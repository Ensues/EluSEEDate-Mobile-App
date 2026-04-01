/**
 * Camera Screen - VisionCamera Native Frame Processor Pipeline
 *
 * High-performance path:
 * - VisionCamera streams raw frames from camera sensor
 * - Worklet resizes to 128x128, converts to RGB CHW, normalizes to [0, 1]
 * - NativeSequenceBuffer assembles [1, 20, 6, 128, 128] tensor views
 * - Tensor is passed directly to react-native-fast-tflite inference service
 */

import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  View,
  Text,
  StyleSheet,
  StatusBar,
  TouchableOpacity,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { NativeStackNavigationProp } from '@react-navigation/native-stack';
import { useIsFocused } from '@react-navigation/native';
import { Camera, useCameraDevice, useCameraFormat, useCameraPermission } from 'react-native-vision-camera';

import { RootStackParamList } from '../navigation/types';
import {
  initializeModel,
  getModelLoadError,
  runPrediction,
  PredictionResult,
  PerformanceMetrics,
} from '../services/convlstmWithoutIntentInference';
import { Detection, YOLOResult } from '../services/yoloInference';
import { ObjectSpeechService } from '../services/ObjectSpeechService';
import { ProcessedTensor } from '../services/preprocessor';
import { FRAME_HEIGHT, FRAME_WIDTH, SEQ_LEN } from '../config/modelConfig';
import {
  CONVLSTM_TENSOR_SHAPE,
  NativeRGBFrame,
  NativeSequenceBuffer,
  useConvLSTMFrameProcessor,
} from '../worklets/convlstmFrameProcessor';

type CameraScreenProps = {
  navigation: NativeStackNavigationProp<RootStackParamList, 'Camera'>;
};

const TARGET_FRAME_PROCESSOR_FPS = 20;
const FALLBACK_CAMERA_FPS = 30;
const UI_UPDATE_INTERVAL_MS = 200;
const YOLO_SPEECH_CONFIDENCE_THRESHOLD = 0.45;

export default function CameraScreen({ navigation }: CameraScreenProps) {
  const isFocused = useIsFocused();
  const { hasPermission, requestPermission } = useCameraPermission();
  const device = useCameraDevice('back');
  const targetFormat = useCameraFormat(device, [
    { fps: TARGET_FRAME_PROCESSOR_FPS },
    { videoResolution: { width: FRAME_WIDTH, height: FRAME_HEIGHT } },
  ]);
  const fallbackFormat = useCameraFormat(device, [
    { fps: FALLBACK_CAMERA_FPS },
    { videoResolution: { width: FRAME_WIDTH, height: FRAME_HEIGHT } },
  ]);

  const cameraRef = useRef<Camera>(null);
  const sequenceBufferRef = useRef<NativeSequenceBuffer>(new NativeSequenceBuffer());
  const objectSpeechServiceRef = useRef<ObjectSpeechService>(
    new ObjectSpeechService({
      confidenceThreshold: YOLO_SPEECH_CONFIDENCE_THRESHOLD,
      sameClassCooldownMs: 4000,
      globalCooldownMs: 1200,
      interruptPriorityDelta: 0.18,
    })
  );
  const isInferencingRef = useRef<boolean>(false);
  const frameCounterRef = useRef<number>(0);
  const droppedCounterRef = useRef<number>(0);
  const lastUiUpdateRef = useRef<number>(0);

  const tensorShape = useMemo<number[]>(() => Array.from(CONVLSTM_TENSOR_SHAPE), []);
  const format = useMemo(() => {
    if (!device) {
      return undefined;
    }

    const supportsTargetFps = Boolean(
      targetFormat
      && targetFormat.minFps <= TARGET_FRAME_PROCESSOR_FPS
      && targetFormat.maxFps >= TARGET_FRAME_PROCESSOR_FPS
    );

    if (supportsTargetFps) {
      return targetFormat;
    }

    const supportsFallbackFps = Boolean(
      fallbackFormat
      && fallbackFormat.minFps <= FALLBACK_CAMERA_FPS
      && fallbackFormat.maxFps >= FALLBACK_CAMERA_FPS
    );

    if (supportsFallbackFps) {
      return fallbackFormat;
    }

    return targetFormat ?? fallbackFormat ?? device.formats[0];
  }, [device, fallbackFormat, targetFormat]);

  const selectedCameraFps = useMemo(() => {
    if (!format) {
      return undefined;
    }

    if (format.minFps <= TARGET_FRAME_PROCESSOR_FPS && format.maxFps >= TARGET_FRAME_PROCESSOR_FPS) {
      return TARGET_FRAME_PROCESSOR_FPS;
    }

    if (format.minFps <= FALLBACK_CAMERA_FPS && format.maxFps >= FALLBACK_CAMERA_FPS) {
      return FALLBACK_CAMERA_FPS;
    }

    return format.maxFps;
  }, [format]);

  const hasVerifiedFormat = Boolean(format && selectedCameraFps);

  const [isModelLoaded, setIsModelLoaded] = useState<boolean>(false);
  const [isCameraReady, setIsCameraReady] = useState<boolean>(false);
  const [isCameraActive, setIsCameraActive] = useState<boolean>(false);

  const [currentPrediction, setCurrentPrediction] = useState<PredictionResult | null>(null);
  const [directionLabel, setDirectionLabel] = useState<string>('Waiting...');
  const [confidence, setConfidence] = useState<number>(0);
  const [metrics, setMetrics] = useState<PerformanceMetrics>({
    preprocessingTimeMs: 0,
    inferenceTimeMs: 0,
    totalLatencyMs: 0,
    fps: 0,
  });

  const [frameCount, setFrameCount] = useState<number>(0);
  const [predictionCount, setPredictionCount] = useState<number>(0);
  const [bufferCount, setBufferCount] = useState<number>(0);
  const [droppedFrames, setDroppedFrames] = useState<number>(0);
  const [frameProcessorTimeMs, setFrameProcessorTimeMs] = useState<number>(0);
  const [yoloInferenceTimeMs, setYoloInferenceTimeMs] = useState<number>(0);
  const [yoloDetections, setYoloDetections] = useState<Detection[]>([]);
  const [frameOrientation, setFrameOrientation] = useState<string>('unknown');
  const [debugStatus, setDebugStatus] = useState<string>('Initializing...');

  const requestCameraPermission = useCallback(async () => {
    try {
      const granted = await requestPermission();
      if (!granted) {
        setDebugStatus('Camera permission denied');
      }
    } catch (error: any) {
      setDebugStatus(`Permission error: ${error?.message || 'unknown error'}`);
    }
  }, [requestPermission]);

  useEffect(() => {
    if (!hasPermission) {
      void requestCameraPermission();
    }
  }, [hasPermission, requestCameraPermission]);

  useEffect(() => {
    let isMounted = true;

    const initConvLSTM = async () => {
      setDebugStatus('Loading ConvLSTM model...');
      const loaded = await initializeModel();

      if (!isMounted) {
        return;
      }

      setIsModelLoaded(loaded);
      if (loaded) {
        setDebugStatus('Model ready | waiting for camera');
      } else {
        setDebugStatus(`Model load failed: ${getModelLoadError() || 'unknown error'}`);
      }
    };

    void initConvLSTM();

    return () => {
      isMounted = false;
      isInferencingRef.current = false;
      sequenceBufferRef.current.clear();
      void objectSpeechServiceRef.current.dispose();
    };
  }, []);

  useEffect(() => {
    const shouldActivate = Boolean(
      isFocused
      && hasPermission
      && isModelLoaded
      && device
      && hasVerifiedFormat
    );

    setIsCameraActive(shouldActivate);

    if (!shouldActivate) {
      setIsCameraReady(false);
      sequenceBufferRef.current.clear();
      setBufferCount(0);
      setYoloDetections([]);
      setYoloInferenceTimeMs(0);
      void objectSpeechServiceRef.current.stop();
    }
  }, [device, hasPermission, hasVerifiedFormat, isFocused, isModelLoaded]);

  useEffect(() => {
    if (!device) {
      return;
    }

    if (!hasVerifiedFormat || !format || !selectedCameraFps) {
      setDebugStatus('Selecting camera format...');
      return;
    }

    const isFallback = selectedCameraFps !== TARGET_FRAME_PROCESSOR_FPS;
    const fpsLabel = isFallback
      ? `${selectedCameraFps} FPS fallback active`
      : `${selectedCameraFps} FPS format ready`;

    setDebugStatus(
      `Format ${format.videoWidth}x${format.videoHeight} | ${fpsLabel}`
    );
  }, [device, format, hasVerifiedFormat, selectedCameraFps]);

  // Invoked from the YOLO inference loop whenever a new YOLOResult is available.
  const handleYOLODetections = useCallback((result: YOLOResult) => {
    setYoloDetections(result.detections);
    setYoloInferenceTimeMs(result.inferenceTimeMs);
  }, []);

  useEffect(() => {
    if (yoloDetections.length === 0) {
      return;
    }

    void objectSpeechServiceRef.current.announceDetections(yoloDetections);
  }, [yoloDetections]);

  const refreshUiStats = useCallback((frame: NativeRGBFrame) => {
    const now = Date.now();
    if (now - lastUiUpdateRef.current < UI_UPDATE_INTERVAL_MS) {
      return;
    }

    lastUiUpdateRef.current = now;
    setFrameCount(frameCounterRef.current);
    setDroppedFrames(droppedCounterRef.current);
    setBufferCount(sequenceBufferRef.current.getFrameCount());
    setFrameProcessorTimeMs(frame.preprocessTimeMs);
    setFrameOrientation(frame.orientation);
  }, []);

  const runInferenceFromBuffer = useCallback(async (latestPreprocessMs: number) => {
    if (!isModelLoaded || !sequenceBufferRef.current.isReady() || isInferencingRef.current) {
      return;
    }

    isInferencingRef.current = true;
    setDebugStatus('Running ConvLSTM inference...');

    try {
      const tensor: ProcessedTensor = {
        data: sequenceBufferRef.current.buildTensorView(),
        shape: tensorShape,
        processingTimeMs: latestPreprocessMs,
      };

      const { prediction, metrics: newMetrics } = await runPrediction(tensor);

      setCurrentPrediction(prediction);
      setDirectionLabel(prediction.className);
      setConfidence(prediction.confidence);
      setMetrics(newMetrics);

      setPredictionCount((previous) => {
        const next = previous + 1;
        setDebugStatus(
          `Prediction #${next}: ${prediction.className} (${(prediction.confidence * 100).toFixed(1)}%)`
        );
        return next;
      });
    } catch (error: any) {
      setDebugStatus(`Inference error: ${error?.message || 'unknown error'}`);
    } finally {
      isInferencingRef.current = false;
    }
  }, [isModelLoaded, tensorShape]);

  const handleNativeFrame = useCallback((frame: NativeRGBFrame) => {
    frameCounterRef.current += 1;

    sequenceBufferRef.current.addFrame(frame);
    refreshUiStats(frame);

    if (!sequenceBufferRef.current.isReady()) {
      setDebugStatus(`Buffering frames: ${sequenceBufferRef.current.getFrameCount()}/${SEQ_LEN}`);
      return;
    }

    if (isInferencingRef.current) {
      droppedCounterRef.current += 1;
      return;
    }

    void runInferenceFromBuffer(frame.preprocessTimeMs);
  }, [refreshUiStats, runInferenceFromBuffer]);

  const frameProcessor = useConvLSTMFrameProcessor({
    enabled: isCameraActive && isCameraReady && hasVerifiedFormat,
    targetFps: TARGET_FRAME_PROCESSOR_FPS,
    onFrame: handleNativeFrame,
  });

  const resolvedFrameProcessor = hasVerifiedFormat ? frameProcessor : undefined;

  const handleBack = useCallback(() => {
    setIsCameraActive(false);
    sequenceBufferRef.current.clear();
    navigation.goBack();
  }, [navigation]);

  if (!hasPermission) {
    return (
      <SafeAreaView style={styles.container}>
        <View style={styles.permissionContainer}>
          <Text style={styles.permissionText}>Camera access is required for real-time prediction.</Text>
          <TouchableOpacity style={styles.permissionButton} onPress={requestCameraPermission}>
            <Text style={styles.permissionButtonText}>Grant Permission</Text>
          </TouchableOpacity>
        </View>
      </SafeAreaView>
    );
  }

  if (!device) {
    return (
      <SafeAreaView style={styles.container}>
        <View style={styles.permissionContainer}>
          <Text style={styles.permissionText}>No rear camera device found.</Text>
          <TouchableOpacity style={styles.permissionButton} onPress={handleBack}>
            <Text style={styles.permissionButtonText}>Back</Text>
          </TouchableOpacity>
        </View>
      </SafeAreaView>
    );
  }

  if (!hasVerifiedFormat || !format || !selectedCameraFps) {
    return (
      <SafeAreaView style={styles.container}>
        <View style={styles.permissionContainer}>
          <Text style={styles.permissionText}>Selecting compatible camera format...</Text>
          <Text style={styles.tensorShapeText}>Target: {FRAME_WIDTH}x{FRAME_HEIGHT} @ {TARGET_FRAME_PROCESSOR_FPS} FPS</Text>
          <TouchableOpacity style={styles.permissionButton} onPress={handleBack}>
            <Text style={styles.permissionButtonText}>Back</Text>
          </TouchableOpacity>
        </View>
      </SafeAreaView>
    );
  }

  return (
    <View style={styles.container}>
      <StatusBar barStyle="light-content" backgroundColor="transparent" translucent />

      <Camera
        ref={cameraRef}
        style={styles.camera}
        device={device}
        format={format}
        isActive={isCameraActive}
        pixelFormat="yuv"
        fps={selectedCameraFps}
        photo={false}
        video={false}
        audio={false}
        frameProcessor={resolvedFrameProcessor}
        onInitialized={() => {
          setIsCameraReady(true);
          setDebugStatus(
            `VisionCamera ready | ${format.videoWidth}x${format.videoHeight} @ ${selectedCameraFps} FPS`
          );
        }}
        onError={(error) => {
          setDebugStatus(`Camera error: ${error.message}`);
        }}
      />

      <View style={styles.overlayContainer}>
        {!isCameraReady && (
          <View style={styles.cameraStatusOverlay}>
            <Text style={styles.cameraStatusText}>Initializing VisionCamera...</Text>
            <Text style={styles.cameraStatusSubtext}>Preparing native frame processor</Text>
          </View>
        )}

        <View style={styles.performanceOverlay}>
          <Text style={styles.performanceTitle}>Performance</Text>
          <Text style={styles.performanceText}>Frame Processor: {frameProcessorTimeMs.toFixed(1)} ms</Text>
          <Text style={styles.performanceText}>Inference: {metrics.inferenceTimeMs.toFixed(1)} ms</Text>
          <Text style={styles.performanceText}>YOLO: {yoloInferenceTimeMs.toFixed(1)} ms</Text>
          <Text style={styles.performanceText}>Total: {metrics.totalLatencyMs.toFixed(1)} ms</Text>
          <Text style={styles.performanceText}>FPS: {metrics.fps.toFixed(1)}</Text>
          <View style={styles.performanceDivider} />
          <Text style={styles.performanceText}>Frames: {frameCount}</Text>
          <Text style={styles.performanceText}>Buffered: {bufferCount}/{SEQ_LEN}</Text>
          <Text style={styles.performanceText}>Predictions: {predictionCount}</Text>
          <Text style={styles.performanceText}>Objects: {yoloDetections.length}</Text>
          <Text style={styles.performanceText}>Dropped (busy): {droppedFrames}</Text>
          <Text style={styles.performanceText}>Orientation: {frameOrientation}</Text>
          <View style={styles.performanceDivider} />
          <Text style={styles.debugText} numberOfLines={2}>{debugStatus}</Text>
        </View>

        <TouchableOpacity style={styles.backButton} onPress={handleBack}>
          <Text style={styles.backButtonText}>X</Text>
        </TouchableOpacity>

        <View style={styles.statusIndicator}>
          <View
            style={[
              styles.statusDot,
              { backgroundColor: (isCameraReady && isCameraActive) ? '#00ff00' : '#666666' },
            ]}
          />
          <Text style={styles.statusText}>
            {isCameraReady && isCameraActive ? 'Streaming native frames' : 'Camera paused'}
          </Text>
          {!isModelLoaded && <Text style={styles.statusText}> | ConvLSTM offline</Text>}
        </View>

        <View style={styles.directionContainer}>
          <Text style={styles.directionLabel}>{directionLabel}</Text>
          {currentPrediction && (
            <Text style={styles.confidenceText}>{(confidence * 100).toFixed(1)}%</Text>
          )}
          <Text style={styles.tensorShapeText}>
            Tensor: [{tensorShape.join(', ')}]
          </Text>
          <Text style={styles.tensorShapeText}>Frame: {FRAME_WIDTH}x{FRAME_HEIGHT} RGB + 3 intent zeros</Text>
        </View>

        <View style={styles.bufferProgress}>
          <View style={styles.bufferContainer}>
            <View
              style={[
                styles.bufferFill,
                { width: `${(bufferCount / SEQ_LEN) * 100}%` },
              ]}
            />
          </View>
          <Text style={styles.bufferText}>Sequence Buffer: {bufferCount}/{SEQ_LEN}</Text>
        </View>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#000000',
  },
  camera: {
    flex: 1,
  },
  overlayContainer: {
    ...StyleSheet.absoluteFillObject,
    pointerEvents: 'box-none',
    zIndex: 100,
  },
  cameraStatusOverlay: {
    position: 'absolute',
    top: '40%',
    left: 0,
    right: 0,
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: 'rgba(0, 0, 0, 0.9)',
    padding: 30,
    marginHorizontal: 40,
    borderRadius: 12,
    borderWidth: 2,
    borderColor: '#444444',
  },
  cameraStatusText: {
    fontSize: 18,
    color: '#ffffff',
    fontWeight: '400',
    marginBottom: 8,
  },
  cameraStatusSubtext: {
    fontSize: 14,
    color: '#888888',
  },
  performanceOverlay: {
    position: 'absolute',
    top: 50,
    left: 16,
    backgroundColor: 'rgba(0, 0, 0, 0.85)',
    padding: 10,
    borderRadius: 4,
    minWidth: 170,
    borderWidth: 1,
    borderColor: '#333333',
  },
  performanceTitle: {
    fontSize: 10,
    fontWeight: '500',
    color: '#888888',
    marginBottom: 4,
    textTransform: 'uppercase',
    letterSpacing: 1,
  },
  performanceText: {
    fontSize: 11,
    color: '#ffffff',
    fontFamily: 'monospace',
    lineHeight: 16,
  },
  debugText: {
    fontSize: 9,
    color: '#00ff00',
    fontFamily: 'monospace',
    lineHeight: 12,
    maxWidth: 180,
  },
  performanceDivider: {
    height: 1,
    backgroundColor: '#333333',
    marginVertical: 4,
  },
  backButton: {
    position: 'absolute',
    top: 50,
    right: 16,
    width: 40,
    height: 40,
    backgroundColor: 'rgba(0, 0, 0, 0.85)',
    borderRadius: 20,
    justifyContent: 'center',
    alignItems: 'center',
    borderWidth: 1,
    borderColor: '#333333',
  },
  backButtonText: {
    fontSize: 18,
    color: '#ffffff',
    fontWeight: '300',
  },
  statusIndicator: {
    position: 'absolute',
    top: 56,
    left: 0,
    right: 0,
    flexDirection: 'row',
    justifyContent: 'center',
    alignItems: 'center',
  },
  statusDot: {
    width: 6,
    height: 6,
    borderRadius: 3,
    marginRight: 6,
  },
  statusText: {
    fontSize: 11,
    color: '#ffffff',
    fontWeight: '400',
  },
  directionContainer: {
    position: 'absolute',
    bottom: 80,
    left: 20,
    right: 20,
    paddingVertical: 20,
    paddingHorizontal: 20,
    borderRadius: 8,
    alignItems: 'center',
    backgroundColor: 'rgba(0, 0, 0, 0.85)',
    borderWidth: 1,
    borderColor: '#333333',
  },
  directionLabel: {
    fontSize: 40,
    fontWeight: '300',
    color: '#ffffff',
    letterSpacing: 6,
  },
  confidenceText: {
    fontSize: 12,
    color: '#888888',
    marginTop: 6,
  },
  tensorShapeText: {
    fontSize: 10,
    color: '#888888',
    marginTop: 4,
    textAlign: 'center',
  },
  bufferProgress: {
    position: 'absolute',
    bottom: 30,
    left: 20,
    right: 20,
    alignItems: 'center',
  },
  bufferContainer: {
    width: '100%',
    height: 2,
    backgroundColor: 'rgba(255, 255, 255, 0.2)',
    borderRadius: 1,
    overflow: 'hidden',
  },
  bufferFill: {
    height: '100%',
    backgroundColor: '#ffffff',
    borderRadius: 1,
  },
  bufferText: {
    fontSize: 10,
    color: 'rgba(255, 255, 255, 0.5)',
    marginTop: 4,
  },
  permissionContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    padding: 40,
  },
  permissionText: {
    fontSize: 16,
    color: '#ffffff',
    textAlign: 'center',
    marginBottom: 20,
  },
  permissionButton: {
    backgroundColor: '#ffffff',
    paddingVertical: 12,
    paddingHorizontal: 24,
    borderRadius: 20,
  },
  permissionButtonText: {
    fontSize: 14,
    color: '#000000',
    fontWeight: '500',
  },
});
