/**
 * Camera Screen - EluSEEdate
 * 
 * Live camera view with real-time turn prediction
 * - Captures frames silently from rear camera (no sound/flash)
 * - Uses simplified capture approach compatible with Expo Go
 * - Shows predicted direction at bottom
 * - Shows inference/latency metrics at top-left
 * 
 * NOTE: takePictureAsync is slow (~200-500ms), so we capture at 2-3 FPS
 * and duplicate frames to fill the 20-frame buffer for inference.
 */

import React, { useEffect, useRef, useState, useCallback } from 'react';
import {
  View,
  Text,
  StyleSheet,
  StatusBar,
  TouchableOpacity,
  Dimensions,
  Alert,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { CameraView, useCameraPermissions } from 'expo-camera';
import { NativeStackNavigationProp } from '@react-navigation/native-stack';
import { RootStackParamList } from '../navigation/types';
import {
  FrameBuffer,
  VideoPreprocessor,
  FrameData,
} from '../services/preprocessor';
import {
  runPrediction,
  initializeModel,
  PredictionResult,
  PerformanceMetrics,
} from '../services/inference';
import { SEQ_LEN, DEVICE_CONFIG, FRAME_WIDTH, FRAME_HEIGHT } from '../config/modelConfig';
import { decodeBase64ToPixels } from '../utils/imageUtils';

type CameraScreenProps = {
  navigation: NativeStackNavigationProp<RootStackParamList, 'Camera'>;
};

// Realistic capture FPS for takePictureAsync (slow but works in Expo Go)
const REALISTIC_CAPTURE_FPS = 2;

export default function CameraScreen({ navigation }: CameraScreenProps) {
  // Camera permission state
  const [permission, requestPermission] = useCameraPermissions();
  
  // Camera reference for frame capture
  const cameraRef = useRef<CameraView>(null);
  
  // Camera mount state
  const [isCameraReady, setIsCameraReady] = useState<boolean>(false);
  
  // Frame buffer for storing captured frames (use realistic FPS)
  const frameBufferRef = useRef<FrameBuffer>(new FrameBuffer(REALISTIC_CAPTURE_FPS));
  
  // Preprocessor instance
  const preprocessorRef = useRef<VideoPreprocessor>(new VideoPreprocessor());
  
  // Prediction state
  const [currentPrediction, setCurrentPrediction] = useState<PredictionResult | null>(null);
  const [directionLabel, setDirectionLabel] = useState<string>('Waiting...');
  const [confidence, setConfidence] = useState<number>(0);
  
  // Performance metrics state
  const [metrics, setMetrics] = useState<PerformanceMetrics>({
    preprocessingTimeMs: 0,
    inferenceTimeMs: 0,
    totalLatencyMs: 0,
    fps: 0,
  });
  
  // Processing state
  const [isModelLoaded, setIsModelLoaded] = useState<boolean>(false);
  const [isCapturing, setIsCapturing] = useState<boolean>(false);
  const [frameCount, setFrameCount] = useState<number>(0);
  const [predictionCount, setPredictionCount] = useState<number>(0);
  const [debugStatus, setDebugStatus] = useState<string>('Initializing...');
  const [lastCaptureTime, setLastCaptureTime] = useState<number>(0);
  
  // Inference lock to prevent concurrent inferences
  const isInferencingRef = useRef<boolean>(false);
  const isCapturingRef = useRef<boolean>(false);
  
  // Capture interval reference (now using setTimeout for async control)
  const captureIntervalRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  /**
   * Initialize model on screen mount
   */
  useEffect(() => {
    console.log('[Camera] Screen mounted');
    console.log('[Camera] Permission status:', permission?.granted ? 'granted' : 'not granted');
    
    const initModel = async () => {
      console.log('[Camera] Initializing model...');
      setDebugStatus('Loading model...');
      const loaded = await initializeModel();
      setIsModelLoaded(loaded);
      if (loaded) {
        console.log('[Camera] Model initialized successfully');
        setDebugStatus('Model ready');
      } else {
        console.log('[Camera] Model failed to load - using demo mode');
        setDebugStatus('Demo mode (no model)');
        // Don't show alert - app will work in demo mode
      }
    };
    
    initModel();
    
    // Cleanup on unmount
    return () => {
      stopCapture();
    };
  }, []);

  /**
   * Start continuous frame capture when permission is granted
   * Start regardless of model status (demo mode works too)
   */
  useEffect(() => {
    if (permission?.granted) {
      // Small delay to ensure camera is ready
      const timer = setTimeout(() => {
        startCapture();
      }, 500);
      return () => clearTimeout(timer);
    }
    
    return () => {
      stopCapture();
    };
  }, [permission?.granted]);

  /**
   * Start continuous frame capture
   */
  const startCapture = useCallback(() => {
    if (captureIntervalRef.current || isCapturingRef.current) return;
    
    isCapturingRef.current = true;
    setIsCapturing(true);
    setDebugStatus('Starting capture...');
    console.log(`[Camera] Starting continuous capture at ${REALISTIC_CAPTURE_FPS} FPS...`);
    
    // Use recursive timeout instead of setInterval for proper async handling
    const captureLoop = async () => {
      if (!isCapturingRef.current) return;
      
      const startTime = Date.now();
      await captureFrame();
      const elapsed = Date.now() - startTime;
      
      // Schedule next capture, accounting for time spent
      const captureInterval = 1000 / REALISTIC_CAPTURE_FPS;
      const delay = Math.max(0, captureInterval - elapsed);
      
      captureIntervalRef.current = setTimeout(captureLoop, delay) as any;
    };
    
    // Start the loop
    captureLoop();
  }, []);

  /**
   * Stop frame capture
   */
  const stopCapture = useCallback(() => {
    isCapturingRef.current = false;
    if (captureIntervalRef.current) {
      clearTimeout(captureIntervalRef.current);
      captureIntervalRef.current = null;
    }
    setIsCapturing(false);
    setDebugStatus('Capture stopped');
    console.log('[Camera] Capture stopped');
  }, []);

  /**
   * Capture a single frame from camera
   * Uses takePictureAsync which is slow but works in Expo Go
   */
  const captureFrame = async () => {
    if (!cameraRef.current) {
      console.log('[Camera] Camera ref not available');
      setDebugStatus('Camera not ready');
      return;
    }
    
    const startTime = Date.now();
    
    try {
      setDebugStatus('Capturing frame...');
      
      // Capture frame silently (no sound, no animation)
      const photo = await cameraRef.current.takePictureAsync({
        quality: 0.1,           // Low quality for faster capture
        base64: true,           // Get base64 for processing
        skipProcessing: true,
        shutterSound: false,    // Disable shutter sound
      });
      
      if (!photo) {
        console.log('[Camera] No photo returned');
        setDebugStatus('Capture failed - no photo');
        return;
      }
      
      const captureTime = Date.now() - startTime;
      setLastCaptureTime(captureTime);
      console.log(`[Camera] Frame captured in ${captureTime}ms`);
      
      // Decode base64 image to pixel data
      let frameData: FrameData;
      
      if (photo.base64) {
        try {
          // Decode the base64 image to actual pixel data
          const decoded = await decodeBase64ToPixels(photo.base64, FRAME_WIDTH, FRAME_HEIGHT);
          
          frameData = {
            data: decoded.data,
            width: decoded.width,
            height: decoded.height,
            timestamp: Date.now(),
          };
          
          console.log(`[Camera] Frame decoded: ${decoded.width}x${decoded.height}, ${decoded.data.length} bytes`);
        } catch (decodeError: any) {
          console.warn('[Camera] Failed to decode image, using fallback:', decodeError?.message);
          
          // Fallback to placeholder if decoding fails
          frameData = {
            data: new Uint8Array(FRAME_WIDTH * FRAME_HEIGHT * 4).fill(128),
            width: FRAME_WIDTH,
            height: FRAME_HEIGHT,
            timestamp: Date.now(),
          };
        }
      } else {
        // No base64 data available - use placeholder
        console.warn('[Camera] No base64 data in photo');
        frameData = {
          data: new Uint8Array(FRAME_WIDTH * FRAME_HEIGHT * 4).fill(128),
          width: FRAME_WIDTH,
          height: FRAME_HEIGHT,
          timestamp: Date.now(),
        };
      }
      
      // Add frame to buffer
      const wasAdded = frameBufferRef.current.addFrame(frameData);
      
      if (wasAdded) {
        // Use functional update to avoid stale closure
        setFrameCount(prev => {
          const newCount = prev + 1;
          const buffer = frameBufferRef.current;
          const bufferCount = buffer.getFrameCount();
          setDebugStatus(`Captured: ${newCount} | Buffer: ${bufferCount}/${SEQ_LEN}`);
          console.log(`[Camera] Frame ${newCount} added to buffer (${bufferCount}/${SEQ_LEN})`);
          return newCount;
        });
        
        // Run inference when buffer is ready (or can predict early with padding)
        const buffer = frameBufferRef.current;
        if (buffer.canPredictEarly() && !isInferencingRef.current) {
          await runInferenceWithPadding();
        }
      }
    } catch (error: any) {
      console.error('[Camera] Frame capture error:', error?.message || error);
      setDebugStatus(`Error: ${error?.message || 'capture failed'}`);
    }
  };

  /**
   * Run model inference with frame padding if buffer not full
   */
  const runInferenceWithPadding = async () => {
    const buffer = frameBufferRef.current;
    
    if (isInferencingRef.current) {
      return;
    }
    
    isInferencingRef.current = true;
    setDebugStatus('Running inference...');
    
    try {
      // Get frames with padding (duplicates last frame to fill SEQ_LEN)
      const frames = buffer.getFramesPadded();
      
      // Preprocess frames
      const preprocessor = preprocessorRef.current;
      const tensor = preprocessor.preprocessFrameSequence(frames);
      
      // Run prediction
      const { prediction, metrics: newMetrics } = await runPrediction(tensor);
      
      // Update state
      setCurrentPrediction(prediction);
      setDirectionLabel(prediction.className);
      setConfidence(prediction.confidence);
      setMetrics(newMetrics);
      
      // Use functional update to avoid stale closure
      setPredictionCount(prev => {
        const newPredCount = prev + 1;
        setDebugStatus(`Prediction #${newPredCount}: ${prediction.className}`);
        console.log(`[Camera] Prediction #${newPredCount}: ${prediction.className} (${(prediction.confidence * 100).toFixed(1)}%)`);
        console.log(`[Camera] Latency: ${newMetrics.totalLatencyMs.toFixed(1)}ms`);
        return newPredCount;
      });
      
    } catch (error: any) {
      console.error('[Camera] Inference error:', error?.message || error);
      setDebugStatus(`Inference error: ${error?.message || 'unknown'}`);
    } finally {
      isInferencingRef.current = false;
    }
  };

  /**
   * Handle back button press
   */
  const handleBack = () => {
    stopCapture();
    navigation.goBack();
  };

  // Permission not determined yet
  if (!permission) {
    console.log('[Camera] Permission not determined yet');
    return (
      <SafeAreaView style={styles.container}>
        <Text style={styles.permissionText}>Requesting camera permission...</Text>
      </SafeAreaView>
    );
  }

  // Permission denied
  if (!permission.granted) {
    console.log('[Camera] Permission denied');
    return (
      <SafeAreaView style={styles.container}>
        <View style={styles.permissionContainer}>
          <Text style={styles.permissionText}>Camera access is required</Text>
          <TouchableOpacity style={styles.permissionButton} onPress={requestPermission}>
            <Text style={styles.permissionButtonText}>Grant Permission</Text>
          </TouchableOpacity>
        </View>
      </SafeAreaView>
    );
  }
  
  console.log('[Camera] Rendering camera view - permission granted');

  return (
    <View style={styles.container}>
      <StatusBar barStyle="light-content" backgroundColor="transparent" translucent />
      
      {/* Camera View - Silent capture mode (no children allowed) */}
      <CameraView
        ref={cameraRef}
        style={styles.camera}
        facing="back"
        mode="picture"
        animateShutter={false}
        enableTorch={false}
        onCameraReady={() => {
          console.log('[Camera] Camera is ready!');
          setIsCameraReady(true);
          setDebugStatus('Camera ready');
        }}
        onMountError={(error) => {
          console.error('[Camera] Mount error:', error);
          setDebugStatus(`Camera error: ${error.message}`);
        }}
      />
      
      {/* Overlay Container - Absolute positioned on top of camera */}
      <View style={styles.overlayContainer}>
        {/* Debug Camera Status - Center of screen */}
        {!isCameraReady && (
          <View style={styles.cameraStatusOverlay}>
            <Text style={styles.cameraStatusText}>📷 Initializing Camera...</Text>
            <Text style={styles.cameraStatusSubtext}>Please wait</Text>
          </View>
        )}
        
        {/* Performance Overlay (Top-Left) */}
        <View style={styles.performanceOverlay}>
          <Text style={styles.performanceTitle}>Performance</Text>
          <Text style={styles.performanceText}>
            Capture: {lastCaptureTime} ms
          </Text>
          <Text style={styles.performanceText}>
            Inference: {metrics.inferenceTimeMs.toFixed(0)} ms
          </Text>
          <Text style={styles.performanceText}>
            Preprocess: {metrics.preprocessingTimeMs.toFixed(0)} ms
          </Text>
          <Text style={styles.performanceText}>
            Total: {metrics.totalLatencyMs.toFixed(0)} ms
          </Text>
          <View style={styles.performanceDivider} />
          <Text style={styles.performanceText}>
            Frames: {frameCount}
          </Text>
          <Text style={styles.performanceText}>
            Predictions: {predictionCount}
          </Text>
          <View style={styles.performanceDivider} />
          <Text style={styles.debugText} numberOfLines={2}>
            {debugStatus}
          </Text>
        </View>

        {/* Back Button (Top-Right) */}
        <TouchableOpacity style={styles.backButton} onPress={handleBack}>
          <Text style={styles.backButtonText}>✕</Text>
        </TouchableOpacity>

        {/* Status Indicator */}
        <View style={styles.statusIndicator}>
          <View style={[
            styles.statusDot,
            { backgroundColor: (isCameraReady && isCapturing) ? '#00ff00' : '#666666' }
          ]} />
          <Text style={styles.statusText}>
            {!isCameraReady ? 'Camera initializing...' : isCapturing ? 'Capturing' : 'Paused'}
          </Text>
          {!isModelLoaded && (
            <Text style={styles.statusText}> | Demo Mode</Text>
          )}
        </View>

        {/* Direction Label (Bottom) */}
        <View style={styles.directionContainer}>
          <Text style={styles.directionLabel}>{directionLabel}</Text>
          {currentPrediction && (
            <Text style={styles.confidenceText}>
              {(confidence * 100).toFixed(1)}%
            </Text>
          )}
        </View>

        {/* Frame Buffer Progress */}
        <View style={styles.bufferProgress}>
          <View style={styles.bufferContainer}>
            <View 
              style={[
                styles.bufferFill,
                { width: `${(frameBufferRef.current.getFrameCount() / SEQ_LEN) * 100}%` }
              ]} 
            />
          </View>
          <Text style={styles.bufferText}>
            Buffer: {frameBufferRef.current.getFrameCount()}/{SEQ_LEN}
          </Text>
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

  // Overlay container - positioned absolutely on top of camera
  overlayContainer: {
    ...StyleSheet.absoluteFillObject,
    pointerEvents: 'box-none',
  },

  // Camera Status Overlay (Center)
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

  // Performance Overlay (Top-Left)
  performanceOverlay: {
    position: 'absolute',
    top: 50,
    left: 16,
    backgroundColor: 'rgba(0, 0, 0, 0.85)',
    padding: 10,
    borderRadius: 4,
    minWidth: 120,
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
    maxWidth: 140,
  },
  performanceDivider: {
    height: 1,
    backgroundColor: '#333333',
    marginVertical: 4,
  },

  // Back Button (Top-Right)
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

  // Status Indicator
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

  // Direction Label (Bottom)
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

  // Buffer Progress
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

  // Permission States
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
