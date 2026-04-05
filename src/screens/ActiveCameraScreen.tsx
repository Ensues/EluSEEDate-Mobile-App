/*
 * Active Camera Screen - EluSEEdate
 *
 * Live camera view with real-time turn prediction
 * - Captures frames silently from rear camera (no sound/flash)
 * - Uses simplified capture approach compatible with Expo Go
 * - Supports two modes via route params:
 *   1) wandering: lightweight pipeline
 *   2) destination: wayfinding pipeline
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
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { CameraView, useCameraPermissions } from 'expo-camera';
import { NativeStackScreenProps } from '@react-navigation/native-stack';
import { RootStackParamList } from '../navigation/types';
import {
  FrameBuffer,
  VideoPreprocessor,
  FrameData,
} from '../services/preprocessor';
import type {
  PredictionResult,
  PerformanceMetrics,
} from '../services/convlstmWithoutIntentInference';
import * as convlstmWithoutIntent from '../services/convlstmWithoutIntentInference';
import * as convlstmWithIntent from '../services/convlstmWithIntentInference';
import {
  runYOLODetection as detectObjects,
  initializeYOLOModel,
  YOLOResult,
  Detection,
} from '../services/yoloInference';
import { ObjectSpeechService } from '../services/ObjectSpeechService';
import BoundingBoxOverlay from '../components/BoundingBoxOverlay';
import {
  SEQ_LEN,
  FRAME_WIDTH,
  FRAME_HEIGHT,
  CLASS_NAMES,
} from '../config/modelConfig';
import { decodeBase64ToPixels, decodeImageUriToPixels } from '../utils/imageUtils';
import { fetchWalkingDirections, maneuverToIntent, DirectionsResult } from '../services/directionsService';
import * as Location from 'expo-location';
import getGeoDistance from 'geolib/es/getDistance';

type ActiveCameraScreenProps = NativeStackScreenProps<RootStackParamList, 'ActiveCamera'>;

// Realistic capture FPS for takePictureAsync (slow but works in Expo Go)
const REALISTIC_CAPTURE_FPS = 2;
const TARGET_CAPTURE_AREA = 640 * 480;
const DIRECTIONS_DESTINATION_CHANGE_THRESHOLD_METERS = 3;

const getDetectionArea = (detection: Detection): number => {
  return detection.boundingBox.width * detection.boundingBox.height;
};

const getLargestAreaDetection = (detections: Detection[]): Detection | null => {
  if (!detections.length) {
    return null;
  }

  return detections.reduce((largest, current) => {
    return getDetectionArea(current) > getDetectionArea(largest) ? current : largest;
  });
};

const getConvLSTMPriority = (prediction: PredictionResult): {
  label: string;
  probability: number;
} => {
  if (!prediction.probabilities.length) {
    return {
      label: prediction.className,
      probability: prediction.confidence,
    };
  }

  let bestIndex = 0;
  for (let index = 1; index < prediction.probabilities.length; index += 1) {
    if (prediction.probabilities[index] > prediction.probabilities[bestIndex]) {
      bestIndex = index;
    }
  }

  return {
    label: CLASS_NAMES[bestIndex] ?? prediction.className,
    probability: prediction.probabilities[bestIndex],
  };
};

const formatConvLSTMTopProbabilities = (probabilities: number[]): string => {
  if (!probabilities.length) {
    return 'none';
  }

  const ranked = probabilities
    .map((value, index) => ({
      label: CLASS_NAMES[index] ?? `Class-${index}`,
      probability: value,
    }))
    .sort((a, b) => b.probability - a.probability)
    .slice(0, 3);

  return ranked
    .map((entry) => `${entry.label}:${entry.probability.toFixed(4)}`)
    .join(' | ');
};

const formatDistanceForOverlay = (meters: number | null | undefined): string => {
  if (typeof meters !== 'number' || !Number.isFinite(meters) || meters < 0) {
    return '0 m';
  }

  if (meters < 100) {
    return `${Math.round(meters)} m`;
  }

  return `${(meters / 1000).toFixed(2)} km`;
};

export default function ActiveCameraScreen({ navigation, route }: ActiveCameraScreenProps) {
  const mode = route.params.mode;
  const modeLabel = mode === 'destination' ? 'Destination' : 'Wandering';
  const useIntentPipeline = mode === 'destination';
  const activePipelineName = mode === 'destination' ? 'Wayfinding pipeline' : 'Wandering pipeline';
  const convlstmService = useIntentPipeline ? convlstmWithIntent : convlstmWithoutIntent;
  const preprocessorChannels: 3 | 6 = useIntentPipeline ? 6 : 3;

  const destinationCoordinates = route.params.destination;
  const destinationLabel = route.params.destinationLabel;
  const routeStepCount = route.params.routeSteps?.length ?? 0;
  const totalDistanceMeters = route.params.totalDistanceMeters;

  // Wayfinding state (live GPS + directions)
  const [userLocation, setUserLocation] = useState<{ latitude: number; longitude: number } | null>(null); // GPS coordinates
  const [directionsCache, setDirectionsCache] = useState<DirectionsResult | null>(null); // Full route from the Map API
  const [currentStepIndex, setCurrentStepIndex] = useState<number>(0); // Step or part index in the route
  const [routeProgress, setRouteProgress] = useState({
    distanceRemaining: 0,
    distanceToStepEnd: 0,
  });
  const [currentDistance, setCurrentDistance] = useState<number | null>(totalDistanceMeters ?? null);
  
  // Camera permission state
  const [permission, requestPermission] = useCameraPermissions();
  
  // Camera reference for frame capture
  const cameraRef = useRef<CameraView>(null);
  
  // Camera mount state
  const [isCameraReady, setIsCameraReady] = useState<boolean>(false);
  const [cameraPictureSize, setCameraPictureSize] = useState<string | undefined>(undefined);
  const [isPictureSizeConfigured, setIsPictureSizeConfigured] = useState<boolean>(false);
  
  // Frame buffer for storing captured frames (use realistic FPS)
  const frameBufferRef = useRef<FrameBuffer>(new FrameBuffer(REALISTIC_CAPTURE_FPS));
  
  // Preprocessor instance
  const preprocessorRef = useRef<VideoPreprocessor>(
    new VideoPreprocessor(FRAME_HEIGHT, FRAME_WIDTH, SEQ_LEN, true, preprocessorChannels)
  );
  
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
  const [debugStatus, setDebugStatus] = useState<string>(`Initializing ${modeLabel} mode...`);
  const [lastCaptureTime, setLastCaptureTime] = useState<number>(0);
  
  // YOLO detection state
  const [isYOLOModelLoaded, setIsYOLOModelLoaded] = useState<boolean>(false);
  const [yoloDetections, setYoloDetections] = useState<Detection[]>([]);
  const [yoloInferenceTime, setYoloInferenceTime] = useState<number>(0);
  const [audioState, setAudioState] = useState<'Ready' | 'Speaking' | 'Error'>('Ready');
  const [lastAnnouncedObject, setLastAnnouncedObject] = useState<string>('None');

  // Object speech service (single instance for the screen lifecycle)
  const objectSpeechServiceRef = useRef<ObjectSpeechService>(new ObjectSpeechService());
  
  // Inference lock to prevent concurrent inferences
  const isInferencingRef = useRef<boolean>(false);
  const isCapturingRef = useRef<boolean>(false);
  const isYOLOInferencingRef = useRef<boolean>(false);
  const isModelLoadedRef = useRef<boolean>(false);
  const isYOLOModelLoadedRef = useRef<boolean>(false);
  const isCameraReadyRef = useRef<boolean>(false);
  const isPictureSizeConfiguredRef = useRef<boolean>(false);
  const hasConfiguredPictureSizeRef = useRef<boolean>(false);
  const isConfiguringPictureSizeRef = useRef<boolean>(false);
  const hasFirstPredictionRef = useRef<boolean>(false);
  const captureSequenceRef = useRef<number>(0);
  const predictionCountRef = useRef<number>(0);
  const isFetchingDirections = useRef<boolean>(false);
  const lastDirectionsDestinationRef = useRef<{ latitude: number; longitude: number } | null>(null);
  
  // Capture interval reference (now using setTimeout for async control)
  const captureIntervalRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const configurePictureSize = useCallback(async () => {
    if (hasConfiguredPictureSizeRef.current || isConfiguringPictureSizeRef.current) {
      return;
    }

    isConfiguringPictureSizeRef.current = true;
    isPictureSizeConfiguredRef.current = false;
    setIsPictureSizeConfigured(false);

    if (!cameraRef.current) {
      isPictureSizeConfiguredRef.current = true;
      hasConfiguredPictureSizeRef.current = true;
      isConfiguringPictureSizeRef.current = false;
      setIsPictureSizeConfigured(true);
      return;
    }

    try {
      const sizes = await cameraRef.current.getAvailablePictureSizesAsync();

      if (!sizes?.length) {
        console.warn('[ActiveCamera] No picture sizes reported by device, using default');
        return;
      }

      const parsedSizes = sizes
        .map((raw) => {
          const match = raw.match(/^(\d+)x(\d+)$/);
          if (!match) return null;

          const width = Number(match[1]);
          const height = Number(match[2]);
          if (!Number.isFinite(width) || !Number.isFinite(height)) return null;

          return { raw, area: width * height };
        })
        .filter((item): item is { raw: string; area: number } => item !== null);

      if (!parsedSizes.length) {
        console.warn('[ActiveCamera] Unable to parse picture sizes, using default');
        return;
      }

      const preferred = parsedSizes.reduce((best, current) => {
        const bestDelta = Math.abs(best.area - TARGET_CAPTURE_AREA);
        const currentDelta = Math.abs(current.area - TARGET_CAPTURE_AREA);
        return currentDelta < bestDelta ? current : best;
      });

      setCameraPictureSize((previous) => (previous === preferred.raw ? previous : preferred.raw));
    } catch (error: any) {
      console.warn('[ActiveCamera] Failed to query picture sizes:', error?.message || error);
    } finally {
      isConfiguringPictureSizeRef.current = false;
      isPictureSizeConfiguredRef.current = true;
      hasConfiguredPictureSizeRef.current = true;
      setIsPictureSizeConfigured(true);
    }
  }, []);

  /**
   * Camera readiness callback from expo-camera.
   * Guarded so one-time camera setup does not repeat on prop-driven remount events.
   */
  const handleCameraReady = useCallback(() => {
    if (!isCameraReadyRef.current) {
      isCameraReadyRef.current = true;
      setIsCameraReady(true);
    }

    if (hasConfiguredPictureSizeRef.current || isConfiguringPictureSizeRef.current) {
      return;
    }

    setDebugStatus('Camera ready | configuring picture size');
    void configurePictureSize();
  }, [configurePictureSize]);

  /**
   * Initialize model on screen mount
   */
  useEffect(() => {
    isCameraReadyRef.current = false;
    isPictureSizeConfiguredRef.current = false;
    hasConfiguredPictureSizeRef.current = false;
    isConfiguringPictureSizeRef.current = false;

    preprocessorRef.current = new VideoPreprocessor(
      FRAME_HEIGHT,
      FRAME_WIDTH,
      SEQ_LEN,
      true,
      preprocessorChannels
    );
    
    const initModels = async () => {
      setDebugStatus('Loading models...');
      
      // Initialize ConvLSTM model
      const convlstmLoaded = await convlstmService.initializeModel();
      setIsModelLoaded(convlstmLoaded);
      isModelLoadedRef.current = convlstmLoaded;
      
      // Initialize YOLO model
      const yoloLoaded = await initializeYOLOModel();
      setIsYOLOModelLoaded(yoloLoaded);
      isYOLOModelLoadedRef.current = yoloLoaded;

      if (convlstmLoaded && yoloLoaded) {
        setDebugStatus('Models ready');
      } else if (!convlstmLoaded && !yoloLoaded) {
        setDebugStatus('Model load failed: ConvLSTM + YOLO unavailable');
      } else if (!convlstmLoaded) {
        setDebugStatus('Model load failed: ConvLSTM unavailable');
      } else {
        setDebugStatus('Model load failed: YOLO unavailable');
      }
    };
    
    initModels();
    
    // Cleanup on unmount — use refs only, no state updates
    return () => {
      isCapturingRef.current = false;
      isCameraReadyRef.current = false;
      isPictureSizeConfiguredRef.current = false;
      hasConfiguredPictureSizeRef.current = false;
      isConfiguringPictureSizeRef.current = false;
      if (captureIntervalRef.current) {
        clearTimeout(captureIntervalRef.current);
        captureIntervalRef.current = null;
      }

      void convlstmService.cleanupModel().catch((error: any) => {
        console.warn('[ActiveCamera] Failed to cleanup ConvLSTM model:', error?.message || error);
      });
    };
  }, [convlstmService, preprocessorChannels]);
  
  useEffect(() => {
    isModelLoadedRef.current = isModelLoaded;
  }, [isModelLoaded]);

  useEffect(() => {
    isYOLOModelLoadedRef.current = isYOLOModelLoaded;
  }, [isYOLOModelLoaded]);

  /**
   * Request location permission and start GPS tracking in destination mode
   */
  useEffect(() => {
    if (mode !== 'destination' || !destinationCoordinates) {
      return;
    }

    let isMounted = true;
    let subscription: Location.LocationSubscription | null = null;

    const startTracking = async () => {
      try {
        const { status } = await Location.requestForegroundPermissionsAsync();
        if (!isMounted) {
          return;
        }

        if (status !== 'granted') {
          console.warn('Location permission denied');
          return;
        }

        const locationSubscription = await Location.watchPositionAsync(
          {
            accuracy: Location.Accuracy.High,
            distanceInterval: 2,             // Updates every 2 meters
            timeInterval: 2000,              // Updates every 2 seconds
          },
          (location) => {
            if (!isMounted) {
              return;
            }

            setUserLocation({
              latitude: location.coords.latitude,
              longitude: location.coords.longitude,
            });
          }
        );

        if (!isMounted) {
          locationSubscription.remove();
          return;
        }

        subscription = locationSubscription;
      } catch (error) {
        console.warn('Failed to start GPS tracking:', error);
      }
    };

    void startTracking();

    return () => {
      isMounted = false;
      subscription?.remove();
      subscription = null;
      console.log('GPS tracking stopped.');
    };
  }, [destinationCoordinates, mode]);

  const fetchDirections = useCallback(async () => {
    if (mode !== 'destination' || !userLocation || !destinationCoordinates) {
      return;
    }

    if (isFetchingDirections.current) {
      return;
    }

    const previousDestination = lastDirectionsDestinationRef.current;
    if (previousDestination && directionsCache) {
      const destinationChangeMeters = getGeoDistance(previousDestination, destinationCoordinates);
      if (destinationChangeMeters < DIRECTIONS_DESTINATION_CHANGE_THRESHOLD_METERS) {
        return;
      }
    }

    isFetchingDirections.current = true;

    try {
      const requestedDestination = {
        latitude: destinationCoordinates.latitude,
        longitude: destinationCoordinates.longitude,
      };

      const directions = await fetchWalkingDirections(userLocation, destinationCoordinates);
      setDirectionsCache(directions);
      setCurrentStepIndex(0); // Start at first step
      lastDirectionsDestinationRef.current = requestedDestination;
      console.log('Directions cached:', directions.steps.length, 'steps');
    } catch (error) {
      console.warn('Failed to fetch directions:', error);
    } finally {
      isFetchingDirections.current = false;
    }
  }, [destinationCoordinates, directionsCache, mode, userLocation]);

  // Directions cache
  useEffect(() => {
    void fetchDirections();
  }, [fetchDirections]);

  /**
   * Recalculate live straight-line distance to destination on every location update.
   */
  useEffect(() => {
    if (mode !== 'destination' || !userLocation || !destinationCoordinates) {
      if (mode !== 'destination') {
        setCurrentDistance(null);
      }

      return;
    }

    const liveDistance = getGeoDistance(userLocation, destinationCoordinates);
    setCurrentDistance(liveDistance);
  }, [destinationCoordinates, mode, userLocation]);

  useEffect(() => {
    if (!directionsCache || !userLocation || !useIntentPipeline) return;
    
    const steps = directionsCache.steps;
    if (currentStepIndex >= steps.length) return;
    
    const checkAndAdvanceStep = () => {
      const currentStep = steps[currentStepIndex];
      const distanceToEnd = getGeoDistance(userLocation, currentStep.endLocation);
      
      if (distanceToEnd < 2) { // Within 2 meters, update to the next step
        setCurrentStepIndex(prev => Math.min(prev + 1, steps.length - 1));
        console.log(`[Route] Advanced to step ${currentStepIndex + 1}`);
      }
    };
    
    checkAndAdvanceStep();
  }, [currentStepIndex, directionsCache, useIntentPipeline, userLocation]);

  /**
   * Calculate distances remaining in the current step of the route and the whole route 
   */
  useEffect(() => {
    if (!directionsCache || !userLocation) return;
    
    // Calculate remaining distance
    let remainingDist = 0;
    for (let i = currentStepIndex; i < directionsCache.steps.length; i++) {
      remainingDist += directionsCache.steps[i].distanceMeters;
    }
    
    // Distance to current step end
    const currentStep = directionsCache.steps[currentStepIndex];
    const distToEnd = getGeoDistance(userLocation, currentStep.endLocation);
    
    setRouteProgress({
      distanceRemaining: remainingDist,
      distanceToStepEnd: distToEnd,
    });
  }, [userLocation, directionsCache, currentStepIndex]);

  /**
   * Start/stop object speech with the screen lifecycle.
   */
  useEffect(() => {
    const speechService = objectSpeechServiceRef.current;

    speechService.setDebugListener((snapshot) => {
      const normalizedAudioState = snapshot.state === 'speaking'
        ? 'Speaking'
        : snapshot.state === 'error'
          ? 'Error'
          : 'Ready';

      setAudioState(normalizedAudioState);
      setLastAnnouncedObject(snapshot.lastAnnouncedLabel ?? 'None');
    });

    speechService.start();

    return () => {
      // Dispose speech engine state to avoid background playback leaks.
      speechService.setDebugListener(undefined);
      void speechService.dispose();
    };
  }, []);

  /**
   * Start continuous frame capture
   */
  const startCapture = useCallback(() => {
    if (captureIntervalRef.current || isCapturingRef.current) return;
    
    isCapturingRef.current = true;
    hasFirstPredictionRef.current = false;
    predictionCountRef.current = 0;
    setPredictionCount(0);
    setIsCapturing(true);
    setDebugStatus('Starting capture...');
    
    // Use recursive timeout instead of setInterval for proper async handling
    const captureLoop = async (): Promise<void> => {
      if (!isCapturingRef.current) {
        captureIntervalRef.current = null;
        return;
      }

      const loopStart = Date.now();

      try {
        await captureFrame();
      } catch (error: any) {
        console.error('[ActiveCamera] Capture loop error:', error?.message || error);
        setDebugStatus(`Loop error: ${error?.message || 'unknown'}`);
      } finally {
        if (!isCapturingRef.current) {
          captureIntervalRef.current = null;
          return;
        }

        const elapsed = Date.now() - loopStart;
        const captureInterval = 1000 / REALISTIC_CAPTURE_FPS;
        const delay = Math.max(0, captureInterval - elapsed);

        captureIntervalRef.current = setTimeout(() => {
          void captureLoop();
        }, delay) as any;
      }
    };
    
    // Start the loop
    void captureLoop();
  // captureFrame is invoked from the timed loop and guarded by refs.
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  /**
   * Start continuous frame capture when permission is granted
   * Start regardless of model status (demo mode works too)
   */
  useEffect(() => {
    if (permission?.granted && isCameraReady && isPictureSizeConfigured) {
      const timer = setTimeout(() => {
        startCapture();
      }, 500);
      return () => clearTimeout(timer);
    }
  }, [permission?.granted, isCameraReady, isPictureSizeConfigured, startCapture]);

  /**
   * Stop frame capture
   */
  const stopCapture = useCallback(() => {
    if (!isCapturingRef.current && !captureIntervalRef.current) return; // already stopped
    isCapturingRef.current = false;
    hasFirstPredictionRef.current = false;
    if (captureIntervalRef.current) {
      clearTimeout(captureIntervalRef.current);
      captureIntervalRef.current = null;
    }
    setIsCapturing(false);
    setDebugStatus('Capture stopped');
  }, []);

  /**
   * Capture a single frame from camera
   * Uses takePictureAsync which is slow but works in Expo Go
   */
  const captureFrame = async () => {
    if (!cameraRef.current || !isCameraReadyRef.current || !isPictureSizeConfiguredRef.current) {
      return;
    }
    
    const startTime = Date.now();
    
    try {
      // Capture frame silently (no sound, no animation)
      const photo = await cameraRef.current.takePictureAsync({
        quality: 0.2,           // Low quality for faster capture
        base64: false,          // Avoid huge base64 payloads on JS thread
        skipProcessing: true,
        shutterSound: false,    // Disable shutter sound
      });
      
      if (!photo) {
        setDebugStatus('Capture failed - no photo');
        return;
      }
      
      const captureTime = Date.now() - startTime;
      setLastCaptureTime(captureTime);
      
      // Decode image to pixel data
      let frameData: FrameData;
      
      if (photo.uri) {
        try {
          // Native downscale first, then decode a tiny JPEG payload.
          const decoded = await decodeImageUriToPixels(photo.uri, FRAME_WIDTH, FRAME_HEIGHT);
          const sequenceId = ++captureSequenceRef.current;
          
          let intent = -1;
          let intentDistance = 0;
        
          if (useIntentPipeline && directionsCache && userLocation && routeProgress) {
            // Set intent for current step
            const currentStep = directionsCache.steps[currentStepIndex];
            if (currentStep) {
              intent = maneuverToIntent(currentStep.maneuver);
              intentDistance = routeProgress.distanceToStepEnd;
            }
          }

          frameData = {
            data: decoded.data,
            width: decoded.width,
            height: decoded.height,
            timestamp: Date.now(),
            sequenceId,
            intent,
            intentDistance
          };
        } catch (decodeError: any) {
          console.warn('[ActiveCamera] Failed to decode image:', decodeError?.message);
          setDebugStatus(`Decode error: ${decodeError?.message || 'invalid frame'}`);
          return;
        }
      } else if (photo.base64) {
        // Fallback path if URI is unavailable on a specific device/runtime.
        const decoded = await decodeBase64ToPixels(photo.base64, FRAME_WIDTH, FRAME_HEIGHT);
        const sequenceId = ++captureSequenceRef.current;

        let intent = -1;
        let intentDistance = 0;
      
        if (useIntentPipeline && directionsCache && userLocation && routeProgress) {
          // Set intent for current step
          const currentStep = directionsCache.steps[currentStepIndex];
          if (currentStep) {
            intent = maneuverToIntent(currentStep.maneuver);
            intentDistance = routeProgress.distanceToStepEnd;
          }
        }

        frameData = {
          data: decoded.data,
          width: decoded.width,
          height: decoded.height,
          timestamp: Date.now(),
          sequenceId,
          intent,
          intentDistance
        };
      } else {
        console.warn('[ActiveCamera] No image URI returned by camera');
        setDebugStatus('Capture error: missing image URI');
        return;
      }
      
      // Add frame to buffer
      const wasAdded = frameBufferRef.current.addFrame(frameData);
      
      if (wasAdded) {
        setFrameCount(prev => prev + 1);
        
        // Run inference when buffer is ready (or can predict early with padding)
        const buffer = frameBufferRef.current;
        if (isModelLoadedRef.current && buffer.canPredictEarly() && !isInferencingRef.current) {
          await runInferenceWithPadding();
        }
        
        // Run YOLO detection on this frame (parallel with ConvLSTM)
        if (isYOLOModelLoadedRef.current && !isYOLOInferencingRef.current) {
          await runYOLODetection(frameData);
        }
      }
    } catch (error: any) {
      console.error('[ActiveCamera] Frame capture error:', error?.message || error);
      setDebugStatus(`Error: ${error?.message || 'capture failed'}`);
    }
  };
  
  /**
   * Run YOLO object detection on single frame
   */
  const runYOLODetection = async (frame: FrameData) => {
    if (!isYOLOModelLoadedRef.current) {
      return;
    }

    if (isYOLOInferencingRef.current) {
      return; // Skip if already running
    }
    
    isYOLOInferencingRef.current = true;
    
    try {
      const result: YOLOResult = await detectObjects(frame);
      setYoloDetections(result.detections);
      setYoloInferenceTime(result.inferenceTimeMs);

      console.log(`[INFERENCE-DEBUG] Mode: [YOLO] | Detections: ${result.detections.length}.`);

      const closestDetection = getLargestAreaDetection(result.detections);
      if (closestDetection) {
        const closestArea = getDetectionArea(closestDetection);
        console.log(
          `[PRIORITY-DEBUG] Closest: [${closestDetection.className}] | Area/Prob: [${closestArea.toFixed(4)}].`
        );
      } else {
        console.log('[PRIORITY-DEBUG] Closest: [None] | Area/Prob: [0.0000].');
      }

      // Non-blocking announcement to avoid slowing camera processing.
      void objectSpeechServiceRef.current.announceDetections(result.detections).catch((error: any) => {
        console.warn('[ObjectSpeech] Announcement error:', error?.message || error);
      });
    } catch (error: any) {
      console.error('[YOLO] Detection error:', error?.message || error);
      setYoloDetections([]);
      setDebugStatus(`YOLO error: ${error?.message || 'inference failed'}`);
    } finally {
      isYOLOInferencingRef.current = false;
    }
  };

  /**
   * Run model inference with frame padding if buffer not full
   */
  const runInferenceWithPadding = async () => {
    if (!isModelLoadedRef.current) {
      return;
    }

    const buffer = frameBufferRef.current;
    
    if (isInferencingRef.current) {
      return;
    }
    
    isInferencingRef.current = true;
    setDebugStatus('Running inference...');
    
    try {
      // Before first prediction, duplicate each buffered frame in order
      // (1,1,2,2,...) to bootstrap sequence length faster.
      const isFirstPrediction = !hasFirstPredictionRef.current;
      const bufferedFrameCount = buffer.getFrameCount();
      const pipelineLabel = activePipelineName;
      console.log(
        `[CONVLSTM-TRACE] Start | Pipeline: [${pipelineLabel}] | Buffered Frames: ${bufferedFrameCount}/${SEQ_LEN} | Bootstrap: ${isFirstPrediction ? 'ON' : 'OFF'}.`
      );

      const frames = isFirstPrediction
        ? buffer.getFramesBootstrapDoubled()
        : buffer.getFramesPadded();

      if (frames.length !== SEQ_LEN) {
        throw new Error(`Invalid frame sequence length: ${frames.length}`);
      }
      
      // Preprocess frames
      const preprocessor = preprocessorRef.current;
      const tensor = preprocessor.preprocessFrameSequence(frames);
      console.log(
        `[CONVLSTM-TRACE] Tensor Ready | Frames Used: ${frames.length} | Shape: [1, ${SEQ_LEN}, ${preprocessorChannels}, ${FRAME_HEIGHT}, ${FRAME_WIDTH}] | Preprocess: ${tensor.processingTimeMs.toFixed(1)} ms.`
      );
      
      // Run prediction
      const { prediction, metrics: newMetrics } = await convlstmService.runPrediction(tensor);
      const priority = getConvLSTMPriority(prediction);
      const topProbabilities = formatConvLSTMTopProbabilities(prediction.probabilities);

      console.log(`[INFERENCE-DEBUG] Mode: [ConvLSTM] | Detections: ${prediction.probabilities.length}.`);
      console.log(
        `[PRIORITY-DEBUG] Closest: [${priority.label}] | Area/Prob: [${priority.probability.toFixed(4)}].`
      );
      console.log(
        `[CONVLSTM-TRACE] Output | Predicted: [${prediction.className}] | Confidence: [${prediction.confidence.toFixed(4)}] | Top: [${topProbabilities}].`
      );
      console.log(
        `[CONVLSTM-TRACE] Timing | Preprocess: ${newMetrics.preprocessingTimeMs.toFixed(1)} ms | Inference: ${newMetrics.inferenceTimeMs.toFixed(1)} ms | Total: ${newMetrics.totalLatencyMs.toFixed(1)} ms | FPS: ${newMetrics.fps.toFixed(2)}.`
      );
      
      // Update state
      setCurrentPrediction(prediction);
      setDirectionLabel(prediction.className);
      setConfidence(prediction.confidence);
      setMetrics(newMetrics);
      hasFirstPredictionRef.current = true;
      
      // Use functional update to avoid stale closure
      setPredictionCount(prev => {
        const newPredCount = prev + 1;
        predictionCountRef.current = newPredCount;
        setDebugStatus(`Prediction #${newPredCount}: ${prediction.className}`);
        return newPredCount;
      });
      
    } catch (error: any) {
      console.error('[ActiveCamera] Inference error:', error?.message || error);
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
    if (navigation.canGoBack && navigation.canGoBack()) {
      navigation.goBack();
    } else if (navigation.navigate) {
      navigation.navigate('MainMenu');
    }
  };

  if (!permission) {
    return (
      <SafeAreaView style={styles.container}>
        <Text style={styles.permissionText}>Requesting camera permission...</Text>
      </SafeAreaView>
    );
  }

  if (!permission.granted) {
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

  return (
    <View style={styles.container}>
      <StatusBar barStyle="light-content" backgroundColor="transparent" translucent />
      
      {/* Camera View - Silent capture mode (no children allowed) */}
      <CameraView
        ref={cameraRef}
        style={styles.camera}
        facing="back"
        mode="picture"
        pictureSize={cameraPictureSize}
        animateShutter={false}
        enableTorch={false}
        onCameraReady={handleCameraReady}
        onMountError={(error) => {
          console.error('[ActiveCamera] Mount error:', error);
          setDebugStatus(`Camera error: ${error.message}`);
        }}
      />
      
      {/* YOLO Bounding Boxes Overlay */}
      <BoundingBoxOverlay 
        detections={yoloDetections}
        containerWidth={Dimensions.get('window').width}
        containerHeight={Dimensions.get('window').height}
      />
      
      {/* Overlay Container - Absolute positioned on top of camera */}
      <View style={styles.overlayContainer}>
        {/* Debug Camera Status - Center of screen */}
        {!isCameraReady && (
          <View style={styles.cameraStatusOverlay}>
            <Text style={styles.cameraStatusText}>ðŸ“· Initializing Camera...</Text>
            <Text style={styles.cameraStatusSubtext}>Please wait</Text>
          </View>
        )}
        
        {/* Performance Overlay (Top-Left) */}
        <View style={styles.performanceOverlay}>
          <Text style={styles.performanceTitle}>Performance</Text>
          <Text style={styles.performanceText}>
            Mode: {modeLabel}
          </Text>
          <Text style={styles.performanceText}>
            ConvLSTM: {activePipelineName}
          </Text>
          {mode === 'destination' && destinationLabel ? (
            <Text style={styles.performanceText} numberOfLines={2}>
              Target: {destinationLabel}
            </Text>
          ) : null}
          {mode === 'destination' ? (
            <Text style={styles.performanceText}>
              Steps: {routeStepCount} | Dist: {formatDistanceForOverlay(currentDistance)}
            </Text>
          ) : null}
          <View style={styles.performanceDivider} />
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
          <Text style={styles.performanceText}>
            YOLO: {yoloInferenceTime.toFixed(0)} ms
          </Text>
          <View style={styles.performanceDivider} />
          <Text style={styles.performanceText}>
            Frames: {frameCount}
          </Text>
          <Text style={styles.performanceText}>
            Objects: {yoloDetections.length}
          </Text>
          <Text style={styles.performanceText}>
            Audio: {audioState}
          </Text>
          <Text style={styles.performanceText} numberOfLines={1}>
            Last Announced: {lastAnnouncedObject}
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
          <Text style={styles.backButtonText}>âœ•</Text>
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
          {(!isModelLoaded || !isYOLOModelLoaded) && (
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
    zIndex: 100, // UI elements on top of everything
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


