/**
 * YOLO Inference Service
 * 
 * Handles model loading and inference for YOLOv12 object detection
 * Uses react-native-fast-tflite for efficient on-device inference
 * 
 * NOTE: Requires a development build (not Expo Go) for native TFLite support
 * Run: npx expo prebuild && npx expo run:android
 */

import { YOLO_NUM_CLASSES, YOLO_CLASS_NAMES } from '../config/modelConfig';
import { FrameData } from './preprocessor';

// TFLite import - requires development build
let loadTensorflowModel: any = null;

// Track if we're in demo mode (Expo Go) or real mode (dev build)
let isDemoMode = true;

// Try to load TFLite (will fail in Expo Go, work in dev build)
try {
  const tfliteModule = require('react-native-fast-tflite');
  loadTensorflowModel = tfliteModule.loadTensorflowModel;
  isDemoMode = false;
  console.log('[YOLO-TFLite] react-native-fast-tflite loaded successfully');
} catch (e) {
  console.log('[YOLO-TFLite] react-native-fast-tflite not available (Expo Go mode)');
  console.log('[YOLO-TFLite] Running in DEMO mode with simulated detections');
  isDemoMode = true;
}

/**
 * Bounding box for detected object
 */
export interface BoundingBox {
  x: number;      // Top-left x coordinate (normalized 0-1)
  y: number;      // Top-left y coordinate (normalized 0-1)
  width: number;  // Width (normalized 0-1)
  height: number; // Height (normalized 0-1)
}

/**
 * Single object detection result
 */
export interface Detection {
  classId: number;           // YOLO class ID
  className: string;         // Human-readable class name
  confidence: number;        // Detection confidence (0-1)
  boundingBox: BoundingBox;  // Object bounding box
}

/**
 * YOLO detection result from model inference
 */
export interface YOLOResult {
  detections: Detection[];    // List of detected objects
  inferenceTimeMs: number;    // Time taken for inference
  frameWidth: number;         // Input frame width
  frameHeight: number;        // Input frame height
}

/**
 * YOLO Model Manager
 * Handles loading and running inference with the YOLOv12 model
 */
class YOLOModelManager {
  private isLoaded: boolean = false;
  private model: any = null;
  private demoMode: boolean = isDemoMode;
  private confidenceThreshold: number = 0.5; // Minimum confidence to report detection

  /**
   * Load the YOLO TFLite model
   * Must be called before running inference
   */
  async loadModel(): Promise<boolean> {
    if (this.isLoaded && this.model) {
      return true;
    }

    // Check if we're in demo mode (Expo Go)
    if (this.demoMode || !loadTensorflowModel) {
      console.log('[YOLO-TFLite] ═══════════════════════════════════════════════');
      console.log('[YOLO-TFLite] ⚠️  Running in DEMO MODE');
      console.log('[YOLO-TFLite] ───────────────────────────────────────────────');
      console.log('[YOLO-TFLite] Object detection is SIMULATED');
      console.log('[YOLO-TFLite] ');
      console.log('[YOLO-TFLite] To use REAL YOLO inference:');
      console.log('[YOLO-TFLite]   1. Replace assets/model/yolo-placeholder.txt');
      console.log('[YOLO-TFLite]   2. With your YOLOv12 .tflite model');
      console.log('[YOLO-TFLite]   3. npx expo prebuild && npx expo run:android');
      console.log('[YOLO-TFLite] ═══════════════════════════════════════════════');
      
      this.isLoaded = false;
      this.demoMode = true;
      return true; // Return true so app continues to function
    }

    try {
      console.log('[YOLO-TFLite] Loading YOLOv12 model from assets...');
      
      // Load model from bundled assets with GPU delegate enabled
      // PLACEHOLDER: Replace with actual model when available
      const modelOptions = {
        useGpu: true, // Enable GPU acceleration
      };
      
      // NOTE: This will fail until real model is added
      // For now, fall back to demo mode
      try {
        this.model = await loadTensorflowModel(
          require('../../assets/model/yolo.tflite'),
          modelOptions
        );
        
        this.isLoaded = true;
        this.demoMode = false;
        console.log('[YOLO-TFLite] ✅ Model loaded successfully with GPU acceleration!');
        console.log('[YOLO-TFLite] YOLOv12 ready for real-time object detection');
        
        // Warm up with dummy inference
        console.log('[YOLO-TFLite] Warming up model...');
        await this.warmUp();
        console.log('[YOLO-TFLite] Model warm-up complete');
        
        return true;
      } catch (loadError: any) {
        console.log('[YOLO-TFLite] Model file not found (expected - using placeholder)');
        console.log('[YOLO-TFLite] Falling back to demo mode');
        this.demoMode = true;
        return true;
      }
    } catch (error: any) {
      console.error('[YOLO-TFLite] ❌ Failed to load model:', error?.message || error);
      console.log('[YOLO-TFLite] Falling back to demo mode');
      this.demoMode = true;
      return true; // Still allow app to run in demo mode
    }
  }

  /**
   * Check if model is loaded (real inference available)
   */
  isModelLoaded(): boolean {
    return this.isLoaded && !this.demoMode;
  }

  /**
   * Check if running in demo mode
   */
  isInDemoMode(): boolean {
    return this.demoMode;
  }

  /**
   * Set confidence threshold for detections
   */
  setConfidenceThreshold(threshold: number): void {
    this.confidenceThreshold = Math.max(0, Math.min(1, threshold));
  }

  /**
   * Run YOLO inference on a single frame
   * 
   * @param frame - Single frame data from camera
   * @returns YOLO detection result with bounding boxes
   */
  async runInference(frame: FrameData): Promise<YOLOResult> {
    const startTime = performance.now();

    try {
      let detections: Detection[];
      
      if (this.demoMode || !this.isLoaded || !this.model) {
        // Demo mode: Use simulated detections
        detections = await this.simulateDetections();
      } else {
        // Real inference with TFLite model
        console.log('[YOLO-TFLite] Running real inference...');
        
        // Preprocess frame for YOLO (resize to model input size, normalize)
        const preprocessed = this.preprocessFrame(frame);
        
        // Run model inference
        const outputTensor = await this.model.run([preprocessed.data]);
        
        // Parse YOLO output (format depends on your specific YOLOv12 model)
        detections = this.parseYOLOOutput(outputTensor, frame.width, frame.height);
        
        console.log('[YOLO-TFLite] Detected', detections.length, 'objects');
      }
      
      const inferenceTimeMs = performance.now() - startTime;

      const modeLabel = this.demoMode ? '[DEMO]' : '[REAL]';
      console.log(`[YOLO-TFLite] ${modeLabel} Detections: ${detections.length} objects in ${inferenceTimeMs.toFixed(1)}ms`);

      return {
        detections,
        inferenceTimeMs,
        frameWidth: frame.width,
        frameHeight: frame.height
      };
    } catch (error: any) {
      console.error('[YOLO-TFLite] Inference failed:', error?.message || error);
      
      // Fallback to empty detections on error
      return {
        detections: [],
        inferenceTimeMs: performance.now() - startTime,
        frameWidth: frame.width,
        frameHeight: frame.height
      };
    }
  }

  /**
   * Preprocess frame for YOLO input
   * Resize to model input size and normalize
   */
  private preprocessFrame(frame: FrameData): { data: Float32Array; width: number; height: number } {
    // PLACEHOLDER: Real preprocessing will depend on your YOLOv12 model requirements
    // For now, create dummy data matching expected input shape
    const inputSize = 128; // Will match ConvLSTM size, adjust when real model arrives
    const channels = 3; // RGB
    
    const data = new Float32Array(inputSize * inputSize * channels);
    
    // TODO: Implement actual frame resizing and normalization
    // This would involve:
    // 1. Resize frame.data from frame.width x frame.height to inputSize x inputSize
    // 2. Convert RGBA to RGB (drop alpha channel)
    // 3. Normalize pixel values (typically to [0,1] or [-1,1] depending on model)
    
    return {
      data,
      width: inputSize,
      height: inputSize
    };
  }

  /**
   * Parse YOLO model output into detection objects
   * NOTE: This is model-specific and will need adjustment for your YOLOv12 model
   */
  private parseYOLOOutput(outputTensor: any, frameWidth: number, frameHeight: number): Detection[] {
    // PLACEHOLDER: Real parsing depends on your specific YOLOv12 output format
    // Typical YOLO output: [batch, num_detections, (x, y, w, h, confidence, class_probs...)]
    
    const detections: Detection[] = [];
    
    // TODO: Implement actual YOLO output parsing
    // This would involve:
    // 1. Extract bounding boxes, confidence scores, and class probabilities
    // 2. Apply NMS (Non-Maximum Suppression) to remove duplicate detections
    // 3. Filter by confidence threshold
    // 4. Convert to Detection objects
    
    return detections;
  }

  /**
   * Simulate detections for demo mode
   * Generates realistic-looking object detections for testing UI
   */
  private async simulateDetections(): Promise<Detection[]> {
    // Simulate processing delay (30-80ms - YOLO is typically faster than ConvLSTM)
    await new Promise(resolve => setTimeout(resolve, 30 + Math.random() * 50));
    
    const detections: Detection[] = [];
    
    // Randomly generate 0-3 detections
    const numDetections = Math.floor(Math.random() * 4);
    
    for (let i = 0; i < numDetections; i++) {
      // Random class (common obstacle types)
      const commonClasses = ['person', 'car', 'bicycle', 'motorcycle', 'truck'];
      const className = commonClasses[Math.floor(Math.random() * commonClasses.length)];
      const classId = commonClasses.indexOf(className);
      
      // Random bounding box (normalized coordinates)
      const x = Math.random() * 0.6; // 0-0.6 (leave room for width)
      const y = Math.random() * 0.6; // 0-0.6 (leave room for height)
      const width = 0.1 + Math.random() * 0.3;  // 0.1-0.4
      const height = 0.1 + Math.random() * 0.3; // 0.1-0.4
      
      // Random confidence (higher for demo to show clearly)
      const confidence = 0.6 + Math.random() * 0.4; // 0.6-1.0
      
      detections.push({
        classId,
        className,
        confidence,
        boundingBox: { x, y, width, height }
      });
    }
    
    return detections;
  }

  /**
   * Warm up the model with dummy inference
   */
  private async warmUp(): Promise<void> {
    if (this.demoMode || !this.model) return;
    
    try {
      const dummyFrame: FrameData = {
        data: new Uint8Array(128 * 128 * 4),
        width: 128,
        height: 128,
        timestamp: Date.now()
      };
      
      await this.runInference(dummyFrame);
      console.log('[YOLO-TFLite] Warm-up successful');
    } catch (error) {
      console.warn('[YOLO-TFLite] Warm-up failed (non-critical):', error);
    }
  }

  /**
   * Unload model and free resources
   */
  async unloadModel(): Promise<void> {
    if (this.model) {
      this.model = null;
      console.log('[YOLO-TFLite] Model unloaded');
    }
    this.isLoaded = false;
  }
}

/**
 * Singleton model manager instance
 */
let yoloModelManager: YOLOModelManager | null = null;

export function getYOLOModelManager(): YOLOModelManager {
  if (!yoloModelManager) {
    yoloModelManager = new YOLOModelManager();
  }
  return yoloModelManager;
}

/**
 * High-level YOLO detection function
 */
export async function runYOLODetection(frame: FrameData): Promise<YOLOResult> {
  const manager = getYOLOModelManager();
  return manager.runInference(frame);
}

/**
 * Initialize the YOLO model (call on app startup)
 */
export async function initializeYOLOModel(): Promise<boolean> {
  const manager = getYOLOModelManager();
  return manager.loadModel();
}

/**
 * Cleanup YOLO model resources (call on app close)
 */
export async function cleanupYOLOModel(): Promise<void> {
  const manager = getYOLOModelManager();
  await manager.unloadModel();
}

/**
 * Check if YOLO is running in demo mode
 */
export function isYOLOInDemoMode(): boolean {
  return getYOLOModelManager().isInDemoMode();
}

/**
 * Set YOLO confidence threshold
 */
export function setYOLOConfidenceThreshold(threshold: number): void {
  getYOLOModelManager().setConfidenceThreshold(threshold);
}
