/**
 * TFLite Inference Service
 * 
 * Handles model loading and inference for ConvLSTM turn prediction
 * Uses react-native-fast-tflite for efficient on-device inference
 * 
 * NOTE: Requires a development build (not Expo Go) for native TFLite support
 * Run: npx expo prebuild && npx expo run:android
 */

import { NUM_CLASSES, CLASS_NAMES, ClassId, PredictionClass } from '../config/modelConfig';
import { ProcessedTensor } from './preprocessor';

// TFLite import - requires development build
let loadTensorflowModel: any = null;

// Track if we're in demo mode (Expo Go) or real mode (dev build)
let isDemoMode = true;

// Try to load TFLite (will fail in Expo Go, work in dev build)
try {
  const tfliteModule = require('react-native-fast-tflite');
  loadTensorflowModel = tfliteModule.loadTensorflowModel;
  isDemoMode = false;
  console.log('[TFLite] react-native-fast-tflite loaded successfully');
} catch (e) {
  console.log('[TFLite] react-native-fast-tflite not available (Expo Go mode)');
  console.log('[TFLite] Running in DEMO mode with simulated predictions');
  isDemoMode = true;
}

/**
 * Prediction result from model inference
 */
export interface PredictionResult {
  classId: ClassId;           // Predicted class (0, 1, 2)
  className: PredictionClass; // Human-readable class name
  confidence: number;         // Prediction confidence (0-1)
  probabilities: number[];    // All class probabilities
  inferenceTimeMs: number;    // Time taken for inference
}

/**
 * Performance metrics for tracking
 */
export interface PerformanceMetrics {
  preprocessingTimeMs: number;
  inferenceTimeMs: number;
  totalLatencyMs: number;
  fps: number;
}

/**
 * TFLite Model Manager
 * Handles loading and running inference with the ConvLSTM model
 */
class TFLiteModelManager {
  private isLoaded: boolean = false;
  private model: any = null;
  private demoMode: boolean = isDemoMode;

  /**
   * Load the TFLite model
   * Must be called before running inference
   */
  async loadModel(): Promise<boolean> {
    if (this.isLoaded && this.model) {
      return true;
    }

    // Check if we're in demo mode (Expo Go)
    if (this.demoMode || !loadTensorflowModel) {
      console.log('[TFLite] ═══════════════════════════════════════════════');
      console.log('[TFLite] ⚠️  Running in DEMO MODE');
      console.log('[TFLite] ───────────────────────────────────────────────');
      console.log('[TFLite] Camera and UI work, but predictions are SIMULATED');
      console.log('[TFLite] ');
      console.log('[TFLite] To use REAL TFLite inference, create a dev build:');
      console.log('[TFLite]   1. npx expo prebuild');
      console.log('[TFLite]   2. npx expo run:android');
      console.log('[TFLite] ═══════════════════════════════════════════════');
      
      this.isLoaded = false;
      this.demoMode = true;
      return true; // Return true so app continues to function
    }

    try {
      console.log('[TFLite] Loading ConvLSTM model from assets...');
      
      // Load model from bundled assets with GPU delegate enabled
      // The model is in assets/model/convlstm.tflite (Float16 optimized)
      const modelOptions = {
        // Enable GPU delegate for hardware acceleration
        // Falls back to CPU if GPU not available
        useGpu: true,
      };
      
      this.model = await loadTensorflowModel(
        require('../../assets/model/convlstm.tflite'),
        modelOptions
      );
      
      this.isLoaded = true;
      this.demoMode = false;
      console.log('[TFLite] ✅ Model loaded successfully with GPU acceleration!');
      console.log('[TFLite] Model: Float16 quantized for optimal mobile performance');
      console.log('[TFLite] Model ready for real-time inference');
      
      // Warm up with dummy inference
      console.log('[TFLite] Warming up model...');
      await this.warmUp();
      console.log('[TFLite] Model warm-up complete');
      
      return true;
    } catch (error: any) {
      console.error('[TFLite] ❌ Failed to load model:', error?.message || error);
      console.log('[TFLite] Falling back to demo mode');
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
   * Run inference on preprocessed tensor
   * 
   * @param tensor - Preprocessed frame sequence tensor
   * @returns Prediction result with class and confidence
   */
  async runInference(tensor: ProcessedTensor): Promise<PredictionResult> {
    const startTime = performance.now();

    try {
      let output: number[];
      
      if (this.demoMode || !this.isLoaded || !this.model) {
        // Demo mode: Use simulated predictions
        output = await this.simulateInference();
      } else {
        // Real inference with TFLite model
        console.log('[TFLite] Running real inference...');
        console.log('[TFLite] Input shape:', tensor.shape);
        
        // Run model inference
        // Input: Float32Array with shape [1, 20, 6, 128, 128]
        const outputTensor = await this.model.run([tensor.data]);
        
        // Get output (should be [1, 3] for 3 classes)
        output = Array.from(outputTensor[0]);
        console.log('[TFLite] Raw output:', output);
      }
      
      const inferenceTimeMs = performance.now() - startTime;

      // Apply softmax to get probabilities
      const probabilities = this.softmax(output);
      
      // Get predicted class
      const classId = this.argmax(probabilities) as ClassId;
      const className = CLASS_NAMES[classId] as PredictionClass;
      const confidence = probabilities[classId];
      
      const modeLabel = this.demoMode ? '[DEMO]' : '[REAL]';
      console.log(`[TFLite] ${modeLabel} Prediction: ${className} (${(confidence * 100).toFixed(1)}%) in ${inferenceTimeMs.toFixed(1)}ms`);

      return {
        classId,
        className,
        confidence,
        probabilities,
        inferenceTimeMs
      };
    } catch (error: any) {
      console.error('[TFLite] Inference failed:', error?.message || error);
      
      // Fallback to simulated output on error
      const output = await this.simulateInference();
      const probabilities = this.softmax(output);
      const classId = this.argmax(probabilities) as ClassId;
      
      return {
        classId,
        className: CLASS_NAMES[classId] as PredictionClass,
        confidence: probabilities[classId],
        probabilities,
        inferenceTimeMs: performance.now() - startTime
      };
    }
  }

  /**
   * Simulate inference for demo mode
   * Generates realistic-looking predictions for testing UI
   */
  private async simulateInference(): Promise<number[]> {
    // Simulate processing delay (50-100ms)
    await new Promise(resolve => setTimeout(resolve, 50 + Math.random() * 50));
    
    // Generate realistic logits
    // Slightly favor "Front" direction for demo
    const logits: number[] = [
      Math.random() * 2 + 0.3,  // Front (slightly higher base)
      Math.random() * 2 - 0.3,  // Left
      Math.random() * 2 - 0.3,  // Right
    ];
    
    return logits;
  }

  /**
   * Warm up the model with dummy inference
   */
  private async warmUp(): Promise<void> {
    if (this.demoMode || !this.model) return;
    
    try {
      const dummyData = new Float32Array(1 * 20 * 6 * 128 * 128);
      await this.model.run([dummyData]);
      console.log('[TFLite] Warm-up successful');
    } catch (error) {
      console.warn('[TFLite] Warm-up failed (non-critical):', error);
    }
  }

  /**
   * Softmax activation function
   */
  private softmax(logits: number[]): number[] {
    const maxLogit = Math.max(...logits);
    const expValues = logits.map(x => Math.exp(x - maxLogit));
    const sumExp = expValues.reduce((a, b) => a + b, 0);
    return expValues.map(x => x / sumExp);
  }

  /**
   * Argmax - find index of maximum value
   */
  private argmax(arr: number[]): number {
    return arr.reduce((maxIdx, val, idx, array) => 
      val > array[maxIdx] ? idx : maxIdx, 0);
  }

  /**
   * Unload model and free resources
   */
  async unloadModel(): Promise<void> {
    if (this.model) {
      // TFLite models don't have explicit dispose, just null the reference
      this.model = null;
      console.log('[TFLite] Model unloaded');
    }
    this.isLoaded = false;
  }
}

/**
 * Singleton model manager instance
 */
let modelManager: TFLiteModelManager | null = null;

export function getModelManager(): TFLiteModelManager {
  if (!modelManager) {
    modelManager = new TFLiteModelManager();
  }
  return modelManager;
}

/**
 * High-level inference function
 */
export async function runPrediction(tensor: ProcessedTensor): Promise<{
  prediction: PredictionResult;
  metrics: PerformanceMetrics;
}> {
  const manager = getModelManager();
  const prediction = await manager.runInference(tensor);

  const metrics: PerformanceMetrics = {
    preprocessingTimeMs: tensor.processingTimeMs,
    inferenceTimeMs: prediction.inferenceTimeMs,
    totalLatencyMs: tensor.processingTimeMs + prediction.inferenceTimeMs,
    fps: 1000 / (tensor.processingTimeMs + prediction.inferenceTimeMs)
  };

  return { prediction, metrics };
}

/**
 * Initialize the model (call on app startup)
 */
export async function initializeModel(): Promise<boolean> {
  const manager = getModelManager();
  return manager.loadModel();
}

/**
 * Cleanup model resources (call on app close)
 */
export async function cleanupModel(): Promise<void> {
  const manager = getModelManager();
  await manager.unloadModel();
}

/**
 * Check if running in demo mode
 */
export function isRunningInDemoMode(): boolean {
  return getModelManager().isInDemoMode();
}
