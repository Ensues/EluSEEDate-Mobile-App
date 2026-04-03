/**
 * TFLite Inference Service - ConvLSTM Without Intent
 * 
 * Handles model loading and inference for ConvLSTM turn prediction
 * Accepts the configured preprocessed tensor shape from the mobile pipeline
 * Uses react-native-fast-tflite for efficient on-device inference
 * 
 * NOTE: Requires a development build (not Expo Go) for native TFLite support
 * Run: npx expo prebuild && npx expo run:android
 */

import {
  NUM_CLASSES,
  CLASS_NAMES,
  ClassId,
  PredictionClass,
  SEQ_LEN,
  CHANNELS,
  FRAME_HEIGHT,
  FRAME_WIDTH,
} from '../config/modelConfig';
import { ProcessedTensor } from './preprocessor';

// TFLite import - requires development build
let loadTensorflowModel: any = null;
let tfliteAvailabilityError: string | null = null;

// Try to load TFLite (will fail in Expo Go, work in dev build)
try {
  const tfliteModule = require('react-native-fast-tflite');
  loadTensorflowModel = tfliteModule.loadTensorflowModel;
  console.log('[ConvLSTM-TFLite] react-native-fast-tflite loaded successfully');
} catch (e: any) {
  tfliteAvailabilityError = e?.message || 'react-native-fast-tflite not available';
  console.error('[ConvLSTM-TFLite] react-native-fast-tflite unavailable:', tfliteAvailabilityError);
  console.error('[ConvLSTM-TFLite] Use a development or standalone build with native modules enabled');
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
  private loadError: string | null = tfliteAvailabilityError;
  private readonly expectedInputShape = [1, SEQ_LEN, CHANNELS, FRAME_HEIGHT, FRAME_WIDTH] as const;

  /**
   * Load the TFLite model
   * Must be called before running inference
   */
  async loadModel(): Promise<boolean> {
    if (this.isLoaded && this.model) {
      return true;
    }

    if (!loadTensorflowModel) {
      this.loadError = tfliteAvailabilityError || 'TFLite loader is unavailable';
      this.model = null;
      this.isLoaded = false;
      console.error('[ConvLSTM-TFLite] Model loader unavailable:', this.loadError);
      return false;
    }

    try {
      console.log('[ConvLSTM-TFLite] Loading ConvLSTM model from assets...');
      
      // Load model from bundled assets with GPU delegate enabled
      // The model is in assets/model/convlstm.tflite (float16 optimized)
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
      this.loadError = null;
      console.log('[ConvLSTM-TFLite] ✅ Model loaded successfully with GPU acceleration!');
      console.log('[ConvLSTM-TFLite] Model: float16 with Global Average Pooling');
      console.log('[ConvLSTM-TFLite] Model ready for real-time inference');
      
      // Warm up with dummy inference
      console.log('[ConvLSTM-TFLite] Warming up model...');
      await this.warmUp();
      console.log('[ConvLSTM-TFLite] Model warm-up complete');
      
      return true;
    } catch (error: any) {
      this.loadError = error?.message || 'Failed to load ConvLSTM model';
      this.model = null;
      this.isLoaded = false;
      console.error('[ConvLSTM-TFLite] ❌ Failed to load model:', this.loadError);
      return false;
    }
  }

  /**
   * Check if model is loaded (real inference available)
   */
  isModelLoaded(): boolean {
    return this.isLoaded && !!this.model;
  }

  /**
   * Get latest model load error
   */
  getLoadError(): string | null {
    return this.loadError;
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
      if (!this.isLoaded || !this.model) {
        throw new Error(this.loadError || 'ConvLSTM model is not loaded');
      }

      console.log('[ConvLSTM-TFLite] Running real inference...');
      console.log('[ConvLSTM-TFLite] Input shape:', tensor.shape);

      if (!this.isExpectedInputShape(tensor.shape)) {
        throw new Error(
          `[ConvLSTM-TFLite] Invalid tensor shape ${JSON.stringify(tensor.shape)}; expected ${JSON.stringify(this.expectedInputShape)}`
        );
      }

      // Input: Float32Array with shape [1, SEQ_LEN, CHANNELS, FRAME_HEIGHT, FRAME_WIDTH]
      const outputTensor = await this.model.run([tensor.data]);
      const output = this.extractOutputVector(outputTensor);
      console.log('[ConvLSTM-TFLite] Raw output:', output);
      
      const inferenceTimeMs = performance.now() - startTime;

      // Apply softmax to get probabilities
      const probabilities = this.softmax(output);
      
      // Get predicted class
      const classId = this.argmax(probabilities) as ClassId;
      const className = CLASS_NAMES[classId] as PredictionClass;
      const confidence = probabilities[classId];

      console.log(`[ConvLSTM-TFLite] Prediction: ${className} (${(confidence * 100).toFixed(1)}%) in ${inferenceTimeMs.toFixed(1)}ms`);

      return {
        classId,
        className,
        confidence,
        probabilities,
        inferenceTimeMs
      };
    } catch (error: any) {
      console.error('[ConvLSTM-TFLite] Inference failed:', error?.message || error);
      throw error;
    }
  }

  /**
   * Extract class logits from variable TFLite output layouts.
   */
  private extractOutputVector(outputTensor: any): number[] {
    const candidate = Array.isArray(outputTensor) ? outputTensor[0] : outputTensor;
    const raw = candidate != null && typeof candidate[Symbol.iterator] === 'function'
      ? Array.from(candidate as Iterable<number>)
      : [];

    const normalized = raw
      .slice(0, NUM_CLASSES)
      .map((value) => (Number.isFinite(value) ? Number(value) : 0));

    if (normalized.length < NUM_CLASSES) {
      while (normalized.length < NUM_CLASSES) {
        normalized.push(0);
      }
      console.warn('[ConvLSTM-TFLite] Output vector shorter than expected, padding with zeros');
    }

    return normalized;
  }

  /**
   * Validate model input tensor shape against current config constants.
   */
  private isExpectedInputShape(shape: number[]): boolean {
    return (
      shape.length === this.expectedInputShape.length &&
      shape.every((value, idx) => value === this.expectedInputShape[idx])
    );
  }

  /**
   * Warm up the model with dummy inference
   */
  private async warmUp(): Promise<void> {
    if (!this.model) return;
    
    try {
      const dummyData = new Float32Array(SEQ_LEN * CHANNELS * FRAME_HEIGHT * FRAME_WIDTH);
      await this.model.run([dummyData]);
      console.log('[ConvLSTM-TFLite] Warm-up successful');
    } catch (error) {
      console.warn('[ConvLSTM-TFLite] Warm-up failed (non-critical):', error);
    }
  }

  /**
   * Softmax activation function
   */
  private softmax(logits: number[]): number[] {
    const safeLogits = logits.map((x) => (Number.isFinite(x) ? x : 0));
    const maxLogit = Math.max(...safeLogits);
    const expValues = safeLogits.map(x => Math.exp(x - maxLogit));
    const sumExp = expValues.reduce((a, b) => a + b, 0);

    if (!Number.isFinite(sumExp) || sumExp <= 0) {
      const uniform = 1 / NUM_CLASSES;
      console.warn('[ConvLSTM-TFLite] Softmax encountered invalid sum; returning uniform probabilities');
      return new Array(NUM_CLASSES).fill(uniform);
    }

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
      console.log('[ConvLSTM-TFLite] Model unloaded');
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
 * Backwards-compatible demo mode check.
 * Returns true when real model inference is unavailable.
 */
export function isRunningInDemoMode(): boolean {
  return !getModelManager().isModelLoaded();
}

/**
 * Get the latest ConvLSTM model load error, if any.
 */
export function getModelLoadError(): string | null {
  return getModelManager().getLoadError();
}
