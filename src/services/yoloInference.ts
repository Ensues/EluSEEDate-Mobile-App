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
let tfliteAvailabilityError: string | null = null;

// Try to load TFLite (will fail in Expo Go, work in dev build)
try {
  const tfliteModule = require('react-native-fast-tflite');
  loadTensorflowModel = tfliteModule.loadTensorflowModel;
  console.log('[YOLO-TFLite] react-native-fast-tflite loaded successfully');
} catch (e: any) {
  tfliteAvailabilityError = e?.message || 'react-native-fast-tflite not available';
  console.error('[YOLO-TFLite] react-native-fast-tflite unavailable:', tfliteAvailabilityError);
  console.error('[YOLO-TFLite] Use a development or standalone build with native modules enabled');
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
  private loadError: string | null = tfliteAvailabilityError;
  private confidenceThreshold: number = 0.35; // Minimum confidence to report detection

  /**
   * Load the YOLO TFLite model
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
      console.error('[YOLO-TFLite] Model loader unavailable:', this.loadError);
      return false;
    }

    try {
      console.log('[YOLO-TFLite] Loading YOLOv12 model from assets...');
      
      // Load model from bundled assets with GPU delegate enabled
      const modelOptions = {
        useGpu: true, // Enable GPU acceleration
      };

      this.model = await loadTensorflowModel(
        require('../../assets/model/yolo.tflite'),
        modelOptions
      );

      this.isLoaded = true;
      this.loadError = null;
      console.log('[YOLO-TFLite] ✅ Model loaded successfully with GPU acceleration!');
      console.log('[YOLO-TFLite] YOLOv12 ready for real-time object detection');

      // Warm up with dummy inference
      console.log('[YOLO-TFLite] Warming up model...');
      await this.warmUp();
      console.log('[YOLO-TFLite] Model warm-up complete');

      return true;
    } catch (error: any) {
      this.loadError = error?.message || 'Failed to load YOLO model';
      this.model = null;
      this.isLoaded = false;
      console.error('[YOLO-TFLite] ❌ Failed to load model:', this.loadError);
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
      if (!this.isLoaded || !this.model) {
        throw new Error(this.loadError || 'YOLO model is not loaded');
      }

      // Real inference with TFLite model
      console.log('[YOLO-TFLite] Running real inference...');

      // Preprocess frame for YOLO (resize to model input size, normalize)
      const preprocessed = this.preprocessFrame(frame);
      console.log('[YOLO-TFLite] Preprocessed data shape:', preprocessed.data.length, 'expected:', 1*3*128*128);

      // Run model inference
      const outputTensor = await this.model.run([preprocessed.data]);
      console.log('[YOLO-TFLite] Model inference complete, output:', typeof outputTensor);

      // Parse YOLO output (format depends on the active YOLOv12 export).
      const detections = this.parseYOLOOutput(outputTensor, frame.width, frame.height);

      console.log('[YOLO-TFLite] Detected', detections.length, 'objects');
      
      const inferenceTimeMs = performance.now() - startTime;

      console.log(`[YOLO-TFLite] Detections: ${detections.length} objects in ${inferenceTimeMs.toFixed(1)}ms`);

      return {
        detections,
        inferenceTimeMs,
        frameWidth: frame.width,
        frameHeight: frame.height
      };
    } catch (error: any) {
      console.error('[YOLO-TFLite] Inference failed:', error?.message || error);
      throw error;
    }
  }

  /**
   * Preprocess frame for YOLO input
   * Input shape: (1, 3, 128, 128) BCHW format
   * Converts RGBA camera data to RGB, resizes, and normalizes to [0, 1]
   */
  private preprocessFrame(frame: FrameData): { data: Float32Array; width: number; height: number } {
    const inputSize = 128;
    const channels = 3;
    
    // Output tensor in BCHW format: (Batch, Channels, Height, Width)
    const data = new Float32Array(1 * channels * inputSize * inputSize);
    
    // Calculate scaling factors
    const scaleX = frame.width / inputSize;
    const scaleY = frame.height / inputSize;
    
    // Resize and convert RGBA to RGB in BCHW format
    // Layout: [R channel (all pixels), G channel (all pixels), B channel (all pixels)]
    for (let y = 0; y < inputSize; y++) {
      for (let x = 0; x < inputSize; x++) {
        // Map to original frame coordinates (nearest neighbor)
        const srcX = Math.min(Math.floor(x * scaleX), frame.width - 1);
        const srcY = Math.min(Math.floor(y * scaleY), frame.height - 1);
        const srcIdx = (srcY * frame.width + srcX) * 4; // RGBA = 4 bytes per pixel
        
        // Output index in BCHW format
        const dstIdx = y * inputSize + x;
        
        // Extract and normalize RGB values (0-255 -> 0-1)
        const r = frame.data[srcIdx] / 255.0;
        const g = frame.data[srcIdx + 1] / 255.0;
        const b = frame.data[srcIdx + 2] / 255.0;
        
        // Store in BCHW format: [all R values, all G values, all B values]
        data[0 * inputSize * inputSize + dstIdx] = r; // R channel
        data[1 * inputSize * inputSize + dstIdx] = g; // G channel
        data[2 * inputSize * inputSize + dstIdx] = b; // B channel
      }
    }
    
    return {
      data,
      width: inputSize,
      height: inputSize
    };
  }

  /**
   * Parse YOLO model output into detection objects
   * Output shape: (1, 84, 336) where 84 = [4 bbox coords + 80 class scores]
   * 336 = number of potential detections from various anchor boxes/scales
   */
  private parseYOLOOutput(outputTensor: any, frameWidth: number, frameHeight: number): Detection[] {
    const detections: Detection[] = [];
    
    try {
      // Output shape: (1, 84, 336)
      // - 84 values per detection: [x, y, w, h, class_scores...80]
      // - 336 potential detections
      const numDetections = 336;
      const numClasses = 80;
      const bboxCoords = 4;
      
      // Access the output data (format depends on TFLite binding)
      const candidateOutput = Array.isArray(outputTensor) ? (outputTensor[0] ?? outputTensor) : outputTensor;
      const outputData = this.flattenNumericData(candidateOutput);
      
      // Detect tensor layout: (1, 84, 336) vs (1, 336, 84)
      const totalValues = outputData?.length || 0;
      const expectedValues = 84 * 336; // 28224
      
      // Infer tensor layout based on typical YOLO output patterns.
      // If length matches, determine whether layout is [84, 336] or [336, 84].
      let isTransposed = true; // Assume [84, 336] by default
      
      // Check whether the data resembles [336, 84] format.
      // In [336, 84] format, each detection uses 84 consecutive values.
      // Layout inference checks value-range continuity; bbox coordinates should exhibit similar channel behavior.
      if (totalValues >= 84 * 2) {
        const val0 = outputData[0];
        const val84 = outputData[84];
        const val1 = outputData[1];
        const val336 = outputData[336];
        
        // If values at stride=84 are more similar than values at stride=336,
        // it's likely [336, 84] format
        if (Math.abs(val0 - val84) < Math.abs(val0 - val336)) {
          // isTransposed = false;
          console.log('[YOLO-DEBUG] Detected non-transposed tensor format [336, 84]');
        } else {
          console.log('[YOLO-DEBUG] Using transposed tensor format [84, 336]');
        }
      }

      if (totalValues < expectedValues) {
        console.warn('[YOLO-TFLite] Output tensor shorter than expected:', totalValues, 'expected:', expectedValues);
        return [];
      }

      const scoreAccessor = (detIndex: number, classIndex: number): number => {
        if (isTransposed) {
          return outputData[(bboxCoords + classIndex) * numDetections + detIndex];
        }
        return outputData[detIndex * 84 + bboxCoords + classIndex];
      };

      // Auto-detect whether class scores are logits (need sigmoid) or probabilities.
      let minSampleScore = Number.POSITIVE_INFINITY;
      let maxSampleScore = Number.NEGATIVE_INFINITY;
      const sampleDetections = Math.min(24, numDetections);

      for (let i = 0; i < sampleDetections; i++) {
        for (let c = 0; c < numClasses; c++) {
          const score = scoreAccessor(i, c);
          if (!Number.isFinite(score)) continue;
          if (score < minSampleScore) minSampleScore = score;
          if (score > maxSampleScore) maxSampleScore = score;
        }
      }

      const scoresLookLikeLogits = minSampleScore < 0 || maxSampleScore > 1;
      if (scoresLookLikeLogits) {
        console.log('[YOLO-DEBUG] Class scores look like logits, applying sigmoid');
      }
      
      // DEBUG: Log tensor info on first few calls
      if (Math.random() < 0.1) { // 10% sampling to avoid spam
        console.log('[YOLO-DEBUG] Output tensor type:', typeof outputData);
        console.log('[YOLO-DEBUG] Output tensor length:', totalValues, '(expected:', expectedValues, ')');
        console.log('[YOLO-DEBUG] Sample values [0-5]:', JSON.stringify({
          "0": outputData[0],
          "1": outputData[1],
          "2": outputData[2],
          "3": outputData[3],
          "4": outputData[4]
        }));
      }
      
      // Sample and log a few detections for debugging
      if (Math.random() < 0.05) {
        console.log('[YOLO-DEBUG] Sample detection data:');
        for (let i = 0; i < Math.min(3, numDetections); i++) {
          let x, y, w, h, firstClassScore;
          
          if (isTransposed) {
            x = outputData[0 * numDetections + i];
            y = outputData[1 * numDetections + i];
            w = outputData[2 * numDetections + i];
            h = outputData[3 * numDetections + i];
            firstClassScore = outputData[4 * numDetections + i];
          } else {
            x = outputData[i * 84 + 0];
            y = outputData[i * 84 + 1];
            w = outputData[i * 84 + 2];
            h = outputData[i * 84 + 3];
            firstClassScore = outputData[i * 84 + 4];
          }
          
          console.log(`  Detection ${i}: x=${x.toFixed(3)}, y=${y.toFixed(3)}, w=${w.toFixed(3)}, h=${h.toFixed(3)}, first_class=${firstClassScore.toFixed(3)}`);
        }
      }
      
      // Parse each detection
      let maxConfidenceFound = 0;
      let detectionCandidates = 0;
      
      for (let i = 0; i < numDetections; i++) {
        // Extract bbox coordinates (x, y, w, h)
        // Handle both transposed [84, 336] and non-transposed [336, 84] layouts
        let x, y, w, h;
        
        if (isTransposed) {
          // Transposed format: [84, 336] - each channel is contiguous
          x = outputData[0 * numDetections + i];      // Center X
          y = outputData[1 * numDetections + i];      // Center Y
          w = outputData[2 * numDetections + i];      // Width
          h = outputData[3 * numDetections + i];      // Height
        } else {
          // Non-transposed format: [336, 84] - each detection is contiguous
          x = outputData[i * 84 + 0];
          y = outputData[i * 84 + 1];
          w = outputData[i * 84 + 2];
          h = outputData[i * 84 + 3];
        }

        if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(w) || !Number.isFinite(h)) {
          continue;
        }
        
        // Find class with highest confidence
        let maxClassScore = -Infinity;
        let maxClassId = 0;
        
        for (let c = 0; c < numClasses; c++) {
          const classScore = scoreAccessor(i, c);
          if (!Number.isFinite(classScore)) {
            continue;
          }
          
          if (classScore > maxClassScore) {
            maxClassScore = classScore;
            maxClassId = c;
          }
        }

        if (!Number.isFinite(maxClassScore)) {
          continue;
        }

        const score = scoresLookLikeLogits ? this.sigmoid(maxClassScore) : maxClassScore;
        const confidence = Math.max(0, Math.min(1, score));
        
        if (confidence > maxConfidenceFound) {
          maxConfidenceFound = confidence;
        }
        
        if (confidence >= 0.25) { // Count candidates at lower threshold for debugging
          detectionCandidates++;
        }
        
        // Filter by confidence threshold (use actual threshold, not temp one)
        if (confidence >= this.confidenceThreshold) {
          const outputScale = Math.max(Math.abs(x), Math.abs(y), Math.abs(w), Math.abs(h)) > 2
            ? 128
            : 1;

          const xNorm = x / outputScale;
          const yNorm = y / outputScale;
          const wNorm = w / outputScale;
          const hNorm = h / outputScale;

          // Convert from center format (x, y, w, h) to corner format (x, y, width, height)
          const boxX = Math.max(0, Math.min(1, xNorm - wNorm / 2)); // Top-left X
          const boxY = Math.max(0, Math.min(1, yNorm - hNorm / 2)); // Top-left Y
          const boxW = Math.max(0, Math.min(1, wNorm));             // Width
          const boxH = Math.max(0, Math.min(1, hNorm));             // Height
          
          // Filter out invalid or tiny boxes (likely false positives)
          const MIN_BOX_SIZE = 0.01; // Minimum 1% of image size
          if (boxW > MIN_BOX_SIZE && boxH > MIN_BOX_SIZE && 
              boxX >= 0 && boxY >= 0 && 
              (boxX + boxW) <= 1 && (boxY + boxH) <= 1) {
            
            detections.push({
              classId: maxClassId,
              className: YOLO_CLASS_NAMES[maxClassId] || `class_${maxClassId}`,
              confidence: confidence,
              boundingBox: {
                x: boxX,
                y: boxY,
                width: boxW,
                height: boxH
              }
            });
          }
        }
      }
      
      // Apply Non-Maximum Suppression (NMS) to remove overlapping boxes
      const nmsDetections = this.applyNMS(detections, 0.45); // IoU threshold = 0.45
      
      // DEBUG: Log detection stats
      console.log(`[YOLO-DEBUG] Pre-NMS detections: ${detections.length}, Post-NMS: ${nmsDetections.length}, Max confidence: ${maxConfidenceFound.toFixed(4)}, Candidates@0.25: ${detectionCandidates}`);
      
      // DEBUG: Check for suspicious identical confidence scores
      if (nmsDetections.length > 5) {
        const confidences = nmsDetections.map(d => d.confidence.toFixed(2));
        const uniqueConfidences = new Set(confidences);
        if (uniqueConfidences.size < nmsDetections.length / 2) {
          console.log('[YOLO-WARNING] Many detections have identical confidence scores!');
          console.log('[YOLO-WARNING] This suggests the model output may not be parsed correctly');
          console.log('[YOLO-WARNING] Unique confidences:', Array.from(uniqueConfidences).join(', '));
        }
      }
      
      return nmsDetections;
    } catch (error: any) {
      console.error('[YOLO-TFLite] Error parsing output:', error?.message || error);
      return [];
    }
  }

  private flattenNumericData(value: any): number[] {
    if (value == null) {
      return [];
    }

    if (typeof value === 'number') {
      return [value];
    }

    if (ArrayBuffer.isView(value)) {
      return Array.from(value as unknown as ArrayLike<number>);
    }

    if (Array.isArray(value)) {
      const flattened: number[] = [];
      for (const item of value) {
        flattened.push(...this.flattenNumericData(item));
      }
      return flattened;
    }

    return [];
  }

  private sigmoid(value: number): number {
    const clamped = Math.max(-60, Math.min(60, value));
    return 1 / (1 + Math.exp(-clamped));
  }

  /**
   * Calculate Intersection over Union (IoU) between two bounding boxes
   */
  private calculateIoU(boxA: BoundingBox, boxB: BoundingBox): number {
    // Calculate intersection area
    const xA = Math.max(boxA.x, boxB.x);
    const yA = Math.max(boxA.y, boxB.y);
    const xB = Math.min(boxA.x + boxA.width, boxB.x + boxB.width);
    const yB = Math.min(boxA.y + boxA.height, boxB.y + boxB.height);
    
    const intersectionWidth = Math.max(0, xB - xA);
    const intersectionHeight = Math.max(0, yB - yA);
    const intersectionArea = intersectionWidth * intersectionHeight;
    
    // Calculate union area
    const boxAArea = boxA.width * boxA.height;
    const boxBArea = boxB.width * boxB.height;
    const unionArea = boxAArea + boxBArea - intersectionArea;
    
    // Return IoU
    return unionArea > 0 ? intersectionArea / unionArea : 0;
  }

  /**
   * Apply Non-Maximum Suppression (NMS) to remove overlapping detections
   */
  private applyNMS(detections: Detection[], iouThreshold: number): Detection[] {
    if (detections.length === 0) return [];
    
    // Sort detections by confidence (highest first)
    const sorted = [...detections].sort((a, b) => b.confidence - a.confidence);
    
    const keep: Detection[] = [];
    const suppressed = new Set<number>();
    
    for (let i = 0; i < sorted.length; i++) {
      if (suppressed.has(i)) continue;
      
      const currentBox = sorted[i];
      keep.push(currentBox);
      
      // Suppress overlapping boxes of the same class
      for (let j = i + 1; j < sorted.length; j++) {
        if (suppressed.has(j)) continue;
        
        const compareBox = sorted[j];
        
        // Only compare boxes of the same class
        if (currentBox.classId === compareBox.classId) {
          const iou = this.calculateIoU(currentBox.boundingBox, compareBox.boundingBox);
          
          if (iou > iouThreshold) {
            suppressed.add(j);
          }
        }
      }
    }
    
    return keep;
  }

  /**
   * Warm up the model with dummy inference
   */
  private async warmUp(): Promise<void> {
    if (!this.model) return;
    
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
 * Backwards-compatible demo mode check.
 * Returns true when real YOLO inference is unavailable.
 */
export function isYOLOInDemoMode(): boolean {
  return !getYOLOModelManager().isModelLoaded();
}

/**
 * Get the latest YOLO model load error, if any.
 */
export function getYOLOModelLoadError(): string | null {
  return getYOLOModelManager().getLoadError();
}

/**
 * Set YOLO confidence threshold
 */
export function setYOLOConfidenceThreshold(threshold: number): void {
  getYOLOModelManager().setConfidenceThreshold(threshold);
}
