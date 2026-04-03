/**
 * Video Preprocessor for Mobile ConvLSTM Turn Prediction
 * 
 * TypeScript port of the Python preprocessor
 * Prepares video frames for inference on mobile devices
 * 
 * Key Features:
 * - Processes frames using the configured sampling rate
 * - Resizes to 128x128
 * - Normalizes pixel values to [0, 1]
 * - Enforces an exact 10-frame FIFO sequence before preprocessing
 * - Returns tensor shape: [batch, seq_len, channels, height, width] = [1, 10, 3, 128, 128]
 */

import {
  SEQ_LEN,
  FPS,
  FRAME_HEIGHT,
  FRAME_WIDTH,
  CHANNELS,
  DEVICE_CONFIG
} from '../config/modelConfig';

/**
 * Frame data structure
 * Holds raw pixel data from camera capture
 */
export interface FrameData {
  data: Uint8Array;      // Raw pixel data (RGBA format from camera)
  width: number;         // Original frame width
  height: number;        // Original frame height
  timestamp: number;     // Capture timestamp in ms
}

/**
 * Processed tensor ready for model inference
 */
export interface ProcessedTensor {
  data: Float32Array;    // Flattened tensor data
  shape: number[];       // Tensor shape [batch, seq_len, channels, height, width]
  processingTimeMs: number; // Time taken to preprocess
}

/**
 * Frame buffer configuration
 */
export interface FrameBufferConfig {
  maxFrames: number;     // Maximum frames to buffer
  samplingRate: number;  // Frame sampling rate (take every Nth frame)
  cameraFps: number;     // Camera's native FPS
}

/**
 * Circular buffer for managing frame sequence
 */
export class FrameBuffer {
  private frames: FrameData[] = [];
  private config: FrameBufferConfig;
  private frameCount: number = 0;

  constructor(cameraFps: number = DEVICE_CONFIG.cameraFps) {
    this.config = {
      maxFrames: SEQ_LEN,
      samplingRate: Math.max(1, Math.floor(cameraFps / FPS)),
      cameraFps: cameraFps
    };
  }

  /**
   * Add a frame to the buffer (with automatic sampling)
   * Returns true if frame was added, false if skipped due to sampling
   */
  addFrame(frame: FrameData): boolean {
    this.frameCount++;
    
    // Sample frames based on camera FPS
    if ((this.frameCount - 1) % this.config.samplingRate !== 0) {
      return false; // Skip this frame
    }

    // Add frame to buffer
    this.frames.push(frame);
    
    // Remove oldest frame if buffer exceeds max size
    if (this.frames.length > this.config.maxFrames) {
      this.frames.shift();
    }

    return true;
  }

  /**
   * Check if buffer has enough frames for inference
   */
  isReady(): boolean {
    return this.frames.length === SEQ_LEN;
  }

  /**
   * Early prediction is intentionally disabled.
   * Inference must wait for a full sequence to avoid temporal inconsistency.
   */
  canPredictEarly(): boolean {
    return false;
  }

  /**
   * Get current frame count in buffer
   */
  getFrameCount(): number {
    return this.frames.length;
  }

  /**
   * Get all frames in buffer
   * Throws if the buffer is not yet ready for strict inference
   */
  getFrames(): FrameData[] {
    if (!this.isReady()) {
      throw new Error(`Frame buffer not ready: expected exactly ${SEQ_LEN} frames, got ${this.frames.length}`);
    }

    return [...this.frames];
  }

  /**
   * Deprecated strict-mode behavior.
   * Padding is not allowed for ConvLSTM inference in the mobile pipeline.
   */
  getFramesPadded(): FrameData[] {
    throw new Error('Padded frame retrieval is disabled. Wait for a full 10-frame sequence before inference.');
  }

  /**
   * Clear the buffer
   */
  clear(): void {
    this.frames = [];
    this.frameCount = 0;
  }

  /**
   * Get buffer status
   */
  getStatus(): { current: number; required: number; ready: boolean } {
    return {
      current: this.frames.length,
      required: SEQ_LEN,
      ready: this.isReady()
    };
  }
}

/**
 * Video Preprocessor Class
 * Handles frame preprocessing for ConvLSTM model inference
 */
export class VideoPreprocessor {
  private height: number;
  private width: number;
  private seqLen: number;
  private normalize: boolean;
  private framePlaneSize: number;
  private frameStride: number;
  private tensorBuffer: Float32Array;
  private outputShape: number[];

  constructor(
    height: number = FRAME_HEIGHT,
    width: number = FRAME_WIDTH,
    seqLen: number = SEQ_LEN,
    normalize: boolean = true
  ) {
    this.height = height;
    this.width = width;
    this.seqLen = seqLen;
    this.normalize = normalize;

    this.framePlaneSize = this.height * this.width;
    this.frameStride = CHANNELS * this.framePlaneSize;

    // Reused backing buffer: [1, seqLen, channels, height, width]
    this.tensorBuffer = new Float32Array(this.seqLen * this.frameStride);
    this.outputShape = [1, this.seqLen, CHANNELS, this.height, this.width];
  }

  /**
   * Preprocess a sequence of frames for model inference
   * 
   * Pipeline:
   * 1. Resize each frame to (height, width)
   * 2. Convert RGBA to RGB
   * 3. Normalize to [0, 1] if enabled
   * 4. Transpose to channels-first format
   * 5. Stack into sequence tensor
   * 
   * @param frames - Array of captured frames
   * @returns ProcessedTensor ready for model inference
   */
  preprocessFrameSequence(frames: FrameData[]): ProcessedTensor {
    const startTime = performance.now();

    if (frames.length !== this.seqLen) {
      throw new Error('Expected ' + this.seqLen + ' frames, got ' + frames.length);
    }

    // Process each frame
    for (let frameIdx = 0; frameIdx < this.seqLen; frameIdx++) {
      this.processFrame(frames[frameIdx], frameIdx, this.tensorBuffer);
    }

    const processingTimeMs = performance.now() - startTime;

    return {
      data: this.tensorBuffer,
      shape: this.outputShape,
      processingTimeMs
    };
  }

  /**
   * Process a single frame through the preprocessing pipeline
   * 
   * Steps:
   * 1. Resize to target dimensions
   * 2. Convert RGBA to RGB (camera captures RGBA)
   * 3. Normalize to [0, 1]
   * 4. Transpose to channels-first format
   */
  private processFrame(
    frame: FrameData,
    frameIdx: number,
    tensorData: Float32Array
  ): void {
    // Calculate offset in tensor for this frame
    // Tensor layout: [batch, seq, channels, height, width]
    // Batch index 0 is filled, so offset = frameIdx * channels * height * width.
    const frameOffset = frameIdx * this.frameStride;

    this.resizeNormalizeAndWriteFrame(frame, frameOffset, tensorData);

    // Tensor is RGB-only and writes channels 0-2 directly.
  }

  /**
   * Resize, normalize, and write directly to final NCHW RGB tensor slots.
   * Uses nearest-neighbor interpolation for lower JS compute cost.
   */
  private resizeNormalizeAndWriteFrame(
    frame: FrameData,
    frameOffset: number,
    tensorData: Float32Array
  ): void {
    const frameWidth = frame.width;
    const frameHeight = frame.height;
    const source = frame.data;

    const channel0Offset = frameOffset;
    const channel1Offset = frameOffset + this.framePlaneSize;
    const channel2Offset = frameOffset + this.framePlaneSize * 2;
    const maxSrcX = frameWidth - 1;
    const maxSrcY = frameHeight - 1;

    for (let y = 0; y < this.height; y++) {
      const rowOffset = y * this.width;
      // Nearest-neighbor row lookup: target y -> source y.
      let srcY = Math.floor((y * frameHeight) / this.height);
      if (srcY > maxSrcY) {
        srcY = maxSrcY;
      }
      const srcRowBase = srcY * frameWidth;

      for (let x = 0; x < this.width; x++) {
        // Nearest-neighbor column lookup: target x -> source x.
        let srcX = Math.floor((x * frameWidth) / this.width);
        if (srcX > maxSrcX) {
          srcX = maxSrcX;
        }

        const srcBase = (srcRowBase + srcX) * 4;

        const pixelOffset = rowOffset + x;

        // Extract RGB from RGBA source.
        const valueR = source[srcBase];
        const valueG = source[srcBase + 1];
        const valueB = source[srcBase + 2];

        if (this.normalize) {
          tensorData[channel0Offset + pixelOffset] = valueR / 255.0;
          tensorData[channel1Offset + pixelOffset] = valueG / 255.0;
          tensorData[channel2Offset + pixelOffset] = valueB / 255.0;
        } else {
          tensorData[channel0Offset + pixelOffset] = valueR;
          tensorData[channel1Offset + pixelOffset] = valueG;
          tensorData[channel2Offset + pixelOffset] = valueB;
        }
      }
    }
  }

  /**
   * Get expected output shape
   */
  getOutputShape(): number[] {
    return this.outputShape;
  }
}

/**
 * Singleton instance for easy access
 */
let preprocessorInstance: VideoPreprocessor | null = null;

export function getPreprocessor(): VideoPreprocessor {
  if (!preprocessorInstance) {
    preprocessorInstance = new VideoPreprocessor();
  }
  return preprocessorInstance;
}

/**
 * Convenience function for quick preprocessing
 */
export function preprocessFrames(frames: FrameData[]): ProcessedTensor {
  return getPreprocessor().preprocessFrameSequence(frames);
}
