/**
 * Video Preprocessor for Mobile ConvLSTM Turn Prediction
 * 
 * TypeScript port of the Python preprocessor
 * Prepares video frames for inference on mobile devices
 * 
 * Key Features:
 * - Processes frames at 10 FPS sampling rate
 * - Resizes to 128x128
 * - Normalizes pixel values to [0, 1]
 * - Adds intent channels (3 additional channels, all zeros for 'no intent')
 * - Returns tensor shape: [seq_len, channels, height, width] = [20, 6, 128, 128]
 */

import {
  SEQ_LEN,
  FPS,
  FRAME_HEIGHT,
  FRAME_WIDTH,
  CHANNELS,
  DEVICE_CONFIG
} from '../config/modelConfig';

type TensorChannelCount = 3 | 6;

const resolveTensorChannelCount = (channels: number): TensorChannelCount => {
  if (channels === 3 || channels === 6) {
    return channels;
  }

  throw new Error('VideoPreprocessor channels must be 3 or 6');
};

/**
 * Frame data structure
 * Holds raw pixel data from camera capture
 */
export interface FrameData {
  data: Uint8Array;         // Raw pixel data (RGBA format from camera)
  width: number;            // Original frame width
  height: number;           // Original frame height
  timestamp: number;        // Capture timestamp in ms
  sequenceId?: number;      // Monotonic capture sequence ID for debugging/tracing
  intent?: number;          // Intent: 0 - Front, 1 - Left, 2 - Right
  intentDistance?: number;  // Distance before intent occurs
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
    return this.frames.length >= SEQ_LEN;
  }

  /**
   * Check if buffer has minimum frames for early prediction (with padding)
   * Early prediction available when at least 50% of required frames are collected
   */
  canPredictEarly(): boolean {
    const minFrames = Math.ceil(SEQ_LEN / 2);
    return this.frames.length >= minFrames;
  }

  /**
   * Get current frame count in buffer
   */
  getFrameCount(): number {
    return this.frames.length;
  }

  /**
   * Get the effective frame count used for UI progress.
   *
   * During early inference, the model input is synthesized to full length
   * using bootstrap doubling/padding. This helper reflects that readiness in
   * the UI while keeping physical buffer data unchanged.
   */
  getEffectiveFrameCount(): number {
    const physicalCount = this.frames.length;
    const requiredCount = this.config.maxFrames;

    if (physicalCount >= requiredCount) {
      return requiredCount;
    }

    const minFramesForEarlyPrediction = Math.ceil(requiredCount / 2);
    if (physicalCount >= minFramesForEarlyPrediction) {
      return Math.min(requiredCount, physicalCount * 2);
    }

    return physicalCount;
  }

  /**
   * Get UI-facing buffer progress values.
   */
  getDisplayProgress(): {
    physicalCount: number;
    effectiveCount: number;
    requiredCount: number;
  } {
    return {
      physicalCount: this.frames.length,
      effectiveCount: this.getEffectiveFrameCount(),
      requiredCount: this.config.maxFrames,
    };
  }

  /**
   * Get all frames in buffer
   * If buffer not full, duplicates last frame to reach SEQ_LEN
   */
  getFrames(): FrameData[] {
    return [...this.frames];
  }

  /**
   * Get frames padded to SEQ_LEN by duplicating the last frame
   * Used for early predictions before buffer is full
   */
  getFramesPadded(): FrameData[] {
    const frames = [...this.frames];
    
    // Pad with duplicate of last frame if needed
    while (frames.length < SEQ_LEN) {
      frames.push(frames[frames.length - 1]);
    }
    
    return frames;
  }

  /**
   * Get bootstrap frames for the first prediction.
   *
   * Strategy:
   * - Take up to the latest half-sequence of unique frames.
   * - Duplicate each frame in order: 1,1,2,2,3,3...
   * - If still short, pad using the last frame.
   */
  getFramesBootstrapDoubled(): FrameData[] {
    if (this.frames.length === 0) {
      return [];
    }

    const maxUniqueFrames = Math.ceil(SEQ_LEN / 2);
    const sourceFrames = this.frames.slice(-maxUniqueFrames);
    const doubled: FrameData[] = [];

    for (const frame of sourceFrames) {
      doubled.push(frame);
      doubled.push(frame);
    }

    while (doubled.length < SEQ_LEN) {
      doubled.push(doubled[doubled.length - 1]);
    }

    return doubled.slice(0, SEQ_LEN);
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
  private channels: TensorChannelCount;
  private injectIntentChannels: boolean;
  private framePlaneSize: number;
  private frameStride: number;
  private tensorBuffer: Float32Array;
  private outputShape: number[];

  constructor(
    height: number = FRAME_HEIGHT,
    width: number = FRAME_WIDTH,
    seqLen: number = SEQ_LEN,
    normalize: boolean = true,
    channels: number = CHANNELS,
    injectIntentChannels: boolean = false
  ) {
    this.height = height;
    this.width = width;
    this.seqLen = seqLen;
    this.normalize = normalize;
    this.channels = resolveTensorChannelCount(channels);
    this.injectIntentChannels = this.channels === 6 && injectIntentChannels;

    this.framePlaneSize = this.height * this.width;
    this.frameStride = this.channels * this.framePlaneSize;

    // Reused backing buffer: [1, seqLen, channels, height, width]
    this.tensorBuffer = new Float32Array(this.seqLen * this.frameStride);
    this.outputShape = [1, this.seqLen, this.channels, this.height, this.width];
  }

  /**
   * Preprocess a sequence of frames for model inference
   * 
   * Pipeline:
   * 1. Resize each frame to (height, width)
   * 2. Convert RGBA to RGB
   * 3. Normalize to [0, 1] if enabled
   * 4. Add intent channels (all zeros for 'no intent')
   * 5. Transpose to channels-first format
   * 6. Stack into sequence tensor
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
   * 5. Add intent channels (all zeros)
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

    // For 6-channel tensors, reset intent channels each frame and optionally populate
    // them from GPS-derived intent metadata.
    this.prepareIntentChannels(frame, frameOffset, tensorData);
    this.resizeNormalizeAndWriteFrame(frame, frameOffset, tensorData);
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

  private prepareIntentChannels(
    frame: FrameData,
    frameOffset: number,
    tensorData: Float32Array
  ): void {
    if (this.channels !== 6) {
      return;
    }

    const intentBaseOffset = frameOffset + this.framePlaneSize * 3;
    const intentRegionEnd = intentBaseOffset + this.framePlaneSize * 3;

    // Always clear intent planes first so no-intent mode stays strictly zeroed and
    // with-intent mode does not leak stale values from previous frames.
    tensorData.fill(0, intentBaseOffset, intentRegionEnd);

    if (!this.injectIntentChannels) {
      return;
    }

    const intentPlaneOffset = this.resolveIntentPlaneOffset(frame, frameOffset);
    tensorData.fill(1, intentPlaneOffset, intentPlaneOffset + this.framePlaneSize);
  }

  private resolveIntentPlaneOffset(frame: FrameData, frameOffset: number): number {
    const intentDistance =
      typeof frame.intentDistance === 'number' && Number.isFinite(frame.intentDistance)
        ? frame.intentDistance
        : Number.POSITIVE_INFINITY;
    const intentClass = frame.intent;

    if (intentDistance <= 5 && (intentClass === 0 || intentClass === 1 || intentClass === 2)) {
      return frameOffset + this.framePlaneSize * (intentClass + 3);
    }

    // Default intent plane is "front".
    return frameOffset + this.framePlaneSize * 3;
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
