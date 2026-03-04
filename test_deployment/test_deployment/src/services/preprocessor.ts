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
  INTENT_FRAMES,
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
      throw new Error(`Expected ${this.seqLen} frames, got ${frames.length}`);
    }

    // Output tensor shape: [1, seq_len, channels, height, width]
    // = [1, 20, 6, 128, 128]
    const batchSize = 1;
    const tensorSize = batchSize * this.seqLen * CHANNELS * this.height * this.width;
    const tensorData = new Float32Array(tensorSize);

    // Process each frame
    for (let frameIdx = 0; frameIdx < this.seqLen; frameIdx++) {
      this.processFrame(frames[frameIdx], frameIdx, tensorData);
    }

    const processingTimeMs = performance.now() - startTime;

    return {
      data: tensorData,
      shape: [batchSize, this.seqLen, CHANNELS, this.height, this.width],
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
    // Get resized RGB data
    const resizedRgb = this.resizeAndConvertFrame(frame);

    // Calculate offset in tensor for this frame
    // Tensor layout: [batch, seq, channels, height, width]
    // We're filling batch 0, so offset = frameIdx * channels * height * width
    const frameOffset = frameIdx * CHANNELS * this.height * this.width;

    // Fill RGB channels (channels 0, 1, 2)
    for (let c = 0; c < 3; c++) {
      const channelOffset = frameOffset + c * this.height * this.width;
      for (let h = 0; h < this.height; h++) {
        for (let w = 0; w < this.width; w++) {
          const pixelIdx = (h * this.width + w) * 3 + c;
          const tensorIdx = channelOffset + h * this.width + w;
          tensorData[tensorIdx] = resizedRgb[pixelIdx];
        }
      }
    }

    // Fill intent channels (channels 3, 4, 5) with zeros (no intent)
    for (let c = 3; c < 6; c++) {
      const channelOffset = frameOffset + c * this.height * this.width;
      for (let h = 0; h < this.height; h++) {
        for (let w = 0; w < this.width; w++) {
          const tensorIdx = channelOffset + h * this.width + w;
          tensorData[tensorIdx] = 0.0; // No intent
        }
      }
    }
  }

  /**
   * Resize frame to target dimensions and convert RGBA to normalized RGB
   * Uses bilinear interpolation for smooth resizing
   */
  private resizeAndConvertFrame(frame: FrameData): Float32Array {
    const outputSize = this.height * this.width * 3;
    const output = new Float32Array(outputSize);

    const scaleX = frame.width / this.width;
    const scaleY = frame.height / this.height;

    for (let y = 0; y < this.height; y++) {
      for (let x = 0; x < this.width; x++) {
        // Calculate source coordinates (bilinear interpolation)
        const srcX = x * scaleX;
        const srcY = y * scaleY;

        // Get integer and fractional parts
        const x0 = Math.floor(srcX);
        const y0 = Math.floor(srcY);
        const x1 = Math.min(x0 + 1, frame.width - 1);
        const y1 = Math.min(y0 + 1, frame.height - 1);
        
        const xFrac = srcX - x0;
        const yFrac = srcY - y0;

        // Bilinear interpolation for each RGB channel
        for (let c = 0; c < 3; c++) {
          // Source pixels (RGBA format, 4 bytes per pixel)
          const idx00 = (y0 * frame.width + x0) * 4 + c;
          const idx01 = (y0 * frame.width + x1) * 4 + c;
          const idx10 = (y1 * frame.width + x0) * 4 + c;
          const idx11 = (y1 * frame.width + x1) * 4 + c;

          // Get pixel values
          const p00 = frame.data[idx00];
          const p01 = frame.data[idx01];
          const p10 = frame.data[idx10];
          const p11 = frame.data[idx11];

          // Interpolate
          const top = p00 * (1 - xFrac) + p01 * xFrac;
          const bottom = p10 * (1 - xFrac) + p11 * xFrac;
          let value = top * (1 - yFrac) + bottom * yFrac;

          // Normalize to [0, 1] if enabled
          if (this.normalize) {
            value = value / 255.0;
          }

          // Store in output (RGB format, 3 values per pixel)
          const outIdx = (y * this.width + x) * 3 + c;
          output[outIdx] = value;
        }
      }
    }

    return output;
  }

  /**
   * Get expected output shape
   */
  getOutputShape(): number[] {
    return [1, this.seqLen, CHANNELS, this.height, this.width];
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
