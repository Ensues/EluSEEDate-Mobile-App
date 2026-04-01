import { runAtTargetFps, useFrameProcessor } from 'react-native-vision-camera';
import { useResizePlugin } from 'vision-camera-resize-plugin';
import { Worklets } from 'react-native-worklets-core';
import { CHANNELS, FRAME_HEIGHT, FRAME_WIDTH, SEQ_LEN } from '../config/modelConfig';

const RGB_CHANNELS = 3;
const FRAME_PLANE_SIZE = FRAME_WIDTH * FRAME_HEIGHT;
const RGB_FRAME_SIZE = RGB_CHANNELS * FRAME_PLANE_SIZE;
const FRAME_STRIDE = CHANNELS * FRAME_PLANE_SIZE;
const TOTAL_TENSOR_SIZE = SEQ_LEN * FRAME_STRIDE;
const DEFAULT_TARGET_FPS = 20;

export const CONVLSTM_TENSOR_SHAPE: [number, number, number, number, number] = [
  1,
  SEQ_LEN,
  CHANNELS,
  FRAME_HEIGHT,
  FRAME_WIDTH,
];

export type CameraFrameOrientation =
  | 'portrait'
  | 'portrait-upside-down'
  | 'landscape-left'
  | 'landscape-right'
  | 'unknown';

export interface NativeRGBFrame {
  chw: Float32Array;
  timestampMs: number;
  preprocessTimeMs: number;
  orientation: CameraFrameOrientation;
}

interface UseConvLSTMFrameProcessorOptions {
  enabled: boolean;
  targetFps?: number;
  onFrame: (frame: NativeRGBFrame) => void;
}

function nowMsWorklet(): number {
  'worklet';
  const perf = (global as any).performance;
  if (perf && typeof perf.now === 'function') {
    return perf.now();
  }
  return Date.now();
}

function normalizeOrientationWorklet(value: unknown): CameraFrameOrientation {
  'worklet';
  if (
    value === 'portrait'
    || value === 'portrait-upside-down'
    || value === 'landscape-left'
    || value === 'landscape-right'
  ) {
    return value;
  }
  return 'unknown';
}

function quarterTurnsToPortrait(orientation: CameraFrameOrientation): 0 | 1 | 2 | 3 {
  'worklet';
  switch (orientation) {
    case 'portrait':
      return 0;
    case 'portrait-upside-down':
      return 2;
    case 'landscape-right':
      return 1;
    case 'landscape-left':
      return 3;
    default:
      return 0;
  }
}

function mapSourceCoordinate(
  x: number,
  y: number,
  size: number,
  quarterTurns: 0 | 1 | 2 | 3
): [number, number] {
  'worklet';
  switch (quarterTurns) {
    case 1:
      return [y, size - 1 - x];
    case 2:
      return [size - 1 - x, size - 1 - y];
    case 3:
      return [size - 1 - y, x];
    default:
      return [x, y];
  }
}

/**
 * VisionCamera frame processor that runs native resize and worklet-side RGB normalization.
 * Output is CHW Float32 in portrait orientation for direct ConvLSTM buffering.
 */
export function useConvLSTMFrameProcessor({
  enabled,
  targetFps = DEFAULT_TARGET_FPS,
  onFrame,
}: UseConvLSTMFrameProcessorOptions) {
  const { resize } = useResizePlugin();
  const onFrameJS = Worklets.createRunOnJS(onFrame);

  return useFrameProcessor(
    (frame) => {
      'worklet';
      if (!enabled) {
        return;
      }

      runAtTargetFps(targetFps, () => {
        'worklet';

        const start = nowMsWorklet();
        const resizedRgb = resize(
          frame,
          {
            scale: {
              width: FRAME_WIDTH,
              height: FRAME_HEIGHT,
            },
            pixelFormat: 'rgb',
            dataType: 'uint8',
          } as any
        ) as Uint8Array;

        if (!resizedRgb || resizedRgb.length < RGB_FRAME_SIZE) {
          return;
        }

        const orientation = normalizeOrientationWorklet((frame as any).orientation);
        const quarterTurns = quarterTurnsToPortrait(orientation);
        const chw = new Float32Array(RGB_FRAME_SIZE);

        for (let y = 0; y < FRAME_HEIGHT; y++) {
          for (let x = 0; x < FRAME_WIDTH; x++) {
            const [srcX, srcY] = mapSourceCoordinate(x, y, FRAME_WIDTH, quarterTurns);
            const srcPixel = (srcY * FRAME_WIDTH + srcX) * RGB_CHANNELS;
            const dstPixel = y * FRAME_WIDTH + x;

            chw[dstPixel] = resizedRgb[srcPixel] / 255.0;
            chw[FRAME_PLANE_SIZE + dstPixel] = resizedRgb[srcPixel + 1] / 255.0;
            chw[(FRAME_PLANE_SIZE * 2) + dstPixel] = resizedRgb[srcPixel + 2] / 255.0;
          }
        }

        onFrameJS({
          chw,
          timestampMs: Number((frame as any).timestamp ?? Date.now()),
          preprocessTimeMs: nowMsWorklet() - start,
          orientation,
        });
      });
    },
    [enabled, onFrameJS, resize, targetFps]
  );
}

/**
 * Ring buffer for assembling [1, 20, 6, 128, 128] tensors from normalized CHW RGB frames.
 * Intent channels (3, 4, 5) remain zero-filled in the persistent native tensor buffers.
 */
export class NativeSequenceBuffer {
  private readonly ringBuffer = new Float32Array(TOTAL_TENSOR_SIZE);
  private readonly tensorView = new Float32Array(TOTAL_TENSOR_SIZE);
  private frameCount = 0;
  private writeIndex = 0;

  addFrame(frame: NativeRGBFrame): void {
    const frameBase = this.writeIndex * FRAME_STRIDE;

    this.ringBuffer.set(frame.chw.subarray(0, FRAME_PLANE_SIZE), frameBase);
    this.ringBuffer.set(
      frame.chw.subarray(FRAME_PLANE_SIZE, FRAME_PLANE_SIZE * 2),
      frameBase + FRAME_PLANE_SIZE
    );
    this.ringBuffer.set(
      frame.chw.subarray(FRAME_PLANE_SIZE * 2, RGB_FRAME_SIZE),
      frameBase + (FRAME_PLANE_SIZE * 2)
    );

    // Intent channels are left untouched and stay at zero.

    this.writeIndex = (this.writeIndex + 1) % SEQ_LEN;
    if (this.frameCount < SEQ_LEN) {
      this.frameCount += 1;
    }
  }

  isReady(): boolean {
    return this.frameCount === SEQ_LEN;
  }

  getFrameCount(): number {
    return this.frameCount;
  }

  clear(): void {
    this.ringBuffer.fill(0);
    this.tensorView.fill(0);
    this.frameCount = 0;
    this.writeIndex = 0;
  }

  /**
   * Returns a contiguous tensor view ordered from oldest->newest frame.
   * Reuses a persistent output buffer to avoid per-inference allocations.
   */
  buildTensorView(): Float32Array {
    if (!this.isReady()) {
      throw new Error('NativeSequenceBuffer requires 20 frames before tensor build.');
    }

    // When the ring is full, writeIndex always points to the oldest frame.
    const oldestFrameIndex = this.writeIndex;

    for (let sequenceIndex = 0; sequenceIndex < SEQ_LEN; sequenceIndex++) {
      const sourceFrameIndex = (oldestFrameIndex + sequenceIndex) % SEQ_LEN;
      const sourceOffset = sourceFrameIndex * FRAME_STRIDE;
      const targetOffset = sequenceIndex * FRAME_STRIDE;

      this.tensorView.set(
        this.ringBuffer.subarray(sourceOffset, sourceOffset + FRAME_STRIDE),
        targetOffset
      );
    }

    return this.tensorView;
  }
}
