/**
 * Image Utilities
 * 
 * Helper functions for decoding and processing camera images
 * Uses real JPEG decoding so models receive true camera pixels.
 */

import { FRAME_WIDTH, FRAME_HEIGHT } from '../config/modelConfig';

const jpeg: any = require('jpeg-js');

export interface RoiRect {
  x: number;
  y: number;
  width: number;
  height: number;
}

export const FLOOR_FOCUS_ROI_NORMALIZED: RoiRect = {
  x: 0.2,
  y: 0.3,
  width: 0.6,
  height: 0.7,
};

interface DecodeImageOptions {
  useFloorFocusRoi?: boolean;
}

export function getFloorFocusRoiPixels(sourceWidth: number, sourceHeight: number): RoiRect {
  const roiWidth = Math.max(1, Math.floor(sourceWidth * FLOOR_FOCUS_ROI_NORMALIZED.width));
  const roiHeight = Math.max(1, Math.floor(sourceHeight * FLOOR_FOCUS_ROI_NORMALIZED.height));
  const roiX = Math.max(0, Math.floor((sourceWidth - roiWidth) * 0.5));
  const roiY = Math.max(0, sourceHeight - roiHeight);

  return {
    x: roiX,
    y: roiY,
    width: roiWidth,
    height: roiHeight,
  };
}

/**
 * Decode a base64 image to raw RGBA pixel data
 * 
 * Strategy:
 * - Decode JPEG bytes with jpeg-js
 * - Resize to target dimensions while preserving real pixel values
 * 
 * @param base64Image - Base64 encoded image string (with or without data URI prefix)
 * @param targetWidth - Target width to resize to
 * @param targetHeight - Target height to resize to
 * @returns Promise<{ data: Uint8Array; width: number; height: number }>
 */
export async function decodeBase64ToPixels(
  base64Image: string,
  targetWidth: number = FRAME_WIDTH,
  targetHeight: number = FRAME_HEIGHT,
  options: DecodeImageOptions = {}
): Promise<{ data: Uint8Array; width: number; height: number }> {
  try {
    // Remove data URI prefix if present
    let base64Data = base64Image;
    if (base64Image.startsWith('data:')) {
      base64Data = base64Image.split(',')[1];
    }
    
    const pixelData = decodeAndResizeJpeg(base64Data, targetWidth, targetHeight, options);
    
    return {
      data: pixelData,
      width: targetWidth,
      height: targetHeight
    };
  } catch (error: any) {
    console.error('[ImageUtils] Failed to decode base64 image:', error?.message || error);
    throw error;
  }
}

/**
 * Decode JPEG bytes to RGBA and resize with nearest-neighbor.
 * This keeps preprocessing fast while preserving real scene content.
 */
function decodeAndResizeJpeg(
  base64: string,
  width: number,
  height: number,
  options: DecodeImageOptions = {}
): Uint8Array {
  const binaryString = atob(base64);
  const jpegBytes = new Uint8Array(binaryString.length);

  for (let i = 0; i < binaryString.length; i++) {
    jpegBytes[i] = binaryString.charCodeAt(i);
  }

  const decoded = jpeg.decode(jpegBytes, { useTArray: true });
  if (!decoded?.data || !decoded.width || !decoded.height) {
    throw new Error('JPEG decode returned empty pixel data');
  }

  const srcData: Uint8Array = decoded.data;
  const srcWidth: number = decoded.width;
  const srcHeight: number = decoded.height;

  const roi = options.useFloorFocusRoi
    ? getFloorFocusRoiPixels(srcWidth, srcHeight)
    : { x: 0, y: 0, width: srcWidth, height: srcHeight };

  if (roi.x === 0 && roi.y === 0 && roi.width === width && roi.height === height) {
    return srcData;
  }

  const out = new Uint8Array(width * height * 4);
  const scaleX = roi.width / width;
  const scaleY = roi.height / height;

  for (let y = 0; y < height; y++) {
    const srcY = roi.y + Math.min(Math.floor(y * scaleY), roi.height - 1);
    for (let x = 0; x < width; x++) {
      const srcX = roi.x + Math.min(Math.floor(x * scaleX), roi.width - 1);
      const srcIdx = (srcY * srcWidth + srcX) * 4;
      const dstIdx = (y * width + x) * 4;

      out[dstIdx] = srcData[srcIdx];
      out[dstIdx + 1] = srcData[srcIdx + 1];
      out[dstIdx + 2] = srcData[srcIdx + 2];
      out[dstIdx + 3] = srcData[srcIdx + 3];
    }
  }

  return out;
}

/**
 * Check if an image string is valid base64
 */
export function isValidBase64Image(base64: string): boolean {
  try {
    const base64Data = base64.replace(/^data:image\/\w+;base64,/, '');
    atob(base64Data.substring(0, Math.min(100, base64Data.length)));
    return base64Data.length > 100; // Reasonable minimum length
  } catch {
    return false;
  }
}
