/**
 * Device Capabilities Detection
 * 
 * Utilities for detecting device hardware capabilities including
 * GPU acceleration, NPU/NNAPI support, and Android version
 */

import { Platform } from 'react-native';

/**
 * TensorFlow Lite delegate types
 */
export type TFLiteDelegate = 'nnapi' | 'gpu' | 'default';

/**
 * Device capability information
 */
export interface DeviceCapabilities {
  platform: 'android' | 'ios' | 'web' | 'unknown';
  androidVersion: number | null;
  supportsNNAPI: boolean;
  supportsGPU: boolean;
  recommendedDelegate: TFLiteDelegate;
}

/**
 * Get Android API level (version)
 * Returns null if not on Android or unable to determine
 */
export function getAndroidVersion(): number | null {
  if (Platform.OS !== 'android') {
    return null;
  }
  
  // React Native provides Android API level via Platform.Version
  const version = Platform.Version;
  return typeof version === 'number' ? version : null;
}

/**
 * Check if NNAPI is supported on this device
 * 
 * NNAPI is available on Android 8.1 (API 27) and above
 * Note: NNAPI is deprecated on Android 15 (API 35) but still functional
 * 
 * @returns true if NNAPI delegate can be used
 */
export function supportsNNAPI(): boolean {
  if (Platform.OS !== 'android') {
    return false;
  }
  
  const apiLevel = getAndroidVersion();
  if (apiLevel === null) {
    return false;
  }
  
  // NNAPI requires Android 8.1 (API 27) or higher
  return apiLevel >= 27;
}

/**
 * Check if GPU delegate is supported
 * 
 * GPU delegate (OpenGL/OpenCL) is generally available on most Android devices
 * with GPU hardware. We assume GPU support is available on Android devices.
 * 
 * @returns true if GPU delegate can be used
 */
export function supportsGPU(): boolean {
  // GPU delegate is available on Android via OpenGL/OpenCL
  // iOS would use Metal/CoreML delegate instead
  return Platform.OS === 'android';
}

/**
 * Get recommended TensorFlow Lite delegate for this device
 * 
 * Decision logic:
 * 1. If NNAPI is supported (Android 8.1+, API 27+), prefer NNAPI
 *    - NNAPI can utilize NPU when available (e.g., Qualcomm Hexagon)
 *    - Falls back to GPU/DSP/CPU automatically
 * 2. Otherwise, use GPU delegate if available
 * 3. Fall back to default (CPU) if neither is available
 * 
 * Note: On Android 15+ (API 35+), NNAPI is deprecated but still functional.
 * GPU delegate has similar performance and better loading time.
 * 
 * @returns Recommended delegate type
 */
export function getRecommendedDelegate(): TFLiteDelegate {
  if (Platform.OS !== 'android') {
    return 'default';
  }
  
  // Check for NNAPI support first (includes NPU when available)
  if (supportsNNAPI()) {
    return 'nnapi';
  }
  
  // Fall back to GPU delegate
  if (supportsGPU()) {
    return 'gpu';
  }
  
  // Ultimate fallback to CPU
  return 'default';
}

/**
 * Get comprehensive device capability information
 * 
 * @returns Device capabilities object
 */
export function getDeviceCapabilities(): DeviceCapabilities {
  const platform = Platform.OS as 'android' | 'ios' | 'web' | 'unknown';
  const androidVersion = getAndroidVersion();
  const nnapiSupport = supportsNNAPI();
  const gpuSupport = supportsGPU();
  const recommendedDelegate = getRecommendedDelegate();
  
  return {
    platform,
    androidVersion,
    supportsNNAPI: nnapiSupport,
    supportsGPU: gpuSupport,
    recommendedDelegate
  };
}

/**
 * Check if NNAPI is deprecated on this device
 * NNAPI is deprecated starting from Android 15 (API 35)
 * 
 * @returns true if NNAPI is deprecated (but still functional)
 */
export function isNNAPIDeprecated(): boolean {
  const apiLevel = getAndroidVersion();
  if (apiLevel === null) {
    return false;
  }
  
  // NNAPI deprecated on Android 15+ (API 35+)
  return apiLevel >= 35;
}

/**
 * Log device capabilities for debugging
 */
export function logDeviceCapabilities(): void {
  const caps = getDeviceCapabilities();
  
  console.log('[Device] ═══════════════════════════════════════');
  console.log('[Device] Device Capabilities');
  console.log('[Device] ───────────────────────────────────────');
  console.log(`[Device] Platform: ${caps.platform}`);
  
  if (caps.androidVersion !== null) {
    console.log(`[Device] Android API Level: ${caps.androidVersion}`);
    
    if (isNNAPIDeprecated()) {
      console.log('[Device] ⚠️  NNAPI is deprecated on Android 15+ (still functional)');
    }
  }
  
  console.log(`[Device] NNAPI Support: ${caps.supportsNNAPI ? '✅ Yes' : '❌ No'}`);
  console.log(`[Device] GPU Support: ${caps.supportsGPU ? '✅ Yes' : '❌ No'}`);
  console.log(`[Device] Recommended Delegate: ${caps.recommendedDelegate.toUpperCase()}`);
  console.log('[Device] ═══════════════════════════════════════');
}
