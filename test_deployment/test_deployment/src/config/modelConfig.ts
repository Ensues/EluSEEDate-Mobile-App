/**
 * Model Configuration for ConvLSTM Turn Prediction
 * Based on Prototype 10 - Mobile-Optimized with Global Average Pooling + ONNX Export
 */

export const MODEL_CONFIG = {
  // Model Architecture Parameters
  model: {
    inputDim: 6,           // RGB (3) + Intent (3) channels
    hiddenDim: [64, 32],   // Hidden dimensions for each ConvLSTM layer
    kernelSize: [3, 3],    // Convolutional kernel size
    numLayers: 2,          // Number of stacked ConvLSTM layers
    height: 128,           // Frame height
    width: 128,            // Frame width
    numClasses: 3,         // Output classes: Front (0), Left (1), Right (2)
    dropoutRate: 0.5,      // Dropout probability
    hasGlobalAvgPool: true // GAP layer present (mobile optimization)
  },

  // Preprocessing Parameters
  preprocessing: {
    seqLen: 20,            // Number of frames per sequence
    fps: 20,               // Frames per second to extract (20 FPS for 1 second)
    duration: 1,           // Video duration in seconds (fast response)
    height: 128,           // Target frame height after resize
    width: 128,            // Target frame width after resize
    channels: 6,           // Total channels: 3 RGB + 3 Intent
    normalize: true,       // Normalize pixel values to [0, 1]
    colorFormat: 'RGB'     // Expected color format
  },

  // Intent Configuration
  intent: {
    numIntentChannels: 3,  // One channel per direction
    intentDuration: 1,     // Intent signal duration in seconds
    intentFrames: 10,      // Intent frames (intentDuration × fps)
    intentProbability: 0.6 // Probability of intent being present
  },

  // Class Labels
  classes: [
    { id: 0, name: 'Front', description: 'Vehicle continues straight' },
    { id: 1, name: 'Left', description: 'Vehicle turns left' },
    { id: 2, name: 'Right', description: 'Vehicle turns right' }
  ],

  // Performance Metrics (Expected)
  performance: {
    modelSizeMb: 1.5,
    parameters: 272000,
    expectedAccuracy: 0.95,
    inferenceTimeMs: {
      highEndDevice: 50,
      midRangeDevice: 100,
      lowEndDevice: 200
    }
  },

  // Deployment Settings
  deployment: {
    targetPlatform: 'Android',
    framework: 'Expo/React Native',
    inferenceBackend: 'TensorFlow Lite',
    preprocessingLocation: 'On-device',
    modelFormat: 'TFLite'
  }
} as const;

// Convenience exports
export const SEQ_LEN = MODEL_CONFIG.preprocessing.seqLen;
export const FPS = MODEL_CONFIG.preprocessing.fps;
export const FRAME_HEIGHT = MODEL_CONFIG.preprocessing.height;
export const FRAME_WIDTH = MODEL_CONFIG.preprocessing.width;
export const CHANNELS = MODEL_CONFIG.preprocessing.channels;
export const NUM_CLASSES = MODEL_CONFIG.model.numClasses;
export const CLASS_NAMES = MODEL_CONFIG.classes.map(c => c.name);
export const INTENT_FRAMES = MODEL_CONFIG.intent.intentFrames;

// Device-specific configuration for Redmi Note 13 Pro 5G
export const DEVICE_CONFIG = {
  name: 'Redmi Note 13 Pro 5G',
  cameraFps: 20,                    // Target capture FPS (20 frames in 1 second)
  screenWidth: 1080,                // Screen width in pixels
  screenHeight: 2400,               // Screen height in pixels
  expectedInferenceMs: 100,         // Expected inference time
  frameInterval: 1,                 // Take every frame at 20fps
  bufferDurationMs: 1000,           // 1 second of frames for fast response
  predictionInterval: 2             // Run prediction every Nth frame (2 = 10 predictions/sec)
} as const;

export type PredictionClass = 'Front' | 'Left' | 'Right';
export type ClassId = 0 | 1 | 2;
