/**
 * Model Configuration for ConvLSTM Turn Prediction
 * Based on Prototype 12 - Mobile-Optimized with Global Average Pooling + ONNX Export
 */

export const MODEL_CONFIG = {
  // Model Architecture Parameters
  model: {
    inputDim: 3,           // RGB channels only (no intent channels)
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
    seqLen: 10,            // Number of frames per FIFO sequence
    fps: 20,               // Frames per second sampling target
    duration: 1,           // Video duration in seconds (fast response)
    height: 128,           // Target frame height after resize
    width: 128,            // Target frame width after resize
    channels: 3,           // RGB channels only
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
    { id: 0, name: 'Front', description: 'User continues straight' },
    { id: 1, name: 'Left', description: 'User turns left' },
    { id: 2, name: 'Right', description: 'User turns right' }
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

// ============================================================================
// YOLO CONFIGURATION
// ============================================================================

/**
 * YOLOv12 Object Detection Configuration
 * For obstacle detection alongside ConvLSTM turn prediction
 * 
 * Model Specifications:
 * - Input: (1, 3, 128, 128) in BCHW format [Batch, Channels, Height, Width]
 * - Output: (1, 84, 336) where 84 = [4 bbox coords + 80 class scores], 336 detections
 */
export const YOLO_CONFIG = {
  // Model Architecture Parameters
  model: {
    inputShape: [1, 3, 128, 128], // Input shape: BCHW format
    outputShape: [1, 84, 336],    // Output shape: [batch, (bbox+classes), num_detections]
    inputSize: 128,               // Input image size (128x128)
    channels: 3,                  // RGB channels
    inputFormat: 'BCHW',          // Batch, Channels, Height, Width
    numClasses: 80,               // Number of COCO classes
    bboxCoords: 4,                // x, y, w, h
    numDetections: 336,           // Maximum number of detections
    confidenceThreshold: 0.5,     // Minimum confidence for detection
    iouThreshold: 0.45,           // IoU threshold for NMS
    
    // Quantization parameters (for int8 quantized models)
    // Note: If using float16, TFLite handles conversion internally
    isQuantized: false,           // Set to true if model uses int8 quantization
    quantizationScale: 0.003921568859368563,  // Dequantization scale (1/255)
    quantizationZeroPoint: -128,  // Dequantization zero point
  },

  // Preprocessing Parameters
  preprocessing: {
    normalize: true,          // Normalize pixel values
    normalizationRange: [0, 1], // Normalization range [min, max]
  },

  // Performance Settings
  performance: {
    inferenceIntervalFrames: 1, // Run YOLO every N frames (1 = every frame)
    expectedInferenceMs: {
      highEndDevice: 30,
      midRangeDevice: 60,
      lowEndDevice: 100
    }
  },

  // Common COCO classes for obstacle detection.
  // NOTE: This list is a placeholder and should be aligned with the active YOLOv12 model classes.
  commonObstacles: [
    'person',
    'bicycle',
    'car',
    'motorcycle',
    'bus',
    'truck',
    'traffic light',
    'stop sign',
    'dog',
    'cat'
  ]
} as const;

// YOLO class names (COCO dataset - 80 classes).
// NOTE: This list should be replaced with the active YOLOv12 model classes.
export const YOLO_CLASS_NAMES = [
  'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
  'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
  'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
  'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
  'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
  'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
  'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
  'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
  'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
  'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
];

export const YOLO_NUM_CLASSES = YOLO_CLASS_NAMES.length;

// Convenience YOLO exports
export const YOLO_INPUT_SIZE = YOLO_CONFIG.model.inputSize;
export const YOLO_CONFIDENCE_THRESHOLD = YOLO_CONFIG.model.confidenceThreshold;
export const YOLO_IOU_THRESHOLD = YOLO_CONFIG.model.iouThreshold;
