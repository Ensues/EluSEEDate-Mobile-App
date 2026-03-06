# EluSEEdate - Mobile Turn Prediction App

React Native/Expo mobile application for real-time turn direction prediction using a ConvLSTM deep learning model with TensorFlow Lite inference.

## Overview

This app uses two AI models working in parallel:

1. **ConvLSTM** (Convolutional Long Short-Term Memory) for turn direction prediction - analyzes sequences of video frames to predict if the user should go **Front**, **Left**, or **Right**
2. **YOLOv12** for real-time obstacle detection - identifies nearby objects (people, cars, bicycles, etc.) and displays bounding boxes on the camera view

**Turn Prediction Model**: Prototype 10 (ConvLSTM with Global Average Pooling)
**Obstacle Detection**: YOLOv12 with TFLite optimization
**Inference Engine**: TensorFlow Lite via `react-native-fast-tflite`

## Two Operating Modes

### 1. Demo Mode (Expo Go)
- **For quick UI/UX testing** without building native code
- Camera and all UI features work normally
- Predictions are **simulated** (random with realistic timing)
- Run with: `npx expo start`

### 2. Production Mode (Development Build)
- **Real TFLite inference** on device
- Actual model predictions from camera frames
- Requires native build
- Run with: `npx expo prebuild && npx expo run:android`

## Target Device

**Redmi Note 13 Pro 5G**
- Screen: 1080 x 2400 pixels (portrait)
- Camera: 200MP main (rear)
- Expected inference time: ~100ms

## Design

Minimalistic black & white palette for a clean, distraction-free interface.

## Features

 - **Main Menu**: Simple start button with voice command support
- **TTS Startup Announcement**: Speaks "Starting EluSEEdate" when the app launches
- **Live Camera**: Real-time camera preview with dual-model prediction
- **Direction Label**: Large direction indicator at the bottom (from ConvLSTM)
- **Obstacle Detection**: Real-time object detection with bounding boxes (from YOLO)
- **Performance Overlay**: Inference times for both models displayed at top-left (in ms)
- **Auto-Prediction**: Sliding window inference - captures at ~2 FPS (limited by takePictureAsync)
- **Dual Inference**: ConvLSTM runs on frame sequences (20 frames), YOLO runs on single frames
- **Priority**: ConvLSTM takes priority if device struggles with both models

## Project Structure

```
├── App.tsx                          # Main entry point
├── package.json                     # Dependencies
├── app.json                         # Expo configuration
├── VERSIONS.txt                     # Dependency versions & rationale
├── tsconfig.json                    # TypeScript config
├── babel.config.js                  # Babel config
├── eas.json                         # EAS Build configuration
├── assets/
│   └── model/
│       ├── convlstm.tflite          # ConvLSTM TFLite model file
│       ├── convlstm.onnx            # ONNX model (backup)
│       └── yolo-placeholder.txt     # Placeholder for YOLOv12 model
├── texts/
│   ├── PACKAGE_DEPENDENCIES.txt     # All dependencies explained
│   ├── TROUBLESHOOTING.txt          # Debugging and diagnostics guide
│   ├── GIT_MERGE_GUIDE.txt          # How to merge branches
│   ├── DATA_DICTIONARY.txt          # Variable documentation
│   ├── DOCUMENTATION_UPDATE_GUIDE.txt # Maintenance guide
│   └── *.txt                        # Other reference docs
└── src/
    ├── components/
    │   ├── ErrorBoundary.tsx        # Error boundary component
    │   └── BoundingBoxOverlay.tsx   # YOLO bounding box renderer
    ├── config/
    │   └── modelConfig.ts           # ConvLSTM & YOLO configuration
    ├── navigation/
    │   └── types.ts                 # Navigation type definitions
    ├── screens/
    │   ├── MainMenuScreen.tsx       # Main menu with Start button
    │   └── CameraScreen.tsx         # Camera with dual-model inference
    ├── services/
    │   ├── preprocessor.ts          # Frame preprocessing (TypeScript)
    │   ├── convlstmWithoutIntentInference.ts  # ConvLSTM inference (no intent)
    │   ├── convlstmWithIntentInference.ts     # ConvLSTM inference (with intent, future)
    │   └── yoloInference.ts         # YOLO object detection inference
    └── utils/
        └── imageUtils.ts            # Image decoding utilities
```

## Model Configuration

### ConvLSTM Turn Prediction

| Parameter | Value |
|-----------|-------|
| Input Shape | [1, 20, 6, 128, 128] |
| Sequence Length | 20 frames |
| Model FPS | 20 frames/second (training) |
| Camera FPS | ~2 frames/second (actual capture rate) |
| Duration | 1 second of capture (20 frames @ 20 FPS) |
| Channels | 6 (3 RGB + 3 Intent) |
| Frame Size | 128 x 128 |
| Output Classes | 3 (Front, Left, Right) |
| Model Type | Float16 Quantized TFLite |
| Model Size | 3.62 MB (optimized from 7.14 MB) |
| GPU Acceleration | Enabled (via GPU delegate) |
| Expected Inference | ~100-200ms (GPU) / ~2-5s (CPU fallback) |

### YOLOv12 Obstacle Detection

| Parameter | Value |
|-----------|-------|
| Input Size | 128 x 128 (matches ConvLSTM, adjustable) |
| Model Type | TFLite (Float16 quantized expected) |
| Output | Bounding boxes with class probabilities |
| Classes | 80 (COCO dataset - adjust based on your model) |
| Confidence Threshold | 0.5 (50% minimum confidence) |
| GPU Acceleration | Enabled (via GPU delegate) |
| Expected Inference | ~30-100ms (faster than ConvLSTM) |
| Status | **Placeholder - awaiting real model** |

**Note**: Currently using a placeholder for YOLO. The app will simulate detections in demo mode until you add your actual YOLOv12 .tflite model to `assets/model/yolo.tflite`.

## Intent Channels

The model expects 6 channels per frame:
- Channels 0-2: RGB color channels [0, 1]
- Channels 3-5: Intent channels (all zeros for 'no intent' mode)

In this app, we always use "no intent" (all zeros for intent channels).

## Getting Started

### Prerequisites

- Node.js (v18 or v20 LTS recommended - v20.18.2 tested; **NOT v24+** due to compatibility issues with Expo SDK 50)
- Expo CLI
- Android Studio (for Android development)

### Installation

```bash
# Install dependencies
npm install

# Start Expo development server
npx expo start

# Run on Android device/emulator
npx expo run:android
```

### Building for Production

```bash
# Create production build using EAS Build (recommended)
npx eas build --platform android --profile preview

# For production release
npx eas build --platform android --profile production
```

## Usage

1. Launch the app — you will hear "Starting EluSEEdate" spoken aloud
2. Tap the **Start** button on the main menu (or say "Start")
3. Grant camera permission when prompted
4. Point the camera in the direction you're moving
5. The app will automatically:
   - Capture frames from the camera
   - Buffer frames until ready for prediction (min 10 frames)
   - Run predictions using the ConvLSTM model
   - Display the predicted direction at the bottom
   - Show performance metrics at the top-left

**Status Indicator**: A green dot means the app is actively capturing; "Demo Mode" label appears when using simulated predictions.

## Performance Metrics

The app displays the following metrics in the top-left corner:

- **Capture**: Time taken to capture a frame (in ms)
- **Inference**: Time taken by the TFLite model (in ms)
- **Preprocess**: Time taken to prepare frames (in ms)
- **Total**: Combined latency (in ms)
- **Frames/Predictions**: Count of captured frames and predictions

## Development Notes

### TFLite Implementation

The app uses `react-native-fast-tflite` for on-device TensorFlow Lite inference. This library:
- Provides GPU delegate support for accelerated inference
- Supports Float32 input tensors (required for ConvLSTM)
- Is registered as an Expo plugin in `app.json`

**Key Implementation Details** (see `src/services/inference.ts`):

```typescript
// Model loading (runs on app startup)
const model = await loadTensorflowModel(
  require('../../assets/model/convlstm.tflite')
);

// Inference (runs for each prediction)
const output = await model.run([tensorData]); // Float32Array [1, 20, 6, 128, 128]
// output[0] contains class probabilities [front, left, right]
```

**Demo Mode Fallback**: If TFLite isn't available (Expo Go), the app automatically switches to demo mode with simulated predictions. Check the status indicator on the camera screen.

### Frame Capture & Preprocessing

The CameraScreen captures frames using `expo-camera`:
- **Capture method**: `takePictureAsync` with base64 output (~200-500ms per frame)
- **Frame processing**: Decodes JPEG to pixel data, resizes to 128x128
- **Buffer management**: Rolling buffer of frames with padding for early predictions
- **Preprocessing**: Normalizes to [0,1], adds intent channels (zeros), transposes to NCHW format

**Preprocessing Pipeline** (see `src/services/preprocessor.ts`):
1. Camera captures JPEG frame → base64
2. Image decoded to RGBA pixel data
3. Resized to 128x128 using bilinear interpolation
4. Normalized to [0, 1] range
5. Intent channels added (all zeros)
6. Transposed to channels-first: [batch, seq, channels, height, width]

For production optimization, consider:
- Using `react-native-vision-camera` with frame processors for real-time 30 FPS capture
- Implementing native modules for direct YUV frame access
- Using GPU-accelerated preprocessing with `expo-gl` shaders

## Architecture

Based on **Prototype 10** - Mobile-Optimized ConvLSTM with Global Average Pooling + ONNX Export:
- 2-layer ConvLSTM: hidden_dim = [64, 32]
- Global Average Pooling (reduces model size by ~80%)
- Dropout (0.5) for regularization
- 3-class classification output

## License

Part of thesis project for ConvLSTM Turn Prediction.
