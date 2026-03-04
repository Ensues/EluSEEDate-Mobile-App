# EluSEEdate - Mobile Turn Prediction App

React Native/Expo mobile application for real-time turn direction prediction using a ConvLSTM deep learning model with TensorFlow Lite inference.

## Overview

This app uses a **ConvLSTM (Convolutional Long Short-Term Memory)** model to predict vehicle turn directions in real-time using the device camera. The model analyzes sequences of video frames and outputs one of three directions: **Front**, **Left**, or **Right**.

**Model Architecture**: Prototype 10 (ConvLSTM with Global Average Pooling)
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

- **Main Menu**: Simple start button
- **Live Camera**: Real-time camera preview with automatic prediction
- **Direction Label**: Large direction indicator at the bottom
- **Performance Overlay**: Inference time and latency metrics at top-left (in ms)
- **Auto-Prediction**: Sliding window inference - captures at 20 FPS, gives new prediction every 2 frames (10 predictions/sec)
- **Fast Response**: First prediction after 1 second (20 frames), then continuous predictions using 20-frame sliding window

## Project Structure

```
test_deployment/
├── App.tsx                          # Main entry point
├── package.json                     # Dependencies
├── app.json                         # Expo configuration
├── VERSIONS.txt                     # Dependency versions & rationale
├── tsconfig.json                    # TypeScript config
├── babel.config.js                  # Babel config
├── eas.json                         # EAS Build configuration
├── assets/
│   └── model/
│       ├── convlstm.tflite          # TFLite model file
│       └── convlstm.onnx            # ONNX model (backup)
└── src/
    ├── config/
    │   └── modelConfig.ts           # Model & device configuration
    ├── navigation/
    │   └── types.ts                 # Navigation type definitions
    ├── screens/
    │   ├── MainMenuScreen.tsx       # Main menu with Start button
    │   └── CameraScreen.tsx         # Camera with inference
    ├── services/
    │   ├── preprocessor.ts          # Frame preprocessing (TypeScript)
    │   └── inference.ts             # TFLite model inference
    └── utils/
        └── imageUtils.ts            # Image decoding utilities
```

## Model Configuration

| Parameter | Value |
|-----------|-------|
| Input Shape | [1, 20, 6, 128, 128] |
| Sequence Length | 20 frames |
| Model FPS | 20 frames/second (training) |
| Camera FPS | 20 frames/second (target capture rate) |
| Duration | 1 second of capture (20 frames @ 20 FPS) |
| Channels | 6 (3 RGB + 3 Intent) |
| Frame Size | 128 x 128 |
| Output Classes | 3 (Front, Left, Right) |
| Model Type | Float16 Quantized TFLite |
| Model Size | 3.62 MB (optimized from 7.14 MB) |
| GPU Acceleration | Enabled (via GPU delegate) |
| Expected Inference | ~100ms (GPU) / ~2-5s (CPU fallback) |
| Prediction Interval | New prediction every 2 frames (~100ms between predictions) |

## Intent Channels

The model expects 6 channels per frame:
- Channels 0-2: RGB color channels [0, 1]
- Channels 3-5: Intent channels (all zeros for 'no intent' mode)

In this app, we always use "no intent" (all zeros for intent channels).

## Getting Started

### Prerequisites

- Node.js 18+
- Expo CLI
- Android Studio (for Android development)

### Installation

```bash
# Navigate to project directory
cd test_deployment

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

1. Launch the app
2. Tap the **Start** button on the main menu
3. Grant camera permission when prompted
4. Point the rear camera at the road ahead
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
