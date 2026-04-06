# EluSEEdate - Mobile Turn Prediction App

![TypeScript](https://img.shields.io/badge/TypeScript-3178C6?style=for-the-badge&logo=typescript&logoColor=white)
![JavaScript](https://shields.io/badge/JavaScript-F7DF1E?logo=JavaScript&logoColor=000&style=for-the-badge)
![React Native](https://img.shields.io/badge/React_Native-20232A?style=for-the-badge&logo=react&logoColor=61DAFB)
![Expo](https://img.shields.io/badge/Expo-000020?style=for-the-badge&logo=expo&logoColor=white)

React Native/Expo mobile application for real-time turn direction prediction using a ConvLSTM deep learning model with TensorFlow Lite inference.

The current runtime uses dual ConvLSTM services with a shared 6-channel tensor format:
1. Wandering mode uses `convlstmWithoutIntentInference` with intent channels kept at zero.
2. Destination mode uses `convlstmWithIntentInference` with route-driven intent channel injection.

## Overview

This app uses two AI models working in parallel:

1. **ConvLSTM** (Convolutional Long Short-Term Memory) for turn direction prediction - analyzes sequences of video frames to predict if the user should go **Front**, **Left**, or **Right**
2. **YOLOv12** for real-time obstacle detection - identifies nearby objects (people, cars, bicycles, etc.) and displays bounding boxes on the camera view

Audio feedback is powered by **ObjectSpeechService**, which converts YOLO detections into spoken obstacle prompts with priority and cooldown controls.

Voice command prompts in MainMenu, Choice, and Wayfinding are now coordinated by a shared hook (`useVoiceInteraction`) with:
1. Reusable TTS -> listening transitions.
2. `Skip` voice command and `Skip Audio` button support.
3. Standardized `[AUDIO-DEBUG]` lifecycle logging for TTS, earcon playback, and listener states.
4. Singleton earcon playback (`assets/sounds/ping.wav`) immediately before each menu/wayfinding TTS prompt.
5. A native-callback fallback timeout in `speakThenListen` that forces listening handoff if `Speech.onDone` is swallowed on-device.

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

- **Voice-first main menu** with Vosk commands (`Start`, `Exit`, `Skip`)
- **Mode selection flow** (`Wandering`, `Destination`, `Back`, `Skip`) with speech prompts
- **Wayfinding flow** for destination geocoding + spoken confirmation (`Yes`, `No`, `Back`, `Skip`)
- **Shared voice orchestration hook** (`src/hooks/useVoiceInteraction.ts`) for prompt timing, listening transitions, and cleanup
- **Accessibility prompt cues** with haptic buzz (`expo-haptics`) and singleton ping earcon (`assets/sounds/ping.wav`) played before TTS on MainMenu/Choice/Wayfinding
- **Unified camera runtime** in `ActiveCameraScreen` for both wandering and destination pipelines
- **Dual-path ConvLSTM selection** between `convlstmWithoutIntentInference` and `convlstmWithIntentInference` based on route mode
- **Real-time destination distance tracking** that recomputes live distance on every GPS fix
- **Live ConvLSTM turn prediction** with rolling frame buffer and low-latency updates
- **Live YOLO obstacle detection** with bounding box overlay
- **Priority-based spoken obstacle feedback** from YOLO detections (one object at a time, with cooldowns and danger interruption)
- **Performance overlay** with capture, preprocessing, ConvLSTM, and YOLO timings
- **Debug logs screen** for in-app runtime diagnostics

## Logs & Diagnostics

Open `LogsScreen` from MainMenu when you need targeted troubleshooting.

How to use filtered logs:
1. Tap the `Category` dropdown and choose the subsystem you want to inspect.
2. Keep `Auto-scroll` enabled for live sessions, or disable it to inspect older entries.
3. Tap `Clear` before reproducing an issue to start from a clean ring buffer.

Category mapping:
1. `All Logs` -> full stream (no filter)
2. `YOLO` -> `[INFERENCE-DEBUG]`, `[PRIORITY-DEBUG]`
3. `ConvLSTM` -> `[CONVLSTM-TRACE]`
4. `Audio` -> `[AUDIO-DEBUG]`, `[AUDIO-TRACE]`
5. `Errors` -> `console.error` and `[ERROR]`

## Spoken Obstacle Feedback Rules

When multiple objects are detected in the same frame:
1. The app announces one object per cycle.
2. It prioritizes the largest bounding box (closest-on-screen proxy).
3. Ties are resolved by higher confidence, then higher danger weight.

Announcement pacing and priority:
1. Same-class cooldown: 5000 ms.
2. Global cooldown: 1500 ms.
3. Danger objects can interrupt lower-priority speech (for example car, bus, truck, train, motorcycle).
4. Speech call is asynchronous so camera/inference processing is not blocked.

## Route Graph

Current stack routes in `App.tsx`:

1. `MainMenu`
2. `Choice`
3. `Wayfinding`
4. `ActiveCamera`
5. `Logs`

Runtime flow:

1. `MainMenu` -> `Choice`
2. `Choice` -> `ActiveCamera` (wandering mode)
3. `Choice` -> `Wayfinding` -> `ActiveCamera` (destination mode with route payload)
4. `MainMenu` -> `Logs` (debug diagnostics)

## Navigation Logic

Pipeline switching:
1. `mode='wandering'` selects `convlstmWithoutIntentInference` and keeps channels 3-5 zeroed.
2. `mode='destination'` selects `convlstmWithIntentInference` and writes intent values into channels 3-5.
3. Performance overlay mirrors the active runtime with `ConvLSTM: Wandering pipeline` or `ConvLSTM: Wayfinding pipeline`.

Real-time destination distance:
1. In destination mode, `ActiveCameraScreen` starts `expo-location` tracking with `watchPositionAsync`.
2. Every location update recalculates straight-line distance from current GPS position to destination coordinates.
3. Overlay `Dist` is refreshed on each fix using dynamic formatting:
  - `< 100 m` shown in meters (for example `84 m`)
  - `>= 100 m` shown in kilometers (for example `0.62 km`)

## Project Structure

```
├── App.tsx                          # Main entry point
├── package.json                     # Dependencies
├── app.json                         # Expo configuration
├── tsconfig.json                    # TypeScript config
├── babel.config.js                  # Babel config
├── eas.json                         # EAS Build configuration
├── assets/
│   ├── model/
│       ├── convlstm.tflite          # ConvLSTM TFLite model file
│       ├── convlstm.onnx            # ONNX model (backup)
│       └── yolo.tflite              # YOLOv12 TFLite model
│   └── sounds/
│       └── ping.wav                 # Earcon played before menu/wayfinding TTS prompts
├── texts/
│   ├── TECHNICAL_APPENDIX.md        # Source-code-truth architecture reference
│   ├── STANDALONE_APK_BUILD.txt     # EAS standalone APK guide
│   └── *.txt                        # Other reference docs
└── src/
    ├── components/
    │   ├── errorBoundary.tsx        # Error boundary component
    │   └── BoundingBoxOverlay.tsx   # YOLO bounding box renderer
    ├── config/
    │   └── modelConfig.ts           # ConvLSTM & YOLO configuration
    ├── navigation/
    │   └── types.ts                 # Navigation type definitions
    ├── screens/
    │   ├── MainMenuScreen.tsx       # Voice-first entry screen
    │   ├── ChoiceScreen.tsx         # Mode selection (Wandering/Destination)
    │   ├── WayfindingScreen.tsx     # Destination speech/geocoding flow
    │   ├── ActiveCameraScreen.tsx   # Unified camera + inference runtime
    │   └── LogsScreen.tsx           # Runtime log viewer
    ├── hooks/
    │   └── useVoiceInteraction.ts   # Shared TTS/STT state machine + skip/cue helpers
    ├── services/
    │   ├── preprocessor.ts          # Frame preprocessing (TypeScript)
    │   ├── convlstmWithoutIntentInference.ts  # ConvLSTM inference (no intent)
    │   ├── convlstmWithIntentInference.ts     # ConvLSTM inference (intent-aware path)
    │   ├── yoloInference.ts         # YOLO object detection inference
    │   ├── geocodingService.ts      # Destination geocoding
    │   ├── directionsService.ts     # Walking route fetcher
    │   └── ObjectSpeechService.ts   # Spoken obstacle announcements
    └── utils/
        ├── imageUtils.ts            # Image decoding utilities
        └── stringUtils.ts           # String formatting helpers for UI display
```

## Model Configuration

### ConvLSTM Turn Prediction

| Parameter | Value |
|-----------|-------|
| Input Shape | Runtime: [1, 20, 6, 128, 128] (both modes) |
| Sequence Length | 20 frames |
| Model FPS | 20 frames/second (training) |
| Camera FPS | ~2 frames/second (actual capture rate) |
| Prediction Start | Early prediction starts at 10 buffered frames (bootstrap + padding to 20) |
| Channels | 6 (RGB + intent planes; channels 3-5 are zeroed in wandering mode) |
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
| Status | Active runtime model loaded from `assets/model/yolo.tflite` in development/preview/production builds |

**Note**: Expo Go cannot run native TFLite inference; real YOLO and ConvLSTM inference require a development build or EAS build.

## Intent Channels

Runtime channel behavior:
1. Wandering path
  - Channels 0-2 for RGB
  - Channels 3-5 explicitly zeroed
  - Packed index: `(frameOffset + channel * (height * width) + (y * width + x))`
2. Destination path
  - Channels 0-2 for RGB
  - Channels 3-5 populated from route step intent metadata
  - Packed index: `(frameOffset + channel * (height * width) + (y * width + x))`

Path selection is driven by:
1. `route.params.mode`

## Getting Started

### Prerequisites

- Node.js (v18 or v20 LTS recommended; v20.18.2 tested)
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

Voice UX notes:
1. `Skip` can be spoken (or tapped as `Skip Audio`) in MainMenu, Choice, and Wayfinding.
2. Skipping immediately stops current prompt playback and advances to active listening state.
3. MainMenu, Choice, and Wayfinding now play `ping.wav` immediately before each TTS prompt.
4. ActiveCamera spoken obstacle announcements (YOLO/ConvLSTM) intentionally do not play earcons to keep navigation audio quieter.
5. `[AUDIO-DEBUG]` logs include `TTS Start`, `TTS Finished`, `Earcon Triggered`, and `Voice Listener Status` entries.
6. `useVoiceInteraction.speakThenListen` includes a duration-estimate timeout fallback so confirmation prompts still return to listening even if native `onDone` does not fire.

## Audio Behavior By Screen

Menu and destination setup screens:
1. MainMenu, Choice, and Wayfinding use `useVoiceInteraction`.
2. A singleton ping earcon is played before each TTS prompt.
3. Voice lifecycle events are logged with `[AUDIO-DEBUG]` and appear in both IDE console and `LogsScreen`.

Navigation camera screen:
1. ActiveCamera uses `ObjectSpeechService` for obstacle alerts.
2. No ping earcon is played for YOLO/ConvLSTM announcements.
3. This avoids unnecessary audio clutter while walking.

**Status Indicator**: A green dot means the app is actively capturing; "Demo Mode" label appears when using simulated predictions.

## Performance Metrics

The app displays the following metrics in the top-left corner:

- **Capture**: Time taken to capture a frame (in ms)
- **Inference (ConvLSTM)**: ConvLSTM inference time (in ms)
- **Preprocess**: Time taken to prepare frames (in ms)
- **YOLO**: YOLO object detection time (in ms)
- **Total**: `Preprocess + Inference (ConvLSTM) + YOLO` (in ms)
- **ConvLSTM Pipeline**: `Wandering pipeline` or `Wayfinding pipeline` based on selected mode
- **Target**: Destination label is visually truncated to the second comma in overlay only
- **Destination Dist**: Live straight-line distance to target, auto-formatted (`m` under 100m, otherwise `km`)
- **Frames/Predictions**: Count of captured frames and predictions

## Development Notes

### TFLite Implementation

The app uses `react-native-fast-tflite` for on-device TensorFlow Lite inference. This library:
- Provides GPU delegate support for accelerated inference
- Supports Float32 input tensors (required for ConvLSTM)
- Is registered as an Expo plugin in `app.json`

**Key Implementation Files**:
1. `src/services/convlstmWithoutIntentInference.ts`
2. `src/services/convlstmWithIntentInference.ts`
3. `src/services/preprocessor.ts`
4. `src/screens/ActiveCameraScreen.tsx`

**Demo Mode Fallback**: If TFLite isn't available (Expo Go), the app automatically switches to demo mode with simulated predictions. Check the status indicator on the camera screen.

### Frame Capture & Preprocessing

The ActiveCameraScreen captures frames using `expo-camera`:
- **Capture method**: `takePictureAsync` (~200-500ms per frame)
- **Frame processing**: Decodes JPEG to pixel data, resizes to 128x128
- **Buffer management**: Rolling buffer of frames with padding for early predictions
- **Preprocessing**: Normalizes to [0,1], then packs into 6-channel RGB+intent-safe layout

**Preprocessing Pipeline** (see `src/services/preprocessor.ts`):
1. Camera captures JPEG frame
2. Image decoded to RGBA pixel data
3. Resized to 128x128 using nearest-neighbor sampling
4. Normalized to [0, 1] range
5. Packed into [1, 20, 6, 128, 128] for both modes
6. In wandering mode, intent slots (channels 3-5) stay zero
7. In destination mode, intent slots receive route-step intent values
8. Output stays channels-first: [batch, seq, channels, height, width]

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
