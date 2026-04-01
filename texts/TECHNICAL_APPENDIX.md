# EluSEEDate Technical Appendix (Source-Code Truth)

## 1. Scope and Method
This appendix is a code-grounded technical reference for the current mobile implementation. The analysis covers all requested files under src and cross-audits them against texts/VERSIONS.txt and texts/DATA_DICTIONARY.txt.

Scanned source files:
1. src/components/boundingBoxOverlay.tsx
2. src/components/errorBoundary.tsx
3. src/config/index.ts
4. src/config/modelConfig.ts
5. src/navigation/index.ts
6. src/navigation/types.ts
7. src/screens/cameraScreen.tsx
8. src/screens/index.ts
9. src/screens/logsScreen.tsx
10. src/screens/mainMenuScreen.tsx
11. src/services/convlstmWithIntentInference.ts
12. src/services/convlstmWithoutIntentInference.ts
13. src/services/index.ts
14. src/services/objectSpeechService.ts
15. src/services/preprocessor.ts
16. src/services/yoloInference.ts
17. src/utils/imageUtils.ts
18. src/utils/index.ts

Audit references:
1. texts/VERSIONS.txt
2. texts/DATA_DICTIONARY.txt

This document prioritizes implementation truth over comments and historical notes.

---

## 2. System Architecture and Runtime Intent
EluSEEDate executes two on-device inference streams from camera data, plus one speech-alert stream:
1. ConvLSTM stream for turn-direction classification across temporal frame sequences.
2. YOLO stream for per-frame obstacle detection.
3. Object speech stream for spoken obstacle alerts based on YOLO detections.

The operational pipeline is:
1. CameraScreen captures JPEG frames via expo-camera takePictureAsync.
2. imageUtils decodes Base64 JPEG into RGBA Uint8Array and resizes to model dimensions.
3. preprocessor FrameBuffer accumulates temporal context.
4. preprocessor VideoPreprocessor writes a Float32 sequence tensor in channels-first layout.
5. ConvLSTM TFLite service performs classification and computes probabilities.
6. YOLO TFLite service performs object detection, score filtering, and class-wise NMS.
7. objectSpeechService selects the highest-priority detection and produces text-to-speech output.
8. CameraScreen updates UI overlays for direction, confidence, performance, and detection boxes.

---

## 3. End-to-End Data Flow (Camera to Model and Back)

### 3.1 Screen Entry and Control Plane
MainMenuScreen navigates to CameraScreen via manual Start button or voice command. CameraScreen initializes both ConvLSTM and YOLO model managers. Capture starts only when all gating conditions are true:
1. Camera permission granted.
2. ConvLSTM model loaded.
3. Camera is mounted and ready.
4. Picture size selection is completed.

### 3.2 Capture Scheduling Model
CameraScreen uses recursive setTimeout instead of setInterval. This is a deliberate drift-control design:
1. Loop start timestamp recorded.
2. Async frame capture and processing executed.
3. Elapsed time subtracted from target interval.
4. Next invocation scheduled as max(0, targetInterval - elapsed).

Given REALISTIC_CAPTURE_FPS = 2, the nominal interval is 500 ms.

### 3.3 Frame Materialization
Each capture returns base64 JPEG (quality 0.1, skipProcessing true). decodeBase64ToPixels:
1. Removes optional data URI prefix.
2. Decodes JPEG bytes through jpeg-js.
3. Resizes RGBA using nearest-neighbor to 128 x 128.

Resulting FrameData fields:
1. data: Uint8Array RGBA pixel buffer.
2. width: 128.
3. height: 128.
4. timestamp: Date.now().

### 3.4 Temporal Buffer Semantics
FrameBuffer in preprocessor.ts:
1. Tracks frameCount globally.
2. Applies sampling every Nth frame where N = max(1, floor(cameraFps / FPS)).
3. Retains at most SEQ_LEN frames through FIFO shift.
4. Declares readiness once frames.length >= SEQ_LEN.

In CameraScreen, FrameBuffer is constructed with cameraFps = 2, while FPS from config is 20. Therefore samplingRate becomes 1 and no skip occurs; every captured frame is accepted.

### 3.5 Tensorization Path
VideoPreprocessor preprocessFrameSequence enforces exact sequence length and writes all frames into a persistent Float32Array buffer.

Tensor layout is [batch, seq, channels, height, width] with batch fixed to 1.

Index mapping used by direct write path:
$$
\text{frameStride} = C \cdot H \cdot W
$$
$$
\text{frameOffset}(f) = f \cdot \text{frameStride}
$$
$$
\text{index}(f,c,y,x) = \text{frameOffset}(f) + c \cdot (H \cdot W) + y \cdot W + x
$$

Source-to-target nearest-neighbor coordinate mapping:
$$
\text{srcX} = \min\left(\left\lfloor x \cdot \frac{\text{srcWidth}}{W} \right\rfloor,\ \text{srcWidth}-1\right)
$$
$$
\text{srcY} = \min\left(\left\lfloor y \cdot \frac{\text{srcHeight}}{H} \right\rfloor,\ \text{srcHeight}-1\right)
$$

RGBA to RGB extraction and normalization:
$$
R' = \frac{R}{255},\quad G' = \frac{G}{255},\quad B' = \frac{B}{255}
$$
if normalize is true; otherwise raw channel values are written.

Channels 3, 4, and 5 (intent channels) are intentionally not written and remain zero due typed array initialization and reuse strategy.

### 3.6 ConvLSTM Inference Path
CameraScreen invokes runPrediction from convlstmWithoutIntentInference.ts once buffer is ready and no active inference lock exists.

Service behavior:
1. Verifies model loaded.
2. Runs TFLite model with input tensor Float32Array.
3. Extracts logits from variable output container shape via extractOutputVector.
4. Applies numerically stable softmax:
$$
p_i = \frac{e^{z_i - z_{max}}}{\sum_j e^{z_j - z_{max}}}
$$
5. Computes argmax class and confidence.
6. Returns metrics where totalLatencyMs = preprocessing + inference and fps = 1000 / totalLatencyMs.

### 3.7 YOLO Inference Path
CameraScreen invokes runYOLODetection on each accepted frame, guarded by an independent YOLO lock.

Service behavior:
1. Applies floor-focus ROI crop before resize using center 60% width and bottom 70% height.
2. Preprocesses cropped ROI to BHWC Float32 tensor [1, 128, 128, 3].
3. Stores normalized ROI metadata with each detection for full-frame remapping.
4. Runs TFLite model.
5. Flattens output and parses detections from expected cardinality 84 x 336.
6. Selects top class per candidate.
7. Optionally applies sigmoid if sampled class values appear to be logits.
8. Converts center box to corner box:
$$
x_{tl} = x - \frac{w}{2},\quad y_{tl} = y - \frac{h}{2}
$$
9. Clamps boxes to [0,1], removes tiny boxes, applies class-wise NMS with IoU threshold 0.45.
10. Returns ROI-relative normalized detections plus ROI metadata to CameraScreen.

Floor-focus ROI in normalized full-frame coordinates:
$$
x_{roi}=0.2,\quad y_{roi}=0.3,\quad w_{roi}=0.6,\quad h_{roi}=0.7
$$

Overlay remap from ROI-normalized box \((x_r,y_r,w_r,h_r)\) to full-frame normalized box:
$$
x_f = x_{roi} + x_r \cdot w_{roi},\quad y_f = y_{roi} + y_r \cdot h_{roi}
$$
$$
w_f = w_r \cdot w_{roi},\quad h_f = h_r \cdot h_{roi}
$$

### 3.8 Object Speech Path
CameraScreen forwards YOLO detections to objectSpeechService when the YOLO model is loaded.

Service behavior:
1. Filters detections below configured speech confidence threshold.
2. Computes candidate priority from normalized box area, confidence, and danger weight.
3. Selects the nearest or largest candidate first, then resolves ties by confidence and danger.
4. Applies same-class and global cooldown windows.
5. Interrupts active speech only when the new candidate exceeds interrupt-priority delta.
6. Speaks direction-aware phrases such as "Person ahead" or "Car on the left".

### 3.9 Rendering and Feedback
CameraScreen pushes two independent visual outputs:
1. Direction card with class label and confidence from ConvLSTM.
2. BoundingBoxOverlay with YOLO normalized boxes projected to screen pixels.

Performance panel reports capture time, preprocessing time, model inference time, total latency, YOLO latency, frame count, object count, and prediction count.

---

## 4. File-by-File Technical Deep Dive

## 4.1 src/components/boundingBoxOverlay.tsx
### File Purpose
This component translates YOLO detections from normalized model space into screen coordinates and draws visual overlays on top of CameraView. It is the last mile between object detection outputs and user-visible hazard context.

### Logic Deep-Dive
1. Input contract is detections plus containerWidth and containerHeight.
2. Empty detection list short-circuits render and returns null, avoiding unnecessary tree cost.
3. For each detection:
   - Reads boundingBox.x, y, width, height in normalized [0,1].
   - Multiplies by container dimensions to obtain pixel-aligned absolute layout values.
4. Renders an absolute-positioned rectangular box and label text with confidence percentage rounded to integer.
5. Parent container uses pointerEvents none to preserve camera interaction semantics.

### Architectural Notes
1. No local state or side effects.
2. Deterministic pure rendering based on props.
3. Confidence display uses toFixed(0), which intentionally hides decimal precision.

---

## 4.2 src/components/errorBoundary.tsx
### File Purpose
This class component is a crash containment layer intended to prevent full-app blank screens from React tree errors by rendering a diagnostic fallback view.

### Logic Deep-Dive
1. getDerivedStateFromError flips hasError state synchronously during reconciliation.
2. componentDidCatch logs both error and component stack, then stores detail in state.
3. render fallback includes:
   - Human-readable error string.
   - JavaScript stack trace if present.
   - React component stack trace if present.
4. Try Again button clears boundary state and retries rendering child tree.

### Architectural Notes
1. Component is currently not wired in App.tsx; therefore it does not actively guard production runtime.
2. Styling favors debugging readability with monospace stack text.

---

## 4.3 src/config/index.ts
### File Purpose
Aggregator entrypoint to re-export model configuration, reducing import verbosity across screens and services.

### Logic Deep-Dive
1. Single statement exports all symbols from modelConfig.ts.
2. No runtime state.

---

## 4.4 src/config/modelConfig.ts
### File Purpose
Central declarative contract for both ConvLSTM and YOLO model assumptions, preprocessing dimensions, class naming, and expected runtime characteristics.

### Logic Deep-Dive
1. MODEL_CONFIG.model defines ConvLSTM architecture assumptions:
   - inputDim 6, hiddenDim [64,32], kernel [3,3], numLayers 2, output classes 3.
2. MODEL_CONFIG.preprocessing encodes sequence design:
   - seqLen 20, fps 20, 128x128 frame target, channels 6, normalize true.
3. MODEL_CONFIG.intent stores intent channel metadata for future use.
4. MODEL_CONFIG.performance and deployment hold expectation-level metadata used for thesis and planning.
5. Convenience constants export scalar values for direct import in runtime code.
6. YOLO_CONFIG defines nominal model IO, thresholds, quantization metadata, and class assumptions.
7. YOLO_CLASS_NAMES exports full COCO-80 list and derived class count.

### Architectural Notes
1. Mixed role file: hard runtime constants and soft expectation metadata coexist.
2. Runtime services partially hardcode values instead of reading all YOLO constants from this file.

---

## 4.5 src/navigation/index.ts
### File Purpose
Navigation barrel export for type definitions.

### Logic Deep-Dive
1. Re-exports all symbols from navigation/types.ts.
2. No runtime behavior.

---

## 4.6 src/navigation/types.ts
### File Purpose
Defines stack route contract for React Navigation with TypeScript safety.

### Logic Deep-Dive
1. RootStackParamList declares MainMenu, Camera, and Logs routes with no params.
2. Global namespace augmentation aligns ReactNavigation RootParamList with app-specific type.

### Architectural Notes
1. This file is the compile-time guard against invalid navigation target names.

---

## 4.7 src/screens/cameraScreen.tsx
### File Purpose
Primary experimental runtime surface for thesis measurements. It orchestrates camera acquisition, dual-model inference, latency tracking, and user feedback.

### Logic Deep-Dive
1. Permissions and camera lifecycle:
   - useCameraPermissions requests and tracks access.
   - onCameraReady sets readiness and triggers picture-size configuration.
2. Capture-size negotiation:
   - Reads available capture sizes from camera API.
   - Parses WxH tokens and computes area.
   - Selects smallest size with area >= TARGET_CAPTURE_AREA (640x480), else largest available.
3. Model initialization effect:
   - Initializes ConvLSTM and YOLO sequentially.
   - Stores separate loaded flags and user-visible debug status.
4. Capture loop:
   - Uses recursive timeout for asynchronous cadence control.
   - Prevents duplicate loop start via isCapturingRef and interval ref checks.
5. Frame capture routine:
   - takePictureAsync options: low quality, base64 true, skipProcessing true, no shutter sound.
   - Decodes base64 JPEG to 128x128 RGBA through imageUtils.
   - Pushes FrameData to FrameBuffer.
6. ConvLSTM gate and execution:
   - Requires model loaded, buffer ready, and no active inference lock.
   - Uses VideoPreprocessor to produce ProcessedTensor.
   - Calls runPrediction and updates prediction state and performance metrics.
7. YOLO gate and execution:
   - Requires YOLO model loaded and no active YOLO lock.
   - Calls detectObjects on current frame.
   - Updates detection list and YOLO inference time.
8. UI overlays:
   - Performance card with timings and counters.
   - Live status indicator.
   - Direction and confidence panel.
   - Buffer fill progress.
   - BoundingBoxOverlay for detections.

### State and Concurrency Model
1. React state stores display values and lifecycle state.
2. Mutable refs implement lock semantics and stale-closure avoidance for long-lived async loop callbacks.
3. captureFrameRef, runInferenceRef, and runYOLORef are reassigned each render to keep loop callbacks current.

### Implementation Implications
1. At 2 FPS, collecting 20 frames requires about 10 seconds before first full-sequence prediction.
2. canPredictEarly and getFramesPadded in FrameBuffer are not used by CameraScreen, so early prediction path is currently dormant.
3. Model cleanup functions are not called on unmount in this screen; model managers remain singleton-loaded until app lifecycle end or explicit cleanup call elsewhere.

---

## 4.8 src/screens/logsScreen.tsx
### File Purpose
In-app observability surface that captures console output into a searchable UI, useful for field testing where native logs are difficult to access.

### Logic Deep-Dive
1. Declares module-level logStorage array, id counter, and listeners array.
2. Overrides console.log, warn, error, info, and debug globally at module load.
3. captureLog serializes arguments to a single message string:
   - Objects are JSON.stringify attempted.
   - Circular or unserializable objects fall back to String conversion.
4. Retains only latest 500 entries to bound memory growth.
5. Notifies subscribers on every appended log entry.
6. Screen features:
   - Filters by all, yolo, convlstm, errors.
   - Optional auto-scroll to bottom on new log.
   - Manual clear operation.

### Architectural Notes
1. Console interception is global side effect and persists beyond this screen module once loaded.
2. Useful for thesis demos but may add overhead under high-frequency logging.

---

## 4.9 src/screens/mainMenuScreen.tsx
### File Purpose
Application entry UX and multimodal launch gate using touch and voice command for accessibility and hands-free workflow.

### Logic Deep-Dive
1. Module-level hasSpokenGreeting ensures TTS greeting only once per app session.
2. Startup useEffect triggers Speech.speak with en-US locale.
3. Vosk model load effect initializes speech recognizer and sets voiceStatus.
4. useFocusEffect lifecycle controls recognition while screen is focused:
   - Starts recognizer with grammar [start, [unk]].
   - Subscribes to Vosk.onResult.
   - Detects start phrase by case-insensitive substring match.
   - Uses hasNavigatedRef to prevent duplicate navigation events.
5. Cleanup:
   - Removes listener when focus is lost.
   - Stops recognizer.
6. Provides direct Start button and Debug Logs navigation button.

### Architectural Notes
1. Static version string displayed as v1.0.5.
2. Voice subsystem degrades gracefully on load or permission failure.

---

## 4.10 src/services/convlstmWithIntentInference.ts
### File Purpose
Future-oriented parallel service intended for intent-aware ConvLSTM path while preserving API shape equivalent to current no-intent service.

### Logic Deep-Dive
1. Implements same manager pattern as no-intent service:
   - Dynamic TFLite loader import.
   - Singleton model manager.
   - Model load, warm-up, inference, softmax, argmax.
2. Input contract expects ProcessedTensor with [1,20,6,128,128].
3. Output contract returns class id, class name, confidence, full probabilities, inference time.
4. High-level runPrediction computes total latency and FPS similarly.

### Architectural Notes
1. Header explicitly states placeholder status and identity with no-intent service.
2. Current CameraScreen imports and uses only convlstmWithoutIntentInference.ts.

---

## 4.11 src/services/convlstmWithoutIntentInference.ts
### File Purpose
Production ConvLSTM classifier service currently driving direction output in CameraScreen.

### Logic Deep-Dive
1. Loader initialization:
   - Attempts require of react-native-fast-tflite.
   - Stores availability error for transparent diagnostics in Expo Go.
2. loadModel:
   - Enables GPU delegate option useGpu true.
   - Loads assets/model/convlstm.tflite.
   - Performs warm-up inference on zero tensor.
3. runInference:
   - Enforces loaded model.
   - Calls model.run with tensor.data.
   - Extracts fixed-length output vector by slicing first NUM_CLASSES numeric values.
4. Numerical robustness:
   - Non-finite output values replaced with zero.
   - Softmax uses max-shift stabilization.
   - Invalid softmax denominator returns uniform distribution.
5. Prediction decision:
   - argmax over probabilities.
   - Maps class index to CLASS_NAMES.
6. Metrics composition:
   - preprocessingTimeMs from input tensor.
   - inferenceTimeMs from measured service run.
   - totalLatencyMs additive.
   - fps reciprocal of total latency.

### Mathematical Notes
Softmax implementation is stable and guards against overflow through subtraction of max logit.

---

## 4.12 src/services/preprocessor.ts
### File Purpose
Temporal and spatial transformation stage converting raw camera frames into ConvLSTM-ready sequence tensors.

### Logic Deep-Dive
1. Interfaces:
   - FrameData encapsulates RGBA pixels and metadata.
   - ProcessedTensor includes flat data, shape vector, and timing.
2. FrameBuffer class:
   - Maintains fixed-capacity FIFO frame list.
   - Applies configurable sampling rate.
   - Exposes ready checks, early-predict checks, padded frame retrieval, and status.
3. VideoPreprocessor constructor:
   - Stores dimensions, sequence length, normalize flag.
   - Precomputes framePlaneSize and frameStride for index arithmetic efficiency.
   - Allocates one persistent Float32 tensorBuffer for reuse.
   - Defines output shape [1, seq, channels, height, width].
4. preprocessFrameSequence:
   - Validates exact frame count.
   - Iterates sequence and processes each frame in-place into tensorBuffer.
   - Returns ProcessedTensor without reallocating data array.
5. processFrame:
   - Computes frameOffset and forwards to resizeNormalizeAndWriteFrame.
6. resizeNormalizeAndWriteFrame:
   - Nearest-neighbor source lookup by floor coordinate mapping.
   - Reads RGBA source but writes only RGB channels.
   - Normalizes to [0,1] depending on normalize flag.
   - Leaves intent channels untouched, preserving zeros.

### Mathematical and Memory Layout Notes
1. Channels-first placement supports ConvLSTM model expectation.
2. Persistent buffer reduces per-inference GC churn.
3. Integer floor mapping plus explicit clamp prevents source index overflow.

---

## 4.13 src/services/yoloInference.ts
### File Purpose
Object detection service managing YOLO model lifecycle, frame preprocessing, raw output interpretation, and NMS filtering.

### Logic Deep-Dive
1. Initialization and lifecycle:
   - Dynamic fast-tflite import with graceful failure diagnostics.
   - Singleton YOLOModelManager.
2. preprocessFrame:
   - Computes floor-focus ROI from source frame geometry.
   - Crops to center 60% width and bottom 70% height.
   - Converts ROI to BHWC Float32 [1,128,128,3] using nearest-neighbor resampling and normalization.
3. runInference:
   - Runs model and delegates output parsing.
4. parseYOLOOutput:
   - Expects 336 candidates, each with 4 box values and 80 class scores.
   - Flattens arbitrary nested typed output structures.
   - Uses score accessor for assumed transposed layout [84,336].
   - Samples score ranges to detect logits versus probabilities.
   - Applies confidence threshold and box validity checks.
   - Converts center boxes to top-left width-height representation.
   - Applies class-wise NMS with IoU threshold 0.45.
   - Attaches ROI metadata to each detection for overlay remapping.
5. Utility math:
   - sigmoid with clamped input domain to avoid overflow.
   - IoU as intersection over union area ratio.

### Notable Implementation Characteristics
1. Runtime confidence threshold default is 0.35, independent of config file nominal 0.5.
2. Layout auto-detection logs non-transposed possibility, but transposed path remains selected in current code path.
3. Contains extensive debug logging and probabilistic sampling logs.

---

## 4.14 src/services/objectSpeechService.ts
### File Purpose
Speech orchestration service that converts YOLO detections into concise obstacle announcements for accessibility and situational awareness.

### Logic Deep-Dive
1. Accepts detection arrays from cameraScreen and exits early when disabled or empty.
2. Filters invalid boxes and low-confidence detections.
3. Computes candidate area and direction labels from normalized box bounds.
4. Prioritizes nearest or largest objects by area-first ranking with confidence and danger tie-breakers.
5. Applies class-level and global cooldown windows to prevent repeated speech spam.
6. Uses interruption policy to stop active speech only for meaningfully higher-priority targets.
7. Sends final phrase to expo-speech and clears active token state on completion or error.

### Architectural Notes
1. Service state is encapsulated in one ref-managed singleton per camera screen instance.
2. Cooldown and interruption parameters are configurable via constructor options.
3. Direction labels are inferred from normalized horizontal box center with left, ahead, and right buckets.

---

## 4.15 src/utils/imageUtils.ts
### File Purpose
Decodes JPEG camera payloads into raw pixel tensors with deterministic resizing for downstream model services.

### Logic Deep-Dive
1. decodeBase64ToPixels:
   - Strips optional prefix.
   - Delegates decode and resize to internal function.
   - Returns RGBA data and target dimensions.
2. decodeAndResizeJpeg:
   - Converts base64 to byte buffer via atob and charCodeAt.
   - Uses jpeg-js decode with typed array output.
   - Fast path returns original decoded data when dimensions already match target.
   - Otherwise executes nearest-neighbor resize into new Uint8Array.
3. isValidBase64Image:
   - Lightweight sanity check by partial decode and minimum payload length heuristic.

### Architectural Notes
1. This utility is the bridge between compressed camera output and raw model inputs.
2. Interpolation method is nearest-neighbor, not bilinear.

---

## 5. Version Audit and Documentation Synchronization

## 5.1 Documentation-Code Mismatch

### Documentation-Code Mismatch 1
Claim source: src/services/preprocessor.ts header comment.
Claim: preprocessing runs at 10 FPS sampling rate.
Code truth: sampling is dynamic and currently tied to FrameBuffer constructor input and config FPS; CameraScreen passes REALISTIC_CAPTURE_FPS = 2.
Impact: thesis text may overstate temporal granularity.
Recommended correction: replace static 10 FPS statement with formula-driven sampling explanation.

### Documentation-Code Mismatch 2
Claim source: src/services/preprocessor.ts header comment.
Claim: returned tensor shape is [20,6,128,128].
Code truth: ProcessedTensor shape is [1,20,6,128,128].
Impact: omission of batch dimension can cause confusion in model interface description.
Recommended correction: update documentation to include batch dimension explicitly.

### Documentation-Code Mismatch 3
Claim source: texts/DATA_DICTIONARY.txt.
Claim: image utility uses bilinearResize function.
Code truth: src/utils/imageUtils.ts uses decodeAndResizeJpeg with nearest-neighbor mapping; no bilinearResize function exists.
Impact: preprocessing methodology in thesis appendix becomes technically incorrect.
Recommended correction: replace bilinear language with nearest-neighbor equations and rationale.

### Documentation-Code Mismatch 4
Claim source: texts/DATA_DICTIONARY.txt navigation section.
Claim: RootStackParamList includes only MainMenu and Camera.
Code truth: src/navigation/types.ts includes MainMenu, Camera, and Logs.
Impact: route graph in thesis diagram becomes incomplete.
Recommended correction: include Logs route and its diagnostics role.

### Documentation-Code Mismatch 5
Claim sources: src/config/modelConfig.ts and texts/DATA_DICTIONARY.txt versus src/services/yoloInference.ts.
Claim: YOLO confidence threshold is 0.5.
Code truth: YOLO service runtime default threshold is 0.35.
Impact: expected precision-recall behavior differs from documented assumptions.
Recommended correction: either align runtime to config value or document dual-threshold behavior and precedence.

### Documentation-Code Mismatch 6
Claim sources: texts/VERSIONS.txt and texts/DATA_DICTIONARY.txt/modelConfig metadata.
Claim: ConvLSTM model size reported as 3.62 MB in VERSIONS and 1.5 MB in DATA_DICTIONARY/modelConfig performance block.
Code truth: current on-disk assets/model/convlstm.tflite size is 2,694,608 bytes (about 2.57 MiB).
Impact: thesis reproducibility and storage budgeting become inconsistent.
Recommended correction: report exact byte size from release artifact and date-stamp measurement.

### Documentation-Code Mismatch 7
Claim source: texts/VERSIONS.txt known limitations section.
Claim: image path uses statistical sampling from JPEG compressed data.
Code truth: src/utils/imageUtils.ts performs full JPEG decode through jpeg-js, then deterministic nearest-neighbor resize.
Impact: perceived fidelity of pixel pipeline may be understated.
Recommended correction: describe full decode path and where losses occur (JPEG compression itself and nearest-neighbor resampling).

### Documentation-Code Mismatch 8
Claim source: component-level expectation around ErrorBoundary role.
Claim: ErrorBoundary catches React errors in app flow.
Code truth: ErrorBoundary is defined but not currently wrapped around App tree in App.tsx.
Impact: crash containment may be assumed but not active in runtime.
Recommended correction: either wire ErrorBoundary in App.tsx or mark component as currently dormant utility.

## 5.2 Items Confirmed as Synchronized
1. App version string in MainMenuScreen and runtime metadata now show 1.0.5.
2. ConvLSTM input dimensionality [1,20,6,128,128] is consistent between config metadata and inference service expectations.
3. takePictureAsync throughput limitation is consistent with CameraScreen comments and VERSIONS performance notes.

---

## 6. Practical Notes for Thesis Narrative
1. Distinguish design-time targets (20 FPS theoretical sequence rate) from runtime capture constraints (2 FPS practical in current capture approach).
2. Clearly separate model contract truth from implementation shortcuts:
   - ConvLSTM channels include three intent channels but intent values are currently all zero.
   - With-intent service exists as API-compatible placeholder, not an active divergent runtime path.
3. For defensibility, include exact preprocessing equations shown in this appendix and state interpolation mode explicitly.
4. Report both latency components separately:
   - preprocessingTimeMs
   - inferenceTimeMs
   - totalLatencyMs
5. Mention that YOLO and ConvLSTM run in parallel but are independently lock-guarded, which avoids same-model concurrency while still allowing dual-model operation per frame.

---

## 7. Recommended Immediate Documentation Updates
1. Update texts/DATA_DICTIONARY.txt sections for image pipeline, route list, and YOLO threshold.
2. Update preprocessor.ts header comments to reflect nearest-neighbor and batched output shape.
3. Reconcile model file size across modelConfig performance metadata and texts/VERSIONS.txt using artifact-measured value.
4. Add explicit note that ErrorBoundary is not currently mounted in App.tsx.
5. Add one architecture diagram in thesis chapter showing dual inference branches from shared capture source.

---

Prepared on: 2026-04-01
Document role: Thesis appendix technical truth reference
