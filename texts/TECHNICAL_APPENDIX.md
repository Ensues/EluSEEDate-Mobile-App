# EluSEEDate Technical Appendix (Source-Code Truth)

Prepared on: 2026-04-06
Purpose: implementation-grounded technical reference for maintenance and release hardening.

---

## 1. Scope

This appendix describes the code as it currently runs in the repository. It reflects active routes, runtime behavior, and data flow in:
- App.tsx
- src/navigation/types.ts
- src/hooks/useVoiceInteraction.ts
- src/screens/MainMenuScreen.tsx
- src/screens/ChoiceScreen.tsx
- src/screens/WayfindingScreen.tsx
- src/screens/ActiveCameraScreen.tsx
- src/screens/LogsScreen.tsx
- src/services/preprocessor.ts
- src/services/convlstmWithoutIntentInference.ts
- src/services/convlstmWithIntentInference.ts
- src/services/yoloInference.ts
- src/services/ObjectSpeechService.ts
- src/services/geocodingService.ts
- src/services/directionsService.ts
- src/utils/imageUtils.ts
- src/utils/stringUtils.ts
- src/config/modelConfig.ts

---

## 2. Runtime Architecture

The app currently has three main runtime branches:
1. Voice-first navigation and mode selection.
2. Destination capture and route construction (wayfinding).
3. Unified camera inference runtime (ActiveCamera) for both wandering and destination modes.

### 2.1 Root Navigation Contract

RootStackParamList:
1. MainMenu
2. Choice
3. Wayfinding
4. ActiveCamera
5. Logs

ActiveCamera route params:
1. mode: wandering | destination (required)
2. origin (optional)
3. destination (optional)
4. destinationLabel (optional)
5. routeSteps (optional)
6. totalDistanceMeters (optional)
7. totalDurationSeconds (optional)

### 2.2 User Flow

Flow A (Wandering):
1. MainMenu
2. Choice
3. ActiveCamera with mode=wandering

Flow B (Destination):
1. MainMenu
2. Choice
3. Wayfinding
4. ActiveCamera with mode=destination and route payload

Flow C (Diagnostics):
1. MainMenu
2. Logs

---

## 3. Voice and Wayfinding Pipeline

### 3.1 MainMenu and Choice

- MainMenu and Choice use react-native-vosk for command-style grammar.
- MainMenu supports Start, Exit, and Skip commands.
- Choice supports Wandering, Destination, Back, and Skip commands.
- Both screens render a Skip Audio button that maps to the same skipSpeech path as the Skip voice command.

### 3.2 WayfindingScreen

Wayfinding is voice-first and uses:
- expo-location for user origin
- expo-speech-recognition for free-form destination and confirmation responses
- shared hook speech controls for TTS playback, listening-state transitions, and cleanup

High-level sequence:
1. Request location permission and resolve current location.
2. Prompt user to speak destination.
3. Geocode spoken destination.
4. Confirm candidate destination via yes/no/back.
5. Enforce MAX_RADIUS_KM = 10.
6. Fetch walking directions from OSRM service.
7. Navigate to ActiveCamera with destination payload.

Hardening behavior:
- Auto-restart voice-listening timeout is tracked and cleared during cleanup to avoid stale delayed restarts.
- Skip command/button can immediately interrupt prompts and move directly to listening state.
- Destination confirmation now sets a transition lock only after explicit Yes confirmation passes validation and route fetch starts, so confirmation listening remains available.
- On successful directions fetch, Wayfinding now navigates immediately to ActiveCamera (destination mode) instead of waiting on a trailing TTS onDone callback.

### 3.3 Shared Voice Interaction Hook

src/hooks/useVoiceInteraction.ts now centralizes prompt/listening state for MainMenu, Choice, and Wayfinding.

Current hook contract provides:
1. speakMessage: one-shot TTS.
2. speakThenListen: prompt followed by delayed transition to ready listening state.
3. skipSpeech: immediate Speech.stop plus state advance to listening.
4. startVoskListening / stopVoskListening for command grammar loops.
5. startExpoListening / stopExpoListening for free-form destination capture.
6. stopAllVoiceActivity for cleanup during blur/unmount.

Handoff reliability behavior:
1. speakThenListen wires native Speech onDone to the listening transition path.
2. A fallback timeout estimates prompt duration with `700 + (message.length * 50)` ms (clamped to 1500-20000 ms), plus configured delay, and forces listening if native onDone is swallowed.
3. The fallback timeout is cleared automatically when native onDone, onError, skipSpeech, or cleanup runs.

Accessibility listening cue behavior:
1. Every transition to listening emits haptic feedback using expo-haptics.
2. A short earcon is played from assets/sounds/ping.wav via expo-av.
3. Cue emission is best-effort and non-fatal (voice flow continues even if cue playback fails).

---

## 4. ActiveCamera Runtime

ActiveCamera is the single camera runtime for both modes.

### 4.1 Mode Selection

Pipeline is selected by:
- route.params.mode

Selection matrix:
1. mode=wandering -> convlstmWithoutIntentInference
2. mode=destination -> convlstmWithIntentInference

Overlay pipeline label:
1. mode=wandering -> "Wandering pipeline"
2. mode=destination -> "Wayfinding pipeline"

Preprocessor channel allocation:
1. wandering path -> channels=6 with intent planes zeroed
2. destination path -> channels=6 with GPS-driven intent-plane injection

### 4.2 Camera Setup and Capture

- Camera implementation: expo-camera CameraView.
- Capture loop: recursive setTimeout (not setInterval).
- Capture frequency: REALISTIC_CAPTURE_FPS = 2.
- Capture method: takePictureAsync with quality 0.2, skipProcessing=true, shutterSound=false.
- Primary decode path: decodeImageUriToPixels.
- Fallback decode path: decodeBase64ToPixels.

Destination-mode route progress wiring in current source:
1. ActiveCamera starts live GPS tracking using expo-location watchPositionAsync in destination mode when destination coordinates are present.
2. ActiveCamera updates userLocation on every GPS fix.
3. ActiveCamera fetches and caches walking directions from current GPS position to route.params.destination once per session.
4. Step advancement is distance-driven: when distance to current step end is less than 2 meters, currentStepIndex advances.
5. Distance checks use getGeoDistance imported from geolib/es/getDistance.
6. routeProgress keeps both distanceRemaining and distanceToStepEnd for intent-aware frame metadata.
7. currentDistance is recalculated on every location update and shown in overlay as meters when under 100 m, otherwise kilometers.

Hardening changes in current source:
1. Camera-ready callback is guarded by refs so one-time picture-size configuration is not repeatedly re-run.
2. Capture gating checks ref-backed readiness flags to avoid stale-closure gating.
3. Structured runtime diagnostics are active in inference paths using INFERENCE-DEBUG, PRIORITY-DEBUG, and CONVLSTM-TRACE log families.
4. Performance overlay now includes audio diagnostics: current audio state and last announced object.
5. ActiveCamera cleanup now calls the selected ConvLSTM service cleanupModel on unmount.
6. Performance overlay Total is now computed as preprocess + ConvLSTM inference + YOLO latency.
7. Destination Target text in overlay is display-truncated to the second comma via truncateToSecondComma without changing route/TTS source data.

### 4.3 Frame Buffer and ConvLSTM Input

Preprocessor output shape:
1. [1, 20, 6, 128, 128] for both wandering and destination pipelines

Channel semantics:
1. Channels 0-2: RGB
2. Channels 3-5: intent channels

Current truth:
1. ConvLSTM preprocessing always allocates six channels.
2. RGB data is written in channel-first layout into channels 0-2.
3. Intent channels 3-5 are cleared to zero for every frame before optional writes.
4. In wandering mode (no-intent pipeline), channels 3-5 remain zero by design.
5. In destination mode (intent pipeline), ActiveCamera writes per-frame intent metadata:
	- intent from maneuverToIntent(currentStep.maneuver)
	- intentDistance from routeProgress.distanceToStepEnd
	and preprocessor fills intent channels by plane:
	- if intentDistance <= 5 meters, set channel (3 + intentClass) to 1
	- otherwise, set Front intent channel (channel 3) to 1

Packing math used by preprocessor:
1. 6-channel channel-first path
	- frameStride = height * width * 6
	- channel base index = frameOffset + channel * (height * width)
	- pixel index = channelBase + (y * width + x)

Buffer behavior:
1. Early inference starts at half sequence length (10 buffered frames when SEQ_LEN=20).
2. First inference uses bootstrap doubling.
3. Subsequent inference uses tail padding.

### 4.4 Unified Inference Diagnostics

Current ActiveCamera diagnostics:
1. YOLO emits INFERENCE-DEBUG count logs and PRIORITY-DEBUG closest-object logs using largest bounding-box area.
2. ConvLSTM emits INFERENCE-DEBUG and PRIORITY-DEBUG logs using highest class probability.
3. ConvLSTM emits CONVLSTM-TRACE logs for:
	- pipeline and buffer status at inference start
	- tensor readiness and preprocessing time
	- predicted label, confidence, and top probability ranking
	- timing summary (preprocess, inference, total latency, and FPS)
4. Wandering mode emits CONVLSTM-NOINTENT-TRACE channel snapshots on a fixed interval,
	including sampled per-channel means/max values and a warning if intent channels are non-zero.
5. ConvLSTM no-intent inference now validates tensor rank, shape, and flattened length
	before model.run, and includes softmax safety guards for non-finite logits.
6. UI overlay surfaces the same runtime intent by showing:
	- Audio: Ready | Speaking | Error
	- Last Announced: most recent spoken object label
6. Performance overlay metric order is:
	- Capture -> Inference (ConvLSTM) -> Preprocess -> YOLO -> Total
7. Overlay Total is user-facing full latency:
	- preprocess + ConvLSTM inference + YOLO
8. Destination overlay Target is visually shortened by truncateToSecondComma.
	- Full destinationLabel remains unchanged for route logic and speech.

---

## 5. YOLO Runtime

YOLO service uses react-native-fast-tflite when native runtime is available.

Current behavior:
1. Model loads from assets/model/yolo.tflite.
2. GPU delegate option is requested.
3. Output parsing supports transposed and non-transposed tensor layouts.
4. Detections are filtered by class allowlist and confidence threshold.
5. Same-class NMS is applied (IoU threshold 0.45).

Hardening updates applied:
1. Removed temporary high-volume debug logging from per-frame inference path.
2. Kept explicit BCHW/BHWC reference comments in preprocessing code; active write path remains BHWC for current runtime.
3. Enabled non-transposed tensor branch assignment in layout detection logic.

### 5.1 ObjectSpeech Runtime (Audio Obstacle Feedback)

ObjectSpeech is now wired directly into the ActiveCamera YOLO loop.

Integration flow:
1. ActiveCamera creates a single ObjectSpeechService instance in a stable useRef.
2. The service is started on screen mount.
3. Immediately after YOLO returns detections, ActiveCamera calls announceDetections(detections).
4. The call is non-blocking (fire-and-forget with Promise error catch) so camera/inference loop timing is not blocked by TTS.
5. The service is disposed on unmount to stop playback and avoid background audio leaks.
6. ActiveCamera subscribes to speech debug snapshots to drive overlay audio state and last-announced label.

Multi-object selection behavior:
1. Only one candidate is spoken per detection cycle.
2. Candidate filtering requires confidence > 0.5 and valid normalized box geometry.
3. Best candidate is selected by largest bounding-box area first (closest-on-screen proxy).
4. If area is tied, higher confidence wins.
5. If area and confidence are tied, higher danger weight wins.

Speech priority and anti-spam policy:
1. sameClassCooldownMs = 5000
2. globalCooldownMs = 1500
3. interruptPriorityDelta = 0.18
4. dangerInterruptThreshold = 0.9
5. High-danger objects (for example car, bus, truck, train, motorcycle) can pre-empt lower-priority active speech.

Audio hardening and tracing:
1. Service attempts local/offline voice selection using getAvailableVoicesAsync and language matching.
2. AUDIO-TRACE logs are emitted for service input and skip reasons (for example low confidence or cooldown).
3. Speech.speak is guarded with error capture, logging code and message when playback fails.
4. Audio runtime state is tracked as ready, speaking, or error.

Class name mapping for TTS:
1. Uses detection.className when meaningful.
2. If className is missing/placeholder (for example class_7), falls back to YOLO class lookup by classId.
3. Final spoken phrase is direction-aware: "<Object> ahead", "<Object> on the left", or "<Object> on the right".

---

## 6. Configuration and Runtime Switches

In src/config/modelConfig.ts:
1. ConvLSTM preprocessing constants and model metadata are centralized.
2. Runtime switch MODEL_CONFIG.runtime.enableIntentMode remains available for configuration-level experimentation.
3. ENABLE_INTENT_MODE is exported but is not currently consumed by ActiveCamera or preprocessor runtime flow; pipeline selection is mode-driven and intent-plane injection is passed explicitly from ActiveCamera to VideoPreprocessor.

---

## 7. Current Operational Notes

1. ActiveCamera is the only camera inference screen in active navigation flow.
2. Destination route payload is generated in Wayfinding and displayed in ActiveCamera performance overlay.
3. Logs screen is still available for diagnostics routing.
4. Runtime diagnostics are intentionally verbose in inference and audio paths to support field debugging.
5. ConvLSTM currently has on-screen output (direction/confidence) and debug logs; spoken obstacle feedback is driven by YOLO detections through ObjectSpeechService.
6. ActiveCamera currently performs in-screen route distance checks using geolib/es/getDistance for step progression.
7. Voice-first navigation screens now share one hook-level cleanup path to reduce listener/timer leak risk.
8. Destination overlay Dist is now live-updated from GPS fixes (not static route-entry distance).
9. Destination overlay Target text is truncated for readability only; internal destination strings remain full fidelity.
10. Overlay Total now includes YOLO latency in addition to ConvLSTM preprocess/inference timings.

---

## 8. Recommended Ongoing Maintenance

1. Keep route and param changes mirrored in src/navigation/types.ts and this appendix.
2. Keep model input format comments synchronized with actual tensor write order in yoloInference.
3. Re-run npx expo-doctor, npx tsc --noEmit, and npx expo lint before each release candidate and before EAS preview/production builds.
4. Maintain changelog entries for each semantic version bump.

---

<<<<<<< HEAD
## 9. Troubleshooting Toolkit Results (2026-04-05, Current Runtime Verification)

Validation run performed after reconfirming ConvLSTM 6-channel preprocessing behavior for both pipelines and refreshing runtime documentation.
=======
## 9. Troubleshooting Toolkit Results (2026-04-06, Overlay Latency + Docs Sync)

Validation run performed after confirming YOLO-inclusive overlay total latency and syncing README/technical appendix docs to runtime truth.
>>>>>>> refactor/performance-metrics

1. Expo Doctor
	- Command: npx expo-doctor
	- Result: 17/17 checks passed. No issues detected.

2. TypeScript No-Emit
	- Command: npx tsc --noEmit
	- Result: Completed with no type errors.

3. Expo Lint
	- Command: npx expo lint
	- Result: Completed with 0 errors and 1 warning.
	- Warning: src/hooks/useVoiceInteraction.ts line 424 (`estimatedSpeechMs` assigned but never used).

Issue summary from this troubleshooting pass:
<<<<<<< HEAD
1. Technical appendix now matches current 6-channel ConvLSTM preprocessing behavior in both wandering and destination modes.
2. Documentation now reflects no-intent channel-zero diagnostics and no-intent service shape/logit guardrails.
3. Runtime switch notes now reflect current implementation (mode-driven pipeline and constructor-driven intent injection).
4. No static analysis or configuration issues were reported by the three toolkits after this verification run.
=======
1. Verified ActiveCamera overlay Total latency now includes YOLO in addition to ConvLSTM preprocess and inference.
2. Verified destination Target label truncation is visual-only and does not alter route/TTS source text.
3. Synced README and this appendix with current 6-channel runtime behavior and overlay metric semantics.
4. Static checks passed except for one pre-existing lint warning in useVoiceInteraction.
>>>>>>> refactor/performance-metrics

---

End of technical appendix.
