# Changelog

All notable changes to this project are documented in this file.

## [1.0.9] - 2026-04-06

### Changed
- Synced ActiveCamera bottom buffer progress UI with effective frame accumulation used by early inference readiness.
- Added UI-side derived frame display logic so when physical buffer reaches early threshold, counter and bar reflect bootstrapped sequence readiness.

### Fixed
- Fixed mismatch where inference could start at 10 frames while the buffer counter still displayed raw accumulation only.

### Validation
- Ran `npx tsc --noEmit` (completed with no type errors).
- Ran `npx expo lint` (0 errors, 1 warning):
  - `src/hooks/useVoiceInteraction.ts`: unused variable warning (pre-existing).
- No hook dependency warnings were reported in capture-loop logic.

## [1.0.8] - 2026-04-06

### Added
- Added `truncateToFirstComma` in `src/utils/stringUtils.ts` for cleaner destination target text in the performance overlay.

### Changed
- Updated ActiveCamera performance metric order to: Capture -> Preprocess -> YOLO -> Inference (ConvLSTM) -> Total.
- Updated overlay total latency formula to include capture time: `lastCaptureTime + preprocessingTimeMs + yoloInferenceTime + inferenceTimeMs`.
- Applied first-comma display truncation to the destination target label in the overlay only.

### Fixed
- Fixed underreported overlay Total latency by including camera capture duration in the displayed end-to-end total.
- Fixed destination target overlay clutter by trimming to the first comma while preserving full underlying destination data.

### Validation
- Ran `npx tsc --noEmit` (completed with no type errors).
- Ran `npx expo lint` (0 errors, 1 warning):
  - `src/hooks/useVoiceInteraction.ts`: unused variable warning (pre-existing).
- No hook dependency warnings were reported.

### Documentation
- Updated `texts/DATA_DICTIONARY.txt` with:
  - Capture-inclusive overlay total formula.
  - New overlay metric order.
  - `truncateToFirstComma` utility definition and visual-only scope.
- Updated `texts/VERSIONS.txt` with:
  - Capture-inclusive overlay total formula.
  - `truncateToFirstComma` target display behavior.

## [1.0.7] - 2026-04-06

### Added
- Added string utility `truncateToSecondComma` in `src/utils/stringUtils.ts` for display-only destination text normalization.

### Changed
- Updated ActiveCamera performance overlay metric flow to: Capture -> Inference (ConvLSTM) -> Preprocess -> YOLO -> Total.
- Updated overlay total latency logic to display `preprocessingTimeMs + inferenceTimeMs + yoloInferenceTime`.
- Applied destination label truncation only in the performance overlay Target field to reduce visual clutter.
- Re-verified that the on-screen overlay Total metric includes YOLO latency.

### Fixed
- Fixed misleading total latency display that excluded YOLO latency from the on-screen Total metric.
- Fixed destination Target overlay clutter by trimming long addresses at the second comma while preserving full internal destination text.

### Validation
- Ran `npx tsc --noEmit` (completed with no type errors).
- Ran `npx expo-doctor` (17/17 checks passed).
- Ran `npx expo lint` (0 errors, 1 warning):
  - `src/hooks/useVoiceInteraction.ts`: unused variable warning (pre-existing).
- Re-ran full validation on 2026-04-06 after metric verification with the same results.

### Documentation
- Updated Technical Appendix entries in `texts/DATA_DICTIONARY.txt` and `texts/VERSIONS.txt` with:
  - Correct overlay Total latency formula.
  - New `truncateToSecondComma` utility behavior and scope boundary.

## [1.0.6] - 2026-04-05

### Fixed
- Wayfinding destination confirmation now transitions directly to ActiveCamera immediately after route fetch succeeds, instead of waiting for trailing TTS completion.
- Added a destination transition lock in Wayfinding so Skip/Back/listener auto-restart cannot interrupt the committed destination handoff.

### Validation
- Ran troubleshooting toolkit 1: `npx expo-doctor` (17/17 checks passed).
- Ran troubleshooting toolkit 2: `npx tsc --noEmit` (completed with no type errors).
- Ran troubleshooting toolkit 3: `npx expo lint` (completed with no lint errors or warnings).

### Documentation
- Updated Technical Appendix with the destination camera-transition fix and toolkit results.
- Updated Wayfinding technical documentation to reflect ActiveCamera destination handoff behavior.

## [1.0.5] - 2026-04-03

### Added
- Added release hardening notes and versioned release entry for the completed sprint.
- Added camera readiness guard documentation in ActiveCamera lifecycle handling.

### Changed
- Unified camera runtime remains route-driven through ActiveCamera with mode values:
  - wandering
  - destination
- Reduced runtime logging noise in camera and YOLO inference paths to improve production performance.
- Updated voice-listening cleanup in Wayfinding to clear auto-restart timers on cleanup.
- Improved Choice voice listener setup to remove any previous listener before registering a new callback.
- Updated technical reference documentation to reflect current source code truth.

### Fixed
- Fixed repeated camera-ready setup behavior by guarding one-time picture-size configuration.
- Fixed potential stale capture gating by using ref-backed readiness checks during frame capture.
- Fixed YOLO tensor-layout detection branch so non-transposed output can be handled correctly.
- Removed commented debug code blocks and temporary console.log statements in recently modified runtime files.

### Documentation
- Updated Technical Appendix to current architecture and runtime behavior.
- Added this changelog for sprint-to-release traceability.

## [1.0.4] - 2026-04-03

### Changed
- Introduced ActiveCamera unified mode routing and global intent toggle.
- Added image URI decode path for optimized camera frame processing.
