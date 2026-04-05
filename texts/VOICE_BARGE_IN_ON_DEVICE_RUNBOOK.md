# Voice Barge-In On-Device Test Runbook

Purpose:
Validate skip/stop/eluseedate interruption behavior during active TTS, plus stress scenarios for rapid interruptions, fallback handoff reliability, and fast screen transitions.

## 1. Preconditions

1. Install and launch a Development Build on Android device (not Expo Go).
2. Grant microphone, location, and camera permissions.
3. Set media volume to at least 60%.
4. Use a quiet room (minimal background speech).
5. Keep phone language and recognizer language set to English (US).
6. Start from app cold launch (fully force-close, then reopen).

## 2. Test Data To Record

- Device model:
- Android version:
- App version/build:
- Date/time:
- Tester name:

## 3. Pass/Fail Rules

- PASS: expected behavior occurs exactly as described.
- FAIL: wrong behavior, delayed behavior beyond timeout, freeze, crash, or wrong navigation.
- BLOCKED: test cannot run because prerequisite failed.

Timeout rules:
- Interruption reaction timeout: 1.0 second after recognized phrase.
- Post-prompt listener readiness timeout: 5 seconds after TTS completion.

## 4. Strict Test Checklist

### T1 - MainMenu barge-in with "skip"

Steps:
1. Cold-launch app and wait for MainMenu greeting TTS to begin.
2. About 1 second into speech, say: skip.

Expected:
1. Current TTS stops quickly.
2. App stays on MainMenu.
3. Voice status indicates interruption or skipped audio.
4. Saying start afterward still navigates to Choice.

Result: [ ] PASS  [ ] FAIL  [ ] BLOCKED
Notes:

### T2 - MainMenu barge-in with "stop"

Steps:
1. Relaunch app to replay greeting.
2. About 1 second into greeting, say: stop.

Expected:
1. Same behavior as T1.
2. Saying start still works after interruption.

Result: [ ] PASS  [ ] FAIL  [ ] BLOCKED
Notes:

### T3 - MainMenu wake-word barge-in with "eluseedate"

Steps:
1. Relaunch app to replay greeting.
2. About 1 second into greeting, say: eluseedate.

Expected:
1. Current TTS interrupts.
2. MainMenu remains stable and listening.
3. Saying start still navigates to Choice.

Result: [ ] PASS  [ ] FAIL  [ ] BLOCKED
Notes:

### T4 - Choice barge-in with "skip"

Steps:
1. From MainMenu, say start to enter Choice.
2. While Choice instruction TTS is speaking, say: skip.

Expected:
1. Choice prompt interrupts.
2. App remains on Choice.
3. Saying wandering or destination still works.

Result: [ ] PASS  [ ] FAIL  [ ] BLOCKED
Notes:

### T5 - Choice barge-in with "stop"

Steps:
1. Re-enter Choice prompt (navigate away/back if needed).
2. During TTS, say: stop.

Expected:
1. Same behavior as T4.
2. No duplicate prompt playback.

Result: [ ] PASS  [ ] FAIL  [ ] BLOCKED
Notes:

### T6 - Choice wake-word barge-in with "eluseedate"

Steps:
1. Re-enter Choice prompt.
2. During TTS, say: eluseedate.

Expected:
1. Prompt interrupts.
2. Choice command handling remains responsive.

Result: [ ] PASS  [ ] FAIL  [ ] BLOCKED
Notes:

### T7 - Wayfinding ask-location prompt barge-in with "stop"

Steps:
1. From Choice, go to Destination (Wayfinding).
2. During ask-location prompt TTS, say: stop.

Expected:
1. Prompt interrupts.
2. App remains in ask-location phase.
3. You can still say a place name or back.

Result: [ ] PASS  [ ] FAIL  [ ] BLOCKED
Notes:

### T8 - Wayfinding confirmation prompt barge-in with "eluseedate"

Steps:
1. In Wayfinding, say a valid destination (example: Ayala Center Cebu).
2. When confirmation TTS starts, say: eluseedate.

Expected:
1. Confirmation TTS interrupts.
2. App stays in confirming phase.
3. Saying yes, no, or back still works.

Result: [ ] PASS  [ ] FAIL  [ ] BLOCKED
Notes:

### T9 - Rapid skip-spam during confirmation

Steps:
1. Trigger confirmation prompt again.
2. During active TTS, rapidly speak: skip skip skip stop stop.

Expected:
1. No crash, freeze, or navigation glitch.
2. Prompt is interrupted once and app remains stable.
3. Confirming commands still work after spam.

Result: [ ] PASS  [ ] FAIL  [ ] BLOCKED
Notes:

### T10 - Delayed onDone fallback proxy check

Steps:
1. Run three prompt cycles (MainMenu, Choice, Wayfinding).
2. For each cycle, let TTS finish naturally (do not interrupt).
3. After TTS ends, wait up to 5 seconds, then speak a valid command.

Expected:
1. Listener is ready within timeout in all cycles.
2. Commands are recognized after TTS completion.
3. No dead state where app never returns to listening.

Result: [ ] PASS  [ ] FAIL  [ ] BLOCKED
Notes:

### T11 - Fast screen-switch stress path

Steps:
1. MainMenu: say start.
2. Choice: say destination.
3. Wayfinding: immediately say back.
4. Choice: say wandering.
5. ActiveCamera opens.
6. Back out to Choice, then back to MainMenu.

Expected:
1. No stale voice command bleed-through from prior screen.
2. No duplicated recognizer sessions.
3. Navigation remains deterministic and stable.

Result: [ ] PASS  [ ] FAIL  [ ] BLOCKED
Notes:

### T12 - Negative phrase safety check

Steps:
1. During active prompt on any voice screen, say: please stop.
2. During active prompt again, say: skip audio.

Expected:
1. Neither phrase is required to trigger barge-in matcher.
2. App should continue normal prompt/listener behavior unless exact supported phrase is recognized.

Result: [ ] PASS  [ ] FAIL  [ ] BLOCKED
Notes:

## 5. Final Sign-Off

- Overall result: [ ] PASS  [ ] FAIL  [ ] BLOCKED
- Critical failures found:
- Suggested fixes:
- Retest required: [ ] Yes  [ ] No
