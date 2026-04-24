import { useCallback, useEffect, useRef, useState } from 'react';
import { Audio, InterruptionModeIOS } from 'expo-av';
import * as Haptics from 'expo-haptics';
import * as Speech from 'expo-speech';
import { ExpoSpeechRecognitionModule } from 'expo-speech-recognition';
import * as Vosk from 'react-native-vosk';
import {
  logAudioDebugEarconTriggered,
  logAudioDebugTtsFinished,
  logAudioDebugTtsStart,
  logAudioDebugVoiceListenerStatus,
} from '../services/audioDebugLogger';

interface SpeakMessageOptions {
  message: string;
  language?: string;
  statusWhileSpeaking?: string;
  playEarcon?: boolean;
  onDone?: () => void;
  onError?: () => void;
}

interface TransitionToListeningOptions {
  statusWhileListening?: string;
  onListeningReady?: () => void | Promise<void>;
  emitCue?: boolean;
}

interface SpeakThenListenOptions extends SpeakMessageOptions, TransitionToListeningOptions {
  delayMs?: number;
}

interface VoskListeningOptions {
  grammar: string[];
  statusWhileListening?: string;
  onResult: (result: string) => void | Promise<void>;
  onErrorMessage?: string;
  emitCueOnStart?: boolean;
}

interface ExpoListeningOptions {
  startOptions?: {
    lang?: string;
    interimResults?: boolean;
    continuous?: boolean;
    contextualStrings?: string[];
    androidIntentOptions?: Record<string, string>;
    iosTaskHint?: 'unspecified' | 'dictation' | 'search' | 'confirmation';
  };
  statusWhileListening?: string;
  onResult?: (event: { isFinal?: boolean; results?: { transcript?: string }[] }) => void | Promise<void>;
  onEnd?: () => void | Promise<void>;
  onError?: (event: { error?: string; message?: string }) => void;
  emitCueOnStart?: boolean;
<<<<<<< HEAD
=======
}
// Barge-in and timing constants
const TTS_HANDOFF_MS_PER_CHAR = 50;
const TTS_HANDOFF_BASE_MS = 700;
const TTS_HANDOFF_MIN_MS = 1500;
const TTS_HANDOFF_MAX_MS = 20000;
const TTS_TO_MIC_BUFFER_MS = 50;
const BARGE_IN_ARM_DELAY_MS = 650;
const MAX_BARGE_IN_COMMAND_WORDS = 3;
function getErrorMessage(error: unknown): string {
  if (error instanceof Error && error.message) {
    return error.message;
  }
  if (typeof error === 'string' && error.length > 0) {
    return error;
  }
  if (typeof error === 'object' && error !== null && 'message' in error) {
    const candidate = (error as { message?: unknown }).message;
    if (typeof candidate === 'string' && candidate.length > 0) {
      return candidate;
    }
  }
  return String(error);
}

function normalizeTranscript(transcript: string): string {
  return transcript
    .toLowerCase()
    .replace(/[^a-z0-9\s]/g, ' ')
    .replace(/\s+/g, ' ')
    .trim();
}

function isBargeInCommandTranscript(transcript: string): boolean {
  const normalized = normalizeTranscript(transcript);
  if (!normalized) {
    return false;
  }
  const words = normalized.split(' ').filter(Boolean);
  if (!words.length || words.length > MAX_BARGE_IN_COMMAND_WORDS) {
    return false;
  }
  const compact = words.join('');
  const isWakeWord = compact.includes('eluseedate');
  const hasStopWord = words.includes('skip') || words.includes('stop');
  if (isWakeWord) {
    return true;
  }
  return words.length === 1 && hasStopWord;
>>>>>>> EluSEEdate-v1
}

interface UseVoiceInteractionOptions {
  initialVoiceStatus?: string;
  defaultLanguage?: string;
  listeningDelayMs?: number;
  ttsEarconEnabled?: boolean;
}

type ListenerWithRemove = { remove: () => void };
type ListenerEngine = 'vosk' | 'expo';

const TTS_HANDOFF_MS_PER_CHAR = 50;
const TTS_HANDOFF_BASE_MS = 700;
const TTS_HANDOFF_MIN_MS = 1500;
const TTS_HANDOFF_MAX_MS = 20000;
const TTS_TO_MIC_BUFFER_MS = 50;

let voiceInteractionInstanceCounter = 0;
let speechOwnerId: number | null = null;
let voskOwnerId: number | null = null;
let expoOwnerId: number | null = null;
let globalVoskResultListener: ListenerWithRemove | null = null;
let globalExpoListeners: ListenerWithRemove[] = [];
let pingSoundSingleton: Audio.Sound | null = null;
let pingSoundLoader: Promise<Audio.Sound | null> | null = null;

function removeListenerSafe(listener: ListenerWithRemove | null | undefined): void {
  if (!listener) {
    return;
  }

  try {
    listener.remove();
  } catch {
    // Ignore listener cleanup errors.
  }
}

function clearGlobalExpoListeners(): void {
  globalExpoListeners.forEach((listener) => {
    removeListenerSafe(listener);
  });
  globalExpoListeners = [];
}

function getErrorMessage(error: unknown): string {
  if (error instanceof Error && error.message) {
    return error.message;
  }

  if (typeof error === 'string' && error.length > 0) {
    return error;
  }

  if (typeof error === 'object' && error !== null && 'message' in error) {
    const candidate = (error as { message?: unknown }).message;
    if (typeof candidate === 'string' && candidate.length > 0) {
      return candidate;
    }
  }

  return String(error);
}

async function getPingSoundSingleton(): Promise<Audio.Sound | null> {
  if (pingSoundSingleton) {
    return pingSoundSingleton;
  }

  if (!pingSoundLoader) {
    pingSoundLoader = Audio.Sound.createAsync(
      require('../../assets/sounds/ping.wav'),
      { shouldPlay: false, volume: 1.0 },
    )
      .then(({ sound }) => {
        pingSoundSingleton = sound;
        return sound;
      })
      .catch((error: unknown) => {
        console.warn(`[AUDIO-TRACE] Earcon preload failed: ${getErrorMessage(error)}`);
        return null;
      });
  }

  return pingSoundLoader;
}

export function useVoiceInteraction(options?: UseVoiceInteractionOptions) {
  const [isListening, setIsListening] = useState(false);
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [readyToListen, setReadyToListen] = useState(false);
  const [voiceStatus, setVoiceStatus] = useState(options?.initialVoiceStatus ?? 'Initializing...');

  const defaultLanguage = options?.defaultLanguage ?? 'en-US';
  const defaultListeningDelayMs = options?.listeningDelayMs ?? 1000;
  const ttsEarconEnabled = options?.ttsEarconEnabled ?? true;
  const interactionIdRef = useRef(++voiceInteractionInstanceCounter);

  const readyTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const handoffFallbackTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const handoffSequenceRef = useRef(0);
  const pendingListeningRef = useRef<TransitionToListeningOptions | null>(null);
  const speechCallIdRef = useRef(0);
  const voskResultListenerRef = useRef<ListenerWithRemove | null>(null);
  const expoListenersRef = useRef<ListenerWithRemove[]>([]);
  const activeListenerEngineRef = useRef<ListenerEngine | null>(null);
  const isVoskListeningRef = useRef(false);
  const isExpoListeningRef = useRef(false);
  const handoffFallbackTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const handoffSequenceRef = useRef(0);
  const speechStartedAtRef = useRef<number | null>(null);

  const clearReadyTimer = useCallback(() => {
    if (readyTimerRef.current) {
      clearTimeout(readyTimerRef.current);
      readyTimerRef.current = null;
    }

    if (handoffFallbackTimerRef.current) {
      clearTimeout(handoffFallbackTimerRef.current);
      handoffFallbackTimerRef.current = null;
    }
  }, []);

  const estimateSpeechDurationMs = useCallback((message: string): number => {
    const estimated = TTS_HANDOFF_BASE_MS + (message.trim().length * TTS_HANDOFF_MS_PER_CHAR);
    return Math.max(TTS_HANDOFF_MIN_MS, Math.min(TTS_HANDOFF_MAX_MS, estimated));
  }, []);

  const estimateSpeechDurationMs = useCallback((message: string): number => {
    const estimated = TTS_HANDOFF_BASE_MS + (message.trim().length * TTS_HANDOFF_MS_PER_CHAR);
    return Math.max(TTS_HANDOFF_MIN_MS, Math.min(TTS_HANDOFF_MAX_MS, estimated));
  }, []);

  const playTtsEarcon = useCallback(async () => {
    const pingSound = await getPingSoundSingleton();
    if (!pingSound) {
      return;
    }
    try {
      logAudioDebugEarconTriggered('ping.wav');
      await pingSound.replayAsync();
    } catch (error: unknown) {
      console.warn(`[AUDIO-TRACE] Earcon playback failed: ${getErrorMessage(error)}`);
    }
  }, []);

  const stopVoskListening = useCallback(async () => {
    const ownsVosk = voskOwnerId === interactionIdRef.current;

    if (voskResultListenerRef.current) {
      removeListenerSafe(voskResultListenerRef.current);

      if (globalVoskResultListener === voskResultListenerRef.current) {
        globalVoskResultListener = null;
      }

      voskResultListenerRef.current = null;
    }

    if (ownsVosk && globalVoskResultListener) {
      removeListenerSafe(globalVoskResultListener);
      globalVoskResultListener = null;
    }

    if (ownsVosk) {
      try {
        await Vosk.stop();
      } catch {
        // Ignore stop failures during cleanup.
      }

      voskOwnerId = null;
    }

    if (isVoskListeningRef.current) {
      logAudioDebugVoiceListenerStatus('Inactive', 'Vosk');
    }

    isVoskListeningRef.current = false;
    if (activeListenerEngineRef.current === 'vosk') {
      activeListenerEngineRef.current = null;
    }

    setIsListening(false);
  }, []);

  const stopExpoListening = useCallback(async () => {
    const ownsExpo = expoOwnerId === interactionIdRef.current;

    expoListenersRef.current.forEach((listener) => {
      removeListenerSafe(listener);
    });
    expoListenersRef.current = [];

    if (ownsExpo) {
      try {
        await Promise.resolve(ExpoSpeechRecognitionModule.abort());
      } catch {
        // Ignore stop failures during cleanup.
      }

      clearGlobalExpoListeners();
      expoOwnerId = null;
    }

    if (isExpoListeningRef.current) {
      logAudioDebugVoiceListenerStatus('Inactive', 'ExpoSpeechRecognition');
    }

    isExpoListeningRef.current = false;
    if (activeListenerEngineRef.current === 'expo') {
      activeListenerEngineRef.current = null;
    }

    setIsListening(false);
  }, []);

  const emitListeningCue = useCallback(async () => {
    await Promise.allSettled([
      Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Medium),
    ]);
  }, []);

  const transitionToListening = useCallback(async (listeningOptions?: TransitionToListeningOptions) => {
    clearReadyTimer();
    pendingListeningRef.current = null;

    setIsSpeaking(false);
    setReadyToListen(true);

    if (listeningOptions?.statusWhileListening) {
      setVoiceStatus(listeningOptions.statusWhileListening);
    }

    if (listeningOptions?.emitCue === true) {
      await emitListeningCue();
    }

    if (listeningOptions?.onListeningReady) {
      await listeningOptions.onListeningReady();
    }
  }, [clearReadyTimer, emitListeningCue]);

  const speakMessage = useCallback((speakOptions: SpeakMessageOptions) => {
    clearReadyTimer();
    pendingListeningRef.current = null;
    setReadyToListen(false);

    if (speakOptions.statusWhileSpeaking) {
      setVoiceStatus(speakOptions.statusWhileSpeaking);
    }

    setIsSpeaking(true);

    const runSpeech = async () => {
      const callId = ++speechCallIdRef.current;
      const previousSpeechOwnerId = speechOwnerId;
      speechOwnerId = interactionIdRef.current;

      if (
        previousSpeechOwnerId !== null
        && previousSpeechOwnerId !== interactionIdRef.current
      ) {
        try {
          await Speech.stop();
        } catch {
          // Ignore interruptions while ownership is transferred.
        }
      }

      if (ttsEarconEnabled && speakOptions.playEarcon !== false) {
        await playTtsEarcon();
      }

      let didFinalize = false;

      const finalizeSpeech = (callback?: () => void) => {
        if (speechCallIdRef.current !== callId) {
          return;
        }

        if (speechOwnerId !== interactionIdRef.current) {
          return;
        }

        if (!didFinalize) {
          didFinalize = true;
          logAudioDebugTtsFinished(speakOptions.message);
        }

        speechOwnerId = null;
        setIsSpeaking(false);

        if (callback) {
          callback();
        }
      };

      try {
        logAudioDebugTtsStart(speakOptions.message);

        Speech.speak(speakOptions.message, {
          language: speakOptions.language ?? defaultLanguage,
          onStart: () => {
            if (speechOwnerId !== interactionIdRef.current) {
              return;
            }

            setIsSpeaking(true);
          },
          onDone: () => {
            finalizeSpeech(() => {
              if (speakOptions.onDone) {
                speakOptions.onDone();
              }
            });
          },
          onStopped: () => {
            finalizeSpeech();
          },
          onError: () => {
            if (speechOwnerId !== interactionIdRef.current) {
              return;
            }

            console.error('[ERROR] TTS playback callback error (expo-speech onError)');
            speechOwnerId = null;
            setIsSpeaking(false);

            if (speakOptions.onError) {
              speakOptions.onError();
            }
          },
        });
      } catch (error: unknown) {
        if (speechOwnerId === interactionIdRef.current) {
          speechOwnerId = null;
        }

        console.error(`[ERROR] TTS initialization failed: ${getErrorMessage(error)}`);
        setIsSpeaking(false);

        if (speakOptions.onError) {
          speakOptions.onError();
        }
      }
    };

    void runSpeech();
  }, [clearReadyTimer, defaultLanguage, playTtsEarcon, ttsEarconEnabled]);

  const speakThenListen = useCallback((speakOptions: SpeakThenListenOptions) => {
    const listeningOptions: TransitionToListeningOptions = {
      statusWhileListening: speakOptions.statusWhileListening,
      onListeningReady: speakOptions.onListeningReady,
      emitCue: false,
    };
    const handoffSequence = ++handoffSequenceRef.current;
    const delayMs = Math.max(0, speakOptions.delayMs ?? defaultListeningDelayMs);
    const handoffDelayMs = delayMs + TTS_TO_MIC_BUFFER_MS;
    const estimatedSpeechMs = estimateSpeechDurationMs(speakOptions.message);
    let handoffCompleted = false;

    const completeHandoff = (source: 'onDone' | 'onError' | 'fallback') => {
      if (handoffCompleted || handoffSequenceRef.current !== handoffSequence) {
        return;
      }
<<<<<<< HEAD

      handoffCompleted = true;

=======
      handoffCompleted = true;
>>>>>>> EluSEEdate-v1
      if (source === 'onDone') {
        clearReadyTimer();
        readyTimerRef.current = setTimeout(() => {
          if (handoffSequenceRef.current !== handoffSequence) {
            return;
          }
<<<<<<< HEAD

=======
>>>>>>> EluSEEdate-v1
          void transitionToListening(pendingListeningRef.current ?? listeningOptions);
        }, handoffDelayMs);
        return;
      }
<<<<<<< HEAD

      clearReadyTimer();

=======
      clearReadyTimer();
>>>>>>> EluSEEdate-v1
      void (async () => {
        if (source === 'fallback' && speechOwnerId === interactionIdRef.current) {
          try {
            await Speech.stop();
          } catch (error: unknown) {
            console.warn(`[AUDIO-TRACE] Fallback Speech.stop failed: ${getErrorMessage(error)}`);
          }
        }
<<<<<<< HEAD

=======
>>>>>>> EluSEEdate-v1
        readyTimerRef.current = setTimeout(() => {
          if (handoffSequenceRef.current !== handoffSequence) {
            return;
          }
<<<<<<< HEAD

=======
>>>>>>> EluSEEdate-v1
          void transitionToListening(pendingListeningRef.current ?? listeningOptions);
        }, TTS_TO_MIC_BUFFER_MS);
      })();
    };

    pendingListeningRef.current = listeningOptions;
    clearReadyTimer();
<<<<<<< HEAD
    handoffFallbackTimerRef.current = setTimeout(() => {
      if (handoffSequenceRef.current !== handoffSequence || handoffCompleted) {
        return;
      }

      console.warn('[AUDIO-TRACE] TTS onDone fallback triggered; forcing listening handoff');
      completeHandoff('fallback');
    }, estimatedSpeechMs + delayMs);
=======
>>>>>>> EluSEEdate-v1

    speakMessage({
      message: speakOptions.message,
      language: speakOptions.language,
      statusWhileSpeaking: speakOptions.statusWhileSpeaking,
      onDone: () => {
        completeHandoff('onDone');
<<<<<<< HEAD

=======
>>>>>>> EluSEEdate-v1
        if (speakOptions.onDone) {
          speakOptions.onDone();
        }
      },
      onError: () => {
        completeHandoff('onError');
        if (speakOptions.onError) {
          speakOptions.onError();
        }
      },
    });
<<<<<<< HEAD
=======

    // Fallback only if TTS is truly stuck (20s max)
    handoffFallbackTimerRef.current = setTimeout(() => {
      if (handoffSequenceRef.current !== handoffSequence || handoffCompleted) {
        return;
      }
      console.warn('[AUDIO-TRACE] TTS onDone fallback triggered; forcing listening handoff');
      completeHandoff('fallback');
    }, 20000); // 20 seconds max, not estimatedSpeechMs
>>>>>>> EluSEEdate-v1
  }, [clearReadyTimer, defaultListeningDelayMs, estimateSpeechDurationMs, speakMessage, transitionToListening]);

  const skipSpeech = useCallback(async () => {
    clearReadyTimer();
    let interruptedSpeech = false;
<<<<<<< HEAD

=======
>>>>>>> EluSEEdate-v1
    if (speechOwnerId === interactionIdRef.current) {
      try {
        await Speech.stop();
        interruptedSpeech = true;
      } catch {
        // Ignore stop failures during user skip.
      }
    }
<<<<<<< HEAD

    if (interruptedSpeech) {
=======
    if (interruptedSpeech) {
      speechStartedAtRef.current = null;
      setIsSpeaking(false);
>>>>>>> EluSEEdate-v1
      await new Promise<void>((resolve) => {
        setTimeout(() => resolve(), TTS_TO_MIC_BUFFER_MS);
      });
    }
<<<<<<< HEAD

=======
>>>>>>> EluSEEdate-v1
    const pendingListening = pendingListeningRef.current;
    await transitionToListening(
      pendingListening ?? {
        statusWhileListening: voiceStatus,
        emitCue: false,
      }
    );
  }, [clearReadyTimer, transitionToListening, voiceStatus]);
  const tryHandleBargeIn = useCallback(async (transcript: string): Promise<boolean> => {
    if (speechOwnerId !== interactionIdRef.current || !isSpeaking) {
      return false;
    }
    const startedAt = speechStartedAtRef.current;
    if (!startedAt || (Date.now() - startedAt) < BARGE_IN_ARM_DELAY_MS) {
      return false;
    }
    if (!isBargeInCommandTranscript(transcript)) {
      return false;
    }
    await skipSpeech();
    return true;
  }, [isSpeaking, skipSpeech]);

  const startVoskListening = useCallback(async (listeningOptions: VoskListeningOptions) => {
    await stopVoskListening();

    if (globalVoskResultListener) {
      removeListenerSafe(globalVoskResultListener);
      globalVoskResultListener = null;
    }

    if (voskOwnerId !== null && voskOwnerId !== interactionIdRef.current) {
      try {
        await Vosk.stop();
      } catch {
        // Ignore ownership handoff failures.
      }
    }

    voskOwnerId = interactionIdRef.current;

    try {
      await Vosk.start({ grammar: listeningOptions.grammar });

      voskResultListenerRef.current = Vosk.onResult((result: string) => {
        if (voskOwnerId !== interactionIdRef.current) {
          return;
        }

        void listeningOptions.onResult(result);
      }) as ListenerWithRemove;

      globalVoskResultListener = voskResultListenerRef.current;
      activeListenerEngineRef.current = 'vosk';
      isVoskListeningRef.current = true;
      isExpoListeningRef.current = false;

      setIsListening(true);
      logAudioDebugVoiceListenerStatus('Active', 'Vosk');

      if (listeningOptions.statusWhileListening) {
        setVoiceStatus(listeningOptions.statusWhileListening);
      }
      if (listeningOptions.emitCueOnStart !== false) {
        await emitListeningCue();
      }

      if (listeningOptions.emitCueOnStart !== false) {
        await emitListeningCue();
      }

      return true;
    } catch (error: any) {
      if (voskOwnerId === interactionIdRef.current) {
        voskOwnerId = null;
      }

      activeListenerEngineRef.current = null;
      isVoskListeningRef.current = false;
      setIsListening(false);
      console.error(`[ERROR] Vosk start failed: ${error?.message ? String(error.message) : String(error)}`);
      logAudioDebugVoiceListenerStatus(
        'Error',
        `Vosk${error?.message ? `: ${String(error.message)}` : ''}`,
      );

      if (listeningOptions.onErrorMessage) {
        setVoiceStatus(listeningOptions.onErrorMessage);
      } else if (error?.message?.toLowerCase?.().includes('permission')) {
        setVoiceStatus('Microphone permission denied');
      } else {
        setVoiceStatus('Voice command disabled');
      }

      return false;
    }
  }, [emitListeningCue, stopVoskListening]);

  const startExpoListening = useCallback(async (listeningOptions: ExpoListeningOptions) => {
    await stopExpoListening();

    if (expoOwnerId !== null && expoOwnerId !== interactionIdRef.current) {
      try {
        await Promise.resolve(ExpoSpeechRecognitionModule.abort());
      } catch {
        // Ignore ownership handoff failures.
      }

      clearGlobalExpoListeners();
    }

    expoOwnerId = interactionIdRef.current;

    try {
      const permission = await ExpoSpeechRecognitionModule.requestPermissionsAsync();
      if (!permission.granted) {
        expoOwnerId = null;
        console.error('[ERROR] ExpoSpeechRecognition permission denied');
        logAudioDebugVoiceListenerStatus('Error', 'ExpoSpeechRecognition: Permission denied');
        setVoiceStatus('Microphone permission denied');
        return false;
      }

      const resultListener = ExpoSpeechRecognitionModule.addListener('result', (event) => {
        if (expoOwnerId !== interactionIdRef.current) {
          return;
        }

        if (listeningOptions.onResult) {
          void listeningOptions.onResult(event);
        }
      });

      const endListener = ExpoSpeechRecognitionModule.addListener('end', () => {
        if (expoOwnerId !== interactionIdRef.current) {
          return;
        }

        expoOwnerId = null;
        clearGlobalExpoListeners();
        activeListenerEngineRef.current = null;
        if (isExpoListeningRef.current) {
          logAudioDebugVoiceListenerStatus('Inactive', 'ExpoSpeechRecognition');
        }

        isExpoListeningRef.current = false;
        setIsListening(false);

        if (listeningOptions.onEnd) {
          void listeningOptions.onEnd();
        }
      });

      const errorListener = ExpoSpeechRecognitionModule.addListener('error', (event) => {
        if (expoOwnerId !== interactionIdRef.current) {
          return;
        }

<<<<<<< HEAD
        console.error(
          `[ERROR] ExpoSpeechRecognition runtime error: ${event.error ? String(event.error) : 'unknown'}${event.message ? ` | ${String(event.message)}` : ''}`,
        );

        expoOwnerId = null;
        clearGlobalExpoListeners();
        activeListenerEngineRef.current = null;
        isExpoListeningRef.current = false;
        setIsListening(false);
=======
        // "no-speech" and "aborted" are non-fatal: the user simply didn't
        // speak in time or the session was intentionally stopped.  Let the
        // "end" listener handle cleanup so the restart loop in onEnd still
        // fires and the user gets another chance to speak.
        const nonFatal = event.error === 'no-speech' || event.error === 'aborted';
>>>>>>> EluSEEdate-v1

        if (!nonFatal) {
          expoOwnerId = null;
          clearGlobalExpoListeners();
          activeListenerEngineRef.current = null;
          isExpoListeningRef.current = false;
          setIsListening(false);

          logAudioDebugVoiceListenerStatus(
            'Error',
            `ExpoSpeechRecognition${event.error ? `: ${String(event.error)}` : ''}`,
          );
        }

        if (listeningOptions.onError) {
          listeningOptions.onError(event);
        }
      });

      expoListenersRef.current = [resultListener, endListener, errorListener];
      clearGlobalExpoListeners();
      globalExpoListeners = [...expoListenersRef.current];

      await Promise.resolve(ExpoSpeechRecognitionModule.start(
        listeningOptions.startOptions ?? {
          lang: defaultLanguage,
          interimResults: false,
          continuous: false,
        }
      ));

      activeListenerEngineRef.current = 'expo';
      isExpoListeningRef.current = true;
      isVoskListeningRef.current = false;
      setIsListening(true);
      logAudioDebugVoiceListenerStatus('Active', 'ExpoSpeechRecognition');

      if (listeningOptions.statusWhileListening) {
        setVoiceStatus(listeningOptions.statusWhileListening);
      }
      if (listeningOptions.emitCueOnStart !== false) {
        await emitListeningCue();
      }

      if (listeningOptions.emitCueOnStart !== false) {
        await emitListeningCue();
      }

      return true;
    } catch (error: any) {
      if (expoOwnerId === interactionIdRef.current) {
        expoOwnerId = null;
      }

      activeListenerEngineRef.current = null;
      isExpoListeningRef.current = false;
      await stopExpoListening();
      setIsListening(false);
      setVoiceStatus('Voice command disabled');
      console.error(`[ERROR] ExpoSpeechRecognition start failed: ${error?.message ? String(error.message) : String(error)}`);

      logAudioDebugVoiceListenerStatus(
        'Error',
        `ExpoSpeechRecognition${error?.message ? `: ${String(error.message)}` : ''}`,
      );

      return false;
    }
  }, [defaultLanguage, emitListeningCue, stopExpoListening]);

  const stopAllVoiceActivity = useCallback(async () => {
    clearReadyTimer();
    pendingListeningRef.current = null;
    setReadyToListen(false);

    const ownsSpeech = speechOwnerId === interactionIdRef.current;

    await Promise.allSettled([
      stopVoskListening(),
      stopExpoListening(),
      ownsSpeech ? Speech.stop() : Promise.resolve(),
    ]);

    if (ownsSpeech) {
      speechOwnerId = null;
    }

    setIsSpeaking(false);
  }, [clearReadyTimer, stopExpoListening, stopVoskListening]);

  useEffect(() => {
    void Audio.setAudioModeAsync({
      allowsRecordingIOS: true,
      playsInSilentModeIOS: true,
      shouldDuckAndroid: true,
<<<<<<< HEAD
      staysActiveInBackground: false,
      playThroughEarpieceAndroid: false,
=======
      staysActiveInBackground: true,
      playThroughEarpieceAndroid: false,
      interruptionModeIOS: InterruptionModeIOS.DoNotMix,
>>>>>>> EluSEEdate-v1
    }).catch((error: unknown) => {
      console.error(`[ERROR] Failed to configure voice audio mode: ${getErrorMessage(error)}`);
    });

    void getPingSoundSingleton();

    return () => {
      clearReadyTimer();
      if (handoffFallbackTimerRef.current) {
        clearTimeout(handoffFallbackTimerRef.current);
        handoffFallbackTimerRef.current = null;
      }
      pendingListeningRef.current = null;
      void stopAllVoiceActivity();
    };
  }, [clearReadyTimer, stopAllVoiceActivity]);

  return {
    isListening,
    isSpeaking,
    readyToListen,
    voiceStatus,
    setVoiceStatus,
    speakMessage,
    speakThenListen,
    transitionToListening,
    skipSpeech,
    tryHandleBargeIn,
    startVoskListening,
    stopVoskListening,
    startExpoListening,
    stopExpoListening,
    stopAllVoiceActivity,
  };
}
