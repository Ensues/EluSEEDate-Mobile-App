/**
 * Wayfinding Screen - EluSEEdate
 *
 * Voice-first destination picker for visually impaired users.
 * The entire interaction flow is driven by TTS (expo-speech) and STT (expo-speech-recognition).
 *
 * Flow:
 *  1. GPS locates the user (expo-location).
 *  2. TTS: "Choose your location. Say the name of your destination."
 *  3. STT listens for a place name (free-form via expo-speech-recognition).
 *  4. Geocode the spoken text → get an address (Nominatim geocoding service).
 *  5. TTS reads it back: "Did you mean <address>? Say yes to confirm, no to try again,
 *     back to return, or skip to interrupt audio."
 *  6. STT listens for "yes" / "no" / "back" / "skip".
 *     - yes  → validate 10 km radius → navigate to Destination (IntentScreen).
 *     - no   → clear & loop back to step 2.
 *     - back → return to ChoiceScreen.
 *     - skip → stop current prompt and keep listening.
 *  7. If out of bounds (> 10 km), TTS informs the user and loops back to step 2.

 */

import React, { useEffect, useState, useRef, useCallback } from 'react';
import {
  View,
  Text,
  StyleSheet,
  StatusBar,
  ActivityIndicator,
  TouchableOpacity,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { NativeStackNavigationProp } from '@react-navigation/native-stack';
import { useFocusEffect } from '@react-navigation/native';
import * as Location from 'expo-location';
import { RootStackParamList } from '../navigation/types';
import { geocodeForward } from '../services/geocodingService';
import { fetchWalkingDirections } from '../services/directionsService';
import { useVoiceInteraction } from '../hooks/useVoiceInteraction';

// ---------- constants ----------
const MAX_RADIUS_KM = 10;

type WayfindingScreenProps = {
  navigation: NativeStackNavigationProp<RootStackParamList, 'Wayfinding'>;
};

type Coordinate = {
  latitude: number;
  longitude: number;
};

/**
 * Conversation phases:
 *  ask_location  – waiting for the user to say a place name
 *  confirming    – geocoded result read back, waiting for yes / no / back
 */
type Phase = 'ask_location' | 'confirming';

// ---------- helpers ----------

/** Haversine distance in km between two lat/lng points. */
function haversineKm(a: Coordinate, b: Coordinate): number {
  const toRad = (deg: number) => (deg * Math.PI) / 180;
  const R = 6371; // Earth radius in km
  const dLat = toRad(b.latitude - a.latitude);
  const dLng = toRad(b.longitude - a.longitude);
  const sinLat = Math.sin(dLat / 2);
  const sinLng = Math.sin(dLng / 2);
  const h =
    sinLat * sinLat +
    Math.cos(toRad(a.latitude)) * Math.cos(toRad(b.latitude)) * sinLng * sinLng;
  return 2 * R * Math.asin(Math.sqrt(h));
}

// ---------- component ----------

export default function WayfindingScreen({ navigation }: WayfindingScreenProps) {
  // ---- GPS state ----
  const [userLocation, setUserLocation] = useState<Coordinate | null>(null);
  const [loading, setLoading] = useState(true);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  // ---- Conversation state ----
  const [phase, setPhase] = useState<Phase>('ask_location');
  const [pendingCoord, setPendingCoord] = useState<Coordinate | null>(null);
  const [pendingLabel, setPendingLabel] = useState<string>('');
  const [pendingDistance, setPendingDistance] = useState<number>(0);

  const hasNavigatedRef = useRef(false);
  const destinationTransitionLockedRef = useRef(false);

  const {
    isListening,
    readyToListen,
    voiceStatus,
    setVoiceStatus,
    speakMessage,
    speakThenListen,
    skipSpeech,
    tryHandleBargeIn,
    startExpoListening,
    stopExpoListening,
    stopAllVoiceActivity,
  } = useVoiceInteraction({
    initialVoiceStatus: 'Initializing...',
    defaultLanguage: 'en-US',
    listeningDelayMs: 1000,
  });

  // ---- Request GPS permission & get current position ----
  useEffect(() => {
    let cancelled = false;

    (async () => {
      try {
        const { status } = await Location.requestForegroundPermissionsAsync();
        if (status !== 'granted') {
          setErrorMsg('Location permission denied');
          setLoading(false);
          return;
        }

        speakMessage({ message: 'Getting your location. Please wait.' });

        const loc = await Location.getCurrentPositionAsync({
          accuracy: Location.Accuracy.High,
        });

        if (!cancelled) {
          setUserLocation({
            latitude: loc.coords.latitude,
            longitude: loc.coords.longitude,
          });
          setLoading(false);
        }
      } catch (err) {
        if (!cancelled) {
          console.error('Location error:', err);
          const msg = 'Unable to get your location';
          setErrorMsg(msg);
          setLoading(false);
          speakMessage({ message: msg });
        }
      }
    })();

    return () => {
      cancelled = true;
    };
  }, [speakMessage]);

  // ---- TTS greeting after GPS ready ----
  useFocusEffect(
    useCallback(() => {
      if (loading || !userLocation) return;

      hasNavigatedRef.current = false;
      destinationTransitionLockedRef.current = false;
      setPhase('ask_location');
      setPendingCoord(null);
      setPendingLabel('');

      speakThenListen({
        message: 'Choose your location. Say the name of your destination. Or say back to return.',
        statusWhileSpeaking: 'Speaking instructions...',
        statusWhileListening: 'Say a place name, "Back", "Skip", or "Stop"',
      });

      return () => {
        void stopAllVoiceActivity();
      };
    }, [loading, speakThenListen, stopAllVoiceActivity, userLocation]),
  );

  // ================================================================
  //  Helper – speak a message then loop back to ask_location phase
  // ================================================================
  const restartAskLocation = useCallback((message: string) => {
    hasNavigatedRef.current = false;
    destinationTransitionLockedRef.current = false;
    setPendingCoord(null);
    setPendingLabel('');
    setPendingDistance(0);
    setPhase('ask_location');
    speakThenListen({
      message,
      statusWhileSpeaking: 'Speaking instructions...',
      statusWhileListening: 'Say a place name, "Back", "Skip", or "Stop"',
    });
  }, [speakThenListen]);

  // ================================================================
  //  Geocode a spoken place name, read it back and enter confirming
  // ================================================================
  const geocodePlace = useCallback(async (placeName: string) => {
    try {
      speakMessage({ message: `Looking up ${placeName}.` });

      const result = await geocodeForward(placeName);
      if (!result) {
        restartAskLocation('I could not find that place. Please say another destination.');
        return;
      }

      const coord: Coordinate = { latitude: result.latitude, longitude: result.longitude };
      const label = result.displayName;

      // Store candidate – do NOT commit yet
      setPendingCoord(coord);
      setPendingLabel(label);

      const dist = userLocation ? haversineKm(userLocation, coord) : 0;
      setPendingDistance(dist);

      // Read back for confirmation
      setPhase('confirming');

      speakThenListen({
        message: `Did you mean ${label}? It is ${dist.toFixed(1)} kilometres away. Say yes to confirm, no to try again, or back to return. You may also say skip or stop.`,
        statusWhileSpeaking: 'Speaking instructions...',
        statusWhileListening: 'Say "Yes", "No", "Back", "Skip", or "Stop"',
      });
    } catch (err) {
      console.error('Geocoding error:', err);
      restartAskLocation('I could not find that place. Please say another destination.');
    }
  }, [restartAskLocation, speakMessage, speakThenListen, userLocation]);

  // ================================================================
  //  Confirmation handlers
  // ================================================================

  /** User said "yes" – validate radius, fetch directions, then navigate or reject. */
  const handleConfirmYes = useCallback(async () => {
    if (!pendingCoord || !userLocation || destinationTransitionLockedRef.current || hasNavigatedRef.current) {
      return;
    }

    // Guard against duplicate "yes" results while we stop the current listener.
    hasNavigatedRef.current = true;

    await stopExpoListening();

    if (pendingDistance > MAX_RADIUS_KM) {
      hasNavigatedRef.current = false;
      restartAskLocation(
        `That location is ${pendingDistance.toFixed(1)} kilometres away. Out of bounds. The maximum walking radius is ${MAX_RADIUS_KM} kilometres. Please choose a closer destination.`,
      );
      return;
    }

    // Lock handoff only after explicit confirmation and route-fetch transition begins.
    destinationTransitionLockedRef.current = true;

    // Fetch walking directions from origin → destination
    speakMessage({ message: 'Destination confirmed. Fetching walking directions. Please wait.' });
    setVoiceStatus('Fetching route...');

    try {
      const directions = await fetchWalkingDirections(userLocation, pendingCoord);

      setVoiceStatus('Opening destination camera...');
      navigation.navigate('ActiveCamera', {
        mode: 'destination',
        origin: userLocation,
        destination: pendingCoord,
        destinationLabel: pendingLabel,
        routeSteps: directions.steps,
        totalDistanceMeters: directions.totalDistanceMeters,
        totalDurationSeconds: directions.totalDurationSeconds,
      });
    } catch (err) {
      console.error('Directions API error:', err);
      hasNavigatedRef.current = false;
      destinationTransitionLockedRef.current = false;
      restartAskLocation(
        'Could not fetch walking directions for that destination. Please try a different location.',
      );
    }
  }, [
    navigation,
    pendingCoord,
    pendingDistance,
    pendingLabel,
    restartAskLocation,
    setVoiceStatus,
    speakMessage,
    stopExpoListening,
    userLocation,
  ]);

  /** User said "no" – discard candidate and ask again. */
  const handleConfirmNo = useCallback(async () => {
    await stopExpoListening();
    restartAskLocation('Okay, say another destination.');
  }, [restartAskLocation, stopExpoListening]);

  // ================================================================
  //  Voice recognition – switches behaviour based on current phase
  //  Uses expo-speech-recognition (Google/Apple cloud speech) for
  //  accurate free-form place name recognition.
  // ================================================================
  useFocusEffect(
    useCallback(() => {
      let restartTimeout: ReturnType<typeof setTimeout> | null = null;

      const startListening = async () => {
        if (!readyToListen) return;

        await startExpoListening({
          statusWhileListening:
            phase === 'confirming'
              ? 'Say "Yes", "No", "Back", "Skip", or "Stop"'
              : 'Say a place name, "Back", "Skip", or "Stop"',
          startOptions: {
            lang: 'en-US',
            interimResults: false,
            continuous: false,
            ...(phase === 'confirming' && {
              contextualStrings: ['yes', 'no', 'back', 'skip', 'stop', 'eluseedate', 'elu see date'],
              androidIntentOptions: { EXTRA_LANGUAGE_MODEL: 'web_search' },
              iosTaskHint: 'confirmation',
            }),
          },
          onResult: async (event) => {
            if (!event.isFinal) return;
            const transcript = event.results?.[0]?.transcript ?? '';
            const lower = transcript.toLowerCase().trim();
            if (hasNavigatedRef.current || destinationTransitionLockedRef.current) return;

            if (await tryHandleBargeIn(lower)) {
              setVoiceStatus(
                phase === 'confirming'
                  ? 'Audio interrupted. Say "Yes", "No", or "Back"'
                  : 'Audio interrupted. Say a place name or "Back"',
              );
              return;
            }

            if (lower.includes('skip') || lower.includes('stop')) {
              await skipSpeech();
              setVoiceStatus(
                phase === 'confirming'
                  ? 'Audio skipped. Say "Yes", "No", or "Back"'
                  : 'Audio skipped. Say a place name or "Back"',
              );
              return;
            }

            // ---- "back" is always honoured ----
            if (lower.includes('back')) {
              hasNavigatedRef.current = true;
              await stopExpoListening();
              speakMessage({
                message: 'Going back',
                onDone: () => navigation.navigate('Choice'),
              });
              return;
            }

            if (phase === 'ask_location') {
              // Treat utterance as a place name
              if (lower.length > 1) {
                await stopExpoListening();
                setVoiceStatus('Looking up location...');
                await geocodePlace(lower);
              }
            } else if (phase === 'confirming') {
              if (lower.includes('yes')) {
                void handleConfirmYes();
              } else if (lower.includes('no')) {
                void handleConfirmNo();
              }
            }
          },
          onEnd: () => {
            if (!hasNavigatedRef.current && !destinationTransitionLockedRef.current && readyToListen) {
              restartTimeout = setTimeout(() => {
                void startListening();
              }, 500);
            }
          },
          onError: (event) => {
            // "aborted" and "no-speech" are expected during normal operation.
            if (event.error !== 'aborted' && event.error !== 'no-speech') {
              console.error(
                `[ERROR] Wayfinding speech recognition error: ${event.error ? String(event.error) : 'unknown'}${event.message ? ` | ${String(event.message)}` : ''}`,
              );
            }
          },
        });
      };

      if (readyToListen) {
        void startListening();
      }

      return () => {
        void stopExpoListening();
        if (restartTimeout) clearTimeout(restartTimeout);
      };
    }, [
      geocodePlace,
      handleConfirmNo,
      handleConfirmYes,
      navigation,
      phase,
      readyToListen,
      setVoiceStatus,
      skipSpeech,
      speakMessage,
      startExpoListening,
      stopExpoListening,
      tryHandleBargeIn,
    ]),
  );

  // ================================================================
  //  Render – voice-first UI for visually impaired users
  // ================================================================

  if (loading) {
    return (
      <SafeAreaView style={styles.container}>
        <StatusBar barStyle="light-content" backgroundColor="#000000" />
        <View style={styles.centerContainer}>
          <ActivityIndicator size="large" color="#ffffff" />
          <Text style={styles.statusText}>Getting your location…</Text>
        </View>
      </SafeAreaView>
    );
  }

  if (errorMsg) {
    return (
      <SafeAreaView style={styles.container}>
        <StatusBar barStyle="light-content" backgroundColor="#000000" />
        <View style={styles.centerContainer}>
          <Text style={styles.errorText}>{errorMsg}</Text>
          <Text style={styles.hintText}>Say Back to return</Text>
        </View>
      </SafeAreaView>
    );
  }

  return (
    <SafeAreaView style={styles.container}>
      <StatusBar barStyle="light-content" backgroundColor="#000000" />

      {/* Header */}
      <View style={styles.headerSection}>
        <Text style={styles.title}>Choose Your Location</Text>
      </View>

      {/* Centre – spoken feedback area */}
      <View style={styles.centerSection}>
        {phase === 'confirming' && pendingLabel ? (
          <>
            <Text style={styles.addressLabel}>{pendingLabel}</Text>
            <Text style={styles.distanceLabel}>
              {pendingDistance.toFixed(1)} km away
            </Text>
            <Text style={styles.promptText}>
              Say Yes to confirm{'\n'}Say No to try again
            </Text>
          </>
        ) : (
          <Text style={styles.promptText}>
            Say the name of your destination
          </Text>
        )}
      </View>

      {/* Voice status indicator */}
      <View style={styles.footerSection}>
        <View style={styles.voiceStatusContainer}>
          <View
            style={[
              styles.voiceIndicator,
              isListening && styles.voiceIndicatorActive,
            ]}
          />
          <Text style={styles.voiceStatusText}>{voiceStatus}</Text>
        </View>
        <TouchableOpacity
          style={styles.skipButton}
          onPress={() => {
            if (destinationTransitionLockedRef.current) {
              return;
            }

            setVoiceStatus(
              phase === 'confirming'
                ? 'Audio skipped. Say "Yes", "No", or "Back"'
                : 'Audio skipped. Say a place name or "Back"',
            );
            void skipSpeech();
          }}
          activeOpacity={0.7}
        >
          <Text style={styles.skipButtonText}>Skip Audio</Text>
        </TouchableOpacity>
        <Text style={styles.hintText}>Say Back to return</Text>
      </View>
    </SafeAreaView>
  );
}

// ---------- styles ----------

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#000000',
  },
  centerContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    paddingHorizontal: 30,
  },
  headerSection: {
    flex: 1,
    justifyContent: 'flex-end',
    alignItems: 'center',
    paddingBottom: 20,
  },
  title: {
    fontSize: 28,
    fontWeight: '300',
    color: '#ffffff',
    letterSpacing: 3,
  },
  centerSection: {
    flex: 3,
    justifyContent: 'center',
    alignItems: 'center',
    paddingHorizontal: 30,
  },
  addressLabel: {
    fontSize: 20,
    fontWeight: '400',
    color: '#ffffff',
    textAlign: 'center',
    marginBottom: 12,
    lineHeight: 28,
  },
  distanceLabel: {
    fontSize: 18,
    fontWeight: '300',
    color: '#aaaaaa',
    marginBottom: 24,
  },
  promptText: {
    fontSize: 18,
    fontWeight: '300',
    color: '#888888',
    textAlign: 'center',
    lineHeight: 26,
  },
  statusText: {
    color: '#ffffff',
    fontSize: 16,
    marginTop: 16,
    letterSpacing: 1,
  },
  errorText: {
    color: '#ff4444',
    fontSize: 18,
    textAlign: 'center',
    marginBottom: 20,
  },
  hintText: {
    color: '#555555',
    fontSize: 13,
    textAlign: 'center',
    marginTop: 8,
  },
  footerSection: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  voiceStatusContainer: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  voiceIndicator: {
    width: 8,
    height: 8,
    borderRadius: 4,
    backgroundColor: '#444444',
    marginRight: 8,
  },
  voiceIndicatorActive: {
    backgroundColor: '#00ff00',
  },
  voiceStatusText: {
    color: '#888888',
    fontSize: 12,
  },
  skipButton: {
    marginTop: 14,
    paddingVertical: 10,
    paddingHorizontal: 18,
    borderRadius: 8,
    borderWidth: 1,
    borderColor: '#555',
    backgroundColor: '#111',
  },
  skipButtonText: {
    fontSize: 13,
    color: '#cccccc',
    letterSpacing: 1,
  },
});
