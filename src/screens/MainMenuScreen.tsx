/**
 * Main Menu Screen
 * 
 * Entry point of the app with a Start button
 * Navigates to CameraScreen when pressed or by saying "Start"
 */

import React, { useEffect, useState, useRef, useCallback } from 'react';
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  StatusBar,
  Dimensions,
  Platform,
  BackHandler,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { NativeStackNavigationProp } from '@react-navigation/native-stack';
import { useFocusEffect } from '@react-navigation/native';
import { RootStackParamList } from '../navigation/types';
import * as Vosk from 'react-native-vosk';
import { useVoiceInteraction } from '../hooks/useVoiceInteraction';

// Module-level flag to ensure TTS only speaks once per app session
let hasSpokenGreeting = false;

type MainMenuScreenProps = {
  navigation: NativeStackNavigationProp<RootStackParamList, 'MainMenu'>;
};

const { width } = Dimensions.get('window');

export default function MainMenuScreen({ navigation }: MainMenuScreenProps) {
  const [modelLoaded, setModelLoaded] = useState(false);
  const hasNavigatedRef = useRef(false);

  const {
    isListening,
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
    stopAllVoiceActivity,
  } = useVoiceInteraction({
    initialVoiceStatus: 'Initializing...',
    defaultLanguage: 'en-US',
    listeningDelayMs: 1000,
  });


  const handleStartPress = () => {
    hasNavigatedRef.current = true;
    void stopVoskListening();
    speakMessage({
      message: 'Starting',
      onDone: () => navigation.navigate('Choice'),
    });
  };

  const handleExitPress = () => {
    hasNavigatedRef.current = true;
    void stopVoskListening();
    speakMessage({ message: 'Exiting' });
    // For mobile apps, exiting is not always supported, but we can try:
    if (Platform.OS === 'android') {
      BackHandler.exitApp();
    }
  };

  // Load Vosk model on component mount
  useEffect(() => {
    let isMounted = true;

    const loadModel = async () => {
      try {
        setVoiceStatus('Loading voice model...');
        await Vosk.loadModel('model-en-us');
        if (isMounted) {
          setModelLoaded(true);
          setVoiceStatus('Say "Start" to begin');
        }
      } catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        console.error(`[ERROR] Failed to load Vosk model: ${message}`);
        if (isMounted) {
          setVoiceStatus('Voice command disabled');
        }
      }
    };

    loadModel();

    return () => {
      isMounted = false;
    };
  }, [setVoiceStatus]);

  // Speak startup greeting once per app session, then prepare listening state.
  useFocusEffect(
    useCallback(() => {
      hasNavigatedRef.current = false;

      if (!modelLoaded) {
        return () => {
          void stopAllVoiceActivity();
        };
      }

      if (!hasSpokenGreeting) {
        hasSpokenGreeting = true;
        speakThenListen({
          message: 'Starting EluSEEdate. You can say Start to begin the app, Exit to exit, or Skip or Stop to interrupt audio prompts.',
          statusWhileSpeaking: 'Speaking instructions...',
          statusWhileListening: 'Say "Start", "Exit", "Skip", or "Stop"',
        });
      } else {
        void transitionToListening({
          statusWhileListening: 'Say "Start", "Exit", "Skip", or "Stop"',
        });
      }

      return () => {
        void stopAllVoiceActivity();
      };
    }, [modelLoaded, speakThenListen, stopAllVoiceActivity, transitionToListening])
  );

  // Start/stop Vosk recognition when listening state is active.
  useFocusEffect(
    useCallback(() => {
      if (!modelLoaded || !readyToListen) {
        return;
      }

      void startVoskListening({
        grammar: ['start', 'exit', 'skip', 'stop', 'eluseedate', '[unk]'],
        statusWhileListening: 'Say "Start", "Exit", "Skip", or "Stop"',
        onResult: async (result: string) => {
          const lowerResult = result.toLowerCase().trim();
          if (hasNavigatedRef.current) {
            return;
          }
          if (await tryHandleBargeIn(lowerResult)) {
            setVoiceStatus('Audio interrupted. Say "Start" or "Exit"');
            return;
          }
          if (lowerResult.includes('skip') || lowerResult.includes('stop')) {
            void skipSpeech();
            setVoiceStatus('Audio skipped. Say "Start" or "Exit"');
            return;
          }
          if (lowerResult.includes('start')) {
            hasNavigatedRef.current = true;
            setVoiceStatus('Starting...');
            void stopVoskListening();
            speakMessage({
              message: 'Starting',
              onDone: () => navigation.navigate('Choice'),
            });
            return;
          }
          if (lowerResult.includes('exit')) {
            hasNavigatedRef.current = true;
            setVoiceStatus('Exiting...');
            void stopVoskListening();
            speakMessage({ message: 'Exiting' });
            if (Platform.OS === 'android') {
              BackHandler.exitApp();
            }
          }
        },
      });

      // Cleanup when screen loses focus
      return () => {
        void stopVoskListening();
      };
    }, [
      modelLoaded,
      navigation,
      readyToListen,
      setVoiceStatus,
      skipSpeech,
      speakMessage,
      startVoskListening,
      stopVoskListening,
      tryHandleBargeIn,
    ])
  );

  return (
    <SafeAreaView style={styles.container}>
      <StatusBar barStyle="light-content" backgroundColor="#000000" />
      
      {/* Header Section */}
      <View style={styles.headerSection}>
        <Text style={styles.title}>EluSEEdate</Text>
        <Text style={styles.subtitle}>Turn Prediction</Text>
        <Text style={styles.version}>v1.0.5</Text>
      </View>

      {/* Center Section with Start and Exit Buttons */}
      <View style={styles.centerSection}>
        <TouchableOpacity
          style={styles.startButton}
          onPress={handleStartPress}
          activeOpacity={0.7}
        >
          <Text style={styles.startButtonText}>Start</Text>
        </TouchableOpacity>
        <TouchableOpacity
          style={[styles.startButton, { marginTop: 20, backgroundColor: '#222' }]}
          onPress={handleExitPress}
          activeOpacity={0.7}
        >
          <Text style={[styles.startButtonText, { color: '#fff' }]}>Exit</Text>
        </TouchableOpacity>
        {/* Voice Status Indicator */}
        <View style={styles.voiceStatusContainer}>
          <View style={[styles.voiceIndicator, isListening && styles.voiceIndicatorActive]} />
          <Text style={styles.voiceStatusText}>{voiceStatus}</Text>
        </View>

        <TouchableOpacity
          style={styles.skipButton}
          onPress={() => {
            setVoiceStatus('Audio skipped. Say "Start" or "Exit"');
            void skipSpeech();
          }}
          activeOpacity={0.7}
        >
          <Text style={styles.skipButtonText}>Skip Audio</Text>
        </TouchableOpacity>
      </View>

      {/* Footer Section */}
      <View style={styles.footerSection}>
        <Text style={styles.footerText}>
          Point camera at the road
        </Text>
        
        {/* Debug Logs Button */}
        <TouchableOpacity
          style={styles.debugButton}
          onPress={() => navigation.navigate('Logs')}
          activeOpacity={0.7}
        >
          <Text style={styles.debugButtonText}>📋 Debug Logs</Text>
        </TouchableOpacity>
      </View>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#000000',
  },
  
  // Header Section
  headerSection: {
    flex: 2,
    justifyContent: 'flex-end',
    alignItems: 'center',
    paddingBottom: 40,
  },
  title: {
    fontSize: 36,
    fontWeight: '300',
    color: '#ffffff',
    letterSpacing: 4,
  },
  subtitle: {
    fontSize: 18,
    fontWeight: '300',
    color: '#888888',
    marginTop: 8,
  },
  version: {
    fontSize: 12,
    color: '#444444',
    marginTop: 16,
  },

  // Center Section
  centerSection: {
    flex: 3,
    justifyContent: 'center',
    alignItems: 'center',
    paddingHorizontal: 40,
  },
  startButton: {
    width: width * 0.5,
    height: 60,
    backgroundColor: '#ffffff',
    borderRadius: 30,
    justifyContent: 'center',
    alignItems: 'center',
  },
  startButtonText: {
    fontSize: 20,
    fontWeight: '500',
    color: '#000000',
    letterSpacing: 2,
  },
  
  // Voice Status
  voiceStatusContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    marginTop: 24,
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
    fontSize: 12,
    color: '#666666',
  },

  // Footer Section
  footerSection: {
    flex: 1,
    justifyContent: 'flex-end',
    alignItems: 'center',
    paddingBottom: 40,
  },
  footerText: {
    fontSize: 12,
    color: '#666666',    marginBottom: 20,
  },
  debugButton: {
    paddingVertical: 12,
    paddingHorizontal: 24,
    backgroundColor: '#1a1a1a',
    borderRadius: 8,
    borderWidth: 1,
    borderColor: '#333',
  },
  debugButtonText: {
    fontSize: 14,
    color: '#888888',
  },
  skipButton: {
    marginTop: 16,
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
