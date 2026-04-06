/**
 * Choice Screen
 * 
 * Allows users to choose between Wandering (NoIntent) and Destination (Intent) modes
 * Supports voice commands: "Wandering", "Destination", "Back", "Skip"
 * 
 * Design: Minimalistic black & white (matches MainMenu)
 */

import React, { useRef, useCallback } from 'react';
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  StatusBar,
  Dimensions,
  InteractionManager,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { NativeStackNavigationProp } from '@react-navigation/native-stack';
import { useFocusEffect } from '@react-navigation/native';
import { RootStackParamList } from '../navigation/types';
import { useVoiceInteraction } from '../hooks/useVoiceInteraction';

type ChoiceScreenProps = {
  navigation: NativeStackNavigationProp<RootStackParamList, 'Choice'>;
};

const { width } = Dimensions.get('window');

export default function ChoiceScreen({ navigation }: ChoiceScreenProps) {
  const hasNavigatedRef = useRef(false);

  const {
    isListening,
    readyToListen,
    voiceStatus,
    setVoiceStatus,
    speakMessage,
    speakThenListen,
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


  const handleWanderingPress = () => {
    hasNavigatedRef.current = true;
    void stopVoskListening();
    speakMessage({
      message: 'Starting wandering mode. Say stop to go back to the previous screen, also Turn Off to disable the G.U.I or Turn On to enable it.',
      onDone: () => navigation.navigate('ActiveCamera', { mode: 'wandering' }),
    });
  };

  const handleDestinationPress = () => {
    hasNavigatedRef.current = true;
    void stopVoskListening();
    speakMessage({
      message: 'Opening wayfinding. Say stop to go back to the previous screen, also Turn Off to disable the G.U.I or Turn On to enable it.',
      onDone: () => navigation.navigate('Wayfinding'),
    });
  };

  const handleBackPress = () => {
    hasNavigatedRef.current = true;
    void stopVoskListening();
    speakMessage({
      message: 'Going back',
      onDone: () => navigation.navigate('MainMenu'),
    });
  };

  // TTS greeting, then enable listening after it finishes + delay
  useFocusEffect(
    useCallback(() => {
      hasNavigatedRef.current = false;

      let isActive = true;
      const startTask = InteractionManager.runAfterInteractions(() => {
        void stopAllVoiceActivity().finally(() => {
          if (!isActive) {
            return;
          }

          speakThenListen({
            message: 'Choose your mode, Wandering or Destination. If you want to return to the main menu say back. You can also say skip or stop to interrupt audio.',
            statusWhileSpeaking: 'Speaking instructions...',
            statusWhileListening: 'Say "Wandering", "Destination", "Back", "Skip", or "Stop"',
          });
        });
      });

      return () => {
        isActive = false;
        startTask.cancel();
        void stopAllVoiceActivity();
      };
    }, [speakThenListen, stopAllVoiceActivity])
  );

  // Start/stop voice recognition when screen is focused and ready
  useFocusEffect(
    useCallback(() => {
      if (!readyToListen) {
        return;
      }

      void startVoskListening({
        grammar: ['wandering', 'destination', 'back', 'skip', 'stop', 'eluseedate', '[unk]'],
        statusWhileListening: 'Say "Wandering", "Destination", "Back", "Skip", or "Stop"',
        onResult: async (result: string) => {
          const lowerResult = result.toLowerCase().trim();
          if (hasNavigatedRef.current) {
            return;
          }
          if (await tryHandleBargeIn(lowerResult)) {
            setVoiceStatus('Audio interrupted. Say "Wandering", "Destination", or "Back"');
            return;
          }
          if (lowerResult.includes('skip') || lowerResult.includes('stop')) {
            void skipSpeech();
            setVoiceStatus('Audio skipped. Say "Wandering", "Destination", or "Back"');
            return;
          }
          if (lowerResult.includes('wandering')) {
            hasNavigatedRef.current = true;
            setVoiceStatus('Starting wandering mode...');
            void stopVoskListening();
            speakMessage({
              message: 'Starting wandering mode. Say stop to go back to the previous screen, also Turn Off to disable the G.U.I or Turn On to enable it.',
              onDone: () => navigation.navigate('ActiveCamera', { mode: 'wandering' }),
            });
            return;
          }
          if (lowerResult.includes('destination')) {
            hasNavigatedRef.current = true;
            setVoiceStatus('Opening wayfinding...');
            void stopVoskListening();
            speakMessage({
              message: 'Opening wayfinding. Say stop to go back to the previous screen, also Turn Off to disable the G.U.I or Turn On to enable it.',
              onDone: () => navigation.navigate('Wayfinding'),
            });
            return;
          }
          if (lowerResult.includes('back')) {
            hasNavigatedRef.current = true;
            setVoiceStatus('Going back...');
            void stopVoskListening();
            speakMessage({
              message: 'Going back',
              onDone: () => navigation.navigate('MainMenu'),
            });
          }
        },
      });

      return () => {
        void stopVoskListening();
      };
    }, [
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
        <Text style={styles.title}>Choose Your Mode</Text>
      </View>

      {/* Center Section with mode buttons */}
      <View style={styles.centerSection}>
        <TouchableOpacity
          style={styles.modeButton}
          onPress={handleWanderingPress}
          activeOpacity={0.7}
        >
          <Text style={styles.modeButtonText}>Wandering</Text>
        </TouchableOpacity>

        <TouchableOpacity
          style={[styles.modeButton, { marginTop: 20 }]}
          onPress={handleDestinationPress}
          activeOpacity={0.7}
        >
          <Text style={styles.modeButtonText}>Destination</Text>
        </TouchableOpacity>

        <TouchableOpacity
          style={[styles.modeButton, { marginTop: 20, backgroundColor: '#222' }]}
          onPress={handleBackPress}
          activeOpacity={0.7}
        >
          <Text style={[styles.modeButtonText, { color: '#fff' }]}>Back</Text>
        </TouchableOpacity>

        {/* Voice Status Indicator */}
        <View style={styles.voiceStatusContainer}>
          <View style={[styles.voiceIndicator, isListening && styles.voiceIndicatorActive]} />
          <Text style={styles.voiceStatusText}>{voiceStatus}</Text>
        </View>

        <TouchableOpacity
          style={styles.skipButton}
          onPress={() => {
            setVoiceStatus('Audio skipped. Say "Wandering", "Destination", or "Back"');
            void skipSpeech();
          }}
          activeOpacity={0.7}
        >
          <Text style={styles.skipButtonText}>Skip Audio</Text>
        </TouchableOpacity>
      </View>

      {/* Footer */}
      <View style={styles.footerSection} />
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#000000',
  },
  headerSection: {
    flex: 2,
    justifyContent: 'flex-end',
    alignItems: 'center',
    paddingBottom: 40,
  },
  title: {
    fontSize: 30,
    fontWeight: '300',
    color: '#ffffff',
    letterSpacing: 3,
  },
  centerSection: {
    flex: 3,
    justifyContent: 'center',
    alignItems: 'center',
    paddingHorizontal: 40,
  },
  modeButton: {
    width: width * 0.5,
    height: 60,
    backgroundColor: '#ffffff',
    borderRadius: 30,
    justifyContent: 'center',
    alignItems: 'center',
  },
  modeButtonText: {
    fontSize: 20,
    fontWeight: '500',
    color: '#000000',
    letterSpacing: 2,
  },
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
  footerSection: {
    flex: 1,
  },
});
