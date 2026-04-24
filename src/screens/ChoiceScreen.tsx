/**
 * Choice Screen
 * 
 * Allows users to choose between Wandering (NoIntent) and Destination (Intent) modes
 * Supports voice commands: "Wandering", "Destination", "Back"
 * 
 * Design: Minimalistic black & white (matches MainMenu)
 */

import React, { useRef, useCallback, useState } from 'react';
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
  const [buttonPressed, setButtonPressed] = useState(false);

  const {
    isListening,
    isSpeaking,
    readyToListen,
    voiceStatus,
    setVoiceStatus,
    speakMessage,
    speakThenListen,
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
    if (buttonPressed) return;
    setButtonPressed(true);
    hasNavigatedRef.current = true;
    void stopVoskListening();
    speakMessage({
      message: 'Starting wandering mode. Say stop to go back to the previous screen, also Turn Off to disable the G.U.I or Turn On to enable it.',
      onDone: () => navigation.navigate('ActiveCamera', { mode: 'wandering' }),
    });
  };

  const handleDestinationPress = () => {
    if (buttonPressed) return;
    setButtonPressed(true);
    hasNavigatedRef.current = true;
    void stopVoskListening();
    speakMessage({
      message: 'Opening wayfinding. Say stop to go back to the previous screen, also Turn Off to disable the G.U.I or Turn On to enable it.',
      onDone: () => navigation.navigate('Wayfinding'),
    });
  };

  const handleBackPress = () => {
    if (buttonPressed) return;
    setButtonPressed(true);
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
      setButtonPressed(false);

      let isActive = true;
      const startTask = InteractionManager.runAfterInteractions(() => {
        void stopAllVoiceActivity().finally(() => {
          if (!isActive) {
            return;
          }

          speakThenListen({
            message: 'Choose your mode, Wandering or Destination. If you want to return to the main menu say back.',
            statusWhileSpeaking: 'Speaking instructions...',
            statusWhileListening: 'Say "Wandering", "Destination", or "Back"',
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
        grammar: ['wandering', 'destination', 'back', 'eluseedate', '[unk]'],
        statusWhileListening: 'Say "Wandering", "Destination", or "Back"',
        onResult: async (result: string) => {
          const lowerResult = result.toLowerCase().trim();
          if (hasNavigatedRef.current) {
            return;
          }
          if (await tryHandleBargeIn(lowerResult)) {
            setVoiceStatus('Audio interrupted. Say "Wandering", "Destination", or "Back"');
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
          disabled={buttonPressed || isSpeaking}
        >
          <Text style={styles.modeButtonText}>Wandering</Text>
        </TouchableOpacity>

        <TouchableOpacity
          style={[styles.modeButton, { marginTop: 20 }]}
          onPress={handleDestinationPress}
          activeOpacity={0.7}
          disabled={buttonPressed || isSpeaking}
        >
          <Text style={styles.modeButtonText}>Destination</Text>
        </TouchableOpacity>

        <TouchableOpacity
          style={[styles.modeButton, { marginTop: 20, backgroundColor: '#222' }]}
          onPress={handleBackPress}
          activeOpacity={0.7}
          disabled={buttonPressed || isSpeaking}
        >
          <Text style={[styles.modeButtonText, { color: '#fff' }]}>Back</Text>
        </TouchableOpacity>

        {/* Voice Status Indicator */}
        <View style={styles.voiceStatusContainer}>
          <View style={[styles.voiceIndicator, isListening && styles.voiceIndicatorActive]} />
          <Text style={styles.voiceStatusText}>{voiceStatus}</Text>
        </View>

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
  footerSection: {
    flex: 1,
  },
});
