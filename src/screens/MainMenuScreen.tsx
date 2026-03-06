/**
 * Main Menu Screen
 * 
 * Entry point of the app with a Start button
 * Navigates to CameraScreen when pressed or by saying "Start"
 * 
 * Design: Minimalistic black & white
 * App Name: EluSEEdate
 */

import React, { useEffect, useState, useRef, useCallback } from 'react';
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  StatusBar,
  Dimensions,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { NativeStackNavigationProp } from '@react-navigation/native-stack';
import { useFocusEffect } from '@react-navigation/native';
import { RootStackParamList } from '../navigation/types';
import * as Vosk from 'react-native-vosk';
import * as Speech from 'expo-speech';

// Module-level flag to ensure TTS only speaks once per app session
let hasSpokenGreeting = false;

type MainMenuScreenProps = {
  navigation: NativeStackNavigationProp<RootStackParamList, 'MainMenu'>;
};

const { width } = Dimensions.get('window');

export default function MainMenuScreen({ navigation }: MainMenuScreenProps) {
  const [isListening, setIsListening] = useState(false);
  const [voiceStatus, setVoiceStatus] = useState('Initializing...');
  const [modelLoaded, setModelLoaded] = useState(false);
  const resultListenerRef = useRef<any>(null);
  const hasNavigatedRef = useRef(false);

  const handleStartPress = () => {
    navigation.navigate('Camera');
  };

  // Speak startup greeting once per app session
  useEffect(() => {
    if (!hasSpokenGreeting) {
      hasSpokenGreeting = true;
      Speech.speak('Starting EluSEEdate', { language: 'en-US' });
    }
  }, []);

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
        console.error('Failed to load Vosk model:', error);
        if (isMounted) {
          setVoiceStatus('Voice command disabled');
        }
      }
    };

    loadModel();

    return () => {
      isMounted = false;
    };
  }, []);

  // Start/stop voice recognition when screen is focused/unfocused
  useFocusEffect(
    useCallback(() => {
      // Reset navigation flag when screen comes into focus
      hasNavigatedRef.current = false;

      const startListening = async () => {
        if (!modelLoaded) return;

        try {
          // Start recognition with grammar for "start" command
          await Vosk.start({
            grammar: ['start', '[unk]'],
          });
          
          setIsListening(true);
          setVoiceStatus('Say "Start" to begin');

          // Set up result listener
          resultListenerRef.current = Vosk.onResult((result: string) => {
            console.log('Voice result:', result);
            
            // Check if user said "start" and we haven't navigated yet
            if (result.toLowerCase().includes('start') && !hasNavigatedRef.current) {
              hasNavigatedRef.current = true;
              setVoiceStatus('Starting...');
              Vosk.stop();
              setIsListening(false);
              navigation.navigate('Camera');
            }
          });
        } catch (error: any) {
          console.error('Failed to start voice recognition:', error);
          // Check if it's a permission error
          if (error?.message?.includes('permission') || error?.message?.includes('Permission')) {
            setVoiceStatus('Microphone permission denied');
          } else {
            setVoiceStatus('Voice command disabled');
          }
          setIsListening(false);
        }
      };

      startListening();

      // Cleanup when screen loses focus
      return () => {
        if (resultListenerRef.current) {
          resultListenerRef.current.remove();
          resultListenerRef.current = null;
        }
        Vosk.stop();
        setIsListening(false);
      };
    }, [modelLoaded, navigation])
  );

  return (
    <SafeAreaView style={styles.container}>
      <StatusBar barStyle="light-content" backgroundColor="#000000" />
      
      {/* Header Section */}
      <View style={styles.headerSection}>
        <Text style={styles.title}>EluSEEdate</Text>
        <Text style={styles.subtitle}>Turn Prediction</Text>
        <Text style={styles.version}>v1.0.4</Text>
      </View>

      {/* Center Section with Start Button */}
      <View style={styles.centerSection}>
        <TouchableOpacity
          style={styles.startButton}
          onPress={handleStartPress}
          activeOpacity={0.7}
        >
          <Text style={styles.startButtonText}>Start</Text>
        </TouchableOpacity>
        
        {/* Voice Status Indicator */}
        <View style={styles.voiceStatusContainer}>
          <View style={[styles.voiceIndicator, isListening && styles.voiceIndicatorActive]} />
          <Text style={styles.voiceStatusText}>{voiceStatus}</Text>
        </View>
      </View>

      {/* Footer Section */}
      <View style={styles.footerSection}>
        <Text style={styles.footerText}>
          Point camera at the road
        </Text>
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
    color: '#666666',
  },
});
