/**
 * EluSEEdate - Turn Prediction App
 * 
 * Main entry point
 * Portrait mode, mobile-optimized for Redmi Note 13 Pro 5G
 * 
 * Design: Minimalistic black & white
 * Model: ConvLSTM Prototype 10
 */

import React from 'react';
import { StatusBar, LogBox } from 'react-native';
import { NavigationContainer } from '@react-navigation/native';
import { createNativeStackNavigator } from '@react-navigation/native-stack';
import { SafeAreaProvider } from 'react-native-safe-area-context';

import MainMenuScreen from './src/screens/MainMenuScreen';
import CameraScreen from './src/screens/CameraScreen';
import { RootStackParamList } from './src/navigation/types';

// Ignore specific warnings (for development)
LogBox.ignoreLogs([
  'Non-serializable values were found in the navigation state',
]);

const Stack = createNativeStackNavigator<RootStackParamList>();

export default function App() {
  return (
    <SafeAreaProvider>
      <NavigationContainer>
        <StatusBar barStyle="light-content" backgroundColor="#000000" />
        <Stack.Navigator
          initialRouteName="MainMenu"
          screenOptions={{
            headerShown: false,
            animation: 'fade',
            contentStyle: { backgroundColor: '#000000' },
          }}
        >
          <Stack.Screen 
            name="MainMenu" 
            component={MainMenuScreen}
          />
          <Stack.Screen 
            name="Camera" 
            component={CameraScreen}
          />
        </Stack.Navigator>
      </NavigationContainer>
    </SafeAreaProvider>
  );
}
