/**
 * Navigation Type Definitions
 * 
 * Type-safe navigation with React Navigation
 */

export type RootStackParamList = {
  MainMenu: undefined;
  Camera: undefined;
};

// Extend the navigation types
declare global {
  namespace ReactNavigation {
    interface RootParamList extends RootStackParamList {}
  }
}
