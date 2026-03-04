# Installation & Setup Guide - ConvLSTM Turn Prediction Mobile App

This guide will walk you through installing dependencies, fixing errors, and running the app on your mobile device.

## Prerequisites

Before starting, ensure you have:

1. **Node.js** (v18 LTS or v20 LTS - **NOT v24+**)
   - Download from: https://nodejs.org/
   - **Use LTS version (v20.18.2 recommended)** - v24+ has compatibility issues with Expo
   - Verify: `node --version`

2. **npm** or **yarn** (comes with Node.js)
   - Verify: `npm --version`

3. **Expo CLI**
   - Install globally: `npm install -g expo-cli`

4. **Android Studio** (for Android development)
   - Download from: https://developer.android.com/studio
   - Required for Android emulator or building APK

5. **Redmi Note 13 Pro 5G or Android device** (for testing)
   - Install **Expo Go** app from Google Play Store

---

## Step 1: Install Dependencies

Navigate to the project directory and install all required packages:

```bash
# Navigate to the project folder
cd test_deployment

# Install all dependencies (this fixes the "Cannot find module" errors)
npm install

# Alternative: use yarn
# yarn install
```

This will install:
- React & React Native
- Expo packages (camera, status bar, etc.)
- React Navigation
- TypeScript types
- All other dependencies listed in package.json

**Expected output:**
```
added 1234 packages in 45s
```

---

## Step 2: Verify Installation

After installation, verify TypeScript can find all modules:

```bash
# Check for TypeScript errors
npx tsc --noEmit

# Expected: No errors (or only warnings)
```

If you still see errors related to module resolution, try:

```bash
# Clear caches
npx expo start -c

# Or manually clear node_modules and reinstall
rm -rf node_modules
npm install
```

---

## Step 3: Running the App

### Option A: Run on Physical Device (Recommended)

This is the easiest way to test on your Redmi Note 13 Pro 5G.

**Steps:**

1. **Install Expo Go on your phone:**
   - Open Google Play Store
   - Search for "Expo Go"
   - Install the app

2. **Start the development server:**
   ```bash
   npx expo start
   ```

3. **Connect your phone:**
   - Make sure your phone and computer are on the **same WiFi network**
   - Open Expo Go app on your phone
   - Scan the QR code displayed in the terminal (or browser)

4. **Grant camera permissions when prompted**

**Troubleshooting:**
- If QR code doesn't work, use the manual connection:
  - Note the URL shown (e.g., `exp://192.168.1.100:8081`)
  - Type it manually in Expo Go app
- Ensure firewall allows connections on port 8081

---

### Option B: Run on Android Emulator

**Requirements:**
- Android Studio installed
- Android emulator configured

**Steps:**

1. **Start Android emulator from Android Studio:**
   - Open Android Studio
   - Go to: Tools → Device Manager
   - Start an emulator (e.g., Pixel 7 with Android 13)

2. **Run the app:**
   ```bash
   npx expo start --android
   ```

   Or manually:
   ```bash
   npx expo start
   # Then press 'a' in the terminal to open in Android
   ```

3. **Wait for build and installation** (first time takes 5-10 minutes)

**Note:** Emulator camera is limited and may not work as expected. Physical device testing is recommended.

---

### Option C: Build Standalone APK

For a production-ready APK that doesn't require Expo Go:

**Using EAS Build (Recommended):**

1. **Install EAS CLI:**
   ```bash
   npm install -g eas-cli
   ```

2. **Login to Expo account:**
   ```bash
   eas login
   # Create free account at expo.dev if needed
   ```

3. **Configure EAS:**
   ```bash
   eas build:configure
   ```

4. **Build APK:**
   ```bash
   # For preview/development
   eas build --platform android --profile preview

   # For production
   eas build --platform android --profile production
   ```

5. **Download and install APK** on your phone

**Note:** EAS Build happens in the cloud and may take 10-20 minutes.

---

## Understanding the Prediction Timing

The app uses a **sliding window** approach for real-time predictions:

1. **Initial Buffer (0-1 second):**
   - Captures frames at 20 FPS (one frame every 50ms)
   - Collects 20 frames total (1 second of video)
   - Shows "Buffering" message with progress

2. **First Prediction (~1 second):**
   - After collecting 20 frames (full buffer), runs first prediction
   - No frame padding - waits for complete buffer for accuracy

3. **Continuous Predictions (after 1 second):**
   - Maintains rolling buffer of last 20 frames
   - Gives new prediction every 2nd frame (10 predictions/second)
   - Example: Frames [1-20] → prediction, [3-22] → prediction, [5-24] → prediction, etc.
   - Sliding window ensures smooth continuous predictions

4. **Performance:**
   - Capture rate: 20 FPS (50ms interval)
   - Prediction rate: 10 predictions/second (every 2 frames)
   - Expected latency: ~100-200ms per prediction on target device
   - Battery-optimized: Reduced prediction frequency saves power

**Customizing Prediction Rate:**

You can adjust how often predictions occur by editing `predictionInterval` in [src/config/modelConfig.ts](src/config/modelConfig.ts):

```typescript
export const DEVICE_CONFIG = {
  // ...
  predictionInterval: 2  // Change this value:
  // 1 = Every frame (20 predictions/sec) - Most responsive
  // 2 = Every 2nd frame (10 predictions/sec) - Balanced (Current)
  // 3 = Every 3rd frame (~7 predictions/sec) - Good battery life
  // 5 = Every 5th frame (4 predictions/sec) - Best battery life
}
```

---

## Step 4: Testing the App

Once the app is running:

1. **Main Menu:**
   - You'll see a black screen with white "ConvLSTM" title
   - Tap the white "Start" button

2. **Camera Screen:**
   - Grant camera permission when prompted
   - Point rear camera at the road/scene
   - The app will automatically:
     - Capture frames at 20 FPS (50ms interval)
     - Buffer 20 frames (1 second of video)
     - Show buffer progress at bottom
     - Run first prediction after 1 second
     - Continue with sliding window predictions every 50ms
     - Display prediction ("Front", "Left", "Right") at bottom center

3. **Performance Metrics (Top-Left):**
   - Inference: Model inference time
   - Preprocess: Frame preprocessing time
   - Total: Combined latency
   - FPS: Processing rate
   - Frames: Total frames captured
   - Predictions: Number of predictions made

4. **Back Button (Top-Right):**
   - Tap "✕" to return to main menu

---

## Common Issues & Solutions

### 0. "npm is not recognized" or "npx is not recognized"

**Cause:** Node.js is not in your PATH environment variable
**Solution:**

**Quick fix (current session only):**
```powershell
$env:Path += ";C:\Program Files\nodejs\"
```

**Permanent fix (recommended):**
1. Press `Win + R`, type `sysdm.cpl`, press Enter
2. Go to **Advanced** tab → **Environment Variables**
3. Under **System variables**, find and select **Path**
4. Click **Edit** → **New**
5. Add: `C:\Program Files\nodejs\`
6. Click **OK** on all dialogs
7. **Restart PowerShell** for changes to take effect

### 0b. "Running scripts is disabled on this system"

**Cause:** PowerShell execution policy is too restrictive
**Solution:**
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### 0c. "ENOENT: no such file or directory, mkdir '...node:sea'"

**Cause:** Node.js v24+ incompatibility with Expo SDK 50 on Windows
**Solution:** Downgrade to Node.js v20 LTS
1. Download Node.js v20 LTS from https://nodejs.org/
2. Install (will replace v24)
3. Verify: `node --version` should show v20.x.x
4. Restart PowerShell/VS Code
5. Run `npm install` again in test_deployment

### 1. "Cannot find module 'react'" or similar errors

**Cause:** Dependencies not installed
**Solution:**
```bash
npm install
```

### 2. "Metro bundler failed to start"

**Solution:**
```bash
# Clear all caches
npx expo start -c

# Or manually
rm -rf .expo node_modules
npm install
npx expo start
```

### 3. "Camera permission denied"

**Solution:**
- On physical device: Go to Settings → Apps → Expo Go → Permissions → Enable Camera
- Restart the app

### 4. "Network response timed out"

**Solution:**
- Ensure phone and computer are on same WiFi
- Disable VPN
- Check firewall settings
- Try USB connection: `npx expo start --tunnel`

### 5. TypeScript errors persist

**Solution:**
```bash
# Restart TypeScript server in VS Code
# Press Ctrl+Shift+P → "TypeScript: Restart TS Server"

# Or check tsconfig.json is properly configured
```

### 6. App crashes on Android

**Check:**
- Android version (should be 10+)
- Available RAM (model requires ~200MB)
- Camera permissions granted
- Check logs: `npx expo start` shows error messages

---

## Development Tips

### Hot Reload
- Changes to code auto-reload in Expo Go
- Shake phone to open developer menu
- Press "r" in terminal to reload manually

### Debugging
- Shake phone → "Debug Remote JS" to use Chrome DevTools
- Or use React Native Debugger
- Console.log statements appear in terminal

### Performance Testing
- Test on actual Redmi Note 13 Pro 5G for accurate metrics
- Emulator performance will be slower
- Expected inference time: ~100ms on target device
- Prediction interval: New prediction every 50ms after initial buffer (using sliding window)
- First prediction: ~1 second (time to collect 20 frames at 20 FPS)

---

## Project Structure Quick Reference

```
test_deployment/
├── package.json         # Dependencies list
├── app.json            # Expo configuration
├── App.tsx             # App entry point
├── src/
│   ├── screens/        # MainMenu & Camera screens
│   ├── services/       # Preprocessor & Inference logic
│   ├── config/         # Model configuration
│   └── navigation/     # Navigation types
└── assets/model/       # convlstm.tflite model file
```

---

## Next Steps

1. **Install dependencies:** `npm install`
2. **Start dev server:** `npx expo start`
3. **Open on phone:** Scan QR with Expo Go
4. **Test the app:** Point camera at road/objects
5. **Monitor metrics:** Check inference time and accuracy

---

## Additional Resources

- **Expo Documentation:** https://docs.expo.dev/
- **React Navigation:** https://reactnavigation.org/
- **React Native Camera:** https://docs.expo.dev/versions/latest/sdk/camera/
- **Troubleshooting:** https://docs.expo.dev/troubleshooting/

---

## Production Deployment

For final deployment without Expo Go:

1. Build standalone APK with EAS Build
2. Test thoroughly on target device
3. Optimize model (INT8 quantization)
4. Implement actual TFLite inference (replace mock)
5. Add error handling and edge cases
6. Consider adding:
   - Settings screen
   - Recording/logging features
   - Multiple camera support
   - Model selection

---

## Step 6: Building a Standalone APK

There are two methods to build an APK for distribution:

### Method 1: EAS Build (Recommended - Cloud-based)

EAS (Expo Application Services) is the easiest way to build production-ready APKs.

**Requirements:**
- Expo account (free tier available)
- Internet connection

**Steps:**

1. **Install EAS CLI:**
   ```bash
   npm install -g eas-cli
   ```

2. **Login to Expo:**
   ```bash
   eas login
   # Or create account: eas register
   ```

3. **Configure EAS:**
   ```bash
   eas build:configure
   ```
   This creates an `eas.json` file with build configurations.

4. **Build APK for Android:**
   ```bash
   # Build APK (for local installation)
   eas build --platform android --profile preview

   # Alternative: Build AAB (for Google Play Store)
   # eas build --platform android --profile production
   ```

5. **Wait for build:**
   - Build happens in the cloud (5-15 minutes)
   - You'll receive a link to download the APK
   - Download and install on your device

**Advantages:**
- No need for Android Studio or local Android SDK
- Handles signing and configuration automatically
- Build logs available online
- Can build for iOS too

**Customizing EAS Build:**

Edit `eas.json` (created in step 3):
```json
{
  "build": {
    "preview": {
      "android": {
        "buildType": "apk"
      }
    },
    "production": {
      "android": {
        "buildType": "aab"
      }
    }
  }
}
```

---

### Method 2: Local Build (Requires Android Studio)

Build the APK locally on your machine.

**Requirements:**
- Android Studio installed
- Android SDK configured
- Java JDK 11 or higher

**Steps:**

1. **Generate Android native code:**
   ```bash
   npx expo prebuild --platform android
   ```
   This creates an `android/` folder with native code.

2. **Install Android dependencies:**
   ```bash
   cd android
   ./gradlew clean
   cd ..
   ```

3. **Build APK:**
   ```bash
   # Debug APK (for testing)
   cd android
   ./gradlew assembleDebug

   # Release APK (for distribution)
   ./gradlew assembleRelease
   ```

4. **Locate the APK:**
   - Debug: `android/app/build/outputs/apk/debug/app-debug.apk`
   - Release: `android/app/build/outputs/apk/release/app-release.apk`

5. **Install on device:**
   ```bash
   # Via ADB
   adb install android/app/build/outputs/apk/debug/app-debug.apk
   
   # Or transfer the APK file to your phone and install manually
   ```

**For Release Build (signed APK):**

You need to create a keystore for signing:

```bash
# Generate keystore
cd android/app
keytool -genkeypair -v -storetype PKCS12 -keystore my-release-key.keystore -alias my-key-alias -keyalg RSA -keysize 2048 -validity 10000
```

Then configure `android/gradle.properties`:
```properties
MYAPP_RELEASE_STORE_FILE=my-release-key.keystore
MYAPP_RELEASE_KEY_ALIAS=my-key-alias
MYAPP_RELEASE_STORE_PASSWORD=your_password
MYAPP_RELEASE_KEY_PASSWORD=your_password
```

And update `android/app/build.gradle`:
```gradle
android {
    ...
    signingConfigs {
        release {
            storeFile file('my-release-key.keystore')
            storePassword System.getenv("MYAPP_RELEASE_STORE_PASSWORD")
            keyAlias System.getenv("MYAPP_RELEASE_KEY_ALIAS")
            keyPassword System.getenv("MYAPP_RELEASE_KEY_PASSWORD")
        }
    }
    buildTypes {
        release {
            signingConfig signingConfigs.release
            minifyEnabled true
            proguardFiles getDefaultProguardFile("proguard-android.txt"), "proguard-rules.pro"
        }
    }
}
```

---

### Quick Comparison

| Feature | EAS Build | Local Build |
|---------|-----------|-------------|
| Setup Time | 5 minutes | 30+ minutes |
| Requirements | Internet only | Android Studio, SDK |
| Build Speed | 5-15 min (cloud) | 2-5 min (local) |
| Difficulty | Easy | Moderate |
| Signing | Automatic | Manual setup |
| Best For | Quick testing | Development |

---

### Recommended Workflow for Your Thesis Project

**For testing on your Redmi Note 13 Pro 5G:**
1. Use **Expo Go** for rapid development (Step 3)
2. Use **EAS Build (preview)** when you need a standalone APK to share

**Commands Summary:**
```bash
# Daily development
npx expo start

# Build APK for testing/demo
eas build --platform android --profile preview

# Check build status
eas build:list
```

**APK File Size:**
- Expect around 40-80 MB for the APK
- Includes React Native, Expo modules, and your model file

---

## Step 7: Updating Your App & Rebuilding

After making changes to your app code, follow these steps to deploy the updated version:

### For Development (Using Expo Go)

Changes are automatically reflected - just save your files:
```bash
# App will hot-reload automatically
# Or manually reload by pressing 'r' in terminal
```

### For Standalone APK (Production Updates)

When you need to rebuild and redistribute your APK:

**1. Update Version Number (Recommended):**

Edit `app.json` to bump the version:
```json
{
  "expo": {
    "version": "1.0.1"  // Change from 1.0.0 to 1.0.1, etc.
  }
}
```

**2. Rebuild with EAS:**
```bash
cd test_deployment

# For testing/preview APK
eas build --platform android --profile preview

# For production (Google Play Store)
eas build --platform android --profile production
```

**3. Check Build Status:**
```bash
# List all builds
eas build:list

# View specific build details
eas build:view [build-id]
```

**4. Download & Install:**
- Once build completes (5-15 minutes), you'll get a download link
- Download the new APK
- Install on your device (may need to uninstall old version first)
- Test your changes

### Quick Update Workflow

```bash
# 1. Make your code changes
# 2. Test locally first
npx expo start

# 3. Once satisfied, update version in app.json
# 4. Build new APK
eas build --platform android --profile preview

# 5. Wait for build, download, and test
```

### Build Profiles Explained

- **preview**: Creates APK files for direct installation (great for testing and demos)
- **production**: Creates AAB files with auto-increment version for Google Play Store
- **development**: Creates development builds with debugging tools enabled

### Troubleshooting Build Issues

**Build failed?**
```bash
# View detailed build logs
eas build:list
# Click on the build URL to see full logs
```

**Need to cancel a build?**
```bash
eas build:cancel
```

**Clear build cache:**
```bash
eas build --platform android --profile preview --clear-cache
```

---

## Support

For issues specific to this project:
- Check console logs: `npx expo start`
- Verify model file exists: `assets/model/convlstm.tflite`
- Ensure TypeScript compiled without errors: `npx tsc --noEmit`
- Test on physical device (emulator limitations)

Good luck with your mobile deployment! 🚀
