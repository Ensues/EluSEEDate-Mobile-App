// Learn more https://docs.expo.io/guides/customizing-metro
const { getDefaultConfig } = require('expo/metro-config');

/** @type {import('expo/metro-config').MetroConfig} */
const config = getDefaultConfig(__dirname);

// Add TFLite file extension support
config.resolver.assetExts = [...(config.resolver.assetExts || []), 'tflite', 'onnx', 'bin'];

// Workaround for Node.js v20+ and Windows path issue with node:sea
// Disable node externals to prevent "ENOENT: no such file or directory, mkdir 'node:sea'" error
config.server = {
  ...config.server,
  enhanceMiddleware: (middleware) => {
    return (req, res, next) => {
      // Skip node externals handling
      if (req.url && req.url.includes('node:')) {
        res.statusCode = 404;
        res.end();
        return;
      }
      return middleware(req, res, next);
    };
  },
};

// Disable experimental features that cause issues with Windows paths
if (config.resolver) {
  config.resolver.unstable_enablePackageExports = false;
  config.resolver.unstable_conditionNames = [];
}

module.exports = config;
