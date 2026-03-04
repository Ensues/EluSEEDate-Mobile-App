# Model Conversion Guide

## Overview

This app uses **TensorFlow Lite (TFLite)** for on-device inference via `react-native-fast-tflite`. The model must be converted from PyTorch to TFLite format.

**Current Status**: The app has a TFLite model at `assets/model/convlstm.tflite` (Float16 quantized, 3.62 MB).

## Conversion Pipeline: PyTorch → ONNX → TensorFlow → TFLite

The conversion follows this path:

```
PyTorch (.pth) → ONNX (.onnx) → TensorFlow (SavedModel) → TFLite (.tflite)
```

---

## Step 1: Export PyTorch Model to ONNX

### Prerequisites
```bash
pip install torch onnx
```

### Export Script

```python
import torch
import onnx
from models.conv_lstm_classifier import ConvLSTMClassifier

# Load trained PyTorch model
model = ConvLSTMClassifier(
    input_dim=6,
    hidden_dim=[64, 32],
    kernel_size=(3, 3),
    num_layers=2,
    seq_len=20,
    height=128,
    width=128,
    num_classes=3,
    dropout_rate=0.5
)
model.load_state_dict(torch.load('notebooks/best_convlstm.pth'))
model.eval()

# Create dummy input: [batch, seq_len, channels, height, width]
dummy_input = torch.randn(1, 20, 6, 128, 128)

# Export to ONNX
torch.onnx.export(
    model,
    dummy_input,
    "conversion/convlstm.onnx",
    export_params=True,
    opset_version=13,  # Use opset 13 for better compatibility
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={
        'input': {0: 'batch_size'},
        'output': {0: 'batch_size'}
    }
)

print("✅ ONNX export successful!")
```

### Verify ONNX Model

```python
import onnx

# Load and check the model
onnx_model = onnx.load("conversion/convlstm.onnx")
onnx.checker.check_model(onnx_model)
print("✅ ONNX model is valid")

# Visualize with Netron (optional)
# pip install netron
# netron conversion/convlstm.onnx
```

---

## Step 2: Convert ONNX to TensorFlow SavedModel

### Prerequisites
```bash
pip install onnx-tf tensorflow
```

### Conversion Script

```bash
# Convert ONNX to TensorFlow SavedModel
onnx-tf convert -i conversion/convlstm.onnx -o conversion/tf_model/

# Expected output:
# conversion/tf_model/
#   ├── saved_model.pb
#   ├── fingerprint.pb
#   └── variables/
```

**Note**: Some PyTorch operations may not have direct TensorFlow equivalents. Check the conversion logs for warnings.

---

## Step 3: Convert TensorFlow SavedModel to TFLite

### Option A: Float32 TFLite (Baseline)

```python
import tensorflow as tf

# Load SavedModel
converter = tf.lite.TFLiteConverter.from_saved_model('conversion/tf_model/')

# Convert to TFLite Float32
tflite_model = converter.convert()

# Save model
with open('conversion/convlstm_float32.tflite', 'wb') as f:
    f.write(tflite_model)

print("✅ Float32 TFLite model created")
print(f"   Size: {len(tflite_model) / (1024*1024):.2f} MB")
```

### Option B: Float16 TFLite (Recommended - Smaller Size)

```python
import tensorflow as tf

# Load SavedModel
converter = tf.lite.TFLiteConverter.from_saved_model('conversion/tf_model/')

# Enable Float16 quantization
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]

# Convert to TFLite Float16
tflite_model = converter.convert()

# Save model
with open('conversion/convlstm_float16.tflite', 'wb') as f:
    f.write(tflite_model)

print("✅ Float16 TFLite model created")
print(f"   Size: {len(tflite_model) / (1024*1024):.2f} MB")
print(f"   Size reduction: ~50% compared to Float32")
```

### Option C: INT8 Quantization (Advanced - Fastest Inference)

```python
import tensorflow as tf
import numpy as np

def representative_dataset():
    """Generate representative data for calibration"""
    for _ in range(100):
        # Generate random data matching input shape
        data = np.random.rand(1, 20, 6, 128, 128).astype(np.float32)
        yield [data]

# Load SavedModel
converter = tf.lite.TFLiteConverter.from_saved_model('conversion/tf_model/')

# Enable INT8 quantization
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8

# Convert
tflite_model = converter.convert()

# Save model
with open('conversion/convlstm_int8.tflite', 'wb') as f:
    f.write(tflite_model)

print("✅ INT8 TFLite model created")
print(f"   Size: {len(tflite_model) / (1024*1024):.2f} MB")
print(f"   Expected: 2-4x faster inference")
```

---

## Step 4: Test TFLite Model

```python
import tensorflow as tf
import numpy as np

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path='conversion/convlstm_float16.tflite')
interpreter.allocate_tensors()

# Get input/output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("Input shape:", input_details[0]['shape'])
print("Output shape:", output_details[0]['shape'])

# Test with dummy data
test_input = np.random.rand(1, 20, 6, 128, 128).astype(np.float32)
interpreter.set_tensor(input_details[0]['index'], test_input)

# Run inference
interpreter.invoke()

# Get output
output = interpreter.get_tensor(output_details[0]['index'])
print("Output:", output)
print("Predicted class:", np.argmax(output))
```

---

## Step 5: Deploy to Mobile App

1. **Copy the TFLite model to the app:**
   ```bash
   cp conversion/convlstm_float16.tflite test_deployment/assets/model/convlstm.tflite
   ```

2. **Build development build** (TFLite requires native modules):
   ```bash
   cd test_deployment
   npx expo prebuild
   npx expo run:android
   ```

3. **Test on device:**
   - Launch app
   - Tap "Start" button
   - Grant camera permission
   - Check if predictions show "[REAL]" instead of "[DEMO]"

---

## File Size Comparison

| Model Type | Size | Accuracy | Inference Time |
|------------|------|----------|----------------|
| PyTorch .pth | 7.14 MB | Baseline | N/A (desktop only) |
| ONNX | ~7 MB | ~Baseline | ~baseline |
| TFLite Float32 | ~7 MB | ~Baseline | 200-500ms |
| TFLite Float16 | ~3.6 MB | 99%+ of baseline | 100-300ms |
| TFLite INT8 | ~2 MB | 95%+ of baseline | 50-150ms |

**Recommendation**: Use Float16 for best balance of size, accuracy, and speed.

---

## Troubleshooting

### Error: "ONNX conversion failed"
- Check PyTorch model architecture for unsupported operations
- Try different ONNX opset versions (11, 12, 13, 14)
- Use `torch.onnx.export` with `verbose=True` for debugging

### Error: "TensorFlow conversion failed"
- Some ONNX ops may not be supported by onnx-tf
- Check conversion logs for specific operation errors
- Consider using alternative operations in PyTorch model

### Error: "TFLite model loads but crashes"
- Verify input shape matches exactly: `[1, 20, 6, 128, 128]`
- Check data type (Float32 vs Float16 vs INT8)
- Ensure preprocessing matches training pipeline

### Model predictions are incorrect
- Verify normalization: pixels should be in [0, 1] range
- Check channel order: RGB (not BGR)
- Ensure intent channels are zeros if not using intent
- Test model in Python first before deploying

### App shows "Demo Mode"
- TFLite requires development build (not Expo Go)
- Run: `npx expo prebuild && npx expo run:android`
- Check that `convlstm.tflite` exists in `assets/model/`
- Check console logs for TFLite loading errors

---

## Need Help?

1. Check the console logs in Metro bundler and device logs
2. Verify each conversion step produces valid output
3. Test model in Python/TensorFlow before deploying to mobile
4. Compare predictions between PyTorch, ONNX, TensorFlow, and TFLite
