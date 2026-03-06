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
   cp conversion/convlstm_float16.tflite assets/model/convlstm.tflite
   ```

2. **Build development build** (TFLite requires native modules):
   ```bash
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
| TFLite Float32 | ~7 MB | ~Baseline | 300-500ms |
| TFLite Float16 | 3.62 MB | 99%+ of baseline | 100-200ms |
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

---

# YOLOv12 Object Detection Model Conversion

## Overview

The app uses YOLOv12 for real-time obstacle detection. This section explains how to convert your YOLOv12 model to TFLite format for mobile deployment.

**Current Status**: The app has a placeholder at `assets/model/yolo-placeholder.txt`. Replace it with your actual `yolo.tflite` model.

---

## Conversion Pipeline: YOLO → TFLite

The conversion path depends on your YOLOv12 source format:

```
PyTorch YOLO → ONNX → TensorFlow → TFLite
     OR
TensorFlow YOLO → TFLite
     OR
Pre-converted YOLO TFLite → (Ready to use)
```

---

## Option 1: Convert from PyTorch YOLOv12

### Prerequisites
```bash
pip install torch torchvision onnx tensorflow tf2onnx
```

### Step 1: Export PyTorch YOLO to ONNX

```python
import torch
from your_yolo_model import YOLOv12  # Replace with your model import

# Load trained model
model = YOLOv12()
model.load_state_dict(torch.load('yolov12.pth'))
model.eval()

# Create dummy input (adjust size based on your model - typically 640x640 or 320x320)
dummy_input = torch.randn(1, 3, 128, 128)  # [batch, channels, height, width]

# Export to ONNX
torch.onnx.export(
    model,
    dummy_input,
    "yolov12.onnx",
    export_params=True,
    opset_version=13,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={
        'input': {0: 'batch_size'},
        'output': {0: 'batch_size'}
    }
)

print("✅ YOLO ONNX export successful!")
```

### Step 2: Convert ONNX to TensorFlow

```bash
# Convert ONNX to TensorFlow SavedModel
onnx-tf convert -i yolov12.onnx -o yolo_tf_model/
```

### Step 3: Convert TensorFlow to TFLite

```python
import tensorflow as tf

# Load SavedModel
converter = tf.lite.TFLiteConverter.from_saved_model('yolo_tf_model/')

# Enable Float16 quantization for smaller size
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]

# Convert
tflite_model = converter.convert()

# Save
with open('yolo.tflite', 'wb') as f:
    f.write(tflite_model)

print(f"✅ YOLO TFLite model created: {len(tflite_model)/(1024*1024):.2f} MB")
```

---

## Option 2: Use Pre-trained YOLOv12 TFLite

If you have a pre-trained YOLOv12 TFLite model:

1. **Verify the model:**
   ```python
   import tensorflow as tf
   
   interpreter = tf.lite.Interpreter(model_path='yolo.tflite')
   interpreter.allocate_tensors()
   
   input_details = interpreter.get_input_details()
   output_details = interpreter.get_output_details()
   
   print("Input shape:", input_details[0]['shape'])
   print("Output shape:", output_details[0]['shape'])
   ```

2. **Copy to assets:**
   ```bash
   cp yolo.tflite assets/model/yolo.tflite
   ```

3. **Update configuration** (if needed):
   - Edit `src/config/modelConfig.ts`
   - Adjust `YOLO_CONFIG.model.inputSize` to match your model
   - Update `YOLO_CLASS_NAMES` if using custom classes

---

## Step 4: Update App Configuration

After adding your YOLO model, you may need to adjust settings in `src/config/modelConfig.ts`:

```typescript
export const YOLO_CONFIG = {
  model: {
    inputSize: 128,           // Change to match your model (e.g., 640, 416, 320)
    numClasses: 80,           // Number of classes your model detects
    confidenceThreshold: 0.5, // Minimum confidence to show detection
    iouThreshold: 0.45,       // IoU threshold for NMS
  },
  // ...
};

// Update class names to match your model
export const YOLO_CLASS_NAMES = [
  'person', 'bicycle', 'car', // ... your model's classes
];
```

### Parsing YOLO Output

You'll need to implement output parsing in `src/services/yoloInference.ts`:

```typescript
private parseYOLOOutput(outputTensor: any, frameWidth: number, frameHeight: number): Detection[] {
  // TODO: Implement based on your YOLOv12 output format
  // Typical YOLO outputs:
  // - Bounding boxes: [x, y, w, h]
  // - Confidence scores
  // - Class probabilities
  // Apply NMS (Non-Maximum Suppression)
  // Filter by confidence threshold
  // Return Detection[] array
}
```

---

## Step 5: Deploy to Mobile

1. **Copy YOLO model:**
   ```bash
   cp yolo.tflite assets/model/yolo.tflite
   rm assets/model/yolo-placeholder.txt  # Remove placeholder
   ```

2. **Rebuild app:**
   ```bash
   npx expo prebuild  # Regenerate native files
   npx expo run:android
   ```

3. **Test on device:**
   - Launch app and tap "Start"
   - Point camera at objects
   - Verify bounding boxes appear with correct labels
   - Check console for "[YOLO-TFLite] [REAL]" (not [DEMO])

---

## Expected Model Specs

| Parameter | Typical Value | Adjustable |
|-----------|---------------|------------|
| Input Size | 128x128, 320x320, 416x416, 640x640 | ✅ Yes |
| Input Format | RGB (normalized [0,1] or [-1,1]) | ✅ Yes |
| Output Format | Bounding boxes + scores + classes | Model-specific |
| Model Size | 5-50 MB (Float16) | Depends on architecture |
| Inference Time | 30-100ms (mobile GPU) | Hardware-dependent |
| Classes | 80 (COCO), or custom | ✅ Yes |

---

## Troubleshooting YOLO

### App still shows demo detections
- Verify `yolo.tflite` exists in `assets/model/`
- Check console logs for model loading errors
- Ensure you rebuilt the app with `npx expo prebuild`

### Bounding boxes are in wrong positions
- Check if coordinates are normalized (0-1) or absolute pixels
- Verify input size matches model expectations
- Ensure preprocessing matches training pipeline

### No objects detected
- Lower `confidenceThreshold` in `modelConfig.ts`
- Check if model expects different normalization range
- Verify class names match model output

### Model loads but crashes during inference
- Check input tensor shape and data type
- Verify output parsing matches your specific YOLOv12 variant
- Test model in Python/TensorFlow first

---

## Performance Tips

1. **Reduce input size**: 128x128 is faster than 640x640 (but less accurate)
2. **Increase confidence threshold**: Fewer false positives, faster rendering
3. **Limit detection frequency**: Run YOLO every N frames if needed
4. **Enable GPU delegate**: Already enabled by default in `yoloInference.ts`
5. **Use INT8 quantization**: Fastest inference, but requires calibration dataset

---

## Demo Mode

Until you add a real YOLO model, the app runs in demo mode:
- Simulates 0-3 random detections per frame
- Shows realistic bounding boxes with labels
- Useful for UI/UX testing
- ConvLSTM continues to work independently

This allows you to develop and test the UI before obtaining the actual YOLOv12 model.
