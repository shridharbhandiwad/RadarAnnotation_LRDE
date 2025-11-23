# ONNX Model Export Guide

This guide explains how to export trained XGBoost and Random Forest models to ONNX format for use in the C++ application.

## Overview

The C++ application uses ONNX Runtime to run XGBoost and Random Forest models. ONNX (Open Neural Network Exchange) is an open format for representing machine learning models that enables model portability across different frameworks and platforms.

## Prerequisites

Install required Python packages:

```bash
# For XGBoost models
pip install xgboost skl2onnx onnx onnxmltools

# For Random Forest models
pip install scikit-learn skl2onnx onnx

# For Neural Network models (optional)
pip install tensorflow tf2onnx onnx
```

## Quick Start

### 1. Train Your Model in Python

Ensure your model outputs 11 binary predictions (multi-output classification):

```python
from sklearn.multioutput import MultiOutputClassifier
import xgboost as xgb
import pickle

# Create base model
base_model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1
)

# Wrap for multi-output
model = MultiOutputClassifier(base_model)

# Train (y should have shape [n_samples, 11])
model.fit(X_train, y_train)

# Save model
with open('xgboost_model.pkl', 'wb') as f:
    pickle.dump(model, f)
```

### 2. Export to ONNX

Use the provided export script:

```bash
python3 export_models_to_onnx.py \
    --model-type xgboost \
    --model-path xgboost_model.pkl \
    --output-path xgboost_model.onnx \
    --metadata-path model_metadata.json \
    --output-metadata xgboost_metadata.json
```

### 3. Use in C++ Application

```bash
./radar_tagger_multioutput \
    --model xgboost_model.onnx \
    --metadata xgboost_metadata.json \
    --model-type xgboost \
    --test-data ../data/test.csv
```

## Detailed Export Instructions

### XGBoost Models

#### Python Training Code

```python
import xgboost as xgb
from sklearn.multioutput import MultiOutputClassifier
import pickle
import json
import numpy as np

# Prepare data
X_train = ...  # Shape: [n_samples, n_features]
y_train = ...  # Shape: [n_samples, 11] - 11 binary outputs

# Create and train model
base_model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    random_state=42
)

model = MultiOutputClassifier(base_model, n_jobs=-1)
model.fit(X_train, y_train)

# Save model
with open('xgboost_multioutput.pkl', 'wb') as f:
    pickle.dump(model, f)

# Create metadata
metadata = {
    'model_type': 'xgboost',
    'multi_output': True,
    'num_outputs': 11,
    'num_features': X_train.shape[1],
    'feature_columns': ['x', 'y', 'z', 'vx', 'vy', 'vz', 'ax', 'ay', 'az',
                        'speed', 'speed_2d', 'heading', 'range', 'range_rate',
                        'curvature', 'accel_magnitude', 'vertical_rate', 'altitude_change'],
    'scaler_mean': np.mean(X_train, axis=0).tolist(),
    'scaler_scale': np.std(X_train, axis=0).tolist(),
    'tag_names': ['incoming', 'outgoing', 'fixed_range_ascending', 
                  'fixed_range_descending', 'level_flight', 'linear', 
                  'curved', 'light_maneuver', 'high_maneuver', 
                  'low_speed', 'high_speed']
}

with open('xgboost_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)
```

#### Export to ONNX

```bash
python3 export_models_to_onnx.py \
    --model-type xgboost \
    --model-path xgboost_multioutput.pkl \
    --output-path xgboost_model.onnx \
    --metadata-path xgboost_metadata.json \
    --output-metadata xgboost_metadata_onnx.json
```

### Random Forest Models

#### Python Training Code

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
import pickle
import json
import numpy as np

# Prepare data
X_train = ...  # Shape: [n_samples, n_features]
y_train = ...  # Shape: [n_samples, 11]

# Create and train model
base_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42
)

model = MultiOutputClassifier(base_model, n_jobs=-1)
model.fit(X_train, y_train)

# Save model
with open('rf_multioutput.pkl', 'wb') as f:
    pickle.dump(model, f)

# Create metadata (same as XGBoost example)
metadata = {
    'model_type': 'random_forest',
    'multi_output': True,
    # ... (same as above)
}

with open('rf_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)
```

#### Export to ONNX

```bash
python3 export_models_to_onnx.py \
    --model-type random_forest \
    --model-path rf_multioutput.pkl \
    --output-path rf_model.onnx \
    --metadata-path rf_metadata.json \
    --output-metadata rf_metadata_onnx.json
```

## Multi-Output Model Structure

### Required Output Format

All models must output **11 binary predictions** (0 or 1) corresponding to:

1. **incoming** - Direction tag
2. **outgoing** - Direction tag
3. **fixed_range_ascending** - Vertical motion
4. **fixed_range_descending** - Vertical motion
5. **level_flight** - Vertical motion
6. **linear** - Path shape
7. **curved** - Path shape
8. **light_maneuver** - Maneuver intensity
9. **high_maneuver** - Maneuver intensity
10. **low_speed** - Speed tag
11. **high_speed** - Speed tag

### Training Data Format

Your training labels should be a binary matrix:

```python
# Shape: [n_samples, 11]
y_train = np.array([
    [1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1],  # Sample 1: incoming, level, linear, light, high_speed
    [0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0],  # Sample 2: outgoing, level, curved, high maneuver, low_speed
    # ...
])
```

## Troubleshooting

### Issue: "Required library not installed"

**Solution:** Install the required packages:
```bash
pip install xgboost skl2onnx onnx onnxmltools scikit-learn
```

### Issue: "Multi-output model conversion failed"

**Solution:** Ensure your model is wrapped with `MultiOutputClassifier`:
```python
from sklearn.multioutput import MultiOutputClassifier
model = MultiOutputClassifier(base_model)
```

### Issue: "Input/output shape mismatch"

**Solution:** Verify your model expects the correct number of features (18) and outputs 11 predictions:
```python
# Check input shape
print("Expected features:", X_train.shape[1])  # Should be 18

# Check output shape
predictions = model.predict(X_test)
print("Output shape:", predictions.shape)  # Should be [n_samples, 11]
```

### Issue: "ONNX Runtime error in C++"

**Solution:** 
1. Verify the ONNX model is valid:
```python
import onnx
model = onnx.load('model.onnx')
onnx.checker.check_model(model)
```

2. Check model input/output names:
```python
import onnx
model = onnx.load('model.onnx')
print("Inputs:", [input.name for input in model.graph.input])
print("Outputs:", [output.name for output in model.graph.output])
```

## Performance Considerations

### Model Size
- **XGBoost**: Typically 1-10 MB depending on number of trees
- **Random Forest**: Typically 5-50 MB depending on number of trees
- **Neural Network**: Typically 0.5-5 MB

### Inference Speed (on CPU)
- **XGBoost**: ~1-5 ms per prediction
- **Random Forest**: ~2-10 ms per prediction
- **Neural Network (TFLite)**: ~0.5-3 ms per prediction

### Recommendations
- For fastest inference: Use simple Neural Networks (TFLite)
- For best accuracy on tabular data: Use XGBoost
- For feature importance and interpretability: Use Random Forest
- Optimize models by reducing tree depth and number of estimators

## ONNX Model Validation

After exporting, validate your ONNX model:

```python
import onnx
import onnxruntime as ort
import numpy as np

# Load and check model
model = onnx.load('model.onnx')
onnx.checker.check_model(model)

# Test inference
session = ort.InferenceSession('model.onnx')

# Create dummy input
input_name = session.get_inputs()[0].name
input_shape = session.get_inputs()[0].shape
dummy_input = np.random.randn(1, 18).astype(np.float32)

# Run inference
output = session.run(None, {input_name: dummy_input})
print("Output shape:", output[0].shape)  # Should be [1, 11]
print("Output sample:", output[0])
```

## Additional Resources

- [ONNX Official Documentation](https://onnx.ai/)
- [ONNX Runtime Documentation](https://onnxruntime.ai/)
- [skl2onnx Documentation](https://onnx.ai/sklearn-onnx/)
- [XGBoost to ONNX Guide](https://github.com/onnx/onnxmltools)

## Support

For issues or questions:
1. Check that all dependencies are installed correctly
2. Verify your model structure matches the multi-output requirements
3. Test the ONNX model in Python before using in C++
4. Check C++ application logs for detailed error messages
