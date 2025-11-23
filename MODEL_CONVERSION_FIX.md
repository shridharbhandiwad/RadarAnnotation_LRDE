# XGBoost Model Conversion Fix

## Issue
When trying to convert an XGBoost model to ONNX/TFLite format, you may encounter:
```
ModuleNotFoundError: No module named 'incoming'
```

## Root Cause
This error occurs when the pickled model file contains references to modules or code that don't exist in the current environment. Specifically:

1. The model was saved using `pickle`/`joblib` which serializes Python objects
2. The serialized model contains references to custom classes from the `src` module (like `MultiOutputDataAdapter`)
3. When loading, Python tries to reconstruct these objects but can't find the modules

## Solution

### Automatic Fix (Already Applied)
The `convert_model_to_tflite.py` script has been updated with:

1. **Better Module Path Handling**: Adds the project directory to Python's path
2. **Improved Loading**: Uses `joblib` (preferred) with fallback to `pickle`
3. **Module Import Recovery**: Automatically imports required `src` modules when loading fails
4. **Better Error Messages**: Provides clear guidance when loading fails

### Usage

#### Convert XGBoost Model to ONNX
```bash
python convert_model_to_tflite.py --model-type xgboost \
    --model-path "D:/Zoppler Projects/RadarAnnotation_LRDE/output/models/xgboost_multi_output/model.pkl" \
    --output-dir cpp_models
```

#### For Linux/Mac paths:
```bash
python convert_model_to_tflite.py --model-type xgboost \
    --model-path output/models/xgboost_multi_output/model.pkl \
    --output-dir cpp_models
```

## What the Fix Does

### 1. Module Path Setup
```python
# Add src directory to path for loading pickled models
_script_dir = Path(__file__).parent
if _script_dir not in sys.path:
    sys.path.insert(0, str(_script_dir))
```

### 2. Smart Model Loading
```python
# Try joblib first (recommended for sklearn/xgboost)
if HAS_JOBLIB:
    data = joblib.load(model_path)
else:
    # Fallback to pickle
    with open(model_path, 'rb') as f:
        data = pickle.load(f)
```

### 3. Module Import Recovery
```python
except ModuleNotFoundError as e:
    # Import required src modules
    from src.multi_output_adapter import MultiOutputDataAdapter
    from src.label_transformer import LabelTransformer
    # Retry loading
    data = joblib.load(model_path)
```

## Dependencies

Make sure you have these installed:
```bash
pip install joblib onnx onnxmltools skl2onnx xgboost
```

## Expected Output

When conversion is successful, you'll see:
```
Loading XGBoost model from: <model_path>
Loading model with joblib...
Found multi-output XGBoost with N models
Tags: [incoming, outgoing, level_flight, ...]
✓ Converted <tag_name> to ONNX: <output_path>
  Model size: XX.XX KB
...
Conversion Complete!
Models saved to: cpp_models
```

## Output Structure

For a multi-output XGBoost model, the script creates:
```
cpp_models/
├── incoming/
│   └── xgboost_incoming.onnx
├── outgoing/
│   └── xgboost_outgoing.onnx
├── level_flight/
│   └── xgboost_level_flight.onnx
├── ...
├── scaler_params.json
└── model_metadata.json
```

## Troubleshooting

### Error: "joblib not available"
```bash
pip install joblib
```

### Error: "ONNX tools not installed"
```bash
pip install onnx onnxmltools skl2onnx
```

### Error: "XGBoost not installed"
```bash
pip install xgboost
```

### Error: "Failed to import src modules"
This means the `src` directory is missing or incomplete. Ensure:
1. You're running the script from the project root directory
2. The `src` directory exists with all necessary Python files
3. `src/__init__.py` exists (can be empty)

### Manual Module Path Fix
If the automatic fix doesn't work, you can manually add the path:
```python
import sys
sys.path.insert(0, '/path/to/your/project')
```

## Alternative: Re-save Model Without Custom Classes

If you continue to have issues, you can re-save the model in a cleaner format:

```python
import joblib
from pathlib import Path

# Load the problematic model
data = joblib.load('model.pkl')

# Extract only the essential components
clean_data = {
    'models': data['models'],  # The actual XGBoost models
    'scaler': data.get('scaler'),  # Scaler for normalization
    # Don't save adapter or other custom classes
}

# Save in a cleaner format
joblib.dump(clean_data, 'model_clean.pkl')
```

Then use `model_clean.pkl` for conversion.

## Next Steps

After successful conversion:

1. **For ONNX models**: Use ONNX Runtime C++ API
   - See: https://onnxruntime.ai/docs/get-started/with-cpp.html

2. **For JSON models** (fallback): Use XGBoost C++ API
   - Link with libxgboost
   - See: https://xgboost.readthedocs.io/

## Support

If you continue to experience issues:
1. Check that all dependencies are installed
2. Verify the model file is not corrupted
3. Try re-training the model with the latest code
4. Check file paths are correct (Windows vs. Linux paths)
