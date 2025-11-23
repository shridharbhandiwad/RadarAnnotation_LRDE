# Model Conversion Fix - Summary

## Problem
When trying to convert an XGBoost model to ONNX/TFLite format, the conversion failed with:
```
ModuleNotFoundError: No module named 'incoming'
```

## Root Cause Analysis

The error occurred because:

1. **Pickle Serialization Issue**: The model was saved using `pickle`/`joblib`, which serializes Python objects including references to the code/modules that created them.

2. **Custom Class References**: The XGBoost multi-output model includes references to custom classes from the `src` module:
   - `MultiOutputDataAdapter` 
   - `LabelTransformer`
   - Other internal classes

3. **Module Resolution Failure**: When the conversion script tried to load the pickled model, Python couldn't find these custom classes, resulting in the `ModuleNotFoundError`.

4. **Misleading Error**: The error message "No module named 'incoming'" was misleading - it wasn't actually looking for a module called 'incoming', but rather failing to resolve references while trying to reconstruct the pickled object.

## Solutions Implemented

### 1. Updated `convert_model_to_tflite.py`

**Changes made:**

#### A. Added Module Path Setup
```python
# Add src directory to path for loading pickled models
_script_dir = Path(__file__).parent
if _script_dir not in sys.path:
    sys.path.insert(0, str(_script_dir))
```

This ensures the `src` module is importable when loading the model.

#### B. Added Joblib Support
```python
try:
    import joblib
    HAS_JOBLIB = True
except ImportError:
    HAS_JOBLIB = False
```

Joblib is the recommended way to serialize scikit-learn and XGBoost models.

#### C. Implemented Smart Loading with Recovery
```python
# Try joblib first (recommended)
if HAS_JOBLIB:
    try:
        data = joblib.load(model_path)
    except ModuleNotFoundError as e:
        # Import required src modules
        from src.multi_output_adapter import MultiOutputDataAdapter
        from src.label_transformer import LabelTransformer
        # Retry loading
        data = joblib.load(model_path)
```

This automatically imports the required modules and retries loading if it fails the first time.

#### D. Better Error Messages
Added clear, actionable error messages that guide users on how to fix issues.

### 2. Created Diagnostic Tool

**New file: `diagnose_model.py`**

This tool helps identify issues with model files before attempting conversion:

```bash
python diagnose_model.py "path/to/model.pkl"
```

**Features:**
- Checks if file exists and is readable
- Analyzes file size and structure
- Detects model type (XGBoost, Random Forest, etc.)
- Identifies multi-output vs single-output models
- Lists all tags/outputs
- Provides specific guidance for fixing issues

### 3. Created Documentation

**New file: `MODEL_CONVERSION_FIX.md`**

Comprehensive guide covering:
- Problem description
- Root cause explanation
- Usage instructions
- Dependencies
- Troubleshooting guide
- Alternative solutions

## How to Use

### Step 1: Diagnose the Model (Optional but Recommended)

```bash
python diagnose_model.py "D:/Zoppler Projects/RadarAnnotation_LRDE/output/models/xgboost_multi_output/model.pkl"
```

This will tell you:
- If the model file is valid
- What type of model it is
- How many output tags it has
- Whether it can be loaded

### Step 2: Convert the Model

```bash
python convert_model_to_tflite.py \
    --model-type xgboost \
    --model-path "D:/Zoppler Projects/RadarAnnotation_LRDE/output/models/xgboost_multi_output/model.pkl" \
    --output-dir cpp_models
```

**For Linux/Mac:**
```bash
python convert_model_to_tflite.py \
    --model-type xgboost \
    --model-path output/models/xgboost_multi_output/model.pkl \
    --output-dir cpp_models
```

### Step 3: Verify Output

Check the output directory:
```
cpp_models/
├── incoming/
│   └── xgboost_incoming.onnx
├── outgoing/
│   └── xgboost_outgoing.onnx
├── level_flight/
│   └── xgboost_level_flight.onnx
├── ... (more tag directories)
├── scaler_params.json
└── model_metadata.json
```

## What Gets Converted

For a multi-output XGBoost model, the conversion creates:

1. **ONNX Model Files**: One `.onnx` file for each output tag (e.g., `incoming`, `outgoing`, `level_flight`, etc.)
2. **Scaler Parameters**: `scaler_params.json` with normalization parameters
3. **Metadata**: `model_metadata.json` with model information

Each ONNX file can be used independently in C++ applications using ONNX Runtime.

## Dependencies

Make sure you have these installed:
```bash
pip install joblib onnx onnxmltools skl2onnx xgboost scikit-learn
```

## Testing the Fix

You can test with a simple Python script:

```python
import sys
from pathlib import Path

# Add project directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Now try loading
import joblib
from src.multi_output_adapter import MultiOutputDataAdapter
from src.label_transformer import LabelTransformer

# Load model
model_path = "output/models/xgboost_multi_output/model.pkl"
data = joblib.load(model_path)

print("Success! Model loaded.")
print(f"Output tags: {data.get('output_tag_names', [])}")
```

## Troubleshooting

### Issue: "joblib not installed"
```bash
pip install joblib
```

### Issue: "Cannot import src modules"
- Make sure you're running from the project root directory
- Ensure `src/` directory exists with all Python files
- Check that `src/__init__.py` exists

### Issue: "ONNX conversion failed"
Some models may not convert to ONNX. The script will automatically fall back to JSON format:
```
xgboost_<tag_name>.json
```

These can be used with the XGBoost C++ API.

### Issue: Model file is corrupted
If the model file is corrupted:
1. Re-train the model using the current code
2. Make sure to save it properly with `joblib.dump()`

## Next Steps

After successful conversion:

### For ONNX Models (Recommended)
Use ONNX Runtime C++ API:
```cpp
#include <onnxruntime/core/session/onnxruntime_cxx_api.h>

Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "XGBoostInference");
Ort::SessionOptions session_options;
Ort::Session session(env, "xgboost_incoming.onnx", session_options);

// Run inference...
```

See: https://onnxruntime.ai/docs/get-started/with-cpp.html

### For JSON Models (Fallback)
Use XGBoost C++ API:
```cpp
#include <xgboost/c_api.h>

BoosterHandle booster;
XGBoosterCreate(NULL, 0, &booster);
XGBoosterLoadModel(booster, "xgboost_incoming.json");

// Run prediction...
```

See: https://xgboost.readthedocs.io/

## Benefits of This Fix

1. **Automatic Recovery**: Script automatically imports required modules
2. **Better Error Messages**: Clear guidance when things go wrong
3. **Diagnostic Tool**: Identify issues before attempting conversion
4. **Multiple Fallbacks**: Tries joblib → pickle → alternative approaches
5. **Comprehensive Documentation**: Complete guide for troubleshooting

## Files Modified/Created

1. **Modified**: `convert_model_to_tflite.py`
   - Added module path setup
   - Improved model loading with joblib
   - Added automatic module import recovery
   - Better error handling and messages

2. **Created**: `diagnose_model.py`
   - Tool to diagnose model file issues
   - Analyzes model structure
   - Provides specific guidance

3. **Created**: `MODEL_CONVERSION_FIX.md`
   - Comprehensive troubleshooting guide
   - Usage instructions
   - Dependency information

4. **Created**: `CONVERSION_FIX_SUMMARY.md` (this file)
   - Summary of changes
   - Quick reference guide

## Support

If you continue to experience issues:
1. Run the diagnostic tool first: `python diagnose_model.py <model_path>`
2. Check that all dependencies are installed
3. Verify you're running from the project root directory
4. Check file paths (Windows vs. Linux path format)
5. Try re-training the model with the latest code

## Success Criteria

The fix is successful when:
1. ✅ Model loads without `ModuleNotFoundError`
2. ✅ ONNX files are created for each output tag
3. ✅ Scaler parameters are exported to JSON
4. ✅ Model metadata is saved
5. ✅ No errors during conversion process

You should see output like:
```
Loading XGBoost model from: <path>
Loading model with joblib...
Found multi-output XGBoost with 11 models
✓ Converted incoming to ONNX: cpp_models/incoming/xgboost_incoming.onnx
✓ Converted outgoing to ONNX: cpp_models/outgoing/xgboost_outgoing.onnx
...
Conversion Complete!
```
