# Quick Model Conversion Guide

## 🚀 Quick Start

### 1. Diagnose (Optional)
```bash
python diagnose_model.py "path/to/model.pkl"
```

### 2. Convert
```bash
python convert_model_to_tflite.py \
    --model-type xgboost \
    --model-path "path/to/model.pkl" \
    --output-dir cpp_models
```

## 📋 Common Commands

### Windows Path
```bash
python convert_model_to_tflite.py ^
    --model-type xgboost ^
    --model-path "D:/Zoppler Projects/RadarAnnotation_LRDE/output/models/xgboost_multi_output/model.pkl" ^
    --output-dir cpp_models
```

### Linux/Mac Path
```bash
python convert_model_to_tflite.py \
    --model-type xgboost \
    --model-path output/models/xgboost_multi_output/model.pkl \
    --output-dir cpp_models
```

## ⚡ Quick Fixes

### Error: "No module named 'incoming'"
**Fix**: Use the updated `convert_model_to_tflite.py` (already fixed)

### Error: "joblib not installed"
```bash
pip install joblib
```

### Error: "ONNX tools not installed"
```bash
pip install onnx onnxmltools skl2onnx
```

### Error: "Cannot import src modules"
**Fix**: Run from project root directory where `src/` folder exists

## 📦 Install All Dependencies
```bash
pip install joblib onnx onnxmltools skl2onnx xgboost scikit-learn tensorflow
```

## ✅ Expected Output

```
cpp_models/
├── incoming/xgboost_incoming.onnx
├── outgoing/xgboost_outgoing.onnx
├── level_flight/xgboost_level_flight.onnx
├── ... (more tags)
├── scaler_params.json
└── model_metadata.json
```

## 🔍 Verify Success

```bash
# Check files were created
ls -lh cpp_models/

# Check metadata
cat cpp_models/model_metadata.json
```

## 📚 Documentation

- **Full Guide**: `MODEL_CONVERSION_FIX.md`
- **Summary**: `CONVERSION_FIX_SUMMARY.md`
- **Diagnostic Tool**: `python diagnose_model.py <model_path>`

## 🆘 Still Having Issues?

1. Run diagnostic: `python diagnose_model.py "path/to/model.pkl"`
2. Check you're in project root directory
3. Verify all dependencies installed
4. Read full troubleshooting in `MODEL_CONVERSION_FIX.md`
