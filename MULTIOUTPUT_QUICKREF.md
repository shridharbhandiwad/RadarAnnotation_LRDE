# Multi-Output Models - Quick Reference

## 🎯 11 Output Tags

```
Direction:          incoming, outgoing
Vertical Motion:    fixed_range_ascending, fixed_range_descending, level_flight
Path Shape:         linear, curved
Maneuver Intensity: light_maneuver, high_maneuver
Speed:              low_speed, high_speed
```

## 🚀 Quick Commands

### Build
```bash
cd cpp_inference && ./build.sh
```

### Run Neural Network
```bash
cd build
./radar_tagger_multioutput --model ../cpp_models/lstm/lstm_model.tflite \
                           --metadata ../cpp_models/lstm/model_metadata.json \
                           --model-type nn
```

### Export XGBoost
```bash
python3 export_xgboost_models.py --model xgb_model.pkl \
                                 --output-dir cpp_models_xgb \
                                 --model-type xgboost --format onnx
```

## 💻 C++ Code Snippets

### Basic Prediction
```cpp
RadarTaggerMultiOutput tagger("model.tflite", "metadata.json", 
                              ModelType::NEURAL_NETWORK);
tagger.initialize();

auto result = tagger.predict(sequence);
std::cout << "Tags: " << result.aggregatedLabel << "\n";
```

### Check Specific Tags
```cpp
if (result.tags.incoming && result.tags.high_speed) {
    alertHighSpeedApproach();
}
```

### Per-Tag Confidences
```cpp
for (const auto& [tag, conf] : result.tags.confidences) {
    std::cout << tag << ": " << conf << "\n";
}
```

## 📊 Key Metrics

| Model | Inference | Throughput | Size | Memory |
|-------|-----------|------------|------|--------|
| NN (TFLite) | 2-5 ms | 200-400/s | 67 KB | <1 MB |
| XGBoost* | 1-3 ms | 300-500/s | ~200 KB | <5 MB |
| RF* | 2-4 ms | 250-400/s | ~1 MB | <10 MB |

*Estimated with ONNX Runtime

## 📁 Files

```
cpp_inference/
├── radar_tagger_multioutput.h    # Header
├── radar_tagger_multioutput.cpp  # Implementation
├── main_multioutput.cpp          # Application
└── CMakeLists.txt                # Build (updated)

export_xgboost_models.py          # Export XGB/RF
MULTI_OUTPUT_CPP_GUIDE.md         # Full guide
MULTI_OUTPUT_COMPLETE.md          # Summary
```

## ✅ Status

- Neural Networks: ✅ Ready
- XGBoost Export: ✅ Ready
- Random Forest Export: ✅ Ready
- XGBoost C++ Inference: ⚠️ Requires ONNX Runtime
- RF C++ Inference: ⚠️ Requires ONNX Runtime

## 📚 Documentation

- Full Guide: `MULTI_OUTPUT_CPP_GUIDE.md`
- Complete Summary: `MULTI_OUTPUT_COMPLETE.md`
- Quick Start: `QUICKSTART_CPP.md`
