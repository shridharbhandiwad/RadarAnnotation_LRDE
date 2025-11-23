# Quick Start: C++ Real-Time Radar Tagging

This guide will help you quickly get started with the C++ real-time radar tagging application.

## Overview

This application demonstrates how to:
1. Load TensorFlow Lite models in C++
2. Perform real-time inference on radar trajectory data
3. Evaluate model performance with comprehensive metrics
4. Integrate ML models into production C++ systems

## Prerequisites

```bash
# Linux/Ubuntu
sudo apt-get update
sudo apt-get install -y build-essential cmake git

# macOS
brew install cmake

# Windows
# Install Visual Studio 2019+ with C++ tools
# Install CMake from https://cmake.org/download/
```

## Step 1: Convert Models

First, convert your trained Keras models to TensorFlow Lite format:

```bash
cd /workspace

# Install Python dependencies (if not already installed)
pip3 install tensorflow numpy

# Convert LSTM model to TFLite
python3 convert_model_to_tflite.py --model-type lstm --output-dir cpp_models
```

This creates:
- `cpp_models/lstm/lstm_model.tflite` - Optimized model (67 KB)
- `cpp_models/lstm/model_metadata.json` - Model configuration
- `cpp_models/lstm/test_data.bin` - Sample test data

## Step 2: Build C++ Application

```bash
cd /workspace/cpp_inference

# Run build script (Linux/Mac)
./build.sh

# Or manually:
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --parallel
```

**Note**: The first build may take 10-15 minutes as it downloads and compiles TensorFlow Lite from source.

## Step 3: Run Inference

### Basic Test (Synthetic Data)

```bash
cd build

./radar_tagger \
    --model ../cpp_models/lstm/lstm_model.tflite \
    --metadata ../cpp_models/lstm/model_metadata.json
```

Expected output:
```
==============================================
  Radar Trajectory Real-Time Tagger
  Using TensorFlow Lite for C++ Deployment
==============================================

Initializing Radar Tagger...
Model loaded successfully
Tensors allocated successfully

=== Model Information ===
Number of classes: 20
Sequence length: 20
Number of features: 18

=== Prediction Result ===
Predicted Class: 10
Class Name: outgoing,level,linear,light_maneuver
Inference Time: 2.450 ms
```

### Test with Binary Data

```bash
./radar_tagger \
    --model ../cpp_models/lstm/lstm_model.tflite \
    --metadata ../cpp_models/lstm/model_metadata.json \
    --test-data ../cpp_models/lstm/test_data.bin \
    --test-binary \
    --samples 10 \
    --seq-length 20 \
    --features 18
```

### Test with Real CSV Data

```bash
./radar_tagger \
    --model ../cpp_models/lstm/lstm_model.tflite \
    --metadata ../cpp_models/lstm/model_metadata.json \
    --test-data ../../data/high_volume_simulation_labeled.csv
```

### Benchmark Mode

Test inference performance:

```bash
./radar_tagger \
    --model ../cpp_models/lstm/lstm_model.tflite \
    --metadata ../cpp_models/lstm/model_metadata.json \
    --test-data ../cpp_models/lstm/test_data.bin \
    --test-binary \
    --benchmark
```

Expected performance on modern CPU:
- Average inference time: 2-5 ms
- Throughput: 200-400 inferences/sec
- Model size: 67 KB

## Step 4: Integrate into Your Application

### Simple Integration Example

```cpp
#include "radar_tagger.h"

int main() {
    // 1. Create and initialize tagger
    RadarTagger tagger("model.tflite", "metadata.json", 4);
    if (!tagger.initialize()) {
        return 1;
    }
    
    // 2. Prepare radar data
    RadarSequence sequence;
    sequence.trackId = 1;
    
    for (int i = 0; i < 20; i++) {
        RadarDataPoint point;
        // Fill in radar measurements
        point.x = 10000.0f + i * 50.0f;
        point.y = 10000.0f;
        point.z = 2000.0f;
        point.vx = 50.0f;
        // ... fill other fields ...
        sequence.points.push_back(point);
    }
    
    // 3. Run inference
    auto result = tagger.predict(sequence);
    
    // 4. Use results
    if (result.success) {
        std::cout << "Class: " << result.className << "\n";
        std::cout << "Confidence: " << 
            result.classProbabilities[result.predictedClass] << "\n";
        std::cout << "Inference time: " << result.inferenceTimeMs << " ms\n";
    }
    
    return 0;
}
```

### Real-Time Processing Loop

```cpp
RadarTagger tagger("model.tflite", "metadata.json");
tagger.initialize();

std::map<int, RadarSequence> activeTrackswhile (system_is_running) {
    // Get new radar measurement
    RadarDataPoint newPoint = radar_system.getNextMeasurement();
    int trackId = (int)newPoint.trackid;
    
    // Add to active track
    activeTracks[trackId].points.push_back(newPoint);
    
    // When we have enough points for classification
    if (activeTracks[trackId].points.size() >= 20) {
        // Run inference
        auto result = tagger.predict(activeTracks[trackId]);
        
        // Process classification
        if (result.success) {
            updateTrackClassification(trackId, result.className);
            
            // Optional: Log performance
            if (tagger.getMetrics().totalInferences % 100 == 0) {
                tagger.getMetrics().print();
            }
        }
        
        // Slide window or reset
        activeTracks[trackId].points.erase(
            activeTracks[trackId].points.begin()
        );
    }
}
```

## Performance Optimization Tips

1. **Thread Count**: Adjust based on your CPU
   ```bash
   --threads 8  # For 8+ core systems
   ```

2. **Batch Processing**: Process multiple sequences together
   ```cpp
   auto results = tagger.predictBatch(sequences);
   ```

3. **Model Optimization**: The TFLite model is already optimized for:
   - Reduced precision (float32)
   - Operator fusion
   - Memory efficiency

## Deployment Checklist

- [x] Convert Keras model to TFLite
- [x] Build C++ application
- [x] Test with sample data
- [x] Verify inference performance
- [ ] Profile on target hardware
- [ ] Integration testing
- [ ] Production deployment

## Troubleshooting

### Build Issues

**Problem**: CMake version too old
```bash
# Solution: Upgrade CMake
pip3 install --upgrade cmake
```

**Problem**: TensorFlow Lite download fails
```bash
# Solution: Check internet connection or use mirrors
# Or download manually and set paths
```

### Runtime Issues

**Problem**: "Model file not found"
```bash
# Solution: Use absolute paths
./radar_tagger --model $(pwd)/../cpp_models/lstm/lstm_model.tflite ...
```

**Problem**: Slow inference
```bash
# Solution: Increase threads, check build type
./radar_tagger --threads 8 ...
# Rebuild with: cmake .. -DCMAKE_BUILD_TYPE=Release
```

## Next Steps

1. **Test with Your Data**: Load your real radar data
2. **Profile Performance**: Run benchmarks on target hardware
3. **Integrate**: Add to your existing C++ application
4. **Deploy**: Package for production deployment

## File Structure

```
workspace/
├── convert_model_to_tflite.py    # Python conversion script
├── cpp_models/                    # Converted models
│   └── lstm/
│       ├── lstm_model.tflite     # TFLite model (67 KB)
│       ├── model_metadata.json   # Model info
│       └── test_data.bin         # Test data
├── cpp_inference/                 # C++ application
│   ├── radar_tagger.h            # Header
│   ├── radar_tagger.cpp          # Implementation
│   ├── main.cpp                  # Main application
│   ├── CMakeLists.txt            # Build config
│   ├── build.sh                  # Build script
│   └── README.md                 # Documentation
└── QUICKSTART_CPP.md             # This file
```

## Additional Resources

- Full documentation: `cpp_inference/README.md`
- TensorFlow Lite C++ Guide: https://www.tensorflow.org/lite/guide/inference
- Parent project README: `README.md`

## Support

For issues or questions:
1. Check `cpp_inference/README.md` for detailed documentation
2. Review error messages carefully
3. Verify model and data file paths
4. Test with sample data first

---

**Ready to deploy ML models in C++!** 🚀
