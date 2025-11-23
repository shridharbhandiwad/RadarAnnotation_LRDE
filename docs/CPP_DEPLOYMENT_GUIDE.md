# C++ Deployment Integration Guide

This guide explains how to use the C++ Deployment feature integrated into the GUI for converting, building, and evaluating models with C++ inference.

## Overview

The C++ Deployment panel provides a complete workflow for deploying trained Keras models to C++ with TensorFlow Lite for high-performance inference. This is ideal for:

- **Production deployment** requiring fast, lightweight inference
- **Edge devices** with limited resources
- **Real-time systems** needing low latency
- **Performance benchmarking** to evaluate inference speed

## Features

✅ **Model Conversion**: Convert Keras (.h5) models to TensorFlow Lite (.tflite)  
✅ **Automatic Build**: Compile C++ inference application with CMake  
✅ **Model Evaluation**: Test models with C++ for accuracy verification  
✅ **Performance Benchmarking**: Measure inference time and throughput  
✅ **Multi-threading**: Configure thread count for optimal performance  
✅ **Test Data Generation**: Automatically create test datasets  

## Quick Start

### 1. Access the C++ Deployment Panel

1. Launch the GUI: `python3 run.py` (or `python run.py` on Windows)
2. In the left sidebar, click **⚙️ C++ Deployment**

### 2. Three-Step Workflow

#### Step 1: Convert Model to TensorFlow Lite

1. Click **📁 Select Keras Model (.h5)**
2. Navigate to your trained model (typically in `output/test_lstm/` or `output/test_transformer/`)
3. Select the model type (LSTM or Transformer) from the dropdown
4. Click **🔄 Convert to TFLite**

**What happens:**
- Model is converted to TensorFlow Lite format
- Metadata (classes, normalization params) is exported
- Test data is automatically generated
- Files are saved to `cpp_models/[model_type]/`

**Output files:**
- `cpp_models/lstm/lstm_model.tflite` - The converted model
- `cpp_models/lstm/model_metadata.json` - Model configuration
- `cpp_models/lstm/test_data.bin` - Binary test data
- `cpp_models/lstm/test_data.csv` - CSV test data

#### Step 2: Build C++ Application

1. After successful conversion, the **🔨 Build C++ Application** button becomes enabled
2. Click to start the build process

**What happens:**
- CMake configures the build system
- Downloads TensorFlow Lite library (first time only)
- Compiles the C++ inference application
- Creates executable in `cpp_inference/build/`

**Requirements:**
- CMake 3.15 or higher
- C++17 compatible compiler (GCC 7+, Clang 5+, MSVC 2019+)
- Internet connection (for downloading dependencies)

#### Step 3: Evaluate Model with C++

1. Configure evaluation options:
   - **Threads**: Number of CPU threads (1-16, default: 4)
   - **🚀 Benchmark Mode**: Enable for detailed performance metrics
2. Click **🎯 Run C++ Evaluation**

**What happens:**
- C++ application runs inference on test data
- Predictions are compared with expected outputs
- Performance metrics are displayed (inference time, throughput)
- Results shown in the output panel

## Output Interpretation

### Standard Evaluation Output

```
=== Model Information ===
Model: lstm_model.tflite
Classes: ['bird', 'drone', 'helicopter']
Sequence Length: 20
Features: 18

=== Predictions ===
Sample 1: Predicted 'drone' (confidence: 0.95)
Sample 2: Predicted 'bird' (confidence: 0.87)
...

=== Performance Metrics ===
Total Inferences: 100
Average Inference Time: 2.45 ms
Min Inference Time: 2.12 ms
Max Inference Time: 3.87 ms
Total Time: 245.32 ms
Throughput: 408.16 inferences/sec
```

### Benchmark Mode Output

When **Benchmark Mode** is enabled, you get additional metrics:

- Warm-up runs (excluded from statistics)
- Detailed timing for each inference
- Memory usage statistics
- Thread utilization information

## Advanced Usage

### Using Custom Test Data

1. In Step 3, click **📄 Select Test Data (Optional)**
2. Choose either:
   - **CSV file**: Standard radar data format with labeled trajectories
   - **Binary file**: Pre-processed binary data (`.bin` format)

### Adjusting Thread Count

The thread count affects performance:
- **1 thread**: Baseline performance, lowest CPU usage
- **4 threads**: Good balance (default)
- **8-16 threads**: Maximum performance on multi-core systems

**Note**: Performance gains diminish beyond 8 threads for most models.

### Command-Line Usage

After building, you can also run the C++ application directly:

```bash
# From workspace root
cd cpp_inference/build

# Basic usage
./radar_tagger --model ../../cpp_models/lstm/lstm_model.tflite \
               --metadata ../../cpp_models/lstm/model_metadata.json

# With test data
./radar_tagger --model ../../cpp_models/lstm/lstm_model.tflite \
               --metadata ../../cpp_models/lstm/model_metadata.json \
               --test-data ../../cpp_models/lstm/test_data.bin \
               --test-binary

# Benchmark mode
./radar_tagger --model ../../cpp_models/lstm/lstm_model.tflite \
               --metadata ../../cpp_models/lstm/model_metadata.json \
               --test-data ../../cpp_models/lstm/test_data.bin \
               --test-binary \
               --benchmark \
               --threads 8
```

## Troubleshooting

### Conversion Fails

**Problem**: Model conversion fails with TensorFlow errors

**Solutions**:
1. Ensure TensorFlow is installed: `pip install tensorflow`
2. Check model file is valid (not corrupted)
3. Verify model was trained successfully
4. Check output panel for specific error messages

### Build Fails

**Problem**: C++ build fails with CMake errors

**Solutions**:
1. Install CMake: `sudo apt install cmake` (Linux) or download from cmake.org
2. Check compiler version: `gcc --version` (need GCC 7+)
3. Ensure internet connection for TensorFlow Lite download
4. Try rebuilding: Delete `cpp_inference/build/` and retry

**Problem**: CMake can't find compiler

**Solutions**:
- Linux: `sudo apt install build-essential`
- Windows: Install Visual Studio 2019 or later with C++ support
- macOS: Install Xcode command-line tools

### Evaluation Fails

**Problem**: Evaluation times out or produces errors

**Solutions**:
1. Verify model was converted successfully
2. Check that metadata file exists
3. Reduce thread count if system resources are limited
4. Check test data format matches expectations

### Performance Issues

**Problem**: Slow inference speed

**Solutions**:
1. Enable optimization in model conversion
2. Increase thread count (Step 3)
3. Use benchmark mode to identify bottlenecks
4. Ensure running in Release mode (not Debug)

## Performance Tips

### Optimization Strategies

1. **Model Size**: Smaller models (LSTM) are faster than larger (Transformer)
2. **Sequence Length**: Shorter sequences reduce inference time
3. **Batch Processing**: Process multiple samples together when possible
4. **Thread Tuning**: Experiment with thread counts for your hardware

### Expected Performance

Typical inference times on modern hardware:

| Model Type  | Sequence Length | Inference Time | Throughput    |
|-------------|----------------|----------------|---------------|
| LSTM        | 20             | 2-4 ms         | 250-500/sec   |
| LSTM        | 50             | 5-10 ms        | 100-200/sec   |
| Transformer | 20             | 5-15 ms        | 65-200/sec    |
| Transformer | 50             | 15-40 ms       | 25-65/sec     |

*Note: Times vary based on hardware, model complexity, and thread count*

## Integration into Applications

The C++ inference code can be integrated into your own applications:

### Basic Integration

```cpp
#include "radar_tagger.h"

int main() {
    // Initialize tagger
    RadarTagger tagger("model.tflite", "metadata.json", 4);
    if (!tagger.initialize()) {
        std::cerr << "Failed to initialize\n";
        return 1;
    }
    
    // Create radar sequence
    RadarSequence sequence;
    // ... populate sequence with data ...
    
    // Predict
    auto result = tagger.predict(sequence);
    
    if (result.success) {
        std::cout << "Class: " << result.className << "\n";
        std::cout << "Confidence: " 
                  << result.classProbabilities[result.predictedClass] << "\n";
    }
    
    return 0;
}
```

### Real-Time Processing

```cpp
// In a real-time loop
while (radarSystem.hasData()) {
    RadarDataPoint point = radarSystem.getNextPoint();
    currentSequence.points.push_back(point);
    
    if (currentSequence.points.size() >= 20) {
        auto result = tagger.predict(currentSequence);
        processClassification(result);
        currentSequence.points.clear();
    }
}
```

## Architecture

### Data Flow

```
[Keras Model .h5]
    ↓ (TensorFlow Lite Converter)
[TFLite Model .tflite] + [Metadata .json]
    ↓ (CMake Build)
[C++ Executable]
    ↓ (Inference)
[Predictions + Performance Metrics]
```

### Component Overview

1. **GUI Panel** (`src/gui.py`): User interface for the workflow
2. **Conversion Script** (`convert_model_to_tflite.py`): Keras → TFLite conversion
3. **C++ Library** (`cpp_inference/radar_tagger.*`): TFLite inference wrapper
4. **C++ Application** (`cpp_inference/main.cpp`): Standalone evaluation tool
5. **Build System** (`cpp_inference/CMakeLists.txt`): CMake configuration

## File Structure

```
workspace/
├── src/
│   └── gui.py                      # GUI with C++ Deployment panel
├── convert_model_to_tflite.py     # Conversion script
├── cpp_inference/
│   ├── CMakeLists.txt             # Build configuration
│   ├── radar_tagger.h             # C++ inference API
│   ├── radar_tagger.cpp           # Implementation
│   ├── main.cpp                   # Standalone application
│   └── build/                     # Build output (created)
│       └── radar_tagger           # Executable
├── cpp_models/                    # Converted models (created)
│   ├── lstm/
│   │   ├── lstm_model.tflite
│   │   ├── model_metadata.json
│   │   ├── test_data.bin
│   │   └── test_data.csv
│   └── transformer/
│       └── (similar structure)
└── output/                        # Trained models
    ├── test_lstm/
    │   └── lstm_model.h5
    └── test_transformer/
        └── transformer_model.h5
```

## Additional Resources

- **C++ README**: See `cpp_inference/README.md` for detailed C++ API documentation
- **Model Training**: Train models using the AI Tagging panel
- **Data Preparation**: Use Data Extraction panel to prepare datasets
- **Performance Analysis**: Use benchmark mode for detailed profiling

## FAQ

**Q: Do I need to rebuild the C++ application every time?**  
A: No, only rebuild if you update the C++ code. You can reuse the same executable for different models.

**Q: Can I deploy this to embedded systems?**  
A: Yes! TensorFlow Lite is designed for edge deployment. Copy the `.tflite` file and metadata to your target system.

**Q: How do I improve inference speed?**  
A: Try increasing threads, using a smaller model, reducing sequence length, or enabling model optimization during conversion.

**Q: What's the difference between LSTM and Transformer models?**  
A: LSTMs are generally faster but less accurate for complex patterns. Transformers are more powerful but slower.

**Q: Can I use this with custom data formats?**  
A: Yes, modify the `RadarDataPoint` structure in `radar_tagger.h` and rebuild. The GUI uses standard CSV/binary formats.

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review the C++ README: `cpp_inference/README.md`
3. Check logs in the GUI output panel
4. Verify all prerequisites are installed

## License

This integration is part of the Radar Data Annotation Application project.
