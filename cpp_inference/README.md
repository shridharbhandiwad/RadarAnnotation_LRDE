# Radar Trajectory Real-Time Tagger - C++ Application

This C++ application provides real-time radar trajectory classification using TensorFlow Lite models converted from Keras.

## Features

- **Real-time Inference**: Fast inference using TensorFlow Lite C++ API
- **Multi-threaded**: Configurable number of threads for optimal performance
- **Flexible Input**: Support for both CSV and binary data formats
- **Comprehensive Metrics**: Performance tracking and evaluation
- **Benchmarking**: Built-in benchmark mode for performance testing

## Requirements

- C++17 compatible compiler (GCC 7+, Clang 5+, MSVC 2019+)
- CMake 3.15 or higher
- Internet connection (for downloading dependencies during build)

## Building

### Quick Build

```bash
# Create build directory
mkdir build && cd build

# Configure
cmake ..

# Build
cmake --build . --config Release

# The executable will be in build/radar_tagger (Linux/Mac) or build/Release/radar_tagger.exe (Windows)
```

### Build Options

```bash
# Use system-installed TensorFlow Lite (if available)
cmake .. -DUSE_SYSTEM_TFLITE=ON

# Specify build type
cmake .. -DCMAKE_BUILD_TYPE=Release  # or Debug

# Specify number of parallel jobs
cmake --build . --parallel 4
```

## Usage

### Basic Usage

```bash
# Run with model and metadata
./radar_tagger --model ../cpp_models/lstm/lstm_model.tflite \
               --metadata ../cpp_models/lstm/model_metadata.json
```

### With Test Data

```bash
# Using CSV file
./radar_tagger --model ../cpp_models/lstm/lstm_model.tflite \
               --metadata ../cpp_models/lstm/model_metadata.json \
               --test-data ../data/high_volume_simulation_labeled.csv

# Using binary file
./radar_tagger --model ../cpp_models/lstm/lstm_model.tflite \
               --metadata ../cpp_models/lstm/model_metadata.json \
               --test-data ../cpp_models/lstm/test_data.bin \
               --test-binary \
               --samples 10 \
               --seq-length 20 \
               --features 18
```

### Benchmark Mode

```bash
./radar_tagger --model ../cpp_models/lstm/lstm_model.tflite \
               --metadata ../cpp_models/lstm/model_metadata.json \
               --test-data ../cpp_models/lstm/test_data.bin \
               --test-binary \
               --benchmark
```

### Command Line Options

- `--model PATH`: Path to TFLite model file (required)
- `--metadata PATH`: Path to model metadata JSON (required)
- `--test-data PATH`: Path to test data file (CSV or binary)
- `--test-binary`: Indicate that test data is in binary format
- `--samples N`: Number of samples in binary file (default: 10)
- `--seq-length N`: Sequence length for binary data (default: 20)
- `--features N`: Number of features per time step (default: 18)
- `--threads N`: Number of threads for inference (default: 4)
- `--benchmark`: Run in benchmark mode
- `--help`: Show help message

## Model Conversion

Before using the C++ application, you need to convert your Keras models to TensorFlow Lite format:

```bash
cd /workspace
python3 convert_model_to_tflite.py --model-type lstm --output-dir cpp_models
```

This will create:
- `cpp_models/lstm/lstm_model.tflite` - The TFLite model
- `cpp_models/lstm/model_metadata.json` - Model metadata (classes, normalization parameters, etc.)
- `cpp_models/lstm/test_data.bin` - Sample test data
- `cpp_models/lstm/test_data.csv` - Sample test data in CSV format

## Performance

The application tracks detailed performance metrics:

- **Average Inference Time**: Mean time per prediction
- **Min/Max Inference Time**: Range of inference times
- **Throughput**: Inferences per second
- **Total Inferences**: Number of predictions made

Example output:

```
=== Performance Metrics ===
Total Inferences: 100
Average Inference Time: 2.45 ms
Min Inference Time: 2.12 ms
Max Inference Time: 3.87 ms
Total Time: 245.32 ms
Throughput: 408.16 inferences/sec
```

## Integration into Your Application

### Basic Integration

```cpp
#include "radar_tagger.h"

int main() {
    // Create tagger
    RadarTagger tagger("model.tflite", "metadata.json", 4);
    
    // Initialize
    if (!tagger.initialize()) {
        return 1;
    }
    
    // Create radar sequence
    RadarSequence sequence;
    // ... fill sequence with radar data ...
    
    // Predict
    auto result = tagger.predict(sequence);
    
    if (result.success) {
        std::cout << "Predicted class: " << result.className << "\n";
        std::cout << "Confidence: " << result.classProbabilities[result.predictedClass] << "\n";
    }
    
    return 0;
}
```

### Real-Time Streaming

```cpp
// In a real-time processing loop
while (radarSystem.hasData()) {
    // Accumulate points for current track
    RadarDataPoint point = radarSystem.getNextPoint();
    currentSequence.points.push_back(point);
    
    // When sequence is complete (e.g., 20 points)
    if (currentSequence.points.size() >= 20) {
        auto result = tagger.predict(currentSequence);
        
        // Process result
        processClassification(result);
        
        // Prepare for next sequence
        currentSequence.points.clear();
    }
}
```

## Input Data Format

### RadarDataPoint Structure

Each radar data point contains:
- Position: `x`, `y`, `z`
- Velocity: `vx`, `vy`, `vz`
- Acceleration: `ax`, `ay`, `az`
- Derived features: `speed`, `speed_2d`, `heading`, `range`, `range_rate`
- Motion features: `curvature`, `accel_magnitude`, `vertical_rate`, `altitude_change`

### CSV Format

```csv
time,trackid,x,y,z,vx,vy,vz,ax,ay,az,speed,speed_2d,heading,range,range_rate,curvature,accel_magnitude,vertical_rate,altitude_change
0.0,1.0,10000.0,10000.0,2000.0,-26.68,22.66,0.0,0.0,0.0,0.0,35.0,35.0,310.34,14282.86,0.0,0.0,0.0,0.0,0.0
...
```

### Binary Format

- Data type: `float32` (4 bytes per value)
- Layout: `[n_samples][sequence_length][n_features]`
- Contiguous memory layout
- Example: 10 samples × 20 timesteps × 18 features = 3600 floats = 14,400 bytes

## Troubleshooting

### Build Errors

1. **TensorFlow Lite download fails**: Check internet connection or use a mirror
2. **Compiler too old**: Upgrade to GCC 7+, Clang 5+, or MSVC 2019+
3. **Out of memory**: Reduce parallel build jobs: `cmake --build . --parallel 1`

### Runtime Errors

1. **Model file not found**: Check that the path to `.tflite` file is correct
2. **Metadata parsing error**: Verify JSON format in metadata file
3. **Input size mismatch**: Ensure sequence length and feature count match model requirements

### Performance Issues

1. **Slow inference**: Try increasing `--threads` parameter
2. **High latency**: Check if running in Debug mode; rebuild with Release mode
3. **Memory usage**: TFLite models are optimized for low memory footprint

## License

This application is part of the Radar Data Annotation project.

## References

- [TensorFlow Lite C++ Guide](https://www.tensorflow.org/lite/guide/inference#load_and_run_a_model_in_c)
- [TensorFlow Lite for Mobile & Edge](https://www.tensorflow.org/lite)
- Parent project: Radar Data Annotation Application
