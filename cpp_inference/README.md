# Radar Trajectory Real-Time Tagger - C++ Application

This C++ application provides real-time radar trajectory classification with multi-output support for Neural Networks (TFLite), XGBoost, and Random Forest models.

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

### Windows MinGW Users - Important Note ⚠️

If you're building on Windows with MinGW and encounter the `eight_bit_int_gemm` compilation error, use the automated fix script:

```batch
fix_and_build_windows.bat clean
```

See [EIGHT_BIT_INT_GEMM_FIX.md](EIGHT_BIT_INT_GEMM_FIX.md) for details.

### Quick Build

**Linux/macOS:**
```bash
# Create build directory
mkdir build && cd build

# Configure
cmake ..

# Build
cmake --build . --config Release

# The executable will be in build/radar_tagger
```

**Windows (MinGW):**
```batch
# Use the automated build script (recommended)
fix_and_build_windows.bat

# Or manual build:
mkdir build
cd build
cmake -G "MinGW Makefiles" -DCMAKE_BUILD_TYPE=Release ..
cmake --build . --config Release

# The executable will be in build\radar_tagger.exe
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
# Neural Network multi-output model
./radar_tagger_multioutput --model <path_to_nn_model.tflite> \
                           --metadata <path_to_metadata.json> \
                           --model-type nn

# XGBoost model (requires additional implementation)
./radar_tagger_multioutput --model <path_to_xgb_model.json> \
                           --metadata <path_to_metadata.json> \
                           --model-type xgboost

# Random Forest model (requires additional implementation)
./radar_tagger_multioutput --model <path_to_rf_model.pkl> \
                           --metadata <path_to_metadata.json> \
                           --model-type rf
```

### With Test Data

```bash
# Using CSV file with ground truth
./radar_tagger_multioutput --model <path_to_model> \
                           --metadata <path_to_metadata.json> \
                           --model-type nn \
                           --test-data ../data/high_volume_simulation_labeled.csv \
                           --load-gt

# Using binary file
./radar_tagger_multioutput --model <path_to_model> \
                           --metadata <path_to_metadata.json> \
                           --model-type nn \
                           --test-data test_data.bin \
                           --test-binary \
                           --samples 10 \
                           --seq-length 20 \
                           --features 18
```

### Benchmark Mode

```bash
./radar_tagger_multioutput --model <path_to_model> \
                           --metadata <path_to_metadata.json> \
                           --model-type nn \
                           --test-data test_data.bin \
                           --test-binary \
                           --benchmark
```

### Command Line Options

- `--model PATH`: Path to model file (.tflite for NN, .json/.pkl for XGBoost/RF) (required)
- `--metadata PATH`: Path to model metadata JSON (required)
- `--model-type TYPE`: Model type: nn, xgboost, or rf (default: nn)
- `--test-data PATH`: Path to test data file (CSV or binary)
- `--test-binary`: Indicate that test data is in binary format
- `--load-gt`: Load ground truth labels from CSV for evaluation
- `--samples N`: Number of samples in binary file (default: 10)
- `--seq-length N`: Sequence length for binary data (default: 20)
- `--features N`: Number of features per time step (default: 18)
- `--threads N`: Number of threads for inference (default: 4)
- `--benchmark`: Run in benchmark mode
- `--help`: Show help message

## Model Requirements

### Supported Models

All models must predict 11 binary outputs corresponding to:
- **Direction**: incoming, outgoing
- **Vertical motion**: fixed_range_ascending, fixed_range_descending, level_flight
- **Path shape**: linear, curved
- **Maneuver intensity**: light_maneuver, high_maneuver
- **Speed**: low_speed, high_speed

1. **Neural Network Multi-Output Models (TFLite)**
   - ✅ **Fully Implemented**
   - Format: `.tflite` (TensorFlow Lite)
   - Converted from Keras/TensorFlow models
   - Best for: Real-time inference with low latency

2. **XGBoost Models (ONNX)**
   - ✅ **Fully Implemented**
   - Format: `.onnx` (ONNX Runtime)
   - Converted from trained XGBoost models
   - Best for: High accuracy with tabular data

3. **Random Forest Models (ONNX)**
   - ✅ **Fully Implemented**
   - Format: `.onnx` (ONNX Runtime)
   - Converted from scikit-learn Random Forest models
   - Best for: Robust predictions with feature importance

### Model Export Instructions

#### Exporting XGBoost Models

```bash
# Export trained XGBoost model to ONNX
python3 export_models_to_onnx.py \
    --model-type xgboost \
    --model-path output/xgboost_multioutput.pkl \
    --output-path cpp_models/xgboost_model.onnx \
    --metadata-path output/xgboost_metadata.json \
    --output-metadata cpp_models/xgboost_metadata.json
```

**Requirements:**
```bash
pip install xgboost skl2onnx onnx onnxmltools
```

#### Exporting Random Forest Models

```bash
# Export trained Random Forest model to ONNX
python3 export_models_to_onnx.py \
    --model-type random_forest \
    --model-path output/random_forest_multioutput.pkl \
    --output-path cpp_models/rf_model.onnx \
    --metadata-path output/rf_metadata.json \
    --output-metadata cpp_models/rf_metadata.json
```

**Requirements:**
```bash
pip install scikit-learn skl2onnx onnx
```

#### Exporting Neural Network Models

You can export Keras models to ONNX or TFLite format:

**Option 1: TFLite (Recommended for best performance)**
```bash
# Export multi-output neural network to TFLite
python3 convert_model_to_tflite.py \
    --model-path output/nn_multioutput.h5 \
    --output-dir cpp_models/nn \
    --metadata-path output/nn_metadata.json
```

**Option 2: ONNX (For cross-platform compatibility)**
```bash
# Export Keras model to ONNX
python3 export_models_to_onnx.py \
    --model-type neural_network \
    --model-path output/nn_multioutput.h5 \
    --output-path cpp_models/nn_model.onnx \
    --metadata-path output/nn_metadata.json \
    --output-metadata cpp_models/nn_metadata.json
```

**Requirements:**
```bash
# For TFLite
pip install tensorflow

# For ONNX
pip install tensorflow tf2onnx onnx
```

**Note:** For multi-output Neural Networks, use simple feed-forward architectures for best C++ deployment performance. TFLite is recommended over ONNX for better inference speed.

### Model Structure Requirements

#### For XGBoost and Random Forest (Multi-Output)

Your Python training code should create models that output 11 binary predictions:

```python
# Example: Training multi-output XGBoost
import xgboost as xgb
from sklearn.multioutput import MultiOutputClassifier

# Create multi-output wrapper
base_model = xgb.XGBClassifier(n_estimators=100, max_depth=6)
model = MultiOutputClassifier(base_model)

# Train on multi-output labels (shape: [n_samples, 11])
model.fit(X_train, y_train_multioutput)

# Save model
import pickle
with open('xgboost_multioutput.pkl', 'wb') as f:
    pickle.dump(model, f)
```

#### For Neural Networks

```python
# Example: Keras multi-output model
import tensorflow as tf

model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(18,)),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(11, activation='sigmoid')  # 11 binary outputs
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.fit(X_train, y_train_multioutput, epochs=50)
model.save('nn_model.h5')
```

**Note:** For best real-time performance in C++ deployment, use:
- **Random Forest or XGBoost**: Best for tabular data, fast inference
- **Feed-forward Neural Networks**: Good balance of accuracy and speed
- Avoid complex architectures (LSTM, Transformer) as they add deployment complexity

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

### Basic Multi-Output Integration

```cpp
#include "radar_tagger_multioutput.h"

int main() {
    // Create tagger
    RadarTaggerMultiOutput tagger("model.tflite", "metadata.json", 
                                   ModelType::NEURAL_NETWORK, 4);
    
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
        std::cout << "Predicted tags: " << result.aggregatedLabel << "\n";
        
        // Access individual tags
        if (result.tags.incoming) {
            std::cout << "Direction: incoming\n";
        }
        if (result.tags.level_flight) {
            std::cout << "Altitude: level flight\n";
        }
        // ... check other tags ...
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
        
        if (result.success) {
            // Process multi-output tags
            std::cout << "Track " << currentSequence.trackId 
                      << ": " << result.aggregatedLabel << "\n";
            
            // Access individual tag confidences
            for (const auto& [tag, confidence] : result.tags.confidences) {
                if (confidence > 0.5f) {
                    std::cout << "  " << tag << ": " << confidence << "\n";
                }
            }
        }
        
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
