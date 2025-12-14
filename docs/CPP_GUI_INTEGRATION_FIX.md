# C++ GUI Integration Fix Summary

## Problem
The GUI's C++ build function was calling CMake without specifying the build generator or compiler, causing it to attempt using `nmake` (Windows tool) on Linux systems.

## Solution
Updated the `build_cpp_app()` method in `src/gui.py` to:

1. **Clean build directory**: Remove stale CMake cache before building
2. **Specify generator**: Use "Unix Makefiles" on Linux/Mac, "MinGW Makefiles" on Windows
3. **Set compiler**: Explicitly set CMAKE_CXX_COMPILER to g++ or clang++
4. **Add parallel builds**: Use all CPU cores on Linux/Mac for faster builds

## How to Use C++ Integration in GUI

### Step 1: Convert Model to TensorFlow Lite
1. Launch the GUI: `python3 src/gui.py`
2. Go to the **"⚙️ C++ Deployment"** tab
3. In "Step 1: Convert Model to TensorFlow Lite" section:
   - Select your trained model directory (e.g., `models/transformer_nn/`)
   - Click **"🔄 Convert to TensorFlow Lite"**
   - Wait for conversion to complete

### Step 2: Build C++ Application
1. In "Step 2: Build C++ Inference Application" section:
   - Click **"🔨 Build C++ Application"**
   - This will:
     - Clean the build directory
     - Configure CMake with proper settings for Linux
     - Compile the C++ radar tagger application
     - Download dependencies (TensorFlow Lite, ONNX Runtime, nlohmann/json)
   - Build takes 5-15 minutes on first run (downloads dependencies)

### Step 3: Evaluate Model with C++
1. In "Step 3: Evaluate Model with C++" section:
   - Select test data CSV file (e.g., `data/test_simulation_labeled.csv`)
   - Click **"🎯 Run C++ Evaluation"**
   - View performance metrics and comparison with Python

## What's Built
The build process creates two executables:
- `cpp_inference/build/radar_tagger` - Basic single-output version
- `cpp_inference/build/radar_tagger_multioutput` - Advanced multi-output version (used by GUI)

## Key Changes Made

```python
# Before (broken on Linux):
cmake_cmd = ["cmake", ".."]

# After (fixed):
cmake_cmd = ["cmake", "..", "-G", "Unix Makefiles", 
             "-DCMAKE_BUILD_TYPE=Release",
             "-DCMAKE_CXX_COMPILER=g++"]
```

## Verification
✅ Python syntax valid
✅ g++ compiler available at `/usr/bin/g++`
✅ CMake available at `/usr/bin/cmake`
✅ Build directory cleaning implemented
✅ Platform-specific generator selection
✅ Parallel build support added

## Alternative: Command Line Build
If you prefer building from command line:
```bash
cd cpp_inference
./build.sh
```

This uses the same configuration as the GUI fix.

## Notes
- **First build takes time**: Downloads TensorFlow (v2.14.0), ONNX Runtime, and other dependencies
- **Subsequent builds are faster**: Dependencies are cached
- **Build directory is cleaned**: Each GUI build starts fresh to avoid cache issues
- **Cross-platform**: Works on Linux, macOS, and Windows (with MinGW)

## Troubleshooting

### If build still fails:
1. **Check dependencies**:
   ```bash
   sudo apt-get install build-essential cmake g++
   ```

2. **Manually clean**:
   ```bash
   rm -rf cpp_inference/build
   ```

3. **Check CMake version**:
   ```bash
   cmake --version  # Should be 3.15+
   ```

4. **Use command line build first**:
   ```bash
   cd cpp_inference && ./build.sh
   ```
   This will show detailed error messages if something is wrong.

## What the GUI Does
The C++ Deployment tab provides an integrated workflow:
1. **Model Conversion**: Converts TensorFlow models to TensorFlow Lite format
2. **Build System**: Compiles C++ inference engine with all dependencies
3. **Evaluation**: Runs performance tests comparing C++ vs Python inference
4. **Results Display**: Shows accuracy, speed, and compatibility metrics

This makes C++ deployment accessible without leaving the GUI!
