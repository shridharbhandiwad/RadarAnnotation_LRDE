# Linux Build Configuration Fix

## Issue

The CMake configuration was failing with the error:
```
/usr/bin/ld: cannot find -lstdc++: No such file or directory
c++: error: linker command failed with exit code 1
```

Additionally, the configuration was showing Windows paths for ONNX Runtime:
```
ONNX Runtime: D:/Zoppler Projects/RadarAnnotation_LRDE/cpp_inference/build/_deps/onnxruntime-src/lib/onnxruntime.lib
```

## Root Cause

The system had Clang 18.1.3 installed, which was trying to use GCC 14's toolchain, but the required C++ standard library development files for GCC 14 (`libstdc++-14-dev`) were not installed. This prevented the compiler from linking C++ programs.

## Solution

Installed the missing C++ standard library development package:

```bash
sudo apt-get update
sudo apt-get install -y libstdc++-14-dev
```

After installation, the build configuration and compilation completed successfully.

## Verification

1. **CMake Configuration**: Successfully completed without errors
   ```
   -- Radar Tagger C++ Configuration:
   --   Version: 1.0.0
   --   C++ Standard: 17
   --   Build Type: 
   --   TensorFlow Lite: tensorflow-lite
   --   ONNX Runtime: /workspace/cpp_inference/build/_deps/onnxruntime-src/lib/libonnxruntime.so
   -- 
   -- Configuring done (95.5s)
   -- Generating done (0.2s)
   ```

2. **Build**: Successfully compiled both executables
   ```
   -rwxr-xr-x 1 ubuntu ubuntu 4.2M Nov 25 12:01 radar_tagger
   -rwxr-xr-x 1 ubuntu ubuntu 4.4M Nov 25 12:01 radar_tagger_multioutput
   ```

## Key Points

- The Windows paths issue was automatically resolved by cleaning the build directory (`rm -rf build && mkdir build`)
- The CMake FetchContent system correctly downloaded the Linux version of ONNX Runtime
- The build now uses correct Linux shared library paths (`.so` instead of `.lib/.dll`)

## Next Steps

To build the project from scratch on Linux:

```bash
cd /workspace/cpp_inference
rm -rf build
mkdir build
cd build
cmake ..
cmake --build . -j$(nproc)
```

The executables will be available in the `build/` directory:
- `radar_tagger` - Single output model inference
- `radar_tagger_multioutput` - Multi-output model inference with ONNX Runtime support
