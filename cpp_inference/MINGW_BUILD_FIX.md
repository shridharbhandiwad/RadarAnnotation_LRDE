# MinGW Build Fix for TensorFlow Lite

## Problem
The CMake build was failing with the error:
```
The filename, directory name, or volume label syntax is incorrect.
```

This was happening during the C compiler test when TensorFlow Lite's CMakeLists.txt called `project()`.

## Root Cause
The issue was caused by problematic compiler flag definitions:
```
-Dmax(a,b)=((a)>(b)?(a):(b)) -Dmin(a,b)=((a)<(b)?(a):(b))
```

These flags contain `>` and `<` characters which Windows interprets as shell redirection operators, causing the build to fail.

## Solution
The fix involves three key changes:

### 1. Removed Global Compiler Flags
Previously, `CMAKE_C_FLAGS` and `CMAKE_CXX_FLAGS` were set globally before TensorFlow Lite was configured. This could interfere with TensorFlow Lite's build process. These have been removed from the global scope.

### 2. Added TensorFlow Lite CMakeLists.txt Patching
Before TensorFlow Lite is added to the build, we now patch its CMakeLists.txt file to remove any problematic max/min macro definitions that contain `>` and `<` characters. The patch removes these definitions from:
- `CMAKE_C_FLAGS` and `CMAKE_CXX_FLAGS` set() calls
- `add_compile_definitions()` calls
- `target_compile_definitions()` calls
- Standalone `-D` definitions

### 3. Applied Flags to Specific Targets Only
The `-Wa,-mbig-obj` flag (needed for MinGW large object files) is now applied only to our specific targets (`radar_tagger` and `radar_tagger_multioutput`) using `target_compile_options()`, preventing it from affecting TensorFlow Lite's build.

## Usage
Simply delete the `build` directory and run CMake again:

```bash
# Windows (MinGW)
cd cpp_inference
rmdir /s /q build
mkdir build
cd build
cmake -G "MinGW Makefiles" -DCMAKE_BUILD_TYPE=Release ..
```

## Result
The build should now proceed without the "filename, directory name, or volume label syntax" error, and TensorFlow Lite should compile successfully on MinGW/Windows.
