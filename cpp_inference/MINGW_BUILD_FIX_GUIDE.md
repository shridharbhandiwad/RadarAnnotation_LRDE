# MinGW Build Fix Guide

This guide addresses two critical build errors when compiling the Radar Tagger C++ project with MinGW on Windows.

## Issues Fixed

### 1. gemmlowp Compilation Error

**Error:**
```
c++.exe: fatal error: cannot specify '-o' with '-c', '-S' or '-E' with multiple files
compilation terminated.
```

**Cause:** The `eight_bit_int_gemm` target in gemmlowp has issues with MinGW's compiler argument handling.

**Solution:** Disable the `eight_bit_int_gemm` target by commenting it out in gemmlowp's CMakeLists.txt.

### 2. cpuinfo Missing max() Function

**Error:**
```
error: implicit declaration of function 'max' [-Wimplicit-function-declaration]
```

**Cause:** Windows/MinGW doesn't provide the `max()` and `min()` macros that cpuinfo expects.

**Solution:** 
- Add `max()` and `min()` macro definitions via compiler flags
- Alternatively, patch the cpuinfo source file directly

## Fix Methods

### Method 1: Automated Python Script (Recommended)

The easiest way to fix these issues:

```bash
# Step 1: Run CMake configuration
mkdir build
cd build
cmake -G "MinGW Makefiles" -DCMAKE_BUILD_TYPE=Release ..

# Step 2: Run the fix script
cd ..
python fix_build_dependencies.py

# Step 3: Build
cd build
mingw32-make -j4
```

The `fix_build_dependencies.py` script automatically:
- Locates gemmlowp and patches its CMakeLists.txt
- Locates cpuinfo and adds missing max/min macros
- Can be run multiple times safely (checks if already patched)

### Method 2: All-in-One Build Script

Use the provided batch script that handles everything:

```bash
build_with_fixes.bat
```

This script:
1. Configures the project with CMake
2. Applies all necessary patches
3. Builds the project
4. Retries with fixes if build fails

For a clean build:
```bash
build_with_fixes.bat clean
```

### Method 3: Manual CMake Flags (Partial Fix)

The CMakeLists.txt includes these fixes for cpuinfo:

```cmake
if(MINGW OR WIN32)
    # Add min/max macros for cpuinfo compatibility
    add_compile_definitions(NOMINMAX)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Dmax(a,b)=((a)>(b)?(a):(b)) -Dmin(a,b)=((a)<(b)?(a):(b))")
endif()
```

However, gemmlowp still needs manual patching after configuration.

## Build Process

### Complete Build Steps

1. **Configure:**
   ```bash
   mkdir build
   cd build
   cmake -G "MinGW Makefiles" -DCMAKE_BUILD_TYPE=Release ..
   ```

2. **Apply Fixes:**
   ```bash
   cd ..
   python fix_build_dependencies.py
   ```

3. **Build:**
   ```bash
   cd build
   mingw32-make -j4
   ```

4. **If Build Fails:**
   - Run `python fix_build_dependencies.py` again
   - Some dependencies might download during the first build attempt
   - Retry: `mingw32-make -j4`

### Output

Successful build produces:
- `build/radar_tagger.exe` - Single-output model inference
- `build/radar_tagger_multioutput.exe` - Multi-output model inference

## Technical Details

### gemmlowp Patch

The patch comments out the problematic target:

```cmake
# Original:
add_library(eight_bit_int_gemm ...)

# Patched:
# DISABLED_FOR_MINGW: add_library(eight_bit_int_gemm ...)
```

This target is not required for TensorFlow Lite inference operations.

### cpuinfo Patch

Adds standard max/min macros:

```c
#ifndef max
#define max(a, b) (((a) > (b)) ? (a) : (b))
#endif
#ifndef min
#define min(a, b) (((a) < (b)) ? (a) : (b))
#endif
```

These are standard C macros that cpuinfo expects to be available.

## Troubleshooting

### "gemmlowp not found"

This is normal if dependencies haven't been downloaded yet:
1. Run CMake configuration completely
2. Wait for TensorFlow Lite download to finish
3. Run the fix script again

### "Build still failing after patch"

Try these steps:
1. Clean build directory: `rm -rf build`
2. Reconfigure: `cmake -G "MinGW Makefiles" ..`
3. Apply patches: `python fix_build_dependencies.py`
4. Build: `mingw32-make -j4`

### Check if Patches Applied

```bash
# Check gemmlowp
grep "DISABLED_FOR_MINGW" build/_deps/gemmlowp-src/CMakeLists.txt

# Check cpuinfo
grep "PATCHED_FOR_MINGW" build/_deps/cpuinfo-src/src/x86/windows/init.c
```

## Alternative: Use Pre-built ONNX Runtime Only

If TensorFlow Lite build continues to fail, you can build with ONNX Runtime only:

```cmake
# In CMakeLists.txt, comment out TensorFlow Lite
# option(USE_SYSTEM_TFLITE "Use system-installed TensorFlow Lite" ON)
```

Note: This limits you to ONNX models (XGBoost, Random Forest) only.

## Updates to CMakeLists.txt

The main CMakeLists.txt includes these MinGW-specific fixes:

1. **Large object files:** `-Wa,-mbig-obj`
2. **Single-threaded build:** Prevents parallel build issues
3. **Disabled SIMD:** `GEMMLOWP_ENABLE_NEON OFF`
4. **max/min macros:** Added via compile flags
5. **NOMINMAX:** Prevents Windows.h conflicts

## Support

If issues persist:
1. Check that you're using a recent MinGW-w64 installation
2. Ensure Python 3.6+ is available for the fix script
3. Try building with `-j1` (single-threaded) instead of `-j4`
4. Check the detailed error messages for specific file paths

## Files

- `fix_build_dependencies.py` - Python script to patch dependencies
- `build_with_fixes.bat` - Windows batch script for complete build
- `fix_dependencies.bat` - Windows batch script to apply patches only
- `CMakeLists.txt` - Includes compile flag fixes for cpuinfo

