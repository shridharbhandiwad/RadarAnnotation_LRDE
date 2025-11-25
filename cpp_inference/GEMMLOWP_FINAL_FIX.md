# Final Fix for gemmlowp eight_bit_int_gemm MinGW Build Error

## Problem Description

**Error:**
```
c++.exe: fatal error: cannot specify '-o' with '-c', '-S' or '-E' with multiple files
compilation terminated.
mingw32-make[2]: *** [_deps\gemmlowp-build\CMakeFiles\eight_bit_int_gemm.dir\build.make:78: ...] Error 1
```

**Root Cause:**
The `gemmlowp` library (a TensorFlow Lite dependency) has a CMake configuration that generates incorrect compiler commands on MinGW/Windows. Specifically, the `eight_bit_int_gemm` target's build rules are malformed, causing the compiler to receive multiple source files with a single output specification.

This is a known issue with gemmlowp on Windows MinGW builds, and the target is not actually needed for TensorFlow Lite inference functionality.

## Why This Keeps Happening

The error persists because:
1. **Timing**: Patches must be applied AFTER gemmlowp is downloaded but BEFORE CMake processes its CMakeLists.txt
2. **CMake Caching**: Once CMake generates build files, they are cached and reused even if the source is patched
3. **Multiple Configuration Passes**: TensorFlow Lite uses FetchContent which downloads and configures dependencies in a specific order

## Comprehensive Solution

This fix applies **multiple layers of protection** to ensure the problematic target is never built:

### Layer 1: Pre-Configuration Patching (CMakeLists.txt)
- Searches for gemmlowp CMakeLists.txt after FetchContent downloads it
- Comments out all `eight_bit_int_gemm` target definitions before CMake processes them

### Layer 2: Post-Configuration Exclusion (CMakeLists.txt)
- Uses `EXCLUDE_FROM_ALL` property to prevent the target from being built
- Uses `cmake_language(DEFER ...)` to apply the fix after all targets are defined

### Layer 3: Direct Python Patching (patch_gemmlowp_direct.py)
- Standalone Python script that directly modifies gemmlowp CMakeLists.txt files
- Can be run independently or as part of the build process
- Handles all possible gemmlowp target commands

### Layer 4: Build Script Orchestration (build_with_gemmlowp_fix.bat/sh)
- Runs CMake configure
- Patches gemmlowp if downloaded
- Re-runs CMake to pick up patches
- Builds with single-threaded compilation for stability

## How to Use

### Method 1: Clean Build (Recommended for Windows/MinGW)

```batch
cd cpp_inference
build_with_gemmlowp_fix.bat clean
```

This will:
1. Delete the build directory
2. Run CMake configure
3. Patch gemmlowp
4. Reconfigure CMake
5. Build the project

### Method 2: Manual Patching

If you already have a partially-built project:

```batch
cd cpp_inference
python patch_gemmlowp_direct.py
cd build
cmake ..
cmake --build . --config Release
```

### Method 3: Linux/Unix

```bash
cd cpp_inference
./build_with_gemmlowp_fix.sh clean
```

## What Gets Patched

The following commands in gemmlowp's CMakeLists.txt are commented out:

```cmake
# DISABLED_FOR_MINGW: add_library(eight_bit_int_gemm ...)
# DISABLED_FOR_MINGW: add_executable(eight_bit_int_gemm ...)
# DISABLED_FOR_MINGW: target_link_libraries(eight_bit_int_gemm ...)
# DISABLED_FOR_MINGW: set_target_properties(eight_bit_int_gemm ...)
# DISABLED_FOR_MINGW: target_compile_options(eight_bit_int_gemm ...)
# DISABLED_FOR_MINGW: target_include_directories(eight_bit_int_gemm ...)
```

## Impact on Functionality

**No impact on TensorFlow Lite functionality:**
- The `eight_bit_int_gemm` target is only a test/benchmark tool
- TensorFlow Lite's internal matrix multiplication still works perfectly
- All inference functionality is preserved
- The main executables (`radar_tagger.exe` and `radar_tagger_multioutput.exe`) will build successfully

## Verification

After building, verify success:

```batch
cd build
dir radar_tagger.exe
dir radar_tagger_multioutput.exe
```

Both executables should exist and be runnable.

## If It Still Fails

If you still see the error after using these fixes:

1. **Completely delete the build directory:**
   ```batch
   cd cpp_inference
   rmdir /s /q build
   ```

2. **Delete CMake cache files:**
   ```batch
   del CMakeCache.txt
   del cmake_install.cmake
   ```

3. **Run the build script with clean:**
   ```batch
   build_with_gemmlowp_fix.bat clean
   ```

4. **Check for stale CMake processes:**
   - Kill any running cmake.exe or mingw32-make.exe processes in Task Manager

## Technical Details

### Why the Error Occurs

The MinGW compiler receives a command like:
```bash
c++.exe -c file1.cc file2.cc -o output.o
```

This is invalid because `-c` (compile only) with `-o` (output file) can only work with a single input file. The error occurs because gemmlowp's CMake configuration doesn't properly handle MinGW's build system generation.

### Why Patching Works

By commenting out the target definition, CMake never generates the problematic build rules. The target simply doesn't exist in the build system, so the malformed compiler commands are never generated.

### Why Multiple Layers Are Needed

1. **Pre-patching** catches the issue at configuration time
2. **Post-exclusion** handles cases where the target still gets created
3. **Python patching** provides a manual override option
4. **Build script** ensures proper ordering and re-configuration

## Files Modified

- `cpp_inference/CMakeLists.txt` - Added pre and post patching logic
- `cpp_inference/patch_gemmlowp_direct.py` - New Python patching script
- `cpp_inference/build_with_gemmlowp_fix.bat` - New Windows build script
- `cpp_inference/build_with_gemmlowp_fix.sh` - New Linux/Unix build script

## Date Applied

2025-11-25

## References

- TensorFlow Issue #42795: gemmlowp build failures on Windows
- CMake Policy CMP0169: FetchContent deprecation handling
- gemmlowp Issue #179: MinGW compilation issues
