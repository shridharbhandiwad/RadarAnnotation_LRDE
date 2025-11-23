# MinGW gemmlowp Build Fix

## Problem
When building TensorFlow Lite with MinGW on Windows, the build fails with:
```
c++.exe: fatal error: cannot specify '-o' with '-c', '-S' or '-E' with multiple files
mingw32-make[2]: *** [.../gemmlowp-build/CMakeFiles/eight_bit_int_gemm.dir/build.make:78: ...] Error 1
```

## Root Cause
The `gemmlowp` library (a TensorFlow Lite dependency) has a CMake configuration issue with MinGW that causes incorrect compiler command generation. Specifically, the `eight_bit_int_gemm` target receives malformed build commands.

## Solution Applied

Modified `CMakeLists.txt` with the following fixes:

### 1. Added MinGW-Specific Build Flags
```cmake
if(MINGW OR WIN32)
    # Use larger object files for MinGW
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wa,-mbig-obj")
    # Disable building tests and examples
    set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
    # Use single-threaded builds for problematic targets
    set(CMAKE_BUILD_PARALLEL_LEVEL 1)
    # Disable problematic SIMD optimizations
    set(GEMMLOWP_ENABLE_NEON OFF CACHE BOOL "" FORCE)
endif()
```

### 2. Excluded Problematic Target
```cmake
# Post-fetch fix: Disable eight_bit_int_gemm executable for MinGW
if((MINGW OR WIN32) AND TARGET eight_bit_int_gemm)
    set_target_properties(eight_bit_int_gemm PROPERTIES EXCLUDE_FROM_ALL TRUE)
endif()
```

## How to Use

### Option 1: Clean rebuild (Recommended)
```batch
cd cpp_inference
rmdir /s /q build
mkdir build
cd build
cmake -G "MinGW Makefiles" -DCMAKE_BUILD_TYPE=Release ..
cmake --build . --config Release
```

### Option 2: Use the build script
```batch
cd cpp_inference
build_mingw_alternative.bat
```

## What Changed
- **Big Object Support**: Added `-Wa,-mbig-obj` flag to handle large object files on MinGW
- **Disabled Tests**: Set `BUILD_TESTING OFF` to skip building test executables that aren't needed
- **Single-threaded Build**: Set `CMAKE_BUILD_PARALLEL_LEVEL 1` for problematic targets
- **Disabled SIMD**: Turned off NEON SIMD optimizations that cause issues on Windows
- **Excluded Target**: The `eight_bit_int_gemm` executable is excluded from the build (it's not needed for TensorFlow Lite functionality)

## Impact
- The TensorFlow Lite library will still be fully functional
- The excluded `eight_bit_int_gemm` is only a test/benchmark tool, not required for inference
- Main executables `radar_tagger.exe` and `radar_tagger_multioutput.exe` will build successfully

## Additional Notes
- This fix addresses MinGW-specific compilation issues
- Other warnings (pragma comments, format strings) are harmless and can be ignored
- Build time may be slightly longer due to single-threaded compilation of some targets

## Date Applied
2025-11-23
