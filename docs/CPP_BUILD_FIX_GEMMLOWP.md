# Gemmlowp Build Fix for MinGW

## Problem
The build was failing with the error:
```
c++.exe: fatal error: cannot specify '-o' with '-c', '-S' or '-E' with multiple files
```

This occurred when trying to compile gemmlowp's `eight_bit_int_gemm` test executable, which has compatibility issues with MinGW.

## Solution
Modified `CMakeLists.txt` to:
1. Download TensorFlow Lite source without configuring it immediately
2. Patch the `gemmlowp.cmake` file to disable the problematic `eight_bit_int_gemm` executable
3. Then proceed with configuration and build

The key changes:
- Use `FetchContent_Populate()` instead of `FetchContent_MakeAvailable()` for more control
- Patch the gemmlowp configuration file before `add_subdirectory()` is called
- Comment out the `add_executable(eight_bit_int_gemm` line that causes issues

## How to Rebuild

### On Windows:
```batch
# Navigate to cpp_inference directory
cd cpp_inference

# Run the clean rebuild script
rebuild_clean.bat
```

### Manual rebuild:
```batch
cd cpp_inference
rmdir /s /q build
mkdir build
cd build
cmake .. -G "MinGW Makefiles"
mingw32-make -j4
```

## What Changed
The `eight_bit_int_gemm` is just a test/benchmark executable for gemmlowp and is not needed for TensorFlow Lite to function. By disabling it, we avoid the MinGW compilation issue while keeping all the functionality needed for the radar tagger application.

## Testing
After rebuilding, test both executables:
```batch
radar_tagger.exe --help
radar_tagger_multioutput.exe --help
```

Both should run without errors and display usage information.
