# gemmlowp Build Fix Applied

## Date
2025-11-23

## Problem
The build was failing with the error:
```
c++.exe: fatal error: cannot specify '-o' with '-c', '-S' or '-E' with multiple files
compilation terminated.
```

This occurred when building the `eight_bit_int_gemm` target from the gemmlowp library (a TensorFlow Lite dependency).

## Root Cause
The gemmlowp library's CMake configuration has issues with MinGW that cause incorrect compiler command generation. The `eight_bit_int_gemm` target is a test/benchmark tool that isn't needed for TensorFlow Lite inference functionality.

## Solution Applied
Modified `/workspace/cpp_inference/CMakeLists.txt` to:

1. **Inject patching code into TensorFlow Lite's CMakeLists.txt**: The patch is now injected directly into TensorFlow Lite's build configuration, so it will automatically patch gemmlowp when TensorFlow Lite fetches it.

2. **Disable the problematic target**: The `eight_bit_int_gemm` target is commented out in gemmlowp's CMakeLists.txt before it gets configured.

3. **Backup exclusion**: As a safety net, if the target somehow still exists after configuration, it's excluded from the build.

## What Changed in CMakeLists.txt

### Added automatic gemmlowp patching (lines 172-215)
- Detects when TensorFlow Lite CMakeLists.txt needs patching
- Injects code that will patch gemmlowp after TensorFlow Lite fetches it
- The injected code comments out all references to `eight_bit_int_gemm`

### Streamlined post-configuration handling (lines 223-231)
- Removed redundant manual patching code
- Kept only a backup exclusion in case the target still exists

## How to Use This Fix

### Step 1: Clean the build directory
Since TensorFlow Lite has already been partially fetched, you need to clean the build directory for the patches to take effect:

```bash
cd /workspace/cpp_inference
rm -rf build
mkdir build
cd build
```

### Step 2: Reconfigure with CMake
```bash
cmake -G "MinGW Makefiles" -DCMAKE_BUILD_TYPE=Release ..
```

You should see messages like:
- "Patching TensorFlow Lite CMakeLists.txt for MinGW compatibility (max/min macros)..."
- "Injecting gemmlowp patching code into TensorFlow Lite CMakeLists.txt..."
- "Patching gemmlowp to disable eight_bit_int_gemm..."

### Step 3: Build
```bash
cmake --build . --config Release
```

Or use the convenience script:
```bash
cd /workspace/cpp_inference
./build.sh
```

## Expected Outcome
✅ Build completes successfully  
✅ `radar_tagger` and `radar_tagger_multioutput` executables are created  
✅ TensorFlow Lite inference functionality is fully preserved  
✅ Only the unused test target `eight_bit_int_gemm` is disabled  

## Impact
- **No functional impact**: The `eight_bit_int_gemm` target is only a test/benchmark tool, not required for inference
- **All TensorFlow Lite features work normally**: The library and all inference capabilities are unaffected
- **Windows/MinGW specific**: This fix only applies to Windows MinGW builds

## Verification
After building, verify the executables exist:
```bash
ls -la /workspace/cpp_inference/build/radar_tagger*
```

## Notes
- This fix is permanent and automatic - it will apply every time you clean and rebuild
- The patches are idempotent - running CMake multiple times won't cause issues
- If you update TensorFlow Lite to a different version, the patches will still apply
