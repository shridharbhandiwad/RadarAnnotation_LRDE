# Fix for eight_bit_int_gemm Compilation Error on Windows/MinGW

## Problem

When building the project with MinGW on Windows, you may encounter this error:

```
c++.exe: fatal error: cannot specify '-o' with '-c', '-S' or '-E' with multiple files
compilation terminated.
mingw32-make[2]: *** [.../eight_bit_int_gemm.cc.obj] Error 1
```

## Root Cause

The `eight_bit_int_gemm` target in Google's gemmlowp library has CMake configuration issues on MinGW that generate incorrect compiler commands. This target is only a test utility and is **not required** for TensorFlow Lite functionality.

## Solution

The fix has been integrated into the CMakeLists.txt file and will be applied automatically. However, if you encounter this error, follow these steps:

### Option 1: Automated Fix (Recommended)

Run the provided Windows batch script:

```batch
cd cpp_inference
fix_and_build_windows.bat clean
```

This script will:
1. Clean the build directory
2. Configure CMake
3. Detect and patch gemmlowp automatically
4. Build the project

### Option 2: Manual Fix

If the automated script doesn't work, follow these manual steps:

1. **Clean the build directory:**
   ```batch
   cd cpp_inference
   rmdir /s /q build
   mkdir build
   cd build
   ```

2. **Run initial CMake configuration:**
   ```batch
   cmake -G "MinGW Makefiles" -DCMAKE_BUILD_TYPE=Release ..
   ```

3. **Wait for gemmlowp to download**, then check if it exists:
   ```batch
   dir gemmlowp\CMakeLists.txt
   ```

4. **If the file exists, apply the patch:**
   ```batch
   copy gemmlowp\CMakeLists.txt gemmlowp\CMakeLists.txt.bak
   ```

   Then edit `gemmlowp\CMakeLists.txt` and comment out these lines:
   - Any line with `add_library(eight_bit_int_gemm`
   - Any line with `add_executable(eight_bit_int_gemm`
   - Any line with `target_link_libraries(eight_bit_int_gemm`
   - Any line with `set_target_properties(eight_bit_int_gemm`

   You can do this automatically with PowerShell:
   ```powershell
   $content = Get-Content 'gemmlowp\CMakeLists.txt' -Raw
   $content = $content -replace 'add_library\(eight_bit_int_gemm', '# DISABLED: add_library(eight_bit_int_gemm'
   $content = $content -replace 'add_executable\(eight_bit_int_gemm', '# DISABLED: add_executable(eight_bit_int_gemm'
   $content = $content -replace 'target_link_libraries\(eight_bit_int_gemm', '# DISABLED: target_link_libraries(eight_bit_int_gemm'
   $content = $content -replace 'set_target_properties\(eight_bit_int_gemm', '# DISABLED: set_target_properties(eight_bit_int_gemm'
   Set-Content 'gemmlowp\CMakeLists.txt' $content
   ```

5. **Re-run CMake to apply the changes:**
   ```batch
   cmake -G "MinGW Makefiles" -DCMAKE_BUILD_TYPE=Release ..
   ```

6. **Build the project:**
   ```batch
   cmake --build . --config Release
   ```

## How the Fix Works

The CMakeLists.txt now includes automatic patching logic:

1. **During CMake configuration**, after TensorFlow Lite is downloaded, the script:
   - Checks if `gemmlowp/CMakeLists.txt` exists
   - Reads the file content
   - Uses regex to comment out all `eight_bit_int_gemm` target definitions
   - Writes the patched file back

2. **As a backup**, there's a post-configuration check that sets:
   ```cmake
   set_target_properties(eight_bit_int_gemm PROPERTIES 
       EXCLUDE_FROM_ALL TRUE
       EXCLUDE_FROM_DEFAULT_BUILD TRUE
   )
   ```

## Verification

After the build completes successfully, you should see:

```
[SUCCESS] radar_tagger.exe built successfully
[SUCCESS] radar_tagger_multioutput.exe built successfully
```

## Impact on Functionality

**None**. The `eight_bit_int_gemm` target is only used for:
- Benchmark testing
- Performance profiling  
- Development/debugging

It is **not used** by TensorFlow Lite runtime. All inference functionality remains intact.

## Troubleshooting

### If the error persists:

1. **Ensure you're using MinGW-w64** (not the older MinGW):
   ```batch
   g++ --version
   ```
   Should show "x86_64-posix-seh" or similar

2. **Check CMake version** (3.16 or newer required):
   ```batch
   cmake --version
   ```

3. **Try building with verbose output** to see the exact error:
   ```batch
   cmake --build . --config Release -- VERBOSE=1
   ```

4. **Clean everything and rebuild:**
   ```batch
   cd cpp_inference
   rmdir /s /q build
   fix_and_build_windows.bat clean
   ```

### If gemmlowp directory doesn't exist:

The gemmlowp library might not have been downloaded yet. This can happen if:
- The TensorFlow download was interrupted
- Network issues prevented the download
- CMake cache is corrupted

Solution:
```batch
cd cpp_inference
rmdir /s /q build
mkdir build
cd build
cmake -G "MinGW Makefiles" -DCMAKE_BUILD_TYPE=Release ..
```

Wait for the download to complete, then check again for the gemmlowp directory.

## Alternative: Use Pre-built Libraries

If the build continues to fail, consider using pre-built TensorFlow Lite libraries:

1. Download pre-built TensorFlow Lite for Windows
2. Set the CMake option:
   ```batch
   cmake -G "MinGW Makefiles" -DUSE_SYSTEM_TFLITE=ON ..
   ```

## References

- [TensorFlow Lite C++ Guide](https://www.tensorflow.org/lite/guide/build_cmake)
- [gemmlowp Repository](https://github.com/google/gemmlowp)
- [MinGW-w64 Downloads](https://sourceforge.net/projects/mingw-w64/)

## Support

If you continue to experience issues:
1. Check the build log for specific error messages
2. Verify all prerequisites are installed
3. Try the manual fix approach step-by-step
4. Open an issue with the complete error log
