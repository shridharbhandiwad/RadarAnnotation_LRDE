# eight_bit_int_gemm Compilation Error - Solution Summary

## Quick Fix (Windows/MinGW Users)

If you're seeing this error:
```
c++.exe: fatal error: cannot specify '-o' with '-c', '-S' or '-E' with multiple files
```

**Run this command:**
```batch
cd cpp_inference
fix_and_build_windows.bat clean
```

That's it! The script will automatically fix the issue and build the project.

---

## What Was Fixed

### Files Modified:
1. **`cpp_inference/CMakeLists.txt`** - Enhanced with automatic gemmlowp patching
2. **`cpp_inference/fix_and_build_windows.bat`** - New automated build script
3. **`cpp_inference/EIGHT_BIT_INT_GEMM_FIX.md`** - Comprehensive documentation

### Changes Made to CMakeLists.txt:

The build process now:

1. **Downloads TensorFlow Lite** using FetchContent
2. **Checks for gemmlowp** after download (gemmlowp is a TF Lite dependency)
3. **Automatically patches** gemmlowp/CMakeLists.txt to disable the problematic `eight_bit_int_gemm` target
4. **Continues building** without the eight_bit_int_gemm target

### The Patch:

The patch comments out these lines in gemmlowp's CMakeLists.txt:
- `add_library(eight_bit_int_gemm ...)`
- `add_executable(eight_bit_int_gemm ...)`
- `target_link_libraries(eight_bit_int_gemm ...)`
- `set_target_properties(eight_bit_int_gemm ...)`

This prevents CMake from creating the problematic target that causes compilation errors on MinGW.

---

## Why This Error Occurs

The `eight_bit_int_gemm` target in Google's gemmlowp library:
- Has CMake configuration issues with MinGW
- Generates incorrect compiler commands (tries to compile multiple files with `-o`)
- **Is only a test/benchmark tool** - not needed for TensorFlow Lite inference
- Works fine on Linux/macOS but fails on Windows/MinGW

---

## Testing the Fix

After running the build script, verify the executables exist:

```batch
cd cpp_inference\build
dir *.exe
```

You should see:
- `radar_tagger.exe`
- `radar_tagger_multioutput.exe`

---

## For Developers

### Manual Application of Fix:

If you need to apply the patch manually:

```batch
cd cpp_inference\build

# Edit gemmlowp\CMakeLists.txt and comment out all lines containing:
# - add_library(eight_bit_int_gemm
# - add_executable(eight_bit_int_gemm  
# - target_link_libraries(eight_bit_int_gemm
# - set_target_properties(eight_bit_int_gemm

# Re-run CMake
cmake -G "MinGW Makefiles" -DCMAKE_BUILD_TYPE=Release ..

# Build
cmake --build . --config Release
```

### Automated via PowerShell:

```powershell
cd cpp_inference\build
$content = Get-Content 'gemmlowp\CMakeLists.txt' -Raw
$content = $content -replace 'add_library\(eight_bit_int_gemm', '# DISABLED: add_library(eight_bit_int_gemm'
$content = $content -replace 'add_executable\(eight_bit_int_gemm', '# DISABLED: add_executable(eight_bit_int_gemm'
$content = $content -replace 'target_link_libraries\(eight_bit_int_gemm', '# DISABLED: target_link_libraries(eight_bit_int_gemm'
$content = $content -replace 'set_target_properties\(eight_bit_int_gemm', '# DISABLED: set_target_properties(eight_bit_int_gemm'
Set-Content 'gemmlowp\CMakeLists.txt' $content
cmake -G "MinGW Makefiles" -DCMAKE_BUILD_TYPE=Release ..
cmake --build . --config Release
```

---

## Impact on Functionality

**✅ NO IMPACT** - The eight_bit_int_gemm target is **not used** by:
- TensorFlow Lite inference
- Model loading
- Prediction/inference operations
- Any runtime functionality

It's only used for benchmarking and testing during gemmlowp development.

---

## Build Process Flow (After Fix)

```
1. CMake Configure
   ↓
2. Download TensorFlow Lite (v2.14.0)
   ↓
3. TensorFlow downloads dependencies (including gemmlowp)
   ↓
4. [FIX APPLIED] Patch gemmlowp/CMakeLists.txt
   ↓
5. Configure TensorFlow Lite
   ↓
6. Build (eight_bit_int_gemm is not built)
   ↓
7. Success! radar_tagger.exe created
```

---

## Troubleshooting

### "gemmlowp not found" error:
```batch
# Clean and rebuild
cd cpp_inference
rmdir /s /q build
fix_and_build_windows.bat clean
```

### Build still fails:
1. Verify MinGW-w64 is installed: `g++ --version`
2. Verify CMake 3.16+: `cmake --version`
3. Check you're in the correct directory: `cd cpp_inference`
4. Try verbose build: `cmake --build build --config Release -- VERBOSE=1`

### Need more details:
See `cpp_inference/EIGHT_BIT_INT_GEMM_FIX.md` for comprehensive documentation.

---

## Platform Notes

- **Windows MinGW**: ✅ Fix applied automatically
- **Windows MSVC**: ⚠️ May not need fix (MSVC handles this differently)
- **Linux**: ✅ No fix needed (compiles fine)
- **macOS**: ✅ No fix needed (compiles fine)

---

## Summary

This is a **known compatibility issue** between gemmlowp and MinGW on Windows. The fix:
- ✅ Has been integrated into CMakeLists.txt
- ✅ Will apply automatically during build
- ✅ Does not affect functionality
- ✅ Works on all platforms (only applies on Windows/MinGW)
- ✅ Is safe and reversible

Just run `fix_and_build_windows.bat clean` and you're good to go! 🚀
