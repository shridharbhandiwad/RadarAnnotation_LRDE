# Windows CMake Configuration Fix

## Issue

CMake configuration fails on Windows with:

```
-- Disabled eight_bit_int_gemm target (method 2)
-- Configuring incomplete, errors occurred!
```

## Root Cause

The CMakeLists.txt was using `cmake_language(DEFER ...)` which requires **CMake 3.19 or higher**, but some Windows systems have older versions (3.16-3.18).

## Solution Applied ✅

The CMakeLists.txt has been updated to check the CMake version before using the `DEFER` command. The fix now gracefully handles older CMake versions.

## How to Build on Windows

### Method 1: Using the Build Script (Recommended)

```batch
cd cpp_inference
build_with_gemmlowp_fix.bat clean
```

### Method 2: Manual Build

```batch
cd cpp_inference

REM Clean previous build
rmdir /s /q build
mkdir build
cd build

REM Configure with MinGW
cmake -G "MinGW Makefiles" -DCMAKE_BUILD_TYPE=Release ..

REM Build
cmake --build . --config Release
```

### Method 3: If Using Visual Studio (MSVC)

```batch
cd cpp_inference

REM Clean previous build
rmdir /s /q build
mkdir build
cd build

REM Configure with Visual Studio
cmake -G "Visual Studio 16 2019" -A x64 ..

REM Build
cmake --build . --config Release
```

## Checking Your CMake Version

To check your CMake version:

```batch
cmake --version
```

### If You Have CMake < 3.16

You need to upgrade CMake. Download from: https://cmake.org/download/

**Recommended version:** CMake 3.20 or higher

## What Changed?

The fix adds a version check before using advanced CMake features:

```cmake
# Old code (caused errors on CMake < 3.19):
cmake_language(DEFER CALL disable_eight_bit_int_gemm)

# New code (works with CMake 3.16+):
if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.19")
    cmake_language(DEFER CALL disable_eight_bit_int_gemm)
else()
    message(STATUS "Skipping deferred fix (requires CMake 3.19+)")
endif()
```

## Multiple Fix Layers

The build system has **3 layers of fixes** for the gemmlowp issue:

1. **Method 1**: Direct target exclusion (works if target exists)
2. **Method 2**: Deferred target fix (only on CMake 3.19+)
3. **Method 3**: Source-level patching (in TensorFlow Lite CMakeLists)

Even if Method 2 is skipped on older CMake, Methods 1 and 3 should still work.

## Troubleshooting

### Issue: Still getting "Configuring incomplete"

1. **Check the full error output** (scroll up in your terminal)
2. **Verify you have internet connection** (downloads TensorFlow Lite)
3. **Check disk space** (needs ~2GB for dependencies)
4. **Try with verbose output:**
   ```batch
   cmake .. -G "MinGW Makefiles" --debug-output
   ```

### Issue: Cannot find compiler

Make sure MinGW is in your PATH:

```batch
where g++
where cmake
```

If not found, add MinGW to your PATH:
```batch
set PATH=C:\MinGW\bin;%PATH%
```

### Issue: TensorFlow Lite download fails

The build downloads TensorFlow Lite from GitHub. If it fails:

1. Check your internet connection
2. Check if GitHub is accessible
3. Try using a VPN if GitHub is blocked
4. Or manually download and extract TensorFlow Lite

### Issue: eight_bit_int_gemm compilation error

This should now be prevented by the patching in CMakeLists.txt. If you still see it:

1. **Delete build directory completely:**
   ```batch
   rmdir /s /q build
   ```

2. **Use the emergency fix script:**
   ```batch
   emergency_fix.bat
   ```

3. **Or patch manually** - see `WINDOWS_MINGW_BUILD_FIX.md`

## Expected Output (Success)

When configuration succeeds, you should see:

```
-- Radar Tagger C++ Configuration:
--   Version: 1.0.0
--   C++ Standard: 17
--   Build Type: Release
--   TensorFlow Lite: tensorflow-lite
--   ONNX Runtime: <path>/onnxruntime.lib
--   ONNX Runtime DLL: <path>/onnxruntime.dll
--
-- Configuring done
-- Generating done
-- Build files have been written to: <path>/build
```

Then build with:
```batch
cmake --build . --config Release
```

And you should see:
```
[100%] Built target radar_tagger
[100%] Built target radar_tagger_multioutput
```

## Verification

After successful build, test the executables:

```batch
cd build

REM Test help output
radar_tagger.exe --help
radar_tagger_multioutput.exe --help
```

You should see usage information for both executables.

## System Requirements

- **Operating System**: Windows 7 or higher
- **CMake**: 3.16 or higher (3.20+ recommended)
- **Compiler**: MinGW-w64 GCC 7+ or MSVC 2019+
- **Disk Space**: ~2GB for dependencies
- **RAM**: 4GB minimum (8GB recommended)
- **Internet**: Required for first build (downloads dependencies)

## Additional Help

- **gemmlowp issues**: See `WINDOWS_MINGW_BUILD_FIX.md`
- **General build help**: See `README.md`
- **Quick start**: See `START_HERE.md`
- **Step-by-step manual fix**: See `QUICK_BUILD_INSTRUCTIONS.txt`

## Date

Fixed: November 25, 2025  
Status: ✅ Resolved - CMake version compatibility issue fixed
