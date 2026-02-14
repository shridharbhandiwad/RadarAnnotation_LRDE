# Windows Build - Complete Guide

## Quick Start (Pick One)

### Option 1: Robust Build Script (Recommended)
```batch
cd cpp_inference
build_windows_robust.bat clean
```
**Best for:** Most users, provides diagnostics and fallback options

### Option 2: ONNX-Only Build (Simplified)
```batch
cd cpp_inference
build_onnx_only.bat clean
```
**Best for:** Quick builds, avoiding TensorFlow Lite issues

### Option 3: MSVC Build
```batch
cd cpp_inference
build_msvc.bat clean
```
**Best for:** Visual Studio users, better debugging

### Option 4: WSL2 (Linux on Windows)
```bash
# In WSL2 Ubuntu
cd cpp_inference
./build.sh
```
**Best for:** Most reliable, uses Linux build system

### Option 5: Pre-built Binaries
Download from GitHub Actions artifacts (see below)

**Best for:** End users, no compilation needed

---

## Understanding Your Error

### Error: "Configuring incomplete, errors occurred!"

This generic error can have many causes. To diagnose:

1. **Check the full output** - scroll up to see the actual error
2. **Look in build/cmake_config_output.txt** for details
3. **Run build_windows_robust.bat** for automatic diagnostics

Common specific errors:

#### A. "cannot find -lstdc++"
**Cause:** Missing C++ standard library  
**Solution:** Install MinGW-w64 properly or use MSVC instead
```batch
build_windows_robust.bat msvc
```

#### B. "CMAKE_CXX_COMPILER not found"
**Cause:** No compiler detected  
**Solution:** Install Visual Studio or MinGW and add to PATH

#### C. "FetchContent failed"
**Cause:** Internet connection or GitHub access issue  
**Solution:** Check connection, use VPN if GitHub is blocked

#### D. "eight_bit_int_gemm" errors
**Cause:** TensorFlow Lite gemmlowp build issue  
**Solution:** Use ONNX-only build
```batch
build_onnx_only.bat clean
```

#### E. "Policy CMP" warnings
**Cause:** CMake version incompatibility  
**Solution:** Upgrade to CMake 3.20+

---

## Prerequisites

### Required Software

1. **CMake 3.16+** (3.20+ recommended)
   - Download: https://cmake.org/download/
   - Install and add to PATH
   - Verify: `cmake --version`

2. **Compiler** (choose one):

   **Option A: MinGW-w64 (GCC for Windows)**
   - Download: https://www.mingw-w64.org/downloads/
   - Or use: https://winlibs.com/
   - Install to C:\MinGW
   - Add C:\MinGW\bin to PATH
   - Verify: `g++ --version`

   **Option B: Visual Studio 2019/2022**
   - Download: https://visualstudio.microsoft.com/
   - Install "Desktop development with C++"
   - Use "x64 Native Tools Command Prompt"
   - Verify: `cl` (should not say "not found")

3. **Python 3.x** (optional, for patch scripts)
   - Download: https://www.python.org/downloads/
   - Verify: `python --version`

4. **Git** (optional, for version control)
   - Download: https://git-scm.com/download/win

### System Requirements

- **OS:** Windows 7 or later (Windows 10/11 recommended)
- **RAM:** 4GB minimum (8GB recommended for compilation)
- **Disk Space:** ~3GB for dependencies and build files
- **Internet:** Required for first build (downloads dependencies)

---

## Detailed Build Instructions

### Method 1: Using build_windows_robust.bat

This is the recommended method with automatic diagnostics.

```batch
REM Navigate to project directory
cd C:\path\to\RadarAnnotation_LRDE\cpp_inference

REM Clean build (recommended for first build or after errors)
build_windows_robust.bat clean

REM Incremental build (faster, after changes)
build_windows_robust.bat

REM Force MSVC
build_windows_robust.bat msvc

REM Force MinGW
build_windows_robust.bat mingw
```

**What it does:**
1. ✅ Checks CMake version
2. ✅ Detects available compiler
3. ✅ Tests internet connection
4. ✅ Configures project
5. ✅ Applies patches automatically
6. ✅ Builds executables
7. ✅ Verifies outputs
8. ✅ Provides detailed error diagnostics

**Expected output:**
```
============================================================================
  BUILD SUCCESSFUL!
============================================================================

Executables are in: C:\...\cpp_inference\build

To test:
  radar_tagger.exe --help
  radar_tagger_multioutput.exe --help
```

### Method 2: ONNX-Only Build (Simplified)

Avoids TensorFlow Lite build issues.

```batch
cd cpp_inference
build_onnx_only.bat clean
```

**What you get:**
- ✅ `radar_tagger_onnx.exe` - Multi-output inference
- ✅ Supports ONNX models (.onnx)
- ❌ No TensorFlow Lite support (.tflite)

**Advantages:**
- Much faster build time
- Fewer build issues
- Smaller executable
- No gemmlowp problems

**When to use:**
- You only need ONNX models
- Standard build is failing
- Quick testing

### Method 3: MSVC Build

```batch
REM Open "x64 Native Tools Command Prompt for VS 2019"
REM (Find in Start Menu under Visual Studio 2019)

cd cpp_inference
build_msvc.bat clean
```

**Advantages:**
- Better Windows compatibility
- Excellent debugging with Visual Studio
- Official Microsoft toolchain

**When to use:**
- You have Visual Studio installed
- MinGW build is failing
- You need to debug the code

### Method 4: Manual CMake (Advanced)

For full control over the build process.

#### With MinGW:
```batch
cd cpp_inference
rmdir /s /q build
mkdir build
cd build

REM Configure
cmake -G "MinGW Makefiles" -DCMAKE_BUILD_TYPE=Release ..

REM Build
cmake --build . --config Release

REM Check output
dir radar_tagger*.exe
```

#### With MSVC:
```batch
cd cpp_inference
rmdir /s /q build
mkdir build
cd build

REM Configure
cmake -G "Visual Studio 16 2019" -A x64 -DCMAKE_BUILD_TYPE=Release ..

REM Build
cmake --build . --config Release

REM Check output
dir Release\radar_tagger*.exe
```

### Method 5: WSL2 (Windows Subsystem for Linux)

Most reliable method, uses Linux build system.

#### Setup WSL2:
```powershell
REM In PowerShell (Administrator)
wsl --install

REM Restart your computer

REM Launch Ubuntu
wsl
```

#### Build in WSL2:
```bash
# Inside WSL2
cd /mnt/c/Users/YourUsername/path/to/cpp_inference

# Install dependencies
sudo apt update
sudo apt install -y build-essential cmake git python3

# Build
chmod +x build.sh
./build.sh

# Binaries will be in build/
ls -lh build/radar_tagger*
```

**Advantages:**
- ⭐ Most reliable
- ⭐ No Windows-specific issues
- ⭐ Faster compilation
- ⭐ Linux binaries (use in WSL2)

---

## Troubleshooting

### Issue: CMake not found

**Error:** `'cmake' is not recognized as an internal or external command`

**Solution:**
1. Install CMake from https://cmake.org/download/
2. During installation, select "Add CMake to system PATH"
3. Or manually add to PATH:
   ```batch
   setx PATH "%PATH%;C:\Program Files\CMake\bin"
   ```
4. Restart your command prompt
5. Verify: `cmake --version`

### Issue: Compiler not found

**Error:** `No CMAKE_CXX_COMPILER could be found`

**Solution:**

For MinGW:
1. Download from https://winlibs.com/ (easiest)
2. Extract to C:\MinGW
3. Add to PATH:
   ```batch
   setx PATH "%PATH%;C:\MinGW\bin"
   ```
4. Restart command prompt
5. Verify: `g++ --version`

For MSVC:
1. Install Visual Studio 2019 or 2022
2. Select "Desktop development with C++"
3. Use "x64 Native Tools Command Prompt for VS"
4. Verify: `cl` (should show version)

### Issue: Internet connection / download failures

**Error:** `Failed to download...` or `FetchContent error`

**Solutions:**
1. Check internet connection: `ping github.com`
2. Use VPN if GitHub is blocked
3. Increase timeout in CMakeLists.txt
4. Download dependencies manually:
   - TensorFlow Lite: https://github.com/tensorflow/tensorflow
   - ONNX Runtime: https://github.com/microsoft/onnxruntime/releases

### Issue: Out of memory during build

**Error:** Build stops with no error, or "out of memory"

**Solutions:**
1. Close other applications
2. Build without parallel jobs:
   ```batch
   cmake --build . --config Release -- -j1
   ```
3. Upgrade RAM (8GB recommended)
4. Use ONNX-only build (smaller)

### Issue: eight_bit_int_gemm compilation errors

**Error:** `error: cannot compile eight_bit_int_gemm`

**Solutions:**

1. **Use ONNX-only build (easiest):**
   ```batch
   build_onnx_only.bat clean
   ```

2. **Clean and rebuild:**
   ```batch
   cd cpp_inference
   rmdir /s /q build
   build_windows_robust.bat clean
   ```

3. **Manual patch:**
   ```batch
   cd cpp_inference
   python patch_gemmlowp_direct.py
   ```

4. **Use WSL2** (most reliable)

### Issue: Permission denied

**Error:** `Access denied` or `Permission denied`

**Solutions:**
1. Run as Administrator
2. Disable antivirus temporarily
3. Check if files are locked by another process
4. Use WSL2 instead

### Issue: DLL not found at runtime

**Error:** `The program can't start because onnxruntime.dll is missing`

**Solution:**
Copy the DLL to the same directory as the executable:
```batch
cd cpp_inference\build
copy _deps\onnxruntime-src\lib\onnxruntime.dll .
```

Or add the directory to PATH:
```batch
setx PATH "%PATH%;C:\path\to\cpp_inference\build\_deps\onnxruntime-src\lib"
```

---

## Getting Pre-built Binaries

If you cannot build from source, you can download pre-built binaries.

### Option 1: GitHub Actions Artifacts

1. Go to the GitHub repository
2. Click "Actions" tab
3. Click on a successful workflow run
4. Download "radar-tagger-windows-msvc-Release" artifact
5. Extract and use

### Option 2: GitHub Releases

1. Go to https://github.com/your-repo/releases
2. Download the latest release ZIP file
3. Extract to your desired location
4. Run the executables

### Option 3: Request from Maintainer

Contact the project maintainer to request pre-built binaries for your platform.

---

## Testing Your Build

After successful build:

```batch
cd cpp_inference\build

REM For MSVC builds
cd Release

REM Test help output
radar_tagger.exe --help
radar_tagger_multioutput.exe --help

REM Verify DLL
dir onnxruntime.dll

REM Check executable sizes (should be several MB)
dir radar_tagger*.exe
```

Expected output:
```
Usage: radar_tagger.exe [options]
...
```

---

## Performance Comparison

| Method | Build Time | Reliability | Complexity |
|--------|-----------|-------------|-----------|
| WSL2 | ⭐⭐⭐⭐⭐ Fast | ⭐⭐⭐⭐⭐ High | ⭐⭐⭐ Medium |
| ONNX-only | ⭐⭐⭐⭐⭐ Fast | ⭐⭐⭐⭐⭐ High | ⭐⭐⭐⭐⭐ Easy |
| MSVC | ⭐⭐⭐ Medium | ⭐⭐⭐⭐ Good | ⭐⭐⭐⭐ Easy |
| MinGW | ⭐⭐ Slow | ⭐⭐⭐ Medium | ⭐⭐⭐ Medium |
| Pre-built | ⭐⭐⭐⭐⭐ Instant | ⭐⭐⭐⭐⭐ High | ⭐⭐⭐⭐⭐ Easy |

---

## Additional Resources

- **WINDOWS_BUILD_ALTERNATIVES.md** - Alternative build methods
- **WINDOWS_CMAKE_VERSION_FIX.md** - CMake version issues
- **WINDOWS_MINGW_BUILD_FIX.md** - MinGW-specific issues
- **START_HERE.md** - General project documentation
- **README.md** - Project overview

---

## Support

If you're still having issues:

1. **Check diagnostics:**
   ```batch
   build_windows_robust.bat
   ```
   This will show what's wrong.

2. **Review logs:**
   - `build/cmake_config_output.txt`
   - `build/CMakeFiles/CMakeError.log`

3. **Try alternatives:**
   - ONNX-only build
   - WSL2
   - Pre-built binaries

4. **Open an issue:**
   Include:
   - Windows version
   - CMake version (`cmake --version`)
   - Compiler version (`g++ --version` or `cl`)
   - Full error log

5. **Contact maintainer** for assistance

---

## Success Criteria

Build is successful when you see:

```
============================================================================
  BUILD SUCCESSFUL!
============================================================================

Executables are in: ...

To test:
  radar_tagger.exe --help
  radar_tagger_multioutput.exe --help
```

And you can run:
```batch
radar_tagger.exe --help
```

Without errors.

---

**Document Version:** 1.0  
**Last Updated:** November 25, 2025  
**Status:** Complete and tested
