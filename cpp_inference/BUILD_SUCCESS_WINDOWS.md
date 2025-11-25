# Windows Build Success Guide

## ✅ Status: Ready for Windows Build

This guide provides comprehensive instructions for building the Radar Tagger C++ inference project on Windows using MinGW.

---

## Prerequisites

### Required Software

1. **CMake** (version 3.16 or later)
   - Download: https://cmake.org/download/
   - During installation, select "Add CMake to system PATH"

2. **MinGW-w64** (GCC compiler for Windows)
   - **Option A - MSYS2** (Recommended):
     - Download: https://www.msys2.org/
     - After installation, open MSYS2 MINGW64 terminal and run:
       ```bash
       pacman -S mingw-w64-x86_64-gcc mingw-w64-x86_64-cmake mingw-w64-x86_64-make
       ```
     - Add to PATH: `C:\msys64\mingw64\bin`
   
   - **Option B - MinGW-w64 Standalone**:
     - Download: https://sourceforge.net/projects/mingw-w64/
     - Add to PATH: `C:\mingw64\bin` (or your installation directory)

3. **Git** (for downloading TensorFlow Lite)
   - Download: https://git-scm.com/download/win

4. **Internet Connection**
   - Required for downloading TensorFlow Lite (~500 MB) and ONNX Runtime (~50 MB)

### Verify Installation

Open Command Prompt or PowerShell and verify:

```cmd
cmake --version
g++ --version
mingw32-make --version
git --version
```

All commands should return version information without errors.

---

## Building the Project

### Method 1: Using the Build Script (Recommended)

1. Open Command Prompt
2. Navigate to the project directory:
   ```cmd
   cd cpp_inference
   ```
3. Run the build script:
   ```cmd
   rebuild_windows.bat
   ```

The script will:
- ✅ Clean any previous build artifacts
- ✅ Configure CMake with MinGW
- ✅ Download dependencies (TensorFlow Lite, ONNX Runtime)
- ✅ Apply Windows/MinGW patches automatically
- ✅ Build the project with parallel compilation
- ✅ Verify the executables

**Expected Build Time:**
- First build: 20-45 minutes (downloads dependencies)
- Subsequent builds: 2-5 minutes (only recompiles changed files)

### Method 2: Manual Build Steps

```cmd
cd cpp_inference
rmdir /s /q build
mkdir build
cd build
cmake -G "MinGW Makefiles" -DCMAKE_BUILD_TYPE=Release ..
cmake --build . --config Release -- -j%NUMBER_OF_PROCESSORS%
```

---

## Build Output

After successful build, you'll find:

```
cpp_inference/build/
├── radar_tagger.exe              (~4 MB)
├── radar_tagger_multioutput.exe  (~4 MB)
├── _deps/
│   ├── onnxruntime-src/lib/onnxruntime.dll
│   └── tensorflow-src/...
└── ...
```

---

## Testing the Build

### Basic Functionality Test

```cmd
cd cpp_inference\build
radar_tagger.exe --help
radar_tagger_multioutput.exe --help
```

Both commands should display usage information without errors.

### Running Inference (Example)

First, export your trained models to TensorFlow Lite format:

```cmd
cd /path/to/project
python convert_model_to_tflite.py
python export_models_to_onnx.py
```

Then run inference:

```cmd
cd cpp_inference\build
radar_tagger.exe ^
    --model ..\..\models\your_model.tflite ^
    --metadata ..\..\models\metadata.json ^
    --test-data ..\..\data\test_data.csv
```

---

## Understanding Windows/MinGW Patches

### Automatic Patching

The `CMakeLists.txt` includes extensive patches (lines 36-291) that automatically activate when building on Windows:

1. **eight_bit_int_gemm Disabled**
   - This TensorFlow Lite test tool has known issues on MinGW
   - It's automatically disabled (not needed for inference)

2. **min/max Macro Fixes**
   - Removes problematic min/max macros that conflict with Windows headers
   - Adds `NOMINMAX` definition

3. **Large Object File Support**
   - Adds `-Wa,-mbig-obj` flag to handle large TensorFlow Lite objects

4. **gemmlowp Compatibility**
   - Patches gemmlowp library CMakeLists.txt automatically

### Patch Detection

When CMake runs, you'll see messages like:

```
-- MinGW/Windows detected - gemmlowp and cpuinfo patches will be applied
-- Patching TensorFlow Lite CMakeLists.txt for MinGW compatibility...
-- Disabled eight_bit_int_gemm target
```

This confirms the patches are being applied correctly.

---

## Common Issues and Solutions

### Issue 1: "mingw32-make not found"

**Cause:** MinGW is not installed or not in PATH

**Solution:**
```cmd
REM Verify MinGW is installed
dir C:\mingw64\bin\mingw32-make.exe

REM Add to PATH (temporary)
set PATH=C:\mingw64\bin;%PATH%

REM Or add permanently via System Environment Variables
```

### Issue 2: "cmake not found"

**Cause:** CMake is not installed or not in PATH

**Solution:**
- Download from https://cmake.org/download/
- During installation, select "Add CMake to system PATH"
- Or manually add: `C:\Program Files\CMake\bin`

### Issue 3: Build fails with "cannot find -lonnxruntime"

**Cause:** ONNX Runtime download incomplete or corrupted

**Solution:**
```cmd
cd cpp_inference
rmdir /s /q build
rebuild_windows.bat
```

### Issue 4: "The system cannot find the path specified"

**Cause:** Path too long or incorrect path separators

**Solution:**
- Use shorter paths (avoid deep nested directories)
- Windows accepts both `\` and `/` as path separators
- Use quotes for paths with spaces: `"C:\My Projects\..."`

### Issue 5: Build takes forever (>1 hour)

**Cause:** Normal for first build (downloading TensorFlow Lite)

**Solution:**
- Be patient (20-45 minutes is normal)
- Subsequent builds are much faster
- Ensure good internet connection

### Issue 6: "fatal error: too many open files"

**Cause:** Parallel compilation overwhelming the system

**Solution:**
```cmd
REM Use fewer parallel jobs
cmake --build . --config Release -- -j2
```

### Issue 7: executable won't run - missing DLL

**Cause:** ONNX Runtime DLL not in PATH

**Solution:**
```cmd
REM Copy DLL to executable directory
copy _deps\onnxruntime-src\lib\onnxruntime.dll .

REM Or add to PATH
set PATH=%CD%\_deps\onnxruntime-src\lib;%PATH%
```

---

## Rebuilding After Changes

### If you modified C++ source files only:

```cmd
cd cpp_inference\build
cmake --build . --config Release
```

⏱️ Takes ~2-5 minutes

### If you modified CMakeLists.txt or want a full clean rebuild:

```cmd
cd cpp_inference
rebuild_windows.bat
```

⏱️ Takes ~20-45 minutes (re-downloads dependencies)

---

## Project Structure

```
cpp_inference/
├── CMakeLists.txt              # Main build configuration
├── rebuild_windows.bat         # Windows build script
├── radar_tagger.h              # Single-output inference header
├── radar_tagger.cpp            # Single-output implementation
├── main.cpp                    # Single-output executable
├── radar_tagger_multioutput.h  # Multi-output inference header
├── radar_tagger_multioutput.cpp # Multi-output implementation
├── main_multioutput.cpp        # Multi-output executable
├── README.md                   # API documentation
└── build/                      # Generated build artifacts
    ├── radar_tagger.exe
    ├── radar_tagger_multioutput.exe
    └── _deps/                  # Downloaded dependencies
```

---

## Dependencies Downloaded During Build

### TensorFlow Lite (v2.14.0)
- **Size:** ~500 MB
- **Source:** GitHub (tensorflow/tensorflow)
- **Built from source:** Yes
- **Includes:** cpuinfo, gemmlowp, flatbuffers, ruy, fft2d, pthreadpool

### ONNX Runtime (v1.16.3)
- **Size:** ~50 MB
- **Source:** GitHub releases (microsoft/onnxruntime)
- **Pre-built:** Yes (Windows x64)
- **File:** onnxruntime.dll

### nlohmann/json (v3.11.2)
- **Size:** ~500 KB
- **Source:** GitHub releases
- **Header-only:** Yes

---

## Integration into Your Application

### Option 1: Use as Executable

```cmd
radar_tagger.exe --model model.tflite --metadata metadata.json --test-data test.csv
```

### Option 2: Link as Library

Include the headers in your C++ project:

```cpp
#include "radar_tagger.h"
// or
#include "radar_tagger_multioutput.h"
```

Link against:
- `tensorflow-lite.lib`
- `onnxruntime.lib`

---

## Performance Tips

1. **Use Release Build**
   ```cmd
   cmake -DCMAKE_BUILD_TYPE=Release ..
   ```
   Release builds are 3-5x faster than Debug builds

2. **Enable Parallel Compilation**
   ```cmd
   cmake --build . -- -j%NUMBER_OF_PROCESSORS%
   ```

3. **Profile Your Model**
   - Use TensorFlow Lite's profiling tools
   - Optimize model size with quantization

---

## Next Steps

1. ✅ **Test the build**
   ```cmd
   cd cpp_inference\build
   radar_tagger.exe --help
   ```

2. ✅ **Export your models**
   ```cmd
   python convert_model_to_tflite.py
   python export_models_to_onnx.py
   ```

3. ✅ **Run inference**
   ```cmd
   radar_tagger.exe --model ..\..\models\model.tflite
   ```

4. ✅ **Integrate into your application**
   - See `radar_tagger.h` for API documentation
   - See `main.cpp` for usage examples

---

## Support and Resources

- **CMakeLists.txt:** Build configuration with MinGW patches
- **WINDOWS_MINGW_BUILD_FIX.md:** Detailed patch explanations
- **rebuild_windows.bat:** Automated build script
- **TensorFlow Lite Docs:** https://www.tensorflow.org/lite
- **ONNX Runtime Docs:** https://onnxruntime.ai/

---

## Build Summary

| Component | Version | Built From |
|-----------|---------|------------|
| CMake | 3.16+ | System |
| MinGW GCC | 8.1+ | System |
| TensorFlow Lite | v2.14.0 | Source |
| ONNX Runtime | v1.16.3 | Pre-built |
| nlohmann/json | v3.11.2 | Header-only |
| C++ Standard | C++17 | Required |

**Build Status:** ✅ Ready for Windows  
**Last Updated:** November 25, 2025  
**Tested On:** Windows 10/11 with MinGW-w64
