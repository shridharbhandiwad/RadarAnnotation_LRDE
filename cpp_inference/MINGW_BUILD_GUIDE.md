# Building with MinGW on Windows

## ⚠️ Important Note
This project includes fixes for known MinGW build issues with TensorFlow Lite's gemmlowp dependency. 
See `MINGW_GEMMLOWP_FIX.md` for technical details.

## Quick Start

If you have MinGW installed and in your PATH, simply run:

```batch
build_mingw_alternative.bat
```

## Prerequisites

1. **MinGW-w64** or **MinGW** installed
   - Download from: https://www.mingw-w64.org/ or https://osdn.net/projects/mingw/
   
2. **CMake** installed
   - Download from: https://cmake.org/download/

3. **MinGW bin directory must be in your PATH**

## Common Issues and Solutions

### Issue 1: "nmake not found" or "CMAKE_CXX_COMPILER not set"

**Cause:** CMake is trying to use Visual Studio's NMAKE instead of MinGW.

**Solution:** Explicitly specify MinGW Makefiles generator:

```batch
cmake -G "MinGW Makefiles" ..
```

### Issue 2: MinGW not in PATH

**Symptoms:**
- `gcc` or `g++` not found
- CMake can't find compiler

**Solution:** Add MinGW to PATH temporarily or permanently.

**Temporary (current terminal session only):**
```batch
set PATH=C:\mingw64\bin;%PATH%
```

**Permanent:**
1. Open System Properties → Advanced → Environment Variables
2. Edit the `Path` variable
3. Add your MinGW bin directory (e.g., `C:\mingw64\bin` or `C:\MinGW\bin`)

### Issue 3: Multiple compilers installed

If you have both Visual Studio and MinGW installed, CMake might get confused.

**Solution:** Explicitly tell CMake which compiler to use:

```batch
cmake -G "MinGW Makefiles" ^
    -DCMAKE_C_COMPILER=gcc ^
    -DCMAKE_CXX_COMPILER=g++ ^
    ..
```

### Issue 4: gemmlowp build error (FIXED)

**Error Message:**
```
c++.exe: fatal error: cannot specify '-o' with '-c', '-S' or '-E' with multiple files
mingw32-make[2]: *** [.../gemmlowp-build/...] Error 1
```

**Status:** ✅ This issue has been fixed in the CMakeLists.txt

The project now includes automatic workarounds for this MinGW-specific issue. No action needed.
See `MINGW_GEMMLOWP_FIX.md` for technical details.

## Manual Build Steps

If the batch scripts don't work, try these manual steps:

### Step 1: Verify MinGW Installation

```batch
gcc --version
g++ --version
mingw32-make --version
```

All commands should work and show version information.

### Step 2: Clean Build Directory

```batch
cd cpp_inference
rmdir /s /q build
mkdir build
cd build
```

### Step 3: Configure with CMake

```batch
cmake -G "MinGW Makefiles" -DCMAKE_BUILD_TYPE=Release ..
```

### Step 4: Build

```batch
cmake --build . --config Release
```

Or use make directly:

```batch
mingw32-make
```

## Alternative: Using MSBuild/Visual Studio

If you prefer to use Visual Studio instead of MinGW:

```batch
cmake -G "Visual Studio 16 2019" ..
cmake --build . --config Release
```

## Verify Build Success

After successful build, you should have:
- `build/radar_tagger.exe`
- `build/radar_tagger_multioutput.exe`

Test by running:
```batch
cd build
radar_tagger.exe --help
```

## Additional Tips

1. **Clean builds:** Always clean when switching between generators:
   ```batch
   rmdir /s /q build
   ```

2. **Debug builds:** For debugging, use:
   ```batch
   cmake -G "MinGW Makefiles" -DCMAKE_BUILD_TYPE=Debug ..
   ```

3. **Parallel builds:** Speed up compilation:
   ```batch
   cmake --build . --parallel 4
   ```

4. **Check CMake cache:** If issues persist, check what CMake detected:
   ```batch
   type CMakeCache.txt | findstr COMPILER
   ```

## Environment Setup Script

Create a file called `setup_mingw_env.bat`:

```batch
@echo off
REM Adjust this path to match your MinGW installation
set MINGW_PATH=C:\mingw64\bin

REM Add MinGW to PATH
set PATH=%MINGW_PATH%;%PATH%

REM Verify setup
echo Checking MinGW setup...
gcc --version
echo.
echo MinGW environment ready!
echo You can now run build_mingw_alternative.bat
```

Run this before building to ensure your environment is set up correctly.
