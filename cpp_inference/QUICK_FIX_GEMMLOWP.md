# QUICK FIX: gemmlowp eight_bit_int_gemm Error

## The Problem

You're seeing this error:
```
c++.exe: fatal error: cannot specify '-o' with '-c', '-S' or '-E' with multiple files
```

## The Solution (3 Steps)

### Step 1: Delete Your Build Directory

**Windows (PowerShell or Command Prompt):**
```batch
cd cpp_inference
rmdir /s /q build
```

**Linux/Mac:**
```bash
cd cpp_inference
rm -rf build
```

### Step 2: Use the New Build Script

**Windows:**
```batch
build_with_gemmlowp_fix.bat clean
```

**Linux/Mac:**
```bash
./build_with_gemmlowp_fix.sh clean
```

### Step 3: Wait for Build to Complete

The script will:
- Configure CMake
- Patch the problematic library
- Re-configure CMake
- Build the project

**This should take 5-15 minutes depending on your system.**

## That's It!

If the build completes successfully, you should see:
```
Build completed successfully!

Executables:
  - radar_tagger.exe
  - radar_tagger_multioutput.exe
```

## Still Not Working?

1. Make sure you have Python installed (Python 3.6+)
2. Make sure you have MinGW in your PATH
3. Try rebooting your computer (to clear any locked files)
4. Check Task Manager for stuck cmake.exe or mingw32-make.exe processes

## What Changed?

We created:
- `patch_gemmlowp_direct.py` - A script that patches the problematic library
- `build_with_gemmlowp_fix.bat/sh` - Build scripts that apply the fix automatically
- Multiple layers of protection in `CMakeLists.txt` to prevent the error

## Why Did This Happen?

The gemmlowp library (a TensorFlow Lite dependency) has a bug in its CMake configuration that affects MinGW/Windows builds. The problematic component (`eight_bit_int_gemm`) is not needed for TensorFlow Lite to work, so we simply disable it during the build.

## Need More Details?

See `GEMMLOWP_FINAL_FIX.md` for complete technical documentation.

---

**Last Updated:** 2025-11-25
