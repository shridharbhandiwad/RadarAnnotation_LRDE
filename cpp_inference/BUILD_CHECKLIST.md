# Build Checklist - Fix eight_bit_int_gemm Error

## ✅ Pre-Flight Check

Before you start, verify these requirements:

- [ ] Python 3.6+ is installed
  ```batch
  python --version
  ```

- [ ] CMake 3.15+ is installed
  ```batch
  cmake --version
  ```

- [ ] MinGW/GCC is installed and in PATH (Windows) or g++ (Linux/Mac)
  ```batch
  g++ --version
  ```

- [ ] Internet connection is working (needed to download dependencies)

## 🚀 Quick Fix (3 Steps)

### Step 1: Clean Your Build Directory

**Windows:**
```batch
cd cpp_inference
rmdir /s /q build
```

**Linux/Mac:**
```bash
cd cpp_inference
rm -rf build
```

### Step 2: Run the Fixed Build Script

**Windows:**
```batch
build_with_gemmlowp_fix.bat clean
```

**Linux/Mac:**
```bash
./build_with_gemmlowp_fix.sh clean
```

### Step 3: Wait for Build to Complete

⏱️ **Expected time:** 10-20 minutes for first build

You should see:
- ✅ "Configuring with CMake..."
- ✅ "Patching gemmlowp (if present)..."
- ✅ "Re-configuring CMake after patching..."
- ✅ "Building project..."
- ✅ "Build completed successfully!"

## 📦 What Gets Built

After successful build, you should have:

```
cpp_inference/build/
  ├── radar_tagger.exe              (or radar_tagger on Linux)
  └── radar_tagger_multioutput.exe  (or radar_tagger_multioutput on Linux)
```

## ⚠️ If Build Fails

### Scenario A: Python Not Found

**Error:** `'python' is not recognized...`

**Fix:**
1. Install Python 3.6+ from https://python.org
2. Add Python to your PATH
3. Try again

### Scenario B: Still Getting eight_bit_int_gemm Error

**Error:** `cannot specify '-o' with '-c'`

**Fix:** Run the emergency fix script

**Windows:**
```batch
emergency_fix.bat
```

**Linux/Mac:**
```bash
./emergency_fix.sh
```

### Scenario C: CMake Fails to Configure

**Error:** `CMake Error at...`

**Fix:**
1. Check CMake version: `cmake --version` (need 3.15+)
2. Delete all cache files:
   ```batch
   del CMakeCache.txt
   del cmake_install.cmake
   ```
3. Try again

### Scenario D: Out of Memory

**Error:** `c++: fatal error: Killed signal terminated program cc1plus`

**Fix:** Use single-threaded build (this is automatic in the fixed script)

### Scenario E: Stuck Processes

**Error:** Build hangs or shows "Permission denied"

**Fix:**
1. Open Task Manager (Windows) or `ps aux` (Linux)
2. Kill any `cmake.exe`, `g++.exe`, or `mingw32-make.exe` processes
3. Delete build directory
4. Try again

## 🧪 Verify Build Success

After build completes:

**Step 1:** Check executables exist
```batch
cd build
dir radar_tagger.exe
dir radar_tagger_multioutput.exe
```

**Step 2:** Try running them
```batch
radar_tagger.exe --help
radar_tagger_multioutput.exe --help
```

Should show usage information without errors.

## 📚 Next Steps

Once build succeeds:

1. **Export your models** to TFLite/ONNX format
   - See `export_models_to_onnx.py` in parent directory
   - See `convert_model_to_tflite.py` for Neural Networks

2. **Test C++ inference**
   ```batch
   radar_tagger_multioutput.exe --model model.tflite --metadata metadata.json --model-type nn
   ```

3. **Read the full documentation**
   - `README.md` - Complete usage guide
   - `GEMMLOWP_FINAL_FIX.md` - Technical details

## 🆘 Still Need Help?

If none of the above fixes work:

1. Check all documentation in `cpp_inference/`:
   - `SOLUTION_SUMMARY.md`
   - `QUICK_FIX_GEMMLOWP.md`
   - `GEMMLOWP_FINAL_FIX.md`

2. Verify your environment:
   - Python 3.6+
   - CMake 3.15+
   - MinGW/GCC with C++17 support

3. Try on a different machine or use WSL (Windows Subsystem for Linux)

## ✨ Success Criteria

You know it worked when:

- [ ] Build completes without errors
- [ ] Both executables exist in `build/` directory
- [ ] Running executables with `--help` shows usage information
- [ ] No "eight_bit_int_gemm" errors appear

---

**Good luck!** 🎉

The build process downloads and compiles TensorFlow Lite from source, so be patient on the first build.
Subsequent builds will be much faster (1-2 minutes).
