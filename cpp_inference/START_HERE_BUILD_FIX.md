# 🚨 START HERE: Fix for "eight_bit_int_gemm" Build Error 🚨

## You're seeing this error, right?

```
c++.exe: fatal error: cannot specify '-o' with '-c', '-S' or '-E' with multiple files
compilation terminated.
mingw32-make[2]: *** [eight_bit_int_gemm] Error 1
```

## ✅ HERE'S THE FIX (30 seconds to apply)

### Windows Users:

```batch
cd cpp_inference
build_with_gemmlowp_fix.bat clean
```

### Linux/Mac Users:

```bash
cd cpp_inference
./build_with_gemmlowp_fix.sh clean
```

**That's it!** ✨

The script will:
1. Clean your build directory
2. Configure CMake
3. Patch the problematic library (gemmlowp)
4. Build everything correctly

**Time:** 10-20 minutes for first build

---

## 🆘 If That Didn't Work

### Option A: Emergency Fix (Mid-Build Failure)

If the build fails partway through:

**Windows:**
```batch
emergency_fix.bat
```

**Linux/Mac:**
```bash
./emergency_fix.sh
```

### Option B: Nuclear Option (Complete Reset)

If nothing else works:

**Windows:**
```batch
cd cpp_inference
rmdir /s /q build
del CMakeCache.txt
build_with_gemmlowp_fix.bat clean
```

**Linux/Mac:**
```bash
cd cpp_inference
rm -rf build
rm -f CMakeCache.txt
./build_with_gemmlowp_fix.sh clean
```

---

## 📖 Documentation Map

Choose your path:

### 🏃 **I just want it to work NOW**
→ [BUILD_CHECKLIST.md](BUILD_CHECKLIST.md)
- Step-by-step checklist
- Pre-flight checks
- Troubleshooting for common errors

### 🚀 **Quick 3-step fix**
→ [QUICK_FIX_GEMMLOWP.md](QUICK_FIX_GEMMLOWP.md)
- Minimal instructions
- What the scripts do
- Why it's happening

### 📚 **I want to understand the problem**
→ [GEMMLOWP_FINAL_FIX.md](GEMMLOWP_FINAL_FIX.md)
- Root cause analysis
- All fix methods explained
- Technical deep-dive

### 📋 **Give me a summary**
→ [SOLUTION_SUMMARY.md](SOLUTION_SUMMARY.md)
- What was done
- What you need to do
- Files created/modified

### 📖 **Complete usage guide**
→ [README.md](README.md)
- Full build instructions
- Usage examples
- Integration guide

---

## 🎯 Success Looks Like This

After the build script finishes, you should see:

```
================================================
  Build completed successfully!
================================================

Executables:
  - radar_tagger.exe
  - radar_tagger_multioutput.exe
```

And you can test them:

```batch
cd build
radar_tagger.exe --help
radar_tagger_multioutput.exe --help
```

---

## ❓ FAQ

### Q: Why does this keep happening?

**A:** The gemmlowp library (a TensorFlow Lite dependency) has a bug in its CMake configuration for Windows MinGW. It generates incorrect compiler commands.

### Q: Is it safe to patch gemmlowp?

**A:** Yes! We're only disabling a test/benchmark tool (`eight_bit_int_gemm`) that isn't needed for TensorFlow Lite to work. All inference functionality is preserved.

### Q: Will I need to do this every time?

**A:** No! Once you build successfully:
- The patched files stay patched
- Subsequent builds are fast (1-2 minutes)
- Only need to re-patch if you delete the `build/` directory

### Q: What if I use a different compiler?

**A:** This issue is specific to MinGW on Windows. If you use:
- **MSVC (Visual Studio)**: No issue, build normally
- **Linux GCC**: No issue, build normally  
- **Mac Clang**: No issue, build normally

### Q: Can I just skip building the C++ part?

**A:** Yes! The Python inference works fine without the C++ build. The C++ version is for:
- Production deployment
- Real-time systems
- Embedded devices
- Maximum performance

---

## 🔧 What Changed?

We created:

### New Build Scripts (Use These!)
- `build_with_gemmlowp_fix.bat` - Fixed Windows build
- `build_with_gemmlowp_fix.sh` - Fixed Linux/Mac build
- `emergency_fix.bat` - Windows emergency fix
- `emergency_fix.sh` - Linux/Mac emergency fix

### Patching Scripts (Automatic)
- `patch_gemmlowp_direct.py` - Patches CMakeLists.txt
- `patch_makefile_direct.py` - Patches Makefiles (last resort)

### Documentation (You Are Here!)
- `START_HERE_BUILD_FIX.md` - This file
- `BUILD_CHECKLIST.md` - Step-by-step guide
- `QUICK_FIX_GEMMLOWP.md` - Quick 3-step fix
- `GEMMLOWP_FINAL_FIX.md` - Technical documentation
- `SOLUTION_SUMMARY.md` - What was done

### Updated Files
- `CMakeLists.txt` - Added 4 layers of patching protection
- `README.md` - Updated build instructions

---

## 🎓 TL;DR

1. Run `build_with_gemmlowp_fix.bat clean` (Windows) or `./build_with_gemmlowp_fix.sh clean` (Linux/Mac)
2. Wait 10-20 minutes
3. Done!

If it doesn't work, run `emergency_fix.bat` (or `.sh`) and try again.

---

**Need help?** Check the documentation files above. One of them has your answer! 📚

**Last Updated:** 2025-11-25
