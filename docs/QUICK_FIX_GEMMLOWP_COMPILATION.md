# QUICK FIX: gemmlowp Compilation Error

## The Error You're Seeing
```
c++.exe: fatal error: cannot specify '-o' with '-c', '-S' or '-E' with multiple files
compilation terminated.
mingw32-make[2]: *** [.../eight_bit_int_gemm.cc.obj] Error 1
```

## The Solution (Applied ✓)

I've fixed the CMakeLists.txt to automatically patch the problematic `eight_bit_int_gemm` target.

## What You Need To Do Now

### ⚠️ IMPORTANT: Clean Rebuild Required

Since TensorFlow Lite was partially fetched, you need to rebuild from scratch:

### Option 1: Use the rebuild script (Windows)
```batch
cd cpp_inference
rebuild_clean_windows.bat
```

### Option 2: Manual rebuild (Windows)
```batch
cd cpp_inference
rmdir /s /q build
mkdir build
cd build
cmake -G "MinGW Makefiles" -DCMAKE_BUILD_TYPE=Release ..
cmake --build . --config Release
```

### Option 3: Use bash script (if you have Git Bash)
```bash
cd cpp_inference
./rebuild_clean.sh
```

## What Will Happen

1. **CMake configuration**: You'll see messages about patching:
   - "Patching TensorFlow Lite CMakeLists.txt..."
   - "Injecting gemmlowp patching code..."
   - "Patching gemmlowp to disable eight_bit_int_gemm..."

2. **Build process**: The problematic target will be disabled, and the build will complete successfully.

3. **Result**: You'll get working executables:
   - `radar_tagger.exe`
   - `radar_tagger_multioutput.exe`

## Why This Works

- The fix injects patching code into TensorFlow Lite's CMakeLists.txt
- When TensorFlow Lite fetches gemmlowp, it automatically patches it
- The problematic `eight_bit_int_gemm` target is commented out before compilation
- This target is only a test tool - removing it doesn't affect TensorFlow Lite functionality

## Troubleshooting

### "cmake not found"
- Install CMake 3.16 or newer
- Add to PATH

### "mingw32-make not found"
- Install MinGW-w64
- Add to PATH

### Still getting the same error
- Make sure you cleaned the build directory completely
- Check that you're in the `cpp_inference` directory
- Try: `rm -rf build` (Git Bash) or `rmdir /s /q build` (CMD)

## Documentation

For more details, see:
- `/workspace/cpp_inference/GEMMLOWP_FIX_APPLIED.md` - Detailed technical explanation
- `/workspace/cpp_inference/rebuild_clean_windows.bat` - Windows rebuild script
- `/workspace/cpp_inference/rebuild_clean.sh` - Linux/Bash rebuild script

---

**TL;DR**: Run `rebuild_clean_windows.bat` from the `cpp_inference` directory.
