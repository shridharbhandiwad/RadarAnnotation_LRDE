# Windows Build Quick Start

## One-Command Build (Recommended)

```batch
cd cpp_inference
rmdir /s /q build 2>nul & mkdir build & cd build & cmake -G "MinGW Makefiles" .. & mingw32-make
```

## Step-by-Step Build

```batch
cd cpp_inference
rmdir /s /q build
mkdir build
cd build
cmake -G "MinGW Makefiles" ..
mingw32-make
```

## What You Should See

During CMake configuration:
```
-- MinGW/Windows detected - gemmlowp and cpuinfo patches will be applied
-- Patching TensorFlow Lite CMakeLists.txt...
-- Injecting gemmlowp patching code...
-- Injecting cpuinfo patching code...
```

Build progress:
```
[  1%] Building CXX object ...
[ 29%] Building C object ...  (← cpuinfo should build without errors)
[100%] Built target radar_tagger
```

## If Build Fails

### Error: "implicit declaration of function 'max'"

**Quick Fix:**
```batch
patch_cpuinfo_manual.bat
cd build
mingw32-make
```

**Details:** See `CPUINFO_FIX.md`

### Error: "COMPUTE was not declared"

**Quick Fix:**
```batch
# Clean and rebuild
rmdir /s /q build
mkdir build
cd build
cmake -G "MinGW Makefiles" ..
mingw32-make
```

**Details:** See `GEMMLOWP_FIX_APPLIED.md`

### Error: "too many sections"

Already fixed automatically. If you still see this, add to CMakeLists.txt:
```cmake
target_compile_options(your_target PRIVATE -Wa,-mbig-obj)
```

## Prerequisites

Make sure you have:
- ✅ MinGW-w64 (gcc, g++, mingw32-make)
- ✅ CMake 3.16 or higher
- ✅ Git (for downloading dependencies)

Check versions:
```batch
gcc --version
cmake --version
mingw32-make --version
```

## Output

If successful, you'll have:
- `build/radar_tagger.exe`
- `build/radar_tagger_multioutput.exe`

## Testing

```batch
cd build
radar_tagger.exe --help
```

## Complete Documentation

- `WINDOWS_BUILD_FIXES_COMPLETE.md` - All fixes explained
- `CPUINFO_FIX.md` - cpuinfo max/min issue
- `GEMMLOWP_FIX_APPLIED.md` - gemmlowp issue
- `BUILD_FIX_SUMMARY.md` - Overview

## Still Having Issues?

1. **Clean everything:**
   ```batch
   rmdir /s /q build
   ```

2. **Check your environment:**
   ```batch
   where gcc
   where cmake
   where mingw32-make
   ```

3. **Try verbose build:**
   ```batch
   cmake -G "MinGW Makefiles" .. -DCMAKE_VERBOSE_MAKEFILE=ON
   mingw32-make VERBOSE=1
   ```

4. **Check the detailed guides** in the docs listed above

## Success Indicators

✅ CMake finishes without errors  
✅ You see patch messages during configuration  
✅ Build reaches 100%  
✅ `radar_tagger.exe` exists in `build/`  

## Support

If you encounter issues not covered here:
1. Check `WINDOWS_BUILD_FIXES_COMPLETE.md`
2. Look at error message carefully
3. Search for error in existing documentation files
4. Try a clean rebuild

---

**Last Updated:** 2025-11-25  
**Tested With:** MinGW-w64 GCC 13.x, CMake 3.28.x
