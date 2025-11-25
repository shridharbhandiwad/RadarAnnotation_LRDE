# Complete Windows/MinGW Build Fixes

This document provides a comprehensive overview of all fixes applied to build this project on Windows with MinGW.

## Quick Start

To build on Windows with MinGW:

```batch
cd cpp_inference
rmdir /s /q build
mkdir build
cd build
cmake -G "MinGW Makefiles" ..
mingw32-make
```

All fixes are now **automatically applied** during CMake configuration.

## Issues Fixed

### 1. ✅ cpuinfo max/min Macro Error

**Error:**
```
error: implicit declaration of function 'max' [-Wimplicit-function-declaration]
```

**Location:** `cpuinfo/src/x86/windows/init.c:130`

**Fix Applied:**
- Automatic patch in `CMakeLists.txt` (lines 225-282)
- Adds `max` and `min` macro definitions to cpuinfo
- Backup manual patch: `patch_cpuinfo_manual.bat`
- Documentation: `CPUINFO_FIX.md`

**How it works:**
- CMake injects patching code into TensorFlow Lite's CMakeLists.txt
- When TensorFlow Lite configures cpuinfo, the patch is applied
- Adds compile definitions: `max(a,b)=((a)>(b)?(a):(b))`

---

### 2. ✅ gemmlowp eight_bit_int_gemm Compilation Error

**Error:**
```
error: 'OMPUTE' was not declared in this scope
```

**Location:** `gemmlowp/eight_bit_int_gemm.cc`

**Fix Applied:**
- Automatic patch in `CMakeLists.txt` (lines 172-223)
- Disables problematic `eight_bit_int_gemm` target on MinGW
- Documentation: `GEMMLOWP_FIX_APPLIED.md`, `MINGW_GEMMLOWP_FIX.md`

**How it works:**
- Comments out `add_library(eight_bit_int_gemm)` and related commands
- Target is excluded from MinGW builds completely

---

### 3. ✅ Max/Min Macro Shell Redirection Issue

**Error:**
```
The system cannot find the file specified.
```
(when CMake tries to add max/min macros with `>` and `<` characters)

**Location:** TensorFlow Lite's CMakeLists.txt

**Fix Applied:**
- Automatic patch in `CMakeLists.txt` (lines 74-170)
- Removes problematic max/min macro definitions from TensorFlow Lite
- Uses `NOMINMAX` instead

**How it works:**
- Regex replacement removes lines with `max(a,b)=(a)>(b)?(a):(b)`
- Prevents Windows cmd.exe from interpreting `>` as redirection

---

### 4. ✅ Big Object Files Error

**Error:**
```
too many sections
```

**Location:** Large compilation units

**Fix Applied:**
- Automatic in `CMakeLists.txt` (lines 354-357)
- Adds `-Wa,-mbig-obj` flag for MinGW

**How it works:**
```cmake
if(MINGW OR WIN32)
    target_compile_options(radar_tagger PRIVATE -Wa,-mbig-obj)
    target_compile_options(radar_tagger_multioutput PRIVATE -Wa,-mbig-obj)
endif()
```

---

## Verification

After running `cmake`, you should see these status messages:

```
-- MinGW/Windows detected - gemmlowp and cpuinfo patches will be applied
-- Patching TensorFlow Lite CMakeLists.txt for MinGW compatibility (max/min macros)...
  -> Successfully patched TensorFlow Lite CMakeLists.txt (max/min macros)
-- Injecting gemmlowp patching code into TensorFlow Lite CMakeLists.txt...
  -> Successfully injected gemmlowp patching code
-- Injecting cpuinfo patching code into TensorFlow Lite CMakeLists.txt...
  -> Successfully injected cpuinfo patching code
```

During TensorFlow Lite configuration:

```
-- Patching gemmlowp to disable eight_bit_int_gemm...
  -> gemmlowp patched successfully
-- Patching cpuinfo CMakeLists.txt to add max/min macros for Windows...
  -> cpuinfo patched successfully with max/min macros
```

## Manual Fixes (If Automatic Fails)

### cpuinfo
```batch
patch_cpuinfo_manual.bat
```

### gemmlowp
Edit `build/_deps/gemmlowp-src/CMakeLists.txt` and comment out:
```cmake
# add_library(eight_bit_int_gemm ...)
```

## Troubleshooting

### Clean Build
```batch
cd cpp_inference
rmdir /s /q build
mkdir build
cd build
cmake -G "MinGW Makefiles" ..
mingw32-make
```

### Check if Patches Applied
```batch
findstr /s "PATCHED_FOR_MINGW_CPUINFO" build\_deps\tensorflow-build\*
findstr /s "PATCHED_FOR_MINGW_GEMMLOWP" build\_deps\tensorflow-build\*
findstr /s "PATCHED_MAX_MIN_MACROS" build\_deps\cpuinfo-src\*
```

### Still Getting Errors?

1. **Delete everything:**
   ```batch
   rmdir /s /q build
   rmdir /s /q cmake-build-*
   ```

2. **Verify MinGW installation:**
   ```batch
   gcc --version
   cmake --version
   mingw32-make --version
   ```

3. **Check CMakeLists.txt:**
   - Ensure lines 36-48 detect MinGW correctly
   - Ensure patching code is present (lines 64-282)

4. **Enable verbose output:**
   ```batch
   cmake -G "MinGW Makefiles" .. -DCMAKE_VERBOSE_MAKEFILE=ON
   mingw32-make VERBOSE=1
   ```

## Architecture

### Build Process Flow

```
1. CMakeLists.txt (main) detects Windows/MinGW
2. Patches prepared for TensorFlow Lite
3. TensorFlow Lite source downloaded
4. TensorFlow Lite CMakeLists.txt patched:
   - Remove problematic max/min macros
   - Inject gemmlowp patching code  
   - Inject cpuinfo patching code
5. TensorFlow Lite configured
6. Dependencies (gemmlowp, cpuinfo) downloaded
7. Injected patches execute:
   - Patch gemmlowp CMakeLists.txt
   - Patch cpuinfo CMakeLists.txt
8. Dependencies built with patches applied
9. Main targets built
```

### Patch Injection Points

**Main CMakeLists.txt** → **TensorFlow Lite CMakeLists.txt** → **Dependency CMakeLists.txt**

- Main knows about Windows/MinGW requirements
- Injects patching logic into TensorFlow Lite
- TensorFlow Lite applies patches to dependencies
- Dependencies build with fixes

## Files Reference

### Documentation
- `CPUINFO_FIX.md` - cpuinfo max/min fix details
- `GEMMLOWP_FIX_APPLIED.md` - gemmlowp fix details
- `MINGW_BUILD_FIX_GUIDE.md` - General MinGW guide
- `BUILD_FIX_SUMMARY.md` - Summary of all fixes

### Scripts
- `patch_cpuinfo_manual.bat` - Manual cpuinfo patch
- `cmake/patch_cpuinfo.cmake` - CMake-based cpuinfo patch
- `rebuild_clean_windows.bat` - Clean rebuild script

### CMake
- `CMakeLists.txt` (lines 36-48) - MinGW detection
- `CMakeLists.txt` (lines 64-170) - TensorFlow max/min patch
- `CMakeLists.txt` (lines 172-223) - gemmlowp patch injection
- `CMakeLists.txt` (lines 225-282) - cpuinfo patch injection
- `CMakeLists.txt` (lines 354-357) - big-obj flags

## Platform Support

| Platform | Status | Notes |
|----------|--------|-------|
| Windows MinGW | ✅ Full | All fixes automatically applied |
| Windows MSVC | ⚠️ Partial | Some fixes not needed, may need adjustments |
| Linux | ✅ Full | Patches not applied (not needed) |
| macOS | ✅ Full | Patches not applied (not needed) |

## Contributing

When adding new dependencies or making changes:

1. Test on MinGW to ensure patches still work
2. Check if new patches are needed
3. Follow the existing pattern: inject → patch → build
4. Update this documentation

## History

- **2025-11-25**: Added cpuinfo max/min fix
- **2025-11-xx**: Added gemmlowp eight_bit_int_gemm fix
- **2025-11-xx**: Added TensorFlow max/min macro fix
- **2025-11-xx**: Added big-obj flag fix

## Related Issues

- [tensorflow/tensorflow#issue] - max/min macro issues
- [pytorch/cpuinfo#issue] - Windows build issues
- [google/gemmlowp#issue] - MinGW compatibility

## License

These patches and fixes are provided as-is under the same license as the main project.
