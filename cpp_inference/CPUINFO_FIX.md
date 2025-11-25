# cpuinfo Build Fix for Windows/MinGW

## Problem

When building TensorFlow Lite on Windows with MinGW, the cpuinfo library fails to compile with the error:

```
error: implicit declaration of function 'max' [-Wimplicit-function-declaration]
```

This occurs in `cpuinfo/src/x86/windows/init.c` at line 130, where the `max` function is used without being defined.

## Root Cause

The cpuinfo library uses the `max()` function in C code, but:
1. C standard library doesn't provide a `max` macro/function by default
2. Windows headers define `max` and `min` macros, but only when `NOMINMAX` is not defined
3. Our build uses `NOMINMAX` to avoid conflicts with C++ `std::max`/`std::min`
4. The cpuinfo source code doesn't define these macros itself

## Solution Applied

The CMakeLists.txt has been updated to automatically patch cpuinfo during the build process. The patch:

1. **Detects Windows/MinGW builds** - Only applies when building on Windows
2. **Injects patch code** into TensorFlow Lite's CMakeLists.txt 
3. **Patches cpuinfo** after it's downloaded but before compilation
4. **Adds max/min macros** as compile definitions:
   ```cmake
   add_compile_definitions(max(a,b)=((a)>(b)?(a):(b)))
   add_compile_definitions(min(a,b)=((a)<(b)?(a):(b)))
   ```

## How to Use

### Method 1: Clean Build (Recommended)

If you've already attempted a build, clean everything first:

```bash
# On Windows
cd cpp_inference
rmdir /s /q build
mkdir build
cd build
cmake -G "MinGW Makefiles" ..
mingw32-make
```

### Method 2: Rebuild from Scratch

```bash
# On Windows - using the provided batch file
cd cpp_inference
rebuild_clean_windows.bat
```

## Verification

After running CMake, you should see these messages:

```
-- MinGW/Windows detected - gemmlowp and cpuinfo patches will be applied
-- Patching TensorFlow Lite CMakeLists.txt for MinGW compatibility (max/min macros)...
-- Injecting cpuinfo patching code into TensorFlow Lite CMakeLists.txt...
-- Patching cpuinfo CMakeLists.txt to add max/min macros for Windows...
-- cpuinfo patched successfully with max/min macros
```

## Alternative Manual Fix

If the automatic patch doesn't work, you can manually patch cpuinfo:

### Option A: Direct Source File Patch

Add these lines at the top of `build/_deps/cpuinfo-src/src/x86/windows/init.c`:

```c
#ifndef max
#define max(a,b) (((a) > (b)) ? (a) : (b))
#endif
#ifndef min
#define min(a,b) (((a) < (b)) ? (a) : (b))
#endif
```

### Option B: CMake Compile Definitions

Add to `build/_deps/cpuinfo-build/CMakeLists.txt` after the `project()` command:

```cmake
if(WIN32 OR MINGW)
  add_compile_definitions(max(a,b)=((a)>(b)?(a):(b)))
  add_compile_definitions(min(a,b)=((a)<(b)?(a):(b)))
endif()
```

## Troubleshooting

### Error persists after applying fix

1. **Delete the build directory completely**: `rmdir /s /q build`
2. **Delete CMake cache in TensorFlow**: `rmdir /s /q build\_deps\tensorflow-build`
3. **Reconfigure from scratch**: Run cmake again

### Patch not being applied

Check if the build directory already has a patched version:
```bash
# In cpp_inference directory
findstr /s "PATCHED_FOR_MINGW_CPUINFO" build\_deps\tensorflow-build\*
```

If found, the patch has been applied. If the build still fails, you may need to manually patch as shown above.

### Different version of cpuinfo

If TensorFlow Lite uses a different version of cpuinfo, the structure might be different. Check:
```bash
dir build\_deps\cpuinfo-src\src\x86\windows\
```

## Technical Details

### Why this approach?

1. **Automatic**: No manual intervention needed
2. **Persistent**: Survives CMake reconfigurations
3. **Safe**: Only applies to Windows/MinGW builds
4. **Minimal**: Doesn't modify source files directly, only CMake configuration

### Patch injection process

1. CMake populates TensorFlow Lite source
2. Our patch modifies TensorFlow Lite's CMakeLists.txt to include cpuinfo patching
3. When TensorFlow Lite is configured, it populates cpuinfo
4. The injected code patches cpuinfo's CMakeLists.txt before it's processed
5. cpuinfo builds with max/min macros defined

## Related Issues

- Similar to gemmlowp `eight_bit_int_gemm` compilation issues
- Part of broader MinGW/Windows compatibility fixes
- Related to NOMINMAX and Windows.h conflicts

## See Also

- `GEMMLOWP_FIX_APPLIED.md` - Similar fix for gemmlowp library
- `MINGW_BUILD_FIX_GUIDE.md` - Comprehensive MinGW build fixes
- `BUILD_FIX_SUMMARY.md` - Overview of all build fixes
