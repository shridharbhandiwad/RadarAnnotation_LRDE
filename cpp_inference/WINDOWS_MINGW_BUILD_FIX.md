# Windows MinGW Build Fix for gemmlowp Issue

## The Problem

When building this project on Windows with MinGW, you may encounter this error:

```
c++.exe: fatal error: cannot specify '-o' with '-c', '-S' or '-E' with multiple files
compilation terminated.
mingw32-make[2]: *** [_deps\gemmlowp-build\CMakeFiles\eight_bit_int_gemm.dir\build.make:78: ...] Error 1
```

This occurs because:
1. TensorFlow Lite depends on the `gemmlowp` library
2. gemmlowp includes a test/benchmark target called `eight_bit_int_gemm`
3. This target has CMake configuration issues that cause malformed compiler commands on MinGW
4. The error occurs during the build phase, causing the entire build to fail

## The Solution

We've implemented **multiple layers of fixes** to resolve this issue:

### Fix #1: CMakeLists.txt Exclusion (Automatic)

The main `CMakeLists.txt` now automatically excludes the `eight_bit_int_gemm` target from the build if it exists:

```cmake
if((MINGW OR WIN32) AND TARGET eight_bit_int_gemm)
    message(STATUS "Excluding eight_bit_int_gemm target from MinGW build")
    set_target_properties(eight_bit_int_gemm PROPERTIES 
        EXCLUDE_FROM_ALL TRUE
        EXCLUDE_FROM_DEFAULT_BUILD TRUE
    )
endif()
```

**However**, this only works if the target is created successfully. If the target has configuration errors, this fix won't help.

### Fix #2: Build Script with Patching (Recommended)

Use the `build_mingw_fixed.bat` script which:
1. Cleans the build directory
2. Runs CMake configuration
3. Detects if gemmlowp was downloaded
4. Patches the gemmlowp CMakeLists.txt to disable eight_bit_int_gemm
5. Reconfigures CMake
6. Builds the project

**Usage:**
```batch
cd cpp_inference
build_mingw_fixed.bat
```

### Fix #3: Manual Patch (If Script Fails)

If the automated script doesn't work, follow these manual steps:

1. **Clean and configure:**
   ```batch
   cd cpp_inference
   rmdir /s /q build
   mkdir build
   cd build
   cmake .. -G "MinGW Makefiles" -DCMAKE_BUILD_TYPE=Release
   ```

2. **Check if gemmlowp was downloaded:**
   ```batch
   dir gemmlowp\CMakeLists.txt
   ```

3. **If gemmlowp exists, patch it:**
   
   Open `build\gemmlowp\CMakeLists.txt` in a text editor and:
   
   - Find any line containing `add_library(eight_bit_int_gemm`
   - Comment it out by adding `#` at the beginning
   - Find any line containing `add_executable(eight_bit_int_gemm`
   - Comment it out by adding `#` at the beginning
   - Find any line containing `target_link_libraries(eight_bit_int_gemm`
   - Comment it out by adding `#` at the beginning

4. **Reconfigure and build:**
   ```batch
   cmake .. -G "MinGW Makefiles" -DCMAKE_BUILD_TYPE=Release
   cmake --build . --config Release
   ```

## Alternative: Use Pre-built TensorFlow Lite

If the above solutions don't work, you can use a pre-built TensorFlow Lite library:

1. Download pre-built TensorFlow Lite for Windows from the official releases
2. Extract it to a known location
3. Configure CMake to use it:
   ```batch
   cmake .. -G "MinGW Makefiles" ^
        -DUSE_SYSTEM_TFLITE=ON ^
        -DTensorFlowLite_DIR=C:\path\to\tflite
   ```

## Why Does This Happen?

The `eight_bit_int_gemm` target is a **test/benchmark program** that:
- Is NOT required for TensorFlow Lite functionality
- Has known issues with MinGW due to how it specifies compiler flags
- Gets included automatically when building TensorFlow Lite from source

The specific error occurs because:
1. The generated `build.make` file has malformed compiler commands
2. MinGW's g++ is being invoked with conflicting flags
3. The `-o` flag is used incorrectly with multiple input files

## Verification

After a successful build, you should see:
```
Built executables:
radar_tagger.exe
radar_tagger_multioutput.exe
```

Both executables are **fully functional** - the excluded `eight_bit_int_gemm` target is not needed for inference.

## Known Warnings (Safe to Ignore)

During the build, you may see these warnings - they are normal:
```
warning: ignoring '#pragma comment ' [-Wunknown-pragmas]
warning: ignoring '#pragma warning ' [-Wunknown-pragmas]
warning: 'HAS_STRPTIME' is not defined, evaluates to '0' [-Wundef]
warning: cast between incompatible function types [-Wcast-function-type]
warning: unknown conversion type character 'z' in format [-Wformat=]
```

These are compatibility warnings between MSVC and GCC and do not affect functionality.

## Troubleshooting

### Issue: "gemmlowp directory not found"
**Solution:** Run the build script twice. The first run downloads dependencies, the second applies patches.

### Issue: "eight_bit_int_gemm target not found"
**Solution:** This is actually good news - the target wasn't created, so there's nothing to exclude. Your build should succeed.

### Issue: Build still fails after patching
**Solution:** 
1. Completely delete the build directory
2. Delete CMake cache: `del CMakeCache.txt`
3. Ensure you have MinGW in PATH: `where g++`
4. Try building with single thread: `cmake --build . --config Release -j 1`

### Issue: Python script fails in batch file
**Solution:** Manually edit `build\gemmlowp\CMakeLists.txt` as described in Fix #3

## Technical Details

The error occurs in the generated `build.make` file at around line 78:

```makefile
_deps/gemmlowp-build/CMakeFiles/eight_bit_int_gemm.dir/__/eight_bit_int_gemm/eight_bit_int_gemm.cc.o: ...
    @$(CMAKE_COMMAND) -E cmake_echo_color ...
    cd /workspace/cpp_inference/build/_deps/gemmlowp-build && /usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT ...
```

The `$(CXX_FLAGS)` or similar variables contain malformed values causing the "multiple files" error.

## Date Updated
2025-11-23

## Related Files
- `CMakeLists.txt` - Main build configuration with automatic exclusion
- `build_mingw_fixed.bat` - Automated build script with patching
- `cmake/patch_gemmlowp.cmake` - Standalone patch module (for reference)
