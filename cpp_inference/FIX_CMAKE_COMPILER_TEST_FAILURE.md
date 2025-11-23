# Fix for CMake C Compiler Test Failure on Windows MinGW

## The Problem

You're seeing this error when running CMake configuration:

```
The C compiler "C:/msys64/mingw64/bin/cc.exe" is not able to compile a simple test program.

The filename, directory name, or volume label syntax is incorrect.
```

The actual problem is in the compile command that contains:
```
-Dmax(a,b)=((a)>(b)?(a):(b)) -Dmin(a,b)=((a)<(b)?(a):(b))
```

These flags contain `<` and `>` characters that Windows `cmd.exe` interprets as **shell redirection operators**, not as part of the compiler flags. This breaks the compilation.

## Root Cause

TensorFlow Lite's CMakeLists.txt sets these flags in `CMAKE_C_FLAGS` before calling `project()`, which means they're active during CMake's C compiler detection test. On Windows, this causes the test to fail.

## The Solution - UPDATED (2025-11-23)

The CMakeLists.txt has been updated with **more aggressive patching** that removes these flags in ALL their forms:

- ✅ Lines containing `-Dmax(a,b)=...` or `-Dmin(a,b)=...`
- ✅ `set(CMAKE_C_FLAGS ...)` with these macros
- ✅ `set(CMAKE_CXX_FLAGS ...)` with these macros  
- ✅ `string(APPEND CMAKE_C_FLAGS ...)` with these macros
- ✅ `string(APPEND CMAKE_CXX_FLAGS ...)` with these macros
- ✅ `add_compile_definitions(...)` with these macros
- ✅ `target_compile_definitions(...)` with these macros

## What You Need to Do

### Step 1: Pull the Latest Code

On your Windows machine:

```batch
git pull origin main
```

Or download the updated `CMakeLists.txt` from this repository.

### Step 2: Complete Clean Build

**CRITICAL**: You MUST delete the build directory to remove cached configuration:

```batch
cd cpp_inference
rmdir /s /q build
mkdir build
cd build
```

### Step 3: Configure CMake

```batch
cmake -G "MinGW Makefiles" -DCMAKE_BUILD_TYPE=Release ..
```

**What to expect:**
- TensorFlow Lite will be downloaded (first time: 5-10 minutes)
- You'll see: `Patching TensorFlow Lite CMakeLists.txt for MinGW compatibility...`
- You'll see: `Successfully patched TensorFlow Lite CMakeLists.txt`
- The C compiler test should now **PASS**: `Check for working C compiler: ... - works`

### Step 4: Build

```batch
cmake --build . --config Release
```

Or use make:

```batch
mingw32-make -j4
```

## Verification - Has It Worked?

### ✅ Success Indicators:

1. **During CMake configuration:**
   ```
   -- Check for working C compiler: C:/msys64/mingw64/bin/cc.exe
   -- Check for working C compiler: C:/msys64/mingw64/bin/cc.exe - works
   ```

2. **Patching messages:**
   ```
   -- Patching TensorFlow Lite CMakeLists.txt for MinGW compatibility...
   --   -> Successfully patched TensorFlow Lite CMakeLists.txt
   ```

3. **Configuration completes:**
   ```
   -- Configuring done
   -- Generating done
   ```

### ❌ Still Failing?

If you still see the error, check:

1. **Did you clean the build directory?**
   ```batch
   dir build
   ```
   Should only show basic CMake files, not tensorflow-src yet.

2. **Is CMakeLists.txt updated?**
   ```batch
   findstr "Remove entire lines containing max macro" CMakeLists.txt
   ```
   Should return a match at line ~83.

3. **Check for environment variables:**
   ```batch
   echo %CMAKE_C_FLAGS%
   ```
   Should be empty or not contain max/min macros.

4. **CMake cache pollution:**
   Even after deleting build/, CMake might have a user cache:
   ```batch
   del %LOCALAPPDATA%\CMake\Cache\*
   ```

## Alternative Solution (If Patching Fails)

If the patching still doesn't work, you can manually edit the TensorFlow Lite CMakeLists.txt:

1. After first CMake run (even if it fails), find:
   ```
   build\_deps\tensorflow-src\tensorflow\lite\CMakeLists.txt
   ```

2. Search for lines containing `-Dmax(a,b)=` or `-Dmin(a,b)=`

3. Delete or comment out those entire lines

4. Re-run CMake configuration

## Technical Details

### Why This Happens

The preprocessor flag `-Dmax(a,b)=((a)>(b)?(a):(b))` is meant to define a macro:

```c
#define max(a,b) ((a)>(b)?(a):(b))
```

On Linux/macOS, the shell properly passes this to the compiler. On Windows `cmd.exe`:
- Sees `>` and tries to create output redirection
- Sees `(b)` as the target filename
- Fails with "syntax is incorrect" because `(b)` is not a valid filename

### Why Patching Is Necessary

TensorFlow Lite sets these flags globally in CMAKE_C_FLAGS before calling `project()`. This means:
1. CMake reads the flags
2. CMake runs C compiler test
3. C compiler test fails on Windows
4. Build stops before we can do anything

The only solution is to patch TensorFlow's CMakeLists.txt to remove these flags **before** `project()` is called.

### Why NOMINMAX Is Better

The updated CMakeLists.txt uses:
```cmake
add_compile_definitions(NOMINMAX)
```

This tells Windows headers not to define min/max macros, avoiding conflicts without using shell-breaking syntax.

## What Gets Built

After successful configuration and build, you'll have:
- ✅ `radar_tagger.exe` - Single model inference
- ✅ `radar_tagger_multioutput.exe` - Multi-model inference

All TensorFlow Lite functionality is preserved!

## Need More Help?

1. **Check build directory structure:**
   ```batch
   dir /s /b build\_deps\tensorflow-src\tensorflow\lite\CMakeLists.txt
   ```

2. **Verify patch was applied:**
   ```batch
   findstr "PATCHED_FOR_MINGW_MAX_MIN" build\_deps\tensorflow-src\tensorflow\lite\CMakeLists.txt
   ```

3. **Clean everything and retry:**
   ```batch
   cd cpp_inference
   rmdir /s /q build
   del CMakeCache.txt
   mkdir build
   cd build
   cmake -G "MinGW Makefiles" -DCMAKE_BUILD_TYPE=Release ..
   ```

## Summary

✅ **Updated:** CMakeLists.txt now has comprehensive patching (2025-11-23)  
✅ **Action:** `git pull` + delete build/ + reconfigure  
✅ **Result:** C compiler test should pass  
✅ **Build:** Should complete successfully

The fix is in the code - you just need to use it with a clean build!
