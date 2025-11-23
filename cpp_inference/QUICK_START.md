# Quick Start Guide - Windows/MinGW Build

## TL;DR - Just Get It Working

```bash
# 1. Clean start
cd /workspace/cpp_inference
rm -rf build
mkdir build

# 2. Run verification (optional but recommended)
bash verify_cmake_config.sh

# 3. Configure
cd build
cmake -G "MinGW Makefiles" -DCMAKE_BUILD_TYPE=Release ..

# 4. Apply fixes
cd ..
python fix_build_dependencies.py

# 5. Build
cd build
mingw32-make -j4
```

If build fails due to missing dependencies, run steps 4-5 again.

---

## What If CMake Configuration Fails?

### Error: "The filename, directory name, or volume label syntax is incorrect"

**Cause**: Problematic CMAKE_C_FLAGS with max/min macros

**Fix**: 
1. Check you don't have this line in CMakeLists.txt:
   ```cmake
   set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Dmax(a,b)=((a)>(b)?(a):(b)) -Dmin(a,b)=((a)<(b)?(a):(b))")
   ```
   
2. If present, DELETE IT completely

3. Clean and reconfigure:
   ```bash
   rm -rf build
   mkdir build
   cd build
   cmake -G "MinGW Makefiles" -DCMAKE_BUILD_TYPE=Release ..
   ```

**For details, see**: `COMPILER_TEST_FAILURE_DIAGNOSIS.md`

---

## What If Build Fails?

### Error: gemmlowp compilation errors

**Fix**:
```bash
python fix_build_dependencies.py
cd build
mingw32-make -j4
```

### Error: cpuinfo missing max() function

**Fix**:
```bash
python fix_build_dependencies.py
cd build  
mingw32-make -j4
```

---

## Success Indicators

### CMake Configuration Success
```
-- Check for working C compiler: C:/msys64/mingw64/bin/cc.exe - works
-- Configuring done
-- Generating done
```

### Build Success
```
[100%] Built target radar_tagger
[100%] Built target radar_tagger_multioutput
```

### Output Files
- `build/radar_tagger.exe`
- `build/radar_tagger_multioutput.exe`

---

## Still Having Issues?

1. **Read the detailed diagnosis**: `COMPILER_TEST_FAILURE_DIAGNOSIS.md`
2. **Follow step-by-step instructions**: `FIX_INSTRUCTIONS.md`
3. **Check original guide** (with warnings): `MINGW_BUILD_FIX_GUIDE.md`

---

## One-Line Build (if using batch script)

```bash
./build_with_fixes.bat clean
```

---

## File Reference

| File | Purpose |
|------|---------|
| `QUICK_START.md` | This file - get building fast |
| `FIX_INSTRUCTIONS.md` | Step-by-step fix procedures |
| `COMPILER_TEST_FAILURE_DIAGNOSIS.md` | Technical deep-dive |
| `MINGW_BUILD_FIX_GUIDE.md` | Original guide with warnings |
| `verify_cmake_config.sh` | Pre-build verification script |
| `fix_build_dependencies.py` | Patch script for dependencies |
