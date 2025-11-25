# Windows Build Documentation - Complete Index

**Last Updated:** November 25, 2025

This is the complete index of all Windows build documentation, fixes, and resources for the Radar Tagger C++ project.

---

## 🚨 Having Build Issues? START HERE

### Quick Diagnosis

**Your error looks like this:**
```
-- Configuring incomplete, errors occurred!
```
→ **See:** [`WINDOWS_CMAKE_VERSION_FIX.md`](WINDOWS_CMAKE_VERSION_FIX.md)

**Your error looks like this:**
```
c++.exe: fatal error: cannot specify '-o' with '-c', '-S' or '-E'
```
→ **See:** [`WINDOWS_MINGW_BUILD_FIX.md`](WINDOWS_MINGW_BUILD_FIX.md)

**Not sure what's wrong?**
→ **Run:** `check_build_system.bat` to diagnose

---

## 📚 Documentation by Category

### 🎯 Quick Start (Read First!)

| File | Purpose | Time | Priority |
|------|---------|------|----------|
| [`START_HERE.md`](START_HERE.md) | **Main entry point** - Overview of all fixes | 2 min | ⭐⭐⭐ |
| [`WINDOWS_BUILD_QUICK_FIX.txt`](WINDOWS_BUILD_QUICK_FIX.txt) | Text-only quick reference | 1 min | ⭐⭐⭐ |
| [`check_build_system.bat`](check_build_system.bat) | **Run this!** System verification script | 1 min | ⭐⭐⭐ |

### 🔧 Build Scripts (Just Run These!)

| Script | Purpose | When to Use |
|--------|---------|-------------|
| [`check_build_system.bat`](check_build_system.bat) | Check CMake, compiler, Python | Before building |
| [`build_with_gemmlowp_fix.bat`](build_with_gemmlowp_fix.bat) | **Recommended** - Full build with all fixes | Main build method |
| [`emergency_fix.bat`](emergency_fix.bat) | Quick patch if build fails mid-way | When build fails |
| [`rebuild_clean_windows.bat`](rebuild_clean_windows.bat) | Clean and rebuild from scratch | After failed builds |

### 🐛 Problem-Specific Guides

| Issue | Documentation | Description |
|-------|--------------|-------------|
| CMake configuration fails | [`WINDOWS_CMAKE_VERSION_FIX.md`](WINDOWS_CMAKE_VERSION_FIX.md) | CMake 3.16-3.18 compatibility |
| eight_bit_int_gemm error | [`WINDOWS_MINGW_BUILD_FIX.md`](WINDOWS_MINGW_BUILD_FIX.md) | gemmlowp compilation issue |
| General build problems | [`README_BUILD_FIX.md`](README_BUILD_FIX.md) | Comprehensive overview |
| Quick manual fix | [`QUICK_BUILD_INSTRUCTIONS.txt`](QUICK_BUILD_INSTRUCTIONS.txt) | Step-by-step commands |

### 📖 Technical Documentation

| File | Purpose | Audience |
|------|---------|----------|
| [`CMAKE_VERSION_FIX_SUMMARY.md`](CMAKE_VERSION_FIX_SUMMARY.md) | Technical details of CMake fix | Developers |
| [`GEMMLOWP_FINAL_FIX.md`](GEMMLOWP_FINAL_FIX.md) | Complete gemmlowp fix explanation | Developers |
| [`FIX_SUMMARY.md`](FIX_SUMMARY.md) | Summary of all applied fixes | Developers |
| [`BUILD_STATUS.txt`](BUILD_STATUS.txt) | Build system status | Reference |

### 📝 General Information

| File | Purpose |
|------|---------|
| [`README.md`](README.md) | Main C++ project documentation |
| [`QUICK_START.md`](QUICK_START.md) | Getting started guide |

---

## 🎯 Common Scenarios

### Scenario 1: First Time Building on Windows

**Steps:**
1. Run `check_build_system.bat` to verify setup
2. Run `build_with_gemmlowp_fix.bat clean`
3. Wait 10-20 minutes (downloads dependencies)
4. Test with `build\radar_tagger.exe --help`

**Documentation:**
- [`START_HERE.md`](START_HERE.md)
- [`WINDOWS_BUILD_QUICK_FIX.txt`](WINDOWS_BUILD_QUICK_FIX.txt)

### Scenario 2: Build Failed with CMake Error

**Error Message:**
```
-- Configuring incomplete, errors occurred!
```

**Steps:**
1. Check CMake version: `cmake --version` (must be 3.16+)
2. If < 3.16: Upgrade CMake
3. If >= 3.16: Read [`WINDOWS_CMAKE_VERSION_FIX.md`](WINDOWS_CMAKE_VERSION_FIX.md)
4. Clean and rebuild: `build_with_gemmlowp_fix.bat clean`

**Documentation:**
- [`WINDOWS_CMAKE_VERSION_FIX.md`](WINDOWS_CMAKE_VERSION_FIX.md)
- [`CMAKE_VERSION_FIX_SUMMARY.md`](CMAKE_VERSION_FIX_SUMMARY.md)

### Scenario 3: Build Failed with Compilation Error

**Error Message:**
```
c++.exe: fatal error: cannot specify '-o' with '-c'
eight_bit_int_gemm compilation error
```

**Steps:**
1. Run `emergency_fix.bat` (if mid-build)
2. OR clean rebuild: `build_with_gemmlowp_fix.bat clean`
3. If still fails: Manual patch (see docs)

**Documentation:**
- [`WINDOWS_MINGW_BUILD_FIX.md`](WINDOWS_MINGW_BUILD_FIX.md)
- [`GEMMLOWP_FINAL_FIX.md`](GEMMLOWP_FINAL_FIX.md)
- [`QUICK_BUILD_INSTRUCTIONS.txt`](QUICK_BUILD_INSTRUCTIONS.txt)

### Scenario 4: Build Succeeds but Executables Missing

**Steps:**
1. Check all possible locations:
   - `build\radar_tagger.exe`
   - `build\Release\radar_tagger.exe`
   - `build\Debug\radar_tagger.exe`
2. Check build output for errors
3. Rebuild with verbose: `cmake --build . --config Release --verbose`

**Documentation:**
- [`README_BUILD_FIX.md`](README_BUILD_FIX.md)
- [`WINDOWS_MINGW_BUILD_FIX.md`](WINDOWS_MINGW_BUILD_FIX.md)

### Scenario 5: Need to Understand What Was Fixed

**Documentation to Read:**
- [`CMAKE_VERSION_FIX_SUMMARY.md`](CMAKE_VERSION_FIX_SUMMARY.md) - CMake compatibility
- [`GEMMLOWP_FINAL_FIX.md`](GEMMLOWP_FINAL_FIX.md) - gemmlowp issue
- [`FIX_SUMMARY.md`](FIX_SUMMARY.md) - All fixes overview

---

## 🛠️ System Requirements

### Minimum Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **CMake** | 3.16 | 3.20+ |
| **Compiler** | MinGW GCC 7+ or MSVC 2019+ | MinGW GCC 11+ or MSVC 2022+ |
| **Python** | 3.6+ (optional) | 3.8+ |
| **Disk Space** | 2 GB | 4 GB |
| **RAM** | 4 GB | 8 GB |
| **Internet** | Required for first build | - |

### Checking Your System

```batch
# Check CMake version
cmake --version

# Check compiler (MinGW)
g++ --version

# Check compiler (MSVC) - in VS Developer Command Prompt
cl

# Check Python (optional)
python --version
```

Or simply run: `check_build_system.bat`

---

## 📊 Fix Status Summary

| Issue | Status | CMake Support | Solution |
|-------|--------|---------------|----------|
| CMake < 3.16 | ❌ Not supported | - | Upgrade CMake |
| CMake 3.16-3.18 compatibility | ✅ Fixed | 3.16-3.18 | Version guard added |
| CMake 3.19+ | ✅ Working | 3.19+ | Full support |
| gemmlowp eight_bit_int_gemm | ✅ Fixed | All | Multi-layer patching |
| MinGW build issues | ✅ Fixed | All | Automated patching |
| MSVC build issues | ✅ Working | All | Native support |

---

## 🔗 Quick Links

### Must-Read Documents
1. [`START_HERE.md`](START_HERE.md) ← **Start here!**
2. [`WINDOWS_BUILD_QUICK_FIX.txt`](WINDOWS_BUILD_QUICK_FIX.txt) ← Quick reference
3. [`check_build_system.bat`](check_build_system.bat) ← Run this script

### Problem-Specific
- CMake issues: [`WINDOWS_CMAKE_VERSION_FIX.md`](WINDOWS_CMAKE_VERSION_FIX.md)
- Build errors: [`WINDOWS_MINGW_BUILD_FIX.md`](WINDOWS_MINGW_BUILD_FIX.md)
- Manual steps: [`QUICK_BUILD_INSTRUCTIONS.txt`](QUICK_BUILD_INSTRUCTIONS.txt)

### Technical Details
- CMake fix: [`CMAKE_VERSION_FIX_SUMMARY.md`](CMAKE_VERSION_FIX_SUMMARY.md)
- gemmlowp fix: [`GEMMLOWP_FINAL_FIX.md`](GEMMLOWP_FINAL_FIX.md)
- All fixes: [`FIX_SUMMARY.md`](FIX_SUMMARY.md)

### Build Scripts (Recommended Order)
1. [`check_build_system.bat`](check_build_system.bat) ← Check your system
2. [`build_with_gemmlowp_fix.bat`](build_with_gemmlowp_fix.bat) ← Build project
3. [`emergency_fix.bat`](emergency_fix.bat) ← If build fails mid-way

---

## 💡 Tips and Best Practices

### Before Building
✅ Run `check_build_system.bat` to verify your setup  
✅ Ensure you have at least 2 GB free disk space  
✅ Close other applications to free RAM  
✅ Ensure stable internet connection (first build downloads ~2 GB)

### During Build
✅ Don't interrupt the first CMake configuration (downloads dependencies)  
✅ Build can take 10-20 minutes on first run  
✅ Warnings are normal - focus on errors  
✅ Subsequent builds are much faster (1-2 minutes)

### If Build Fails
✅ Read the actual error message (scroll up in terminal)  
✅ Check the documentation index for your specific error  
✅ Try `emergency_fix.bat` if mid-build  
✅ Try clean rebuild: `build_with_gemmlowp_fix.bat clean`  
✅ Check CMake and compiler versions

### After Successful Build
✅ Test executables: `build\radar_tagger.exe --help`  
✅ Keep the build directory for faster rebuilds  
✅ Only clean if you need to troubleshoot

---

## 🎓 Understanding the Fixes

### CMake Version Fix (November 2025)

**Problem:** Used `cmake_language(DEFER ...)` which requires CMake 3.19+, but minimum was 3.16.

**Solution:** Added version check to use DEFER only on 3.19+, graceful fallback for older versions.

**Impact:** Users with CMake 3.16-3.18 can now build successfully.

**Details:** [`CMAKE_VERSION_FIX_SUMMARY.md`](CMAKE_VERSION_FIX_SUMMARY.md)

### gemmlowp Fix

**Problem:** TensorFlow Lite dependency (gemmlowp) has a test target that fails on MinGW.

**Solution:** Multi-layer fix:
1. Direct target exclusion (all CMake versions)
2. Deferred fix (CMake 3.19+ only)
3. Source-level patching (all versions)
4. Build script automation

**Impact:** eight_bit_int_gemm target is properly excluded/patched.

**Details:** [`GEMMLOWP_FINAL_FIX.md`](GEMMLOWP_FINAL_FIX.md)

---

## 📞 Getting Help

### Self-Service (Try These First)

1. **Run diagnostics:** `check_build_system.bat`
2. **Read quick fix:** [`WINDOWS_BUILD_QUICK_FIX.txt`](WINDOWS_BUILD_QUICK_FIX.txt)
3. **Find your error:** Use this index to locate relevant documentation
4. **Try emergency fix:** `emergency_fix.bat`
5. **Clean rebuild:** `build_with_gemmlowp_fix.bat clean`

### Reporting Issues

If you still have problems, provide:
- Complete error output (full terminal text)
- CMake version: `cmake --version`
- Compiler version: `g++ --version` or `cl`
- Python version: `python --version`
- Output of `check_build_system.bat`

---

## 📅 Version History

| Date | Change | Impact |
|------|--------|--------|
| Nov 25, 2025 | CMake 3.16 compatibility fix | Supports older CMake versions |
| Nov 23, 2025 | gemmlowp fix implementation | Fixes MinGW build errors |
| Earlier | Initial build system | - |

---

## ✅ Current Status

**Build System:** ✅ Fully functional on Windows  
**CMake Support:** ✅ 3.16+ (3.20+ recommended)  
**Compiler Support:** ✅ MinGW GCC 7+ and MSVC 2019+  
**Documentation:** ✅ Complete and up-to-date  
**Automation:** ✅ Build scripts handle everything  

**Recommended Build Command:**
```batch
cd cpp_inference
build_with_gemmlowp_fix.bat clean
```

---

**This index will be updated as new issues are discovered and fixed.**
