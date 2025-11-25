# Windows Build Documentation Index

Quick navigation for all Windows/MinGW build fixes and documentation.

## 🚀 Quick Start (Start Here!)

| Document | Purpose | When to Use |
|----------|---------|-------------|
| **[WINDOWS_BUILD_QUICKSTART.md](WINDOWS_BUILD_QUICKSTART.md)** | Quick build instructions | Your first build, or need quick commands |
| **[README.md](README.md)** | Main project documentation | Overview of the entire project |

## 🔧 Build Fixes (Automatic)

All these fixes are **automatically applied** during CMake configuration. No manual steps needed!

| Issue | Document | Status |
|-------|----------|--------|
| **cpuinfo max/min error** | [CPUINFO_FIX.md](CPUINFO_FIX.md) | ✅ Auto-fixed |
| **gemmlowp compilation** | [GEMMLOWP_FIX_APPLIED.md](GEMMLOWP_FIX_APPLIED.md) | ✅ Auto-fixed |
| **Max/min macro shell error** | [WINDOWS_BUILD_FIXES_COMPLETE.md](WINDOWS_BUILD_FIXES_COMPLETE.md) | ✅ Auto-fixed |
| **Big object files** | [WINDOWS_BUILD_FIXES_COMPLETE.md](WINDOWS_BUILD_FIXES_COMPLETE.md) | ✅ Auto-fixed |

## 📚 Detailed Documentation

### Comprehensive Guides

| Document | Description | Read Time |
|----------|-------------|-----------|
| **[WINDOWS_BUILD_FIXES_COMPLETE.md](WINDOWS_BUILD_FIXES_COMPLETE.md)** | Complete guide to all Windows fixes | 10 min |
| **[MINGW_BUILD_FIX_GUIDE.md](MINGW_BUILD_FIX_GUIDE.md)** | General MinGW compatibility guide | 8 min |
| **[BUILD_FIX_SUMMARY.md](BUILD_FIX_SUMMARY.md)** | Overview of build system fixes | 5 min |

### Specific Issue Documentation

| Document | Issue | Details |
|----------|-------|---------|
| **[CPUINFO_FIX.md](CPUINFO_FIX.md)** | cpuinfo max/min | Problem, solution, alternatives |
| **[CPUINFO_FIX_SUMMARY.txt](CPUINFO_FIX_SUMMARY.txt)** | cpuinfo max/min | Technical implementation details |
| **[GEMMLOWP_FIX_APPLIED.md](GEMMLOWP_FIX_APPLIED.md)** | gemmlowp eight_bit_int_gemm | Problem, solution, verification |
| **[MINGW_GEMMLOWP_FIX.md](MINGW_GEMMLOWP_FIX.md)** | gemmlowp details | Detailed technical explanation |
| **[EIGHT_BIT_INT_GEMM_FIX.md](EIGHT_BIT_INT_GEMM_FIX.md)** | gemmlowp alternative | Alternative fix approach |

## 🛠️ Manual Fix Scripts

Use these if automatic fixing doesn't work (rare):

| Script | Purpose | When to Use |
|--------|---------|-------------|
| **[patch_cpuinfo_manual.bat](patch_cpuinfo_manual.bat)** | Manually patch cpuinfo | If automatic cpuinfo patch fails |
| **[cmake/patch_cpuinfo.cmake](cmake/patch_cpuinfo.cmake)** | CMake-based cpuinfo patch | Alternative manual patch method |
| **fix_and_build_windows.bat** | General fix and build | Older all-in-one script |

## 📖 Reference Documentation

### Build System

| Document | Content |
|----------|---------|
| **[CMAKE_IF_MACRO_FIX.md](CMAKE_IF_MACRO_FIX.md)** | CMake macro fixing |
| **[FIX_CMAKE_COMPILER_TEST_FAILURE.md](FIX_CMAKE_COMPILER_TEST_FAILURE.md)** | Compiler test issues |
| **[COMPILER_TEST_FAILURE_DIAGNOSIS.md](COMPILER_TEST_FAILURE_DIAGNOSIS.md)** | Diagnosing compiler tests |

### Change Logs

| Document | Content |
|----------|---------|
| **[FIX_CHANGELOG.txt](FIX_CHANGELOG.txt)** | History of all fixes |
| **[BUILD_STATUS_UPDATED.md](BUILD_STATUS_UPDATED.md)** | Build status updates |
| **[BUILD_STATUS.txt](BUILD_STATUS.txt)** | Current build status |

## 🎯 Common Scenarios

### Scenario 1: First Time Building
1. Read: [WINDOWS_BUILD_QUICKSTART.md](WINDOWS_BUILD_QUICKSTART.md)
2. Run the build commands
3. If it works: ✅ Done!
4. If it fails: Check the error and go to Scenario 2

### Scenario 2: Error with "max was not declared"
1. Check: [CPUINFO_FIX.md](CPUINFO_FIX.md)
2. Try: Clean rebuild (delete build directory)
3. If still fails: Run [patch_cpuinfo_manual.bat](patch_cpuinfo_manual.bat)
4. See: [CPUINFO_FIX_SUMMARY.txt](CPUINFO_FIX_SUMMARY.txt) for details

### Scenario 3: Error with "COMPUTE was not declared"
1. Check: [GEMMLOWP_FIX_APPLIED.md](GEMMLOWP_FIX_APPLIED.md)
2. Try: Clean rebuild (delete build directory)
3. See: [MINGW_GEMMLOWP_FIX.md](MINGW_GEMMLOWP_FIX.md) for technical details

### Scenario 4: Understanding the Build System
1. Start: [WINDOWS_BUILD_FIXES_COMPLETE.md](WINDOWS_BUILD_FIXES_COMPLETE.md)
2. Read: Architecture section
3. See: [BUILD_FIX_SUMMARY.md](BUILD_FIX_SUMMARY.md)
4. Deep dive: [CPUINFO_FIX_SUMMARY.txt](CPUINFO_FIX_SUMMARY.txt)

### Scenario 5: Contributing or Modifying
1. Read: [WINDOWS_BUILD_FIXES_COMPLETE.md](WINDOWS_BUILD_FIXES_COMPLETE.md)
2. Understand: Patch injection architecture
3. Review: CMakeLists.txt lines 36-282
4. Test: On clean Windows/MinGW environment

## 🔍 Finding Information

### By Error Message

| Error Contains | Check Document |
|----------------|----------------|
| `implicit declaration of function 'max'` | [CPUINFO_FIX.md](CPUINFO_FIX.md) |
| `'COMPUTE' was not declared` | [GEMMLOWP_FIX_APPLIED.md](GEMMLOWP_FIX_APPLIED.md) |
| `eight_bit_int_gemm` | [EIGHT_BIT_INT_GEMM_FIX.md](EIGHT_BIT_INT_GEMM_FIX.md) |
| `too many sections` | [WINDOWS_BUILD_FIXES_COMPLETE.md](WINDOWS_BUILD_FIXES_COMPLETE.md) |
| `The system cannot find the file` | [FIX_CMAKE_COMPILER_TEST_FAILURE.md](FIX_CMAKE_COMPILER_TEST_FAILURE.md) |

### By Component

| Component | Document |
|-----------|----------|
| **cpuinfo** | [CPUINFO_FIX.md](CPUINFO_FIX.md), [CPUINFO_FIX_SUMMARY.txt](CPUINFO_FIX_SUMMARY.txt) |
| **gemmlowp** | [GEMMLOWP_FIX_APPLIED.md](GEMMLOWP_FIX_APPLIED.md), [MINGW_GEMMLOWP_FIX.md](MINGW_GEMMLOWP_FIX.md) |
| **TensorFlow Lite** | [WINDOWS_BUILD_FIXES_COMPLETE.md](WINDOWS_BUILD_FIXES_COMPLETE.md) |
| **CMake** | [CMAKE_IF_MACRO_FIX.md](CMAKE_IF_MACRO_FIX.md) |

### By Skill Level

| Level | Recommended Reading |
|-------|---------------------|
| **Beginner** | [WINDOWS_BUILD_QUICKSTART.md](WINDOWS_BUILD_QUICKSTART.md) |
| **Intermediate** | [WINDOWS_BUILD_FIXES_COMPLETE.md](WINDOWS_BUILD_FIXES_COMPLETE.md), [CPUINFO_FIX.md](CPUINFO_FIX.md) |
| **Advanced** | [CPUINFO_FIX_SUMMARY.txt](CPUINFO_FIX_SUMMARY.txt), CMakeLists.txt source |

## 📊 Documentation Stats

- **Total Documentation Files**: 15+ Windows-specific docs
- **Quick Start Guides**: 2
- **Detailed Fixes**: 4 major issues
- **Manual Scripts**: 3 batch/cmake files
- **Troubleshooting Guides**: 5+

## ✅ Verification Checklist

After reading docs and building:

- [ ] Read [WINDOWS_BUILD_QUICKSTART.md](WINDOWS_BUILD_QUICKSTART.md)
- [ ] Ran clean build (deleted build directory)
- [ ] CMake completed without errors
- [ ] Saw patch messages during cmake
- [ ] Build completed to 100%
- [ ] `radar_tagger.exe` exists in `build/`
- [ ] Tested executable runs

## 🆘 Still Need Help?

1. **Check error message** against "By Error Message" table above
2. **Try clean rebuild**: `rmdir /s /q build && mkdir build && cd build && cmake -G "MinGW Makefiles" .. && mingw32-make`
3. **Read relevant doc** from the tables above
4. **Check CMake version**: `cmake --version` (need 3.16+)
5. **Check GCC version**: `gcc --version` (need 7.0+)
6. **Enable verbose**: Add `CMAKE_VERBOSE_MAKEFILE=ON` and `VERBOSE=1`

## 📝 Document Updates

This index was last updated: **2025-11-25**

New documents added:
- CPUINFO_FIX.md
- CPUINFO_FIX_SUMMARY.txt
- WINDOWS_BUILD_FIXES_COMPLETE.md
- WINDOWS_BUILD_QUICKSTART.md
- WINDOWS_BUILD_INDEX.md (this file)
- cmake/patch_cpuinfo.cmake
- patch_cpuinfo_manual.bat

## 🔗 External Resources

- [TensorFlow Lite C++ Guide](https://www.tensorflow.org/lite/guide/inference)
- [CMake Documentation](https://cmake.org/documentation/)
- [MinGW-w64](https://www.mingw-w64.org/)
- [cpuinfo GitHub](https://github.com/pytorch/cpuinfo)
- [gemmlowp GitHub](https://github.com/google/gemmlowp)

---

**Navigation Tip**: Press `Ctrl+F` in your editor to search this index for keywords like "cpuinfo", "gemmlowp", "error", "patch", etc.
