# Windows Build - Start Here! 🪟

## ✅ Status: Ready for Windows Build

This project has been configured for **Windows 10/11** with **MinGW-w64**.

---

## 🚀 Quick Start (2 minutes)

### 1. Install Prerequisites

**Required Software:**
- [CMake](https://cmake.org/download/) - Build system
- [MinGW-w64](https://www.mingw-w64.org/) or [MSYS2](https://www.msys2.org/) - Compiler
- [Git](https://git-scm.com/download/win) - Version control

**Verify Installation:**
```cmd
cmake --version
g++ --version
mingw32-make --version
```

### 2. Build the Project

```cmd
cd cpp_inference
rebuild_windows.bat
```

**That's it!** The script will:
- ✅ Clean old builds
- ✅ Configure CMake
- ✅ Download dependencies (~550 MB)
- ✅ Apply Windows/MinGW patches
- ✅ Build executables

**Build Time:** 20-45 minutes (first time only)

### 3. Test the Build

```cmd
cd cpp_inference\build
radar_tagger.exe --help
radar_tagger_multioutput.exe --help
```

---

## 📚 Documentation

| File | Purpose | Read if... |
|------|---------|-----------|
| **WINDOWS_BUILD_QUICK_FIX.md** | Quick fixes | Build fails |
| **BUILD_SUCCESS_WINDOWS.md** | Complete guide | Want details |
| **BUILD_FIXED_WINDOWS.txt** | Status overview | Quick reference |
| **CHANGES_FOR_WINDOWS.md** | All changes | Want full changelog |
| **WINDOWS_BUILD_CORRECTION_SUMMARY.md** | Technical summary | Developer |

---

## 🎯 What You Get

After successful build:

```
cpp_inference/build/
├── radar_tagger.exe              (~4 MB)
├── radar_tagger_multioutput.exe  (~4 MB)
└── _deps/
    └── onnxruntime-src/lib/onnxruntime.dll
```

---

## 🐛 Troubleshooting

### "mingw32-make not found"
```cmd
REM Install MinGW-w64 and add to PATH
set PATH=C:\mingw64\bin;%PATH%
```

### "cmake not found"
Download from https://cmake.org/download/  
Select "Add CMake to system PATH" during install

### Build fails
```cmd
cd cpp_inference
rmdir /s /q build
rebuild_windows.bat
```

### Still stuck?
Read **WINDOWS_BUILD_QUICK_FIX.md** for more solutions

---

## 🔑 Key Features

### Automatic Windows/MinGW Patches
The build system automatically:
- ✅ Disables problematic `eight_bit_int_gemm` target
- ✅ Removes min/max macro conflicts
- ✅ Adds `NOMINMAX` definition
- ✅ Adds large object file support
- ✅ Patches TensorFlow Lite for MinGW

**You don't need to do anything!** Patches apply automatically.

---

## 🏗️ Build System

```
CMake detects Windows/MinGW
        ↓
Applies automatic patches (lines 36-291 in CMakeLists.txt)
        ↓
Downloads TensorFlow Lite v2.14.0 (~500 MB)
        ↓
Downloads ONNX Runtime v1.16.3 (~50 MB)
        ↓
Compiles with MinGW GCC
        ↓
Creates radar_tagger.exe and radar_tagger_multioutput.exe
```

---

## 📊 Build Times

| Build Type | Duration |
|------------|----------|
| **First clean build** | 20-45 minutes |
| **Incremental build** | 2-5 minutes |
| **CMake reconfigure** | 30-60 seconds |

---

## ✨ Usage Example

```cmd
REM Export your trained model
python convert_model_to_tflite.py

REM Run inference
cd cpp_inference\build
radar_tagger.exe ^
    --model ..\..\models\model.tflite ^
    --metadata ..\..\models\metadata.json ^
    --test-data ..\..\data\test_data.csv
```

---

## 📦 What Gets Downloaded

During the first build, CMake downloads:

| Package | Version | Size | Type |
|---------|---------|------|------|
| TensorFlow Lite | v2.14.0 | ~500 MB | Built from source |
| ONNX Runtime | v1.16.3 | ~50 MB | Pre-built (Windows x64) |
| nlohmann/json | v3.11.2 | ~500 KB | Header-only |

**Total download:** ~550 MB  
**Once downloaded, subsequent builds are much faster!**

---

## 🔍 What Changed

This correction replaced Linux-focused files with Windows equivalents:

### Added ✅
- `BUILD_FIXED_WINDOWS.txt`
- `WINDOWS_BUILD_QUICK_FIX.md`
- `WINDOWS_BUILD_CORRECTION_SUMMARY.md`
- `cpp_inference/BUILD_SUCCESS_WINDOWS.md`
- `cpp_inference/rebuild_windows.bat`
- `CHANGES_FOR_WINDOWS.md`
- `README_WINDOWS_BUILD.md` (this file)

### Removed ❌
- `BUILD_FIXED.txt` (Linux)
- `CMAKE_ERROR_RESOLUTION_SUMMARY.md` (Linux)
- `LINUX_BUILD_QUICK_FIX.md` (Linux)
- `cpp_inference/BUILD_SUCCESS_LINUX.md` (Linux)
- `cpp_inference/rebuild_linux.sh` (Linux)

### Unchanged 📋
- `cpp_inference/CMakeLists.txt` (already supports Windows!)
- All C++ source files (platform-independent)
- All Python training scripts

---

## 🎓 Learning Resources

- **CMakeLists.txt lines 36-291:** Windows/MinGW patch code
- **TensorFlow Lite:** https://www.tensorflow.org/lite
- **ONNX Runtime:** https://onnxruntime.ai/
- **MinGW-w64:** https://www.mingw-w64.org/

---

## 💡 Pro Tips

1. **Use the build script** - It handles everything automatically
2. **First build is slow** - Downloads dependencies (normal!)
3. **Subsequent builds are fast** - Only recompiles changes
4. **Check prerequisites first** - Save time debugging
5. **Clean rebuild if stuck** - `rmdir /s /q build`

---

## 🛠️ System Requirements

- **OS:** Windows 10 or Windows 11
- **RAM:** 4 GB minimum, 8 GB recommended
- **Disk:** 2 GB free space
- **Network:** Internet connection (for dependencies)
- **Compiler:** MinGW-w64 8.1.0 or later

---

## ✅ Verification Checklist

Before building, verify:

- [ ] CMake installed (version 3.16+)
- [ ] MinGW-w64 installed
- [ ] Git installed
- [ ] All tools in PATH
- [ ] Internet connection available
- [ ] 2 GB free disk space
- [ ] Not using WSL (use native Windows CMD/PowerShell)

---

## 🎯 Next Steps

1. **Install prerequisites** (if not done)
2. **Run `rebuild_windows.bat`**
3. **Wait 20-45 minutes** (grab coffee ☕)
4. **Test with `radar_tagger.exe --help`**
5. **Export your models** with Python
6. **Run inference** on your data

---

## 📞 Support

**Having issues?**

1. Read **WINDOWS_BUILD_QUICK_FIX.md** first
2. Check **BUILD_SUCCESS_WINDOWS.md** for details
3. Verify prerequisites are installed and in PATH
4. Try clean rebuild: `rmdir /s /q build && rebuild_windows.bat`

---

## 📈 Status

**✅ Windows build ready**  
**✅ All patches included**  
**✅ Documentation complete**  
**✅ Build script tested**

**Platform:** Windows 10/11  
**Compiler:** MinGW-w64 (GCC)  
**Build System:** CMake 3.16+  
**Last Updated:** November 25, 2025

---

## 🚀 Ready to Build?

```cmd
cd cpp_inference
rebuild_windows.bat
```

**Good luck! 🎉**
