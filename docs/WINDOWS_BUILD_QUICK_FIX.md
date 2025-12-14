# Windows Build - Quick Fix Reference

## 🚀 Quick Start (60 seconds)

```cmd
cd cpp_inference
rebuild_windows.bat
```

**Wait 20-45 minutes on first build** (downloads dependencies)

---

## ✅ Prerequisites Checklist

- [ ] CMake installed (https://cmake.org/download/)
- [ ] MinGW-w64 installed (https://www.mingw-w64.org/)
- [ ] Git installed (https://git-scm.com/download/win)
- [ ] All added to PATH

**Verify:**
```cmd
cmake --version
g++ --version
mingw32-make --version
```

---

## 🔧 Manual Build (if script fails)

```cmd
cd cpp_inference
rmdir /s /q build
mkdir build
cd build
cmake -G "MinGW Makefiles" -DCMAKE_BUILD_TYPE=Release ..
cmake --build . --config Release -- -j%NUMBER_OF_PROCESSORS%
```

---

## 🧪 Test Build

```cmd
cd cpp_inference\build
radar_tagger.exe --help
radar_tagger_multioutput.exe --help
```

---

## 🐛 Common Issues

### "mingw32-make not found"
```cmd
set PATH=C:\mingw64\bin;%PATH%
```

### "cmake not found"
Install from https://cmake.org/download/ and add to PATH

### Build fails - clean rebuild
```cmd
cd cpp_inference
rmdir /s /q build
rebuild_windows.bat
```

### Build takes >1 hour
This is normal for first build (downloads TensorFlow Lite)

---

## 📊 Expected Output

```
cpp_inference/build/
├── radar_tagger.exe              (~4 MB)
└── radar_tagger_multioutput.exe  (~4 MB)
```

---

## 🎯 Next Steps

1. Export models:
   ```cmd
   python convert_model_to_tflite.py
   ```

2. Run inference:
   ```cmd
   radar_tagger.exe --model model.tflite --metadata metadata.json
   ```

---

## 📚 Full Documentation

- **BUILD_SUCCESS_WINDOWS.md** - Comprehensive guide
- **WINDOWS_MINGW_BUILD_FIX.md** - MinGW patch details
- **cpp_inference/README.md** - API documentation

---

## ⏱️ Build Times

| Build Type | Time |
|------------|------|
| First build (clean) | 20-45 min |
| Incremental build | 2-5 min |
| CMake config only | 30-60 sec |

---

**Status:** ✅ Windows build ready  
**Platform:** Windows 10/11  
**Compiler:** MinGW-w64 (GCC)  
**Last Updated:** November 25, 2025
