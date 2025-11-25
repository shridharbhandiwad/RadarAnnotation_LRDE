# Quick Fix: CMake Configuration Error on Linux

## ⚠️ Error You're Seeing

```
-- ONNX Runtime: D:/Zoppler Projects/... (Windows path on Linux!)
-- Disabled eight_bit_int_gemm target (method 2)
-- Configuring incomplete, errors occurred!
```

## ✅ Quick Fix (30 seconds)

```bash
cd /workspace/cpp_inference
rm -rf build && mkdir build && cd build
cmake -G "Unix Makefiles" -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=g++ ..
make -j$(nproc)
```

**That's it!** ✨

## What This Does

1. **Removes** old Windows/MinGW build artifacts
2. **Configures** CMake for native Linux build  
3. **Builds** both executables with parallel compilation

## Build Time

- First build: ~5-10 minutes (downloads dependencies)
- Subsequent: ~1-2 minutes

## Expected Output

```
✅ /workspace/cpp_inference/build/radar_tagger (4.0 MB)
✅ /workspace/cpp_inference/build/radar_tagger_multioutput (4.1 MB)
```

## Why This Happens

The build directory had leftover artifacts from a **Windows build attempt**, causing CMake to detect the wrong platform and apply MinGW-specific patches.

**Solution**: Clean build directory = fresh Linux build.

## Alternative: One-Line Build Script

Create this script for easy rebuilding:

```bash
#!/bin/bash
# rebuild_linux.sh
cd "$(dirname "$0")/cpp_inference"
rm -rf build && mkdir build && cd build
cmake -G "Unix Makefiles" -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=g++ ..
make -j$(nproc)
```

Make it executable:
```bash
chmod +x rebuild_linux.sh
./rebuild_linux.sh
```

## Verify Build Success

```bash
# Check executables exist
ls -lh /workspace/cpp_inference/build/radar_tagger*

# Test help output
/workspace/cpp_inference/build/radar_tagger --help
/workspace/cpp_inference/build/radar_tagger_multioutput --help
```

## Notes

- ✅ **No special patches needed** on Linux (only Windows/MinGW needs them)
- ✅ **No `eight_bit_int_gemm` errors** on Linux (GCC handles it fine)  
- ✅ **Uses native Linux libraries** (libonnxruntime.so, not .dll)

## Detailed Documentation

See `/workspace/cpp_inference/BUILD_SUCCESS_LINUX.md` for complete details.

---

**Remember**: When in doubt, clean rebuild! 🧹
