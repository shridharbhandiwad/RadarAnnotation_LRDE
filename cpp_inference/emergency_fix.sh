#!/bin/bash
# Emergency fix script for eight_bit_int_gemm build errors
# Run this if you encounter the "cannot specify '-o' with '-c'" error during build

set -e

echo "========================================"
echo "  EMERGENCY FIX for eight_bit_int_gemm"
echo "========================================"
echo ""

cd "$(dirname "$0")"

echo "Step 1: Patching CMakeLists.txt files..."
python3 patch_gemmlowp_direct.py || python patch_gemmlowp_direct.py
echo ""

echo "Step 2: Patching Makefiles..."
python3 patch_makefile_direct.py || python patch_makefile_direct.py
echo ""

echo "Step 3: Recreating CMake cache..."
cd build
rm -f CMakeCache.txt
cmake -DCMAKE_BUILD_TYPE=Release ..
echo ""

echo "Step 4: Re-patching after CMake regeneration..."
cd ..
python3 patch_makefile_direct.py || python patch_makefile_direct.py
echo ""

echo "========================================"
echo "  Fix applied. Now attempting build..."
echo "========================================"
echo ""

cd build
if cmake --build . --config Release -- -j$(nproc 2>/dev/null || echo 4); then
    echo ""
    echo "========================================"
    echo "  SUCCESS! Build completed."
    echo "========================================"
else
    echo ""
    echo "========================================"
    echo "  Build still failing. Try these steps:"
    echo "========================================"
    echo ""
    echo "1. Completely delete the build directory:"
    echo "   rm -rf build"
    echo ""
    echo "2. Run the clean build script:"
    echo "   ./build_with_gemmlowp_fix.sh clean"
    echo ""
    echo "3. If that fails, check:"
    echo "   - Python 3 is installed"
    echo "   - CMake 3.16+ is installed"
    echo "   - Compiler is properly configured"
    echo ""
    exit 1
fi
