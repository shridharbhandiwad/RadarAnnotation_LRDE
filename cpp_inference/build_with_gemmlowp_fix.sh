#!/bin/bash
# Build script for Linux/MinGW with aggressive gemmlowp fixing
# This script ensures gemmlowp is properly patched before building

set -e

echo "================================================"
echo "  Radar Tagger C++ Build (with gemmlowp fix)"
echo "================================================"
echo ""

cd "$(dirname "$0")"

# Step 1: Clean build if requested
if [ "$1" = "clean" ]; then
    echo "Cleaning build directory..."
    rm -rf build
    mkdir -p build
    echo "Done."
    echo ""
fi

# Step 2: Run CMake configure
echo "Configuring with CMake..."
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
echo ""

# Step 3: Patch gemmlowp if it was downloaded
echo "Patching gemmlowp (if present)..."
cd ..
python3 patch_gemmlowp_direct.py || python patch_gemmlowp_direct.py
echo ""

# Step 4: Reconfigure to pick up patches
echo "Re-configuring CMake after patching..."
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
echo ""

# Step 5: Build
echo "Building project..."
cmake --build . --config Release -- -j$(nproc 2>/dev/null || echo 4)

echo ""
echo "================================================"
echo "  Build completed successfully!"
echo "================================================"
echo ""
echo "Executables:"
[ -f radar_tagger ] && echo "  - radar_tagger"
[ -f radar_tagger_multioutput ] && echo "  - radar_tagger_multioutput"
echo ""
