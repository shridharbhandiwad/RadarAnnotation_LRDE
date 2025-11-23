#!/bin/bash
# Clean rebuild script for MinGW gemmlowp fix
# This script cleans the build directory and rebuilds from scratch

set -e  # Exit on error

echo "========================================="
echo "  Clean Rebuild with gemmlowp Fix"
echo "========================================="
echo ""

cd "$(dirname "$0")"

echo "[1/4] Cleaning build directory..."
if [ -d "build" ]; then
    rm -rf build
    echo "  ✓ Build directory removed"
else
    echo "  ✓ Build directory doesn't exist (clean slate)"
fi

echo ""
echo "[2/4] Creating build directory..."
mkdir -p build
cd build
echo "  ✓ Build directory created"

echo ""
echo "[3/4] Configuring with CMake..."
echo "  This will download dependencies and apply patches..."
cmake -G "MinGW Makefiles" -DCMAKE_BUILD_TYPE=Release ..

echo ""
echo "[4/4] Building project..."
echo "  This may take 20-45 minutes on first build..."
cmake --build . --config Release

echo ""
echo "========================================="
echo "  Build Complete!"
echo "========================================="
echo ""
echo "Executables:"
if [ -f "radar_tagger.exe" ]; then
    echo "  ✓ radar_tagger.exe"
else
    echo "  ✗ radar_tagger.exe (NOT FOUND)"
fi

if [ -f "radar_tagger_multioutput.exe" ]; then
    echo "  ✓ radar_tagger_multioutput.exe"
else
    echo "  ✗ radar_tagger_multioutput.exe (NOT FOUND)"
fi

echo ""
echo "Location: $(pwd)"
echo ""
