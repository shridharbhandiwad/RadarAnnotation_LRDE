#!/bin/bash
#
# Linux Build Script for Radar Tagger C++ Inference
# This script performs a clean rebuild for Linux systems
#

set -e  # Exit on error

echo "=========================================="
echo "  Radar Tagger C++ - Linux Clean Build"
echo "=========================================="
echo ""

# Get the script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Check if we're on Linux
if [[ "$OSTYPE" != "linux-gnu"* ]]; then
    echo "⚠️  Warning: This script is designed for Linux."
    echo "   Current OS: $OSTYPE"
    echo ""
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Check for required tools
echo "🔍 Checking prerequisites..."
command -v cmake >/dev/null 2>&1 || { echo "❌ CMake not found. Install with: sudo apt install cmake"; exit 1; }
command -v g++ >/dev/null 2>&1 || { echo "❌ g++ not found. Install with: sudo apt install g++"; exit 1; }
command -v make >/dev/null 2>&1 || { echo "❌ make not found. Install with: sudo apt install build-essential"; exit 1; }

CMAKE_VERSION=$(cmake --version | head -n1)
GCC_VERSION=$(g++ --version | head -n1)
echo "   ✅ $CMAKE_VERSION"
echo "   ✅ $GCC_VERSION"
echo ""

# Clean build directory
echo "🧹 Cleaning build directory..."
if [ -d "build" ]; then
    rm -rf build
    echo "   ✅ Removed old build directory"
fi
mkdir build
echo "   ✅ Created fresh build directory"
echo ""

# Configure with CMake
echo "⚙️  Configuring with CMake..."
cd build
cmake -G "Unix Makefiles" \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CXX_COMPILER=g++ \
      -DCMAKE_C_COMPILER=gcc \
      .. || { echo "❌ CMake configuration failed!"; exit 1; }
echo ""

# Build
echo "🔨 Building (using $(nproc) parallel jobs)..."
START_TIME=$(date +%s)
make -j$(nproc) || { echo "❌ Build failed!"; exit 1; }
END_TIME=$(date +%s)
BUILD_TIME=$((END_TIME - START_TIME))
echo ""

# Verify executables
echo "✅ Build completed successfully!"
echo ""
echo "📦 Built executables:"
if [ -f "radar_tagger" ]; then
    ls -lh radar_tagger | awk '{print "   ✅ radar_tagger (" $5 ")"}'
else
    echo "   ❌ radar_tagger not found!"
    exit 1
fi

if [ -f "radar_tagger_multioutput" ]; then
    ls -lh radar_tagger_multioutput | awk '{print "   ✅ radar_tagger_multioutput (" $5 ")"}'
else
    echo "   ❌ radar_tagger_multioutput not found!"
    exit 1
fi
echo ""

# Test executables
echo "🧪 Testing executables..."
if ./radar_tagger --help >/dev/null 2>&1; then
    echo "   ✅ radar_tagger is functional"
else
    echo "   ⚠️  radar_tagger help test failed"
fi

if ./radar_tagger_multioutput --help >/dev/null 2>&1; then
    echo "   ✅ radar_tagger_multioutput is functional"
else
    echo "   ⚠️  radar_tagger_multioutput help test failed"
fi
echo ""

# Summary
echo "=========================================="
echo "  Build Summary"
echo "=========================================="
echo "Status: ✅ SUCCESS"
echo "Build time: ${BUILD_TIME}s"
echo "Location: $(pwd)"
echo ""
echo "To use:"
echo "  ./build/radar_tagger --help"
echo "  ./build/radar_tagger_multioutput --help"
echo ""
echo "For detailed usage, see:"
echo "  cpp_inference/README.md"
echo "  cpp_inference/BUILD_SUCCESS_LINUX.md"
echo "=========================================="
