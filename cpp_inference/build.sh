#!/bin/bash
# Build script for Radar Tagger C++ Application

set -e

echo "====================================="
echo "  Radar Tagger C++ Build Script"
echo "====================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check for CMake
if ! command -v cmake &> /dev/null; then
    echo -e "${RED}Error: CMake is not installed${NC}"
    echo "Please install CMake 3.15 or higher"
    exit 1
fi

CMAKE_VERSION=$(cmake --version | head -n1 | cut -d' ' -f3)
echo -e "${GREEN}Found CMake version: $CMAKE_VERSION${NC}"

# Create build directory
BUILD_DIR="build"
if [ -d "$BUILD_DIR" ]; then
    echo -e "${YELLOW}Build directory exists. Cleaning...${NC}"
    rm -rf "$BUILD_DIR"
fi

mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# Configure
echo ""
echo "Configuring project..."
cmake .. -DCMAKE_BUILD_TYPE=Release

# Build
echo ""
echo "Building project..."
cmake --build . --config Release --parallel $(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)

# Check if build succeeded
if [ -f "radar_tagger" ] || [ -f "Release/radar_tagger.exe" ]; then
    echo ""
    echo -e "${GREEN}✓ Build successful!${NC}"
    echo ""
    echo "Executable location:"
    if [ -f "radar_tagger" ]; then
        echo "  $(pwd)/radar_tagger"
    else
        echo "  $(pwd)/Release/radar_tagger.exe"
    fi
    echo ""
    echo "To run the application:"
    echo "  cd build"
    echo "  ./radar_tagger --model ../cpp_models/lstm/lstm_model.tflite \\"
    echo "                 --metadata ../cpp_models/lstm/model_metadata.json"
    echo ""
else
    echo -e "${RED}✗ Build failed${NC}"
    exit 1
fi
