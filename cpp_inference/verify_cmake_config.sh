#!/bin/bash

# Script to verify CMake configuration is correct for Windows/MinGW builds
# Run this before attempting to build

echo "==================================="
echo "CMake Configuration Verification"
echo "==================================="
echo ""

ERRORS=0

# Check 1: Verify CMakeLists.txt doesn't have problematic CMAKE_C_FLAGS
echo "[1/5] Checking CMakeLists.txt for problematic CMAKE_C_FLAGS..."
if grep -q "CMAKE_C_FLAGS.*max\|CMAKE_C_FLAGS.*min" CMakeLists.txt 2>/dev/null; then
    echo "❌ FAIL: Found problematic CMAKE_C_FLAGS with max/min macros"
    echo "   These will cause 'filename, directory name, or volume label syntax' errors on Windows"
    echo "   Remove lines like: set(CMAKE_C_FLAGS ... -Dmax(a,b)=... )"
    ERRORS=$((ERRORS + 1))
else
    echo "✅ PASS: No problematic CMAKE_C_FLAGS found"
fi
echo ""

# Check 2: Verify NOMINMAX is defined
echo "[2/5] Checking for NOMINMAX definition..."
if grep -q "add_compile_definitions(NOMINMAX)" CMakeLists.txt 2>/dev/null; then
    echo "✅ PASS: NOMINMAX is defined (prevents Windows.h conflicts)"
else
    echo "⚠️  WARN: NOMINMAX not found - you may encounter min/max conflicts with Windows.h"
fi
echo ""

# Check 3: Verify compiler is accessible
echo "[3/5] Checking C compiler..."
if command -v cc &> /dev/null || command -v gcc &> /dev/null; then
    COMPILER=$(command -v cc || command -v gcc)
    echo "✅ PASS: Compiler found at $COMPILER"
    
    # Test compile
    echo "int main() { return 0; }" > /tmp/test_compile_$$.c
    if $COMPILER -c /tmp/test_compile_$$.c -o /tmp/test_compile_$$.o 2>/dev/null; then
        echo "✅ PASS: Compiler can compile simple programs"
        rm -f /tmp/test_compile_$$.*
    else
        echo "❌ FAIL: Compiler cannot compile simple programs"
        rm -f /tmp/test_compile_$$.*
        ERRORS=$((ERRORS + 1))
    fi
else
    echo "❌ FAIL: No C compiler found (cc or gcc)"
    echo "   Install MinGW-w64 or ensure it's in your PATH"
    ERRORS=$((ERRORS + 1))
fi
echo ""

# Check 4: Check for CMake
echo "[4/5] Checking CMake..."
if command -v cmake &> /dev/null; then
    CMAKE_VERSION=$(cmake --version | head -n1)
    echo "✅ PASS: $CMAKE_VERSION"
else
    echo "❌ FAIL: CMake not found"
    ERRORS=$((ERRORS + 1))
fi
echo ""

# Check 5: Check environment variables
echo "[5/5] Checking environment variables..."
if [ -n "$CMAKE_C_FLAGS" ]; then
    echo "⚠️  WARN: CMAKE_C_FLAGS environment variable is set:"
    echo "   CMAKE_C_FLAGS=$CMAKE_C_FLAGS"
    if echo "$CMAKE_C_FLAGS" | grep -q "max\|min"; then
        echo "❌ FAIL: CMAKE_C_FLAGS contains max/min macros - this will cause build failures"
        echo "   Run: unset CMAKE_C_FLAGS"
        ERRORS=$((ERRORS + 1))
    fi
else
    echo "✅ PASS: CMAKE_C_FLAGS not set (good)"
fi
echo ""

# Check build directory
if [ -d "build" ]; then
    echo "📁 Build directory exists"
    if [ -f "build/CMakeCache.txt" ]; then
        echo "   Checking CMake cache for problematic flags..."
        if grep "CMAKE_C_FLAGS.*max\|CMAKE_C_FLAGS.*min" build/CMakeCache.txt 2>/dev/null | grep -v "^#" | grep -q "max\|min"; then
            echo "⚠️  WARN: CMake cache may contain old problematic flags"
            echo "   Recommendation: rm -rf build && mkdir build"
        else
            echo "   ✅ CMake cache looks clean"
        fi
    fi
else
    echo "📁 No build directory (will be created during configuration)"
fi
echo ""

# Summary
echo "==================================="
echo "Summary"
echo "==================================="
if [ $ERRORS -eq 0 ]; then
    echo "✅ All checks passed! You can proceed with:"
    echo ""
    echo "   mkdir -p build && cd build"
    echo "   cmake -G \"MinGW Makefiles\" -DCMAKE_BUILD_TYPE=Release .."
    echo "   cd .. && python fix_build_dependencies.py"
    echo "   cd build && mingw32-make -j4"
    echo ""
    exit 0
else
    echo "❌ Found $ERRORS error(s) - fix these before building"
    echo ""
    echo "See FIX_INSTRUCTIONS.md for detailed guidance"
    echo ""
    exit 1
fi
