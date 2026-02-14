@echo off
REM ============================================================================
REM   Radar Tagger C++ - ONNX-Only Build (Simplified)
REM ============================================================================
REM
REM This script builds a simplified version that uses only ONNX Runtime.
REM TensorFlow Lite is excluded to avoid Windows build issues.
REM
REM Usage: build_onnx_only.bat [clean]
REM ============================================================================

setlocal enabledelayedexpansion

cd /d "%~dp0"

echo.
echo ============================================================================
echo   Radar Tagger C++ - ONNX-Only Build
echo ============================================================================
echo.
echo This build uses ONLY ONNX Runtime (no TensorFlow Lite)
echo.
echo Benefits:
echo   + Much simpler and faster build
echo   + Avoids gemmlowp and TensorFlow Lite issues
echo   + Works reliably on Windows
echo.
echo Limitations:
echo   - Cannot use TensorFlow Lite models (^.tflite files^)
echo   - Only creates radar_tagger_onnx.exe
echo.

REM Check if user wants to continue
set /p CONTINUE="Continue? (Y/N): "
if /i not "%CONTINUE%"=="Y" (
    echo Build cancelled.
    exit /b 0
)

echo.

REM Clean build directory if requested
if "%1"=="clean" (
    echo Cleaning build directory...
    if exist build_onnx rmdir /s /q build_onnx
    mkdir build_onnx
    echo Done.
    echo.
) else (
    if not exist build_onnx mkdir build_onnx
)

REM Check for CMake
where cmake >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: CMake not found!
    echo Please install CMake from: https://cmake.org/download/
    pause
    exit /b 1
)

REM Detect compiler
where g++ >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    set GENERATOR=MinGW Makefiles
    echo Using MinGW compiler
) else (
    set GENERATOR=Visual Studio 16 2019
    echo Using MSVC compiler
)

echo.
echo Configuring with CMake...
echo Generator: !GENERATOR!
echo.

cd build_onnx

REM Use the simplified CMakeLists.txt
cmake -G "!GENERATOR!" -DCMAKE_BUILD_TYPE=Release -C ../CMakeLists_onnx_only.txt ..

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ERROR: CMake configuration failed!
    echo.
    echo Check:
    echo   1. Internet connection (needs to download ONNX Runtime)
    echo   2. CMake version (needs 3.16+)
    echo   3. Compiler is installed
    echo.
    cd ..
    pause
    exit /b 1
)

echo.
echo Building...
echo.

cmake --build . --config Release

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ERROR: Build failed!
    echo.
    echo For help, see WINDOWS_BUILD_ALTERNATIVES.md
    echo.
    cd ..
    pause
    exit /b 1
)

echo.
echo ============================================================================
echo   Build Successful!
echo ============================================================================
echo.
echo Executable created: build_onnx\radar_tagger_onnx.exe
echo.
echo To test:
echo   cd build_onnx
echo   radar_tagger_onnx.exe --help
echo.
echo Note: This executable can only use ONNX models (not TensorFlow Lite)
echo.

cd ..
pause
exit /b 0
