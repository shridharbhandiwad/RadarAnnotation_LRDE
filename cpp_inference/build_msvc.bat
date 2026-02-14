@echo off
REM ============================================================================
REM   Radar Tagger C++ - MSVC Build Script
REM ============================================================================
REM
REM This script builds using Microsoft Visual C++ (MSVC) instead of MinGW.
REM MSVC often has better Windows compatibility.
REM
REM Prerequisites:
REM   - Visual Studio 2019 or 2022 installed
REM   - Run from "x64 Native Tools Command Prompt for VS"
REM
REM Usage: build_msvc.bat [clean]
REM ============================================================================

setlocal enabledelayedexpansion

cd /d "%~dp0"

echo.
echo ============================================================================
echo   Radar Tagger C++ - MSVC Build
echo ============================================================================
echo.

REM Check if we're in a Visual Studio environment
if not defined VSINSTALLDIR (
    echo WARNING: Visual Studio environment not detected!
    echo.
    echo Please run this script from:
    echo   "x64 Native Tools Command Prompt for VS 2019"
    echo   or
    echo   "x64 Native Tools Command Prompt for VS 2022"
    echo.
    echo You can find this in the Start Menu under Visual Studio.
    echo.
    set /p CONTINUE="Continue anyway? (Y/N): "
    if /i not "!CONTINUE!"=="Y" (
        exit /b 1
    )
    echo.
)

REM Check for CMake
where cmake >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: CMake not found!
    echo Please install CMake from: https://cmake.org/download/
    pause
    exit /b 1
)

echo CMake found: 
cmake --version | findstr /C:"version"
echo.

REM Detect Visual Studio version
set VS_GENERATOR=Visual Studio 16 2019

if defined VisualStudioVersion (
    if "%VisualStudioVersion:~0,2%"=="17" (
        set VS_GENERATOR=Visual Studio 17 2022
        echo Detected: Visual Studio 2022
    ) else if "%VisualStudioVersion:~0,2%"=="16" (
        set VS_GENERATOR=Visual Studio 16 2019
        echo Detected: Visual Studio 2019
    ) else (
        echo Detected: Visual Studio (version %VisualStudioVersion%)
    )
) else (
    echo Using default: Visual Studio 2019
)

echo Generator: !VS_GENERATOR!
echo.

REM Clean build directory if requested
if "%1"=="clean" (
    echo Cleaning build directory...
    if exist build rmdir /s /q build
    mkdir build
    echo Done.
    echo.
) else (
    if not exist build mkdir build
)

echo Configuring with CMake...
echo.

cd build

REM Configure with MSVC
cmake -G "!VS_GENERATOR!" -A x64 -DCMAKE_BUILD_TYPE=Release ..

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ============================================================================
    echo   Configuration Failed!
    echo ============================================================================
    echo.
    echo Common Issues:
    echo.
    echo 1. Visual Studio not properly installed
    echo    Solution: Install Visual Studio with "Desktop development with C++"
    echo.
    echo 2. CMake version too old
    echo    Solution: Upgrade to CMake 3.20+ from https://cmake.org/download/
    echo.
    echo 3. Internet connection required
    echo    Solution: CMake needs to download TensorFlow Lite and ONNX Runtime
    echo.
    echo 4. TensorFlow Lite build issues
    echo    Solution: Try the ONNX-only build: build_onnx_only.bat
    echo.
    echo For more help, see WINDOWS_BUILD_ALTERNATIVES.md
    echo.
    cd ..
    pause
    exit /b 1
)

echo.
echo ============================================================================
echo   Configuration Successful!
echo ============================================================================
echo.
echo Building... (this may take 10-30 minutes)
echo.

REM Build the project
cmake --build . --config Release

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ============================================================================
    echo   Build Failed!
    echo ============================================================================
    echo.
    echo Common Issues:
    echo.
    echo 1. eight_bit_int_gemm compilation error
    echo    Solution: The patches may not have been applied correctly
    echo    Try: Delete build directory and run 'build_msvc.bat clean'
    echo.
    echo 2. Out of memory
    echo    Solution: Close other applications and try again
    echo.
    echo 3. Linker errors
    echo    Solution: Make sure all dependencies downloaded correctly
    echo    Check: build\_deps directory should exist
    echo.
    echo Alternative: Try ONNX-only build with 'build_onnx_only.bat'
    echo.
    echo For more help, see WINDOWS_BUILD_ALTERNATIVES.md
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

REM Find and display the executables
if exist Release\radar_tagger.exe (
    echo Created: Release\radar_tagger.exe
    for %%A in (Release\radar_tagger.exe) do echo   Size: %%~zA bytes
)

if exist Release\radar_tagger_multioutput.exe (
    echo Created: Release\radar_tagger_multioutput.exe
    for %%A in (Release\radar_tagger_multioutput.exe) do echo   Size: %%~zA bytes
)

if exist Release\onnxruntime.dll (
    echo Created: Release\onnxruntime.dll
)

echo.
echo To test the executables:
echo   cd build\Release
echo   radar_tagger.exe --help
echo   radar_tagger_multioutput.exe --help
echo.

cd ..
pause
exit /b 0
