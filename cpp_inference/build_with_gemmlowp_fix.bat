@echo off
REM Build script for MinGW with aggressive gemmlowp fixing
REM This script ensures gemmlowp is properly patched before building

echo ================================================
echo   Radar Tagger C++ Build (MinGW with Fixes)
echo ================================================
echo.

cd /d "%~dp0"

REM Step 1: Clean build if requested
if "%1"=="clean" (
    echo Cleaning build directory...
    if exist build rmdir /s /q build
    mkdir build
    echo Done.
    echo.
)

REM Step 2: Run CMake configure
echo Configuring with CMake...
cd build
cmake -G "MinGW Makefiles" -DCMAKE_BUILD_TYPE=Release ..
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ERROR: CMake configuration failed!
    echo.
    pause
    exit /b 1
)
echo.

REM Step 3: Patch gemmlowp if it was downloaded
echo Patching gemmlowp (if present)...
cd ..
python patch_gemmlowp_direct.py
echo.

REM Step 4: Reconfigure to pick up patches
echo Re-configuring CMake after patching...
cd build
cmake -G "MinGW Makefiles" -DCMAKE_BUILD_TYPE=Release ..
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ERROR: CMake re-configuration failed!
    echo.
    pause
    exit /b 1
)
echo.

REM Step 5: Build
echo Building project...
cmake --build . --config Release -- -j1
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ERROR: Build failed!
    echo.
    echo If you're still seeing eight_bit_int_gemm errors, try:
    echo   1. Delete the build directory completely
    echo   2. Run this script with 'clean' argument: build_with_gemmlowp_fix.bat clean
    echo.
    pause
    exit /b 1
)

echo.
echo ================================================
echo   Build completed successfully!
echo ================================================
echo.
echo Executables:
if exist radar_tagger.exe echo   - radar_tagger.exe
if exist radar_tagger_multioutput.exe echo   - radar_tagger_multioutput.exe
echo.
pause
