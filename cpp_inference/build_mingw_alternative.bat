@echo off
REM Alternative build script for MinGW - auto-detects MinGW installation

echo === Building C++ Application with MinGW (Auto-detect) ===

REM Check if MinGW is in PATH
where gcc >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo ✗ Error: gcc not found in PATH
    echo Please ensure MinGW's bin directory is in your PATH
    echo Example: C:\MinGW\bin or C:\mingw64\bin
    echo.
    echo You can add it temporarily with:
    echo set PATH=C:\mingw64\bin;%%PATH%%
    exit /b 1
)

where g++ >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo ✗ Error: g++ not found in PATH
    exit /b 1
)

where mingw32-make >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    set MAKE_PROGRAM=mingw32-make
) else (
    where make >nul 2>&1
    if %ERRORLEVEL% EQU 0 (
        set MAKE_PROGRAM=make
    ) else (
        echo ✗ Error: Neither mingw32-make nor make found in PATH
        exit /b 1
    )
)

echo ✓ Found gcc: 
gcc --version | findstr gcc
echo ✓ Found g++: 
g++ --version | findstr g++
echo ✓ Using make program: %MAKE_PROGRAM%
echo.

REM Clean previous build
if exist build (
    echo Cleaning previous build...
    rmdir /s /q build
)

REM Create build directory
echo Creating build directory...
mkdir build
cd build

REM Configure with CMake
echo Configuring with CMake...
cmake -G "MinGW Makefiles" ^
    -DCMAKE_MAKE_PROGRAM=%MAKE_PROGRAM% ^
    -DCMAKE_BUILD_TYPE=Release ^
    ..

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ✗ Configuration failed
    echo.
    echo Troubleshooting tips:
    echo 1. Ensure MinGW is properly installed
    echo 2. Add MinGW's bin directory to PATH
    echo 3. Try running: gcc --version
    cd ..
    exit /b 1
)

echo ✓ Configuration successful
echo.

REM Build the project
echo Building with %MAKE_PROGRAM%...
%MAKE_PROGRAM%

if %ERRORLEVEL% NEQ 0 (
    echo ✗ Build failed
    cd ..
    exit /b 1
)

echo.
echo ✓ Build successful
cd ..

echo.
echo ======================================
echo Build completed successfully!
echo ======================================
echo Executables are in: cpp_inference\build\
echo   - radar_tagger.exe
echo   - radar_tagger_multioutput.exe
