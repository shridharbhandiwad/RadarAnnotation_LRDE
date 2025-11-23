@echo off
REM Clean rebuild script for Windows/MinGW

echo ===================================
echo Clean Rebuild Script for MinGW
echo ===================================
echo.

echo Cleaning build directory...
if exist build (
    rmdir /s /q build
    echo Build directory cleaned.
) else (
    echo No existing build directory found.
)

echo.
echo Creating new build directory...
mkdir build
cd build

echo.
echo Configuring with CMake...
cmake .. -G "MinGW Makefiles"

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo *** Configuration failed! ***
    pause
    exit /b 1
)

echo.
echo Building application...
mingw32-make -j4

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo *** Build failed! ***
    pause
    exit /b 1
)

echo.
echo ===================================
echo Build completed successfully!
echo ===================================
echo.
echo Executables are in: build\radar_tagger.exe and build\radar_tagger_multioutput.exe
echo.
pause
