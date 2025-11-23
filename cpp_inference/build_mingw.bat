@echo off
REM Build script for RadarTagger C++ with MinGW

echo === Building C++ Application with MinGW ===

REM Clean previous build
if exist build (
    echo Cleaning previous build...
    rmdir /s /q build
)

REM Create build directory
echo Creating build directory...
mkdir build
cd build

REM Configure with CMake using MinGW Makefiles
echo Configuring with CMake and MinGW...
cmake -G "MinGW Makefiles" ^
    -DCMAKE_C_COMPILER=gcc ^
    -DCMAKE_CXX_COMPILER=g++ ^
    -DCMAKE_MAKE_PROGRAM=mingw32-make ^
    -DCMAKE_BUILD_TYPE=Release ^
    ..

if %ERRORLEVEL% NEQ 0 (
    echo ✗ Configuration failed
    cd ..
    exit /b 1
)

echo ✓ Configuration successful

REM Build the project
echo Building...
cmake --build . --config Release

if %ERRORLEVEL% NEQ 0 (
    echo ✗ Build failed
    cd ..
    exit /b 1
)

echo ✓ Build successful
cd ..

echo.
echo Executables are in: cpp_inference\build\
echo   - radar_tagger.exe
echo   - radar_tagger_multioutput.exe
