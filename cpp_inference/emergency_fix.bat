@echo off
REM Emergency fix script for eight_bit_int_gemm build errors
REM Run this if you encounter the "cannot specify '-o' with '-c'" error during build

echo ========================================
echo   EMERGENCY FIX for eight_bit_int_gemm
echo ========================================
echo.

cd /d "%~dp0"

echo Step 1: Patching CMakeLists.txt files...
python patch_gemmlowp_direct.py
echo.

echo Step 2: Patching Makefiles...
python patch_makefile_direct.py
echo.

echo Step 3: Recreating CMake cache...
cd build
del /q CMakeCache.txt 2>nul
cmake -G "MinGW Makefiles" -DCMAKE_BUILD_TYPE=Release ..
echo.

echo Step 4: Re-patching after CMake regeneration...
cd ..
python patch_makefile_direct.py
echo.

echo ========================================
echo   Fix applied. Now attempting build...
echo ========================================
echo.

cd build
cmake --build . --config Release -- -j1

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ========================================
    echo   SUCCESS! Build completed.
    echo ========================================
) else (
    echo.
    echo ========================================
    echo   Build still failing. Try these steps:
    echo ========================================
    echo.
    echo 1. Completely delete the build directory:
    echo    rmdir /s /q build
    echo.
    echo 2. Run the clean build script:
    echo    build_with_gemmlowp_fix.bat clean
    echo.
    echo 3. If that fails, check:
    echo    - Python 3 is installed
    echo    - MinGW is in your PATH
    echo    - No cmake.exe processes stuck in Task Manager
    echo.
)

pause
