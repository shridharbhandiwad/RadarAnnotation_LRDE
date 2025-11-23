@echo off
REM ============================================================================
REM  RadarTagger C++ Build Script for MinGW (with gemmlowp fix)
REM ============================================================================
REM  This script handles the problematic eight_bit_int_gemm target
REM  by patching it after initial configuration
REM ============================================================================

setlocal enabledelayedexpansion

echo ============================================================================
echo RadarTagger C++ Build for MinGW (with automatic gemmlowp fix)
echo ============================================================================
echo.

cd /d "%~dp0"

REM Clean build directory
if exist build (
    echo Cleaning old build directory...
    rmdir /s /q build 2>nul
    if exist build (
        echo Warning: Some files could not be deleted. Trying again...
        timeout /t 2 >nul
        rmdir /s /q build 2>nul
    )
)

echo Creating fresh build directory...
mkdir build
cd build

echo.
echo ============================================================================
echo Step 1: Configuring with CMake
echo ============================================================================
echo.

cmake .. -G "MinGW Makefiles" -DCMAKE_BUILD_TYPE=Release
if errorlevel 1 (
    echo [ERROR] CMake configuration failed!
    goto :error
)

echo.
echo ============================================================================
echo Step 2: Applying gemmlowp fix if needed...
echo ============================================================================
echo.

REM Check if gemmlowp was downloaded
if exist "gemmlowp\CMakeLists.txt" (
    echo Found gemmlowp, applying MinGW compatibility patch...
    
    REM Create a Python script to patch the file
    echo import re > patch_gemmlowp.py
    echo import sys >> patch_gemmlowp.py
    echo. >> patch_gemmlowp.py
    echo cmake_file = "gemmlowp/CMakeLists.txt" >> patch_gemmlowp.py
    echo. >> patch_gemmlowp.py
    echo try: >> patch_gemmlowp.py
    echo     with open(cmake_file, 'r') as f: >> patch_gemmlowp.py
    echo         content = f.read() >> patch_gemmlowp.py
    echo. >> patch_gemmlowp.py
    echo     # Comment out add_library for eight_bit_int_gemm >> patch_gemmlowp.py
    echo     content = re.sub(r'add_library\(eight_bit_int_gemm', >> patch_gemmlowp.py
    echo                      '# DISABLED_FOR_MINGW: add_library(eight_bit_int_gemm', >> patch_gemmlowp.py
    echo                      content) >> patch_gemmlowp.py
    echo. >> patch_gemmlowp.py
    echo     # Comment out add_executable for eight_bit_int_gemm >> patch_gemmlowp.py
    echo     content = re.sub(r'add_executable\(eight_bit_int_gemm', >> patch_gemmlowp.py
    echo                      '# DISABLED_FOR_MINGW: add_executable(eight_bit_int_gemm', >> patch_gemmlowp.py
    echo                      content) >> patch_gemmlowp.py
    echo. >> patch_gemmlowp.py
    echo     # Comment out target_link_libraries for eight_bit_int_gemm >> patch_gemmlowp.py
    echo     content = re.sub(r'target_link_libraries\(eight_bit_int_gemm', >> patch_gemmlowp.py
    echo                      '# DISABLED_FOR_MINGW: target_link_libraries(eight_bit_int_gemm', >> patch_gemmlowp.py
    echo                      content) >> patch_gemmlowp.py
    echo. >> patch_gemmlowp.py
    echo     with open(cmake_file, 'w') as f: >> patch_gemmlowp.py
    echo         f.write(content) >> patch_gemmlowp.py
    echo. >> patch_gemmlowp.py
    echo     print("Successfully patched gemmlowp CMakeLists.txt") >> patch_gemmlowp.py
    echo except Exception as e: >> patch_gemmlowp.py
    echo     print(f"Error patching: {e}") >> patch_gemmlowp.py
    echo     sys.exit(1) >> patch_gemmlowp.py
    
    REM Run the patch script
    python patch_gemmlowp.py
    if errorlevel 1 (
        echo [WARNING] Failed to patch gemmlowp, continuing anyway...
    ) else (
        echo Successfully patched gemmlowp!
        echo.
        echo Reconfiguring CMake with patched files...
        cmake .. -G "MinGW Makefiles" -DCMAKE_BUILD_TYPE=Release
        if errorlevel 1 (
            echo [ERROR] CMake reconfiguration failed!
            goto :error
        )
    )
    
    del patch_gemmlowp.py 2>nul
) else (
    echo gemmlowp directory not found yet, will rely on CMakeLists.txt fix...
)

echo.
echo ============================================================================
echo Step 3: Building project
echo ============================================================================
echo.

cmake --build . --config Release --parallel 4
if errorlevel 1 (
    echo.
    echo [ERROR] Build failed!
    echo.
    echo If you see errors about eight_bit_int_gemm, the patch may not have been applied.
    echo Please run this script again.
    goto :error
)

echo.
echo ============================================================================
echo Build completed successfully!
echo ============================================================================
echo.

REM List the built executables
echo Built executables:
dir /b *.exe 2>nul
if errorlevel 1 (
    echo [WARNING] No .exe files found in build directory
    echo Checking subdirectories...
    dir /s /b *.exe 2>nul | findstr /i "radar_tagger"
)

echo.
echo ============================================================================
echo SUCCESS!
echo ============================================================================
cd ..
exit /b 0

:error
echo.
echo ============================================================================
echo BUILD FAILED
echo ============================================================================
echo.
echo Please check the error messages above.
echo If the error is related to eight_bit_int_gemm:
echo   1. Delete the build directory
echo   2. Run this script again
echo.
cd ..
exit /b 1
