@echo off
REM ============================================================================
REM   Radar Tagger C++ - Robust Windows Build Script with Diagnostics
REM ============================================================================
REM
REM This script:
REM   1. Checks system requirements
REM   2. Provides detailed diagnostics
REM   3. Handles CMake errors gracefully
REM   4. Offers alternative build methods
REM
REM Usage: build_windows_robust.bat [clean|msvc|mingw]
REM ============================================================================

setlocal enabledelayedexpansion

cd /d "%~dp0"

echo.
echo ============================================================================
echo   Radar Tagger C++ - Windows Build (Robust Edition)
echo ============================================================================
echo.

REM ============================================================================
REM Step 1: System Diagnostics
REM ============================================================================

echo [STEP 1/6] Checking system requirements...
echo.

REM Check CMake
where cmake >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] CMake not found!
    echo.
    echo Please install CMake from: https://cmake.org/download/
    echo Minimum version: 3.16 ^(3.20+ recommended^)
    echo.
    goto :error_exit
)

REM Get CMake version
for /f "tokens=3" %%i in ('cmake --version 2^>^&1 ^| findstr /r "cmake.version"') do set CMAKE_VERSION=%%i
echo [OK] CMake found: %CMAKE_VERSION%

REM Check for compiler based on argument
set BUILD_GENERATOR=MinGW Makefiles
set COMPILER_TYPE=mingw

if "%1"=="msvc" (
    set BUILD_GENERATOR=Visual Studio 16 2019
    set COMPILER_TYPE=msvc
    echo [INFO] Using MSVC compiler ^(Visual Studio^)
) else (
    REM Check MinGW
    where g++ >nul 2>&1
    if !ERRORLEVEL! NEQ 0 (
        echo [WARNING] MinGW g++ not found in PATH
        echo.
        echo Attempting to use MSVC instead...
        set BUILD_GENERATOR=Visual Studio 16 2019
        set COMPILER_TYPE=msvc
    ) else (
        echo [OK] MinGW g++ compiler found
    )
)

REM Check Python (for patching scripts)
where python >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [WARNING] Python not found - patch scripts may not work
) else (
    echo [OK] Python found
)

REM Check internet connection
echo [INFO] Checking internet connection...
ping -n 1 github.com >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [WARNING] Cannot reach github.com - dependency downloads may fail
) else (
    echo [OK] Internet connection available
)

echo.
echo [SUMMARY]
echo   Compiler: %COMPILER_TYPE%
echo   Generator: %BUILD_GENERATOR%
echo   CMake: %CMAKE_VERSION%
echo.

pause
echo.

REM ============================================================================
REM Step 2: Clean build directory
REM ============================================================================

if "%1"=="clean" (
    echo [STEP 2/6] Cleaning build directory...
    if exist build (
        echo Removing old build directory...
        rmdir /s /q build
    )
    mkdir build
    echo [OK] Build directory cleaned
    echo.
) else (
    echo [STEP 2/6] Skipping clean ^(use 'build_windows_robust.bat clean' to clean^)
    if not exist build mkdir build
    echo.
)

REM ============================================================================
REM Step 3: CMake Configuration (First Attempt)
REM ============================================================================

echo [STEP 3/6] Configuring project with CMake...
echo.
echo Command: cmake -G "%BUILD_GENERATOR%" -DCMAKE_BUILD_TYPE=Release ..
echo.

cd build

REM Capture CMake output to a file for diagnostics
cmake -G "%BUILD_GENERATOR%" -DCMAKE_BUILD_TYPE=Release .. > cmake_config_output.txt 2>&1
set CMAKE_RESULT=%ERRORLEVEL%

REM Display the output
type cmake_config_output.txt
echo.

if %CMAKE_RESULT% NEQ 0 (
    echo [ERROR] CMake configuration failed!
    echo.
    echo Analyzing error...
    echo.
    
    REM Check for specific error patterns
    findstr /C:"cannot find -lstdc++" cmake_config_output.txt >nul 2>&1
    if !ERRORLEVEL! EQU 0 (
        echo [DIAGNOSIS] Missing C++ standard library
        echo [SOLUTION] Install MinGW-w64 with libstdc++
        echo   Download from: https://www.mingw-w64.org/downloads/
        goto :error_with_alternatives
    )
    
    findstr /C:"CMAKE_CXX_COMPILER" cmake_config_output.txt >nul 2>&1
    if !ERRORLEVEL! EQU 0 (
        echo [DIAGNOSIS] Compiler not found or not working
        echo [SOLUTION] Ensure compiler is in PATH
        goto :error_with_alternatives
    )
    
    findstr /C:"FetchContent" cmake_config_output.txt >nul 2>&1
    if !ERRORLEVEL! EQU 0 (
        echo [DIAGNOSIS] Failed to download dependencies
        echo [SOLUTION] Check internet connection and try again
        goto :error_with_alternatives
    )
    
    findstr /C:"Policy CMP" cmake_config_output.txt >nul 2>&1
    if !ERRORLEVEL! EQU 0 (
        echo [DIAGNOSIS] CMake version incompatibility
        echo [SOLUTION] Upgrade CMake to version 3.20 or higher
        goto :error_with_alternatives
    )
    
    echo [DIAGNOSIS] Unknown error - see cmake_config_output.txt for details
    goto :error_with_alternatives
) else (
    echo [OK] CMake configuration succeeded!
)

echo.

REM ============================================================================
REM Step 4: Apply patches (if needed)
REM ============================================================================

echo [STEP 4/6] Applying patches to dependencies...
echo.

cd ..
if exist patch_gemmlowp_direct.py (
    python patch_gemmlowp_direct.py
    if !ERRORLEVEL! EQU 0 (
        echo [OK] Patches applied successfully
    ) else (
        echo [WARNING] Patch script failed - continuing anyway
    )
) else (
    echo [INFO] No patch script found - skipping
)
echo.

cd build

REM ============================================================================
REM Step 5: Build
REM ============================================================================

echo [STEP 5/6] Building project...
echo.

if "%COMPILER_TYPE%"=="msvc" (
    echo Building with MSVC...
    cmake --build . --config Release
) else (
    echo Building with MinGW...
    cmake --build . --config Release -- -j1
)

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] Build failed!
    echo.
    echo Common issues:
    echo   - eight_bit_int_gemm errors: The patches may not have been applied
    echo   - Out of memory: Try building without -j flag
    echo   - Linker errors: Check that all dependencies are available
    echo.
    goto :error_with_alternatives
)

echo.
echo [OK] Build completed successfully!
echo.

REM ============================================================================
REM Step 6: Verification
REM ============================================================================

echo [STEP 6/6] Verifying build outputs...
echo.

if exist radar_tagger.exe (
    echo [OK] radar_tagger.exe created
    for %%A in (radar_tagger.exe) do echo       Size: %%~zA bytes
) else (
    echo [WARNING] radar_tagger.exe not found
)

if exist radar_tagger_multioutput.exe (
    echo [OK] radar_tagger_multioutput.exe created  
    for %%A in (radar_tagger_multioutput.exe) do echo       Size: %%~zA bytes
) else (
    echo [WARNING] radar_tagger_multioutput.exe not found
)

if exist onnxruntime.dll (
    echo [OK] onnxruntime.dll copied
) else (
    echo [INFO] onnxruntime.dll will be copied on next build
)

echo.
echo ============================================================================
echo   BUILD SUCCESSFUL!
echo ============================================================================
echo.
echo Executables are in: %CD%
echo.
echo To test:
echo   radar_tagger.exe --help
echo   radar_tagger_multioutput.exe --help
echo.

cd ..
goto :end

REM ============================================================================
REM Error Handlers
REM ============================================================================

:error_with_alternatives
cd ..
echo.
echo ============================================================================
echo   BUILD FAILED - Alternative Options
echo ============================================================================
echo.
echo Option 1: Try different compiler
echo   - If you used MinGW, try: build_windows_robust.bat msvc
echo   - If you used MSVC, try: build_windows_robust.bat mingw
echo.
echo Option 2: Use Docker
echo   Run the build in a controlled environment:
echo   docker run --rm -v %CD%:/work -w /work/cpp_inference ...
echo.
echo Option 3: Use WSL2 (Windows Subsystem for Linux)
echo   1. Install WSL2: wsl --install
echo   2. cd cpp_inference
echo   3. chmod +x build.sh
echo   4. ./build.sh
echo.
echo Option 4: Manual dependency management
echo   See WINDOWS_BUILD_ALTERNATIVES.md for details
echo.
echo Option 5: Request pre-built binaries
echo   Contact the maintainer for pre-compiled executables
echo.
echo For detailed diagnostics, check:
echo   build/cmake_config_output.txt
echo.

:error_exit
echo ============================================================================
echo.
pause
exit /b 1

:end
pause
exit /b 0
