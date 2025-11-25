@echo off
REM Build System Verification Script for Windows
REM Checks CMake, compiler, and provides recommendations

echo ========================================================
echo   Radar Tagger Build System Check
echo ========================================================
echo.

REM Check CMake
echo [1/4] Checking CMake...
where cmake >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [X] CMake NOT FOUND
    echo     Download from: https://cmake.org/download/
    echo.
    goto :compiler_check
)

echo [OK] CMake found
cmake --version | findstr /C:"cmake version"
echo.

REM Get CMake version (basic check)
for /f "tokens=3" %%i in ('cmake --version ^| findstr /C:"cmake version"') do set CMAKE_VER=%%i
echo     Your CMake version: %CMAKE_VER%

REM Simple version comparison (just check first digit after 3.)
for /f "tokens=2 delims=." %%i in ("%CMAKE_VER%") do set CMAKE_MINOR=%%i

if %CMAKE_MINOR% LSS 16 (
    echo [!] WARNING: CMake version is too old
    echo     Minimum required: 3.16
    echo     Recommended: 3.20+
    echo     Please upgrade from: https://cmake.org/download/
) else if %CMAKE_MINOR% LSS 19 (
    echo [!] NOTICE: CMake 3.%CMAKE_MINOR% detected
    echo     Build will work, but some optimizations disabled
    echo     Recommended: 3.20+ for full feature support
) else if %CMAKE_MINOR% LSS 20 (
    echo [OK] CMake 3.%CMAKE_MINOR% - Fully supported
) else (
    echo [OK] CMake 3.%CMAKE_MINOR% - Excellent! (Recommended version)
)
echo.

:compiler_check
REM Check for MinGW
echo [2/4] Checking compiler...
where g++ >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo [OK] MinGW g++ found
    g++ --version | findstr "g++"
    echo.
    set COMPILER_FOUND=1
    goto :git_check
)

REM Check for MSVC (cl.exe)
where cl >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo [OK] MSVC compiler found
    cl 2>&1 | findstr "Compiler"
    echo.
    set COMPILER_FOUND=1
    goto :git_check
)

echo [X] No C++ compiler found
echo     Install one of:
echo     - MinGW-w64: https://www.mingw-w64.org/downloads/
echo     - Visual Studio 2019+: https://visualstudio.microsoft.com/
echo.
set COMPILER_FOUND=0

:git_check
REM Check for Git (optional but useful)
echo [3/4] Checking Git (optional)...
where git >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [!] Git not found (optional - only needed for version control)
    echo.
) else (
    echo [OK] Git found
    git --version
    echo.
)

:python_check
REM Check for Python (needed for some build scripts)
echo [4/4] Checking Python (for patching scripts)...
where python >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [!] Python not found
    echo     Some build scripts may not work
    echo     Install from: https://www.python.org/downloads/
    echo.
) else (
    echo [OK] Python found
    python --version
    echo.
)

echo ========================================================
echo   Summary and Recommendations
echo ========================================================
echo.

if %CMAKE_MINOR% LSS 16 (
    echo [ACTION REQUIRED]
    echo   1. Upgrade CMake to 3.20+ from https://cmake.org/download/
    echo.
)

if %COMPILER_FOUND% EQU 0 (
    echo [ACTION REQUIRED]
    echo   1. Install a C++ compiler:
    echo      - MinGW-w64: https://www.mingw-w64.org/downloads/
    echo      - OR Visual Studio 2019+: https://visualstudio.microsoft.com/
    echo.
) else (
    echo [READY TO BUILD]
    echo   Your system has the required tools installed!
    echo.
    echo   To build the project:
    echo     1. cd cpp_inference
    echo     2. build_with_gemmlowp_fix.bat clean
    echo.
    
    if %CMAKE_MINOR% LSS 19 (
        echo   Note: You have CMake 3.%CMAKE_MINOR%
        echo   - Build will work correctly
        echo   - Some optimization features will be skipped
        echo   - Consider upgrading to CMake 3.20+ for best experience
        echo.
    )
)

echo ========================================================
echo   Documentation
echo ========================================================
echo.
echo   - Windows CMake issues: WINDOWS_CMAKE_VERSION_FIX.md
echo   - gemmlowp build issues: WINDOWS_MINGW_BUILD_FIX.md
echo   - Quick start guide: START_HERE.md
echo   - General information: README.md
echo.

pause
