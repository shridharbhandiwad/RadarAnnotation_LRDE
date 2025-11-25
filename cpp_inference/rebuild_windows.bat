@echo off
REM
REM Windows Build Script for Radar Tagger C++ Inference
REM This script performs a clean rebuild for Windows/MinGW systems
REM

setlocal enabledelayedexpansion

echo ==========================================
echo   Radar Tagger C++ - Windows Clean Build
echo ==========================================
echo.

REM Get the script directory
cd /d "%~dp0"

REM Check if we're using MinGW
where mingw32-make >nul 2>&1
if errorlevel 1 (
    echo [WARNING] mingw32-make not found in PATH
    echo Please install MinGW-w64 from: https://www.mingw-w64.org/
    echo Or MSYS2 from: https://www.msys2.org/
    echo.
    pause
    exit /b 1
)

REM Check for required tools
echo [*] Checking prerequisites...
where cmake >nul 2>&1
if errorlevel 1 (
    echo [X] CMake not found. Download from: https://cmake.org/download/
    pause
    exit /b 1
)

where g++ >nul 2>&1
if errorlevel 1 (
    echo [X] g++ not found. Install MinGW-w64
    pause
    exit /b 1
)

for /f "tokens=*" %%i in ('cmake --version ^| findstr /R "version"') do set CMAKE_VER=%%i
for /f "tokens=*" %%i in ('g++ --version ^| findstr /R "g++"') do set GCC_VER=%%i
echo    [OK] %CMAKE_VER%
echo    [OK] %GCC_VER%
echo.

REM Clean build directory
echo [*] Cleaning build directory...
if exist "build" (
    rmdir /s /q build
    echo    [OK] Removed old build directory
)
mkdir build
echo    [OK] Created fresh build directory
echo.

REM Configure with CMake
echo [*] Configuring with CMake...
echo    This will download dependencies and apply Windows/MinGW patches...
cd build
cmake -G "MinGW Makefiles" -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=g++ -DCMAKE_C_COMPILER=gcc ..
if errorlevel 1 (
    echo [X] CMake configuration failed!
    pause
    exit /b 1
)
echo.

REM Build
echo [*] Building (this may take 20-45 minutes on first build)...
set START_TIME=%TIME%
cmake --build . --config Release -- -j%NUMBER_OF_PROCESSORS%
if errorlevel 1 (
    echo [X] Build failed!
    pause
    exit /b 1
)
set END_TIME=%TIME%
echo.

REM Verify executables
echo [OK] Build completed successfully!
echo.
echo [*] Built executables:
if exist "radar_tagger.exe" (
    for %%F in (radar_tagger.exe) do echo    [OK] radar_tagger.exe (%%~zF bytes^)
) else (
    echo    [X] radar_tagger.exe not found!
    exit /b 1
)

if exist "radar_tagger_multioutput.exe" (
    for %%F in (radar_tagger_multioutput.exe) do echo    [OK] radar_tagger_multioutput.exe (%%~zF bytes^)
) else (
    echo    [X] radar_tagger_multioutput.exe not found!
    exit /b 1
)
echo.

REM Test executables
echo [*] Testing executables...
radar_tagger.exe --help >nul 2>&1
if errorlevel 1 (
    echo    [!] radar_tagger.exe help test failed
) else (
    echo    [OK] radar_tagger.exe is functional
)

radar_tagger_multioutput.exe --help >nul 2>&1
if errorlevel 1 (
    echo    [!] radar_tagger_multioutput.exe help test failed
) else (
    echo    [OK] radar_tagger_multioutput.exe is functional
)
echo.

REM Summary
echo ==========================================
echo   Build Summary
echo ==========================================
echo Status: [OK] SUCCESS
echo Location: %CD%
echo.
echo To use:
echo   .\build\radar_tagger.exe --help
echo   .\build\radar_tagger_multioutput.exe --help
echo.
echo For detailed usage, see:
echo   cpp_inference\README.md
echo   cpp_inference\BUILD_SUCCESS_WINDOWS.md
echo ==========================================
echo.

endlocal
pause
