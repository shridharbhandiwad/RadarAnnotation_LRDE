@echo off
REM Clean rebuild script for MinGW gemmlowp fix
REM This script cleans the build directory and rebuilds from scratch

echo =========================================
echo   Clean Rebuild with gemmlowp Fix
echo =========================================
echo.

cd /d "%~dp0"

echo [1/4] Cleaning build directory...
if exist build (
    rmdir /s /q build
    echo   [OK] Build directory removed
) else (
    echo   [OK] Build directory doesn't exist (clean slate)
)

echo.
echo [2/4] Creating build directory...
mkdir build
cd build
echo   [OK] Build directory created

echo.
echo [3/4] Configuring with CMake...
echo   This will download dependencies and apply patches...
cmake -G "MinGW Makefiles" -DCMAKE_BUILD_TYPE=Release ..
if errorlevel 1 (
    echo   [ERROR] CMake configuration failed!
    pause
    exit /b 1
)

echo.
echo [4/4] Building project...
echo   This may take 20-45 minutes on first build...
cmake --build . --config Release
if errorlevel 1 (
    echo   [ERROR] Build failed!
    pause
    exit /b 1
)

echo.
echo =========================================
echo   Build Complete!
echo =========================================
echo.
echo Executables:
if exist radar_tagger.exe (
    echo   [OK] radar_tagger.exe
) else (
    echo   [X] radar_tagger.exe (NOT FOUND)
)

if exist radar_tagger_multioutput.exe (
    echo   [OK] radar_tagger_multioutput.exe
) else (
    echo   [X] radar_tagger_multioutput.exe (NOT FOUND)
)

echo.
echo Location: %CD%
echo.
pause
