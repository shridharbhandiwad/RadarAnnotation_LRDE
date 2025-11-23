@echo off
REM Complete build script with automatic dependency patching

echo ============================================================================
echo Radar Tagger C++ Build Script with Automatic Fixes
echo ============================================================================
echo.

REM Step 1: Clean build directory if requested
if "%1"=="clean" (
    echo Cleaning build directory...
    if exist build rmdir /s /q build
    mkdir build
    echo Clean complete.
    echo.
)

REM Step 2: Configure with CMake
echo ============================================================================
echo Step 1: Configuring project with CMake...
echo ============================================================================
cd build
cmake -G "MinGW Makefiles" -DCMAKE_BUILD_TYPE=Release ..
if %errorlevel% neq 0 (
    echo ERROR: CMake configuration failed!
    cd ..
    exit /b 1
)
cd ..

REM Step 3: Apply patches to dependencies
echo.
echo ============================================================================
echo Step 2: Applying dependency fixes...
echo ============================================================================
call fix_dependencies.bat

REM Step 4: Build the project
echo.
echo ============================================================================
echo Step 3: Building project...
echo ============================================================================
cd build
mingw32-make -j4
if %errorlevel% neq 0 (
    echo.
    echo ERROR: Build failed!
    echo.
    echo Trying to apply fixes and rebuild...
    cd ..
    call fix_dependencies.bat
    cd build
    mingw32-make -j4
    if %errorlevel% neq 0 (
        echo.
        echo ERROR: Build still failing after fixes!
        cd ..
        exit /b 1
    )
)
cd ..

echo.
echo ============================================================================
echo Build complete!
echo ============================================================================
echo.
echo Executables:
echo   - build\radar_tagger.exe
echo   - build\radar_tagger_multioutput.exe
echo.
