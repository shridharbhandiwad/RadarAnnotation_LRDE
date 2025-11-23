@echo off
REM ========================================================================
REM Windows MinGW Build Script with eight_bit_int_gemm Fix
REM ========================================================================
REM This script fixes the eight_bit_int_gemm compilation error and builds
REM the project on Windows with MinGW.
REM
REM USAGE:
REM   fix_and_build_windows.bat [clean]
REM
REM OPTIONS:
REM   clean - Remove build directory before building
REM ========================================================================

echo.
echo ========================================================================
echo   Radar Tagger C++ - Windows MinGW Build with eight_bit_int_gemm Fix
echo ========================================================================
echo.

REM Check if MinGW is available
where mingw32-make >nul 2>&1
if errorlevel 1 (
    echo ERROR: mingw32-make not found in PATH
    echo Please install MinGW-w64 and add it to your PATH
    echo Download from: https://sourceforge.net/projects/mingw-w64/
    exit /b 1
)

where cmake >nul 2>&1
if errorlevel 1 (
    echo ERROR: cmake not found in PATH
    echo Please install CMake and add it to your PATH
    echo Download from: https://cmake.org/download/
    exit /b 1
)

REM Parse command line arguments
if "%1"=="clean" (
    echo Cleaning build directory...
    if exist build (
        rmdir /s /q build
    )
    echo Build directory cleaned.
    echo.
)

REM Create build directory
if not exist build (
    mkdir build
)

cd build

REM Step 1: Configure with CMake
echo ========================================================================
echo Step 1/4: Configuring CMake...
echo ========================================================================
echo.
cmake -G "MinGW Makefiles" -DCMAKE_BUILD_TYPE=Release ..
if errorlevel 1 (
    echo.
    echo ERROR: CMake configuration failed
    cd ..
    exit /b 1
)

REM Step 2: Apply gemmlowp patch if needed
echo.
echo ========================================================================
echo Step 2/4: Checking for gemmlowp and applying patch if needed...
echo ========================================================================
echo.

if exist gemmlowp\CMakeLists.txt (
    echo Found gemmlowp CMakeLists.txt
    findstr /C:"DISABLED_FOR_MINGW" gemmlowp\CMakeLists.txt >nul 2>&1
    if errorlevel 1 (
        echo Applying patch to disable eight_bit_int_gemm target...
        
        REM Backup original file
        copy /y gemmlowp\CMakeLists.txt gemmlowp\CMakeLists.txt.bak >nul
        
        REM Apply patch using PowerShell
        powershell -Command "$content = Get-Content 'gemmlowp\CMakeLists.txt' -Raw; $content = $content -replace 'add_library\(eight_bit_int_gemm', '# DISABLED_FOR_MINGW: add_library(eight_bit_int_gemm'; $content = $content -replace 'add_executable\(eight_bit_int_gemm', '# DISABLED_FOR_MINGW: add_executable(eight_bit_int_gemm'; $content = $content -replace 'target_link_libraries\(eight_bit_int_gemm', '# DISABLED_FOR_MINGW: target_link_libraries(eight_bit_int_gemm'; $content = $content -replace 'set_target_properties\(eight_bit_int_gemm', '# DISABLED_FOR_MINGW: set_target_properties(eight_bit_int_gemm'; Set-Content 'gemmlowp\CMakeLists.txt' $content"
        
        if errorlevel 1 (
            echo ERROR: Failed to apply patch
            cd ..
            exit /b 1
        )
        
        echo Patch applied successfully!
        echo.
        echo Re-running CMake configuration...
        cmake -G "MinGW Makefiles" -DCMAKE_BUILD_TYPE=Release ..
        if errorlevel 1 (
            echo ERROR: CMake re-configuration failed
            cd ..
            exit /b 1
        )
    ) else (
        echo Patch already applied, skipping...
    )
) else (
    echo gemmlowp not yet downloaded. Patch will be applied automatically during build.
)

REM Step 3: Build
echo.
echo ========================================================================
echo Step 3/4: Building project...
echo ========================================================================
echo.
echo This may take several minutes on first build...
echo.

cmake --build . --config Release -- -j4
if errorlevel 1 (
    echo.
    echo ========================================================================
    echo Build failed!
    echo ========================================================================
    echo.
    echo If you see errors about eight_bit_int_gemm:
    echo   1. Run: fix_and_build_windows.bat clean
    echo   2. This will clean and rebuild from scratch with the patch
    echo.
    cd ..
    exit /b 1
)

REM Step 4: Verify build
echo.
echo ========================================================================
echo Step 4/4: Verifying build...
echo ========================================================================
echo.

if exist radar_tagger.exe (
    echo [SUCCESS] radar_tagger.exe built successfully
) else (
    echo [WARNING] radar_tagger.exe not found
)

if exist radar_tagger_multioutput.exe (
    echo [SUCCESS] radar_tagger_multioutput.exe built successfully  
) else (
    echo [WARNING] radar_tagger_multioutput.exe not found
)

echo.
echo ========================================================================
echo BUILD COMPLETE!
echo ========================================================================
echo.
echo Executables are located in: cpp_inference\build\
echo.
echo To run:
echo   cd build
echo   .\radar_tagger.exe
echo   .\radar_tagger_multioutput.exe
echo.

cd ..
exit /b 0
