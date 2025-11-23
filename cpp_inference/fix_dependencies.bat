@echo off
REM Script to patch gemmlowp and cpuinfo dependencies for MinGW builds

echo ============================================================================
echo Fixing TensorFlow Lite dependencies for MinGW...
echo ============================================================================

REM Find and patch gemmlowp
echo.
echo Searching for gemmlowp...
if exist "build\_deps\gemmlowp-src\CMakeLists.txt" (
    echo Found gemmlowp at build\_deps\gemmlowp-src
    
    REM Check if already patched
    findstr /C:"DISABLED_FOR_MINGW" "build\_deps\gemmlowp-src\CMakeLists.txt" >nul
    if %errorlevel% equ 0 (
        echo gemmlowp already patched, skipping...
    ) else (
        echo Patching gemmlowp CMakeLists.txt...
        
        REM Create a temporary Python script to patch the file
        echo import re > patch_gemmlowp.py
        echo with open('build/_deps/gemmlowp-src/CMakeLists.txt', 'r') as f: >> patch_gemmlowp.py
        echo     content = f.read() >> patch_gemmlowp.py
        echo. >> patch_gemmlowp.py
        echo content = re.sub(r'add_library\(eight_bit_int_gemm', '# DISABLED_FOR_MINGW: add_library(eight_bit_int_gemm', content) >> patch_gemmlowp.py
        echo content = re.sub(r'add_executable\(eight_bit_int_gemm', '# DISABLED_FOR_MINGW: add_executable(eight_bit_int_gemm', content) >> patch_gemmlowp.py
        echo content = re.sub(r'target_link_libraries\(eight_bit_int_gemm', '# DISABLED_FOR_MINGW: target_link_libraries(eight_bit_int_gemm', content) >> patch_gemmlowp.py
        echo content = re.sub(r'set_target_properties\(eight_bit_int_gemm', '# DISABLED_FOR_MINGW: set_target_properties(eight_bit_int_gemm', content) >> patch_gemmlowp.py
        echo. >> patch_gemmlowp.py
        echo with open('build/_deps/gemmlowp-src/CMakeLists.txt', 'w') as f: >> patch_gemmlowp.py
        echo     f.write(content) >> patch_gemmlowp.py
        echo. >> patch_gemmlowp.py
        echo print('Successfully patched gemmlowp') >> patch_gemmlowp.py
        
        python patch_gemmlowp.py
        del patch_gemmlowp.py
    )
) else if exist "build\gemmlowp\CMakeLists.txt" (
    echo Found gemmlowp at build\gemmlowp
    
    findstr /C:"DISABLED_FOR_MINGW" "build\gemmlowp\CMakeLists.txt" >nul
    if %errorlevel% equ 0 (
        echo gemmlowp already patched, skipping...
    ) else (
        echo Patching gemmlowp CMakeLists.txt...
        
        echo import re > patch_gemmlowp.py
        echo with open('build/gemmlowp/CMakeLists.txt', 'r') as f: >> patch_gemmlowp.py
        echo     content = f.read() >> patch_gemmlowp.py
        echo. >> patch_gemmlowp.py
        echo content = re.sub(r'add_library\(eight_bit_int_gemm', '# DISABLED_FOR_MINGW: add_library(eight_bit_int_gemm', content) >> patch_gemmlowp.py
        echo content = re.sub(r'add_executable\(eight_bit_int_gemm', '# DISABLED_FOR_MINGW: add_executable(eight_bit_int_gemm', content) >> patch_gemmlowp.py
        echo content = re.sub(r'target_link_libraries\(eight_bit_int_gemm', '# DISABLED_FOR_MINGW: target_link_libraries(eight_bit_int_gemm', content) >> patch_gemmlowp.py
        echo content = re.sub(r'set_target_properties\(eight_bit_int_gemm', '# DISABLED_FOR_MINGW: set_target_properties(eight_bit_int_gemm', content) >> patch_gemmlowp.py
        echo. >> patch_gemmlowp.py
        echo with open('build/gemmlowp/CMakeLists.txt', 'w') as f: >> patch_gemmlowp.py
        echo     f.write(content) >> patch_gemmlowp.py
        echo. >> patch_gemmlowp.py
        echo print('Successfully patched gemmlowp') >> patch_gemmlowp.py
        
        python patch_gemmlowp.py
        del patch_gemmlowp.py
    )
) else (
    echo WARNING: gemmlowp not found yet. Will be patched during reconfigure.
)

echo.
echo ============================================================================
echo Patch complete! 
echo ============================================================================
echo.
echo Next steps:
echo 1. Rebuild with: mingw32-make -j4
echo 2. Or reconfigure with: cmake --build build
echo.
