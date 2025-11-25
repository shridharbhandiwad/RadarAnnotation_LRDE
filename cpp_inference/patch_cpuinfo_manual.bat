@echo off
REM Manual patch script for cpuinfo max/min macro issue
REM Run this if the automatic patch doesn't work

echo ===== Manual cpuinfo Patch for Windows/MinGW =====
echo.

REM Check if build directory exists
if not exist "build" (
    echo ERROR: build directory not found!
    echo Please run cmake first to populate dependencies.
    pause
    exit /b 1
)

REM Find cpuinfo directory
set CPUINFO_DIR=
if exist "build\_deps\cpuinfo-src" set CPUINFO_DIR=build\_deps\cpuinfo-src
if exist "build\cpuinfo" set CPUINFO_DIR=build\cpuinfo

if "%CPUINFO_DIR%"=="" (
    echo ERROR: cpuinfo directory not found!
    echo Expected at: build\_deps\cpuinfo-src or build\cpuinfo
    echo Make sure TensorFlow Lite has been downloaded by running cmake.
    pause
    exit /b 1
)

echo Found cpuinfo at: %CPUINFO_DIR%
echo.

REM Patch CMakeLists.txt
set CMAKE_FILE=%CPUINFO_DIR%\CMakeLists.txt
if not exist "%CMAKE_FILE%" (
    echo ERROR: CMakeLists.txt not found at %CMAKE_FILE%
    pause
    exit /b 1
)

echo Checking if already patched...
findstr /C:"PATCHED_MAX_MIN_MACROS" "%CMAKE_FILE%" >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo CMakeLists.txt already patched!
    echo.
    goto patch_source
)

echo Patching CMakeLists.txt...
echo.

REM Create a temporary patch file
set PATCH_FILE=%TEMP%\cpuinfo_patch.txt
(
echo.
echo # PATCHED_MAX_MIN_MACROS: Add max/min macros for Windows compatibility
echo if^(WIN32 OR MINGW^)
echo     add_compile_definitions^(max^(a,b^)=^(^(a^)^>^(b^)?^(a^):^(b^)^)^)
echo     add_compile_definitions^(min^(a,b^)=^(^(a^)^<^(b^)?^(a^):^(b^)^)^)
echo     message^(STATUS "cpuinfo: Added max/min macro definitions for Windows/MinGW"^)
echo endif^(^)
echo.
) > "%PATCH_FILE%"

REM Find project() line and insert patch after it
powershell -Command "$content = Get-Content '%CMAKE_FILE%' -Raw; $projectPos = $content.IndexOf('project('); if ($projectPos -eq -1) { $projectPos = $content.IndexOf('cmake_minimum_required'); } if ($projectPos -ne -1) { $newlinePos = $content.IndexOf(\"`n\", $projectPos); if ($newlinePos -ne -1) { $insertPos = $newlinePos + 1; $patchContent = Get-Content '%PATCH_FILE%' -Raw; $before = $content.Substring(0, $insertPos); $after = $content.Substring($insertPos); $newContent = $before + $patchContent + $after; Set-Content -Path '%CMAKE_FILE%' -Value $newContent; Write-Host 'Successfully patched CMakeLists.txt'; } else { Write-Host 'ERROR: Could not find insertion point'; exit 1; } } else { Write-Host 'ERROR: Could not find project() or cmake_minimum_required'; exit 1; }"

if %ERRORLEVEL% NEQ 0 (
    echo Failed to patch CMakeLists.txt
    pause
    exit /b 1
)

:patch_source
REM Also patch the source file directly
set INIT_C_FILE=%CPUINFO_DIR%\src\x86\windows\init.c
if not exist "%INIT_C_FILE%" (
    echo WARNING: init.c not found at %INIT_C_FILE%
    echo This is expected if cpuinfo hasn't been fully populated yet.
    goto done
)

echo.
echo Checking init.c source file...
findstr /C:"PATCHED_MAX_MIN_SOURCE" "%INIT_C_FILE%" >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo init.c already patched!
    goto done
)

echo Patching init.c source file...

REM Create source file patch
set SOURCE_PATCH_FILE=%TEMP%\cpuinfo_source_patch.txt
(
echo /* PATCHED_MAX_MIN_SOURCE: Add max/min macros for Windows */
echo #ifndef max
echo #define max^(a,b^) ^(^(^(a^) ^> ^(b^)^) ? ^(a^) : ^(b^)^)
echo #endif
echo #ifndef min
echo #define min^(a,b^) ^(^(^(a^) ^< ^(b^)^) ? ^(a^) : ^(b^)^)
echo #endif
echo.
) > "%SOURCE_PATCH_FILE%"

REM Insert patch before first #include
powershell -Command "$content = Get-Content '%INIT_C_FILE%' -Raw; $includePos = $content.IndexOf('#include'); if ($includePos -ne -1) { $patchContent = Get-Content '%SOURCE_PATCH_FILE%' -Raw; $before = $content.Substring(0, $includePos); $after = $content.Substring($includePos); $newContent = $before + $patchContent + $after; Set-Content -Path '%INIT_C_FILE%' -Value $newContent; Write-Host 'Successfully patched init.c'; } else { Write-Host 'WARNING: Could not find #include in init.c'; }"

:done
echo.
echo ===== Patch complete! =====
echo.
echo Next steps:
echo 1. If you haven't configured yet, run: cmake -G "MinGW Makefiles" ..
echo 2. Build the project: mingw32-make
echo.
echo The max/min macro definitions have been added to cpuinfo.
echo This should fix the "implicit declaration of function 'max'" error.
echo.
pause
