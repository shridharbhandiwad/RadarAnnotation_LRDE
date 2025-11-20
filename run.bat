@echo off
REM Quick launch script for Radar Data Annotation Application (Windows)

echo ================================================================================
echo  Radar Data Annotation Application
echo ================================================================================
echo.
echo Starting application...
echo.
python -m src.gui
if errorlevel 1 (
    echo.
    echo *** Application exited with an error ***
    echo.
)
pause
