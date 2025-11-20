@echo off
REM Installation script for GUI dependencies (Windows)

echo ================================================================================
echo  Radar Data Annotation Application - GUI Installation
echo ================================================================================
echo.
echo This script will install the required GUI packages:
echo   - PyQt6 (GUI framework)
echo   - pyqtgraph (Plotting library)
echo.
echo If you want to install ALL project dependencies, use:
echo   pip install -r requirements.txt
echo.
pause

echo.
echo Installing PyQt6 and pyqtgraph...
echo.

pip install PyQt6 pyqtgraph

if errorlevel 1 (
    echo.
    echo ================================================================================
    echo  ERROR: Installation failed!
    echo ================================================================================
    echo.
    echo Please check that:
    echo   1. Python and pip are properly installed
    echo   2. You have an active internet connection
    echo   3. You have proper permissions to install packages
    echo.
    echo Try running this command manually:
    echo   pip install PyQt6 pyqtgraph
    echo.
    pause
    exit /b 1
)

echo.
echo ================================================================================
echo  SUCCESS! GUI packages installed successfully
echo ================================================================================
echo.
echo You can now run the application with:
echo   run.bat
echo.
echo Or verify the installation with:
echo   python -c "from PyQt6.QtWidgets import QApplication; print('PyQt6 OK')"
echo.
pause
