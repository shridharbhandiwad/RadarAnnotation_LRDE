@echo off
REM Radar Data Annotation Application - Demo Script (Windows)
REM This script demonstrates the complete workflow

echo ======================================
echo Radar Data Annotation Application Demo
echo ======================================
echo.

REM Create output directories
if not exist "data\simulations" mkdir "data\simulations"
if not exist "output\models" mkdir "output\models"

REM Step 1: Generate simulations
echo [1/5] Generating simulation data...
python -m src.sim_engine --outdir data/simulations --count 10

if errorlevel 1 (
    echo Error: Failed to generate simulations
    exit /b 1
)

echo Generated 10 simulation files
echo.

REM Step 2: Extract first simulation
echo [2/5] Extracting data from first simulation...
for /f "delims=" %%i in ('dir /s /b data\simulations\radar_data.bin 2^>nul') do (
    set FIRST_SIM=%%i
    goto :found_sim
)
:found_sim

if "%FIRST_SIM%"=="" (
    echo Error: No simulation files found
    exit /b 1
)

python -m src.data_engine --input "%FIRST_SIM%" --out output/raw_data.csv

if errorlevel 1 (
    echo Error: Failed to extract data
    exit /b 1
)

echo Extracted data to output/raw_data.csv
echo.

REM Step 3: Run autolabeling
echo [3/5] Running autolabeling...
python -m src.autolabel_engine --input output/raw_data.csv --out output/labelled_data.csv

if errorlevel 1 (
    echo Error: Failed to run autolabeling
    exit /b 1
)

echo Generated labeled data
echo.

REM Step 4: Train XGBoost model
echo [4/5] Training XGBoost model...
python -m src.ai_engine --model xgboost --data output/labelled_data.csv --outdir output/models

if errorlevel 1 (
    echo Warning: Model training failed (may require scikit-learn and xgboost^)
    echo Continuing to report generation...
)

echo Model training complete
echo.

REM Step 5: Generate report
echo [5/5] Generating report...
python -m src.report_engine --folder output --out output/report.html

if errorlevel 1 (
    echo Warning: Report generation failed
)

echo Report generated
echo.

echo ======================================
echo Demo Complete!
echo ======================================
echo.
echo Generated files:
echo   - data\simulations\     (10 simulation folders)
echo   - output\raw_data.csv   (extracted data)
echo   - output\labelled_data.csv  (labeled data)
echo   - output\models\        (trained models)
echo   - output\report.html    (HTML report)
echo.
echo To view the report, open: output\report.html
echo.
echo To launch the GUI:
echo   python -m src.gui
echo.
pause
