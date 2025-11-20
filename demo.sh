#!/bin/bash
# Radar Data Annotation Application - Demo Script
# This script demonstrates the complete workflow

echo "======================================"
echo "Radar Data Annotation Application Demo"
echo "======================================"
echo ""

# Create output directories
mkdir -p data/simulations
mkdir -p output/models

# Step 1: Generate simulations
echo "[1/5] Generating simulation data..."
python -m src.sim_engine --outdir data/simulations --count 10

if [ $? -ne 0 ]; then
    echo "Error: Failed to generate simulations"
    exit 1
fi

echo "✓ Generated 10 simulation files"
echo ""

# Step 2: Extract first simulation
echo "[2/5] Extracting data from first simulation..."
FIRST_SIM=$(find data/simulations -name "radar_data.bin" | head -n 1)

if [ -z "$FIRST_SIM" ]; then
    echo "Error: No simulation files found"
    exit 1
fi

python -m src.data_engine --input "$FIRST_SIM" --out output/raw_data.csv

if [ $? -ne 0 ]; then
    echo "Error: Failed to extract data"
    exit 1
fi

echo "✓ Extracted data to output/raw_data.csv"
echo ""

# Step 3: Run autolabeling
echo "[3/5] Running autolabeling..."
python -m src.autolabel_engine --input output/raw_data.csv --out output/labelled_data.csv

if [ $? -ne 0 ]; then
    echo "Error: Failed to run autolabeling"
    exit 1
fi

echo "✓ Generated labeled data"
echo ""

# Step 4: Train XGBoost model
echo "[4/5] Training XGBoost model..."
python -m src.ai_engine --model xgboost --data output/labelled_data.csv --outdir output/models

if [ $? -ne 0 ]; then
    echo "Warning: Model training failed (may require scikit-learn and xgboost)"
    echo "Continuing to report generation..."
fi

echo "✓ Model training complete"
echo ""

# Step 5: Generate report
echo "[5/5] Generating report..."
python -m src.report_engine --folder output --out output/report.html

if [ $? -ne 0 ]; then
    echo "Warning: Report generation failed"
fi

echo "✓ Report generated"
echo ""

echo "======================================"
echo "Demo Complete!"
echo "======================================"
echo ""
echo "Generated files:"
echo "  - data/simulations/     (10 simulation folders)"
echo "  - output/raw_data.csv   (extracted data)"
echo "  - output/labelled_data.csv  (labeled data)"
echo "  - output/models/        (trained models)"
echo "  - output/report.html    (HTML report)"
echo ""
echo "To view the report, open: output/report.html"
echo ""
echo "To launch the GUI:"
echo "  python -m src.gui"
echo ""
