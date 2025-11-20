# Quick Start Guide

## Installation (5 minutes)

### Option 1: Using pip (Recommended)

```bash
# Install Python 3.10+ if not already installed
# Download from: https://www.python.org/downloads/

# Navigate to project directory
cd radar-annotator

# Install all dependencies
pip install -r requirements.txt
```

### Option 2: Using conda

```bash
conda create -n radar-annotator python=3.10
conda activate radar-annotator
pip install -r requirements.txt
```

## First Run (2 minutes)

### 1. Generate Sample Data

```bash
python -m src.sim_engine --outdir data/simulations --count 3
```

This creates 3 simulation folders with sample radar data.

### 2. Launch the Application

```bash
python -m src.gui
```

Or use the convenience script:
- Linux/Mac: `./run.sh`
- Windows: `run.bat`

## Using the GUI (5 minutes)

### Extract Data

1. Click **"Data Extraction"** in left panel
2. Click **"Select Binary File"**
3. Navigate to `data/simulations/sim_01_*/radar_data.bin`
4. Click **"Extract Data"**
5. Click **"Save Extracted Data"** → Save as `test_data.csv`

### Auto-Label Data

1. Click **"AutoLabeling"** in left panel
2. Click **"Select CSV File"** → Choose `test_data.csv`
3. Click **"Run Auto-Labeling"**
4. Review the annotation results table
5. Click **"Save Labeled Data"** → Save as `labeled_data.csv`

### Visualize Data

1. Click **"Visualization"** in left panel
2. Click **"Load Data for Visualization"**
3. Choose `labeled_data.csv`
4. Explore interactive plots:
   - **PPI Plot**: Polar view of tracks
   - **Time Series**: Altitude, speed, curvature over time
   - Use mouse to zoom and pan

### Generate Report

1. Click **"Report"** in left panel
2. Click **"Select Data Folder"**
3. Choose a folder containing your processed CSV files
4. Click **"Generate Report"**
5. Open the generated `report.html` in your browser

## Running the Full Demo

To see the complete workflow in action:

```bash
# Linux/Mac:
./demo.sh

# Windows:
demo.bat
```

This will:
- Generate 10 simulations
- Extract and label data
- Train an ML model
- Generate a report

Total time: ~2-3 minutes

## Common Commands

### Generate more simulations
```bash
python -m src.sim_engine --outdir data/simulations --count 10
```

### Extract binary data
```bash
python -m src.data_engine --input path/to/file.bin --out output.csv
```

### Auto-label data
```bash
python -m src.autolabel_engine --input raw.csv --out labeled.csv
```

### Train ML model
```bash
python -m src.ai_engine --model xgboost --data labeled.csv --outdir models/
```

### Generate report
```bash
python -m src.report_engine --folder output/ --out report.html
```

## Troubleshooting

### "Module not found" errors
```bash
pip install -r requirements.txt
```

### GUI doesn't start
```bash
pip install PyQt6 pyqtgraph
```

### TensorFlow errors (for LSTM)
- LSTM models are optional
- Use XGBoost model instead: faster and no TensorFlow required
- Or install TensorFlow: `pip install tensorflow`

### Can't find simulation files
- Ensure you ran the simulation engine first
- Check `data/simulations/` directory exists

## Next Steps

1. **Customize Configuration**: Edit `config/default_config.json`
2. **Add Your Data**: Place binary files in a folder and extract
3. **Tune Thresholds**: Adjust autolabeling thresholds in GUI
4. **Train Models**: Experiment with XGBoost and LSTM models
5. **Export Reports**: Generate professional reports for analysis

## Tips

- Start with small datasets (1-3 simulations) for faster testing
- Use XGBoost for quick model training
- The visualization panel supports zoom (mouse wheel) and pan (drag)
- Reports include embedded plots - no external files needed
- Each simulation type represents a different flight pattern

## Need Help?

Refer to the full README.md for:
- Detailed feature descriptions
- Configuration options
- Advanced usage
- API documentation
- Troubleshooting guide

---

**Time to fully operational**: ~15 minutes from fresh install
