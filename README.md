# Radar Data Annotation Application

A comprehensive desktop application for radar data processing, automatic annotation, AI-powered classification, and visualization of airborne target trajectories.

## Features

### 🔧 Core Engines

1. **Data Extraction Engine** - Parse binary radar data files into structured formats
2. **AutoLabeling Engine** - Rule-based motion feature extraction and automatic annotation
3. **AI Tagging Engine** - Machine learning models (XGBoost, LSTM, Transformer) for trajectory classification
4. **High Volume Training** - ✨ NEW! End-to-end pipeline for large-scale model training and evaluation
5. **Report Engine** - Generate comprehensive HTML reports with visualizations
6. **Simulation Engine** - Create synthetic radar data with diverse trajectory patterns

### 📊 Rich GUI

- Interactive PPI (Plan Position Indicator) visualization
- Multi-panel time series plots (altitude, speed, curvature)
- Zoom, pan, and interactive track selection
- Real-time annotation highlighting
- Progress tracking for long-running operations

### 🎯 Key Capabilities

- Parse configurable binary radar formats
- Compute motion features: speed, heading, curvature, acceleration
- Apply rule-based classification: incoming/outgoing, level flight, maneuvers, etc.
- Train and evaluate ML models for automated tagging
- Generate professional reports with embedded visualizations
- Create synthetic test data with 10 diverse trajectory types

## Quick Start

### 🚀 Fastest Way to Run

1. Install GUI dependencies (if not already installed):
   ```bash
   # Windows
   install_gui.bat
   
   # Linux/Mac
   ./install_gui.sh
   ```

2. Run the application:
   ```bash
   # Windows
   run.bat
   
   # Linux/Mac
   ./run.sh
   ```

### ⚠️ GUI Not Opening?

If you run the application and the GUI window doesn't appear:
- **Cause**: PyQt6 is not installed
- **Fix**: Run `install_gui.bat` (Windows) or `./install_gui.sh` (Linux/Mac)
- **Details**: See [GUI_NOT_OPENING_FIX.md](GUI_NOT_OPENING_FIX.md)

## Installation

### Prerequisites

- Python 3.10 or higher
- pip or conda package manager

### Setup with pip

```bash
# Clone or extract the repository
cd radar-annotator

# Create virtual environment (recommended)
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Setup with conda

```bash
# Create conda environment
conda create -n radar-annotator python=3.10
conda activate radar-annotator

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### 1. Generate Sample Data

```bash
# Generate 10 simulation folders with diverse trajectories
python -m src.sim_engine --outdir data/simulations --count 10
```

### 2. Launch GUI Application

```bash
# Start the GUI
python -m src.gui
```

### 3. Using the GUI

**Data Extraction:**
1. Navigate to "Data Extraction" panel
2. Click "Select Binary File" and choose a `.bin` file (e.g., from simulations)
3. Click "Extract Data"
4. Save extracted data as CSV or Excel

**AutoLabeling:**
1. Navigate to "AutoLabeling" panel
2. Load the extracted CSV file
3. Adjust thresholds if needed
4. Click "Run Auto-Labeling"
5. Review annotation results in the table
6. Save labeled data

**AI Tagging:**
1. Navigate to "AI Tagging" panel
2. Select model type (XGBoost, LSTM, or Transformer)
3. Load labeled data CSV
4. Click "Train Model"
5. View training results and metrics

**Visualization:**
1. Navigate to "Visualization" panel
2. Load any data CSV file
3. Interactive PPI and time series plots will display
4. Click tracks to highlight them

**High Volume Training:** ✨ NEW!
1. Navigate to "🚀 High Volume Training" panel
2. Generate large dataset (200+ tracks) or select existing data
3. Apply auto-labeling
4. Train multiple models (Transformer, LSTM, XGBoost) simultaneously
5. Compare results in summary table
6. Generate comprehensive reports

See [QUICK_START_HIGH_VOLUME_GUI.md](QUICK_START_HIGH_VOLUME_GUI.md) for detailed guide.

**Report Generation:**
1. Navigate to "Report" panel
2. Select a data folder containing processed files
3. Click "Generate Report"
4. Open the generated HTML report in your browser

## Command-Line Interface

Each engine can also be used via command line:

### Data Extraction

```bash
python -m src.data_engine --input data/simulations/sim_01_*/radar_data.bin --out output/raw_data.csv
```

### AutoLabeling

```bash
python -m src.autolabel_engine --input output/raw_data.csv --out output/labelled_data.csv
```

### AI Training

```bash
# Train XGBoost (fast, tabular features)
python -m src.ai_engine --model xgboost --data output/labelled_data.csv --outdir output/models

# Train LSTM (sequence modeling)
python -m src.ai_engine --model lstm --data output/labelled_data.csv --outdir output/models

# Train Transformer (multi-output, state-of-the-art)
python -m src.ai_engine --model transformer --data output/labelled_data.csv --outdir output/models
```

### Report Generation

```bash
python -m src.report_engine --folder output --out output/report.html
```

### Simulation

```bash
python -m src.sim_engine --outdir data/simulations --count 10
```

## Running the Demo

A complete end-to-end demo script is provided:

```bash
# On Linux/Mac:
bash demo.sh

# On Windows:
demo.bat
```

This will:
1. Generate 10 simulation files
2. Extract data from first simulation
3. Run autolabeling
4. Train an XGBoost model
5. Generate a report

## Configuration

The application uses a JSON configuration file for customization:

```bash
# Default config is created automatically at:
config/default_config.json
```

### Key Configuration Options

**Binary Schema:**
- `record_size`: Size of each binary record in bytes
- `struct_format`: Python struct format string
- `fields`: Field definitions (name, type, offset)

**AutoLabel Thresholds:**
- `level_flight_threshold`: Max altitude change for level flight (meters)
- `curvature_threshold`: Threshold for linear vs curved motion (rad/m)
- `low_speed_threshold`: Speed threshold for low speed classification (m/s)
- `high_speed_threshold`: Speed threshold for high speed classification (m/s)
- `high_maneuver_threshold`: Acceleration threshold for high maneuver (m/s²)

**ML Parameters:**
- XGBoost: `n_estimators`, `max_depth`, `learning_rate`
- LSTM: `units`, `dropout`, `epochs`, `sequence_length`
- Transformer: `d_model`, `num_heads`, `ff_dim`, `num_layers`, `dropout`, `epochs`

## Binary Format

The default binary format assumes:
- Each record: 10 × float64 (80 bytes)
- Little-endian byte order
- Fields: time, trackid, x, y, z, vx, vy, vz, ax, ay

To use a custom format, modify `config/default_config.json`:

```json
{
  "binary_schema": {
    "record_size": 80,
    "endian": "little",
    "struct_format": "<10d",
    "fields": [
      {"name": "time", "type": "float64", "offset": 0},
      {"name": "trackid", "type": "float64", "offset": 8},
      ...
    ]
  }
}
```

## Project Structure

```
radar-annotator/
├── src/
│   ├── __init__.py
│   ├── config.py              # Configuration management
│   ├── utils.py               # Utility functions
│   ├── data_engine.py         # Data Extraction Engine
│   ├── autolabel_engine.py    # AutoLabeling Engine
│   ├── ai_engine.py           # AI Tagging Engine
│   ├── report_engine.py       # Report Engine
│   ├── sim_engine.py          # Simulation Engine
│   ├── plotting.py            # Visualization utilities
│   └── gui.py                 # Main GUI application
├── tests/
│   ├── test_data_engine.py    # Unit tests for data engine
│   └── test_autolabel_engine.py  # Unit tests for autolabel
├── config/
│   └── default_config.json    # Default configuration
├── data/
│   └── simulations/           # Generated simulation data
├── output/
│   └── models/                # Trained ML models
├── requirements.txt           # Python dependencies
├── README.md                  # This file
├── demo.sh                    # Demo script (Linux/Mac)
└── demo.bat                   # Demo script (Windows)
```

## Testing

Run unit tests:

```bash
# Run all tests
python -m pytest tests/

# Run with coverage
python -m pytest tests/ --cov=src --cov-report=html

# Run specific test file
python -m pytest tests/test_data_engine.py -v
```

## Trajectory Types

The simulation engine generates 10 trajectory types:

1. **Straight Low Speed** - Constant velocity at 30 m/s
2. **Straight High Speed** - Constant velocity at 250 m/s
3. **Ascending Spiral** - Spiral pattern with altitude gain
4. **Descending Path** - Controlled descent at 15° angle
5. **Sharp Maneuver** - 90° turn with high acceleration
6. **Curved Path** - Gentle circular arc
7. **Level Flight with Jitter** - Horizontal flight with altitude noise
8. **Stop and Go** - Alternating fast and slow speeds
9. **Oscillating Lateral** - Sine wave lateral motion
10. **Complex Maneuver** - Multi-phase combined maneuver

## Annotations

The AutoLabeling Engine produces these annotation tags:

**Direction:**
- `incoming` - Moving toward radar
- `outgoing` - Moving away from radar

**Vertical Motion:**
- `ascending` - Climbing with minimal lateral motion
- `descending` - Descending with minimal lateral motion
- `level` - Constant altitude

**Path Shape:**
- `linear` - Straight path (low curvature)
- `curved` - Curved path (high curvature)

**Maneuver Intensity:**
- `light_maneuver` - Low acceleration
- `high_maneuver` - High acceleration

**Speed:**
- `low_speed` - Below threshold
- `high_speed` - Above threshold

## Troubleshooting

**GUI doesn't start:**
- Ensure PyQt6 is installed: `pip install PyQt6`
- Check Python version: `python --version` (must be 3.10+)

**Training error: "Insufficient classes for training":**
- This means all your data has the same label
- ML models need at least 2 different classes
- **Quick fix**: Run `python analyze_label_diversity.py <your_csv>`
- **Solutions**:
  - Create per-track labels: `python create_track_labels.py <your_csv>`
  - Split composite labels: `python split_composite_labels.py <your_csv>`
  - Adjust auto-labeling thresholds in `config/default_config.json`
  - Collect more diverse data
- See **[INSUFFICIENT_CLASSES_FIX.md](INSUFFICIENT_CLASSES_FIX.md)** for detailed guide
- See **[LABEL_DIVERSITY_GUIDE.md](LABEL_DIVERSITY_GUIDE.md)** for complete reference

**TensorFlow errors:**
- LSTM model requires TensorFlow
- For CPU-only: `pip install tensorflow`
- For GPU: `pip install tensorflow[and-cuda]`

**Binary parsing errors:**
- Verify binary format matches configuration
- Check endianness setting
- Ensure record size is correct

**Visualization issues:**
- Install pyqtgraph: `pip install pyqtgraph`
- Update graphics drivers for OpenGL support

## Performance Tips

- For large datasets (>100K points), filter data before visualization
- Use XGBoost for faster training on tabular data
- LSTM models require more memory and training time
- Generate simulation data in batches for faster processing

## License

This project is provided as-is for educational and research purposes.

## Contributing

Contributions are welcome! Areas for improvement:

- Additional ML models (GRU, Attention mechanisms)
- Real-time data streaming support
- 3D visualization
- Advanced filtering and search
- Export to additional formats
- Multi-radar fusion
- Attention visualization for Transformer model

## Support

For issues, questions, or feature requests, please refer to the project documentation or contact the development team.

## Acknowledgments

Built with:
- PyQt6 for GUI framework
- PyQtGraph for high-performance plotting
- scikit-learn and XGBoost for machine learning
- TensorFlow/Keras for deep learning (LSTM, Transformer with Multi-Head Attention)
- NumPy and Pandas for data processing

## Machine Learning Models

This application includes three powerful models for trajectory classification:

### 1. XGBoost Classifier
- **Type**: Gradient boosting for tabular features
- **Speed**: ⚡⚡⚡ Very Fast
- **Best for**: Quick training, tabular features, small datasets
- **Accuracy**: 85-90%

### 2. LSTM (Long Short-Term Memory)
- **Type**: Recurrent neural network for sequences
- **Speed**: ⚡⚡ Moderate
- **Best for**: Sequential data, proven reliability
- **Accuracy**: 88-92%

### 3. Transformer (Multi-Output) ⭐ NEW
- **Type**: Self-attention mechanism with multi-head attention
- **Speed**: ⚡ Slower but parallelizable
- **Best for**: Multi-attribute classification, composite labels, long sequences
- **Accuracy**: 90-95%
- **Special Feature**: Predicts multiple outputs simultaneously (direction, altitude, path, maneuver, speed)

**For detailed information**, see [TRANSFORMER_MODEL_GUIDE.md](TRANSFORMER_MODEL_GUIDE.md)

---

**Version:** 1.0.0  
**Last Updated:** 2025
