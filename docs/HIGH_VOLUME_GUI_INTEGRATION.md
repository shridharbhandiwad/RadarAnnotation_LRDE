# High Volume Data Training - GUI Integration

## Overview

The GUI now includes a comprehensive **High Volume Training** panel that integrates all high-volume data processing, model training, evaluation, and reporting in a single workflow.

## Features

### 🚀 Complete End-to-End Workflow

The High Volume Training panel provides a streamlined 3-step process:

1. **Dataset Generation** - Create large-scale simulation datasets
2. **Auto-Labeling** - Apply intelligent rule-based annotations
3. **Model Training** - Train and compare multiple AI models

### 📊 Integrated Components

- **Dataset Generation**
  - Generate 10-500 trajectory tracks
  - Configurable duration (1-30 minutes per track)
  - Diverse trajectory patterns (straight, spiral, maneuvers, etc.)
  - Supports existing high-volume data files

- **Auto-Labeling**
  - Automatic motion feature computation
  - Rule-based classification
  - Composite label generation
  - Label diversity analysis

- **Multi-Model Training**
  - Transformer (attention-based, multi-output capable)
  - LSTM (sequence modeling)
  - XGBoost (gradient boosting)
  - Parallel training with automatic recovery
  - Label transformation for insufficient diversity

- **Results Comparison**
  - Side-by-side model comparison table
  - Training accuracy, test accuracy, F1 score
  - Training time analysis
  - Automatic best model identification

- **Report Generation**
  - Comprehensive HTML reports
  - Visualization plots (PPI, altitude, speed)
  - Confusion matrices
  - Classification reports
  - Export results to JSON

## Usage

### Accessing the Panel

1. Launch the GUI:
   ```bash
   python -m src.gui
   ```

2. Select **"🚀 High Volume Training"** from the left panel

### Step-by-Step Guide

#### Step 1: Generate Dataset

1. Configure dataset parameters:
   - **Number of Tracks**: 10-500 (default: 200)
   - **Duration**: 1-30 minutes per track (default: 10)

2. Click **"Generate Dataset"**
   - Progress bar shows generation status
   - Dataset saved to `data/high_volume_simulation.csv`
   - Summary shows tracks, records, and duration

**Alternative**: Click **"Or Select Existing CSV File"** to use existing data

#### Step 2: Apply Auto-Labeling

1. Click **"Apply Auto-Labeling"**
   - Computes motion features (speed, heading, curvature, etc.)
   - Applies rule-based classification
   - Generates composite labels

2. Review labeling results:
   - Valid record count
   - Number of unique annotations
   - Top 5 most common annotations

3. Labeled data saved to `data/high_volume_simulation_labeled.csv`

#### Step 3: Train Models

1. Select models to train:
   - **🧠 Transformer** (default: ON) - Advanced attention mechanism
   - **🔁 LSTM** (default: ON) - Sequence modeling
   - **🚀 XGBoost** (default: OFF) - Fast gradient boosting

2. Click **"Train Selected Models"**
   - Training runs in background (non-blocking)
   - Progress bar shows activity
   - Each model trains independently

3. View results in the **Results Summary** table:
   - Model name
   - Train accuracy
   - Test accuracy
   - F1 score
   - Training time

4. Status log shows detailed metrics for each model
5. Best model automatically identified with 🏆 indicator

#### Step 4: Generate Reports and Export

**Generate HTML Report:**
1. Click **"📄 Generate Report"**
2. Comprehensive report saved to `output/high_volume_training_report.html`
3. Option to open in browser immediately

**Export Results:**
1. Click **"💾 Export Results"**
2. Save training metrics to JSON file
3. Useful for analysis, comparison, and documentation

## Technical Details

### Data Flow

```
Raw Data → Feature Computation → Auto-Labeling → Model Training → Evaluation → Report
```

### Files Generated

```
data/
├── high_volume_simulation.csv          # Raw trajectory data
└── high_volume_simulation_labeled.csv  # Labeled data with annotations

output/
├── models/
│   ├── transformer_highvolume/
│   │   ├── transformer_model.h5
│   │   └── transformer_metrics.json
│   ├── lstm_highvolume/
│   │   ├── lstm_model.h5
│   │   └── lstm_metrics.json
│   └── xgboost_highvolume/
│       ├── xgboost_model.pkl
│       └── xgboost_metrics.json
└── high_volume_training_report.html
```

### Model Specifications

#### Transformer Model
- **Architecture**: Multi-head attention with positional encoding
- **Supports**: Multi-output prediction (direction, altitude, path, maneuver, speed)
- **Best for**: Complex temporal patterns, composite labels
- **Parameters**:
  - d_model: 128
  - num_heads: 8
  - num_layers: 3
  - sequence_length: 30

#### LSTM Model
- **Architecture**: Bidirectional LSTM with dropout
- **Best for**: Sequential pattern recognition
- **Parameters**:
  - units: 128
  - dropout: 0.3
  - sequence_length: 30

#### XGBoost Model
- **Architecture**: Gradient boosting decision trees
- **Best for**: Fast training, tabular features
- **Parameters**:
  - n_estimators: 200
  - max_depth: 8
  - learning_rate: 0.1

### Automatic Label Recovery

The system includes automatic label diversity recovery:

1. **Detection**: Identifies insufficient label diversity
2. **Analysis**: Analyzes label distribution
3. **Transformation**: Applies appropriate transformation strategy:
   - Per-track labels
   - Binary decomposition
   - Feature-based labels
4. **Retry**: Automatically retries training with transformed data

### Thread Safety

All long-running operations use `WorkerThread`:
- Non-blocking UI
- Progress indication
- Error handling
- Graceful cancellation

## Best Practices

### Dataset Size

- **Small test**: 10-50 tracks, 2-5 minutes
- **Medium**: 100-150 tracks, 5-10 minutes
- **Large**: 200+ tracks, 10+ minutes
- **Production**: 500+ tracks, 15-30 minutes

### Model Selection

- **Start with Transformer + LSTM** for comprehensive comparison
- **Add XGBoost** if you need fast baseline
- **Transformer** excels with composite labels
- **LSTM** good for pure sequence modeling
- **XGBoost** fastest training, good for simple patterns

### Performance Tips

1. Generate dataset once, reuse for multiple experiments
2. Save labeled data to skip re-labeling
3. Train models in parallel (already automated)
4. Export results for offline analysis
5. Generate reports after all training complete

## Troubleshooting

### Issue: "No dataset available"
**Solution**: Complete Step 1 or select existing CSV file

### Issue: "No labeled data available"
**Solution**: Complete Step 2 (Auto-Labeling)

### Issue: Model training fails with "Insufficient classes"
**Solution**: Automatic recovery should handle this. If not:
- Check data diversity
- Verify labeling produced varied annotations
- Try different dataset parameters

### Issue: Training takes too long
**Solution**: 
- Reduce number of tracks
- Reduce duration per track
- Disable XGBoost (slower than deep learning models)
- Use smaller sequence_length

### Issue: Out of memory
**Solution**:
- Reduce batch_size in model parameters
- Reduce sequence_length
- Generate smaller dataset
- Train models one at a time

## Integration with Other Panels

The High Volume Training panel complements existing panels:

- **📊 Data Extraction**: For binary radar data
- **🏷️ AutoLabeling**: Individual labeling control
- **🤖 AI Tagging**: Single model training
- **📈 Report**: Detailed reporting
- **🔬 Simulation**: Basic simulation
- **📉 Visualization**: Interactive data viewing

## API Usage

You can also use the high-volume training programmatically:

```python
from src.sim_engine import create_large_training_dataset
from src.autolabel_engine import compute_motion_features, apply_rules_and_flags
from src.ai_engine import train_model

# Generate dataset
csv_path = create_large_training_dataset(
    output_path="data/my_dataset.csv",
    n_tracks=200,
    duration_min=10
)

# Label data
df = pd.read_csv(csv_path)
df = compute_motion_features(df)
df = apply_rules_and_flags(df)
df.to_csv("data/my_dataset_labeled.csv", index=False)

# Train models
models = ['transformer', 'lstm', 'xgboost']
results = {}

for model_name in models:
    model, metrics = train_model(
        model_name,
        "data/my_dataset_labeled.csv",
        f"output/models/{model_name}",
        auto_transform=True
    )
    results[model_name] = metrics

# Compare results
for name, metrics in results.items():
    print(f"{name}: {metrics['test']['accuracy']:.4f}")
```

## Advanced Features

### Custom Parameters

You can customize training parameters by modifying `config/default_config.json`:

```json
{
  "ml_params": {
    "transformer": {
      "epochs": 100,
      "batch_size": 64,
      "d_model": 128,
      "num_heads": 8
    },
    "lstm": {
      "epochs": 100,
      "batch_size": 64,
      "units": 128
    },
    "xgboost": {
      "n_estimators": 200,
      "max_depth": 8
    }
  }
}
```

### Multi-Output Predictions

The Transformer model automatically detects composite labels and uses multi-output architecture:

- **Direction**: incoming/outgoing (binary)
- **Altitude**: ascending/descending/level (3-class)
- **Path**: linear/curved (binary)
- **Maneuver**: light/high (binary)
- **Speed**: low/high (binary)

Results show accuracy and F1 score for each output independently.

## Future Enhancements

Potential additions:
- Real-time training progress visualization
- Hyperparameter tuning interface
- Model ensemble creation
- Cross-validation support
- Custom trajectory pattern designer
- Batch dataset processing
- Model versioning and comparison
- Export to ONNX format

## Support

For issues or questions:
1. Check the Status Log in the panel for detailed error messages
2. Review the generated report for insights
3. Export results and examine JSON for debugging
4. Check console output for detailed stack traces
5. Refer to other documentation files in the project

---

**Last Updated**: 2025-11-22  
**Version**: 1.0  
**Compatibility**: PyQt6, TensorFlow 2.x, XGBoost 1.x
