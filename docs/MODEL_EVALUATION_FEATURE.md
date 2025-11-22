# Model Evaluation Feature

## Overview

The AI engine now includes a powerful **Model Evaluation** feature that allows you to:
- Load a trained model
- Apply it to unlabeled or new data files
- Auto-generate labels/annotations using the model
- Save and visualize the results

## Implementation Details

### Backend Functions (ai_engine.py)

#### 1. `load_trained_model(model_path: str)`
Automatically detects and loads any trained model from disk:
- **Random Forest** models (.pkl files with "random_forest" in name)
- **Gradient Boosting/XGBoost** models (.pkl files with "gradient" or "xgboost")
- **Neural Network/Transformer** models (.h5 files or "neural_network"/"transformer" in name)

Returns: `(model_object, model_type)`

#### 2. `predict_and_label(model_path: str, input_csv_path: str, output_csv_path: str = None)`
Predicts labels for unlabeled data using a trained model.

**Process:**
1. Loads the trained model
2. Reads input CSV (can be raw unlabeled data)
3. Computes necessary motion features automatically
4. Uses the model to predict labels
5. Saves results with predicted annotations

**Supports all model types:**
- **Random Forest & Gradient Boosting**: Uses tabular features with scaling
- **Neural Network/Transformer**: Uses sequence-based predictions
  - Handles both single-output and multi-output architectures
  - Automatically reconstructs composite labels for multi-output models

Returns: DataFrame with predicted annotations

### GUI Panel (gui.py)

#### ModelEvaluationPanel
A user-friendly 3-step interface:

**Step 1: Select Trained Model**
- Browse for model files (.pkl or .h5)
- Default location: `output/models/`
- Displays selected model name

**Step 2: Select Input Data File**
- Browse for CSV files with trajectory data
- Can be labeled or unlabeled
- Shows basic file statistics (records, tracks)
- Warns if file already has annotations

**Step 3: Run Prediction**
- Executes prediction in background thread
- Shows progress indicator
- Displays results summary table
- Shows prediction statistics

**Results Actions:**
- **Save Labeled Data**: Export predictions to custom location
- **Visualize Results**: Automatically switch to visualization panel with predicted data

## Usage

### Via GUI

1. Launch the GUI application:
   ```bash
   python -m src.gui
   ```

2. Navigate to **"🔮 Model Evaluation"** panel

3. Follow the 3-step workflow:
   - Select a trained model
   - Select your input CSV file
   - Click "Predict and Auto-Label"

4. View results and optionally:
   - Save to custom location
   - Visualize in the plotting panel

### Via Python Script

```python
from src import ai_engine

# Predict labels for new data
df_labeled = ai_engine.predict_and_label(
    model_path='output/models/random_forest_model.pkl',
    input_csv_path='data/my_new_data.csv',
    output_csv_path='data/my_new_data_predicted.csv'
)

print(f"Predicted {len(df_labeled)} records")
print(df_labeled['Annotation'].value_counts())
```

### Via Command Line (Future Enhancement)

```bash
# Could add CLI support in ai_engine.py
python -m src.ai_engine --predict \
    --model output/models/random_forest_model.pkl \
    --input data/new_data.csv \
    --output data/new_data_labeled.csv
```

## Testing

A comprehensive test script is provided:

```bash
python test_model_evaluation.py
```

This test script:
1. Generates test simulation data
2. Auto-labels the data
3. Trains a Random Forest model
4. Generates new unlabeled data
5. Uses the model to predict labels
6. Displays prediction results

## Input Data Requirements

The input CSV file must contain the following columns:
- `trackid`: Track identifier
- `time`: Timestamp
- `x`, `y`, `z`: Position coordinates
- `vx`, `vy`, `vz`: Velocity components

Optional columns:
- `ax`, `ay`, `az`: Acceleration (auto-initialized to 0 if missing)
- Any other columns are preserved in output

## Output

The output CSV contains all original columns plus:
- **Computed features**: speed, heading, range, curvature, etc.
- **Annotation**: Predicted label for each record
- **Feature flags**: incoming, outgoing, linear, curved, etc.

## Features

✅ **Automatic Feature Computation**: No need to pre-compute features
✅ **Model Auto-Detection**: Automatically identifies model type from filename
✅ **Multi-Output Support**: Handles composite labels for Transformer models
✅ **Thread-Safe GUI**: Non-blocking predictions in background
✅ **Comprehensive Results**: Displays statistics and distribution tables
✅ **Visualization Integration**: Seamlessly switches to visualization panel
✅ **Error Handling**: Clear error messages for invalid inputs

## Example Workflow

### Complete End-to-End Example

```python
from src import sim_engine, autolabel_engine, ai_engine

# 1. Generate training data
sim_engine.create_large_training_dataset(
    output_path='data/train.csv',
    n_tracks=100,
    duration_min=5.0
)

# 2. Auto-label training data
df = pd.read_csv('data/train.csv')
df = autolabel_engine.compute_motion_features(df)
df = autolabel_engine.apply_rules_and_flags(df)
df.to_csv('data/train_labeled.csv', index=False)

# 3. Train model
model, metrics = ai_engine.train_model(
    'random_forest',
    'data/train_labeled.csv',
    'output/my_model'
)

# 4. Generate new unlabeled data
sim_engine.create_large_training_dataset(
    output_path='data/new_data.csv',
    n_tracks=20,
    duration_min=2.0
)

# 5. Evaluate model on new data
df_predicted = ai_engine.predict_and_label(
    'output/my_model/random_forest_model.pkl',
    'data/new_data.csv',
    'data/new_data_predicted.csv'
)

# 6. Analyze results
print(df_predicted['Annotation'].value_counts())
```

## Navigation in GUI

The Model Evaluation panel is accessible from the main navigation:

```
📊 Data Extraction
🏷️ AutoLabeling
🤖 AI Tagging
🔮 Model Evaluation      ← NEW!
🚀 High Volume Training
📈 Report
🔬 Simulation
📉 Visualization
⚙️ Settings
```

## Benefits

1. **No Feature Engineering Required**: Automatically computes all necessary features
2. **Universal Model Support**: Works with RF, XGBoost, and Neural Network models
3. **Production Ready**: Apply trained models to real-world data instantly
4. **User Friendly**: Simple 3-step process in GUI
5. **Flexible**: Can be used via GUI, Python API, or potentially CLI

## Future Enhancements

Potential improvements:
- [ ] Batch processing for multiple files
- [ ] Confidence scores for predictions
- [ ] Model performance comparison
- [ ] Real-time prediction mode
- [ ] REST API endpoint for model serving
- [ ] Export predictions in multiple formats (JSON, Excel, etc.)

## Technical Notes

### Multi-Output Model Handling

For Transformer models with multi-output architecture:
- Predictions are made for each output head (direction, altitude, path, maneuver, speed)
- Results are automatically combined into composite label strings
- Format matches the training data structure

### Memory Efficiency

The prediction process is optimized for large datasets:
- Streaming data processing where possible
- Efficient feature computation using vectorized operations
- Proper memory cleanup after predictions

### Error Recovery

The system handles various edge cases:
- Missing feature columns (auto-initialized)
- Insufficient sequence length (padding)
- Invalid/corrupted models (clear error messages)
- Malformed input data (validation checks)

## Summary

The Model Evaluation feature completes the machine learning workflow by enabling easy application of trained models to new data. With automatic feature computation and a user-friendly interface, it's now trivial to:
1. Train a model on labeled data
2. Apply it to unlabeled real-world data
3. Get instant predictions with comprehensive statistics
4. Visualize and analyze the results

This makes the entire radar data annotation pipeline truly end-to-end and production-ready! 🚀
