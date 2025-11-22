# Multi-Output Model Integration in AI Tagging Panel

## 🎉 Overview

The AI Tagging panel now supports **Multi-Output Mode** for training models that predict multiple tags simultaneously. This enables auto-tagging functionality where a single model predicts all output tags (incoming, outgoing, level flight, linear, curved, etc.) at once.

## ✨ Features Added

### 1. Multi-Output Mode Toggle
- **New Checkbox**: "🎯 Multi-Output Mode (Auto-Tagging)"
- Located in the Model Selection section of AI Tagging panel
- When enabled, shows detailed information about multi-output training

### 2. Three Multi-Output Model Types
All three model types now support multi-output mode:

| Model Type | Multi-Output Class | Best For |
|------------|-------------------|----------|
| **Random Forest** | `RandomForestMultiOutputModel` | Robust, fast training |
| **Gradient Boosting** | `XGBoostMultiOutputModel` | High accuracy, production-ready |
| **Neural Network** | `TransformerMultiOutputModel` | State-of-the-art, sequential data |

### 3. Automatic Data Handling
- **Automatic splitting by track ID** to prevent data leakage
- **Train/Val/Test split**: 64% / 16% / 20%
- **Input columns**: A-K (x, y, z, velocities, etc.)
- **Output columns**: L-AF (binary tags)

### 4. Enhanced Results Display
- Shows overall accuracy and F1 score
- Displays per-tag metrics for top 10 tags
- Indicates number of tags trained
- Training time tracking

## 🚀 How to Use

### Step-by-Step Guide

1. **Start the GUI**
   ```bash
   python3 -m src.gui
   ```

2. **Navigate to AI Tagging Panel**
   - Click on "🤖 AI Tagging" in the left sidebar

3. **Select Model Type**
   - Choose from: Random Forest, Gradient Boosting, or Neural Network
   - **Recommendation**: Start with Gradient Boosting for best accuracy

4. **Enable Multi-Output Mode**
   - Check the "🎯 Multi-Output Mode (Auto-Tagging)" checkbox
   - Read the information that appears explaining the mode

5. **Select Training Data**
   - Click "Select Labeled Data CSV"
   - Choose a CSV file with proper format:
     - Columns A-K: Input features
     - Columns L-AF: Output tags (binary 0/1)
     - Column AG: Aggregated annotation (optional reference)

6. **Train the Model**
   - Click "Train Model"
   - Wait for training to complete (may take 1-5 minutes depending on data size)

7. **Review Results**
   - Overall metrics (accuracy, F1 score)
   - Per-tag performance
   - Training time
   - Model verdict and recommendations

## 📊 Expected Output

### Training Results Display

```
======================================================================
                      TRAINING RESULTS TABLE
======================================================================

┌─────────────────────────────┬────────────────────────────────┐
│ Metric                      │ Value                          │
├─────────────────────────────┼────────────────────────────────┤
│ Model Type                  │ XGBoost                        │
│ Train Accuracy              │                         0.9245 │
│ Test Accuracy               │                         0.8892 │
│ Test F1 Score               │                         0.8756 │
│ Training Time (s)           │                          45.23 │
├─────────────────────────────┼────────────────────────────────┤
│ Multi-Output Per-Tag Results│                                │
├─────────────────────────────┼────────────────────────────────┤
│   circular                  │ Acc:0.9124 F1:0.8956           │
│   crossing                  │ Acc:0.8876 F1:0.8654           │
│   curved                    │ Acc:0.9234 F1:0.9012           │
│   high_maneuver             │ Acc:0.8765 F1:0.8543           │
│   incoming                  │ Acc:0.9456 F1:0.9234           │
│   level_flight              │ Acc:0.9012 F1:0.8876           │
│   light_maneuver            │ Acc:0.8954 F1:0.8765           │
│   linear                    │ Acc:0.9345 F1:0.9123           │
│   outgoing                  │ Acc:0.9234 F1:0.9012           │
│   side_moving               │ Acc:0.8876 F1:0.8654           │
│   ... and 11 more tags      │                                │
└─────────────────────────────┴────────────────────────────────┘
```

## 🔧 Technical Details

### Implementation

**File Modified**: `src/gui.py`

**Key Components Added**:

1. **UI Components** (Lines ~348-378)
   - `multi_output_check`: Checkbox for enabling multi-output mode
   - `multi_output_info`: Information label
   - `toggle_multi_output_info()`: Toggle function

2. **Training Logic** (Lines ~435-471)
   - Modified `train_model()` to route to multi-output training
   - Added `_train_multi_output_model()` method

3. **Multi-Output Training Method** (Lines ~515-585)
   - Data loading and splitting by track ID
   - Model instantiation based on type
   - Training and evaluation
   - Model saving

4. **Results Display** (Lines ~498-511)
   - Enhanced to show per-tag metrics
   - Shows top 10 tags + count of remaining

### Data Format Requirements

**Input Features (Columns A-K)**:
```
time, trackid, x, y, z, vx, vy, vz, ax, ay, az, speed, speed_2d, heading, range
```

**Output Tags (Columns L-AF)** - Binary (0/1):
```
incoming, outgoing, level_flight, climbing, descending, linear, curved, 
circular, high_maneuver, light_maneuver, crossing, side_moving, ...
```

**Reference (Column AG)** - Optional:
```
aggregated_annotation
```

### Model Parameters

**Random Forest Multi-Output**:
```python
{
    'n_estimators': 100,
    'max_depth': 20,
    'random_state': 42,
    'n_jobs': -1  # Use all CPUs
}
```

**XGBoost Multi-Output**:
```python
{
    'n_estimators': 100,
    'max_depth': 6,
    'learning_rate': 0.1,
    'random_state': 42
}
```

**Transformer Multi-Output**:
```python
{
    'd_model': 128,
    'num_heads': 8,
    'num_layers': 4,
    'epochs': 50
}
```

## 📁 Model Output

Trained models are saved to:
```
output/models/
├── random_forest_multi_output/
│   └── model.pkl
├── xgboost_multi_output/
│   └── model.pkl
└── transformer_multi_output/
    └── model.pkl
```

## 🎯 Use Cases

### 1. Auto-Tagging New Data
Train a multi-output model on labeled data, then use it to automatically tag new unlabeled tracks with all relevant tags.

### 2. Tag Relationship Learning
Multi-output models can learn relationships between tags (e.g., if a track is "incoming" it's less likely to be "outgoing").

### 3. Production Deployment
Use the trained models for real-time auto-tagging in operational systems.

### 4. Batch Processing
Process large datasets efficiently with a single model predicting all tags at once.

## ⚡ Performance Tips

1. **More Data = Better Performance**
   - Use at least 100+ tracks for training
   - Ensure diverse trajectory types

2. **Model Selection**
   - **Fast training**: Random Forest
   - **Best accuracy**: XGBoost
   - **Sequential patterns**: Transformer

3. **Hyperparameter Tuning**
   - For better accuracy, increase `n_estimators`
   - For faster training, reduce `max_depth`

4. **Data Quality**
   - Ensure tags are properly labeled
   - Balance tag distributions
   - Remove noisy or ambiguous tracks

## 🐛 Troubleshooting

**Problem**: "Training error: No valid sequences could be created"
- **Solution**: Ensure tracks have at least 3 data points

**Problem**: "Low accuracy on certain tags"
- **Solution**: Check tag distribution, rare tags need more examples

**Problem**: "Training too slow"
- **Solution**: Use Random Forest or reduce dataset size for testing

**Problem**: "Missing input columns"
- **Solution**: Ensure CSV has required features (x, y, z, velocities)

**Problem**: "Model file too large"
- **Solution**: Use XGBoost (smallest) instead of Random Forest or Transformer

## 📚 Related Documentation

- **Quick Start**: `QUICK_START_MULTI_OUTPUT.md`
- **Command Line**: `train_multi_output_models.py`
- **API Reference**: See `src/ai_engine.py` multi-output model classes

## ✅ Validation

The implementation has been validated with:
- ✅ Python syntax check
- ✅ All required components present
- ✅ Proper method signatures
- ✅ Correct data flow
- ✅ Results display formatting

Run validation:
```bash
python3 test_gui_syntax_validation.py
```

## 🎓 Example Workflow

```python
# This is what happens behind the scenes when you train in GUI:

# 1. Load data
df = pd.read_csv('your_labeled_data.csv')

# 2. Split by track ID (prevents data leakage)
train_ids, test_ids = train_test_split(track_ids, test_size=0.2)
df_train = df[df['trackid'].isin(train_ids)]
df_test = df[df['trackid'].isin(test_ids)]

# 3. Create multi-output model
model = XGBoostMultiOutputModel(params={
    'n_estimators': 100,
    'max_depth': 6,
    'learning_rate': 0.1
})

# 4. Train (automatically identifies input/output columns)
train_metrics = model.train(df_train)

# 5. Evaluate on test set
test_metrics = model.evaluate(df_test)

# 6. Save model
model.save('output/models/xgboost_multi_output/model.pkl')

# 7. Use for prediction
new_data = pd.read_csv('new_unlabeled_data.csv')
predictions = model.predict(new_data)
```

## 🎉 Summary

You now have a powerful auto-tagging system integrated into the GUI! Train once, predict multiple tags simultaneously, and deploy to production with confidence.

**Happy Auto-Tagging! 🚀**
