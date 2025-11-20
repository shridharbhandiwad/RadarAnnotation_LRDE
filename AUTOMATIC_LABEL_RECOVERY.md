# Automatic Label Recovery System

## 🎯 Overview

Your radar annotation system now has **automatic label diversity recovery** built-in! When the training process detects insufficient label diversity (like having only one composite label), it automatically transforms the data and retries training.

## ✨ Key Improvements

### 1. **Vectorized Multi-Label Classification**
Instead of treating `'incoming,level,linear,light_maneuver,low_speed'` as a single class, the system:
- Automatically splits it into individual binary labels
- Creates one-hot encoded vectors
- Trains on each component separately

### 2. **Intelligent Label Transformation**
The system analyzes your data and chooses the best strategy:
- **Multi-label binary**: Split composite labels into separate classifiers
- **Primary extraction**: Extract the most important tag (hierarchical)
- **Per-track labels**: Aggregate point-level labels to track-level

### 3. **Smart Auto-Recovery**
Training now automatically:
1. Attempts normal training first
2. Detects label diversity issues
3. Analyzes the data structure
4. Applies optimal transformation
5. Retries training with transformed data
6. Reports success with transformation details

## 🚀 How It Works

### Automatic Recovery (Default Behavior)

```python
from src.ai_engine import train_model

# This now has automatic recovery built-in!
model, metrics = train_model(
    model_name='xgboost',
    data_path='labelled_data_1.csv',
    output_dir='output/models',
    auto_transform=True  # Default, automatically recovers from label issues
)

# Check if transformation was applied
if 'label_transformation' in metrics:
    print(f"✅ Auto-recovery applied: {metrics['label_transformation']['transformation']}")
    print(f"   Created {metrics['label_transformation']['n_labels']} labels")
```

### Manual Label Transformation

You can also use the label transformer directly:

```bash
# Analyze labels without transforming
python -m src.label_transformer your_data.csv --analyze-only

# Auto-transform with intelligent strategy selection
python -m src.label_transformer your_data.csv -o fixed_data.csv --strategy auto

# Use specific strategies
python -m src.label_transformer your_data.csv -o fixed_data.csv --strategy multi_label
python -m src.label_transformer your_data.csv -o fixed_data.csv --strategy primary
python -m src.label_transformer your_data.csv -o fixed_data.csv --strategy track
```

### Python API

```python
from src.label_transformer import LabelTransformer, quick_fix_labels
import pandas as pd

# Quick fix for label issues
df_fixed, info = quick_fix_labels('your_data.csv', strategy='auto')

# Or use the transformer class for more control
transformer = LabelTransformer()

# Analyze label diversity
df = pd.read_csv('your_data.csv')
analysis = transformer.analyze_label_diversity(df['Annotation'])
print(f"Unique labels: {analysis['n_unique_labels']}")
print(f"Recommended strategy: {analysis['recommended_strategy']}")

# Auto-transform with intelligent selection
df_transformed, info = transformer.auto_transform(df)

# Or use specific transformations
df_multi = transformer.transform_to_multi_label(df)[0]  # Returns (df, array, names)
df_primary = transformer.extract_primary_labels(df, strategy='hierarchy')
df_tracks = transformer.create_per_track_labels(df, strategy='primary')
```

## 📊 Transformation Strategies

### 1. Multi-Label Binary Classification

**When**: Composite labels with multiple unique tags
**Example**: `'incoming,level,linear,light_maneuver,low_speed'`

**Result**:
```
Original:    'incoming,level,linear,light_maneuver,low_speed'
Transformed: 
  - label_incoming: True
  - label_level: True
  - label_linear: True
  - label_light_maneuver: True
  - label_low_speed: True
```

Each tag becomes a binary classifier. Great for learning individual characteristics!

### 2. Primary Label Extraction

**When**: Need simplified single-label classification
**Strategy**: Hierarchical priority

**Priority Order**:
1. Direction: incoming/outgoing
2. Vertical: ascending/descending/level
3. Path: curved/linear
4. Maneuver: high_maneuver/light_maneuver
5. Speed: high_speed/low_speed

**Example**:
```
Original:    'incoming,level,linear,light_maneuver,low_speed'
Transformed: 'incoming'  (highest priority)
```

### 3. Per-Track Label Aggregation

**When**: Multiple tracks with same point-level labels
**Strategy**: Aggregate points to track-level

**Example**:
```
Track 1 (all points): 'incoming,level,linear'
Track 2 (all points): 'outgoing,level,curved'
Track 3 (all points): 'incoming,descending,linear'

After transformation:
Track 1: 'incoming'
Track 2: 'outgoing'
Track 3: 'incoming'

Result: 2 unique classes (incoming, outgoing)
```

## 🎨 GUI Integration

The GUI automatically uses the auto-recovery system:

1. Click "Train Model" in the AI Training panel
2. If label diversity issues are detected:
   - System automatically analyzes the data
   - Applies optimal transformation
   - Saves transformed data to `output/models/transformed_training_data.csv`
   - Retries training
   - Displays success message
3. View transformation details in the results panel

## 🔧 Configuration

You can disable auto-recovery if needed:

```python
# Disable automatic recovery (use original behavior)
model, metrics = train_model(
    model_name='xgboost',
    data_path='labelled_data_1.csv',
    output_dir='output/models',
    auto_transform=False  # Disable auto-recovery
)
```

## 📝 Output Files

When auto-recovery is applied, you'll get:

```
output/models/
├── xgboost_model.pkl                    # Trained model
├── xgboost_metrics.json                 # Training metrics (includes transformation info)
└── transformed_training_data.csv        # Transformed data used for training
```

The metrics file includes transformation details:
```json
{
  "model_name": "xgboost",
  "train": {...},
  "test": {...},
  "label_transformation": {
    "transformation": "multi_label_binary",
    "n_labels": 5,
    "binary_label_columns": ["incoming", "level", "linear", "light_maneuver", "low_speed"],
    "success": true
  },
  "original_data_path": "labelled_data_1.csv",
  "transformed_data_path": "output/models/transformed_training_data.csv"
}
```

## 🎯 Use Cases

### Case 1: All Data Has Same Composite Label

**Problem**: Auto-labeling creates `'incoming,level,linear,light_maneuver,low_speed'` for all points

**Auto-Recovery**: 
- Detects composite labels with insufficient diversity
- Splits into 5 binary classifiers
- Each learns to distinguish presence/absence of that characteristic
- Training succeeds!

### Case 2: Multiple Tracks with Identical Point Labels

**Problem**: 3 tracks, all points in each track have same label

**Auto-Recovery**:
- Detects track-level structure
- Aggregates to per-track labels
- Extracts primary characteristic per track
- Creates diverse track-level classes
- Training succeeds!

### Case 3: Truly Uniform Data

**Problem**: All data is genuinely identical (same speed, direction, etc.)

**Auto-Recovery**:
- Attempts transformations
- Detects that even after transformation, no diversity exists
- Provides clear error message explaining the issue
- Suggests collecting more diverse data

## 🔍 Troubleshooting

### "Cannot recover from label diversity issue"

This means your data is genuinely uniform across all dimensions:
- All trajectories have the same motion characteristics
- **Solution**: Collect more diverse data with different:
  - Directions (incoming vs outgoing)
  - Speeds (slow vs fast)
  - Flight patterns (level vs ascending/descending)
  - Paths (straight vs turning)

### "Transformation failed"

Check the error details:
- Missing required columns (trackid, Annotation)
- Corrupt or empty data
- Unsupported data format

## 📚 Advanced Usage

### Custom Transformation Pipeline

```python
from src.label_transformer import LabelTransformer
import pandas as pd

df = pd.read_csv('your_data.csv')
transformer = LabelTransformer()

# Analyze first
analysis = transformer.analyze_label_diversity(df['Annotation'])
print(f"Analysis: {analysis}")

# Choose transformation based on your needs
if analysis['is_composite'] and analysis['n_unique_tags'] >= 2:
    # Use multi-label for maximum information
    df_out, binary_array, label_names = transformer.transform_to_multi_label(df)
    print(f"Created binary classifiers for: {label_names}")
    
elif 'trackid' in df.columns:
    # Use per-track aggregation
    df_out = transformer.create_per_track_labels(df, strategy='voting')
    print(f"Created {df_out['Annotation'].nunique()} track-level labels")
    
else:
    # Extract primary label
    df_out = transformer.extract_primary_labels(df, strategy='hierarchy')
    print(f"Extracted {df_out['Annotation'].nunique()} primary labels")

# Save and train
df_out.to_csv('transformed_data.csv', index=False)
```

### Vectorization for High Performance

The multi-label transformation uses scikit-learn's `MultiLabelBinarizer` for efficient vectorization:

```python
from src.label_transformer import LabelTransformer
import numpy as np

transformer = LabelTransformer()
df_out, binary_labels, label_names = transformer.transform_to_multi_label(df)

# binary_labels is a numpy array: (n_samples, n_labels)
# Each row is a binary vector: [1, 0, 1, 1, 0] 
# Highly efficient for training!

print(f"Shape: {binary_labels.shape}")
print(f"Labels: {label_names}")
print(f"Sample vector: {binary_labels[0]}")
```

## ✅ Benefits

1. **Zero Manual Intervention**: Training "just works" even with problematic labels
2. **Intelligent**: Automatically selects the best transformation strategy
3. **Efficient**: Vectorized operations for fast processing
4. **Transparent**: Saves all transformation details and intermediate files
5. **Flexible**: Can be used manually or automatically
6. **Recoverable**: Original data is preserved, transformed data is saved separately

## 🎓 Summary

The new automatic recovery system transforms this:

**Before**:
```
❌ Training error: Insufficient classes for training.
   Found 1 unique class: ['incoming,level,linear,light_maneuver,low_speed']
   [Manual intervention required]
```

**After**:
```
⚠️  Insufficient label diversity detected - attempting automatic recovery...
📊 Analysis: 1 unique labels found
   Recommended strategy: multi_label_binary
🔄 Applied transformation: multi_label_binary
   Created 5 unique labels
✅ Saved transformed data to output/models/transformed_training_data.csv
🔁 Retrying training with transformed labels...
✅ Training succeeded with automatic label transformation!
```

Your training pipeline is now intelligent, robust, and automatic! 🚀
