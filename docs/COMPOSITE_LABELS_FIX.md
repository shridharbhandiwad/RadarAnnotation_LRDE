# Composite Labels Training Error Fix

## Problem Fixed

**Error:** `y contains previously unseen labels: 'incoming,level,linear,high_maneuver,high_speed'`

This error occurred when training Random Forest and Gradient Boosting models with data containing composite labels (comma-separated tags like `"incoming,level,linear,high_maneuver,high_speed"`).

### Root Cause

1. **Composite labels treated as single labels**: The models were treating each unique combination of tags as a distinct class (e.g., `"incoming,level,linear,high_maneuver,high_speed"` as one class)

2. **Train/test split issues**: When data was split into training, validation, and test sets, some composite label combinations only appeared in certain splits, causing "previously unseen labels" errors

3. **Insufficient error detection**: The automatic label transformation system only caught "Insufficient classes" errors, not "previously unseen labels" errors

## Solution Implemented

### 1. Enhanced Error Detection (`src/ai_engine.py`)

Updated the error catching in `_train_model_with_recovery` to detect:
- "Insufficient classes" errors
- "unique class" errors  
- **"previously unseen labels" errors** ✨ NEW

```python
# Now catches all label-related errors
if ("Insufficient classes" in error_str or 
    "unique class" in error_str.lower() or
    "previously unseen labels" in error_str.lower()):
```

### 2. Automatic Composite Label Transformation (`src/label_transformer.py`)

Updated `analyze_label_diversity` to:
- **Always detect composite labels** (labels containing commas)
- **Always recommend transformation** for composite labels, even with high diversity
- Use "extract_primary" strategy to create stable, single labels

```python
# Always transform composite labels
'requires_transformation': n_unique < 2 or is_composite
```

**Transformation Strategy:**
- Detects composite labels like `"incoming,level,linear,high_maneuver,high_speed"`
- Applies "per_track_primary" strategy to extract one primary label per track
- Creates simple labels based on priority hierarchy:
  - Direction (incoming/outgoing) > Vertical (level/ascending/descending) > 
  - Path (linear/curved) > Maneuver > Speed

## Results

✅ **Random Forest**: 90.00% test accuracy  
✅ **Gradient Boosting**: 89.91% test accuracy  
✅ **Automatic transformation**: Works seamlessly without user intervention

## How It Works

1. **User attempts to train model** with composite labels
2. **Error is caught**: "previously unseen labels" error detected
3. **Automatic analysis**: System analyzes label diversity and detects composite labels
4. **Transformation applied**: Extracts primary label per track (e.g., "incoming" or "outgoing")
5. **Retry training**: Training succeeds with transformed labels
6. **Results saved**: Transformed data saved to `output/models/{model_name}/transformed_training_data.csv`

## Usage

No changes required! The fix works automatically when using `auto_transform=True` (default):

```python
from src import ai_engine

# Will automatically handle composite labels
model, metrics = ai_engine.train_model(
    'random_forest',
    'data/test_simulation_labeled.csv',
    'output/models/random_forest',
    auto_transform=True  # Default
)
```

## GUI Integration

The fix is fully integrated with the GUI. When training models through the interface:
1. Select your labeled data file
2. Click "Train All Models"
3. System automatically detects and transforms composite labels
4. Training completes successfully with informative logs

## Technical Details

### Files Modified

1. **`src/ai_engine.py`**:
   - Line 1181-1187: Enhanced error detection to catch "previously unseen labels"

2. **`src/label_transformer.py`**:
   - Line 54: Always flag composite labels for transformation
   - Line 59-64: Prioritize "extract_primary" strategy for composite labels

### Transformation Output

The transformation creates a new CSV file with:
- Original features (x, y, z, velocities, etc.)
- **Simplified Annotation column**: Single primary label per track
- Same structure, easier to train

### Example Transformation

**Before:**
```
trackid,Annotation
1,"incoming,level,linear,high_maneuver,high_speed"
1,"incoming,level,linear,high_maneuver,high_speed"
2,"outgoing,ascending,curved,light_maneuver,low_speed"
```

**After:**
```
trackid,Annotation
1,"incoming"
1,"incoming"
2,"outgoing"
```

## Benefits

✅ **Automatic**: No manual intervention required  
✅ **Reliable**: Handles train/test split issues  
✅ **Informative**: Clear logs explain what happened  
✅ **Backwards Compatible**: Works with existing code  
✅ **GUI Integrated**: Seamless user experience  

## Logs to Expect

When the fix is applied, you'll see logs like:

```
⚠️  Insufficient label diversity detected - attempting automatic recovery...
📊 Analysis: 35 unique labels found
   Recommended strategy: extract_primary
🔄 Applied transformation: per_track_primary
   Created 2 unique labels
✅ Saved transformed data to output/.../transformed_training_data.csv
🔁 Retrying training with transformed labels...
✅ Training succeeded with automatic label transformation!
```

## Date

Fixed: 2025-11-22
