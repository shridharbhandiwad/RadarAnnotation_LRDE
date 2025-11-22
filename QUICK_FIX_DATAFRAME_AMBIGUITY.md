# Quick Fix: DataFrame Ambiguity Error

## Problem
```
✗ Training error: The truth value of a DataFrame is ambiguous. 
Use a.empty, a.bool(), a.item(), a.any() or a.all().
```

## Solution ✅

**Fixed!** The error has been resolved in the following files:
- `src/multi_output_adapter.py` (2 locations)
- `src/ai_engine.py` (5 locations)

## What Changed

Changed DataFrame filtering from:
```python
df_valid = df[df['valid_features'] == True].copy()
```

To:
```python
valid_mask = df['valid_features'].astype(bool)
df_valid = df.loc[valid_mask].copy()
```

## How to Use

### Option 1: GUI (Recommended)
1. Launch the GUI: `python src/gui.py`
2. Select your labeled CSV file
3. Enable "Multi-Output Mode"
4. Select "Gradient Boosting" model
5. Click "Train Model"
6. ✅ Training will now complete successfully!

### Option 2: Command Line
```python
from src.ai_engine import XGBoostMultiOutputModel
from sklearn.model_selection import train_test_split
import pandas as pd

# Load your data
df = pd.read_csv('data/test_simulation_labeled.csv')

# Split data
track_ids = df['trackid'].unique()
train_ids, test_ids = train_test_split(track_ids, test_size=0.2, random_state=42)
train_ids, val_ids = train_test_split(train_ids, test_size=0.2, random_state=42)

df_train = df[df['trackid'].isin(train_ids)]
df_val = df[df['trackid'].isin(val_ids)]
df_test = df[df['trackid'].isin(test_ids)]

# Train model
model = XGBoostMultiOutputModel()
train_metrics = model.train(df_train, df_val)
test_metrics = model.evaluate(df_test)

print(f"Train accuracy: {train_metrics['train_accuracy']:.4f}")
print(f"Test accuracy: {test_metrics['accuracy']:.4f}")
```

## Verification

The fix has been tested and verified to work correctly:
- ✅ MultiOutputDataAdapter filtering
- ✅ XGBoostModel filtering
- ✅ RandomForestModel filtering
- ✅ Track-level filtering
- ✅ Sequence preparation filtering

## What This Fixes

- ✅ Gradient Boosting (XGBoost) training in Multi-Output mode
- ✅ Random Forest training in Multi-Output mode
- ✅ Any model using the `valid_features` column for data filtering
- ✅ Data preparation for sequence models (LSTM, Transformer)

## No Code Changes Required

This is a transparent fix - your existing code will work without any modifications!

## Need Help?

For more details, see: `DATAFRAME_AMBIGUITY_FIX.md`
