# DataFrame Ambiguity Error Fix

## Issue Summary

**Error:** "The truth value of a DataFrame is ambiguous. Use a.empty, a.bool(), a.item(), a.any() or a.all()."

**Context:** This error occurred when training Gradient Boosting models in Multi-Output mode.

## Root Cause

The error was caused by ambiguous DataFrame boolean filtering in several locations where the code used:

```python
df_valid = df[df['valid_features'] == True].copy()
```

In certain edge cases (e.g., column naming issues, pandas version differences), the expression `df['valid_features'] == True` could return a DataFrame instead of a Series, causing pandas to raise an ambiguity error when trying to use it as a boolean mask.

## Solution

Replaced all instances of the problematic filtering pattern with an explicit approach:

```python
valid_mask = df['valid_features'].astype(bool)
df_valid = df.loc[valid_mask].copy()
```

This ensures:
1. The mask is always a Series (not a DataFrame)
2. Boolean conversion is explicit
3. `.loc` indexing is used for clarity and safety

## Files Modified

### 1. `src/multi_output_adapter.py`
- **Line 103-105:** Fixed `prepare_data` method filtering
- **Line 183-184:** Fixed `prepare_sequences` method filtering

### 2. `src/ai_engine.py`
- **Line 72:** Fixed filtering in sequence preparation for Transformer models
- **Line 192-194:** Fixed filtering in `XGBoostModel.prepare_features`
- **Line 653:** Fixed filtering in `RandomForestModel.prepare_features`
- **Line 1941:** Fixed filtering in `predict_with_model` function
- **Line 1996:** Fixed filtering in sequence preparation for predictions

## Testing

Created and ran test scripts to verify the fix:

1. **MultiOutputDataAdapter Test:** Verified DataFrame filtering works correctly with various data patterns
2. **AI Engine Test:** Verified filtering logic works correctly in all modified locations

All tests passed successfully.

## Impact

This fix resolves the training error for:
- ✅ Gradient Boosting (XGBoost) in Multi-Output mode
- ✅ Random Forest in Multi-Output mode
- ✅ All models using the `valid_features` column for filtering

## Usage

The fix is transparent to users. Simply train models as usual:

```python
# In GUI: Select "Multi-Output Mode" and train Gradient Boosting
# The DataFrame ambiguity error will no longer occur

# Or programmatically:
from src.ai_engine import XGBoostMultiOutputModel

model = XGBoostMultiOutputModel(params={
    'n_estimators': 100,
    'max_depth': 6,
    'learning_rate': 0.1
})

train_metrics = model.train(df_train, df_val)
```

## Technical Details

### Why `.astype(bool)` + `.loc`?

1. **`.astype(bool)`:** Explicitly converts the column to boolean type, ensuring we get a Series
2. **`.loc[mask]`:** Uses label-based indexing with the boolean mask, which is the recommended pandas approach
3. **Copy safety:** The `.copy()` ensures we don't have view vs. copy issues

### Alternative Approaches (Not Used)

- `df[df['valid_features'].eq(True)]` - Less readable
- `df[df['valid_features'] == True]` - Original problematic pattern
- `df.query('valid_features == True')` - String-based, harder to debug

## Verification

To verify the fix is working, check that:
1. Multi-Output Gradient Boosting training completes without errors
2. The console shows training progress and metrics
3. Models are saved successfully to the output directory

## Related Issues

This fix also prevents similar errors in:
- Data validation workflows
- Feature engineering pipelines
- Any code that filters DataFrames based on boolean columns
