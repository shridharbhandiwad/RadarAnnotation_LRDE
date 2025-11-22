# XGBoost num_class Parameter Fix

## Issue
Training XGBoost models was failing with the error:
```
✗ Training error: value 0 for Parameter num_class should be greater equal to 1
num_class: Number of output class in the multi-class classification.
```

## Root Cause Analysis

The error occurred due to multiple related issues in the XGBoost training pipeline:

1. **Data Filtering Without Validation**: The `prepare_features()` method filtered data using the `valid_features` column but didn't validate that any data remained after filtering.

2. **No Class Count Validation**: The training code didn't check if there were sufficient classes (minimum 2) before attempting to train.

3. **Incorrect num_class Configuration**: The code had logic errors in setting the `num_class` parameter:
   - For binary classification (2 classes): `num_class` should NOT be set
   - For multi-class (3+ classes): `num_class` MUST be set to the number of classes
   - The original code didn't properly handle the case where config had `objective='multi:softmax'` but data was binary

4. **Config Override Issues**: The default config had `objective='multi:softmax'`, which was not being overridden for binary classification, causing XGBoost to expect `num_class` even for 2-class problems.

## Solution Implemented

### 1. Enhanced `prepare_features()` Method (Lines 141-198)

**Added proper handling of `valid_features` column:**
```python
# Filter valid features only
# Use .get() with Series to properly handle missing column
if 'valid_features' in df.columns:
    df_valid = df[df['valid_features'] == True].copy()
else:
    # If no valid_features column, use all data
    df_valid = df.copy()
```

**Added validation for empty data:**
```python
# Validate we have data after filtering
if len(df_valid) == 0:
    raise ValueError(
        "No valid data remaining after filtering. "
        "All rows were filtered out by 'valid_features' column. \n\n"
        "This usually happens when: \n"
        "  1. The auto-labeling engine marked all data as invalid \n"
        "  2. The data doesn't have enough points per track \n\n"
        "Suggestions: \n"
        "  - Check your input data has sufficient trajectory points \n"
        "  - Try using raw labeled data without auto-labeling \n"
        f"  - Original data had {len(df)} rows, all were filtered out"
    )
```

### 2. Enhanced `train()` Method (Lines 200-255)

**Added training data validation:**
```python
# Validate training data
if len(X_train) == 0 or len(y_train) == 0:
    raise ValueError(
        "Training data is empty. Cannot train model with 0 samples.\n\n"
        "Suggestions:\n"
        "  - Verify your CSV file contains data rows\n"
        "  - Check that data wasn't filtered out by 'valid_features' column\n"
        "  - Ensure the data has required feature columns"
    )
```

**Added class count validation:**
```python
# Validate minimum number of classes
if n_classes < 2:
    raise ValueError(
        f"Insufficient classes for training. Found {n_classes} unique class(es): {unique_labels}\n\n"
        "Machine learning models require at least 2 different classes to train.\n\n"
        "Suggestions:\n"
        "  1. Check your 'Annotation' column has multiple different labels\n"
        "  2. If using auto-labeling, verify it generated diverse labels\n"
        "  3. Manually review and add variety to your annotations\n"
        f"  4. Current unique classes: {unique_labels}"
    )
```

**Fixed objective and num_class configuration:**
```python
# Set appropriate objective based on number of classes
# Always override objective based on actual data, regardless of config
params = self.params.copy()

if n_classes == 2:
    # Binary classification
    params['objective'] = 'binary:logistic'
    # Remove num_class if present (not used for binary classification)
    if 'num_class' in params:
        del params['num_class']
    logger.info(f"Detected {n_classes} classes, using binary classification (objective=binary:logistic)")
else:
    # Multi-class classification (3+ classes)
    params['objective'] = 'multi:softmax'
    # Always set num_class for multi-class
    params['num_class'] = n_classes
    logger.info(f"Detected {n_classes} classes, using multi-class classification (objective=multi:softmax, num_class={n_classes})")
```

## Key Improvements

### 1. Proper Data Validation
- Checks for empty DataFrames before processing
- Validates data remains after `valid_features` filtering
- Provides clear error messages with actionable suggestions

### 2. Class Count Validation
- Ensures at least 2 classes exist for training
- Lists the unique classes found to help users diagnose issues
- Explains what needs to be fixed

### 3. Intelligent Objective Selection
- **Binary Classification (2 classes)**:
  - Sets `objective='binary:logistic'`
  - Explicitly removes `num_class` parameter (not used for binary)
  - Prevents the "num_class should be >= 1" error
  
- **Multi-class Classification (3+ classes)**:
  - Sets `objective='multi:softmax'`
  - Always sets `num_class` to the detected number of classes
  - Ensures consistency between data and parameters

### 4. Config Override
- Always determines objective from actual data, not config
- Prevents mismatches between configured and actual problem type
- Logs the detection for transparency

## Testing

Created comprehensive test suite covering all edge cases:

1. ✅ **Empty DataFrame**: Properly caught with helpful error message
2. ✅ **Single Class Only**: Caught with clear explanation of minimum requirements
3. ✅ **Binary Classification (2 classes)**: Trains successfully with `binary:logistic`
4. ✅ **Multi-class Classification (3+ classes)**: Trains successfully with `num_class` set correctly
5. ✅ **Filtered Data**: Catches when all rows are marked invalid

**Test Results**: 5/5 tests passed ✅

## Files Modified

- `src/ai_engine.py`:
  - Enhanced `prepare_features()` method (lines 141-198)
  - Enhanced `train()` method (lines 200-280)

## Error Messages Guide

### "No valid data remaining after filtering"
**Cause**: All data was filtered out by the `valid_features` column  
**Solution**: 
- Check your data has enough trajectory points per track
- Try using raw labeled data instead of auto-labeled data

### "Insufficient classes for training. Found 1 unique class"
**Cause**: All annotations are the same (e.g., all "normal")  
**Solution**:
- Add variety to your annotations
- Check auto-labeling generated diverse labels
- Manually review and correct annotations

### "Training data is empty"
**Cause**: No data available after all filtering steps  
**Solution**:
- Verify CSV file contains data rows
- Check required feature columns exist
- Ensure `valid_features` isn't filtering everything out

## XGBoost Parameter Reference

| Scenario | objective | num_class | Notes |
|----------|-----------|-----------|-------|
| 2 classes | `binary:logistic` | NOT SET | Binary classification, num_class not used |
| 3+ classes | `multi:softmax` | Set to # of classes | Multi-class classification |
| 1 class | ERROR | N/A | Cannot train, need at least 2 classes |
| 0 classes | ERROR | N/A | No data available |

## Usage Example

The fix is automatic and requires no changes to existing code. The training will now:

1. Validate your data has sufficient classes
2. Automatically detect binary vs multi-class
3. Configure XGBoost parameters correctly
4. Provide helpful error messages if issues are found

```python
# This will now work correctly for both binary and multi-class
model, metrics = train_model('xgboost', 'labeled_data.csv', 'output/models')
```

## User Impact

✅ **Automatic Fix**: No code changes needed, works out-of-the-box  
✅ **Better Errors**: Clear, actionable error messages  
✅ **Binary Support**: Properly handles 2-class problems  
✅ **Multi-class Support**: Correctly configures for 3+ classes  
✅ **Data Validation**: Catches issues early with helpful guidance  

## Related Documentation

- `XGBOOST_TRAINING_FIXES.md` - Previous XGBoost objective function fix
- `XGBOOST_TRAINING_FIX.md` - Small dataset handling fix
- `validate_training_data.py` - Data validation utility script

## Next Steps

If training still fails after this fix:
1. Run `python validate_training_data.py <your_csv>` to check data quality
2. Check the error message for specific guidance
3. Verify your CSV has:
   - An 'Annotation' column with at least 2 different labels
   - A 'trackid' column for grouping trajectories
   - Sufficient feature columns (x, y, z, speed, etc.)
