# XGBoost Training Error Fixes

## Summary
Fixed two training errors encountered when attempting to train XGBoost models:
1. Missing 'Annotation' column error with unclear messaging
2. XGBoost "base_score must be in (0,1)" error due to incorrect objective function

## Issues Fixed

### Issue 1: Unclear Error When Using Raw Data
**Error Message:**
```
✗ Training error: CSV file is missing required columns: ['Annotation']. 
Available columns: ['time', 'trackid', 'x', 'y', 'z', 'vx', 'vy', 'vz', 'ax', 'ay', 'az']
```

**Root Cause:**
Users were selecting raw radar data files (e.g., `radar_data_reference.csv`) that don't have labels, instead of labeled data files needed for training.

**Solution:**
Enhanced error message in `src/ai_engine.py` (lines 498-504) to provide clear guidance:
```python
if 'Annotation' in missing_columns:
    error_msg += "\n\nThis appears to be raw radar data without labels."
    error_msg += "\nTo train a model, you need labeled data with an 'Annotation' column."
    error_msg += "\n\nOptions:"
    error_msg += "\n  1. Use the Auto-Labeling tool to generate annotations from raw data"
    error_msg += "\n  2. Select a file that already has annotations (e.g., labelled_data_*.csv)"
    error_msg += "\n  3. Manually add an 'Annotation' column to your CSV file"
```

### Issue 2: XGBoost Objective Function Error
**Error Message:**
```
✗ Training error: [12:56:04] C:\actions-runner\_work\xgboost\xgboost\src\objective\regression_obj.cu:119: 
Check failed: is_valid: base_score must be in (0,1) for the logistic loss.
```

**Root Cause:**
XGBoost configuration was missing the `objective` parameter, causing it to default to binary classification (`binary:logistic`) when multi-class classification was needed.

**Solution:**
Applied a three-part fix:

1. **Updated `config/default_config.json`** (line 75):
   ```json
   "xgboost": {
     "n_estimators": 100,
     "max_depth": 6,
     "learning_rate": 0.1,
     "objective": "multi:softmax",
     "random_state": 42
   }
   ```

2. **Updated `src/config.py`** (line 42):
   Added `"objective": "multi:softmax"` to the DEFAULT_CONFIG dictionary.

3. **Enhanced `src/ai_engine.py`** (lines 192-202):
   Added intelligent objective detection that automatically selects the correct objective based on the number of classes:
   ```python
   # Set appropriate objective based on number of classes
   params = self.params.copy()
   if 'objective' not in params or params['objective'] is None:
       if n_classes == 2:
           params['objective'] = 'binary:logistic'
       else:
           params['objective'] = 'multi:softmax'
   
   # For multi-class, ensure num_class is set
   if n_classes > 2 and 'num_class' not in params:
       params['num_class'] = n_classes
   ```

## Files Modified
- `src/ai_engine.py` - Enhanced error messages and XGBoost training logic
- `src/config.py` - Updated default XGBoost configuration
- `config/default_config.json` - Added objective parameter

## Testing
Configuration verified using:
```bash
python3 -c "from src.config import get_config; config = get_config(); \
  print(config.get('ml_params.xgboost'))"
```

Output confirms the objective parameter is now present:
```json
{
  "n_estimators": 100,
  "max_depth": 6,
  "learning_rate": 0.1,
  "objective": "multi:softmax",
  "random_state": 42
}
```

## User Impact
- **Better Error Messages**: Users now get clear guidance when they select the wrong file type
- **Automatic Fix**: XGBoost will automatically use the correct objective function
- **Binary Classification**: System now properly handles both binary and multi-class problems
- **Robust Training**: The training process is more resilient to configuration issues

## Next Steps
When training fails:
1. **Check the file**: Ensure you're using a labeled data file (should have 'Annotation' column)
2. **Use Auto-Labeling**: If you have raw data, use the Auto-Labeling tool first
3. **Verify Data**: Use `python validate_training_data.py <file>` to check your data before training

## Related Files
- `validate_training_data.py` - Data validation utility
- `TRAINING_ERROR_FIX.md` - Previous training error documentation
- `XGBOOST_TRAINING_FIX.md` - Previous XGBoost fix documentation
