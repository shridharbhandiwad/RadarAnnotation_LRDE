# Fix Summary: XGBoost num_class Parameter Error

## Problem
User reported training error when using XGBoost:
```
✗ Training error: value 0 for Parameter num_class should be greater equal to 1
num_class: Number of output class in the multi-class classification.
```

File: `D:/Zoppler Projects/RadarAnnotation_LRDE/Database/labelled_data_1.csv`

## Root Cause
The XGBoost training code had multiple issues:
1. No validation that data remained after `valid_features` filtering
2. No check for minimum number of classes (need at least 2)
3. Incorrect handling of `num_class` parameter for binary vs multi-class
4. Config override issues where default `multi:softmax` wasn't being changed for binary data

Result: `num_class=0` was being passed to XGBoost, causing the error.

## Solution
Enhanced `src/ai_engine.py` with comprehensive validation and proper parameter configuration:

### Changes Made

#### 1. `prepare_features()` Method (Lines 141-198)
- ✅ Properly handles missing `valid_features` column
- ✅ Validates data exists after filtering
- ✅ Provides clear error messages with actionable suggestions
- ✅ Prevents empty dataset from reaching training

#### 2. `train()` Method (Lines 200-280)
- ✅ Validates training data is not empty
- ✅ Checks for minimum 2 classes (requirement for ML)
- ✅ **Key Fix**: Intelligent objective selection based on number of classes:
  - **2 classes** → `objective='binary:logistic'`, `num_class` NOT set
  - **3+ classes** → `objective='multi:softmax'`, `num_class` set to class count
- ✅ Always overrides config based on actual data
- ✅ Logs detection for transparency

## Testing
Created and ran comprehensive test suite (all tests passed):
- ✅ Empty DataFrame handling
- ✅ Single class detection and rejection
- ✅ Binary classification (2 classes)
- ✅ Multi-class classification (3+ classes)
- ✅ Filtered data validation

**Result**: 5/5 tests passed ✅

## Files Modified
- `src/ai_engine.py` - Enhanced with validation and proper parameter handling

## Files Created
- `NUM_CLASS_PARAMETER_FIX.md` - Detailed technical documentation
- `FIX_SUMMARY_NUM_CLASS.md` - This summary

## Error Messages Now Available

### 1. No Valid Data After Filtering
```
No valid data remaining after filtering. All rows were filtered out by 'valid_features' column.

This usually happens when: 
  1. The auto-labeling engine marked all data as invalid 
  2. The data doesn't have enough points per track 

Suggestions: 
  - Check your input data has sufficient trajectory points 
  - Try using raw labeled data without auto-labeling 
```

### 2. Insufficient Classes
```
Insufficient classes for training. Found 1 unique class(es): ['normal']

Machine learning models require at least 2 different classes to train.

Suggestions:
  1. Check your 'Annotation' column has multiple different labels
  2. If using auto-labeling, verify it generated diverse labels
  3. Manually review and add variety to your annotations
```

### 3. Empty Training Data
```
Training data is empty. Cannot train model with 0 samples.

Suggestions:
  - Verify your CSV file contains data rows
  - Check that data wasn't filtered out by 'valid_features' column
  - Ensure the data has required feature columns
```

## What This Means for Users

### The fix is automatic - no code changes needed!

The training will now:
1. ✅ Automatically detect if problem is binary or multi-class
2. ✅ Configure XGBoost parameters correctly
3. ✅ Provide helpful error messages if data has issues
4. ✅ Work correctly for both 2-class and 3+-class problems

### To Use:
Simply retry your training with the GUI or CLI:
```python
# In GUI: Click "Train Model" again
# In CLI:
python3 -m src.ai_engine --model xgboost --data labeled_data.csv --outdir output/models
```

## Validation Tool
To check your data before training:
```bash
python3 validate_training_data.py path/to/your/data.csv
```

## Status
✅ **COMPLETE** - Fix implemented, tested, and documented

## What To Do Next

### If the Error Was Due to Parameter Configuration:
The fix should resolve it automatically. Just retry training.

### If Your Data Actually Has Issues:
You'll now get a clear error message telling you exactly what to fix:
- **Only 1 class**: Add more variety to your annotations
- **No data after filtering**: Check your trajectory data has enough points
- **Empty CSV**: Verify your data file has content

### Still Having Issues?
1. Run: `python3 validate_training_data.py your_file.csv`
2. Check the detailed error message
3. Follow the suggestions provided
4. Ensure your CSV has:
   - 'Annotation' column with 2+ different labels
   - 'trackid' column
   - Feature columns (x, y, z, speed, etc.)

## Technical Details
See `NUM_CLASS_PARAMETER_FIX.md` for complete technical documentation.

---
**Fix Date**: 2025-11-20  
**Branch**: cursor/fix-xgboost-num-class-parameter-error-9245  
**Files Modified**: 1 (src/ai_engine.py)  
**Tests**: All passed (5/5)  
**Linting**: No errors  
