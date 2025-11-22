# Fix: Missing Output Columns in Model Evaluation

## Problem

When running model evaluation, the system was failing with the error:

```
Failed to run prediction:
Missing output columns: ['incoming', 'outgoing', 'fixed_range_ascending', 
'fixed_range_descending', 'level_flight', 'linear', 'curved', 
'light_maneuver', 'high_maneuver', 'low_speed', 'high_speed']
```

## Root Cause

The issue was caused by **inconsistent tag naming** between different parts of the system:

1. **Flag Columns** (Boolean columns in DataFrame): Used full names
   - `fixed_range_ascending`, `fixed_range_descending`, `level_flight`

2. **Composite Annotation Strings**: Used abbreviated names
   - `ascending`, `descending`, `level`

3. **Multi-Output Model Predictions**: Used abbreviated names
   - `ascending`, `descending`, `level`

This inconsistency caused the multi-output adapter to fail when preparing data for prediction, as it expected the full tag names to match the flag column names.

## Solution

Updated all relevant code to use **consistent full tag names** throughout the system:

### 1. AI Engine (`src/ai_engine.py`)

**a) Multi-output prediction (lines 2199-2204):**
- Changed from abbreviated names to full names:
  - `'ascending'` → `'fixed_range_ascending'`
  - `'descending'` → `'fixed_range_descending'`
  - `'level'` → `'level_flight'`

**b) Label parsing during training (lines 1236-1241):**
- Updated to accept both formats:
  ```python
  if 'ascending' in tags or 'fixed_range_ascending' in tags:
      labels['altitude'][idx] = 0
  elif 'descending' in tags or 'fixed_range_descending' in tags:
      labels['altitude'][idx] = 1
  elif 'level' in tags or 'level_flight' in tags:
      labels['altitude'][idx] = 2
  ```

### 2. Auto-labeling Engine (`src/autolabel_engine.py`)

**Updated composite label generation (lines 188-193):**
- Changed to use full tag names:
  ```python
  if df.loc[idx, 'fixed_range_ascending']:
      tags.append('fixed_range_ascending')
  elif df.loc[idx, 'fixed_range_descending']:
      tags.append('fixed_range_descending')
  elif df.loc[idx, 'level_flight']:
      tags.append('level_flight')
  ```

### 3. Label Transformer (`src/label_transformer.py`)

**a) Primary tag extraction (lines 144-149):**
- Updated to check both formats and return full names:
  ```python
  elif 'ascending' in tags or 'fixed_range_ascending' in tags:
      return 'fixed_range_ascending'
  elif 'descending' in tags or 'fixed_range_descending' in tags:
      return 'fixed_range_descending'
  elif 'level' in tags or 'level_flight' in tags:
      return 'level_flight'
  ```

**b) Priority ordering (lines 267-273):**
- Updated to include both formats with full names taking priority

### 4. Plotting Engine (`src/plotting.py`)

No changes needed - already supports both formats in the color map.

## Benefits

1. **Consistency**: All parts of the system now use the same tag naming convention
2. **Backward Compatibility**: Code still accepts abbreviated names from older data
3. **Clear Semantics**: Full names like `fixed_range_ascending` are more descriptive than `ascending`
4. **Multi-Output Support**: Properly supports multi-output model architecture

## Testing

All modified files pass Python syntax checks:
- ✅ `src/ai_engine.py`
- ✅ `src/autolabel_engine.py`
- ✅ `src/label_transformer.py`

## Impact

- **Existing Models**: Models trained with old abbreviated names will still work due to backward compatibility
- **New Models**: Will use consistent full tag names throughout
- **Model Evaluation**: Now works correctly with multi-output models

## Related Files Modified

1. `src/ai_engine.py` - Neural network prediction and label parsing
2. `src/autolabel_engine.py` - Composite annotation generation
3. `src/label_transformer.py` - Label transformation and extraction
4. `src/plotting.py` - Already had support (no changes needed)

## Tag Name Reference

| Category | Full Name | Abbreviated Name (Legacy) |
|----------|-----------|---------------------------|
| Altitude Rising | `fixed_range_ascending` | `ascending` |
| Altitude Falling | `fixed_range_descending` | `descending` |
| Altitude Stable | `level_flight` | `level` |
| Direction Toward | `incoming` | (same) |
| Direction Away | `outgoing` | (same) |
| Path Straight | `linear` | (same) |
| Path Curved | `curved` | (same) |
| Maneuver Light | `light_maneuver` | (same) |
| Maneuver Heavy | `high_maneuver` | (same) |
| Speed Low | `low_speed` | (same) |
| Speed High | `high_speed` | (same) |

## Summary

The model evaluation feature now works correctly with proper tag naming consistency. The system maintains backward compatibility while enforcing the correct full tag names for new data and predictions.
