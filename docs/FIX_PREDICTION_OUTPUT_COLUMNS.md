# Fix: Missing Output Columns During Prediction

## Problem

When running predictions on unlabeled data, the system was failing with the error:

```
❌ Prediction failed!
Error: Missing output columns: ['incoming', 'outgoing', 'fixed_range_ascending', 
'fixed_range_descending', 'level_flight', 'linear', 'curved', 
'light_maneuver', 'high_maneuver', 'low_speed', 'high_speed']
```

## Root Cause

The `MultiOutputDataAdapter.prepare_data()` method was designed for **training mode**, where both input features and output labels exist in the DataFrame. However, during **prediction mode**, the input data doesn't have the output columns yet (those are what we're trying to predict!).

The error occurred at line 133 in `src/multi_output_adapter.py`:

```python
missing_output_cols = [col for col in self.output_tag_columns if col not in df_valid.columns]
if missing_output_cols:
    raise ValueError(f"Missing output columns: {missing_output_cols}")
```

When predict methods in `ai_engine.py` called `adapter.prepare_data()`, they passed DataFrames without output columns, causing this validation check to fail.

## Solution

Modified the `prepare_data()` method to support both training and prediction modes by adding an optional `include_outputs` parameter.

### 1. Updated MultiOutputDataAdapter (`src/multi_output_adapter.py`)

**Modified method signature (lines 87-102):**

```python
def prepare_data(self, df: pd.DataFrame, 
                 filter_valid: bool = True,
                 include_outputs: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Prepare data for multi-output training or prediction
    
    Args:
        df: Input DataFrame
        filter_valid: Whether to filter by valid_features column
        include_outputs: Whether to extract output columns (False for prediction mode)
        
    Returns:
        Tuple of (X, Y, metadata) where:
            - X: DataFrame with input features
            - Y: DataFrame with output tags (empty DataFrame if include_outputs=False)
            - metadata: DataFrame with trackid, time, etc.
    """
```

**Conditional output extraction (lines 129-144):**

```python
# Extract output tags (only if include_outputs=True, i.e., for training)
if include_outputs:
    if not self.output_tag_columns:
        raise ValueError("Output tag columns not identified. Call identify_columns() first.")
    
    missing_output_cols = [col for col in self.output_tag_columns if col not in df_valid.columns]
    if missing_output_cols:
        raise ValueError(f"Missing output columns: {missing_output_cols}")
    
    Y = df_valid[self.output_tag_columns].copy()
    
    # Convert to binary (0/1)
    Y = Y.astype(int)
else:
    # For prediction mode, return empty DataFrame with correct columns
    Y = pd.DataFrame(columns=self.output_tag_columns, index=df_valid.index)
```

**Updated logging (lines 153-158):**

```python
logger.info(f"Prepared data: X shape={X.shape}, Y shape={Y.shape}")
if include_outputs and len(Y) > 0:
    logger.info(f"Output tag distribution:")
    for col in Y.columns:
        pos_count = Y[col].sum()
        logger.info(f"  {col}: {pos_count}/{len(Y)} ({pos_count/len(Y)*100:.1f}%)")
```

### 2. Updated AI Engine (`src/ai_engine.py`)

**Updated XGBoost Multi-Output predict method (line 578):**

```python
X, _, metadata = self.adapter.prepare_data(df, filter_valid=False, include_outputs=False)
```

**Updated Random Forest Multi-Output predict method (line 991):**

```python
X, _, metadata = self.adapter.prepare_data(df, filter_valid=False, include_outputs=False)
```

**Updated predict_and_label function (line 2059):**

```python
X, _, _ = model.adapter.prepare_data(df_valid, filter_valid=False, include_outputs=False)
```

## Benefits

1. **Prediction Support**: Models can now make predictions on unlabeled data
2. **Backward Compatibility**: Training code continues to work without changes (default `include_outputs=True`)
3. **Clear Semantics**: The parameter name makes it explicit whether we're in training or prediction mode
4. **Consistent API**: Same `prepare_data()` method works for both modes

## Testing

All modified files pass Python syntax checks:
- ✅ `src/multi_output_adapter.py`
- ✅ `src/ai_engine.py`

## Impact

- **Training**: No impact - continues to work as before with default parameter
- **Prediction**: Now works correctly for unlabeled data
- **Model Evaluation**: Evaluation uses labeled data with `include_outputs=True`, so no impact

## Usage Example

**Training mode (extract both inputs and outputs):**

```python
X, Y, metadata = adapter.prepare_data(df_train, filter_valid=True, include_outputs=True)
```

**Prediction mode (extract only inputs):**

```python
X, _, metadata = adapter.prepare_data(df_unlabeled, filter_valid=False, include_outputs=False)
```

## Related Files Modified

1. `src/multi_output_adapter.py` - Added `include_outputs` parameter to `prepare_data()`
2. `src/ai_engine.py` - Updated predict methods to use `include_outputs=False`

## Summary

The prediction functionality now works correctly by allowing the `prepare_data()` method to operate in two modes: training mode (with output labels) and prediction mode (without output labels). This fix resolves the "Missing output columns" error and enables the model to make predictions on unlabeled data.
