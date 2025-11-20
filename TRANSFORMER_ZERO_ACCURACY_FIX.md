# Transformer Model Zero Accuracy Fix

## Problem Identified

The transformer model was training but achieving **0.0000 accuracy** on both training and test sets. This indicated a fundamental issue with the data preparation pipeline.

## Root Cause

The bug was in the **shape mismatch** between sequences and labels when using multi-output architecture:

### What Was Happening (BROKEN):

1. **Sequence Creation** (`prepare_sequences`):
   - Creates sequences using a sliding window approach
   - For a DataFrame with 100 rows and sequence_length=20, this might create 31 sequences
   - Returns: `X_train` (31 sequences), `y_train` (31 labels), `track_ids` (31 IDs)

2. **Multi-Output Label Creation** (`prepare_multi_output_labels`):
   - Was called with the **original DataFrame** (100 rows)
   - Created multi-output labels for 100 samples
   - Returns: `y_train_multi` with 100 labels

3. **Training**:
   - Model.fit was called with `X_train` (31 samples) and `y_train_multi` (100 labels)
   - **SHAPE MISMATCH**: 31 ≠ 100
   - Model couldn't learn due to misaligned data → 0.0000 accuracy

### The Fix:

Modified the pipeline to ensure labels match the actual sequences:

1. **Added `return_label_strings` parameter** to `prepare_sequences()`:
   - When `True`, returns raw label strings instead of encoded integers
   - Allows label strings to be processed AFTER sequence creation

2. **Modified `prepare_multi_output_labels()`**:
   - Changed signature from `prepare_multi_output_labels(df: pd.DataFrame)` 
   - To: `prepare_multi_output_labels(label_strings: np.ndarray)`
   - Now accepts an array of label strings (one per sequence)
   - Creates multi-output labels for the correct number of sequences

3. **Updated training flow**:
   ```python
   # OLD (BROKEN):
   X_train, y_train, _ = prepare_sequences(df_train, feature_cols)
   y_train_multi = prepare_multi_output_labels(df_train)  # Wrong shape!
   
   # NEW (FIXED):
   X_train, y_train_strings, _ = prepare_sequences(df_train, feature_cols, return_label_strings=True)
   y_train_multi = prepare_multi_output_labels(y_train_strings)  # Correct shape!
   ```

## Changes Made

### File: `src/ai_engine.py`

#### 1. Modified `SequenceDataGenerator.prepare_sequences()` (lines 47-102):
- Added `return_label_strings` parameter
- Returns raw label strings when requested instead of encoded integers
- Ensures labels correspond exactly to created sequences

#### 2. Modified `TransformerModel.prepare_multi_output_labels()` (lines 554-615):
- Changed from accepting DataFrame to accepting label_strings array
- Now creates labels for the correct number of sequences
- Added whitespace stripping for tags
- Added support for 'level_flight' tag variant

#### 3. Updated `TransformerModel.train()` (lines 617-734):
- Multi-output path now uses `return_label_strings=True`
- Creates multi-output labels from sequence label strings
- Single-output path explicitly uses `return_label_strings=False`
- Both paths ensure X and y shapes match

#### 4. Updated `TransformerModel.evaluate()` (lines 759-835):
- Multi-output evaluation uses `return_label_strings=True`
- Single-output evaluation uses `return_label_strings=False`
- Consistent with training approach

## Verification

The fix ensures:

✅ **Shape Consistency**: X and y always have matching first dimensions
✅ **Label Accuracy**: Labels correspond to actual sequences, not DataFrame rows
✅ **Multi-Output Support**: Each output head gets correct labels
✅ **Single-Output Support**: Traditional encoding still works correctly

## Expected Results After Fix

Instead of:
```
Train Accuracy: 0.0000
Test Accuracy: 0.0000
Test F1 Score: 0.0000
```

You should now see realistic accuracy values (typically 60-95% depending on data quality):
```
Train Accuracy: 0.7500 (or higher)
Test Accuracy: 0.7200 (or similar)
Test F1 Score: 0.7100 (or similar)
```

## Testing the Fix

To test the fix, simply re-run your transformer model training:

```bash
# From the GUI, select "Train Transformer Model" with your labeled data
# Or via command line:
python -m src.ai_engine --model transformer --data path/to/labeled_data.csv --outdir output/models
```

## Technical Details

### Why Sequence Count ≠ DataFrame Row Count

The sequence generation uses a sliding window approach:

- **Short tracks** (< sequence_length): Padded to create 1 sequence
- **Long tracks** (≥ sequence_length): Creates multiple overlapping windows
  - Track with 50 points + sequence_length=20 → Creates 31 sequences (50-20+1)
  - Each window slides by 1 point

This is why we can't use the original DataFrame length for labels!

### Multi-Output Architecture

For composite labels like `"incoming,ascending,linear,light_maneuver,low_speed"`, the model:

1. Parses each label string into tags
2. Creates 5 separate output heads:
   - **Direction**: incoming(0) or outgoing(1)
   - **Altitude**: ascending(0), descending(1), or level(2)
   - **Path**: linear(0) or curved(1)
   - **Maneuver**: light_maneuver(0) or high_maneuver(1)
   - **Speed**: low_speed(0) or high_speed(1)
3. Trains each head simultaneously with appropriate loss functions

## Backwards Compatibility

- ✅ Single-output models (simple labels) continue to work
- ✅ Multi-output models (composite labels) now work correctly
- ✅ No changes to saved model format or API
- ✅ No impact on XGBoost or LSTM models

## Summary

The fix resolves the critical shape mismatch bug by ensuring that multi-output labels are created from the actual sequence labels, not from the original DataFrame. This allows the transformer model to properly learn from the data and achieve realistic accuracy metrics.
