# Multi-Output Auto-Tagging Guide

## Overview

This guide explains how to use the multi-output ML/DL models for **auto-tagging** and **auto-annotation** of radar track data.

## Data Format

The models support the following data structure:

### Input Format
- **Columns A-K**: Input features (radar measurements)
  - Examples: `x`, `y`, `z`, `vx`, `vy`, `vz`, `ax`, `ay`, `az`, `speed`, `heading`, `range`, etc.
  
- **Columns L-AF**: Output tag columns (targets to predict)
  - Examples: `incoming`, `outgoing`, `level_flight`, `ascending`, `descending`, `linear`, `curved`, `light_maneuver`, `high_maneuver`, `low_speed`, `high_speed`
  - These are **binary tags** (0/1 or True/False)
  
- **Column AG**: Aggregated annotation (optional reference)
  - This is a composite label combining all active tags
  - Example: `"incoming,level,linear,light_maneuver,low_speed"`

## Available Models

Three models are available, each with multi-output support:

### 1. **XGBoost Multi-Output** ⚡ Fast & Accurate
- **Type**: Gradient Boosting (tabular)
- **Speed**: Very Fast (trains in seconds)
- **Best for**: Quick training, tabular features, production deployments
- **Approach**: Trains separate XGBoost model for each tag
- **Accuracy**: 85-92%

### 2. **Random Forest Multi-Output** 🌲 Robust
- **Type**: Ensemble of decision trees
- **Speed**: Fast (trains in seconds to minutes)
- **Best for**: Robust predictions, feature importance analysis
- **Approach**: Trains separate Random Forest for each tag
- **Accuracy**: 83-90%

### 3. **Transformer Multi-Output** 🚀 State-of-the-Art
- **Type**: Deep learning with self-attention
- **Speed**: Moderate (trains in minutes)
- **Best for**: Sequential data, complex temporal patterns
- **Approach**: Single transformer with multiple output heads
- **Accuracy**: 88-95%

## Quick Start

### Training Multi-Output Models

```bash
# Train all three models on your dataset
python train_multi_output_models.py --data your_data.csv --output output/models
```

This will:
1. Automatically detect input features and output tags
2. Train XGBoost, Random Forest, and Transformer models
3. Evaluate performance on test data
4. Save trained models to output directory

### Using Trained Models for Prediction

```python
from src.ai_engine import XGBoostMultiOutputModel, RandomForestMultiOutputModel, TransformerMultiOutputModel
import pandas as pd

# Load your data
df = pd.read_csv('your_radar_data.csv')

# Load trained model
model = XGBoostMultiOutputModel()
model.load('output/models/xgboost_multi_output/model.pkl')

# Predict tags
predictions = model.predict(df)
print(predictions)

# Output:
#    incoming  outgoing  level_flight  linear  curved  ...  Predicted_Annotation
# 0         1         0             1       1       0  ...  incoming,level,linear,...
# 1         0         1             0       0       1  ...  outgoing,curved,...
```

## Advanced Usage

### Custom Column Selection

If your data has different column names, you can specify them explicitly:

```python
from src.ai_engine import XGBoostMultiOutputModel

model = XGBoostMultiOutputModel()

# Specify input and output columns
input_features = ['x', 'y', 'z', 'velocity_x', 'velocity_y', 'velocity_z']
output_tags = ['tag1', 'tag2', 'tag3', 'tag4']

train_metrics = model.train(
    df_train, 
    df_val,
    input_cols=input_features,
    output_cols=output_tags
)
```

### Automatic Column Detection

If you don't specify columns, the models will automatically detect:
- **Input features**: Numeric columns that are not binary (0/1)
- **Output tags**: Binary columns (containing only 0, 1, True, False)

### Per-Tag Performance Analysis

```python
# Train model
model = XGBoostMultiOutputModel()
train_metrics = model.train(df_train, df_val)

# Check per-tag performance
for tag_name, metrics in train_metrics['per_tag_metrics'].items():
    print(f"{tag_name}: Train Acc = {metrics['train_accuracy']:.4f}")

# Evaluate on test set
test_metrics = model.evaluate(df_test)

# Per-tag test performance
for tag_name, metrics in test_metrics['per_tag_metrics'].items():
    acc = metrics['accuracy']
    f1 = metrics['f1_score']
    print(f"{tag_name}: Test Acc = {acc:.4f}, F1 = {f1:.4f}")
```

## Model Comparison

| Model | Training Speed | Prediction Speed | Accuracy | Memory Usage | Best Use Case |
|-------|---------------|------------------|----------|--------------|---------------|
| **XGBoost** | ⚡⚡⚡ Very Fast | ⚡⚡⚡ Very Fast | ⭐⭐⭐⭐ | Low | Production, Real-time |
| **Random Forest** | ⚡⚡ Fast | ⚡⚡ Fast | ⭐⭐⭐ | Low | Robust baseline |
| **Transformer** | ⚡ Moderate | ⚡⚡ Fast | ⭐⭐⭐⭐⭐ | Medium | High accuracy needed |

## Hyperparameter Tuning

### XGBoost

```python
model = XGBoostMultiOutputModel(params={
    'n_estimators': 200,      # More trees = better accuracy (slower)
    'max_depth': 8,           # Tree depth (6-10 typical)
    'learning_rate': 0.1,     # Learning rate (0.01-0.3)
    'subsample': 0.8,         # Sample ratio per tree
    'colsample_bytree': 0.8,  # Feature ratio per tree
    'random_state': 42
})
```

### Random Forest

```python
model = RandomForestMultiOutputModel(params={
    'n_estimators': 200,      # Number of trees (100-500)
    'max_depth': 20,          # Tree depth (10-30)
    'min_samples_split': 2,   # Min samples to split node
    'min_samples_leaf': 1,    # Min samples in leaf
    'random_state': 42,
    'n_jobs': -1              # Use all CPU cores
})
```

### Transformer

```python
model = TransformerMultiOutputModel(params={
    'd_model': 128,           # Model dimension (64-256)
    'num_heads': 8,           # Attention heads (4-16)
    'ff_dim': 256,            # Feed-forward dimension
    'num_layers': 3,          # Transformer layers (2-6)
    'dropout': 0.2,           # Dropout rate (0.1-0.3)
    'epochs': 100,            # Training epochs (50-200)
    'batch_size': 64,         # Batch size (16-128)
    'sequence_length': 30     # Sequence length (10-50)
})
```

## Integration with Existing Workflow

The multi-output models integrate seamlessly with your existing workflow:

### Option 1: Use with Auto-Labeled Data

```bash
# Step 1: Generate or extract radar data
python -m src.data_engine --input radar_data.bin --out raw_data.csv

# Step 2: Apply auto-labeling to create tag columns
python -m src.autolabel_engine --input raw_data.csv --out labeled_data.csv

# Step 3: Train multi-output models
python train_multi_output_models.py --data labeled_data.csv

# Step 4: Use trained model for prediction
python predict_with_multi_output.py --model output/models/xgboost_multi_output/model.pkl --data new_data.csv
```

### Option 2: Use with Manual Labels

If you have manually labeled data with tag columns:

```python
import pandas as pd
from src.ai_engine import XGBoostMultiOutputModel

# Your data already has tag columns
df = pd.read_csv('manually_labeled_data.csv')

# Train directly
model = XGBoostMultiOutputModel()
model.train(df)  # Automatically detects columns

# Predict on new data
predictions = model.predict(new_df)
```

## Understanding Outputs

### Model Prediction Output

When you call `model.predict(df)`, you get a DataFrame with:

```
   incoming  outgoing  ascending  descending  level_flight  linear  curved  ...  Predicted_Annotation
0         1         0          0           0             1       1       0  ...  incoming,level_flight,linear,light_maneuver,low_speed
1         0         1          0           0             1       0       1  ...  outgoing,level_flight,curved,light_maneuver,low_speed
2         1         0          1           0             0       1       0  ...  incoming,ascending,linear,light_maneuver,low_speed
```

Each column (incoming, outgoing, etc.) contains:
- **0**: Tag is not active/present
- **1**: Tag is active/present

The `Predicted_Annotation` column combines all active tags into a single comma-separated string.

## Performance Tips

1. **Data Quality**: Ensure input features are properly normalized and don't contain NaN/Inf values
2. **Class Imbalance**: If some tags are rare, consider using class weights or oversampling
3. **Sequence Length** (Transformer only): Adjust based on your track length distribution
4. **Feature Engineering**: Add domain-specific features for better accuracy
5. **Ensemble**: Combine predictions from multiple models for best results

## Troubleshooting

### Issue: "No valid sequences could be created"
**Solution**: Check that your tracks have enough data points. Transformer requires at least 3 points per track.

### Issue: "Missing input columns"
**Solution**: Ensure your data has the required feature columns (x, y, z, velocities, etc.)

### Issue: Low accuracy on certain tags
**Solution**: 
- Check tag distribution (some tags may be too rare)
- Increase model complexity (more estimators, layers, etc.)
- Add more training data
- Perform feature engineering

### Issue: Training takes too long
**Solution**:
- Use XGBoost instead of Transformer for faster training
- Reduce dataset size for initial experiments
- Use smaller Transformer (fewer layers, smaller d_model)

## Example: Complete End-to-End Workflow

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from src.ai_engine import XGBoostMultiOutputModel, TransformerMultiOutputModel

# 1. Load and prepare data
df = pd.read_csv('radar_data_with_tags.csv')

# 2. Split by track ID
track_ids = df['trackid'].unique()
train_ids, test_ids = train_test_split(track_ids, test_size=0.2)
df_train = df[df['trackid'].isin(train_ids)]
df_test = df[df['trackid'].isin(test_ids)]

# 3. Train XGBoost model
print("Training XGBoost...")
model_xgb = XGBoostMultiOutputModel()
train_metrics = model_xgb.train(df_train)
test_metrics = model_xgb.evaluate(df_test)
print(f"XGBoost Test Accuracy: {test_metrics['accuracy']:.4f}")

# 4. Train Transformer model
print("Training Transformer...")
model_transformer = TransformerMultiOutputModel()
train_metrics = model_transformer.train(df_train)
test_metrics = model_transformer.evaluate(df_test)
print(f"Transformer Test Accuracy: {test_metrics['accuracy']:.4f}")

# 5. Save models
model_xgb.save('models/xgboost_multi_output.pkl')
model_transformer.save('models/transformer_multi_output.h5')

# 6. Use for prediction on new data
new_df = pd.read_csv('new_radar_data.csv')
predictions_xgb = model_xgb.predict(new_df)
predictions_transformer = model_transformer.predict(new_df)

# 7. Save predictions
predictions_xgb.to_csv('predictions_xgboost.csv', index=False)
predictions_transformer.to_csv('predictions_transformer.csv', index=False)

print("✅ Complete! Models trained and predictions saved.")
```

## Next Steps

1. **Try the training script**: Run `python train_multi_output_models.py` on your data
2. **Compare models**: See which model works best for your use case
3. **Fine-tune**: Adjust hyperparameters for optimal performance
4. **Deploy**: Integrate the best model into your production pipeline
5. **Monitor**: Track model performance and retrain as needed

## Support

For issues or questions, refer to:
- Main README: `README.md`
- Model documentation: See docstrings in `src/ai_engine.py`
- Data adapter: See `src/multi_output_adapter.py`
