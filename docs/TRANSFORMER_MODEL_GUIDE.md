# Transformer-based Multi-output Model Guide

## Overview

The Transformer-based Multi-output Model is a state-of-the-art deep learning model for trajectory classification that uses **self-attention mechanisms** to capture temporal dependencies in radar trajectory sequences. Unlike traditional recurrent models (LSTM), the Transformer processes entire sequences in parallel, making it both faster and more effective at capturing long-range dependencies.

### Key Features

✅ **Multi-Head Self-Attention** - Captures complex temporal patterns in trajectory data  
✅ **Multi-Output Classification** - Predicts multiple attributes simultaneously  
✅ **Positional Encoding** - Maintains temporal order information  
✅ **Parallel Processing** - Faster training than sequential models  
✅ **Automatic Composite Label Detection** - Intelligently handles comma-separated labels  

## Architecture

### Model Components

1. **Input Projection Layer**
   - Projects input features to `d_model` dimensions (default: 64)
   - Enables uniform processing across all features

2. **Positional Encoding**
   - Adds learnable position embeddings to maintain sequence order
   - Critical for capturing temporal patterns in trajectories

3. **Transformer Blocks** (Stacked)
   - **Multi-Head Attention**: Captures relationships between time steps
   - **Feed-Forward Network**: Processes attended features
   - **Layer Normalization**: Stabilizes training
   - **Residual Connections**: Enables deep architectures

4. **Global Average Pooling**
   - Aggregates sequence-level features
   - Provides fixed-size representation regardless of sequence length

5. **Output Heads** (Single or Multiple)
   - **Single Output**: One softmax layer for standard classification
   - **Multi-Output**: Separate heads for each attribute (direction, altitude, path, maneuver, speed)

### Architecture Diagram

```
Input Sequence (20 x N_features)
    ↓
Linear Projection (20 x 64)
    ↓
+ Positional Encoding
    ↓
Transformer Block 1 (Multi-Head Attention + FFN)
    ↓
Transformer Block 2 (Multi-Head Attention + FFN)
    ↓
Global Average Pooling (64)
    ↓
Dense Layer (64, ReLU)
    ↓
┌─────────┬──────────┬──────┬──────────┬───────┐
↓         ↓          ↓      ↓          ↓       ↓
Direction Altitude   Path   Maneuver   Speed
(2 classes)(3 classes)(2)    (2)       (2)
```

## Configuration

The model is configured via `config/default_config.json`:

```json
{
  "ml_params": {
    "transformer": {
      "d_model": 64,           // Model dimension
      "num_heads": 4,          // Number of attention heads
      "ff_dim": 128,           // Feed-forward dimension
      "num_layers": 2,         // Number of transformer blocks
      "dropout": 0.1,          // Dropout rate
      "epochs": 50,            // Training epochs
      "batch_size": 32,        // Batch size
      "sequence_length": 20    // Input sequence length
    }
  }
}
```

### Parameter Guide

| Parameter | Description | Recommended Range | Default |
|-----------|-------------|-------------------|---------|
| `d_model` | Model embedding dimension | 32-128 | 64 |
| `num_heads` | Number of attention heads | 2-8 | 4 |
| `ff_dim` | Feed-forward network size | 64-256 | 128 |
| `num_layers` | Number of transformer blocks | 1-4 | 2 |
| `dropout` | Regularization dropout | 0.0-0.3 | 0.1 |
| `epochs` | Training iterations | 20-100 | 50 |
| `batch_size` | Training batch size | 16-64 | 32 |
| `sequence_length` | Trajectory points per sequence | 10-50 | 20 |

**Note**: `d_model` must be divisible by `num_heads`

## Usage

### 1. Command Line Interface

```bash
# Train transformer model
python -m src.ai_engine --model transformer --data labeled_data.csv --outdir output/models

# With custom parameters
python -m src.ai_engine --model transformer --data labeled_data.csv --outdir output/models --params '{"epochs": 100, "d_model": 128}'
```

### 2. Python API

```python
from src.ai_engine import TransformerModel, train_model
import pandas as pd

# Option 1: Using high-level API (recommended)
model, metrics = train_model('transformer', 'labeled_data.csv', 'output/models')

print(f"Test Accuracy: {metrics['test']['accuracy']:.4f}")
print(f"Test F1 Score: {metrics['test']['f1_score']:.4f}")

# Option 2: Direct model usage (advanced)
model = TransformerModel(params={'epochs': 100, 'd_model': 128})

df_train = pd.read_csv('train.csv')
df_val = pd.read_csv('val.csv')

# Train model
train_metrics = model.train(df_train, df_val, use_multi_output=True)

# Evaluate model
df_test = pd.read_csv('test.csv')
test_metrics = model.evaluate(df_test)

# Save model
model.save('output/models/transformer_model.h5')

# Load model
model.load('output/models/transformer_model.h5')
```

### 3. GUI Application

1. Launch the GUI:
   ```bash
   python -m src.gui
   ```

2. Navigate to **AI Tagging** panel

3. Select **Model Type**: `transformer`

4. Click **Select Labeled Data CSV** and choose your labeled data file

5. Click **Train Model**

6. View results in the output panel

## Multi-Output Classification

### What is Multi-Output?

Instead of predicting a single label, the transformer model can predict **multiple attributes simultaneously**. This is automatically enabled when the model detects composite labels (comma-separated tags).

### Composite Label Format

```
incoming,level,linear,light_maneuver,low_speed
outgoing,ascending,curved,high_maneuver,high_speed
```

### Output Categories

The model predicts 5 separate attributes:

| Output | Type | Classes | Description |
|--------|------|---------|-------------|
| **direction** | Binary | `incoming`, `outgoing` | Movement toward/away from radar |
| **altitude** | Multi-class | `ascending`, `descending`, `level` | Vertical motion pattern |
| **path** | Binary | `linear`, `curved` | Path curvature |
| **maneuver** | Binary | `light_maneuver`, `high_maneuver` | Maneuver intensity |
| **speed** | Binary | `low_speed`, `high_speed` | Speed classification |

### Multi-Output Advantages

✅ **Better Feature Learning**: Each output head learns specialized representations  
✅ **Shared Representations**: Lower layers capture common trajectory patterns  
✅ **Handles Composite Labels**: No need for label transformation  
✅ **Detailed Predictions**: Provides granular classification per attribute  
✅ **Robust to Missing Labels**: Can train even if some attributes are missing  

### Example Output

```python
{
  'accuracy': 0.92,  # Overall accuracy
  'f1_score': 0.91,
  'multi_output': True,
  'outputs': {
    'direction': {'accuracy': 0.95, 'f1_score': 0.94},
    'altitude': {'accuracy': 0.88, 'f1_score': 0.87},
    'path': {'accuracy': 0.93, 'f1_score': 0.92},
    'maneuver': {'accuracy': 0.90, 'f1_score': 0.89},
    'speed': {'accuracy': 0.94, 'f1_score': 0.93}
  }
}
```

## When to Use Transformer vs LSTM vs XGBoost

### Use **Transformer** when:
- ✅ You have **composite labels** (multi-attribute classification)
- ✅ You want to capture **long-range dependencies** (20+ time steps)
- ✅ Training time is not critical (Transformer is faster than LSTM but slower than XGBoost)
- ✅ You have sufficient data (>1000 sequences recommended)
- ✅ You want **state-of-the-art performance** on sequence data

### Use **LSTM** when:
- ✅ You have sequential data with temporal dependencies
- ✅ You need a proven, reliable model
- ✅ You have moderate amounts of data (500+ sequences)
- ✅ Single-output classification is sufficient

### Use **XGBoost** when:
- ✅ You want **fast training** and inference
- ✅ You have tabular/aggregated features (per-track statistics)
- ✅ Interpretability is important
- ✅ You have limited data (<500 samples)
- ✅ Sequence information is less critical

## Performance Comparison

| Model | Training Speed | Accuracy | Memory Usage | Best For |
|-------|---------------|----------|--------------|----------|
| XGBoost | ⚡⚡⚡ Fast | 85-90% | Low | Tabular features, small datasets |
| LSTM | ⚡⚡ Moderate | 88-92% | Medium | Sequential data, proven reliability |
| Transformer | ⚡ Slower | 90-95% | Medium-High | Multi-output, long sequences, SOTA |

*Performance varies based on dataset quality and size*

## Training Tips

### 1. Data Requirements

- **Minimum**: 500 sequences (tracks) for basic training
- **Recommended**: 2,000+ sequences for good generalization
- **Optimal**: 5,000+ sequences for best performance

### 2. Hyperparameter Tuning

**For small datasets** (<1000 sequences):
```json
{
  "d_model": 32,
  "num_heads": 2,
  "num_layers": 1,
  "dropout": 0.2,
  "epochs": 30
}
```

**For medium datasets** (1000-5000 sequences):
```json
{
  "d_model": 64,
  "num_heads": 4,
  "num_layers": 2,
  "dropout": 0.1,
  "epochs": 50
}
```

**For large datasets** (5000+ sequences):
```json
{
  "d_model": 128,
  "num_heads": 8,
  "num_layers": 3,
  "dropout": 0.1,
  "epochs": 100
}
```

### 3. Avoiding Overfitting

- Increase `dropout` (0.2-0.3)
- Reduce `d_model` and `ff_dim`
- Reduce `num_layers`
- Use more training data
- Enable data augmentation (if available)

### 4. Improving Performance

- Increase `d_model` (64 → 128)
- Add more transformer layers (2 → 3-4)
- Increase `sequence_length` if trajectories are long
- Tune learning rate (not exposed in config, defaults to Adam optimizer)
- Use validation split to monitor training

## Troubleshooting

### Issue: "TensorFlow is required for Transformer model"

**Solution**: Install TensorFlow
```bash
pip install tensorflow
# Or for GPU support:
pip install tensorflow[and-cuda]
```

### Issue: Low accuracy on multi-output predictions

**Diagnosis**: Check if all output categories have balanced data

**Solutions**:
1. Analyze label distribution:
   ```bash
   python analyze_label_diversity.py labeled_data.csv
   ```
2. Ensure each output category has examples in training data
3. Try single-output mode if multi-output is not needed

### Issue: Training is very slow

**Solutions**:
1. Reduce `batch_size` if running out of memory
2. Reduce `d_model` and `ff_dim`
3. Reduce `num_layers`
4. Consider using GPU acceleration
5. Use fewer epochs for initial experiments

### Issue: Model doesn't detect composite labels

**Diagnosis**: Check if labels are properly formatted with commas

**Example Correct Format**:
```
incoming,level,linear,light_maneuver,low_speed
```

**Solution**: Ensure `Annotation` column uses comma-separated format

### Issue: "d_model must be divisible by num_heads"

**Solution**: Adjust parameters so `d_model % num_heads == 0`

Valid combinations:
- `d_model=64, num_heads=4` ✅
- `d_model=64, num_heads=8` ✅
- `d_model=128, num_heads=4` ✅
- `d_model=64, num_heads=5` ❌ (64 not divisible by 5)

## Advanced Features

### Custom Loss Weights

For multi-output training, you can weight different outputs differently (requires code modification):

```python
# In TransformerModel.build_model()
self.model.compile(
    optimizer='adam',
    loss=losses,
    loss_weights={
        'direction': 1.0,
        'altitude': 1.5,  # Higher weight for altitude
        'path': 1.0,
        'maneuver': 1.2,  # Higher weight for maneuver
        'speed': 1.0
    },
    metrics=metrics_dict
)
```

### Attention Visualization

Extract attention weights to understand what the model focuses on:

```python
# Get attention layer
attention_layer = model.model.layers[X]  # Find the TransformerBlock layer

# Get attention weights during prediction
attention_weights = attention_layer.att.attention_weights
```

### Transfer Learning

Fine-tune a pre-trained model on new data:

```python
# Load pre-trained model
model = TransformerModel()
model.load('pretrained_transformer.h5')

# Freeze lower layers
for layer in model.model.layers[:-2]:
    layer.trainable = False

# Train on new data
model.train(df_new_train, df_new_val)
```

## Model Artifacts

After training, the following files are saved:

```
output/models/
├── transformer_model.h5              # Keras model (architecture + weights)
├── transformer_model_metadata.pkl    # Scaler, encoder, parameters
└── transformer_metrics.json          # Training metrics
```

## Citation

If you use the Transformer model in your research, please cite:

```
Vaswani, A., et al. (2017). "Attention is All You Need."
Advances in Neural Information Processing Systems.
```

## Contributing

Contributions to improve the Transformer model are welcome:

- Add attention visualization tools
- Implement custom attention mechanisms
- Add support for variable-length sequences
- Optimize multi-output architecture
- Add more output categories

## Version History

- **v1.0** (2025) - Initial implementation with multi-output support

---

**Questions or Issues?** Refer to `README.md` or contact the development team.
