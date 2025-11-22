# High-Volume Simulation Dataset Guide

## Dataset Overview

Successfully generated a high-volume simulation dataset for training, testing, and evaluating different models.

### Dataset Specifications

- **Total Tracks**: 10 random trajectories
- **Sampling Interval**: 100ms (0.1 seconds)
- **Flight Duration**: 10 minutes per track
- **Total Data Points**: 60,000 records
- **Points per Track**: 6,000 each

### Files Generated

1. **Raw Dataset**: `data/high_volume_simulation.csv` (7.8 MB)
   - Contains basic trajectory data: time, trackid, x, y, z, vx, vy, vz, ax, ay, az

2. **Labeled Dataset**: `data/high_volume_simulation_labeled.csv` (21 MB)
   - Contains 33 features including:
     - Motion features: speed, heading, range, range_rate, curvature, etc.
     - Auto-labeled annotations with 27 unique behavior patterns
     - Classification flags for: incoming/outgoing, level flight, maneuvers, speed categories

### Data Quality

- **Valid Records**: 59,980 out of 60,000 (99.97%)
- **Annotation Distribution**: Well-balanced with 27 unique behavioral patterns
- **Top Patterns**:
  - `outgoing,level,linear,light_maneuver,high_speed`: 23.43%
  - `outgoing,linear,light_maneuver`: 14.12%
  - `outgoing,level,linear,high_maneuver`: 13.08%
  - And 24 more diverse patterns

---

## Training Models

### Recommended Data Split

```
Training Set:   70% (~42,000 records)
Validation Set: 15% (~9,000 records)
Testing Set:    15% (~9,000 records)
```

### 1. Train Transformer Model

```bash
python3 -c "from src.ai_engine import train_model; train_model('transformer', 'data/high_volume_simulation_labeled.csv', 'output/models/transformer_hv', params={'epochs': 100, 'batch_size': 64, 'sequence_length': 30})"
```

**Recommended Parameters:**
- `d_model`: 128
- `num_heads`: 8
- `ff_dim`: 256
- `num_layers`: 3
- `dropout`: 0.2
- `epochs`: 100
- `batch_size`: 64
- `sequence_length`: 30

### 2. Train LSTM Model

```bash
python3 -c "from src.ai_engine import train_model; train_model('lstm', 'data/high_volume_simulation_labeled.csv', 'output/models/lstm_hv', params={'epochs': 100, 'batch_size': 64})"
```

**Recommended Parameters:**
- `units`: 128
- `dropout`: 0.3
- `epochs`: 100
- `batch_size`: 64
- `sequence_length`: 30

### 3. Train XGBoost Model

```bash
python3 -c "from src.ai_engine import train_model; train_model('xgboost', 'data/high_volume_simulation_labeled.csv', 'output/models/xgboost_hv')"
```

**Recommended Parameters:**
- `n_estimators`: 200
- `max_depth`: 8
- `learning_rate`: 0.1

### 4. Train and Compare All Models

Create a comprehensive comparison script:

```python
# compare_models.py
from src.ai_engine import train_model
import logging

logging.basicConfig(level=logging.INFO)

data_path = 'data/high_volume_simulation_labeled.csv'

# Train all models
models = {}
metrics = {}

print("Training Transformer...")
models['transformer'], metrics['transformer'] = train_model(
    'transformer', data_path, 'output/models/transformer_hv',
    params={'epochs': 100, 'batch_size': 64}
)

print("\nTraining LSTM...")
models['lstm'], metrics['lstm'] = train_model(
    'lstm', data_path, 'output/models/lstm_hv',
    params={'epochs': 100, 'batch_size': 64}
)

print("\nTraining XGBoost...")
models['xgboost'], metrics['xgboost'] = train_model(
    'xgboost', data_path, 'output/models/xgboost_hv'
)

# Compare results
print("\n" + "="*80)
print("MODEL COMPARISON")
print("="*80)
for name, metric in metrics.items():
    print(f"\n{name.upper()}:")
    print(f"  Test Accuracy: {metric['test'].get('accuracy', 0):.4f}")
    print(f"  Test F1 Score: {metric['test'].get('f1_score', 0):.4f}")
    print(f"  Training Time: {metric['train'].get('training_time', 0):.2f}s")
```

Run with:
```bash
python3 compare_models.py
```

---

## Evaluation and Testing

### Load and Explore Data

```python
import pandas as pd

# Load labeled data
df = pd.read_csv('data/high_volume_simulation_labeled.csv')

# Basic statistics
print(f"Total records: {len(df)}")
print(f"Unique tracks: {df['trackid'].nunique()}")
print(f"Features: {len(df.columns)}")
print(f"\nColumn names:\n{df.columns.tolist()}")

# Annotation distribution
print(f"\nAnnotation distribution:")
print(df['Annotation'].value_counts().head(10))

# Feature statistics
print(f"\nSpeed statistics:")
print(df['speed'].describe())

print(f"\nAltitude statistics:")
print(df['z'].describe())
```

### Visualize Trajectories

```python
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('data/high_volume_simulation.csv')

# Plot all tracks
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 2D trajectory plot
for trackid in df['trackid'].unique():
    track = df[df['trackid'] == trackid]
    axes[0, 0].plot(track['x'], track['y'], label=f'Track {int(trackid)}')
axes[0, 0].set_xlabel('X (m)')
axes[0, 0].set_ylabel('Y (m)')
axes[0, 0].set_title('2D Trajectories')
axes[0, 0].legend()
axes[0, 0].grid(True)

# Altitude profiles
for trackid in df['trackid'].unique():
    track = df[df['trackid'] == trackid]
    axes[0, 1].plot(track['time'], track['z'], label=f'Track {int(trackid)}')
axes[0, 1].set_xlabel('Time (s)')
axes[0, 1].set_ylabel('Altitude (m)')
axes[0, 1].set_title('Altitude Profiles')
axes[0, 1].legend()
axes[0, 1].grid(True)

# Speed profiles
for trackid in df['trackid'].unique():
    track = df[df['trackid'] == trackid]
    speed = np.sqrt(track['vx']**2 + track['vy']**2 + track['vz']**2)
    axes[1, 0].plot(track['time'], speed, label=f'Track {int(trackid)}')
axes[1, 0].set_xlabel('Time (s)')
axes[1, 0].set_ylabel('Speed (m/s)')
axes[1, 0].set_title('Speed Profiles')
axes[1, 0].legend()
axes[1, 0].grid(True)

# 3D trajectory (first track only for clarity)
from mpl_toolkits.mplot3d import Axes3D
ax = fig.add_subplot(224, projection='3d')
for trackid in list(df['trackid'].unique())[:3]:  # Show first 3 tracks
    track = df[df['trackid'] == trackid]
    ax.plot(track['x'], track['y'], track['z'], label=f'Track {int(trackid)}')
ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_zlabel('Z (m)')
ax.set_title('3D Trajectories (First 3 tracks)')
ax.legend()

plt.tight_layout()
plt.savefig('trajectories_visualization.png', dpi=300)
print("Visualization saved to trajectories_visualization.png")
```

### Use GUI for Interactive Visualization

```bash
python3 -m src.gui
```

Then load `data/high_volume_simulation.csv` in the GUI for interactive exploration.

---

## Model Performance Evaluation

### Evaluate Trained Model

```python
from src.ai_engine import train_model
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# Load model and test data
# (This is automatically done during training, but for manual evaluation:)

df = pd.read_csv('data/high_volume_simulation_labeled.csv')

# Load trained model
# Example for transformer:
from tensorflow import keras
model = keras.models.load_model('output/models/transformer_hv/model.h5')

# Perform predictions and evaluate
# (Note: You'll need to prepare the data in the same format as during training)
```

### Cross-Validation

```python
from sklearn.model_selection import TimeSeriesSplit
import pandas as pd

df = pd.read_csv('data/high_volume_simulation_labeled.csv')

# Time series cross-validation
tscv = TimeSeriesSplit(n_splits=5)

for train_index, test_index in tscv.split(df):
    train_data = df.iloc[train_index]
    test_data = df.iloc[test_index]
    
    # Train and evaluate model
    # ... (your training code)
```

---

## Regenerating Dataset

To generate a new dataset with different parameters:

```bash
# Edit generate_high_volume_simulation.py or use directly:
python3 -c "
from src.sim_engine import create_large_training_dataset
csv_path = create_large_training_dataset(
    output_path='data/custom_simulation.csv',
    n_tracks=20,      # Change number of tracks
    duration_min=5    # Change duration
)
print(f'Created: {csv_path}')
"
```

Or use the dedicated script:

```bash
python3 generate_high_volume_simulation.py
```

---

## Dataset Statistics

### Trajectory Types Included

The dataset includes diverse trajectory patterns:
1. Straight low-speed flight
2. Straight high-speed flight
3. Ascending spiral
4. Descending path
5. Sharp maneuvers
6. Curved paths
7. Level flight with jitter
8. Stop-and-go patterns
9. Oscillating lateral movement
10. Complex multi-phase maneuvers

### Feature List (33 total)

**Basic Features (11):**
- time, trackid, x, y, z, vx, vy, vz, ax, ay, az

**Computed Features (10):**
- speed, speed_2d, heading, range, range_rate
- curvature, accel_magnitude, vertical_rate, altitude_change, valid_features

**Classification Flags (11):**
- incoming, outgoing
- fixed_range_ascending, fixed_range_descending
- level_flight, linear, curved
- light_maneuver, high_maneuver
- low_speed, high_speed

**Label:**
- Annotation (composite label string)

---

## Tips for Best Results

1. **Data Preprocessing**: The labeled dataset already includes normalized features and valid flags
2. **Sequence Length**: For temporal models (LSTM/Transformer), use sequence_length=30-50 for best results
3. **Batch Size**: 64 works well for this dataset size
4. **Epochs**: 100-150 epochs should be sufficient for convergence
5. **Validation**: Use time-based splitting to avoid data leakage
6. **Feature Selection**: All 33 features are useful, but you can experiment with subsets

---

## Troubleshooting

### Out of Memory Errors
- Reduce batch_size (try 32 or 16)
- Reduce sequence_length
- Use data generators instead of loading all data at once

### Poor Model Performance
- Check annotation distribution (should be balanced)
- Increase training epochs
- Tune learning rate
- Add data augmentation
- Use ensemble methods

### Slow Training
- Use GPU acceleration (TensorFlow will auto-detect CUDA)
- Reduce model complexity
- Use smaller batch sizes with gradient accumulation
- Enable mixed precision training

---

## Additional Resources

- Original dataset generation script: `generate_high_volume_simulation.py`
- Training utilities: `src/ai_engine.py`
- Auto-labeling engine: `src/autolabel_engine.py`
- Simulation engine: `src/sim_engine.py`
- GUI for visualization: `src/gui.py`

For questions or issues, refer to the main README.md or project documentation.
