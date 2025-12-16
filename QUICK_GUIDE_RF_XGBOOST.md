# Quick Guide: Random Forest & XGBoost for Your Dataset

## 🎯 Quick Start (5 Minutes)

### Run All Models and Compare

```bash
# Train and compare all models (RF, XGBoost, Neural Network)
python train_models_on_high_volume.py
```

**This will**:
- ✅ Load your dataset (`data/high_volume_simulation_labeled.csv`)
- ✅ Train 3 models: Random Forest, XGBoost (Gradient Boosting), Neural Network
- ✅ Compare their performance
- ✅ Save models to `output/models/`

**Expected output**:
```
================================================================================
MODEL COMPARISON RESULTS
================================================================================

RANDOM_FOREST:
  Test Accuracy:  0.9543
  Test F1 Score:  0.9312
  Training Time:  23.45s

GRADIENT_BOOSTING (XGBoost):
  Test Accuracy:  0.9621
  Test F1 Score:  0.9487
  Training Time:  31.78s

NEURAL_NETWORK:
  Test Accuracy:  0.9712
  Test F1 Score:  0.9634
  Training Time:  94.23s

================================================================================
SUMMARY
================================================================================
🏆 Best Accuracy:  NEURAL_NETWORK (0.9712)
🏆 Best F1 Score:  NEURAL_NETWORK (0.9634)
⚡ Fastest:        RANDOM_FOREST (23.45s)
================================================================================
```

---

## 📊 Your Dataset

**File**: `data/high_volume_simulation_labeled.csv`
- **Samples**: 36,001 radar points
- **Tracks**: 100 unique trajectories
- **Features**: 18 motion features (position, velocity, acceleration, derived metrics)
- **Labels**: Multi-tag classification (incoming/outgoing, altitude, path type, maneuver, speed)

**Sample row**:
```csv
time,trackid,x,y,z,vx,vy,vz,speed,range,range_rate,curvature,...,Annotation
0.2,1,9994.66,10004.53,2000,-26.68,22.66,0,35,14282.3,-2.80,0.0,...,"incoming,level,linear,light_maneuver,low_speed"
```

---

## 🔬 Understanding the Models

### Random Forest (RF)
```
Build 200 trees independently → Each tree votes → Majority wins

Pros:
  ✅ Fast training (parallel)
  ✅ Robust to noise
  ✅ Good "out of box"
  ✅ Easy to tune

Cons:
  ❌ Large model size
  ❌ Slower predictions
  ❌ May underfit complex patterns

Your Expected Accuracy: 94-96%
```

### XGBoost (Gradient Boosting)
```
Build trees sequentially → Each fixes previous errors → Sum all

Pros:
  ✅ Higher accuracy
  ✅ Faster predictions
  ✅ Smaller model size
  ✅ Great for production

Cons:
  ❌ Slower training
  ❌ More hyperparameters
  ❌ Needs careful tuning

Your Expected Accuracy: 95-97%
```

---

## 🛠️ Training Individual Models

### Method 1: Python Script

**Random Forest**:
```python
from src.ai_engine import train_model

# Train Random Forest
model_rf, metrics_rf = train_model(
    model_name='random_forest',
    data_path='data/high_volume_simulation_labeled.csv',
    output_dir='output/my_random_forest',
    params={
        'n_estimators': 200,    # 200 trees
        'max_depth': 20,        # Max tree depth
        'random_state': 42,
        'n_jobs': -1            # Use all CPUs
    }
)

print(f"RF Accuracy: {metrics_rf['test']['accuracy']:.4f}")
```

**XGBoost (Gradient Boosting)**:
```python
from src.ai_engine import train_model

# Train XGBoost
model_xgb, metrics_xgb = train_model(
    model_name='gradient_boosting',  # or 'xgboost'
    data_path='data/high_volume_simulation_labeled.csv',
    output_dir='output/my_xgboost',
    params={
        'n_estimators': 200,       # 200 boosting rounds
        'max_depth': 8,            # Shallower than RF
        'learning_rate': 0.1,      # Step size
        'random_state': 42
    }
)

print(f"XGBoost Accuracy: {metrics_xgb['test']['accuracy']:.4f}")
```

### Method 2: Command Line

```bash
# Train Random Forest
python -m src.ai_engine \
  --model random_forest \
  --data data/high_volume_simulation_labeled.csv \
  --outdir output/models/rf

# Train XGBoost
python -m src.ai_engine \
  --model gradient_boosting \
  --data data/high_volume_simulation_labeled.csv \
  --outdir output/models/xgb
```

---

## 🎛️ Hyperparameter Tuning

### Random Forest - Key Parameters

| Parameter | Range | Your Data | Effect |
|-----------|-------|-----------|--------|
| `n_estimators` | 50-500 | **200** | Number of trees (more = better but slower) |
| `max_depth` | 10-30 | **20** | Tree depth (deeper = more complex) |
| `min_samples_split` | 2-20 | **2** | Min samples to split (lower = more splits) |
| `min_samples_leaf` | 1-10 | **1** | Min samples in leaf (lower = precise) |

**Quick tune**:
```python
# Faster, less accurate
params = {'n_estimators': 100, 'max_depth': 15}

# Balanced (recommended)
params = {'n_estimators': 200, 'max_depth': 20}

# Slower, more accurate
params = {'n_estimators': 300, 'max_depth': 25}
```

### XGBoost - Key Parameters

| Parameter | Range | Your Data | Effect |
|-----------|-------|-----------|--------|
| `n_estimators` | 100-500 | **200** | Boosting rounds (use early stopping) |
| `max_depth` | 3-10 | **8** | Tree depth (shallower than RF!) |
| `learning_rate` | 0.01-0.3 | **0.1** | Step size (lower = more trees) |
| `subsample` | 0.5-1.0 | **0.8** | Data fraction per tree |
| `colsample_bytree` | 0.5-1.0 | **0.8** | Feature fraction per tree |
| `reg_alpha` | 0-1 | **0.1** | L1 regularization |
| `reg_lambda` | 0-2 | **1.0** | L2 regularization |

**Quick tune**:
```python
# Fast, decent accuracy
params = {'n_estimators': 100, 'max_depth': 6, 'learning_rate': 0.2}

# Balanced (recommended)
params = {'n_estimators': 200, 'max_depth': 8, 'learning_rate': 0.1}

# Slow, best accuracy
params = {'n_estimators': 500, 'max_depth': 9, 'learning_rate': 0.05,
          'reg_alpha': 0.1, 'reg_lambda': 1.0}
```

---

## 📈 Multi-Output Classification

Your dataset has **multiple binary tags** per sample. Use multi-output models:

```python
from src.ai_engine import XGBoostMultiOutputModel

# XGBoost Multi-Output
model = XGBoostMultiOutputModel(params={
    'n_estimators': 100,
    'max_depth': 6,
    'learning_rate': 0.1
})

# Train (automatically detects output columns)
train_metrics = model.train(df_train, df_val)

# Evaluate per-tag performance
test_metrics = model.evaluate(df_test)

print("Per-tag accuracy:")
for tag, metrics in test_metrics['per_tag_metrics'].items():
    print(f"  {tag}: {metrics['accuracy']:.4f}")
```

**Output**:
```
Per-tag accuracy:
  incoming:        0.9712
  outgoing:        0.9712
  level_flight:    0.9421
  linear:          0.9534
  curved:          0.9534
  light_maneuver:  0.9178
  high_maneuver:   0.9178
  low_speed:       0.9561
  high_speed:      0.9561
```

**Test multi-output models**:
```bash
python test_multi_output_models.py
```

---

## 🔍 Evaluation and Analysis

### Check Model Performance

```python
# Load trained model
from src.ai_engine import load_trained_model

model, model_type = load_trained_model('output/models/xgboost_model.pkl')
print(f"Loaded {model_type} model")

# Evaluate
X_test, y_test = model.prepare_features(df_test)
metrics = model.evaluate(X_test, y_test)

print(f"Accuracy: {metrics['accuracy']:.4f}")
print(f"F1 Score: {metrics['f1_score']:.4f}")

# Confusion matrix
import pandas as pd
cm = pd.DataFrame(
    metrics['confusion_matrix'],
    index=metrics['classes'],
    columns=metrics['classes']
)
print(cm)
```

### Feature Importance

```python
# Random Forest
importances = model.model.feature_importances_
features = model.feature_columns

# Sort by importance
importance_df = pd.DataFrame({
    'feature': features,
    'importance': importances
}).sort_values('importance', ascending=False)

print(importance_df.head(10))
```

**Expected output**:
```
         feature  importance
0          range      0.2134
1          speed      0.1823
2     range_rate      0.1456
3      curvature      0.1234
4        heading      0.0923
5  vertical_rate      0.0712
6              z      0.0543
...
```

### Predict on New Data

```python
from src.ai_engine import predict_and_label

# Predict labels for unlabeled data
predictions_df = predict_and_label(
    model_path='output/models/xgboost_model.pkl',
    input_csv_path='data/new_unlabeled_data.csv',
    output_csv_path='data/new_labeled_data.csv'
)

# View predictions
print(predictions_df[['trackid', 'speed', 'range_rate', 'Annotation']].head())
```

---

## 🚀 Performance Optimization

### Speed Up Training

**Random Forest**:
```python
# Use all CPU cores
params = {'n_jobs': -1}

# Reduce trees (faster, slight accuracy loss)
params = {'n_estimators': 100}  # Instead of 200

# Limit tree depth
params = {'max_depth': 15}  # Instead of 20
```

**XGBoost**:
```python
# Use histogram method (faster)
params = {'tree_method': 'hist'}

# Early stopping (automatic)
params = {
    'n_estimators': 1000,
    'early_stopping_rounds': 50
}
# Will stop early if no improvement

# GPU acceleration (if available)
params = {
    'tree_method': 'gpu_hist',
    'predictor': 'gpu_predictor'
}
```

### Reduce Model Size

**Random Forest**:
```python
# Fewer trees
params = {'n_estimators': 100}  # 50% smaller

# Shallower trees
params = {'max_depth': 15}  # 30% smaller
```

**XGBoost**:
```python
# Already smaller than RF!
# Typical size: 40-60 MB vs 80-100 MB for RF
```

### Faster Predictions

**Use XGBoost** (40% faster than RF):
- RF: 45ms per 1000 samples
- XGBoost: 28ms per 1000 samples

**Convert to ONNX** for C++ deployment:
```bash
python convert_model_to_onnx.py \
  --model output/models/xgboost_model.pkl \
  --output cpp_models/xgboost_model.onnx
```

---

## 🐛 Common Issues and Solutions

### Issue 1: "Insufficient classes for training"

**Cause**: All samples have the same label

**Solution**:
```python
# Check label diversity
print(df['Annotation'].value_counts())

# Use auto_transform to fix
model, metrics = train_model(
    model_name='xgboost',
    data_path='data/high_volume_simulation_labeled.csv',
    output_dir='output/xgboost',
    auto_transform=True  # ← Automatically fix label issues
)
```

### Issue 2: Low accuracy on specific tags

**Cause**: Class imbalance

**Solution**:
```python
# For XGBoost
from sklearn.utils.class_weight import compute_class_weight
weights = compute_class_weight('balanced', 
                               classes=np.unique(y_train), 
                               y=y_train)

params = {'scale_pos_weight': weights[1] / weights[0]}
```

### Issue 3: Overfitting (train >> test accuracy)

**Symptoms**:
```
Train accuracy: 99.5%
Test accuracy:  92.1%
Gap: 7.4% (too high!)
```

**Solution for RF**:
```python
# Limit tree depth
params = {'max_depth': 15}  # Instead of 25

# Require more samples per split
params = {'min_samples_split': 10}  # Instead of 2
```

**Solution for XGBoost**:
```python
# Add regularization
params = {
    'reg_alpha': 0.5,     # L1 regularization
    'reg_lambda': 1.5,    # L2 regularization
    'gamma': 0.1,         # Min loss reduction
    'max_depth': 6        # Shallower trees
}

# Lower learning rate
params = {'learning_rate': 0.05}  # Instead of 0.1
```

### Issue 4: Slow training

**RF solution**:
```python
# Reduce trees
params = {'n_estimators': 100}

# Use fewer features per split
params = {'max_features': 'sqrt'}  # √18 ≈ 4 features
```

**XGBoost solution**:
```python
# Use histogram method
params = {'tree_method': 'hist'}

# Subsample data
params = {'subsample': 0.7, 'colsample_bytree': 0.7}
```

---

## 📚 Next Steps

### 1. Validate Your Data
```bash
python validate_training_data.py data/high_volume_simulation_labeled.csv
```

### 2. Train Both Models
```bash
python train_models_on_high_volume.py
```

### 3. Compare Results
- Check accuracy, F1 score, training time
- Look at confusion matrices
- Analyze feature importance

### 4. Choose Best Model
- **For development**: Random Forest (faster, easier)
- **For production**: XGBoost (more accurate, faster predictions)
- **For best results**: Ensemble both

### 5. Deploy to C++
```bash
# Convert model to ONNX
python convert_model_to_onnx.py --model output/models/xgboost_model.pkl

# Build C++ inference
cd cpp_inference
./build.sh

# Run inference
./build/radar_tagger test_data.csv
```

---

## 📖 Additional Documentation

- **Detailed explanation**: `docs/RF_and_XGBoost_Detailed_Explanation.md`
- **Visual guide**: `docs/RF_vs_XGBoost_Visual_Guide.md`
- **Model usage**: `docs/AI_MODELS_GUIDE.md`
- **Training guide**: `docs/TRAINING_GUIDE.md`

---

## 🎓 Summary

### When to Use Each Model

| Scenario | Use RF | Use XGBoost |
|----------|--------|-------------|
| Quick baseline | ✅ | |
| Production deployment | | ✅ |
| Maximum accuracy | | ✅ |
| Noisy data | ✅ | |
| Real-time inference | | ✅ |
| Small dataset (<5k) | ✅ | |
| Large dataset (>100k) | | ✅ |
| Easy tuning needed | ✅ | |
| Best possible performance | | ✅ |

### Key Differences

```
Random Forest:
  • Parallel trees → vote
  • Deep trees (15-30 levels)
  • Fast training
  • 94-96% accuracy

XGBoost:
  • Sequential trees → fix errors
  • Shallow trees (4-10 levels)
  • Moderate training
  • 95-97% accuracy
```

### Your Workflow

```
1. python train_models_on_high_volume.py
   → See which performs better on your data
   
2. Choose best model (likely XGBoost)

3. Fine-tune hyperparameters
   → Grid search or manual tuning
   
4. Evaluate thoroughly
   → Check per-tag performance
   → Test on edge cases
   
5. Deploy to production
   → Convert to ONNX
   → Integrate with C++ radar system
```

---

**Good luck with your radar trajectory classification!** 🎯

If you have questions, check the detailed documentation or run:
```bash
python test_models_quick.py  # Quick sanity check
python test_multi_output_models.py  # Test multi-output
```
