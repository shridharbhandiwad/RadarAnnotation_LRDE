# Machine Learning Models - Hyperparameter Tuning Guide

## 📊 Overview

This document provides comprehensive hyperparameter tuning values for all three machine learning models used in the Radar Data Annotation System:
- **XGBoost** (Gradient Boosting)
- **Random Forest** (Ensemble Learning)
- **Transformer** (Deep Learning with Self-Attention)

---

## 🎯 Model Comparison

| Model | Type | Speed | Accuracy | Best Use Case |
|-------|------|-------|----------|---------------|
| **XGBoost** | Gradient Boosting | ⚡⚡⚡ Fast | 85-90% | Quick inference, tabular data |
| **Random Forest** | Ensemble Trees | ⚡⚡ Moderate | 83-88% | Robust, interpretable results |
| **Transformer** | Self-Attention | ⚡ Slower | 90-95% | Multi-output, complex patterns |

---

## 🔧 XGBoost Hyperparameters

### Current Configuration
```json
{
  "n_estimators": 100,
  "max_depth": 6,
  "learning_rate": 0.1,
  "objective": "multi:softmax",
  "random_state": 42
}
```

### Detailed Parameters

| Parameter | Current | Tuning Range | Impact | Priority |
|-----------|---------|--------------|--------|----------|
| `n_estimators` | 100 | 50-300 | Number of boosting rounds. Higher = more complex model, risk of overfitting | HIGH |
| `max_depth` | 6 | 3-10 | Maximum tree depth. Controls model complexity | HIGH |
| `learning_rate` | 0.1 | 0.01-0.3 | Step size for weight updates. Lower = slower but more precise | HIGH |
| `subsample` | 1.0 | 0.7-1.0 | Fraction of samples for each tree. <1.0 reduces overfitting | MEDIUM |
| `colsample_bytree` | 1.0 | 0.7-1.0 | Fraction of features per tree. Adds randomness | MEDIUM |
| `min_child_weight` | 1 | 1-10 | Minimum sum of weights in child node | MEDIUM |
| `gamma` | 0 | 0-5 | Minimum loss reduction for split. Higher = more conservative | LOW |
| `reg_alpha` | 0 | 0-1 | L1 regularization term | LOW |
| `reg_lambda` | 1 | 0-10 | L2 regularization term | LOW |
| `random_state` | 42 | Fixed | Reproducibility seed | - |

### Recommended Tuning Strategy

**Phase 1: Coarse Grid Search**
```python
{
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 6, 9],
    'learning_rate': [0.01, 0.1, 0.3]
}
```

**Phase 2: Fine-Tuning**
```python
{
    'n_estimators': [80, 100, 120],
    'max_depth': [5, 6, 7],
    'learning_rate': [0.08, 0.1, 0.12],
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.8, 0.9, 1.0]
}
```

---

## 🌲 Random Forest Hyperparameters

### Current Configuration
```json
{
  "n_estimators": 100,
  "max_depth": 15,
  "min_samples_split": 2,
  "min_samples_leaf": 1,
  "random_state": 42,
  "n_jobs": -1
}
```

### Detailed Parameters

| Parameter | Current | Tuning Range | Impact | Priority |
|-----------|---------|--------------|--------|----------|
| `n_estimators` | 100 | 50-300 | Number of trees in forest. More = better but slower | HIGH |
| `max_depth` | 15 | 10-30 | Maximum tree depth. None = unlimited (risk overfitting) | HIGH |
| `min_samples_split` | 2 | 2-20 | Minimum samples required to split internal node | MEDIUM |
| `min_samples_leaf` | 1 | 1-10 | Minimum samples required at leaf node | MEDIUM |
| `max_features` | 'sqrt' | 'sqrt'/'log2'/None | Features to consider for best split | MEDIUM |
| `bootstrap` | True | True/False | Whether to use bootstrap samples | LOW |
| `max_samples` | None | 0.5-1.0 | Fraction of samples for each tree (if bootstrap=True) | LOW |
| `criterion` | 'gini' | 'gini'/'entropy' | Split quality measure | LOW |
| `n_jobs` | -1 | Fixed | Parallel processing (all cores) | - |
| `random_state` | 42 | Fixed | Reproducibility seed | - |

### Recommended Tuning Strategy

**Phase 1: Coarse Grid Search**
```python
{
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 15, 20, None],
    'min_samples_split': [2, 5, 10]
}
```

**Phase 2: Fine-Tuning**
```python
{
    'n_estimators': [80, 100, 120],
    'max_depth': [12, 15, 18],
    'min_samples_split': [2, 3, 4],
    'min_samples_leaf': [1, 2, 3],
    'max_features': ['sqrt', 'log2']
}
```

---

## 🤖 Transformer Hyperparameters

### Current Configuration
```json
{
  "d_model": 64,
  "num_heads": 4,
  "ff_dim": 128,
  "num_layers": 2,
  "dropout": 0.1,
  "epochs": 50,
  "batch_size": 32,
  "sequence_length": 20
}
```

### Architecture Parameters

| Parameter | Current | Tuning Range | Impact | Priority |
|-----------|---------|--------------|--------|----------|
| `d_model` | 64 | 32-256 | Model dimension. Higher = more capacity but slower | HIGH |
| `num_heads` | 4 | 2-16 | Number of attention heads. Must divide d_model evenly | HIGH |
| `num_layers` | 2 | 1-6 | Number of transformer blocks. Deeper = more complex | HIGH |
| `ff_dim` | 128 | 64-512 | Feed-forward network dimension. Usually 2-4x d_model | MEDIUM |
| `dropout` | 0.1 | 0.1-0.5 | Dropout rate for regularization | MEDIUM |
| `sequence_length` | 20 | 10-50 | Input sequence length. Longer = more context | MEDIUM |

### Training Parameters

| Parameter | Current | Tuning Range | Impact | Priority |
|-----------|---------|--------------|--------|----------|
| `epochs` | 50 | 20-100 | Training iterations. Monitor validation loss | HIGH |
| `batch_size` | 32 | 16-128 | Samples per batch. Higher = faster but more memory | HIGH |
| `learning_rate` | 0.001 | 0.0001-0.01 | Optimizer step size. Critical for convergence | HIGH |
| `optimizer` | 'adam' | adam/adamw/sgd | Optimization algorithm | MEDIUM |
| `weight_decay` | 0.0001 | 0-0.01 | L2 regularization strength | MEDIUM |
| `warmup_steps` | 0 | 0-1000 | Learning rate warmup period | LOW |
| `patience` | 10 | 5-20 | Early stopping patience (epochs) | LOW |

### Recommended Tuning Strategy

**Phase 1: Architecture Search**
```python
{
    'd_model': [32, 64, 128],
    'num_heads': [2, 4, 8],
    'num_layers': [1, 2, 3]
}
```

**Phase 2: Fine-Tuning**
```python
{
    'd_model': [56, 64, 72],
    'num_heads': [4, 6, 8],
    'ff_dim': [96, 128, 160],
    'dropout': [0.1, 0.2, 0.3]
}
```

**Phase 3: Training Optimization**
```python
{
    'learning_rate': [0.0005, 0.001, 0.002],
    'batch_size': [16, 32, 64],
    'epochs': [30, 50, 70]
}
```

---

## 📈 Hyperparameter Tuning Methodology

### 1. Grid Search
- **Method**: Exhaustive search over specified parameter grid
- **Pros**: Comprehensive, guaranteed to find best in grid
- **Cons**: Computationally expensive, scales poorly
- **Use for**: 2-3 parameters with small ranges

### 2. Random Search
- **Method**: Random sampling from parameter distributions
- **Pros**: More efficient than grid search, good for large spaces
- **Cons**: May miss optimal combinations
- **Use for**: Initial exploration, 4+ parameters

### 3. Bayesian Optimization
- **Method**: Probabilistic model guides search (e.g., using Optuna, Hyperopt)
- **Pros**: Smart convergence, fewer iterations needed
- **Cons**: Complex setup, requires additional libraries
- **Use for**: Final optimization, expensive models (Transformer)

### 4. Cross-Validation Strategy
```python
from sklearn.model_selection import StratifiedKFold

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
```

---

## 🎯 Performance Metrics

### Primary Metrics
1. **Accuracy**: Overall classification accuracy
2. **F1-Score**: Harmonic mean of precision and recall (per class)
3. **Macro-averaged F1**: Average F1 across all classes
4. **Per-Tag Accuracy**: Individual tag prediction accuracy

### Secondary Metrics
1. **Training Time**: Time to train model
2. **Inference Speed**: Prediction time per sample
3. **Model Size**: Disk space for saved model
4. **Memory Usage**: RAM during training/inference

---

## ⚙️ Training Configuration

### Data Split Strategy
```python
{
    "train_test_split": 0.8,  # 80% train, 20% test
    "validation_split": 0.2,  # 20% of train for validation
    "stratify": True,         # Maintain class distribution
    "random_state": 42        # Reproducibility
}
```

### Hardware Recommendations

| Model | CPU | GPU | RAM | Training Time |
|-------|-----|-----|-----|---------------|
| XGBoost | ✅ Excellent | ⚡ Optional | 2-4 GB | 2-5 min |
| Random Forest | ✅ Good | ❌ N/A | 4-8 GB | 5-10 min |
| Transformer | ⚠️ Slow | ✅ Recommended | 8-16 GB | 10-30 min |

---

## 🚀 Quick Start: Hyperparameter Tuning

### Example: XGBoost with Grid Search

```python
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier

# Define parameter grid
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 6, 9],
    'learning_rate': [0.01, 0.1, 0.3],
    'subsample': [0.8, 0.9, 1.0]
}

# Create model
model = XGBClassifier(random_state=42)

# Grid search with cross-validation
grid_search = GridSearchCV(
    model, 
    param_grid, 
    cv=5, 
    scoring='f1_macro',
    n_jobs=-1,
    verbose=2
)

# Fit
grid_search.fit(X_train, y_train)

# Best parameters
print("Best parameters:", grid_search.best_params_)
print("Best score:", grid_search.best_score_)
```

### Example: Transformer with Manual Tuning

```python
from src.ai_engine import TransformerMultiOutputModel

# Test different configurations
configs = [
    {'d_model': 32, 'num_heads': 2, 'num_layers': 2},
    {'d_model': 64, 'num_heads': 4, 'num_layers': 2},
    {'d_model': 128, 'num_heads': 8, 'num_layers': 3},
]

best_accuracy = 0
best_config = None

for config in configs:
    model = TransformerMultiOutputModel(params=config)
    model.fit(X_train, y_train)
    accuracy = model.evaluate(X_test, y_test)['accuracy']
    
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_config = config
    
    print(f"Config: {config}, Accuracy: {accuracy:.4f}")

print(f"\nBest Config: {best_config}")
print(f"Best Accuracy: {best_accuracy:.4f}")
```

---

## 📊 Expected Results

### Baseline Performance (Current Hyperparameters)
- XGBoost: 85-90% accuracy
- Random Forest: 83-88% accuracy
- Transformer: 90-95% accuracy

### After Tuning (Expected Improvement)
- XGBoost: +3-5% accuracy
- Random Forest: +2-4% accuracy
- Transformer: +2-3% accuracy

### Target Goals
- Overall Accuracy: ≥90%
- Per-Tag F1-Score: ≥0.85
- Training Time: <30 minutes
- Inference Speed: <100ms per prediction

---

## ⚠️ Common Pitfalls & Solutions

### 1. Overfitting
**Symptoms**: High training accuracy, low validation accuracy
**Solutions**:
- Increase dropout (Transformer)
- Reduce max_depth (XGBoost, Random Forest)
- Add L1/L2 regularization
- Use early stopping
- Increase training data

### 2. Underfitting
**Symptoms**: Low training and validation accuracy
**Solutions**:
- Increase model complexity (d_model, num_layers)
- More training epochs
- Lower learning rate
- More features

### 3. Slow Training
**Solutions**:
- Reduce batch_size
- Decrease sequence_length
- Use GPU for Transformer
- Parallelize with n_jobs=-1

### 4. Class Imbalance
**Solutions**:
- Use class_weight='balanced'
- Apply SMOTE oversampling
- Use stratified sampling
- Adjust threshold for predictions

---

## 📚 References

### Code Locations
- **Configuration**: `/workspace/config/default_config.json`
- **AI Engine**: `/workspace/src/ai_engine.py`
- **Training Script**: `/workspace/train_multi_output_models.py`
- **GUI Integration**: `/workspace/src/gui.py`

### Documentation
- `README.md` - System overview
- `QUICK_START_MODELS.md` - Model training guide
- `QUICK_START_HIGH_VOLUME_GUI.md` - High-volume training
- `TRANSFORMER_MODEL_GUIDE.md` - Transformer architecture details

### External Resources
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)
- [TensorFlow Tutorials](https://www.tensorflow.org/tutorials)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Original Transformer paper

---

## 🔄 Implementation Timeline

### Week 1: Baseline Validation
- Verify current configuration
- Establish baseline metrics
- Set up experiment tracking

### Week 2-3: Grid/Random Search
- XGBoost tuning (3 days)
- Random Forest tuning (2 days)
- Transformer architecture search (5 days)

### Week 4-5: Fine-Tuning
- Bayesian optimization
- Cross-validation
- Ensemble experiments

### Week 6: Validation & Documentation
- Final model validation
- Performance benchmarking
- Documentation & deployment

---

## ✅ Success Criteria

- [ ] Accuracy ≥90% on validation set
- [ ] F1-Score ≥0.85 for all tag categories
- [ ] Training time <30 minutes
- [ ] Inference speed <100ms per track
- [ ] Model size <500 MB
- [ ] Reproducible results (fixed seeds)
- [ ] Complete documentation

---

**Last Updated**: November 2025  
**Version**: 1.0  
**Status**: Ready for Implementation
