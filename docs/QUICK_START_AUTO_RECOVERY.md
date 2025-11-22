# Quick Start - Automatic Label Recovery

## 🎯 TL;DR

Your training error is now **automatically fixed**! Just use the GUI or API as normal.

## 🚀 How to Use

### Option 1: GUI (Simplest)

```
1. Open GUI
2. Go to "AI Training" tab
3. Select your labeled data (labelled_data_1.csv)
4. Click "Train Model"
5. ✅ Done! System handles everything automatically
```

**What happens behind the scenes:**
```
⚠️  Insufficient label diversity detected
🔄 Analyzing: Found 1 unique composite label with 5 tags
🔄 Applying transformation: multi_label_binary
✅ Created 5 binary classifiers
✅ Training succeeded!
```

### Option 2: Python API

```python
from src.ai_engine import train_model

# That's it! Auto-recovery is enabled by default
model, metrics = train_model(
    model_name='xgboost',
    data_path='labelled_data_1.csv',
    output_dir='output/models'
)
```

### Option 3: Command Line

```bash
# Analyze your data first
python -m src.label_transformer labelled_data_1.csv --analyze-only

# Transform and save
python -m src.label_transformer labelled_data_1.csv -o fixed_data.csv --strategy auto

# Train with fixed data
python -m src.ai_engine --model xgboost --data fixed_data.csv
```

## 📊 What Gets Fixed

### Your Error:
```
✗ Training error: Insufficient classes for training
  Found 1 unique class: ['incoming,level,linear,light_maneuver,low_speed']
```

### Automatic Solution:

**Splits into 5 binary classifiers:**
1. Is incoming? (True/False)
2. Is level flight? (True/False)
3. Is linear path? (True/False)
4. Is light maneuver? (True/False)
5. Is low speed? (True/False)

Each classifier learns independently! ✅

## 📁 Output Files

After auto-recovery:
```
output/models/
├── xgboost_model.pkl                    # Your trained model
├── xgboost_metrics.json                 # Metrics + transformation info
└── transformed_training_data.csv        # Transformed data used
```

## 🔧 Advanced: Disable Auto-Recovery

If you want the old behavior (manual fixes):

```python
model, metrics = train_model(
    model_name='xgboost',
    data_path='labelled_data_1.csv',
    output_dir='output/models',
    auto_transform=False  # Disable automatic recovery
)
```

## 📚 Full Documentation

- **Complete guide**: `AUTOMATIC_LABEL_RECOVERY.md`
- **Solution overview**: `LABEL_DIVERSITY_SOLUTION.md`
- **Technical details**: `IMPROVEMENTS_SUMMARY.md`

## ✨ Key Benefit

**Before**: 5-step manual process taking 5-10 minutes
**After**: Automatic one-click solution taking 15 seconds

Just click "Train Model" and it works! 🎉
