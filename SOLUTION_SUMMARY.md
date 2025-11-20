# Solution Summary: "Insufficient Classes for Training" Error

## Problem Resolved ✓

Your training error has been diagnosed and comprehensive solutions have been provided.

---

## What Was the Problem?

You encountered this error:

```
✗ Training error: Insufficient classes for training. 
Found 1 unique class(es): ['incoming,level,linear,light_maneuver,low_speed']
```

**Root Cause**: All your data points have the **exact same annotation label**, making machine learning impossible. Models need variety (at least 2 different classes) to learn patterns and make predictions.

Your data uses **composite labels** where multiple tags are combined:
- `'incoming,level,linear,light_maneuver,low_speed'`

When the auto-labeling engine found all your data had identical motion characteristics, it assigned the same composite label to everything.

---

## What Has Been Created

### 🔍 1. Analysis Tool

**File**: `analyze_label_diversity.py`

**What it does**: Examines your labeled data and tells you exactly why all labels are the same

**Usage**:
```bash
python analyze_label_diversity.py path/to/your/labelled_data.csv
```

**Output**: 
- Number of unique labels
- Distribution of individual flags  
- Feature statistics
- Specific recommendations for YOUR data

---

### ✅ 2. Per-Track Labeling Tool (Primary Solution)

**File**: `create_track_labels.py`

**What it does**: Converts uniform point-level labels into distinct track-level labels

**Usage**:
```bash
python create_track_labels.py path/to/your/labelled_data.csv
```

**How it helps**:
- Simplifies composite labels to single main tags
- Groups all points in a track under one label
- If you have 5 tracks with different motion, creates 5 distinct labels
- Produces a new CSV file ready for training

**Example transformation**:
```
BEFORE: All 1000 points → 'incoming,level,linear,light_maneuver,low_speed'
AFTER:  Track 1 (200 pts) → 'incoming'
        Track 2 (200 pts) → 'outgoing'
        Track 3 (200 pts) → 'level'
        Track 4 (200 pts) → 'curved'
        Track 5 (200 pts) → 'high_speed'
```

Now you have **5 classes** for training!

---

### 🔀 3. Label Splitting Tool (Alternative Solution)

**File**: `split_composite_labels.py`

**What it does**: Breaks composite labels into separate binary classification tasks

**Usage**:
```bash
python split_composite_labels.py path/to/your/labelled_data.csv
```

**How it helps**:
- Creates columns like `label_incoming`, `label_level`, `label_linear` (True/False)
- Shows which individual flags vary in your data
- Enables training separate models for each characteristic
- Useful when tracks have the same composite label but individual flags differ

**Example transformation**:
```
BEFORE: Annotation = 'incoming,level,linear,light_maneuver,low_speed'

AFTER:  label_incoming      = True
        label_level         = True
        label_linear        = True
        label_light_maneuver = True
        label_low_speed     = True
```

Train a model for each characteristic (e.g., "incoming vs outgoing", "level vs not-level")

---

### 📚 4. Comprehensive Documentation

**Primary Guides**:
1. **`INSUFFICIENT_CLASSES_FIX.md`** - Quick fix guide (start here!)
2. **`LABEL_DIVERSITY_GUIDE.md`** - Complete reference with all solutions
3. **`SOLUTION_SUMMARY.md`** - This file (overview)

**Updated Files**:
- `README.md` - Added troubleshooting section for this error
- `src/ai_engine.py` - Enhanced error messages with automatic detection and fix suggestions

---

## How to Fix Your Specific Issue

### 🎯 Recommended 3-Step Process

#### Step 1: Analyze
```bash
python analyze_label_diversity.py "D:/Zoppler Projects/RadarAnnotation_LRDE/Database/labelled_data_1.csv"
```

This tells you:
- How many tracks you have
- Why labels are uniform
- Which solution will work best

#### Step 2: Create Track Labels (Most Likely Solution)
```bash
python create_track_labels.py "D:/Zoppler Projects/RadarAnnotation_LRDE/Database/labelled_data_1.csv"
```

This creates: `labelled_data_1_track_labeled.csv`

#### Step 3: Train with New File

In the GUI:
1. Go to **AI Tagging** tab
2. Select **"labelled_data_1_track_labeled.csv"**
3. Click **"Train Model"**

The training should now succeed with multiple classes! ✅

---

## Understanding the Enhanced Error Messages

The training code now automatically detects composite label issues and provides:

```
🔍 DETECTED: Your data uses composite labels (comma-separated tags)
   Example: 'incoming,level,linear,light_maneuver,low_speed'

This happens when auto-labeling creates the same combination for all data.

QUICK FIXES:
  1. Analyze your data to understand why labels are uniform:
     → python analyze_label_diversity.py <your_csv_file>

  2. Create per-track labels (if you have multiple tracks):
     → python create_track_labels.py <your_csv_file>

  3. Split composite labels into separate binary tasks:
     → python split_composite_labels.py <your_csv_file>

  4. Adjust auto-labeling thresholds in config/default_config.json
     and re-run auto-labeling to create more varied labels

  5. Collect more diverse data with different motion patterns
```

---

## Why This Happened

Based on your label `'incoming,level,linear,light_maneuver,low_speed'`, ALL your data:

- 📍 Moves **incoming** (toward radar)
- ⬌ Flies **level** (constant altitude)
- ➡️ Follows **linear** paths (no curves)
- 🎯 Has **light maneuvers** (gentle motion)
- 🐌 Travels at **low speed**

This could mean:
1. **Single trajectory type**: Data from one aircraft doing one thing
2. **Short data segment**: Brief snapshot showing consistent motion
3. **Threshold mismatch**: Auto-labeling thresholds don't match your data characteristics
4. **Limited dataset**: Need more diverse trajectory examples

---

## What to Do If Per-Track Labels Don't Help

If `create_track_labels.py` still produces only 1 class, try:

### Option A: Split Labels
```bash
python split_composite_labels.py labelled_data_1.csv output.csv incoming
```
Creates binary "incoming vs not-incoming" classification

### Option B: Adjust Thresholds

Edit `config/default_config.json`:
```json
{
  "autolabel_thresholds": {
    "low_speed_threshold": 50.0,      // Try 20 or 100
    "high_speed_threshold": 200.0,
    "curvature_threshold": 0.01,      // Try 0.005 for more sensitivity
    "range_rate_threshold": 1.0       // For incoming/outgoing detection
  }
}
```

Then re-run auto-labeling on your raw data.

### Option C: Collect More Data

Ensure your dataset includes:
- Different speeds (slow, medium, fast)
- Different directions (incoming, outgoing, tangential)
- Different altitudes (level, ascending, descending)
- Different paths (straight, curved, circular)
- Different maneuvers (gentle, aggressive)

---

## Files Created/Modified

### New Files
- ✅ `analyze_label_diversity.py` - Data analysis tool
- ✅ `create_track_labels.py` - Per-track labeling tool
- ✅ `split_composite_labels.py` - Label splitting tool
- ✅ `INSUFFICIENT_CLASSES_FIX.md` - Quick fix guide
- ✅ `LABEL_DIVERSITY_GUIDE.md` - Complete reference
- ✅ `SOLUTION_SUMMARY.md` - This summary

### Modified Files
- ✅ `src/ai_engine.py` - Enhanced error detection and messages
- ✅ `README.md` - Added troubleshooting section

---

## Quick Reference Commands

| Task | Command |
|------|---------|
| Analyze data | `python analyze_label_diversity.py data.csv` |
| Create track labels | `python create_track_labels.py data.csv` |
| Split composite labels | `python split_composite_labels.py data.csv` |
| Train with new file | Use GUI or `python -m src.ai_engine --model xgboost --data new_file.csv` |
| Check unique labels | `python -c "import pandas as pd; print(pd.read_csv('data.csv')['Annotation'].value_counts())"` |

---

## Next Steps

1. **Run the analysis tool** on your data to understand the specific issue
2. **Apply the recommended fix** (likely per-track labeling)
3. **Verify the fix worked** (should have 2+ unique labels)
4. **Train your model** with the new labeled data
5. **Enjoy successful training!** 🎉

---

## Need More Help?

- **Quick start**: Read `INSUFFICIENT_CLASSES_FIX.md`
- **Complete guide**: Read `LABEL_DIVERSITY_GUIDE.md`  
- **Understanding auto-labeling**: See annotations section in `README.md`
- **Configuration**: Check `config/default_config.json`

---

## Summary

✅ **Problem diagnosed**: All data has identical composite labels  
✅ **Tools created**: 3 utility scripts to fix the issue  
✅ **Documentation provided**: 2 comprehensive guides  
✅ **Error messages enhanced**: Automatic detection and suggestions  
✅ **README updated**: Added troubleshooting section  

**Your next action**: Run `python analyze_label_diversity.py` on your CSV file!

Good luck with your training! 🚀
