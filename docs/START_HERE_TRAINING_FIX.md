# 🚀 START HERE: Fix Your Training Error

## You Got This Error ❌

```
✗ Training error: Insufficient classes for training. 
Found 1 unique class(es): ['incoming,level,linear,light_maneuver,low_speed']
```

---

## Quick Fix in 3 Steps ✅

### Step 1: Copy Your CSV to Linux (if needed)

If your file is on Windows (`D:/Zoppler Projects/...`), copy it to Linux first:

```bash
# Create a data directory
mkdir -p /workspace/data

# Copy from Windows drive (adjust path as needed)
cp "/mnt/d/Zoppler Projects/RadarAnnotation_LRDE/Database/labelled_data_1.csv" /workspace/data/

# Or if the path has spaces, use quotes
cp "/mnt/d/Zoppler Projects/RadarAnnotation_LRDE/Database/labelled_data_1.csv" /workspace/data/labelled_data_1.csv
```

---

### Step 2: Analyze Your Data

```bash
python3 analyze_label_diversity.py /workspace/data/labelled_data_1.csv
```

**What this shows you:**
- ✓ How many unique labels you have (currently 1, need 2+)
- ✓ How many tracks in your data
- ✓ What motion characteristics are uniform
- ✓ Specific fix recommendations

---

### Step 3: Apply the Fix

**Most Common Solution** - Create per-track labels:

```bash
python3 create_track_labels.py /workspace/data/labelled_data_1.csv
```

This creates: `/workspace/data/labelled_data_1_track_labeled.csv`

**Verify it worked:**

```bash
python3 -c "import pandas as pd; df = pd.read_csv('/workspace/data/labelled_data_1_track_labeled.csv'); print(f'Unique labels: {df[\"Annotation\"].nunique()}'); print(df['Annotation'].value_counts())"
```

You should see **2 or more unique labels** now! 🎉

---

### Step 4: Train with Fixed Data

**Option A: Use GUI**
1. Open the application
2. Go to **AI Tagging** tab
3. Click **"Select Labeled Data CSV"**
4. Choose: `labelled_data_1_track_labeled.csv`
5. Click **"Train Model"**

**Option B: Use Command Line**
```bash
python3 -m src.ai_engine --model xgboost --data /workspace/data/labelled_data_1_track_labeled.csv --outdir output/models
```

Training should now work! ✅

---

## What Each Tool Does

### 🔍 analyze_label_diversity.py
**Purpose**: Diagnose why all labels are the same

**Usage**: 
```bash
python3 analyze_label_diversity.py <input.csv>
```

**Output**:
- Unique annotation count
- Distribution of each label
- Individual flag statistics
- Feature value distributions
- Specific recommendations for YOUR data

---

### ✅ create_track_labels.py (PRIMARY FIX)
**Purpose**: Convert uniform point-labels into distinct track-labels

**Usage**:
```bash
python3 create_track_labels.py <input.csv> [output.csv] [strategy]
```

**Strategies**:
- `primary` (default) - Extract most important tag
- `majority` - Use most common annotation per track
- `first` - Use first point's annotation
- `last` - Use last point's annotation

**Example**:
```bash
# Use default strategy
python3 create_track_labels.py data.csv

# Specify output and strategy
python3 create_track_labels.py data.csv output.csv primary
```

**What it does**:
```
BEFORE: All 1000 points have 'incoming,level,linear,light_maneuver,low_speed'

AFTER:  Track 1 → 'incoming'
        Track 2 → 'outgoing'  
        Track 3 → 'level'
        Track 4 → 'curved'
        Track 5 → 'high_speed'
```

---

### 🔀 split_composite_labels.py (ALTERNATIVE FIX)
**Purpose**: Split composite labels into separate binary tasks

**Usage**:
```bash
python3 split_composite_labels.py <input.csv> [output.csv] [target_label]
```

**Examples**:
```bash
# Analyze all labels
python3 split_composite_labels.py data.csv

# Create binary classification for 'incoming'
python3 split_composite_labels.py data.csv output.csv incoming
```

**What it does**:
- Creates columns: `label_incoming`, `label_level`, `label_linear`, etc.
- Each column is True/False
- Shows which flags vary in your data
- Enables training separate models for each characteristic

---

## Alternative Solutions

### Solution A: Adjust Auto-Labeling Thresholds

If your data HAS variation but thresholds are wrong:

1. **Edit config file**:
   ```bash
   nano config/default_config.json
   ```

2. **Modify thresholds** (example changes):
   ```json
   {
     "autolabel_thresholds": {
       "low_speed_threshold": 20.0,      // Was 50.0
       "high_speed_threshold": 100.0,    // Was 200.0
       "curvature_threshold": 0.005,     // Was 0.01
       "range_rate_threshold": 0.5       // Was 1.0
     }
   }
   ```

3. **Re-run auto-labeling** on raw data
4. **Check diversity** again:
   ```bash
   python3 analyze_label_diversity.py new_labeled_data.csv
   ```

---

### Solution B: Collect More Diverse Data

If your data is genuinely uniform, you need variety:

**What to include:**
- ✈️ Different directions (incoming, outgoing, tangential)
- ⬆️ Different altitudes (level, ascending, descending)
- 🏃 Different speeds (slow, medium, fast)
- 🔄 Different paths (straight, curved, circular)
- 💨 Different maneuvers (gentle, aggressive)

**Minimum requirements:**
- At least 2 distinct motion types
- At least 3 tracks per type
- At least 10 points per track

---

## Understanding Your Error

### Why All Labels Are the Same

Your label: `'incoming,level,linear,light_maneuver,low_speed'`

This means ALL your data points:
- 📍 Move **incoming** (toward radar)
- ⬌ Fly **level** (constant altitude)
- ➡️ Follow **linear** paths
- 🎯 Have **light maneuvers**
- 🐌 Travel at **low speed**

**Possible reasons:**
1. Single trajectory type (one aircraft, one motion)
2. Short data segment (brief snapshot)
3. Incorrect thresholds for your data
4. Limited training dataset

---

## Verification Commands

### Check number of unique labels:
```bash
python3 -c "import pandas as pd; df = pd.read_csv('data.csv'); print(f'Unique: {df[\"Annotation\"].nunique()}')"
```

### Check label distribution:
```bash
python3 -c "import pandas as pd; df = pd.read_csv('data.csv'); print(df['Annotation'].value_counts())"
```

### Check number of tracks:
```bash
python3 -c "import pandas as pd; df = pd.read_csv('data.csv'); print(f'Tracks: {df[\"trackid\"].nunique()}')"
```

### Check feature statistics:
```bash
python3 -c "import pandas as pd; df = pd.read_csv('data.csv'); print(df[['speed', 'range', 'curvature']].describe())"
```

---

## Troubleshooting

### "File not found" error
- Make sure you copied the CSV to Linux filesystem
- Use absolute paths: `/workspace/data/file.csv`
- Check spaces in filename (use quotes)

### Still only 1 class after fix
- Run analysis tool to understand why
- Try split_composite_labels.py instead
- Adjust auto-labeling thresholds
- Collect more diverse data

### "ModuleNotFoundError: pandas"
- Make sure you're in the virtual environment
- Install dependencies: `pip install -r requirements.txt`

---

## Complete Documentation

For detailed information, see:

1. **[INSUFFICIENT_CLASSES_FIX.md](INSUFFICIENT_CLASSES_FIX.md)** - Complete guide with examples
2. **[LABEL_DIVERSITY_GUIDE.md](LABEL_DIVERSITY_GUIDE.md)** - In-depth reference
3. **[SOLUTION_SUMMARY.md](SOLUTION_SUMMARY.md)** - What was fixed and why
4. **[README.md](README.md)** - Updated with troubleshooting section

---

## Quick Reference

| Task | Command |
|------|---------|
| Analyze problem | `python3 analyze_label_diversity.py data.csv` |
| Fix (track labels) | `python3 create_track_labels.py data.csv` |
| Fix (split labels) | `python3 split_composite_labels.py data.csv` |
| Verify fix | Check unique labels (see commands above) |
| Train model | Use GUI or `python3 -m src.ai_engine ...` |

---

## Summary

1. ✅ Your error means all data has the same label
2. ✅ Run `analyze_label_diversity.py` to diagnose
3. ✅ Run `create_track_labels.py` to fix (most common)
4. ✅ Verify you have 2+ unique labels
5. ✅ Train with the new CSV file
6. ✅ Success! 🎉

**Need help?** Read the full guides linked above!

Good luck! 🚀
