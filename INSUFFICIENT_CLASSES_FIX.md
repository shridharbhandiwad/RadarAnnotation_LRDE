# Fix: Insufficient Classes for Training

## Your Error

```
✗ Training error: Insufficient classes for training. 
Found 1 unique class(es): ['incoming,level,linear,light_maneuver,low_speed']
```

## What This Means

Your training data has **only ONE unique label** for all data points. Machine learning models need **at least 2 different classes** to train (e.g., "incoming" vs "outgoing", or "fast" vs "slow").

All your data has been labeled as: `'incoming,level,linear,light_maneuver,low_speed'`

This is a **composite label** (multiple tags combined with commas) created by the auto-labeling engine.

---

## Quick Fix (3 Steps)

### Step 1: Analyze Your Data

Run this command to understand the problem:

```bash
python analyze_label_diversity.py "D:/Zoppler Projects/RadarAnnotation_LRDE/Database/labelled_data_1.csv"
```

**Note for Linux users**: If you copied the file to Linux, use the local path:
```bash
python analyze_label_diversity.py /path/to/labelled_data_1.csv
```

This will show you:
- Why all labels are the same
- How many tracks you have
- Which individual flags vary
- Specific recommendations

---

### Step 2: Create Per-Track Labels (Most Common Fix)

If you have multiple tracks, this usually works:

```bash
python create_track_labels.py "D:/Zoppler Projects/RadarAnnotation_LRDE/Database/labelled_data_1.csv"
```

This creates a new file: `labelled_data_1_track_labeled.csv`

**What it does:**
- Simplifies the composite label to a single main tag
- Groups all points in a track under one label
- Creates distinct labels for different tracks

**Example:**
- Track 1: All points → `'incoming'`
- Track 2: All points → `'outgoing'`
- Track 3: All points → `'level'`

Now you have 3 classes instead of 1!

---

### Step 3: Train with New File

In the GUI:
1. Go to **AI Tagging** tab
2. Click **"Select Labeled Data CSV"**
3. Choose the new file: `labelled_data_1_track_labeled.csv`
4. Click **"Train Model"**

Or via command line:
```bash
python -m src.ai_engine --model xgboost --data labelled_data_1_track_labeled.csv --outdir output/models
```

---

## Alternative Solutions

### If Per-Track Labels Don't Work

#### Option A: Split Composite Labels

Create separate binary classification tasks:

```bash
python split_composite_labels.py "D:/Zoppler Projects/RadarAnnotation_LRDE/Database/labelled_data_1.csv"
```

This shows which individual flags (incoming, level, linear, etc.) have variation in your data.

For binary classification (e.g., "incoming" vs "not incoming"):
```bash
python split_composite_labels.py input.csv output.csv incoming
```

#### Option B: Adjust Auto-Labeling Thresholds

Your data might have variation that the auto-labeling isn't detecting.

1. Edit `config/default_config.json`
2. Adjust values in the `autolabel_thresholds` section:
   ```json
   {
     "autolabel_thresholds": {
       "low_speed_threshold": 50.0,   // Try different values: 20, 30, 100
       "high_speed_threshold": 200.0,
       "curvature_threshold": 0.01,   // Try 0.005 for more sensitivity
       "range_rate_threshold": 1.0    // For incoming/outgoing detection
     }
   }
   ```
3. Re-run auto-labeling on your raw data
4. Check if you get more diverse labels

#### Option C: Collect More Diverse Data

If your data genuinely represents only one type of motion, you need:
- Different speeds (slow, medium, fast)
- Different directions (incoming, outgoing)
- Different flight patterns (level, ascending, descending)
- Different paths (straight, curved)

---

## Understanding the Problem

### Why Do All Points Have the Same Label?

The auto-labeling engine analyzes motion features and assigns tags:

- **Direction**: incoming or outgoing
- **Vertical**: level, ascending, or descending  
- **Path**: linear or curved
- **Maneuver**: light_maneuver or high_maneuver
- **Speed**: low_speed or high_speed

These combine into: `'incoming,level,linear,light_maneuver,low_speed'`

If **all your data** has:
- Same direction
- Same altitude change
- Same curvature
- Same acceleration
- Same speed range

Then **all points get the same label**!

### Your Data Characteristics

Based on your label `'incoming,level,linear,light_maneuver,low_speed'`:

- ✈️ All targets are **incoming** (approaching radar)
- ⬌ All are **level** (constant altitude)
- ➡️ All follow **linear** paths (no curves)
- 🎯 All have **light maneuvers** (gentle motion)
- 🐌 All are **low speed**

This could mean:
1. Your data comes from one type of aircraft doing the same thing
2. Your data is a short segment showing consistent motion
3. The thresholds need adjustment for your specific data

---

## Troubleshooting

### "File not found" Error

If running on Linux but file is on Windows drive:
1. Copy the CSV file to your Linux filesystem first:
   ```bash
   cp "/mnt/d/Zoppler Projects/RadarAnnotation_LRDE/Database/labelled_data_1.csv" /workspace/
   ```
2. Then run the scripts:
   ```bash
   python analyze_label_diversity.py /workspace/labelled_data_1.csv
   ```

### Still Only 1 Class After Per-Track Labels

This means all your tracks have the same characteristics. Options:
1. Try splitting composite labels (Option A above)
2. Adjust thresholds (Option B above)
3. Use different/additional data (Option C above)

### "No variation in individual flags"

Your data is truly uniform. You must either:
- Adjust thresholds to create more granular distinctions
- Collect data with more diversity

---

## Testing Your Fix

After creating new labeled data, verify it worked:

```bash
# Check how many unique labels you have
python -c "import pandas as pd; df = pd.read_csv('labelled_data_1_track_labeled.csv'); print(f'Unique labels: {df[\"Annotation\"].nunique()}'); print(df['Annotation'].value_counts())"
```

**Expected output:**
```
Unique labels: 3
incoming    120
outgoing     80
level        50
```

If you see 2+ unique labels, you're ready to train! 🎉

---

## Need More Help?

See the complete guide: **`LABEL_DIVERSITY_GUIDE.md`**

Or run the analysis script - it gives specific recommendations for your data!

```bash
python analyze_label_diversity.py <your_csv_file>
```
