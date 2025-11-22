# Label Diversity Guide

## Problem: "Insufficient classes for training" Error

If you see this error when training:

```
✗ Training error: Insufficient classes for training. Found 1 unique class(es): ['incoming,level,linear,light_maneuver,low_speed']
```

This means **all your data has the same label**, making machine learning impossible. Models need variety to learn patterns!

---

## Understanding the Problem

### Why does this happen?

1. **Composite Labels**: The auto-labeling engine creates labels by combining multiple tags:
   - Example: `'incoming,level,linear,light_maneuver,low_speed'`
   - If all your data has identical motion characteristics, all rows get the same label

2. **Uniform Data**: Your trajectory data might all represent the same type of motion:
   - All moving at the same speed
   - All flying in the same direction
   - All following similar paths

3. **Threshold Settings**: Auto-labeling thresholds might be set incorrectly for your data

---

## Solutions

### 🔍 Step 1: Analyze Your Data

**Run the analysis script to understand what's wrong:**

```bash
python analyze_label_diversity.py path/to/your/labelled_data.csv
```

This will show you:
- How many unique labels you have
- Distribution of individual flags (incoming, level, etc.)
- Feature statistics (speed, range, curvature, etc.)
- Specific recommendations for your data

**Example output:**
```
📊 ANNOTATION ANALYSIS
Unique annotations: 1

  'incoming,level,linear,light_maneuver,low_speed': 1000 rows (100.0%)

⚠️  PROBLEM IDENTIFIED: All rows have the same annotation!

💡 SOLUTIONS
1️⃣  USE PER-TRACK LABELS (Recommended)
2️⃣  ADJUST AUTO-LABELING THRESHOLDS
3️⃣  USE INDIVIDUAL FLAGS FOR CLASSIFICATION
4️⃣  COLLECT MORE DIVERSE DATA
```

---

### ✅ Step 2: Choose a Fix

#### Option A: Create Per-Track Labels (Recommended)

If you have multiple tracks but they all have the same point-level labels, aggregate to track level:

```bash
python create_track_labels.py path/to/your/labelled_data.csv
```

**What this does:**
- Simplifies composite labels to single tags (e.g., `'incoming,level,linear,...'` → `'incoming'`)
- Assigns one label per track (all points in a track get the same label)
- Creates a new CSV file ready for training

**Strategies available:**
```bash
# Extract most important tag (default)
python create_track_labels.py data.csv output.csv primary

# Use most common annotation in track
python create_track_labels.py data.csv output.csv majority

# Use first point's annotation
python create_track_labels.py data.csv output.csv first
```

**When to use this:**
- You have 2+ tracks with different characteristics
- You want to classify entire trajectories (not individual points)

---

#### Option B: Split Composite Labels

Convert multi-label classification into multiple binary classification tasks:

```bash
python split_composite_labels.py path/to/your/labelled_data.csv
```

**What this does:**
- Creates separate columns like `label_incoming`, `label_level`, `label_linear`, etc.
- Each column is True/False for whether that tag applies
- Shows which individual flags have variation in your data

**Example for binary classification:**
```bash
# Create a dataset for "incoming vs not incoming" classification
python split_composite_labels.py data.csv output.csv incoming
```

**When to use this:**
- Your composite labels are identical but individual flags might vary
- You want to solve multiple simpler problems instead of one complex one
- You want to train separate models for each characteristic

---

#### Option C: Adjust Auto-Labeling Thresholds

If your data has variation but thresholds are wrong, adjust them:

1. **Edit the configuration file:**
   ```bash
   nano config/default_config.json
   # or use any text editor
   ```

2. **Modify threshold values** in the `autolabel_thresholds` section:
   ```json
   {
     "autolabel_thresholds": {
       "level_flight_threshold": 5.0,        // Try 2.0 for tighter criteria
       "curvature_threshold": 0.01,          // Try 0.005 for more sensitivity
       "low_speed_threshold": 50.0,          // Try 20.0 or 100.0
       "high_speed_threshold": 200.0,        // Adjust based on your data
       "light_maneuver_threshold": 2.0,      // Lower for more sensitivity
       "high_maneuver_threshold": 5.0,       // Adjust as needed
       "range_rate_threshold": 1.0           // Adjust for incoming/outgoing
     }
   }
   ```

3. **Re-run auto-labeling** with new thresholds:
   - In GUI: Select your raw data and run auto-labeling again
   - Or use CLI:
     ```bash
     python -m src.autolabel_engine --input raw_data.csv --out labelled_data_new.csv
     ```

4. **Verify diversity:**
   ```bash
   python analyze_label_diversity.py labelled_data_new.csv
   ```

**When to use this:**
- You know your data has variation but all points are being labeled the same
- You understand your data's characteristics (speed ranges, maneuver types, etc.)
- You're willing to experiment with different threshold values

---

#### Option D: Collect More Diverse Data

If your data genuinely represents only one type of motion, you need more variety:

**What to collect:**
- ✈️ Different directions: incoming, outgoing, tangential
- ⬆️ Different vertical motion: level flight, ascending, descending
- 🏃 Different speeds: slow, medium, fast
- 🔄 Different paths: straight lines, curves, circles
- 💨 Different maneuvers: gentle, aggressive

**How much data:**
- Minimum: 2 distinct classes (e.g., incoming and outgoing)
- Better: 5+ classes with 20+ tracks each
- Best: 10+ classes with 50+ tracks each

---

## Quick Reference

| Your Situation | Solution | Command |
|---------------|----------|---------|
| Multiple tracks, same labels | Per-track labels | `python create_track_labels.py data.csv` |
| Composite labels identical | Split labels | `python split_composite_labels.py data.csv` |
| Data has variation | Adjust thresholds | Edit `config/default_config.json` |
| Data lacks variation | Collect more data | Get diverse trajectories |
| Not sure | Analyze first | `python analyze_label_diversity.py data.csv` |

---

## Workflow Example

Here's a complete workflow for fixing the "insufficient classes" error:

```bash
# 1. Analyze the problem
python analyze_label_diversity.py labelled_data_1.csv

# Output shows: "All 1000 rows have same annotation, but 5 tracks found"

# 2. Create per-track labels
python create_track_labels.py labelled_data_1.csv labelled_data_tracks.csv

# Output shows: "Created 3 unique labels from 5 tracks"
# Labels: incoming (2 tracks), outgoing (2 tracks), level (1 track)

# 3. Train model with new file
# In GUI: Select "labelled_data_tracks.csv" for training
# Or CLI:
python -m src.ai_engine --model xgboost --data labelled_data_tracks.csv --outdir output/models

# 4. Success! Model trains with 3 classes
```

---

## Understanding Auto-Labeling

### How Composite Labels Are Created

The auto-labeling engine analyzes each data point and assigns multiple tags:

1. **Direction** (mutually exclusive):
   - `incoming`: Moving toward radar (negative range rate)
   - `outgoing`: Moving away from radar (positive range rate)

2. **Vertical Motion** (mutually exclusive):
   - `level`: Altitude change < threshold
   - `ascending`: Moving up with low lateral speed
   - `descending`: Moving down with low lateral speed

3. **Path Shape** (mutually exclusive):
   - `linear`: Low curvature
   - `curved`: High curvature

4. **Maneuver Intensity**:
   - `light_maneuver`: Low acceleration
   - `high_maneuver`: High acceleration

5. **Speed**:
   - `low_speed`: Speed below threshold
   - `high_speed`: Speed above threshold

These tags are combined with commas: `'incoming,level,linear,light_maneuver,low_speed'`

### Why All Data Gets the Same Label

If your data has:
- Same direction (e.g., all incoming)
- Same altitude change (e.g., all level)
- Same path curvature (e.g., all linear)
- Same acceleration (e.g., all light maneuvers)
- Same speed range (e.g., all low speed)

Then all points will get the exact same composite label!

---

## Need Help?

If you're still stuck after trying these solutions:

1. Check your data has actual variation:
   ```bash
   # Quick check for data diversity
   python -c "import pandas as pd; df = pd.read_csv('data.csv'); print(df[['speed', 'range', 'vx', 'vy']].describe())"
   ```

2. Review the analysis output carefully - it tells you exactly what's wrong

3. Start simple - try per-track labels first, it's the easiest fix

4. Remember: You MUST have at least 2 different classes for ML to work!

---

## Additional Resources

- **Configuration Reference**: See `config/default_config.json` for all threshold settings
- **Auto-Labeling Details**: See `src/autolabel_engine.py` for the labeling logic
- **Training Guide**: See `QUICK_START.md` for general training workflow
- **Project Overview**: See `PROJECT_OVERVIEW.md` for system architecture
