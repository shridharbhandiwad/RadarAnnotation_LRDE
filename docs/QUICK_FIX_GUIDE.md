# Quick Fix Guide - Colored Segments & PPI Layout

## ✅ What Was Fixed

### 1. Colored Track Segments Now Work!
The "Track Segments (Colored by Annotation)" feature now properly displays different colored segments within each track based on annotations.

**Problem**: Colors weren't showing because annotation format mismatch  
**Fixed**: Now handles both formats (`"level,high_speed"` and `"LevelFlight+HighSpeed"`)

### 2. PPI Plot is Now Mandatory (Always Visible)
**Before**: PPI and time series in a split view  
**After**: PPI plot always visible, time series optional with toggle button

## 🚀 Quick Start

### Step 1: Launch GUI
```bash
python3 -m src.gui
# OR
./run.sh    # Linux/Mac
run.bat     # Windows
```

### Step 2: Go to Visualization Panel
Click **📉 Visualization** in the left sidebar

### Step 3: Load Data
1. Click **"Load Data for Visualization"**
2. Select: `data/test_simulation_labeled.csv`

### Step 4: Enable Colored Segments
In **"Color By:"** dropdown, select:
- **"Track Segments (Colored by Annotation)"** ✨

### Step 5 (Optional): Show Time Series
Click **"Show Time Series Charts"** button to see altitude, speed, and curvature plots

## 🎨 Color Reference

| Annotation | Color | What It Means |
|------------|-------|---------------|
| level / level_flight | 🔵 Blue | Level flight |
| high_speed | 🔴 Red | High speed |
| low_speed | 🟢 Green | Low speed |
| ascending | 🟠 Orange | Climbing |
| descending | 🌸 Pink | Descending |
| outgoing | 🔷 Turquoise | Moving away |
| incoming | 🟤 Dark Orange | Approaching |
| curved | 🟡 Yellow/Gold | Turning |
| linear | 🟩 Mint Green | Straight path |
| light_maneuver | 🟣 Purple | Light maneuver |
| high_maneuver | 🟪 Magenta | High maneuver |

**Composite annotations**: Colors automatically blend!

## 📊 Display Options

### Color By Modes:
1. **Track ID**: Each track = different color
2. **Annotation**: All same annotation = same color  
3. **Track Segments (Colored by Annotation)**: ✨ Different colored segments per track

### Display Modes:
1. **Radar View (Circular)**: Traditional PPI (Default) ⭐
2. **Cartesian (X, Y)**: Standard X-Y coordinates
3. **Polar (Range, Azimuth)**: Polar plot

## 🎯 What You'll See

### PPI View Layout:
```
┌────────────────────────────────────────────┐
│ [Load] [Mode] [Color] [Filter] [Toggle TS]│ ← Controls
├────────────────────────────────────────────┤
│          ╭─────────────────╮               │
│         ╱   🎯 Radar View   ╲              │
│        │                     │             │
│        │    Track with       │             │
│        │    🔵 Blue segment  │             │ ← PPI (Always Visible)
│        │    🔴 Red segment   │             │
│        │    🟢 Green segment │             │
│         ╲                   ╱              │
│          ╰─────────────────╯               │
└────────────────────────────────────────────┘
        [Show Time Series Charts] ← Click to show
```

### With Time Series (Optional):
```
┌────────────────────────────────────────────┐
│              PPI View (Above)               │
├────────────────────────────────────────────┤
│  Altitude vs Time   ┌────┐                 │
│  Speed vs Time      │📈  │                 │ ← Time Series
│  Curvature vs Time  └────┘                 │  (Optional)
└────────────────────────────────────────────┘
        [Hide Time Series Charts] ← Click to hide
```

## ✨ Key Features

- ✅ **Colored segments work**: Segments colored by annotation
- ✅ **PPI always visible**: No need to scroll to see radar view
- ✅ **Optional time series**: Toggle on/off as needed
- ✅ **Better performance**: Time series only updates when visible
- ✅ **Both formats supported**: Old and new annotation formats work

## 🧪 Test It

1. Load `data/test_simulation_labeled.csv`
2. Select "Track Segments (Colored by Annotation)"
3. Look for different colored segments on tracks
4. Click "Show Time Series Charts" to see additional plots
5. Filter by individual tracks using "Filter Track ID" dropdown

## 📝 Files Changed

- `src/plotting.py` - Color mapping fixed
- `src/gui.py` - Layout reorganized

## 📚 Full Documentation

See `VISUALIZATION_FIX_SUMMARY.md` for complete technical details.

---
**Quick Access**: This guide is designed for rapid reference. For detailed information, see the full documentation.
