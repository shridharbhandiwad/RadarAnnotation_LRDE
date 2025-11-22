# Quick Start: Colored Track Segments on PPI

## What's New? 🎨

Your PPI radar view now supports **colored segments** where each track can show different colors based on annotation changes. The radar view is also **70% larger** for better visibility!

## Quick Start (3 Steps)

### Step 1: Launch & Load
```bash
python -m src.gui
```
1. Click **"📉 Visualization"** in left sidebar
2. Click **"Load Data for Visualization"**
3. Select your labeled CSV file

### Step 2: Select Color Mode
In the toolbar, find **"Color By:"** dropdown and select:
- **"Track Segments (Colored by Annotation)"** ← NEW FEATURE!

### Step 3: View Results
Watch as each track displays with different colored segments representing different flight behaviors!

## Color Mode Options

| Mode | What You See |
|------|--------------|
| **Track ID** | Each track has one solid color |
| **Annotation** | All same behaviors grouped by color |
| **Track Segments** ⭐ | Individual tracks with colored segments per annotation |

## Example Visualization

Imagine Track 1 has this flight profile:
```
Time 0-10s:  LevelFlight (Blue segment)
Time 10-20s: Climbing (Orange segment)
Time 20-30s: HighSpeed (Red segment)
```

**Old Behavior**: Entire track shown in one color  
**New Behavior**: Track shows blue → orange → red segments! 🌈

## Display Modes

| Mode | Description | Best For |
|------|-------------|----------|
| **Radar View (Circular)** ⭐ | Traditional PPI with range rings | Air traffic control style |
| **Cartesian (X, Y)** | Standard X-Y coordinates | General analysis |
| **Polar (Range, Azimuth)** | Range vs Azimuth plot | Range-based analysis |

## Size Changes

The PPI radar view is now **70% of the screen** (was 40%):
- **Before**: Small radar view
- **After**: Large, prominent radar display
- **Benefit**: Better visibility of track details and annotations

## Color Theme 🎨

All colors are now consistent across the app:

### Single Annotations
- 🔵 **LevelFlight**: Blue
- 🟠 **Climbing**: Orange
- 🔴 **HighSpeed**: Red
- 🟢 **LowSpeed**: Green
- 🟡 **Turning**: Yellow/Gold
- 🟣 **HighManeuver**: Magenta
- 💙 **Outgoing**: Turquoise

### Composite Annotations
Colors automatically blend for combinations:
- **LevelFlight+HighSpeed**: Light Red
- **Climbing+HighSpeed**: Burnt Orange
- **Turning+LowSpeed**: Yellow-Green
- And many more...

## Tips & Tricks 💡

1. **Hover for Details**: Move mouse over any point to see full information
2. **Filter Tracks**: Use "Filter Track ID" to focus on specific tracks
3. **Compare Modes**: Switch between color modes to validate annotations
4. **Dark Background**: Colors are optimized for the dark radar screen
5. **Legend**: Check the legend to identify annotation colors

## Common Use Cases

### 1. Validate Auto-Labeling
**Goal**: Check if annotations are correct  
**Steps**: 
- Load auto-labeled data
- Select "Track Segments" mode
- Look for smooth color transitions
- Verify colors match expected behavior

### 2. Analyze Track Behavior
**Goal**: Understand how tracks change over time  
**Steps**:
- Load tracked data
- Select "Track Segments" mode
- Follow color changes along track
- Correlate with time series plots below

### 3. Present to Stakeholders
**Goal**: Show professional radar visualization  
**Steps**:
- Use "Radar View (Circular)" mode
- Select "Track Segments" for detail
- Leverage the larger 70% display
- Colors match professional theme

## Troubleshooting

### No Colors Showing?
- Ensure your CSV has an "Annotation" column
- Check that annotations are not all "invalid"
- Try "Annotation" mode first to verify colors work

### Colors Look Wrong?
- Colors are optimized for dark background
- Each annotation has a specific color
- Combinations blend parent colors
- Check the summary document for color mapping

### PPI Still Small?
- Make sure you're using the latest version
- PPI should take ~70% of vertical space
- Time series plots should be ~30% below

## File Locations

- **Summary**: `PPI_COLORED_SEGMENTS_SUMMARY.md` - Full technical details
- **Source**: `src/plotting.py` - Color definitions and plotting logic
- **GUI**: `src/gui.py` - Interface and controls

## Need Help?

Check these files for more information:
- `PPI_COLORED_SEGMENTS_SUMMARY.md` - Complete technical documentation
- `QUICK_START.md` - General application guide
- `README.md` - Project overview

---

**Enjoy your enhanced radar visualization!** 🎉
