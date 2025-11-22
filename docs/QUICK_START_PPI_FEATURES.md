# Quick Start: Enhanced PPI Visualization

## What's New

The PPI (Plan Position Indicator) visualization now includes:

✨ **Hover Tooltips** - See track details when you hover over points  
🎨 **Smart Color Coding** - Tracks colored by annotation type  
💅 **Modern GUI Design** - Beautiful, professional interface  

## Getting Started in 3 Steps

### Step 1: Start the Application

```bash
# Linux/Mac
./run.sh

# Windows
run.bat
```

### Step 2: Load Your Data

1. Click on **"Visualization"** in the left sidebar
2. Click the green **"Load Data for Visualization"** button
3. Select your CSV file (try `data/test_simulation_labeled.csv`)

### Step 3: Explore!

**Try Hover Tooltips:**
- Move your mouse over any track point
- A tooltip will appear showing:
  - Track ID
  - Time (seconds)
  - Position (X, Y in km)
  - Annotation type

**Try Color Modes:**
- Use the **"Color By"** dropdown:
  - **Track ID** = Different color for each track
  - **Annotation** = Color based on flight behavior

## Color Guide

### Flight Patterns:

| What You See | Color | Meaning |
|--------------|-------|---------|
| 🔵 Sky Blue | LevelFlight | Aircraft flying level |
| 🔴 Red | HighSpeed | High-speed movement |
| 🟢 Green | LowSpeed | Slow movement |
| 🟡 Yellow | Turning | Aircraft turning |
| 🟠 Orange | Climbing | Gaining altitude |
| 🩷 Pink | Descending | Losing altitude |
| 🟣 Purple | HighManeuver+Turning | Aggressive turning |

### Common Combinations:

- **Light Red**: Level flight at high speed (cruise)
- **Gold**: Fast turn (intercept maneuver)
- **Light Green**: Level flight at low speed (approach)
- **Deep Orange**: Climbing at high speed (takeoff)

## Tips & Tricks

### Best Practices:
1. **Use Annotation coloring** for labeled data to see flight patterns
2. **Hover near the start/end** of tracks to see time progression
3. **Check time series plots below** the PPI for detailed track behavior

### Troubleshooting:

**Tooltip not showing?**
- Make sure you loaded data first
- Hover closer to a track point (within 0.5 km)

**Colors all the same?**
- Switch to "Color By: Annotation" in the dropdown
- Ensure your CSV has an 'Annotation' column

**GUI looks plain?**
- The stylesheet should auto-apply
- Try restarting the application

## Sample Data

Try these files included in the project:
- `data/test_simulation_labeled.csv` - Pre-labeled test data
- `data/test_simulation.csv` - Raw simulation data

## GUI Overview

### New Look:

**Left Panel** (Dark):
- Clean navigation
- Hover effects
- Blue selection highlight

**Buttons**:
- Blue for standard actions
- Green for primary actions (like "Load Data")
- Rounded corners
- Hover effects

**Input Fields**:
- Clean borders
- Smooth focus effects
- Consistent styling

**Tables & Lists**:
- Dark headers
- Easy-to-read rows
- Selection highlighting

## Advanced Features

### Multiple Tracks:
- Each track automatically gets a unique color in "Track ID" mode
- In "Annotation" mode, all tracks with the same behavior share a color

### Time Series Sync:
- The plots below the PPI show altitude, speed, and curvature
- All plots share the same time axis for easy comparison

### Data Format:
Your CSV should include these columns:
- `trackid` - Track identifier
- `x`, `y` - Position (meters)
- `time` - Timestamp (seconds)
- `Annotation` - Flight behavior (optional, for color coding)

## Next Steps

1. **Load your own data** - Any CSV with track information works
2. **Run auto-labeling** - Use the AutoLabeling panel to annotate raw data
3. **Train models** - Use the AI Tagging panel to train classifiers
4. **Generate reports** - Create visual reports of your analysis

## Keyboard Shortcuts (Coming Soon)

Future enhancements may include:
- Space: Play/pause animation
- Arrow keys: Navigate time
- +/- : Zoom in/out
- R: Reset view

## Feedback

For detailed technical information, see `PPI_ENHANCEMENTS.md`

---

**Enjoy the enhanced visualization! 🚀**
