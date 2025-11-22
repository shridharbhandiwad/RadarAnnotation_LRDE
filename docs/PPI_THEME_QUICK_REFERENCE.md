# PPI Theme Support - Quick Reference

## What Changed? 🎨

### ✅ Default Theme: WHITE
- Application now starts with **white theme** by default
- PPI plots display on **white background** with vibrant colors
- More suitable for bright office environments

### ✅ PPI Plots Are Now Theme-Aware
- **PPI (Plan Position Indicator)** plots adapt to selected theme
- **Time Series** plots adapt to selected theme
- **Range rings, labels, tooltips** all adjust colors automatically

## How to Use

### On Startup:
1. Launch application: `python -m src.gui` or `./run.sh`
2. Application starts with **white theme** (light interface)
3. Navigate to **Visualization** tab
4. Load data - PPI shows white background with vibrant colored tracks

### Switching Themes:
1. Go to **⚙️ Settings** tab
2. Click **⚫ Black Theme** or **⚪ White Theme**
3. Return to **Visualization** tab
4. PPI plot updates instantly with new theme

### Theme Persistence:
- Selected theme is saved automatically
- Next time you open the app, it remembers your choice
- Saved in: `config/default_config.json`

## Visual Differences

### White Theme (Default):
```
Background: White (#ffffff)
Tracks: Vibrant colors (red, green, blue, purple, orange, turquoise, yellow)
Range Rings: Light gray
Labels: Dark gray / black
Tooltips: Black text on white background
Best For: Bright offices, presentations, daytime use
```

### Black Theme:
```
Background: Dark slate (#1c2329)
Tracks: Muted slate tones (light grays, blue-grays)
Range Rings: Medium slate
Labels: Light gray / white
Tooltips: White text on dark background
Best For: Low-light environments, tactical displays, extended viewing
```

## Benefits

✅ **Consistency** - Entire application (GUI + PPI) uses the same theme  
✅ **Flexibility** - Switch themes anytime without reloading data  
✅ **Professional** - Proper contrast in both light and dark modes  
✅ **User-Friendly** - White theme is more familiar to most users  
✅ **Accessibility** - Choose theme based on lighting conditions  

## Technical Details

### Files Modified:
- `config/default_config.json` - Changed default theme to "white"
- `src/config.py` - Updated DEFAULT_CONFIG
- `src/gui.py` - Added theme propagation to visualization panels
- `src/plotting.py` - Added theme support to PPI and Time Series widgets

### New Methods:
- `PPIPlotWidget.set_theme(theme)` - Update PPI plot theme
- `TimeSeriesPlotWidget.set_theme(theme)` - Update time series theme
- `VisualizationPanel.set_theme(theme)` - Update both plot widgets

### Theme Values:
- `'white'` - Light theme (default)
- `'black'` - Dark theme

## Troubleshooting

**Q: Theme not changing?**  
A: Make sure to navigate back to Visualization tab after changing theme in Settings.

**Q: Want to force a specific theme?**  
A: Edit `config/default_config.json` and set `"theme": "white"` or `"theme": "black"`

**Q: PPI still showing dark background?**  
A: Check Settings tab to verify which theme is selected. Click the theme button to apply.

## Quick Test

```bash
# 1. Start the application
python -m src.gui

# 2. Check default theme is white
# - GUI should be light/white
# - Go to Visualization, load data
# - PPI background should be white

# 3. Switch to black theme
# - Go to Settings
# - Click "⚫ Black Theme"
# - Go to Visualization
# - PPI background should be dark

# 4. Restart and verify persistence
# - Close application
# - Reopen
# - Should start with last selected theme
```

---

**Status:** ✅ Complete  
**Version:** 1.0  
**Date:** 2025-11-21
