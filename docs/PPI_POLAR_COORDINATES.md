# PPI Polar Coordinates Display

## Overview
The PPI (Plan Position Indicator) visualization now supports displaying radar data in both **Cartesian** and **Polar** coordinate systems.

## New Features

### 1. Coordinate Mode Selector
A new dropdown control in the Visualization panel allows you to switch between:
- **Cartesian (X, Y)**: Traditional X and Y position in kilometers
- **Polar (Range, Azimuth)**: Range in kilometers and Azimuth in degrees

### 2. Dynamic Display Switching
The PPI plot automatically updates when you switch coordinate modes:
- **Cartesian Mode**: Displays X vs Y position
  - X-axis: X Position (km)
  - Y-axis: Y Position (km)
  - Title: "PPI - Plan Position Indicator"

- **Polar Mode**: Displays Azimuth vs Range
  - X-axis: Azimuth (degrees)
  - Y-axis: Range (km)
  - Title: "PPI - Range vs Azimuth"

### 3. Enhanced Tooltips
When hovering over track points, tooltips now display **both** coordinate systems simultaneously:
- Track ID
- Time (seconds)
- **Cartesian coordinates**: (X, Y) in km
- **Polar coordinates**: Range in km, Azimuth in degrees
- Annotation type

This allows you to see both representations at once, regardless of the display mode.

## How to Use

### Step 1: Start the Application
```bash
# Linux/Mac
./run.sh

# Windows
run.bat
```

### Step 2: Load Data
1. Navigate to **"Visualization"** in the left sidebar
2. Click **"Load Data for Visualization"** (green button)
3. Select your CSV file (e.g., `data/test_simulation_labeled.csv`)

### Step 3: Switch Coordinate Modes
1. Use the **"Coordinates:"** dropdown near the top
2. Select:
   - **"Cartesian (X, Y)"** for traditional X-Y plot
   - **"Polar (Range, Azimuth)"** for range-azimuth display

### Step 4: Explore
- Hover over any track point to see both coordinate representations
- Use other controls (Color By, Track Filter) as usual
- All features work in both coordinate modes

## Coordinate Conversion

The conversion between coordinate systems uses these formulas:

### Cartesian to Polar:
- **Range** = √(x² + y²)
- **Azimuth** = arctan2(y, x) in degrees

### Polar to Cartesian:
- **X** = Range × cos(Azimuth)
- **Y** = Range × sin(Azimuth)

## Technical Details

### Modified Files:
1. **`src/plotting.py`**:
   - Added `coordinate_mode` attribute to `PPIPlotWidget`
   - Added `set_coordinate_mode()` method
   - Updated `plot_tracks()` to compute and plot polar coordinates
   - Enhanced `on_mouse_moved()` tooltip to show both coordinate systems
   - Dynamic axis labels based on coordinate mode

2. **`src/gui.py`**:
   - Added coordinate mode selector dropdown in `VisualizationPanel`
   - Connected selector to visualization update logic
   - Integrated with existing color-by and filter controls

3. **`src/utils.py`** (existing functions used):
   - `cartesian_to_polar()`: Converts X,Y to Range, Azimuth
   - `polar_to_cartesian()`: Converts Range, Azimuth to X,Y

### Hover Detection:
- **Cartesian mode**: Tooltip appears within 0.5 km of a point
- **Polar mode**: Tooltip appears within 5.0 units (adjusted for azimuth scaling)

## Use Cases

### When to Use Cartesian Mode:
- Analyzing spatial patterns in X-Y plane
- Understanding geographic/positional relationships
- Correlating with map overlays
- Analyzing rectangular coverage areas

### When to Use Polar Mode:
- Analyzing radar coverage patterns
- Understanding range and bearing relationships
- Correlating with radar parameters (beam width, range gates)
- Analyzing circular coverage patterns
- Range-dependent phenomena analysis

## Examples

### Cartesian Display:
```
Track appears at (10.5, 23.2) km
Good for: "Where is the target in space?"
```

### Polar Display:
```
Same track appears at Range=25.5 km, Azimuth=65.7°
Good for: "How far is the target and at what angle?"
```

### Tooltip Shows Both:
```
Track ID: 5
Time: 12.45 s
Cartesian: (10.50, 23.20) km
Polar: Range=25.48 km, Az=65.7°
Annotation: LevelFlight+HighSpeed
```

## Compatibility

- Works with all existing features:
  - ✓ Color by Track ID
  - ✓ Color by Annotation
  - ✓ Track filtering
  - ✓ Time series plots (always show time domain)
  - ✓ Hover tooltips
  - ✓ Track highlighting

- Data Requirements:
  - CSV file with `x`, `y`, `trackid` columns (minimum)
  - Optional: `Annotation` column for color coding
  - Optional: `time`, `z` columns for additional features

## Performance

- Coordinate conversion is computed once when loading data
- Switching modes is instant (no recomputation needed)
- Both coordinate systems stored in memory for fast access
- No impact on visualization performance

## Future Enhancements

Potential improvements:
1. **True Polar Plot**: Use polar plot widget instead of cartesian axes
2. **Grid Lines**: Add range rings and azimuth rays in polar mode
3. **Azimuth Reference**: Allow setting North/reference direction
4. **Range Units**: Support nautical miles, statute miles
5. **Elevation Angle**: Add 3D polar coordinates (range, azimuth, elevation)

## Troubleshooting

**Tooltip not showing correct coordinates?**
- Ensure data has valid x, y values
- Check that coordinates are in meters (converted to km internally)

**Polar plot looks stretched?**
- This is normal - azimuth spans -180° to +180°, while range is always positive
- Consider this when interpreting angular relationships

**Switching modes resets zoom?**
- This is expected behavior - axes scales change between modes
- Use mouse wheel to zoom after switching modes

## Version History

**v2.0** (Current):
- Added polar coordinate display mode
- Enhanced tooltips to show both coordinate systems
- Added coordinate mode selector in GUI

**v1.0**:
- Initial implementation with Cartesian coordinates only

---

**Related Documentation:**
- `PPI_ENHANCEMENTS.md` - Color coding and tooltips
- `QUICK_START_PPI_FEATURES.md` - Quick start guide
- `README.md` - Main project documentation
