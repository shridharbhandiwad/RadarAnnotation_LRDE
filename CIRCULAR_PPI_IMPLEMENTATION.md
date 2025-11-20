# Circular PPI Implementation - Radar View with Range Rings and Azimuth Segments

## Overview

The PPI (Plan Position Indicator) now features a **true circular radar display** with range rings (concentric circles) and azimuth segments (radial angle lines), providing an authentic radar scope visualization.

## Features

### 1. Circular Radar Display
- **Round PPI scope** resembling traditional radar displays
- **Dark radar screen background** (#0a0a0a) for authentic look
- **Center-origin display** with radar at (0, 0)

### 2. Range Rings (Concentric Circles)
- **5 range circles** automatically scaled to data
- **Green dashed lines** (RGB: 0, 150, 0) for clear visibility
- **Range labels** displayed at top of each ring
- **Automatic scaling**: Rounds to nearest 10 km for clean divisions
- **Example**: For 47 km max range → displays 0, 10, 20, 30, 40, 50 km rings

### 3. Azimuth Segments (Radial Lines)
- **12 radial lines** at 30° intervals (0°, 30°, 60°, 90°, etc.)
- **Cardinal directions labeled**: N (North), E (East), S (South), W (West)
- **Angle labels** at intermediate positions
- **Standard orientation**: 
  - 0° / E = East (right)
  - 90° / N = North (top)
  - 180° / W = West (left)
  - 270° / S = South (bottom)

### 4. Track Plotting
- Tracks plotted in **Cartesian coordinates** on circular background
- **Color-coded by Track ID** or **Annotation type**
- **Interactive tooltips** showing position in both coordinate systems
- **Real-time hover information**

## Visual Layout

```
         N (90°)
          |
    150°  |  30°
       \  |  /
  W ---- [O] ---- E (0°)
       /  |  \
    210°  |  300°
          |
         S (270°)

[O] = Radar origin (center)
Concentric circles = Range rings
Radial lines = Azimuth segments
```

## Display Modes

The visualization now supports **3 display modes**:

### 1. Radar View (Circular) - NEW! ⭐
- **Circular PPI** with range rings and azimuth segments
- **Authentic radar display** appearance
- **Default mode** for radar applications
- Best for: Radar operators, air traffic control, tactical displays

### 2. Cartesian (X, Y)
- Traditional X-Y plot
- Standard grid lines
- Best for: Spatial analysis, map overlays

### 3. Polar (Range, Azimuth)
- Range vs Azimuth plot (not circular)
- Linear axes
- Best for: Data analysis, range-bearing correlations

## Usage

### Starting the Application

```bash
# Linux/Mac
./run.sh

# Windows
run.bat

# Or directly
python3 -m src.gui
```

### Loading Data and Viewing Circular PPI

1. **Launch the GUI** and navigate to the **"📉 Visualization"** panel

2. **Load Data**: Click the green **"Load Data for Visualization"** button
   - Select your CSV file (e.g., `data/test_simulation_labeled.csv`)
   - Data must have `x`, `y`, `trackid` columns

3. **Select Display Mode**: Use the **"Display Mode"** dropdown
   - Choose **"Radar View (Circular)"** for the new circular PPI
   - Default is already set to Radar View

4. **Explore**: 
   - Zoom with mouse wheel
   - Pan by dragging
   - Hover over tracks to see tooltips
   - Use **"Color By"** to switch between Track ID and Annotation coloring

### Switching Between Modes

You can dynamically switch between display modes:

```
Display Mode Dropdown:
├── Radar View (Circular)  ← New circular PPI with range rings
├── Cartesian (X, Y)       ← Traditional X-Y plot
└── Polar (Range, Azimuth) ← Linear range-azimuth plot
```

## Technical Implementation

### Modified Files

#### 1. `src/plotting.py`
**New Methods:**
- `draw_circular_ppi_background(max_range_km)`: Draws range rings and azimuth lines
  - Creates 5 concentric range circles
  - Draws 12 radial azimuth lines (every 30°)
  - Adds range and azimuth labels
  - Sets appropriate view bounds

**Updated Methods:**
- `__init__()`: Added circular display mode and background storage
- `set_coordinate_mode()`: Added 'polar_circular' mode support
- `clear()`: Now clears range rings and azimuth elements
- `plot_tracks()`: Automatically draws circular background when in polar_circular mode

**New Attributes:**
- `self.range_rings`: List of circle plot items
- `self.azimuth_lines`: List of radial line plot items
- `self.range_labels`: List of range text labels
- `self.azimuth_labels`: List of azimuth text labels
- `self.coordinate_mode`: Now supports 'polar_circular' mode

#### 2. `src/gui.py`
**Updated:**
- Display mode dropdown now includes "Radar View (Circular)" as first option
- `update_visualization()`: Detects and applies 'polar_circular' mode
- Default display mode is Radar View

### Algorithm Details

#### Range Ring Calculation
```python
num_rings = 5
max_range_km = ceil(max(sqrt(x^2 + y^2)) / 10) * 10  # Round to nearest 10
ring_spacing = max_range_km / num_rings

for i in 1 to 5:
    radius = i * ring_spacing
    draw_circle(radius)
    label_at(0, radius)
```

#### Azimuth Line Calculation
```python
angles = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]
labels = ['E', '30°', '60°', 'N', '120°', '150°', 'W', '210°', '240°', 'S', '300°', '330°']

for angle, label in zip(angles, labels):
    x_end = max_range * cos(angle)
    y_end = max_range * sin(angle)
    draw_line(0, 0, x_end, y_end)
    label_at(x_end * 1.05, y_end * 1.05)
```

### Color Scheme

| Element | Color | RGB | Purpose |
|---------|-------|-----|---------|
| Background | Very Dark Gray | (10, 10, 10) | Radar screen |
| Range Rings | Green (Dashed) | (0, 150, 0) | Distance markers |
| Azimuth Lines | Green (Solid) | (0, 150, 0) | Angle markers |
| Outer Boundary | Bright Green | (0, 255, 0) | Display edge |
| Range Labels | Green | (0, 200, 0) | Distance text |
| Azimuth Labels | Light Green | (100, 255, 100) | Angle text |
| Track Points | Various | Per track/annotation | Data visualization |

## Data Requirements

### Minimum Requirements
- CSV file with columns: `x`, `y`, `trackid`
- Position values in meters (converted to km automatically)

### Recommended Columns
- `x`, `y`: Position in meters
- `trackid`: Track identifier
- `time`: Timestamp for time series
- `Annotation`: For color-by-annotation mode
- `z`: Altitude (for time series display)

### Example Data Format
```csv
trackid,time,x,y,z,vx,vy,vz,Annotation
1,0.0,5000,10000,3000,150,200,5,LevelFlight+HighSpeed
1,1.0,5150,10200,3005,150,200,5,LevelFlight+HighSpeed
2,0.0,-8000,15000,4000,180,-50,10,Climbing+HighSpeed
```

## Performance

- **Fast rendering**: Range rings drawn once per data load
- **Instant mode switching**: Background redrawn only when needed
- **Efficient updates**: Track points use scatter plot optimization
- **Scalable**: Handles 1000+ track points smoothly

## Advantages of Circular PPI

### Vs. Traditional Cartesian Plot:
✅ **More intuitive** for radar operators  
✅ **Shows coverage** (circular scan pattern)  
✅ **Authentic** radar display appearance  
✅ **Clear range indication** with concentric circles  
✅ **Natural azimuth representation**

### Vs. Linear Polar Plot:
✅ **True-to-life** circular representation  
✅ **Better angular perception** (not distorted)  
✅ **Professional radar display**  
✅ **Easier to correlate** with real radar systems

## Customization

### Changing Number of Range Rings
Edit `src/plotting.py`, line ~175:
```python
num_rings = 5  # Change this value (e.g., 4, 6, 8)
```

### Changing Azimuth Line Intervals
Edit `src/plotting.py`, line ~192:
```python
# Current: Every 30°
angles_deg = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]

# For every 45°:
angles_deg = [0, 45, 90, 135, 180, 225, 270, 315]

# For every 15°:
angles_deg = list(range(0, 360, 15))
```

### Changing Colors
Edit color values in `draw_circular_ppi_background()`:
```python
# Range rings
pen=pg.mkPen(color=(0, 150, 0), ...)  # RGB color tuple

# Background
self.plot_widget.setBackground('#0a0a0a')  # Hex color
```

## Tooltip Information

Hover tooltips display **both coordinate systems** simultaneously:

```
Track ID: 5
Time: 12.45 s
Cartesian: (10.50, 23.20) km
Polar: Range=25.48 km, Az=65.7°
Annotation: LevelFlight+HighSpeed
```

## Compatibility

✅ Works with all existing features:
- Color by Track ID
- Color by Annotation
- Track filtering
- Time series plots
- Hover tooltips
- Track highlighting

✅ Backward compatible:
- Old visualization modes still work
- Existing data files compatible
- No changes to data format needed

## Use Cases

### Air Traffic Control
- Monitor aircraft positions on radar-style display
- Track multiple targets with range/bearing
- Identify approach patterns

### Military/Defense
- Tactical situation display
- Target tracking and identification
- Range and bearing to threats

### Marine Radar
- Ship navigation and collision avoidance
- Port traffic monitoring
- Weather pattern visualization

### Research & Training
- Radar system simulation
- Operator training
- Data analysis and replay

## Troubleshooting

### Problem: Circles not showing
**Solution**: Ensure data is loaded. Range rings draw automatically when data is plotted.

### Problem: Display looks stretched
**Solution**: Ensure aspect ratio is locked (default behavior). Check window size.

### Problem: Labels overlapping
**Solution**: Zoom in/out to adjust label spacing. Labels scale with view.

### Problem: Tracks outside visible area
**Solution**: Range is auto-scaled to data. Use mouse wheel to zoom out if needed.

## Future Enhancements

Potential improvements:
1. **Customizable range rings**: User-selectable number and spacing
2. **Bearing cursor**: Click-and-drag bearing line tool
3. **Range/bearing measurement**: Click two points to measure
4. **Sector highlighting**: Highlight specific azimuth sectors
5. **Range gates**: Configurable range bands with colors
6. **Sweep animation**: Animated radar sweep line
7. **Target trails**: Show historical track positions
8. **Clutter filtering**: Hide low-priority tracks
9. **Alert zones**: Configurable alert boundaries

## Version History

### v3.0 (Current) - Circular PPI
- ✅ Circular PPI display with range rings
- ✅ Azimuth segments every 30°
- ✅ Radar-style dark background
- ✅ Cardinal direction labels
- ✅ Auto-scaling range rings
- ✅ Three display modes

### v2.0 - Polar Coordinates
- Added polar coordinate display
- Tooltips with both coordinate systems

### v1.0 - Initial Release
- Basic Cartesian PPI display

---

## Quick Reference

### Keyboard Shortcuts (Standard PyQtGraph)
- **Mouse Wheel**: Zoom in/out
- **Left Click + Drag**: Pan view
- **Right Click**: Context menu (export, etc.)
- **Middle Click + Drag**: Zoom to rectangle

### Display Modes
1. **Radar View (Circular)** - Circular PPI ⭐ NEW
2. **Cartesian (X, Y)** - Traditional X-Y plot
3. **Polar (Range, Azimuth)** - Linear polar plot

### Color Modes
1. **Track ID** - Each track gets unique color
2. **Annotation** - Color-coded by behavior/annotation

---

**For more information, see:**
- `README.md` - Main project documentation
- `PPI_POLAR_COORDINATES.md` - Coordinate conversion details
- `PPI_ENHANCEMENTS.md` - Color coding and tooltips
- `QUICK_START.md` - Quick start guide

**Questions or issues?** Check the documentation or examine the source code in `src/plotting.py`.
