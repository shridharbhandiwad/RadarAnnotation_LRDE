# PPI Visualization Enhancements

## Overview
This document describes the enhancements made to the PPI (Plan Position Indicator) visualization and GUI in the Radar Data Annotation Application.

## Features Implemented

### 1. Hover Data Tips (Tooltips)
**Location**: `src/plotting.py` - `PPIPlotWidget` class

When hovering over track paths in the PPI display, a tooltip now appears showing:
- **Track ID**: Unique identifier for the track
- **Time**: Timestamp of the data point (in seconds)
- **Position**: X, Y coordinates in kilometers
- **Annotation**: Current annotation(s) for that point

**Implementation Details**:
- The tooltip uses `pyqtgraph.TextItem` with a semi-transparent black background
- Mouse movement is tracked via `sigMouseMoved` signal
- Nearest point detection uses Euclidean distance (threshold: 0.5 km)
- Tooltip appears when hovering within 0.5 km of any track point

**Code Location**:
```python
def on_mouse_moved(self, pos):
    """Handle mouse movement for tooltip display"""
    # In src/plotting.py, lines ~207-252
```

### 2. Annotation-Based Color Coding
**Location**: `src/plotting.py` - `get_annotation_color()` function

Tracks are now color-coded based on their annotation combinations:

#### Single Annotation Colors:
| Annotation | Color | RGB |
|------------|-------|-----|
| LevelFlight | Sky Blue | (0, 150, 255) |
| Climbing | Orange | (255, 128, 0) |
| Descending | Pink | (255, 0, 128) |
| HighSpeed | Red | (255, 0, 0) |
| LowSpeed | Green | (0, 255, 0) |
| Turning | Yellow | (255, 255, 0) |
| Straight | Light Green | (100, 200, 100) |
| LightManeuver | Light Blue | (150, 150, 255) |
| HighManeuver | Magenta | (255, 0, 255) |
| Incoming | Orange | (255, 165, 0) |
| Outgoing | Cyan | (0, 255, 255) |
| FixedRange | Gray | (128, 128, 128) |

#### Composite Annotation Colors:
| Annotation Combination | Color | RGB |
|------------------------|-------|-----|
| LevelFlight+HighSpeed | Light Red | (255, 100, 100) |
| LevelFlight+LowSpeed | Light Green | (100, 255, 100) |
| Climbing+HighSpeed | Deep Orange | (255, 150, 0) |
| Descending+HighSpeed | Hot Pink | (255, 50, 150) |
| Turning+HighSpeed | Gold | (255, 200, 0) |
| Turning+LowSpeed | Yellow-Green | (200, 255, 100) |
| LevelFlight+Straight | Bright Sky Blue | (100, 180, 255) |
| HighManeuver+Turning | Purple | (200, 0, 255) |

**Smart Color Blending**:
- For unknown combinations, the function blends colors from individual annotations
- Falls back to gray (128, 128, 128) for completely unknown annotations

### 3. Color-By Selector
**Location**: `src/gui.py` - `VisualizationPanel` class

Added a dropdown selector in the Visualization panel:
- **Track ID**: Colors tracks by unique track identifier (traditional view)
- **Annotation**: Colors points by annotation type/combination (new feature)

The selector allows users to switch between coloring modes dynamically without reloading data.

### 4. Modern GUI Stylesheet
**Location**: `src/gui.py` - `MainWindow.apply_stylesheet()` method

Implemented a comprehensive stylesheet with modern design principles:

#### Design System:
- **Primary Color**: Blue (#3498db)
- **Success Color**: Green (#27ae60)
- **Dark Background**: Navy (#2c3e50)
- **Light Background**: White/Off-white (#f5f5f5)
- **Border Color**: Light gray (#bdc3c7)

#### Styled Components:
1. **Navigation Panel** (Engine Selector):
   - Dark navy background
   - Hover effects with lighter shade
   - Selected item highlighted in blue

2. **Buttons**:
   - Rounded corners (6px)
   - Blue primary buttons
   - Green accent for primary actions
   - Hover and pressed states
   - Disabled state styling

3. **Input Widgets**:
   - Clean borders with hover effects
   - Consistent padding and sizing
   - Custom combo box styling

4. **Tables**:
   - Dark headers
   - Hover/selection highlighting
   - Clean grid lines

5. **Progress Bars**:
   - Blue progress chunks
   - Clean rounded design

6. **Scroll Bars**:
   - Rounded custom scrollbars
   - Hover effects

7. **Group Boxes**:
   - Rounded borders
   - Clear section separation
   - White backgrounds

## Usage

### Starting the GUI:
```bash
# Linux/Mac
./run.sh

# Windows
run.bat
```

### Using the PPI Visualization:

1. Navigate to the **Visualization** panel in the left sidebar

2. Click **"Load Data for Visualization"** (green button)

3. Select a CSV file with track data (e.g., `data/test_simulation_labeled.csv`)

4. Use the **"Color By"** dropdown to switch between:
   - **Track ID**: Traditional track-based coloring
   - **Annotation**: Annotation-based color coding

5. **Hover over track points** to see data tips showing:
   - Track ID
   - Time
   - Position
   - Current annotation

### Tips:
- The tooltip appears when you're within 0.5 km of a track point
- Annotation coloring works best with labeled data
- Use the time series plots below the PPI to see track behavior over time

## Technical Details

### Dependencies:
- PyQt6: GUI framework
- pyqtgraph: High-performance plotting
- pandas: Data handling
- numpy: Numerical computations

### Performance Optimizations:
- Efficient nearest-point search using numpy vectorization
- Tooltip updates only when mouse moves
- Plot items reused when possible

### File Structure:
```
src/
├── gui.py           # Main GUI application with stylesheet
├── plotting.py      # PPI and time series visualization
└── config.py        # Configuration management
```

## Future Enhancements

Potential improvements for future versions:

1. **Interactive Annotation Editing**: Click on points to change annotations
2. **Animation**: Replay track data over time
3. **3D Visualization**: View altitude in 3D space
4. **Custom Color Schemes**: User-defined color mappings
5. **Export Visualizations**: Save plots as images
6. **Track Filtering**: Show/hide specific tracks or annotations
7. **QML Integration**: More advanced UI components using Qt Quick/QML

## Testing

To verify the implementation:

1. Load the test data: `data/test_simulation_labeled.csv`
2. Switch between color modes
3. Hover over different track points to see tooltips
4. Verify colors match annotation types
5. Check that the GUI styling is consistent across all panels

## Color Customization

To add or modify annotation colors, edit the `get_annotation_color()` function in `src/plotting.py`:

```python
color_map = {
    'YourAnnotation': (R, G, B),  # Add custom colors here
    # ...
}
```

RGB values should be in range 0-255.

## Troubleshooting

**Tooltips not appearing**:
- Ensure you're hovering close enough to a point (< 0.5 km)
- Check that data has been loaded
- Verify pyqtgraph is installed

**Colors not showing correctly**:
- Ensure data has an 'Annotation' column
- Select "Annotation" in the Color By dropdown
- Check annotation naming matches the color map

**GUI styling issues**:
- Verify PyQt6 is installed correctly
- Check that the stylesheet is applied in MainWindow.__init__()

## Version History

**v1.0** (Current):
- Initial implementation of hover tooltips
- Annotation-based color coding
- Modern GUI stylesheet
- Color-by selector in visualization panel

---

*For questions or issues, please refer to the main README.md or contact the development team.*
