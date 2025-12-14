# Visualization Chronology Markers

## Summary
Added start and end point markers to the visualization to clearly indicate track chronology.

## Changes Made

### 1. PPI Plot (Plan Position Indicator)
**File**: `src/plotting.py`
**Method**: `PPIPlotWidget.plot_tracks()`

#### Start Point Markers
- **Symbol**: Green star (★)
- **Color**: Green (46, 204, 113)
- **Border**: Dark green (0, 100, 0)
- **Size**: 12 pixels
- **Purpose**: Indicates where each track begins (earliest time point)

#### End Point Markers
- **Symbol**: Red triangle (▲)
- **Color**: Red (231, 76, 60)
- **Border**: Dark red (139, 0, 0)
- **Size**: 12 pixels
- **Purpose**: Indicates where each track ends (latest time point)

#### Features
- Markers appear in the legend as "Start Points" and "End Points"
- Markers are displayed on top of all other plot elements (z-index: 100)
- Markers are hoverable with white highlight borders
- Works in all coordinate modes:
  - Radar View (Circular)
  - Cartesian (X, Y)
  - Polar (Range, Azimuth)

### 2. Time Series Plots
**File**: `src/plotting.py`
**Method**: `TimeSeriesPlotWidget.plot_tracks()`

#### Implementation
Start and end markers are added to all three time series plots:
- **Altitude vs Time**
- **Speed vs Time**
- **Curvature vs Time**

#### Marker Style
- Same symbols and colors as PPI plot:
  - **Start**: Green star
  - **End**: Red triangle
- Markers appear only once in the legend (on the first track)
- Markers are displayed on top of plot elements (z-index: 100)

## Benefits

1. **Clear Chronology**: Users can immediately see the direction of track movement
2. **Multi-Track Visualization**: Essential when viewing multiple overlapping tracks
3. **Consistent Design**: Same markers across PPI and time series plots
4. **Visual Clarity**: Distinct colors (green for start, red for end) follow common conventions
5. **Interactive**: Markers support hover effects for better visibility

## Usage

The markers will automatically appear whenever data is visualized through the Visualization panel:

1. Load data using the "Load Data" button
2. Start points (green stars) and end points (red triangles) will automatically appear
3. Works with all color modes:
   - Track ID
   - Annotation
   - Track Segments (Colored by Annotation)
4. Works with track filtering (when viewing individual tracks)

## Technical Details

### Chronology Determination
- Tracks are sorted by the 'time' column
- First point (earliest time) = Start point
- Last point (latest time) = End point

### Coordinate Handling
- In Cartesian/Circular modes: Uses x_km, y_km coordinates
- In Polar mode: Uses azimuth_deg, range_km coordinates
- Time series: Uses time on x-axis, various metrics on y-axis

### Legend Integration
- Start and end points appear in the plot legend
- Can be toggled on/off through the legend (if PyQtGraph supports it)
- Legend entries created only once to avoid duplication

## Example Visual

```
PPI Plot:
  ★ (green)  = Track starts here
  •••••••••• = Track trajectory
  ▲ (red)    = Track ends here
```

## Compatibility

- Works with existing themes (white and black)
- Compatible with all visualization features:
  - Zoom/Pan
  - Magnifier
  - Export to image
  - Undo/Redo view history
  - Track highlighting
