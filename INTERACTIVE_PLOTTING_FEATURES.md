# Interactive Plotting Features Guide

## Overview

The visualization panel now includes advanced interactive features to enhance plot exploration and analysis. These features provide comprehensive control over the plot view, history management, magnification, and export capabilities.

## New Features

### 1. **Panning** 🖱️
- **How to use**: Click and drag on any plot to pan around
- **Description**: Move the plot view in any direction without changing zoom level
- **Availability**: Enabled by default on all plots (PPI and Time Series)

### 2. **Zoom Controls** 🔍

#### Zoom In
- **Button**: "➕ Zoom In"
- **Action**: Zooms in by 20% centered on the current view
- **Keyboard**: Mouse scroll wheel up (native pyqtgraph support)

#### Zoom Out
- **Button**: "➖ Zoom Out"
- **Action**: Zooms out by 20% to see more data
- **Keyboard**: Mouse scroll wheel down (native pyqtgraph support)

#### Reset View
- **Button**: "🔄 Reset View"
- **Action**: Automatically scales the plot to show all data
- **Use case**: Quick return to full data view after zooming

#### Rectangle Zoom
- **How to use**: Right-click and drag to select a rectangular region
- **Action**: Zooms to fit the selected rectangle
- **Use case**: Precise zoom to a specific area of interest

### 3. **View History (Undo/Redo)** ⏮️⏭️

#### Undo
- **Button**: "⬅ Undo"
- **Action**: Returns to the previous view state
- **History**: Maintains up to 50 previous view states
- **Smart saving**: Automatically saves view state after 500ms of no changes (debounced)

#### Redo
- **Button**: "➡ Redo"
- **Action**: Moves forward to the next view state (after undo)
- **Use case**: Restore view after accidental undo

**Note**: View history tracks pan and zoom operations, allowing you to navigate through your exploration history.

### 4. **Magnifier Lens** 🔎

#### Enable/Disable Magnifier
- **Button**: "🔍 Enable Magnifier" (toggles to "🔍 Disable Magnifier")
- **Action**: Shows a circular magnifier lens that follows your mouse cursor
- **Visual**: Yellow circular outline with semi-transparent center

#### Magnifier Zoom Factor
- **Control**: Spin box (Range: 1.5x to 10.0x, Default: 3.0x)
- **Action**: Adjusts the magnification level of the lens
- **Use case**: Fine-tune magnification for detailed inspection

#### How It Works
1. Click "Enable Magnifier" button
2. Move mouse over plot - lens follows cursor
3. Adjust zoom factor as needed
4. Click button again to disable

**Pro Tip**: The magnifier is excellent for inspecting closely-spaced data points or examining fine details without changing the main plot view.

### 5. **Plot Size Customization** 📐

#### Width and Height Controls
- **Width Range**: 400-3000 pixels (Default: 800px)
- **Height Range**: 300-2000 pixels (Default: 600px)
- **Button**: "Apply Size"

#### Use Cases
- **Presentations**: Increase size for better visibility
- **Multi-monitor**: Optimize for your screen layout
- **Export**: Set specific dimensions before exporting
- **Detail work**: Larger plots show more detail

### 6. **Export to Image** 💾

#### Export PPI Plot
- **Button**: "💾 Save PPI Plot"
- **Formats**: PNG, JPEG
- **Quality**: High-resolution export of current view
- **Includes**: All visible elements (data points, grids, labels, legend)

#### Export Time Series
- **Button**: "💾 Save Time Series"
- **Formats**: PNG, JPEG
- **Content**: Exports all three time series plots (Altitude, Speed, Curvature)
- **Use case**: Reports, presentations, documentation

## Usage Examples

### Example 1: Detailed Region Inspection
1. Load your data
2. Right-click and drag to zoom to region of interest
3. Enable magnifier to inspect specific points
4. Adjust magnifier zoom to 5.0x for very detailed view
5. Export the view for documentation

### Example 2: Presentation Preparation
1. Set plot size to 1920x1080 for full HD
2. Apply the size
3. Zoom and pan to desired view
4. Export as PNG
5. Use undo if you need to adjust

### Example 3: Exploratory Analysis
1. Start with reset view to see all data
2. Zoom in to interesting areas
3. Use undo/redo to compare different regions
4. Pan around to explore neighboring areas
5. Save multiple exports of different regions

## Interactive Controls Layout

### Top Bar (Always Visible)
- Load Data
- Display Mode selector
- Color By selector
- Track Filter
- Time Series toggle

### Interactive Controls Bar (New)
```
┌─────────────────────────────────────────────────────────────────────┐
│ [Zoom & View]  [History]  [Magnifier]  [Export]                     │
│                                                                       │
│ ➕ Zoom In    ⬅ Undo     🔍 Enable    💾 Save PPI                   │
│ ➖ Zoom Out   ➡ Redo        Zoom: 3.0  💾 Save TS                   │
│ 🔄 Reset                                                             │
└─────────────────────────────────────────────────────────────────────┘
```

### Plot Size Controls
```
Plot Size:  Width: [800] px  Height: [600] px  [Apply Size]
```

## Technical Details

### View History Management
- **Storage**: Up to 50 view states in memory
- **Efficiency**: Debounced saving (500ms delay) to avoid excessive history entries
- **Data Stored**: X and Y ranges for each state
- **Memory**: Minimal overhead (~1KB per state)

### Magnifier Implementation
- **Type**: Graphics overlay with circular shape
- **Performance**: Hardware-accelerated rendering
- **Z-Index**: High value (1000) to always appear on top
- **Follow Mode**: Smooth cursor tracking

### Export Quality
- **Resolution**: Native plot resolution (adjustable via plot size)
- **Format**: Lossless PNG or compressed JPEG
- **DPI**: Screen DPI by default
- **Anti-aliasing**: Enabled for smooth lines and text

## Keyboard Shortcuts (Native PyQtGraph)

While the GUI provides buttons, you can also use these native shortcuts:

- **Pan**: Click and drag
- **Zoom In**: Mouse wheel up
- **Zoom Out**: Mouse wheel down
- **Rectangle Zoom**: Right-click and drag
- **Auto Range**: Middle mouse button (if available)

## Tips and Best Practices

1. **Use View History**: Don't be afraid to explore - you can always undo back to where you started

2. **Magnifier for Details**: Perfect for examining overlapping data points or precise measurements

3. **Export Before Closing**: Save interesting views as images for later reference

4. **Custom Sizes for Export**: Set your desired export size before saving the image

5. **Reset Often**: Use "Reset View" to quickly return to full data view when disoriented

6. **Combine Features**: Use magnifier with zoom for the best of both worlds - zoomed view with local magnification

7. **History Limit**: Remember that history is limited to 50 states, so very old views may be lost

## Troubleshooting

### Magnifier Not Showing
- Ensure the "Enable Magnifier" button is pressed (toggled on)
- Move mouse over the plot area
- Check that magnifier zoom is set above 1.5x

### Export Failed
- Verify write permissions in the target directory
- Check available disk space
- Try a different file format (PNG vs JPEG)

### Undo/Redo Not Available
- History is only saved after pan/zoom operations
- Wait 500ms after zoom/pan for state to be saved
- History is cleared when loading new data

### Plot Size Not Applied
- Click "Apply Size" button after changing dimensions
- Very large sizes may take a moment to render
- Some systems may have maximum window size limits

## System Requirements

- **PyQt6**: Required for GUI framework
- **PyQtGraph**: Required for plotting (with exporters)
- **Python**: 3.7+
- **Memory**: Minimal overhead (< 50MB for history)
- **Display**: Any resolution (responsive design)

## Future Enhancements (Potential)

- Keyboard shortcuts for undo/redo
- Multiple magnifier lenses simultaneously
- Crosshair cursor for precise coordinate reading
- Snapshot gallery to save multiple views
- Animated transitions between history states
- Custom zoom levels (50%, 100%, 200%, etc.)
- Region annotations and measurements

---

## Quick Reference Card

| Feature | Button/Control | Shortcut |
|---------|---------------|----------|
| Pan | Click & Drag | - |
| Zoom In | ➕ Zoom In | Scroll Up |
| Zoom Out | ➖ Zoom Out | Scroll Down |
| Rectangle Zoom | - | Right-Click & Drag |
| Reset View | 🔄 Reset | - |
| Undo | ⬅ Undo | - |
| Redo | ➡ Redo | - |
| Magnifier | 🔍 Toggle | - |
| Export PPI | 💾 Save PPI | - |
| Export Time Series | 💾 Save TS | - |
| Apply Size | Apply Size | - |

---

**Last Updated**: 2025-11-21  
**Version**: 1.0  
**Compatibility**: All radar data visualization modes (Radar View, Cartesian, Polar)
