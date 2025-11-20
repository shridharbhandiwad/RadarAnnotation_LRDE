# Quick Start: Circular PPI Radar View

## What's New? 🎉

Your PPI display is now a **true circular radar scope** with:
- ⭕ **Round display** (not rectangular)
- 🎯 **Range rings** - concentric circles showing distance (0-50 km, auto-scaled)
- 📐 **Azimuth segments** - radial lines every 30° (N, E, S, W + angles)
- 🖥️ **Radar-style dark background** - authentic radar screen appearance

## Quick Start (3 Steps)

### Step 1: Launch the Application
```bash
# Linux/Mac
./run.sh

# Windows
run.bat
```

### Step 2: Load Data
1. Click on **"📉 Visualization"** in the left sidebar
2. Click the green button: **"Load Data for Visualization"**
3. Select: `data/test_simulation_labeled.csv`

### Step 3: View the Circular PPI
The display automatically shows **Radar View (Circular)** mode!

You should now see:
- ✅ A circular radar display (round, not square)
- ✅ Green concentric circles (range rings)
- ✅ Green radial lines from center (azimuth segments)
- ✅ Labels: N, E, S, W for cardinal directions
- ✅ Your track data plotted on this circular background

## Visual Reference

```
         N (90°)
          │
    150°  │  30°
       ╲  │  ╱
  W ─────(O)───── E (0°)
       ╱  │  ╲
    210°  │  300°
          │
         S (270°)

Legend:
(O) = Radar center (your position)
│─╲╱ = Azimuth lines (angle markers)
Circles = Range rings (distance markers)
● = Track points (your data)
```

## Display Modes

Switch between modes using the **"Display Mode"** dropdown:

| Mode | Description | Use Case |
|------|-------------|----------|
| **Radar View (Circular)** ⭐ | Round PPI with range rings | Default - Best for radar operators |
| Cartesian (X, Y) | Traditional X-Y plot | Spatial analysis |
| Polar (Range, Azimuth) | Linear polar plot | Data analysis |

## Features You Can Use

### 1. Color Coding
Use **"Color By"** dropdown:
- **Track ID**: Each track has unique color
- **Annotation**: Colors based on behavior (HighSpeed=Red, LevelFlight=Blue, etc.)

### 2. Track Filtering
Use **"Filter Track ID"** dropdown:
- Select "All Tracks" to see everything
- Select specific track (e.g., "Track 5") to focus on one

### 3. Interactive Tooltips
Hover your mouse over any track point to see:
- Track ID
- Time
- Position in Cartesian (X, Y in km)
- Position in Polar (Range in km, Azimuth in degrees)
- Annotation/behavior

### 4. Zoom and Pan
- **Zoom**: Scroll mouse wheel
- **Pan**: Click and drag
- **Reset**: Right-click → "View All"

## Understanding the Circular PPI

### Range Rings (Circles)
The concentric circles show distance from center:
```
Innermost circle = 10 km
Next circle = 20 km
... and so on ...
Outermost circle = 50 km (auto-scaled to your data)
```

Range is labeled at the **top** (North) of each circle.

### Azimuth Segments (Radial Lines)
The lines radiating from center show angles:
```
0° (E) = East (right)
90° (N) = North (top)
180° (W) = West (left)
270° (S) = South (bottom)

Plus intermediate angles: 30°, 60°, 120°, 150°, 210°, 240°, 300°, 330°
```

### Cardinal Directions
- **N** = North (top)
- **E** = East (right)
- **S** = South (bottom)
- **W** = West (left)

## Example: Reading the Display

If you see a track point at:
- **Between the 2nd and 3rd circle** → Range ≈ 20-30 km
- **On the line marked "60°"** → Bearing 60° (between E and N)
- **Tooltip says**: "Range=25.5 km, Az=65.7°" → Exact position

## Tips and Tricks

### Tip 1: Auto-Scaling
Range rings automatically adjust to your data:
- Max track at 47 km → Display shows 0-50 km (rounds up to 10)
- Max track at 123 km → Display shows 0-130 km

### Tip 2: Multiple Tracks
Different colors help distinguish tracks:
- Red, Green, Blue, Yellow, Magenta, Cyan, etc.
- Or use annotation coloring for behavior-based colors

### Tip 3: Time Series
The plots below the PPI show:
- **Top**: Altitude vs Time
- **Middle**: Speed vs Time
- **Bottom**: Curvature vs Time

These sync with the PPI display!

### Tip 4: Professional Look
The dark background (#0a0a0a) mimics real radar displays:
- High contrast green on black
- Easy on eyes during long monitoring sessions
- Professional military/ATC appearance

## Common Questions

**Q: Can I switch back to the old rectangular view?**  
A: Yes! Use "Display Mode" dropdown → "Cartesian (X, Y)"

**Q: The circles look oval/stretched**  
A: Resize the window - aspect ratio is locked, so make it square-ish

**Q: I want more/fewer range rings**  
A: Edit `src/plotting.py`, line ~175: `num_rings = 5` (change to 4, 6, etc.)

**Q: Can I change angle intervals?**  
A: Yes! Edit `src/plotting.py`, line ~192. See customization guide in main docs.

**Q: Why is center at (0,0)?**  
A: This represents YOUR radar position. All tracks are relative to you.

**Q: What if my tracks are off-center?**  
A: The display shows all tracks. If they're not centered, your data has an offset.

## Testing with Sample Data

The included test data `data/test_simulation_labeled.csv` shows:
- Multiple aircraft tracks
- Various behaviors (level flight, climbing, turning, etc.)
- Range: 0-50 km (approximately)
- Perfect for testing the circular PPI

You should see:
- ✅ Tracks of different colors
- ✅ Some tracks moving outward (outgoing)
- ✅ Some tracks moving inward (incoming)
- ✅ Various flight patterns

## Next Steps

1. ✅ **Explore**: Try zooming, panning, and hovering over tracks
2. ✅ **Experiment**: Switch between display modes
3. ✅ **Customize**: Try different color-by options
4. ✅ **Load your data**: Use your own CSV files (must have x, y, trackid columns)

## Complete Documentation

For detailed technical information, see:
- **CIRCULAR_PPI_IMPLEMENTATION.md** - Full implementation details
- **PPI_ENHANCEMENTS.md** - Color coding and tooltips
- **PPI_POLAR_COORDINATES.md** - Coordinate systems
- **README.md** - Main project documentation

## Keyboard Reference

| Action | Method |
|--------|--------|
| Zoom In/Out | Mouse wheel |
| Pan | Left-click + drag |
| Reset View | Right-click → "View All" |
| Tooltip | Hover over track point |

---

## Summary

You now have a **professional circular radar display** with:
- ⭕ Round scope (authentic radar appearance)
- 🎯 5 range rings (auto-scaled)
- 📐 12 azimuth segments (every 30°)
- 🎨 Color-coded tracks
- 💬 Interactive tooltips
- 🖱️ Zoom and pan controls

**Enjoy your new circular PPI! 🎉**
