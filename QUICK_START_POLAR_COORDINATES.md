# Quick Start: PPI Polar Coordinates

## 🎯 What's New

The PPI can now display radar data as **Range and Azimuth** (polar coordinates) in addition to the traditional X-Y view!

## 🚀 3-Second Quick Start

1. Open GUI → Visualization
2. Load your data
3. Change **"Coordinates:"** dropdown to **"Polar (Range, Azimuth)"**

Done! Your PPI now shows Range vs Azimuth.

## 📊 What You'll See

### Cartesian Mode (Default):
```
╔════════════════════════════════════╗
║  PPI - Plan Position Indicator     ║
║                                    ║
║  Y Position (km)                   ║
║    ▲                               ║
║    │    • Track points             ║
║    │  •   in X-Y space             ║
║    │•                              ║
║    └────────────► X Position (km)  ║
╚════════════════════════════════════╝
```

### Polar Mode (New!):
```
╔════════════════════════════════════╗
║  PPI - Range vs Azimuth            ║
║                                    ║
║  Range (km)                        ║
║    ▲                               ║
║    │    • Track points             ║
║    │  •   in Range-Azimuth space   ║
║    │•                              ║
║    └────────────► Azimuth (degrees)║
╚════════════════════════════════════╝
```

## 🎨 Control Panel

The Visualization panel now has these controls:

```
┌────────────────────────────────────────────────┐
│ [Load Data for Visualization] (Green Button)  │
├────────────────────────────────────────────────┤
│ Coordinates:  [Cartesian (X, Y)          ▼]   │ ← NEW!
│ Color By:     [Track ID                  ▼]   │
│ Filter:       [All Tracks                ▼]   │
└────────────────────────────────────────────────┘
```

Simply change **"Coordinates:"** to switch modes!

## 💡 Hover Tooltip Shows Both!

When you hover over any track point, the tooltip shows **both** coordinate systems:

```
┌───────────────────────────────────┐
│ Track ID: 5                       │
│ Time: 12.45 s                     │
│ Cartesian: (10.50, 23.20) km     │ ← X, Y
│ Polar: Range=25.48 km, Az=65.7°  │ ← Range, Azimuth
│ Annotation: LevelFlight+HighSpeed│
└───────────────────────────────────┘
```

**You always see both** - no matter which mode you're in!

## 🔍 When to Use Each Mode

### Use **Cartesian (X, Y)** when:
- ✅ You want to see geographic/spatial layout
- ✅ Analyzing movements in X-Y plane
- ✅ Correlating with maps or terrain
- ✅ Understanding relative positions

### Use **Polar (Range, Azimuth)** when:
- ✅ You want to see radar-centric view
- ✅ Analyzing range-dependent effects
- ✅ Understanding angular coverage
- ✅ Correlating with radar parameters
- ✅ Analyzing approach/departure patterns

## 📐 Coordinate Systems Explained

### Cartesian Coordinates (X, Y):
```
      Y (North)
      ▲
      │
      │     • Target at (10, 20) km
      │    ╱
      │   ╱
      │  ╱
      │ ╱
      └─────────────► X (East)
```

### Polar Coordinates (Range, Azimuth):
```
    Same target:
    Range = √(10² + 20²) = 22.4 km
    Azimuth = arctan(20/10) = 63.4°
    
         Range
           ▲
           │
         22.4 km
           │  ╱) 63.4°
           │ ╱
           │╱
           └─────────► Azimuth (0° = East)
```

## 🎬 Step-by-Step Tutorial

### Step 1: Start the GUI
```bash
# Linux/Mac
./run.sh

# Windows
run.bat
```

### Step 2: Navigate to Visualization
Click **"📉 Visualization"** in the left sidebar

### Step 3: Load Your Data
1. Click **"Load Data for Visualization"** (green button)
2. Select a CSV file (e.g., `data/test_simulation_labeled.csv`)
3. Data appears in default Cartesian mode

### Step 4: Switch to Polar Mode
1. Find the **"Coordinates:"** dropdown (top of the panel)
2. Click it and select **"Polar (Range, Azimuth)"**
3. Watch the plot instantly update!

### Step 5: Explore Both Views
1. Switch back to **"Cartesian (X, Y)"** to compare
2. Hover over points to see both coordinate systems
3. Use **"Color By"** and **"Filter"** as normal

## 🔧 Tips & Tricks

### Tip 1: Quick Comparison
Toggle between modes to understand the data from different perspectives:
- Cartesian: "Where is it in space?"
- Polar: "How far and at what angle?"

### Tip 2: Use Hover Tooltips
Tooltips show both systems, so you can:
- Display in Polar mode (for radar view)
- Hover to get Cartesian coords (for map plotting)

### Tip 3: Combine with Color Coding
```
Coordinates: Polar (Range, Azimuth)
Color By: Annotation
```
Great for seeing which behaviors occur at different ranges/angles!

### Tip 4: Filter Tracks
```
Coordinates: Polar (Range, Azimuth)
Filter: Track 3
```
Focus on one track's range-azimuth evolution

## 📊 Example Scenarios

### Scenario 1: Range Analysis
**Question**: "At what range do aircraft typically start turning?"

**Solution**:
1. Set Coordinates: **Polar (Range, Azimuth)**
2. Set Color By: **Annotation**
3. Look for where "Turning" points appear on the Range axis

### Scenario 2: Angular Coverage
**Question**: "What azimuth sectors have the most activity?"

**Solution**:
1. Set Coordinates: **Polar (Range, Azimuth)**
2. Set Color By: **Track ID**
3. Observe concentration of points on Azimuth axis

### Scenario 3: Approach Pattern
**Question**: "How do aircraft approach the radar?"

**Solution**:
1. Set Coordinates: **Polar (Range, Azimuth)**
2. Filter: **Track 5** (one approaching aircraft)
3. Watch range decrease over time (points move down)

## ❓ Troubleshooting

**Q: Polar mode looks different from Cartesian**
- ✅ This is expected! Same data, different coordinate system
- ✅ Use tooltips to verify - coordinates match mathematically

**Q: Some tracks look "stretched" in Polar mode**
- ✅ Normal - azimuth is in degrees (-180° to +180°)
- ✅ Range is in km, creating different scaling

**Q: Tooltip not showing polar coordinates**
- ✅ Check that data has x, y columns
- ✅ Polar coords are computed automatically from x, y

**Q: Can I use both modes at once?**
- ❌ No, but you can:
  - Quickly toggle between modes
  - Use tooltips to see both (they always show both)

## 🎓 Understanding the Math

The conversion is straightforward:

### From Cartesian to Polar:
```python
Range = √(x² + y²)
Azimuth = arctan2(y, x)  # in degrees
```

### From Polar to Cartesian:
```python
x = Range × cos(Azimuth)
y = Range × sin(Azimuth)
```

Both are computed automatically - you don't need to do anything!

## ✨ Feature Benefits

1. **Dual Perspective**: See data how you need it
2. **No Data Loss**: Full information in both modes
3. **Instant Switching**: No reload required
4. **Smart Tooltips**: Always see both systems
5. **Full Integration**: Works with all existing features

## 📚 Related Documentation

- **PPI_POLAR_COORDINATES.md** - Complete technical documentation
- **PPI_ENHANCEMENTS.md** - Color coding and tooltip features
- **QUICK_START_PPI_FEATURES.md** - General PPI features guide

## 🎉 You're Ready!

Start exploring your radar data in both Cartesian and Polar coordinates. The best way to learn is to try both modes and see which perspective helps you understand your specific data better!

---

**Pro Tip**: Most radar analysts find Polar mode more intuitive for radar-specific analysis, while Cartesian is better for spatial/geographic understanding. Use both!
