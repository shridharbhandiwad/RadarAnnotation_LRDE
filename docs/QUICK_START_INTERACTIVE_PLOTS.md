# Quick Start: Interactive Plotting Features

## What's New? 🎉

Your radar data visualization now has **advanced interactive features** that make plot exploration much more powerful and intuitive!

## 5-Minute Quick Start

### 1. Launch the Application
```bash
# Linux/Mac
./run.sh

# Windows
run.bat
```

### 2. Navigate to Visualization
- Click on **"📉 Visualization"** in the left sidebar

### 3. Load Your Data
- Click **"Load Data"** button
- Select a CSV file with radar track data
- Plot will automatically display

### 4. Try the Interactive Features!

#### 🖱️ **Pan Around**
- Simply **click and drag** anywhere on the plot
- Move the view to explore different regions

#### 🔍 **Zoom In/Out**
- Click **"➕ Zoom In"** or **"➖ Zoom Out"** buttons
- Or use your **mouse scroll wheel**
- Right-click and drag to **zoom to a specific rectangle**

#### ⏮️ **Undo/Redo Your View**
- Explored too far? Click **"⬅ Undo"** to go back
- Changed your mind? Click **"➡ Redo"** to go forward
- View history remembers your last 50 zoom/pan operations!

#### 🔎 **Use the Magnifier**
1. Click **"🔍 Enable Magnifier"**
2. Move your mouse over the plot
3. See a magnified view following your cursor
4. Adjust the zoom factor (1.5x to 10.0x) as needed

#### 💾 **Export Your Plots**
- Click **"💾 Save PPI Plot"** to save the radar view
- Click **"💾 Save Time Series"** to save the time series charts
- Choose PNG (best quality) or JPEG (smaller size)

#### 📐 **Customize Plot Size**
1. Set width and height (in pixels)
2. Click **"Apply Size"**
3. Plot resizes to your specifications
4. Perfect for preparing presentations!

## Feature Showcase

### Scenario 1: Finding Specific Track Behavior
```
1. Load data → 2. Zoom to region of interest → 3. Enable magnifier
4. Inspect individual points → 5. Export the view
```

### Scenario 2: Comparing Multiple Regions
```
1. Zoom to first region → 2. Study it → 3. Zoom to second region
4. Use "Undo" to go back → 5. Use "Redo" to return
```

### Scenario 3: Preparing for Presentation
```
1. Set plot size to 1920x1080 → 2. Apply Size
3. Adjust view to highlight key data → 4. Export as PNG
```

## All Interactive Controls at a Glance

| Feature | Location | Function |
|---------|----------|----------|
| **Panning** | Click & Drag | Move plot view around |
| **Zoom In** | Zoom & View group | Zoom in 20% |
| **Zoom Out** | Zoom & View group | Zoom out 20% |
| **Reset View** | Zoom & View group | Show all data |
| **Undo** | History group | Previous view state |
| **Redo** | History group | Next view state |
| **Magnifier Toggle** | Magnifier group | Enable/disable lens |
| **Magnifier Zoom** | Magnifier group | 1.5x to 10.0x |
| **Export PPI** | Export group | Save radar plot |
| **Export Time Series** | Export group | Save time charts |
| **Plot Width** | Plot Size row | Set width (400-3000px) |
| **Plot Height** | Plot Size row | Set height (300-2000px) |
| **Apply Size** | Plot Size row | Apply dimensions |

## Tips for Best Results

✅ **DO:**
- Use undo/redo freely - your exploration is saved
- Try the magnifier for detailed point inspection
- Export at larger sizes for better image quality
- Use reset view when you get lost

❌ **AVOID:**
- Very large plot sizes (>2000px) on slow computers
- Magnifier zoom above 8x (can be too sensitive)
- Rapid undo/redo clicking (wait for view to update)

## Keyboard Shortcuts (Built-in)

- **Mouse Wheel Up**: Zoom in
- **Mouse Wheel Down**: Zoom out
- **Right-Click + Drag**: Zoom to rectangle
- **Left-Click + Drag**: Pan

## What Works Where?

| Feature | PPI Plot | Time Series |
|---------|----------|-------------|
| Panning | ✅ Yes | ✅ Yes |
| Zoom In/Out | ✅ Yes | ✅ Yes |
| Reset View | ✅ Yes | ✅ Yes |
| Undo/Redo | ✅ Yes | ❌ No |
| Magnifier | ✅ Yes | ❌ No |
| Export | ✅ Yes | ✅ Yes |
| Custom Size | ✅ Yes | ❌ No |

## Troubleshooting

**Q: Magnifier isn't showing**  
A: Make sure the button says "🔍 Disable Magnifier" (enabled) and move mouse over plot

**Q: Undo button does nothing**  
A: History is only saved after zoom/pan. Try zooming first.

**Q: Export creates empty/black image**  
A: Load data first, then export. Ensure plot has content.

**Q: Plot size won't change**  
A: Click "Apply Size" button after setting width/height values.

**Q: Can I use keyboard shortcuts?**  
A: Mouse wheel and click-drag work natively. Button-based features need mouse clicks.

## Need More Help?

See the full guide: `INTERACTIVE_PLOTTING_FEATURES.md`

## Demo Video Script (Try This!)

1. **Start**: Launch GUI, go to Visualization
2. **Load**: Select `data/test_simulation_labeled.csv`
3. **Explore**: Click and drag to pan around
4. **Zoom**: Right-click drag a rectangle around interesting tracks
5. **Magnify**: Enable magnifier, move mouse to inspect points
6. **Navigate**: Click Undo to go back, Redo to return
7. **Reset**: Click Reset View to see everything again
8. **Export**: Save PPI plot as `my_radar_view.png`
9. **Customize**: Set size to 1600x1200, Apply, see the difference
10. **Share**: Open the exported PNG - perfect for reports!

---

**Total Time to Master**: ~5 minutes  
**Complexity**: Beginner-friendly  
**Requirements**: PyQt6, PyQtGraph installed

**Ready to explore?** Fire up the GUI and start interacting with your radar data like never before! 🚀
