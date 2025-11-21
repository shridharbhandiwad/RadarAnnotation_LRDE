# Mono-Color Theme for Defense Application

## Overview
The application now features a **professional slate blue-gray mono-color theme** designed specifically for defense applications. This theme maintains a clean, sophisticated look while reducing visual distraction.

## Color Palette

### Base Colors (Slate Blue-Gray)
- **Darkest**: `#1c2329` - Main background, radar display
- **Dark**: `#2b3440` - Window background, secondary surfaces
- **Medium-Dark**: `#3d4a58` - Borders, containers
- **Medium**: `#4a5a6b` - Interactive elements
- **Medium-Light**: `#5a6b7d` - Active states, buttons
- **Light**: `#6a7b8d` - Hover states
- **Lighter**: `#7a8a9b` - Highlights
- **Text Light**: `#b8c5d6` - Primary text
- **Text Lighter**: `#c5d1df` - Labels
- **Text Lightest**: `#d5dfe9` - Emphasized text

## Design Principles

### 1. **Simplicity**
- Single color family (slate blue-gray)
- No bright accent colors
- Consistent visual hierarchy through shading

### 2. **Professional Appearance**
- Military/defense aesthetic
- Subdued, non-distracting interface
- Clean lines and subtle gradients

### 3. **Excellent Readability**
- High contrast between text and backgrounds
- Carefully selected text colors for different contexts
- Clear visual separation between UI elements

### 4. **Functional Elegance**
- Hover states show clear feedback
- Selected items are distinctly visible
- Disabled states are appropriately muted

## Component Styling

### Main Window
- Background: Dark slate (`#2b3440`)
- Creates a unified, immersive environment

### Sidebar (Engine Selector)
- Gradient from darkest to dark slate
- Selected items: Medium slate with light border accent
- Hover: Subtle transparency overlay

### Buttons
- Primary: Medium to medium-light slate gradient
- Hover: Lighter slate gradient
- Icons: Smaller buttons with same theme
- Disabled: Muted dark slate

### Input Fields
- Background: Dark slate (`#343e4c`)
- Border: Medium-dark slate
- Focus: Lighter slate border
- Text: Light slate

### Data Tables
- Background: Dark slate
- Headers: Dark gradient with light text
- Selection: Medium slate highlight
- Hover: Subtle background change

### Radar Display (PPI)
- Background: Darkest slate (`#1c2329`)
- Range rings: Medium-dark slate dashed lines
- Azimuth lines: Medium-dark slate
- Labels: Light slate text
- Tracks: Various slate shades for distinction

### Progress Bars
- Background: Dark slate
- Progress: Medium to light slate gradient
- Text: Light slate

### Scroll Bars
- Background: Dark slate
- Handle: Medium slate
- Hover: Medium-light slate
- Active: Light slate

## Track & Annotation Colors

All visualization colors use different shades of slate:

### Track Colors (10 distinct shades)
1. Light slate: `rgb(138, 154, 171)`
2. Medium-light slate: `rgb(106, 123, 141)`
3. Medium slate: `rgb(90, 107, 125)`
4. Medium-dark slate: `rgb(74, 90, 107)`
5. Very light slate: `rgb(184, 197, 214)`
6. Mid-light slate: `rgb(122, 138, 155)`
7. Dark slate: `rgb(61, 74, 88)`
8. Light-medium slate: `rgb(155, 167, 183)`
9. Lightest slate: `rgb(197, 205, 217)`
10. Darkest slate: `rgb(45, 56, 68)`

### Annotation Mapping
- **Level Flight**: Light slate
- **Ascending**: Very light slate
- **Descending**: Medium-light slate
- **High Speed**: Lightest slate
- **Low Speed**: Mid-light slate
- **Turning/Curved**: Medium slate
- **Straight/Linear**: Very light slate
- **Maneuvers**: Medium-dark to medium-light slate
- **Invalid**: Darkest slate

## Visual Features

### Hover Effects
- All interactive elements brighten on hover
- Consistent across all components
- Subtle but noticeable feedback

### Focus States
- Lighter borders indicate focus
- Slightly lighter backgrounds
- Clear indication of active input

### Selection
- Medium slate background
- Light text for contrast
- Border accents for additional clarity

### Disabled States
- Darker backgrounds
- Muted text colors
- No hover effects

## Benefits

1. **Eye Comfort**: Mono-color reduces eye strain during long operations
2. **Focus**: No distracting colors, attention stays on data
3. **Professional**: Military/defense-grade appearance
4. **Consistency**: Unified look across all panels
5. **Accessibility**: High contrast maintains readability
6. **Modern**: Clean, contemporary design language

## Usage

The theme is automatically applied when the application launches. No configuration needed.

To view the new theme:
```bash
python -m src.gui
```

The mono-color slate theme provides a sophisticated, distraction-free environment perfect for defense and tactical applications.

---

**Theme Version**: 1.0  
**Applied Date**: 2025-11-21  
**Color Family**: Slate Blue-Gray Mono-Color
