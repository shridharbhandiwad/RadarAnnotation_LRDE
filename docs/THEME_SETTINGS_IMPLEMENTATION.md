# Theme Settings Implementation

## Overview
The Radar Data Annotation Application now includes a **Settings Tab** with theme selection functionality, allowing users to switch between **Black (Dark)** and **White (Light)** themes.

## Features Implemented

### 1. Settings Tab
- New "⚙️ Settings" option added to the main engine selector
- Dedicated settings panel with theme customization options
- Clean, intuitive interface for theme selection

### 2. Theme Options

#### Black Theme (Default)
- **Purpose**: Dark professional interface optimized for low-light environments
- **Best For**: Defense and tactical applications, extended viewing sessions
- **Colors**: Slate blue-gray mono-color palette
- **Features**:
  - Dark backgrounds (#2b3440, #1c2329)
  - Light text for contrast (#b8c5d6, #c5d1df)
  - Subtle gradients for depth
  - Professional military aesthetic

#### White Theme
- **Purpose**: Light clean interface for bright environments  
- **Best For**: Well-lit offices, presentations, daytime operations
- **Colors**: Light grayscale with subtle accents
- **Features**:
  - Light backgrounds (#ffffff, #f5f5f5)
  - Dark text for readability (#2b3440, #1c2329)
  - Clean, modern appearance
  - Reduced eye strain in bright conditions

### 3. Theme UI Components

Both themes include complete styling for:
- Main window and backgrounds
- List widgets (engine selector)
- Group boxes and containers
- Buttons (standard, primary, icon, theme selection)
- Labels and text
- Text edit fields
- Combo boxes and dropdowns
- Spin boxes
- Tables and data grids
- Progress bars
- Scroll bars
- Splitters
- Sliders
- All interactive elements with hover/focus/pressed states

### 4. Theme Persistence
- Selected theme is automatically saved to configuration file
- Theme preference persists between application sessions
- Saved in: `config/default_config.json` under the `"theme"` key
- Valid values: `"black"` or `"white"`

## Usage

### Switching Themes via GUI
1. Launch the application
2. Navigate to "⚙️ Settings" in the left sidebar
3. Click on either:
   - **⚫ Black Theme** button
   - **⚪ White Theme** button
4. Theme applies instantly
5. Status message confirms successful application
6. Theme preference is automatically saved

### Default Theme
- The application defaults to the **Black Theme**
- First-time users will see the dark interface
- Theme can be changed at any time

### Manual Configuration
You can also manually set the theme in `config/default_config.json`:

```json
{
  "theme": "black",
  ...
}
```

Or:

```json
{
  "theme": "white",
  ...
}
```

## Technical Implementation

### Files Modified
1. **src/gui.py**:
   - Added `SettingsPanel` class (lines 523-614)
   - Modified `MainWindow` class with theme methods
   - Added `set_theme()` method for switching themes
   - Added `get_theme_stylesheet()` method
   - Added `get_black_theme()` method with complete dark theme
   - Added `get_white_theme()` method with complete light theme
   - Updated engine list to include Settings
   - Updated panel stack to include SettingsPanel

2. **src/config.py**:
   - Added `"theme": "black"` to DEFAULT_CONFIG
   - Modified `save_default_config()` to accept config data parameter
   - Enables theme persistence

### Key Methods

#### `SettingsPanel.apply_theme(theme_name)`
- Applies selected theme
- Finds main window and calls `set_theme()`
- Updates button states
- Shows status message

#### `MainWindow.set_theme(theme_name)`
- Sets current theme
- Saves preference to config file
- Applies stylesheet immediately

#### `MainWindow.get_theme_stylesheet()`
- Returns appropriate stylesheet based on current theme
- Delegates to `get_black_theme()` or `get_white_theme()`

#### `MainWindow.get_black_theme()`
- Returns complete CSS stylesheet for dark theme
- ~400 lines of styling rules

#### `MainWindow.get_white_theme()`
- Returns complete CSS stylesheet for light theme
- ~400 lines of styling rules

## UI Components Styling

### Theme Selection Buttons
- Large, prominent buttons (60px height)
- Emoji indicators (⚫ Black, ⚪ White)
- Checkable state shows active theme
- Visual feedback on hover and selection
- Special `#themeButton` styling in both themes

### Hover Effects
- All interactive elements provide visual feedback
- Consistent behavior across both themes
- Color shifts indicate interactivity

### Focus States
- Input fields show clear focus indication
- Helps with keyboard navigation
- Different colors for each theme

### Selection States
- Lists, tables, and dropdowns show selected items clearly
- High contrast for readability
- Theme-appropriate colors

## Benefits

### Black Theme Benefits
1. **Eye Comfort**: Reduced eye strain in low-light conditions
2. **Professional**: Military/defense-grade appearance
3. **Focus**: Dark backgrounds keep attention on data
4. **Energy Saving**: Less screen brightness needed
5. **Tactical**: Suitable for control room environments

### White Theme Benefits
1. **Brightness**: Better visibility in well-lit spaces
2. **Familiarity**: Traditional application appearance
3. **Presentations**: More suitable for projectors/presentations
4. **Accessibility**: Some users prefer light themes
5. **Daytime Use**: Optimized for office environments

## Accessibility

Both themes maintain:
- High contrast ratios for readability
- Clear visual hierarchy
- Consistent interaction patterns
- Keyboard navigation support
- Hover and focus indicators

## Testing

To test the theme switching:

```bash
# Run the application
python -m src.gui

# Or use the convenience scripts
./run.sh      # Linux/Mac
run.bat       # Windows
```

Then:
1. Navigate to Settings tab
2. Click Black Theme button - observe dark theme
3. Click White Theme button - observe light theme
4. Restart application - verify theme persists

## Future Enhancements

Potential additions:
- Custom color schemes
- Theme scheduling (auto-switch based on time of day)
- Per-panel theme overrides
- High contrast mode
- Color blind friendly themes
- Import/export theme configurations

## Notes

- Theme switching is instant (no restart required)
- All panels and components update simultaneously
- Theme preference saved automatically
- Robust error handling prevents theme application failures
- Logging included for debugging theme issues

---

**Implementation Date**: 2025-11-21  
**Version**: 1.0  
**Status**: ✅ Complete and Tested
