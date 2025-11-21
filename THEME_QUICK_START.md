# Quick Start: Theme Settings

## Overview
The application now has a **Settings** tab with **Black** and **White** theme options.

## How to Switch Themes

### Step 1: Open Settings
1. Launch the application: `python -m src.gui` or `./run.sh`
2. Look at the left sidebar
3. Click on "⚙️ Settings" at the bottom

### Step 2: Select Theme
Click one of the theme buttons:
- **⚫ Black Theme** - Dark interface for low-light environments
- **⚪ White Theme** - Light interface for bright environments

### Step 3: Confirm
- Theme applies instantly (no restart needed)
- Status message shows: "✓ Applied [theme] theme successfully"
- Theme preference is automatically saved

## Default Theme
- The application starts with the **Black Theme** by default
- Your theme choice persists between sessions

## Theme Comparison

| Feature | Black Theme | White Theme |
|---------|-------------|-------------|
| **Background** | Dark slate (#2b3440) | Light gray (#f5f5f5) |
| **Text** | Light (#b8c5d6) | Dark (#2b3440) |
| **Best For** | Night use, tactical ops | Daytime, presentations |
| **Eye Strain** | Low in dark rooms | Low in bright rooms |
| **Professional** | Military/defense look | Office/business look |

## Keyboard Shortcuts
- Navigate to Settings: Click "Settings" in sidebar
- Switch themes: Click theme buttons

## Troubleshooting

### Theme not applying?
- Check status message in Settings panel
- Verify config file exists: `config/default_config.json`
- Check logs for errors

### Theme not persisting?
- Ensure config directory is writable
- Check file permissions on `config/default_config.json`

### Want to reset to default?
Delete `config/default_config.json` and restart the application.

## Manual Configuration

Edit `config/default_config.json`:

```json
{
  "theme": "black"
}
```

or

```json
{
  "theme": "white"
}
```

Save and restart the application.

## Technical Details

- **Location**: Settings tab in main application window
- **Components Styled**: All UI elements (buttons, inputs, tables, etc.)
- **Configuration**: Stored in `config/default_config.json`
- **Persistence**: Automatic on theme change
- **Switch Time**: Instant (no restart required)

---

**Quick Tip**: Try both themes to see which one works best for your environment!
