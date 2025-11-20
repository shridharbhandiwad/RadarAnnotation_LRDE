# GUI Not Opening - Fix Guide

## Problem

When you run `run.bat` or `python -m src.gui`, you see:
- TensorFlow initialization messages
- No GUI window appears
- Script exits with "Press any key to continue..."

## Root Cause

**PyQt6 is not installed** in your Python environment. The application loads TensorFlow (from the AI engine module), but when it tries to create the GUI window, it detects PyQt6 is missing.

## Solution

You need to install PyQt6 and pyqtgraph. Choose one of the options below:

### Option 1: Use the Installation Helper (Easiest)

**Windows:**
```bash
install_gui.bat
```

**Linux/Mac:**
```bash
./install_gui.sh
```

### Option 2: Manual Installation (Recommended)

Install just the GUI packages:
```bash
pip install PyQt6 pyqtgraph
```

### Option 3: Install All Dependencies

Install everything including ML libraries:
```bash
pip install -r requirements.txt
```

### Option 4: Using Conda (Anaconda Users)

```bash
conda install -c conda-forge pyqt
pip install pyqtgraph
```

## Verification

After installation, verify PyQt6 is working:

```bash
python -c "from PyQt6.QtWidgets import QApplication; print('PyQt6 is installed correctly!')"
```

If you see "PyQt6 is installed correctly!", you're good to go!

## Running the Application

After installing PyQt6, run the application:

**Windows:**
```bash
run.bat
```

**Linux/Mac:**
```bash
./run.sh
```

Or directly:
```bash
python -m src.gui
```

## Expected Behavior

When properly configured, you should see:
1. TensorFlow initialization messages (brief)
2. A GUI window opens with the Radar Data Annotation Application
3. Left sidebar with engine options:
   - Data Extraction
   - AutoLabeling
   - AI Tagging
   - Report
   - Simulation
   - Visualization

## Troubleshooting

### Still Not Working?

1. **Check Python version** (should be 3.10+):
   ```bash
   python --version
   ```

2. **Verify pip is working**:
   ```bash
   pip --version
   ```

3. **Check if you're in the correct virtual environment**:
   - Windows: `radarenv\Scripts\activate`
   - Linux/Mac: `source radarenv/bin/activate`

4. **Try upgrading pip first**:
   ```bash
   python -m pip install --upgrade pip
   pip install PyQt6 pyqtgraph
   ```

### Error: "No module named 'PyQt6'"

This means the installation didn't work or you're using a different Python environment. Make sure:
- You're using the same Python that runs the app
- You're in the correct virtual environment (if using one)
- The installation completed without errors

### GUI Opens But Is Blank/Frozen

This could be a graphics driver issue. Try:
1. Updating your graphics drivers
2. Setting a different Qt platform:
   ```bash
   set QT_QPA_PLATFORM=windows  # Windows
   export QT_QPA_PLATFORM=xcb   # Linux
   ```

### Permission Errors During Installation

Try:
```bash
pip install --user PyQt6 pyqtgraph
```

## What Changed

The code has been updated to:
1. Show a clear, visible error message if PyQt6 is missing
2. Wait for user input before exiting (Windows)
3. Provide better error handling for GUI startup failures
4. Include installation helper scripts

## Support

If you continue to have issues:
1. Check that TensorFlow loads successfully (you should see its messages)
2. Ensure no firewall/antivirus is blocking Python
3. Try creating a fresh virtual environment:
   ```bash
   python -m venv new_env
   new_env\Scripts\activate  # Windows
   pip install PyQt6 pyqtgraph pandas numpy
   ```

## Related Files

- `src/gui.py` - Main GUI module (improved error handling)
- `run.bat` / `run.sh` - Launch scripts (updated)
- `install_gui.bat` / `install_gui.sh` - New installation helpers
- `requirements.txt` - Full project dependencies
