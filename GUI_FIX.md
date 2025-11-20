# GUI Error Fix - PyQt6 Installation Issue

## Problem
When running `python -m src.gui`, you encountered the error:
```
TypeError: NoneType takes no arguments
```

This was caused by PyQt6 not being installed, which made `QWidget` and other Qt classes become `None`, causing Python to fail when trying to define classes that inherit from them.

## Solution Applied

### Code Changes in `src/gui.py`

1. **Fixed stub classes** (lines 41-53):
   - Changed stub classes from `None` to actual `_QtStub` class instances
   - This allows the module to import successfully even when PyQt6 is not installed
   - Classes can now inherit from the stubs without causing `TypeError`

2. **Improved error message** (lines 689-708):
   - Added a clear, detailed error message when PyQt6 is missing
   - Provides multiple installation options for different environments

### The Fix

**Before:**
```python
# This caused TypeError when PyQt6 was not installed
QWidget = None  # Can't inherit from None!
```

**After:**
```python
# Stub class that can be inherited from
class _QtStub:
    def __init__(self, *args, **kwargs):
        pass
    def __call__(self, *args, **kwargs):
        return self
    def __getattr__(self, name):
        return lambda *args, **kwargs: None

QWidget = _QtStub  # Can inherit from this!
```

## What You Need To Do

To fix the error on your Windows machine, install PyQt6 and pyqtgraph:

### Option 1: Install GUI packages only (Recommended)
```bash
pip install PyQt6 pyqtgraph
```

### Option 2: Install all project requirements
```bash
pip install -r requirements.txt
```

### Option 3: Using Conda (if you're using Anaconda)
```bash
conda install -c conda-forge pyqt
pip install pyqtgraph
```

## After Installation

Once PyQt6 is installed, run the GUI again:
```bash
python -m src.gui
```

The application should now launch successfully with a modern GUI interface.

## Verification

After installing PyQt6, you can verify it's working by running:
```bash
python -c "from PyQt6.QtWidgets import QApplication; print('PyQt6 is installed correctly!')"
```

## Notes

- The fix ensures the module can be imported without errors even when PyQt6 is missing
- You'll get a clear, helpful error message with installation instructions
- No more confusing `TypeError: NoneType takes no arguments` errors
- The application requires PyQt6 to run, so installation is necessary for the GUI to work

## Related Files
- `src/gui.py` - Main GUI module (fixed)
- `requirements.txt` - Contains PyQt6>=6.5.0 requirement
- `src/plotting.py` - Visualization module (also uses PyQtGraph)
