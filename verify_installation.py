#!/usr/bin/env python3
"""Verify Radar Annotation Application installation"""

import sys
import importlib
from pathlib import Path

def check_module(module_name, package_name=None):
    """Check if a module can be imported"""
    try:
        importlib.import_module(module_name)
        print(f"  ✓ {package_name or module_name}")
        return True
    except ImportError:
        print(f"  ✗ {package_name or module_name} - MISSING")
        return False

def main():
    print("=" * 60)
    print("Radar Data Annotation Application - Installation Verification")
    print("=" * 60)
    print()
    
    # Check Python version
    print("1. Checking Python version...")
    version = sys.version_info
    if version.major == 3 and version.minor >= 10:
        print(f"  ✓ Python {version.major}.{version.minor}.{version.micro}")
    else:
        print(f"  ✗ Python {version.major}.{version.minor} (requires 3.10+)")
        return False
    print()
    
    # Check required packages
    print("2. Checking required packages...")
    required = [
        ('numpy', 'numpy'),
        ('pandas', 'pandas'),
        ('scipy', 'scipy'),
        ('PyQt6', 'PyQt6'),
        ('pyqtgraph', 'pyqtgraph'),
        ('openpyxl', 'openpyxl'),
        ('sklearn', 'scikit-learn'),
        ('xgboost', 'xgboost'),
        ('matplotlib', 'matplotlib'),
        ('joblib', 'joblib'),
    ]
    
    optional = [
        ('tensorflow', 'tensorflow (optional for LSTM)'),
        ('docx', 'python-docx (optional)'),
    ]
    
    all_ok = True
    for module, name in required:
        if not check_module(module, name):
            all_ok = False
    
    print("\n3. Checking optional packages...")
    for module, name in optional:
        check_module(module, name)
    
    print()
    
    # Check project structure
    print("4. Checking project structure...")
    required_files = [
        'src/__init__.py',
        'src/config.py',
        'src/utils.py',
        'src/data_engine.py',
        'src/autolabel_engine.py',
        'src/ai_engine.py',
        'src/report_engine.py',
        'src/sim_engine.py',
        'src/plotting.py',
        'src/gui.py',
        'tests/__init__.py',
        'config/default_config.json',
        'requirements.txt',
        'README.md',
    ]
    
    for file_path in required_files:
        if Path(file_path).exists():
            print(f"  ✓ {file_path}")
        else:
            print(f"  ✗ {file_path} - MISSING")
            all_ok = False
    
    print()
    
    # Check if src modules can be imported
    print("5. Checking application modules...")
    app_modules = [
        'src.config',
        'src.utils',
        'src.data_engine',
        'src.autolabel_engine',
        'src.sim_engine',
    ]
    
    for module in app_modules:
        if check_module(module):
            pass
        else:
            all_ok = False
    
    print()
    print("=" * 60)
    
    if all_ok:
        print("✓ Installation verification PASSED")
        print()
        print("Ready to use! Run the application with:")
        print("  python -m src.gui")
        print()
        print("Or run the demo:")
        print("  bash demo.sh    (Linux/Mac)")
        print("  demo.bat        (Windows)")
    else:
        print("✗ Installation verification FAILED")
        print()
        print("Please install missing packages:")
        print("  pip install -r requirements.txt")
    
    print("=" * 60)
    
    return all_ok

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
