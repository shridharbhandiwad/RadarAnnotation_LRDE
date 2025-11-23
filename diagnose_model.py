#!/usr/bin/env python3
"""Diagnose Model File Issues

This script helps identify problems with pickled model files that might
prevent them from being loaded or converted.
"""

import sys
import pickle
import os
from pathlib import Path

def diagnose_model(model_path: str):
    """Diagnose a pickled model file
    
    Args:
        model_path: Path to the .pkl model file
    """
    print("=" * 70)
    print("MODEL FILE DIAGNOSTIC TOOL")
    print("=" * 70)
    print()
    
    # Check file exists
    if not os.path.exists(model_path):
        print(f"❌ ERROR: File not found: {model_path}")
        print()
        print("Please check:")
        print("  1. The file path is correct")
        print("  2. You have permission to read the file")
        print("  3. The file hasn't been moved or deleted")
        return 1
    
    print(f"✓ File exists: {model_path}")
    
    # Check file size
    file_size = os.path.getsize(model_path)
    print(f"✓ File size: {file_size:,} bytes ({file_size / 1024 / 1024:.2f} MB)")
    
    if file_size < 1000:
        print("  ⚠️  Warning: File is very small, may be corrupted or empty")
    
    print()
    
    # Try to get file metadata
    try:
        import pickle
        print("Attempting to analyze pickle structure...")
        
        # Add project directory to path
        script_dir = Path(__file__).parent
        if script_dir not in sys.path:
            sys.path.insert(0, str(script_dir))
        
        # Try loading with pickle
        try:
            with open(model_path, 'rb') as f:
                # Peek at the pickle without fully loading
                unpickler = pickle.Unpickler(f)
                
                # Try to get the first object
                try:
                    data = unpickler.load()
                    print("✓ Successfully loaded with pickle")
                    
                    # Analyze structure
                    print()
                    print("Model Structure:")
                    print("-" * 70)
                    
                    if isinstance(data, dict):
                        print(f"✓ Model is a dictionary with {len(data)} keys")
                        print(f"  Keys: {list(data.keys())}")
                        print()
                        
                        # Check for expected keys
                        expected_keys = ['models', 'scaler', 'adapter', 'output_tag_names', 'params']
                        for key in expected_keys:
                            if key in data:
                                print(f"  ✓ Found '{key}'")
                                
                                # Get more details
                                if key == 'models' and isinstance(data[key], dict):
                                    print(f"    - Number of models: {len(data[key])}")
                                    print(f"    - Model names: {list(data[key].keys())}")
                                elif key == 'output_tag_names' and isinstance(data[key], list):
                                    print(f"    - Number of tags: {len(data[key])}")
                                    print(f"    - Tags: {data[key]}")
                                elif key == 'scaler':
                                    print(f"    - Type: {type(data[key]).__name__}")
                                elif key == 'adapter':
                                    print(f"    - Type: {type(data[key]).__name__}")
                            else:
                                print(f"  ℹ️  Missing '{key}' (may not be required)")
                        
                        print()
                        
                        # Check for model type
                        if 'models' in data and isinstance(data['models'], dict):
                            # Multi-output model
                            print("✓ Detected: Multi-Output Model")
                            print(f"  - {len(data['models'])} output tags")
                            
                            # Check first model type
                            first_model_key = list(data['models'].keys())[0]
                            first_model = data['models'][first_model_key]
                            model_type = type(first_model).__name__
                            print(f"  - Model type: {model_type}")
                            
                            if 'XGB' in model_type or 'Booster' in model_type:
                                print("  - Algorithm: XGBoost (Gradient Boosting)")
                            elif 'Forest' in model_type:
                                print("  - Algorithm: Random Forest")
                            else:
                                print(f"  - Algorithm: {model_type}")
                                
                        elif 'model' in data:
                            # Single-output model
                            print("✓ Detected: Single-Output Model")
                            model_type = type(data['model']).__name__
                            print(f"  - Model type: {model_type}")
                        
                    else:
                        print(f"⚠️  Unexpected structure: {type(data)}")
                        print("   Expected a dictionary")
                    
                    print()
                    print("=" * 70)
                    print("DIAGNOSIS: ✓ Model file appears valid")
                    print("=" * 70)
                    print()
                    print("You can convert this model using:")
                    print(f"  python convert_model_to_tflite.py \\")
                    print(f"    --model-type xgboost \\")
                    print(f"    --model-path \"{model_path}\" \\")
                    print(f"    --output-dir cpp_models")
                    
                    return 0
                    
                except ModuleNotFoundError as e:
                    print(f"❌ ERROR: Missing module during unpickling: {e}")
                    print()
                    print("This usually means the model references custom classes.")
                    print()
                    print("SOLUTION:")
                    print("  1. Make sure you're running from the project directory")
                    print("  2. Ensure the 'src' directory exists with all Python files")
                    print("  3. Try using the updated convert_model_to_tflite.py script")
                    print("     which includes automatic module import recovery")
                    print()
                    print("Try running:")
                    print("  python convert_model_to_tflite.py \\")
                    print("    --model-type xgboost \\")
                    print(f"    --model-path \"{model_path}\" \\")
                    print("    --output-dir cpp_models")
                    
                    return 1
                    
                except Exception as e:
                    print(f"❌ ERROR: Failed to load pickle: {e}")
                    print()
                    print("Possible causes:")
                    print("  1. File is corrupted")
                    print("  2. File was created with incompatible Python version")
                    print("  3. File requires specific modules to be installed")
                    print()
                    print("Try:")
                    print("  1. Re-train the model with the current code")
                    print("  2. Check if all dependencies are installed")
                    print("  3. Use 'pip install joblib scikit-learn xgboost'")
                    
                    return 1
                    
        except Exception as e:
            print(f"❌ ERROR: Cannot open file: {e}")
            return 1
            
    except Exception as e:
        print(f"❌ ERROR: Diagnostic failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


def main():
    """Main entry point"""
    if len(sys.argv) < 2:
        print("Usage: python diagnose_model.py <model_path>")
        print()
        print("Example:")
        print('  python diagnose_model.py "output/models/xgboost_multi_output/model.pkl"')
        return 1
    
    model_path = sys.argv[1]
    return diagnose_model(model_path)


if __name__ == '__main__':
    sys.exit(main())
