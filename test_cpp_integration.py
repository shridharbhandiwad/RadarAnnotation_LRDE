#!/usr/bin/env python3
"""Test script to verify C++ deployment integration in GUI"""

import sys
import os
from pathlib import Path

def test_gui_structure():
    """Test that GUI structure is correct"""
    print("Testing GUI structure...")
    
    # Parse GUI file and check for CppDeploymentPanel
    with open('src/gui.py', 'r') as f:
        gui_content = f.read()
    
    # Check for CppDeploymentPanel class
    if 'class CppDeploymentPanel' not in gui_content:
        print("✗ CppDeploymentPanel class not found")
        return False
    print("✓ CppDeploymentPanel class found")
    
    # Check for panel methods
    required_methods = [
        'select_model',
        'convert_model',
        'build_cpp_app',
        'run_evaluation',
        'on_conversion_complete',
        'on_build_complete',
        'on_evaluation_complete'
    ]
    
    for method in required_methods:
        if f'def {method}' not in gui_content:
            print(f"✗ Method {method} not found")
            return False
    print(f"✓ All {len(required_methods)} required methods found")
    
    # Check that panel is added to stack
    if 'CppDeploymentPanel()' not in gui_content:
        print("✗ CppDeploymentPanel not added to stack")
        return False
    print("✓ CppDeploymentPanel added to stack")
    
    # Check that menu item exists
    if '⚙️ C++ Deployment' not in gui_content:
        print("✗ C++ Deployment menu item not found")
        return False
    print("✓ C++ Deployment menu item found")
    
    return True


def test_conversion_script():
    """Test that conversion script exists and is valid"""
    print("\nTesting conversion script...")
    
    script_path = Path('convert_model_to_tflite.py')
    if not script_path.exists():
        print("✗ convert_model_to_tflite.py not found")
        return False
    print("✓ Conversion script exists")
    
    # Check for required functions
    with open(script_path, 'r') as f:
        script_content = f.read()
    
    required_functions = [
        'convert_keras_to_tflite',
        'export_metadata',
        'create_test_data'
    ]
    
    for func in required_functions:
        if f'def {func}' not in script_content:
            print(f"✗ Function {func} not found in conversion script")
            return False
    print(f"✓ All {len(required_functions)} required functions found")
    
    return True


def test_cpp_infrastructure():
    """Test that C++ infrastructure is in place"""
    print("\nTesting C++ infrastructure...")
    
    # Check for C++ directory
    cpp_dir = Path('cpp_inference')
    if not cpp_dir.exists():
        print("✗ cpp_inference directory not found")
        return False
    print("✓ cpp_inference directory exists")
    
    # Check for required C++ files
    required_files = [
        'CMakeLists.txt',
        'radar_tagger.h',
        'radar_tagger.cpp',
        'main.cpp',
        'README.md'
    ]
    
    for file in required_files:
        file_path = cpp_dir / file
        if not file_path.exists():
            print(f"✗ Required file {file} not found")
            return False
    print(f"✓ All {len(required_files)} required C++ files found")
    
    return True


def test_workflow_readiness():
    """Test that the complete workflow is ready"""
    print("\nTesting workflow readiness...")
    
    # Check for model directories
    output_dir = Path('output')
    if not output_dir.exists():
        print("⚠ output directory doesn't exist yet (expected if no models trained)")
    else:
        print("✓ output directory exists")
    
    # Check for cpp_models directory structure
    cpp_models_dir = Path('cpp_models')
    if not cpp_models_dir.exists():
        print("⚠ cpp_models directory doesn't exist yet (will be created on first conversion)")
    else:
        print("✓ cpp_models directory exists")
    
    return True


def main():
    """Run all tests"""
    print("="*60)
    print("C++ Integration Test Suite")
    print("="*60)
    
    tests = [
        ("GUI Structure", test_gui_structure),
        ("Conversion Script", test_conversion_script),
        ("C++ Infrastructure", test_cpp_infrastructure),
        ("Workflow Readiness", test_workflow_readiness)
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"✗ Test {name} failed with exception: {e}")
            results.append((name, False))
    
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {name}")
    
    all_passed = all(r[1] for r in results)
    
    print("\n" + "="*60)
    if all_passed:
        print("✓ All tests passed!")
        print("\nThe C++ deployment integration is ready to use.")
        print("\nUsage:")
        print("1. Launch the GUI: python3 run.py")
        print("2. Navigate to '⚙️ C++ Deployment' in the left menu")
        print("3. Follow the 3-step process:")
        print("   - Convert your Keras model to TensorFlow Lite")
        print("   - Build the C++ inference application")
        print("   - Run evaluation and benchmarking")
        return 0
    else:
        print("✗ Some tests failed. Please review the output above.")
        return 1


if __name__ == '__main__':
    sys.exit(main())
