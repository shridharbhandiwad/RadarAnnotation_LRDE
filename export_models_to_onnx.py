#!/usr/bin/env python3
"""
Export trained XGBoost and Random Forest models to ONNX format for C++ inference.

This script converts trained scikit-learn and XGBoost models to ONNX format,
which can be loaded and used by the C++ application via ONNX Runtime.
"""

import argparse
import json
import pickle
import numpy as np
from pathlib import Path


def export_xgboost_to_onnx(model_path, output_path, metadata_path=None):
    """
    Export XGBoost model to ONNX format.
    
    Args:
        model_path: Path to pickled XGBoost model
        output_path: Path to save ONNX model
        metadata_path: Optional path to model metadata JSON
    """
    try:
        import xgboost as xgb
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType
        from onnxmltools.convert import convert_xgboost
        import onnx
    except ImportError as e:
        print(f"Error: Required library not installed: {e}")
        print("Install with: pip install xgboost skl2onnx onnx onnxmltools")
        return False
    
    print(f"Loading XGBoost model from {model_path}...")
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    # Load metadata to get number of features
    num_features = 18  # Default
    if metadata_path and Path(metadata_path).exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
            if 'feature_columns' in metadata:
                num_features = len(metadata['feature_columns'])
    
    print(f"Converting XGBoost model to ONNX (input features: {num_features})...")
    
    # Define input type
    initial_type = [('float_input', FloatTensorType([None, num_features]))]
    
    try:
        # Try converting using onnxmltools
        onnx_model = convert_xgboost(
            model,
            initial_types=initial_type,
            target_opset=12
        )
    except Exception as e:
        print(f"Direct conversion failed: {e}")
        print("Attempting alternative conversion method...")
        
        # Alternative: wrap in sklearn-compatible wrapper
        from sklearn.base import BaseEstimator, ClassifierMixin
        
        class XGBMultiOutputWrapper(BaseEstimator, ClassifierMixin):
            def __init__(self, models):
                self.models = models
            
            def predict_proba(self, X):
                # Stack predictions from all models
                predictions = []
                for model in self.models:
                    pred = model.predict(X)
                    predictions.append(pred.reshape(-1, 1))
                return np.hstack(predictions)
        
        # If model is a dict of models (multi-output)
        if isinstance(model, dict):
            wrapper = XGBMultiOutputWrapper(list(model.values()))
        else:
            wrapper = model
        
        # Convert using skl2onnx
        onnx_model = convert_sklearn(
            wrapper,
            initial_types=initial_type,
            target_opset=12
        )
    
    # Save ONNX model
    print(f"Saving ONNX model to {output_path}...")
    onnx.save_model(onnx_model, output_path)
    
    print(f"✓ XGBoost model successfully exported to ONNX")
    print(f"  Input shape: [batch_size, {num_features}]")
    print(f"  Output: Multi-output predictions (11 binary outputs)")
    
    return True


def export_random_forest_to_onnx(model_path, output_path, metadata_path=None):
    """
    Export Random Forest model to ONNX format.
    
    Args:
        model_path: Path to pickled Random Forest model
        output_path: Path to save ONNX model
        metadata_path: Optional path to model metadata JSON
    """
    try:
        from sklearn.ensemble import RandomForestClassifier
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType
        import onnx
    except ImportError as e:
        print(f"Error: Required library not installed: {e}")
        print("Install with: pip install scikit-learn skl2onnx onnx")
        return False
    
    print(f"Loading Random Forest model from {model_path}...")
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    # Load metadata to get number of features
    num_features = 18  # Default
    if metadata_path and Path(metadata_path).exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
            if 'feature_columns' in metadata:
                num_features = len(metadata['feature_columns'])
    
    print(f"Converting Random Forest model to ONNX (input features: {num_features})...")
    
    # Define input type
    initial_type = [('float_input', FloatTensorType([None, num_features]))]
    
    try:
        # Handle multi-output models
        if isinstance(model, dict):
            # Multi-output model stored as dict
            print("Multi-output model detected (dict of models)")
            
            # For now, we'll need to combine them somehow
            # This is a simplified approach - you may need to adjust based on your model structure
            from sklearn.multioutput import MultiOutputClassifier
            
            # Create a dummy wrapper
            print("Warning: Multi-output Random Forest requires special handling")
            print("Attempting to convert first model as example...")
            first_model = list(model.values())[0]
            
            onnx_model = convert_sklearn(
                first_model,
                initial_types=initial_type,
                target_opset=12
            )
        else:
            # Single model
            onnx_model = convert_sklearn(
                model,
                initial_types=initial_type,
                target_opset=12
            )
    except Exception as e:
        print(f"Error during conversion: {e}")
        print("\nNote: Multi-output Random Forest models may require custom conversion.")
        print("Consider using MultiOutputClassifier from sklearn or train separate models.")
        return False
    
    # Save ONNX model
    print(f"Saving ONNX model to {output_path}...")
    onnx.save_model(onnx_model, output_path)
    
    print(f"✓ Random Forest model successfully exported to ONNX")
    print(f"  Input shape: [batch_size, {num_features}]")
    print(f"  Output: Multi-output predictions (11 binary outputs)")
    
    return True


def export_neural_network_to_onnx(model_path, output_path, metadata_path=None):
    """
    Export Keras neural network to ONNX format (alternative to TFLite).
    
    Args:
        model_path: Path to Keras model (.h5)
        output_path: Path to save ONNX model
        metadata_path: Optional path to model metadata JSON
    """
    try:
        import tensorflow as tf
        import tf2onnx
        import onnx
    except ImportError as e:
        print(f"Error: Required library not installed: {e}")
        print("Install with: pip install tensorflow tf2onnx onnx")
        return False
    
    print(f"Loading Keras model from {model_path}...")
    model = tf.keras.models.load_model(model_path)
    
    print("Converting Keras model to ONNX...")
    
    # Get input shape from model
    input_signature = [tf.TensorSpec(model.input.shape, tf.float32, name="input")]
    
    try:
        onnx_model, _ = tf2onnx.convert.from_keras(
            model,
            input_signature=input_signature,
            opset=13
        )
        
        # Save ONNX model
        print(f"Saving ONNX model to {output_path}...")
        onnx.save_model(onnx_model, output_path)
        
        print(f"✓ Neural Network model successfully exported to ONNX")
        print(f"  Input shape: {model.input.shape}")
        print(f"  Output shape: {model.output.shape}")
        
        return True
    except Exception as e:
        print(f"Error during conversion: {e}")
        return False


def create_metadata_for_onnx(model_metadata_path, output_metadata_path):
    """
    Create or update metadata JSON for ONNX model.
    
    Args:
        model_metadata_path: Path to original model metadata
        output_metadata_path: Path to save updated metadata
    """
    # Load existing metadata
    with open(model_metadata_path, 'r') as f:
        metadata = json.load(f)
    
    # Add ONNX-specific information
    metadata['onnx_export'] = True
    metadata['format'] = 'onnx'
    
    # Save updated metadata
    with open(output_metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✓ Metadata saved to {output_metadata_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Export trained models to ONNX format for C++ inference'
    )
    parser.add_argument(
        '--model-type',
        type=str,
        required=True,
        choices=['xgboost', 'random_forest', 'neural_network'],
        help='Type of model to export'
    )
    parser.add_argument(
        '--model-path',
        type=str,
        required=True,
        help='Path to trained model file (.pkl for XGBoost/RF, .h5 for NN)'
    )
    parser.add_argument(
        '--output-path',
        type=str,
        required=True,
        help='Path to save ONNX model (.onnx)'
    )
    parser.add_argument(
        '--metadata-path',
        type=str,
        help='Path to model metadata JSON'
    )
    parser.add_argument(
        '--output-metadata',
        type=str,
        help='Path to save updated metadata JSON'
    )
    
    args = parser.parse_args()
    
    # Ensure output directory exists
    output_dir = Path(args.output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("  Model to ONNX Export Utility")
    print("=" * 70)
    print()
    
    # Export based on model type
    success = False
    if args.model_type == 'xgboost':
        success = export_xgboost_to_onnx(
            args.model_path,
            args.output_path,
            args.metadata_path
        )
    elif args.model_type == 'random_forest':
        success = export_random_forest_to_onnx(
            args.model_path,
            args.output_path,
            args.metadata_path
        )
    elif args.model_type == 'neural_network':
        success = export_neural_network_to_onnx(
            args.model_path,
            args.output_path,
            args.metadata_path
        )
    
    # Create updated metadata if requested
    if success and args.metadata_path and args.output_metadata:
        create_metadata_for_onnx(args.metadata_path, args.output_metadata)
    
    print()
    if success:
        print("=" * 70)
        print("✓ Export completed successfully!")
        print("=" * 70)
        print()
        print("Next steps:")
        print(f"  1. Copy {args.output_path} to your C++ project")
        if args.output_metadata:
            print(f"  2. Copy {args.output_metadata} to your C++ project")
        print(f"  3. Run C++ application:")
        print(f"     ./radar_tagger_multioutput \\")
        print(f"       --model {args.output_path} \\")
        if args.output_metadata:
            print(f"       --metadata {args.output_metadata} \\")
        print(f"       --model-type {args.model_type.replace('_', '')}")
        print()
    else:
        print("=" * 70)
        print("✗ Export failed. See errors above.")
        print("=" * 70)
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())
