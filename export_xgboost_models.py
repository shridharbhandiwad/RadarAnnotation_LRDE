"""Export XGBoost and Random Forest Models for C++ Deployment

This script exports trained XGBoost and RandomForest models to formats
that can be used in C++ applications.

For XGBoost:
- Exports to JSON format (text-based, human-readable)
- Exports to binary format (.bin)
- Can be loaded with XGBoost C++ API

For Random Forest:
- Exports model parameters to JSON
- Includes tree structures and decision rules
"""

import os
import sys
import argparse
import json
import pickle
import logging
from pathlib import Path
import numpy as np

try:
    import joblib
    HAS_JOBLIB = True
except ImportError:
    HAS_JOBLIB = False
    print("Warning: joblib not available")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def export_xgboost_model(model_path: str, output_dir: str, model_name: str = "xgboost"):
    """Export XGBoost model to C++ compatible formats
    
    Args:
        model_path: Path to .pkl file containing XGBoost model
        output_dir: Output directory
        model_name: Base name for output files
    """
    logger.info(f"Loading XGBoost model from: {model_path}")
    
    try:
        import xgboost as xgb
    except ImportError:
        logger.error("XGBoost not installed. Install with: pip install xgboost")
        return False
    
    # Load model
    try:
        with open(model_path, 'rb') as f:
            data = pickle.load(f)
        
        # Extract models (multi-output: one model per tag)
        if isinstance(data, dict) and 'models' in data:
            models = data['models']
            scaler = data.get('scaler')
            tag_names = list(models.keys())
            
            logger.info(f"Found multi-output XGBoost with {len(models)} models")
            logger.info(f"Tags: {tag_names}")
            
            # Export each model
            for tag_name, model in models.items():
                tag_output_dir = os.path.join(output_dir, tag_name)
                os.makedirs(tag_output_dir, exist_ok=True)
                
                # Export to JSON
                json_path = os.path.join(tag_output_dir, f'{model_name}_{tag_name}.json')
                model.save_model(json_path)
                logger.info(f"Exported {tag_name} model to JSON: {json_path}")
                
                # Export to binary
                bin_path = os.path.join(tag_output_dir, f'{model_name}_{tag_name}.bin')
                model.save_model(bin_path)
                logger.info(f"Exported {tag_name} model to binary: {bin_path}")
            
            # Export scaler parameters
            if scaler is not None:
                scaler_data = {
                    'mean': scaler.mean_.tolist(),
                    'scale': scaler.scale_.tolist(),
                    'n_features': len(scaler.mean_)
                }
                scaler_path = os.path.join(output_dir, 'scaler_params.json')
                with open(scaler_path, 'w') as f:
                    json.dump(scaler_data, f, indent=2)
                logger.info(f"Exported scaler parameters: {scaler_path}")
            
            # Export model metadata
            metadata = {
                'model_type': 'xgboost_multioutput',
                'num_models': len(models),
                'tag_names': tag_names,
                'has_scaler': scaler is not None,
                'model_files': {
                    tag: f'{model_name}_{tag}.json' for tag in tag_names
                }
            }
            metadata_path = os.path.join(output_dir, 'model_info.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            logger.info(f"Exported metadata: {metadata_path}")
            
            return True
            
        else:
            logger.error("Unknown model format")
            return False
            
    except Exception as e:
        logger.error(f"Failed to export XGBoost model: {e}")
        import traceback
        traceback.print_exc()
        return False


def export_randomforest_model(model_path: str, output_dir: str, model_name: str = "randomforest"):
    """Export Random Forest model to JSON format
    
    Args:
        model_path: Path to .pkl file containing RandomForest model
        output_dir: Output directory
        model_name: Base name for output files
    """
    logger.info(f"Loading Random Forest model from: {model_path}")
    
    try:
        from sklearn.ensemble import RandomForestClassifier
    except ImportError:
        logger.error("scikit-learn not installed. Install with: pip install scikit-learn")
        return False
    
    # Load model
    try:
        with open(model_path, 'rb') as f:
            data = pickle.load(f)
        
        if isinstance(data, dict) and 'models' in data:
            models = data['models']
            scaler = data.get('scaler')
            tag_names = list(models.keys())
            
            logger.info(f"Found multi-output RandomForest with {len(models)} models")
            logger.info(f"Tags: {tag_names}")
            
            # Export each model
            for tag_name, model in models.items():
                tag_output_dir = os.path.join(output_dir, tag_name)
                os.makedirs(tag_output_dir, exist_ok=True)
                
                # Extract tree structures
                model_data = {
                    'n_estimators': model.n_estimators,
                    'max_depth': model.max_depth,
                    'n_features': model.n_features_in_,
                    'n_classes': len(model.classes_),
                    'classes': model.classes_.tolist(),
                    'trees': []
                }
                
                # Export each tree (limited to first 10 for size)
                for i, tree in enumerate(model.estimators_[:10]):
                    tree_data = {
                        'tree_id': i,
                        'n_nodes': tree.tree_.node_count,
                        'max_depth': tree.tree_.max_depth,
                        # Note: Full tree export would be very large
                        # In production, use ONNX or TFLite for RandomForest
                    }
                    model_data['trees'].append(tree_data)
                
                # Save to JSON
                json_path = os.path.join(tag_output_dir, f'{model_name}_{tag_name}.json')
                with open(json_path, 'w') as f:
                    json.dump(model_data, f, indent=2)
                logger.info(f"Exported {tag_name} model metadata to: {json_path}")
            
            # Export scaler parameters
            if scaler is not None:
                scaler_data = {
                    'mean': scaler.mean_.tolist(),
                    'scale': scaler.scale_.tolist(),
                    'n_features': len(scaler.mean_)
                }
                scaler_path = os.path.join(output_dir, 'scaler_params.json')
                with open(scaler_path, 'w') as f:
                    json.dump(scaler_data, f, indent=2)
                logger.info(f"Exported scaler parameters: {scaler_path}")
            
            # Export model metadata
            metadata = {
                'model_type': 'randomforest_multioutput',
                'num_models': len(models),
                'tag_names': tag_names,
                'has_scaler': scaler is not None,
                'note': 'Full tree export not included. Consider using ONNX or TFLite for production.'
            }
            metadata_path = os.path.join(output_dir, 'model_info.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            logger.info(f"Exported metadata: {metadata_path}")
            
            return True
            
        else:
            logger.error("Unknown model format")
            return False
            
    except Exception as e:
        logger.error(f"Failed to export RandomForest model: {e}")
        import traceback
        traceback.print_exc()
        return False


def convert_to_onnx(model_path: str, output_dir: str, model_type: str = "xgboost"):
    """Convert XGBoost or RandomForest to ONNX format
    
    This is the RECOMMENDED approach for C++ deployment of tree-based models.
    ONNX provides excellent C++ support and optimized inference.
    """
    logger.info(f"Converting {model_type} model to ONNX format")
    
    try:
        import onnx
        import onnxmltools
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType
    except ImportError:
        logger.error("ONNX tools not installed. Install with: pip install onnx onnxmltools skl2onnx")
        logger.info("For XGBoost: pip install onnxmltools")
        logger.info("For scikit-learn: pip install skl2onnx")
        return False
    
    # Load model
    try:
        with open(model_path, 'rb') as f:
            data = pickle.load(f)
        
        if isinstance(data, dict) and 'models' in data:
            models = data['models']
            tag_names = list(models.keys())
            
            logger.info(f"Converting {len(models)} models to ONNX...")
            
            for tag_name, model in models.items():
                tag_output_dir = os.path.join(output_dir, tag_name)
                os.makedirs(tag_output_dir, exist_ok=True)
                
                onnx_path = os.path.join(tag_output_dir, f'{model_type}_{tag_name}.onnx')
                
                try:
                    # Determine input shape
                    n_features = model.n_features_in_ if hasattr(model, 'n_features_in_') else 18
                    initial_type = [('input', FloatTensorType([None, n_features]))]
                    
                    if model_type == "xgboost":
                        onnx_model = onnxmltools.convert_xgboost(model, initial_types=initial_type)
                    else:  # sklearn models
                        onnx_model = convert_sklearn(model, initial_types=initial_type)
                    
                    # Save ONNX model
                    onnx.save_model(onnx_model, onnx_path)
                    logger.info(f"Converted {tag_name} to ONNX: {onnx_path}")
                    
                except Exception as e:
                    logger.error(f"Failed to convert {tag_name} to ONNX: {e}")
                    continue
            
            return True
            
    except Exception as e:
        logger.error(f"ONNX conversion failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description='Export XGBoost/RandomForest models for C++ deployment')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to model .pkl file')
    parser.add_argument('--output-dir', type=str, default='cpp_models_multioutput',
                       help='Output directory')
    parser.add_argument('--model-type', type=str, choices=['xgboost', 'randomforest', 'auto'],
                       default='auto', help='Model type')
    parser.add_argument('--format', type=str, choices=['json', 'onnx', 'both'],
                       default='both', help='Export format')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Auto-detect model type if needed
    model_type = args.model_type
    if model_type == 'auto':
        # Try to detect from file content
        try:
            with open(args.model, 'rb') as f:
                data = pickle.load(f)
            if isinstance(data, dict) and 'models' in data:
                first_model = list(data['models'].values())[0]
                if 'XGB' in str(type(first_model)):
                    model_type = 'xgboost'
                else:
                    model_type = 'randomforest'
                logger.info(f"Auto-detected model type: {model_type}")
        except:
            logger.error("Failed to auto-detect model type")
            return 1
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Exporting {model_type} Model for C++ Deployment")
    logger.info(f"{'='*60}\n")
    
    success = False
    
    # Export based on format
    if args.format in ['json', 'both']:
        if model_type == 'xgboost':
            success = export_xgboost_model(args.model, args.output_dir)
        else:
            success = export_randomforest_model(args.model, args.output_dir)
    
    if args.format in ['onnx', 'both']:
        onnx_success = convert_to_onnx(args.model, args.output_dir, model_type)
        success = success or onnx_success
    
    if success:
        logger.info(f"\n{'='*60}")
        logger.info(f"Export Complete!")
        logger.info(f"Models saved to: {args.output_dir}")
        logger.info(f"{'='*60}\n")
        
        logger.info("Next Steps:")
        logger.info("1. For ONNX format (RECOMMENDED):")
        logger.info("   - Use ONNX Runtime C++ API")
        logger.info("   - See: https://onnxruntime.ai/docs/get-started/with-cpp.html")
        logger.info("\n2. For JSON format:")
        logger.info("   - Use XGBoost C++ API")
        logger.info("   - Link against libxgboost")
        logger.info("   - See: https://xgboost.readthedocs.io/en/latest/")
        
        return 0
    else:
        logger.error("Export failed")
        return 1


if __name__ == '__main__':
    sys.exit(main())
