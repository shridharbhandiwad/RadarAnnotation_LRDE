"""Convert Models to Deployment Formats for C++

This script converts trained models to formats suitable for C++ deployment:
- Keras models (LSTM/Transformer) -> TensorFlow Lite
- XGBoost models -> ONNX format (recommended for tree-based models)
"""

import os
import sys
import argparse
import json
import pickle
import logging
from pathlib import Path
import numpy as np

# Add src directory to path for loading pickled models that reference src modules
_script_dir = Path(__file__).parent
if _script_dir not in sys.path:
    sys.path.insert(0, str(_script_dir))

try:
    import joblib
    HAS_JOBLIB = True
except ImportError:
    HAS_JOBLIB = False

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    HAS_TF = True
except ImportError:
    HAS_TF = False

try:
    import onnx
    import onnxmltools
    from onnxmltools.convert.common.data_types import FloatTensorType
    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# Define custom layers that were used in model training
class TransformerBlock(keras.layers.Layer):
    """Transformer block with multi-head attention"""
    
    def __init__(self, d_model: int, num_heads: int, ff_dim: int, dropout: float = 0.1, dropout_rate: float = None, **kwargs):
        super(TransformerBlock, self).__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        # Handle both 'dropout' and 'dropout_rate' parameters for compatibility
        self.dropout_rate = dropout_rate if dropout_rate is not None else dropout
        
    def build(self, input_shape):
        """Build the layer"""
        self.att = layers.MultiHeadAttention(
            num_heads=self.num_heads, 
            key_dim=self.d_model // self.num_heads
        )
        self.ffn = keras.Sequential([
            layers.Dense(self.ff_dim, activation='relu'),
            layers.Dense(self.d_model)
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(self.dropout_rate)
        self.dropout2 = layers.Dropout(self.dropout_rate)
        super(TransformerBlock, self).build(input_shape)
        
    def call(self, inputs, training=False):
        """Forward pass"""
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)
        
        return out2
    
    def get_config(self):
        """Get layer configuration"""
        config = super(TransformerBlock, self).get_config()
        config.update({
            'd_model': self.d_model,
            'num_heads': self.num_heads,
            'ff_dim': self.ff_dim,
            'dropout_rate': self.dropout_rate
        })
        return config


def convert_keras_to_tflite(model_path: str, output_path: str, model_name: str = "model"):
    """Convert a Keras model to TensorFlow Lite format
    
    Args:
        model_path: Path to .h5 Keras model file
        output_path: Output path for .tflite file
        model_name: Name for the model
    """
    logger.info(f"Loading Keras model from: {model_path}")
    
    # Load Keras model with custom objects
    custom_objects = {'TransformerBlock': TransformerBlock}
    
    try:
        with keras.utils.custom_object_scope(custom_objects):
            model = keras.models.load_model(model_path, compile=False)
    except Exception as e:
        logger.warning(f"Failed to load with custom objects: {e}")
        logger.info("Trying to load model without custom objects...")
        model = keras.models.load_model(model_path, compile=False)
    
    # Print model summary
    logger.info("Model architecture:")
    model.summary()
    
    # Get input/output shapes
    input_shape = model.input_shape
    output_shape = model.output_shape
    logger.info(f"Input shape: {input_shape}")
    logger.info(f"Output shape: {output_shape}")
    
    # Convert to TFLite
    logger.info("Converting to TensorFlow Lite format...")
    
    try:
        # Create TFLite converter
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        
        # Enable TensorFlow SELECT_TF_OPS for LSTM models
        # This allows use of TensorFlow ops not available in TFLite builtins
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,  # Enable TensorFlow Lite ops
            tf.lite.OpsSet.SELECT_TF_OPS     # Enable TensorFlow ops
        ]
        converter._experimental_lower_tensor_list_ops = False
        
        # Optimize for size and latency
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        # Convert
        tflite_model = converter.convert()
        
        # Save the model
        with open(output_path, 'wb') as f:
            f.write(tflite_model)
            
        logger.info(f"Successfully converted model to: {output_path}")
        logger.info(f"Model size: {len(tflite_model) / 1024:.2f} KB")
        
    except Exception as e:
        logger.error(f"Conversion failed: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    return output_path


def export_metadata(model_dir: str, output_dir: str):
    """Export model metadata for C++ application
    
    Args:
        model_dir: Directory containing model files
        output_dir: Output directory for metadata
    """
    metadata_pkl = os.path.join(model_dir, os.path.basename(model_dir) + '_model_metadata.pkl')
    metrics_json = os.path.join(model_dir, os.path.basename(model_dir) + '_metrics.json')
    
    metadata = {}
    
    # Load pickle metadata if exists
    if os.path.exists(metadata_pkl):
        logger.info(f"Loading metadata from: {metadata_pkl}")
        with open(metadata_pkl, 'rb') as f:
            pkl_data = pickle.load(f)
            
            # Extract relevant information
            if 'scaler' in pkl_data:
                scaler = pkl_data['scaler']
                metadata['scaler_mean'] = scaler.mean_.tolist()
                metadata['scaler_scale'] = scaler.scale_.tolist()
            
            if 'label_encoder' in pkl_data:
                label_encoder = pkl_data['label_encoder']
                metadata['classes'] = label_encoder.classes_.tolist()
                metadata['n_classes'] = len(label_encoder.classes_)
            
            if 'feature_columns' in pkl_data:
                metadata['feature_columns'] = pkl_data['feature_columns']
                
            if 'sequence_length' in pkl_data:
                metadata['sequence_length'] = pkl_data['sequence_length']
    
    # Load metrics if exists
    if os.path.exists(metrics_json):
        logger.info(f"Loading metrics from: {metrics_json}")
        with open(metrics_json, 'r') as f:
            metrics = json.load(f)
            metadata['metrics'] = metrics
    
    # Save combined metadata
    output_file = os.path.join(output_dir, 'model_metadata.json')
    logger.info(f"Saving metadata to: {output_file}")
    with open(output_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return metadata


def create_test_data(output_dir: str, n_samples: int = 10, sequence_length: int = 20, n_features: int = 18):
    """Create test data for C++ application validation
    
    Args:
        output_dir: Output directory
        n_samples: Number of test samples
        sequence_length: Sequence length
        n_features: Number of features
    """
    logger.info(f"Creating {n_samples} test samples...")
    
    # Generate random test data (normalized)
    test_data = np.random.randn(n_samples, sequence_length, n_features).astype(np.float32)
    
    # Save as binary file
    output_file = os.path.join(output_dir, 'test_data.bin')
    test_data.tofile(output_file)
    
    # Also save as CSV for easier inspection
    csv_file = os.path.join(output_dir, 'test_data.csv')
    # Reshape for CSV (flatten sequences)
    test_data_flat = test_data.reshape(n_samples, -1)
    np.savetxt(csv_file, test_data_flat, delimiter=',', fmt='%.6f')
    
    # Save dimensions info
    dims_info = {
        'n_samples': n_samples,
        'sequence_length': sequence_length,
        'n_features': n_features,
        'dtype': 'float32',
        'shape': [n_samples, sequence_length, n_features],
        'total_elements': n_samples * sequence_length * n_features
    }
    
    with open(os.path.join(output_dir, 'test_data_info.json'), 'w') as f:
        json.dump(dims_info, f, indent=2)
    
    logger.info(f"Test data saved to: {output_file}")
    logger.info(f"Test data CSV: {csv_file}")
    logger.info(f"Shape: {test_data.shape}")
    
    return test_data


def convert_xgboost_to_onnx(model_path: str, output_dir: str, model_name: str = "xgboost"):
    """Convert XGBoost model to ONNX format
    
    Args:
        model_path: Path to .pkl file containing XGBoost model
        output_dir: Output directory for ONNX files
        model_name: Base name for output files
    
    Returns:
        List of paths to converted ONNX files
    """
    if not HAS_ONNX:
        logger.error("ONNX tools not installed. Install with:")
        logger.error("  pip install onnx onnxmltools skl2onnx")
        return []
    
    logger.info(f"Loading XGBoost model from: {model_path}")
    
    try:
        import xgboost as xgb
    except ImportError:
        logger.error("XGBoost not installed. Install with: pip install xgboost")
        return []
    
    # Load model - try joblib first (preferred), then pickle
    try:
        # Try joblib first (recommended for sklearn/xgboost models)
        if HAS_JOBLIB:
            try:
                logger.info("Loading model with joblib...")
                data = joblib.load(model_path)
            except ModuleNotFoundError as e:
                # Missing module during unpickling - likely needs src module classes
                logger.warning(f"Module not found during joblib load: {e}")
                logger.info("Attempting to import required src modules...")
                
                # Try importing common src modules that might be needed
                try:
                    from src.multi_output_adapter import MultiOutputDataAdapter
                    from src.label_transformer import LabelTransformer
                    logger.info("Successfully imported src modules, retrying...")
                    data = joblib.load(model_path)
                except ImportError as ie:
                    logger.error(f"Failed to import src modules: {ie}")
                    logger.error("The model file may have been created with a different version of the code.")
                    logger.error("Please ensure the 'src' directory is available and contains all necessary modules.")
                    raise
            except Exception as e:
                logger.warning(f"Failed to load with joblib: {e}")
                logger.info("Trying with standard pickle...")
                with open(model_path, 'rb') as f:
                    data = pickle.load(f)
        else:
            # Joblib not available, use pickle
            logger.info("Joblib not available, using pickle...")
            with open(model_path, 'rb') as f:
                data = pickle.load(f)
        
        # Extract models (multi-output: one model per tag)
        if isinstance(data, dict) and 'models' in data:
            models = data['models']
            scaler = data.get('scaler')
            tag_names = list(models.keys())
            
            logger.info(f"Found multi-output XGBoost with {len(models)} models")
            logger.info(f"Tags: {tag_names}")
            
            onnx_files = []
            json_files = []
            
            # Convert each model
            for tag_name, model in models.items():
                tag_output_dir = os.path.join(output_dir, tag_name)
                os.makedirs(tag_output_dir, exist_ok=True)
                
                onnx_path = os.path.join(tag_output_dir, f'{model_name}_{tag_name}.onnx')
                json_path = os.path.join(tag_output_dir, f'{model_name}_{tag_name}.json')
                
                # Try ONNX conversion first
                onnx_success = False
                try:
                    # Determine input shape
                    n_features = model.n_features_in_ if hasattr(model, 'n_features_in_') else 18
                    initial_type = [('input', FloatTensorType([None, n_features]))]
                    
                    # Get the booster from the XGBoost model
                    # This is more reliable for ONNX conversion
                    booster = model.get_booster()
                    
                    # Convert to ONNX
                    onnx_model = onnxmltools.convert_xgboost(booster, initial_types=initial_type)
                    
                    # Save ONNX model
                    onnx.save_model(onnx_model, onnx_path)
                    logger.info(f"✓ Converted {tag_name} to ONNX: {onnx_path}")
                    
                    # Get file size
                    file_size = os.path.getsize(onnx_path) / 1024
                    logger.info(f"  Model size: {file_size:.2f} KB")
                    
                    onnx_files.append(onnx_path)
                    onnx_success = True
                    
                except Exception as e:
                    logger.warning(f"ONNX conversion failed for {tag_name}: {e}")
                    logger.info(f"  Falling back to JSON format...")
                
                # Fall back to JSON export if ONNX fails
                if not onnx_success:
                    try:
                        model.save_model(json_path)
                        logger.info(f"✓ Exported {tag_name} to JSON: {json_path}")
                        
                        # Get file size
                        file_size = os.path.getsize(json_path) / 1024
                        logger.info(f"  Model size: {file_size:.2f} KB")
                        
                        json_files.append(json_path)
                        
                    except Exception as e:
                        logger.error(f"Failed to export {tag_name} to JSON: {e}")
                        continue
            
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
            model_files = {}
            export_format = 'mixed' if (onnx_files and json_files) else ('onnx' if onnx_files else 'json')
            
            for tag in tag_names:
                if any(tag in path for path in onnx_files):
                    model_files[tag] = f'{tag}/{model_name}_{tag}.onnx'
                else:
                    model_files[tag] = f'{tag}/{model_name}_{tag}.json'
            
            metadata = {
                'model_type': 'xgboost_multioutput',
                'format': export_format,
                'num_models': len(models),
                'tag_names': tag_names,
                'has_scaler': scaler is not None,
                'model_files': model_files,
                'onnx_models': len(onnx_files),
                'json_models': len(json_files)
            }
            metadata_path = os.path.join(output_dir, 'model_metadata.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            logger.info(f"Exported metadata: {metadata_path}")
            
            return onnx_files + json_files
            
        else:
            logger.error("Unknown model format. Expected dict with 'models' key.")
            return []
            
    except Exception as e:
        logger.error(f"Failed to convert XGBoost model: {e}")
        import traceback
        traceback.print_exc()
        return []


def main():
    parser = argparse.ArgumentParser(description='Convert models to formats suitable for C++ deployment')
    parser.add_argument('--model-type', type=str, choices=['lstm', 'transformer', 'both', 'xgboost'],
                       default='lstm', help='Which model to convert')
    parser.add_argument('--model-path', type=str,
                       help='Path to model file (required for xgboost)')
    parser.add_argument('--output-dir', type=str, default='cpp_models',
                       help='Output directory for converted models')
    
    args = parser.parse_args()
    
    # Check TensorFlow requirement for neural network models
    if args.model_type in ['lstm', 'transformer', 'both'] and not HAS_TF:
        logger.error("ERROR: TensorFlow is required for LSTM/Transformer model conversion")
        logger.error("Install with: pip install tensorflow")
        return 1
    
    # Check ONNX requirement for XGBoost models
    if args.model_type == 'xgboost' and not HAS_ONNX:
        logger.error("ERROR: ONNX tools are required for XGBoost model conversion")
        logger.error("Install with: pip install onnx onnxmltools skl2onnx")
        return 1
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Handle XGBoost conversion
    if args.model_type == 'xgboost':
        if not args.model_path:
            logger.error("ERROR: --model-path is required for xgboost model type")
            return 1
        
        if not os.path.exists(args.model_path):
            logger.error(f"ERROR: Model file not found: {args.model_path}")
            return 1
        
        logger.info(f"\n{'='*60}")
        logger.info("Converting XGBoost Model to ONNX Format")
        logger.info(f"{'='*60}")
        logger.info("")
        logger.info("Note: XGBoost models are tree-based and cannot be converted to TensorFlow Lite.")
        logger.info("ONNX format is the recommended deployment format for XGBoost in C++.")
        logger.info("")
        
        model_files = convert_xgboost_to_onnx(args.model_path, args.output_dir)
        
        if model_files:
            logger.info(f"\n{'='*60}")
            logger.info("Conversion Complete!")
            logger.info(f"{'='*60}")
            logger.info(f"Models saved to: {args.output_dir}")
            logger.info(f"Number of models converted: {len(model_files)}")
            
            # Count ONNX vs JSON files
            onnx_count = sum(1 for f in model_files if f.endswith('.onnx'))
            json_count = sum(1 for f in model_files if f.endswith('.json'))
            
            if onnx_count > 0:
                logger.info(f"  - ONNX models: {onnx_count}")
            if json_count > 0:
                logger.info(f"  - JSON models: {json_count}")
            
            logger.info("")
            logger.info("Next Steps:")
            if onnx_count > 0:
                logger.info("For ONNX models:")
                logger.info("  1. Use ONNX Runtime C++ API for inference")
                logger.info("  2. See: https://onnxruntime.ai/docs/get-started/with-cpp.html")
            if json_count > 0:
                logger.info("For JSON models:")
                logger.info("  1. Use XGBoost C++ API to load and run inference")
                logger.info("  2. Link with libxgboost")
                logger.info("  3. See: https://xgboost.readthedocs.io/en/latest/")
            return 0
        else:
            logger.error("Conversion failed")
            return 1
    
    # Handle Keras model conversion (LSTM/Transformer)
    models_to_convert = []
    
    if args.model_type in ['transformer', 'both']:
        transformer_dir = 'output/test_transformer'
        if os.path.exists(transformer_dir):
            models_to_convert.append(('transformer', transformer_dir))
    
    if args.model_type in ['lstm', 'both']:
        lstm_dir = 'output/test_lstm'
        if os.path.exists(lstm_dir):
            models_to_convert.append(('lstm', lstm_dir))
    
    if not models_to_convert:
        logger.error(f"No models found to convert. Check that model directories exist.")
        return 1
    
    # Convert each model
    for model_name, model_dir in models_to_convert:
        logger.info(f"\n{'='*60}")
        logger.info(f"Converting {model_name.upper()} model")
        logger.info(f"{'='*60}")
        
        model_h5 = os.path.join(model_dir, f'{model_name}_model.h5')
        
        if not os.path.exists(model_h5):
            logger.warning(f"Model file not found: {model_h5}")
            continue
        
        # Create model-specific output directory
        model_output_dir = os.path.join(args.output_dir, model_name)
        os.makedirs(model_output_dir, exist_ok=True)
        
        # Convert model
        tflite_path = os.path.join(model_output_dir, f'{model_name}_model.tflite')
        try:
            convert_keras_to_tflite(model_h5, tflite_path, model_name)
        except Exception as e:
            logger.error(f"Failed to convert {model_name} model: {e}")
            continue
        
        # Export metadata
        try:
            metadata = export_metadata(model_dir, model_output_dir)
            logger.info(f"Metadata exported successfully")
            
            # Create test data
            sequence_length = metadata.get('sequence_length', 20)
            n_features = len(metadata.get('feature_columns', [])) or 18
            create_test_data(model_output_dir, n_samples=10, 
                           sequence_length=sequence_length, n_features=n_features)
            
        except Exception as e:
            logger.error(f"Failed to export metadata: {e}")
            import traceback
            traceback.print_exc()
    
    logger.info(f"\n{'='*60}")
    logger.info("Conversion complete!")
    logger.info(f"TensorFlow Lite models saved to: {args.output_dir}")
    logger.info(f"{'='*60}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
