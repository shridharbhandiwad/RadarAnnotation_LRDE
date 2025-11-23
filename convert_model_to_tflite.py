"""Convert Keras Models to TensorFlow Lite Format for C++ Deployment

TensorFlow Lite provides excellent C++ support and is optimized for deployment.
This script converts trained Keras models to TFLite format.
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
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    HAS_TF = True
except ImportError:
    HAS_TF = False
    print("ERROR: TensorFlow is required for model conversion")
    print("Install with: pip install tensorflow")
    sys.exit(1)

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


def main():
    parser = argparse.ArgumentParser(description='Convert Keras models to TensorFlow Lite for C++ deployment')
    parser.add_argument('--model-type', type=str, choices=['lstm', 'transformer', 'both'],
                       default='lstm', help='Which model to convert')
    parser.add_argument('--output-dir', type=str, default='cpp_models',
                       help='Output directory for TFLite models')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
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
