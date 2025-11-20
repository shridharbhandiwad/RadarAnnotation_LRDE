"""Test script for Transformer model"""
import numpy as np
import pandas as pd
import tempfile
import shutil
from pathlib import Path

def test_transformer_model():
    """Test basic functionality of Transformer model"""
    print("=" * 60)
    print("Testing Transformer-based Multi-output Model")
    print("=" * 60)
    
    try:
        from src.ai_engine import TransformerModel
        print("✓ TransformerModel imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import TransformerModel: {e}")
        return False
    
    # Create synthetic test data with composite labels
    print("\n1. Creating synthetic test data...")
    np.random.seed(42)
    
    n_tracks = 10
    points_per_track = 30
    data = []
    
    labels = [
        'incoming,level,linear,light_maneuver,low_speed',
        'outgoing,ascending,curved,high_maneuver,high_speed',
        'incoming,descending,linear,light_maneuver,high_speed',
        'outgoing,level,curved,high_maneuver,low_speed'
    ]
    
    for track_id in range(n_tracks):
        label = labels[track_id % len(labels)]
        for t in range(points_per_track):
            data.append({
                'trackid': track_id,
                'time': t * 0.1,
                'x': np.random.randn() * 100 + track_id * 50,
                'y': np.random.randn() * 100,
                'z': 1000 + t * 10 + np.random.randn() * 10,
                'vx': 50 + np.random.randn() * 5,
                'vy': 30 + np.random.randn() * 5,
                'vz': 5 + np.random.randn(),
                'ax': np.random.randn(),
                'ay': np.random.randn(),
                'az': np.random.randn() * 0.1,
                'speed': 60 + np.random.randn() * 10,
                'heading': np.random.rand() * 360,
                'range': 5000 + t * 100,
                'curvature': np.random.rand() * 0.02,
                'Annotation': label
            })
    
    df = pd.DataFrame(data)
    print(f"✓ Created {len(df)} data points across {n_tracks} tracks")
    print(f"✓ Labels: {df['Annotation'].unique()}")
    
    # Split data
    print("\n2. Splitting data...")
    train_tracks = list(range(8))
    test_tracks = list(range(8, 10))
    
    df_train = df[df['trackid'].isin(train_tracks)]
    df_test = df[df['trackid'].isin(test_tracks)]
    print(f"✓ Train: {len(df_train)} samples, Test: {len(df_test)} samples")
    
    # Initialize model
    print("\n3. Initializing Transformer model...")
    try:
        params = {
            'd_model': 32,
            'num_heads': 2,
            'ff_dim': 64,
            'num_layers': 1,
            'dropout': 0.1,
            'epochs': 3,  # Quick test
            'batch_size': 4,
            'sequence_length': 10
        }
        model = TransformerModel(params)
        print("✓ Model initialized successfully")
    except Exception as e:
        print(f"✗ Model initialization failed: {e}")
        return False
    
    # Test single-output mode
    print("\n4. Testing single-output mode...")
    try:
        # Modify labels to be simple for single-output test
        df_train_single = df_train.copy()
        df_test_single = df_test.copy()
        df_train_single['Annotation'] = df_train_single['Annotation'].apply(lambda x: x.split(',')[0])
        df_test_single['Annotation'] = df_test_single['Annotation'].apply(lambda x: x.split(',')[0])
        
        metrics = model.train(df_train_single, use_multi_output=False)
        print(f"✓ Single-output training completed in {metrics['training_time']:.2f}s")
        print(f"  Train accuracy: {metrics.get('train_accuracy', 0):.4f}")
        
        eval_metrics = model.evaluate(df_test_single)
        print(f"✓ Single-output evaluation: Accuracy={eval_metrics['accuracy']:.4f}")
    except Exception as e:
        print(f"✗ Single-output training failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test multi-output mode
    print("\n5. Testing multi-output mode...")
    try:
        model2 = TransformerModel(params)
        metrics = model2.train(df_train, use_multi_output=True)
        print(f"✓ Multi-output training completed in {metrics['training_time']:.2f}s")
        print(f"  Multi-output: {metrics.get('multi_output', False)}")
        
        # Check if we have per-output metrics
        for key in metrics:
            if 'accuracy' in key:
                print(f"  {key}: {metrics[key]:.4f}")
        
        eval_metrics = model2.evaluate(df_test)
        print(f"✓ Multi-output evaluation: Overall Accuracy={eval_metrics['accuracy']:.4f}")
        
        if 'outputs' in eval_metrics:
            print("  Per-output metrics:")
            for output_name, output_metrics in eval_metrics['outputs'].items():
                print(f"    {output_name}: Acc={output_metrics['accuracy']:.4f}")
    except Exception as e:
        print(f"✗ Multi-output training failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test save/load
    print("\n6. Testing save/load functionality...")
    try:
        temp_dir = tempfile.mkdtemp()
        model_path = Path(temp_dir) / "test_transformer.h5"
        
        model2.save(str(model_path))
        print(f"✓ Model saved to {model_path}")
        
        model3 = TransformerModel(params)
        model3.load(str(model_path))
        print("✓ Model loaded successfully")
        
        # Clean up
        shutil.rmtree(temp_dir)
        print("✓ Cleanup completed")
    except Exception as e:
        print(f"✗ Save/load failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "=" * 60)
    print("✓ All tests passed!")
    print("=" * 60)
    return True


if __name__ == "__main__":
    import sys
    
    # Check if TensorFlow is available
    try:
        import tensorflow as tf
        print(f"TensorFlow version: {tf.__version__}")
    except ImportError:
        print("✗ TensorFlow is not installed!")
        print("  Install with: pip install tensorflow")
        sys.exit(1)
    
    success = test_transformer_model()
    sys.exit(0 if success else 1)
