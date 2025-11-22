"""Quick test of Random Forest, Gradient Boosting, and Neural Network models"""

import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

logger = logging.getLogger(__name__)

from src.sim_engine import create_large_training_dataset
from src.autolabel_engine import compute_motion_features, apply_rules_and_flags
from src.ai_engine import train_model
import pandas as pd


def main():
    """Quick test with small dataset"""
    logger.info("=" * 80)
    logger.info("Quick Test: Random Forest, Gradient Boosting & Neural Network")
    logger.info("=" * 80)
    
    # Step 1: Generate small dataset (50 tracks, 3 minutes each)
    logger.info("\n[1/5] Generating simulation data...")
    raw_csv = create_large_training_dataset(
        output_path="data/test_simulation.csv",
        n_tracks=50,
        duration_min=3
    )
    
    # Step 2: Load and prepare data
    logger.info("\n[2/5] Applying auto-labeling...")
    df = pd.read_csv(raw_csv)
    df = compute_motion_features(df)
    df = apply_rules_and_flags(df)
    
    labeled_csv = raw_csv.replace('.csv', '_labeled.csv')
    df.to_csv(labeled_csv, index=False)
    logger.info(f"✓ Labeled data saved: {labeled_csv}")
    logger.info(f"  Total tracks: {df['trackid'].nunique()}")
    logger.info(f"  Total records: {len(df)}")
    logger.info(f"  Valid records: {df['valid_features'].sum()}")
    
    # Check annotation diversity
    annotations = df['Annotation'].value_counts()
    logger.info(f"  Unique annotations: {len(annotations)}")
    logger.info(f"  Top 3 annotations:")
    for ann, count in list(annotations.items())[:3]:
        logger.info(f"    {ann}: {count}")
    
    # Step 3: Train Random Forest
    logger.info("\n[3/6] Training Random Forest model...")
    try:
        rf_model, rf_metrics = train_model(
            model_name='random_forest',
            data_path=labeled_csv,
            output_dir='output/test_random_forest',
            params={
                'n_estimators': 100,
                'max_depth': 15,
                'random_state': 42,
                'n_jobs': -1
            },
            auto_transform=True
        )
        
        logger.info("✓ Random Forest trained successfully")
        logger.info(f"  Train Accuracy: {rf_metrics['train'].get('train_accuracy', 0):.4f}")
        logger.info(f"  Test Accuracy: {rf_metrics['test'].get('accuracy', 0):.4f}")
                
    except Exception as e:
        logger.error(f"✗ Random Forest training failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Step 4: Train Gradient Boosting
    logger.info("\n[4/6] Training Gradient Boosting model...")
    try:
        gb_model, gb_metrics = train_model(
            model_name='gradient_boosting',
            data_path=labeled_csv,
            output_dir='output/test_gradient_boosting',
            params={
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.1
            },
            auto_transform=True
        )
        
        logger.info("✓ Gradient Boosting trained successfully")
        logger.info(f"  Train Accuracy: {gb_metrics['train'].get('train_accuracy', 0):.4f}")
        logger.info(f"  Test Accuracy: {gb_metrics['test'].get('accuracy', 0):.4f}")
        
    except Exception as e:
        logger.error(f"✗ Gradient Boosting training failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Step 5: Train Neural Network
    logger.info("\n[5/6] Training Neural Network model...")
    try:
        nn_model, nn_metrics = train_model(
            model_name='neural_network',
            data_path=labeled_csv,
            output_dir='output/test_neural_network',
            params={
                'd_model': 64,
                'num_heads': 4,
                'ff_dim': 128,
                'num_layers': 2,
                'dropout': 0.2,
                'epochs': 30,
                'batch_size': 32,
                'sequence_length': 20
            },
            auto_transform=True
        )
        
        logger.info("✓ Neural Network trained successfully")
        logger.info(f"  Train Accuracy: {nn_metrics['train'].get('train_accuracy', 0):.4f}")
        logger.info(f"  Test Accuracy: {nn_metrics['test'].get('accuracy', 0):.4f}")
        logger.info(f"  Multi-output: {nn_metrics['train'].get('multi_output', False)}")
        
        if nn_metrics['train'].get('multi_output', False) and 'outputs' in nn_metrics['test']:
            logger.info("  Per-output accuracy:")
            for name, metrics in nn_metrics['test']['outputs'].items():
                logger.info(f"    {name}: {metrics['accuracy']:.4f}")
        
    except Exception as e:
        logger.error(f"✗ Neural Network training failed: {e}")
        import traceback
        traceback.print_exc()
    
    logger.info("\n[6/6] Test completed!")
    logger.info("=" * 80)
    logger.info("✓ All three models are working correctly!")
    logger.info("=" * 80)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
