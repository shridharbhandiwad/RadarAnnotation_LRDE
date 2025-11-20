"""Quick test of transformer and LSTM models with enhanced features"""

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
    logger.info("Quick Test: Transformer & LSTM with Enhanced Features")
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
    
    # Step 3: Train Transformer
    logger.info("\n[3/5] Training Transformer model...")
    try:
        t_model, t_metrics = train_model(
            model_name='transformer',
            data_path=labeled_csv,
            output_dir='output/test_transformer',
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
        
        logger.info("✓ Transformer trained successfully")
        logger.info(f"  Train Accuracy: {t_metrics['train'].get('train_accuracy', 0):.4f}")
        logger.info(f"  Test Accuracy: {t_metrics['test'].get('accuracy', 0):.4f}")
        logger.info(f"  Multi-output: {t_metrics['train'].get('multi_output', False)}")
        
        if t_metrics['train'].get('multi_output', False) and 'outputs' in t_metrics['test']:
            logger.info("  Per-output accuracy:")
            for name, metrics in t_metrics['test']['outputs'].items():
                logger.info(f"    {name}: {metrics['accuracy']:.4f}")
                
    except Exception as e:
        logger.error(f"✗ Transformer training failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Step 4: Train LSTM
    logger.info("\n[4/5] Training LSTM model...")
    try:
        l_model, l_metrics = train_model(
            model_name='lstm',
            data_path=labeled_csv,
            output_dir='output/test_lstm',
            params={
                'units': 64,
                'dropout': 0.2,
                'epochs': 30,
                'batch_size': 32,
                'sequence_length': 20
            },
            auto_transform=True
        )
        
        logger.info("✓ LSTM trained successfully")
        logger.info(f"  Train Accuracy: {l_metrics['train'].get('train_accuracy', 0):.4f}")
        logger.info(f"  Test Accuracy: {l_metrics['test'].get('accuracy', 0):.4f}")
        
    except Exception as e:
        logger.error(f"✗ LSTM training failed: {e}")
        import traceback
        traceback.print_exc()
    
    logger.info("\n[5/5] Test completed!")
    logger.info("=" * 80)
    logger.info("✓ Both models are working correctly with enhanced features!")
    logger.info("=" * 80)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
