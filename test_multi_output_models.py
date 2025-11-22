"""Quick test script for multi-output models"""

import pandas as pd
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

from src.ai_engine import XGBoostMultiOutputModel

def test_multi_output():
    """Test multi-output models with sample data"""
    
    # Load sample data
    data_path = 'data/high_volume_simulation_labeled.csv'
    if not Path(data_path).exists():
        logger.error(f"Test data not found: {data_path}")
        return 1
    
    logger.info(f"Loading test data from {data_path}")
    df = pd.read_csv(data_path, nrows=5000)  # Use subset for quick test
    logger.info(f"Loaded {len(df)} rows")
    
    # Show data info
    logger.info(f"Columns: {len(df.columns)}")
    logger.info(f"Unique tracks: {df['trackid'].nunique()}")
    
    # Split data
    from sklearn.model_selection import train_test_split
    track_ids = df['trackid'].unique()
    
    if len(track_ids) < 3:
        logger.error("Need at least 3 tracks for train/val/test split")
        return 1
    
    train_ids, test_ids = train_test_split(track_ids, test_size=0.3, random_state=42)
    df_train = df[df['trackid'].isin(train_ids)]
    df_test = df[df['trackid'].isin(test_ids)]
    
    logger.info(f"Train: {len(train_ids)} tracks, {len(df_train)} samples")
    logger.info(f"Test: {len(test_ids)} tracks, {len(df_test)} samples")
    
    # Test XGBoost Multi-Output Model
    logger.info("\n" + "="*60)
    logger.info("Testing XGBoost Multi-Output Model")
    logger.info("="*60)
    
    try:
        model = XGBoostMultiOutputModel(params={
            'n_estimators': 50,
            'max_depth': 5,
            'learning_rate': 0.1
        })
        
        logger.info("Training model...")
        train_metrics = model.train(df_train, df_val=None)
        
        logger.info(f"✓ Training completed in {train_metrics['training_time']:.2f}s")
        logger.info(f"  Overall train accuracy: {train_metrics['train_accuracy']:.4f}")
        logger.info(f"  Number of tags: {train_metrics['n_tags']}")
        logger.info(f"  Tag names: {train_metrics['tag_names']}")
        
        logger.info("\nEvaluating on test set...")
        test_metrics = model.evaluate(df_test)
        
        logger.info(f"✓ Evaluation completed")
        logger.info(f"  Overall test accuracy: {test_metrics['accuracy']:.4f}")
        logger.info(f"  Overall test F1: {test_metrics['f1_score']:.4f}")
        
        logger.info("\nPer-tag performance:")
        for tag_name, metrics in test_metrics['per_tag_metrics'].items():
            logger.info(f"  {tag_name:<25} Acc: {metrics['accuracy']:.4f}  F1: {metrics['f1_score']:.4f}")
        
        logger.info("\nTesting prediction...")
        predictions = model.predict(df_test.head(10))
        logger.info(f"✓ Predictions shape: {predictions.shape}")
        logger.info(f"  Columns: {list(predictions.columns)}")
        logger.info("\nSample predictions:")
        logger.info(predictions[['incoming', 'outgoing', 'level_flight', 'linear', 'Predicted_Annotation']].head())
        
        logger.info("\n" + "="*60)
        logger.info("✅ ALL TESTS PASSED!")
        logger.info("="*60)
        return 0
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(test_multi_output())
