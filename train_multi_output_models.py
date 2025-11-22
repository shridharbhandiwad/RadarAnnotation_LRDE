"""Train Multi-Output Models for Auto-Tagging

This script trains XGBoost, Random Forest, and Transformer models in multi-output mode
for auto-tagging and auto-annotation based on the data format:
- Columns A-K: Input features (radar measurements)
- Columns L-AF: Output tags (to be predicted)
- Column AG: Aggregated annotation (reference)

All three models will predict multiple tag columns simultaneously.
"""

import logging
import sys
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

from src.ai_engine import (
    XGBoostMultiOutputModel,
    RandomForestMultiOutputModel,
    TransformerMultiOutputModel
)


def train_multi_output_models(data_path: str = "data/high_volume_simulation_labeled.csv",
                               output_dir: str = "output/multi_output_models"):
    """Train all three multi-output models and compare performance
    
    Args:
        data_path: Path to labeled CSV dataset
        output_dir: Output directory for trained models
    """
    
    # Check if dataset exists
    if not Path(data_path).exists():
        logger.error(f"Dataset not found: {data_path}")
        logger.info("Please provide a valid dataset path")
        return 1
    
    logger.info("=" * 80)
    logger.info("TRAINING MULTI-OUTPUT MODELS FOR AUTO-TAGGING")
    logger.info("=" * 80)
    logger.info(f"Dataset: {data_path}")
    logger.info("Data format:")
    logger.info("  - Columns A-K: Input features (x, y, z, velocities, etc.)")
    logger.info("  - Columns L-AF: Output tags (incoming, outgoing, level, etc.)")
    logger.info("  - Column AG: Aggregated annotation (reference)")
    logger.info("=" * 80)
    
    # Load data
    logger.info("Loading data...")
    df = pd.read_csv(data_path)
    logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")
    logger.info(f"Unique tracks: {df['trackid'].nunique()}")
    
    # Split data by track ID
    track_ids = df['trackid'].unique()
    train_ids, test_ids = train_test_split(track_ids, test_size=0.2, random_state=42)
    train_ids, val_ids = train_test_split(train_ids, test_size=0.2, random_state=42)
    
    df_train = df[df['trackid'].isin(train_ids)]
    df_val = df[df['trackid'].isin(val_ids)]
    df_test = df[df['trackid'].isin(test_ids)]
    
    logger.info(f"Data split:")
    logger.info(f"  Train: {len(train_ids)} tracks, {len(df_train)} samples")
    logger.info(f"  Val:   {len(val_ids)} tracks, {len(df_val)} samples")
    logger.info(f"  Test:  {len(test_ids)} tracks, {len(df_test)} samples")
    logger.info("")
    
    models = {}
    metrics = {}
    
    # ========================================================================
    # 1. Train XGBoost Multi-Output Model
    # ========================================================================
    logger.info("=" * 80)
    logger.info("1/3: TRAINING XGBOOST MULTI-OUTPUT MODEL")
    logger.info("=" * 80)
    try:
        model_xgb = XGBoostMultiOutputModel(params={
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'random_state': 42
        })
        
        train_metrics_xgb = model_xgb.train(df_train, df_val)
        test_metrics_xgb = model_xgb.evaluate(df_test)
        
        # Save model
        model_path_xgb = Path(output_dir) / 'xgboost_multi_output' / 'model.pkl'
        model_xgb.save(str(model_path_xgb))
        
        models['xgboost'] = model_xgb
        metrics['xgboost'] = {
            'train': train_metrics_xgb,
            'test': test_metrics_xgb
        }
        
        logger.info("✓ XGBoost Multi-Output training completed")
        logger.info(f"  Overall Test Accuracy: {test_metrics_xgb['accuracy']:.4f}")
        logger.info(f"  Overall Test F1: {test_metrics_xgb['f1_score']:.4f}")
        logger.info("")
        
    except Exception as e:
        logger.error(f"✗ XGBoost training failed: {e}")
        import traceback
        traceback.print_exc()
        metrics['xgboost'] = None
    
    # ========================================================================
    # 2. Train Random Forest Multi-Output Model
    # ========================================================================
    logger.info("=" * 80)
    logger.info("2/3: TRAINING RANDOM FOREST MULTI-OUTPUT MODEL")
    logger.info("=" * 80)
    try:
        model_rf = RandomForestMultiOutputModel(params={
            'n_estimators': 100,
            'max_depth': 15,
            'min_samples_split': 2,
            'random_state': 42,
            'n_jobs': -1
        })
        
        train_metrics_rf = model_rf.train(df_train, df_val)
        test_metrics_rf = model_rf.evaluate(df_test)
        
        # Save model
        model_path_rf = Path(output_dir) / 'random_forest_multi_output' / 'model.pkl'
        model_rf.save(str(model_path_rf))
        
        models['random_forest'] = model_rf
        metrics['random_forest'] = {
            'train': train_metrics_rf,
            'test': test_metrics_rf
        }
        
        logger.info("✓ Random Forest Multi-Output training completed")
        logger.info(f"  Overall Test Accuracy: {test_metrics_rf['accuracy']:.4f}")
        logger.info(f"  Overall Test F1: {test_metrics_rf['f1_score']:.4f}")
        logger.info("")
        
    except Exception as e:
        logger.error(f"✗ Random Forest training failed: {e}")
        import traceback
        traceback.print_exc()
        metrics['random_forest'] = None
    
    # ========================================================================
    # 3. Train Transformer Multi-Output Model
    # ========================================================================
    logger.info("=" * 80)
    logger.info("3/3: TRAINING TRANSFORMER MULTI-OUTPUT MODEL")
    logger.info("=" * 80)
    try:
        model_transformer = TransformerMultiOutputModel(params={
            'd_model': 64,
            'num_heads': 4,
            'ff_dim': 128,
            'num_layers': 2,
            'dropout': 0.2,
            'epochs': 50,
            'batch_size': 32,
            'sequence_length': 20
        })
        
        train_metrics_transformer = model_transformer.train(df_train, df_val)
        test_metrics_transformer = model_transformer.evaluate(df_test)
        
        # Save model
        model_path_transformer = Path(output_dir) / 'transformer_multi_output' / 'model.h5'
        model_transformer.save(str(model_path_transformer))
        
        models['transformer'] = model_transformer
        metrics['transformer'] = {
            'train': train_metrics_transformer,
            'test': test_metrics_transformer
        }
        
        logger.info("✓ Transformer Multi-Output training completed")
        logger.info(f"  Overall Test Accuracy: {test_metrics_transformer['accuracy']:.4f}")
        logger.info(f"  Overall Test F1: {test_metrics_transformer['f1_score']:.4f}")
        logger.info("")
        
    except Exception as e:
        logger.error(f"✗ Transformer training failed: {e}")
        import traceback
        traceback.print_exc()
        metrics['transformer'] = None
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    logger.info("")
    logger.info("=" * 80)
    logger.info("MULTI-OUTPUT MODEL COMPARISON RESULTS")
    logger.info("=" * 80)
    logger.info("")
    
    # Create summary table
    print("┌─────────────────────────────┬──────────────┬──────────────┬──────────────┐")
    print("│ Model                       │   Accuracy   │   F1 Score   │ Training Time│")
    print("├─────────────────────────────┼──────────────┼──────────────┼──────────────┤")
    
    results = []
    for name, metric in metrics.items():
        if metric and metric['test']:
            test_acc = metric['test'].get('accuracy', 0)
            test_f1 = metric['test'].get('f1_score', 0)
            train_time = metric['train'].get('training_time', 0)
            
            results.append({
                'model': name,
                'accuracy': test_acc,
                'f1_score': test_f1,
                'time': train_time
            })
            
            print(f"│ {name.upper():<27} │   {test_acc:>8.4f}   │   {test_f1:>8.4f}   │   {train_time:>8.2f}s │")
    
    print("└─────────────────────────────┴──────────────┴──────────────┴──────────────┘")
    print("")
    
    # Per-tag breakdown
    logger.info("=" * 80)
    logger.info("PER-TAG PERFORMANCE BREAKDOWN")
    logger.info("=" * 80)
    
    for name, metric in metrics.items():
        if metric and metric['test'] and 'per_tag_metrics' in metric['test']:
            logger.info(f"\n{name.upper()}:")
            per_tag = metric['test']['per_tag_metrics']
            for tag_name, tag_metric in per_tag.items():
                acc = tag_metric['accuracy']
                f1 = tag_metric['f1_score']
                logger.info(f"  {tag_name:<25} Acc: {acc:.4f}  F1: {f1:.4f}")
    
    # Best models
    if results:
        logger.info("")
        logger.info("=" * 80)
        logger.info("SUMMARY")
        logger.info("=" * 80)
        
        best_by_accuracy = max(results, key=lambda x: x['accuracy'])
        best_by_f1 = max(results, key=lambda x: x['f1_score'])
        fastest = min(results, key=lambda x: x['time'])
        
        logger.info(f"🏆 Best Accuracy:  {best_by_accuracy['model'].upper()} ({best_by_accuracy['accuracy']:.4f})")
        logger.info(f"🏆 Best F1 Score:  {best_by_f1['model'].upper()} ({best_by_f1['f1_score']:.4f})")
        logger.info(f"⚡ Fastest:        {fastest['model'].upper()} ({fastest['time']:.2f}s)")
        logger.info("=" * 80)
        logger.info("")
        logger.info(f"✅ All models saved to: {output_dir}/")
        logger.info("   - xgboost_multi_output/model.pkl")
        logger.info("   - random_forest_multi_output/model.pkl")
        logger.info("   - transformer_multi_output/model.h5")
        logger.info("")
        logger.info("💡 These models can now predict multiple tags simultaneously for auto-tagging!")
    
    return 0


def main():
    """Main execution function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Multi-Output Models for Auto-Tagging')
    parser.add_argument('--data', default='data/high_volume_simulation_labeled.csv',
                       help='Path to labeled CSV dataset')
    parser.add_argument('--output', default='output/multi_output_models',
                       help='Output directory for trained models')
    
    args = parser.parse_args()
    
    try:
        return train_multi_output_models(args.data, args.output)
    except Exception as e:
        logger.error(f"Training pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
