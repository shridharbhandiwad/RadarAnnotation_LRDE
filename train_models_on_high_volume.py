"""Train and Compare Multiple Models on High-Volume Dataset

This script trains Random Forest, Gradient Boosting, and Neural Network models on the high-volume
simulation dataset and compares their performance.
"""

import logging
import sys
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

from src.ai_engine import train_model


def train_and_compare_models(data_path: str = "data/high_volume_simulation_labeled.csv"):
    """Train multiple models and compare performance
    
    Args:
        data_path: Path to labeled CSV dataset
    """
    
    # Check if dataset exists
    if not Path(data_path).exists():
        logger.error(f"Dataset not found: {data_path}")
        logger.info("Please run: python3 generate_high_volume_simulation.py")
        return 1
    
    logger.info("=" * 80)
    logger.info("TRAINING MULTIPLE MODELS ON HIGH-VOLUME DATASET")
    logger.info("=" * 80)
    logger.info(f"Dataset: {data_path}")
    logger.info("Models: Random Forest, Gradient Boosting, Neural Network")
    logger.info("=" * 80)
    
    models = {}
    metrics = {}
    
    # Train Random Forest
    logger.info("\n" + "=" * 80)
    logger.info("1/3: TRAINING RANDOM FOREST MODEL")
    logger.info("=" * 80)
    try:
        models['random_forest'], metrics['random_forest'] = train_model(
            model_name='random_forest',
            data_path=data_path,
            output_dir='output/models/random_forest_high_volume',
            params={
                'n_estimators': 200,
                'max_depth': 20,
                'min_samples_split': 2,
                'min_samples_leaf': 1,
                'random_state': 42,
                'n_jobs': -1
            },
            auto_transform=True
        )
        logger.info("✓ Random Forest training completed")
    except Exception as e:
        logger.error(f"✗ Random Forest training failed: {e}")
        metrics['random_forest'] = None
    
    # Train Gradient Boosting
    logger.info("\n" + "=" * 80)
    logger.info("2/3: TRAINING GRADIENT BOOSTING MODEL")
    logger.info("=" * 80)
    try:
        models['gradient_boosting'], metrics['gradient_boosting'] = train_model(
            model_name='gradient_boosting',
            data_path=data_path,
            output_dir='output/models/gradient_boosting_high_volume',
            params={
                'n_estimators': 200,
                'max_depth': 8,
                'learning_rate': 0.1
            },
            auto_transform=True
        )
        logger.info("✓ Gradient Boosting training completed")
    except Exception as e:
        logger.error(f"✗ Gradient Boosting training failed: {e}")
        metrics['gradient_boosting'] = None
    
    # Train Neural Network
    logger.info("\n" + "=" * 80)
    logger.info("3/3: TRAINING NEURAL NETWORK MODEL")
    logger.info("=" * 80)
    try:
        models['neural_network'], metrics['neural_network'] = train_model(
            model_name='neural_network',
            data_path=data_path,
            output_dir='output/models/neural_network_high_volume',
            params={
                'd_model': 128,
                'num_heads': 8,
                'ff_dim': 256,
                'num_layers': 3,
                'dropout': 0.2,
                'epochs': 100,
                'batch_size': 64,
                'sequence_length': 30
            },
            auto_transform=True
        )
        logger.info("✓ Neural Network training completed")
    except Exception as e:
        logger.error(f"✗ Neural Network training failed: {e}")
        metrics['neural_network'] = None
    
    # Compare results
    logger.info("\n" + "=" * 80)
    logger.info("MODEL COMPARISON RESULTS")
    logger.info("=" * 80)
    
    results = []
    for name, metric in metrics.items():
        if metric:
            test_acc = metric['test'].get('accuracy', 0)
            test_f1 = metric['test'].get('f1_score', 0)
            train_time = metric['train'].get('training_time', 0)
            
            results.append({
                'model': name,
                'accuracy': test_acc,
                'f1_score': test_f1,
                'time': train_time
            })
            
            logger.info(f"\n{name.upper()}:")
            logger.info(f"  Test Accuracy:  {test_acc:.4f}")
            logger.info(f"  Test F1 Score:  {test_f1:.4f}")
            logger.info(f"  Training Time:  {train_time:.2f}s ({train_time/60:.2f}m)")
    
    if results:
        # Find best model
        best_by_accuracy = max(results, key=lambda x: x['accuracy'])
        best_by_f1 = max(results, key=lambda x: x['f1_score'])
        fastest = min(results, key=lambda x: x['time'])
        
        logger.info("\n" + "=" * 80)
        logger.info("SUMMARY")
        logger.info("=" * 80)
        logger.info(f"🏆 Best Accuracy:  {best_by_accuracy['model'].upper()} ({best_by_accuracy['accuracy']:.4f})")
        logger.info(f"🏆 Best F1 Score:  {best_by_f1['model'].upper()} ({best_by_f1['f1_score']:.4f})")
        logger.info(f"⚡ Fastest:        {fastest['model'].upper()} ({fastest['time']:.2f}s)")
        logger.info("=" * 80)
        
        logger.info("\nModel outputs saved to:")
        logger.info("  - output/models/random_forest_high_volume/")
        logger.info("  - output/models/gradient_boosting_high_volume/")
        logger.info("  - output/models/neural_network_high_volume/")
    
    return 0


def main():
    """Main execution function"""
    try:
        return train_and_compare_models()
    except Exception as e:
        logger.error(f"Training pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
