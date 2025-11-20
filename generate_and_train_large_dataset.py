"""Generate large simulation dataset and train transformer/LSTM models

This script:
1. Creates a large simulation dataset with 200+ tracks and diverse patterns
2. Applies auto-labeling to generate composite labels
3. Trains both transformer and LSTM models
4. Evaluates and compares performance
"""

import logging
import sys
from pathlib import Path
import pandas as pd

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

from src.sim_engine import create_large_training_dataset
from src.autolabel_engine import compute_motion_features, apply_rules_and_flags, get_annotation_summary
from src.ai_engine import train_model, TransformerModel, LSTMModel


def generate_large_dataset(n_tracks=200, duration_min=10):
    """Generate large simulation dataset
    
    Args:
        n_tracks: Number of tracks to generate
        duration_min: Duration of each track in minutes
        
    Returns:
        Path to generated CSV
    """
    logger.info("=" * 80)
    logger.info("STEP 1: Generating Large Simulation Dataset")
    logger.info("=" * 80)
    
    output_path = "data/large_simulation_training.csv"
    
    # Create large dataset
    csv_path = create_large_training_dataset(
        output_path=output_path,
        n_tracks=n_tracks,
        duration_min=duration_min
    )
    
    logger.info(f"✓ Generated large dataset: {csv_path}")
    
    return csv_path


def apply_autolabeling(csv_path):
    """Apply auto-labeling to generate annotations
    
    Args:
        csv_path: Path to raw CSV data
        
    Returns:
        Path to labeled CSV
    """
    logger.info("=" * 80)
    logger.info("STEP 2: Applying Auto-Labeling")
    logger.info("=" * 80)
    
    # Load data
    df = pd.read_csv(csv_path)
    logger.info(f"Loaded {len(df)} records from {csv_path}")
    
    # Compute motion features
    logger.info("Computing motion features...")
    df = compute_motion_features(df)
    
    # Apply rule-based flags
    logger.info("Applying rule-based classification...")
    df = apply_rules_and_flags(df)
    
    # Get summary
    summary = get_annotation_summary(df)
    logger.info(f"✓ Labeled {summary['valid_records']}/{summary['total_records']} records")
    
    # Print annotation distribution
    logger.info("\nAnnotation Distribution:")
    for annotation, data in list(summary['annotation_distribution'].items())[:10]:
        logger.info(f"  {annotation}: {data['count']} ({data['percentage']:.2f}%)")
    
    # Save labeled data
    labeled_path = csv_path.replace('.csv', '_labeled.csv')
    df.to_csv(labeled_path, index=False)
    logger.info(f"✓ Saved labeled data: {labeled_path}")
    
    return labeled_path


def train_transformer(labeled_csv, output_dir="output/models/transformer_large"):
    """Train transformer model on labeled data
    
    Args:
        labeled_csv: Path to labeled CSV
        output_dir: Output directory for model
        
    Returns:
        Tuple of (model, metrics)
    """
    logger.info("=" * 80)
    logger.info("STEP 3: Training Transformer Model")
    logger.info("=" * 80)
    
    try:
        model, metrics = train_model(
            model_name='transformer',
            data_path=labeled_csv,
            output_dir=output_dir,
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
        
        logger.info("✓ Transformer training completed")
        logger.info(f"  Train Accuracy: {metrics['train'].get('train_accuracy', 0):.4f}")
        logger.info(f"  Test Accuracy: {metrics['test'].get('accuracy', 0):.4f}")
        logger.info(f"  Test F1 Score: {metrics['test'].get('f1_score', 0):.4f}")
        
        if metrics['train'].get('multi_output', False):
            logger.info("  Multi-output metrics:")
            if 'outputs' in metrics['test']:
                for output_name, output_metrics in metrics['test']['outputs'].items():
                    logger.info(f"    {output_name}: Acc={output_metrics['accuracy']:.4f}, F1={output_metrics['f1_score']:.4f}")
        
        return model, metrics
        
    except Exception as e:
        logger.error(f"✗ Transformer training failed: {e}")
        return None, None


def train_lstm(labeled_csv, output_dir="output/models/lstm_large"):
    """Train LSTM model on labeled data
    
    Args:
        labeled_csv: Path to labeled CSV
        output_dir: Output directory for model
        
    Returns:
        Tuple of (model, metrics)
    """
    logger.info("=" * 80)
    logger.info("STEP 4: Training LSTM Model")
    logger.info("=" * 80)
    
    try:
        model, metrics = train_model(
            model_name='lstm',
            data_path=labeled_csv,
            output_dir=output_dir,
            params={
                'units': 128,
                'dropout': 0.3,
                'epochs': 100,
                'batch_size': 64,
                'sequence_length': 30
            },
            auto_transform=True
        )
        
        logger.info("✓ LSTM training completed")
        logger.info(f"  Train Accuracy: {metrics['train'].get('train_accuracy', 0):.4f}")
        logger.info(f"  Test Accuracy: {metrics['test'].get('accuracy', 0):.4f}")
        logger.info(f"  Test F1 Score: {metrics['test'].get('f1_score', 0):.4f}")
        
        return model, metrics
        
    except Exception as e:
        logger.error(f"✗ LSTM training failed: {e}")
        return None, None


def compare_models(transformer_metrics, lstm_metrics):
    """Compare transformer and LSTM performance
    
    Args:
        transformer_metrics: Transformer model metrics
        lstm_metrics: LSTM model metrics
    """
    logger.info("=" * 80)
    logger.info("STEP 5: Model Comparison")
    logger.info("=" * 80)
    
    if transformer_metrics and lstm_metrics:
        logger.info("\n{:<20} {:<15} {:<15}".format("Metric", "Transformer", "LSTM"))
        logger.info("-" * 50)
        
        t_train_acc = transformer_metrics['train'].get('train_accuracy', 0)
        l_train_acc = lstm_metrics['train'].get('train_accuracy', 0)
        logger.info("{:<20} {:<15.4f} {:<15.4f}".format("Train Accuracy", t_train_acc, l_train_acc))
        
        t_test_acc = transformer_metrics['test'].get('accuracy', 0)
        l_test_acc = lstm_metrics['test'].get('accuracy', 0)
        logger.info("{:<20} {:<15.4f} {:<15.4f}".format("Test Accuracy", t_test_acc, l_test_acc))
        
        t_f1 = transformer_metrics['test'].get('f1_score', 0)
        l_f1 = lstm_metrics['test'].get('f1_score', 0)
        logger.info("{:<20} {:<15.4f} {:<15.4f}".format("Test F1 Score", t_f1, l_f1))
        
        t_time = transformer_metrics['train'].get('training_time', 0)
        l_time = lstm_metrics['train'].get('training_time', 0)
        logger.info("{:<20} {:<15.2f}s {:<15.2f}s".format("Training Time", t_time, l_time))
        
        logger.info("\n" + "=" * 80)
        
        # Determine winner
        if t_test_acc > l_test_acc:
            logger.info("🏆 Transformer model performed better!")
        elif l_test_acc > t_test_acc:
            logger.info("🏆 LSTM model performed better!")
        else:
            logger.info("🤝 Both models performed equally well!")


def main():
    """Main execution function"""
    logger.info("=" * 80)
    logger.info("Large-Scale Transformer & LSTM Training Pipeline")
    logger.info("=" * 80)
    
    try:
        # Step 1: Generate large dataset
        raw_csv = generate_large_dataset(n_tracks=200, duration_min=10)
        
        # Step 2: Apply auto-labeling
        labeled_csv = apply_autolabeling(raw_csv)
        
        # Step 3: Train transformer
        transformer_model, transformer_metrics = train_transformer(labeled_csv)
        
        # Step 4: Train LSTM
        lstm_model, lstm_metrics = train_lstm(labeled_csv)
        
        # Step 5: Compare models
        compare_models(transformer_metrics, lstm_metrics)
        
        logger.info("=" * 80)
        logger.info("✓ Pipeline completed successfully!")
        logger.info("=" * 80)
        logger.info(f"\nGenerated files:")
        logger.info(f"  Raw data: {raw_csv}")
        logger.info(f"  Labeled data: {labeled_csv}")
        logger.info(f"  Transformer model: output/models/transformer_large/")
        logger.info(f"  LSTM model: output/models/lstm_large/")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
