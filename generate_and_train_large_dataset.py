"""Generate large simulation dataset and train Random Forest, Gradient Boosting, and Neural Network models

This script:
1. Creates a large simulation dataset with 200+ tracks and diverse patterns
2. Applies auto-labeling to generate composite labels
3. Trains Random Forest, Gradient Boosting, and Neural Network models
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
from src.ai_engine import train_model


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


def train_random_forest(labeled_csv, output_dir="output/models/random_forest_large"):
    """Train Random Forest model on labeled data
    
    Args:
        labeled_csv: Path to labeled CSV
        output_dir: Output directory for model
        
    Returns:
        Tuple of (model, metrics)
    """
    logger.info("=" * 80)
    logger.info("STEP 3: Training Random Forest Model")
    logger.info("=" * 80)
    
    try:
        model, metrics = train_model(
            model_name='random_forest',
            data_path=labeled_csv,
            output_dir=output_dir,
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
        logger.info(f"  Train Accuracy: {metrics['train'].get('train_accuracy', 0):.4f}")
        logger.info(f"  Test Accuracy: {metrics['test'].get('accuracy', 0):.4f}")
        logger.info(f"  Test F1 Score: {metrics['test'].get('f1_score', 0):.4f}")
        
        return model, metrics
        
    except Exception as e:
        logger.error(f"✗ Random Forest training failed: {e}")
        return None, None


def train_gradient_boosting(labeled_csv, output_dir="output/models/gradient_boosting_large"):
    """Train Gradient Boosting model on labeled data
    
    Args:
        labeled_csv: Path to labeled CSV
        output_dir: Output directory for model
        
    Returns:
        Tuple of (model, metrics)
    """
    logger.info("=" * 80)
    logger.info("STEP 4: Training Gradient Boosting Model")
    logger.info("=" * 80)
    
    try:
        model, metrics = train_model(
            model_name='gradient_boosting',
            data_path=labeled_csv,
            output_dir=output_dir,
            params={
                'n_estimators': 200,
                'max_depth': 8,
                'learning_rate': 0.1
            },
            auto_transform=True
        )
        
        logger.info("✓ Gradient Boosting training completed")
        logger.info(f"  Train Accuracy: {metrics['train'].get('train_accuracy', 0):.4f}")
        logger.info(f"  Test Accuracy: {metrics['test'].get('accuracy', 0):.4f}")
        logger.info(f"  Test F1 Score: {metrics['test'].get('f1_score', 0):.4f}")
        
        return model, metrics
        
    except Exception as e:
        logger.error(f"✗ Gradient Boosting training failed: {e}")
        return None, None


def train_neural_network(labeled_csv, output_dir="output/models/neural_network_large"):
    """Train Neural Network model on labeled data
    
    Args:
        labeled_csv: Path to labeled CSV
        output_dir: Output directory for model
        
    Returns:
        Tuple of (model, metrics)
    """
    logger.info("=" * 80)
    logger.info("STEP 5: Training Neural Network Model")
    logger.info("=" * 80)
    
    try:
        model, metrics = train_model(
            model_name='neural_network',
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
        
        logger.info("✓ Neural Network training completed")
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
        logger.error(f"✗ Neural Network training failed: {e}")
        return None, None


def compare_models(rf_metrics, gb_metrics, nn_metrics):
    """Compare model performance
    
    Args:
        rf_metrics: Random Forest model metrics
        gb_metrics: Gradient Boosting model metrics
        nn_metrics: Neural Network model metrics
    """
    logger.info("=" * 80)
    logger.info("STEP 6: Model Comparison")
    logger.info("=" * 80)
    
    if rf_metrics and gb_metrics and nn_metrics:
        logger.info("\n{:<20} {:<15} {:<15} {:<15}".format("Metric", "Random Forest", "Grad Boosting", "Neural Net"))
        logger.info("-" * 65)
        
        rf_train_acc = rf_metrics['train'].get('train_accuracy', 0)
        gb_train_acc = gb_metrics['train'].get('train_accuracy', 0)
        nn_train_acc = nn_metrics['train'].get('train_accuracy', 0)
        logger.info("{:<20} {:<15.4f} {:<15.4f} {:<15.4f}".format("Train Accuracy", rf_train_acc, gb_train_acc, nn_train_acc))
        
        rf_test_acc = rf_metrics['test'].get('accuracy', 0)
        gb_test_acc = gb_metrics['test'].get('accuracy', 0)
        nn_test_acc = nn_metrics['test'].get('accuracy', 0)
        logger.info("{:<20} {:<15.4f} {:<15.4f} {:<15.4f}".format("Test Accuracy", rf_test_acc, gb_test_acc, nn_test_acc))
        
        rf_f1 = rf_metrics['test'].get('f1_score', 0)
        gb_f1 = gb_metrics['test'].get('f1_score', 0)
        nn_f1 = nn_metrics['test'].get('f1_score', 0)
        logger.info("{:<20} {:<15.4f} {:<15.4f} {:<15.4f}".format("Test F1 Score", rf_f1, gb_f1, nn_f1))
        
        rf_time = rf_metrics['train'].get('training_time', 0)
        gb_time = gb_metrics['train'].get('training_time', 0)
        nn_time = nn_metrics['train'].get('training_time', 0)
        logger.info("{:<20} {:<15.2f}s {:<15.2f}s {:<15.2f}s".format("Training Time", rf_time, gb_time, nn_time))
        
        logger.info("\n" + "=" * 80)
        
        # Determine winner
        accuracies = {'Random Forest': rf_test_acc, 'Gradient Boosting': gb_test_acc, 'Neural Network': nn_test_acc}
        best_model = max(accuracies, key=accuracies.get)
        logger.info(f"🏆 {best_model} model performed best with {accuracies[best_model]:.4f} accuracy!")


def main():
    """Main execution function"""
    logger.info("=" * 80)
    logger.info("Large-Scale Model Training Pipeline")
    logger.info("=" * 80)
    
    try:
        # Step 1: Generate large dataset
        raw_csv = generate_large_dataset(n_tracks=200, duration_min=10)
        
        # Step 2: Apply auto-labeling
        labeled_csv = apply_autolabeling(raw_csv)
        
        # Step 3: Train Random Forest
        rf_model, rf_metrics = train_random_forest(labeled_csv)
        
        # Step 4: Train Gradient Boosting
        gb_model, gb_metrics = train_gradient_boosting(labeled_csv)
        
        # Step 5: Train Neural Network
        nn_model, nn_metrics = train_neural_network(labeled_csv)
        
        # Step 6: Compare models
        compare_models(rf_metrics, gb_metrics, nn_metrics)
        
        logger.info("=" * 80)
        logger.info("✓ Pipeline completed successfully!")
        logger.info("=" * 80)
        logger.info(f"\nGenerated files:")
        logger.info(f"  Raw data: {raw_csv}")
        logger.info(f"  Labeled data: {labeled_csv}")
        logger.info(f"  Random Forest model: output/models/random_forest_large/")
        logger.info(f"  Gradient Boosting model: output/models/gradient_boosting_large/")
        logger.info(f"  Neural Network model: output/models/neural_network_large/")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
