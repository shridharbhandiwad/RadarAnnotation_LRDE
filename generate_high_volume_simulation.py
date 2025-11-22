"""Generate High-Volume Simulation Dataset

This script generates:
- 1 single track with ALL trajectory types
- 100ms sampling interval (0.1 seconds)
- 60 minutes (1 hour) flight time
- Total: ~36,000 data points

The dataset includes all possible trajectory types for better model accuracy.
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

from src.sim_engine import create_single_track_all_trajectories
from src.autolabel_engine import compute_motion_features, apply_rules_and_flags, get_annotation_summary


def generate_high_volume_dataset():
    """Generate high-volume simulation dataset with single track
    
    Parameters:
    - 1 single track with ALL trajectory types
    - 100ms interval (configured in config)
    - 60 minutes (1 hour) total flight time
    - 10 different trajectory types, 6 minutes each
    - Total: ~36,000 data points
    
    Returns:
        Tuple of (raw_csv_path, labeled_csv_path)
    """
    logger.info("=" * 80)
    logger.info("HIGH-VOLUME SIMULATION DATASET GENERATION")
    logger.info("=" * 80)
    logger.info("Configuration:")
    logger.info("  - Tracks: 1 (single track with all trajectory types)")
    logger.info("  - Sample Interval: 100ms (0.1 seconds)")
    logger.info("  - Total flight duration: 60 minutes (1 hour)")
    logger.info("  - Trajectory types: 10 (6 minutes each)")
    logger.info("  - Expected data points: ~36,000")
    logger.info("=" * 80)
    
    # Generate raw simulation data
    output_path = "data/high_volume_simulation.csv"
    
    logger.info("\nSTEP 1: Generating raw simulation data...")
    csv_path = create_single_track_all_trajectories(
        output_path=output_path,
        duration_min=60
    )
    
    # Load and verify
    df = pd.read_csv(csv_path)
    logger.info(f"\n✓ Raw simulation dataset created: {csv_path}")
    logger.info(f"  Total records: {len(df):,}")
    logger.info(f"  Track ID: {df['trackid'].unique()[0]}")
    logger.info(f"  Duration: {df['time'].max():.2f} seconds ({df['time'].max()/60:.2f} minutes)")
    
    # Calculate time intervals
    time_diffs = df['time'].diff().dropna()
    avg_interval_ms = time_diffs.mean() * 1000
    logger.info(f"  Avg sampling interval: {avg_interval_ms:.1f}ms")
    logger.info(f"  Total data points: {len(df):,}")
    
    # Apply auto-labeling
    logger.info("\nSTEP 2: Applying auto-labeling...")
    df = compute_motion_features(df)
    df = apply_rules_and_flags(df)
    
    # Get labeling summary
    summary = get_annotation_summary(df)
    logger.info(f"\n✓ Auto-labeling completed")
    logger.info(f"  Valid records: {summary['valid_records']:,}/{summary['total_records']:,}")
    logger.info(f"  Unique annotations: {len(summary['annotation_distribution'])}")
    
    # Show top annotations
    logger.info("\n  Top 15 Annotations:")
    for annotation, data in list(summary['annotation_distribution'].items())[:15]:
        logger.info(f"    {annotation}: {data['count']:,} ({data['percentage']:.2f}%)")
    
    # Save labeled dataset
    labeled_path = csv_path.replace('.csv', '_labeled.csv')
    df.to_csv(labeled_path, index=False)
    logger.info(f"\n✓ Labeled dataset saved: {labeled_path}")
    
    # Generate dataset summary
    logger.info("\n" + "=" * 80)
    logger.info("DATASET SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Raw Dataset: {csv_path}")
    logger.info(f"  - Size: {Path(csv_path).stat().st_size / (1024*1024):.2f} MB")
    logger.info(f"  - Records: {len(df):,}")
    logger.info(f"\nLabeled Dataset: {labeled_path}")
    logger.info(f"  - Size: {Path(labeled_path).stat().st_size / (1024*1024):.2f} MB")
    logger.info(f"  - Records: {len(df):,}")
    logger.info(f"  - Features: {len(df.columns)}")
    logger.info(f"\nData Split Recommendations:")
    logger.info(f"  - Training (70%): ~{int(len(df)*0.7):,} records")
    logger.info(f"  - Validation (15%): ~{int(len(df)*0.15):,} records")
    logger.info(f"  - Testing (15%): ~{int(len(df)*0.15):,} records")
    logger.info("=" * 80)
    
    return csv_path, labeled_path


def print_usage_examples(raw_csv, labeled_csv):
    """Print usage examples for the generated data"""
    logger.info("\n" + "=" * 80)
    logger.info("USAGE EXAMPLES")
    logger.info("=" * 80)
    logger.info("\n1. Train Transformer Model:")
    logger.info(f"   python -c \"from src.ai_engine import train_model; train_model('transformer', '{labeled_csv}', 'output/models/transformer')\"")
    
    logger.info("\n2. Train LSTM Model:")
    logger.info(f"   python -c \"from src.ai_engine import train_model; train_model('lstm', '{labeled_csv}', 'output/models/lstm')\"")
    
    logger.info("\n3. Train XGBoost Model:")
    logger.info(f"   python -c \"from src.ai_engine import train_model; train_model('xgboost', '{labeled_csv}', 'output/models/xgboost')\"")
    
    logger.info("\n4. Train and Compare Multiple Models:")
    logger.info(f"   python generate_and_train_large_dataset.py")
    logger.info(f"   (Then modify to use '{labeled_csv}')")
    
    logger.info("\n5. Load and Explore Data:")
    logger.info(f"   python -c \"import pandas as pd; df = pd.read_csv('{labeled_csv}'); print(df.info()); print(df.head())\"")
    
    logger.info("\n6. Visualize with GUI:")
    logger.info(f"   python -m src.gui")
    logger.info(f"   (Then load '{raw_csv}' in the GUI)")
    
    logger.info("\n" + "=" * 80)


def main():
    """Main execution function"""
    try:
        # Generate dataset
        raw_csv, labeled_csv = generate_high_volume_dataset()
        
        # Print usage examples
        print_usage_examples(raw_csv, labeled_csv)
        
        logger.info("\n✓ High-volume simulation dataset generation completed successfully!")
        logger.info("=" * 80)
        
        return 0
        
    except Exception as e:
        logger.error(f"✗ Dataset generation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
