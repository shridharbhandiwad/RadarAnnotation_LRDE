"""Train Multi-Output Models for Auto-Tagging

This script trains XGBoost, Random Forest, and Transformer models in multi-output mode
for auto-tagging and auto-annotation based on the data format:
- Columns A-K: Input features (radar measurements)
- Columns L-AF: Output tags (to be predicted)
- Column AG: Aggregated annotation (reference)

All three models will predict multiple tag columns simultaneously.

IMPORTANT: This script now delegates to the ai_engine.train_multi_output_models() function.
           The main training logic is now integrated into the AI engine.
"""

import logging
import sys

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

from src.ai_engine import train_multi_output_models


def main():
    """Main execution function - delegates to ai_engine.train_multi_output_models()"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Multi-Output Models for Auto-Tagging')
    parser.add_argument('--data', default='data/high_volume_simulation_labeled.csv',
                       help='Path to labeled CSV dataset')
    parser.add_argument('--output', default='output/multi_output_models',
                       help='Output directory for trained models')
    
    args = parser.parse_args()
    
    try:
        # Call the integrated function from ai_engine
        return train_multi_output_models(args.data, args.output)
    except Exception as e:
        logger.error(f"Training pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
