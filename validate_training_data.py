#!/usr/bin/env python3
"""
Utility script to validate training data CSV files before attempting to train models.
This helps identify issues early and provides clear guidance on what needs to be fixed.
"""

import sys
import os
from pathlib import Path
import pandas as pd


def validate_training_data(csv_path):
    """Validate a CSV file for ML model training
    
    Args:
        csv_path: Path to CSV file
        
    Returns:
        bool: True if valid, False otherwise
    """
    print(f"\n{'='*80}")
    print(f"Validating Training Data: {csv_path}")
    print(f"{'='*80}\n")
    
    # Check 1: File exists
    print("✓ Checking file existence...")
    if not Path(csv_path).exists():
        print(f"  ✗ FAILED: File not found at: {csv_path}")
        print(f"\n  Suggestions:")
        print(f"  - Verify the file path is correct")
        print(f"  - If copying from Windows, ensure the path is updated for Linux")
        print(f"  - Check for typos in the filename")
        return False
    print(f"  ✓ File exists")
    
    # Check 2: File readable
    print("\n✓ Checking file permissions...")
    if not os.access(csv_path, os.R_OK):
        print(f"  ✗ FAILED: File is not readable")
        print(f"\n  Suggestions:")
        print(f"  - Check file permissions: chmod +r {csv_path}")
        print(f"  - Verify you have read access to the directory")
        return False
    print(f"  ✓ File is readable")
    
    # Check 3: Read CSV
    print("\n✓ Reading CSV file...")
    try:
        df = pd.read_csv(csv_path)
        print(f"  ✓ CSV loaded successfully")
    except Exception as e:
        print(f"  ✗ FAILED: Could not read CSV file")
        print(f"  Error: {str(e)}")
        print(f"\n  Suggestions:")
        print(f"  - Ensure the file is a valid CSV format")
        print(f"  - Check for encoding issues (should be UTF-8)")
        print(f"  - Verify the file is not corrupted")
        return False
    
    # Check 4: Not empty
    print("\n✓ Checking data content...")
    if len(df) == 0:
        print(f"  ✗ FAILED: CSV file is empty (0 rows)")
        print(f"\n  Suggestions:")
        print(f"  - Ensure the CSV contains data rows")
        print(f"  - Check if data extraction completed successfully")
        return False
    print(f"  ✓ Contains {len(df)} rows")
    
    # Check 5: Required columns
    print("\n✓ Checking required columns...")
    required_columns = ['trackid', 'Annotation']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        print(f"  ✗ FAILED: Missing required columns: {missing_columns}")
        print(f"\n  Available columns: {list(df.columns)}")
        print(f"\n  Suggestions:")
        if 'trackid' in missing_columns:
            print(f"  - 'trackid' column is required to group trajectory data")
            print(f"  - Run data extraction or ensure your CSV has track identifiers")
        if 'Annotation' in missing_columns:
            print(f"  - 'Annotation' column contains the labels for training")
            print(f"  - Run auto-labeling engine to generate annotations")
            print(f"  - Manually add annotations to your CSV")
        return False
    print(f"  ✓ All required columns present: {required_columns}")
    
    # Check 6: Track IDs
    print("\n✓ Analyzing tracks...")
    track_ids = df['trackid'].unique()
    n_tracks = len(track_ids)
    print(f"  ✓ Found {n_tracks} unique track(s)")
    
    if n_tracks == 0:
        print(f"  ✗ WARNING: No tracks found")
        return False
    elif n_tracks < 3:
        print(f"  ⚠ WARNING: Only {n_tracks} track(s) available")
        print(f"    - Training will use all data without validation/test split")
        print(f"    - For proper evaluation, at least 3 tracks recommended")
        print(f"    - For good performance, 10+ tracks recommended")
    else:
        print(f"  ✓ Sufficient tracks for train/validation/test split")
    
    # Check 7: Annotations
    print("\n✓ Checking annotations...")
    if 'Annotation' in df.columns:
        annotations = df['Annotation'].value_counts()
        print(f"  Annotation distribution:")
        for label, count in annotations.items():
            percentage = (count / len(df)) * 100
            print(f"    - {label}: {count} ({percentage:.1f}%)")
        
        # Check for balanced classes
        if len(annotations) == 1:
            print(f"  ⚠ WARNING: Only one class present in data")
            print(f"    - Model may not generalize well")
            print(f"    - Consider collecting more diverse data")
        
        # Check for very imbalanced data
        min_count = annotations.min()
        max_count = annotations.max()
        if max_count / min_count > 10:
            print(f"  ⚠ WARNING: Highly imbalanced classes detected")
            print(f"    - Consider balancing your dataset")
            print(f"    - Model may be biased towards majority class")
    
    # Check 8: Feature columns (optional check)
    print("\n✓ Checking feature columns...")
    common_features = ['x', 'y', 'z', 'vx', 'vy', 'vz', 'speed', 'heading', 'range']
    available_features = [col for col in common_features if col in df.columns]
    
    if len(available_features) < 3:
        print(f"  ⚠ WARNING: Very few feature columns detected")
        print(f"    Available features: {available_features}")
        print(f"    - Consider adding more features for better model performance")
    else:
        print(f"  ✓ Found {len(available_features)} common feature columns")
    
    # Summary
    print(f"\n{'='*80}")
    print(f"✓ VALIDATION PASSED")
    print(f"{'='*80}")
    print(f"\nSummary:")
    print(f"  - File: {csv_path}")
    print(f"  - Rows: {len(df)}")
    print(f"  - Tracks: {n_tracks}")
    print(f"  - Columns: {len(df.columns)}")
    print(f"  - Classes: {len(annotations) if 'Annotation' in df.columns else 'N/A'}")
    print(f"\n✓ This file is ready for training!")
    
    if n_tracks < 3:
        print(f"\n⚠ RECOMMENDATION: Collect more tracks (3+) for better model evaluation")
    
    return True


def main():
    """Main entry point"""
    if len(sys.argv) < 2:
        print("Usage: python validate_training_data.py <path_to_csv>")
        print("\nExample:")
        print("  python validate_training_data.py data/labeled_data.csv")
        sys.exit(1)
    
    csv_path = sys.argv[1]
    
    try:
        is_valid = validate_training_data(csv_path)
        sys.exit(0 if is_valid else 1)
    except Exception as e:
        print(f"\n✗ UNEXPECTED ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
