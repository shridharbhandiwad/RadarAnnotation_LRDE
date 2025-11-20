#!/usr/bin/env python3
"""
Utility to split composite labels into individual binary classification tasks.
Instead of treating 'incoming,level,linear,light_maneuver,low_speed' as one class,
this creates separate label columns for each component.
"""

import sys
import pandas as pd
from pathlib import Path


def split_composite_labels(input_path, output_path=None, target_label=None):
    """Split composite labels into individual binary columns
    
    Args:
        input_path: Path to labeled CSV file
        output_path: Path to save output (optional)
        target_label: Specific label to extract (e.g., 'incoming', 'level'), or None for all
    """
    print(f"\n{'='*80}")
    print(f"Splitting Composite Labels into Binary Columns")
    print(f"{'='*80}\n")
    print(f"Input: {input_path}\n")
    
    # Load data
    df = pd.read_csv(input_path)
    print(f"✓ Loaded {len(df)} rows")
    
    if 'Annotation' not in df.columns:
        print(f"✗ Error: 'Annotation' column not found")
        return None
    
    # Get all unique tags across all annotations
    all_tags = set()
    for annotation in df['Annotation'].dropna():
        tags = [tag.strip() for tag in str(annotation).split(',')]
        all_tags.update(tags)
    
    all_tags = sorted([tag for tag in all_tags if tag and tag != 'invalid'])
    print(f"✓ Found {len(all_tags)} unique tags: {', '.join(all_tags)}\n")
    
    # Create binary columns for each tag
    df_out = df.copy()
    
    for tag in all_tags:
        df_out[f'label_{tag}'] = df_out['Annotation'].apply(
            lambda x: tag in str(x).split(',') if pd.notna(x) else False
        )
    
    # If target_label specified, create a simplified Annotation column
    if target_label:
        if target_label not in all_tags:
            print(f"⚠️  Warning: Target label '{target_label}' not found in data")
            print(f"   Available labels: {', '.join(all_tags)}")
        else:
            df_out['Annotation'] = df_out[f'label_{target_label}'].apply(
                lambda x: target_label if x else f'not_{target_label}'
            )
            print(f"✓ Created binary classification for '{target_label}'")
    
    # Print distribution for each label
    print(f"📊 Binary Label Distribution:")
    print(f"{'='*80}")
    
    for tag in all_tags:
        col = f'label_{tag}'
        true_count = df_out[col].sum()
        false_count = len(df_out) - true_count
        true_pct = (true_count / len(df_out)) * 100
        false_pct = 100 - true_pct
        
        print(f"\n  {tag}:")
        print(f"    True:  {true_count:6d} ({true_pct:5.1f}%)")
        print(f"    False: {false_count:6d} ({false_pct:5.1f}%)")
        
        # Check if suitable for training
        if true_count == 0 or false_count == 0:
            print(f"    ⚠️  Cannot use for training (only one class)")
        elif min(true_count, false_count) < 5:
            print(f"    ⚠️  Very imbalanced (minority class < 5 samples)")
        elif true_pct < 5 or true_pct > 95:
            print(f"    ⚠️  Highly imbalanced (consider balancing)")
        else:
            print(f"    ✓  Suitable for binary classification")
    
    # Save output
    if output_path is None:
        input_stem = Path(input_path).stem
        input_parent = Path(input_path).parent
        output_path = input_parent / f"{input_stem}_split.csv"
    
    df_out.to_csv(output_path, index=False)
    print(f"\n✓ Saved to: {output_path}")
    
    # Provide usage instructions
    print(f"\n💡 USAGE")
    print(f"{'='*80}")
    print(f"\nYou can now train separate models for each label:\n")
    
    usable_labels = [tag for tag in all_tags 
                    if 0 < df_out[f'label_{tag}'].sum() < len(df_out)]
    
    if usable_labels:
        print(f"Usable labels (with variation): {', '.join(usable_labels)}\n")
        print(f"To train a model for a specific label:")
        print(f"  1. Create a new CSV with that label as 'Annotation'")
        print(f"  2. For example, for 'incoming' classification:\n")
        print(f"     python split_composite_labels.py '{input_path}' output.csv incoming\n")
    else:
        print(f"⚠️  No labels have variation - all are always True or always False")
        print(f"   This means your data has uniform characteristics")
        print(f"   You need more diverse data for training\n")
    
    return df_out


def main():
    """Main entry point"""
    if len(sys.argv) < 2:
        print("Usage: python split_composite_labels.py <input_csv> [output_csv] [target_label]")
        print("\nExamples:")
        print("  # Analyze all labels")
        print("  python split_composite_labels.py data/labelled_data_1.csv")
        print()
        print("  # Create binary classification for 'incoming' vs 'not_incoming'")
        print("  python split_composite_labels.py data/labelled_data_1.csv output.csv incoming")
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    target_label = sys.argv[3] if len(sys.argv) > 3 else None
    
    if not Path(input_path).exists():
        print(f"✗ File not found: {input_path}")
        sys.exit(1)
    
    try:
        df = split_composite_labels(input_path, output_path, target_label)
        sys.exit(0 if df is not None else 1)
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
