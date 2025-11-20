#!/usr/bin/env python3
"""
Utility to analyze labeled data and provide suggestions for improving label diversity.
This helps diagnose why all data points have the same annotation.
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path


def analyze_label_diversity(csv_path):
    """Analyze a labeled CSV file to understand why labels might be uniform
    
    Args:
        csv_path: Path to labeled CSV file
    """
    print(f"\n{'='*80}")
    print(f"Label Diversity Analysis: {csv_path}")
    print(f"{'='*80}\n")
    
    # Load data
    try:
        df = pd.read_csv(csv_path)
        print(f"✓ Loaded {len(df)} rows from CSV")
    except Exception as e:
        print(f"✗ Error loading CSV: {e}")
        return False
    
    # Check annotations
    if 'Annotation' not in df.columns:
        print(f"✗ No 'Annotation' column found")
        return False
    
    unique_annotations = df['Annotation'].unique()
    print(f"\n📊 ANNOTATION ANALYSIS")
    print(f"{'='*80}")
    print(f"Unique annotations: {len(unique_annotations)}")
    print(f"\nAnnotation distribution:")
    
    annotation_counts = df['Annotation'].value_counts()
    for annotation, count in annotation_counts.items():
        percentage = (count / len(df)) * 100
        print(f"  '{annotation}': {count} rows ({percentage:.1f}%)")
    
    # If only one unique annotation, this is the problem
    if len(unique_annotations) == 1:
        print(f"\n⚠️  PROBLEM IDENTIFIED: All rows have the same annotation!")
        print(f"   This makes machine learning impossible - models need variety.\n")
        
        # Analyze the individual flags
        analyze_individual_flags(df)
        
        # Analyze feature distributions
        analyze_feature_distributions(df)
        
        # Provide solutions
        provide_solutions(df)
        
        return False
    elif len(unique_annotations) < 5:
        print(f"\n⚠️  WARNING: Very few unique annotations ({len(unique_annotations)})")
        print(f"   More diversity would improve model performance.\n")
    else:
        print(f"\n✓ Good diversity: {len(unique_annotations)} unique annotations")
    
    return True


def analyze_individual_flags(df):
    """Analyze individual boolean flags to see if they vary"""
    print(f"\n📋 INDIVIDUAL FLAG ANALYSIS")
    print(f"{'='*80}")
    
    flag_columns = [
        'incoming', 'outgoing', 'fixed_range_ascending', 'fixed_range_descending',
        'level_flight', 'linear', 'curved', 'light_maneuver', 'high_maneuver',
        'low_speed', 'high_speed'
    ]
    
    flags_present = [col for col in flag_columns if col in df.columns]
    
    if not flags_present:
        print("  No individual flag columns found in data")
        print("  → This is expected if using composite annotations only")
        return
    
    print("Flag distributions (% of rows where flag is True):")
    for flag in flags_present:
        true_count = df[flag].sum()
        percentage = (true_count / len(df)) * 100
        print(f"  {flag:25s}: {true_count:6d} rows ({percentage:5.1f}%)")
    
    # Check if any flags vary
    varying_flags = [flag for flag in flags_present 
                     if 0 < df[flag].sum() < len(df)]
    
    if not varying_flags:
        print(f"\n⚠️  No flags show variation (all are always True or always False)")
        print(f"   This means your data has uniform motion characteristics")
    else:
        print(f"\n✓ Flags with variation: {len(varying_flags)}")
        print(f"   → These could be used for classification instead of composite labels")


def analyze_feature_distributions(df):
    """Analyze feature value distributions"""
    print(f"\n📈 FEATURE DISTRIBUTION ANALYSIS")
    print(f"{'='*80}")
    
    feature_columns = ['speed', 'range', 'curvature', 'accel_magnitude', 
                      'altitude_change', 'range_rate', 'heading']
    
    features_present = [col for col in feature_columns if col in df.columns]
    
    if not features_present:
        print("  No feature columns found")
        return
    
    print("Feature statistics (to understand why all labels are the same):\n")
    
    for feature in features_present:
        values = df[feature].dropna()
        if len(values) == 0:
            continue
        
        min_val = values.min()
        max_val = values.max()
        mean_val = values.mean()
        std_val = values.std()
        
        print(f"  {feature:20s}: min={min_val:8.2f}, max={max_val:8.2f}, "
              f"mean={mean_val:8.2f}, std={std_val:8.2f}")
        
        # Check if feature is essentially constant
        if std_val < 0.01 * abs(mean_val) and mean_val != 0:
            print(f"    ⚠️  Near-constant feature (low variation)")
        elif min_val == max_val:
            print(f"    ⚠️  Constant feature (no variation at all)")
    
    print()


def provide_solutions(df):
    """Provide actionable solutions based on the analysis"""
    print(f"\n💡 SOLUTIONS")
    print(f"{'='*80}\n")
    
    print("Your data has only ONE unique annotation, which prevents ML training.")
    print("Here are your options:\n")
    
    # Solution 1: Use per-track labels
    if 'trackid' in df.columns:
        num_tracks = df['trackid'].nunique()
        print(f"1️⃣  USE PER-TRACK LABELS (Recommended)")
        print(f"   - Your data has {num_tracks} track(s)")
        print(f"   - Instead of labeling every point, label whole tracks")
        print(f"   - Create a simplified label like 'incoming' or 'level_flight'")
        print(f"   - Use the track-based labeling script (see below)")
        print()
    
    # Solution 2: Adjust thresholds
    print(f"2️⃣  ADJUST AUTO-LABELING THRESHOLDS")
    print(f"   - Your data may need different threshold values")
    print(f"   - Edit config/default_config.json → 'autolabel_thresholds'")
    print(f"   - Example: Lower 'low_speed_threshold' from 50 to 20")
    print(f"   - Re-run auto-labeling with adjusted thresholds")
    print()
    
    # Solution 3: Use individual flags
    flag_columns = ['incoming', 'outgoing', 'level_flight', 'linear', 
                   'curved', 'light_maneuver', 'high_maneuver', 'low_speed', 'high_speed']
    flags_present = [col for col in flag_columns if col in df.columns]
    
    if flags_present:
        # Check if any individual flag varies
        varying_flags = [flag for flag in flags_present 
                        if col in df.columns and 0 < df[flag].sum() < len(df)]
        
        if varying_flags:
            print(f"3️⃣  USE INDIVIDUAL FLAGS FOR CLASSIFICATION")
            print(f"   - The following flags show variation in your data:")
            for flag in varying_flags:
                true_pct = (df[flag].sum() / len(df)) * 100
                print(f"     • {flag}: {true_pct:.1f}% True, {100-true_pct:.1f}% False")
            print(f"   - Use the flag splitting script (see below)")
            print()
    
    # Solution 4: Get more diverse data
    print(f"4️⃣  COLLECT MORE DIVERSE DATA")
    print(f"   - Current data may represent only one type of motion")
    print(f"   - Include trajectories with:")
    print(f"     • Different speeds (slow, medium, fast)")
    print(f"     • Different directions (incoming, outgoing)")
    print(f"     • Different flight patterns (level, ascending, descending)")
    print(f"     • Different maneuvers (straight, turning)")
    print()
    
    print(f"\n📝 QUICK FIX SCRIPT")
    print(f"{'='*80}")
    print(f"\nRun this to create a per-track labeled dataset:\n")
    print(f"  python create_track_labels.py '{sys.argv[1]}'\n")


def main():
    """Main entry point"""
    if len(sys.argv) < 2:
        print("Usage: python analyze_label_diversity.py <path_to_labeled_csv>")
        print("\nExample:")
        print("  python analyze_label_diversity.py data/labelled_data_1.csv")
        sys.exit(1)
    
    csv_path = sys.argv[1]
    
    if not Path(csv_path).exists():
        print(f"✗ File not found: {csv_path}")
        sys.exit(1)
    
    try:
        success = analyze_label_diversity(csv_path)
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
