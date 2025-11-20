#!/usr/bin/env python3
"""
Utility to create per-track labels from point-level annotations.
This aggregates all points in a track to a single label, which can help when
all points have the same annotation but you have multiple tracks.
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter


def extract_primary_label(composite_label):
    """Extract the most important tag from a composite label
    
    Args:
        composite_label: String like 'incoming,level,linear,light_maneuver,low_speed'
        
    Returns:
        Simplified label based on priority
    """
    if pd.isna(composite_label) or composite_label == '':
        return 'unknown'
    
    tags = [tag.strip() for tag in str(composite_label).split(',')]
    
    # Priority order for classification
    # Direction is most important
    if 'incoming' in tags:
        return 'incoming'
    elif 'outgoing' in tags:
        return 'outgoing'
    
    # Then vertical motion
    if 'ascending' in tags:
        return 'ascending'
    elif 'descending' in tags:
        return 'descending'
    elif 'level' in tags or 'level_flight' in tags:
        return 'level'
    
    # Then path shape
    if 'curved' in tags:
        return 'curved'
    elif 'linear' in tags:
        return 'linear'
    
    # Then maneuver intensity
    if 'high_maneuver' in tags:
        return 'high_maneuver'
    elif 'light_maneuver' in tags:
        return 'light_maneuver'
    
    # Finally speed
    if 'high_speed' in tags:
        return 'high_speed'
    elif 'low_speed' in tags:
        return 'low_speed'
    
    # Default
    return 'normal'


def create_track_labels(input_path, output_path=None, strategy='primary'):
    """Create per-track labels from point-level data
    
    Args:
        input_path: Path to labeled CSV file
        output_path: Path to save output (optional, defaults to input_path with _track_labeled suffix)
        strategy: Label creation strategy:
            - 'primary': Extract most important tag from composite label
            - 'majority': Use most common annotation in track
            - 'first': Use annotation from first point
            - 'last': Use annotation from last point
    """
    print(f"\n{'='*80}")
    print(f"Creating Per-Track Labels")
    print(f"{'='*80}\n")
    print(f"Input: {input_path}")
    print(f"Strategy: {strategy}\n")
    
    # Load data
    df = pd.read_csv(input_path)
    print(f"✓ Loaded {len(df)} rows")
    
    if 'trackid' not in df.columns:
        print(f"✗ Error: 'trackid' column not found")
        return None
    
    if 'Annotation' not in df.columns:
        print(f"✗ Error: 'Annotation' column not found")
        return None
    
    num_tracks = df['trackid'].nunique()
    print(f"✓ Found {num_tracks} unique tracks")
    
    # Create track-level labels
    df_out = df.copy()
    track_labels = {}
    
    for trackid in df['trackid'].unique():
        track_df = df[df['trackid'] == trackid]
        
        if strategy == 'primary':
            # Extract primary tag from composite labels
            annotations = track_df['Annotation'].dropna()
            if len(annotations) > 0:
                # Use most common annotation in track
                most_common = annotations.mode()[0] if len(annotations) > 0 else 'unknown'
                # Extract primary label
                label = extract_primary_label(most_common)
            else:
                label = 'unknown'
                
        elif strategy == 'majority':
            # Use most common annotation
            annotations = track_df['Annotation'].dropna()
            label = annotations.mode()[0] if len(annotations) > 0 else 'unknown'
            
        elif strategy == 'first':
            # Use first point's annotation
            label = track_df.iloc[0]['Annotation'] if len(track_df) > 0 else 'unknown'
            if pd.isna(label):
                label = 'unknown'
                
        elif strategy == 'last':
            # Use last point's annotation
            label = track_df.iloc[-1]['Annotation'] if len(track_df) > 0 else 'unknown'
            if pd.isna(label):
                label = 'unknown'
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        track_labels[trackid] = label
    
    # Apply track labels to all points
    df_out['Track_Label'] = df_out['trackid'].map(track_labels)
    
    # Replace Annotation with Track_Label
    df_out['Annotation'] = df_out['Track_Label']
    df_out = df_out.drop(columns=['Track_Label'])
    
    # Print label distribution
    print(f"\n📊 Track Label Distribution:")
    label_counts = df_out.groupby('trackid')['Annotation'].first().value_counts()
    
    for label, count in label_counts.items():
        percentage = (count / num_tracks) * 100
        print(f"  {label:20s}: {count:3d} tracks ({percentage:5.1f}%)")
    
    unique_labels = len(label_counts)
    print(f"\n✓ Created {unique_labels} unique label(s) from {num_tracks} tracks")
    
    # Save output
    if output_path is None:
        input_stem = Path(input_path).stem
        input_parent = Path(input_path).parent
        output_path = input_parent / f"{input_stem}_track_labeled.csv"
    
    df_out.to_csv(output_path, index=False)
    print(f"✓ Saved to: {output_path}")
    
    # Check if suitable for training
    if unique_labels < 2:
        print(f"\n⚠️  WARNING: Only {unique_labels} unique label(s)")
        print(f"   ML models need at least 2 different classes")
        print(f"   Your tracks all have the same label - you need more diverse data")
    elif unique_labels >= 2:
        print(f"\n✅ READY FOR TRAINING!")
        print(f"   You have {unique_labels} classes from {num_tracks} tracks")
        print(f"   This data can now be used to train ML models")
        
        if num_tracks < 10:
            print(f"\n💡 TIP: {num_tracks} tracks is quite small")
            print(f"   For better model performance, collect 20+ tracks")
    
    return df_out


def main():
    """Main entry point"""
    if len(sys.argv) < 2:
        print("Usage: python create_track_labels.py <input_csv> [output_csv] [strategy]")
        print("\nStrategies:")
        print("  primary  - Extract most important tag from composite label (default)")
        print("  majority - Use most common annotation in track")
        print("  first    - Use annotation from first point")
        print("  last     - Use annotation from last point")
        print("\nExample:")
        print("  python create_track_labels.py data/labelled_data_1.csv")
        print("  python create_track_labels.py data/labelled_data_1.csv output.csv primary")
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 and not sys.argv[2] in ['primary', 'majority', 'first', 'last'] else None
    strategy = sys.argv[3] if len(sys.argv) > 3 else (sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] in ['primary', 'majority', 'first', 'last'] else 'primary')
    
    if not Path(input_path).exists():
        print(f"✗ File not found: {input_path}")
        sys.exit(1)
    
    try:
        df = create_track_labels(input_path, output_path, strategy)
        sys.exit(0 if df is not None else 1)
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
