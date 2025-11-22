"""AutoLabeling Engine - Rule-based motion feature extraction and annotation"""
import pandas as pd
import numpy as np
import logging
from typing import Dict, Any
from .utils import compute_speed, compute_heading, compute_curvature, compute_acceleration_magnitude
from .config import get_config

logger = logging.getLogger(__name__)


def compute_motion_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute motion features from position and velocity data
    
    Args:
        df: DataFrame with columns: trackid, time, x, y, z, vx, vy, vz, ax, ay, az
        
    Returns:
        DataFrame with additional computed features
    """
    if len(df) == 0:
        return df
    
    df = df.copy()
    df = df.sort_values(['trackid', 'time']).reset_index(drop=True)
    
    config = get_config()
    min_points = config.get('autolabel_thresholds.min_points_per_track', 3)
    
    # Initialize feature columns
    df['speed'] = 0.0
    df['speed_2d'] = 0.0
    df['heading'] = 0.0
    df['range'] = 0.0
    df['range_rate'] = 0.0
    df['curvature'] = 0.0
    df['accel_magnitude'] = 0.0
    df['vertical_rate'] = 0.0
    df['altitude_change'] = 0.0
    df['valid_features'] = False
    
    # Process each track separately
    for trackid in df['trackid'].unique():
        track_mask = df['trackid'] == trackid
        track_df = df[track_mask].copy()
        
        if len(track_df) < min_points:
            logger.warning(f"Track {trackid} has only {len(track_df)} points, skipping feature computation")
            continue
        
        # Compute speed
        speed = compute_speed(track_df['vx'].values, track_df['vy'].values, track_df['vz'].values)
        speed_2d = compute_speed(track_df['vx'].values, track_df['vy'].values, np.zeros_like(track_df['vz'].values))
        
        # Compute heading
        heading = compute_heading(track_df['vx'].values, track_df['vy'].values)
        
        # Compute range from radar origin (assumed at 0,0,0)
        range_val = np.sqrt(track_df['x']**2 + track_df['y']**2 + track_df['z']**2)
        
        # Compute range rate (derivative of range)
        dt = track_df['time'].diff().fillna(1.0)
        range_rate = range_val.diff() / dt
        range_rate = range_rate.fillna(0)
        
        # Compute curvature
        curvature = compute_curvature(track_df)
        
        # Compute acceleration magnitude
        accel_mag = compute_acceleration_magnitude(track_df['ax'].values, track_df['ay'].values, track_df['az'].values)
        
        # Compute vertical rate
        vertical_rate = track_df['z'].diff() / dt
        vertical_rate = vertical_rate.fillna(0)
        
        # Compute cumulative altitude change
        altitude_change = track_df['z'] - track_df['z'].iloc[0]
        
        # Mark first 2 points as invalid for derived features
        valid_features = np.ones(len(track_df), dtype=bool)
        valid_features[:2] = False
        
        # Assign computed features back to main dataframe
        df.loc[track_mask, 'speed'] = speed
        df.loc[track_mask, 'speed_2d'] = speed_2d
        df.loc[track_mask, 'heading'] = heading
        df.loc[track_mask, 'range'] = range_val.values
        df.loc[track_mask, 'range_rate'] = range_rate.values
        df.loc[track_mask, 'curvature'] = curvature.values
        df.loc[track_mask, 'accel_magnitude'] = accel_mag
        df.loc[track_mask, 'vertical_rate'] = vertical_rate.values
        df.loc[track_mask, 'altitude_change'] = altitude_change.values
        df.loc[track_mask, 'valid_features'] = valid_features
    
    logger.info(f"Computed motion features for {len(df)} records across {df['trackid'].nunique()} tracks")
    
    return df


def apply_rules_and_flags(df: pd.DataFrame, rules_config: Dict[str, Any] = None) -> pd.DataFrame:
    """Apply rule-based classification flags
    
    Args:
        df: DataFrame with computed motion features
        rules_config: Configuration for rule thresholds (uses config default if None)
        
    Returns:
        DataFrame with boolean flag columns and combined Annotation column
    """
    if len(df) == 0:
        return df
    
    if rules_config is None:
        config = get_config()
        rules_config = config.get('autolabel_thresholds')
    
    df = df.copy()
    
    # Extract thresholds
    level_flight_thresh = rules_config.get('level_flight_threshold', 5.0)
    curvature_thresh = rules_config.get('curvature_threshold', 0.01)
    low_speed_thresh = rules_config.get('low_speed_threshold', 50.0)
    high_speed_thresh = rules_config.get('high_speed_threshold', 200.0)
    light_maneuver_thresh = rules_config.get('light_maneuver_threshold', 2.0)
    high_maneuver_thresh = rules_config.get('high_maneuver_threshold', 5.0)
    range_rate_thresh = rules_config.get('range_rate_threshold', 1.0)
    fixed_range_thresh = rules_config.get('fixed_range_threshold', 10.0)
    
    # Only apply rules to points with valid features
    valid_mask = df.get('valid_features', True)
    
    # Initialize all flags as False
    df['incoming'] = False
    df['outgoing'] = False
    df['fixed_range_ascending'] = False
    df['fixed_range_descending'] = False
    df['level_flight'] = False
    df['linear'] = False
    df['curved'] = False
    df['light_maneuver'] = False
    df['high_maneuver'] = False
    df['low_speed'] = False
    df['high_speed'] = False
    
    # Apply rules only to valid points
    if valid_mask.any():
        # Incoming/Outgoing based on range rate
        df.loc[valid_mask & (df['range_rate'] < -range_rate_thresh), 'incoming'] = True
        df.loc[valid_mask & (df['range_rate'] > range_rate_thresh), 'outgoing'] = True
        
        # Fixed range with vertical motion
        small_lateral_motion = df['speed_2d'] < low_speed_thresh
        df.loc[valid_mask & small_lateral_motion & (df['vertical_rate'] > 1.0), 'fixed_range_ascending'] = True
        df.loc[valid_mask & small_lateral_motion & (df['vertical_rate'] < -1.0), 'fixed_range_descending'] = True
        
        # Level flight
        df.loc[valid_mask & (np.abs(df['altitude_change']) < level_flight_thresh), 'level_flight'] = True
        
        # Linear vs Curved motion
        df.loc[valid_mask & (df['curvature'] < curvature_thresh), 'linear'] = True
        df.loc[valid_mask & (df['curvature'] >= curvature_thresh), 'curved'] = True
        
        # Maneuver intensity based on acceleration
        df.loc[valid_mask & (df['accel_magnitude'] < light_maneuver_thresh), 'light_maneuver'] = True
        df.loc[valid_mask & (df['accel_magnitude'] >= high_maneuver_thresh), 'high_maneuver'] = True
        
        # Speed categories
        df.loc[valid_mask & (df['speed'] < low_speed_thresh), 'low_speed'] = True
        df.loc[valid_mask & (df['speed'] >= high_speed_thresh), 'high_speed'] = True
    
    # Create combined annotation string
    df['Annotation'] = ''
    
    for idx in df.index:
        if not df.loc[idx, 'valid_features']:
            df.loc[idx, 'Annotation'] = 'invalid'
            continue
        
        tags = []
        
        # Direction
        if df.loc[idx, 'incoming']:
            tags.append('incoming')
        elif df.loc[idx, 'outgoing']:
            tags.append('outgoing')
        
        # Vertical motion
        if df.loc[idx, 'fixed_range_ascending']:
            tags.append('fixed_range_ascending')
        elif df.loc[idx, 'fixed_range_descending']:
            tags.append('fixed_range_descending')
        elif df.loc[idx, 'level_flight']:
            tags.append('level_flight')
        
        # Path shape
        if df.loc[idx, 'linear']:
            tags.append('linear')
        elif df.loc[idx, 'curved']:
            tags.append('curved')
        
        # Maneuver intensity
        if df.loc[idx, 'high_maneuver']:
            tags.append('high_maneuver')
        elif df.loc[idx, 'light_maneuver']:
            tags.append('light_maneuver')
        
        # Speed
        if df.loc[idx, 'low_speed']:
            tags.append('low_speed')
        elif df.loc[idx, 'high_speed']:
            tags.append('high_speed')
        
        df.loc[idx, 'Annotation'] = ','.join(tags) if tags else 'normal'
    
    logger.info(f"Applied rule-based flags to {len(df)} records")
    
    return df


def get_annotation_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """Get summary statistics of annotations
    
    Args:
        df: DataFrame with annotation flags
        
    Returns:
        Dictionary with annotation counts and percentages
    """
    flag_columns = ['incoming', 'outgoing', 'fixed_range_ascending', 'fixed_range_descending',
                   'level_flight', 'linear', 'curved', 'light_maneuver', 'high_maneuver',
                   'low_speed', 'high_speed']
    
    summary = {
        'total_records': len(df),
        'valid_records': df.get('valid_features', pd.Series([True]*len(df))).sum(),
        'flag_counts': {},
        'annotation_distribution': {}
    }
    
    # Count each flag
    for flag in flag_columns:
        if flag in df.columns:
            count = df[flag].sum()
            percentage = (count / len(df) * 100) if len(df) > 0 else 0
            summary['flag_counts'][flag] = {
                'count': int(count),
                'percentage': round(percentage, 2)
            }
    
    # Count annotation strings
    if 'Annotation' in df.columns:
        annotation_counts = df['Annotation'].value_counts()
        for annotation, count in annotation_counts.items():
            percentage = (count / len(df) * 100) if len(df) > 0 else 0
            summary['annotation_distribution'][annotation] = {
                'count': int(count),
                'percentage': round(percentage, 2)
            }
    
    return summary


def autolabel_pipeline(input_path: str, output_path: str, rules_config: Dict[str, Any] = None) -> pd.DataFrame:
    """Complete autolabeling pipeline
    
    Args:
        input_path: Path to input CSV file (raw data)
        output_path: Path to output CSV file (labeled data)
        rules_config: Rule configuration (optional)
        
    Returns:
        Labeled DataFrame
    """
    logger.info(f"Starting autolabeling pipeline: {input_path} -> {output_path}")
    
    # Load data
    df = pd.read_csv(input_path)
    
    # Compute features
    df = compute_motion_features(df)
    
    # Apply rules
    df = apply_rules_and_flags(df, rules_config)
    
    # Save labeled data
    df.to_csv(output_path, index=False)
    logger.info(f"Saved labeled data to {output_path}")
    
    # Print summary
    summary = get_annotation_summary(df)
    logger.info(f"Annotation summary: {summary['valid_records']}/{summary['total_records']} valid records")
    
    return df


# CLI interface
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='AutoLabeling Engine')
    parser.add_argument('--input', required=True, help='Input CSV file (raw data)')
    parser.add_argument('--out', required=True, help='Output CSV file (labeled data)')
    parser.add_argument('--config', help='Config JSON file for thresholds (optional)')
    
    args = parser.parse_args()
    
    # Load config if provided
    rules_config = None
    if args.config:
        import json
        with open(args.config, 'r') as f:
            rules_config = json.load(f).get('autolabel_thresholds')
    
    # Run pipeline
    df = autolabel_pipeline(args.input, args.out, rules_config)
    
    # Print summary
    summary = get_annotation_summary(df)
    print("\nAnnotation Summary:")
    print(f"  Total records: {summary['total_records']}")
    print(f"  Valid records: {summary['valid_records']}")
    print("\nFlag Counts:")
    for flag, data in summary['flag_counts'].items():
        print(f"  {flag}: {data['count']} ({data['percentage']}%)")
