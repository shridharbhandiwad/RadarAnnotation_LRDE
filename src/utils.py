"""Utility functions for Radar Annotation Application"""
import struct
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Any
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def read_binary_file(file_path: str, schema: Dict[str, Any]) -> List[Dict[str, float]]:
    """Read binary radar data file according to schema
    
    Args:
        file_path: Path to binary file
        schema: Schema dictionary with format information
        
    Returns:
        List of record dictionaries
    """
    records = []
    record_size = schema['record_size']
    struct_format = schema['struct_format']
    fields = schema['fields']
    
    try:
        with open(file_path, 'rb') as f:
            data = f.read()
            
        num_records = len(data) // record_size
        logger.info(f"Reading {num_records} records from {file_path}")
        
        for i in range(num_records):
            offset = i * record_size
            record_bytes = data[offset:offset + record_size]
            
            if len(record_bytes) < record_size:
                logger.warning(f"Incomplete record at position {i}, skipping")
                break
            
            values = struct.unpack(struct_format, record_bytes)
            
            # Map values to field names
            record = {field['name']: values[idx] for idx, field in enumerate(fields)}
            records.append(record)
            
    except Exception as e:
        logger.error(f"Error reading binary file: {e}")
        raise
    
    return records


def write_binary_file(file_path: str, records: List[Dict[str, float]], schema: Dict[str, Any]) -> None:
    """Write records to binary file according to schema
    
    Args:
        file_path: Path to output binary file
        records: List of record dictionaries
        schema: Schema dictionary with format information
    """
    struct_format = schema['struct_format']
    fields = schema['fields']
    
    try:
        with open(file_path, 'wb') as f:
            for record in records:
                values = tuple(record[field['name']] for field in fields)
                packed = struct.pack(struct_format, *values)
                f.write(packed)
                
        logger.info(f"Wrote {len(records)} records to {file_path}")
        
    except Exception as e:
        logger.error(f"Error writing binary file: {e}")
        raise


def cartesian_to_polar(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Convert Cartesian coordinates to polar (range, azimuth)
    
    Args:
        x: X coordinates (meters)
        y: Y coordinates (meters)
        
    Returns:
        Tuple of (range in meters, azimuth in degrees)
    """
    r = np.sqrt(x**2 + y**2)
    theta = np.degrees(np.arctan2(y, x))
    return r, theta


def polar_to_cartesian(r: np.ndarray, theta_deg: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Convert polar coordinates to Cartesian
    
    Args:
        r: Range in meters
        theta_deg: Azimuth in degrees
        
    Returns:
        Tuple of (x, y) in meters
    """
    theta_rad = np.radians(theta_deg)
    x = r * np.cos(theta_rad)
    y = r * np.sin(theta_rad)
    return x, y


def compute_speed(vx: np.ndarray, vy: np.ndarray, vz: np.ndarray) -> np.ndarray:
    """Compute 3D speed magnitude
    
    Args:
        vx, vy, vz: Velocity components
        
    Returns:
        Speed magnitude
    """
    return np.sqrt(vx**2 + vy**2 + vz**2)


def compute_acceleration_magnitude(ax: np.ndarray, ay: np.ndarray, az: np.ndarray) -> np.ndarray:
    """Compute 3D acceleration magnitude
    
    Args:
        ax, ay, az: Acceleration components
        
    Returns:
        Acceleration magnitude
    """
    return np.sqrt(ax**2 + ay**2 + az**2)


def compute_heading(vx: np.ndarray, vy: np.ndarray) -> np.ndarray:
    """Compute heading angle from velocity components
    
    Args:
        vx, vy: Velocity components
        
    Returns:
        Heading in degrees (0-360, clockwise from North)
    """
    heading = np.degrees(np.arctan2(vy, vx))
    # Convert to 0-360 range
    heading = (90 - heading) % 360
    return heading


def compute_curvature(df: pd.DataFrame) -> pd.Series:
    """Compute path curvature for trajectory
    
    Args:
        df: DataFrame with x, y, vx, vy columns (must be sorted by time)
        
    Returns:
        Series with curvature values
    """
    # Curvature = |v x a| / |v|^3
    # For 2D: curvature = (vx*ay - vy*ax) / (vx^2 + vy^2)^(3/2)
    
    curvature = np.zeros(len(df))
    
    if len(df) < 3:
        return pd.Series(curvature, index=df.index)
    
    # Compute acceleration from velocity
    dt = df['time'].diff().fillna(1.0)
    ax = df['vx'].diff() / dt
    ay = df['vy'].diff() / dt
    
    speed_2d = np.sqrt(df['vx']**2 + df['vy']**2)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        curvature = np.abs(df['vx'] * ay - df['vy'] * ax) / (speed_2d**3)
        curvature = np.nan_to_num(curvature, 0.0)
    
    return pd.Series(curvature, index=df.index)


def ensure_dir(path: str) -> Path:
    """Ensure directory exists, create if not
    
    Args:
        path: Directory path
        
    Returns:
        Path object
    """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def get_track_statistics(df: pd.DataFrame, trackid: float) -> Dict[str, Any]:
    """Get statistics for a specific track
    
    Args:
        df: DataFrame with track data
        trackid: Track ID to analyze
        
    Returns:
        Dictionary of statistics
    """
    track_df = df[df['trackid'] == trackid].copy()
    
    if len(track_df) == 0:
        return {}
    
    track_df = track_df.sort_values('time')
    
    stats = {
        'trackid': trackid,
        'num_points': len(track_df),
        'duration': track_df['time'].max() - track_df['time'].min(),
        'start_time': track_df['time'].min(),
        'end_time': track_df['time'].max(),
        'x_range': (track_df['x'].min(), track_df['x'].max()),
        'y_range': (track_df['y'].min(), track_df['y'].max()),
        'z_range': (track_df['z'].min(), track_df['z'].max()),
        'avg_speed': compute_speed(track_df['vx'].values, track_df['vy'].values, track_df['vz'].values).mean(),
        'max_speed': compute_speed(track_df['vx'].values, track_df['vy'].values, track_df['vz'].values).max(),
        'avg_altitude': track_df['z'].mean(),
        'altitude_change': track_df['z'].max() - track_df['z'].min()
    }
    
    return stats
