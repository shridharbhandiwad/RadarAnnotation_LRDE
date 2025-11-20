"""Data Extraction Engine - Parse binary radar data"""
import pandas as pd
import logging
from pathlib import Path
from typing import Dict, Any
from .utils import read_binary_file, ensure_dir
from .config import get_config

logger = logging.getLogger(__name__)


def extract_binary_to_dataframe(binary_path: str, schema: Dict[str, Any] = None) -> pd.DataFrame:
    """Extract binary radar data to pandas DataFrame
    
    Args:
        binary_path: Path to binary file
        schema: Binary schema dictionary (uses config default if None)
        
    Returns:
        DataFrame with columns: trackid, time, x, y, z, vx, vy, vz, ax, ay, az
    """
    if schema is None:
        config = get_config()
        schema = config.get('binary_schema')
    
    logger.info(f"Extracting binary data from {binary_path}")
    
    # Read binary records
    records = read_binary_file(binary_path, schema)
    
    if not records:
        logger.warning("No records extracted from binary file")
        return pd.DataFrame()
    
    # Convert to DataFrame
    df = pd.DataFrame(records)
    
    # Ensure correct column order
    expected_columns = ['time', 'trackid', 'x', 'y', 'z', 'vx', 'vy', 'vz', 'ax', 'ay', 'az']
    available_columns = [col for col in expected_columns if col in df.columns]
    df = df[available_columns]
    
    # Convert trackid to int if it's effectively an integer
    if 'trackid' in df.columns:
        df['trackid'] = df['trackid'].astype(int)
    
    # Sort by track and time
    df = df.sort_values(['trackid', 'time']).reset_index(drop=True)
    
    logger.info(f"Extracted {len(df)} records, {df['trackid'].nunique()} unique tracks")
    
    return df


def save_dataframe(df: pd.DataFrame, out_path: str, fmt: str = 'csv') -> str:
    """Save DataFrame to file
    
    Args:
        df: DataFrame to save
        out_path: Output file path
        fmt: Format ('csv' or 'xlsx')
        
    Returns:
        Path to saved file
    """
    ensure_dir(Path(out_path).parent)
    
    if fmt == 'csv':
        df.to_csv(out_path, index=False)
        logger.info(f"Saved {len(df)} records to CSV: {out_path}")
    elif fmt == 'xlsx':
        df.to_excel(out_path, index=False, engine='openpyxl')
        logger.info(f"Saved {len(df)} records to Excel: {out_path}")
    else:
        raise ValueError(f"Unsupported format: {fmt}")
    
    return out_path


def load_dataframe(path: str) -> pd.DataFrame:
    """Load DataFrame from CSV or Excel file
    
    Args:
        path: File path
        
    Returns:
        DataFrame
    """
    if path.endswith('.csv'):
        df = pd.read_csv(path)
    elif path.endswith('.xlsx'):
        df = pd.read_excel(path, engine='openpyxl')
    else:
        raise ValueError(f"Unsupported file format: {path}")
    
    logger.info(f"Loaded {len(df)} records from {path}")
    return df


def get_data_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """Get summary statistics for the data
    
    Args:
        df: DataFrame with radar data
        
    Returns:
        Dictionary with summary statistics
    """
    summary = {
        'total_records': len(df),
        'num_tracks': df['trackid'].nunique() if 'trackid' in df.columns else 0,
        'time_range': (df['time'].min(), df['time'].max()) if 'time' in df.columns else (0, 0),
        'duration_seconds': df['time'].max() - df['time'].min() if 'time' in df.columns else 0,
        'spatial_extent': {
            'x': (df['x'].min(), df['x'].max()) if 'x' in df.columns else (0, 0),
            'y': (df['y'].min(), df['y'].max()) if 'y' in df.columns else (0, 0),
            'z': (df['z'].min(), df['z'].max()) if 'z' in df.columns else (0, 0)
        },
        'columns': list(df.columns)
    }
    
    return summary


# CLI interface
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Data Extraction Engine')
    parser.add_argument('--input', required=True, help='Input binary file')
    parser.add_argument('--out', required=True, help='Output CSV file')
    parser.add_argument('--schema', help='Schema JSON file (optional)')
    parser.add_argument('--format', default='csv', choices=['csv', 'xlsx'], help='Output format')
    
    args = parser.parse_args()
    
    # Load schema if provided
    schema = None
    if args.schema:
        import json
        with open(args.schema, 'r') as f:
            schema = json.load(f)
    
    # Extract data
    df = extract_binary_to_dataframe(args.input, schema)
    
    # Save data
    save_dataframe(df, args.out, args.format)
    
    # Print summary
    summary = get_data_summary(df)
    print("\nData Summary:")
    print(f"  Total records: {summary['total_records']}")
    print(f"  Number of tracks: {summary['num_tracks']}")
    print(f"  Duration: {summary['duration_seconds']:.2f} seconds")
