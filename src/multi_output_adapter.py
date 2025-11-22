"""Multi-Output Data Adapter for Tag Prediction

This module handles data preparation for multi-output classification where:
- Columns A-K: Input features (radar measurements)
- Columns L-AF: Individual tag columns (outputs to predict)
- Column AG: Aggregated annotation (optional reference)
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional

logger = logging.getLogger(__name__)


class MultiOutputDataAdapter:
    """Adapter for multi-output tag prediction data format"""
    
    def __init__(self):
        """Initialize the adapter"""
        self.input_feature_columns = []
        self.output_tag_columns = []
        self.aggregated_label_column = None
        
    def identify_columns(self, df: pd.DataFrame, 
                         input_cols: Optional[List[str]] = None,
                         output_cols: Optional[List[str]] = None,
                         aggregated_col: str = 'Annotation') -> None:
        """Identify input features, output tags, and aggregated label columns
        
        Args:
            df: Input DataFrame
            input_cols: List of input feature column names (if None, auto-detect)
            output_cols: List of output tag column names (if None, auto-detect)
            aggregated_col: Name of aggregated annotation column
        """
        if input_cols is not None:
            self.input_feature_columns = input_cols
        else:
            # Auto-detect input features (numeric columns excluding tags)
            # Typical radar features: time, x, y, z, vx, vy, vz, ax, ay, az, speed, etc.
            exclude_cols = ['trackid', 'valid_features', aggregated_col]
            
            # Find numeric columns that are not boolean (tags are boolean)
            potential_inputs = []
            for col in df.columns:
                if col in exclude_cols:
                    continue
                if col not in df.columns:
                    continue
                
                # Check if column is numeric and not boolean (0/1 only)
                if pd.api.types.is_numeric_dtype(df[col]):
                    unique_vals = df[col].dropna().unique()
                    # If column has only 0, 1, True, False -> it's likely a tag
                    if not (set(unique_vals).issubset({0, 1, True, False, 0.0, 1.0})):
                        potential_inputs.append(col)
            
            self.input_feature_columns = potential_inputs
        
        if output_cols is not None:
            self.output_tag_columns = output_cols
        else:
            # Auto-detect output tags (boolean columns)
            potential_outputs = []
            for col in df.columns:
                if col in ['trackid', 'time', 'valid_features', aggregated_col]:
                    continue
                if col in self.input_feature_columns:
                    continue
                    
                # Check if column is boolean or binary (0/1)
                if pd.api.types.is_bool_dtype(df[col]) or \
                   (pd.api.types.is_numeric_dtype(df[col]) and 
                    set(df[col].dropna().unique()).issubset({0, 1, True, False, 0.0, 1.0})):
                    potential_outputs.append(col)
            
            self.output_tag_columns = potential_outputs
        
        self.aggregated_label_column = aggregated_col if aggregated_col in df.columns else None
        
        logger.info(f"Identified {len(self.input_feature_columns)} input features: {self.input_feature_columns[:5]}...")
        logger.info(f"Identified {len(self.output_tag_columns)} output tags: {self.output_tag_columns}")
        logger.info(f"Aggregated label column: {self.aggregated_label_column}")
    
    def prepare_data(self, df: pd.DataFrame, 
                     filter_valid: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Prepare data for multi-output training
        
        Args:
            df: Input DataFrame
            filter_valid: Whether to filter by valid_features column
            
        Returns:
            Tuple of (X, Y, metadata) where:
                - X: DataFrame with input features
                - Y: DataFrame with output tags
                - metadata: DataFrame with trackid, time, etc.
        """
        # Filter valid features if requested
        if filter_valid and 'valid_features' in df.columns:
            # Use .loc to avoid DataFrame ambiguity issues
            valid_mask = df['valid_features'].astype(bool)
            df_valid = df.loc[valid_mask].copy()
            logger.info(f"Filtered from {len(df)} to {len(df_valid)} valid samples")
        else:
            df_valid = df.copy()
        
        if len(df_valid) == 0:
            raise ValueError("No valid samples found after filtering")
        
        # Extract input features
        if not self.input_feature_columns:
            raise ValueError("Input feature columns not identified. Call identify_columns() first.")
        
        missing_input_cols = [col for col in self.input_feature_columns if col not in df_valid.columns]
        if missing_input_cols:
            raise ValueError(f"Missing input columns: {missing_input_cols}")
        
        X = df_valid[self.input_feature_columns].copy()
        
        # Handle NaN and Inf in input features
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(0)
        
        # Extract output tags
        if not self.output_tag_columns:
            raise ValueError("Output tag columns not identified. Call identify_columns() first.")
        
        missing_output_cols = [col for col in self.output_tag_columns if col not in df_valid.columns]
        if missing_output_cols:
            raise ValueError(f"Missing output columns: {missing_output_cols}")
        
        Y = df_valid[self.output_tag_columns].copy()
        
        # Convert to binary (0/1)
        Y = Y.astype(int)
        
        # Extract metadata
        metadata_cols = ['trackid', 'time'] if 'trackid' in df_valid.columns else []
        if self.aggregated_label_column and self.aggregated_label_column in df_valid.columns:
            metadata_cols.append(self.aggregated_label_column)
        
        metadata = df_valid[metadata_cols].copy() if metadata_cols else pd.DataFrame(index=df_valid.index)
        
        logger.info(f"Prepared data: X shape={X.shape}, Y shape={Y.shape}")
        logger.info(f"Output tag distribution:")
        for col in Y.columns:
            pos_count = Y[col].sum()
            logger.info(f"  {col}: {pos_count}/{len(Y)} ({pos_count/len(Y)*100:.1f}%)")
        
        return X, Y, metadata
    
    def prepare_sequences(self, df: pd.DataFrame, 
                         sequence_length: int = 20,
                         filter_valid: bool = True) -> Tuple[np.ndarray, np.ndarray, List]:
        """Prepare sequence data for LSTM/Transformer models
        
        Args:
            df: Input DataFrame
            sequence_length: Length of sequences
            filter_valid: Whether to filter by valid_features column
            
        Returns:
            Tuple of (X_sequences, Y_sequences, track_info) where:
                - X_sequences: Array of shape (n_sequences, seq_len, n_features)
                - Y_sequences: Array of shape (n_sequences, n_output_tags)
                - track_info: List of (trackid, indices) for each sequence
        """
        if not self.input_feature_columns or not self.output_tag_columns:
            raise ValueError("Columns not identified. Call identify_columns() first.")
        
        X_sequences = []
        Y_sequences = []
        track_info = []
        
        for trackid in df['trackid'].unique():
            track_df = df[df['trackid'] == trackid].sort_values('time').copy()
            
            # Filter valid features
            if filter_valid and 'valid_features' in track_df.columns:
                valid_mask = track_df['valid_features'].astype(bool)
                track_df = track_df.loc[valid_mask]
            
            if len(track_df) < 3:
                continue
            
            # Get features
            X_track = track_df[self.input_feature_columns].copy()
            X_track = X_track.replace([np.inf, -np.inf], np.nan)
            X_track = X_track.fillna(0)
            
            # Get tags (use mode across sequence as label)
            Y_track = track_df[self.output_tag_columns].copy()
            
            if len(track_df) < sequence_length:
                # Pad if too short
                n_pad = sequence_length - len(track_df)
                
                # Pad features with zeros
                pad_X = pd.DataFrame(
                    np.zeros((n_pad, len(self.input_feature_columns))),
                    columns=self.input_feature_columns
                )
                X_padded = pd.concat([pad_X, X_track], ignore_index=True)
                
                # Use mode of available labels
                Y_label = Y_track.mode().iloc[0] if len(Y_track) > 0 else pd.Series(0, index=self.output_tag_columns)
                
                X_sequences.append(X_padded.values)
                Y_sequences.append(Y_label.values.astype(int))
                track_info.append((trackid, track_df.index))
            else:
                # Use sliding window
                stride = max(1, sequence_length // 4)
                for i in range(0, len(track_df) - sequence_length + 1, stride):
                    window_indices = track_df.iloc[i:i+sequence_length].index
                    
                    X_window = X_track.iloc[i:i+sequence_length]
                    Y_window = Y_track.iloc[i:i+sequence_length]
                    
                    # Use mode for labels in window
                    Y_label = Y_window.mode().iloc[0] if len(Y_window) > 0 else pd.Series(0, index=self.output_tag_columns)
                    
                    X_sequences.append(X_window.values)
                    Y_sequences.append(Y_label.values.astype(int))
                    track_info.append((trackid, window_indices))
        
        if len(X_sequences) == 0:
            raise ValueError("No sequences could be created from the data")
        
        X_sequences = np.array(X_sequences)
        Y_sequences = np.array(Y_sequences)
        
        logger.info(f"Created {len(X_sequences)} sequences of length {sequence_length}")
        logger.info(f"Sequence shape: X={X_sequences.shape}, Y={Y_sequences.shape}")
        
        return X_sequences, Y_sequences, track_info
    
    def create_aggregated_labels(self, Y: pd.DataFrame) -> pd.Series:
        """Create aggregated annotation labels from tag columns
        
        Args:
            Y: DataFrame with tag columns
            
        Returns:
            Series with aggregated labels (comma-separated tag names)
        """
        aggregated = []
        for idx, row in Y.iterrows():
            active_tags = [col for col in Y.columns if row[col] == 1]
            if not active_tags:
                aggregated.append('no_tags')
            else:
                aggregated.append(','.join(sorted(active_tags)))
        
        return pd.Series(aggregated, index=Y.index, name='Aggregated_Annotation')


# Default column mappings for typical radar data
DEFAULT_INPUT_FEATURES = [
    'x', 'y', 'z', 'vx', 'vy', 'vz', 'ax', 'ay', 'az',
    'speed', 'speed_2d', 'heading', 'range', 'range_rate',
    'curvature', 'accel_magnitude', 'vertical_rate', 'altitude_change'
]

DEFAULT_OUTPUT_TAGS = [
    'incoming', 'outgoing', 
    'fixed_range_ascending', 'fixed_range_descending', 'level_flight',
    'linear', 'curved',
    'light_maneuver', 'high_maneuver',
    'low_speed', 'high_speed'
]
