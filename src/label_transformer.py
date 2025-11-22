"""
Intelligent Label Transformer - Automatically handles label diversity issues
Provides vectorized multi-label classification and automatic label splitting
"""
import pandas as pd
import numpy as np
from typing import Tuple, List, Dict, Any, Optional
import logging
from sklearn.preprocessing import MultiLabelBinarizer
from pathlib import Path

logger = logging.getLogger(__name__)


class LabelTransformer:
    """Transforms labels to handle diversity issues automatically"""
    
    def __init__(self):
        self.mlb = MultiLabelBinarizer()
        self.label_columns = []
        self.transformation_applied = None
        self.original_labels = None
        
    def analyze_label_diversity(self, labels: pd.Series) -> Dict[str, Any]:
        """Analyze label diversity and recommend transformation
        
        Args:
            labels: Series of annotation labels
            
        Returns:
            Dictionary with analysis results and recommendations
        """
        unique_labels = labels.unique()
        n_unique = len(unique_labels)
        
        # Check if composite labels (contains commas)
        is_composite = any(',' in str(label) for label in unique_labels)
        
        # Extract all unique tags from composite labels
        all_tags = set()
        if is_composite:
            for label in labels.dropna():
                tags = [tag.strip() for tag in str(label).split(',')]
                all_tags.update(tags)
            all_tags.discard('invalid')
            all_tags.discard('')
        
        analysis = {
            'n_unique_labels': n_unique,
            'unique_labels': list(unique_labels),
            'is_composite': is_composite,
            'n_unique_tags': len(all_tags),
            'unique_tags': sorted(list(all_tags)),
            'requires_transformation': n_unique < 2 or is_composite,  # Always transform composite labels
            'recommended_strategy': None
        }
        
        # Recommend transformation strategy
        if is_composite:
            # Composite labels should be transformed to extract primary label
            # This avoids "previously unseen labels" errors when data is split
            # and works with traditional ML models that expect single labels
            analysis['recommended_strategy'] = 'extract_primary'
            analysis['reason'] = f'Composite labels with {len(all_tags)} individual tags - extracting primary label for single-output models'
        elif n_unique < 2:
            analysis['recommended_strategy'] = 'manual_labeling_required'
            analysis['reason'] = 'Insufficient diversity in simple labels - need more data'
        else:
            analysis['recommended_strategy'] = 'use_as_is'
            analysis['reason'] = f'{n_unique} unique labels - sufficient for training'
            
        return analysis
    
    def transform_to_multi_label(self, df: pd.DataFrame, 
                                 label_column: str = 'Annotation') -> Tuple[pd.DataFrame, np.ndarray, List[str]]:
        """Transform composite labels to multi-label binary format
        
        Args:
            df: DataFrame with label column
            label_column: Name of label column
            
        Returns:
            Tuple of (dataframe, binary_labels_array, label_names)
        """
        logger.info("Transforming composite labels to multi-label binary format")
        
        # Parse composite labels into lists of tags
        label_lists = []
        for label in df[label_column]:
            if pd.isna(label) or label == 'invalid' or label == '':
                tags = []
            else:
                tags = [tag.strip() for tag in str(label).split(',')]
                tags = [tag for tag in tags if tag and tag != 'invalid']
            label_lists.append(tags)
        
        # Fit and transform to binary format
        binary_labels = self.mlb.fit_transform(label_lists)
        self.label_columns = list(self.mlb.classes_)
        self.transformation_applied = 'multi_label_binary'
        self.original_labels = df[label_column].copy()
        
        # Add binary columns to dataframe
        df_out = df.copy()
        for i, label_name in enumerate(self.label_columns):
            df_out[f'label_{label_name}'] = binary_labels[:, i]
        
        logger.info(f"Created {len(self.label_columns)} binary label columns: {self.label_columns}")
        
        return df_out, binary_labels, self.label_columns
    
    def extract_primary_labels(self, df: pd.DataFrame, 
                              label_column: str = 'Annotation',
                              strategy: str = 'hierarchy') -> pd.DataFrame:
        """Extract primary label from composite labels
        
        Args:
            df: DataFrame with composite labels
            label_column: Name of label column
            strategy: Extraction strategy ('hierarchy', 'first', 'most_common')
            
        Returns:
            DataFrame with simplified labels
        """
        logger.info(f"Extracting primary labels using '{strategy}' strategy")
        
        df_out = df.copy()
        self.transformation_applied = 'extract_primary'
        self.original_labels = df[label_column].copy()
        
        if strategy == 'hierarchy':
            # Priority-based extraction
            def extract_primary(label):
                if pd.isna(label) or label == '' or label == 'invalid':
                    return 'unknown'
                
                tags = [tag.strip() for tag in str(label).split(',')]
                
                # Priority order: Direction > Vertical > Path > Maneuver > Speed
                if 'incoming' in tags:
                    return 'incoming'
                elif 'outgoing' in tags:
                    return 'outgoing'
                elif 'ascending' in tags:
                    return 'ascending'
                elif 'descending' in tags:
                    return 'descending'
                elif 'level' in tags or 'level_flight' in tags:
                    return 'level'
                elif 'curved' in tags:
                    return 'curved'
                elif 'linear' in tags:
                    return 'linear'
                elif 'high_maneuver' in tags:
                    return 'high_maneuver'
                elif 'light_maneuver' in tags:
                    return 'light_maneuver'
                elif 'high_speed' in tags:
                    return 'high_speed'
                elif 'low_speed' in tags:
                    return 'low_speed'
                return 'normal'
            
            df_out[label_column] = df[label_column].apply(extract_primary)
            
        elif strategy == 'first':
            # Use first tag
            df_out[label_column] = df[label_column].apply(
                lambda x: str(x).split(',')[0].strip() if pd.notna(x) else 'unknown'
            )
            
        elif strategy == 'most_common':
            # Use most common tag across dataset
            all_tags = []
            for label in df[label_column].dropna():
                all_tags.extend([tag.strip() for tag in str(label).split(',')])
            
            if all_tags:
                from collections import Counter
                most_common_tag = Counter(all_tags).most_common(1)[0][0]
                df_out[label_column] = df[label_column].apply(
                    lambda x: most_common_tag if most_common_tag in str(x) else str(x).split(',')[0].strip()
                )
        
        unique_labels = df_out[label_column].nunique()
        logger.info(f"Extracted primary labels - {unique_labels} unique labels created")
        
        return df_out
    
    def create_per_track_labels(self, df: pd.DataFrame, 
                               label_column: str = 'Annotation',
                               track_column: str = 'trackid',
                               strategy: str = 'primary') -> pd.DataFrame:
        """Create per-track labels from point-level annotations
        
        Args:
            df: DataFrame with point-level labels
            label_column: Name of label column
            track_column: Name of track ID column
            strategy: Aggregation strategy ('primary', 'majority', 'voting')
            
        Returns:
            DataFrame with track-level labels
        """
        logger.info(f"Creating per-track labels using '{strategy}' strategy")
        
        if track_column not in df.columns:
            logger.warning(f"Track column '{track_column}' not found, skipping track aggregation")
            return df
        
        df_out = df.copy()
        self.transformation_applied = 'per_track'
        self.original_labels = df[label_column].copy()
        
        track_labels = {}
        
        for trackid in df[track_column].unique():
            track_df = df[df[track_column] == trackid]
            annotations = track_df[label_column].dropna()
            
            if len(annotations) == 0:
                track_labels[trackid] = 'unknown'
                continue
            
            if strategy == 'primary':
                # Extract primary from most common annotation
                most_common = annotations.mode()[0] if len(annotations) > 0 else 'unknown'
                # Apply hierarchical extraction
                primary = self._extract_primary_tag(most_common)
                track_labels[trackid] = primary
                
            elif strategy == 'majority':
                # Use most common annotation as-is
                track_labels[trackid] = annotations.mode()[0]
                
            elif strategy == 'voting':
                # Count individual tags across all points
                all_tags = []
                for ann in annotations:
                    tags = [tag.strip() for tag in str(ann).split(',')]
                    all_tags.extend(tags)
                
                from collections import Counter
                if all_tags:
                    most_common_tag = Counter(all_tags).most_common(1)[0][0]
                    track_labels[trackid] = most_common_tag
                else:
                    track_labels[trackid] = 'unknown'
        
        # Apply track labels to all points
        df_out[label_column] = df_out[track_column].map(track_labels)
        
        unique_labels = df_out[label_column].nunique()
        n_tracks = df_out[track_column].nunique()
        logger.info(f"Created {unique_labels} unique labels from {n_tracks} tracks")
        
        return df_out
    
    def _extract_primary_tag(self, label: str) -> str:
        """Helper to extract primary tag from composite label"""
        if pd.isna(label) or label == '' or label == 'invalid':
            return 'unknown'
        
        tags = [tag.strip() for tag in str(label).split(',')]
        
        # Priority order
        priority = [
            'incoming', 'outgoing',
            'ascending', 'descending', 'level', 'level_flight',
            'curved', 'linear',
            'high_maneuver', 'light_maneuver',
            'high_speed', 'low_speed'
        ]
        
        for priority_tag in priority:
            if priority_tag in tags:
                return priority_tag
        
        return tags[0] if tags else 'normal'
    
    def auto_transform(self, df: pd.DataFrame, 
                      label_column: str = 'Annotation',
                      track_column: str = 'trackid') -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Automatically select and apply best transformation
        
        Args:
            df: DataFrame with labels
            label_column: Name of label column
            track_column: Name of track column (optional)
            
        Returns:
            Tuple of (transformed_dataframe, transformation_info)
        """
        # Analyze labels
        analysis = self.analyze_label_diversity(df[label_column])
        
        info = {
            'analysis': analysis,
            'transformation': None,
            'success': False
        }
        
        if not analysis['requires_transformation']:
            logger.info("Labels have sufficient diversity - no transformation needed")
            info['transformation'] = 'none'
            info['success'] = True
            return df, info
        
        strategy = analysis['recommended_strategy']
        logger.info(f"Applying automatic transformation: {strategy}")
        
        try:
            if strategy == 'multi_label_binary':
                df_out, binary_labels, label_names = self.transform_to_multi_label(df, label_column)
                info['transformation'] = 'multi_label_binary'
                info['binary_label_columns'] = label_names
                info['n_labels'] = len(label_names)
                info['success'] = True
                return df_out, info
                
            elif strategy == 'extract_primary':
                # Try track-level first if available
                if track_column in df.columns and df[track_column].nunique() > 1:
                    df_out = self.create_per_track_labels(df, label_column, track_column, 'primary')
                    if df_out[label_column].nunique() >= 2:
                        info['transformation'] = 'per_track_primary'
                        info['n_labels'] = df_out[label_column].nunique()
                        info['success'] = True
                        return df_out, info
                
                # Fall back to primary extraction
                df_out = self.extract_primary_labels(df, label_column, 'hierarchy')
                info['transformation'] = 'extract_primary'
                info['n_labels'] = df_out[label_column].nunique()
                info['success'] = df_out[label_column].nunique() >= 2
                return df_out, info
                
            else:
                logger.error(f"Cannot automatically transform - {analysis['reason']}")
                info['transformation'] = 'failed'
                info['reason'] = analysis['reason']
                return df, info
                
        except Exception as e:
            logger.error(f"Transformation failed: {e}")
            info['transformation'] = 'failed'
            info['error'] = str(e)
            return df, info


def quick_fix_labels(csv_path: str, output_path: Optional[str] = None, 
                    strategy: str = 'auto') -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Quick fix for label diversity issues
    
    Args:
        csv_path: Path to CSV file with labels
        output_path: Output path (optional, auto-generated if None)
        strategy: Strategy to use ('auto', 'multi_label', 'primary', 'track')
        
    Returns:
        Tuple of (fixed_dataframe, info_dict)
    """
    df = pd.read_csv(csv_path)
    
    if 'Annotation' not in df.columns:
        raise ValueError("CSV must have 'Annotation' column")
    
    transformer = LabelTransformer()
    
    if strategy == 'auto':
        df_fixed, info = transformer.auto_transform(df)
    elif strategy == 'multi_label':
        df_fixed, _, _ = transformer.transform_to_multi_label(df)
        info = {'transformation': 'multi_label_binary', 'success': True}
    elif strategy == 'primary':
        df_fixed = transformer.extract_primary_labels(df)
        info = {'transformation': 'extract_primary', 'success': True}
    elif strategy == 'track':
        df_fixed = transformer.create_per_track_labels(df)
        info = {'transformation': 'per_track', 'success': True}
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    
    # Save if output path provided
    if output_path:
        df_fixed.to_csv(output_path, index=False)
        logger.info(f"Saved transformed data to {output_path}")
        info['output_path'] = output_path
    
    return df_fixed, info


# CLI interface
if __name__ == "__main__":
    import argparse
    import sys
    
    parser = argparse.ArgumentParser(
        description='Intelligent Label Transformer - Fix label diversity issues automatically'
    )
    parser.add_argument('input', help='Input CSV file with labels')
    parser.add_argument('--output', '-o', help='Output CSV file (optional)')
    parser.add_argument('--strategy', '-s', 
                       choices=['auto', 'multi_label', 'primary', 'track'],
                       default='auto',
                       help='Transformation strategy (default: auto)')
    parser.add_argument('--analyze-only', '-a', action='store_true',
                       help='Only analyze labels without transforming')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s'
    )
    
    if not Path(args.input).exists():
        print(f"✗ File not found: {args.input}")
        sys.exit(1)
    
    try:
        if args.analyze_only:
            # Analysis only
            df = pd.read_csv(args.input)
            transformer = LabelTransformer()
            analysis = transformer.analyze_label_diversity(df['Annotation'])
            
            print(f"\n{'='*80}")
            print(f"Label Diversity Analysis")
            print(f"{'='*80}\n")
            print(f"Unique labels: {analysis['n_unique_labels']}")
            print(f"Is composite: {analysis['is_composite']}")
            if analysis['is_composite']:
                print(f"Unique tags: {analysis['n_unique_tags']}")
                print(f"Tags: {', '.join(analysis['unique_tags'])}")
            print(f"\nRequires transformation: {analysis['requires_transformation']}")
            print(f"Recommended strategy: {analysis['recommended_strategy']}")
            print(f"Reason: {analysis['reason']}\n")
        else:
            # Transform
            df_fixed, info = quick_fix_labels(args.input, args.output, args.strategy)
            
            print(f"\n{'='*80}")
            print(f"Label Transformation Complete")
            print(f"{'='*80}\n")
            print(f"Strategy applied: {info.get('transformation', 'unknown')}")
            print(f"Success: {info.get('success', False)}")
            
            if info.get('success'):
                if 'n_labels' in info:
                    print(f"Unique labels created: {info['n_labels']}")
                if 'binary_label_columns' in info:
                    print(f"Binary label columns: {', '.join(info['binary_label_columns'])}")
                if 'output_path' in info:
                    print(f"Saved to: {info['output_path']}")
                print("\n✅ Ready for training!")
            else:
                print(f"⚠️ Transformation unsuccessful: {info.get('reason', 'unknown')}")
        
        sys.exit(0 if info.get('success', True) else 1)
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
