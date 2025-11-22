"""AI Tagging Engine - Machine Learning models for trajectory classification"""
import numpy as np
import pandas as pd
import logging
import os
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional
import joblib
import json

# ML imports
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

# Deep learning imports
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    HAS_TENSORFLOW = True
except ImportError:
    HAS_TENSORFLOW = False
    logging.warning("TensorFlow not available, LSTM and Transformer models will not work")

from .config import get_config
from .utils import ensure_dir
from .label_transformer import LabelTransformer
from .multi_output_adapter import MultiOutputDataAdapter, DEFAULT_INPUT_FEATURES, DEFAULT_OUTPUT_TAGS

logger = logging.getLogger(__name__)


class SequenceDataGenerator:
    """Generate sequence data for LSTM/Transformer models"""
    
    def __init__(self, sequence_length: int = 20):
        """Initialize sequence generator
        
        Args:
            sequence_length: Length of sequences
        """
        self.sequence_length = sequence_length
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
    def prepare_sequences(self, df: pd.DataFrame, feature_columns: List[str], 
                         label_column: str = 'Annotation', return_label_strings: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Prepare sequence data from DataFrame with enhanced feature engineering
        
        Args:
            df: Input DataFrame
            feature_columns: List of feature column names
            label_column: Label column name
            return_label_strings: If True, return raw label strings instead of encoded integers
            
        Returns:
            Tuple of (sequences, labels, track_ids)
            If return_label_strings=True, labels are strings; otherwise they are encoded integers
        """
        sequences = []
        labels = []
        track_ids = []
        
        for trackid in df['trackid'].unique():
            track_df = df[df['trackid'] == trackid].sort_values('time').copy()
            
            # Filter valid features only
            if 'valid_features' in track_df.columns:
                valid_mask = track_df['valid_features'].astype(bool)
                track_df = track_df.loc[valid_mask]
            
            if len(track_df) < 3:  # Skip tracks with insufficient data
                continue
            
            # Ensure feature columns exist
            available_features = [col for col in feature_columns if col in track_df.columns]
            if len(available_features) == 0:
                continue
            
            # Handle NaN and Inf values
            track_df[available_features] = track_df[available_features].replace([np.inf, -np.inf], np.nan)
            track_df[available_features] = track_df[available_features].fillna(0)
            
            if len(track_df) < self.sequence_length:
                # Pad if too short
                n_pad = self.sequence_length - len(track_df)
                pad_df = pd.DataFrame(
                    np.zeros((n_pad, len(available_features))),
                    columns=available_features
                )
                track_features = pd.concat([pad_df, track_df[available_features]], ignore_index=True)
                track_label = track_df[label_column].mode()[0] if len(track_df) > 0 else 'normal'
            else:
                # Use sliding window with stride
                stride = max(1, self.sequence_length // 4)  # 75% overlap
                for i in range(0, len(track_df) - self.sequence_length + 1, stride):
                    window = track_df.iloc[i:i+self.sequence_length]
                    track_features = window[available_features]
                    track_label = window[label_column].mode()[0]
                    
                    sequences.append(track_features.values)
                    labels.append(track_label)
                    track_ids.append(trackid)
                continue
            
            sequences.append(track_features.values)
            labels.append(track_label)
            track_ids.append(trackid)
        
        if len(sequences) == 0:
            raise ValueError("No valid sequences could be created from the data. Ensure tracks have sufficient points.")
        
        sequences = np.array(sequences)
        track_ids = np.array(track_ids)
        
        # Encode labels unless raw strings requested
        if return_label_strings:
            labels = np.array(labels)
        else:
            labels = self.label_encoder.fit_transform(labels)
        
        return sequences, labels, track_ids
    
    def normalize_sequences(self, sequences: np.ndarray, fit: bool = True) -> np.ndarray:
        """Normalize sequence data
        
        Args:
            sequences: Input sequences (n_samples, seq_len, n_features)
            fit: Whether to fit the scaler
            
        Returns:
            Normalized sequences
        """
        n_samples, seq_len, n_features = sequences.shape
        
        # Reshape for scaling
        sequences_flat = sequences.reshape(-1, n_features)
        
        if fit:
            sequences_scaled = self.scaler.fit_transform(sequences_flat)
        else:
            sequences_scaled = self.scaler.transform(sequences_flat)
        
        # Reshape back
        sequences_scaled = sequences_scaled.reshape(n_samples, seq_len, n_features)
        
        return sequences_scaled


class XGBoostModel:
    """XGBoost classifier for tabular features"""
    
    def __init__(self, params: Dict[str, Any] = None):
        """Initialize XGBoost model
        
        Args:
            params: Model parameters
        """
        config = get_config()
        default_params = config.get('ml_params.xgboost', {})
        self.params = {**default_params, **(params or {})}
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_columns = None
        
    def prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare tabular features from DataFrame
        
        Args:
            df: Input DataFrame
            
        Returns:
            Tuple of (features, labels)
        """
        # Select feature columns (exclude identifiers and labels)
        exclude_cols = ['trackid', 'time', 'Annotation', 'valid_features']
        feature_cols = [col for col in df.columns if col not in exclude_cols and 
                       not col.startswith('incoming') and not col.startswith('outgoing') and
                       not col.startswith('fixed_range') and not col.startswith('level_flight') and
                       not col.startswith('linear') and not col.startswith('curved') and
                       not col.startswith('light_maneuver') and not col.startswith('high_maneuver') and
                       not col.startswith('low_speed') and not col.startswith('high_speed')]
        
        self.feature_columns = feature_cols
        
        # Filter valid features only
        # Use .loc with boolean mask to avoid DataFrame ambiguity
        if 'valid_features' in df.columns:
            valid_mask = df['valid_features'].astype(bool)
            df_valid = df.loc[valid_mask].copy()
        else:
            # If no valid_features column, use all data
            df_valid = df.copy()
        
        # Validate we have data after filtering
        if len(df_valid) == 0:
            raise ValueError(
                "No valid data remaining after filtering. "
                "All rows were filtered out by 'valid_features' column. \n\n"
                "This usually happens when: \n"
                "  1. The auto-labeling engine marked all data as invalid \n"
                "  2. The data doesn't have enough points per track \n\n"
                "Suggestions: \n"
                "  - Check your input data has sufficient trajectory points \n"
                "  - Try using raw labeled data without auto-labeling \n"
                f"  - Original data had {len(df)} rows, all were filtered out"
            )
        
        X = df_valid[feature_cols].values
        y = df_valid['Annotation'].values
        
        # Validate we have labels
        if len(y) == 0:
            raise ValueError("No labels found in data after filtering")
        
        return X, y
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
             X_val: np.ndarray = None, y_val: np.ndarray = None) -> Dict[str, Any]:
        """Train XGBoost model
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            
        Returns:
            Training metrics
        """
        import time
        start_time = time.time()
        
        # Validate training data
        if len(X_train) == 0 or len(y_train) == 0:
            raise ValueError(
                "Training data is empty. Cannot train model with 0 samples.\n\n"
                "Suggestions:\n"
                "  - Verify your CSV file contains data rows\n"
                "  - Check that data wasn't filtered out by 'valid_features' column\n"
                "  - Ensure the data has required feature columns"
            )
        
        # Normalize features
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Encode labels
        y_train_encoded = self.label_encoder.fit_transform(y_train)
        n_classes = len(self.label_encoder.classes_)
        unique_labels = list(self.label_encoder.classes_)
        
        # Validate minimum number of classes
        if n_classes < 2:
            # Check if it's a composite label issue
            is_composite = any(',' in str(label) for label in unique_labels)
            
            error_msg = (
                f"Insufficient classes for training. Found {n_classes} unique class(es): {unique_labels}\n\n"
                "Machine learning models require at least 2 different classes to train.\n\n"
            )
            
            if is_composite:
                error_msg += (
                    "🔍 DETECTED: Your data uses composite labels (comma-separated tags)\n"
                    "   Example: 'incoming,level,linear,light_maneuver,low_speed'\n\n"
                    "This happens when auto-labeling creates the same combination for all data.\n\n"
                    "💡 TIP: Use auto_transform=True in train_model() for automatic recovery\n\n"
                    "MANUAL FIXES:\n"
                    "  1. Analyze your data to understand why labels are uniform:\n"
                    "     → python analyze_label_diversity.py <your_csv_file>\n\n"
                    "  2. Auto-transform labels with intelligent recovery:\n"
                    "     → python -m src.label_transformer <your_csv_file> -o fixed_data.csv\n\n"
                    "  3. Create per-track labels (if you have multiple tracks):\n"
                    "     → python create_track_labels.py <your_csv_file>\n\n"
                    "  4. Split composite labels into separate binary tasks:\n"
                    "     → python split_composite_labels.py <your_csv_file>\n\n"
                    "  5. Adjust auto-labeling thresholds in config/default_config.json\n"
                    "     and re-run auto-labeling to create more varied labels\n"
                )
            else:
                error_msg += (
                    "Suggestions:\n"
                    "  1. Check your 'Annotation' column has multiple different labels\n"
                    "  2. If using auto-labeling, verify it generated diverse labels\n"
                    "  3. Manually review and add variety to your annotations\n"
                    f"  4. Current unique classes: {unique_labels}\n"
                )
            
            raise ValueError(error_msg)
        
        # Set appropriate objective based on number of classes
        # Always override objective based on actual data, regardless of config
        params = self.params.copy()
        
        if n_classes == 2:
            # Binary classification
            params['objective'] = 'binary:logistic'
            # Remove num_class if present (not used for binary classification)
            if 'num_class' in params:
                del params['num_class']
            logger.info(f"Detected {n_classes} classes, using binary classification (objective=binary:logistic)")
        else:
            # Multi-class classification (3+ classes)
            params['objective'] = 'multi:softmax'
            # Always set num_class for multi-class
            params['num_class'] = n_classes
            logger.info(f"Detected {n_classes} classes, using multi-class classification (objective=multi:softmax, num_class={n_classes})")
        
        # Train model
        self.model = xgb.XGBClassifier(**params)
        
        eval_set = None
        if X_val is not None and y_val is not None:
            X_val_scaled = self.scaler.transform(X_val)
            y_val_encoded = self.label_encoder.transform(y_val)
            eval_set = [(X_val_scaled, y_val_encoded)]
        
        self.model.fit(X_train_scaled, y_train_encoded, eval_set=eval_set, verbose=False)
        
        training_time = time.time() - start_time
        
        # Evaluate on training data
        y_train_pred = self.model.predict(X_train_scaled)
        train_accuracy = accuracy_score(y_train_encoded, y_train_pred)
        
        metrics = {
            'training_time': training_time,
            'train_accuracy': train_accuracy,
            'n_classes': len(self.label_encoder.classes_),
            'classes': list(self.label_encoder.classes_)
        }
        
        if eval_set:
            y_val_pred = self.model.predict(X_val_scaled)
            val_accuracy = accuracy_score(y_val_encoded, y_val_pred)
            metrics['val_accuracy'] = val_accuracy
        
        logger.info(f"XGBoost training completed in {training_time:.2f}s, accuracy: {train_accuracy:.4f}")
        
        return metrics
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """Evaluate model on test data
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Evaluation metrics
        """
        X_test_scaled = self.scaler.transform(X_test)
        y_test_encoded = self.label_encoder.transform(y_test)
        
        y_pred = self.model.predict(X_test_scaled)
        
        accuracy = accuracy_score(y_test_encoded, y_pred)
        f1 = f1_score(y_test_encoded, y_pred, average='weighted')
        
        # Confusion matrix
        cm = confusion_matrix(y_test_encoded, y_pred)
        
        # Classification report
        report = classification_report(y_test_encoded, y_pred, 
                                      target_names=self.label_encoder.classes_,
                                      output_dict=True, zero_division=0)
        
        metrics = {
            'accuracy': accuracy,
            'f1_score': f1,
            'confusion_matrix': cm.tolist(),
            'classification_report': report,
            'classes': list(self.label_encoder.classes_)
        }
        
        logger.info(f"XGBoost evaluation - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
        
        return metrics
    
    def save(self, path: str) -> None:
        """Save model to disk"""
        ensure_dir(Path(path).parent)
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'feature_columns': self.feature_columns,
            'params': self.params
        }, path)
        logger.info(f"Saved XGBoost model to {path}")
    
    def load(self, path: str) -> None:
        """Load model from disk"""
        data = joblib.load(path)
        self.model = data['model']
        self.scaler = data['scaler']
        self.label_encoder = data['label_encoder']
        self.feature_columns = data['feature_columns']
        self.params = data['params']
        logger.info(f"Loaded XGBoost model from {path}")


class XGBoostMultiOutputModel:
    """XGBoost Multi-Output Classifier for tag prediction
    
    Trains separate XGBoost models for each output tag column.
    Suitable for the data format where columns A-K are inputs and columns L-AF are output tags.
    """
    
    def __init__(self, params: Dict[str, Any] = None):
        """Initialize XGBoost Multi-Output model
        
        Args:
            params: Model parameters
        """
        config = get_config()
        default_params = config.get('ml_params.xgboost', {})
        self.params = {**default_params, **(params or {})}
        self.models = {}  # Dictionary of models, one per output tag
        self.scaler = StandardScaler()
        self.adapter = MultiOutputDataAdapter()
        self.output_tag_names = []
        
    def train(self, df_train: pd.DataFrame, df_val: pd.DataFrame = None,
             input_cols: List[str] = None, output_cols: List[str] = None) -> Dict[str, Any]:
        """Train XGBoost models for multi-output prediction
        
        Args:
            df_train: Training DataFrame
            df_val: Validation DataFrame (optional)
            input_cols: List of input feature column names (if None, auto-detect)
            output_cols: List of output tag column names (if None, auto-detect)
            
        Returns:
            Training metrics
        """
        import time
        start_time = time.time()
        
        # Identify columns
        self.adapter.identify_columns(df_train, input_cols, output_cols)
        self.output_tag_names = self.adapter.output_tag_columns
        
        # Prepare data
        X_train, Y_train, _ = self.adapter.prepare_data(df_train, filter_valid=True)
        
        # Normalize features
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Prepare validation data if available
        X_val_scaled, Y_val = None, None
        if df_val is not None:
            X_val, Y_val, _ = self.adapter.prepare_data(df_val, filter_valid=True)
            X_val_scaled = self.scaler.transform(X_val)
        
        # Train a model for each output tag
        tag_metrics = {}
        for tag_name in self.output_tag_names:
            logger.info(f"Training XGBoost model for tag: {tag_name}")
            
            y_train = Y_train[tag_name].values
            
            # Set up parameters for binary classification
            params = self.params.copy()
            params['objective'] = 'binary:logistic'
            if 'num_class' in params:
                del params['num_class']
            # Set base_score to valid value for binary:logistic (must be in (0,1))
            params['base_score'] = 0.5
            
            # Train model
            model = xgb.XGBClassifier(**params)
            
            eval_set = None
            if X_val_scaled is not None and Y_val is not None:
                y_val = Y_val[tag_name].values
                eval_set = [(X_val_scaled, y_val)]
            
            model.fit(X_train_scaled, y_train, eval_set=eval_set, verbose=False)
            self.models[tag_name] = model
            
            # Evaluate
            y_train_pred = model.predict(X_train_scaled)
            train_acc = accuracy_score(y_train, y_train_pred)
            
            tag_metrics[tag_name] = {'train_accuracy': train_acc}
            
            if eval_set:
                y_val_pred = model.predict(X_val_scaled)
                val_acc = accuracy_score(y_val, y_val_pred)
                tag_metrics[tag_name]['val_accuracy'] = val_acc
            
            logger.info(f"  {tag_name}: train_acc={train_acc:.4f}" + 
                       (f", val_acc={val_acc:.4f}" if eval_set else ""))
        
        training_time = time.time() - start_time
        
        # Calculate overall metrics
        overall_train_acc = np.mean([m['train_accuracy'] for m in tag_metrics.values()])
        overall_val_acc = np.mean([m['val_accuracy'] for m in tag_metrics.values() if 'val_accuracy' in m]) if df_val is not None and len(df_val) > 0 else None
        
        metrics = {
            'training_time': training_time,
            'train_accuracy': overall_train_acc,
            'per_tag_metrics': tag_metrics,
            'n_tags': len(self.output_tag_names),
            'tag_names': self.output_tag_names
        }
        
        if overall_val_acc is not None:
            metrics['val_accuracy'] = overall_val_acc
        
        logger.info(f"XGBoost Multi-Output training completed in {training_time:.2f}s")
        logger.info(f"Overall train accuracy: {overall_train_acc:.4f}")
        if overall_val_acc:
            logger.info(f"Overall val accuracy: {overall_val_acc:.4f}")
        
        return metrics
    
    def evaluate(self, df_test: pd.DataFrame) -> Dict[str, Any]:
        """Evaluate model on test data
        
        Args:
            df_test: Test DataFrame
            
        Returns:
            Evaluation metrics
        """
        X_test, Y_test, _ = self.adapter.prepare_data(df_test, filter_valid=True)
        X_test_scaled = self.scaler.transform(X_test)
        
        tag_metrics = {}
        y_pred_all = {}
        
        for tag_name in self.output_tag_names:
            y_true = Y_test[tag_name].values
            model = self.models[tag_name]
            y_pred = model.predict(X_test_scaled)
            
            acc = accuracy_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred, average='binary', zero_division=0)
            
            tag_metrics[tag_name] = {
                'accuracy': float(acc),
                'f1_score': float(f1)
            }
            y_pred_all[tag_name] = y_pred
        
        # Overall metrics
        overall_acc = np.mean([m['accuracy'] for m in tag_metrics.values()])
        overall_f1 = np.mean([m['f1_score'] for m in tag_metrics.values()])
        
        metrics = {
            'accuracy': float(overall_acc),
            'f1_score': float(overall_f1),
            'per_tag_metrics': tag_metrics,
            'multi_output': True,
            'tag_names': self.output_tag_names
        }
        
        logger.info(f"XGBoost Multi-Output evaluation - Accuracy: {overall_acc:.4f}, F1: {overall_f1:.4f}")
        
        return metrics
    
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Predict tags for input data
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with predicted tags
        """
        X, _, metadata = self.adapter.prepare_data(df, filter_valid=False)
        X_scaled = self.scaler.transform(X)
        
        predictions = {}
        for tag_name in self.output_tag_names:
            model = self.models[tag_name]
            predictions[tag_name] = model.predict(X_scaled)
        
        pred_df = pd.DataFrame(predictions, index=X.index)
        
        # Add aggregated annotation
        pred_df['Predicted_Annotation'] = self.adapter.create_aggregated_labels(pred_df)
        
        return pred_df
    
    def save(self, path: str) -> None:
        """Save model to disk"""
        ensure_dir(Path(path).parent)
        joblib.dump({
            'models': self.models,
            'scaler': self.scaler,
            'adapter': self.adapter,
            'output_tag_names': self.output_tag_names,
            'params': self.params
        }, path)
        logger.info(f"Saved XGBoost Multi-Output model to {path}")
    
    def load(self, path: str) -> None:
        """Load model from disk"""
        data = joblib.load(path)
        self.models = data['models']
        self.scaler = data['scaler']
        self.adapter = data['adapter']
        self.output_tag_names = data['output_tag_names']
        self.params = data['params']
        logger.info(f"Loaded XGBoost Multi-Output model from {path}")


class RandomForestModel:
    """Random Forest classifier for tabular features"""
    
    def __init__(self, params: Dict[str, Any] = None):
        """Initialize Random Forest model
        
        Args:
            params: Model parameters
        """
        config = get_config()
        default_params = config.get('ml_params.random_forest', {})
        self.params = {**default_params, **(params or {})}
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_columns = None
        
    def prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare tabular features from DataFrame
        
        Args:
            df: Input DataFrame
            
        Returns:
            Tuple of (features, labels)
        """
        # Select feature columns (exclude identifiers and labels)
        exclude_cols = ['trackid', 'time', 'Annotation', 'valid_features']
        feature_cols = [col for col in df.columns if col not in exclude_cols and 
                       not col.startswith('incoming') and not col.startswith('outgoing') and
                       not col.startswith('fixed_range') and not col.startswith('level_flight') and
                       not col.startswith('linear') and not col.startswith('curved') and
                       not col.startswith('light_maneuver') and not col.startswith('high_maneuver') and
                       not col.startswith('low_speed') and not col.startswith('high_speed')]
        
        self.feature_columns = feature_cols
        
        # Filter valid features only
        if 'valid_features' in df.columns:
            valid_mask = df['valid_features'].astype(bool)
            df_valid = df.loc[valid_mask].copy()
        else:
            df_valid = df.copy()
        
        # Validate we have data after filtering
        if len(df_valid) == 0:
            raise ValueError(
                "No valid data remaining after filtering. "
                "All rows were filtered out by 'valid_features' column. \n\n"
                "This usually happens when: \n"
                "  1. The auto-labeling engine marked all data as invalid \n"
                "  2. The data doesn't have enough points per track \n\n"
                "Suggestions: \n"
                "  - Check your input data has sufficient trajectory points \n"
                "  - Try using raw labeled data without auto-labeling \n"
                f"  - Original data had {len(df)} rows, all were filtered out"
            )
        
        X = df_valid[feature_cols].values
        y = df_valid['Annotation'].values
        
        # Validate we have labels
        if len(y) == 0:
            raise ValueError("No labels found in data after filtering")
        
        return X, y
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
             X_val: np.ndarray = None, y_val: np.ndarray = None) -> Dict[str, Any]:
        """Train Random Forest model
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            
        Returns:
            Training metrics
        """
        import time
        start_time = time.time()
        
        # Validate training data
        if len(X_train) == 0 or len(y_train) == 0:
            raise ValueError(
                "Training data is empty. Cannot train model with 0 samples.\n\n"
                "Suggestions:\n"
                "  - Verify your CSV file contains data rows\n"
                "  - Check that data wasn't filtered out by 'valid_features' column\n"
                "  - Ensure the data has required feature columns"
            )
        
        # Normalize features
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Encode labels
        y_train_encoded = self.label_encoder.fit_transform(y_train)
        n_classes = len(self.label_encoder.classes_)
        unique_labels = list(self.label_encoder.classes_)
        
        # Validate minimum number of classes
        if n_classes < 2:
            is_composite = any(',' in str(label) for label in unique_labels)
            
            error_msg = (
                f"Insufficient classes for training. Found {n_classes} unique class(es): {unique_labels}\n\n"
                "Machine learning models require at least 2 different classes to train.\n\n"
            )
            
            if is_composite:
                error_msg += (
                    "🔍 DETECTED: Your data uses composite labels (comma-separated tags)\n"
                    "   Example: 'incoming,level,linear,light_maneuver,low_speed'\n\n"
                    "This happens when auto-labeling creates the same combination for all data.\n\n"
                    "💡 TIP: Use auto_transform=True in train_model() for automatic recovery\n\n"
                )
            else:
                error_msg += (
                    "Suggestions:\n"
                    "  1. Check your 'Annotation' column has multiple different labels\n"
                    "  2. If using auto-labeling, verify it generated diverse labels\n"
                    "  3. Manually review and add variety to your annotations\n"
                    f"  4. Current unique classes: {unique_labels}\n"
                )
            
            raise ValueError(error_msg)
        
        # Train model
        self.model = RandomForestClassifier(**self.params)
        self.model.fit(X_train_scaled, y_train_encoded)
        
        training_time = time.time() - start_time
        
        # Evaluate on training data
        y_train_pred = self.model.predict(X_train_scaled)
        train_accuracy = accuracy_score(y_train_encoded, y_train_pred)
        
        metrics = {
            'training_time': training_time,
            'train_accuracy': train_accuracy,
            'n_classes': len(self.label_encoder.classes_),
            'classes': list(self.label_encoder.classes_)
        }
        
        if X_val is not None and y_val is not None:
            X_val_scaled = self.scaler.transform(X_val)
            y_val_encoded = self.label_encoder.transform(y_val)
            y_val_pred = self.model.predict(X_val_scaled)
            val_accuracy = accuracy_score(y_val_encoded, y_val_pred)
            metrics['val_accuracy'] = val_accuracy
        
        logger.info(f"Random Forest training completed in {training_time:.2f}s, accuracy: {train_accuracy:.4f}")
        
        return metrics
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """Evaluate model on test data
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Evaluation metrics
        """
        X_test_scaled = self.scaler.transform(X_test)
        y_test_encoded = self.label_encoder.transform(y_test)
        
        y_pred = self.model.predict(X_test_scaled)
        
        accuracy = accuracy_score(y_test_encoded, y_pred)
        f1 = f1_score(y_test_encoded, y_pred, average='weighted')
        
        # Confusion matrix
        cm = confusion_matrix(y_test_encoded, y_pred)
        
        # Classification report
        report = classification_report(y_test_encoded, y_pred, 
                                      target_names=self.label_encoder.classes_,
                                      output_dict=True, zero_division=0)
        
        metrics = {
            'accuracy': accuracy,
            'f1_score': f1,
            'confusion_matrix': cm.tolist(),
            'classification_report': report,
            'classes': list(self.label_encoder.classes_)
        }
        
        logger.info(f"Random Forest evaluation - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
        
        return metrics
    
    def save(self, path: str) -> None:
        """Save model to disk"""
        ensure_dir(Path(path).parent)
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'feature_columns': self.feature_columns,
            'params': self.params
        }, path)
        logger.info(f"Saved Random Forest model to {path}")
    
    def load(self, path: str) -> None:
        """Load model from disk"""
        data = joblib.load(path)
        self.model = data['model']
        self.scaler = data['scaler']
        self.label_encoder = data['label_encoder']
        self.feature_columns = data['feature_columns']
        self.params = data['params']
        logger.info(f"Loaded Random Forest model from {path}")


class RandomForestMultiOutputModel:
    """Random Forest Multi-Output Classifier for tag prediction
    
    Uses sklearn's MultiOutputClassifier with Random Forest for each output tag column.
    Suitable for the data format where columns A-K are inputs and columns L-AF are output tags.
    """
    
    def __init__(self, params: Dict[str, Any] = None):
        """Initialize Random Forest Multi-Output model
        
        Args:
            params: Model parameters
        """
        config = get_config()
        default_params = config.get('ml_params.random_forest', {})
        self.params = {**default_params, **(params or {})}
        self.models = {}  # Dictionary of models, one per output tag
        self.scaler = StandardScaler()
        self.adapter = MultiOutputDataAdapter()
        self.output_tag_names = []
        
    def train(self, df_train: pd.DataFrame, df_val: pd.DataFrame = None,
             input_cols: List[str] = None, output_cols: List[str] = None) -> Dict[str, Any]:
        """Train Random Forest models for multi-output prediction
        
        Args:
            df_train: Training DataFrame
            df_val: Validation DataFrame (optional)
            input_cols: List of input feature column names (if None, auto-detect)
            output_cols: List of output tag column names (if None, auto-detect)
            
        Returns:
            Training metrics
        """
        import time
        start_time = time.time()
        
        # Identify columns
        self.adapter.identify_columns(df_train, input_cols, output_cols)
        self.output_tag_names = self.adapter.output_tag_columns
        
        # Prepare data
        X_train, Y_train, _ = self.adapter.prepare_data(df_train, filter_valid=True)
        
        # Normalize features
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Prepare validation data if available
        X_val_scaled, Y_val = None, None
        if df_val is not None:
            X_val, Y_val, _ = self.adapter.prepare_data(df_val, filter_valid=True)
            X_val_scaled = self.scaler.transform(X_val)
        
        # Train a model for each output tag
        tag_metrics = {}
        for tag_name in self.output_tag_names:
            logger.info(f"Training Random Forest model for tag: {tag_name}")
            
            y_train = Y_train[tag_name].values
            
            # Train model
            model = RandomForestClassifier(**self.params)
            model.fit(X_train_scaled, y_train)
            self.models[tag_name] = model
            
            # Evaluate
            y_train_pred = model.predict(X_train_scaled)
            train_acc = accuracy_score(y_train, y_train_pred)
            
            tag_metrics[tag_name] = {'train_accuracy': train_acc}
            
            if X_val_scaled is not None and Y_val is not None:
                y_val = Y_val[tag_name].values
                y_val_pred = model.predict(X_val_scaled)
                val_acc = accuracy_score(y_val, y_val_pred)
                tag_metrics[tag_name]['val_accuracy'] = val_acc
                logger.info(f"  {tag_name}: train_acc={train_acc:.4f}, val_acc={val_acc:.4f}")
            else:
                logger.info(f"  {tag_name}: train_acc={train_acc:.4f}")
        
        training_time = time.time() - start_time
        
        # Calculate overall metrics
        overall_train_acc = np.mean([m['train_accuracy'] for m in tag_metrics.values()])
        overall_val_acc = np.mean([m['val_accuracy'] for m in tag_metrics.values() if 'val_accuracy' in m]) if df_val is not None and len(df_val) > 0 else None
        
        metrics = {
            'training_time': training_time,
            'train_accuracy': overall_train_acc,
            'per_tag_metrics': tag_metrics,
            'n_tags': len(self.output_tag_names),
            'tag_names': self.output_tag_names
        }
        
        if overall_val_acc is not None:
            metrics['val_accuracy'] = overall_val_acc
        
        logger.info(f"Random Forest Multi-Output training completed in {training_time:.2f}s")
        logger.info(f"Overall train accuracy: {overall_train_acc:.4f}")
        if overall_val_acc:
            logger.info(f"Overall val accuracy: {overall_val_acc:.4f}")
        
        return metrics
    
    def evaluate(self, df_test: pd.DataFrame) -> Dict[str, Any]:
        """Evaluate model on test data
        
        Args:
            df_test: Test DataFrame
            
        Returns:
            Evaluation metrics
        """
        X_test, Y_test, _ = self.adapter.prepare_data(df_test, filter_valid=True)
        X_test_scaled = self.scaler.transform(X_test)
        
        tag_metrics = {}
        y_pred_all = {}
        
        for tag_name in self.output_tag_names:
            y_true = Y_test[tag_name].values
            model = self.models[tag_name]
            y_pred = model.predict(X_test_scaled)
            
            acc = accuracy_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred, average='binary', zero_division=0)
            
            tag_metrics[tag_name] = {
                'accuracy': float(acc),
                'f1_score': float(f1)
            }
            y_pred_all[tag_name] = y_pred
        
        # Overall metrics
        overall_acc = np.mean([m['accuracy'] for m in tag_metrics.values()])
        overall_f1 = np.mean([m['f1_score'] for m in tag_metrics.values()])
        
        metrics = {
            'accuracy': float(overall_acc),
            'f1_score': float(overall_f1),
            'per_tag_metrics': tag_metrics,
            'multi_output': True,
            'tag_names': self.output_tag_names
        }
        
        logger.info(f"Random Forest Multi-Output evaluation - Accuracy: {overall_acc:.4f}, F1: {overall_f1:.4f}")
        
        return metrics
    
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Predict tags for input data
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with predicted tags
        """
        X, _, metadata = self.adapter.prepare_data(df, filter_valid=False)
        X_scaled = self.scaler.transform(X)
        
        predictions = {}
        for tag_name in self.output_tag_names:
            model = self.models[tag_name]
            predictions[tag_name] = model.predict(X_scaled)
        
        pred_df = pd.DataFrame(predictions, index=X.index)
        
        # Add aggregated annotation
        pred_df['Predicted_Annotation'] = self.adapter.create_aggregated_labels(pred_df)
        
        return pred_df
    
    def save(self, path: str) -> None:
        """Save model to disk"""
        ensure_dir(Path(path).parent)
        joblib.dump({
            'models': self.models,
            'scaler': self.scaler,
            'adapter': self.adapter,
            'output_tag_names': self.output_tag_names,
            'params': self.params
        }, path)
        logger.info(f"Saved Random Forest Multi-Output model to {path}")
    
    def load(self, path: str) -> None:
        """Load model from disk"""
        data = joblib.load(path)
        self.models = data['models']
        self.scaler = data['scaler']
        self.adapter = data['adapter']
        self.output_tag_names = data['output_tag_names']
        self.params = data['params']
        logger.info(f"Loaded Random Forest Multi-Output model from {path}")


class TransformerBlock(keras.layers.Layer):
    """Transformer block with multi-head attention"""
    
    def __init__(self, d_model: int, num_heads: int, ff_dim: int, dropout: float = 0.1):
        """Initialize Transformer block
        
        Args:
            d_model: Dimension of the model
            num_heads: Number of attention heads
            ff_dim: Dimension of feed-forward network
            dropout: Dropout rate
        """
        super(TransformerBlock, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout_rate = dropout
        
    def build(self, input_shape):
        """Build the layer"""
        self.att = layers.MultiHeadAttention(
            num_heads=self.num_heads, 
            key_dim=self.d_model // self.num_heads
        )
        self.ffn = keras.Sequential([
            layers.Dense(self.ff_dim, activation='relu'),
            layers.Dense(self.d_model)
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(self.dropout_rate)
        self.dropout2 = layers.Dropout(self.dropout_rate)
        super(TransformerBlock, self).build(input_shape)
        
    def call(self, inputs, training=False):
        """Forward pass"""
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)
        
        return out2
    
    def get_config(self):
        """Get layer configuration"""
        config = super(TransformerBlock, self).get_config()
        config.update({
            'd_model': self.d_model,
            'num_heads': self.num_heads,
            'ff_dim': self.ff_dim,
            'dropout_rate': self.dropout_rate
        })
        return config


class TransformerModel:
    """Transformer-based Multi-output Model for trajectory classification
    
    This model uses self-attention mechanisms to capture temporal dependencies
    in trajectory sequences and can predict multiple output labels simultaneously
    (e.g., direction, altitude pattern, maneuver type, speed class).
    """
    
    def __init__(self, params: Dict[str, Any] = None):
        """Initialize Transformer model
        
        Args:
            params: Model parameters
        """
        if not HAS_TENSORFLOW:
            raise RuntimeError("TensorFlow is required for Transformer model")
        
        config = get_config()
        default_params = config.get('ml_params.transformer', {})
        self.params = {**default_params, **(params or {})}
        self.model = None
        self.sequence_generator = SequenceDataGenerator(self.params['sequence_length'])
        self.history = None
        self.multi_output = False
        self.output_names = []
        
    def build_model(self, input_shape: Tuple[int, int], n_classes: int, 
                    multi_output: bool = False, output_dims: Dict[str, int] = None) -> None:
        """Build Transformer model architecture
        
        Args:
            input_shape: (sequence_length, n_features)
            n_classes: Number of output classes (for single output)
            multi_output: Whether to use multi-output architecture
            output_dims: Dictionary of output names to number of classes (for multi-output)
        """
        self.multi_output = multi_output
        
        # Input layer
        inputs = keras.Input(shape=input_shape)
        
        # Linear projection to d_model dimensions
        x = layers.Dense(self.params['d_model'])(inputs)
        
        # Add positional encoding
        seq_len = input_shape[0]
        positions = tf.range(start=0, limit=seq_len, delta=1)
        position_embedding = layers.Embedding(
            input_dim=seq_len, 
            output_dim=self.params['d_model']
        )(positions)
        x = x + position_embedding
        
        # Stack Transformer blocks
        for _ in range(self.params['num_layers']):
            x = TransformerBlock(
                d_model=self.params['d_model'],
                num_heads=self.params['num_heads'],
                ff_dim=self.params['ff_dim'],
                dropout=self.params['dropout']
            )(x)
        
        # Global average pooling
        x = layers.GlobalAveragePooling1D()(x)
        x = layers.Dropout(self.params['dropout'])(x)
        x = layers.Dense(self.params['d_model'], activation='relu')(x)
        x = layers.Dropout(self.params['dropout'])(x)
        
        # Output layer(s)
        if multi_output and output_dims:
            # Multi-output architecture
            outputs = {}
            for output_name, n_out in output_dims.items():
                if n_out == 2:
                    # Binary output
                    outputs[output_name] = layers.Dense(1, activation='sigmoid', name=output_name)(x)
                else:
                    # Multi-class output
                    outputs[output_name] = layers.Dense(n_out, activation='softmax', name=output_name)(x)
            self.output_names = list(output_dims.keys())
            
            self.model = keras.Model(inputs=inputs, outputs=outputs)
            
            # Compile with multiple losses
            losses = {}
            metrics_dict = {}
            for output_name, n_out in output_dims.items():
                if n_out == 2:
                    losses[output_name] = 'binary_crossentropy'
                    metrics_dict[output_name] = ['accuracy']
                else:
                    losses[output_name] = 'sparse_categorical_crossentropy'
                    metrics_dict[output_name] = ['accuracy']
            
            self.model.compile(
                optimizer='adam',
                loss=losses,
                metrics=metrics_dict
            )
        else:
            # Single output architecture
            outputs = layers.Dense(n_classes, activation='softmax')(x)
            self.model = keras.Model(inputs=inputs, outputs=outputs)
            
            self.model.compile(
                optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
        
        logger.info(f"Built Transformer model with input shape {input_shape}, "
                   f"multi_output={multi_output}")
    
    def prepare_multi_output_labels(self, label_strings: np.ndarray) -> Dict[str, np.ndarray]:
        """Prepare multi-output labels from composite annotation strings
        
        Args:
            label_strings: Array of annotation strings (e.g., from sequences)
            
        Returns:
            Dictionary of label arrays for each output
        """
        # Define output categories
        direction_map = {'incoming': 0, 'outgoing': 1}
        altitude_map = {'ascending': 0, 'descending': 1, 'level': 2}
        path_map = {'linear': 0, 'curved': 1}
        maneuver_map = {'light_maneuver': 0, 'high_maneuver': 1}
        speed_map = {'low_speed': 0, 'high_speed': 1}
        
        n_samples = len(label_strings)
        labels = {
            'direction': np.zeros(n_samples, dtype=np.float32),
            'altitude': np.zeros(n_samples, dtype=np.int32),
            'path': np.zeros(n_samples, dtype=np.float32),
            'maneuver': np.zeros(n_samples, dtype=np.float32),
            'speed': np.zeros(n_samples, dtype=np.float32)
        }
        
        for idx, annotation in enumerate(label_strings):
            tags = str(annotation).split(',')
            tags = [tag.strip() for tag in tags]  # Strip whitespace
            
            # Direction
            if 'incoming' in tags:
                labels['direction'][idx] = 0
            elif 'outgoing' in tags:
                labels['direction'][idx] = 1
            
            # Altitude
            if 'ascending' in tags:
                labels['altitude'][idx] = 0
            elif 'descending' in tags:
                labels['altitude'][idx] = 1
            elif 'level' in tags or 'level_flight' in tags:
                labels['altitude'][idx] = 2
            
            # Path
            if 'linear' in tags:
                labels['path'][idx] = 0
            elif 'curved' in tags:
                labels['path'][idx] = 1
            
            # Maneuver
            if 'light_maneuver' in tags:
                labels['maneuver'][idx] = 0
            elif 'high_maneuver' in tags:
                labels['maneuver'][idx] = 1
            
            # Speed
            if 'low_speed' in tags:
                labels['speed'][idx] = 0
            elif 'high_speed' in tags:
                labels['speed'][idx] = 1
        
        return labels
    
    def train(self, df_train: pd.DataFrame, df_val: pd.DataFrame = None,
             use_multi_output: bool = False) -> Dict[str, Any]:
        """Train Transformer model
        
        Args:
            df_train: Training DataFrame
            df_val: Validation DataFrame (optional)
            use_multi_output: Whether to use multi-output architecture
            
        Returns:
            Training metrics
        """
        import time
        start_time = time.time()
        
        # Prepare enhanced feature columns for transformer
        feature_cols = [
            'x', 'y', 'z', 'vx', 'vy', 'vz', 'ax', 'ay', 'az',
            'speed', 'speed_2d', 'heading', 'range', 'range_rate', 
            'curvature', 'accel_magnitude', 'vertical_rate', 'altitude_change'
        ]
        feature_cols = [col for col in feature_cols if col in df_train.columns]
        
        if len(feature_cols) == 0:
            raise ValueError("No valid feature columns found in the training data")
        
        logger.info(f"Using {len(feature_cols)} features for transformer model: {feature_cols}")
        
        # Check if data has composite labels
        if use_multi_output or self._has_composite_labels(df_train):
            use_multi_output = True
            logger.info("Using multi-output architecture for composite labels")
        
        if use_multi_output:
            # Generate sequences with label strings (not encoded)
            X_train, y_train_strings, _ = self.sequence_generator.prepare_sequences(
                df_train, feature_cols, return_label_strings=True
            )
            X_train = self.sequence_generator.normalize_sequences(X_train, fit=True)
            
            # Prepare multi-output labels from the sequence label strings
            y_train_multi = self.prepare_multi_output_labels(y_train_strings)
            
            # Build multi-output model
            output_dims = {
                'direction': 2,
                'altitude': 3,
                'path': 2,
                'maneuver': 2,
                'speed': 2
            }
            self.build_model((X_train.shape[1], X_train.shape[2]), 0, 
                           multi_output=True, output_dims=output_dims)
            
            # Prepare validation data
            validation_data = None
            if df_val is not None:
                X_val, y_val_strings, _ = self.sequence_generator.prepare_sequences(
                    df_val, feature_cols, return_label_strings=True
                )
                X_val = self.sequence_generator.normalize_sequences(X_val, fit=False)
                y_val_multi = self.prepare_multi_output_labels(y_val_strings)
                validation_data = (X_val, y_val_multi)
            
            # Train
            self.history = self.model.fit(
                X_train, y_train_multi,
                epochs=self.params['epochs'],
                batch_size=self.params['batch_size'],
                validation_data=validation_data,
                verbose=0
            )
        else:
            # Single output training - use encoded labels
            X_train, y_train, _ = self.sequence_generator.prepare_sequences(
                df_train, feature_cols, return_label_strings=False
            )
            X_train = self.sequence_generator.normalize_sequences(X_train, fit=True)
            
            n_classes = len(self.sequence_generator.label_encoder.classes_)
            self.build_model((X_train.shape[1], X_train.shape[2]), n_classes)
            
            # Prepare validation data
            validation_data = None
            if df_val is not None:
                X_val, y_val, _ = self.sequence_generator.prepare_sequences(
                    df_val, feature_cols, return_label_strings=False
                )
                X_val = self.sequence_generator.normalize_sequences(X_val, fit=False)
                validation_data = (X_val, y_val)
            
            # Train
            self.history = self.model.fit(
                X_train, y_train,
                epochs=self.params['epochs'],
                batch_size=self.params['batch_size'],
                validation_data=validation_data,
                verbose=0
            )
        
        training_time = time.time() - start_time
        
        # Collect metrics
        metrics = {
            'training_time': training_time,
            'multi_output': use_multi_output,
            'history': {}
        }
        
        if use_multi_output:
            # Multi-output metrics
            for output_name in self.output_names:
                acc_key = f'{output_name}_accuracy'
                loss_key = f'{output_name}_loss'
                if acc_key in self.history.history:
                    metrics['history'][acc_key] = [float(x) for x in self.history.history[acc_key]]
                    metrics['history'][loss_key] = [float(x) for x in self.history.history[loss_key]]
                    metrics[f'train_{output_name}_accuracy'] = float(self.history.history[acc_key][-1])
            
            if validation_data:
                for output_name in self.output_names:
                    val_acc_key = f'val_{output_name}_accuracy'
                    val_loss_key = f'val_{output_name}_loss'
                    if val_acc_key in self.history.history:
                        metrics['history'][val_acc_key] = [float(x) for x in self.history.history[val_acc_key]]
                        metrics['history'][val_loss_key] = [float(x) for x in self.history.history[val_loss_key]]
                        metrics[f'val_{output_name}_accuracy'] = float(self.history.history[val_acc_key][-1])
        else:
            # Single output metrics
            metrics['train_accuracy'] = float(self.history.history['accuracy'][-1])
            metrics['n_classes'] = len(self.sequence_generator.label_encoder.classes_)
            metrics['classes'] = list(self.sequence_generator.label_encoder.classes_)
            metrics['history']['accuracy'] = [float(x) for x in self.history.history['accuracy']]
            metrics['history']['loss'] = [float(x) for x in self.history.history['loss']]
            
            if validation_data:
                metrics['val_accuracy'] = float(self.history.history['val_accuracy'][-1])
                metrics['history']['val_accuracy'] = [float(x) for x in self.history.history['val_accuracy']]
                metrics['history']['val_loss'] = [float(x) for x in self.history.history['val_loss']]
        
        logger.info(f"Transformer training completed in {training_time:.2f}s")
        
        return metrics
    
    def _has_composite_labels(self, df: pd.DataFrame) -> bool:
        """Check if DataFrame has composite labels"""
        if 'Annotation' not in df.columns:
            return False
        sample_label = str(df['Annotation'].iloc[0])
        return ',' in sample_label
    
    def evaluate(self, df_test: pd.DataFrame) -> Dict[str, Any]:
        """Evaluate Transformer model with enhanced features
        
        Args:
            df_test: Test DataFrame
            
        Returns:
            Evaluation metrics
        """
        feature_cols = [
            'x', 'y', 'z', 'vx', 'vy', 'vz', 'ax', 'ay', 'az',
            'speed', 'speed_2d', 'heading', 'range', 'range_rate', 
            'curvature', 'accel_magnitude', 'vertical_rate', 'altitude_change'
        ]
        feature_cols = [col for col in feature_cols if col in df_test.columns]
        
        if self.multi_output:
            # Multi-output evaluation - get label strings for sequences
            X_test, y_test_strings, _ = self.sequence_generator.prepare_sequences(
                df_test, feature_cols, return_label_strings=True
            )
            X_test = self.sequence_generator.normalize_sequences(X_test, fit=False)
            
            # Prepare multi-output labels from sequence label strings
            y_test_multi = self.prepare_multi_output_labels(y_test_strings)
            
            # Predictions
            y_pred_multi = self.model.predict(X_test, verbose=0)
            
            metrics = {'multi_output': True, 'outputs': {}}
            
            for output_name in self.output_names:
                y_true = y_test_multi[output_name]
                y_pred = y_pred_multi[output_name]
                
                if y_pred.shape[-1] == 1:
                    # Binary classification
                    y_pred_class = (y_pred > 0.5).astype(int).flatten()
                    accuracy = accuracy_score(y_true, y_pred_class)
                    f1 = f1_score(y_true, y_pred_class, average='binary', zero_division=0)
                else:
                    # Multi-class classification
                    y_pred_class = np.argmax(y_pred, axis=1)
                    accuracy = accuracy_score(y_true, y_pred_class)
                    f1 = f1_score(y_true, y_pred_class, average='weighted', zero_division=0)
                
                metrics['outputs'][output_name] = {
                    'accuracy': float(accuracy),
                    'f1_score': float(f1)
                }
            
            # Overall accuracy (average across outputs)
            overall_acc = np.mean([m['accuracy'] for m in metrics['outputs'].values()])
            metrics['accuracy'] = float(overall_acc)
            metrics['f1_score'] = float(np.mean([m['f1_score'] for m in metrics['outputs'].values()]))
            
        else:
            # Single output evaluation - use encoded labels
            X_test, y_test, _ = self.sequence_generator.prepare_sequences(
                df_test, feature_cols, return_label_strings=False
            )
            X_test = self.sequence_generator.normalize_sequences(X_test, fit=False)
            
            y_pred_proba = self.model.predict(X_test, verbose=0)
            y_pred = np.argmax(y_pred_proba, axis=1)
            
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            
            cm = confusion_matrix(y_test, y_pred)
            
            classes = self.sequence_generator.label_encoder.classes_
            # Get labels present in y_test and y_pred
            labels_present = np.unique(np.concatenate([y_test, y_pred]))
            target_names = [classes[i] for i in labels_present if i < len(classes)]
            
            report = classification_report(y_test, y_pred,
                                          labels=labels_present,
                                          target_names=target_names if len(target_names) == len(labels_present) else None,
                                          output_dict=True, zero_division=0)
            
            metrics = {
                'accuracy': float(accuracy),
                'f1_score': float(f1),
                'confusion_matrix': cm.tolist(),
                'classification_report': report,
                'classes': list(classes),
                'multi_output': False
            }
        
        logger.info(f"Transformer evaluation - Accuracy: {metrics['accuracy']:.4f}, "
                   f"F1: {metrics['f1_score']:.4f}")
        
        return metrics
    
    def save(self, path: str) -> None:
        """Save model to disk"""
        ensure_dir(Path(path).parent)
        self.model.save(path)
        
        # Save additional data
        metadata_path = Path(path).parent / f"{Path(path).stem}_metadata.pkl"
        joblib.dump({
            'sequence_generator': self.sequence_generator,
            'params': self.params,
            'multi_output': self.multi_output,
            'output_names': self.output_names
        }, metadata_path)
        
        logger.info(f"Saved Transformer model to {path}")
    
    def load(self, path: str) -> None:
        """Load model from disk"""
        # Register custom layer
        custom_objects = {'TransformerBlock': TransformerBlock}
        self.model = keras.models.load_model(path, custom_objects=custom_objects)
        
        metadata_path = Path(path).parent / f"{Path(path).stem}_metadata.pkl"
        data = joblib.load(metadata_path)
        self.sequence_generator = data['sequence_generator']
        self.params = data['params']
        self.multi_output = data.get('multi_output', False)
        self.output_names = data.get('output_names', [])
        
        logger.info(f"Loaded Transformer model from {path}")


class TransformerMultiOutputModel:
    """Transformer Multi-Output Model for tag prediction
    
    Uses a single transformer architecture with multiple output heads,
    one for each output tag column. Suitable for the data format where
    columns A-K are inputs and columns L-AF are output tags.
    """
    
    def __init__(self, params: Dict[str, Any] = None):
        """Initialize Transformer Multi-Output model
        
        Args:
            params: Model parameters
        """
        if not HAS_TENSORFLOW:
            raise RuntimeError("TensorFlow is required for Transformer model")
        
        config = get_config()
        default_params = config.get('ml_params.transformer', {})
        self.params = {**default_params, **(params or {})}
        self.model = None
        self.adapter = MultiOutputDataAdapter()
        self.history = None
        self.output_tag_names = []
        
    def train(self, df_train: pd.DataFrame, df_val: pd.DataFrame = None,
             input_cols: List[str] = None, output_cols: List[str] = None) -> Dict[str, Any]:
        """Train Transformer model for multi-output prediction
        
        Args:
            df_train: Training DataFrame
            df_val: Validation DataFrame (optional)
            input_cols: List of input feature column names (if None, auto-detect)
            output_cols: List of output tag column names (if None, auto-detect)
            
        Returns:
            Training metrics
        """
        import time
        start_time = time.time()
        
        # Identify columns
        self.adapter.identify_columns(df_train, input_cols, output_cols)
        self.output_tag_names = self.adapter.output_tag_columns
        
        # Prepare sequences
        X_train, Y_train, _ = self.adapter.prepare_sequences(
            df_train, 
            sequence_length=self.params['sequence_length'],
            filter_valid=True
        )
        
        # Normalize sequences
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        n_samples, seq_len, n_features = X_train.shape
        X_train_flat = X_train.reshape(-1, n_features)
        X_train_scaled = scaler.fit_transform(X_train_flat)
        X_train = X_train_scaled.reshape(n_samples, seq_len, n_features)
        
        self.scaler = scaler
        
        # Prepare validation data if available
        validation_data = None
        if df_val is not None:
            X_val, Y_val, _ = self.adapter.prepare_sequences(
                df_val,
                sequence_length=self.params['sequence_length'],
                filter_valid=True
            )
            X_val_flat = X_val.reshape(-1, n_features)
            X_val_scaled = scaler.transform(X_val_flat)
            X_val = X_val_scaled.reshape(X_val.shape[0], seq_len, n_features)
            
            # Prepare Y_val as dictionary
            Y_val_dict = {tag_name: Y_val[:, i] for i, tag_name in enumerate(self.output_tag_names)}
            validation_data = (X_val, Y_val_dict)
        
        # Build model
        self._build_model(input_shape=(X_train.shape[1], X_train.shape[2]))
        
        # Prepare Y_train as dictionary
        Y_train_dict = {tag_name: Y_train[:, i] for i, tag_name in enumerate(self.output_tag_names)}
        
        # Train
        self.history = self.model.fit(
            X_train, Y_train_dict,
            epochs=self.params['epochs'],
            batch_size=self.params['batch_size'],
            validation_data=validation_data,
            verbose=0
        )
        
        training_time = time.time() - start_time
        
        # Collect metrics
        metrics = {
            'training_time': training_time,
            'per_tag_metrics': {},
            'n_tags': len(self.output_tag_names),
            'tag_names': self.output_tag_names
        }
        
        # Extract per-tag metrics from history
        for tag_name in self.output_tag_names:
            acc_key = f'{tag_name}_accuracy'
            if acc_key in self.history.history:
                train_acc = float(self.history.history[acc_key][-1])
                metrics['per_tag_metrics'][tag_name] = {'train_accuracy': train_acc}
                
                if validation_data:
                    val_acc_key = f'val_{tag_name}_accuracy'
                    if val_acc_key in self.history.history:
                        val_acc = float(self.history.history[val_acc_key][-1])
                        metrics['per_tag_metrics'][tag_name]['val_accuracy'] = val_acc
        
        # Calculate overall metrics
        overall_train_acc = np.mean([m['train_accuracy'] for m in metrics['per_tag_metrics'].values()])
        metrics['train_accuracy'] = overall_train_acc
        
        if validation_data:
            overall_val_acc = np.mean([m['val_accuracy'] for m in metrics['per_tag_metrics'].values() if 'val_accuracy' in m])
            metrics['val_accuracy'] = overall_val_acc
        
        logger.info(f"Transformer Multi-Output training completed in {training_time:.2f}s")
        logger.info(f"Overall train accuracy: {overall_train_acc:.4f}")
        
        return metrics
    
    def _build_model(self, input_shape: Tuple[int, int]) -> None:
        """Build Transformer model architecture with multiple output heads
        
        Args:
            input_shape: (sequence_length, n_features)
        """
        # Input layer
        inputs = keras.Input(shape=input_shape)
        
        # Linear projection to d_model dimensions
        x = layers.Dense(self.params['d_model'])(inputs)
        
        # Add positional encoding
        seq_len = input_shape[0]
        positions = tf.range(start=0, limit=seq_len, delta=1)
        position_embedding = layers.Embedding(
            input_dim=seq_len,
            output_dim=self.params['d_model']
        )(positions)
        x = x + position_embedding
        
        # Stack Transformer blocks
        for _ in range(self.params['num_layers']):
            x = TransformerBlock(
                d_model=self.params['d_model'],
                num_heads=self.params['num_heads'],
                ff_dim=self.params['ff_dim'],
                dropout=self.params['dropout']
            )(x)
        
        # Global average pooling
        x = layers.GlobalAveragePooling1D()(x)
        x = layers.Dropout(self.params['dropout'])(x)
        x = layers.Dense(self.params['d_model'], activation='relu')(x)
        x = layers.Dropout(self.params['dropout'])(x)
        
        # Create output head for each tag (all binary classification)
        outputs = {}
        for tag_name in self.output_tag_names:
            outputs[tag_name] = layers.Dense(1, activation='sigmoid', name=tag_name)(x)
        
        self.model = keras.Model(inputs=inputs, outputs=outputs)
        
        # Compile with multiple losses
        losses = {tag_name: 'binary_crossentropy' for tag_name in self.output_tag_names}
        metrics_dict = {tag_name: ['accuracy'] for tag_name in self.output_tag_names}
        
        self.model.compile(
            optimizer='adam',
            loss=losses,
            metrics=metrics_dict
        )
        
        logger.info(f"Built Transformer Multi-Output model with {len(self.output_tag_names)} output heads")
    
    def evaluate(self, df_test: pd.DataFrame) -> Dict[str, Any]:
        """Evaluate model on test data
        
        Args:
            df_test: Test DataFrame
            
        Returns:
            Evaluation metrics
        """
        X_test, Y_test, _ = self.adapter.prepare_sequences(
            df_test,
            sequence_length=self.params['sequence_length'],
            filter_valid=True
        )
        
        # Normalize
        n_samples, seq_len, n_features = X_test.shape
        X_test_flat = X_test.reshape(-1, n_features)
        X_test_scaled = self.scaler.transform(X_test_flat)
        X_test = X_test_scaled.reshape(n_samples, seq_len, n_features)
        
        # Predict
        Y_pred_dict = self.model.predict(X_test, verbose=0)
        
        # Evaluate each tag
        tag_metrics = {}
        for i, tag_name in enumerate(self.output_tag_names):
            y_true = Y_test[:, i]
            y_pred_prob = Y_pred_dict[tag_name].flatten()
            y_pred = (y_pred_prob > 0.5).astype(int)
            
            acc = accuracy_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred, average='binary', zero_division=0)
            
            tag_metrics[tag_name] = {
                'accuracy': float(acc),
                'f1_score': float(f1)
            }
        
        # Overall metrics
        overall_acc = np.mean([m['accuracy'] for m in tag_metrics.values()])
        overall_f1 = np.mean([m['f1_score'] for m in tag_metrics.values()])
        
        metrics = {
            'accuracy': float(overall_acc),
            'f1_score': float(overall_f1),
            'per_tag_metrics': tag_metrics,
            'multi_output': True,
            'tag_names': self.output_tag_names
        }
        
        logger.info(f"Transformer Multi-Output evaluation - Accuracy: {overall_acc:.4f}, F1: {overall_f1:.4f}")
        
        return metrics
    
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Predict tags for input data
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with predicted tags
        """
        X, _, track_info = self.adapter.prepare_sequences(
            df,
            sequence_length=self.params['sequence_length'],
            filter_valid=False
        )
        
        # Normalize
        n_samples, seq_len, n_features = X.shape
        X_flat = X.reshape(-1, n_features)
        X_scaled = self.scaler.transform(X_flat)
        X = X_scaled.reshape(n_samples, seq_len, n_features)
        
        # Predict
        Y_pred_dict = self.model.predict(X, verbose=0)
        
        # Convert to binary predictions
        predictions = {}
        for tag_name in self.output_tag_names:
            y_pred_prob = Y_pred_dict[tag_name].flatten()
            predictions[tag_name] = (y_pred_prob > 0.5).astype(int)
        
        pred_df = pd.DataFrame(predictions)
        
        # Add aggregated annotation
        pred_df['Predicted_Annotation'] = self.adapter.create_aggregated_labels(pred_df)
        
        return pred_df
    
    def save(self, path: str) -> None:
        """Save model to disk"""
        ensure_dir(Path(path).parent)
        self.model.save(path)
        
        # Save additional data
        metadata_path = Path(path).parent / f"{Path(path).stem}_metadata.pkl"
        joblib.dump({
            'adapter': self.adapter,
            'scaler': self.scaler,
            'output_tag_names': self.output_tag_names,
            'params': self.params
        }, metadata_path)
        
        logger.info(f"Saved Transformer Multi-Output model to {path}")
    
    def load(self, path: str) -> None:
        """Load model from disk"""
        # Register custom layer
        custom_objects = {'TransformerBlock': TransformerBlock}
        self.model = keras.models.load_model(path, custom_objects=custom_objects)
        
        metadata_path = Path(path).parent / f"{Path(path).stem}_metadata.pkl"
        data = joblib.load(metadata_path)
        self.adapter = data['adapter']
        self.scaler = data['scaler']
        self.output_tag_names = data['output_tag_names']
        self.params = data['params']
        
        logger.info(f"Loaded Transformer Multi-Output model from {path}")


def load_trained_model(model_path: str) -> Tuple[Any, str]:
    """Load a trained model from disk
    
    Args:
        model_path: Path to the saved model file
        
    Returns:
        Tuple of (model_object, model_type)
        
    Raises:
        FileNotFoundError: If model file doesn't exist
        ValueError: If model type cannot be determined
    """
    model_path = Path(model_path)
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Determine model type from filename, parent directory, or extension
    model_name = model_path.stem.lower()
    parent_dir = model_path.parent.name.lower()
    full_path_lower = str(model_path).lower()
    
    # Priority 1: Check for .h5 files (Keras/TensorFlow models)
    if model_path.suffix == '.h5':
        if not HAS_TENSORFLOW:
            raise RuntimeError("TensorFlow is required to load neural network models")
        
        # Use TransformerModel for all neural network models
        model = TransformerModel()
        model.load(str(model_path))
        return model, 'neural_network'
    
    # Priority 2: Check for metadata pickle files (shouldn't be loaded directly)
    if '_metadata.pkl' in model_path.name or '_metadata' in model_name:
        # User selected a metadata file instead of the main model file
        # Try to find the corresponding .h5 file
        h5_name = model_name.replace('_metadata', '') + '.h5'
        h5_path = model_path.parent / h5_name
        
        if h5_path.exists():
            raise ValueError(
                f"You selected a metadata file. Please select the main model file instead:\n{h5_path}"
            )
        else:
            raise ValueError(
                f"You selected a metadata file ({model_path.name}), which cannot be loaded directly. "
                f"Please select the corresponding .h5 model file from the same directory."
            )
    
    # Priority 3: Check for neural network models by path indicators
    if 'neural_network' in full_path_lower or 'transformer' in full_path_lower or 'lstm' in full_path_lower:
        if not HAS_TENSORFLOW:
            raise RuntimeError("TensorFlow is required to load neural network models")
        
        # Look for .h5 file in the same directory
        h5_files = list(model_path.parent.glob('*.h5'))
        if h5_files:
            # Use the first .h5 file found (TransformerModel handles all neural network types)
            model = TransformerModel()
            model.load(str(h5_files[0]))
            return model, 'neural_network'
        else:
            raise ValueError(
                f"Model directory appears to be for a neural network model, but no .h5 file found. "
                f"Please select the .h5 model file."
            )
    
    # Priority 4: Check for Random Forest models
    if 'random_forest' in model_name or 'random_forest' in parent_dir or (model_path.suffix == '.pkl' and 'forest' in full_path_lower):
        model = RandomForestModel()
        model.load(str(model_path))
        return model, 'random_forest'
    
    # Priority 5: Check for XGBoost models
    elif 'gradient_boosting' in full_path_lower or 'xgboost' in full_path_lower or 'gradient' in full_path_lower:
        model = XGBoostModel()
        model.load(str(model_path))
        return model, 'gradient_boosting'
    
    # Priority 6: Try to determine from .pkl file contents
    elif model_path.suffix == '.pkl':
        import joblib
        try:
            data = joblib.load(str(model_path))
            
            # Check what keys are in the pickle file
            if isinstance(data, dict):
                keys = set(data.keys())
                
                # Check if it's an XGBoost/RandomForest model file
                if 'model' in keys and 'scaler' in keys and 'label_encoder' in keys:
                    # Could be either, default to XGBoost
                    model = XGBoostModel()
                    model.load(str(model_path))
                    return model, 'gradient_boosting'
                
                # Check if it's a multi-output model
                elif 'models' in keys and 'adapter' in keys:
                    # Multi-output model (could be XGBoost or RandomForest)
                    model = XGBoostMultiOutputModel()
                    model.load(str(model_path))
                    return model, 'gradient_boosting'
                
                # Check if it's metadata only (no 'model' or 'models' key)
                elif 'sequence_generator' in keys or 'output_tag_names' in keys:
                    raise ValueError(
                        f"This appears to be a metadata file for a neural network model. "
                        f"Please select the corresponding .h5 model file from the same directory."
                    )
            
            raise ValueError(f"Pickle file format not recognized: {model_path}")
            
        except Exception as e:
            if 'ValueError' in str(type(e).__name__):
                raise
            raise ValueError(f"Failed to load model file: {str(e)}")
    
    else:
        raise ValueError(
            f"Cannot determine model type from filename: {model_path}\n"
            f"Supported file types: .h5 (neural networks), .pkl (Random Forest/XGBoost)"
        )


def predict_and_label(model_path: str, input_csv_path: str, output_csv_path: str = None) -> pd.DataFrame:
    """Predict labels for unlabeled data using a trained model
    
    This function:
    1. Loads a trained model
    2. Reads the input CSV (which may be raw unlabeled data)
    3. Computes necessary motion features
    4. Uses the model to predict labels
    5. Saves results with predicted annotations
    
    Args:
        model_path: Path to trained model file
        input_csv_path: Path to input CSV (can be unlabeled)
        output_csv_path: Path to save labeled output (optional, defaults to input_labeled.csv)
        
    Returns:
        DataFrame with predicted annotations
        
    Raises:
        FileNotFoundError: If model or input file doesn't exist
        ValueError: If input data is invalid
    """
    logger.info(f"Loading model from {model_path}")
    model, model_type = load_trained_model(model_path)
    
    logger.info(f"Reading input data from {input_csv_path}")
    if not Path(input_csv_path).exists():
        raise FileNotFoundError(f"Input file not found: {input_csv_path}")
    
    df = pd.read_csv(input_csv_path)
    
    # Check required columns for feature computation
    required_cols = ['trackid', 'time', 'x', 'y', 'z', 'vx', 'vy', 'vz']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Input CSV is missing required columns: {missing_cols}")
    
    # Add acceleration columns if missing (initialize to zero)
    for col in ['ax', 'ay', 'az']:
        if col not in df.columns:
            df[col] = 0.0
    
    logger.info("Computing motion features...")
    # Import autolabel_engine to compute features
    from . import autolabel_engine
    df_features = autolabel_engine.compute_motion_features(df)
    
    # Prepare predictions based on model type
    logger.info(f"Generating predictions using {model_type} model...")
    
    if model_type in ['random_forest', 'gradient_boosting']:
        # For tabular models (RF, XGBoost)
        # Check if this is a multi-output model
        is_multi_output = hasattr(model, 'models') and hasattr(model, 'adapter')
        
        if is_multi_output:
            # Multi-output model prediction
            logger.info("Using multi-output model for prediction")
            
            # Filter to only valid features
            valid_mask = df_features['valid_features'].astype(bool)
            df_valid = df_features.loc[valid_mask].copy()
            
            if len(df_valid) == 0:
                logger.warning("No valid features found after filtering. Using all data.")
                df_valid = df_features.copy()
            
            # Use adapter to get input features
            X, _, _ = model.adapter.prepare_data(df_valid, filter_valid=False)
            
            # Scale features
            X_scaled = model.scaler.transform(X)
            
            # Predict each output tag
            predictions = {}
            for tag_name, tag_model in model.models.items():
                predictions[tag_name] = tag_model.predict(X_scaled)
            
            # Convert predictions back to composite labels
            composite_labels = []
            for i in range(len(X_scaled)):
                tags = []
                for tag_name in model.output_tag_names:
                    if tag_name in predictions and predictions[tag_name][i] == 1:
                        tags.append(tag_name)
                composite_labels.append(','.join(tags) if tags else 'normal')
            
            # Assign predictions back to dataframe
            df_features.loc[df_valid.index, 'Annotation'] = composite_labels
            
            # Mark invalid rows
            invalid_mask = df_features['valid_features'] == False
            df_features.loc[invalid_mask, 'Annotation'] = 'invalid'
            
        else:
            # Single-output model prediction
            logger.info("Using single-output model for prediction")
            
            # Filter to only valid features
            valid_mask = df_features['valid_features'].astype(bool)
            df_valid = df_features.loc[valid_mask].copy()
            
            if len(df_valid) == 0:
                logger.warning("No valid features found after filtering. Using all data.")
                df_valid = df_features.copy()
            
            # Get feature columns from the model
            if not hasattr(model, 'feature_columns') or model.feature_columns is None:
                raise ValueError("Model does not have feature_columns defined. Model may be corrupted.")
            
            feature_cols = model.feature_columns
            
            # Ensure all required features exist
            missing_features = [col for col in feature_cols if col not in df_valid.columns]
            if missing_features:
                raise ValueError(f"Input data is missing required features: {missing_features}")
            
            # Extract features
            X = df_valid[feature_cols].values
            
            # Scale features
            X_scaled = model.scaler.transform(X)
            
            # Predict
            y_pred_encoded = model.model.predict(X_scaled)
            
            # Decode labels
            y_pred = model.label_encoder.inverse_transform(y_pred_encoded)
            
            # Assign predictions back to dataframe
            df_features.loc[df_valid.index, 'Annotation'] = y_pred
            
            # Mark invalid rows
            invalid_mask = df_features['valid_features'] == False
            df_features.loc[invalid_mask, 'Annotation'] = 'invalid'
        
    elif model_type == 'neural_network':
        # For sequence models (Transformer/LSTM)
        feature_cols = [
            'x', 'y', 'z', 'vx', 'vy', 'vz', 'ax', 'ay', 'az',
            'speed', 'speed_2d', 'heading', 'range', 'range_rate', 
            'curvature', 'accel_magnitude', 'vertical_rate', 'altitude_change'
        ]
        feature_cols = [col for col in feature_cols if col in df_features.columns]
        
        # Prepare sequences (without labels - we'll predict them)
        sequences = []
        track_info = []  # Store (trackid, indices) for each sequence
        
        sequence_length = model.sequence_generator.sequence_length
        
        for trackid in df_features['trackid'].unique():
            track_df = df_features[df_features['trackid'] == trackid].sort_values('time').copy()
            
            # Filter valid features
            valid_mask = track_df['valid_features'].astype(bool)
            track_df = track_df.loc[valid_mask]
            
            if len(track_df) < 3:
                continue
            
            # Handle NaN and Inf
            track_df[feature_cols] = track_df[feature_cols].replace([np.inf, -np.inf], np.nan)
            track_df[feature_cols] = track_df[feature_cols].fillna(0)
            
            if len(track_df) < sequence_length:
                # Pad if too short
                n_pad = sequence_length - len(track_df)
                pad_df = pd.DataFrame(
                    np.zeros((n_pad, len(feature_cols))),
                    columns=feature_cols
                )
                track_features = pd.concat([pad_df, track_df[feature_cols]], ignore_index=True)
                sequences.append(track_features.values)
                track_info.append((trackid, track_df.index))
            else:
                # Use sliding window
                stride = max(1, sequence_length // 4)
                for i in range(0, len(track_df) - sequence_length + 1, stride):
                    window = track_df.iloc[i:i+sequence_length]
                    sequences.append(window[feature_cols].values)
                    track_info.append((trackid, window.index))
        
        if len(sequences) == 0:
            raise ValueError("No valid sequences could be created from the data")
        
        sequences = np.array(sequences)
        sequences = model.sequence_generator.normalize_sequences(sequences, fit=False)
        
        # Predict
        if model.multi_output:
            # Multi-output prediction
            y_pred_multi = model.model.predict(sequences, verbose=0)
            
            # Convert multi-output predictions back to composite labels
            predicted_labels = []
            for idx in range(len(sequences)):
                tags = []
                
                # Direction
                direction_pred = y_pred_multi['direction'][idx][0] if len(y_pred_multi['direction'][idx].shape) > 0 else y_pred_multi['direction'][idx]
                if direction_pred < 0.5:
                    tags.append('incoming')
                else:
                    tags.append('outgoing')
                
                # Altitude
                altitude_pred = np.argmax(y_pred_multi['altitude'][idx]) if len(y_pred_multi['altitude'][idx].shape) > 0 else int(y_pred_multi['altitude'][idx])
                if altitude_pred == 0:
                    tags.append('ascending')
                elif altitude_pred == 1:
                    tags.append('descending')
                else:
                    tags.append('level')
                
                # Path
                path_pred = y_pred_multi['path'][idx][0] if len(y_pred_multi['path'][idx].shape) > 0 else y_pred_multi['path'][idx]
                if path_pred < 0.5:
                    tags.append('linear')
                else:
                    tags.append('curved')
                
                # Maneuver
                maneuver_pred = y_pred_multi['maneuver'][idx][0] if len(y_pred_multi['maneuver'][idx].shape) > 0 else y_pred_multi['maneuver'][idx]
                if maneuver_pred < 0.5:
                    tags.append('light_maneuver')
                else:
                    tags.append('high_maneuver')
                
                # Speed
                speed_pred = y_pred_multi['speed'][idx][0] if len(y_pred_multi['speed'][idx].shape) > 0 else y_pred_multi['speed'][idx]
                if speed_pred < 0.5:
                    tags.append('low_speed')
                else:
                    tags.append('high_speed')
                
                predicted_labels.append(','.join(tags))
        else:
            # Single output prediction
            y_pred_proba = model.model.predict(sequences, verbose=0)
            y_pred_encoded = np.argmax(y_pred_proba, axis=1)
            predicted_labels = model.sequence_generator.label_encoder.inverse_transform(y_pred_encoded)
        
        # Assign predictions to dataframe
        # For each sequence, assign the predicted label to all points in that sequence
        df_features['Annotation'] = 'invalid'  # Default for unpredicted rows
        
        for idx, (trackid, indices) in enumerate(track_info):
            df_features.loc[indices, 'Annotation'] = predicted_labels[idx]
    
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    # Set default output path if not provided
    if output_csv_path is None:
        input_path = Path(input_csv_path)
        output_csv_path = str(input_path.parent / f"{input_path.stem}_labeled.csv")
    
    # Save results
    logger.info(f"Saving labeled data to {output_csv_path}")
    df_features.to_csv(output_csv_path, index=False)
    
    # Print summary
    annotation_counts = df_features['Annotation'].value_counts()
    logger.info(f"Prediction summary:")
    logger.info(f"  Total records: {len(df_features)}")
    logger.info(f"  Unique labels: {len(annotation_counts)}")
    logger.info(f"  Top 5 labels:")
    for label, count in annotation_counts.head(5).items():
        logger.info(f"    {label}: {count} ({count/len(df_features)*100:.1f}%)")
    
    return df_features


def train_model(model_name: str, data_path: str, output_dir: str, params: Dict[str, Any] = None, 
                auto_transform: bool = True) -> Tuple[Any, Dict[str, Any]]:
    """Train a model and save results with automatic label transformation
    
    Args:
        model_name: Model type ('random_forest', 'gradient_boosting', 'neural_network')
        data_path: Path to labeled data CSV
        output_dir: Output directory for model and results
        params: Model parameters (optional)
        auto_transform: Automatically transform labels if insufficient diversity (default: True)
        
    Returns:
        Tuple of (model, metrics)
    
    Note:
        The 'neural_network' (Transformer) model automatically detects composite labels (comma-separated)
        and uses multi-output architecture to predict each component separately.
    """
    logger.info(f"Training {model_name} model from {data_path}")
    
    # Attempt training with auto-transformation if enabled
    if auto_transform:
        try:
            return _train_model_with_recovery(model_name, data_path, output_dir, params)
        except Exception as e:
            # If auto-transform fails, fall through to original error handling
            logger.warning(f"Auto-transformation failed: {e}")
            logger.info("Attempting standard training...")
    
    # Original training logic (without auto-transform)
    return _train_model_impl(model_name, data_path, output_dir, params)


def _train_model_with_recovery(model_name: str, data_path: str, output_dir: str, 
                               params: Dict[str, Any] = None) -> Tuple[Any, Dict[str, Any]]:
    """Train model with automatic label diversity recovery
    
    Args:
        model_name: Model type
        data_path: Path to labeled data CSV
        output_dir: Output directory
        params: Model parameters
        
    Returns:
        Tuple of (model, metrics)
    """
    try:
        # Try normal training first
        return _train_model_impl(model_name, data_path, output_dir, params)
    except (ValueError, Exception) as e:
        error_str = str(e)
        
        # Check if it's a label diversity or unseen labels error
        if ("Insufficient classes" in error_str or 
            "unique class" in error_str.lower() or
            "previously unseen labels" in error_str.lower()):
            logger.warning("⚠️  Insufficient label diversity detected - attempting automatic recovery...")
            
            # Load data and analyze
            df = pd.read_csv(data_path)
            
            if 'Annotation' not in df.columns:
                raise
            
            # Apply automatic transformation
            transformer = LabelTransformer()
            analysis = transformer.analyze_label_diversity(df['Annotation'])
            
            logger.info(f"📊 Analysis: {analysis['n_unique_labels']} unique labels found")
            logger.info(f"   Recommended strategy: {analysis['recommended_strategy']}")
            
            # Transform data
            df_transformed, transform_info = transformer.auto_transform(df)
            
            if not transform_info.get('success', False):
                logger.error("❌ Automatic transformation failed")
                raise ValueError(
                    f"Cannot recover from label diversity issue.\n"
                    f"Reason: {transform_info.get('reason', 'Unknown')}\n\n"
                    f"Original error:\n{error_str}"
                )
            
            # Save transformed data
            transformed_path = Path(output_dir) / 'transformed_training_data.csv'
            ensure_dir(Path(output_dir))
            df_transformed.to_csv(transformed_path, index=False)
            logger.info(f"✅ Saved transformed data to {transformed_path}")
            
            # Log transformation details
            logger.info(f"🔄 Applied transformation: {transform_info['transformation']}")
            if 'n_labels' in transform_info:
                logger.info(f"   Created {transform_info['n_labels']} unique labels")
            if 'binary_label_columns' in transform_info:
                logger.info(f"   Binary columns: {', '.join(transform_info['binary_label_columns'])}")
            
            # Retry training with transformed data
            logger.info("🔁 Retrying training with transformed labels...")
            model, metrics = _train_model_impl(model_name, str(transformed_path), output_dir, params)
            
            # Add transformation info to metrics
            metrics['label_transformation'] = transform_info
            metrics['original_data_path'] = data_path
            metrics['transformed_data_path'] = str(transformed_path)
            
            logger.info("✅ Training succeeded with automatic label transformation!")
            
            return model, metrics
        else:
            # Not a label diversity issue - re-raise
            raise


def _train_model_impl(model_name: str, data_path: str, output_dir: str, 
                      params: Dict[str, Any] = None) -> Tuple[Any, Dict[str, Any]]:
    """Internal implementation of model training
    
    Args:
        model_name: Model type ('random_forest', 'gradient_boosting', 'neural_network')
        data_path: Path to labeled data CSV
        output_dir: Output directory for model and results
        params: Model parameters (optional)
        
    Returns:
        Tuple of (model, metrics)
        
    Supported Models:
        - random_forest: Random Forest classifier for tabular features
        - gradient_boosting: XGBoost gradient boosting for tabular features
        - neural_network: Transformer with multi-head attention and multi-output support
    """
    
    # Validate input file exists
    if not Path(data_path).exists():
        error_msg = f"Training data file not found: {data_path}"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)
    
    # Validate file is readable
    if not os.access(data_path, os.R_OK):
        error_msg = f"Training data file is not readable: {data_path}"
        logger.error(error_msg)
        raise PermissionError(error_msg)
    
    # Load data
    try:
        df = pd.read_csv(data_path)
    except Exception as e:
        error_msg = f"Failed to read CSV file {data_path}: {str(e)}"
        logger.error(error_msg)
        raise ValueError(error_msg) from e
    
    # Validate required columns
    required_columns = ['trackid', 'Annotation']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        error_msg = f"CSV file is missing required columns: {missing_columns}. Available columns: {list(df.columns)}"
        if 'Annotation' in missing_columns:
            error_msg += "\n\nThis appears to be raw radar data without labels."
            error_msg += "\nTo train a model, you need labeled data with an 'Annotation' column."
            error_msg += "\n\nOptions:"
            error_msg += "\n  1. Use the Auto-Labeling tool to generate annotations from raw data"
            error_msg += "\n  2. Select a file that already has annotations (e.g., labelled_data_*.csv)"
            error_msg += "\n  3. Manually add an 'Annotation' column to your CSV file"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    # Check if data is empty
    if len(df) == 0:
        error_msg = f"CSV file is empty: {data_path}"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    # Get unique tracks
    track_ids = df['trackid'].unique()
    n_tracks = len(track_ids)
    
    logger.info(f"Dataset contains {n_tracks} unique track(s) with {len(df)} total samples")
    
    # Check minimum requirements
    if n_tracks < 1:
        raise ValueError("Dataset must contain at least 1 track")
    
    # Handle small datasets
    config = get_config()
    test_size = 1.0 - config.get('ml_params.train_test_split', 0.8)
    
    if n_tracks < 3:
        # Too few tracks to split - use all data for training
        logger.warning(f"Only {n_tracks} track(s) available. Using all data for training without validation/test split.")
        logger.warning("For proper model evaluation, at least 3 tracks are recommended.")
        
        df_train_sub = df.copy()
        df_val = None
        df_test = None
        train_tracks_sub = track_ids
        val_tracks = []
        test_tracks = []
    else:
        # Normal splitting with sufficient data
        train_tracks, test_tracks = train_test_split(track_ids, test_size=test_size, random_state=42)
        
        df_train = df[df['trackid'].isin(train_tracks)]
        df_test = df[df['trackid'].isin(test_tracks)]
        
        # Further split train into train/val if enough tracks
        if len(train_tracks) < 2:
            # Not enough tracks for validation split
            logger.warning(f"Only {len(train_tracks)} training track(s). Skipping validation split.")
            df_train_sub = df_train.copy()
            df_val = None
            train_tracks_sub = train_tracks
            val_tracks = []
        else:
            train_tracks_sub, val_tracks = train_test_split(train_tracks, test_size=0.2, random_state=42)
            df_train_sub = df[df['trackid'].isin(train_tracks_sub)]
            df_val = df[df['trackid'].isin(val_tracks)]
    
    logger.info(f"Data split - Train: {len(train_tracks_sub)} tracks, Val: {len(val_tracks)} tracks, Test: {len(test_tracks)} tracks")
    
    ensure_dir(output_dir)
    
    # Train model
    if model_name == 'random_forest':
        model = RandomForestModel(params)
        X_train, y_train = model.prepare_features(df_train_sub)
        
        # Prepare validation data if available
        X_val, y_val = None, None
        if df_val is not None and len(df_val) > 0:
            X_val, y_val = model.prepare_features(df_val)
        
        train_metrics = model.train(X_train, y_train, X_val, y_val)
        
        # Evaluate on test set if available
        if df_test is not None and len(df_test) > 0:
            X_test, y_test = model.prepare_features(df_test)
            test_metrics = model.evaluate(X_test, y_test)
        else:
            # No test data - use training metrics as placeholder
            logger.warning("No test data available. Using training set for evaluation (not recommended).")
            test_metrics = {
                'accuracy': train_metrics.get('train_accuracy', 0.0),
                'f1_score': 0.0,
                'confusion_matrix': [],
                'classification_report': {},
                'classes': train_metrics.get('classes', []),
                'note': 'Evaluated on training set due to insufficient data for test split'
            }
        
        model_path = Path(output_dir) / 'random_forest_model.pkl'
        model.save(str(model_path))
        
    elif model_name == 'gradient_boosting' or model_name == 'xgboost':
        model = XGBoostModel(params)
        X_train, y_train = model.prepare_features(df_train_sub)
        
        # Prepare validation data if available
        X_val, y_val = None, None
        if df_val is not None and len(df_val) > 0:
            X_val, y_val = model.prepare_features(df_val)
        
        train_metrics = model.train(X_train, y_train, X_val, y_val)
        
        # Evaluate on test set if available
        if df_test is not None and len(df_test) > 0:
            X_test, y_test = model.prepare_features(df_test)
            test_metrics = model.evaluate(X_test, y_test)
        else:
            # No test data - use training metrics as placeholder
            logger.warning("No test data available. Using training set for evaluation (not recommended).")
            test_metrics = {
                'accuracy': train_metrics.get('train_accuracy', 0.0),
                'f1_score': 0.0,
                'confusion_matrix': [],
                'classification_report': {},
                'classes': train_metrics.get('classes', []),
                'note': 'Evaluated on training set due to insufficient data for test split'
            }
        
        model_path = Path(output_dir) / 'gradient_boosting_model.pkl'
        model.save(str(model_path))
        
    elif model_name == 'neural_network' or model_name == 'transformer':
        if not HAS_TENSORFLOW:
            raise RuntimeError("TensorFlow is required for Transformer model")
        
        model = TransformerModel(params)
        
        # Check if we should use multi-output mode
        use_multi_output = False
        if 'Annotation' in df_train_sub.columns:
            sample_label = str(df_train_sub['Annotation'].iloc[0])
            use_multi_output = ',' in sample_label
        
        train_metrics = model.train(df_train_sub, df_val, use_multi_output=use_multi_output)
        
        # Evaluate on test set if available
        if df_test is not None and len(df_test) > 0:
            test_metrics = model.evaluate(df_test)
        else:
            logger.warning("No test data available. Using training set for evaluation (not recommended).")
            test_metrics = {
                'accuracy': train_metrics.get('train_accuracy', 0.0),
                'f1_score': 0.0,
                'confusion_matrix': [],
                'classification_report': {},
                'classes': train_metrics.get('classes', []),
                'multi_output': train_metrics.get('multi_output', False),
                'note': 'Evaluated on training set due to insufficient data for test split'
            }
        
        model_path = Path(output_dir) / 'neural_network_model.h5'
        model.save(str(model_path))
        
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    
    # Combine metrics
    metrics = {
        'model_name': model_name,
        'train': train_metrics,
        'test': test_metrics
    }
    
    # Save metrics
    metrics_path = Path(output_dir) / f'{model_name}_metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    logger.info(f"Training complete. Model saved to {model_path}")
    
    return model, metrics


def train_multi_output_models(data_path: str = "data/high_volume_simulation_labeled.csv",
                               output_dir: str = "output/multi_output_models") -> int:
    """Train all three multi-output models (XGBoost, Random Forest, Transformer) and compare performance
    
    This function trains all three multi-output models for auto-tagging and auto-annotation
    based on the data format:
    - Columns A-K: Input features (radar measurements)
    - Columns L-AF: Output tags (to be predicted)
    - Column AG: Aggregated annotation (reference)
    
    All three models will predict multiple tag columns simultaneously.
    
    Args:
        data_path: Path to labeled CSV dataset
        output_dir: Output directory for trained models
        
    Returns:
        0 on success, 1 on failure
    """
    
    # Check if dataset exists
    if not Path(data_path).exists():
        logger.error(f"Dataset not found: {data_path}")
        logger.info("Please provide a valid dataset path")
        return 1
    
    logger.info("=" * 80)
    logger.info("TRAINING MULTI-OUTPUT MODELS FOR AUTO-TAGGING")
    logger.info("=" * 80)
    logger.info(f"Dataset: {data_path}")
    logger.info("Data format:")
    logger.info("  - Columns A-K: Input features (x, y, z, velocities, etc.)")
    logger.info("  - Columns L-AF: Output tags (incoming, outgoing, level, etc.)")
    logger.info("  - Column AG: Aggregated annotation (reference)")
    logger.info("=" * 80)
    
    # Load data
    logger.info("Loading data...")
    df = pd.read_csv(data_path)
    logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")
    logger.info(f"Unique tracks: {df['trackid'].nunique()}")
    
    # Split data by track ID
    track_ids = df['trackid'].unique()
    train_ids, test_ids = train_test_split(track_ids, test_size=0.2, random_state=42)
    train_ids, val_ids = train_test_split(train_ids, test_size=0.2, random_state=42)
    
    df_train = df[df['trackid'].isin(train_ids)]
    df_val = df[df['trackid'].isin(val_ids)]
    df_test = df[df['trackid'].isin(test_ids)]
    
    logger.info(f"Data split:")
    logger.info(f"  Train: {len(train_ids)} tracks, {len(df_train)} samples")
    logger.info(f"  Val:   {len(val_ids)} tracks, {len(df_val)} samples")
    logger.info(f"  Test:  {len(test_ids)} tracks, {len(df_test)} samples")
    logger.info("")
    
    models = {}
    metrics = {}
    
    # ========================================================================
    # 1. Train XGBoost Multi-Output Model
    # ========================================================================
    logger.info("=" * 80)
    logger.info("1/3: TRAINING XGBOOST MULTI-OUTPUT MODEL")
    logger.info("=" * 80)
    try:
        model_xgb = XGBoostMultiOutputModel(params={
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'random_state': 42
        })
        
        train_metrics_xgb = model_xgb.train(df_train, df_val)
        test_metrics_xgb = model_xgb.evaluate(df_test)
        
        # Save model
        model_path_xgb = Path(output_dir) / 'xgboost_multi_output' / 'model.pkl'
        model_xgb.save(str(model_path_xgb))
        
        models['xgboost'] = model_xgb
        metrics['xgboost'] = {
            'train': train_metrics_xgb,
            'test': test_metrics_xgb
        }
        
        logger.info("✓ XGBoost Multi-Output training completed")
        logger.info(f"  Overall Test Accuracy: {test_metrics_xgb['accuracy']:.4f}")
        logger.info(f"  Overall Test F1: {test_metrics_xgb['f1_score']:.4f}")
        logger.info("")
        
    except Exception as e:
        logger.error(f"✗ XGBoost training failed: {e}")
        import traceback
        traceback.print_exc()
        metrics['xgboost'] = None
    
    # ========================================================================
    # 2. Train Random Forest Multi-Output Model
    # ========================================================================
    logger.info("=" * 80)
    logger.info("2/3: TRAINING RANDOM FOREST MULTI-OUTPUT MODEL")
    logger.info("=" * 80)
    try:
        model_rf = RandomForestMultiOutputModel(params={
            'n_estimators': 100,
            'max_depth': 15,
            'min_samples_split': 2,
            'random_state': 42,
            'n_jobs': -1
        })
        
        train_metrics_rf = model_rf.train(df_train, df_val)
        test_metrics_rf = model_rf.evaluate(df_test)
        
        # Save model
        model_path_rf = Path(output_dir) / 'random_forest_multi_output' / 'model.pkl'
        model_rf.save(str(model_path_rf))
        
        models['random_forest'] = model_rf
        metrics['random_forest'] = {
            'train': train_metrics_rf,
            'test': test_metrics_rf
        }
        
        logger.info("✓ Random Forest Multi-Output training completed")
        logger.info(f"  Overall Test Accuracy: {test_metrics_rf['accuracy']:.4f}")
        logger.info(f"  Overall Test F1: {test_metrics_rf['f1_score']:.4f}")
        logger.info("")
        
    except Exception as e:
        logger.error(f"✗ Random Forest training failed: {e}")
        import traceback
        traceback.print_exc()
        metrics['random_forest'] = None
    
    # ========================================================================
    # 3. Train Transformer Multi-Output Model
    # ========================================================================
    logger.info("=" * 80)
    logger.info("3/3: TRAINING TRANSFORMER MULTI-OUTPUT MODEL")
    logger.info("=" * 80)
    try:
        model_transformer = TransformerMultiOutputModel(params={
            'd_model': 64,
            'num_heads': 4,
            'ff_dim': 128,
            'num_layers': 2,
            'dropout': 0.2,
            'epochs': 50,
            'batch_size': 32,
            'sequence_length': 20
        })
        
        train_metrics_transformer = model_transformer.train(df_train, df_val)
        test_metrics_transformer = model_transformer.evaluate(df_test)
        
        # Save model
        model_path_transformer = Path(output_dir) / 'transformer_multi_output' / 'model.h5'
        model_transformer.save(str(model_path_transformer))
        
        models['transformer'] = model_transformer
        metrics['transformer'] = {
            'train': train_metrics_transformer,
            'test': test_metrics_transformer
        }
        
        logger.info("✓ Transformer Multi-Output training completed")
        logger.info(f"  Overall Test Accuracy: {test_metrics_transformer['accuracy']:.4f}")
        logger.info(f"  Overall Test F1: {test_metrics_transformer['f1_score']:.4f}")
        logger.info("")
        
    except Exception as e:
        logger.error(f"✗ Transformer training failed: {e}")
        import traceback
        traceback.print_exc()
        metrics['transformer'] = None
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    logger.info("")
    logger.info("=" * 80)
    logger.info("MULTI-OUTPUT MODEL COMPARISON RESULTS")
    logger.info("=" * 80)
    logger.info("")
    
    # Create summary table
    print("┌─────────────────────────────┬──────────────┬──────────────┬──────────────┐")
    print("│ Model                       │   Accuracy   │   F1 Score   │ Training Time│")
    print("├─────────────────────────────┼──────────────┼──────────────┼──────────────┤")
    
    results = []
    for name, metric in metrics.items():
        if metric and metric['test']:
            test_acc = metric['test'].get('accuracy', 0)
            test_f1 = metric['test'].get('f1_score', 0)
            train_time = metric['train'].get('training_time', 0)
            
            results.append({
                'model': name,
                'accuracy': test_acc,
                'f1_score': test_f1,
                'time': train_time
            })
            
            print(f"│ {name.upper():<27} │   {test_acc:>8.4f}   │   {test_f1:>8.4f}   │   {train_time:>8.2f}s │")
    
    print("└─────────────────────────────┴──────────────┴──────────────┴──────────────┘")
    print("")
    
    # Per-tag breakdown
    logger.info("=" * 80)
    logger.info("PER-TAG PERFORMANCE BREAKDOWN")
    logger.info("=" * 80)
    
    for name, metric in metrics.items():
        if metric and metric['test'] and 'per_tag_metrics' in metric['test']:
            logger.info(f"\n{name.upper()}:")
            per_tag = metric['test']['per_tag_metrics']
            for tag_name, tag_metric in per_tag.items():
                acc = tag_metric['accuracy']
                f1 = tag_metric['f1_score']
                logger.info(f"  {tag_name:<25} Acc: {acc:.4f}  F1: {f1:.4f}")
    
    # Best models
    if results:
        logger.info("")
        logger.info("=" * 80)
        logger.info("SUMMARY")
        logger.info("=" * 80)
        
        best_by_accuracy = max(results, key=lambda x: x['accuracy'])
        best_by_f1 = max(results, key=lambda x: x['f1_score'])
        fastest = min(results, key=lambda x: x['time'])
        
        logger.info(f"🏆 Best Accuracy:  {best_by_accuracy['model'].upper()} ({best_by_accuracy['accuracy']:.4f})")
        logger.info(f"🏆 Best F1 Score:  {best_by_f1['model'].upper()} ({best_by_f1['f1_score']:.4f})")
        logger.info(f"⚡ Fastest:        {fastest['model'].upper()} ({fastest['time']:.2f}s)")
        logger.info("=" * 80)
        logger.info("")
        logger.info(f"✅ All models saved to: {output_dir}/")
        logger.info("   - xgboost_multi_output/model.pkl")
        logger.info("   - random_forest_multi_output/model.pkl")
        logger.info("   - transformer_multi_output/model.h5")
        logger.info("")
        logger.info("💡 These models can now predict multiple tags simultaneously for auto-tagging!")
    
    return 0


# CLI interface
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='AI Tagging Engine')
    parser.add_argument('--model', required=True, choices=['random_forest', 'gradient_boosting', 'neural_network'], 
                       help='Model type')
    parser.add_argument('--data', required=True, help='Path to labeled data CSV')
    parser.add_argument('--outdir', default='output/models', help='Output directory')
    
    args = parser.parse_args()
    
    model, metrics = train_model(args.model, args.data, args.outdir)
    
    # Display results in a formatted table
    print("\n" + "=" * 80)
    print(" " * 25 + f"TRAINING RESULTS - {args.model.upper()}")
    print("=" * 80)
    print("")
    
    print("┌─────────────────────────────────┬──────────────────────────────────┐")
    print("│ Metric                          │ Value                            │")
    print("├─────────────────────────────────┼──────────────────────────────────┤")
    print(f"│ Model Type                      │ {args.model:<32} │")
    print(f"│ Train Accuracy                  │ {metrics['train'].get('train_accuracy', 0):>32.4f} │")
    print(f"│ Test Accuracy                   │ {metrics['test'].get('accuracy', 0):>32.4f} │")
    print(f"│ Test F1 Score                   │ {metrics['test'].get('f1_score', 0):>32.4f} │")
    print(f"│ Training Time (s)               │ {metrics['train'].get('training_time', 0):>32.2f} │")
    
    if metrics['train'].get('multi_output', False):
        print("├─────────────────────────────────┼──────────────────────────────────┤")
        print("│ Multi-Output Results            │                                  │")
        print("├─────────────────────────────────┼──────────────────────────────────┤")
        if 'outputs' in metrics['test']:
            for output_name, output_metrics in metrics['test']['outputs'].items():
                print(f"│   {output_name:<27} │ Acc: {output_metrics['accuracy']:.4f} F1: {output_metrics['f1_score']:.4f}     │")
    
    print("└─────────────────────────────────┴──────────────────────────────────┘")
    print("")
    
    # Verdict
    print("=" * 80)
    print(" " * 32 + "VERDICT")
    print("=" * 80)
    print("")
    
    test_acc = metrics['test'].get('accuracy', 0)
    if test_acc > 0.95:
        print("🏆 EXCELLENT: Model achieved outstanding performance (>95% accuracy)")
        print("   ✅ Production-ready and highly reliable")
    elif test_acc > 0.85:
        print("✅ GOOD: Model shows strong performance (>85% accuracy)")
        print("   ✅ Suitable for deployment with monitoring")
    elif test_acc > 0.75:
        print("⚠️  MODERATE: Model has acceptable performance (>75% accuracy)")
        print("   💡 Consider collecting more training data or tuning hyperparameters")
    else:
        print("❌ NEEDS IMPROVEMENT: Model performance is below expectations (<75% accuracy)")
        print("   💡 Recommendations:")
        print("      • Collect more diverse training data")
        print("      • Feature engineering - add more relevant features")
        print("      • Try different model architectures")
        print("      • Check for data quality issues")
    
    # Check for overfitting
    train_acc = metrics['train'].get('train_accuracy', 0)
    overfit_gap = train_acc - test_acc
    print("")
    if overfit_gap > 0.15:
        print(f"⚠️  HIGH OVERFITTING DETECTED: Train-test gap = {overfit_gap:.4f}")
        print("   💡 Model may be memorizing training data. Try:")
        print("      • Increase regularization")
        print("      • Use more training data")
        print("      • Reduce model complexity")
    elif overfit_gap > 0.05:
        print(f"⚠️  SLIGHT OVERFITTING: Train-test gap = {overfit_gap:.4f}")
        print("   💡 Model is fitting well but could generalize better")
    else:
        print(f"✅ GOOD GENERALIZATION: Train-test gap = {overfit_gap:.4f}")
        print("   Model generalizes well to unseen data")
    
    print("")
    print("=" * 80)
