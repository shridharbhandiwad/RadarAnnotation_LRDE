"""AI Tagging Engine - Machine Learning models for trajectory classification"""
import numpy as np
import pandas as pd
import logging
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional
import joblib
import json

# ML imports
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
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
                         label_column: str = 'Annotation') -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Prepare sequence data from DataFrame
        
        Args:
            df: Input DataFrame
            feature_columns: List of feature column names
            label_column: Label column name
            
        Returns:
            Tuple of (sequences, labels, track_ids)
        """
        sequences = []
        labels = []
        track_ids = []
        
        for trackid in df['trackid'].unique():
            track_df = df[df['trackid'] == trackid].sort_values('time')
            
            if len(track_df) < self.sequence_length:
                # Pad if too short
                n_pad = self.sequence_length - len(track_df)
                pad_df = pd.DataFrame(
                    np.zeros((n_pad, len(feature_columns))),
                    columns=feature_columns
                )
                track_features = pd.concat([pad_df, track_df[feature_columns]], ignore_index=True)
                track_label = track_df[label_column].mode()[0] if len(track_df) > 0 else 'normal'
            else:
                # Use sliding window
                for i in range(len(track_df) - self.sequence_length + 1):
                    window = track_df.iloc[i:i+self.sequence_length]
                    track_features = window[feature_columns]
                    track_label = window[label_column].mode()[0]
                    
                    sequences.append(track_features.values)
                    labels.append(track_label)
                    track_ids.append(trackid)
                continue
            
            sequences.append(track_features.values)
            labels.append(track_label)
            track_ids.append(trackid)
        
        sequences = np.array(sequences)
        track_ids = np.array(track_ids)
        
        # Encode labels
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
        df_valid = df[df.get('valid_features', True)].copy()
        
        X = df_valid[feature_cols].values
        y = df_valid['Annotation'].values
        
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
        
        # Normalize features
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Encode labels
        y_train_encoded = self.label_encoder.fit_transform(y_train)
        
        # Train model
        self.model = xgb.XGBClassifier(**self.params)
        
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


class LSTMModel:
    """LSTM classifier for sequence data"""
    
    def __init__(self, params: Dict[str, Any] = None):
        """Initialize LSTM model
        
        Args:
            params: Model parameters
        """
        if not HAS_TENSORFLOW:
            raise RuntimeError("TensorFlow is required for LSTM model")
        
        config = get_config()
        default_params = config.get('ml_params.lstm', {})
        self.params = {**default_params, **(params or {})}
        self.model = None
        self.sequence_generator = SequenceDataGenerator(self.params['sequence_length'])
        self.history = None
        
    def build_model(self, input_shape: Tuple[int, int], n_classes: int) -> None:
        """Build LSTM model architecture
        
        Args:
            input_shape: (sequence_length, n_features)
            n_classes: Number of output classes
        """
        self.model = keras.Sequential([
            layers.LSTM(self.params['units'], return_sequences=True, input_shape=input_shape),
            layers.Dropout(self.params['dropout']),
            layers.LSTM(self.params['units'] // 2),
            layers.Dropout(self.params['dropout']),
            layers.Dense(64, activation='relu'),
            layers.Dense(n_classes, activation='softmax')
        ])
        
        self.model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        logger.info(f"Built LSTM model with input shape {input_shape}")
    
    def train(self, df_train: pd.DataFrame, df_val: pd.DataFrame = None) -> Dict[str, Any]:
        """Train LSTM model
        
        Args:
            df_train: Training DataFrame
            df_val: Validation DataFrame (optional)
            
        Returns:
            Training metrics
        """
        import time
        start_time = time.time()
        
        # Prepare feature columns
        feature_cols = ['x', 'y', 'z', 'vx', 'vy', 'vz', 'ax', 'ay', 'az', 
                       'speed', 'heading', 'range', 'curvature']
        feature_cols = [col for col in feature_cols if col in df_train.columns]
        
        # Generate sequences
        X_train, y_train, _ = self.sequence_generator.prepare_sequences(df_train, feature_cols)
        X_train = self.sequence_generator.normalize_sequences(X_train, fit=True)
        
        # Build model
        n_classes = len(self.sequence_generator.label_encoder.classes_)
        self.build_model((X_train.shape[1], X_train.shape[2]), n_classes)
        
        # Prepare validation data
        validation_data = None
        if df_val is not None:
            X_val, y_val, _ = self.sequence_generator.prepare_sequences(df_val, feature_cols)
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
        
        metrics = {
            'training_time': training_time,
            'train_accuracy': float(self.history.history['accuracy'][-1]),
            'n_classes': n_classes,
            'classes': list(self.sequence_generator.label_encoder.classes_),
            'history': {
                'accuracy': [float(x) for x in self.history.history['accuracy']],
                'loss': [float(x) for x in self.history.history['loss']]
            }
        }
        
        if validation_data:
            metrics['val_accuracy'] = float(self.history.history['val_accuracy'][-1])
            metrics['history']['val_accuracy'] = [float(x) for x in self.history.history['val_accuracy']]
            metrics['history']['val_loss'] = [float(x) for x in self.history.history['val_loss']]
        
        logger.info(f"LSTM training completed in {training_time:.2f}s, accuracy: {metrics['train_accuracy']:.4f}")
        
        return metrics
    
    def evaluate(self, df_test: pd.DataFrame) -> Dict[str, Any]:
        """Evaluate LSTM model
        
        Args:
            df_test: Test DataFrame
            
        Returns:
            Evaluation metrics
        """
        feature_cols = ['x', 'y', 'z', 'vx', 'vy', 'vz', 'ax', 'ay', 'az', 
                       'speed', 'heading', 'range', 'curvature']
        feature_cols = [col for col in feature_cols if col in df_test.columns]
        
        X_test, y_test, _ = self.sequence_generator.prepare_sequences(df_test, feature_cols)
        X_test = self.sequence_generator.normalize_sequences(X_test, fit=False)
        
        # Predictions
        y_pred_proba = self.model.predict(X_test, verbose=0)
        y_pred = np.argmax(y_pred_proba, axis=1)
        
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        cm = confusion_matrix(y_test, y_pred)
        
        classes = self.sequence_generator.label_encoder.classes_
        report = classification_report(y_test, y_pred, target_names=classes, 
                                      output_dict=True, zero_division=0)
        
        metrics = {
            'accuracy': accuracy,
            'f1_score': f1,
            'confusion_matrix': cm.tolist(),
            'classification_report': report,
            'classes': list(classes)
        }
        
        logger.info(f"LSTM evaluation - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
        
        return metrics
    
    def save(self, path: str) -> None:
        """Save model to disk"""
        ensure_dir(Path(path).parent)
        self.model.save(path)
        
        # Save additional data
        metadata_path = Path(path).parent / f"{Path(path).stem}_metadata.pkl"
        joblib.dump({
            'sequence_generator': self.sequence_generator,
            'params': self.params
        }, metadata_path)
        
        logger.info(f"Saved LSTM model to {path}")
    
    def load(self, path: str) -> None:
        """Load model from disk"""
        self.model = keras.models.load_model(path)
        
        metadata_path = Path(path).parent / f"{Path(path).stem}_metadata.pkl"
        data = joblib.load(metadata_path)
        self.sequence_generator = data['sequence_generator']
        self.params = data['params']
        
        logger.info(f"Loaded LSTM model from {path}")


def train_model(model_name: str, data_path: str, output_dir: str, params: Dict[str, Any] = None) -> Tuple[Any, Dict[str, Any]]:
    """Train a model and save results
    
    Args:
        model_name: Model type ('xgboost', 'lstm', 'transformer')
        data_path: Path to labeled data CSV
        output_dir: Output directory for model and results
        params: Model parameters (optional)
        
    Returns:
        Tuple of (model, metrics)
    """
    logger.info(f"Training {model_name} model from {data_path}")
    
    # Load data
    df = pd.read_csv(data_path)
    
    # Split by track
    config = get_config()
    test_size = 1.0 - config.get('ml_params.train_test_split', 0.8)
    
    track_ids = df['trackid'].unique()
    train_tracks, test_tracks = train_test_split(track_ids, test_size=test_size, random_state=42)
    
    df_train = df[df['trackid'].isin(train_tracks)]
    df_test = df[df['trackid'].isin(test_tracks)]
    
    # Further split train into train/val
    train_tracks_sub, val_tracks = train_test_split(train_tracks, test_size=0.2, random_state=42)
    df_train_sub = df[df['trackid'].isin(train_tracks_sub)]
    df_val = df[df['trackid'].isin(val_tracks)]
    
    logger.info(f"Data split - Train: {len(train_tracks_sub)} tracks, Val: {len(val_tracks)} tracks, Test: {len(test_tracks)} tracks")
    
    ensure_dir(output_dir)
    
    # Train model
    if model_name == 'xgboost':
        model = XGBoostModel(params)
        X_train, y_train = model.prepare_features(df_train_sub)
        X_val, y_val = model.prepare_features(df_val)
        train_metrics = model.train(X_train, y_train, X_val, y_val)
        
        X_test, y_test = model.prepare_features(df_test)
        test_metrics = model.evaluate(X_test, y_test)
        
        model_path = Path(output_dir) / 'xgboost_model.pkl'
        model.save(str(model_path))
        
    elif model_name == 'lstm':
        if not HAS_TENSORFLOW:
            raise RuntimeError("TensorFlow is required for LSTM model")
        
        model = LSTMModel(params)
        train_metrics = model.train(df_train_sub, df_val)
        test_metrics = model.evaluate(df_test)
        
        model_path = Path(output_dir) / 'lstm_model.h5'
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


# CLI interface
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='AI Tagging Engine')
    parser.add_argument('--model', required=True, choices=['xgboost', 'lstm'], 
                       help='Model type')
    parser.add_argument('--data', required=True, help='Path to labeled data CSV')
    parser.add_argument('--outdir', default='output/models', help='Output directory')
    
    args = parser.parse_args()
    
    model, metrics = train_model(args.model, args.data, args.outdir)
    
    print(f"\nTraining Results for {args.model}:")
    print(f"  Train Accuracy: {metrics['train'].get('train_accuracy', 0):.4f}")
    print(f"  Test Accuracy: {metrics['test'].get('accuracy', 0):.4f}")
    print(f"  Test F1 Score: {metrics['test'].get('f1_score', 0):.4f}")
