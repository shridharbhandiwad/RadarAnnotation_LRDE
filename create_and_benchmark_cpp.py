#!/usr/bin/env python3
"""
Comprehensive C++ Model Benchmark Script
Creates models, converts to TFLite, and runs detailed C++ benchmarks
"""

import os
import sys
import json
import pickle
import subprocess
import time
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    HAS_TF = True
except ImportError:
    print("ERROR: TensorFlow not installed. Install with: pip install tensorflow")
    HAS_TF = False
    sys.exit(1)

try:
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.model_selection import train_test_split
except ImportError:
    print("ERROR: scikit-learn not installed. Install with: pip install scikit-learn")
    sys.exit(1)


class ModelBenchmark:
    """Comprehensive model training and C++ benchmarking"""
    
    def __init__(self, output_dir='cpp_models', data_dir='data'):
        self.output_dir = Path(output_dir)
        self.data_dir = Path(data_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Model configuration
        self.sequence_length = 20
        self.n_features = 18
        self.n_classes = 2
        
        self.feature_columns = [
            'x', 'y', 'z', 'vx', 'vy', 'vz', 'ax', 'ay', 'az',
            'speed', 'speed_2d', 'heading', 'range', 'range_rate',
            'curvature', 'accel_magnitude', 'vertical_rate', 'altitude_change'
        ]
        
    def load_and_prepare_data(self) -> Tuple[np.ndarray, np.ndarray, StandardScaler, LabelEncoder]:
        """Load training data and prepare sequences"""
        print("=" * 80)
        print("STEP 1: Loading and Preparing Data")
        print("=" * 80)
        
        # Look for available data files
        data_files = list(self.data_dir.glob("*.csv"))
        if not data_files:
            print("No data files found, creating synthetic data...")
            return self._create_synthetic_data()
        
        # Use the first available data file
        data_file = data_files[0]
        print(f"Loading data from: {data_file}")
        
        df = pd.read_csv(data_file)
        print(f"Loaded {len(df)} rows")
        print(f"Columns: {list(df.columns)}")
        
        # Check if we have the required columns
        has_features = all(col in df.columns for col in self.feature_columns)
        has_label = any(col in df.columns for col in ['label', 'direction', 'class'])
        
        if not has_features or not has_label:
            print("Data doesn't have required format, creating synthetic data...")
            return self._create_synthetic_data()
        
        # Find label column
        label_col = None
        for col in ['label', 'direction', 'class']:
            if col in df.columns:
                label_col = col
                break
        
        print(f"Using label column: {label_col}")
        
        # Create sequences grouped by track_id
        sequences = []
        labels = []
        
        if 'trackid' in df.columns or 'track_id' in df.columns:
            track_col = 'trackid' if 'trackid' in df.columns else 'track_id'
            
            for track_id in df[track_col].unique():
                track_data = df[df[track_col] == track_id]
                
                if len(track_data) >= self.sequence_length:
                    # Take first sequence_length points
                    features = track_data[self.feature_columns].iloc[:self.sequence_length].values
                    sequences.append(features)
                    # Use the most common label for this track
                    labels.append(track_data[label_col].mode()[0])
        else:
            # No track_id, create sliding windows
            for i in range(0, len(df) - self.sequence_length, self.sequence_length):
                features = df[self.feature_columns].iloc[i:i+self.sequence_length].values
                sequences.append(features)
                labels.append(df[label_col].iloc[i])
        
        X = np.array(sequences)
        y = np.array(labels)
        
        print(f"Created {len(X)} sequences")
        print(f"Sequence shape: {X.shape}")
        print(f"Labels shape: {y.shape}")
        print(f"Unique labels: {np.unique(y)}")
        
        # Normalize features
        scaler = StandardScaler()
        X_reshaped = X.reshape(-1, self.n_features)
        X_scaled = scaler.fit_transform(X_reshaped)
        X = X_scaled.reshape(-1, self.sequence_length, self.n_features)
        
        # Encode labels
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)
        
        print(f"Encoded labels: {label_encoder.classes_}")
        
        return X, y_encoded, scaler, label_encoder
    
    def _create_synthetic_data(self) -> Tuple[np.ndarray, np.ndarray, StandardScaler, LabelEncoder]:
        """Create synthetic radar trajectory data"""
        print("Creating synthetic radar trajectories...")
        
        n_samples = 1000
        X = []
        y_labels = []
        
        for i in range(n_samples):
            # Create two types of trajectories: incoming and outgoing
            is_incoming = i % 2 == 0
            
            sequence = []
            for t in range(self.sequence_length):
                if is_incoming:
                    # Incoming: moving towards origin, decreasing range
                    x_pos = 10000 - t * 500 + np.random.randn() * 100
                    y_pos = 5000 + np.random.randn() * 100
                    vx = -50 + np.random.randn() * 5
                    range_rate = -30 + np.random.randn() * 3
                else:
                    # Outgoing: moving away from origin, increasing range
                    x_pos = 5000 + t * 500 + np.random.randn() * 100
                    y_pos = 5000 + np.random.randn() * 100
                    vx = 50 + np.random.randn() * 5
                    range_rate = 30 + np.random.randn() * 3
                
                z = 2000 + np.random.randn() * 50
                vy = np.random.randn() * 10
                vz = np.random.randn() * 2
                
                point = [
                    x_pos, y_pos, z,  # position
                    vx, vy, vz,  # velocity
                    np.random.randn(), np.random.randn(), np.random.randn(),  # acceleration
                    np.sqrt(vx**2 + vy**2 + vz**2),  # speed
                    np.sqrt(vx**2 + vy**2),  # speed_2d
                    np.arctan2(vy, vx),  # heading
                    np.sqrt(x_pos**2 + y_pos**2 + z**2),  # range
                    range_rate,  # range_rate
                    0.0001,  # curvature
                    np.sqrt(np.random.randn()**2),  # accel_magnitude
                    vz,  # vertical_rate
                    0.0  # altitude_change
                ]
                sequence.append(point)
            
            X.append(sequence)
            y_labels.append(0 if is_incoming else 1)
        
        X = np.array(X, dtype=np.float32)
        y = np.array(y_labels)
        
        # Normalize
        scaler = StandardScaler()
        X_reshaped = X.reshape(-1, self.n_features)
        X_scaled = scaler.fit_transform(X_reshaped)
        X = X_scaled.reshape(-1, self.sequence_length, self.n_features)
        
        # Create label encoder
        label_encoder = LabelEncoder()
        label_encoder.classes_ = np.array(['incoming', 'outgoing'])
        
        print(f"Created {len(X)} synthetic sequences")
        return X, y, scaler, label_encoder
    
    def train_lstm_model(self, X_train, y_train, X_val, y_val) -> keras.Model:
        """Train LSTM model for trajectory classification"""
        print("\n" + "=" * 80)
        print("STEP 2: Training LSTM Model")
        print("=" * 80)
        
        model = keras.Sequential([
            layers.Input(shape=(self.sequence_length, self.n_features)),
            layers.LSTM(64, return_sequences=True),
            layers.Dropout(0.2),
            layers.LSTM(32),
            layers.Dropout(0.2),
            layers.Dense(16, activation='relu'),
            layers.Dense(self.n_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print(model.summary())
        
        start_time = time.time()
        
        history = model.fit(
            X_train, y_train,
            epochs=20,
            batch_size=32,
            validation_data=(X_val, y_val),
            verbose=1
        )
        
        training_time = time.time() - start_time
        
        print(f"\nTraining completed in {training_time:.2f} seconds")
        print(f"Final training accuracy: {history.history['accuracy'][-1]:.4f}")
        print(f"Final validation accuracy: {history.history['val_accuracy'][-1]:.4f}")
        
        return model
    
    def convert_to_tflite(self, model: keras.Model, output_path: Path) -> Path:
        """Convert Keras model to TensorFlow Lite"""
        print("\n" + "=" * 80)
        print("STEP 3: Converting to TensorFlow Lite")
        print("=" * 80)
        
        tflite_path = output_path / "radar_model.tflite"
        
        # Convert to TFLite
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        
        # Enable SELECT_TF_OPS for LSTM support
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,
            tf.lite.OpsSet.SELECT_TF_OPS
        ]
        converter._experimental_lower_tensor_list_ops = False
        
        # Optimize
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        tflite_model = converter.convert()
        
        # Save
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)
        
        model_size_kb = len(tflite_model) / 1024
        print(f"TFLite model saved: {tflite_path}")
        print(f"Model size: {model_size_kb:.2f} KB")
        
        return tflite_path
    
    def save_metadata(self, scaler, label_encoder, output_path: Path):
        """Save model metadata for C++ inference"""
        print("\n" + "=" * 80)
        print("STEP 4: Saving Model Metadata")
        print("=" * 80)
        
        metadata = {
            'scaler_mean': scaler.mean_.tolist(),
            'scaler_scale': scaler.scale_.tolist(),
            'classes': label_encoder.classes_.tolist(),
            'n_classes': len(label_encoder.classes_),
            'feature_columns': self.feature_columns,
            'sequence_length': self.sequence_length,
            'n_features': self.n_features
        }
        
        metadata_path = output_path / "model_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Metadata saved: {metadata_path}")
        return metadata_path
    
    def create_test_data(self, X_test, y_test, output_path: Path):
        """Create test data for C++ benchmarking"""
        print("\n" + "=" * 80)
        print("STEP 5: Creating Test Data")
        print("=" * 80)
        
        # Save binary test data
        bin_path = output_path / "test_data.bin"
        X_test.astype(np.float32).tofile(bin_path)
        
        # Save CSV test data
        csv_path = output_path / "test_data.csv"
        
        # Flatten sequences and add track_id
        rows = []
        for i, seq in enumerate(X_test[:100]):  # Limit to 100 samples for CSV
            for j, point in enumerate(seq):
                row = [j * 0.1, i] + point.tolist()  # time, trackid, features
                rows.append(row)
        
        df = pd.DataFrame(rows, columns=['time', 'trackid'] + self.feature_columns)
        df.to_csv(csv_path, index=False)
        
        # Save metadata
        test_info = {
            'n_samples': len(X_test),
            'sequence_length': self.sequence_length,
            'n_features': self.n_features,
            'binary_file': str(bin_path),
            'csv_file': str(csv_path),
            'shape': list(X_test.shape)
        }
        
        info_path = output_path / "test_data_info.json"
        with open(info_path, 'w') as f:
            json.dump(test_info, f, indent=2)
        
        print(f"Binary test data: {bin_path} ({X_test.nbytes / 1024:.2f} KB)")
        print(f"CSV test data: {csv_path}")
        print(f"Test info: {info_path}")
        print(f"Number of test samples: {len(X_test)}")
        
        return bin_path, csv_path
    
    def run_cpp_benchmark(self, model_path: Path, metadata_path: Path, 
                         test_data_path: Path) -> Dict:
        """Run C++ inference benchmark"""
        print("\n" + "=" * 80)
        print("STEP 6: Running C++ Benchmarks")
        print("=" * 80)
        
        cpp_executable = Path("cpp_inference/build/radar_tagger")
        
        if not cpp_executable.exists():
            print(f"ERROR: C++ executable not found at {cpp_executable}")
            print("Please build the C++ project first:")
            print("  cd cpp_inference/build && make")
            return None
        
        results = {}
        
        # Test 1: Single inference with CSV data
        print("\n--- Test 1: Single Inference (CSV) ---")
        cmd = [
            str(cpp_executable),
            "--model", str(model_path),
            "--metadata", str(metadata_path),
            "--test-data", str(test_data_path.parent / "test_data.csv"),
            "--threads", "1"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("Errors:", result.stderr)
        results['single_thread_csv'] = result.stdout
        
        # Test 2: Multi-threaded inference with binary data
        print("\n--- Test 2: Multi-threaded Inference (Binary) ---")
        cmd = [
            str(cpp_executable),
            "--model", str(model_path),
            "--metadata", str(metadata_path),
            "--test-data", str(test_data_path),
            "--test-binary",
            "--samples", "100",
            "--seq-length", str(self.sequence_length),
            "--features", str(self.n_features),
            "--threads", "4"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("Errors:", result.stderr)
        results['multi_thread_binary'] = result.stdout
        
        # Test 3: Benchmark mode
        print("\n--- Test 3: Benchmark Mode (100 iterations) ---")
        cmd = [
            str(cpp_executable),
            "--model", str(model_path),
            "--metadata", str(metadata_path),
            "--test-data", str(test_data_path),
            "--test-binary",
            "--samples", "50",
            "--seq-length", str(self.sequence_length),
            "--features", str(self.n_features),
            "--threads", "4",
            "--benchmark"
        ]
        
        start_time = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        benchmark_time = time.time() - start_time
        
        print(result.stdout)
        if result.stderr:
            print("Errors:", result.stderr)
        results['benchmark'] = result.stdout
        results['total_benchmark_time'] = benchmark_time
        
        # Thread scaling test
        print("\n--- Test 4: Thread Scaling Analysis ---")
        for n_threads in [1, 2, 4, 8]:
            print(f"\nTesting with {n_threads} thread(s)...")
            cmd = [
                str(cpp_executable),
                "--model", str(model_path),
                "--metadata", str(metadata_path),
                "--test-data", str(test_data_path),
                "--test-binary",
                "--samples", "100",
                "--seq-length", str(self.sequence_length),
                "--features", str(self.n_features),
                "--threads", str(n_threads)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            results[f'threads_{n_threads}'] = result.stdout
            
            # Extract timing info
            for line in result.stdout.split('\n'):
                if 'Average Inference Time' in line:
                    print(f"  {n_threads} threads: {line.strip()}")
        
        return results
    
    def parse_cpp_output(self, output: str) -> Dict:
        """Parse C++ benchmark output to extract metrics"""
        metrics = {}
        
        for line in output.split('\n'):
            if 'Average Inference Time' in line:
                try:
                    time_str = line.split(':')[1].strip().replace(' ms', '')
                    metrics['avg_inference_ms'] = float(time_str)
                except:
                    pass
            elif 'Min Inference Time' in line:
                try:
                    time_str = line.split(':')[1].strip().replace(' ms', '')
                    metrics['min_inference_ms'] = float(time_str)
                except:
                    pass
            elif 'Max Inference Time' in line:
                try:
                    time_str = line.split(':')[1].strip().replace(' ms', '')
                    metrics['max_inference_ms'] = float(time_str)
                except:
                    pass
            elif 'Throughput' in line:
                try:
                    throughput_str = line.split(':')[1].strip().replace(' inferences/sec', '')
                    metrics['throughput_per_sec'] = float(throughput_str)
                except:
                    pass
            elif 'Total Inferences' in line:
                try:
                    count_str = line.split(':')[1].strip()
                    metrics['total_inferences'] = int(count_str)
                except:
                    pass
        
        return metrics
    
    def generate_report(self, results: Dict, output_path: Path):
        """Generate comprehensive benchmark report"""
        print("\n" + "=" * 80)
        print("STEP 7: Generating Benchmark Report")
        print("=" * 80)
        
        report = []
        report.append("=" * 80)
        report.append("C++ INFERENCE BENCHMARK REPORT")
        report.append("=" * 80)
        report.append("")
        
        # Summary table
        report.append("## Performance Summary")
        report.append("")
        
        metrics_table = []
        for test_name, output in results.items():
            if isinstance(output, str):
                metrics = self.parse_cpp_output(output)
                if metrics:
                    metrics_table.append({
                        'Test': test_name,
                        **metrics
                    })
        
        if metrics_table:
            report.append("| Test | Avg Time (ms) | Min Time (ms) | Max Time (ms) | Throughput (inf/s) |")
            report.append("|------|---------------|---------------|---------------|-------------------|")
            
            for m in metrics_table:
                test = m.get('Test', 'N/A')
                avg = m.get('avg_inference_ms', 'N/A')
                min_t = m.get('min_inference_ms', 'N/A')
                max_t = m.get('max_inference_ms', 'N/A')
                throughput = m.get('throughput_per_sec', 'N/A')
                
                report.append(f"| {test} | {avg} | {min_t} | {max_t} | {throughput} |")
        
        report.append("")
        report.append("## Key Metrics")
        report.append("")
        
        # Extract benchmark metrics
        if 'benchmark' in results:
            benchmark_metrics = self.parse_cpp_output(results['benchmark'])
            if benchmark_metrics:
                report.append(f"- **Single Inference Time**: {benchmark_metrics.get('avg_inference_ms', 'N/A')} ms")
                report.append(f"- **Min Inference Time**: {benchmark_metrics.get('min_inference_ms', 'N/A')} ms")
                report.append(f"- **Max Inference Time**: {benchmark_metrics.get('max_inference_ms', 'N/A')} ms")
                report.append(f"- **Throughput**: {benchmark_metrics.get('throughput_per_sec', 'N/A')} inferences/second")
                
                if 'avg_inference_ms' in benchmark_metrics:
                    latency_us = benchmark_metrics['avg_inference_ms'] * 1000
                    report.append(f"- **Latency (microseconds)**: {latency_us:.0f} µs")
        
        report.append("")
        report.append("## Real-time Performance Analysis")
        report.append("")
        
        if 'benchmark' in results:
            benchmark_metrics = self.parse_cpp_output(results['benchmark'])
            if 'avg_inference_ms' in benchmark_metrics:
                avg_ms = benchmark_metrics['avg_inference_ms']
                
                # Calculate real-time capabilities
                fps_100ms = 100 / avg_ms
                fps_50ms = 50 / avg_ms
                fps_10ms = 10 / avg_ms
                
                report.append(f"- Can process **{fps_100ms:.1f} trajectories** within 100ms update cycle")
                report.append(f"- Can process **{fps_50ms:.1f} trajectories** within 50ms update cycle")
                report.append(f"- Can process **{fps_10ms:.1f} trajectories** within 10ms update cycle")
                report.append("")
                
                if avg_ms < 1.0:
                    report.append("✅ **EXCELLENT**: Sub-millisecond inference, suitable for high-frequency real-time systems")
                elif avg_ms < 10.0:
                    report.append("✅ **GOOD**: Low-latency inference, suitable for real-time radar tracking")
                elif avg_ms < 50.0:
                    report.append("⚠️ **ACCEPTABLE**: Moderate latency, suitable for near real-time applications")
                else:
                    report.append("❌ **SLOW**: High latency, may need optimization for real-time use")
        
        report.append("")
        report.append("## Model Information")
        report.append("")
        report.append(f"- **Model Type**: LSTM Neural Network (TensorFlow Lite)")
        report.append(f"- **Sequence Length**: {self.sequence_length} timesteps")
        report.append(f"- **Features**: {self.n_features}")
        report.append(f"- **Classes**: {self.n_classes}")
        report.append(f"- **Framework**: TensorFlow Lite C++ API")
        
        report.append("")
        report.append("=" * 80)
        
        report_text = "\n".join(report)
        print(report_text)
        
        # Save report
        report_path = output_path / "benchmark_report.txt"
        with open(report_path, 'w') as f:
            f.write(report_text)
        
        print(f"\nReport saved to: {report_path}")
        
        # Save JSON metrics
        json_path = output_path / "benchmark_metrics.json"
        json_metrics = {}
        for test_name, output in results.items():
            if isinstance(output, str):
                json_metrics[test_name] = self.parse_cpp_output(output)
        
        with open(json_path, 'w') as f:
            json.dump(json_metrics, f, indent=2)
        
        print(f"JSON metrics saved to: {json_path}")
        
        return report_text
    
    def run(self):
        """Run complete benchmark pipeline"""
        print("\n")
        print("=" * 80)
        print(" COMPREHENSIVE C++ INFERENCE BENCHMARK")
        print(" TensorFlow Lite Model Performance Analysis")
        print("=" * 80)
        print("\n")
        
        # Step 1: Load and prepare data
        X, y, scaler, label_encoder = self.load_and_prepare_data()
        
        # Split data
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
        )
        
        print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        
        # Step 2: Train model
        model = self.train_lstm_model(X_train, y_train, X_val, y_val)
        
        # Step 3: Convert to TFLite
        tflite_path = self.convert_to_tflite(model, self.output_dir)
        
        # Step 4: Save metadata
        metadata_path = self.save_metadata(scaler, label_encoder, self.output_dir)
        
        # Step 5: Create test data
        bin_path, csv_path = self.create_test_data(X_test, y_test, self.output_dir)
        
        # Step 6: Run C++ benchmarks
        results = self.run_cpp_benchmark(tflite_path, metadata_path, bin_path)
        
        if results:
            # Step 7: Generate report
            self.generate_report(results, self.output_dir)
        
        print("\n")
        print("=" * 80)
        print(" BENCHMARK COMPLETE")
        print("=" * 80)
        print(f"\nAll files saved to: {self.output_dir}")
        print("\nGenerated files:")
        print(f"  - TFLite model: {tflite_path}")
        print(f"  - Metadata: {metadata_path}")
        print(f"  - Test data: {bin_path}")
        print(f"  - Benchmark report: {self.output_dir}/benchmark_report.txt")
        print(f"  - JSON metrics: {self.output_dir}/benchmark_metrics.json")


if __name__ == '__main__':
    benchmark = ModelBenchmark()
    benchmark.run()
