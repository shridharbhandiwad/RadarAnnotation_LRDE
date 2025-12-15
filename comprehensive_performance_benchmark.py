#!/usr/bin/env python3
"""
Comprehensive Performance Benchmark for Radar Trajectory Classification
Measures inference time, throughput, and provides detailed metrics for both
Python and projected C++ performance
"""

import os
import sys
import json
import time
import numpy as np
import tensorflow as tf
from pathlib import Path
from typing import Dict, List
from statistics import mean, stdev, median

print("=" * 80)
print(" COMPREHENSIVE PERFORMANCE BENCHMARK")
print(" Radar Trajectory Classification - Inference Metrics")
print("=" * 80)
print()

# Load the TFLite model
model_path = "cpp_models/radar_model.tflite"
metadata_path = "cpp_models/model_metadata.json"

if not os.path.exists(model_path):
    print(f"ERROR: Model not found at {model_path}")
    print("Please run create_and_benchmark_cpp.py first")
    sys.exit(1)

# Load model
print(f"Loading TFLite model from: {model_path}")
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print(f"Input shape: {input_details[0]['shape']}")
print(f"Output shape: {output_details[0]['shape']}")
print()

# Load metadata
with open(metadata_path, 'r') as f:
    metadata = json.load(f)

print(f"Classes: {metadata['classes']}")
print(f"Features: {metadata['n_features']}")
print(f"Sequence length: {metadata['sequence_length']}")
print()

# Create test data
test_data_path = "cpp_models/test_data.bin"
if os.path.exists(test_data_path):
    print(f"Loading test data from: {test_data_path}")
    test_data = np.fromfile(test_data_path, dtype=np.float32)
    n_samples = 150
    seq_length = metadata['sequence_length']
    n_features = metadata['n_features']
    test_data = test_data.reshape(n_samples, seq_length, n_features)
    print(f"Loaded {n_samples} test samples")
else:
    print("Creating synthetic test data...")
    n_samples = 150
    seq_length = metadata['sequence_length']
    n_features = metadata['n_features']
    test_data = np.random.randn(n_samples, seq_length, n_features).astype(np.float32)
    print(f"Created {n_samples} synthetic samples")

print()
print("=" * 80)
print(" BENCHMARK 1: Single Inference Performance")
print("=" * 80)
print()

# Warm-up
print("Warming up model...")
for i in range(10):
    interpreter.set_tensor(input_details[0]['index'], test_data[i:i+1])
    interpreter.invoke()

print("Running single inference benchmark...")
times = []
for i in range(100):
    start = time.perf_counter()
    interpreter.set_tensor(input_details[0]['index'], test_data[i % n_samples:i % n_samples + 1])
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    end = time.perf_counter()
    times.append((end - start) * 1000)  # Convert to ms

print(f"✓ Completed 100 inferences")
print()

# Statistics
avg_time = mean(times)
med_time = median(times)
min_time = min(times)
max_time = max(times)
std_time = stdev(times)

print(f"Average inference time: {avg_time:.4f} ms")
print(f"Median inference time:  {med_time:.4f} ms")
print(f"Min inference time:     {min_time:.4f} ms")
print(f"Max inference time:     {max_time:.4f} ms")
print(f"Std deviation:          {std_time:.4f} ms")
print(f"Throughput:             {1000/avg_time:.2f} inferences/sec")
print()

print("=" * 80)
print(" BENCHMARK 2: Batch Processing Performance")
print("=" * 80)
print()

batch_sizes = [1, 10, 50, 100]
batch_results = {}

for batch_size in batch_sizes:
    print(f"Testing batch size: {batch_size}")
    times = []
    
    num_batches = min(50, n_samples // batch_size)
    for i in range(num_batches):
        start_idx = (i * batch_size) % (n_samples - batch_size)
        batch_data = test_data[start_idx:start_idx + 1]  # TFLite expects batch=1
        
        batch_times = []
        for j in range(batch_size):
            start = time.perf_counter()
            interpreter.set_tensor(input_details[0]['index'], test_data[(start_idx + j) % n_samples:(start_idx + j) % n_samples + 1])
            interpreter.invoke()
            output = interpreter.get_tensor(output_details[0]['index'])
            end = time.perf_counter()
            batch_times.append((end - start) * 1000)
        
        times.extend(batch_times)
    
    avg_time = mean(times)
    throughput = 1000 / avg_time
    
    print(f"  Avg time per sample: {avg_time:.4f} ms")
    print(f"  Throughput: {throughput:.2f} samples/sec")
    print()
    
    batch_results[batch_size] = {
        'avg_time_ms': avg_time,
        'throughput_per_sec': throughput
    }

print("=" * 80)
print(" BENCHMARK 3: Latency Distribution Analysis")
print("=" * 80)
print()

# Run more samples for distribution
print("Running 1000 inferences for latency distribution...")
latency_samples = []
for i in range(1000):
    start = time.perf_counter()
    interpreter.set_tensor(input_details[0]['index'], test_data[i % n_samples:i % n_samples + 1])
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    end = time.perf_counter()
    latency_samples.append((end - start) * 1000)

# Compute percentiles
p50 = np.percentile(latency_samples, 50)
p90 = np.percentile(latency_samples, 90)
p95 = np.percentile(latency_samples, 95)
p99 = np.percentile(latency_samples, 99)

print(f"Latency Percentiles (ms):")
print(f"  P50 (median): {p50:.4f} ms")
print(f"  P90:          {p90:.4f} ms")
print(f"  P95:          {p95:.4f} ms")
print(f"  P99:          {p99:.4f} ms")
print()

print("=" * 80)
print(" MODEL ANALYSIS")
print("=" * 80)
print()

# Get model size
model_size_kb = os.path.getsize(model_path) / 1024
model_size_mb = model_size_kb / 1024

print(f"Model Size:")
print(f"  {model_size_kb:.2f} KB ({model_size_mb:.2f} MB)")
print()

# Model complexity
print(f"Model Complexity:")
print(f"  Input size: {seq_length} timesteps × {n_features} features = {seq_length * n_features} values")
print(f"  Output size: {metadata['n_classes']} classes")
print(f"  Memory per inference: {seq_length * n_features * 4} bytes (input) + {metadata['n_classes'] * 4} bytes (output)")
print()

print("=" * 80)
print(" REAL-TIME PERFORMANCE ANALYSIS")
print("=" * 80)
print()

# Calculate real-time capabilities
radar_update_rates = [10, 20, 50, 100]  # Hz

print(f"Real-time Processing Capability (based on {avg_time:.4f} ms avg latency):")
print()

for rate_hz in radar_update_rates:
    update_period_ms = 1000 / rate_hz
    max_tracks = int(update_period_ms / avg_time)
    print(f"  {rate_hz} Hz update rate ({update_period_ms:.1f} ms period):")
    print(f"    Can process: {max_tracks} trajectories per update")
    print(f"    Total throughput: {max_tracks * rate_hz} classifications/sec")
    print()

print("=" * 80)
print(" C++ PERFORMANCE PROJECTION")
print("=" * 80)
print()

# Estimate C++ performance improvement
# Typical speedup: 2-5x for optimized C++ vs Python TFLite
cpp_speedup_factors = {
    'conservative': 2.0,
    'typical': 3.5,
    'optimistic': 5.0
}

print("Projected C++ Performance (TensorFlow Lite C++ API):")
print()

for scenario, factor in cpp_speedup_factors.items():
    cpp_time = avg_time / factor
    cpp_throughput = 1000 / cpp_time
    
    print(f"{scenario.upper()} ({factor}x speedup):")
    print(f"  Inference time: {cpp_time:.4f} ms ({cpp_time * 1000:.1f} µs)")
    print(f"  Throughput: {cpp_throughput:.2f} inferences/sec")
    
    # Real-time capability at 50 Hz
    rate_hz = 50
    update_period_ms = 1000 / rate_hz
    max_tracks = int(update_period_ms / cpp_time)
    print(f"  Tracks @ 50 Hz: {max_tracks} trajectories/update")
    print()

print("=" * 80)
print(" MEMORY FOOTPRINT ANALYSIS")
print("=" * 80)
print()

input_size_bytes = seq_length * n_features * 4  # float32
output_size_bytes = metadata['n_classes'] * 4  # float32
model_size_bytes = os.path.getsize(model_path)

print(f"Memory Requirements:")
print(f"  Model (on disk):     {model_size_bytes:,} bytes ({model_size_bytes/1024:.1f} KB)")
print(f"  Model (in memory):   ~{model_size_bytes * 1.5:,.0f} bytes (with runtime overhead)")
print(f"  Input tensor:        {input_size_bytes} bytes")
print(f"  Output tensor:       {output_size_bytes} bytes")
print(f"  Per-inference:       ~{input_size_bytes + output_size_bytes + 1024:.0f} bytes (including temp buffers)")
print(f"  Total estimated:     ~{model_size_bytes * 1.5 + input_size_bytes + output_size_bytes + 1024:,.0f} bytes")
print()

print("=" * 80)
print(" DEPLOYMENT RECOMMENDATIONS")
print("=" * 80)
print()

if avg_time < 1.0:
    rating = "EXCELLENT"
    symbol = "✅"
    desc = "Sub-millisecond latency - ideal for high-frequency real-time systems"
elif avg_time < 5.0:
    rating = "VERY GOOD"
    symbol = "✅"
    desc = "Low latency - suitable for real-time radar tracking applications"
elif avg_time < 10.0:
    rating = "GOOD"
    symbol = "✓"
    desc = "Good latency - suitable for most real-time applications"
elif avg_time < 50.0:
    rating = "ACCEPTABLE"
    symbol = "⚠"
    desc = "Moderate latency - suitable for near real-time applications"
else:
    rating = "NEEDS OPTIMIZATION"
    symbol = "❌"
    desc = "High latency - may need model optimization"

print(f"{symbol} Performance Rating: {rating}")
print(f"   {desc}")
print()

print("Deployment Scenarios:")
print()
print(f"1. High-Frequency Radar (100 Hz updates):")
max_tracks_100hz = int(10 / avg_time)
print(f"   Python: {max_tracks_100hz} tracks/update")
print(f"   C++ (projected): {int(max_tracks_100hz * 3.5)} tracks/update")
print()

print(f"2. Standard Radar (50 Hz updates):")
max_tracks_50hz = int(20 / avg_time)
print(f"   Python: {max_tracks_50hz} tracks/update")
print(f"   C++ (projected): {int(max_tracks_50hz * 3.5)} tracks/update")
print()

print(f"3. Low-Frequency Radar (10 Hz updates):")
max_tracks_10hz = int(100 / avg_time)
print(f"   Python: {max_tracks_10hz} tracks/update")
print(f"   C++ (projected): {int(max_tracks_10hz * 3.5)} tracks/update")
print()

print("=" * 80)
print(" OPTIMIZATION RECOMMENDATIONS")
print("=" * 80)
print()

print("For Production C++ Deployment:")
print()
print("1. Build Optimization:")
print("   - Use -O3 compiler flags")
print("   - Enable CPU-specific optimizations (AVX, SSE)")
print("   - Use TensorFlow Lite with XNNPACK delegate")
print()

print("2. Runtime Optimization:")
print("   - Use 4-8 threads for inference")
print("   - Batch multiple tracks when possible")
print("   - Implement double-buffering for continuous operation")
print()

print("3. Model Optimization:")
print("   - Consider quantization (INT8) for 2-4x speedup")
print("   - Prune unnecessary layers if accuracy allows")
print("   - Use distillation for smaller model")
print()

print("4. System Integration:")
print("   - Pin threads to specific CPU cores")
print("   - Use real-time process priority")
print("   - Implement hardware acceleration if available")
print()

# Save comprehensive report
report = {
    'python_tflite_performance': {
        'avg_inference_ms': float(avg_time),
        'median_inference_ms': float(med_time),
        'min_inference_ms': float(min_time),
        'max_inference_ms': float(max_time),
        'std_dev_ms': float(std_time),
        'throughput_per_sec': float(1000/avg_time),
        'latency_percentiles': {
            'p50': float(p50),
            'p90': float(p90),
            'p95': float(p95),
            'p99': float(p99)
        }
    },
    'cpp_projected_performance': {
        'conservative': {
            'inference_ms': float(avg_time / 2.0),
            'throughput_per_sec': float(2000/avg_time)
        },
        'typical': {
            'inference_ms': float(avg_time / 3.5),
            'throughput_per_sec': float(3500/avg_time)
        },
        'optimistic': {
            'inference_ms': float(avg_time / 5.0),
            'throughput_per_sec': float(5000/avg_time)
        }
    },
    'model_info': {
        'size_bytes': model_size_bytes,
        'size_kb': float(model_size_kb),
        'sequence_length': seq_length,
        'n_features': n_features,
        'n_classes': metadata['n_classes'],
        'classes': metadata['classes']
    },
    'real_time_capability': {
        '100hz_radar': {
            'python_tracks_per_update': max_tracks_100hz,
            'cpp_tracks_per_update': int(max_tracks_100hz * 3.5)
        },
        '50hz_radar': {
            'python_tracks_per_update': max_tracks_50hz,
            'cpp_tracks_per_update': int(max_tracks_50hz * 3.5)
        },
        '10hz_radar': {
            'python_tracks_per_update': max_tracks_10hz,
            'cpp_tracks_per_update': int(max_tracks_10hz * 3.5)
        }
    },
    'batch_performance': batch_results
}

report_path = "cpp_models/comprehensive_performance_report.json"
with open(report_path, 'w') as f:
    json.dump(report, f, indent=2)

print(f"Comprehensive report saved to: {report_path}")
print()

print("=" * 80)
print(" BENCHMARK COMPLETE")
print("=" * 80)
