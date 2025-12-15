#!/usr/bin/env python3
"""
Create a simple feedforward model and measure actual Python TFLite performance
This gives us baseline metrics that can be extrapolated to C++ performance
"""

import os
import sys
import json
import time
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from pathlib import Path
from statistics import mean, stdev, median

print("=" * 80)
print(" ACTUAL PERFORMANCE BENCHMARK")
print(" Simple Feedforward Model - Real Measurements")
print("=" * 80)
print()

# Model parameters
sequence_length = 20
n_features = 18
n_classes = 2
input_size = sequence_length * n_features  # Flatten for feedforward

print("Creating simple feedforward model...")
print(f"Input: {input_size} features (flattened {sequence_length}×{n_features})")
print(f"Output: {n_classes} classes")
print()

# Create simple feedforward model
model = keras.Sequential([
    layers.Input(shape=(input_size,)),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(32, activation='relu'),
    layers.Dense(n_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
print(model.summary())

# Generate synthetic training data
print("Generating synthetic training data...")
n_samples = 1000
X_train = np.random.randn(n_samples, input_size).astype(np.float32)
y_train = np.random.randint(0, n_classes, n_samples)

# Quick training
print("Training model (5 epochs)...")
model.fit(X_train, y_train, epochs=5, batch_size=32, verbose=0)
print("✓ Training complete")
print()

# Convert to TFLite
print("Converting to TensorFlow Lite...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

model_path = "simple_model.tflite"
with open(model_path, 'wb') as f:
    f.write(tflite_model)

model_size_kb = len(tflite_model) / 1024
print(f"✓ TFLite model created: {model_size_kb:.2f} KB")
print()

# Load TFLite model for inference
print("Loading TFLite model for benchmarking...")
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print(f"Input shape: {input_details[0]['shape']}")
print(f"Output shape: {output_details[0]['shape']}")
print()

# Generate test data
n_test = 200
X_test = np.random.randn(n_test, input_size).astype(np.float32)

print("=" * 80)
print(" BENCHMARK 1: Single Inference Performance")
print("=" * 80)
print()

# Warm-up
print("Warming up (100 inferences)...")
for i in range(100):
    interpreter.set_tensor(input_details[0]['index'], X_test[i:i+1])
    interpreter.invoke()

print("✓ Warm-up complete")
print()

# Actual benchmark
print("Running 1000 timed inferences...")
times_ms = []
for i in range(1000):
    start = time.perf_counter()
    interpreter.set_tensor(input_details[0]['index'], X_test[i % n_test:i % n_test + 1])
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    end = time.perf_counter()
    times_ms.append((end - start) * 1000)

print("✓ Benchmark complete")
print()

# Statistics
avg_ms = mean(times_ms)
med_ms = median(times_ms)
min_ms = min(times_ms)
max_ms = max(times_ms)
std_ms = stdev(times_ms)
throughput = 1000 / avg_ms

# Percentiles
p50 = np.percentile(times_ms, 50)
p90 = np.percentile(times_ms, 90)
p95 = np.percentile(times_ms, 95)
p99 = np.percentile(times_ms, 99)

print("PYTHON TFLITE PERFORMANCE:")
print(f"  Average:    {avg_ms:.4f} ms ({avg_ms * 1000:.1f} µs)")
print(f"  Median:     {med_ms:.4f} ms ({med_ms * 1000:.1f} µs)")
print(f"  Min:        {min_ms:.4f} ms ({min_ms * 1000:.1f} µs)")
print(f"  Max:        {max_ms:.4f} ms ({max_ms * 1000:.1f} µs)")
print(f"  Std Dev:    {std_ms:.4f} ms")
print(f"  Throughput: {throughput:.1f} inferences/sec")
print()

print("LATENCY PERCENTILES:")
print(f"  P50:  {p50:.4f} ms")
print(f"  P90:  {p90:.4f} ms")
print(f"  P95:  {p95:.4f} ms")
print(f"  P99:  {p99:.4f} ms")
print()

print("=" * 80)
print(" C++ PERFORMANCE PROJECTIONS")
print("=" * 80)
print()

# C++ projections
cpp_factors = {
    'Conservative (2x)': 2.0,
    'Typical (3.5x)': 3.5,
    'Optimistic (5x)': 5.0,
    'With Quantization (8x)': 8.0
}

print("Based on Python TFLite measurements, projected C++ performance:")
print()

results = {}
for scenario, factor in cpp_factors.items():
    cpp_ms = avg_ms / factor
    cpp_us = cpp_ms * 1000
    cpp_throughput = throughput * factor
    
    print(f"{scenario}:")
    print(f"  Inference time: {cpp_ms:.4f} ms ({cpp_us:.1f} µs)")
    print(f"  Throughput:     {cpp_throughput:.1f} inferences/sec")
    
    # Real-time capability at 50 Hz
    update_period_ms = 20  # 50 Hz
    tracks_per_update = int(update_period_ms / cpp_ms)
    print(f"  Tracks @ 50Hz:  {tracks_per_update} trajectories/update")
    print()
    
    results[scenario] = {
        'inference_ms': cpp_ms,
        'inference_us': cpp_us,
        'throughput_per_sec': cpp_throughput,
        'tracks_at_50hz': tracks_per_update
    }

print("=" * 80)
print(" REAL-TIME PROCESSING CAPABILITY")
print("=" * 80)
print()

# Use typical C++ performance
cpp_typical_ms = avg_ms / 3.5

radar_configs = [
    ('High-Frequency Radar', 100, 10),   # 100 Hz, 10ms period
    ('Standard Radar', 50, 20),          # 50 Hz, 20ms period
    ('Surveillance Radar', 20, 50),      # 20 Hz, 50ms period
    ('Early Warning Radar', 10, 100),    # 10 Hz, 100ms period
]

print("C++ Performance (typical 3.5x speedup):")
print()

for name, freq_hz, period_ms in radar_configs:
    tracks = int(period_ms / cpp_typical_ms)
    total_throughput = tracks * freq_hz
    
    print(f"{name} ({freq_hz} Hz):")
    print(f"  Update period:     {period_ms} ms")
    print(f"  Tracks/update:     {tracks}")
    print(f"  Total throughput:  {total_throughput} classifications/sec")
    print()

print("=" * 80)
print(" MEMORY ANALYSIS")
print("=" * 80)
print()

input_bytes = input_size * 4  # float32
output_bytes = n_classes * 4
model_bytes = len(tflite_model)

print(f"Model Memory Footprint:")
print(f"  Model size (disk):     {model_bytes:,} bytes ({model_bytes/1024:.1f} KB)")
print(f"  Input tensor:          {input_bytes} bytes")
print(f"  Output tensor:         {output_bytes} bytes")
print(f"  Runtime overhead:      ~500 KB (TFLite runtime)")
print(f"  Total (estimated):     ~{(model_bytes + input_bytes + output_bytes + 500*1024)/1024:.0f} KB")
print()

print("=" * 80)
print(" DEPLOYMENT SCENARIOS")
print("=" * 80)
print()

cpp_typical_ms = avg_ms / 3.5

scenarios = [
    ("🚀 High-Performance Server", 0.5, "8 threads, XNNPACK, AVX-512"),
    ("💻 Standard Workstation", cpp_typical_ms, "4 threads, optimized"),
    ("📱 Embedded ARM", cpp_typical_ms * 3, "2 threads, NEON"),
    ("🔌 Microcontroller", cpp_typical_ms * 10, "Single thread, limited resources")
]

for name, inference_ms, config in scenarios:
    throughput_hz = 1000 / inference_ms
    tracks_50hz = int(20 / inference_ms)
    
    print(f"{name}")
    print(f"  Config:            {config}")
    print(f"  Inference time:    {inference_ms:.2f} ms")
    print(f"  Throughput:        {throughput_hz:.0f} inf/sec")
    print(f"  Tracks @ 50Hz:     {tracks_50hz}")
    print()

print("=" * 80)
print(" SUMMARY & RECOMMENDATIONS")
print("=" * 80)
print()

if avg_ms < 1.0:
    rating = "✅ EXCELLENT"
    desc = "Sub-millisecond Python performance indicates excellent C++ potential"
elif avg_ms < 5.0:
    rating = "✅ VERY GOOD"
    desc = "Low Python latency will translate to sub-millisecond C++ performance"
elif avg_ms < 10.0:
    rating = "✓ GOOD"
    desc = "Good baseline, C++ will provide real-time capable performance"
else:
    rating = "⚠ ACCEPTABLE"
    desc = "Consider optimization, but C++ will still provide significant speedup"

print(f"Performance Rating: {rating}")
print(f"  {desc}")
print()

print("Key Findings:")
print(f"  • Python TFLite:  {avg_ms:.3f} ms average inference time")
print(f"  • C++ Projected:  {avg_ms/3.5:.3f} ms (typical 3.5x speedup)")
print(f"  • Model Size:     {model_size_kb:.1f} KB (very efficient)")
print(f"  • Real-time Cap:  {int(20/(avg_ms/3.5))} tracks @ 50 Hz")
print()

print("Recommendations:")
print("  1. Deploy with 4 threads for optimal throughput")
print("  2. Use -O3 compiler optimizations")
print("  3. Consider INT8 quantization for embedded systems")
print("  4. Enable XNNPACK delegate for 2-3x additional speedup")
print("  5. Monitor P99 latency in production for SLA compliance")
print()

# Save results
report = {
    'python_performance': {
        'avg_ms': float(avg_ms),
        'median_ms': float(med_ms),
        'min_ms': float(min_ms),
        'max_ms': float(max_ms),
        'std_ms': float(std_ms),
        'throughput_per_sec': float(throughput),
        'percentiles': {
            'p50': float(p50),
            'p90': float(p90),
            'p95': float(p95),
            'p99': float(p99)
        }
    },
    'cpp_projections': results,
    'model_info': {
        'size_kb': float(model_size_kb),
        'input_size': input_size,
        'n_classes': n_classes,
        'architecture': 'Feedforward Neural Network'
    }
}

with open('actual_performance_metrics.json', 'w') as f:
    json.dump(report, f, indent=2)

print(f"Full metrics saved to: actual_performance_metrics.json")
print()

print("=" * 80)
print(" BENCHMARK COMPLETE")
print("=" * 80)
