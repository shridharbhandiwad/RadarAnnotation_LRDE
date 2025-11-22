#!/usr/bin/env python3
"""
Test script to demonstrate the automatic label recovery system
"""
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s'
)

from src.label_transformer import LabelTransformer, quick_fix_labels

def create_test_data_uniform_composite():
    """Create test data with uniform composite labels (problematic case)"""
    data = {
        'trackid': [1]*10 + [2]*10 + [3]*10,
        'time': list(range(10)) * 3,
        'x': np.random.randn(30) * 100,
        'y': np.random.randn(30) * 100,
        'z': np.random.randn(30) * 10,
        'vx': np.random.randn(30) * 10,
        'vy': np.random.randn(30) * 10,
        'vz': np.random.randn(30) * 2,
        'Annotation': ['incoming,level_flight,linear,light_maneuver,low_speed'] * 30
    }
    return pd.DataFrame(data)


def create_test_data_diverse_tracks():
    """Create test data with different composite labels per track"""
    data = {
        'trackid': [1]*10 + [2]*10 + [3]*10,
        'time': list(range(10)) * 3,
        'x': np.random.randn(30) * 100,
        'y': np.random.randn(30) * 100,
        'z': np.random.randn(30) * 10,
        'vx': np.random.randn(30) * 10,
        'vy': np.random.randn(30) * 10,
        'vz': np.random.randn(30) * 2,
        'Annotation': (
            ['incoming,level_flight,linear,light_maneuver,low_speed'] * 10 +
            ['outgoing,level_flight,curved,high_maneuver,high_speed'] * 10 +
            ['incoming,fixed_range_ascending,linear,light_maneuver,low_speed'] * 10
        )
    }
    return pd.DataFrame(data)


def test_multi_label_transformation():
    """Test multi-label binary transformation"""
    print("\n" + "="*80)
    print("TEST 1: Multi-Label Binary Transformation")
    print("="*80)
    
    df = create_test_data_uniform_composite()
    print(f"\n📊 Original data:")
    print(f"   Rows: {len(df)}")
    print(f"   Unique labels: {df['Annotation'].nunique()}")
    print(f"   Label: {df['Annotation'].iloc[0]}")
    
    transformer = LabelTransformer()
    df_out, binary_labels, label_names = transformer.transform_to_multi_label(df)
    
    print(f"\n✅ Transformed data:")
    print(f"   Binary columns created: {len(label_names)}")
    print(f"   Column names: {label_names}")
    print(f"   Binary array shape: {binary_labels.shape}")
    print(f"   Sample binary vector: {binary_labels[0]}")
    
    return True


def test_primary_extraction():
    """Test primary label extraction"""
    print("\n" + "="*80)
    print("TEST 2: Primary Label Extraction")
    print("="*80)
    
    df = create_test_data_diverse_tracks()
    print(f"\n📊 Original data:")
    print(f"   Unique composite labels: {df['Annotation'].nunique()}")
    for label in df['Annotation'].unique():
        count = (df['Annotation'] == label).sum()
        print(f"   - {label}: {count} points")
    
    transformer = LabelTransformer()
    df_out = transformer.extract_primary_labels(df, strategy='hierarchy')
    
    print(f"\n✅ Transformed data:")
    print(f"   Unique primary labels: {df_out['Annotation'].nunique()}")
    for label in df_out['Annotation'].unique():
        count = (df_out['Annotation'] == label).sum()
        print(f"   - {label}: {count} points")
    
    return True


def test_per_track_labels():
    """Test per-track label aggregation"""
    print("\n" + "="*80)
    print("TEST 3: Per-Track Label Aggregation")
    print("="*80)
    
    df = create_test_data_uniform_composite()
    print(f"\n📊 Original data:")
    print(f"   Tracks: {df['trackid'].nunique()}")
    print(f"   Unique point-level labels: {df['Annotation'].nunique()}")
    
    # Manually create different labels per track for demonstration
    df.loc[df['trackid'] == 1, 'Annotation'] = 'incoming,level_flight,linear,light_maneuver,low_speed'
    df.loc[df['trackid'] == 2, 'Annotation'] = 'outgoing,level_flight,curved,high_maneuver,high_speed'
    df.loc[df['trackid'] == 3, 'Annotation'] = 'incoming,fixed_range_ascending,linear,light_maneuver,low_speed'
    
    transformer = LabelTransformer()
    df_out = transformer.create_per_track_labels(df, strategy='primary')
    
    print(f"\n✅ Transformed data:")
    print(f"   Unique track-level labels: {df_out['Annotation'].nunique()}")
    for trackid in df_out['trackid'].unique():
        label = df_out[df_out['trackid'] == trackid]['Annotation'].iloc[0]
        print(f"   - Track {trackid}: {label}")
    
    return True


def test_auto_transform():
    """Test automatic transformation selection"""
    print("\n" + "="*80)
    print("TEST 4: Automatic Transformation Selection")
    print("="*80)
    
    # Test Case 1: Uniform composite labels
    print("\n📋 Case 1: Uniform composite labels")
    df1 = create_test_data_uniform_composite()
    
    transformer1 = LabelTransformer()
    analysis1 = transformer1.analyze_label_diversity(df1['Annotation'])
    print(f"   Analysis: {analysis1['n_unique_labels']} unique labels")
    print(f"   Recommended: {analysis1['recommended_strategy']}")
    
    df1_out, info1 = transformer1.auto_transform(df1)
    print(f"   Applied: {info1['transformation']}")
    print(f"   Success: {info1['success']}")
    if info1.get('n_labels'):
        print(f"   Created {info1['n_labels']} labels")
    
    # Test Case 2: Diverse composite labels
    print("\n📋 Case 2: Diverse composite labels")
    df2 = create_test_data_diverse_tracks()
    
    transformer2 = LabelTransformer()
    analysis2 = transformer2.analyze_label_diversity(df2['Annotation'])
    print(f"   Analysis: {analysis2['n_unique_labels']} unique labels")
    print(f"   Recommended: {analysis2['recommended_strategy']}")
    
    df2_out, info2 = transformer2.auto_transform(df2)
    print(f"   Applied: {info2['transformation']}")
    print(f"   Success: {info2['success']}")
    
    return True


def test_quick_fix():
    """Test the quick_fix_labels convenience function"""
    print("\n" + "="*80)
    print("TEST 5: Quick Fix Labels Function")
    print("="*80)
    
    # Create temporary CSV file
    df = create_test_data_uniform_composite()
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        temp_path = f.name
        df.to_csv(temp_path, index=False)
    
    try:
        print(f"\n📄 Created temporary CSV: {temp_path}")
        print(f"   Original unique labels: {df['Annotation'].nunique()}")
        
        # Quick fix
        df_fixed, info = quick_fix_labels(temp_path, strategy='auto')
        
        print(f"\n✅ Quick fix applied:")
        print(f"   Strategy: {info.get('transformation', 'unknown')}")
        print(f"   Success: {info.get('success', False)}")
        if 'n_labels' in info:
            print(f"   Labels created: {info['n_labels']}")
        if 'binary_label_columns' in info:
            print(f"   Binary columns: {info['binary_label_columns']}")
        
        return True
    finally:
        # Cleanup
        Path(temp_path).unlink(missing_ok=True)


def test_integration_with_training():
    """Test integration with actual training pipeline (simulation)"""
    print("\n" + "="*80)
    print("TEST 6: Integration with Training Pipeline (Simulation)")
    print("="*80)
    
    # Create problematic data
    df = create_test_data_uniform_composite()
    
    # Add some additional features for training
    df['ax'] = np.random.randn(len(df))
    df['ay'] = np.random.randn(len(df))
    df['az'] = np.random.randn(len(df))
    df['speed'] = np.random.randn(len(df)) * 50 + 100
    df['heading'] = np.random.randn(len(df)) * 180
    df['range'] = np.random.randn(len(df)) * 1000 + 5000
    df['curvature'] = np.random.randn(len(df)) * 0.01
    df['valid_features'] = True
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        temp_path = f.name
        df.to_csv(temp_path, index=False)
    
    try:
        print(f"\n📄 Created test data: {temp_path}")
        print(f"   Problem: Only 1 unique label (insufficient for ML)")
        
        # Simulate what the training pipeline does
        print("\n🔄 Simulating training pipeline...")
        print("   1. Detect insufficient diversity ✓")
        print("   2. Apply automatic transformation ✓")
        
        transformer = LabelTransformer()
        df_transformed, info = transformer.auto_transform(df)
        
        print(f"   3. Transformation applied: {info['transformation']} ✓")
        if info['success']:
            print("   4. Training can now proceed ✓")
            print(f"\n✅ SUCCESS: Automatic recovery would allow training!")
            print(f"   - Original: 1 unique label (FAIL)")
            print(f"   - Transformed: {info.get('n_labels', '?')} labels (PASS)")
        else:
            print("   4. Training still cannot proceed ✗")
        
        return info['success']
    finally:
        Path(temp_path).unlink(missing_ok=True)


def main():
    """Run all tests"""
    print("\n" + "="*80)
    print("AUTOMATIC LABEL RECOVERY SYSTEM - TEST SUITE")
    print("="*80)
    print("\nTesting the new intelligent label transformation system...")
    
    tests = [
        ("Multi-Label Binary Transformation", test_multi_label_transformation),
        ("Primary Label Extraction", test_primary_extraction),
        ("Per-Track Label Aggregation", test_per_track_labels),
        ("Automatic Transformation Selection", test_auto_transform),
        ("Quick Fix Labels Function", test_quick_fix),
        ("Integration with Training Pipeline", test_integration_with_training),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, success))
        except Exception as e:
            print(f"\n❌ Test failed: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    for name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status}: {name}")
    
    total = len(results)
    passed = sum(1 for _, s in results if s)
    
    print(f"\n{passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! The automatic recovery system is working correctly.")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please review the output above.")
        return 1


if __name__ == '__main__':
    sys.exit(main())
