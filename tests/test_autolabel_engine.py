"""Tests for AutoLabeling Engine"""
import unittest
import numpy as np
import pandas as pd

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src import autolabel_engine


class TestAutoLabelEngine(unittest.TestCase):
    """Test AutoLabeling Engine"""
    
    def setUp(self):
        """Setup test fixtures"""
        # Create test trajectory data
        t = np.arange(0, 10, 0.1)
        self.df = pd.DataFrame({
            'trackid': np.ones(len(t), dtype=int),
            'time': t,
            'x': 100 * t,  # Constant velocity in x
            'y': 200 * t,  # Constant velocity in y
            'z': np.full(len(t), 1000.0),  # Constant altitude
            'vx': np.full(len(t), 100.0),
            'vy': np.full(len(t), 200.0),
            'vz': np.zeros(len(t)),
            'ax': np.zeros(len(t)),
            'ay': np.zeros(len(t))
        })
    
    def test_compute_motion_features(self):
        """Test motion feature computation"""
        df_features = autolabel_engine.compute_motion_features(self.df)
        
        # Check that features are computed
        self.assertIn('speed', df_features.columns)
        self.assertIn('heading', df_features.columns)
        self.assertIn('range', df_features.columns)
        self.assertIn('curvature', df_features.columns)
        
        # Check speed is approximately correct
        expected_speed = np.sqrt(100**2 + 200**2)
        valid_mask = df_features['valid_features']
        computed_speed = df_features.loc[valid_mask, 'speed'].mean()
        self.assertAlmostEqual(computed_speed, expected_speed, delta=10)
    
    def test_apply_rules_and_flags(self):
        """Test rule-based flag application"""
        df_features = autolabel_engine.compute_motion_features(self.df)
        df_labeled = autolabel_engine.apply_rules_and_flags(df_features)
        
        # Check that flag columns exist
        self.assertIn('level_flight', df_labeled.columns)
        self.assertIn('linear', df_labeled.columns)
        self.assertIn('Annotation', df_labeled.columns)
        
        # Level flight should be true (constant altitude)
        valid_mask = df_labeled['valid_features']
        self.assertTrue(df_labeled.loc[valid_mask, 'level_flight'].any())
        
        # Linear motion should be true (no curvature)
        self.assertTrue(df_labeled.loc[valid_mask, 'linear'].any())
    
    def test_annotation_summary(self):
        """Test annotation summary generation"""
        df_features = autolabel_engine.compute_motion_features(self.df)
        df_labeled = autolabel_engine.apply_rules_and_flags(df_features)
        
        summary = autolabel_engine.get_annotation_summary(df_labeled)
        
        self.assertIn('total_records', summary)
        self.assertIn('valid_records', summary)
        self.assertIn('annotation_distribution', summary)
        
        self.assertEqual(summary['total_records'], len(self.df))
        self.assertGreater(summary['valid_records'], 0)
    
    def test_curved_trajectory(self):
        """Test detection of curved trajectory"""
        # Create circular trajectory
        t = np.arange(0, 10, 0.1)
        radius = 1000.0
        omega = 0.1  # Angular velocity
        
        df_circle = pd.DataFrame({
            'trackid': np.ones(len(t), dtype=int),
            'time': t,
            'x': radius * np.cos(omega * t),
            'y': radius * np.sin(omega * t),
            'z': np.full(len(t), 1000.0),
            'vx': -radius * omega * np.sin(omega * t),
            'vy': radius * omega * np.cos(omega * t),
            'vz': np.zeros(len(t)),
            'ax': -radius * omega**2 * np.cos(omega * t),
            'ay': -radius * omega**2 * np.sin(omega * t)
        })
        
        df_features = autolabel_engine.compute_motion_features(df_circle)
        df_labeled = autolabel_engine.apply_rules_and_flags(df_features)
        
        # Should detect curved motion
        valid_mask = df_labeled['valid_features']
        self.assertTrue(df_labeled.loc[valid_mask, 'curved'].any())


if __name__ == '__main__':
    unittest.main()
