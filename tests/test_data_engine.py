"""Tests for Data Extraction Engine"""
import unittest
import numpy as np
import pandas as pd
import tempfile
import os
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src import data_engine, utils
from src.config import get_config


class TestDataEngine(unittest.TestCase):
    """Test Data Extraction Engine"""
    
    def setUp(self):
        """Setup test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.config = get_config()
        self.schema = self.config.get('binary_schema')
    
    def test_binary_parsing(self):
        """Test binary file parsing"""
        # Create test records
        records = [
            {
                'time': 0.0,
                'trackid': 1.0,
                'x': 1000.0,
                'y': 2000.0,
                'z': 3000.0,
                'vx': 10.0,
                'vy': 20.0,
                'vz': 5.0,
                'ax': 0.1,
                'ay': 0.2
            },
            {
                'time': 0.1,
                'trackid': 1.0,
                'x': 1001.0,
                'y': 2002.0,
                'z': 3000.5,
                'vx': 10.0,
                'vy': 20.0,
                'vz': 5.0,
                'ax': 0.1,
                'ay': 0.2
            }
        ]
        
        # Write binary file
        binary_path = os.path.join(self.temp_dir, 'test.bin')
        utils.write_binary_file(binary_path, records, self.schema)
        
        # Read back
        df = data_engine.extract_binary_to_dataframe(binary_path, self.schema)
        
        # Verify
        self.assertEqual(len(df), 2)
        self.assertIn('trackid', df.columns)
        self.assertIn('x', df.columns)
        self.assertAlmostEqual(df.iloc[0]['x'], 1000.0, places=2)
    
    def test_save_load_csv(self):
        """Test CSV save and load"""
        df = pd.DataFrame({
            'trackid': [1, 1, 2],
            'time': [0.0, 0.1, 0.0],
            'x': [1000.0, 1010.0, 2000.0],
            'y': [2000.0, 2020.0, 3000.0],
            'z': [3000.0, 3000.0, 4000.0],
            'vx': [10.0, 10.0, 15.0],
            'vy': [20.0, 20.0, 25.0],
            'vz': [0.0, 0.0, 5.0],
            'ax': [0.0, 0.0, 0.0],
            'ay': [0.0, 0.0, 0.0]
        })
        
        csv_path = os.path.join(self.temp_dir, 'test.csv')
        data_engine.save_dataframe(df, csv_path, 'csv')
        
        df_loaded = data_engine.load_dataframe(csv_path)
        
        self.assertEqual(len(df_loaded), len(df))
        pd.testing.assert_frame_equal(df, df_loaded)
    
    def test_data_summary(self):
        """Test data summary generation"""
        df = pd.DataFrame({
            'trackid': [1, 1, 2],
            'time': [0.0, 1.0, 0.0],
            'x': [1000.0, 1010.0, 2000.0],
            'y': [2000.0, 2020.0, 3000.0],
            'z': [3000.0, 3100.0, 4000.0]
        })
        
        summary = data_engine.get_data_summary(df)
        
        self.assertEqual(summary['total_records'], 3)
        self.assertEqual(summary['num_tracks'], 2)
        self.assertEqual(summary['duration_seconds'], 1.0)


class TestUtils(unittest.TestCase):
    """Test utility functions"""
    
    def test_coordinate_conversion(self):
        """Test coordinate conversions"""
        x = np.array([100.0, 0.0, -100.0])
        y = np.array([0.0, 100.0, 0.0])
        
        r, theta = utils.cartesian_to_polar(x, y)
        
        self.assertAlmostEqual(r[0], 100.0, places=2)
        self.assertAlmostEqual(theta[0], 0.0, places=2)
        self.assertAlmostEqual(theta[1], 90.0, places=2)
    
    def test_speed_computation(self):
        """Test speed computation"""
        vx = np.array([3.0, 0.0])
        vy = np.array([4.0, 0.0])
        vz = np.array([0.0, 5.0])
        
        speed = utils.compute_speed(vx, vy, vz)
        
        self.assertAlmostEqual(speed[0], 5.0, places=2)
        self.assertAlmostEqual(speed[1], 5.0, places=2)
    
    def test_heading_computation(self):
        """Test heading computation"""
        vx = np.array([1.0, 0.0, -1.0])
        vy = np.array([0.0, 1.0, 0.0])
        
        heading = utils.compute_heading(vx, vy)
        
        # Verify heading is in 0-360 range
        self.assertTrue(np.all(heading >= 0))
        self.assertTrue(np.all(heading < 360))


if __name__ == '__main__':
    unittest.main()
