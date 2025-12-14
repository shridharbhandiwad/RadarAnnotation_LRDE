#!/usr/bin/env python3
"""
Test script for chronology markers in visualization.

This script demonstrates the new start/end point markers that show
track chronology in the PPI and time series plots.
"""

import pandas as pd
import numpy as np
from pathlib import Path


def create_test_data():
    """Create simple test data with multiple tracks"""
    
    # Track 1: Moving North-East
    track1_times = np.linspace(0, 10, 20)
    track1_x = np.linspace(0, 5000, 20)  # Moving East
    track1_y = np.linspace(0, 5000, 20)  # Moving North
    track1_z = np.linspace(1000, 2000, 20)  # Climbing
    
    track1 = pd.DataFrame({
        'trackid': [1] * 20,
        'time': track1_times,
        'x': track1_x,
        'y': track1_y,
        'z': track1_z,
        'speed': np.random.uniform(100, 150, 20),
        'curvature': np.random.uniform(0.001, 0.01, 20)
    })
    
    # Track 2: Moving South-West  
    track2_times = np.linspace(5, 15, 20)
    track2_x = np.linspace(8000, 3000, 20)  # Moving West
    track2_y = np.linspace(8000, 3000, 20)  # Moving South
    track2_z = np.linspace(2000, 1500, 20)  # Descending
    
    track2 = pd.DataFrame({
        'trackid': [2] * 20,
        'time': track2_times,
        'x': track2_x,
        'y': track2_y,
        'z': track2_z,
        'speed': np.random.uniform(80, 120, 20),
        'curvature': np.random.uniform(0.005, 0.015, 20)
    })
    
    # Track 3: Circular motion
    track3_times = np.linspace(2, 12, 30)
    angles = np.linspace(0, 2*np.pi, 30)
    radius = 4000
    track3_x = 6000 + radius * np.cos(angles)
    track3_y = 6000 + radius * np.sin(angles)
    track3_z = np.linspace(1500, 1500, 30)  # Level flight
    
    track3 = pd.DataFrame({
        'trackid': [3] * 30,
        'time': track3_times,
        'x': track3_x,
        'y': track3_y,
        'z': track3_z,
        'speed': np.random.uniform(90, 110, 30),
        'curvature': np.random.uniform(0.01, 0.02, 30)
    })
    
    # Combine all tracks
    df = pd.concat([track1, track2, track3], ignore_index=True)
    
    return df


def main():
    """Main test function"""
    
    print("="*70)
    print("CHRONOLOGY MARKERS TEST")
    print("="*70)
    print()
    
    # Create test data
    print("Creating test data with 3 tracks...")
    df = create_test_data()
    print(f"✓ Created {len(df)} data points across {df['trackid'].nunique()} tracks")
    print()
    
    # Save to CSV
    output_path = Path('data/chronology_test.csv')
    output_path.parent.mkdir(exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"✓ Saved test data to: {output_path}")
    print()
    
    # Print track summary
    print("Track Summary:")
    print("-" * 70)
    for trackid in sorted(df['trackid'].unique()):
        track = df[df['trackid'] == trackid].sort_values('time')
        start_time = track['time'].iloc[0]
        end_time = track['time'].iloc[-1]
        start_pos = (track['x'].iloc[0]/1000, track['y'].iloc[0]/1000)
        end_pos = (track['x'].iloc[-1]/1000, track['y'].iloc[-1]/1000)
        
        print(f"Track {int(trackid)}:")
        print(f"  Time: {start_time:.2f}s → {end_time:.2f}s")
        print(f"  Start: ({start_pos[0]:.2f}, {start_pos[1]:.2f}) km  [GREEN STAR ★]")
        print(f"  End:   ({end_pos[0]:.2f}, {end_pos[1]:.2f}) km  [RED TRIANGLE ▲]")
        print()
    
    print("="*70)
    print("INSTRUCTIONS TO VIEW CHRONOLOGY MARKERS:")
    print("="*70)
    print()
    print("1. Run the GUI application:")
    print("   python3 -m src.gui")
    print()
    print("2. Navigate to the 'Visualization' tab")
    print()
    print("3. Click 'Load Data' and select:")
    print(f"   {output_path.absolute()}")
    print()
    print("4. Observe the markers:")
    print("   ★ GREEN STARS     = Start points (earliest time)")
    print("   ▲ RED TRIANGLES   = End points (latest time)")
    print()
    print("5. Try different views:")
    print("   - Display: Radar View, Cartesian, or Polar")
    print("   - Color: Track ID, Annotation, or Track Segments")
    print("   - Track filter: All or individual tracks")
    print()
    print("6. Enable Time Series to see markers on time plots")
    print()
    print("="*70)
    print()
    print("✓ Test data ready! Launch the GUI to see chronology markers.")
    print()


if __name__ == '__main__':
    main()
