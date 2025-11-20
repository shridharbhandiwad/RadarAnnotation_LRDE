"""Simulation Engine - Generate sample radar data"""
import numpy as np
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple
from .utils import write_binary_file, ensure_dir
from .config import get_config

logger = logging.getLogger(__name__)


class TrajectoryGenerator:
    """Generate various trajectory types for simulation"""
    
    def __init__(self, sample_rate_ms: int = 100, duration_min: float = 5, radar_origin: List[float] = None):
        """Initialize trajectory generator
        
        Args:
            sample_rate_ms: Sample rate in milliseconds
            duration_min: Flight duration in minutes
            radar_origin: Radar origin [x, y, z]
        """
        self.sample_rate_s = sample_rate_ms / 1000.0
        self.duration_s = duration_min * 60.0
        self.num_samples = int(self.duration_s / self.sample_rate_s)
        self.radar_origin = radar_origin or [0, 0, 0]
        
    def straight_constant_velocity(self, speed: float, altitude: float, start_pos: Tuple[float, float]) -> np.ndarray:
        """Generate straight constant velocity trajectory"""
        t = np.arange(0, self.num_samples) * self.sample_rate_s
        
        # Direction: moving towards radar initially, then away
        angle = np.random.uniform(0, 2 * np.pi)
        vx = speed * np.cos(angle)
        vy = speed * np.sin(angle)
        
        x = start_pos[0] + vx * t
        y = start_pos[1] + vy * t
        z = np.full_like(t, altitude)
        
        vx_arr = np.full_like(t, vx)
        vy_arr = np.full_like(t, vy)
        vz_arr = np.zeros_like(t)
        
        ax_arr = np.zeros_like(t)
        ay_arr = np.zeros_like(t)
        az_arr = np.zeros_like(t)
        
        return np.column_stack([t, x, y, z, vx_arr, vy_arr, vz_arr, ax_arr, ay_arr, az_arr])
    
    def ascending_spiral(self, radius: float, angular_vel: float, climb_rate: float, center: Tuple[float, float], start_alt: float) -> np.ndarray:
        """Generate ascending spiral trajectory"""
        t = np.arange(0, self.num_samples) * self.sample_rate_s
        
        theta = angular_vel * t
        x = center[0] + radius * np.cos(theta)
        y = center[1] + radius * np.sin(theta)
        z = start_alt + climb_rate * t
        
        vx = -radius * angular_vel * np.sin(theta)
        vy = radius * angular_vel * np.cos(theta)
        vz = np.full_like(t, climb_rate)
        
        ax = -radius * angular_vel**2 * np.cos(theta)
        ay = -radius * angular_vel**2 * np.sin(theta)
        az = np.zeros_like(t)
        
        return np.column_stack([t, x, y, z, vx, vy, vz, ax, ay, az])
    
    def descending_path(self, speed: float, descent_angle: float, start_pos: Tuple[float, float, float]) -> np.ndarray:
        """Generate descending trajectory"""
        t = np.arange(0, self.num_samples) * self.sample_rate_s
        
        angle = np.random.uniform(0, 2 * np.pi)
        horizontal_speed = speed * np.cos(descent_angle)
        vx = horizontal_speed * np.cos(angle)
        vy = horizontal_speed * np.sin(angle)
        vz = -speed * np.sin(descent_angle)
        
        x = start_pos[0] + vx * t
        y = start_pos[1] + vy * t
        z = start_pos[2] + vz * t
        
        # Prevent going below ground
        z = np.maximum(z, 100)
        
        vx_arr = np.full_like(t, vx)
        vy_arr = np.full_like(t, vy)
        vz_arr = np.full_like(t, vz)
        
        ax_arr = np.zeros_like(t)
        ay_arr = np.zeros_like(t)
        az_arr = np.zeros_like(t)
        
        return np.column_stack([t, x, y, z, vx_arr, vy_arr, vz_arr, ax_arr, ay_arr, az_arr])
    
    def sharp_maneuver(self, speed: float, altitude: float, turn_time: float) -> np.ndarray:
        """Generate sharp 90-degree turn maneuver"""
        t = np.arange(0, self.num_samples) * self.sample_rate_s
        
        # Start position
        x = np.zeros_like(t)
        y = np.zeros_like(t)
        z = np.full_like(t, altitude)
        
        # Initial velocity: moving in +x direction
        vx = np.full_like(t, speed)
        vy = np.zeros_like(t)
        
        # Find turn start index
        turn_start_idx = int((self.duration_s / 2 - turn_time / 2) / self.sample_rate_s)
        turn_end_idx = int((self.duration_s / 2 + turn_time / 2) / self.sample_rate_s)
        
        # Calculate positions and velocities
        for i in range(len(t)):
            if i == 0:
                x[i] = 10000  # Start 10km away
                y[i] = 0
            else:
                dt = self.sample_rate_s
                
                if turn_start_idx <= i < turn_end_idx:
                    # During turn: gradually change velocity direction
                    progress = (i - turn_start_idx) / (turn_end_idx - turn_start_idx)
                    angle = progress * np.pi / 2  # 90 degrees
                    vx[i] = speed * np.cos(angle)
                    vy[i] = speed * np.sin(angle)
                elif i >= turn_end_idx:
                    # After turn: moving in +y direction
                    vx[i] = 0
                    vy[i] = speed
                
                x[i] = x[i-1] + vx[i-1] * dt
                y[i] = y[i-1] + vy[i-1] * dt
        
        vz = np.zeros_like(t)
        
        # Calculate accelerations
        ax = np.gradient(vx, self.sample_rate_s)
        ay = np.gradient(vy, self.sample_rate_s)
        az = np.zeros_like(t)
        
        return np.column_stack([t, x, y, z, vx, vy, vz, ax, ay, az])
    
    def curved_path(self, speed: float, radius: float, altitude: float, center: Tuple[float, float]) -> np.ndarray:
        """Generate gentle curved path"""
        t = np.arange(0, self.num_samples) * self.sample_rate_s
        
        angular_vel = speed / radius
        theta = angular_vel * t + np.pi  # Start from left side
        
        x = center[0] + radius * np.cos(theta)
        y = center[1] + radius * np.sin(theta)
        z = np.full_like(t, altitude)
        
        vx = -speed * np.sin(theta)
        vy = speed * np.cos(theta)
        vz = np.zeros_like(t)
        
        ax = -speed**2 / radius * np.cos(theta)
        ay = -speed**2 / radius * np.sin(theta)
        az = np.zeros_like(t)
        
        return np.column_stack([t, x, y, z, vx, vy, vz, ax, ay, az])
    
    def level_flight_with_jitter(self, speed: float, altitude: float, jitter_mag: float) -> np.ndarray:
        """Generate level flight with altitude jitter"""
        t = np.arange(0, self.num_samples) * self.sample_rate_s
        
        angle = np.random.uniform(0, 2 * np.pi)
        vx = speed * np.cos(angle)
        vy = speed * np.sin(angle)
        
        x = 5000 + vx * t
        y = 5000 + vy * t
        
        # Add altitude jitter (noise)
        z = altitude + jitter_mag * np.random.randn(len(t))
        
        vx_arr = np.full_like(t, vx)
        vy_arr = np.full_like(t, vy)
        vz_arr = np.gradient(z, self.sample_rate_s)
        
        ax_arr = np.zeros_like(t)
        ay_arr = np.zeros_like(t)
        az_arr = np.gradient(vz_arr, self.sample_rate_s)
        
        return np.column_stack([t, x, y, z, vx_arr, vy_arr, vz_arr, ax_arr, ay_arr, az_arr])
    
    def stop_and_go(self, max_speed: float, altitude: float) -> np.ndarray:
        """Generate stop-and-go pattern with speed changes"""
        t = np.arange(0, self.num_samples) * self.sample_rate_s
        
        # Create speed profile: alternating between fast and slow
        speed = np.zeros_like(t)
        for i in range(len(t)):
            phase = (t[i] % 60) / 60  # 60 second cycle
            if phase < 0.4:
                speed[i] = max_speed
            elif phase < 0.5:
                speed[i] = max_speed * (1 - (phase - 0.4) / 0.1)  # Deceleration
            elif phase < 0.9:
                speed[i] = max_speed * 0.3
            else:
                speed[i] = max_speed * 0.3 + max_speed * 0.7 * (phase - 0.9) / 0.1  # Acceleration
        
        # Calculate position
        angle = np.pi / 4
        x = np.zeros_like(t)
        y = np.zeros_like(t)
        x[0] = 8000
        y[0] = 8000
        
        for i in range(1, len(t)):
            dt = self.sample_rate_s
            x[i] = x[i-1] + speed[i] * np.cos(angle) * dt
            y[i] = y[i-1] + speed[i] * np.sin(angle) * dt
        
        z = np.full_like(t, altitude)
        
        vx = speed * np.cos(angle)
        vy = speed * np.sin(angle)
        vz = np.zeros_like(t)
        
        ax = np.gradient(vx, self.sample_rate_s)
        ay = np.gradient(vy, self.sample_rate_s)
        az = np.zeros_like(t)
        
        return np.column_stack([t, x, y, z, vx, vy, vz, ax, ay, az])
    
    def oscillating_lateral(self, speed: float, amplitude: float, period: float, altitude: float) -> np.ndarray:
        """Generate trajectory with lateral oscillation (sine wave)"""
        t = np.arange(0, self.num_samples) * self.sample_rate_s
        
        # Main direction: +x
        x = speed * t + 3000
        
        # Lateral oscillation in y
        y = amplitude * np.sin(2 * np.pi * t / period)
        
        z = np.full_like(t, altitude)
        
        vx = np.full_like(t, speed)
        vy = amplitude * (2 * np.pi / period) * np.cos(2 * np.pi * t / period)
        vz = np.zeros_like(t)
        
        ax = np.zeros_like(t)
        ay = -amplitude * (2 * np.pi / period)**2 * np.sin(2 * np.pi * t / period)
        az = np.zeros_like(t)
        
        return np.column_stack([t, x, y, z, vx, vy, vz, ax, ay, az])
    
    def complex_maneuver(self, altitude: float) -> np.ndarray:
        """Generate complex multi-phase maneuver"""
        t = np.arange(0, self.num_samples) * self.sample_rate_s
        
        x = np.zeros_like(t)
        y = np.zeros_like(t)
        z = np.zeros_like(t)
        vx = np.zeros_like(t)
        vy = np.zeros_like(t)
        vz = np.zeros_like(t)
        
        # Initial position
        x[0] = 15000
        y[0] = 15000
        z[0] = altitude
        
        speed = 150.0
        
        for i in range(1, len(t)):
            dt = self.sample_rate_s
            phase = (t[i] % 120) / 120  # 2-minute cycle
            
            if phase < 0.25:
                # Straight flight
                vx[i] = speed
                vy[i] = 0
            elif phase < 0.5:
                # Turn
                turn_progress = (phase - 0.25) / 0.25
                angle = turn_progress * np.pi
                vx[i] = speed * np.cos(angle)
                vy[i] = speed * np.sin(angle)
            elif phase < 0.75:
                # Descend
                vx[i] = 0
                vy[i] = speed
                vz[i] = -10
            else:
                # Climb and turn
                turn_progress = (phase - 0.75) / 0.25
                angle = np.pi + turn_progress * np.pi / 2
                vx[i] = speed * np.cos(angle)
                vy[i] = speed * np.sin(angle)
                vz[i] = 15
            
            x[i] = x[i-1] + vx[i-1] * dt
            y[i] = y[i-1] + vy[i-1] * dt
            z[i] = z[i-1] + vz[i-1] * dt
            z[i] = np.clip(z[i], 500, 5000)  # Altitude limits
        
        ax = np.gradient(vx, self.sample_rate_s)
        ay = np.gradient(vy, self.sample_rate_s)
        az = np.gradient(vz, self.sample_rate_s)
        
        return np.column_stack([t, x, y, z, vx, vy, vz, ax, ay, az])


def create_simulation_folders(base_dir: str = "data/simulations", n_folders: int = 10) -> List[str]:
    """Create simulation folders with diverse trajectory types
    
    Args:
        base_dir: Base directory for simulations
        n_folders: Number of simulation folders to create
        
    Returns:
        List of created folder paths
    """
    config = get_config()
    sample_rate_ms = config.get('simulation.sample_rate_ms', 100)
    duration_min = config.get('simulation.flight_duration_min', 5)
    
    generator = TrajectoryGenerator(sample_rate_ms, duration_min)
    
    ensure_dir(base_dir)
    
    # Define 10 trajectory types
    trajectories = [
        ("straight_low_speed", lambda: generator.straight_constant_velocity(30.0, 1000, (20000, 10000))),
        ("straight_high_speed", lambda: generator.straight_constant_velocity(250.0, 3000, (25000, 15000))),
        ("ascending_spiral", lambda: generator.ascending_spiral(2000, 0.1, 5, (10000, 10000), 500)),
        ("descending_path", lambda: generator.descending_path(150.0, np.radians(15), (30000, 30000, 4000))),
        ("sharp_maneuver", lambda: generator.sharp_maneuver(180.0, 2000, 30)),
        ("curved_path", lambda: generator.curved_path(120.0, 8000, 2500, (5000, 5000))),
        ("level_flight_jitter", lambda: generator.level_flight_with_jitter(100.0, 1500, 10)),
        ("stop_and_go", lambda: generator.stop_and_go(200.0, 1800)),
        ("oscillating_lateral", lambda: generator.oscillating_lateral(140.0, 500, 40, 2200)),
        ("complex_maneuver", lambda: generator.complex_maneuver(2000))
    ]
    
    created_folders = []
    schema = config.get('binary_schema')
    
    for idx in range(min(n_folders, len(trajectories))):
        name, traj_func = trajectories[idx]
        
        # Create folder
        folder_path = Path(base_dir) / f"sim_{idx+1:02d}_{name}"
        ensure_dir(folder_path)
        
        # Generate trajectory
        logger.info(f"Generating trajectory: {name}")
        trajectory = traj_func()
        
        # Convert to records
        trackid = idx + 1
        records = []
        for row in trajectory:
            record = {
                'time': row[0],
                'trackid': float(trackid),
                'x': row[1],
                'y': row[2],
                'z': row[3],
                'vx': row[4],
                'vy': row[5],
                'vz': row[6],
                'ax': row[7],
                'ay': row[8],
                'az': row[9]
            }
            records.append(record)
        
        # Write binary file
        binary_path = folder_path / "radar_data.bin"
        write_binary_file(str(binary_path), records, schema)
        
        # Write metadata
        metadata = {
            'trajectory_type': name,
            'trackid': trackid,
            'num_records': len(records),
            'duration_seconds': trajectory[-1, 0],
            'sample_rate_ms': sample_rate_ms,
            'description': f"Simulated {name} trajectory"
        }
        
        metadata_path = folder_path / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Also save as CSV for reference
        import pandas as pd
        df = pd.DataFrame(records)
        csv_path = folder_path / "radar_data_reference.csv"
        df.to_csv(csv_path, index=False)
        
        logger.info(f"Created simulation: {folder_path}")
        created_folders.append(str(folder_path))
    
    logger.info(f"Created {len(created_folders)} simulation folders in {base_dir}")
    return created_folders


def create_large_training_dataset(output_path: str = "data/large_simulation_training.csv", 
                                  n_tracks: int = 200, duration_min: float = 10) -> str:
    """Create a large combined training dataset with diverse trajectories
    
    Args:
        output_path: Path to save the combined CSV
        n_tracks: Number of tracks to generate
        duration_min: Duration of each track in minutes
        
    Returns:
        Path to the created CSV file
    """
    config = get_config()
    sample_rate_ms = config.get('simulation.sample_rate_ms', 100)
    
    logger.info(f"Creating large training dataset with {n_tracks} tracks...")
    
    generator = TrajectoryGenerator(sample_rate_ms, duration_min)
    
    # Define trajectory templates with variations
    trajectory_templates = [
        ("straight_low_speed", lambda: generator.straight_constant_velocity(
            np.random.uniform(20, 50), np.random.uniform(500, 2000), 
            (np.random.uniform(15000, 25000), np.random.uniform(8000, 15000)))),
        ("straight_high_speed", lambda: generator.straight_constant_velocity(
            np.random.uniform(200, 300), np.random.uniform(2500, 4000), 
            (np.random.uniform(20000, 30000), np.random.uniform(10000, 20000)))),
        ("ascending_spiral", lambda: generator.ascending_spiral(
            np.random.uniform(1500, 3000), np.random.uniform(0.05, 0.15), 
            np.random.uniform(3, 10), (np.random.uniform(5000, 15000), np.random.uniform(5000, 15000)), 
            np.random.uniform(300, 800))),
        ("descending_path", lambda: generator.descending_path(
            np.random.uniform(120, 180), np.radians(np.random.uniform(10, 20)), 
            (np.random.uniform(25000, 35000), np.random.uniform(25000, 35000), np.random.uniform(3500, 4500)))),
        ("sharp_maneuver", lambda: generator.sharp_maneuver(
            np.random.uniform(150, 220), np.random.uniform(1500, 2500), 
            np.random.uniform(20, 40))),
        ("curved_path", lambda: generator.curved_path(
            np.random.uniform(100, 150), np.random.uniform(6000, 10000), 
            np.random.uniform(2000, 3000), (np.random.uniform(3000, 8000), np.random.uniform(3000, 8000)))),
        ("level_flight_jitter", lambda: generator.level_flight_with_jitter(
            np.random.uniform(80, 120), np.random.uniform(1200, 1800), 
            np.random.uniform(5, 15))),
        ("stop_and_go", lambda: generator.stop_and_go(
            np.random.uniform(150, 250), np.random.uniform(1500, 2200))),
        ("oscillating_lateral", lambda: generator.oscillating_lateral(
            np.random.uniform(120, 160), np.random.uniform(300, 700), 
            np.random.uniform(30, 50), np.random.uniform(2000, 2500))),
        ("complex_maneuver", lambda: generator.complex_maneuver(
            np.random.uniform(1800, 2200)))
    ]
    
    all_records = []
    
    for trackid in range(1, n_tracks + 1):
        # Select random trajectory type
        name, traj_func = trajectory_templates[trackid % len(trajectory_templates)]
        
        try:
            # Generate trajectory with variations
            trajectory = traj_func()
            
            # Convert to records
            for row in trajectory:
                record = {
                    'time': row[0],
                    'trackid': float(trackid),
                    'x': row[1],
                    'y': row[2],
                    'z': row[3],
                    'vx': row[4],
                    'vy': row[5],
                    'vz': row[6],
                    'ax': row[7],
                    'ay': row[8],
                    'az': row[9]
                }
                all_records.append(record)
            
            if trackid % 20 == 0:
                logger.info(f"Generated {trackid}/{n_tracks} tracks...")
                
        except Exception as e:
            logger.warning(f"Failed to generate track {trackid}: {e}")
            continue
    
    # Convert to DataFrame
    import pandas as pd
    df = pd.DataFrame(all_records)
    
    # Ensure output directory exists
    ensure_dir(Path(output_path).parent)
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    logger.info(f"Created large training dataset: {output_path}")
    logger.info(f"  Total tracks: {df['trackid'].nunique()}")
    logger.info(f"  Total records: {len(df)}")
    logger.info(f"  Duration: {df['time'].max():.2f} seconds")
    
    return output_path


# CLI interface
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Simulation Engine - Generate radar trajectory data')
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Folders command
    folders_parser = subparsers.add_parser('folders', help='Create simulation folders')
    folders_parser.add_argument('--outdir', default='data/simulations', help='Output directory')
    folders_parser.add_argument('--count', type=int, default=10, help='Number of simulations to create')
    
    # Large dataset command
    large_parser = subparsers.add_parser('large', help='Create large training dataset')
    large_parser.add_argument('--output', default='data/large_simulation_training.csv', help='Output CSV path')
    large_parser.add_argument('--tracks', type=int, default=200, help='Number of tracks to generate')
    large_parser.add_argument('--duration', type=float, default=10, help='Duration per track (minutes)')
    
    args = parser.parse_args()
    
    if args.command == 'folders' or args.command is None:
        # Default behavior - create folders
        outdir = getattr(args, 'outdir', 'data/simulations')
        count = getattr(args, 'count', 10)
        folders = create_simulation_folders(outdir, count)
        print(f"\nCreated {len(folders)} simulation folders:")
        for folder in folders:
            print(f"  {folder}")
    
    elif args.command == 'large':
        # Create large dataset
        csv_path = create_large_training_dataset(args.output, args.tracks, args.duration)
        print(f"\nCreated large training dataset:")
        print(f"  Path: {csv_path}")
        print(f"  Tracks: {args.tracks}")
        print(f"  Duration: {args.duration} minutes per track")
