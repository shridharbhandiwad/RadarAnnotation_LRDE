"""Configuration management for Radar Annotation Application"""
import json
import os
from pathlib import Path
from typing import Dict, Any

# Default configuration
DEFAULT_CONFIG = {
    "binary_schema": {
        "record_size": 80,  # 10 x float64 (8 bytes each)
        "endian": "little",
        "fields": [
            {"name": "time", "type": "float64", "offset": 0},
            {"name": "trackid", "type": "float64", "offset": 8},
            {"name": "x", "type": "float64", "offset": 16},
            {"name": "y", "type": "float64", "offset": 24},
            {"name": "z", "type": "float64", "offset": 32},
            {"name": "vx", "type": "float64", "offset": 40},
            {"name": "vy", "type": "float64", "offset": 48},
            {"name": "vz", "type": "float64", "offset": 56},
            {"name": "ax", "type": "float64", "offset": 64},
            {"name": "ay", "type": "float64", "offset": 72}
        ],
        "struct_format": "<10d"  # little-endian, 10 doubles
    },
    "autolabel_thresholds": {
        "level_flight_threshold": 5.0,  # meters vertical change
        "curvature_threshold": 0.01,  # radians per meter
        "low_speed_threshold": 50.0,  # m/s
        "high_speed_threshold": 200.0,  # m/s
        "light_maneuver_threshold": 2.0,  # m/s^2 acceleration
        "high_maneuver_threshold": 5.0,  # m/s^2 acceleration
        "range_rate_threshold": 1.0,  # m/s for incoming/outgoing
        "fixed_range_threshold": 10.0,  # meters
        "min_points_per_track": 3
    },
    "ml_params": {
        "xgboost": {
            "n_estimators": 100,
            "max_depth": 6,
            "learning_rate": 0.1,
            "objective": "multi:softmax",
            "random_state": 42
        },
        "lstm": {
            "units": 64,
            "dropout": 0.2,
            "epochs": 50,
            "batch_size": 32,
            "sequence_length": 20
        },
        "transformer": {
            "d_model": 64,
            "num_heads": 4,
            "ff_dim": 128,
            "num_layers": 2,
            "dropout": 0.1,
            "epochs": 50,
            "batch_size": 32,
            "sequence_length": 20
        },
        "train_test_split": 0.8
    },
    "visualization": {
        "ppi_range_km": 100,
        "color_map": "viridis",
        "point_size": 5
    },
    "simulation": {
        "sample_rate_ms": 100,  # 100ms between samples
        "flight_duration_min": 5,  # 5 minutes
        "radar_origin": [0, 0, 0]
    }
}


class Config:
    """Configuration manager"""
    
    def __init__(self, config_path: str = None):
        """Initialize configuration
        
        Args:
            config_path: Path to JSON config file (optional)
        """
        self.config = DEFAULT_CONFIG.copy()
        
        if config_path and os.path.exists(config_path):
            self.load(config_path)
    
    def load(self, path: str) -> None:
        """Load configuration from JSON file"""
        with open(path, 'r') as f:
            user_config = json.load(f)
            self._deep_update(self.config, user_config)
    
    def save(self, path: str) -> None:
        """Save configuration to JSON file"""
        with open(path, 'w') as f:
            json.dump(self.config, f, indent=2)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by dot-separated key"""
        keys = key.split('.')
        value = self.config
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k, default)
            else:
                return default
        return value
    
    def set(self, key: str, value: Any) -> None:
        """Set configuration value by dot-separated key"""
        keys = key.split('.')
        config = self.config
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        config[keys[-1]] = value
    
    @staticmethod
    def _deep_update(base: dict, update: dict) -> None:
        """Recursively update nested dictionary"""
        for key, value in update.items():
            if isinstance(value, dict) and key in base and isinstance(base[key], dict):
                Config._deep_update(base[key], value)
            else:
                base[key] = value


# Global config instance
_global_config = None


def get_config(config_path: str = None) -> Config:
    """Get global configuration instance"""
    global _global_config
    if _global_config is None:
        _global_config = Config(config_path)
    return _global_config


def save_default_config(path: str = "config/default_config.json") -> None:
    """Save default configuration to file"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(DEFAULT_CONFIG, f, indent=2)
