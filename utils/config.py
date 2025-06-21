"""
Configuration module for Sonit
Manages application settings and constants
"""

import os
import json
from typing import Dict, Any, Optional


class Config:
    """Configuration manager for Sonit application"""
    
    def __init__(self, config_file='data/config.json'):
        """
        Initialize configuration
        
        Args:
            config_file (str): Path to configuration file
        """
        self.config_file = config_file
        self.config = self._load_default_config()
        self._load_config()
    
    def _load_default_config(self) -> Dict[str, Any]:
        """Load default configuration"""
        return {
            'audio': {
                'sample_rate': 16000,
                'chunk_size': 1024,
                'channels': 1,
                'format': 'int16',
                'device_index': None
            },
            'model': {
                'type': 'neural_network',
                'confidence_threshold': 0.5,
                'max_predictions': 5,
                'hidden_size': 128,
                'dropout_rate': 0.3
            },
            'training': {
                'min_samples': 5,
                'max_samples': 1000,
                'validation_split': 0.2,
                'batch_size': 32,
                'learning_rate': 0.001,
                'num_epochs': 100,
                'early_stopping_patience': 10
            },
            'ui': {
                'auto_save': True,
                'show_confidence': True,
                'dark_mode': False,
                'window_width': 800,
                'window_height': 600
            },
            'features': {
                'n_mfcc': 13,
                'n_mels': 128,
                'n_chroma': 12,
                'n_fft': 2048,
                'hop_length': 512
            },
            'paths': {
                'data_dir': 'data',
                'recordings_dir': 'data/recordings',
                'models_dir': 'data/models',
                'plots_dir': 'data/plots',
                'database_file': 'data/sonit.db'
            }
        }
    
    def _load_config(self):
        """Load configuration from file"""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    file_config = json.load(f)
                    self._merge_config(file_config)
        except Exception as e:
            print(f"Error loading config: {e}")
    
    def _merge_config(self, file_config: Dict[str, Any]):
        """Merge file configuration with default config"""
        for section, values in file_config.items():
            if section in self.config:
                if isinstance(values, dict):
                    self.config[section].update(values)
                else:
                    self.config[section] = values
            else:
                self.config[section] = values
    
    def save_config(self):
        """Save current configuration to file"""
        try:
            os.makedirs(os.path.dirname(self.config_file), exist_ok=True)
            with open(self.config_file, 'w') as f:
                json.dump(self.config, f, indent=2)
        except Exception as e:
            print(f"Error saving config: {e}")
    
    def get(self, key: str, default=None):
        """
        Get configuration value using dot notation
        
        Args:
            key (str): Configuration key (e.g., 'audio.sample_rate')
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        keys = key.split('.')
        value = self.config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key: str, value: Any):
        """
        Set configuration value using dot notation
        
        Args:
            key (str): Configuration key (e.g., 'audio.sample_rate')
            value: Value to set
        """
        keys = key.split('.')
        config = self.config
        
        # Navigate to the parent of the target key
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        # Set the value
        config[keys[-1]] = value
    
    def get_audio_config(self) -> Dict[str, Any]:
        """Get audio configuration"""
        return self.config['audio']
    
    def get_model_config(self) -> Dict[str, Any]:
        """Get model configuration"""
        return self.config['model']
    
    def get_training_config(self) -> Dict[str, Any]:
        """Get training configuration"""
        return self.config['training']
    
    def get_ui_config(self) -> Dict[str, Any]:
        """Get UI configuration"""
        return self.config['ui']
    
    def get_features_config(self) -> Dict[str, Any]:
        """Get features configuration"""
        return self.config['features']
    
    def get_paths_config(self) -> Dict[str, Any]:
        """Get paths configuration"""
        return self.config['paths']
    
    def ensure_directories(self):
        """Ensure all required directories exist"""
        paths = self.get_paths_config()
        
        for path_key, path_value in paths.items():
            if path_key.endswith('_dir'):
                os.makedirs(path_value, exist_ok=True)
    
    def update_from_settings(self, settings_dict: Dict[str, Any]):
        """
        Update configuration from settings dictionary
        
        Args:
            settings_dict (dict): Settings dictionary from settings screen
        """
        # Update audio settings
        if 'audio' in settings_dict:
            for key, value in settings_dict['audio'].items():
                self.set(f'audio.{key}', value)
        
        # Update model settings
        if 'model' in settings_dict:
            for key, value in settings_dict['model'].items():
                self.set(f'model.{key}', value)
        
        # Update UI settings
        if 'ui' in settings_dict:
            for key, value in settings_dict['ui'].items():
                self.set(f'ui.{key}', value)
        
        # Update training settings
        if 'training' in settings_dict:
            for key, value in settings_dict['training'].items():
                self.set(f'training.{key}', value)
        
        # Save updated configuration
        self.save_config()
    
    def get_all_config(self) -> Dict[str, Any]:
        """Get all configuration"""
        return self.config.copy()
    
    def reset_to_defaults(self):
        """Reset configuration to defaults"""
        self.config = self._load_default_config()
        self.save_config()


# Global configuration instance
config = Config()


def get_config() -> Config:
    """Get global configuration instance"""
    return config 