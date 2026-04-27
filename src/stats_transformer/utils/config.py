"""Configuration handling for the stats-transformer library."""

import yaml
from typing import Dict, Any, Optional
import logging


class Config:
    """
    Unified configuration system for the stats-transformer library.
    
    This class handles configuration from both YAML files and programmatic
    settings, providing a consistent interface for all components.
    """
    
    def __init__(self, config_path: Optional[str] = None, **kwargs):
        """
        Initialize configuration.
        
        Args:
            config_path: Optional path to YAML configuration file
            **kwargs: Additional configuration parameters to override or supplement
        """
        self.logger = logging.getLogger(__name__)
        self.config = {}
        
        # Load from file if provided
        if config_path:
            self.load_from_file(config_path)
        
        # Update with any keyword arguments
        if kwargs:
            self.update(kwargs)
    
    def load_from_file(self, config_path: str) -> None:
        """
        Load configuration from YAML file.
        
        Args:
            config_path: Path to YAML file
            
        Raises:
            FileNotFoundError: If the configuration file is not found
            yaml.YAMLError: If the file is not valid YAML
        """
        try:
            with open(config_path, "r") as f:
                file_config = yaml.safe_load(f)
            if file_config:
                self.config.update(file_config)
            self.logger.info(f"Configuration loaded from {config_path}")
        except FileNotFoundError:
            self.logger.error(f"Configuration file {config_path} not found")
            raise
        except yaml.YAMLError as e:
            self.logger.error(f"Error parsing configuration file {config_path}: {e}")
            raise
    
    def update(self, new_config: Dict[str, Any]) -> None:
        """
        Update configuration with new values.
        
        Args:
            new_config: Dictionary of configuration values to update
        """
        self.config.update(new_config)
        self.logger.debug(f"Configuration updated with {len(new_config)} values")
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by key.
        
        Args:
            key: Configuration key (can use dot notation for nested keys)
            default: Default value if key is not found
            
        Returns:
            Configuration value or default
        """
        # Support dot notation for nested keys
        if "." in key:
            keys = key.split(".")
            value = self.config
            for k in keys:
                if isinstance(value, dict) and k in value:
                    value = value[k]
                else:
                    return default
            return value
        else:
            return self.config.get(key, default)
    
    def set(self, key: str, value: Any) -> None:
        """
        Set configuration value.
        
        Args:
            key: Configuration key (can use dot notation for nested keys)
            value: Value to set
        """
        # Support dot notation for nested keys
        if "." in key:
            keys = key.split(".")
            current = self.config
            for k in keys[:-1]:
                if k not in current:
                    current[k] = {}
                current = current[k]
            current[keys[-1]] = value
        else:
            self.config[key] = value
        
        self.logger.debug(f"Configuration set: {key} = {value}")
    
    def get_featurization_config(self) -> Dict[str, Any]:
        """
        Get featurization configuration.
        
        Returns:
            Dictionary with featurization settings
        """
        return self.get("featurization", {})
    
    def get_model_config(self) -> Dict[str, Any]:
        """
        Get model configuration.
        
        Returns:
            Dictionary with model settings
        """
        return self.get("model", {})
    
    def get_visualization_config(self) -> Dict[str, Any]:
        """
        Get visualization configuration.
        
        Returns:
            Dictionary with visualization settings
        """
        return self.get("visualization", {})
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Get configuration as dictionary.
        
        Returns:
            Complete configuration dictionary
        """
        return self.config.copy()
    
    def save_to_file(self, config_path: str) -> None:
        """
        Save configuration to YAML file.
        
        Args:
            config_path: Path to save configuration file
        """
        try:
            with open(config_path, "w") as f:
                yaml.dump(self.config, f, default_flow_style=False)
            self.logger.info(f"Configuration saved to {config_path}")
        except Exception as e:
            self.logger.error(f"Error saving configuration to {config_path}: {e}")
            raise
