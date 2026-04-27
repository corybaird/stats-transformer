"""
Base feature engineering class.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union


class BaseFeatureEngineer(ABC):
    """
    Abstract base class for feature engineering pipelines.

    This class defines the interface for feature engineering transformations
    that can be applied to time series data with entity-based grouping.
    """

    def __init__(self, params_path: Optional[str] = None, **kwargs):
        """
        Initialize the base feature engineer.

        Args:
            params_path: Optional path to YAML configuration file
            **kwargs: Additional configuration parameters
        """
        self.params = {}
        if params_path:
            self.params = self._load_params(params_path)
        
        # Apply any kwargs to override params
        for key, value in kwargs.items():
            if key in self.params:
                self.params[key] = value
        
        # Initialize logger
        self.logger = logging.getLogger(self.__class__.__name__)
        
    def _load_params(self, params_path: str) -> Dict:
        """
        Load parameters from YAML file.
        
        Args:
            params_path: Path to YAML file
            
        Returns:
            Dictionary of parameters
        """
        import yaml
        try:
            with open(params_path, "r") as f:
                return yaml.safe_load(f)
        except Exception as e:
            self.logger.error(f"Error loading parameters from {params_path}: {e}")
            raise
    
    @abstractmethod
    def fit(self, df):
        """Fit the feature engineer to data."""
        pass
    
    @abstractmethod
    def transform(self, df):
        """Transform the data."""
        pass
    
    @abstractmethod
    def fit_transform(self, df):
        """Fit and transform the data."""
        pass
