import logging
from .feature_engineering import FeatureEngineer
from .data_merger import DataMerger

__all__ = ["FeatureEngineer", "DataMerger", "suppress_logging"]

def suppress_logging():
    logging.getLogger("stats_transformer.featurization.feature_engineering").setLevel(logging.CRITICAL + 1)
