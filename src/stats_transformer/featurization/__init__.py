import logging
from .feature_engineering import FeatureEngineer
from .data_merger import DataMerger
from .event_study import EventStudyBuilder

__all__ = ["FeatureEngineer", "DataMerger", "EventStudyBuilder", "suppress_logging"]

def suppress_logging():
    logging.getLogger("stats_transformer.featurization.feature_engineering").setLevel(logging.CRITICAL + 1)
