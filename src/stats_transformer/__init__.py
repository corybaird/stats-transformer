import importlib.metadata

try:
    __version__ = importlib.metadata.version("stats-transformer")
except importlib.metadata.PackageNotFoundError:
    __version__ = "unknown"

from .featurization import FeatureEngineer
from .models import RegressionModel
from .pipeline import Pipeline
from .visualization import (
    BaseVisualizer, DataVisualizer, ModelVisualizer, RegressionVisualizer,
    CoefficientBarChart, GroupedBarChart, StackedBarChart,
    TimeSeriesPlot, IRFPlot, FacetedTimeSeries,
    BinnedScatterPlot, ScatterWithRegression, CorrelationHeatmap
)

__all__ = [
    "FeatureEngineer",
    "RegressionModel",
    "Pipeline",
    "BaseVisualizer",
    "DataVisualizer",
    "ModelVisualizer",
    "RegressionVisualizer",
    "CoefficientBarChart",
    "GroupedBarChart",
    "StackedBarChart",
    "TimeSeriesPlot",
    "IRFPlot",
    "FacetedTimeSeries",
    "BinnedScatterPlot",
    "ScatterWithRegression",
    "CorrelationHeatmap",
]
