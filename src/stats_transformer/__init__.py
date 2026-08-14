import importlib.metadata

try:
    __version__ = importlib.metadata.version("stats-transformer")
except importlib.metadata.PackageNotFoundError:
    __version__ = "unknown"

from .featurization import FeatureEngineer, EventStudyBuilder
from .models import RegressionModel, IV2SLSModel, PanelIV2SLSModel
from .pipeline import Pipeline
from .visualization import (
    BaseVisualizer, DataVisualizer, ModelVisualizer, RegressionVisualizer, TimeSeriesVisualizer,
    CoefficientBarChart, GroupedBarChart, StackedBarChart,
    TimeSeriesPlot, IRFPlot, FacetedTimeSeries,
    BinnedScatterPlot, ScatterWithRegression, CorrelationHeatmap,
    HistoricalDecompositionChart, ImpulseResponseChart, VarianceDecompositionChart
)
from .visualization.tables import TableGenerator
from .models.timeseries.reduced_form.local_projections import LocalProjectionsModel
from .models.timeseries.reduced_form.local_projections_iv import LocalProjectionsIVModel
from .reporting import TimeSeriesReporter

__all__ = [
    "FeatureEngineer",
    "RegressionModel",
    "IV2SLSModel",
    "PanelIV2SLSModel",
    "Pipeline",
    "BaseVisualizer",
    "DataVisualizer",
    "ModelVisualizer",
    "RegressionVisualizer",
    "TimeSeriesVisualizer",
    "CoefficientBarChart",
    "GroupedBarChart",
    "StackedBarChart",
    "TimeSeriesPlot",
    "IRFPlot",
    "FacetedTimeSeries",
    "BinnedScatterPlot",
    "ScatterWithRegression",
    "CorrelationHeatmap",
    "HistoricalDecompositionChart",
    "ImpulseResponseChart",
    "VarianceDecompositionChart",
    "TableGenerator",
    "LocalProjectionsModel",
    "EventStudyBuilder",
    "TimeSeriesReporter",
]
