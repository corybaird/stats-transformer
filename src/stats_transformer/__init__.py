import importlib.metadata

try:
    __version__ = importlib.metadata.version("stats-transformer")
except importlib.metadata.PackageNotFoundError:
    __version__ = "unknown"

from .featurization import FeatureEngineer, EventStudyBuilder
from .models import (
    RegressionModel,
    RobustOLSModel,
    IV2SLSModel,
    PanelRegressionModel,
    PanelIV2SLSModel,
    PCAModel,
    VARModel,
    SVARModel,
    ProxySVARModel,
    SignZeroSVARModel,
    BlanchardQuahModel,
    VECMModel,
    BVARModel,
    DynamicFactorModel,
    TVARModel,
    VARForecaster,
    VARLagSelector,
    LocalProjectionsModel,
    LocalProjectionsIVModel,
)
from .reporting.tables import TableGenerator
from .pipeline import Pipeline
from .reporting import (
    BaseVisualizer, DataVisualizer, ModelVisualizer, RegressionVisualizer, TimeSeriesVisualizer,
    CoefficientBarChart, GroupedBarChart, StackedBarChart,
    TimeSeriesPlot, IRFPlot, FacetedTimeSeries,
    BinnedScatterPlot, ScatterWithRegression, CorrelationHeatmap,
    HistoricalDecompositionChart, ImpulseResponseChart, VarianceDecompositionChart,
    ReportExporter, TimeSeriesReporter
)
__all__ = [
    "FeatureEngineer",
    "RegressionModel",
    "RobustOLSModel",
    "IV2SLSModel",
    "PanelRegressionModel",
    "PanelIV2SLSModel",
    "PCAModel",
    "VARModel",
    "SVARModel",
    "ProxySVARModel",
    "SignZeroSVARModel",
    "BlanchardQuahModel",
    "VECMModel",
    "BVARModel",
    "DynamicFactorModel",
    "TVARModel",
    "VARForecaster",
    "VARLagSelector",
    "LocalProjectionsModel",
    "LocalProjectionsIVModel",
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
    "EventStudyBuilder",
    "ReportExporter",
    "TimeSeriesReporter",
]
