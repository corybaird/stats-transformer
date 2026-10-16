import importlib
import importlib.metadata
import warnings

try:
    __version__ = importlib.metadata.version("stats-transformer")
except importlib.metadata.PackageNotFoundError:
    __version__ = "unknown"

from .data import load_example
from .featurization import FeatureEngineer, EventStudyBuilder
from .pipeline import Pipeline

# Top-level exports trimmed to the core API. Keep the deprecated names
# importable for one release by lazily importing them from their subpackage.
_DEPRECATED_SUBPACKAGE = {
    "RegressionModel": "stats_transformer.models",
    "RobustOLSModel": "stats_transformer.models",
    "IV2SLSModel": "stats_transformer.models",
    "PanelRegressionModel": "stats_transformer.models",
    "PanelIV2SLSModel": "stats_transformer.models",
    "PCAModel": "stats_transformer.models",
    "VARModel": "stats_transformer.models",
    "SVARModel": "stats_transformer.models",
    "ProxySVARModel": "stats_transformer.models",
    "SignZeroSVARModel": "stats_transformer.models",
    "BlanchardQuahModel": "stats_transformer.models",
    "VECMModel": "stats_transformer.models",
    "BVARModel": "stats_transformer.models",
    "DynamicFactorModel": "stats_transformer.models",
    "TVARModel": "stats_transformer.models",
    "VARForecaster": "stats_transformer.models",
    "VARLagSelector": "stats_transformer.models",
    "LocalProjectionsModel": "stats_transformer.models",
    "LocalProjectionsIVModel": "stats_transformer.models",
    "TableGenerator": "stats_transformer.reporting.tables",
    "BaseVisualizer": "stats_transformer.reporting",
    "DataVisualizer": "stats_transformer.reporting",
    "ModelVisualizer": "stats_transformer.reporting",
    "RegressionVisualizer": "stats_transformer.reporting",
    "TimeSeriesVisualizer": "stats_transformer.reporting",
    "CoefficientBarChart": "stats_transformer.reporting",
    "GroupedBarChart": "stats_transformer.reporting",
    "StackedBarChart": "stats_transformer.reporting",
    "TimeSeriesPlot": "stats_transformer.reporting",
    "IRFPlot": "stats_transformer.reporting",
    "FacetedTimeSeries": "stats_transformer.reporting",
    "BinnedScatterPlot": "stats_transformer.reporting",
    "ScatterWithRegression": "stats_transformer.reporting",
    "CorrelationHeatmap": "stats_transformer.reporting",
    "HistoricalDecompositionChart": "stats_transformer.reporting",
    "ImpulseResponseChart": "stats_transformer.reporting",
    "VarianceDecompositionChart": "stats_transformer.reporting",
    "ReportExporter": "stats_transformer.reporting",
    "TimeSeriesReporter": "stats_transformer.reporting",
}


def __getattr__(name):
    subpackage = _DEPRECATED_SUBPACKAGE.get(name)
    if subpackage is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    warnings.warn(
        f"stats_transformer.{name} is deprecated; import it from {subpackage}",
        DeprecationWarning,
        stacklevel=2,
    )
    return getattr(importlib.import_module(subpackage), name)


__all__ = [
    "Pipeline",
    "FeatureEngineer",
    "EventStudyBuilder",
    "load_example",
    "__version__",
]