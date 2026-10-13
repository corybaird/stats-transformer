"""Deprecated alias for :mod:`stats_transformer.reporting`.

The ``visualization`` package was merged into ``reporting`` in 1.7.0. Old import
paths such as ``stats_transformer.visualization.charts.bar`` resolve to the same
module objects under ``reporting`` and will be removed in a future release.
"""
import importlib
import sys
import warnings

warnings.warn(
    "`stats_transformer.visualization` is deprecated and will be removed in a future release; import from `stats_transformer.reporting` instead.",
    DeprecationWarning,
    stacklevel=2,
)

_MOVED_MODULES = [
    "base",
    "charts", "charts.bar", "charts.heatmap", "charts.scatter", "charts.time_series",
    "charts.timeseries", "charts.timeseries.structural", "charts.timeseries_reporting",
    "defaults", "defaults.colors", "defaults.labels",
    "eda", "eda.data_viz", "eda.eda",
    "formatters", "formatters.significance", "formatters.style",
    "models", "models.model_viz", "models.regression_viz", "models.timeseries_viz",
    "tables", "tables.descriptive_stats", "tables.table_generator", "tables.timeseries",
]

for _name in _MOVED_MODULES:
    _module = importlib.import_module(f"stats_transformer.reporting.{_name}")
    sys.modules[f"{__name__}.{_name}"] = _module
    if "." not in _name:
        globals()[_name] = _module

from stats_transformer.reporting import *  # noqa: E402,F401,F403
from stats_transformer.reporting import __all__  # noqa: E402,F401
