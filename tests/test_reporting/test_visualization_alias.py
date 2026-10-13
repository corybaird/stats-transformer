import importlib
import sys

import pytest


def _drop_visualization_modules():
    for name in [m for m in sys.modules if m == "stats_transformer.visualization" or m.startswith("stats_transformer.visualization.")]:
        del sys.modules[name]


def test_legacy_visualization_imports_warn_and_resolve_to_reporting_modules():
    _drop_visualization_modules()
    with pytest.warns(DeprecationWarning, match="stats_transformer.reporting"):
        legacy = importlib.import_module("stats_transformer.visualization")

    from stats_transformer.reporting import CoefficientBarChart
    from stats_transformer.reporting.charts import bar
    from stats_transformer.visualization.charts.bar import CoefficientBarChart as LegacyCoefficientBarChart
    import stats_transformer.visualization.charts.bar as legacy_bar

    assert legacy.CoefficientBarChart is CoefficientBarChart
    assert LegacyCoefficientBarChart is CoefficientBarChart
    assert legacy_bar is bar
    assert legacy.eda.eda.EDAVisualizer is importlib.import_module("stats_transformer.reporting.eda.eda").EDAVisualizer
