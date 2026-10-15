import matplotlib
matplotlib.use("Agg")

import os
import numpy as np
import pandas as pd
import pytest
from stats_transformer.models.regression.robust_ols import RobustOLSModel
from stats_transformer.models.regression.panel import PanelRegressionModel
from stats_transformer.reporting.models.regression_viz import RegressionVisualizer
from stats_transformer.reporting.charts.bar import CoefficientForestPlot


def _fit_ols():
    df = pd.DataFrame({
        "y": [1, 2, 3, 4, 5, 6, 7],
        "x1": [2, 4, 6, 8, 10, 12, 14],
        "x2": [1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0],
    })
    model = RobustOLSModel(target="y", independent_variables=["x1", "x2"])
    model.fit(df)
    return model.get_model_metadata()


def test_forest_plot_for_ols(tmp_path):
    metadata = _fit_ols()
    viz = RegressionVisualizer(output_dir=str(tmp_path))
    paths = viz.create_coefficient_plot_from_json(metadata, subdir="regression")
    assert paths
    assert all(os.path.exists(p) for p in paths)


def test_forest_plot_for_panel(tmp_path):
    rng = np.random.RandomState(0)
    n_entities, n_periods = 10, 5
    n = n_entities * n_periods
    df = pd.DataFrame({
        "entity": np.repeat(np.arange(n_entities), n_periods),
        "date": np.tile(np.arange(n_periods), n_entities),
        "y": rng.random(size=n),
        "x1": rng.random(size=n),
    })
    model = PanelRegressionModel(
        target="y", independent_variables=["x1"], entity_column="entity", time_column="date"
    )
    model.fit(df)
    viz = RegressionVisualizer(output_dir=str(tmp_path))
    paths = viz.create_coefficient_plot_from_json(model.get_model_metadata(), subdir="regression")
    assert paths
    assert all(os.path.exists(p) for p in paths)


def test_ci_whiskers_equal_coef_plus_minus_1_96_se():
    chart = CoefficientForestPlot()
    fig, ax = chart.plot(
        labels=["a", "b"],
        coefficients=[2.0, -1.0],
        std_errors=[0.5, 0.25],
        p_values=[0.001, 0.2],
    )
    # The main whisker line for each point spans coef - 1.96*se .. coef + 1.96*se.
    whiskers = sorted(
        tuple(sorted((line.get_xdata()[0], line.get_xdata()[-1])))
        for line in ax.lines
        if len(set(line.get_ydata())) == 1 and len(set(line.get_xdata())) > 1  # horizontal segments only
    )
    assert whiskers == pytest.approx([(-1.49, -0.51), (1.02, 2.98)])


def test_intercept_excluded_by_default_and_includable():
    metadata = {
        "coefficients": {
            "const": {"value": 1.0, "std_err": 0.1},
            "var1": {"value": 2.5, "std_err": 0.5},
        }
    }
    viz = RegressionVisualizer(output_dir="unused")
    captured = []
    viz.save_figure = lambda fig, *args, **kwargs: (captured.append(fig), [])[-1]

    viz.create_coefficient_plot_from_json(metadata, display_only=True)
    fig = captured[-1]
    ax = fig.axes[0]
    # Default: intercept excluded, model order kept.
    assert [t.get_text() for t in ax.get_yticklabels()] == ["Var1"]

    viz.create_coefficient_plot_from_json(metadata, display_only=True, include_intercept=True)
    fig = captured[-1]
    ax = fig.axes[0]
    assert [t.get_text() for t in ax.get_yticklabels()] == ["Const", "Var1"]


def test_kind_bar_still_works(tmp_path):
    metadata = _fit_ols()
    viz = RegressionVisualizer(output_dir=str(tmp_path))
    paths = viz.create_coefficient_plot_from_json(metadata, subdir="regression", kind="bar")
    assert paths
    assert all(os.path.exists(p) for p in paths)
    assert all(p.endswith(".png") for p in paths)