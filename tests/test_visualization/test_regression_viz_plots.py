import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
from stats_transformer.visualization.models.regression_viz import RegressionVisualizer


@pytest.fixture
def fitted_model():
    gen = np.random.default_rng(11)
    n = 80
    X = pd.DataFrame({"x1": gen.normal(size=n), "x2": gen.normal(size=n)})
    y = pd.Series(2.0 * X["x1"] - 1.0 * X["x2"] + gen.normal(scale=0.5, size=n), name="y")
    X_const = sm.add_constant(X)
    return sm.OLS(y, X_const).fit(), X, y


@pytest.fixture
def viz(tmp_path):
    return RegressionVisualizer(output_dir=str(tmp_path))


def _close():
    plt.close("all")


def test_partial_regression_plots_write_one_file_per_regressor(viz, fitted_model):
    model, X, y = fitted_model
    try:
        saved = viz.create_partial_regression_plots(model, X, y)
        assert len(saved) == 2
        assert all(path for path in saved)
    finally:
        _close()


def test_partial_regression_plots_accept_numpy_input(viz, fitted_model):
    # The method builds X1/X2 column names when handed a bare array.
    model, X, y = fitted_model
    try:
        saved = viz.create_partial_regression_plots(model, X.to_numpy(), y)
        assert len(saved) == 2
    finally:
        _close()


def test_component_plus_residual_plots(viz, fitted_model):
    model, X, y = fitted_model
    try:
        saved = viz.create_component_plus_residual_plots(model, X)
        assert len(saved) == 2
    finally:
        _close()


def test_residuals_vs_fitted(viz, fitted_model):
    model, X, y = fitted_model
    try:
        result = viz.create_residuals_vs_fitted(model)
        assert result is not None
    finally:
        _close()


def test_actual_vs_predicted(viz, fitted_model):
    model, X, y = fitted_model
    try:
        result = viz.create_actual_vs_predicted(model)
        assert result is not None
    finally:
        _close()


def test_prediction_intervals(viz, fitted_model):
    model, X, y = fitted_model
    try:
        result = viz.create_prediction_intervals(model, sm.add_constant(X), y)
        assert result is not None
    finally:
        _close()


def test_prediction_intervals_returns_none_without_get_prediction(viz):
    class Bare:
        pass

    assert viz.create_prediction_intervals(Bare(), None, None) is None


def test_variance_decomposition(viz, fitted_model):
    model, X, y = fitted_model
    try:
        result = viz.create_variance_decomposition(model, X)
        assert result is not None
    finally:
        _close()


def test_display_only_does_not_write_files(viz, fitted_model, tmp_path):
    model, X, y = fitted_model
    try:
        viz.create_residuals_vs_fitted(model, display_only=True)
        written = list(tmp_path.rglob("*.png"))
        assert written == []
    finally:
        _close()
