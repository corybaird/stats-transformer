import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from stats_transformer.visualization.charts.timeseries.structural import RestrictionHeatmap, SwathePlot


@pytest.fixture
def restrictions():
    return [
        {"shock": "supply", "response": "gdp", "type": "sign", "value": "+", "horizon": 0},
        {"shock": "supply", "response": "cpi", "type": "sign", "value": "-", "horizon": 0},
        {"shock": "demand", "response": "gdp", "type": "zero", "horizon": 0},
        {"shock": "demand", "response": "cpi", "type": "sign", "value": "+", "horizon": 4},
    ]


def _bootstrap_results(n_draws=20, horizons=6, k=2):
    gen = np.random.default_rng(3)
    return [{"irf": gen.normal(scale=0.3, size=(horizons, k, k))} for _ in range(n_draws)]


def test_restriction_heatmap_plots_and_titles(restrictions):
    chart = RestrictionHeatmap(restrictions, variables=["gdp", "cpi"], shocks=["supply", "demand"])
    ax = chart.plot()
    try:
        assert ax.get_title() == "Impact Matrix Restrictions (h=0)"
        assert [t.get_text() for t in ax.get_xticklabels()] == ["supply", "demand"]
    finally:
        plt.close("all")


def test_restriction_heatmap_accepts_supplied_axis(restrictions):
    fig, ax = plt.subplots()
    try:
        chart = RestrictionHeatmap(restrictions, variables=["gdp", "cpi"], shocks=["supply", "demand"])
        returned = chart.plot(ax=ax)
        assert returned is ax
    finally:
        plt.close("all")


def test_restriction_heatmap_ignores_nonzero_horizon_restrictions():
    # Only horizon-0 restrictions belong on the impact matrix; the horizon=4
    # entry must not be drawn.
    only_late = [{"shock": "demand", "response": "cpi", "type": "sign", "value": "+", "horizon": 4}]
    chart = RestrictionHeatmap(only_late, variables=["gdp", "cpi"], shocks=["supply", "demand"])
    ax = chart.plot()
    try:
        # Nothing annotated, since every cell stayed NaN and is masked.
        assert len(ax.texts) == 0
    finally:
        plt.close("all")


def test_restriction_heatmap_skips_unknown_variables_and_shocks():
    unknown = [{"shock": "not_a_shock", "response": "not_a_var", "type": "sign", "value": "+", "horizon": 0}]
    chart = RestrictionHeatmap(unknown, variables=["gdp"], shocks=["supply"])
    ax = chart.plot()
    try:
        assert len(ax.texts) == 0
    finally:
        plt.close("all")


def test_swathe_plot_builds_grid_of_axes():
    chart = SwathePlot(_bootstrap_results(), {"irf": np.zeros((6, 2, 2))}, variables=["gdp", "cpi"], shocks=["supply", "demand"])
    axes = chart.plot()
    try:
        assert axes.shape == (2, 2)
        assert axes[0, 0].get_title() == "Shock: supply"
        assert axes[0, 0].get_ylabel() == "Response: gdp"
    finally:
        plt.close("all")


def test_swathe_plot_draws_bands_and_median_line():
    chart = SwathePlot(_bootstrap_results(), {"irf": np.zeros((6, 2, 2))}, variables=["gdp", "cpi"], shocks=["supply", "demand"])
    axes = chart.plot()
    try:
        ax = axes[0, 0]
        # two fill_between bands (68% and 90%)
        assert len(ax.collections) == 2
        # median line plus the zero reference line
        assert len(ax.lines) == 2
    finally:
        plt.close("all")


def test_swathe_plot_handles_single_variable_single_shock():
    # Exercises the K==1 and S==1 axis-reshaping branch, which is the easiest
    # thing to break when this plotting code is refactored.
    chart = SwathePlot(_bootstrap_results(k=1), {"irf": np.zeros((6, 1, 1))}, variables=["gdp"], shocks=["supply"])
    axes = chart.plot()
    try:
        assert axes.shape == (1, 1)
    finally:
        plt.close("all")


def test_swathe_plot_accepts_supplied_axes():
    fig, axes = plt.subplots(2, 2)
    try:
        chart = SwathePlot(_bootstrap_results(), {"irf": np.zeros((6, 2, 2))}, variables=["gdp", "cpi"], shocks=["supply", "demand"])
        returned = chart.plot(axes=axes)
        assert returned is axes
    finally:
        plt.close("all")
