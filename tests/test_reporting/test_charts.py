import matplotlib
matplotlib.use("Agg")

import pytest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from stats_transformer.reporting.charts import (
    CoefficientBarChart, GroupedBarChart, StackedBarChart,
    TimeSeriesPlot, IRFPlot, FacetedTimeSeries,
    BinnedScatterPlot, ScatterWithRegression, CorrelationHeatmap
)


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def test_coefficient_bar_chart():
    chart = CoefficientBarChart()
    coefficients = [0.5, -0.2, 0.1]
    fig, ax = chart.plot(
        labels=["A", "B", "C"],
        coefficients=coefficients,
        std_errors=[0.1, 0.05, 0.02],
        p_values=[0.01, 0.2, 0.05],
        ylabel="coefficient",
        title="Coefficients"
    )
    # one bar per coefficient, drawn at the coefficient value
    np.testing.assert_allclose([p.get_height() for p in ax.patches], coefficients)
    assert [t.get_text() for t in ax.get_xticklabels()] == ["A", "B", "C"]
    assert ax.get_ylabel() == "coefficient"
    assert ax.get_title() == "Coefficients"
    # zero reference line
    assert any(np.array_equal(line.get_ydata(), [0, 0]) for line in ax.lines)
    # significance stars for p < 0.05 and p < 0.10, none for p = 0.2
    assert [t.get_text() for t in ax.texts] == ["**", "*"]


def test_grouped_bar_chart():
    df = pd.DataFrame({
        'group': ['G1', 'G1', 'G2', 'G2'],
        'x': ['A', 'B', 'A', 'B'],
        'y': [1, 2, 3, 4],
        'err': [0.1, 0.2, 0.1, 0.2],
        'pval': [0.01, 0.05, 0.2, 0.001]
    })
    chart = GroupedBarChart()
    fig, ax = chart.plot(df, x_col='x', y_col='y', group_col='group', error_col='err', pval_col='pval', ylabel="y")
    # two groups side by side at each of two x positions, one bar per row value
    assert [p.get_height() for p in ax.patches] == [1, 2, 3, 4]
    assert [t.get_text() for t in ax.get_xticklabels()] == ["A", "B"]
    assert [t.get_text() for t in ax.get_legend().get_texts()] == ["G1", "G2"]
    assert ax.get_ylabel() == "y"


def test_stacked_bar_chart():
    df = pd.DataFrame({
        'x': ['A', 'B', 'C'],
        'y1': [1, 2, 3],
        'y2': [4, 5, 6]
    })
    chart = StackedBarChart()
    fig, ax = chart.plot(df, x_col='x', y_cols=['y1', 'y2'])
    # two series x three categories, second series stacked on top of the first
    assert [p.get_height() for p in ax.patches] == [1, 2, 3, 4, 5, 6]
    assert [p.get_y() for p in ax.patches] == [0, 0, 0, 1, 2, 3]
    assert [t.get_text() for t in ax.get_xticklabels()] == ["A", "B", "C"]
    assert [t.get_text() for t in ax.get_legend().get_texts()] == ["Y1", "Y2"]


def test_time_series_plot():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        'date': pd.date_range('2020-01-01', periods=10),
        'y1': rng.normal(size=10),
        'y2': rng.normal(size=10)
    })
    chart = TimeSeriesPlot()
    fig, ax = chart.plot(df, x_col='date', y_cols=['y1', 'y2'])
    assert len(ax.lines) == 2
    np.testing.assert_allclose(ax.lines[0].get_ydata(), df['y1'])
    np.testing.assert_allclose(ax.lines[1].get_ydata(), df['y2'])
    assert [line.get_label() for line in ax.lines] == ["Y1", "Y2"]


def test_irf_plot():
    rng = np.random.default_rng(1)
    df = pd.DataFrame({
        'var': ['V1']*5 + ['V2']*5,
        'horizon': list(range(5)) * 2,
        'coef': rng.normal(size=10),
        'ci_lower': rng.normal(size=10) - 1,
        'ci_upper': rng.normal(size=10) + 1
    })
    chart = IRFPlot()
    fig, axes = chart.plot(df, horizon_col='horizon', coef_col='coef', ci_lower_col='ci_lower', ci_upper_col='ci_upper', group_col='var')
    # one panel per variable, each with the response line plus a CI fill
    assert len(axes) == 2
    for i, group in enumerate(['V1', 'V2']):
        ax = axes[i]
        expected = df.loc[df['var'] == group, 'coef'].to_numpy()
        np.testing.assert_allclose(ax.lines[0].get_ydata(), expected)
        assert len(ax.collections) == 1
        assert ax.get_title() == group
        assert ax.get_xlabel() == "Horizon"
    assert axes[0].get_ylabel() == "IRF Coefficient"


def test_faceted_time_series():
    rng = np.random.default_rng(2)
    df = pd.DataFrame({
        'date': list(pd.date_range('2020-01-01', periods=5)) * 2,
        'entity': ['E1']*5 + ['E2']*5,
        'val': rng.normal(size=10)
    })
    chart = FacetedTimeSeries()
    fig, axes = chart.plot(df, date_col='date', value_cols=['val'], facet_col='entity')
    # one panel per entity, unused grid cells hidden
    assert len(axes) == 3
    assert [ax.get_visible() for ax in axes] == [True, True, False]
    for i, entity in enumerate(['E1', 'E2']):
        expected = df.loc[df['entity'] == entity, 'val'].to_numpy()
        np.testing.assert_allclose(axes[i].lines[0].get_ydata(), expected)
        assert axes[i].get_title() == entity
        assert axes[i].get_xlabel() == "Date"
    assert axes[0].get_ylabel() == "Value"


def test_binned_scatter_plot():
    rng = np.random.default_rng(3)
    df = pd.DataFrame({
        'x': rng.normal(size=100),
        'y': rng.normal(size=100)
    })
    chart = BinnedScatterPlot()
    fig, ax = chart.plot(df, x_col='x', y_col='y', n_bins=5)
    # raw points, five binned means (20 observations per bin), and the error bars
    assert [len(c.get_offsets()) for c in ax.collections[:2]] == [100, 5]
    assert len(ax.collections) == 3
    # trend line spans the regressor range and its legend label is the OLS slope
    trend = [line for line in ax.lines if line.get_label().startswith("slope = ")]
    assert len(trend) == 1
    assert len(trend[0].get_ydata()) == 100
    slope = np.polyfit(df['x'], df['y'], 1)[0]
    assert trend[0].get_label() == f"slope = {slope:.3f}"
    assert ax.get_xlabel() == "X"
    assert ax.get_ylabel() == "Y"


def test_scatter_with_regression():
    rng = np.random.default_rng(4)
    df = pd.DataFrame({
        'x': rng.normal(size=50),
        'y': rng.normal(size=50)
    })
    chart = ScatterWithRegression()
    fig, ax = chart.plot(df, x_col='x', y_col='y')
    # all points drawn plus one fitted regression line
    assert len(ax.collections[0].get_offsets()) == 50
    assert len(ax.lines) == 1
    corr = df['x'].corr(df['y'])
    assert [t.get_text() for t in ax.texts] == [f"Corr: {corr:.3f}"]


def test_correlation_heatmap():
    rng = np.random.default_rng(5)
    df = pd.DataFrame(rng.normal(size=(10, 4)), columns=['A', 'B', 'C', 'D'])
    chart = CorrelationHeatmap()
    fig, ax = chart.plot(df)
    corr = df.corr()
    # one mesh holding the correlation matrix, upper triangle (incl. diagonal) masked
    assert len(ax.collections) == 1
    drawn = np.ma.filled(ax.collections[0].get_array().reshape(corr.shape), np.nan)
    expected = corr.to_numpy().copy()
    expected[np.triu_indices_from(expected)] = np.nan
    np.testing.assert_allclose(drawn, expected)
    # only the six unmasked cells are annotated
    assert len(ax.texts) == 6
    assert ax.get_title() == "Pearson Correlation Matrix"
    assert [t.get_text() for t in ax.get_xticklabels()] == ["A", "B", "C", "D"]
