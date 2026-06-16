import pytest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from stats_transformer.visualization.charts import (
    CoefficientBarChart, GroupedBarChart, StackedBarChart,
    TimeSeriesPlot, IRFPlot, FacetedTimeSeries,
    BinnedScatterPlot, ScatterWithRegression, CorrelationHeatmap
)

def test_coefficient_bar_chart():
    chart = CoefficientBarChart()
    fig, ax = chart.plot(
        labels=["A", "B", "C"],
        coefficients=[0.5, -0.2, 0.1],
        std_errors=[0.1, 0.05, 0.02],
        p_values=[0.01, 0.2, 0.05]
    )
    assert fig is not None
    assert ax is not None
    assert len(ax.patches) == 3

def test_grouped_bar_chart():
    df = pd.DataFrame({
        'group': ['G1', 'G1', 'G2', 'G2'],
        'x': ['A', 'B', 'A', 'B'],
        'y': [1, 2, 3, 4],
        'err': [0.1, 0.2, 0.1, 0.2],
        'pval': [0.01, 0.05, 0.2, 0.001]
    })
    chart = GroupedBarChart()
    fig, ax = chart.plot(df, x_col='x', y_col='y', group_col='group', error_col='err', pval_col='pval')
    assert fig is not None
    assert ax is not None

def test_stacked_bar_chart():
    df = pd.DataFrame({
        'x': ['A', 'B', 'C'],
        'y1': [1, 2, 3],
        'y2': [4, 5, 6]
    })
    chart = StackedBarChart()
    fig, ax = chart.plot(df, x_col='x', y_cols=['y1', 'y2'])
    assert fig is not None
    assert ax is not None

def test_time_series_plot():
    df = pd.DataFrame({
        'date': pd.date_range('2020-01-01', periods=10),
        'y1': np.random.randn(10),
        'y2': np.random.randn(10)
    })
    chart = TimeSeriesPlot()
    fig, ax = chart.plot(df, x_col='date', y_cols=['y1', 'y2'])
    assert fig is not None
    assert ax is not None

def test_irf_plot():
    df = pd.DataFrame({
        'var': ['V1']*5 + ['V2']*5,
        'horizon': list(range(5)) * 2,
        'coef': np.random.randn(10),
        'ci_lower': np.random.randn(10) - 1,
        'ci_upper': np.random.randn(10) + 1
    })
    chart = IRFPlot()
    fig, axes = chart.plot(df, horizon_col='horizon', coef_col='coef', ci_lower_col='ci_lower', ci_upper_col='ci_upper', group_col='var')
    assert fig is not None
    assert len(axes) > 0

def test_faceted_time_series():
    df = pd.DataFrame({
        'date': list(pd.date_range('2020-01-01', periods=5)) * 2,
        'entity': ['E1']*5 + ['E2']*5,
        'val': np.random.randn(10)
    })
    chart = FacetedTimeSeries()
    fig, axes = chart.plot(df, date_col='date', value_cols=['val'], facet_col='entity')
    assert fig is not None
    assert len(axes) > 0

def test_binned_scatter_plot():
    df = pd.DataFrame({
        'x': np.random.randn(100),
        'y': np.random.randn(100)
    })
    chart = BinnedScatterPlot()
    fig, ax = chart.plot(df, x_col='x', y_col='y', n_bins=5)
    assert fig is not None
    assert ax is not None

def test_scatter_with_regression():
    df = pd.DataFrame({
        'x': np.random.randn(50),
        'y': np.random.randn(50)
    })
    chart = ScatterWithRegression()
    fig, ax = chart.plot(df, x_col='x', y_col='y')
    assert fig is not None
    assert ax is not None

def test_correlation_heatmap():
    df = pd.DataFrame(np.random.randn(10, 4), columns=['A', 'B', 'C', 'D'])
    chart = CorrelationHeatmap()
    fig, ax = chart.plot(df)
    assert fig is not None
    assert ax is not None
