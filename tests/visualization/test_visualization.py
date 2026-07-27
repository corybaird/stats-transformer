import pytest
import pandas as pd
import numpy as np
import os
import shutil
from stats_transformer.models.regression.robust_ols import RobustOLSModel
from stats_transformer.visualization.models.regression_viz import RegressionVisualizer
from stats_transformer.visualization.eda.data_viz import DataVisualizer
from stats_transformer.visualization.models.model_viz import ModelVisualizer

def test_regression_visualizer():
    # 1. Create a model and fit it to get metadata
    df = pd.DataFrame({
        "y": [1, 2, 3, 4, 5],
        "x1": [2, 4, 6, 8, 10],
        "x2": [1, 1, 2, 2, 3]
    })
    
    model = RobustOLSModel(
        target="y",
        independent_variables=["x1", "x2"]
    )
    model.fit(df)
    metadata = model.get_model_metadata()
    
    # 2. Test visualizer
    viz_dir = "data/temp/visualizations"
    os.makedirs(viz_dir, exist_ok=True)
    
    viz = RegressionVisualizer(output_dir=viz_dir)
    results_with_files = viz.visualize_from_json(model_summary=metadata, display_only=False)
    
    assert "coefficient_plot" in results_with_files
    assert "model_summary" in results_with_files
    
    # Verify files exist
    assert os.path.exists(results_with_files["coefficient_plot"])
    assert os.path.exists(results_with_files["model_summary"])

    # Clean up
    if os.path.exists(viz_dir):
        shutil.rmtree(viz_dir)

def test_eda_visualizer():
    from stats_transformer.visualization.eda.eda import EDAVisualizer
    
    df = pd.DataFrame({
        "a": [1, 2, np.nan, 4],
        "b": [10, 11, 12, 13],
        "c": ["x", "y", "z", "w"]
    })
    
    viz_dir = "data/temp/eda_viz"
    os.makedirs(viz_dir, exist_ok=True)
    
    viz = EDAVisualizer(output_dir=viz_dir)
    results = viz.create_visualization(df, display_only=False)
    
    assert "missingness" in results
    assert "distributions" in results
    assert os.path.exists(results["missingness"])
    assert os.path.exists(results["distributions"])
    
    # Clean up
    if os.path.exists(viz_dir):
        shutil.rmtree(viz_dir)

def test_data_visualizer():
    df = pd.DataFrame({
        "x": np.random.randn(100),
        "y": np.random.randn(100),
        "cat": np.random.choice(["A", "B"], 100),
        "date": pd.date_range("2020-01-01", periods=100)
    })
    
    viz_dir = "data/temp/data_viz"
    os.makedirs(viz_dir, exist_ok=True)
    viz = DataVisualizer(output_dir=viz_dir)
    
    # Test histogram
    hist_files = viz.create_visualization(df, feature_list=["x"], viz_type="histogram", display_only=False)
    assert len(hist_files) == 1
    assert os.path.exists(hist_files[0])
    
    # Test correlation
    corr_file = viz.create_visualization(df, feature_list=["x", "y"], viz_type="correlation", display_only=False)
    assert os.path.exists(corr_file)
    
    # Test time series
    ts_file = viz.create_time_series_plot(df, date_column="date", display_only=False)
    assert os.path.exists(ts_file)
    
    # Clean up
    if os.path.exists(viz_dir):
        shutil.rmtree(viz_dir)

def test_model_visualizer():
    model_summary = {
        "coefficients": {
            "const": {"value": 1.0, "std_err": 0.1},
            "var1": {"value": 2.5, "std_err": 0.5},
            "var2": {"value": -1.2, "std_err": 0.3}
        }
    }
    
    viz_dir = "data/temp/model_viz"
    os.makedirs(viz_dir, exist_ok=True)
    viz = ModelVisualizer(output_dir=viz_dir)
    
    coef_plot = viz.create_visualization(model_summary, visualization_type="coefficient", display_only=False)
    assert os.path.exists(coef_plot)
    
    # Clean up
    if os.path.exists(viz_dir):
        shutil.rmtree(viz_dir)
