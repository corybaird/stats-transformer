import pytest
import pandas as pd
import numpy as np
import os
from stats_transformer.models.regression.robust_ols import RobustOLSModel
from stats_transformer.visualization.models.regression_viz import RegressionVisualizer

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
        import shutil
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
        import shutil
        shutil.rmtree(viz_dir)
