import pytest
import pandas as pd
import os
import shutil
from stats_transformer.pipeline import Pipeline

def test_pipeline_run_regression():
    # Setup
    config_path = "references/configs/test_pipeline.yaml"
    temp_dir = "data/temp"
    os.makedirs(temp_dir, exist_ok=True)
    
    pipeline = Pipeline(params_path=config_path)
    
    # Run the regression stage
    results = pipeline.run(stage="regression")
    
    assert results is not None
    assert "metrics" in results
    assert "r_squared" in results["metrics"]
    assert os.path.exists("data/temp/test_summary.json")
    
    # Clean up
    if os.path.exists("data/temp/test_summary.json"):
        os.remove("data/temp/test_summary.json")

def test_pipeline_fit_transform():
    config_path = "references/configs/test_pipeline.yaml"
    pipeline = Pipeline(params_path=config_path)
    
    df = pd.read_csv("tests/data/test_data.csv")
    transformed = pipeline.fit_transform(df)
    
    assert transformed is not None
    assert "y" in transformed.columns
    assert "x1" in transformed.columns
    assert pipeline.model_results is not None
    assert "metrics" in pipeline.model_results
