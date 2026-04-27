import pytest
import pandas as pd
import numpy as np

# Importing directly from the module to avoid global package import failures in the project
from stats_transformer.featurization.feature_engineering import FeatureEngineer

def test_feature_engineering_transform():
    fe = FeatureEngineer(
        params_path=None,
        transformations=["changepct", "log", "lag1"],
        entity_column="country",
        date_column="date",
        period="annual",
        data_columns=["val"],
        verbose=False
    )
    
    df = pd.DataFrame({
        "country": ["USA", "USA", "USA"],
        "date": ["2020-01-01", "2021-01-01", "2022-01-01"],
        "val": [10.0, 11.0, 12.1]
    })
    
    res = fe.transform(df)
    
    assert "val_changepct" in res.columns
    assert "val_log" in res.columns
    assert "val_lag1" in res.columns
    
    # Check log transformation values
    assert np.isclose(res["val_log"].iloc[0], np.log(10.0))
    
    # Check lag value (lag1 of index 1 should be index 0's value)
    assert res["val_lag1"].iloc[1] == 10.0

def test_feature_engineering_truncate_data():
    fe = FeatureEngineer(
        params_path=None,
        entity_column="country",
        country_list=["USA", "CAN"],
        verbose=False
    )
    
    df = pd.DataFrame({
        "country": ["USA", "CAN", "MEX"],
        "val": [1, 2, 3]
    })
    
    truncated = fe.truncate_data(df)
    assert len(truncated) == 2
    assert "MEX" not in truncated["country"].values
