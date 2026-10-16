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

def test_unsorted_dates_match_sorted_features():
    # Shuffled dates must give the same lag/diff/changepct values as sorted input.
    fe = FeatureEngineer(
        params_path=None,
        transformations=["changeraw", "changepct", "lag1"],
        entity_column="country",
        date_column="date",
        data_columns=["val"],
        verbose=False
    )
    shuffled = pd.DataFrame({
        "country": ["A"] * 4,
        "date": ["2020-03-01", "2020-01-01", "2020-04-01", "2020-02-01"],
        "val": [30.0, 10.0, 40.0, 20.0],
    })
    sorted_df = shuffled.sort_values("date").reset_index(drop=True)
    res_shuffled = fe.transform(shuffled).sort_values("date").reset_index(drop=True)
    res_sorted = fe.transform(sorted_df)
    for col in ["val_changeraw", "val_changepct", "val_lag1"]:
        assert res_shuffled[col].equals(res_sorted[col])

def test_no_entity_path_sorts_by_date():
    fe = FeatureEngineer(
        params_path=None,
        transformations=["changeraw"],
        entity_column=None,
        date_column="date",
        data_columns=["val"],
        verbose=False
    )
    df = pd.DataFrame({
        "date": ["2020-03-01", "2020-01-01", "2020-02-01"],
        "val": [30.0, 10.0, 20.0],
    })
    out = fe.transform(df)
    assert np.isnan(out["val_changeraw"].iloc[0])
    assert out["val_changeraw"].iloc[1:].tolist() == [10.0, 10.0]

def test_failing_transformation_raises():
    fe = FeatureEngineer(
        params_path=None,
        transformations=["log"],
        entity_column="country",
        date_column="date",
        data_columns=["val"],
        verbose=False
    )
    df = pd.DataFrame({
        "country": ["A", "A"],
        "date": ["2020-01-01", "2021-01-01"],
        "val": [10.0, -5.0],
    })
    with pytest.raises(ValueError, match="column 'val' in entity 'A'"):
        fe.transform(df)


@pytest.mark.parametrize("suffix", [".csv", ".parquet"])
def test_save_features_writes_output_path(tmp_path, suffix):
    df = pd.DataFrame({"date": pd.date_range("2020-01-01", periods=3, freq="MS"), "x": [1.0, 2.0, 3.0]})
    fe = FeatureEngineer(params_path=None, transformations=[], date_column="date", data_columns=["x"], verbose=False)
    out = tmp_path / "nested" / f"features{suffix}"
    fe.save_features(df, str(out))
    back = pd.read_csv(out) if suffix == ".csv" else pd.read_parquet(out)
    assert back["x"].tolist() == [1.0, 2.0, 3.0]
