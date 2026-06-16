import pytest
import pandas as pd
from unittest.mock import patch

# Importing directly from the module to avoid global package import failures in the project
from stats_transformer.featurization.data_merger import DataMerger

@patch("stats_transformer.featurization.data_merger.DataMerger._load_config", return_value={})
def test_data_merger_merge(mock_load_config):
    merger = DataMerger("dummy.yaml")
    
    df1 = pd.DataFrame({
        "country": ["USA", "GBR"],
        "date": ["2020-01-01", "2020-01-01"],
        "val1": [1, 2]
    })
    
    df2 = pd.DataFrame({
        "country": ["USA", "GBR"],
        "date": ["2020-01-01", "2020-01-01"],
        "val2": [3, 4]
    })
    
    merged = merger.merge(df1, df2, on=["country", "date"])
    assert len(merged) == 2
    assert "val1" in merged.columns
    assert "val2" in merged.columns

@patch("stats_transformer.featurization.data_merger.DataMerger._load_config", return_value={})
def test_data_merger_standardize_entity(mock_load_config):
    merger = DataMerger("dummy.yaml")
    
    df = pd.DataFrame({
        "iso2": ["US", "GB", "AF"],
        "val": [1, 2, 3]
    })
    
    # Needs to match ISO2 to ISO3 mapping present in dict_country_converter
    with patch("stats_transformer.featurization.data_merger.ISO2_TO_ISO3", {"US": "USA", "GB": "GBR", "AF": "AFG"}):
        standardized = merger.standardize_entity(df, "iso2")
        
        assert "country" in standardized.columns
        assert standardized["country"].iloc[0] == "USA"
        assert standardized["country"].iloc[1] == "GBR"
