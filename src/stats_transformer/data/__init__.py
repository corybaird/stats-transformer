"""Data utilities for the stats-transformer library."""

import pandas as pd
from importlib.resources import files


def load_sample_data():
    """Load the sample macroeconomic dataset.
    
    Returns:
        DataFrame with columns: country, date, inflation, gdp
    
    Example:
        >>> from stats_transformer.data import load_sample_data
        >>> df = load_sample_data()
        >>> print(df.head())
    """
    try:
        data_path = files("stats_transformer.data").joinpath("macrodb_gdp_inflation.parquet")
        return pd.read_parquet(data_path)
    except Exception as e:
        # Fallback for development or testing
        import os
        fallback_path = os.path.join(os.path.dirname(__file__), "macrodb_gdp_inflation.parquet")
        if os.path.exists(fallback_path):
            return pd.read_parquet(fallback_path)
        else:
            raise FileNotFoundError(
                "Sample data file not found. Make sure the package was installed correctly."
            ) from e


def get_sample_data_description():
    """Get description of the sample dataset.
    
    Returns:
        Dictionary with dataset description
    """
    return {
        "name": "macrodb_gdp_inflation",
        "description": "Macroeconomic dataset with GDP and inflation data for multiple countries",
        "columns": {
            "country": "Country code (ISO 3-letter)",
            "date": "Year of observation",
            "inflation": "Inflation rate (annual %)",
            "gdp": "GDP per capita (current US$)"
        },
        "shape": (11490, 4),
        "time_period": "Annual data",
        "source": "Derived from World Bank and IMF data",
        "notes": "Includes data for multiple countries across different time periods"
    }
