"""Data utilities for the stats-transformer library."""

import pandas as pd
from importlib.resources import files


EXAMPLE_DATASETS = {
    "macrodb_gdp_inflation": {
        "filename": "macrodb_gdp_inflation.parquet",
        "description": "Macroeconomic dataset with GDP and inflation data for multiple countries",
        "columns": {
            "country": "Country code (ISO 3-letter)",
            "date": "Year of observation",
            "inflation": "Inflation rate (annual %)",
            "gdp": "GDP per capita (current US$)",
        },
        "shape": (11490, 4),
        "time_period": "Annual data",
        "source": "Derived from World Bank and IMF data",
        "notes": "Includes data for multiple countries across different time periods",
    }
}


def list_examples():
    """List packaged example datasets."""
    return sorted(EXAMPLE_DATASETS)


def describe_example(name="macrodb_gdp_inflation"):
    """Return metadata for a packaged example dataset."""
    if name not in EXAMPLE_DATASETS:
        available = ", ".join(list_examples())
        raise ValueError(f"Unknown example dataset '{name}'. Available examples: {available}")

    metadata = EXAMPLE_DATASETS[name].copy()
    metadata.pop("filename", None)
    metadata["name"] = name
    return metadata


def load_example(name="macrodb_gdp_inflation"):
    """Load a packaged example dataset."""
    if name not in EXAMPLE_DATASETS:
        available = ", ".join(list_examples())
        raise ValueError(f"Unknown example dataset '{name}'. Available examples: {available}")

    filename = EXAMPLE_DATASETS[name]["filename"]
    try:
        data_path = files("stats_transformer.data").joinpath(filename)
        return pd.read_parquet(data_path)
    except Exception as e:
        # Fallback for development or testing
        import os
        fallback_path = os.path.join(os.path.dirname(__file__), filename)
        if os.path.exists(fallback_path):
            return pd.read_parquet(fallback_path)
        else:
            raise FileNotFoundError(
                f"Example data file '{filename}' not found. Make sure the package was installed correctly."
            ) from e


def load_sample_data():
    """Load the default sample macroeconomic dataset.

    Returns:
        DataFrame with columns: country, date, inflation, gdp

    Example:
        >>> from stats_transformer.data import load_sample_data
        >>> df = load_sample_data()
        >>> print(df.head())
    """
    return load_example("macrodb_gdp_inflation")


def get_sample_data_description():
    """Get description of the sample dataset.

    Returns:
        Dictionary with dataset description
    """
    return describe_example("macrodb_gdp_inflation")


__all__ = [
    "EXAMPLE_DATASETS",
    "describe_example",
    "get_sample_data_description",
    "list_examples",
    "load_example",
    "load_sample_data",
]
