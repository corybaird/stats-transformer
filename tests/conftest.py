import pandas as pd
import pytest
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent


@pytest.fixture
def project_root() -> Path:
    return PROJECT_ROOT


@pytest.fixture
def test_df() -> pd.DataFrame:
    return pd.read_csv(PROJECT_ROOT / "tests" / "data" / "test_data.csv")


@pytest.fixture
def panel_df() -> pd.DataFrame:
    return pd.read_csv(PROJECT_ROOT / "tests" / "data" / "test_data.csv")


@pytest.fixture
def macro_ts_df() -> pd.DataFrame:
    return pd.read_parquet(PROJECT_ROOT / "tests" / "data" / "macrodb_gdp_inflation.parquet")
