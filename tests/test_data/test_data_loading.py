import pytest
import pandas as pd
from pathlib import Path

# Provide paths to some sample data for robust testing
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"

@pytest.fixture
def annual_panel_path():
    return DATA_DIR / "panel" / "annual" / "jst_panel.parquet"

@pytest.fixture
def sample_raw_data_path():
    return DATA_DIR / "raw" / "monthly" / "imf_ifs_monthly.parquet"

def test_load_annual_panel(annual_panel_path):
    if not annual_panel_path.exists():
        pytest.skip(f"Dataset not found at {annual_panel_path}")
    
    df = pd.read_parquet(annual_panel_path)
    assert not df.empty, "Annual panel dataset should not be empty"
    # Basic check to see if it acts like a panel (e.g. contains common columns if we know them, but right now we just check it loads)

def test_stats_transformer_import():
    try:
        import stats_transformer
    except ImportError:
        pytest.fail("Failed to import stats_transformer package")
