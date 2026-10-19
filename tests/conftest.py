import numpy as np
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
def var1_data():
    """Factory for the stationary bivariate VAR(1) sample used by the model/reporting tests."""

    def make(observations: int = 120, seed: int = 42) -> pd.DataFrame:
        generator = np.random.default_rng(seed)
        shocks = generator.normal(size=(observations, 2))
        values = np.zeros((observations, 2))
        transition = np.array([[0.55, 0.10], [0.15, 0.40]])
        for index in range(1, observations):
            values[index] = transition @ values[index - 1] + shocks[index]
        return pd.DataFrame(values, columns=["output", "inflation"])

    return make
