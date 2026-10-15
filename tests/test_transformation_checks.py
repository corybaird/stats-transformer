"""Transform checks backing paper Table 2: FeatureEngineer vs direct pandas reference.

Rows 1-2: Bauer & Swanson (2023) loadmacrodata.m (log-diff of nonfarm payrolls,
one-period lag of unemployment). Rows 3-4: Bauer, Bernanke & Milstein (2023)
GSW yields (daily diff of SVENY01, daily pct change of SVENY10).
Row 5 (Nakamura-Steinsson NY1 changeraw) lives in test_academic_replications.py.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from stats_transformer.featurization.feature_engineering import FeatureEngineer

DATA_DIR = Path(__file__).parent.parent / "data" / "examples" / "academic"


def _engineer(transformations, data_columns, date_column, period):
    return FeatureEngineer(
        params_path=None,
        transformations=transformations,
        entity_column="country",
        date_column=date_column,
        period=period,
        data_columns=data_columns,
        verbose=False,
    )


def _load_bauer_swanson(filename):
    df = pd.read_csv(
        DATA_DIR / "bauer_swanson_2023" / filename,
        sep=r"\s+",
        header=None,
        names=["year", "month", "value"],
    )
    df["date"] = pd.to_datetime(df["year"].astype(str) + "-" + df["month"].astype(str) + "-01")
    df["country"] = "USA"
    return df


def _load_bbm():
    df = pd.read_csv(DATA_DIR / "bbm_2023" / "feds_subset.csv", skiprows=9)
    df["Date"] = pd.to_datetime(df["Date"])
    df["country"] = "USA"
    for col in ["SVENY01", "SVENY10"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(subset=["SVENY01", "SVENY10"]).sort_values("Date").reset_index(drop=True)


def _compare(result, column, reference):
    """Merge FeatureEngineer output with the pandas reference on date, drop warm-up NaNs."""
    key = "Date" if "Date" in reference.columns else "date"
    merged = pd.merge(result.dropna(subset=[column]), reference, on=key)
    errors = (merged[column] - merged["reference"]).abs()
    return len(merged), float(errors.max())


def test_bauer_swanson_nfp_log_diff():
    """Nonfarm payrolls: FeatureEngineer log then changeraw equals np-log diff (n = 473)."""
    df = _load_bauer_swanson("NonfarmPayrolls.txt")
    reference = pd.DataFrame({"date": df["date"], "reference": np.log(df["value"]).diff()})

    df_log = _engineer(["log"], ["value"], "date", "monthly").fit_transform(df)
    result = _engineer(["changeraw"], ["value_log"], "date", "monthly").fit_transform(df_log)
    n, max_error = _compare(result, "value_log_changeraw", reference)
    assert n == 473
    assert max_error < 1e-10


def test_bauer_swanson_unemployment_lag():
    """Unemployment: FeatureEngineer lag1 equals shift(1) (n = 473)."""
    df = _load_bauer_swanson("Unemployment.txt")
    reference = pd.DataFrame({"date": df["date"], "reference": df["value"].shift(1)})

    result = _engineer(["lag1"], ["value"], "date", "monthly").fit_transform(df)
    n, max_error = _compare(result, "value_lag1", reference)
    assert n == 473
    assert max_error < 1e-10


@pytest.mark.parametrize("column,transform,reference_fn", [
    ("SVENY01", "changeraw", lambda df: df["SVENY01"].diff()),
    ("SVENY10", "changepct", lambda df: df["SVENY10"].pct_change(fill_method=None)),
])
def test_bbm_yield_transforms(column, transform, reference_fn):
    """BBM GSW yields: FeatureEngineer changeraw/changepct equal diff/pct_change (n = 958)."""
    df = _load_bbm()
    reference = pd.DataFrame({"Date": df["Date"], "reference": reference_fn(df)})

    result = _engineer([transform], [column], "Date", "daily").fit_transform(df)
    n, max_error = _compare(result, f"{column}_{transform}", reference)
    assert n == 958
    assert max_error < 1e-10