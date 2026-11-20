import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from stats_transformer import Pipeline

ROOT = Path(__file__).resolve().parents[2]
REF = ROOT / "tests/external_benchmarks/dataverse/brennan2024_shocks.csv"
SERIES = ["mp1", "ff4", "nsmethod_nsdata", "brwmethod_brwdata", "brwmethod_nsdata", "brwmethod_nsdataplustreasuries", "nsmethod_brwdata"]


def _published():
    df = pd.read_csv(REF)
    df.columns = [c.lower() for c in df.columns]
    df["date"] = pd.to_datetime(df["date"])
    return df


def test_brennan2024_shocks_featurization_matches_pandas(tmp_path):
    df = _published()
    data = tmp_path / "shocks.csv"
    df.to_csv(data, index=False)

    cfg = {
        "data": {"raw_data_file": str(data), "featurization": {"date_column": "date", "transformations": ["lag1", "zscore"]}},
        "model": {"model_type": "var", "target_variables": ["brwmethod_brwdata", "nsmethod_brwdata", "ff4"], "maxlags": 2, "summary_output_path": str(tmp_path / "summary.json")},
        "output": {"root": str(tmp_path / "runs")},
    }
    p = tmp_path / "params.yaml"
    p.write_text(yaml.dump(cfg))

    pl = Pipeline(params_path=str(p))
    pl.run(stage="features")
    t = pl.transformed_data
    ref = df.sort_values("date").reset_index(drop=True)
    # complete series have no NaNs; compare them strictly
    for col in ["brwmethod_brwdata", "nsmethod_brwdata"]:
        exp_lag = ref[col].shift(1)
        np.testing.assert_allclose(t[f"{col}_lag1"][~exp_lag.isna()], exp_lag[~exp_lag.isna()])
        np.testing.assert_allclose(t[f"{col}_zscore"], (ref[col] - ref[col].mean()) / ref[col].std())


def test_brennan2024_var_matches_statsmodels(tmp_path):
    from statsmodels.tsa.api import VAR

    df = _published()
    complete = df[["brwmethod_brwdata", "nsmethod_brwdata", "ff4"]].dropna().reset_index(drop=True)
    ref = VAR(complete).fit(2)

    data = tmp_path / "shocks.csv"
    df.to_csv(data, index=False)
    cfg = {
        "data": {"raw_data_file": str(data), "featurization": {"date_column": "date", "transformations": ["lag1", "zscore"]}},
        "model": {"model_type": "var", "target_variables": ["brwmethod_brwdata", "nsmethod_brwdata", "ff4"], "maxlags": 2, "summary_output_path": str(tmp_path / "summary.json")},
        "output": {"root": str(tmp_path / "runs")},
    }
    p = tmp_path / "params.yaml"
    p.write_text(yaml.dump(cfg))

    pl = Pipeline(params_path=str(p))
    pl.fit_transform(complete, fit_model=True)
    d = pl.model.get_model_metadata()
    for eq in ["brwmethod_brwdata", "nsmethod_brwdata", "ff4"]:
        for term in ["L1.brwmethod_brwdata", "const"]:
            assert abs(d["coefficients"][eq][term]["value"] - ref.params.loc[term, eq]) < 1e-10, (eq, term)
