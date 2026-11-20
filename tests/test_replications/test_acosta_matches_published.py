import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from stats_transformer import Pipeline

ROOT = Path(__file__).resolve().parents[2]
LIB_PARQUET = ROOT / "data/examples/academic/acosta_brennan_jacobson_2024/sofr_surprises.parquet.gzip"
PUBLISHED_CSV = ROOT / "tests/external_benchmarks/dataverse/acosta2024_sofr_surprises.csv"

COL_MAP = {"GSS_target": "gss_target", "GSS_path": "gss_path", "NS": "ns"}


def _published():
    pub = pd.read_csv(PUBLISHED_CSV)
    pub["date"] = pd.to_datetime(pub["date"])
    return pub


def _library_flat():
    return pd.read_parquet(LIB_PARQUET).reset_index()


def test_acosta_sofr_surprises_match_published():
    pub = _published()
    lib = _library_flat()
    assert len(pub) == len(lib) == 234
    assert (pd.to_datetime(lib["date"]).to_numpy() == pub["date"].to_numpy()).all()
    for pub_col, lib_col in COL_MAP.items():
        # The committed reference CSV round-trips at 17 significant figures, so
        # tolerate only last-bit float64 noise; any real drift is far larger.
        np.testing.assert_allclose(lib[lib_col].to_numpy(), pub[pub_col].to_numpy(), rtol=1e-10, atol=1e-14)


def _run_pipeline(tmp_path):
    cfg = {
        "data": {
            "raw_data_file": str(LIB_PARQUET),
            "featurization": {"date_column": "date", "entity_column": "country", "transformations": ["lag1", "zscore"]},
        },
        "model": {
            "model_type": "var",
            "target_variables": ["gss_target", "gss_path", "ns"],
            "maxlags": 2,
            "summary_output_path": str(tmp_path / "summary.json"),
        },
        "output": {"root": str(tmp_path / "runs")},
    }
    p = tmp_path / "params.yaml"
    p.write_text(yaml.dump(cfg))
    return Pipeline(params_path=str(p))


def test_acosta_featurization_matches_pandas(tmp_path):
    pl = _run_pipeline(tmp_path)
    pl.run(stage="features")
    t = pl.transformed_data
    ref = _library_flat().sort_values("date").reset_index(drop=True)
    for col in ["gss_target", "gss_path", "ns"]:
        exp_lag = ref[col].shift(1)
        got_lag = t[f"{col}_lag1"]
        np.testing.assert_allclose(got_lag[~exp_lag.isna()], exp_lag[~exp_lag.isna()])
        np.testing.assert_allclose(t[f"{col}_zscore"], (ref[col] - ref[col].mean()) / ref[col].std())


def test_acosta_var_matches_statsmodels(tmp_path):
    from statsmodels.tsa.api import VAR

    pl = _run_pipeline(tmp_path)
    pl.run(stage="model")
    d = json.loads((tmp_path / "summary.json").read_text())

    ref = VAR(_library_flat()[["gss_target", "gss_path", "ns"]]).fit(2)
    for eq in ["gss_target", "gss_path", "ns"]:
        for term in ["L1.gss_target", "L2.ns", "const"]:
            got = d["coefficients"][eq][term]["value"]
            want = ref.params.loc[term, eq]
            assert abs(got - want) < 1e-10, (eq, term, got, want)
