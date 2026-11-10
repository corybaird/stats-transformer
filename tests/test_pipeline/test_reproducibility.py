import json

import pandas as pd
import yaml

from stats_transformer.pipeline import Pipeline


def _write_config(tmp_path):
    raw = tmp_path / "raw.csv"
    n = 20
    x1 = [i for i in range(n)]
    x2 = [i * i for i in range(n)]  # non-collinear with x1
    y = [2.0 * a + 1.5 * b + 0.5 for a, b in zip(x1, x2)]
    pd.DataFrame({
        "date": pd.date_range("2020-01-01", periods=n, freq="MS"),
        "x1": x1,
        "x2": x2,
        "y": y,
    }).to_csv(raw, index=False)
    config = {
        "data": {"raw_data_file": str(raw), "featurization": {"date_column": "date"}},
        "model": {
            "model_type": "ols",
            "target_variable": "y",
            "independent_variables": ["x1", "x2"],
            "summary_output_path": str(tmp_path / "summary.json"),
        },
        "output": {"root": str(tmp_path / "out")},
    }
    path = tmp_path / "params.yaml"
    path.write_text(yaml.dump(config))
    return path


def _coefs(tmp_path):
    return json.loads((tmp_path / "summary.json").read_text())["coefficients"]


def test_rerun_produces_identical_coefficients(tmp_path):
    config = _write_config(tmp_path)
    runs = []
    for _ in range(2):
        Pipeline(params_path=str(config)).run(stage="model")
        runs.append(_coefs(tmp_path))
    assert runs[0] == runs[1]


def test_rerun_is_independent_of_leftover_merged_artifact(tmp_path, monkeypatch):
    # A stale shared merged parquet must not change a single-source config's result.
    monkeypatch.chdir(tmp_path)
    stale = tmp_path / "data" / "pipeline"
    stale.mkdir(parents=True)
    pd.DataFrame({"wrong_col": [1.0]}).to_parquet(stale / "resampled_merged.parquet")

    config = _write_config(tmp_path)
    Pipeline(params_path=str(config)).run(stage="model")
    coefs = _coefs(tmp_path)
    assert abs(coefs["x1"]["value"] - 2.0) < 1e-10
    assert abs(coefs["x2"]["value"] - 1.5) < 1e-10
