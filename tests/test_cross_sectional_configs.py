"""Cross-sectional configs (no entity column) must run the features and model stages.

Regression tests for the paper's Mincer example (docs/wrapup triage P0):
FeatureEngineer used to demand [None, 'date'] when a config declared neither
entity nor date columns.
"""

import glob
import json
import os

import pandas as pd
import yaml
import pytest
from stats_transformer import FeatureEngineer, Pipeline

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

ALL_CONFIGS = sorted(glob.glob(os.path.join(REPO_ROOT, "references", "configs", "**", "*.yaml"), recursive=True))


def _prepare_config(tmp_path, rel_path):
    """Copy a repo config into tmp_path, resolving input file paths to the repo."""
    with open(os.path.join(REPO_ROOT, rel_path)) as f:
        doc = yaml.safe_load(f)
    doc = _redirect_inputs(doc)
    config_path = tmp_path / "params.yaml"
    config_path.write_text(yaml.dump(doc))
    return config_path


def _redirect_inputs(obj):
    if isinstance(obj, dict):
        return {k: _redirect_inputs(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_redirect_inputs(x) for x in obj]
    if isinstance(obj, str) and not obj.startswith("/") and os.path.isfile(os.path.join(REPO_ROOT, obj)):
        return os.path.join(REPO_ROOT, obj)
    return obj


def test_mincer_wage_features_and_model(tmp_path, monkeypatch):
    config_path = _prepare_config(tmp_path, "references/configs/regression/mincer_wage.yaml")
    monkeypatch.chdir(tmp_path)  # run_model writes ./reports relative to CWD
    pipeline = Pipeline(params_path=str(config_path))
    transformed = pipeline.run_features()
    results = pipeline.run_model()
    assert transformed.shape[1] > 4
    coefficients = results["coefficients"]
    assert len(coefficients) == 5  # const + 4 regressors
    assert coefficients["educ"]["value"] > 0
    assert coefficients["expersq"]["value"] < 0
    assert results["metrics"]["num_observations"] == 526


def test_mincer_wage_full_run(tmp_path, monkeypatch):
    """Configs without data.datasets must run the full pipeline off raw_data_file.

    Regression: a full run always called run_data(), which crashed with
    IndexError on resampled[0] when no datasets were configured.
    """
    config_path = _prepare_config(tmp_path, "references/configs/regression/mincer_wage.yaml")
    monkeypatch.chdir(tmp_path)  # all outputs land in tmp_path relative to CWD
    pipeline = Pipeline(params_path=str(config_path))
    transformed = pipeline.run()
    assert isinstance(transformed, pd.DataFrame)
    assert transformed.shape[1] > 4
    # Model summary in the per-run folder, manifest lists exactly what ran.
    run_dir = next(iter((tmp_path / "data" / "pipeline" / "params").iterdir()))
    assert (run_dir / "03_model" / "model_summary.json").is_file()
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["stages_run"] == ["features", "model", "reporting"]


def test_run_data_stage_without_datasets_raises(tmp_path, monkeypatch):
    config_path = _prepare_config(tmp_path, "references/configs/regression/mincer_wage.yaml")
    monkeypatch.chdir(tmp_path)
    pipeline = Pipeline(params_path=str(config_path))
    with pytest.raises(ValueError, match="data.datasets"):
        pipeline.run(stage="data")
    # The failed stage leaves nothing behind: no run folder, no output file.
    assert not (tmp_path / "data").exists()
    assert not list(tmp_path.rglob("manifest.json"))


def test_transform_whole_frame_without_entity():
    # Cross-sectional frame in row order: lag/changepct operate on the whole frame.
    fe = FeatureEngineer(transformations=["lag1", "changepct"], entity_column=None)
    import pandas as pd
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0]})
    out = fe.fit_transform(df)
    assert out["x_lag1"].iloc[0] != out["x_lag1"].iloc[0]  # NaN from shift
    assert out["x_lag1"][1:].tolist() == [1.0, 2.0, 3.0]
    changepct = out["x_changepct"].tolist()
    assert changepct[0] != changepct[0]  # NaN
    assert changepct[1:] == pytest.approx([1.0, 0.5, 1.0 / 3.0])


def test_country_list_without_entity_raises():
    import pandas as pd
    fe = FeatureEngineer(transformations=["log"], entity_column=None, country_list=["a"], data_columns=["x"])
    with pytest.raises(ValueError, match="entity_column"):
        fe.fit_transform(pd.DataFrame({"x": [1.0, 2.0]}))


def test_start_date_without_date_column_raises():
    import pandas as pd
    fe = FeatureEngineer(transformations=[], entity_column=None, date_column=None, start_date="2000-01-01", data_columns=["x"])
    with pytest.raises(ValueError, match="date_column"):
        fe.fit_transform(pd.DataFrame({"x": [1.0, 2.0]}))


def _single_series_feature_engineer():
    return FeatureEngineer(entity_column=None, date_column="date")


def test_resample_downsample_without_entity():
    # Monthly single series -> annual mean; no 'country' column invented.
    import pandas as pd
    fe = _single_series_feature_engineer()
    df = pd.DataFrame({"date": pd.date_range("2020-01-01", periods=24, freq="MS"), "x": range(24)})
    out = fe.resample_dataset(df, {"date_column": "date", "source_period": "monthly", "resample": {"target_period": "annual", "agg": "mean"}})
    assert "country" not in out.columns
    assert list(out.columns) == ["date", "x_Mmean"]
    assert len(out) == 2  # 2020: months 0-11 (mean 5.5), 2021: months 12-23 (mean 17.5)
    assert out["x_Mmean"].tolist() == pytest.approx([5.5, 17.5])


def test_resample_upsample_without_entity():
    # Annual single series -> monthly, forward-filled; no 'country' column invented.
    import pandas as pd
    fe = _single_series_feature_engineer()
    df = pd.DataFrame({"date": ["2020", "2021"], "x": [10.0, 20.0]})
    out = fe.resample_dataset(df, {"date_column": "date", "source_period": "annual", "resample": {"target_period": "monthly", "agg": "ffill"}})
    assert "country" not in out.columns
    # asfreq spans only up to the last observed date (2020-01 + 12 months to 2021-01).
    assert len(out) == 13
    assert out["x_Yffill"].tolist() == [10.0] * 12 + [20.0]


def test_resample_no_duplicate_columns_for_numeric_entity_id():
    # A numeric entity column is also in numeric_cols; make sure it is not selected twice.
    import pandas as pd
    fe = FeatureEngineer(entity_column="firm", date_column="year")
    df = pd.DataFrame({"firm": [1, 1, 2, 2], "year": [2020, 2021, 2020, 2021], "inv": [1.0, 2.0, 3.0, 4.0]})
    out = fe.resample_dataset(df, {"entity_column": "firm", "date_column": "year", "source_period": "annual", "resample": {"target_period": "annual"}})
    assert len(out.columns) == len(set(out.columns))
    assert [c for c in out.columns if c == "country"] == ["country"]


XFAIL_CONFIGS = {
    "timeseries/identification/structural_restrictions.yaml": "Restrictions-only SVAR input file, not a standalone params.yaml (no data:/model: sections)",
    "timeseries/reduced_form/local_projections.yaml": "Config declares date_column: date but sofr_surprises.parquet.gzip has no date column",
}


def _config_params():
    return [
        pytest.param(f"references/configs/{rel}", marks=pytest.mark.xfail(strict=True, reason=reason))
        for rel, reason in XFAIL_CONFIGS.items()
    ] + [
        pytest.param(os.path.relpath(c, REPO_ROOT), id=os.path.relpath(c, REPO_ROOT))
        for c in ALL_CONFIGS
        if os.path.relpath(c, os.path.join(REPO_ROOT, "references", "configs")) not in XFAIL_CONFIGS
    ]

@pytest.mark.parametrize("config", _config_params())
def test_config_features_and_model_smoke(config, tmp_path, monkeypatch):
    """Every YAML under references/configs runs the features and model stages.

    Input paths resolve to the repo's data files; everything the pipeline
    writes (features CSV, model summary, ./reports) lands in tmp_path.
    """
    config_path = _prepare_config(tmp_path, config)
    monkeypatch.chdir(tmp_path)
    pipeline = Pipeline(params_path=str(config_path))
    pipeline.run_features()
    pipeline.run_model()


@pytest.mark.parametrize("config", _config_params())
def test_config_full_run_smoke(config, tmp_path, monkeypatch):
    """Every YAML under references/configs also runs the full pipeline (stage=None)."""
    config_path = _prepare_config(tmp_path, config)
    monkeypatch.chdir(tmp_path)
    Pipeline(params_path=str(config_path)).run()
