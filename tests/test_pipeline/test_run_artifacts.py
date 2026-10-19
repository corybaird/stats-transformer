import json
import os

import numpy as np
import pandas as pd
import pytest
import yaml
from pathlib import Path

from stats_transformer.models.regression.panel import PanelRegressionModel
from stats_transformer.models.timeseries import LocalProjectionsModel, VARModel
from stats_transformer.pipeline import Pipeline


def _repo_root() -> Path:
    """Walk up from this file to the first directory holding pyproject.toml."""
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").exists():
            return parent
    raise RuntimeError(f"no pyproject.toml above {__file__}")


# Test paths are resolved from the repository root, never from the CWD.
REPO_ROOT = _repo_root()


def _write_dataset_config(tmp_path, datasets, model, wide=False):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir(exist_ok=True)
    dates = ["2020-01-01", "2021-01-01", "2022-01-01"]
    df1_cols = ["country", "date", "y", "x1"] if wide else ["country", "date", "y"]
    df1 = pd.DataFrame({"country": ["USA", "USA", "USA", "CAN", "CAN", "CAN"], "date": dates * 2, "y": [10.0, 11.0, 12.0, 9.0, 10.0, 11.0]})
    if wide:
        df1["x1"] = [2.0, 3.0, 4.0, 1.0, 2.0, 3.0]
    df1 = df1[df1_cols]
    df2 = pd.DataFrame({"country": ["USA", "USA", "USA", "CAN", "CAN", "CAN"], "date": dates * 2, "x1": [2.0, 3.0, 4.0, 1.0, 2.0, 3.0]})
    df1.to_csv(raw_dir / "y.csv", index=False)
    df2.to_csv(raw_dir / "x1.csv", index=False)
    # Explicit merge output_path inside tmp_path keeps canonical copies out of the repo;
    # the run folder must still receive its own 01_data/merged.parquet.
    data = {
        "datasets": [{"name": name, "path": str(raw_dir / f"{name}.csv"), "entity_column": "country", "date_column": "date", "source_period": "annual", "resample": {"target_period": "annual"}} for name in datasets],
        "featurization": {"entity_column": "country", "date_column": "date"},
        "merge": {"on": ["country", "date"], "how": "outer", "output_path": str(tmp_path / "merged.parquet")},
    }
    model["summary_output_path"] = str(tmp_path / "summary.json")
    config = {
        "output": {"root": str(tmp_path / "output_root")},
        "model": model,
        "data": data,
        "reporting": {"output_dir": str(tmp_path / "viz"), "regression": {"plots": []}},
    }
    config_path = tmp_path / "params.yaml"
    config_path.write_text(yaml.dump(config))
    return config_path


def _full_run(tmp_path, datasets=("y", "x1"), model=None, wide=False):
    if model is None:
        model = {"model_type": "ols", "target_variable": "y_Y", "independent_variables": ["x1_Y"]}
    config_path = _write_dataset_config(tmp_path, list(datasets), model, wide=wide)
    pipeline = Pipeline(params_path=str(config_path))
    pipeline.run()
    run_dir = next((tmp_path / "output_root" / "params").iterdir())
    return pipeline, run_dir


def test_multi_source_configs_produce_raw_files(tmp_path):
    _, run_dir = _full_run(tmp_path)
    raw_dir = run_dir / "01_data" / "raw"
    assert sorted(p.name for p in raw_dir.glob("*.parquet")) == ["x1.parquet", "y.parquet"]
    # Raw copies are pre-resample: no period suffix on the data columns.
    raw_y = pd.read_parquet(raw_dir / "y.parquet")
    assert list(raw_y.columns) == ["country", "date", "y"]


def test_single_source_config_produces_one_raw_file(tmp_path):
    # Same model as the multi-source runs, but the y dataset also carries x1, so one dataset suffices.
    _, run_dir = _full_run(tmp_path, datasets=("y",), wide=True)
    raw_dir = run_dir / "01_data" / "raw"
    assert [p.name for p in raw_dir.glob("*.parquet")] == ["y.parquet"]


def test_raw_source_name_is_sanitised(tmp_path):
    # A dataset name with spaces/slashes must yield a safe filename.
    config_path = _write_dataset_config(tmp_path, ["y"], {"model_type": "ols", "target_variable": "y_Y", "independent_variables": ["x1_Y"]}, wide=True)
    params = yaml.safe_load(config_path.read_text())
    params["data"]["datasets"][0]["name"] = "my/data set"
    config_path.write_text(yaml.dump(params))
    Pipeline(params_path=str(config_path)).run()
    run_dir = next((tmp_path / "output_root" / "params").iterdir())
    assert [p.name for p in (run_dir / "01_data" / "raw").glob("*.parquet")] == ["my_data_set.parquet"]


def test_merged_parquet_written_when_output_path_set(tmp_path):
    _, run_dir = _full_run(tmp_path)
    merged = pd.read_parquet(run_dir / "01_data" / "merged.parquet")
    # resample_dataset suffixes numeric columns with the resample period ("_Y" for annual)
    assert {"country", "date", "y_Y", "x1_Y"} <= set(merged.columns)
    # The canonical chaining copy is still written to the configured output_path.
    assert os.path.exists(tmp_path / "merged.parquet")


def test_model_data_is_used_columns_only_and_without_nan(tmp_path):
    _, run_dir = _full_run(tmp_path)
    model_data = pd.read_parquet(run_dir / "03_model" / "model_data.parquet")
    # required columns = regressor + target + entity (order from _get_required_columns)
    assert list(model_data.columns) == ["x1_Y", "y_Y", "country"]
    assert not model_data.isna().any().any()
    assert len(model_data) == 6


def test_manifest_lists_new_files_and_shapes(tmp_path):
    _, run_dir = _full_run(tmp_path)
    manifest = json.loads((run_dir / "manifest.json").read_text())
    files = [os.path.basename(f) for f in manifest["files"]]
    for name in ["y.parquet", "merged.parquet", "features.parquet", "model_data.parquet", "model.joblib", "model_summary.json"]:
        assert name in files, f"{name} missing from manifest file list"
    shapes = manifest["shapes"]
    assert shapes["raw"]["y"] == [6, 3]
    assert shapes["raw"]["x1"] == [6, 3]
    assert shapes["merged"] == [6, 4]
    assert shapes["features"] == [6, 4]
    assert shapes["model_data"] == [6, 3]


def test_model_joblib_roundtrips_ols(tmp_path, test_df):
    pipeline = Pipeline(entity_column="country", target="y", features=["x1", "x2"], transformations=[], model_type="ols")
    pipeline.fit_transform(test_df)
    model = pipeline.model

    path = tmp_path / "model.joblib"
    model.save(str(path))
    loaded = model.load(str(path))

    np.testing.assert_allclose(loaded.model.params, model.model.params)
    pd.testing.assert_series_equal(loaded.predict(test_df), model.predict(test_df))


def test_model_joblib_roundtrips_panel_ols(tmp_path):
    df = pd.read_csv(REPO_ROOT / "data" / "examples" / "regression" / "grunfeld.csv")
    model = PanelRegressionModel(target="invest", independent_variables=["value", "capital"], entity_column="firm", time_column="year")
    model.fit(df)

    path = tmp_path / "model.joblib"
    model.save(str(path))
    loaded = model.load(str(path))

    np.testing.assert_allclose(loaded.model.params, model.model.params)
    np.testing.assert_allclose(np.asarray(loaded.model.fitted_values), np.asarray(model.model.fitted_values))


def test_model_joblib_roundtrips_var(tmp_path, test_df):
    model = VARModel(target_variables=["y", "x1", "x2"], date_column="date")
    model.fit(test_df)

    path = tmp_path / "model.joblib"
    model.save(str(path))
    loaded = model.load(str(path))

    np.testing.assert_allclose(loaded.model.params, model.model.params)
    np.testing.assert_allclose(np.asarray(loaded.model.fittedvalues), np.asarray(model.model.fittedvalues))


def test_model_joblib_roundtrips_local_projections(tmp_path):
    # Synthetic shock/target series with a date column (the packaged LP data has no date column).
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"date": pd.date_range("2000-01-01", periods=80, freq="D"), "gss_target": rng.normal(size=80), "ns": rng.normal(size=80)})
    model = LocalProjectionsModel(target="ns", shock_var="gss_target", controls=[], horizon=6, date_column="date")
    model.fit(df)

    path = tmp_path / "model.joblib"
    model.save(str(path))
    loaded = model.load(str(path))

    pd.testing.assert_frame_equal(loaded.compute_irf(), model.compute_irf())