import hashlib
import json
import os
import subprocess
from pathlib import Path

import pandas as pd
import pytest
import yaml

from stats_transformer.pipeline import Pipeline

REPO_ROOT = Path(__file__).resolve().parents[1]


def _write_full_config(tmp_path, with_explicit_paths=False, with_processed=False):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir(exist_ok=True)
    dates = ["2020-01-01", "2021-01-01", "2022-01-01"]
    df1 = pd.DataFrame({"country": ["USA", "USA", "USA", "CAN", "CAN", "CAN"], "date": dates * 2, "y": [10.0, 11.0, 12.0, 9.0, 10.0, 11.0]})
    df2 = pd.DataFrame({"country": ["USA", "USA", "USA", "CAN", "CAN", "CAN"], "date": dates * 2, "x1": [2.0, 3.0, 4.0, 1.0, 2.0, 3.0]})
    df1.to_csv(raw_dir / "y.csv", index=False)
    df2.to_csv(raw_dir / "x1.csv", index=False)

    # resample_dataset suffixes numeric columns with the resample period ("_Y" for annual)
    processed_path = tmp_path / "processed.csv"
    if with_processed:
        pd.DataFrame({
            "country": ["USA"] * 3 + ["CAN"] * 3, "date": dates * 2,
            "y_Y": [10.0, 11.0, 12.0, 9.0, 10.0, 11.0], "x1_Y": [2.0, 3.0, 4.0, 1.0, 2.0, 3.0],
        }).to_csv(processed_path, index=False)

    data = {
        "datasets": [
            {"name": "y", "path": str(raw_dir / "y.csv"), "entity_column": "country", "date_column": "date", "source_period": "annual", "resample": {"target_period": "annual"}},
            {"name": "x1", "path": str(raw_dir / "x1.csv"), "entity_column": "country", "date_column": "date", "source_period": "annual", "resample": {"target_period": "annual"}},
        ],
        "featurization": {"entity_column": "country", "date_column": "date"},
        "raw_data_file": str(raw_dir / "y.csv"),
        "merge": {"on": ["country", "date"], "how": "outer"},
    }
    if with_processed or with_explicit_paths:
        data["featurization"]["output_path"] = str(processed_path)
    if with_explicit_paths:
        data["merge"]["output_path"] = str(tmp_path / "merged.parquet")

    model = {"model_type": "ols", "target_variable": "y_Y", "independent_variables": ["x1_Y"], "summary_output_path": str(tmp_path / "summary.json")} if with_explicit_paths else {"model_type": "ols", "target_variable": "y_Y", "independent_variables": ["x1_Y"]}

    # output.root inside tmp_path keeps per-run artefacts out of the repo
    config = {
        "output": {"root": str(tmp_path / "output_root")},
        "model": model,
        "data": data,
        "reporting": {"output_dir": str(tmp_path / "viz")},
    }
    config_path = tmp_path / "params.yaml"
    config_path.write_text(yaml.dump(config))
    return config_path, raw_dir


def _repo_snapshot():
    """Snapshot the repo tree (git status including ignored files, plus file paths + mtimes)."""
    status = subprocess.run(["git", "-C", str(REPO_ROOT), "status", "--porcelain", "--ignored"], capture_output=True, text=True).stdout
    files = set()
    for p in REPO_ROOT.rglob("*"):
        if {".git", ".venv", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"} & set(p.parts):
            continue
        if p.is_file():
            files.add((str(p), p.stat().st_mtime_ns))
    return status, files


def _single_run_dir(output_root: Path) -> Path:
    """Return <root>/<config_stem>/<timestamp>/ for the only run under output_root."""
    stem_dirs = list(output_root.iterdir())
    assert len(stem_dirs) == 1, f"expected exactly one config folder under {output_root}, got {stem_dirs}"
    run_dirs = [d for d in stem_dirs[0].iterdir() if d.is_dir()]
    assert len(run_dirs) == 1, f"expected exactly one run folder, got {run_dirs}"
    return run_dirs[0]


# --- full run and single-stage runs land in a per-run folder; nothing in reports/ ---

@pytest.mark.parametrize("stage", [None, "data", "features", "model"])
def test_run_writes_run_folder_and_nothing_in_reports(tmp_path, monkeypatch, stage):
    monkeypatch.chdir(tmp_path)
    config_path, _ = _write_full_config(tmp_path, with_processed=stage == "model")

    pipeline = Pipeline(params_path=str(config_path))
    pipeline.run(stage=stage)

    run_dir = _single_run_dir(tmp_path / "output_root")
    if stage in (None, "data"):
        assert (run_dir / "01_data" / "merged.parquet").exists()
    if stage in (None, "features"):
        assert (run_dir / "02_features" / "features.parquet").exists()
    assert (run_dir / "manifest.json").exists()
    if stage in (None, "model"):
        assert (run_dir / "03_model" / "model_summary.json").exists()
    # The hardcoded reports/ write is gone.
    assert not (tmp_path / "reports").exists()
    # Default (non-explicit) paths write only the run-folder artefacts.
    if stage == "features":
        assert not (tmp_path / "processed.csv").exists()


# --- nothing leaks outside tmp_path (repo tree snapshot before/after) ---

@pytest.mark.parametrize("stage", [None, "data", "features", "model", "reporting"])
def test_run_writes_nothing_outside_tmp_path(tmp_path, monkeypatch, stage):
    monkeypatch.chdir(tmp_path)
    if stage == "reporting":
        # Give the reporting stage readable disk artefacts so it actually draws
        test_df = pd.read_csv(REPO_ROOT / "tests" / "data" / "test_data.csv")
        test_df.to_csv(tmp_path / "processed.csv", index=False)
        (tmp_path / "summary.json").write_text(json.dumps({"metrics": {"r_squared": 0.5}}))

    config_path, _ = _write_full_config(tmp_path, with_explicit_paths=stage == "reporting", with_processed=stage == "model")
    before_status, before_files = _repo_snapshot()

    pipeline = Pipeline(params_path=str(config_path))
    pipeline.run(stage=stage)

    after_status, after_files = _repo_snapshot()
    # Snapshot covers tracked files, ignored dirs (data/pipeline/, reports/) and mtimes
    assert after_status == before_status, f"repo tree changed:\n{after_status}"
    assert {name for name, _ in after_files} == {name for name, _ in before_files}
    assert after_files == before_files


def test_run_with_repo_cwd_writes_nothing_into_repo(tmp_path, monkeypatch):
    # Mirrors the leak that produced the stray CSVs in reports/: tests whose config
    # lives in tmp_path but whose CWD stays at the repo root. On main,
    # run_model() wrote reports/transformed_data_*.csv relative to the repo CWD;
    # these writes must go to the output.root run folder or the explicit paths
    # instead. Deliberately does not chdir; explicit paths keep the branch side
    # of the comparison clean (the default merge path is repo-CWD-relative).
    before_status, before_files = _repo_snapshot()

    config_path, _ = _write_full_config(tmp_path, with_explicit_paths=True)
    pipeline = Pipeline(params_path=str(config_path))
    pipeline.run()

    after_status, after_files = _repo_snapshot()
    assert after_status == before_status, f"repo tree changed:\n{after_status}"
    assert {name for name, _ in after_files} == {name for name, _ in before_files}
    assert after_files == before_files


# --- single-stage data run with explicit paths still gets a run manifest ---

def test_run_data_stage_with_explicit_merge_path_still_gets_run_manifest(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config_path, _ = _write_full_config(tmp_path, with_explicit_paths=True)
    pipeline = Pipeline(params_path=str(config_path))
    pipeline.run(stage="data")

    run_dir = _single_run_dir(tmp_path / "output_root")
    assert (run_dir / "manifest.json").exists()
    # merged.parquet is always copied into the run folder, even with an explicit merge output_path.
    assert (run_dir / "01_data" / "merged.parquet").exists()
    with open(run_dir / "manifest.json") as f:
        manifest = json.load(f)
    assert manifest["stages_run"] == ["data"]
    assert str(tmp_path / "merged.parquet") in manifest["files"]
    assert (tmp_path / "merged.parquet").exists()


# --- stage chaining across separate Pipeline instances (explicit paths unset) ---

def test_data_stage_then_fresh_pipeline_features_stage(tmp_path, monkeypatch):
    # The README/DVC way to run single stages: one Pipeline invocation per stage.
    # The canonical merged output stays at the explicit path or the data/pipeline
    # default so a fresh instance finds it, as on main.
    monkeypatch.chdir(tmp_path)
    config_path, _ = _write_full_config(tmp_path)
    Pipeline(params_path=str(config_path)).run(stage="data")

    assert (tmp_path / "data" / "pipeline" / "resampled_merged.parquet").exists()

    transformed = Pipeline(params_path=str(config_path)).run(stage="features")
    assert transformed is not None


# --- manifest contents ---

def test_full_run_manifest_contents(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config_path, raw_dir = _write_full_config(tmp_path)
    pipeline = Pipeline(params_path=str(config_path))
    pipeline.run()

    with open(_single_run_dir(tmp_path / "output_root") / "manifest.json") as f:
        manifest = json.load(f)

    assert set(manifest) == {"stages_run", "started_at_utc", "finished_at_utc", "config", "config_sha256", "inputs_sha256", "git_sha", "package_version", "files", "shapes"}
    assert manifest["stages_run"] == ["data", "features", "model", "reporting"]
    assert manifest["finished_at_utc"] >= manifest["started_at_utc"]
    assert manifest["finished_at_utc"].endswith("+00:00")
    with open(config_path, "rb") as f:
        assert manifest["config_sha256"] == hashlib.sha256(f.read()).hexdigest()
    with open(config_path) as f:
        assert manifest["config"] == yaml.safe_load(f)
    for csv_name in ("y.csv", "x1.csv"):
        csv_path = raw_dir / csv_name
        with open(csv_path, "rb") as f:
            assert manifest["inputs_sha256"][str(csv_path)] == hashlib.sha256(f.read()).hexdigest()
    assert manifest["package_version"]
    file_names = {Path(p).name for p in manifest["files"]}
    assert {"merged.parquet", "features.parquet", "model_summary.json"} <= file_names


def test_manifest_config_hash_tracks_config_edits(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config_path, _ = _write_full_config(tmp_path)
    pipeline = Pipeline(params_path=str(config_path))
    pipeline.run()
    first_dir = _single_run_dir(tmp_path / "output_root")

    # A second run lands in a fresh run folder; editing the config changes its hash
    config_path.write_text(config_path.read_text() + "# edited\n")
    pipeline2 = Pipeline(params_path=str(config_path))
    pipeline2.run()

    run_dirs = sorted((tmp_path / "output_root" / "params").iterdir())
    assert len(run_dirs) == 2
    with open(first_dir / "manifest.json") as f:
        old_manifest = json.load(f)
    second_dir = run_dirs[1]
    assert second_dir.name != first_dir.name
    with open(second_dir / "manifest.json") as f:
        new_manifest = json.load(f)
    assert new_manifest["config_sha256"] != old_manifest["config_sha256"]


# --- explicit paths keep working; the run folder is an addition ---

def test_explicit_output_paths_still_honoured(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config_path, _ = _write_full_config(tmp_path, with_explicit_paths=True)
    pipeline = Pipeline(params_path=str(config_path))
    pipeline.run()

    assert (tmp_path / "merged.parquet").exists()
    assert (tmp_path / "processed.csv").exists()
    assert (tmp_path / "summary.json").exists()
    # Run folder written as usual; 01_data/merged.parquet is always written (the
    # explicit merge output_path keeps its canonical copy too).
    run_dir = _single_run_dir(tmp_path / "output_root")
    assert (run_dir / "01_data" / "merged.parquet").exists()
    assert (run_dir / "02_features" / "features.parquet").exists()
    assert (run_dir / "03_model" / "model_summary.json").exists()
    assert (run_dir / "manifest.json").exists()
    with open(run_dir / "manifest.json") as f:
        manifest = json.load(f)
    assert manifest["stages_run"] == ["data", "features", "model", "reporting"]
    assert str(tmp_path / "merged.parquet") in manifest["files"]
    assert not (tmp_path / "reports").exists()


def test_full_run_features_parquet_keeps_date_dtype(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config_path, _ = _write_full_config(tmp_path)
    pipeline = Pipeline(params_path=str(config_path))
    pipeline.run()

    run_dir = _single_run_dir(tmp_path / "output_root")
    features = pd.read_parquet(run_dir / "02_features" / "features.parquet")
    transformed = pipeline.transformed_data
    assert features["date"].dtype == transformed["date"].dtype


# --- save_results public method is unchanged ---

def test_save_results_public_api_unchanged(tmp_path, test_df):
    pipeline = Pipeline(entity_column="country", target="y", features=["x1", "x2"], transformations=[], model_type="ols")
    pipeline.fit_transform(test_df)
    results = pipeline.save_results(output_dir=str(tmp_path))
    assert os.path.exists(results["transformed_data"])
    assert os.path.exists(results["model_summary"])