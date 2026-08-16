import json
import os
import pandas as pd
import pytest
import yaml
from stats_transformer.pipeline import Pipeline
from stats_transformer.models.regression.panel import PanelRegressionModel
from stats_transformer.models.regression.iv import IV2SLSModel
from stats_transformer.models.regression.panel_iv import PanelIV2SLSModel
from stats_transformer.models.unsupervised.unsupervised import PCAModel
from stats_transformer.models.timeseries import BlanchardQuahModel, LocalProjectionsModel, LocalProjectionsIVModel


def _write_config(tmp_path, model, data=None, visualization=None):
    config = {"model": model}
    if data is not None:
        config["data"] = data
    if visualization is not None:
        config["visualization"] = visualization
    config_path = tmp_path / "params.yaml"
    config_path.write_text(yaml.dump(config))
    return config_path


# --- _initialize_from_args, kinds not covered by test_pipeline.py's params-path tests ---

def test_initialize_from_args_panel(test_df):
    pipeline = Pipeline(entity_column="country", target="y", features=["x1", "x2"], transformations=[], model_type="panel_ols")
    pipeline._initialize_from_args()
    assert isinstance(pipeline.model, PanelRegressionModel)


def test_initialize_from_args_iv(test_df):
    pipeline = Pipeline(entity_column="country", target="y", features=["x1"], transformations=[], model_type="iv", endogenous=["x2"], instruments=["x1"])
    pipeline._initialize_from_args()
    assert isinstance(pipeline.model, IV2SLSModel)
    assert pipeline.model.endogenous == ["x2"]
    assert pipeline.model.instruments == ["x1"]


def test_initialize_from_args_panel_iv(test_df):
    pipeline = Pipeline(entity_column="country", target="y", features=["x1"], transformations=[], model_type="panel_iv", endogenous=["x2"], instruments=["x1"])
    pipeline._initialize_from_args()
    assert isinstance(pipeline.model, PanelIV2SLSModel)


def test_initialize_from_args_unsupervised():
    pipeline = Pipeline(entity_column="country", features=["x1", "x2"], transformations=[], model_type="pca")
    pipeline._initialize_from_args()
    assert isinstance(pipeline.model, PCAModel)


def test_initialize_from_args_svar_family():
    pipeline = Pipeline(entity_column="country", target="y", features=["x1", "x2"], transformations=[], model_type="blanchard_quah")
    pipeline._initialize_from_args()
    assert isinstance(pipeline.model, BlanchardQuahModel)
    assert pipeline.model.target_variables == ["y", "x1", "x2"]


def test_initialize_from_args_lp_iv():
    pipeline = Pipeline(entity_column="country", target="y", features=["x1"], transformations=[], model_type="lp_iv")
    pipeline._initialize_from_args()
    assert isinstance(pipeline.model, LocalProjectionsIVModel)


def test_initialize_from_args_local_projections():
    pipeline = Pipeline(entity_column="country", target="y", features=["x1", "x2"], transformations=[], model_type="local_projections", horizon=4)
    pipeline._initialize_from_args()
    assert isinstance(pipeline.model, LocalProjectionsModel)
    assert pipeline.model.shock_var == "x1"
    assert pipeline.model.controls == ["x2"]
    assert pipeline.model.horizon == 4


def test_initialize_from_params_local_projections(tmp_path):
    config_path = _write_config(tmp_path, {"model_type": "local_projections", "target_variable": "y", "shock_variable": "x2", "independent_variables": ["x1", "x2"], "horizon": 3, "date_column": "date"})
    pipeline = Pipeline(params_path=str(config_path))
    pipeline._initialize_from_params()
    assert isinstance(pipeline.model, LocalProjectionsModel)
    assert pipeline.model.shock_var == "x2"
    assert pipeline.model.controls == ["x1"]
    assert pipeline.model.horizon == 3
    assert pipeline.model.date_column == "date"


def test_initialize_from_args_requires_entity_column():
    pipeline = Pipeline(target="y", features=["x1"], model_type="ols")
    with pytest.raises(ValueError, match="entity_column must be specified"):
        pipeline._initialize_from_args()


def test_initialize_from_args_noop_without_target_or_features():
    pipeline = Pipeline(entity_column="country", transformations=[])
    pipeline._initialize_from_args()
    assert pipeline.model is None


# --- fit_transform: no feature_engineer initialized ---

def test_fit_transform_raises_without_feature_engineer(test_df):
    pipeline = Pipeline(entity_column="country", target="y", features=["x1", "x2"], transformations=[], model_type="ols")
    pipeline.feature_engineer = None
    pipeline._initialize_from_args = lambda: None  # bypass so feature_engineer stays None
    with pytest.raises(ValueError, match="No feature engineering component initialized"):
        pipeline.fit_transform(test_df)


# --- save_results / save_model_summary ---

def test_save_results_writes_transformed_data_and_model_summary(tmp_path, test_df):
    pipeline = Pipeline(entity_column="country", target="y", features=["x1", "x2"], transformations=[], model_type="ols")
    pipeline.fit_transform(test_df)

    results = pipeline.save_results(output_dir=str(tmp_path))

    assert "transformed_data" in results
    assert os.path.exists(results["transformed_data"])
    assert "model_summary" in results
    assert os.path.exists(results["model_summary"])


def test_save_results_empty_when_nothing_to_save(tmp_path):
    pipeline = Pipeline()
    results = pipeline.save_results(output_dir=str(tmp_path))
    assert results == {}


def test_save_model_summary_noop_when_no_results(tmp_path):
    pipeline = Pipeline()
    output_path = tmp_path / "summary.json"
    pipeline.save_model_summary(str(output_path))
    assert not output_path.exists()


def test_save_model_summary_writes_json(tmp_path, test_df):
    pipeline = Pipeline(entity_column="country", target="y", features=["x1", "x2"], transformations=[], model_type="ols")
    pipeline.fit_transform(test_df)

    output_path = tmp_path / "nested" / "summary.json"
    pipeline.save_model_summary(str(output_path))

    assert output_path.exists()
    with open(output_path) as f:
        data = json.load(f)
    assert "metrics" in data


# --- create_visualizations ---

def test_create_visualizations_empty_without_results():
    pipeline = Pipeline()
    assert pipeline.create_visualizations() == {}


def test_create_visualizations_produces_time_series_for_transformed_data(tmp_path, test_df):
    pipeline = Pipeline(entity_column="country", date_column="date", target="y", features=["x1", "x2"], transformations=[], model_type="ols")
    pipeline.fit_transform(test_df)

    results = pipeline.create_visualizations(output_dir=str(tmp_path), display_only=True)
    assert "time_series" in results


# --- predict ---

def test_predict_raises_when_model_not_fitted():
    pipeline = Pipeline()
    with pytest.raises(ValueError, match="Model must be fitted"):
        pipeline.predict()


# --- run(): resample stage ---

def test_run_resample_stage_merges_and_writes_parquet(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    df1 = pd.DataFrame({"country": ["USA", "USA"], "date": ["2020-01-01", "2021-01-01"], "gdp": [1.0, 2.0]})
    df2 = pd.DataFrame({"country": ["USA", "USA"], "date": ["2020-01-01", "2021-01-01"], "inflation": [0.1, 0.2]})
    df1.to_csv(raw_dir / "gdp.csv", index=False)
    df2.to_csv(raw_dir / "inflation.csv", index=False)

    output_path = tmp_path / "merged.parquet"
    config = {
        "model": {"model_type": "ols", "target_variable": "gdp", "independent_variables": ["inflation"]},
        "data": {
            "datasets": [
                {"name": "gdp", "path": str(raw_dir / "gdp.csv"), "entity_column": "country", "date_column": "date", "source_period": "annual", "resample": {"target_period": "annual"}},
                {"name": "inflation", "path": str(raw_dir / "inflation.csv"), "entity_column": "country", "date_column": "date", "source_period": "annual", "resample": {"target_period": "annual"}},
            ],
            "merge": {"on": ["country", "date"], "how": "outer", "output_path": str(output_path)},
        },
    }
    config_path = tmp_path / "params.yaml"
    config_path.write_text(yaml.dump(config))

    pipeline = Pipeline(params_path=str(config_path))
    merged = pipeline.run(stage="resample")

    # resample_dataset suffixes numeric columns with the resample period
    # (e.g. "_Y" for annual), so the merged frame carries gdp_Y/inflation_Y.
    assert output_path.exists()
    assert "gdp_Y" in merged.columns
    assert "inflation_Y" in merged.columns


# --- run(): eda stage ---

def test_run_eda_stage_calls_eda_visualizer(tmp_path, test_df, monkeypatch):
    data_path = tmp_path / "data.csv"
    test_df.to_csv(data_path, index=False)

    config_path = _write_config(
        tmp_path,
        model={"model_type": "ols", "target_variable": "y", "independent_variables": ["x1", "x2"]},
        data={"raw_data_file": str(data_path), "featurization": {"entity_column": "country", "date_column": "date"}},
        visualization={"output_dir": str(tmp_path / "viz")},
    )

    calls = {}

    def fake_run(self, data_path=None, output_path=None):
        calls["data_path"] = data_path
        return {"ran": True}

    monkeypatch.setattr("stats_transformer.visualization.eda.eda.EDAVisualizer.run", fake_run)

    pipeline = Pipeline(params_path=str(config_path))
    result = pipeline.run(stage="eda")

    assert result == {"ran": True}
    assert calls["data_path"] == str(data_path)


def test_run_eda_stage_raises_without_data_path(tmp_path):
    config_path = _write_config(
        tmp_path,
        model={"model_type": "ols", "target_variable": "y", "independent_variables": ["x1"]},
        data={},
    )
    pipeline = Pipeline(params_path=str(config_path))
    with pytest.raises(ValueError, match="No data path found for eda stage"):
        pipeline.run(stage="eda")


# --- run(): features stage raises without data path ---

def test_run_features_stage_raises_without_data_path(tmp_path):
    config_path = _write_config(
        tmp_path,
        model={"model_type": "ols", "target_variable": "y", "independent_variables": ["x1"]},
        data={},
    )
    pipeline = Pipeline(params_path=str(config_path))
    with pytest.raises(ValueError, match="No data path found for features stage"):
        pipeline.run(stage="features")


# --- run(): visualization stage, loading from disk ---

def test_run_visualization_stage_loads_transformed_data_and_summary_from_disk(tmp_path, test_df):
    processed_path = tmp_path / "processed.csv"
    test_df.to_csv(processed_path, index=False)

    summary_path = tmp_path / "summary.json"
    summary_path.write_text(json.dumps({"metrics": {"r_squared": 0.5}}))

    config_path = _write_config(
        tmp_path,
        model={"model_type": "ols", "target_variable": "y", "independent_variables": ["x1", "x2"], "summary_output_path": str(summary_path)},
        data={"raw_data_file": str(tmp_path / "unused.csv"), "featurization": {"entity_column": "country", "date_column": "date", "output_path": str(processed_path)}},
        visualization={"output_dir": str(tmp_path / "viz")},
    )

    pipeline = Pipeline(params_path=str(config_path))
    results = pipeline.run(stage="visualization")

    assert pipeline.transformed_data is not None
    assert pipeline.model_results == {"metrics": {"r_squared": 0.5}}
    assert "time_series" in results


# --- run(): full run (stage=None) ---

def test_run_full_pipeline(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    df1 = pd.DataFrame({"country": ["USA", "USA", "CAN"], "date": ["2020-01-01", "2021-01-01", "2020-01-01"], "y": [10.0, 12.0, 9.0]})
    df2 = pd.DataFrame({"country": ["USA", "USA", "CAN"], "date": ["2020-01-01", "2021-01-01", "2020-01-01"], "x1": [2.0, 3.0, 1.0]})
    df1.to_csv(raw_dir / "y.csv", index=False)
    df2.to_csv(raw_dir / "x1.csv", index=False)

    merged_path = tmp_path / "merged.parquet"
    processed_path = tmp_path / "processed.csv"
    summary_path = tmp_path / "summary.json"

    config = {
        # resample_dataset suffixes numeric columns with the resample period
        # ("_Y" for annual), so the merged/featurized frame carries y_Y/x1_Y.
        "model": {"model_type": "ols", "target_variable": "y_Y", "independent_variables": ["x1_Y"], "summary_output_path": str(summary_path)},
        "data": {
            "datasets": [
                {"name": "y", "path": str(raw_dir / "y.csv"), "entity_column": "country", "date_column": "date", "source_period": "annual", "resample": {"target_period": "annual"}},
                {"name": "x1", "path": str(raw_dir / "x1.csv"), "entity_column": "country", "date_column": "date", "source_period": "annual", "resample": {"target_period": "annual"}},
            ],
            "merge": {"on": ["country", "date"], "how": "outer", "output_path": str(merged_path)},
            "featurization": {"entity_column": "country", "date_column": "date", "output_path": str(processed_path)},
        },
        "visualization": {"output_dir": str(tmp_path / "viz")},
    }
    config_path = tmp_path / "params.yaml"
    config_path.write_text(yaml.dump(config))

    pipeline = Pipeline(params_path=str(config_path))
    transformed = pipeline.run(stage=None)

    assert merged_path.exists()
    assert processed_path.exists()
    assert summary_path.exists()
    assert transformed is not None
    assert pipeline.model_results is not None
