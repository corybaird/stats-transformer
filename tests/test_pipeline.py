import pytest
import pandas as pd
import os
import shutil
import yaml
from pathlib import Path
from stats_transformer.pipeline import Pipeline
from stats_transformer.models.registry import MODEL_REGISTRY
from stats_transformer.models.regression.iv import IV2SLSModel

def test_pipeline_run_regression():
    # Setup
    config_path = "references/configs/test_pipeline.yaml"
    temp_dir = "data/temp"
    os.makedirs(temp_dir, exist_ok=True)
    
    pipeline = Pipeline(params_path=config_path)
    
    # Run the regression stage
    results = pipeline.run(stage="regression")
    
    assert results is not None
    assert "metrics" in results
    assert "r_squared" in results["metrics"]
    assert os.path.exists("data/temp/test_summary.json")
    
    # Clean up
    if os.path.exists("data/temp/test_summary.json"):
        os.remove("data/temp/test_summary.json")

def test_pipeline_fit_transform():
    config_path = "references/configs/test_pipeline.yaml"
    pipeline = Pipeline(params_path=config_path)
    
    df = pd.read_csv("tests/data/test_data.csv")
    transformed = pipeline.fit_transform(df)
    
    assert transformed is not None
    assert "y" in transformed.columns
    assert "x1" in transformed.columns
    assert pipeline.model_results is not None
    assert "metrics" in pipeline.model_results

def test_pipeline_fit_transform_from_constructor_args():
    pipeline = Pipeline(
        entity_column="country",
        target="y",
        features=["x1", "x2"],
        transformations=[],
        model_type="ols",
    )

    df = pd.read_csv("tests/data/test_data.csv")
    transformed = pipeline.fit_transform(df)

    assert transformed is not None
    assert pipeline.model_results is not None
    assert "metrics" in pipeline.model_results


MODEL_TYPE_DISPATCH = [(model_type, entry["cls"]) for model_type, entry in MODEL_REGISTRY.items()]


@pytest.mark.parametrize("model_type,expected_cls", MODEL_TYPE_DISPATCH)
def test_pipeline_dispatches_model_type_from_params(tmp_path, model_type, expected_cls):
    config_path = tmp_path / "params.yaml"
    model_config = {"model_type": model_type, "target_variable": "y", "independent_variables": ["x1"]}
    data_config = {}
    if model_type in ("panel_ols", "panel_iv"):
        data_config = {"featurization": {"entity_column": "country"}}
    if model_type == "iv":
        model_config["endogenous"] = ["x2"]
        model_config["instruments"] = ["z1"]
    elif model_type == "panel_iv":
        model_config["panel_iv"] = {"endogenous": ["x2"], "instruments": ["z1"]}
    config_path.write_text(yaml.dump({"model": model_config, "data": data_config}))

    pipeline = Pipeline(params_path=str(config_path))
    pipeline._initialize_from_params()

    assert isinstance(pipeline.model, expected_cls)


def test_pipeline_construction_validates_config_before_dispatch():
    config_path = "references/configs/mroz_iv.yaml"
    pipeline = Pipeline(params_path=config_path)
    # _get_config() runs Config.validate(); a config with an unknown or
    # underspecified model_type must fail here, before any model is built.
    pipeline._get_config()


@pytest.mark.parametrize("config_path", sorted(Path("references/configs").glob("*.yaml")))
def test_pipeline_configs_pass_validation_or_are_not_pipeline_configs(config_path):
    with open(config_path, "r") as f:
        params = yaml.safe_load(f)
    if (params or {}).get("model", {}).get("target_variable") is None and (params or {}).get("model", {}).get("model_type") not in MODEL_REGISTRY:
        pytest.skip(f"{config_path} is not a Pipeline params file")

    pipeline = Pipeline(params_path=str(config_path))
    pipeline._get_config()  # must not raise


def test_mroz_iv_config_dispatches_iv_with_endogenous_and_instruments():
    pipeline = Pipeline(params_path="references/configs/mroz_iv.yaml")
    pipeline._initialize_from_params()

    assert isinstance(pipeline.model, IV2SLSModel)
    assert pipeline.model.endogenous == ["educ"]
    assert pipeline.model.instruments == ["motheduc", "fatheduc"]


def test_pipeline_raises_on_unknown_model_type(tmp_path):
    config_path = tmp_path / "params.yaml"
    config_path.write_text(yaml.dump({"model": {"model_type": "not_a_real_model"}}))

    pipeline = Pipeline(params_path=str(config_path))
    with pytest.raises(ValueError, match="Unknown model_type"):
        pipeline._initialize_from_params()


def test_pipeline_raises_on_unknown_stage():
    pipeline = Pipeline(params_path="references/configs/test_pipeline.yaml")
    with pytest.raises(ValueError, match="Unknown stage"):
        pipeline.run(stage="regresion")


def test_pipeline_predict_raises_not_implemented_for_unsupported_model():
    # No model in stats_transformer currently implements predict(); calling
    # Pipeline.predict() must fail explicitly rather than with a confusing
    # AttributeError from inside the model.
    pipeline = Pipeline(params_path="references/configs/test_pipeline.yaml")
    df = pd.read_csv("tests/data/test_data.csv")
    pipeline.fit_transform(df)
    with pytest.raises(NotImplementedError, match="does not implement predict"):
        pipeline.predict(df)


def test_models_star_import_does_not_raise():
    import stats_transformer.models as models_module
    for name in models_module.__all__:
        assert hasattr(models_module, name), f"__all__ entry '{name}' is not importable"


@pytest.mark.parametrize("config_path", sorted(Path("references/configs").glob("*.yaml")))
def test_pipeline_configs_dispatch_to_declared_model_type(config_path):
    with open(config_path, "r") as f:
        params = yaml.safe_load(f)
    model_type = (params or {}).get("model", {}).get("model_type")
    if model_type is None:
        pytest.skip(f"{config_path} has no model.model_type (not a Pipeline params file)")

    expected_by_type = dict(MODEL_TYPE_DISPATCH)
    expected_cls = expected_by_type.get(model_type)
    if expected_cls is None:
        pytest.fail(f"{config_path} uses model_type '{model_type}', which is not a recognized dispatch key")

    pipeline = Pipeline(params_path=str(config_path))
    pipeline._initialize_from_params()

    assert isinstance(pipeline.model, expected_cls), f"{config_path}: model_type '{model_type}' dispatched to {type(pipeline.model).__name__}, expected {expected_cls.__name__}"
