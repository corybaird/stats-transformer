import numpy as np
import pandas as pd
import pytest
from stats_transformer.models.discrete.probit import ProbitModel


def _binary_frame(n=300):
    gen = np.random.default_rng(17)
    df = pd.DataFrame({"x1": gen.normal(size=n), "x2": gen.normal(size=n)})
    latent = 1.5 * df["x1"] - 0.8 * df["x2"] + gen.normal(size=n)
    df["y"] = (latent > 0).astype(int)
    return df


def test_probit_fit_returns_metrics():
    model = ProbitModel(target="y", independent_variables=["x1", "x2"])
    metrics = model.fit(_binary_frame())

    assert set(metrics) == {"pseudo_r_squared", "llr_pvalue", "aic", "bic", "num_observations"}
    assert 0.0 < metrics["pseudo_r_squared"] < 1.0
    assert metrics["num_observations"] == 300


def test_probit_recovers_coefficient_signs():
    model = ProbitModel(target="y", independent_variables=["x1", "x2"])
    model.fit(_binary_frame())

    assert model.model.params["x1"] > 0
    assert model.model.params["x2"] < 0


def test_probit_differs_from_logit_on_same_data():
    from stats_transformer.models.discrete.logit import LogitModel

    df = _binary_frame()
    probit = ProbitModel(target="y", independent_variables=["x1", "x2"])
    probit.fit(df)
    logit = LogitModel(target="y", independent_variables=["x1", "x2"])
    logit.fit(df)

    # Same sign, different scale -- confirms Probit is actually being used
    # rather than silently delegating to Logit.
    assert probit.model.params["x1"] > 0 and logit.model.params["x1"] > 0
    assert probit.model.params["x1"] != pytest.approx(logit.model.params["x1"], rel=1e-6)


def test_probit_raises_on_missing_columns():
    model = ProbitModel(target="y", independent_variables=["x1", "nonexistent"])
    with pytest.raises(ValueError, match="Missing columns"):
        model.fit(_binary_frame())


def test_probit_raises_before_fit():
    model = ProbitModel(target="y", independent_variables=["x1"])
    with pytest.raises(ValueError, match="Model not trained"):
        model.get_model_metrics()
    with pytest.raises(ValueError, match="Model not trained"):
        model.get_summary()


def test_probit_metadata_has_coefficients():
    model = ProbitModel(target="y", independent_variables=["x1", "x2"])
    model.fit(_binary_frame())
    metadata = model.get_model_metadata()

    assert metadata["summary"]["dependent_variable"] == "y"
    assert "x1" in metadata["coefficients"]


def test_probit_is_dispatchable_from_pipeline(tmp_path):
    import yaml
    from stats_transformer.pipeline import Pipeline

    config_path = tmp_path / "params.yaml"
    config_path.write_text(yaml.dump({"model": {"model_type": "probit", "target_variable": "y", "independent_variables": ["x1", "x2"]}}))

    pipeline = Pipeline(params_path=str(config_path))
    pipeline._initialize_from_params()
    assert isinstance(pipeline.model, ProbitModel)


def test_probit_exported_from_models_package():
    from stats_transformer.models import ProbitModel as Exported
    assert Exported is ProbitModel
