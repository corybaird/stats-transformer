import pytest
from stats_transformer.utils.config import Config

def test_config_initialization():
    config = Config()
    assert config.config == {}
    
    config = Config(key="value", another=123)
    assert config.get("key") == "value"
    assert config.get("another") == 123

def test_config_get_set():
    config = Config()
    config.set("test_key", "test_value")
    assert config.get("test_key") == "test_value"
    
    # Nested
    config.set("nested.key", "nested_value")
    assert config.get("nested.key") == "nested_value"

def test_config_sections():
    config = Config(
        data={"featurization": {"feature1": True}},
        model={"type": "rf"},
        visualization={"theme": "dark"}
    )

    assert config.get_featurization_config()["feature1"] is True
    assert config.get_model_config()["type"] == "rf"
    assert config.get_visualization_config()["theme"] == "dark"


def test_validate_raises_on_unknown_model_type():
    config = Config(model={"model_type": "not_a_real_model", "target_variable": "y", "independent_variables": ["x1"]})
    with pytest.raises(ValueError, match="Unknown model_type"):
        config.validate()


def test_validate_resolves_alias_before_checking_registry():
    config = Config(model={"model_type": "iv_2sls", "target_variable": "y", "independent_variables": ["x1"], "endogenous": ["x2"], "instruments": ["z1"]})
    config.validate()  # should not raise


def test_validate_raises_on_missing_target_variable():
    config = Config(model={"model_type": "ols", "independent_variables": ["x1"]})
    with pytest.raises(ValueError, match="requires model.target_variable"):
        config.validate()


def test_validate_raises_on_missing_independent_variables():
    config = Config(model={"model_type": "ols", "target_variable": "y"})
    with pytest.raises(ValueError, match="requires model.independent_variables"):
        config.validate()


def test_validate_raises_on_iv_missing_endogenous_and_instruments():
    config = Config(model={"model_type": "iv", "target_variable": "y", "independent_variables": ["x1"]})
    with pytest.raises(ValueError, match="requires endogenous"):
        config.validate()


def test_validate_passes_for_svar_family_without_target_or_independent_variables():
    config = Config(model={"model_type": "blanchard_quah"})
    config.validate()  # svar_family models don't use target_variable/independent_variables


def test_validate_defaults_to_ols_when_model_type_absent():
    config = Config(model={"target_variable": "y", "independent_variables": ["x1"]})
    config.validate()  # should not raise; model_type defaults to "ols"
