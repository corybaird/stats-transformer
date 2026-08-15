import pytest
import yaml
from stats_transformer.featurization.base import BaseFeatureEngineer


class _ConcreteFeatureEngineer(BaseFeatureEngineer):
    def fit(self, df):
        return self

    def transform(self, df):
        return df

    def fit_transform(self, df):
        return df


def test_cannot_instantiate_abstract_base_directly():
    with pytest.raises(TypeError):
        BaseFeatureEngineer()


def test_concrete_subclass_initializes_without_params_path():
    fe = _ConcreteFeatureEngineer()
    assert fe.params == {}


def test_concrete_subclass_loads_params_from_yaml(tmp_path):
    config_path = tmp_path / "params.yaml"
    config_path.write_text(yaml.dump({"entity_column": "country"}))

    fe = _ConcreteFeatureEngineer(params_path=str(config_path))
    assert fe.params == {"entity_column": "country"}


def test_kwargs_override_existing_param_keys(tmp_path):
    config_path = tmp_path / "params.yaml"
    config_path.write_text(yaml.dump({"entity_column": "country"}))

    fe = _ConcreteFeatureEngineer(params_path=str(config_path), entity_column="iso3")
    assert fe.params["entity_column"] == "iso3"


def test_kwargs_not_in_existing_params_are_ignored(tmp_path):
    config_path = tmp_path / "params.yaml"
    config_path.write_text(yaml.dump({"entity_column": "country"}))

    fe = _ConcreteFeatureEngineer(params_path=str(config_path), unrelated_key="value")
    assert "unrelated_key" not in fe.params


def test_load_params_raises_on_missing_file():
    with pytest.raises(FileNotFoundError):
        _ConcreteFeatureEngineer(params_path="/nonexistent/path.yaml")
