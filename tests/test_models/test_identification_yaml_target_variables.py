import yaml
import pytest
from stats_transformer.models.timeseries.identification.svar import SVARModel
from stats_transformer.models.timeseries.identification.proxy_svar import ProxySVARModel
from stats_transformer.models.timeseries.identification.volatility import VolatilitySVARModel
from stats_transformer.models.timeseries.identification.blanchard_quah import BlanchardQuahModel
from stats_transformer.models.timeseries.identification.sign_zero import SignZeroSVARModel
from stats_transformer.models.timeseries.identification.independence import IndependenceSVARModel


@pytest.mark.parametrize("model_cls", [
    SVARModel,
    ProxySVARModel,
    VolatilitySVARModel,
    BlanchardQuahModel,
    SignZeroSVARModel,
    IndependenceSVARModel,
])
def test_identification_models_read_target_variables_from_params_yaml(tmp_path, model_cls):
    config_path = tmp_path / "params.yaml"
    config_path.write_text(yaml.dump({
        "model": {
            "model_type": model_cls.__name__,
            "target_variables": ["y1", "y2"]
        }
    }))

    model = model_cls(params_path=str(config_path))
    assert model.target_variables == ["y1", "y2"]