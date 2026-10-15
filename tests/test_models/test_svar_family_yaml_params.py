"""SVAR-family models must honour every scalar param in the YAML model section.

Regression (follow-up to #107/#110): Pipeline builds kind=="svar_family" as
cls(params_path=...), but several identification models only read
target_variables from YAML; everything else (maxlags, n_starts,
regime_column, ...) was silently ignored.

Precedence enforced by ModelBase._param: explicit kwarg > YAML > default.
"""

import glob
import os

import yaml
import pytest
from stats_transformer import Pipeline

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

IDENTIFICATION_CONFIGS = [
    p for p in sorted(glob.glob(os.path.join(REPO_ROOT, "references", "configs", "timeseries", "identification", "*.yaml")))
    if os.path.basename(p) != "structural_restrictions.yaml"  # restrictions-only input file, no model: section
]

# Model-section keys handled elsewhere (registry dispatch / target wiring),
# not plain scalar attributes of the model.
NON_ATTR_KEYS = {"model_type", "target_variables", "target_variable", "independent_variables"}


@pytest.mark.parametrize("config_path", IDENTIFICATION_CONFIGS, ids=lambda p: os.path.basename(p))
def test_identification_config_params_land_on_model(config_path):
    with open(config_path) as f:
        doc = yaml.safe_load(f)
    model_params = doc.get("model") or {}
    pipeline = Pipeline(params_path=config_path)
    pipeline._initialize_from_params()
    for key, value in model_params.items():
        if key in NON_ATTR_KEYS:
            continue
        assert getattr(pipeline.model, key) == value


def test_explicit_kwarg_beats_yaml(tmp_path):
    """Precedence: explicit constructor kwarg > YAML model section > default."""
    config_path = tmp_path / "params.yaml"
    config_path.write_text(yaml.dump({
        "model": {
            "model_type": "blanchard_quah",
            "target_variables": ["y1", "y2"],
            "maxlags": 3,
        }
    }))
    from stats_transformer.models.timeseries.identification.blanchard_quah import BlanchardQuahModel

    assert BlanchardQuahModel(params_path=str(config_path)).maxlags == 3  # YAML
    assert BlanchardQuahModel(params_path=str(config_path), maxlags=2).maxlags == 2  # explicit kwarg
    assert BlanchardQuahModel(params_path=None, target_variables=["y1", "y2"], maxlags=4).maxlags == 4  # no YAML