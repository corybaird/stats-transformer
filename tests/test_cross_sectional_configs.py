"""Cross-sectional configs (no entity column) must run the features and model stages.

Regression tests for the paper's Mincer example (docs/wrapup triage P0):
FeatureEngineer used to demand [None, 'date'] when a config declared neither
entity nor date columns.
"""

import glob
import os
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


@pytest.mark.parametrize(
    "config",
    [
        pytest.param(
            "references/configs/timeseries/identification/structural_restrictions.yaml",
            marks=pytest.mark.xfail(strict=True, reason="Restrictions-only SVAR input file, not a standalone params.yaml (no data:/model: sections)"),
        ),
        pytest.param(
            "references/configs/timeseries/reduced_form/local_projections.yaml",
            marks=pytest.mark.xfail(strict=True, reason="Config declares date_column: date but sofr_surprises.parquet.gzip has no date column"),
        ),
    ] + [pytest.param(os.path.relpath(c, REPO_ROOT), id=os.path.relpath(c, REPO_ROOT))
         for c in ALL_CONFIGS
         if os.path.basename(c) not in ("structural_restrictions.yaml", "local_projections.yaml")],
)
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