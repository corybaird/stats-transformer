"""LaTeX tables (descriptive stats + regression) and their placement in the run folder."""

import json
import os

import pandas as pd
import pytest
import yaml
from stats_transformer import Pipeline
from stats_transformer.reporting.tables.descriptive_stats import DescriptiveStatsTable
from stats_transformer.reporting.tables.regression_table import RegressionTable
from stats_transformer.reporting.tables.table_generator import TableGenerator

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _toy_frame():
    return pd.DataFrame({
        "price": [1.0, 2.0, 3.0, 4.0, 5.0],
        "x_lag1": [10.0, 20.0, None, 40.0, 50.0],
        "label": ["a", "b", "a", "b", "a"],  # non-numeric: must be excluded
    })


def _ols_metadata():
    return {
        "coefficients": {
            "const": {"value": 1.0, "std_err": 0.1, "p_value": 0.0},
            "educ_lag": {"value": 0.05, "std_err": 0.012, "p_value": 0.032},
            "expersq": {"value": -0.000688, "std_err": 0.000110, "p_value": 0.6},
            "tenure": {"value": -0.2, "std_err": None, "p_value": 0.5},
        },
        "summary": {"dependent_variable": "lwage", "nobs": 526.0, "rsquared": 0.42, "rsquared_adj": 0.41},
    }


def _var_metadata():
    return {
        "coefficients": {
            "realgdp": {"D.realgdp.L1": {"value": 0.5, "std_err": 0.1, "p_value": 0.001}},
        },
        "summary": {"variables": ["realgdp", "realcons"], "model_type": "VARResults"},
    }


def _gen(tmp_path):
    return TableGenerator(output_dir=tmp_path)


# --- booktabs output ---

def test_descriptive_stats_tex_has_booktabs_rules_and_escapes_underscores(tmp_path):
    table = DescriptiveStatsTable(table_generator=_gen(tmp_path))
    path = table.export_dataframe(_toy_frame(), "descriptive_stats")

    tex = path.read_text()
    body = tex.split("\\midrule", 1)[1]
    assert "\\toprule" in tex and "\\midrule" in tex and "\\bottomrule" in tex
    assert "x\\_lag1" in tex  # underscores escaped for LaTeX
    assert "price" in body
    assert "label" not in tex  # non-numeric column excluded
    assert "5" in body and "1,200" not in body  # N present


def test_regression_table_tex_has_stars_parenthesised_se_and_n_row(tmp_path):
    table = RegressionTable(table_generator=_gen(tmp_path))
    path = table.export(_ols_metadata(), "regression_table")

    tex = path.read_text()
    assert "\\toprule" in tex and "\\bottomrule" in tex
    assert "0.0500**" in tex  # star from the 0.05 threshold
    assert "\\midrule" in tex.split("N &", 1)[0]  # break between coefficients and N row
    assert "(0.0120)" in tex  # parenthesised SE below the coefficient
    assert "-0.000688" in tex and "(0.000110)" in tex  # 3 significant digits, never 0.000
    assert "educ\\_lag" in tex  # underscores escaped
    assert "526" in tex  # N row
    assert "0.420" in tex and "0.410" in tex
    # Constant last, note inside the tabular as the final row
    body_rows = [l for l in tex.splitlines() if l and not l.startswith("\\")]
    const_idx = next(i for i, l in enumerate(body_rows) if l.startswith("Constant"))
    n_idx = next(i for i, l in enumerate(body_rows) if l.startswith("N &"))
    assert const_idx < n_idx
    assert "\\par\\smallskip" not in tex
    assert r"\multicolumn{2}{l}{\footnotesize * p$<$0.1, ** p$<$0.05, *** p$<$0.01}" in tex
    assert tex.rindex("p$<$0.01") < tex.rindex("\\end{tabular}")


def test_regression_table_skips_non_regression_models(caplog, tmp_path):
    table = RegressionTable(table_generator=_gen(tmp_path))
    with caplog.at_level("INFO"):
        assert table.build(_var_metadata()) is None
    assert "Skipping regression table" in caplog.text


def test_regression_table_supports_two_models_side_by_side(tmp_path):
    table = RegressionTable(table_generator=_gen(tmp_path))
    combined = table.build([_ols_metadata(), _var_metadata()])
    assert list(combined.columns) == ["Variable", "lwage", "Model 2"]
    assert combined["lwage"].iloc[0] != ""  # regression contributes coefficients
    assert combined["Model 2"].iloc[0] == ""  # VAR column empty
    variables = list(combined["Variable"])
    assert variables.index("Constant") > variables.index("educ_lag")  # constant listed last among coefficients


# --- descriptive stats numbers ---

def test_from_dataframe_matches_pandas_describe():
    df = _toy_frame()
    stats = DescriptiveStatsTable.from_dataframe(df).set_index("Variable")

    desc = df.describe()
    for var in ["price", "x_lag1"]:
        assert stats.loc[var, "N"] == df[var].dropna().size
        assert stats.loc[var, "Mean"] == pytest.approx(desc.loc["mean", var])
        assert stats.loc[var, "SD"] == pytest.approx(desc.loc["std", var])
        assert stats.loc[var, "Min"] == pytest.approx(desc.loc["min", var])
        assert stats.loc[var, "Max"] == pytest.approx(desc.loc["max", var])


def test_from_dataframe_respects_requested_variables_and_non_numeric():
    df = _toy_frame()
    stats = DescriptiveStatsTable.from_dataframe(df, variables=["price"])
    assert stats["Variable"].tolist() == ["price"]
    assert "label" not in DescriptiveStatsTable.from_dataframe(df)["Variable"].tolist()

    assert DescriptiveStatsTable.from_dataframe(pd.DataFrame()).empty


# --- pipeline wiring ---

def _redirect_inputs(obj):
    if isinstance(obj, dict):
        return {k: _redirect_inputs(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_redirect_inputs(x) for x in obj]
    if isinstance(obj, str) and not obj.startswith("/") and os.path.isfile(os.path.join(REPO_ROOT, obj)):
        return os.path.join(REPO_ROOT, obj)
    return obj


def _prepare_config(tmp_path, rel_path):
    with open(os.path.join(REPO_ROOT, rel_path)) as f:
        doc = yaml.safe_load(f)
    doc = _redirect_inputs(doc)
    config_path = tmp_path / "params.yaml"
    config_path.write_text(yaml.dump(doc))
    return config_path


def _single_run_dir(output_root):
    stem_dirs = list(output_root.iterdir())
    run_dirs = [d for d in stem_dirs[0].iterdir() if d.is_dir()]
    assert len(run_dirs) == 1, f"expected exactly one run folder, got {run_dirs}"
    return run_dirs[0]


def test_mincer_full_run_writes_both_tables_and_charts_under_run_folder(tmp_path, monkeypatch):
    # Config has no reporting: key -> default generates both tables; charts move
    # into 04_reporting (no explicit output_dir configured).
    config_path = _prepare_config(tmp_path, "references/configs/regression/mincer_wage.yaml")
    monkeypatch.chdir(tmp_path)
    pipeline = Pipeline(params_path=str(config_path))
    pipeline.run()

    run_dir = _single_run_dir(tmp_path / "data" / "pipeline")
    tables_dir = run_dir / "04_reporting" / "tables"
    stats_tex = (tables_dir / "descriptive_stats.tex").read_text()
    reg_tex = (tables_dir / "regression_table.tex").read_text()

    # Dependent variable first, then regressors in model order
    assert stats_tex.index("lwage") < stats_tex.index("educ &") < stats_tex.index("exper") < stats_tex.index("expersq") < stats_tex.index("tenure")
    assert "\\toprule" in stats_tex
    assert "Constant &" in reg_tex  # constant row present
    assert "\\midrule" in reg_tex.split("N &", 1)[0]
    charts = list((run_dir / "04_reporting").rglob("*.png"))
    assert charts, "expected charts under 04_reporting"

    manifest = json.loads((run_dir / "manifest.json").read_text())
    for name in ["04_reporting/tables/descriptive_stats.tex", "04_reporting/tables/regression_table.tex"]:
        assert any(p.endswith(name) for p in manifest["files"])
    assert any(p.endswith(".png") for p in manifest["files"])


def test_mincer_full_run_honours_explicit_output_dir(tmp_path, monkeypatch):
    config_path = _prepare_config(tmp_path, "references/configs/regression/mincer_wage.yaml")
    with open(config_path) as f:
        doc = yaml.safe_load(f)
    doc["reporting"] = {"output_dir": str(tmp_path / "viz")}
    with open(config_path, "w") as f:
        yaml.dump(doc, f)
    monkeypatch.chdir(tmp_path)
    pipeline = Pipeline(params_path=str(config_path))
    pipeline.run()

    run_dir = _single_run_dir(tmp_path / "data" / "pipeline")
    charts = list((run_dir / "04_reporting").rglob("*.png"))
    explicit = list((tmp_path / "viz").rglob("*.png"))
    assert charts and explicit
    assert {p.name for p in charts} == {p.name for p in explicit}


def test_var_config_writes_descriptive_stats_only(tmp_path, monkeypatch):
    config_path = _prepare_config(tmp_path, "references/configs/timeseries/reduced_form/macro_var.yaml")
    monkeypatch.chdir(tmp_path)
    pipeline = Pipeline(params_path=str(config_path))
    pipeline.run()

    run_dir = _single_run_dir(tmp_path / "data" / "pipeline")
    tables_dir = run_dir / "04_reporting" / "tables"
    assert (tables_dir / "descriptive_stats.tex").is_file()
    assert not (tables_dir / "regression_table.tex").exists()

    stats_tex = (tables_dir / "descriptive_stats.tex").read_text()
    for var in ["realgdp", "realcons", "realinv"]:
        assert var in stats_tex

    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert any(p.endswith("descriptive_stats.tex") for p in manifest["files"])
    assert not any(p.endswith("regression_table.tex") for p in manifest["files"])