"""Generate the paper's empirical-application artefacts from real pipeline runs.

For each YAML application under references/configs/ this script runs the full
pipeline in a temporary directory (CWD = tmp, config copy with input paths
redirected to the repo's data files) and renders, into --out-dir:

- applications.tex           booktabs table body rows ending in \\bottomrule (the paper owns \\begin{table})
- app_<name>_flow.tex        one 4-box flow strip per YAML application
                             (config -> data -> pipeline -> outputs)
- app_<name>_config.tex      the committed YAML of each application, so the
                             paper can \\input one per subsection

Everything in the .tex is generated from the runs (estimates from
03_model/model_summary.json); counts (Observations) are exact integers with a
thousands separator by fmt_count. applications table has five cells: Application, Data, Config, Estimator / SE,
Checked-against. The flow strips' last box says just "Outputs" (the paper does
not list run files).
"""

import argparse
import json
import os
import tempfile
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]

# Per-application metadata: descriptive labels (source, frequency, parity test).
# Numbers (N, effective observations, outputs) always come from the run.
APPLICATIONS = [
    {
        "name": "mincer_wage",
        "config": "references/configs/regression/mincer_wage.yaml",
        "label": "Mincer wage",
        "source": "Wooldridge wage1",
        "frequency": "cross-section",
        "checked": "Python: \\texttt{statsmodels}",
    },
    {
        "name": "grunfeld_panel",
        "config": "references/configs/regression/grunfeld_panel.yaml",
        "label": "Two-way FE panel",
        "source": "Grunfeld",
        "frequency": "annual",
        "checked": "R: \\texttt{plm::plm}; Stata: \\texttt{xtreg, fe}",
    },
    {
        "name": "macro_var",
        "config": "references/configs/timeseries/reduced_form/macro_var.yaml",
        "label": "Macro VAR",
        "source": "statsmodels macrodata",
        "frequency": "quarterly",
        "checked": "Stata: \\texttt{var}; R: \\texttt{vars::VAR}",
    },
    {
        "name": "blanchard_quah",
        "config": "references/configs/timeseries/identification/blanchard_quah_1989.yaml",
        "label": "Blanchard--Quah SVAR",
        "source": "Blanchard & Quah (1989)",
        "frequency": "quarterly",
        "checked": "MATLAB: VAR-Toolbox 4.0; R: \\texttt{vars::BQ}",
    },
]


def fmt_count(value) -> str:
    """Counts (observations, countries, ...) are exact integers with a thousands separator."""
    return f"{int(value):,}".replace(",", "{,}")


def _tex_escape(text: str) -> str:
    return text.replace("&", "\\&").replace("%", "\\%").replace("#", "\\#").replace("_", "\\_")


def _code(text: str) -> str:
    return f"\\texttt{{{_tex_escape(text)}}}"


def _redirect_inputs(obj):
    """Rewrite relative input paths in a config to absolute repo paths."""
    if isinstance(obj, dict):
        return {k: _redirect_inputs(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_redirect_inputs(v) for v in obj]
    if isinstance(obj, str) and not os.path.isabs(obj) and os.path.isfile(ROOT / obj):
        return str(ROOT / obj)
    return obj


def run_yaml_application(app: dict, run_root: Path) -> dict:
    """Run one YAML application's full pipeline with CWD inside run_root.

    Returns the run's model summary plus the manifest. Never writes outside
    run_root.
    """
    original_cwd = os.getcwd()
    work = run_root / app["name"]
    work.mkdir(parents=True)
    os.chdir(work)
    try:
        config_rel = app["config"]
        committed = yaml.safe_load((ROOT / config_rel).read_text())
        config = _redirect_inputs(committed)
        config_path = Path(config_rel).name
        Path(config_path).write_text(yaml.dump(config, sort_keys=False))

        from stats_transformer.pipeline import Pipeline

        Pipeline(params_path=config_path).run()

        run_dir = max((Path("data") / "pipeline" / Path(config_rel).stem).iterdir())
        manifest = json.loads((run_dir / "manifest.json").read_text())
        run_files = sorted(os.path.relpath(path, run_dir) for path in manifest["files"])
        summary = json.loads((run_dir / "03_model" / "model_summary.json").read_text())
        model_config = committed.get("model", {})
        return {
            "config_rel": config_rel,
            "data_rel": committed["data"]["raw_data_file"],
            "n_obs": len(pd.read_csv(ROOT / committed["data"]["raw_data_file"])),
            "model_type": model_config.get("model_type"),
            "maxlags": model_config.get("maxlags"),
            "cov_type": model_config.get("robust_ols", {}).get("cov_type", "HC3"),
            "summary": summary,
            "manifest": manifest,
            "run_files": run_files,
        }
    finally:
        os.chdir(original_cwd)


def build_records(run_root: Path) -> list[tuple[dict, dict]]:
    """Run every application; returns (app, record) pairs in table order."""
    return [(app, run_yaml_application(app, run_root)) for app in APPLICATIONS]


def _estimator_se(app: dict, record: dict) -> str:
    """Estimator / SE cell, generated from the config and the fit."""
    model_type = record["model_type"]
    if model_type == "robust_ols":
        return f"{_code(model_type)}, {_code(record['cov_type'])}"
    if model_type == "panel_ols":
        fe = record["summary"]["summary"]
        effects = " + ".join(e for e, on in (("entity", fe["entity_effects"]), ("time", fe["time_effects"])) if on)
        return f"{_code(model_type)}, {effects} FE"
    if model_type in ("var", "blanchard_quah"):
        return f"VAR({record['maxlags']}), constant"
    return _code(model_type)


def _data_cell(app: dict, record: dict) -> str:
    return f"{_tex_escape(app['source'])}, {fmt_count(record['n_obs'])} obs, {app['frequency']}"


def render_rows(records: list[tuple[dict, dict]]) -> str:
    """Booktabs body rows of the applications table, ending in \\bottomrule."""
    rows = []
    for app, record in records:
        config_cell = _code(Path(app["config"]).name)
        cells = [app["label"], _data_cell(app, record), config_cell, _estimator_se(app, record), app["checked"]]
        rows.append(" & ".join(cells) + " \\\\")
    return "\n".join(rows) + "\n\\bottomrule\n"


def render_flow(app: dict, record: dict) -> str:
    """Compact 4-box flow strip: config -> data -> pipeline -> outputs."""
    nodes = [_code(Path(app["config"]).name), _code(Path(record["data_rel"]).name), "stats-transformer pipeline", "Outputs"]
    lines = [
        f"% Generated by scripts/generate_paper_applications.py for {_tex_escape(app['config'])}:",
        "\\begin{tikzpicture}[>=Stealth, node distance=2mm,",
        "  box/.style={draw, rounded corners=1pt, font=\\tiny, align=center, inner sep=1pt, text width=3.2cm}]",
        f"\\node[box] (n0) {{{nodes[0]}}};",
    ]
    for i, label in enumerate(nodes[1:], start=1):
        lines.append(f"\\node[box, right=of n{i - 1}] (n{i}) {{{label}}};")
    for i in range(len(nodes) - 1):
        lines.append(f"\\draw[->] (n{i}) -- (n{i + 1});")
    lines.append("\\end{tikzpicture}")
    return "\n".join(lines) + "\n"


def render_config(app: dict) -> str:
    """Committed YAML of one application as a verbatim block (no heading)."""
    body = (ROOT / app["config"]).read_text().rstrip()
    return f"\\begin{{verbatim}}\n{body}\n\\end{{verbatim}}"


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", required=True, metavar="DIR", help="directory for the generated .tex files")
    args = parser.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="paper_applications_") as tmp:
        records = build_records(Path(tmp))

    (out_dir / "applications.tex").write_text(render_rows(records))
    for app, record in records:
        (out_dir / f"app_{app['name']}_flow.tex").write_text(render_flow(app, record))
        (out_dir / f"app_{app['name']}_config.tex").write_text(render_config(app) + "\n")
    print(f"Wrote applications.tex, {len(records)} flow strips and {len(records)} per-application configs to {out_dir}")


if __name__ == "__main__":
    main()