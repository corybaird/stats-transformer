"""The paper's empirical-application artefacts are generated from real runs.

scripts/generate_paper_applications.py renders the applications table body,
per-application TikZ flow strips, and an appendix of the committed YAML
configs. These tests pin the contract the paper relies on: one row per
application, exact integer counts, estimates/statistics rounded to at most
3 significant figures (scientific notation only inside math mode), flow-strip
files taken from each run's manifest, short TikZ node labels, and estimates
equal to the model summary values rounded by the same helper.
"""

import re
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import generate_paper_applications as gen

NUM = re.compile(r"-?\d+(?:\.\d+)?")
COUNT = re.compile(r"\d{1,3}(?:\{,\}\d{3})+")  # exact counts: excluded from the 3-sig-fig rule


def sig_figs(token: str) -> int:
    """Count significant digits: leading zeros are not significant."""
    return len(token.lstrip("-").replace(".", "").lstrip("0"))


def run_generator(tmp_path, monkeypatch) -> Path:
    out_dir = tmp_path / "tex"
    monkeypatch.chdir(tmp_path)  # any stray pipeline writes must not reach the repo
    gen.main(["--out-dir", str(out_dir)])
    return out_dir


def test_generator_writes_one_row_per_application(tmp_path, monkeypatch):
    out_dir = run_generator(tmp_path, monkeypatch)
    tex = (out_dir / "applications.tex").read_text()
    rows = [row for row in tex.splitlines() if not row.startswith("\\multicolumn")]
    assert len(rows) == len(gen.APPLICATIONS) == 5
    assert all(row.endswith(" \\\\") for row in rows)
    for app in gen.APPLICATIONS:
        assert app["label"] in tex
    # Exact counts, never rounded or in scientific notation.
    for exact in ("526", "304", "220", "202", "11{,}490", "207", "225", "223"):
        assert exact in tex, tex
    # Grunfeld two-way FE made it into the config and the generated row; the
    # parity caveat is disclosed.
    assert "entity + time FE" in tex
    assert "one-way FE parity" in tex
    # Mincer/Okun state what was actually compared, not the Longley benchmark.
    assert tex.count("statsmodels OLS (HC3), same design matrix") == 2
    # The Macro VAR row is marked and a note row discloses the API fallback.
    assert "\\dag~Macro VAR key estimates were fitted through the Python API" in tex
    assert "multivariate configs" in tex
    # Config cells print basenames only; the appendix holds the full paths.
    assert "references/configs" not in tex
    assert tex.count("VAR(2), constant") == 2
    # The four YAML applications have flow strips; every config appears in the
    # appendix, in table order.
    for app in gen.APPLICATIONS:
        if app["config"]:
            assert (out_dir / f"app_{app['name']}_flow.tex").is_file()
            assert gen._tex_escape(app["config"]) in (out_dir / "application_configs.tex").read_text()


def test_estimates_have_three_significant_figures_and_math_wrapped_scientific_notation(tmp_path, monkeypatch):
    out_dir = run_generator(tmp_path, monkeypatch)
    tex = (out_dir / "applications.tex").read_text()
    estimates = COUNT.sub("", tex)  # counts are exact by test above
    worst = max(sig_figs(match) for match in NUM.findall(estimates))
    assert worst <= 3, tex
    # \times, ^ and any minus directly before a digit may appear only in math mode.
    for match in re.finditer(r"\\times|\^|-(?=\d)", tex):
        assert tex[: match.start()].count("$") % 2 == 1, tex


def test_flow_files_come_from_the_manifest(tmp_path, monkeypatch):
    app = gen.APPLICATIONS[0]  # mincer wage
    monkeypatch.chdir(tmp_path)
    record = gen.run_yaml_application(app, tmp_path / "runs")
    flow = gen.render_flow(app, record)
    assert flow.count("{") == flow.count("}")
    # Stages that ran come from manifest.stages_run, in order.
    for stage in record["manifest"]["stages_run"]:
        assert stage in flow
    # Every file the strip references exists in the run (manifest) or is an input.
    unescaped = flow.replace("\\_", "_")
    referenced = set(re.findall(r"[\w.-]+\.(?:parquet|json|joblib|csv|yaml)", unescaped))
    assert Path(record["data_rel"]).name in referenced
    assert {Path(f).name for f in record["run_files"]} <= referenced


def test_flow_nodes_use_short_labels(tmp_path, monkeypatch):
    for app in gen.APPLICATIONS:
        if not app["config"]:
            continue  # flow strips need a manifest; the API application has none
        monkeypatch.chdir(tmp_path)
        record = gen.run_yaml_application(app, tmp_path / app["name"])
        flow = gen.render_flow(app, record)
        for line in flow.splitlines():
            match = re.match(r"\\node\[box.*\] \(n\d+\) \{(.*)\};$", line)
            if not match:
                continue
            label = match.group(1)
            for part in label.replace("\\\\", "\n").split("\n"):
                visible = (
                    part.replace("\\texttt{", "").replace("\\_", "_").replace("\\&", "&").replace("{", "").replace("}", "")
                )
                assert len(visible) <= 28, (app["name"], part)


def test_key_estimates_equal_summary_values_rounded_by_fmt3(tmp_path, monkeypatch):
    app = gen.APPLICATIONS[0]
    monkeypatch.chdir(tmp_path)
    record = gen.run_yaml_application(app, tmp_path / "runs")
    row = gen.render_rows([(app, record)])
    coefficients = record["summary"]["coefficients"]
    for term, _ in app["key"]:
        entry = coefficients[term]
        assert f"{gen.fmt(entry['value'])} ({gen.fmt(entry['std_err'])})" in row


def test_fmt3_and_counts_are_the_fixed_rounding_rules():
    assert gen.fmt(0.08534893684706603) == "$0.0853$"
    assert gen.fmt(0.16986) == "$0.17$"
    assert gen.fmt(-0.474) == "$-0.474$"
    assert gen.fmt(1.234e-05) == "$1.23\\times 10^{-5}$"
    assert gen.fmt(220) == "$220$"
    assert gen.fmt_count(526) == "526"
    assert gen.fmt_count(11490) == "11{,}490"