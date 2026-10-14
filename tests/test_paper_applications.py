"""The paper's empirical-application artefacts are generated from real runs.

scripts/generate_paper_applications.py renders the applications table body,
per-application TikZ flow strips, and an appendix of the committed YAML
configs. These tests pin the contract the paper relies on: one row per
application, no number with more than 3 significant figures, flow-strip files
taken from each run's manifest, and estimates equal to the model summary
values rounded by the same helper.
"""

import re
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import generate_paper_applications as gen

NUM = re.compile(r"-?\d+(?:\.\d+)?")


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
    rows = tex.splitlines()
    assert len(rows) == len(gen.APPLICATIONS) == 5
    assert all(row.endswith(" \\\\") for row in rows)
    for app in gen.APPLICATIONS:
        assert app["label"] in tex
    # Grunfeld two-way FE made it into the config and the generated row.
    assert "entity + time FE" in tex
    # The four YAML applications have flow strips; every config appears in the
    # appendix, in table order.
    for app in gen.APPLICATIONS:
        if app["config"]:
            assert (out_dir / f"app_{app['name']}_flow.tex").is_file()
            assert gen._tex_escape(app["config"]) in (out_dir / "application_configs.tex").read_text()


def test_no_number_exceeds_three_significant_figures(tmp_path, monkeypatch):
    out_dir = run_generator(tmp_path, monkeypatch)
    tex = (out_dir / "applications.tex").read_text()
    worst = max(sig_figs(match) for match in NUM.findall(tex))
    assert worst <= 3, tex


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
    referenced = set(re.findall(r"\d{2}_[a-z]+/[\w.]+", flow))
    manifest_files = set(record["run_files"])
    assert referenced <= manifest_files
    assert gen._tex_escape(record["data_rel"]) in flow


def test_key_estimates_equal_summary_values_rounded_by_fmt3(tmp_path, monkeypatch):
    app = gen.APPLICATIONS[0]
    monkeypatch.chdir(tmp_path)
    record = gen.run_yaml_application(app, tmp_path / "runs")
    row = gen.render_row(app, record)
    coefficients = record["summary"]["coefficients"]
    for term, _ in app["key"]:
        entry = coefficients[term]
        assert f"{gen.fmt(entry['value'])} ({gen.fmt(entry['std_err'])})" in row


def test_fmt3_is_the_fixed_rounding_rule():
    assert gen.fmt(0.08534893684706603) == "0.0853"
    assert gen.fmt(0.16986) == "0.17"
    assert gen.fmt(11490) == "1.15\\times 10^{4}"
    assert gen.fmt(220) == "220"