"""The paper's empirical-application artefacts are generated from real runs.

scripts/generate_paper_applications.py renders the applications table body,
per-application TikZ flow strips, and per-application verbatim config files.
These tests pin the contract the paper relies on: one row per application,
exact integer counts, exact integer counts with a thousands separator, flow
strips limited to four boxes (config, data, pipeline, outputs) with short node
labels, and per-application config files in table order.
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
    rows = [row for row in tex.splitlines() if row.endswith(" \\\\")]
    assert len(rows) == len(gen.APPLICATIONS) == 4
    assert all(row.endswith(" \\\\") for row in rows)
    for app in gen.APPLICATIONS:
        assert app["label"] in tex
    # Okun's law and the country-panel VAR are gone (and with them the
    # footnote machinery and the Key-estimates column).
    assert "Okun" not in tex and "okuns" not in tex
    assert "Country-panel" not in tex and "Python API" not in tex
    assert "\\multicolumn" not in tex and "\\dag" not in tex
    # Exact counts, never rounded or in scientific notation.
    for exact in ("526", "220", "202", "159"):
        assert exact in tex, tex
    assert tex.count("statsmodels OLS (HC3), same design matrix") == 1
    # Grunfeld two-way FE made it into the config and the generated row; the
    # parity caveat is disclosed.
    assert "entity + time FE" in tex
    assert "one-way FE parity" in tex
    # The row is the whole table body: no note rows, nothing but ' \\\\' lines.
    assert all(line.endswith(" \\\\") for line in tex.splitlines())
    # Config cells print basenames only; the appendix holds the full paths.
    assert "references/configs" not in tex
    assert tex.count("VAR(2), constant") == 1
    assert "VAR(8), constant" in tex
    # Output cells show bare file names, no stage folders (file names appear
    # inside \texttt cells, dots unescaped).
    assert "02\\_features" not in tex and "03\\_model" not in tex
    assert tex.count("features.parquet") == len(gen.APPLICATIONS)
    assert tex.count("model\\_summary.json") == len(gen.APPLICATIONS)
    # One flow strip and one verbatim config per application; 
    # application_configs.tex is no longer emitted.
    for name in (app["name"] for app in gen.APPLICATIONS):
        assert (out_dir / f"app_{name}_flow.tex").is_file()
        assert (out_dir / f"app_{name}_config.tex").is_file()
    assert not (out_dir / "application_configs.tex").exists()
    configs = [(out_dir / f"app_{app['name']}_config.tex").read_text() for app in gen.APPLICATIONS]
    assert "references" not in "".join(c.splitlines()[0] for c in configs)
    assert all(c.startswith("\\begin{verbatim}") for c in configs)


def test_estimates_have_three_significant_figures_and_math_wrapped_scientific_notation(tmp_path, monkeypatch):
    out_dir = run_generator(tmp_path, monkeypatch)
    tex = (out_dir / "applications.tex").read_text()
    # Numbers inside \texttt cells are identifiers (test names, file names),
    # not statistics; only numbers outside them follow the 3-sig-fig rule.
    tex = re.sub(r"\\texttt\{[^}]*\}", "", tex)
    estimates = COUNT.sub("", tex)  # counts are exact by test above
    estimates = re.sub(r"\(\d{4}\)", "", estimates)  # citation years (Blanchard & Quah 1989)
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
    # Four boxes: config, data, the pipeline, the outputs.
    nodes = [line for line in flow.splitlines() if line.startswith("\\node[box")]
    assert len(nodes) == 4
    assert "\\node[box] (n0)" in flow
    # The pipeline box is the third one.
    assert nodes[2].startswith("\\node[box, right=of n1] (n2) {stats-transformer pipeline};")
    # Every file the strip references exists in the run (manifest) or is an input.
    unescaped = flow.replace("\\_", "_")
    referenced = set(re.findall(r"[\w.-]+\.(?:parquet|json|csv|yaml)", unescaped))
    assert Path(record["data_rel"]).name in referenced
    assert {Path(f).name for f in record["run_files"]} <= referenced


def test_flow_nodes_use_short_labels(tmp_path, monkeypatch):
    for app in gen.APPLICATIONS:
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


def test_fmt_count_and_config_body_are_the_fixed_rounding_and_source_rules():
    assert gen.fmt_count(526) == "526"
    assert gen.fmt_count(11490) == "11{,}490"
    # The verbatim block is exactly the committed YAML file.
    app = gen.APPLICATIONS[0]
    assert gen.render_config(app) == f"\\begin{{verbatim}}\n{(gen.ROOT / app['config']).read_text().rstrip()}\n\\end{{verbatim}}"