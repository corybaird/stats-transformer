"""Tests that the model validation status docs stay in sync with model_status.yaml."""

import importlib.util
import re
from pathlib import Path

import yaml

from stats_transformer.utils.config import Config

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs"
BENCHMARKS_DOC = DOCS / "validation" / "benchmarks.md"
README = ROOT / "README.md"

_spec = importlib.util.spec_from_file_location("generate_model_status", ROOT / "scripts" / "generate_model_status.py")
generator = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(generator)

ENTRIES = yaml.safe_load((DOCS / "validation" / "model_status.yaml").read_text())


def _generated_section():
    text = BENCHMARKS_DOC.read_text()
    start = text.index(generator.BEGIN_MARK) + len(generator.BEGIN_MARK)
    end = text.index(generator.END_MARK)
    return text[start:end].strip("\n")


def test_benchmarks_verified_section_matches_generator():
    """Hand-editing the generated section of benchmarks.md fails this test."""
    assert _generated_section() == generator.render_markdown(ENTRIES)


def test_every_table1_row_is_rendered_in_latex():
    tex = generator.render_tex(ENTRIES)
    for entry in ENTRIES:
        for model in entry["models"]:
            assert f"\\texttt{{{model}}}" in tex, model
        assert entry["family"] in tex


def test_verified_and_known_issue_entries_name_existing_tests():
    for entry in ENTRIES:
        if entry["status"] not in ("verified", "known_issue"):
            continue
        assert entry.get("tests"), f"{entry['models']} claims status '{entry['status']}' without a backing test"
        for node_id in entry["tests"]:
            path, _, function = node_id.partition("::")
            test_file = ROOT / path
            assert test_file.is_file(), f"missing test file for {node_id}"
            name = function.split("[")[0]
            assert re.search(rf"^def {name}\(", test_file.read_text(), re.MULTILINE), node_id


def test_yaml_statuses_are_known():
    for entry in ENTRIES:
        assert entry["status"] in generator.STATUS_WORDS, entry["models"]
    models = {model for entry in ENTRIES for model in entry["models"]}
    assert "PandemicBVARModel" in models  # missing from the paper's Table 1 before


def test_readme_quickstart_yaml_loads_and_points_at_real_data(tmp_path):
    text = README.read_text()
    start = text.index("#### 3.2.1.")
    fence_start = text.index("```yaml", start) + len("```yaml\n")
    fence_end = text.index("```", fence_start)
    config_path = tmp_path / "params.yaml"
    config_path.write_text(text[fence_start:fence_end])
    config = Config(config_path=str(config_path))
    config.validate()
    raw = config.get("data.raw_data_file")
    assert raw and (ROOT / raw).is_file(), f"quickstart data file does not exist: {raw}"