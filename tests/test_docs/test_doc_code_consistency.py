import importlib
import pkgutil
import re
import pytest
from pathlib import Path

import stats_transformer.models

DOCS_DIR = Path(__file__).parent.parent.parent / "docs"
MODELS_DOC = DOCS_DIR / "library" / "models.md"
ROADMAP_DOC = DOCS_DIR / "roadmap.md"
BENCHMARKS_DOC = DOCS_DIR / "validation" / "benchmarks.md"


def _importable_names():
    names = set()
    package = stats_transformer.models
    for module_info in pkgutil.walk_packages(package.__path__, prefix=package.__name__ + "."):
        try:
            module = importlib.import_module(module_info.name)
        except Exception:
            continue
        for attr in dir(module):
            if not attr.startswith("_"):
                names.add(attr)
    return names


def _benchmark_rows():
    rows = []
    in_section_2 = False
    for line in BENCHMARKS_DOC.read_text().splitlines():
        if "## 2. Software Parity Benchmarks" in line:
            in_section_2 = True
            continue
        if in_section_2:
            if "## 3." in line:
                break
            if not line.startswith("|") or line.startswith("| ---"):
                continue
            cells = [c.strip() for c in line.split("|")[1:-1]]
            if len(cells) < 5 or cells[1] == "Model":
                continue
            class_match = re.search(r'`([A-Za-z_][A-Za-z0-9_]*)`', cells[1])
            if not class_match:
                continue
            rows.append((class_match.group(1), cells[-1]))
    return rows


def test_benchmarks_doc_has_parseable_rows():
    rows = _benchmark_rows()
    assert len(rows) > 10, f"Parsed only {len(rows)} benchmark rows; the table format may have changed"


@pytest.mark.parametrize("class_name,status", [r for r in _benchmark_rows() if "Verified" in r[1]])
def test_benchmark_verified_classes_are_importable(class_name, status):
    assert class_name in _importable_names(), (
        f"benchmarks.md marks {class_name} as {status}, but it is not importable from "
        f"stats_transformer.models."
    )


def test_models_doc_sections_reference_real_classes():
    importable = _importable_names()
    problems = []
    for line in MODELS_DOC.read_text().splitlines():
        if not line.startswith("### "):
            continue
        for name in re.findall(r'`([A-Za-z_][A-Za-z0-9_]*)`', line):
            if name not in importable:
                problems.append(f"{line.strip()} -> `{name}` is not importable")
    assert not problems, "models.md references classes that do not exist:\n" + "\n".join(problems)
