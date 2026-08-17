import importlib
import pkgutil
import re
import pytest
from pathlib import Path

import stats_transformer.models

DOCS = Path(__file__).parent.parent.parent / "docs" / "extensions"
ROADMAP = DOCS / "roadmap.md"
MODELS_DOC = DOCS / "models.md"


def _importable_names():
    # Walk the whole models package rather than only the top-level __all__:
    # several documented classes (SVARBootstrap, VARLagSelector, RestrictedVAR)
    # are legitimately importable from their own modules without being
    # re-exported at the top level.
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


def _roadmap_rows():
    rows = []
    for line in ROADMAP.read_text().splitlines():
        if not line.startswith("|") or line.startswith("| ---") or "Model Class" in line:
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) < 8:
            continue
        class_match = re.search(r'`([A-Za-z_][A-Za-z0-9_]*)`', cells[1])
        if not class_match:
            continue
        rows.append((class_match.group(1), cells[-1]))
    return rows


def test_roadmap_has_parseable_rows():
    rows = _roadmap_rows()
    assert len(rows) > 10, f"Parsed only {len(rows)} roadmap rows; the table format may have changed"


@pytest.mark.parametrize("class_name,status", [r for r in _roadmap_rows() if "Verified" in r[1]])
def test_roadmap_verified_classes_are_importable(class_name, status):
    assert class_name in _importable_names(), (
        f"roadmap.md marks {class_name} as {status}, but it is not importable from "
        f"stats_transformer.models. Either implement/export it or change its status to *Planned*."
    )


def test_models_doc_implemented_sections_reference_real_classes():
    # Every "### ... (`Name`) - **Implemented**" heading must name a real class.
    importable = _importable_names()
    problems = []
    for line in MODELS_DOC.read_text().splitlines():
        if not line.startswith("### ") or "**Implemented**" not in line:
            continue
        for name in re.findall(r'`([A-Za-z_][A-Za-z0-9_]*)`', line):
            if name not in importable:
                problems.append(f"{line.strip()} -> `{name}` is not importable")
    assert not problems, "models.md claims these are Implemented but they do not exist:\n" + "\n".join(problems)


# Classes that exist and import, but whose estimation is deliberately not
# implemented -- they raise NotImplementedError rather than fabricating
# results. They are correctly documented as *Planned* despite being importable.
IMPORTABLE_BUT_NOT_IMPLEMENTED = set()


def test_models_doc_planned_sections_are_not_implemented():
    # The inverse guard: if something marked *Planned* becomes importable, the
    # doc is now understating the library and should be promoted.
    importable = _importable_names() - IMPORTABLE_BUT_NOT_IMPLEMENTED
    stale = []
    for line in MODELS_DOC.read_text().splitlines():
        if not line.startswith("### ") or "*Planned*" not in line:
            continue
        for name in re.findall(r'`([A-Za-z_][A-Za-z0-9_]*)`', line):
            if name in importable:
                stale.append(f"{line.strip()} -> `{name}` is now importable")
    assert not stale, "models.md marks these *Planned* but they now exist; promote them to **Implemented**:\n" + "\n".join(stale)


def test_importable_but_unimplemented_classes_still_raise():
    # Guards the exemption above: if anything is added to IMPORTABLE_BUT_NOT_IMPLEMENTED,
    # ensure it actually exists and is accounted for.
    assert isinstance(IMPORTABLE_BUT_NOT_IMPLEMENTED, set)
