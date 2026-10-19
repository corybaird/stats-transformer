import tomllib
from pathlib import Path

from stats_transformer.data.registry import EXAMPLE_DATASETS

# Walk up from this file to the checkout root (the dir holding pyproject.toml).
PROJECT_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())

def _registry_file_paths():
    paths = []
    for dataset in EXAMPLE_DATASETS.values():
        for spec in dataset["files"].values():
            if "repository_path" in spec:
                continue  # shipped inside the package itself, not force-included
            paths.append(spec["path"])
    return paths

def test_registry_datasets_force_included_in_sdist_and_wheel():
    pyproject = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text())
    targets = pyproject["tool"]["hatch"]["build"]["targets"]
    for target in ("sdist", "wheel"):
        keys = [key.removeprefix("data/examples/") for key in targets[target]["force-include"]]
        for path in _registry_file_paths():
            assert any(
                path == key or path.startswith(key.rstrip("/") + "/") for key in keys
            ), f"{path} is not force-included in the {target} build"