"""Data utilities for the stats-transformer library."""

from contextlib import contextmanager
from importlib.resources import as_file, files
from pathlib import Path

import pandas as pd

from stats_transformer.data.registry import EXAMPLE_DATASETS


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


def list_examples():
    """List registered example datasets."""
    return sorted(EXAMPLE_DATASETS)


def _get_example(name):
    if name not in EXAMPLE_DATASETS:
        available = ", ".join(list_examples())
        raise ValueError(f"Unknown example dataset '{name}'. Available examples: {available}")
    return EXAMPLE_DATASETS[name]


def list_example_files(name):
    """List the named files belonging to an example dataset."""
    return sorted(_get_example(name)["files"])


def describe_example(name="macrodb_gdp_inflation"):
    """Return metadata for a registered example dataset."""
    dataset = _get_example(name)
    metadata = {key: value for key, value in dataset.items() if key != "files"}
    metadata["files"] = {
        member: spec.get("repository_path", f"data/examples/{spec['path']}")
        for member, spec in dataset["files"].items()
    }
    metadata["file_count"] = len(dataset["files"])
    metadata["name"] = name
    return metadata


@contextmanager
def _example_path(spec):
    repository_path = REPOSITORY_ROOT / spec.get("repository_path", f"data/examples/{spec['path']}")
    if repository_path.is_file():
        yield repository_path
        return

    package_path = spec.get("package_path", f"examples/{spec['path']}")
    resource = files("stats_transformer.data")
    for part in package_path.split("/"):
        resource = resource.joinpath(part)
    if resource.is_file():
        with as_file(resource) as materialized_path:
            yield materialized_path
        return

    raise FileNotFoundError(
        f"Example data file '{spec['path']}' was not found in the repository or installed package."
    )


def _read_example_file(spec, read_kwargs):
    path_name = spec["path"].lower()
    options = dict(spec.get("read_kwargs", {}))
    options.update(read_kwargs)
    if path_name.endswith(".txt"):
        options = {"sep": r"\s+", "header": None, "names": ["year", "month", "value"], **options}

    with _example_path(spec) as data_path:
        if path_name.endswith(".parquet") or path_name.endswith(".parquet.gzip"):
            return pd.read_parquet(data_path, **options)
        if path_name.endswith(".csv") or path_name.endswith(".txt"):
            return pd.read_csv(data_path, **options)
        if path_name.endswith(".dta"):
            return pd.read_stata(data_path, **options)
        if path_name.endswith(".xlsx") or path_name.endswith(".xls"):
            return pd.read_excel(data_path, **options)
    raise ValueError(f"Unsupported example data format: {spec['path']}")


def load_example(name="macrodb_gdp_inflation", member=None, **read_kwargs):
    """Load one registered dataset or one of its named files."""
    dataset = _get_example(name)
    dataset_files = dataset["files"]

    if member is not None:
        if member not in dataset_files:
            available = ", ".join(sorted(dataset_files))
            raise ValueError(f"Unknown member '{member}' for example '{name}'. Available members: {available}")
        return _read_example_file(dataset_files[member], read_kwargs)

    if len(dataset_files) == 1:
        return _read_example_file(next(iter(dataset_files.values())), read_kwargs)

    if read_kwargs:
        raise ValueError("read options for a multi-file dataset require selecting member=...")
    return {
        file_name: _read_example_file(spec, {})
        for file_name, spec in dataset_files.items()
    }


def load_sample_data():
    """Load the default sample macroeconomic dataset.

    Returns:
        DataFrame with columns: country, date, inflation, gdp

    Example:
        >>> from stats_transformer.data import load_sample_data
        >>> df = load_sample_data()
        >>> print(df.head())
    """
    return load_example("macrodb_gdp_inflation")


def get_sample_data_description():
    """Get description of the sample dataset.

    Returns:
        Dictionary with dataset description
    """
    return describe_example("macrodb_gdp_inflation")


__all__ = [
    "EXAMPLE_DATASETS",
    "describe_example",
    "get_sample_data_description",
    "list_example_files",
    "list_examples",
    "load_example",
    "load_sample_data",
]
