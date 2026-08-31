import pandas as pd
import pytest

from stats_transformer.data import (
    EXAMPLE_DATASETS,
    describe_example,
    get_sample_data_description,
    list_example_files,
    list_examples,
    load_example,
    load_sample_data,
)


EXPECTED_EXAMPLES = {
    "adrr_2018",
    "bauer_swanson_2023",
    "bbm_2023",
    "bq_1989",
    "fomc_surprises",
    "ghysels_ch1",
    "ghysels_ch2",
    "ghysels_ch6",
    "ghysels_ch7",
    "gk_2015",
    "global_factor",
    "greenbook_forecast_errors",
    "grunfeld",
    "jt_2025",
    "longley",
    "macrodata",
    "macrodb_gdp_inflation",
    "mincer_wage",
    "mroz",
    "nakamura_steinsson_2018",
    "news_sentiment",
    "okuns_law",
    "pmu_data",
    "policy_loans",
    "sofr_surprises",
    "spector_logit",
    "sw_2001",
    "synthetic_nakamura",
    "tariffs",
    "uhlig_2005",
    "var_panel",
}


def test_list_examples_includes_every_registered_dataset():
    assert set(list_examples()) == EXPECTED_EXAMPLES


def test_load_example_returns_packaged_dataframe():
    df = load_example("macrodb_gdp_inflation")

    assert not df.empty
    assert {"country", "date", "inflation", "gdp"}.issubset(df.columns)


@pytest.mark.parametrize("name", sorted(EXPECTED_EXAMPLES))
def test_every_registered_example_loads(name):
    loaded = load_example(name)
    frames = loaded.values() if isinstance(loaded, dict) else [loaded]

    assert all(isinstance(frame, pd.DataFrame) and not frame.empty for frame in frames)


def test_registry_covers_every_repository_data_file(project_root):
    actual_paths = {
        path.relative_to(project_root).as_posix()
        for path in (project_root / "data" / "examples").rglob("*")
        if path.is_file() and path.suffix != ".py"
    }
    actual_paths.add("src/stats_transformer/data/macrodb_gdp_inflation.parquet")
    registered_paths = {
        spec.get("repository_path", f"data/examples/{spec['path']}")
        for dataset in EXAMPLE_DATASETS.values()
        for spec in dataset["files"].values()
    }

    assert registered_paths == actual_paths


def test_multi_file_example_can_load_one_member():
    assert list_example_files("bauer_swanson_2023") == ["cpi", "nonfarm_payrolls", "unemployment"]

    cpi = load_example("bauer_swanson_2023", member="cpi")

    assert list(cpi.columns) == ["year", "month", "value"]
    assert not cpi.empty


def test_multi_file_example_requires_member_for_read_options():
    with pytest.raises(ValueError, match="require selecting member"):
        load_example("bauer_swanson_2023", nrows=2)


def test_load_example_rejects_unknown_member():
    with pytest.raises(ValueError, match="Unknown member"):
        load_example("bauer_swanson_2023", member="missing")


def test_sample_data_helpers_delegate_to_example_registry():
    sample = load_sample_data()
    metadata = get_sample_data_description()

    assert not sample.empty
    assert metadata == describe_example("macrodb_gdp_inflation")
    assert metadata["name"] == "macrodb_gdp_inflation"
    assert metadata["file_count"] == 1


def test_load_example_rejects_unknown_dataset():
    with pytest.raises(ValueError, match="Unknown example dataset"):
        load_example("missing")
