import pytest

from stats_transformer.data import (
    describe_example,
    get_sample_data_description,
    list_examples,
    load_example,
    load_sample_data,
)


def test_list_examples_includes_packaged_macro_dataset():
    assert list_examples() == ["macrodb_gdp_inflation"]


def test_load_example_returns_packaged_dataframe():
    df = load_example("macrodb_gdp_inflation")

    assert not df.empty
    assert {"country", "date", "inflation", "gdp"}.issubset(df.columns)


def test_sample_data_helpers_delegate_to_example_registry():
    sample = load_sample_data()
    metadata = get_sample_data_description()

    assert not sample.empty
    assert metadata == describe_example("macrodb_gdp_inflation")
    assert metadata["name"] == "macrodb_gdp_inflation"


def test_load_example_rejects_unknown_dataset():
    with pytest.raises(ValueError, match="Unknown example dataset"):
        load_example("missing")
