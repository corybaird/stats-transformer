import pytest
from stats_transformer.utils.dict_country_converter import dict_country

def test_dict_country_converter():
    assert "iso2_to_iso3" in dict_country
    assert "iso3_to_iso2" in dict_country
    assert "iso2_to_name_short" in dict_country
    assert "iso3_to_name_short" in dict_country
    
    # Check some basic mappings
    assert dict_country["iso2_to_iso3"]["US"] == "USA"
    assert dict_country["iso3_to_iso2"]["USA"] == "US"
    assert dict_country["iso2_to_iso3"]["AF"] == "AFG"
    assert dict_country["iso3_to_iso2"]["AFG"] == "AF"
