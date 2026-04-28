import pytest
from stats_transformer.utils.config import Config

def test_config_initialization():
    config = Config()
    assert config.config == {}
    
    config = Config(key="value", another=123)
    assert config.get("key") == "value"
    assert config.get("another") == 123

def test_config_get_set():
    config = Config()
    config.set("test_key", "test_value")
    assert config.get("test_key") == "test_value"
    
    # Nested
    config.set("nested.key", "nested_value")
    assert config.get("nested.key") == "nested_value"

def test_config_sections():
    config = Config(
        featurization={"feature1": True},
        model={"type": "rf"},
        visualization={"theme": "dark"}
    )
    
    assert config.get_featurization_config()["feature1"] is True
    assert config.get_model_config()["type"] == "rf"
    assert config.get_visualization_config()["theme"] == "dark"
