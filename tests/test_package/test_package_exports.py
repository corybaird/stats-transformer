import warnings

import stats_transformer

def test_all_names_importable_from_package():
    for name in stats_transformer.__all__:
        assert hasattr(stats_transformer, name), f"{name!r} listed in __all__ but not importable from the package"

def test_all_is_trimmed():
    assert stats_transformer.__all__ == [
        "Pipeline",
        "FeatureEngineer",
        "EventStudyBuilder",
        "load_example",
        "__version__",
    ]

def test_deprecated_name_resolves_with_warning_from_models():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        var_model = stats_transformer.VARModel
    assert var_model is stats_transformer.models.VARModel
    messages = [str(w.message) for w in caught if issubclass(w.category, DeprecationWarning)]
    assert messages == [
        "stats_transformer.VARModel is deprecated; import it from stats_transformer.models"
    ]

def test_deprecated_reporting_name_resolves_with_warning():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        table_generator = stats_transformer.TableGenerator
    assert table_generator is stats_transformer.reporting.tables.TableGenerator
    messages = [str(w.message) for w in caught if issubclass(w.category, DeprecationWarning)]
    assert messages == [
        "stats_transformer.TableGenerator is deprecated; import it from stats_transformer.reporting.tables"
    ]

def test_unknown_name_raises_attribute_error():
    try:
        stats_transformer.NotARealExport
    except AttributeError:
        pass
    else:
        raise AssertionError("expected AttributeError for unknown name")