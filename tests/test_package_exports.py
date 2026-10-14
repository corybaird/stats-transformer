import stats_transformer

def test_all_names_importable_from_package():
    for name in stats_transformer.__all__:
        assert hasattr(stats_transformer, name), f"{name!r} listed in __all__ but not importable from the package"