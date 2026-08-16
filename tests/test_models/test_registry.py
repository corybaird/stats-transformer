import pytest
import stats_transformer.models as models_module
from stats_transformer.models.base import ModelBase
from stats_transformer.models.registry import MODEL_REGISTRY, MODEL_TYPE_ALIASES, NOT_PIPELINE_EXPOSED


def _all_model_base_subclasses(cls=ModelBase):
    subclasses = set()
    for subclass in cls.__subclasses__():
        subclasses.add(subclass)
        subclasses.update(_all_model_base_subclasses(subclass))
    return subclasses


def test_every_model_base_subclass_is_registered_or_explicitly_excluded():
    import stats_transformer.pipeline  # noqa: F401  ensure every model module is imported

    registered_classes = {entry["cls"] for entry in MODEL_REGISTRY.values()}
    unaccounted = []
    for subclass in _all_model_base_subclasses():
        if subclass in registered_classes:
            continue
        if subclass.__name__ in NOT_PIPELINE_EXPOSED:
            continue
        unaccounted.append(subclass.__name__)

    assert not unaccounted, (
        f"ModelBase subclasses not in MODEL_REGISTRY and not in NOT_PIPELINE_EXPOSED: {unaccounted}. "
        "Register them in stats_transformer/models/registry.py or add them to NOT_PIPELINE_EXPOSED with a reason."
    )


def test_registry_kinds_are_all_handled():
    from stats_transformer.pipeline import Pipeline
    handled_kinds = {"single_equation", "panel", "iv", "panel_iv", "unsupervised", "svar_family", "lp", "lp_iv"}
    registry_kinds = {entry["kind"] for entry in MODEL_REGISTRY.values()}
    assert registry_kinds <= handled_kinds, f"Registry uses kinds not handled by Pipeline: {registry_kinds - handled_kinds}"


def test_aliases_point_at_real_registry_keys():
    for alias, target in MODEL_TYPE_ALIASES.items():
        assert target in MODEL_REGISTRY, f"Alias '{alias}' points at unregistered model_type '{target}'"


def test_not_pipeline_exposed_entries_are_real_classes():
    known_names = {subclass.__name__ for subclass in _all_model_base_subclasses()}
    for name in NOT_PIPELINE_EXPOSED:
        assert name in known_names, f"NOT_PIPELINE_EXPOSED lists '{name}', which is not a known ModelBase subclass"
