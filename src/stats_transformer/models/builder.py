"""Registry-driven construction of the pipeline's estimator."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from stats_transformer.models.base import ModelBase
from stats_transformer.models.registry import MODEL_REGISTRY, MODEL_TYPE_ALIASES


@dataclass(frozen=True)
class ModelSpec:
    """Everything needed to construct a pipeline model, from config or programmatic arguments."""

    params_path: Optional[str] = None
    model_config: Dict[str, Any] = field(default_factory=dict)
    target: Optional[str] = None
    features: Optional[List[str]] = None
    entity_column: Optional[str] = None
    date_column: str = "date"
    add_entity_fixed_effects: bool = False
    kwargs: Dict[str, Any] = field(default_factory=dict)


def has_model_target(target: Optional[str], model: Optional[ModelBase]) -> bool:
    """True when the configured model defines its outcome(s) — via a single
    target (univariate) or target_variables on the model (multivariate:
    VAR/VECM/SVAR set target_variables instead of target)."""
    if target:
        return True
    return bool(getattr(model, "target_variables", None) or getattr(model, "target_variable", None))


class ModelBuilder:
    """Builds the configured model from a ModelSpec via the registry."""

    def __init__(self, spec: ModelSpec) -> None:
        self.spec = spec

    def build(self, model_type: str) -> ModelBase:
        """Resolve a (possibly aliased) model_type in the registry and construct the model."""
        model_type = MODEL_TYPE_ALIASES.get(model_type, model_type)
        entry = MODEL_REGISTRY.get(model_type)
        if entry is None:
            valid = ", ".join(sorted(MODEL_REGISTRY))
            raise ValueError(f"Unknown model_type '{model_type}'. Valid values: {valid}")
        if self.spec.params_path:
            return self._from_params(entry)
        return self._from_args(entry)

    def _from_params(self, entry: Dict[str, Any]) -> ModelBase:
        """Instantiate the model for a params_path-configured pipeline."""
        spec = self.spec
        cls, kind = entry["cls"], entry["kind"]
        if kind in ("iv", "svar_family", "lp_iv"):
            return cls(params_path=spec.params_path)
        if kind in ("panel", "panel_iv"):
            return cls(params_path=spec.params_path, entity_column=spec.entity_column)
        if kind == "single_equation":
            return cls(
                params_path=spec.params_path,
                add_entity_fixed_effects=spec.add_entity_fixed_effects,
            )
        if kind == "unsupervised":
            return cls(params_path=spec.params_path, features=spec.features)
        if kind == "lp":
            model_params = spec.model_config
            shock_var = model_params.get("shock_variable") or (spec.features[0] if spec.features else None)
            controls = [f for f in (spec.features or []) if f != shock_var]
            return cls(
                params_path=spec.params_path,
                target=spec.target,
                shock_var=shock_var,
                controls=controls,
                horizon=model_params.get("horizon", 8),
                date_column=model_params.get("date_column"),
            )
        if kind == "did":
            model_params = spec.model_config
            return cls(
                entity_column=spec.entity_column,
                time_column=model_params.get("time_column", spec.date_column),
                cohort_column=model_params.get("cohort_column"),
                outcome_column=model_params.get("target_variable", spec.target),
                control_variables=model_params.get("independent_variables", spec.features),
                control_group=model_params.get("control_group", "never_treated"),
            )
        raise ValueError(f"Unhandled registry kind '{kind}' for {cls.__name__}")

    def _from_args(self, entry: Dict[str, Any]) -> ModelBase:
        """Instantiate the model for a programmatically-configured pipeline."""
        spec, kwargs = self.spec, self.spec.kwargs
        cls, kind = entry["cls"], entry["kind"]
        if kind == "single_equation":
            return cls(
                params_path=None,
                target=spec.target,
                independent_variables=spec.features,
                add_entity_fixed_effects=spec.add_entity_fixed_effects,
                entity_column=spec.entity_column,
            )
        if kind == "panel":
            return cls(
                params_path=None,
                target=spec.target,
                independent_variables=spec.features,
                entity_column=spec.entity_column,
            )
        if kind == "iv":
            return cls(
                params_path=None,
                target=spec.target,
                independent_variables=spec.features,
                endogenous=kwargs.get("endogenous"),
                instruments=kwargs.get("instruments"),
            )
        if kind == "panel_iv":
            return cls(
                params_path=None,
                target=spec.target,
                independent_variables=spec.features,
                endogenous=kwargs.get("endogenous"),
                instruments=kwargs.get("instruments"),
                entity_column=spec.entity_column,
            )
        if kind == "unsupervised":
            return cls(params_path=None, features=spec.features)
        if kind == "svar_family":
            return cls(params_path=None, target_variables=[spec.target] + (spec.features or []))
        if kind == "lp_iv":
            shock_var = spec.features[0] if spec.features else None
            return cls(params_path=None, target_variable=spec.target, shock_variable=shock_var)
        if kind == "lp":
            shock_var = kwargs.get("shock_variable") or (spec.features[0] if spec.features else None)
            controls = [f for f in (spec.features or []) if f != shock_var]
            return cls(
                params_path=None,
                target=spec.target,
                shock_var=shock_var,
                controls=controls,
                horizon=kwargs.get("horizon", 8),
                date_column=kwargs.get("date_column"),
            )
        if kind == "did":
            return cls(
                entity_column=spec.entity_column,
                time_column=kwargs.get("time_column", spec.date_column),
                cohort_column=kwargs.get("cohort_column"),
                outcome_column=spec.target,
                control_variables=spec.features,
                control_group=kwargs.get("control_group", "never_treated"),
            )
        raise ValueError(f"Unhandled registry kind '{kind}' for {cls.__name__}")