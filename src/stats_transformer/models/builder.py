"""Registry-driven construction of the pipeline's estimator."""

from typing import TYPE_CHECKING, Any, Dict

from stats_transformer.models.base import ModelBase
from stats_transformer.models.registry import MODEL_REGISTRY, MODEL_TYPE_ALIASES

if TYPE_CHECKING:
    from stats_transformer.pipeline import Pipeline


class ModelBuilder:
    """Builds the configured model from the registry, reading settings from its Pipeline."""

    def __init__(self, pipeline: "Pipeline") -> None:
        self._p = pipeline

    def resolve_registry_entry(self, model_type: str) -> Dict[str, Any]:
        """Resolve a (possibly aliased) model_type to its registry entry."""
        model_type = MODEL_TYPE_ALIASES.get(model_type, model_type)
        entry = MODEL_REGISTRY.get(model_type)
        if entry is None:
            valid = ", ".join(sorted(MODEL_REGISTRY))
            raise ValueError(f"Unknown model_type '{model_type}'. Valid values: {valid}")
        return entry

    def build_from_params(self, entry: Dict[str, Any]) -> ModelBase:
        """Instantiate the model for a params_path-configured pipeline."""
        cls, kind = entry["cls"], entry["kind"]
        if kind in ("iv", "svar_family", "lp_iv"):
            return cls(params_path=self._p.params_path)
        if kind in ("panel", "panel_iv"):
            return cls(params_path=self._p.params_path, entity_column=self._p.entity_column)
        if kind == "single_equation":
            return cls(params_path=self._p.params_path, add_entity_fixed_effects=self._p.add_entity_fixed_effects)
        if kind == "unsupervised":
            return cls(params_path=self._p.params_path, features=self._p.features)
        if kind == "lp":
            model_params = self._p._get_config().to_dict().get("model", {})
            shock_var = model_params.get("shock_variable") or (self._p.features[0] if self._p.features else None)
            controls = [f for f in (self._p.features or []) if f != shock_var]
            return cls(params_path=self._p.params_path, target=self._p.target, shock_var=shock_var, controls=controls, horizon=model_params.get("horizon", 8), date_column=model_params.get("date_column"))
        if kind == "did":
            model_params = self._p._get_config().to_dict().get("model", {})
            return cls(entity_column=self._p.entity_column, time_column=model_params.get("time_column", self._p.date_column), cohort_column=model_params.get("cohort_column"), outcome_column=model_params.get("target_variable", self._p.target), control_variables=model_params.get("independent_variables", self._p.features), control_group=model_params.get("control_group", "never_treated"))
        raise ValueError(f"Unhandled registry kind '{kind}' for {cls.__name__}")

    def build_from_args(self, entry: Dict[str, Any]) -> ModelBase:
        """Instantiate the model for a programmatically-configured pipeline."""
        cls, kind = entry["cls"], entry["kind"]
        p, kwargs = self._p, self._p.kwargs
        if kind == "single_equation":
            return cls(params_path=None, target=p.target, independent_variables=p.features, add_entity_fixed_effects=p.add_entity_fixed_effects, entity_column=p.entity_column)
        if kind == "panel":
            return cls(params_path=None, target=p.target, independent_variables=p.features, entity_column=p.entity_column)
        if kind == "iv":
            return cls(params_path=None, target=p.target, independent_variables=p.features, endogenous=kwargs.get("endogenous"), instruments=kwargs.get("instruments"))
        if kind == "panel_iv":
            return cls(params_path=None, target=p.target, independent_variables=p.features, endogenous=kwargs.get("endogenous"), instruments=kwargs.get("instruments"), entity_column=p.entity_column)
        if kind == "unsupervised":
            return cls(params_path=None, features=p.features)
        if kind == "svar_family":
            return cls(params_path=None, target_variables=[p.target] + (p.features or []))
        if kind == "lp_iv":
            return cls(params_path=None, target_variable=p.target, shock_variable=p.features[0] if p.features else None)
        if kind == "lp":
            shock_var = kwargs.get("shock_variable") or (p.features[0] if p.features else None)
            controls = [f for f in (p.features or []) if f != shock_var]
            return cls(params_path=None, target=p.target, shock_var=shock_var, controls=controls, horizon=kwargs.get("horizon", 8), date_column=kwargs.get("date_column"))
        if kind == "did":
            return cls(entity_column=p.entity_column, time_column=kwargs.get("time_column", p.date_column), cohort_column=kwargs.get("cohort_column"), outcome_column=p.target, control_variables=p.features, control_group=kwargs.get("control_group", "never_treated"))
        raise ValueError(f"Unhandled registry kind '{kind}' for {cls.__name__}")

    def has_model_target(self) -> bool:
        """True when the configured model defines its outcome(s) — via a single
        target (univariate) or target_variables on the model (multivariate:
        VAR/VECM/SVAR set target_variables instead of target)."""
        if self._p.target:
            return True
        return bool(getattr(self._p.model, "target_variables", None) or getattr(self._p.model, "target_variable", None))