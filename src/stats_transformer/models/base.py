import json
import logging
import os
import pandas as pd
import sys
import yaml
from datetime import datetime
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

class ModelBase(ABC):

    time_column: Optional[str] = None
    # Multivariate (system) models have no dependent/independent split: every
    # variable is endogenous. They set this True to opt out of the
    # single-equation target/independent_variables contract, and report a
    # symmetric "variables" list in metadata instead.
    _is_multivariate: bool = False

    def __init__(self, params_path: Optional[str] = None, target: Optional[str] = None, independent_variables: Optional[List[str]] = None, add_entity_fixed_effects: bool = False, entity_column: Optional[str] = None, **kwargs: Any) -> None:
        self._setup_logging()
        self.params: Dict[str, Any] = {}

        if params_path:
            self.params = self._load_params(params_path)
            model_params = self.params.get("model", {})
            self.target = model_params.get("target_variable", target)
            self.independent_variables = model_params.get("independent_variables", independent_variables or [])
            self.target_variables = model_params.get("target_variables") or self.independent_variables or []
            self.add_entity_fixed_effects = model_params.get("ols", {}).get("add_entity_fixed_effects", add_entity_fixed_effects)
            feat_params = self.params.get("data", {}).get("featurization", self.params.get("featurization", {}))
            self.entity_column = feat_params.get("entity_column", entity_column)
        else:
            self.target = target
            self.independent_variables = independent_variables or []
            self.add_entity_fixed_effects = add_entity_fixed_effects
            self.entity_column = entity_column
            if kwargs:
                self.params.update(kwargs)

        if not self._is_multivariate:
            if not self.target:
                raise ValueError("Target variable must be specified")
            if not self.independent_variables:
                raise ValueError("Independent variables must be specified")
            
        self.df: Optional[pd.DataFrame] = None
        self.df_clean: Optional[pd.DataFrame] = None
        self.X: Optional[pd.DataFrame] = None
        self.y: Optional[Union[pd.Series, pd.DataFrame]] = None
        self.model: Optional[Any] = None
        self.model_version = datetime.now().strftime("%Y%m%d_%H%M%S")

    def _is_running_in_jupyter(self) -> bool:
        try:
            from IPython import get_ipython
            return get_ipython() is not None
        except ImportError:
            return False

    def _setup_logging(self) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)
        if self._is_running_in_jupyter():
            self.logger.setLevel(logging.CRITICAL + 1)
            for handler in self.logger.handlers[:]:
                self.logger.removeHandler(handler)
            self.logger.addHandler(logging.NullHandler())
        else:
            self.logger.setLevel(logging.INFO)
            if not self.logger.handlers:
                handler = logging.StreamHandler(sys.stdout)
                handler.setLevel(logging.INFO)
                handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
                self.logger.addHandler(handler)

    def _load_params(self, params_path: str) -> Dict[str, Any]:
        try:
            with open(params_path, "r") as f:
                params = yaml.safe_load(f)
            return params
        except FileNotFoundError:
            raise FileNotFoundError(f"Parameter file {params_path} not found.")

    def load_data(self, data: Union[str, pd.DataFrame]) -> pd.DataFrame:
        self.logger.info("Loading data")
        if isinstance(data, str):
            self.df = pd.read_csv(data) if data.endswith(".csv") else pd.read_parquet(data)
        else:
            self.df = data.copy()

        required_columns = self._get_required_columns()
        missing_columns = [col for col in required_columns if col not in self.df.columns]
        if missing_columns:
            raise ValueError(f"Required columns missing: {missing_columns}")

        # Models that declare a time_column keep it as a column: they sort and
        # index by it themselves, and some (e.g. narrative restrictions) read
        # it back by name. Everything else preserves the historical behavior of
        # consuming a column literally named "date" into the index.
        index_col = None if getattr(self, "time_column", None) else ('date' if 'date' in self.df.columns else None)

        if self.entity_column and self.entity_column in self.df.columns and index_col:
            if self.add_entity_fixed_effects and self.entity_column in required_columns:
                self.df_clean = self.df[required_columns].dropna()
            else:
                cols = [c for c in required_columns if c not in [self.entity_column, index_col]]
                self.df_clean = self.df.set_index([self.entity_column, index_col])[cols].dropna()
        elif index_col:
            cols = [c for c in required_columns if c != index_col]
            self.df_clean = self.df.set_index([index_col])[cols].dropna()
        else:
            self.df_clean = self.df[required_columns].dropna()
            
        if self.df_clean.empty:
            raise ValueError("DataFrame is empty after dropping NaNs.")
        return self.df_clean

    def _get_required_columns(self) -> List[str]:
        if self._is_multivariate:
            columns = list(getattr(self, "target_variables", []) or [])
        else:
            columns = list(self.independent_variables) + [self.target]
        if getattr(self, "entity_column", None) and self.entity_column not in columns:
            columns.append(self.entity_column)
        if getattr(self, "time_column", None) and self.time_column not in columns:
            columns.append(self.time_column)
        return columns

    @abstractmethod
    def build_model(self) -> Any:
        pass

    @abstractmethod
    def get_summary(self) -> Any:
        pass

    @abstractmethod
    def get_model_metrics(self) -> Dict[str, Any]:
        pass

    def _extract_single_equation_coefficients(self) -> Dict[str, Any]:
        assert self.model is not None  # guarded by the caller
        conf_int = self.model.conf_int() if hasattr(self.model, "conf_int") else None
        coefficients = {}
        for var in self.model.params.index:
            coefficients[var] = {
                "value": float(self.model.params[var]),
                "std_err": float(self.model.bse[var]) if hasattr(self.model, "bse") else None,
                "t_value": float(self.model.tvalues[var]) if hasattr(self.model, "tvalues") else None,
                "p_value": float(self.model.pvalues[var]) if hasattr(self.model, "pvalues") else None,
                "ci_lower": float(conf_int.loc[var, 0]) if isinstance(conf_int, pd.DataFrame) else None,
                "ci_upper": float(conf_int.loc[var, 1]) if isinstance(conf_int, pd.DataFrame) else None
            }
        return coefficients

    def _extract_system_coefficients(self) -> Dict[str, Any]:
        # System models (VAR/VECM/SVAR) report one column of coefficients per
        # equation, so params is a DataFrame indexed by term. Mirrors the
        # accessor pattern in reporting/timeseries/adapters.py.
        assert self.model is not None  # guarded by the caller
        params = self.model.params
        stderr = getattr(self.model, "stderr", None)
        tvalues = getattr(self.model, "tvalues", None)
        pvalues = getattr(self.model, "pvalues", None)

        coefficients: Dict[str, Any] = {}
        for equation in params.columns:
            terms = {}
            for term in params.index:
                terms[term] = {
                    "value": float(params.loc[term, equation]),
                    "std_err": float(stderr.loc[term, equation]) if isinstance(stderr, pd.DataFrame) else None,
                    "t_value": float(tvalues.loc[term, equation]) if isinstance(tvalues, pd.DataFrame) else None,
                    "p_value": float(pvalues.loc[term, equation]) if isinstance(pvalues, pd.DataFrame) else None
                }
            coefficients[str(equation)] = terms
        return coefficients

    def get_model_metadata(self, metrics: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if metrics is None:
            metrics = self.get_model_metrics()

        coefficients: Dict[str, Any] = {}
        if hasattr(self, "model") and self.model is not None and hasattr(self.model, "params"):
            try:
                if isinstance(self.model.params, pd.DataFrame):
                    coefficients = self._extract_system_coefficients()
                else:
                    coefficients = self._extract_single_equation_coefficients()
            except (AttributeError, KeyError, TypeError, ValueError) as e:
                self.logger.warning(f"Could not extract coefficients: {e}")
                coefficients = {"error": f"{type(e).__name__}: {e}"}

        summary_stats: Dict[str, Any] = {}
        if hasattr(self, "model") and self.model is not None:
            if self._is_multivariate:
                summary_stats = {
                    "variables": list(getattr(self, "target_variables", []) or []),
                    "model_type": type(self.model).__name__
                }
            else:
                summary_stats = {
                    "dependent_variable": self.target,
                    "independent_variables": self.independent_variables,
                    "model_type": type(self.model).__name__
                }
            for attr in ["rsquared", "rsquared_adj", "fvalue", "f_pvalue", "llf", "aic", "bic", "nobs", "df_resid"]:
                if hasattr(self.model, attr):
                    summary_stats[attr] = float(getattr(self.model, attr)) if attr not in ["nobs", "df_resid"] else int(getattr(self.model, attr))

        metadata = {
            "model_version": self.model_version,
            "creation_timestamp": datetime.now().isoformat(),
            "params": self.params.get("model", {}) if self.params else {},
            "metrics": metrics,
            "coefficients": coefficients,
            "summary": summary_stats
        }
        return metadata

    def save_model_metadata(self, metrics: Dict[str, Any], output_dir: str = "models") -> str:
        os.makedirs(output_dir, exist_ok=True)
        metadata = self.get_model_metadata(metrics)
        metadata_path = os.path.join(output_dir, f"model_{self.model_version}_metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=4)
        return metadata_path

    def fit(self, data: Union[str, pd.DataFrame]) -> Dict[str, Any]:
        self.load_data(data)
        if hasattr(self, 'df_clean') and self.df_clean is not None:
            self.df_clean = self.df_clean.dropna()
        
        try:
            self.build_model()
        except Exception as e:
            if "NaN" in str(e) or "inf" in str(e):
                if hasattr(self, 'X') and self.X is not None and hasattr(self, 'y') and self.y is not None:
                    nan_mask = self.X.isna().any(axis=1) | self.y.isna()
                    if nan_mask.any():
                        self.X = self.X[~nan_mask]
                        self.y = self.y[~nan_mask]
                        self.build_model()
                    else:
                        raise
                else:
                    self.build_model()
            else:
                raise
        return self.get_model_metrics()
