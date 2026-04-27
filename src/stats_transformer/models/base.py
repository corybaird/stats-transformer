import json
import logging
import os
import pandas as pd
import sys
import yaml
from datetime import datetime
from abc import ABC, abstractmethod

class ModelBase(ABC):

    def __init__(self, params_path=None, target=None, independent_variables=None, add_entity_fixed_effects=False, entity_column=None, **kwargs):
        self._setup_logging()
        self.params = {}
        
        if params_path:
            self.params = self._load_params(params_path)
            self.target = self.params.get("model", {}).get("target_variable", target)
            self.independent_variables = self.params.get("model", {}).get("independent_variables", independent_variables or [])
            self.add_entity_fixed_effects = self.params.get("model", {}).get("ols", {}).get("add_entity_fixed_effects", add_entity_fixed_effects)
            feat_params = self.params.get("data", {}).get("featurization", self.params.get("featurization", {}))
            self.entity_column = feat_params.get("entity_column", entity_column)
        else:
            self.target = target
            self.independent_variables = independent_variables or []
            self.add_entity_fixed_effects = add_entity_fixed_effects
            self.entity_column = entity_column
            if kwargs:
                self.params.update(kwargs)

        if not self.target:
            raise ValueError("Target variable must be specified")
        if not self.independent_variables:
            raise ValueError("Independent variables must be specified")
            
        self.df = None
        self.df_clean = None
        self.X = None
        self.y = None
        self.model = None
        self.model_version = datetime.now().strftime("%Y%m%d_%H%M%S")

    def _is_running_in_jupyter(self):
        try:
            from IPython import get_ipython
            return get_ipython() is not None
        except ImportError:
            return False

    def _setup_logging(self):
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

    def _load_params(self, params_path):
        try:
            with open(params_path, "r") as f:
                params = yaml.safe_load(f)
            return params
        except FileNotFoundError:
            raise FileNotFoundError(f"Parameter file {params_path} not found.")

    def load_data(self, data):
        self.logger.info("Loading data")
        if type(data) == str:
            self.df = pd.read_csv(data) if data.endswith(".csv") else pd.read_parquet(data)
        else:
            self.df = data.copy()

        required_columns = self._get_required_columns()
        missing_columns = [col for col in required_columns if col not in self.df.columns]
        if missing_columns:
            raise ValueError(f"Required columns missing: {missing_columns}")

        if self.entity_column and self.entity_column in self.df.columns and 'date' in self.df.columns:
            if self.add_entity_fixed_effects and self.entity_column in required_columns:
                self.df_clean = self.df[required_columns].dropna()
            else:
                self.df_clean = self.df.set_index([self.entity_column, 'date'])[required_columns].dropna()
        elif 'date' in self.df.columns:
            self.df_clean = self.df.set_index(['date'])[required_columns].dropna()
        else:
            self.df_clean = self.df[required_columns].dropna()
            
        if self.df_clean.empty:
            raise ValueError("DataFrame is empty after dropping NaNs.")
        return self.df_clean

    @abstractmethod
    def _get_required_columns(self):
        pass

    @abstractmethod
    def build_model(self):
        pass

    @abstractmethod
    def get_summary(self):
        pass

    @abstractmethod
    def get_model_metrics(self):
        pass

    def get_model_metadata(self, metrics=None):
        if metrics is None:
            metrics = self.get_model_metrics()

        coefficients = {}
        if hasattr(self, "model") and self.model is not None and hasattr(self.model, "params"):
            try:
                conf_int = self.model.conf_int()
                for var in self.model.params.index:
                    coefficients[var] = {
                        "value": float(self.model.params[var]),
                        "std_err": float(self.model.bse[var]) if hasattr(self.model, "bse") else None,
                        "t_value": float(self.model.tvalues[var]) if hasattr(self.model, "tvalues") else None,
                        "p_value": float(self.model.pvalues[var]) if hasattr(self.model, "pvalues") else None,
                        "ci_lower": float(conf_int.loc[var, 0]) if type(conf_int) == pd.DataFrame else None,
                        "ci_upper": float(conf_int.loc[var, 1]) if type(conf_int) == pd.DataFrame else None
                    }
            except Exception as e:
                self.logger.warning(f"Could not extract coefficients: {e}")

        summary_stats = {}
        if hasattr(self, "model") and self.model is not None:
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

    def save_model_metadata(self, metrics, output_dir="models"):
        os.makedirs(output_dir, exist_ok=True)
        metadata = self.get_model_metadata(metrics)
        metadata_path = os.path.join(output_dir, f"model_{self.model_version}_metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=4)
        return metadata_path

    def fit(self, data):
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
