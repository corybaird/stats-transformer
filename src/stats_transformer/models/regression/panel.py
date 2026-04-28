import pandas as pd
import numpy as np
from linearmodels.panel import PanelOLS
from src.stats_transformer.models.base import ModelBase
from datetime import datetime

class PanelRegressionModel(ModelBase):
    # Panel data modeling using entity and time fixed effects

    def __init__(self, params_path=None, target=None, independent_variables=None, entity_column=None, time_column="date", entity_effects=True, time_effects=False, check_rank=True, **kwargs):
        super().__init__(params_path=params_path, target=target, independent_variables=independent_variables, entity_column=entity_column, **kwargs)
        self.time_column = time_column
        if self.params:
            panel_config = self.params.get("model", {}).get("panel_ols", {})
            self.entity_effects = panel_config.get("entity_effects", entity_effects)
            self.time_effects = panel_config.get("time_effects", time_effects)
            self.check_rank = panel_config.get("check_rank", check_rank)
        else:
            self.entity_effects = entity_effects
            self.time_effects = time_effects
            self.check_rank = check_rank

    def _get_required_columns(self):
        req = [self.target] + self.independent_variables
        if self.entity_column and self.entity_column not in req:
            req.append(self.entity_column)
        if self.time_column and self.time_column not in req:
            req.append(self.time_column)
        return req

    def load_data(self, data):
        self.logger.info("Loading data for PanelRegressionModel")
        df = pd.read_csv(data) if isinstance(data, str) and data.endswith(".csv") else (pd.read_parquet(data) if isinstance(data, str) else data.copy())
        
        req = self._get_required_columns()
        missing = [c for c in req if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")

        self.df_clean = df.dropna(subset=req).copy()
        
        if self.time_column in self.df_clean.columns:
            if not np.issubdtype(self.df_clean[self.time_column].dtype, np.datetime64):
                try:
                    self.df_clean[self.time_column] = pd.to_datetime(self.df_clean[self.time_column])
                except:
                    pass

        self.df_clean = self.df_clean.set_index([self.entity_column, self.time_column])
        
        self.y = self.df_clean[[self.target]]
        self.X = self.df_clean[self.independent_variables]
        import statsmodels.api as sm
        self.X = sm.add_constant(self.X)
        return self.df_clean

    def build_model(self):
        if self.X is None or self.y is None:
            raise ValueError("Data not loaded, X or y is None")
        
        self.model_spec = PanelOLS(self.y, self.X, entity_effects=self.entity_effects, time_effects=self.time_effects, check_rank=self.check_rank)
        self.model = self.model_spec.fit()
        return self.model

    def get_summary(self):
        return str(self.model.summary) if self.model else "Panel Model not trained"

    def get_model_metrics(self):
        if not self.model:
            return {}
        return {
            "rsquared": float(self.model.rsquared),
            "rsquared_between": float(self.model.rsquared_between),
            "rsquared_within": float(self.model.rsquared_within),
            "f_statistic": float(self.model.f_statistic.stat),
            "f_pvalue": float(self.model.f_statistic.pval),
            "nobs": int(self.model.nobs)
        }

    def get_model_metadata(self, metrics=None):
        if metrics is None:
            metrics = self.get_model_metrics()

        coefficients = {}
        if self.model:
            conf_int = self.model.conf_int()
            for var in self.model.params.index:
                coefficients[var] = {
                    "value": float(self.model.params[var]),
                    "std_err": float(self.model.std_errors[var]),
                    "t_value": float(self.model.tstats[var]),
                    "p_value": float(self.model.pvalues[var]),
                    "ci_lower": float(conf_int.iloc[:, 0].loc[var]),
                    "ci_upper": float(conf_int.iloc[:, 1].loc[var])
                }

        summary_stats = {
            "dependent_variable": self.target,
            "independent_variables": self.independent_variables,
            "model_type": "PanelOLS",
            "entity_effects": self.entity_effects,
            "time_effects": self.time_effects
        }
        for attr in ["rsquared", "rsquared_between", "rsquared_within", "nobs"]:
            if hasattr(self.model, attr):
                val = getattr(self.model, attr)
                summary_stats[attr] = float(val) if "nobs" not in attr else int(val)

        metadata = {
            "model_version": self.model_version,
            "creation_timestamp": datetime.now().isoformat(),
            "params": self.params.get("model", {}) if self.params else {},
            "metrics": metrics,
            "coefficients": coefficients,
            "summary": summary_stats
        }
        return metadata

    def run(self, data_path=None, output_path=None):
        self.fit(data_path)
        if output_path:
            self.save_model_metadata(self.get_model_metrics(), output_dir=output_path)
        return self.get_model_metadata()
