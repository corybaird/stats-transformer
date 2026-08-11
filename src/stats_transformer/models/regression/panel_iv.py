import numpy as np
import pandas as pd
import statsmodels.api as sm
from datetime import datetime
from linearmodels.iv import IV2SLS
from stats_transformer.models.base import ModelBase


class PanelIV2SLSModel(ModelBase):
    # Panel 2SLS with entity and time fixed effects

    def __init__(self, params_path=None, 
                 target=None, independent_variables=None, 
                 # IV
                 instruments=None, endogenous=None, 
                 # Panel TWFE
                 entity_column=None, time_column="date", entity_effects=True, time_effects=False, cov_type="robust", cluster_by="entity", **kwargs):
        super().__init__(params_path=params_path, target=target, independent_variables=independent_variables, entity_column=entity_column, **kwargs)

        panel_iv_config = self.params.get("model", {}).get("panel_iv", {}) if self.params else {}
        feature_config = self.params.get("data", {}).get("featurization", {}) if self.params else {}

        self.instruments = panel_iv_config.get("instruments", instruments or [])
        self.endogenous = panel_iv_config.get("endogenous", endogenous or [])
        self.time_column = panel_iv_config.get("time_column", feature_config.get("date_column", time_column))
        self.entity_effects = panel_iv_config.get("entity_effects", entity_effects)
        self.time_effects = panel_iv_config.get("time_effects", time_effects)
        self.cov_type = panel_iv_config.get("cov_type", cov_type)
        self.cluster_by = panel_iv_config.get("cluster_by", cluster_by)

        if not self.entity_column:
            raise ValueError("entity_column must be specified")
        if not self.endogenous:
            raise ValueError("Endogenous variables must be specified")
        if not self.instruments:
            raise ValueError("Instruments must be specified")
        if len(self.instruments) < len(self.endogenous):
            raise ValueError("The number of instruments must be at least the number of endogenous variables")
        if self.cov_type == "clustered" and self.cluster_by not in ["entity", "time"]:
            raise ValueError("cluster_by must be 'entity' or 'time'")

    def _get_required_columns(self):
        columns = super()._get_required_columns()
        for column in self.endogenous + self.instruments:
            if column not in columns:
                columns.append(column)
        return columns

    def load_data(self, data):
        self.logger.info("Loading data for PanelIV2SLSModel")
        df = pd.read_csv(data) if isinstance(data, str) and data.endswith(".csv") else (pd.read_parquet(data) if isinstance(data, str) else data.copy())

        req = self._get_required_columns()
        missing = [column for column in req if column not in df.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")


        self.df_clean = df[req].replace([np.inf, -np.inf], np.nan).dropna().copy()
        if self.df_clean.empty:
            raise ValueError("DataFrame is empty after dropping NaNs")

        self.df_clean = self.df_clean.set_index([self.entity_column, self.time_column]).sort_index()
        return self.df_clean

    def _add_fixed_effects(self, exog):
        # we add fixed effects as dummy variables to exog variables, following Hansen (2021; 628)
        frames = [exog]

        if self.entity_effects:
            entities = pd.Series(self.df_clean.index.get_level_values(self.entity_column), index=self.df_clean.index)
            frames.append(pd.get_dummies(entities, prefix="entity", drop_first=True, dtype=float))

        if self.time_effects:
            periods = pd.Series(self.df_clean.index.get_level_values(self.time_column), index=self.df_clean.index)
            frames.append(pd.get_dummies(periods, prefix="time", drop_first=True, dtype=float))

        return sm.add_constant(pd.concat(frames, axis=1), has_constant="add")

    def _get_clusters(self):
        level = self.entity_column if self.cluster_by == "entity" else self.time_column
        values = self.df_clean.index.get_level_values(level)
        return pd.Series(pd.Categorical(values).codes, index=self.df_clean.index)

    def build_model(self):
        self.y = self.df_clean[self.target]
        self.X_exog = self._add_fixed_effects(self.df_clean[self.independent_variables])
        self.X_endog = self.df_clean[self.endogenous]
        self.Z = self.df_clean[self.instruments]
        self.X = self.X_exog

        self.model_spec = IV2SLS(dependent=self.y, exog=self.X_exog, endog=self.X_endog, instruments=self.Z)
        
        fit_options = {
            "cov_type": self.cov_type
        }
        
        if self.cov_type == "clustered":
            fit_options["clusters"] = self._get_clusters()

        self.model = self.model_spec.fit(**fit_options)
        return self.model

    def get_summary(self):
        if self.model is None:
            raise ValueError("Model not trained")
        return str(self.model.summary)

    def get_model_metrics(self):
        if self.model is None:
            raise ValueError("Model not trained")

        first_stage = self.model.first_stage.diagnostics.replace({np.nan: None}).to_dict(orient="index")
        for variable, diagnostics in first_stage.items():
            if diagnostics.get("f.stat") is not None and diagnostics["f.stat"] < 10:
                self.logger.warning(f"Weak instrument warning for {variable}: first-stage statistic is below 10")

        return {
            "r_squared": float(self.model.rsquared),
            "f_statistic": float(self.model.f_statistic.stat),
            "f_pvalue": float(self.model.f_statistic.pval),
            "wu_hausman_stat": float(self.model.wu_hausman().stat),
            "num_observations": int(self.model.nobs),
            "first_stage": first_stage
        }

    def get_model_metadata(self, metrics=None):
        if metrics is None:
            metrics = self.get_model_metrics()

        coefficients = {}
        conf_int = self.model.conf_int()
        reported_variables = ["const"] + self.independent_variables + self.endogenous
        for variable in reported_variables:
            if variable in self.model.params.index:
                coefficients[variable] = {
                    "value": float(self.model.params[variable]),
                    "std_err": float(self.model.std_errors[variable]),
                    "t_value": float(self.model.tstats[variable]),
                    "p_value": float(self.model.pvalues[variable]),
                    "ci_lower": float(conf_int.iloc[:, 0].loc[variable]),
                    "ci_upper": float(conf_int.iloc[:, 1].loc[variable])
                }

        return {
            "model_version": self.model_version,
            "creation_timestamp": datetime.now().isoformat(),
            "params": self.params.get("model", {}) if self.params else {},
            "metrics": metrics,
            "coefficients": coefficients,
            "summary": {
                "dependent_variable": self.target,
                "independent_variables": self.independent_variables,
                "endogenous": self.endogenous,
                "instruments": self.instruments,
                "model_type": "PanelIV2SLS",
                "entity_effects": self.entity_effects,
                "time_effects": self.time_effects,
                "cov_type": self.cov_type
            }
        }

    def run(self, data_path=None, output_path=None):
        self.fit(data_path)
        if output_path:
            self.save_model_metadata(self.get_model_metrics(), output_dir=output_path)
        return self.get_model_metadata()
