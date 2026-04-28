import json
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import sys
import yaml
from datetime import datetime
from src.stats_transformer.models.base import ModelBase

class RegressionResults:

    def __init__(self, model):
        self.model = model
        
    @property
    def summary(self):
        return str(self.model.summary())
    
    @property
    def metrics(self):
        return {
            "r_squared": self.model.rsquared,
            "adj_r_squared": self.model.rsquared_adj,
            "f_statistic": self.model.fvalue,
            "f_pvalue": self.model.f_pvalue,
            "aic": self.model.aic,
            "bic": self.model.bic,
            "condition_number": self.model.condition_number,
            "num_observations": self.model.nobs,
        }
    
    @property
    def params(self):
        return self.model.params
    
    @property
    def pvalues(self):
        return self.model.pvalues
    
    @property
    def conf_int(self):
        return self.model.conf_int()

class RegressionModel(ModelBase):

    def __init__(self, params_path=None, target=None, independent_variables=None, add_entity_fixed_effects=False, entity_column=None, **kwargs):
        super().__init__(params_path=params_path, target=target, independent_variables=independent_variables, add_entity_fixed_effects=add_entity_fixed_effects, entity_column=entity_column, **kwargs)
        if self.add_entity_fixed_effects and not self.entity_column:
            self.logger.warning("Entity fixed effects requested but entity_column not specified.")

    def _get_required_columns(self):
        columns = self.independent_variables + [self.target]
        if self.add_entity_fixed_effects and self.entity_column:
            columns.append(self.entity_column)
        return columns

    def split_xy(self, drop_na=True):
        if self.df_clean is None:
            raise ValueError("No cleaned data available")

        numeric_df = self.df_clean.select_dtypes(include=[np.number])
        if drop_na:
            df_clean = self.df_clean.copy()
            for col in numeric_df.columns:
                df_clean[col] = df_clean[col].replace([np.inf, -np.inf], np.nan)
            df_clean = df_clean.dropna()
            self.df_clean = df_clean

        self.y = self.df_clean[self.target]
        self.X = self.df_clean[self.independent_variables]
        
        if self.add_entity_fixed_effects and self.entity_column in self.X.columns:
            entity_dummies = pd.get_dummies(self.X[self.entity_column], prefix="entity", dtype=int)
            self.X = pd.concat([self.X.drop(self.entity_column, axis=1), entity_dummies], axis=1)
        else:
            self.X = sm.add_constant(self.X)

        return self.X, self.y

    def build_model(self, drop_na=True):
        if self.X is None or self.y is None:
            self.split_xy(drop_na=drop_na)

        x_numeric = self.X.select_dtypes(include=[np.number])
        if np.isnan(x_numeric.values).any() or np.isinf(x_numeric.values).any() or np.isnan(self.y.values).any() or np.isinf(self.y.values).any():
            if drop_na:
                mask = ~(np.isnan(x_numeric.values).any(axis=1) | np.isinf(x_numeric.values).any(axis=1) | np.isnan(self.y.values) | np.isinf(self.y.values))
                self.X = self.X[mask]
                self.y = self.y[mask]
            else:
                raise ValueError("exog contains inf or nans")

        self.model = sm.OLS(self.y, self.X).fit()
        return self.model

    def fit(self, df, drop_na=True):
        required_cols = self._get_required_columns()
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing columns: {missing_cols}")

        self.df_clean = df.copy()
        if drop_na:
            for col in required_cols:
                if col in self.df_clean.columns:
                    self.df_clean[col] = self.df_clean[col].replace([np.inf, -np.inf], np.nan)
            self.df_clean = self.df_clean.dropna(subset=required_cols)

        self.build_model(drop_na=False)
        return self.get_model_metrics()

    def get_summary(self):
        if self.model is None:
            raise ValueError("Model not trained")
        return str(self.model.summary())

    def get_model_metrics(self):
        if self.model is None:
            raise ValueError("Model not trained")
        return {
            "r_squared": self.model.rsquared,
            "adj_r_squared": self.model.rsquared_adj,
            "f_statistic": self.model.fvalue,
            "f_pvalue": self.model.f_pvalue,
            "aic": self.model.aic,
            "bic": self.model.bic,
            "condition_number": self.model.condition_number,
            "num_observations": self.model.nobs,
        }

    @property
    def results(self):
        return RegressionResults(self.model) if self.model else None

    def summary(self):
        return self.get_summary()

    def get_results(self):
        return self.results

    def save_summary_to_json(self, output_path="regression_summary.json"):
        if self.model is None:
            raise ValueError("Model not trained")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        summary_dict = {}
        for table in self.model.summary().tables:
            for row in table.data:
                summary_dict[row[0]] = row[1:]
        summary_dict["metrics"] = self.get_model_metrics()
        summary_dict["model_version"] = self.model_version
        summary_dict["timestamp"] = datetime.now().isoformat()
        with open(output_path, "w") as f:
            json.dump(summary_dict, f, indent=4)

    def save_latex_table(self, output_path="regression_table.tex", float_format="%.3f", caption=None, label=None):
        if self.model is None:
            raise ValueError("Model not trained")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        results_df = pd.concat([self.model.params, self.model.bse, self.model.pvalues, self.model.tvalues, self.model.conf_int()[0], self.model.conf_int()[1]], axis=1)
        results_df.columns = ["Coefficient", "Std Error", "p-values", "t-values", "CI Lower", "CI Upper"]
        results_df.index = [x.replace("_", "-") for x in results_df.index]
        results_df.index.name = "Variables"
        if caption is None:
            caption = f"Regression Results for {self.target}"
        if label is None:
            label = "tab:regression_results"
        latex_table = results_df.to_latex(index=True, float_format=float_format, caption=caption, label=label, escape=False, column_format="l|cccccc")
        with open(output_path, "w") as f:
            f.write(latex_table)
        return results_df

    def run(self, data_path=None, output_path=None):
        if not data_path and self.params:
            data_path = self.params.get("data", {}).get("featurization", {}).get("output_path")
        if not output_path and self.params:
            output_path = self.params.get("model", {}).get("summary_output_path")
        if not data_path:
            raise ValueError("No data path found")
        df = pd.read_csv(data_path)
        results = self.fit(df)
        if output_path:
            self.save_summary_to_json(output_path)
        return results

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="params.yaml")
    args = parser.parse_args()
    model = RegressionModel(params_path=args.config)
    model.run()
