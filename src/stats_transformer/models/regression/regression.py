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
from typing import Any, Dict, List, Optional, Tuple
from stats_transformer.models.base import ModelBase

class RegressionResults:

    """Accessors around a fitted statsmodels results object."""
    def __init__(self, model: Any) -> None:
        self.model = model

    @property
    def summary(self) -> str:
        """Return the statsmodels summary text."""
        return str(self.model.summary())

    @property
    def metrics(self) -> Dict[str, Any]:
        """Return R-squared, adjusted R-squared, AIC and BIC as a dictionary."""
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
    def params(self) -> pd.Series:
        """Return the estimated coefficients."""
        return self.model.params

    @property
    def pvalues(self) -> pd.Series:
        """Return the coefficient p-values."""
        return self.model.pvalues

    @property
    def conf_int(self) -> pd.DataFrame:
        """Return the coefficient confidence intervals."""
        return self.model.conf_int()

class RegressionModel(ModelBase):

    """Regression with a single-equation target and metric reporting."""
    def __init__(self, params_path: Optional[str] = None, target: Optional[str] = None, independent_variables: Optional[List[str]] = None, add_entity_fixed_effects: bool = False, entity_column: Optional[str] = None, **kwargs: Any) -> None:
        super().__init__(params_path=params_path, target=target, independent_variables=independent_variables, add_entity_fixed_effects=add_entity_fixed_effects, entity_column=entity_column, **kwargs)
        if self.add_entity_fixed_effects and not self.entity_column:
            self.logger.warning("Entity fixed effects requested but entity_column not specified.")


    def split_xy(self, drop_na: bool = True) -> Tuple[pd.DataFrame, pd.Series]:
        """Split the cleaned data into X and y for the target variable."""
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
        
        if self.add_entity_fixed_effects and self.entity_column in self.df_clean.columns:
            entity_dummies = pd.get_dummies(self.df_clean[self.entity_column], prefix="entity", dtype=int)
            self.X = pd.concat([self.X, entity_dummies], axis=1)
        else:
            self.X = sm.add_constant(self.X)

        return self.X, self.y

    def build_model(self, drop_na: bool = True) -> Any:
        """Build the underlying model object from the loaded data."""
        if self.X is None or self.y is None:
            self.split_xy(drop_na=drop_na)
        assert self.X is not None and self.y is not None  # set by split_xy above

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

    def fit(self, df: pd.DataFrame, drop_na: bool = True, compute_metrics: bool = True) -> Dict[str, Any]:
        """Fit the model on the data and store the results."""
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
        return self.get_model_metrics() if compute_metrics else {}

    def predict(self, new_data: Any) -> pd.Series:
        """Generate predictions for new_data using fitted model coefficients."""
        if self.model is None:
            raise ValueError("Model not trained. Call fit() before predict().")

        params = self.model.params

        if isinstance(new_data, pd.Series):
            new_data = new_data.to_frame().T

        if isinstance(new_data, pd.DataFrame):
            missing = [c for c in self.independent_variables if c not in new_data.columns]
            if missing:
                raise ValueError(f"Missing independent variables in new_data: {missing}")

            X_new = new_data[self.independent_variables].copy()
            if self.add_entity_fixed_effects and self.entity_column in new_data.columns:
                trained_entities = {
                    col.removeprefix("entity_") for col in params.index if col.startswith("entity_")
                }
                unseen = sorted(
                    {str(v) for v in new_data[self.entity_column]} - trained_entities
                )
                if unseen:
                    raise ValueError(
                        f"Entities not seen in training: {unseen}. "
                        "Predictions with entity fixed effects cannot include new entities."
                    )
                entity_dummies = pd.get_dummies(new_data[self.entity_column], prefix="entity", dtype=int)
                X_new = pd.concat([X_new, entity_dummies], axis=1)
                for col in params.index:
                    if col not in X_new.columns:
                        X_new[col] = 0
                X_new = X_new[params.index]
            elif self.add_entity_fixed_effects:
                raise ValueError(
                    f"Entity column '{self.entity_column}' is required in new_data because "
                    "the model was fit with entity fixed effects."
                )
            else:
                if "const" in params.index:
                    X_new = sm.add_constant(X_new, has_constant="add")
                X_new = X_new[params.index]

            preds = np.dot(X_new.values, params.values)
            return pd.Series(preds, index=new_data.index, name="prediction")
        elif isinstance(new_data, (np.ndarray, list)):
            arr = np.asarray(new_data)
            if "const" in params.index and arr.shape[-1] == len(params) - 1:
                arr = sm.add_constant(arr, has_constant="add")
            preds = np.dot(arr, params.values)
            return pd.Series(preds, name="prediction")
        else:
            raise TypeError(f"Unsupported data type for predict: {type(new_data)}")

    def get_summary(self) -> str:
        """Return a text summary of the fitted model."""
        if self.model is None:
            raise ValueError("Model not trained")
        return str(self.model.summary())

    def get_model_metrics(self) -> Dict[str, Any]:
        """Return key fit metrics as a dictionary."""
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
    def results(self) -> Optional[RegressionResults]:
        """Return the RegressionResults or None before fitting."""
        return RegressionResults(self.model) if self.model else None

    def summary(self) -> str:
        """Return the summary text or None before fitting."""
        return self.get_summary()

    def get_results(self) -> Optional[RegressionResults]:
        """Return the RegressionResults wrapper."""
        return self.results

    def save_summary_to_json(self, output_path: str = "regression_summary.json") -> None:
        """Save the model summary to a JSON file."""
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

    def save_latex_table(self, output_path: str = "regression_table.tex", float_format: str = "%.3f", caption: Optional[str] = None, label: Optional[str] = None) -> pd.DataFrame:
        """Save a LaTeX table of the regression results."""
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

    def run(self, data_path: Optional[str] = None, output_path: Optional[str] = None) -> Dict[str, Any]:
        """Run the full workflow end to end."""
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
