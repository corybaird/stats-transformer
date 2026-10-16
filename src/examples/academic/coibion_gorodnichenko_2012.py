"""Replication of Coibion & Gorodnichenko (2012), Table 1.

Paper: "Why Are Target Interest Rate Changes So Persistent?", American
Economic Journal: Macroeconomics 4(4): 126-162.

Replicates the nested Taylor-rule regressions (columns 1-5 of Table 1) and the
motivate regressions reported in the paper's text, following the authors'
replication files: the series in their EViews workfile greenbookquarterly.wf1
(converted to parquet by scripts/convert_eviews_wf1.py) and the equations
reg_motivate01-03 and basic_nested0_AR1 to basic_nested4_AR1 in
regressions_for_tables.prg. Sample 1987:4-2006:4; each column regresses the
federal funds rate on a constant, expected inflation, the output gap series
ex, the one-quarter-ahead real GDP growth forecast egy0 (dropped in column 4),
and in the nested columns the lagged federal funds rate; epia is regenerated
per the program's "genr epia=(epi1+epi2)/2". Columns with interest smoothing
only are fitted with RegressionModel (OLS); columns with AR(1) errors are
fitted with the library's ARIMAModel, extended here to carry exogenous
regressors by exact maximum likelihood.

Approximation: EViews' ls fits the AR(1)-error specification by Gauss-Newton
least squares with backcast initial residuals, while ARIMAModel uses exact ML,
so the AR(1)-error coefficients match the published ones approximately.
"""
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from examples._print import format_metrics
from stats_transformer.models.regression.regression import RegressionModel
from stats_transformer.models.timeseries.arima import ARIMAModel

COLUMNS: Dict[str, Dict[str, Any]] = {
    "reg_motivate01": {"regressors": ["EPIA", "EX", "EGY0"], "errors": "ols"},
    "reg_motivate02": {"regressors": ["EPIA", "EX", "EGY0", "ffr_l1"], "errors": "ols"},
    "reg_motivate03": {"regressors": ["EPIA", "EX", "EGY0"], "errors": "ar1"},
    "table1_col1": {"regressors": ["EPIA", "EX", "EGY0", "ffr_l1"], "errors": "ar1"},
    "table1_col2": {"regressors": ["EPI1", "EX", "EGY0", "ffr_l1"], "errors": "ar1"},
    "table1_col3": {"regressors": ["EPIA", "EX", "EGY1", "ffr_l1"], "errors": "ar1"},
    "table1_col4": {"regressors": ["EPIA", "EX", "ffr_l1"], "errors": "ar1"},
    "table1_col5": {"regressors": ["EPI0", "EX", "EGY0", "ffr_l1"], "errors": "ar1"},
}

SAMPLE_START = "1987-10-01"  # 1987:4 in the authors' quarterly workfile
SAMPLE_END = "2006-10-01"  # 2006:4


class CoibionGorodnichenko2012Replication:

    def __init__(self, data_path: Optional[Path] = None):
        if data_path:
            self.data_path = Path(data_path)
        else:
            self.data_path = Path(__file__).resolve().parents[3] / "data/examples/academic/coibion_gorodnichenko_2012/greenbook_quarterly.parquet"

    def _load_data(self) -> pd.DataFrame:
        df = pd.read_parquet(self.data_path)
        df.index = pd.DatetimeIndex(df.index, freq="QS")
        # regressions_for_tables.prg: genr epia=(epi1+epi2)/2 over the full sample
        df["epia"] = (df["EPI1"] + df["EPI2"]) / 2
        df["ffr_l1"] = df["FFR"].shift(1)
        return df.loc[SAMPLE_START:SAMPLE_END]

    def _fit_column(self, spec: Dict[str, Any], df: pd.DataFrame) -> tuple[Dict[str, Any], Dict[str, Any]]:
        if spec["errors"] == "ols":
            model = RegressionModel(target="FFR", independent_variables=spec["regressors"])
            metrics = model.fit(df)
            params = model.model.params
            metrics["r_squared"] = model.model.rsquared
        else:
            model = ARIMAModel(target="FFR", independent_variables=spec["regressors"], order=(1, 0, 0), trend="c", date_column=None)
            metrics = model.fit(df)
            params = model.model.params
        metrics["estimator"] = "ols" if spec["errors"] == "ols" else "ar_1_ml"
        metrics = {key: float(value) if hasattr(value, "__float__") else value for key, value in metrics.items()}
        coefficients = {name: float(value) for name, value in params.items() if name != "sigma2"}
        return metrics, coefficients

    def run(self) -> Dict[str, Any]:
        df = self._load_data()
        metrics: Dict[str, Any] = {}
        coefficients: Dict[str, Any] = {}
        for name, spec in COLUMNS.items():
            col_metrics, col_coefficients = self._fit_column(spec, df)
            metrics[name] = col_metrics
            coefficients[name] = col_coefficients
        print("Coibion & Gorodnichenko (2012), Table 1, sample 1987:4-2006:4")
        print(pd.DataFrame(coefficients).round(4).to_string())
        print("fit metrics:")
        format_metrics(metrics)
        return {"coefficients": coefficients, "metrics": metrics}


if __name__ == "__main__":
    CoibionGorodnichenko2012Replication().run()