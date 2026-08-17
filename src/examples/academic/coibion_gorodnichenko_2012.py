import pandas as pd
from pathlib import Path
from stats_transformer.models.regression.gmm import GMMModel

class CoibionGorodnichenko2012Replication:

    def __init__(self, data_path=None):
        if data_path:
            self.data_path = Path(data_path)
        else:
            self.data_path = Path("data/examples/academic/coibion_gorodnichenko_2012/greenbook_forecast_errors.parquet.gzip")
        self.gmm_model = None

    def _load_data(self):
        df = pd.read_parquet(self.data_path).reset_index()
        # Forecast error at the nowcast horizon, the ex-ante revision to next
        # quarter's forecast, and a lagged (public-information) backcast
        # revision as the instrument -- a simplified single-series analogue
        # of Coibion & Gorodnichenko (2012)'s "errors are predictable from
        # revisions" test, adapted to the fields available in the Greenbook
        # CPI extract (there is no exact meeting-to-meeting target alignment
        # to reconstruct their original FOMC-panel specification).
        df["forecast_error"] = df["cpi_nowcast"] - df["cpi_backcast_h1"]
        df["forecast_revision"] = df["cpi_forecast_h1"] - df["cpi_nowcast"]
        df["backcast_revision"] = df["cpi_backcast_h1"] - df["cpi_backcast_h2"]
        df["prior_inflation_level"] = df["cpi_backcast_h2"]
        return df

    def run(self):
        df = self._load_data()
        self.gmm_model = GMMModel(target="forecast_error", independent_variables=["prior_inflation_level"], endogenous=["forecast_revision"], instruments=["backcast_revision"], method="two_step", weighting="hac")
        metrics = self.gmm_model.fit(df)
        print("Coibion & Gorodnichenko (2012) GMM Metrics:", metrics)
        return {"metrics": metrics}
