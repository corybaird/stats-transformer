import numpy as np
import pandas as pd
from pathlib import Path
from stats_transformer.models.timeseries.nonlinear.tvar import TVARModel


class Balke2000TVARReplication:

    def __init__(self, data_path=None):
        if data_path:
            self.data_path = Path(data_path)
        else:
            self.data_path = Path("data/examples/timeseries/macrodata.csv")
        self.model = None

    def _load_data(self):
        df = pd.read_csv(self.data_path)
        df["gdp_growth"] = df["realgdp"]
        df["inv_growth"] = df["realinv"]
        df["spread_proxy"] = df["realcons"] - df["realgdp"]
        return df

    def run(self):
        df = self._load_data()
        self.model = TVARModel(
            target_variables=["gdp_growth", "inv_growth"],
            threshold_variable="spread_proxy",
            lags=2,
            delay=1,
            trim=0.15
        )
        metrics = self.model.fit(df)
        print("Balke (2000) TVAR Metrics:", metrics)
        print("\nOptimal Threshold Gamma:", self.model.gamma)
        print("Regime 1 Share:", metrics["regime_1_share"])
        print("Regime 2 Share:", metrics["regime_2_share"])
        return {"metrics": metrics, "gamma": self.model.gamma}


if __name__ == "__main__":
    Balke2000TVARReplication().run()
