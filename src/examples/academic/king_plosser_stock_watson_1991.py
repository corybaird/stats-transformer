import numpy as np
import pandas as pd
from pathlib import Path
from stats_transformer.models.timeseries.structural.svec import SVECModel


class KingPlosserStockWatson1991Replication:

    def __init__(self, data_path=None):
        if data_path:
            self.data_path = Path(data_path)
        else:
            # data/examples lives under the repo root (src/examples/academic/x.py)
            self.data_path = Path(__file__).resolve().parents[3] / "data/examples/timeseries/macrodata.csv"
        self.model = None

    def _load_data(self):
        df = pd.read_csv(self.data_path)
        df["y"] = np.cumsum(df["realgdp"])
        df["c"] = np.cumsum(df["realcons"])
        df["i"] = np.cumsum(df["realinv"])
        return df

    def run(self):
        df = self._load_data()
        # King, Plosser, Stock & Watson (1991): one permanent (balanced-growth)
        # shock, identified by the two transitory shocks having zero long-run
        # effects (LR columns 1 and 2 of the long-run impact matrix Xi B; King
        # et al. 1991 eq. (7)), plus one contemporaneous zero restriction
        # B[1,2] to disentangle the two transitory shocks (Bruggemann 2006,
        # SFB 649 DP 2006-021 sec. 4). With K = 3, r = 2 this is just-identified.
        SR = np.array([
            [np.nan, np.nan, np.nan],
            [np.nan, np.nan, 0.0],
            [np.nan, np.nan, np.nan]
        ])
        LR = np.array([
            [np.nan, 0.0, 0.0],
            [np.nan, 0.0, 0.0],
            [np.nan, 0.0, 0.0]
        ])

        self.model = SVECModel(
            target_variables=["y", "c", "i"],
            k_ar_diff=2,
            coint_rank=2,
            SR=SR,
            LR=LR
        )
        metrics = self.model.fit(df)
        print("King, Plosser, Stock, & Watson (1991) SVEC Metrics:", metrics)
        print("\nEstimated Structural B (SR):\n", self.model.B_0)
        return {"metrics": metrics, "SR": self.model.B_0}


if __name__ == "__main__":
    KingPlosserStockWatson1991Replication().run()
