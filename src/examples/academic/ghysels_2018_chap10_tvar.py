import pandas as pd
from pathlib import Path
from stats_transformer.models.timeseries.nonlinear.tvar import TVARModel


class Ghysels2018Chap10TVARExample:

    def __init__(self, data_path=None):
        if data_path:
            self.data_path = Path(data_path)
        else:
            self.data_path = Path("data/examples/academic/ghysels_2018/Ch_10/simulated_ch10_sec5.csv")
        self.model = None

    def _load_data(self):
        df = pd.read_csv(self.data_path)
        df_clean = df.iloc[100:600].reset_index(drop=True)
        return df_clean

    def run(self):
        df = self._load_data()
        self.model = TVARModel(
            target_variables=["y"],
            threshold_variable="y",
            lags=2,
            delay=1,
            trim=0.15
        )
        metrics = self.model.fit(df)
        print("Ghysels (2018) Ch. 10 TVAR Metrics:", metrics)
        print("Estimated Threshold Gamma:", self.model.gamma)
        return {"metrics": metrics, "gamma": self.model.gamma}


if __name__ == "__main__":
    Ghysels2018Chap10TVARExample().run()
