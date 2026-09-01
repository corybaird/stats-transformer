import pandas as pd
from pathlib import Path
from stats_transformer.models.timeseries.reduced_form.bvar import BVARModel


class Ghysels2018Chap8BVARExample:

    def __init__(self, data_path=None):
        if data_path:
            self.data_path = Path(data_path)
        else:
            self.data_path = Path("data/examples/timeseries/ghysels_ch8/bvar_simulated_ch8_sec4.csv")
        self.model = None

    def _load_data(self):
        df = pd.read_csv(self.data_path)
        df_clean = df[["x", "y"]].iloc[100:600].reset_index(drop=True)
        return df_clean

    def run(self):
        df = self._load_data()
        self.model = BVARModel(
            target_variables=["x", "y"],
            lags=4,
            lambda1=0.2,
            lambda2=0.99,
            lambda3=1.0
        )
        metrics = self.model.fit(df)
        print("Ghysels (2018) Ch. 8 BVAR Metrics:", metrics)
        return {"metrics": metrics}


if __name__ == "__main__":
    Ghysels2018Chap8BVARExample().run()
