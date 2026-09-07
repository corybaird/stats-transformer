import pandas as pd
from pathlib import Path
from stats_transformer.models.timeseries.arima import ARIMAModel


class Ghysels2018Chap5ARIMAExample:

    def __init__(self, data_path=None):
        if data_path:
            self.data_path = Path(data_path)
        else:
            self.data_path = Path("data/examples/timeseries/ghysels_ch5/arma_inven.csv")
        self.model = None

    def _load_data(self):
        df = pd.read_csv(self.data_path)
        return df

    def run(self):
        df = self._load_data()
        target_col = "rcpi" if "rcpi" in df.columns else (df.columns[1] if len(df.columns) > 1 else df.columns[0])
        self.model = ARIMAModel(
            target=target_col,
            order=(1, 0, 1)
        )
        metrics = self.model.fit(df)
        print("Ghysels (2018) Ch. 5 ARIMA Metrics:", metrics)
        return {"metrics": metrics}


if __name__ == "__main__":
    Ghysels2018Chap5ARIMAExample().run()
