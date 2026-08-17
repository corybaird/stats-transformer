import pandas as pd
from pathlib import Path
from stats_transformer.models.timeseries.reduced_form.local_projections import LocalProjectionsModel


class KilianLutkepohl2017Chap5LocalProjectionsReplication:

    def __init__(self, data_path=None):
        if data_path:
            self.data_path = Path(data_path)
        else:
            self.data_path = Path("data/examples/timeseries/macrodata.csv")
        self.model = None

    def _load_data(self):
        df = pd.read_csv(self.data_path)
        return df

    def run(self):
        df = self._load_data()
        self.model = LocalProjectionsModel(
            target="realgdp",
            shock_var="realinv",
            controls=["realcons"],
            horizon=4
        )
        metrics = self.model.fit(df)
        print("Kilian & Lütkepohl (2017) Ch. 5 Local Projections Metrics:", metrics)
        print("IRF Results across horizons:")
        for irf in self.model.irf_results:
            print(f"  h={irf['horizon']}: effect={irf['effect']:.4f}, se={irf['stderr']:.4f}")
        return {"metrics": metrics, "irfs": self.model.irf_results}


if __name__ == "__main__":
    KilianLutkepohl2017Chap5LocalProjectionsReplication().run()
