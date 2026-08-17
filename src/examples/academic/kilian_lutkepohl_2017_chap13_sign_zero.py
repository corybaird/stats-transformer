import numpy as np
import pandas as pd
from pathlib import Path
from stats_transformer.models.timeseries.identification.sign_zero import SignZeroSVARModel


class KilianLutkepohl2017Chap13SignZeroReplication:

    def __init__(self, data_path=None, config_path=None):
        if data_path:
            self.data_path = Path(data_path)
        else:
            self.data_path = Path("data/examples/timeseries/macrodata.csv")

        if config_path:
            self.config_path = Path(config_path)
        else:
            self.config_path = Path("references/configs/timeseries/identification/structural_restrictions.yaml")
        self.model = None

    def _load_data(self):
        df = pd.read_csv(self.data_path)
        df_model = pd.DataFrame({
            "output": df["realgdp"],
            "inflation": df["realcons"],
            "interest_rate": df["realinv"]
        })
        return df_model

    def run(self):
        df = self._load_data()
        self.model = SignZeroSVARModel(
            target_variables=["output", "inflation", "interest_rate"],
            config_path=str(self.config_path),
            maxlags=1,
            max_draws=1000,
            required_accepts=10,
            seed=42
        )
        self.model.narrative_restrictions = []
        metrics = self.model.fit(df)
        print("Kilian & Lütkepohl (2017) Ch. 13 Sign/Zero SVAR Metrics:", metrics)
        print(f"Accepted Rotations: {len(self.model.accepted_rotations)}")
        return {"metrics": metrics, "n_accepted": len(self.model.accepted_rotations)}


if __name__ == "__main__":
    KilianLutkepohl2017Chap13SignZeroReplication().run()
