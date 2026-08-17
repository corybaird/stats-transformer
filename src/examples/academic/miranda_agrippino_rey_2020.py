import numpy as np
import pandas as pd
from pathlib import Path
from stats_transformer.models.timeseries.reduced_form.dynamic_factor import DynamicFactorModel

class MirandaAgrippinoRey2020Replication:

    def __init__(self, data_path=None):
        if data_path:
            self.data_path = Path(data_path)
        else:
            self.data_path = Path("data/examples/academic/miranda_agrippino_rey_2020/global_factor.parquet.gzip")
        self.dfm_model = None

    def _load_reference_factor(self):
        df = pd.read_parquet(self.data_path)
        return df.reset_index(drop=True)

    def _simulate_risky_asset_panel(self, n_periods=300, n_series=20):
        np.random.seed(42)
        global_factor = np.zeros(n_periods)
        for t in range(1, n_periods):
            global_factor[t] = 0.6 * global_factor[t - 1] + np.random.normal(scale=0.5)
        loadings = np.random.uniform(0.4, 1.6, size=n_series) * np.random.choice([-1, 1], size=n_series)
        panel = global_factor[:, None] @ loadings[None, :] + np.random.normal(scale=np.sqrt(0.5), size=(n_periods, n_series))
        columns = [f"asset_{i}" for i in range(n_series)]
        df = pd.DataFrame(panel, columns=columns)
        df["date"] = pd.date_range("1990-01-01", periods=n_periods, freq="ME")
        return df, columns, global_factor

    def run(self):
        # The Miranda-Agrippino & Rey (2020) Global Factor is estimated from an
        # unbalanced panel of ~858-1002 risky asset prices (equities, commodities,
        # corporate bonds) that is not available in this repository -- only the
        # paper's published factor series is. This runner therefore demonstrates
        # DynamicFactorModel mechanics on a small synthetic risky-asset-style
        # panel with a known factor, and separately loads the published series
        # purely as a labeled reference overlay. It is not a formal replication.
        panel_df, columns, true_factor = self._simulate_risky_asset_panel()
        self.dfm_model = DynamicFactorModel(target_variables=columns, date_column="date", n_factors=1, factor_lags=1, max_iter=150, tol=1e-8)
        metrics = self.dfm_model.fit(panel_df)

        extracted = self.dfm_model.compute_factors()
        correlation = float(np.corrcoef(self.dfm_model.factors_[:, 0], true_factor)[0, 1])

        reference_df = self._load_reference_factor()

        print("Miranda, Agrippino & Rey (2020) Global Factor - illustrative DFM extraction:", metrics)
        print("Correlation between extracted and simulated true factor:", round(correlation, 4))
        print("Published Global Factor reference series (not used as model input):")
        print(reference_df.tail())

        return {"metrics": metrics, "extracted_factor": extracted, "true_factor_correlation": correlation, "reference_series": reference_df}
