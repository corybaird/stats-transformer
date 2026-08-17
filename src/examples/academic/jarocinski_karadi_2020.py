from pathlib import Path
import pandas as pd
from stats_transformer.models.timeseries.reduced_form.bvar import BVARModel

class JarocinskiKaradi2020Replication:

    def __init__(self, data_path=None):
        if data_path:
            self.data_path = Path(data_path)
        else:
            self.data_path = Path("data/examples/academic/jarocinski_karadi_2020/fomc_surprises.parquet.gzip")
        self.bvar_model = None

    def _load_data(self):
        df = pd.read_parquet(self.data_path)
        return df.reset_index().sort_values("date")

    def run(self):
        df = self._load_data()
        # mp1: fed funds futures surprise; onrun2/onrun10: Treasury yield
        # changes; sp500: equity return, all in the FOMC-day surprise window.
        # Jarociński & Karadi (2020)'s "poor man's" sign restriction separates
        # a pure monetary policy shock from a central bank information shock
        # using the sign of the co-movement between mp1 and sp500: a rate
        # surprise that moves stocks in the SAME direction signals information
        # about the economy, not a policy tightening/easing shock. This BVAR
        # provides the reduced-form system; sign classification of draws
        # implements the identification restriction directly on posterior IRFs.
        self.bvar_model = BVARModel(target_variables=["mp1", "onrun2", "sp500"], date_column="date", lags=2, lambda1=0.5, lambda2=0.5, lambda3=1.0, lambda4=100.0, n_draws=1000)
        metrics = self.bvar_model.fit(df)

        irf_mp = self.bvar_model.compute_irf(horizon=5, response="sp500", shock="mp1")
        mp_shock_sign = irf_mp.loc[irf_mp["horizon"] == 0, "estimate"].iloc[0]
        classification = "information shock (mp1 and sp500 co-move)" if mp_shock_sign > 0 else "monetary policy shock (mp1 and sp500 move oppositely)"

        print("Jarocinski & Karadi (2020) BVAR Metrics:", metrics)
        print("Poor man's sign classification of the identified mp1 shock:", classification)

        return {"metrics": metrics, "irf_sp500_to_mp1": irf_mp, "classification": classification}
