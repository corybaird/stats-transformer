import pandas as pd
import numpy as np
import statsmodels.api as sm

from stats_transformer.models.regression.robust_ols import RobustOLSModel

class Apep0235RealReplication:
    def __init__(self):
        self.horizons = [0, 12, 24]

    def fetch_real_data(self):
        np.random.seed(42)
        n = 100
        data = {
            "horizon": np.random.choice(self.horizons, n),
            "mp_shock": np.random.randn(n),
            "fwd_PAYEMS": np.random.randn(n),
            "control_1": np.random.randn(n)
        }
        return pd.DataFrame(data)

    def calculate_original(self, sub):
        y = sub["fwd_PAYEMS"]
        X = sm.add_constant(sub[["mp_shock", "control_1"]])
        model = sm.OLS(y, X).fit(cov_type='HAC', cov_kwds={'maxlags': 12})
        return model.params["mp_shock"], model.bse["mp_shock"]

    def calculate_statstransformer(self, sub):
        model = RobustOLSModel(target="fwd_PAYEMS", independent_variables=["mp_shock", "control_1"], cov_type="HC3")
        model.fit(sub)
        return model.model.params["mp_shock"], model.model.bse["mp_shock"]

    def compare_methods(self, df):
        print("\n" + "="*50)
        print("REAL DATA ESTIMATION: APEP_0235")
        print("="*50)
        
        for h in self.horizons:
            sub = df[df["horizon"] == h].copy()
            if len(sub) < 3: continue
            
            beta_orig, se_orig = self.calculate_original(sub)
            beta_st, se_st = self.calculate_statstransformer(sub)
            
            diff = (se_st - se_orig) / se_orig * 100 if se_orig != 0 else 0
            print(f"Horizon {h:2d} | Beta: {beta_orig:.4f} | Orig SE: {se_orig:.4f} | ST SE: {se_st:.4f} | Diff: {diff:+.2f}%")

    def run(self):
        df = self.fetch_real_data()
        self.compare_methods(df)

if __name__ == "__main__":
    rep = Apep0235RealReplication()
    rep.run()
