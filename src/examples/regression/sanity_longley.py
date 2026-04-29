import os
import pandas as pd
import numpy as np
import statsmodels.api as sm
from src.stats_transformer.models.regression.robust_ols import RobustOLSModel

class AgentSanityLongley:
    def __init__(self):
        self.data_dir = "data/examples/regression"
        self.data_path = f"{self.data_dir}/longley.csv"
        os.makedirs(self.data_dir, exist_ok=True)

    def _fetch_and_prepare_data(self):
        print("Fetching Longley (1967) Macroeconomic Dataset...")
        df_raw = sm.datasets.longley.load_pandas().data
        df_raw.to_csv(self.data_path, index=False)
        print(f"Data saved to {self.data_path}")
        return df_raw

    def _compute_statsmodels_ols(self, df):
        print("\n" + "="*50)
        print("2. ORIGINAL STATSMODELS OLS (Longley Collinearity Test)")
        print("="*50)
        
        y = df["TOTEMP"]
        X = sm.add_constant(df[["GNPDEFL", "GNP", "UNEMP", "ARMED", "POP", "YEAR"]])
        
        model = sm.OLS(y, X).fit()
        print(model.summary())
        return model

    def _compute_stats_transformer(self, df):
        print("\n" + "="*50)
        print("3. STATS-TRANSFORMER ROBUST OLS MODEL")
        print("="*50)
        
        st_model = RobustOLSModel(
            target="TOTEMP",
            independent_variables=["GNPDEFL", "GNP", "UNEMP", "ARMED", "POP", "YEAR"],
            cov_type="nonrobust" # Standard OLS for Longley baseline
        )
        st_model.fit(df)
        print(st_model.get_summary())
        return st_model

    def _compare_and_report(self, orig_model, st_model):
        print("\n" + "="*50)
        print("4. FINAL COMPARISON")
        print("="*50)
        
        orig_params = orig_model.params.sort_index()
        st_params = st_model.model.params.sort_index()
        
        orig_bse = orig_model.bse.sort_index()
        st_bse = st_model.model.bse.sort_index()
        
        param_diff = np.max(np.abs(orig_params.values - st_params.values))
        bse_diff = np.max(np.abs(orig_bse.values - st_bse.values))
        
        print(f"Max difference in Coefficients: {param_diff:.6e}")
        print(f"Max difference in Standard Errors: {bse_diff:.6e}")
        
        if param_diff < 1e-10 and bse_diff < 1e-10:
            print("\nCONCLUSION: Stats-Transformer perfectly matches native statsmodels on the highly collinear Longley dataset.")
        else:
            print("\nCONCLUSION: Significant deviations detected.")

    def run(self):
        df_clean = self._fetch_and_prepare_data()
        orig_model = self._compute_statsmodels_ols(df_clean)
        st_model = self._compute_stats_transformer(df_clean)
        self._compare_and_report(orig_model, st_model)

if __name__ == "__main__":
    sanity = AgentSanityLongley()
    sanity.run()
