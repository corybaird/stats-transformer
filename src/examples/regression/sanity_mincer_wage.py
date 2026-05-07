import os
import pandas as pd
import numpy as np
import statsmodels.api as sm
from stats_transformer.models.regression.robust_ols import RobustOLSModel

class AgentSanityMincerWage:
    def __init__(self):
        self.data_url = "https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/master/csv/wooldridge/wage1.csv"
        self.data_dir = "data/raw/examples/regression"
        self.data_path = f"{self.data_dir}/mincer_wage.csv"
        os.makedirs(self.data_dir, exist_ok=True)

    def _fetch_and_prepare_data(self):
        print("Fetching Mincer Wage Data (1976 CPS from Wooldridge)...")
        # Read from Rdatasets
        df_raw = pd.read_csv(self.data_url)
        
        # Mincer Equation: log(wage) = b0 + b1*educ + b2*exper + b3*exper^2 + b4*tenure
        # Create experience squared
        df_raw["expersq"] = df_raw["exper"] ** 2
        
        # Save data
        df_raw.to_csv(self.data_path, index=False)
        print(f"Data saved to {self.data_path}")
        print("\n" + "="*50)
        print("1. PREPARED DATA (Last 5 Rows)")
        print("="*50)
        print(df_raw[["lwage", "educ", "exper", "expersq", "tenure"]].tail())
        
        return df_raw

    def _compute_statsmodels_ols(self, df):
        print("\n" + "="*50)
        print("2. ORIGINAL STATSMODELS OLS (HC3 Robust SE)")
        print("="*50)
        
        y = df["lwage"]
        X = sm.add_constant(df[["educ", "exper", "expersq", "tenure"]])
        
        model = sm.OLS(y, X).fit(cov_type="HC3")
        
        print(model.summary())
        return model

    def _compute_stats_transformer(self, df):
        print("\n" + "="*50)
        print("3. STATS-TRANSFORMER ROBUST OLS MODEL")
        print("="*50)
        
        st_model = RobustOLSModel(
            target="lwage",
            independent_variables=["educ", "exper", "expersq", "tenure"],
            cov_type="HC3"
        )
        st_model.fit(df)
        
        print(st_model.get_summary())
        return st_model

    def _compare_and_report(self, sm_model, st_model):
        print("\n" + "="*50)
        print("4. FINAL COMPARISON")
        print("="*50)
        
        sm_params = sm_model.params.values
        st_params = st_model.model.params.values
        
        sm_bse = sm_model.bse.values
        st_bse = st_model.model.bse.values
        
        param_diff = np.max(np.abs(sm_params - st_params))
        bse_diff = np.max(np.abs(sm_bse - st_bse))
        
        print(f"Max difference in Coefficients: {param_diff:.6e}")
        print(f"Max difference in Standard Errors: {bse_diff:.6e}")
        
        if param_diff < 1e-10 and bse_diff < 1e-10:
            print("\nCONCLUSION: Stats-Transformer perfectly matches native statsmodels robust OLS for the Mincer wage equation.")
        else:
            print("\nCONCLUSION: Significant deviations detected.")

    def run(self):
        df_clean = self._fetch_and_prepare_data()
        sm_model = self._compute_statsmodels_ols(df_clean)
        st_model = self._compute_stats_transformer(df_clean)
        self._compare_and_report(sm_model, st_model)

if __name__ == "__main__":
    sanity = AgentSanityMincerWage()
    sanity.run()
