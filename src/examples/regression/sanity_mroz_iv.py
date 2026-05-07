import os
import pandas as pd
import numpy as np
from linearmodels.iv import IV2SLS
import statsmodels.api as sm
from src.stats_transformer.models.regression.iv import IV2SLSModel

class AgentSanityMrozIV:
    def __init__(self):
        self.data_url = "https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/master/csv/wooldridge/mroz.csv"
        self.data_dir = "data/raw/examples/regression"
        self.data_path = f"{self.data_dir}/mroz.csv"
        os.makedirs(self.data_dir, exist_ok=True)

    def _fetch_and_prepare_data(self):
        print("Fetching Mroz (1987) Female Labor Supply Data...")
        df_raw = pd.read_csv(self.data_url)
        
        # We only use women who are in the labor force (wage > 0)
        df_clean = df_raw[df_raw['wage'] > 0].copy()
        
        # We will regress log wage on experience, experience squared, instrumenting education with parents' education
        df_clean["expersq"] = df_clean["exper"] ** 2
        df_clean = df_clean.dropna(subset=["lwage", "exper", "expersq", "educ", "motheduc", "fatheduc"])
        
        df_clean.to_csv(self.data_path, index=False)
        print(f"Data saved to {self.data_path}")
        return df_clean

    def _compute_linearmodels_iv(self, df):
        print("\n" + "="*50)
        print("2. ORIGINAL LINEARMODELS IV/2SLS")
        print("="*50)
        
        y = df["lwage"]
        X_exog = sm.add_constant(df[["exper", "expersq"]])
        X_endog = df[["educ"]]
        Z = df[["motheduc", "fatheduc"]]
        
        model = IV2SLS(dependent=y, exog=X_exog, endog=X_endog, instruments=Z).fit(cov_type="robust")
        print(model.summary)
        return model

    def _compute_stats_transformer_iv(self, df):
        print("\n" + "="*50)
        print("3. STATS-TRANSFORMER IV/2SLS MODEL")
        print("="*50)
        
        st_model = IV2SLSModel(
            target="lwage",
            independent_variables=["exper", "expersq"],
            endogenous=["educ"],
            instruments=["motheduc", "fatheduc"],
            cov_type="robust"
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
        
        orig_bse = orig_model.std_errors.sort_index()
        st_bse = st_model.model.std_errors.sort_index()
        
        param_diff = np.max(np.abs(orig_params.values - st_params.values))
        bse_diff = np.max(np.abs(orig_bse.values - st_bse.values))
        
        print(f"Max difference in Coefficients: {param_diff:.6e}")
        print(f"Max difference in Standard Errors: {bse_diff:.6e}")
        
        if param_diff < 1e-10 and bse_diff < 1e-10:
            print("\nCONCLUSION: Stats-Transformer perfectly matches native linearmodels IV2SLS.")
        else:
            print("\nCONCLUSION: Significant deviations detected.")

    def run(self):
        df_clean = self._fetch_and_prepare_data()
        orig_model = self._compute_linearmodels_iv(df_clean)
        st_model = self._compute_stats_transformer_iv(df_clean)
        self._compare_and_report(orig_model, st_model)

if __name__ == "__main__":
    sanity = AgentSanityMrozIV()
    sanity.run()
