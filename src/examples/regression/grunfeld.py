import os
import pandas as pd
import numpy as np
import statsmodels.api as sm
from stats_transformer.models.regression.regression import RegressionModel
from src.examples.base import BaseExample

class GrunfeldPanelExample(BaseExample):
    def __init__(self):
        self.data_dir = "data/examples/regression"
        self.data_path = f"{self.data_dir}/grunfeld.csv"
        os.makedirs(self.data_dir, exist_ok=True)

    def _fetch_and_prepare_data(self):
        print("Fetching Grunfeld Investment Panel Data...")
        # Load directly from statsmodels
        df_raw = sm.datasets.grunfeld.load_pandas().data
        
        # Save data
        df_raw.to_csv(self.data_path, index=False)
        print(f"Data saved to {self.data_path}")
        print("\n" + "="*50)
        print("1. PREPARED DATA (Last 5 Rows)")
        print("="*50)
        print(df_raw.tail())
        
        return df_raw

    def _compute_baseline(self, df):
        print("\n" + "="*50)
        print("2. ORIGINAL STATSMODELS OLS (LSDV - Least Squares Dummy Variable)")
        print("="*50)
        
        y = df["invest"]
        # Include firm as categorical for fixed effects (Dummy Variables), without intercept
        X = df[["value", "capital"]].copy()
        entity_dummies = pd.get_dummies(df["firm"], prefix="entity", dtype=int)
        X = pd.concat([X, entity_dummies], axis=1)
        
        model = sm.OLS(y, X).fit()
        
        print(model.summary())
        return model

    def _compute_stats_transformer(self, df):
        print("\n" + "="*50)
        print("3. STATS-TRANSFORMER FIXED EFFECTS MODEL")
        print("="*50)
        
        st_model = RegressionModel(
            target="invest",
            independent_variables=["value", "capital"],
            add_entity_fixed_effects=True,
            entity_column="firm"
        )
        st_model.fit(df)
        
        print(st_model.get_summary())
        return st_model

    def _compare_and_report(self, sm_model, st_model):
        print("\n" + "="*50)
        print("4. FINAL COMPARISON")
        print("="*50)
        
        # Align indexes as column order might differ
        sm_params = sm_model.params.sort_index()
        st_params = st_model.model.params.sort_index()
        
        sm_bse = sm_model.bse.sort_index()
        st_bse = st_model.model.bse.sort_index()
        
        param_diff = np.max(np.abs(sm_params.values - st_params.values))
        bse_diff = np.max(np.abs(sm_bse.values - st_bse.values))
        
        print(f"Max difference in Coefficients: {param_diff:.6e}")
        print(f"Max difference in Standard Errors: {bse_diff:.6e}")
        
        if param_diff < 1e-10 and bse_diff < 1e-10:
            print("\nCONCLUSION: Stats-Transformer perfectly matches native statsmodels Fixed Effects (LSDV) regression.")
        else:
            print("\nCONCLUSION: Significant deviations detected.")

if __name__ == "__main__":
    example = GrunfeldPanelExample()
    example.run()
