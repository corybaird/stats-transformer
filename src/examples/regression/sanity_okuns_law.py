import os
import requests
import pandas as pd
import numpy as np
import statsmodels.api as sm
from src.stats_transformer.models.regression.robust_ols import RobustOLSModel

class AgentSanityOkunsLaw:
    def __init__(self):
        self.api_key = self._load_api_key()
        self.base_url = "https://api.stlouisfed.org/fred/series/observations"
        self.data_dir = "data/examples/regression"
        self.data_path = f"{self.data_dir}/okuns_law.csv"
        os.makedirs(self.data_dir, exist_ok=True)

    def _load_api_key(self):
        env_path = ".env"
        if os.path.exists(env_path):
            with open(env_path) as f:
                for line in f:
                    if line.startswith("API_FRED"):
                        return line.split("=", 1)[1].strip()
        return os.environ.get("API_FRED")

    def _fetch_series(self, series_id, frequency="q"):
        if not self.api_key:
            raise ValueError("FRED API key not found. Please set API_FRED in .env")
        params = {
            "series_id": series_id,
            "api_key": self.api_key,
            "file_type": "json",
            "units": "lin",
            "observation_start": "1948-01-01",
            "observation_end": "2024-01-01",
            "frequency": frequency
        }
        response = requests.get(self.base_url, params=params)
        response.raise_for_status()
        data = response.json().get("observations", [])
        df = pd.DataFrame(data)
        df["date"] = pd.to_datetime(df["date"])
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df = df.rename(columns={"value": series_id})
        return df[["date", series_id]]

    def _fetch_and_prepare_data(self):
        print("Fetching FRED data for Okun's Law (GDPC1, UNRATE)...")
        # GDP is quarterly, UNRATE is monthly, so we fetch both as quarterly.
        df_gdp = self._fetch_series("GDPC1", frequency="q")
        df_unrate = self._fetch_series("UNRATE", frequency="q")
        
        df_raw = df_gdp.merge(df_unrate, on="date", how="inner").sort_values("date").dropna()
        
        # Calculate transformations for Okun's law
        # Target: Change in Unemployment Rate (diff)
        # Independent: Percentage Change in Real GDP (pct_change)
        
        df_raw["unrate_diff"] = df_raw["UNRATE"].diff()
        df_raw["gdp_pct_change"] = df_raw["GDPC1"].pct_change() * 100
        
        df_clean = df_raw.dropna().copy()
        
        # Save data
        df_clean.to_csv(self.data_path, index=False)
        print(f"Data saved to {self.data_path}")
        print("\n" + "="*50)
        print("1. PREPARED DATA (Last 5 Rows)")
        print("="*50)
        print(df_clean[["date", "unrate_diff", "gdp_pct_change"]].tail())
        
        return df_clean

    def _compute_statsmodels_ols(self, df):
        print("\n" + "="*50)
        print("2. ORIGINAL STATSMODELS OLS (HC3 Robust SE)")
        print("="*50)
        
        y = df["unrate_diff"]
        X = sm.add_constant(df["gdp_pct_change"])
        
        model = sm.OLS(y, X).fit(cov_type="HC3")
        
        print(model.summary())
        return model

    def _compute_stats_transformer(self, df):
        print("\n" + "="*50)
        print("3. STATS-TRANSFORMER ROBUST OLS MODEL")
        print("="*50)
        
        # We initialize RobustOLSModel with HC3
        st_model = RobustOLSModel(
            target="unrate_diff",
            independent_variables=["gdp_pct_change"],
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
            print("\nCONCLUSION: Stats-Transformer perfectly matches native statsmodels robust OLS.")
        else:
            print("\nCONCLUSION: Significant deviations detected.")

    def run(self):
        df_clean = self._fetch_and_prepare_data()
        sm_model = self._compute_statsmodels_ols(df_clean)
        st_model = self._compute_stats_transformer(df_clean)
        self._compare_and_report(sm_model, st_model)

if __name__ == "__main__":
    sanity = AgentSanityOkunsLaw()
    sanity.run()
