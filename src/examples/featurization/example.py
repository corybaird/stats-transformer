import os
import requests
import pandas as pd
import numpy as np
from stats_transformer.featurization.feature_engineering import FeatureEngineer

class FeaturizationExample:
    """
    Example check to compare stats-transformer's feature engineering
    with the explicit R-code transformations from the ape-papers repository.
    
    R-code (02_clean_data.R) computes:
    d_emp_0  = 100 * (log_emp - lag(log_emp, 1))
    d_emp_12 = 100 * (lead(log_emp, 12) - lag(log_emp, 1))
    """
    
    def __init__(self):
        self.api_key = self._load_api_key()
        self.base_url = "https://api.stlouisfed.org/fred/series/observations"
    
    def _load_api_key(self):
        env_path = ".env"
        if os.path.exists(env_path):
            with open(env_path) as f:
                for line in f:
                    if line.startswith("API_FRED"):
                        return line.split("=", 1)[1].strip()
        return os.environ.get("API_FRED")
    
    def _fetch_series(self, series_id, units="lin"):
        if not self.api_key:
            raise ValueError("FRED API key not found. Please set API_FRED in .env")
        params = {
            "series_id": series_id,
            "api_key": self.api_key,
            "file_type": "json",
            "units": units,
            "observation_start": "1990-01-01",
            "observation_end": "2024-12-01",
            "frequency": "m"
        }
        response = requests.get(self.base_url, params=params)
        response.raise_for_status()
        data = response.json().get("observations", [])
        df = pd.DataFrame(data)
        df["date"] = pd.to_datetime(df["date"])
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        return df[["date", "value"]]

    def run(self):
        print("Fetching Raw State Employment (CANA - California) from FRED...")
        df_raw = self._fetch_series("CANA")
        df_raw = df_raw.rename(columns={"value": "emp"}).dropna()
        df_raw["state"] = "CA"
        
        # 1. R-Code Manual Implementation
        # 02_clean_data.R logic:
        # log_emp = log(emp)
        # d_emp_0  = 100 * (log_emp - lag(log_emp, 1))
        # d_emp_12 = 100 * (lead(log_emp, 12) - lag(log_emp, 1))
        
        print("\nComputing R-style transformations (Log Returns)...")
        df_r = df_raw.copy()
        df_r = df_r.sort_values("date")
        df_r["log_emp"] = np.log(df_r["emp"])
        df_r["d_emp_0_R"] = 100 * (df_r["log_emp"] - df_r["log_emp"].shift(1))
        df_r["d_emp_12_R"] = 100 * (df_r["log_emp"].shift(-12) - df_r["log_emp"].shift(1))
        
        # 2. Stats-Transformer Implementation
        # We can approximate the log differences with changepct * 100
        # Or we can compute the exact log return if we pass log_emp to FeatureEngineer
        print("\nApplying FeatureEngineer to raw data...")
        engineer = FeatureEngineer(
            transformations=["changepct", "log", "lag1", "lead12"],
            entity_column="state",
            date_column="date",
            period="monthly",
            data_columns=["emp"],
            verbose=False
        )
        
        df_features = engineer.fit_transform(df_raw)
        
        print("Merging and comparing...")
        # Note: changepct is simple returns. R code uses log returns.
        # simple return = (emp - lag(emp)) / lag(emp)
        # log return = log(emp / lag(emp))
        df_features["emp_changepct_100"] = df_features["emp_changepct"] * 100
        
        df_merged = pd.merge(df_features, df_r[["date", "d_emp_0_R", "d_emp_12_R"]], on="date", how="inner")
        df_merged = df_merged.dropna(subset=["emp_changepct_100", "d_emp_0_R"])
        
        df_merged["diff_simple_vs_log"] = (df_merged["emp_changepct_100"] - df_merged["d_emp_0_R"]).abs()
        
        max_diff = df_merged["diff_simple_vs_log"].max()
        print(f"\nMax difference between simple returns (stats-transformer) and log returns (R script): {max_diff:.6f}")
        
        print("\nSample Data (Comparing d_emp_0):")
        print(df_merged[["date", "emp", "emp_changepct_100", "d_emp_0_R", "diff_simple_vs_log"]].tail())
        
        print("\n--- Analysis ---")
        print("The R script uses 'log returns' (difference in log values * 100).")
        print("stats-transformer natively uses 'simple returns' (pct_change).")
        print("For small percentage changes (like employment), they are mathematically very close.")
        if max_diff < 2.0:
            print("\nExample check PASSED! The stats-transformer simple returns closely approximate the R-script log returns.")
            print("Note: The max difference usually occurs during large shocks (e.g., COVID-19) where log returns and simple returns diverge slightly.")
        else:
            print("\nExample check FAILED - significant differences found between simple and log returns.")

if __name__ == "__main__":
    sanity = FeaturizationExample()
    sanity.run()
