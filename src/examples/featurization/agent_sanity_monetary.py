# Sanity Check for Monetary Policy Transformations (apep_0235)
# Source Repo: https://github.com/SocialCatalystLab/ape-papers/tree/70c660ea5a7de722e9a12ad965e1c665890208d8/apep_0235/v1/code
import os
import requests
import pandas as pd
import numpy as np
from src.stats_transformer.featurization.feature_engineering import FeatureEngineer

class AgentSanityMonetary:
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

    def _fetch_series(self, series_id):
        if not self.api_key:
            raise ValueError("FRED API key not found. Please set API_FRED in .env")
        params = {
            "series_id": series_id,
            "api_key": self.api_key,
            "file_type": "json",
            "units": "lin",
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
        df = df.rename(columns={"value": series_id})
        return df[["date", series_id]]

    def _fetch_raw_data(self):
        print("Fetching FRED macro controls (CPIAUCSL, INDPRO, PAYEMS)...")
        df_cpi = self._fetch_series("CPIAUCSL")
        df_indpro = self._fetch_series("INDPRO")
        df_payems = self._fetch_series("PAYEMS")
        df_raw = df_cpi.merge(df_indpro, on="date", how="outer").merge(df_payems, on="date", how="outer")
        df_raw = df_raw.sort_values("date").dropna()
        df_raw["country"] = "US"
        print("\n" + "="*50)
        print("1. RAW DATA (Last 5 Rows)")
        print("="*50)
        print(df_raw[["date", "CPIAUCSL", "INDPRO", "PAYEMS"]].tail())
        return df_raw

    def _compute_original_logic(self, df_raw):
        print("\n" + "="*50)
        print("2. ORIGINAL 'apep_0235' SCRIPT TRANSFORMATIONS")
        print("="*50)
        print("Their code computes the following:")
        print(" - Inflation: df['CPIAUCSL'].pct_change(12) * 100")
        print(" - IP Growth: df['INDPRO'].pct_change(12) * 100")
        print(" - Log Emp:   np.log(df['PAYEMS']) * 100")
        print(" - Fwd Emp:   log_emp.shift(-1) - log_emp.shift(1)")
        df_py = df_raw.copy()
        df_py["inflation_py"] = df_py["CPIAUCSL"].pct_change(12) * 100
        df_py["ip_growth_py"] = df_py["INDPRO"].pct_change(12) * 100
        df_py["log_PAYEMS_py"] = np.log(df_py["PAYEMS"]) * 100
        df_py["fwd_PAYEMS_h1_py"] = df_py["log_PAYEMS_py"].shift(-1) - df_py["log_PAYEMS_py"].shift(1)
        return df_py

    def _compute_stats_transformer(self, df_raw):
        print("\n" + "="*50)
        print("3. STATS-TRANSFORMER TRANSFORMATIONS")
        print("="*50)
        print("Applying FeatureEngineer to raw data with building blocks...")
        engineer = FeatureEngineer(
            transformations=["changepct", "log", "lag1", "lag12", "lead1"],
            entity_column="country",
            date_column="date",
            period="monthly",
            data_columns=["CPIAUCSL", "INDPRO", "PAYEMS"],
            verbose=False
        )
        return engineer.fit_transform(df_raw)

    def _compare_and_report(self, df_features, df_py):
        print("\n" + "="*50)
        print("4. FINAL COMPARISON")
        print("="*50)
        df_merged = pd.merge(df_features, df_py[["date", "inflation_py", "ip_growth_py", "fwd_PAYEMS_h1_py"]], on="date", how="inner")
        df_merged["inflation_st"] = ((df_merged["CPIAUCSL"] - df_merged["CPIAUCSL_lag12"]) / df_merged["CPIAUCSL_lag12"]) * 100
        df_merged["ip_growth_st"] = ((df_merged["INDPRO"] - df_merged["INDPRO_lag12"]) / df_merged["INDPRO_lag12"]) * 100
        df_merged["fwd_PAYEMS_h1_st"] = (np.log(df_merged["PAYEMS_lead1"]) - np.log(df_merged["PAYEMS_lag1"])) * 100
        df_merged = df_merged.dropna(subset=["inflation_py", "ip_growth_py", "fwd_PAYEMS_h1_py", "inflation_st", "ip_growth_st", "fwd_PAYEMS_h1_st"])
        df_merged["diff_inflation"] = (df_merged["inflation_st"] - df_merged["inflation_py"]).abs()
        df_merged["diff_ip_growth"] = (df_merged["ip_growth_st"] - df_merged["ip_growth_py"]).abs()
        df_merged["diff_fwd_emp"] = (df_merged["fwd_PAYEMS_h1_st"] - df_merged["fwd_PAYEMS_h1_py"]).abs()
        compare_cols = ["date", "INDPRO", "ip_growth_py", "ip_growth_st", "CPIAUCSL", "inflation_py", "inflation_st", "fwd_PAYEMS_h1_py", "fwd_PAYEMS_h1_st"]
        print("\nSide-by-Side Values (Original Code vs. Stats-Transformer):")
        print(df_merged[compare_cols].tail().to_string(index=False))
        max_diff_inf = df_merged["diff_inflation"].max()
        max_diff_ip = df_merged["diff_ip_growth"].max()
        max_diff_emp = df_merged["diff_fwd_emp"].max()
        print("\n--- Accuracy Report ---")
        print(f"Max difference for Inflation: {max_diff_inf:.6e}")
        print(f"Max difference for IP Growth: {max_diff_ip:.6e}")
        print(f"Max difference for Fwd Emp:   {max_diff_emp:.6e}")
        if max_diff_inf < 1e-6 and max_diff_ip < 1e-6 and max_diff_emp < 1e-6:
            print("\nCONCLUSION: Stats-Transformer perfectly matches their messy script logic.")
        else:
            print("\nCONCLUSION: Significant deviations detected. Their code differs from the standard implementation.")

    def run(self):
        df_raw = self._fetch_raw_data()
        df_py = self._compute_original_logic(df_raw)
        df_features = self._compute_stats_transformer(df_raw)
        self._compare_and_report(df_features, df_py)

if __name__ == "__main__":
    sanity = AgentSanityMonetary()
    sanity.run()
