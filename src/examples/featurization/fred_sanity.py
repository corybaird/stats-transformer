import os
import requests
import pandas as pd
from src.stats_transformer.featurization.feature_engineering import FeatureEngineer

class FredFeaturizationSanity:
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
            raise ValueError("FRED API key not found.")
        params = {
            "series_id": series_id,
            "api_key": self.api_key,
            "file_type": "json",
            "units": units
        }
        response = requests.get(self.base_url, params=params)
        response.raise_for_status()
        data = response.json().get("observations", [])
        df = pd.DataFrame(data)
        df["date"] = pd.to_datetime(df["date"])
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        return df[["date", "value"]]

    def run(self):
        print("Fetching Raw GDP...")
        df_raw = self._fetch_series("GDP")
        df_raw = df_raw.rename(columns={"value": "gdp"}).dropna()
        df_raw["country"] = "USA"
        
        print("Fetching FRED Percent Change GDP (units=pch)...")
        df_fred_pc = self._fetch_series("GDP", units="pch")
        df_fred_pc = df_fred_pc.rename(columns={"value": "fred_gdp_pc"}).dropna()
        
        print("Applying FeatureEngineer...")
        engineer = FeatureEngineer(
            transformations=["changepct", "changeraw"],
            entity_column="country",
            date_column="date",
            period="quarterly",
            data_columns=["gdp"],
            verbose=False
        )
        
        df_features = engineer.fit_transform(df_raw)
        
        print("Merging and comparing...")
        # Note: changepct is fraction, FRED pc is percentage (fraction * 100)
        df_features["gdp_changepct_100"] = df_features["gdp_changepct"] * 100
        
        df_merged = pd.merge(df_features, df_fred_pc, on="date", how="inner")
        df_merged = df_merged.dropna(subset=["gdp_changepct_100", "fred_gdp_pc"])
        
        df_merged["diff"] = (df_merged["gdp_changepct_100"] - df_merged["fred_gdp_pc"]).abs()
        
        max_diff = df_merged["diff"].max()
        print(f"Max difference between calculated and FRED pre-calculated percent change: {max_diff:.6f}")
        
        # Display a sample
        print("\nSample Data:")
        print(df_merged[["date", "gdp", "gdp_changepct_100", "fred_gdp_pc", "diff"]].tail())
        
        if max_diff < 1e-4:
            print("\nSanity check PASSED!")
        else:
            print("\nSanity check FAILED - significant differences found.")

if __name__ == "__main__":
    sanity = FredFeaturizationSanity()
    sanity.run()
