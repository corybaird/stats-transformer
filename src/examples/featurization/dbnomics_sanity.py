import requests
import pandas as pd
from src.stats_transformer.featurization.feature_engineering import FeatureEngineer

class DbnomicsFeaturizationSanity:
    def __init__(self):
        self.base_url = "https://api.db.nomics.world/v22/series/Eurostat"
    
    def _fetch_series(self, dataset, series_code):
        url = f"{self.base_url}/{dataset}/{series_code}?observations=1"
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        series_data = data["series"]["docs"][0]
        
        df = pd.DataFrame({
            "date": pd.to_datetime(series_data["period"]),
            "value": pd.to_numeric(series_data["value"], errors="coerce")
        })
        return df.dropna()

    def run(self):
        countries = {"FR": "France", "DE": "Germany"}
        
        raw_dfs = []
        dbnomics_pc_dfs = []
        
        for code, name in countries.items():
            print(f"Fetching data for {name} ({code})...")
            
            # Fetch Raw Index (prc_hicp_midx)
            df_raw = self._fetch_series("prc_hicp_midx", f"M.I15.CP00.{code}")
            df_raw = df_raw.rename(columns={"value": "hicp"})
            df_raw["country"] = code
            raw_dfs.append(df_raw)
            
            # Fetch Pre-calculated Monthly Rate of Change (prc_hicp_mmor)
            df_pc = self._fetch_series("prc_hicp_mmor", f"M.RCH_M.CP00.{code}")
            df_pc = df_pc.rename(columns={"value": "dbnomics_hicp_pc"})
            df_pc["country"] = code
            dbnomics_pc_dfs.append(df_pc)
            
        df_raw_panel = pd.concat(raw_dfs, ignore_index=True)
        df_dbnomics_panel = pd.concat(dbnomics_pc_dfs, ignore_index=True)
        
        print("\nApplying FeatureEngineer to Panel Data...")
        engineer = FeatureEngineer(
            transformations=["changepct"],
            entity_column="country",
            date_column="date",
            period="monthly",
            data_columns=["hicp"],
            verbose=False
        )
        
        df_features = engineer.fit_transform(df_raw_panel)
        
        print("\nMerging and comparing...")
        # Note: changepct is fraction, DBnomics HICP rate of change is percentage (fraction * 100)
        df_features["hicp_changepct_100"] = df_features["hicp_changepct"] * 100
        
        df_merged = pd.merge(df_features, df_dbnomics_panel, on=["country", "date"], how="inner")
        df_merged = df_merged.dropna(subset=["hicp_changepct_100", "dbnomics_hicp_pc"])
        
        df_merged["diff"] = (df_merged["hicp_changepct_100"] - df_merged["dbnomics_hicp_pc"]).abs()
        
        max_diff = df_merged["diff"].max()
        print(f"Max difference between calculated and DBnomics pre-calculated percent change: {max_diff:.6f}")
        
        print("\nSample Data (France):")
        print(df_merged[df_merged["country"] == "FR"][["date", "country", "hicp", "hicp_changepct_100", "dbnomics_hicp_pc", "diff"]].tail())
        
        print("\nSample Data (Germany):")
        print(df_merged[df_merged["country"] == "DE"][["date", "country", "hicp", "hicp_changepct_100", "dbnomics_hicp_pc", "diff"]].tail())
        
        # We allow a slightly higher tolerance for DBnomics due to rounding in published index values vs published rates
        if max_diff < 1e-1:
            print("\nSanity check PASSED!")
            if max_diff > 1e-4:
                print("(Note: Small differences are due to Eurostat rounding the raw index to 1 or 2 decimal places before publishing, while calculating the official rate from unrounded data).")
        else:
            print("\nSanity check FAILED - significant differences found.")

if __name__ == "__main__":
    sanity = DbnomicsFeaturizationSanity()
    sanity.run()
