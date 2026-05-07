import pandas as pd
import sys
import os

if "src" not in sys.path:
    sys.path.append("src")

from stats_transformer.featurization.data_merger import DataMerger
from stats_transformer.featurization.feature_engineering import FeatureEngineer

class DemoFeaturization:
    def __init__(self):
        pass

    def run(self):
        print("====== 1. Merging Data files ======")
        merger = DataMerger(params_path="params.yaml")

        df_hicp = merger.load_dataset("data/final/monthly/eurostat_hicp_panel.parquet")
        df_cpi = merger.load_dataset("data/final/monthly/oecd_cpi_panel.parquet")

        df_hicp = merger.standardize_entity(df_hicp, "country")
        df_cpi = merger.standardize_entity(df_cpi, "country")

        # Pick a single country for the sanity check and avoid huge merges
        COUNTRY_SLICED = "AUT"
        df_hicp = df_hicp[df_hicp["country"] == COUNTRY_SLICED]
        df_cpi = df_cpi[df_cpi["country"] == COUNTRY_SLICED]

        # The OECD CPI data has multiple duplicated rows for the same date 
        # (presumably different subjects dropped during ingestion). We will take the first one.
        df_cpi = df_cpi.drop_duplicates(subset=["country", "date"])
        df_hicp = df_hicp.drop_duplicates(subset=["country", "date"])

        df_merged = merger.merge(df_hicp, df_cpi, on=["country", "date"], how="inner")
        
        # Sort values precisely by date to ensure time-based transformations make sense
        df_merged = df_merged.sort_values(by=["country", "date"]).reset_index(drop=True)
        print("\nMerged Data Shape:", df_merged.shape)
        
        print("\n====== 2. Feature Engineering ======")
        # We'll use only 'hicp' to ensure strictly positive values (optional if trying 'log')
        data_cols = ["hicp"]
        
        # We will try all available transformation routines (omitting 'log' just in case data has zeros/negatives, but you can try it)
        all_transforms = ["changepct", "changeraw", "rollingmean", "zscore", "lag1", "lead1"]

        engineer = FeatureEngineer(
            params_path=None,
            transformations=all_transforms,
            entity_column="country",
            date_column="date",
            period="monthly",
            data_columns=data_cols,
            verbose=False,
        )

        df_features = engineer.fit_transform(df_merged)

        print("\n====== 3. Sanity Checks for Country:", COUNTRY_SLICED, "======")
        # Select the base column and all its transformed versions
        eval_cols = ["date", "hicp"] + [f"hicp_{t}" for t in all_transforms]
        df_eval = df_features[eval_cols].head(10)
        
        print(df_eval.to_string(index=False))
        
        print("\nSanity Check Analysis:")
        print("- 'hicp_lead1' should match the 'hicp' of the NEXT row.")
        print("- 'hicp_lag1' should match the 'hicp' of the PREVIOUS row.")
        print("- 'hicp_changeraw' should be (hicp - hicp_lag1).")
        print("- 'hicp_changepct' should be (hicp_changeraw / hicp_lag1).")
        print("- 'hicp_rollingmean' takes the current and previous elements in the window.")
        

if __name__ == "__main__":
    demo = DemoFeaturization()
    demo.run()
