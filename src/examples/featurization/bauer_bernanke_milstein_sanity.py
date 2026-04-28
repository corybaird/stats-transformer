import pandas as pd
import sys
import numpy as np
from pathlib import Path

if "src" not in sys.path:
    sys.path.append("src")

from src.stats_transformer.featurization.feature_engineering import FeatureEngineer

class BBMSanity:
    def __init__(self):
        self.data_dir = Path("data") / "examples" / "BBM_2023"
        self.raw_path = self.data_dir / "feds_subset.csv"

    def load_raw(self):
        print(f"Loading raw BBM data from {self.raw_path}...")
        df = pd.read_csv(self.raw_path, skiprows=9)
        df["Date"] = pd.to_datetime(df["Date"])
        df["country"] = "USA"
        # Ensure numeric and handle potential 'NA' strings from GSW file
        for col in ["SVENY01", "SVENY10"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=["SVENY01", "SVENY10"]).sort_values("Date").reset_index(drop=True)
        return df

    def transform_manual(self, df):
        print("Applying BBM-style manual transformations...")
        df_manual = df.copy()
        df_manual["SVENY01_diff"] = df_manual["SVENY01"].diff()
        df_manual["SVENY10_pct"] = df_manual["SVENY10"].pct_change(fill_method=None)
        return df_manual

    def transform_library(self, df):
        print("Applying FeatureEngineer transformations...")
        engineer = FeatureEngineer(
            params_path=None,
            transformations=["changeraw", "changepct"],
            entity_column="country",
            date_column="Date",
            period="daily",
            data_columns=["SVENY01", "SVENY10"],
            verbose=True, # Enable logging for debugging
        )
        return engineer.fit_transform(df)

    def compare(self, df_my, df_gt):
        print("\n====== Comparing BBM-style Transformations ======")
        col01 = next((c for c in df_my.columns if "SVENY01" in c and "changeraw" in c), None)
        col10 = next((c for c in df_my.columns if "SVENY10" in c and "changepct" in c), None)
        
        if not col01 or not col10:
            print(f"ERROR: Could not find transformed columns in library output: {df_my.columns.tolist()}")
            return

        merged = pd.merge(
            df_my[["Date", col01, col10]],
            df_gt[["Date", "SVENY01_diff", "SVENY10_pct"]],
            on="Date"
        )
        
        comparison = merged.dropna()
        diff01 = (comparison[col01] - comparison["SVENY01_diff"]).abs().max()
        diff10 = (comparison[col10] - comparison["SVENY10_pct"]).abs().max()
        
        print(f"SVENY01 (Diff) Max Error: {diff01:.12f}")
        print(f"SVENY10 (PctChange) Max Error: {diff10:.12f}")
        
        if max(diff01, diff10) < 1e-10:
            print("SUCCESS: Library transformations perfectly match BBM's Python logic.")
        else:
            print("WARNING: Differences detected.")

    def run(self):
        df_raw = self.load_raw()
        df_gt = self.transform_manual(df_raw.copy())
        df_my = self.transform_library(df_raw.copy())
        self.compare(df_my, df_gt)

if __name__ == "__main__":
    demo = BBMSanity()
    demo.run()
