import pandas as pd
import sys
import numpy as np
from pathlib import Path

if "src" not in sys.path:
    sys.path.append("src")

from stats_transformer.featurization.feature_engineering import FeatureEngineer

class NakamuraSteinssonSanity:
    def __init__(self):
        self.data_dir = Path("data") / "examples" / "Nakamura_Steinsson_2018"
        self.raw_path = self.data_dir / "NominalYields.csv"
        self.transformed_path = self.data_dir / "master.dta"

    def load_raw(self, path):
        # This method loads the raw daily nominal yields from the Nakamura & Steinsson (2018) replication package.
        # Data Source: Stata/Data_Orig/NominalYields.csv (originally from Gurkaynak, Sack, and Wright JME 2007).
        print(f"Loading raw data from {path}...")
        df = pd.read_csv(path)
        # SVENY01 is the 1-year zero-coupon nominal yield, which authors rename to NY1.
        df.rename(columns={df.columns[0]: "date", "SVENY01": "NY1"}, inplace=True)
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)
        df["country"] = "USA"
        return df

    def load_transformed(self, path):
        # This method loads the official Stata binary .dta file produced by the Nakamura & Steinsson scripts.
        # Data Source: Stata/IntermediateFiles/master.dta.
        # This requires the 'pyreadstat' library to be installed.
        print(f"Loading official Stata .dta ground truth from {path}...")
        df = pd.read_stata(path)
        # master.dta uses 'date_daily' as the primary time index.
        df["date"] = pd.to_datetime(df["date_daily"])
        return df

    def transform_raw(self, df):
        # This method applies the library's FeatureEngineer to the raw yield data to calculate first differences.
        # The goal is to replicate the 'D.NY1' (first difference) logic found in Nakamura & Steinsson's Stata code.
        print("Applying FeatureEngineer transformations to raw data...")
        engineer = FeatureEngineer(
            params_path=None,
            transformations=["changeraw"],
            entity_column="country",
            date_column="date",
            period="daily",
            data_columns=["NY1"],
            verbose=False,
        )
        df_transformed = engineer.fit_transform(df)
        return df_transformed

    def compare_transformed(self, df_my, df_gt):
        # This method validates our library's output by merging it with the researchers' original .dta data.
        # It calculates the absolute error between 'NY1_changeraw' (ours) and 'DNY1' (theirs).
        print("\n====== Comparing Transformations against Master .dta ======")
        merged = pd.merge(
            df_my[["date", "NY1_changeraw"]],
            df_gt[["date", "DNY1"]],
            on="date",
            suffixes=("_mine", "_gt")
        )
        
        # Calculate absolute difference
        merged["absolute_error"] = (merged["NY1_changeraw"] - merged["DNY1"]).abs()
        
        # master.dta might have gaps or different sample ranges, so we drop NaNs.
        comparison = merged.dropna()
        
        print(f"Number of overlapping observation dates: {len(comparison)}")
        
        if len(comparison) == 0:
            print("ERROR: No overlapping dates found between raw data and master .dta.")
            return

        max_error = comparison["absolute_error"].max()
        mean_error = comparison["absolute_error"].mean()
        
        print(f"Maximum Absolute Error: {max_error:.8f}")
        print(f"Mean Absolute Error: {mean_error:.8f}")
        
        if max_error < 1e-6:
            print("SUCCESS: Transformations align with Nakamura & Steinsson's original .dta output.")
        else:
            print("WARNING: Significant differences detected between transformations.")
            
        print("\nSample Comparisons (Top 10):")
        print(comparison[["date", "NY1_changeraw", "DNY1", "absolute_error"]].head(10).to_string(index=False))

    def run(self):
        # Entry point for the sanity check orchestrating the loading, transformation, and comparison.
        df_raw = self.load_raw(self.raw_path)
        df_gt = self.load_transformed(self.transformed_path)
        df_my = self.transform_raw(df_raw)
        self.compare_transformed(df_my, df_gt)

if __name__ == "__main__":
    demo = NakamuraSteinssonSanity()
    demo.run()
