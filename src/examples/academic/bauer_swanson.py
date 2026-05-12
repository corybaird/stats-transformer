import pandas as pd
import sys
import numpy as np
from pathlib import Path

if "src" not in sys.path:
    sys.path.append("src")

from stats_transformer.featurization.feature_engineering import FeatureEngineer

class BauerSwansonExample:
    def __init__(self):
        self.data_dir = Path("data") / "examples" / "academic" / "bauer_swanson_2023"
        self.nfp_path = self.data_dir / "NonfarmPayrolls.txt"
        self.unemp_path = self.data_dir / "Unemployment.txt"

    def load_raw(self, path, name):
        # Loads raw data from Bauer & Swanson (2023) text files.
        print(f"Loading raw {name} data from {path}...")
        df = pd.read_csv(path, sep=r"\s+", header=None, names=["year", "month", "value"])
        df["date"] = pd.to_datetime(df["year"].astype(str) + "-" + df["month"].astype(str) + "-01")
        df["country"] = "USA"
        return df

    def transform_manual(self, df, type):
        # Manually implements the transformation logic from Bauer_Swanson_2023/MATLAB/loadmacrodata.m.
        if type == "nfp":
            df["log_val"] = np.log(df["value"])
            df["target"] = df["log_val"].diff()
        elif type == "unemp":
            df["target"] = df["value"].shift(1)
        return df

    def transform_library(self, df, type):
        # Uses FeatureEngineer to replicate the paper's logic.
        if type == "nfp":
            # Step 1: Log transformation
            engineer_log = FeatureEngineer(
                params_path=None,
                transformations=["log"],
                entity_column="country",
                date_column="date",
                period="monthly",
                data_columns=["value"],
                verbose=False,
            )
            df_log = engineer_log.fit_transform(df)
            
            # Step 2: First difference of the log column
            # Note: We must ensure 'country' and 'date' are preserved
            engineer_diff = FeatureEngineer(
                params_path=None,
                transformations=["changeraw"],
                entity_column="country",
                date_column="date",
                period="monthly",
                data_columns=["value_log"],
                verbose=False,
            )
            return engineer_diff.fit_transform(df_log)
        else:
            # Replicating lag
            engineer = FeatureEngineer(
                params_path=None,
                transformations=["lag1"],
                entity_column="country",
                date_column="date",
                period="monthly",
                data_columns=["value"],
                verbose=False,
            )
            return engineer.fit_transform(df)

    def compare(self, df_my, df_gt, label):
        print(f"\n====== Comparing {label} Transformations ======")
        
        cols = [c for c in df_my.columns if any(t in c for t in ["log_changeraw", "lag1"])]
        if not cols:
            print(f"Error: Could not find transformed column in {df_my.columns}")
            return
        col_mine = cols[0]

        merged = pd.merge(
            df_my[["date", col_mine]],
            df_gt[["date", "target"]],
            on="date"
        )
        
        merged["absolute_error"] = (merged[col_mine] - merged["target"]).abs()
        comparison = merged.dropna()
        
        max_error = comparison["absolute_error"].max()
        print(f"Number of observations: {len(comparison)}")
        print(f"Maximum Absolute Error: {max_error:.12f}")
        
        if max_error < 1e-10:
            print(f"SUCCESS: {label} transformations perfectly match Matlab logic.")
        else:
            print(f"WARNING: Differences detected in {label}.")

    def run(self):
        # 1. Nonfarm Payrolls (Log-Difference)
        df_nfp_raw = self.load_raw(self.nfp_path, "NFP")
        df_nfp_gt = self.transform_manual(df_nfp_raw.copy(), "nfp")
        df_nfp_lib = self.transform_library(df_nfp_raw.copy(), "nfp")
        self.compare(df_nfp_lib, df_nfp_gt, "Nonfarm Payrolls (Log-Diff)")

        # 2. Unemployment (Lag)
        df_unemp_raw = self.load_raw(self.unemp_path, "Unemployment")
        df_unemp_gt = self.transform_manual(df_unemp_raw.copy(), "unemp")
        df_unemp_lib = self.transform_library(df_unemp_raw.copy(), "unemp")
        self.compare(df_unemp_lib, df_unemp_gt, "Unemployment (Lag)")

if __name__ == "__main__":
    demo = BauerSwansonExample()
    demo.run()
