import logging
import os
import pandas as pd
import yaml
from references.dictionaries.COUNTRY_MAPPING import ISO2_TO_ISO3

class DataMerger:

    def __init__(self, params_path="params.yaml"):
        self.params_path = params_path
        self.logger = logging.getLogger(__name__)
        self._setup_logging()
        self.config = self._load_config()

    def _setup_logging(self):
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
            self.logger.addHandler(handler)

    def _load_config(self):
        with open(self.params_path, "r") as f:
            return yaml.safe_load(f)

    def load_dataset(self, path):
        if path.endswith(".parquet"):
            return pd.read_parquet(path)
        return pd.read_csv(path)

    def _log_merge_diagnostics(self, df_left, df_right, merged_df, entity_col):
        rows_left, rows_right, rows_merged = len(df_left), len(df_right), len(merged_df)
        if entity_col in df_left.columns:
            countries_left = set(df_left[entity_col].unique())
        else:
            countries_left = set(df_left.index.get_level_values(entity_col).unique())
        if entity_col in df_right.columns:
            countries_right = set(df_right[entity_col].unique())
        else:
            countries_right = set(df_right.index.get_level_values(entity_col).unique())
        if entity_col in merged_df.columns:
            countries_merged = set(merged_df[entity_col].unique())
        elif merged_df.index.names and entity_col in merged_df.index.names:
            countries_merged = set(merged_df.index.get_level_values(entity_col).unique())
        else:
            countries_merged = set()
        dropped = countries_left - countries_merged
        self.logger.info(f"Left:   {rows_left:>7} rows, {len(countries_left):>4} countries")
        self.logger.info(f"Right:  {rows_right:>7} rows, {len(countries_right):>4} countries")
        self.logger.info(f"Merged: {rows_merged:>7} rows, {len(countries_merged):>4} countries")
        if dropped:
            self.logger.warning(f"Dropped countries from left: {sorted(dropped)}")

    def merge(self, df_left, df_right, on, how="inner"):
        merged_df = pd.merge(df_left, df_right, on=on, how=how)
        entity_col = on[0] if on else "country"
        self._log_merge_diagnostics(df_left, df_right, merged_df, entity_col)
        return merged_df

    def merge_on_index(self, df_left, df_right, how="inner"):
        merged_df = pd.merge(df_left, df_right, how=how, left_index=True, right_index=True)
        entity_col = "country"
        if "country" in df_left.index.names:
            self._log_merge_diagnostics(df_left, df_right, merged_df, entity_col)
        return merged_df

    def merge_asof(self, df_left, df_right, left_on, right_on, by="country", direction="backward"):
        df_left_sorted = df_left.sort_values(by=left_on)
        df_right_sorted = df_right.sort_values(by=right_on)
        merged_df = pd.merge_asof(df_left_sorted, df_right_sorted, left_on=left_on, right_on=right_on, by=by, direction=direction)
        self._log_merge_diagnostics(df_left, df_right, merged_df, by)
        return merged_df

    def standardize_entity(self, df, entity_col):
        df = df.rename(columns={entity_col: "country"})
        df["country"] = df["country"].map(lambda x: ISO2_TO_ISO3.get(x, x))
        return df

    def run(self):
        data_config = self.config.get("data", {})
        datasets_config = data_config.get("datasets", [])

        if not datasets_config:
            self.logger.warning("No datasets found in config to merge")
            return None

        loaded_datasets = []
        for ds_cfg in datasets_config:
            name = ds_cfg.get("name")
            self.logger.info(f"Loading dataset: {name}")
            df = self.load_dataset(ds_cfg.get("path"))
            entity_col = ds_cfg.get("entity_column", "country")
            date_col = ds_cfg.get("date_column", "date")
            if entity_col != "country":
                df = self.standardize_entity(df, entity_col)
            if date_col != "date":
                df = df.rename(columns={date_col: "date"})
            loaded_datasets.append(df)

        if not loaded_datasets:
            return None

        merge_config = data_config.get("merge", {})
        merge_on = merge_config.get("on", ["country", "date"])
        merge_how = merge_config.get("how", "outer")
        output_path = merge_config.get("output_path", "data/pipeline/resampled_merged.parquet")

        merged_df = loaded_datasets[0]
        for i, next_df in enumerate(loaded_datasets[1:], start=1):
            self.logger.info(f"Merging dataset {i + 1} of {len(loaded_datasets)}")
            merged_df = self.merge(merged_df, next_df, on=merge_on, how=merge_how)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        merged_df.to_parquet(output_path)
        self.logger.info(f"Merged data saved to {output_path} with shape {merged_df.shape}")
        return merged_df

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="params.yaml")
    args = parser.parse_args()
    dm = DataMerger(params_path=args.config)
    dm.run()
