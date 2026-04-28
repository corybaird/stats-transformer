import logging
import numpy as np
import os
import pandas as pd
import sys
import yaml
from datetime import datetime

SUFFIX_TO_MONTHS = {"Y": 12, "Q": 3, "M": 1, "D": 1}
TARGET_TO_MONTHS = {"annual": 12, "quarterly": 3, "monthly": 1, "daily": 1 / 30}
PERIOD_TO_SUFFIX = {"daily": "D", "monthly": "M", "quarterly": "Q", "annual": "Y"}
FREQ_ORDER = {"daily": 0, "monthly": 1, "quarterly": 2, "annual": 3}

class FeatureEngineer:

    def __init__(self, params_path=None, transformations=None, entity_column=None, date_column="date", period="annual", country_list=None, start_date=None, data_columns=None, verbose=True):
        self.verbose = verbose
        self._setup_logging()
        self.params = {}
        
        if params_path:
            self.params = self._load_params(params_path)
            feat_cfg = self.params.get("data", {}).get("featurization", self.params.get("featurization", {}))
            self.transformations = feat_cfg.get("transformations", [])
            self.period = feat_cfg.get("period", "annual").lower()
            self.entity_column = feat_cfg.get("entity_column", None)
            self.date_column = feat_cfg.get("date_column", "date")
            self.country_list = feat_cfg.get("country_list", [])
            self.start_date = feat_cfg.get("start_date", None)
            self.data_columns = feat_cfg.get("transformation_data", [])
        else:
            self.transformations = transformations or []
            self.period = period.lower()
            self.entity_column = entity_column
            self.date_column = date_column
            self.country_list = country_list or []
            self.start_date = start_date
            self.data_columns = data_columns or []

        if not self.transformations:
            self.logger.warning("No transformations specified")
        if not self.entity_column:
            self.logger.warning("No entity column specified")
        if not self.data_columns:
            self.logger.warning("No data columns specified for transformations")

    def _is_running_in_jupyter(self):
        try:
            from IPython import get_ipython
            return get_ipython() is not None
        except ImportError:
            return False

    def _setup_logging(self):
        self.logger = logging.getLogger(__name__)
        should_be_verbose = self.verbose and not self._is_running_in_jupyter()
        
        if should_be_verbose:
            self.logger.setLevel(logging.INFO)
            if not self.logger.handlers:
                handler = logging.StreamHandler(sys.stdout)
                handler.setLevel(logging.INFO)
                handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
                self.logger.addHandler(handler)
        else:
            self.logger.setLevel(logging.CRITICAL + 1)
            for handler in self.logger.handlers[:]:
                self.logger.removeHandler(handler)
            self.logger.addHandler(logging.NullHandler())

    def _load_params(self, params_path):
        try:
            with open(params_path, "r") as f:
                params = yaml.safe_load(f)
            return params
        except FileNotFoundError:
            error_msg = f"Parameter file {params_path} not found."
            self.logger.error(error_msg)
            raise FileNotFoundError(error_msg)
        except yaml.YAMLError as e:
            error_msg = f"Error parsing parameter file {params_path}: {str(e)}"
            self.logger.error(error_msg)
            raise

    def load_data(self, data_path):
        self.logger.info(f"Loading data from {data_path}")
        if not data_path.endswith(".csv") and not data_path.endswith(".parquet"):
            error_msg = f"File {data_path} is not supported. Use CSV or Parquet."
            self.logger.error(error_msg)
            raise ValueError(error_msg)

        try:
            if data_path.endswith(".parquet"):
                df = pd.read_parquet(data_path)
            else:
                df = pd.read_csv(data_path)
            self.logger.info(f"Data loaded successfully with shape: {df.shape}")
            return df
        except FileNotFoundError:
            error_msg = f"File {data_path} not found."
            self.logger.error(error_msg)
            raise FileNotFoundError(error_msg)
        except Exception as e:
            error_msg = f"Error loading data from {data_path}: {str(e)}"
            self.logger.error(error_msg)
            raise

    def _validate_data_structure(self, df):
        required_columns = [self.entity_column, self.date_column] + self.data_columns
        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            error_msg = f"Required columns are missing from the data: {missing_columns}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)

    def truncate_data(self, df):
        if not self.country_list:
            self.logger.info("No country list specified, skipping truncation")
            return df

        self.logger.info(f"Truncating data to entities: {self.country_list}")
        available_countries = set(df[self.entity_column].unique())
        requested_countries = set(self.country_list)
        missing_countries = requested_countries - available_countries

        if missing_countries:
            self.logger.warning(f"The following entities were not found in the data: {missing_countries}")

        matching_countries = requested_countries.intersection(available_countries)
        if not matching_countries:
            error_msg = f"None of the specified entities {self.country_list} were found in the data."
            self.logger.error(error_msg)
            raise ValueError(error_msg)

        filtered_df = df[df[self.entity_column].isin(self.country_list)]
        self.logger.info(f"Data truncated to shape: {filtered_df.shape}")
        return filtered_df

    def filter_by_date(self, df):
        if not self.start_date:
            self.logger.info("No start date specified, skipping date filtering")
            return df

        self.logger.info(f"Filtering data from start date: {self.start_date}")

        if self.period == "annual":
            datetime_format = "%Y"
        elif self.period == "monthly":
            datetime_format = "%Y-%m"
        elif self.period == "quarterly":
            datetime_format = "%Y-Q%q"
        elif self.period == "daily":
            datetime_format = "%Y-%m-%d"
        else:
            datetime_format = None

        try:
            start_date_dt = pd.to_datetime(self.start_date)
            try:
                df[self.date_column] = pd.to_datetime(df[self.date_column])
            except Exception as e:
                error_msg = f"Error converting '{self.date_column}' column to datetime: {str(e)}"
                self.logger.error(error_msg)
                raise ValueError(error_msg)

            filtered_df = df[df[self.date_column] >= start_date_dt]
            if filtered_df.empty:
                error_msg = f"No data remains after filtering for dates >= {self.start_date}"
                self.logger.error(error_msg)
                raise ValueError(error_msg)

            self.logger.info(f"Data filtered to shape: {filtered_df.shape}")
            return filtered_df
        except ValueError as e:
            if "start_date" in str(e) or "cannot convert" in str(e).lower():
                error_msg = f"Invalid start date format: {self.start_date}. Use YYYY-MM-DD format."
                self.logger.error(error_msg)
                raise ValueError(error_msg)
            else:
                raise

    def _get_source_periods(self, column):
        import re
        match = re.search(r"_([YQMD])(?:[a-z]+)?(?:_|$)", column)
        if match:
            freq_letter = match.group(1)
            source_months = SUFFIX_TO_MONTHS.get(freq_letter, 1)
            target_months = TARGET_TO_MONTHS.get(self.period, 1)
            return max(1, int(round(source_months / target_months)))
        return 1

    def _apply_entity_transformations(self, group):
        if self.entity_column in group.columns:
            entity = group[self.entity_column].iloc[0]
        elif hasattr(group, "name"):
            entity = group.name
        else:
            entity = "Unknown"
        transformed_group = group.copy()

        for column in self.data_columns:
            try:
                source_periods = self._get_source_periods(column)

                if "changepct" in self.transformations:
                    transformed_group[f"{column}_changepct"] = transformed_group[column].pct_change(periods=source_periods, fill_method=None)

                if "changeraw" in self.transformations:
                    transformed_group[f"{column}_changeraw"] = transformed_group[column].diff(periods=source_periods)

                if "rollingmean" in self.transformations:
                    window = 3 * source_periods
                    self.logger.info(f"Calculating rolling mean for {column} with window {window} (source_periods={source_periods}) in entity {entity}")
                    transformed_group[f"{column}_rollingmean"] = transformed_group[column].rolling(window=window).mean()

                if "log" in self.transformations:
                    if (transformed_group[column] <= 0).any():
                        error_msg = f"Non-positive values found in {column} for entity {entity}. Cannot apply log transformation."
                        self.logger.error(error_msg)
                        raise ValueError(error_msg)
                    transformed_group[f"{column}_log"] = np.log(transformed_group[column])

                if "zscore" in self.transformations:
                    mean = transformed_group[column].mean()
                    std = transformed_group[column].std()
                    if std == 0:
                        self.logger.warning(f"Standard deviation is zero for {column} in entity {entity}. Z-score not calculated.")
                    else:
                        transformed_group[f"{column}_zscore"] = (transformed_group[column] - mean) / std

                for transform in self.transformations:
                    if transform.startswith("lag"):
                        try:
                            lag = int(transform[3:])
                            transformed_group[f"{column}_{transform}"] = transformed_group[column].shift(lag * source_periods)
                        except ValueError:
                            raise ValueError(f"Invalid lag format: {transform}")

                for transform in self.transformations:
                    if transform.startswith("lead"):
                        try:
                            lead = int(transform[4:])
                            transformed_group[f"{column}_{transform}"] = transformed_group[column].shift(-lead * source_periods)
                        except ValueError:
                            raise ValueError(f"Invalid lead format: {transform}")
            except Exception as e:
                self.logger.error(f"Error calculating transformations for column '{column}' in entity '{entity}': {str(e)}")

        return transformed_group

    def transform(self, df):
        self.logger.info("Starting feature transformations.")
        missing_columns = [f for f in self.data_columns if f not in df.columns]
        if missing_columns:
            error_msg = f"The following features are not in the dataframe: {missing_columns}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)

        valid_base_transformations = ["changepct", "changeraw", "rollingmean", "log", "zscore"]
        valid_transformations = valid_base_transformations.copy()
        for transform in self.transformations:
            if transform.startswith("lag"):
                suffix = transform[3:]
                if suffix.isdigit():
                    valid_transformations.append(transform)
            elif transform.startswith("lead"):
                suffix = transform[4:]
                if suffix.isdigit():
                    valid_transformations.append(transform)

        invalid_transformations = [t for t in self.transformations if t not in valid_transformations]
        if invalid_transformations:
            error_msg = f"Invalid transformations requested: {invalid_transformations}."
            self.logger.error(error_msg)
            raise ValueError(error_msg)

        try:
            grouped = df.groupby(self.entity_column)
            transformed_df = grouped.apply(self._apply_entity_transformations).reset_index(drop=True)
            self.logger.info("Feature transformations completed.")
            return transformed_df
        except Exception as e:
            self.logger.error(f"An error occurred during the transformation process: {str(e)}")
            raise

    def fit_transform(self, df):
        try:
            self.logger.info("Starting feature engineering pipeline.")
            if not self.data_columns:
                exclude_cols = [self.entity_column, self.date_column]
                numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
                self.data_columns = [col for col in numeric_cols if col not in exclude_cols]
            
            self._validate_data_structure(df)
            df = self.truncate_data(df)
            df = self.filter_by_date(df)
            transformed_df = self.transform(df)
            self.logger.info("Feature engineering pipeline completed successfully.")
            return transformed_df
        except Exception as e:
            self.logger.error(f"Feature engineering pipeline failed: {str(e)}")
            raise

    def _log_data_quality_metrics(self, df):
        self.logger.info(f"Final dataframe shape: {df.shape}")
        self.logger.info(f"Missing values: {df.isnull().sum().sum()}")
        if self.entity_column in df.columns:
            num_entities = df[self.entity_column].nunique()
            self.logger.info(f"Number of entities: {num_entities}")
        if self.date_column in df.columns:
            min_date = df[self.date_column].min()
            max_date = df[self.date_column].max()
            self.logger.info(f"Date range: {min_date} to {max_date}")

    def _add_frequency_suffix(self, df, source_period, agg_method=None, resampled=False):
        suffix = PERIOD_TO_SUFFIX.get(source_period.lower(), "")
        if not suffix:
            self.logger.warning(f"Unknown source_period '{source_period}', no suffix added")
            return df
        if resampled and agg_method:
            suffix = f"{suffix}{agg_method}"
        skip_cols = {self.entity_column, self.date_column, "country", "date", "countryname"}
        rename_map = {col: f"{col}_{suffix}" for col in df.columns if col not in skip_cols}
        self.logger.info(f"Adding suffix '_{suffix}' to {len(rename_map)} data columns")
        return df.rename(columns=rename_map)

    def _upsample_entity(self, group, entity_col, date_col, freq, fill_method):
        entity_value = group[entity_col].iloc[0]
        group = group.drop(columns=[entity_col]).set_index(date_col).sort_index()
        upsampled = group.resample(freq).asfreq()
        if fill_method == "interpolate":
            upsampled = upsampled.interpolate()
        elif fill_method == "bfill":
            upsampled = upsampled.bfill()
        else:
            upsampled = upsampled.ffill()
        upsampled = upsampled.reset_index()
        upsampled[entity_col] = entity_value
        return upsampled

    def resample_dataset(self, df, dataset_config):
        entity_col = dataset_config.get("entity_column", self.entity_column)
        date_col = dataset_config.get("date_column", self.date_column)
        source_period = dataset_config.get("source_period", "annual")
        target_period = dataset_config.get("resample", {}).get("target_period", "monthly")
        agg_method = dataset_config.get("resample", {}).get("agg", "mean")

        freq_map = {"annual": "YS", "quarterly": "QS", "monthly": "MS", "daily": "D"}
        freq = freq_map.get(target_period.lower(), "MS")

        if df[date_col].dtype == "float64":
            df[date_col] = df[date_col].fillna(0).astype(int).astype(str)
        elif df[date_col].dtype == "int64":
            df[date_col] = df[date_col].astype(str)
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.dropna(subset=[date_col])

        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        cols_to_keep = list(set([entity_col, date_col] + numeric_cols))
        df = df[cols_to_keep]

        source_order = FREQ_ORDER.get(source_period.lower(), 3)
        target_order = FREQ_ORDER.get(target_period.lower(), 1)
        is_upsampling = source_order > target_order

        if is_upsampling:
            self.logger.info(f"Upsampling from {source_period} to {target_period} using {agg_method}")
            resampled_df = df.groupby(entity_col, group_keys=False).apply(self._upsample_entity, entity_col=entity_col, date_col=date_col, freq=freq, fill_method=agg_method).reset_index(drop=True)
        elif source_order == target_order:
            self.logger.info(f"No resampling needed for {source_period} to {target_period}")
            resampled_df = df.copy()
        else:
            self.logger.info(f"Downsampling from {source_period} to {target_period} using {agg_method}")
            grouper = [entity_col, pd.Grouper(key=date_col, freq=freq)]
            if agg_method == "sum":
                resampled_df = df.groupby(grouper).sum().reset_index()
            elif agg_method == "first":
                resampled_df = df.groupby(grouper).first().reset_index()
            elif agg_method == "last":
                resampled_df = df.groupby(grouper).last().reset_index()
            else:
                resampled_df = df.groupby(grouper).mean().reset_index()

        resampled_df = resampled_df.rename(columns={entity_col: "country", date_col: "date"})
        was_resampled = source_order != target_order
        resampled_df = self._add_frequency_suffix(resampled_df, source_period, agg_method=agg_method, resampled=was_resampled)
        self.logger.info(f"Resampled {dataset_config.get('name', 'dataset')}: {resampled_df.shape}")
        return resampled_df

    def save_features(self, df, output_path):
        try:
            self._log_data_quality_metrics(df)
        except Exception as e:
            error_msg = f"Error saving features to {output_path}: {str(e)}"
            self.logger.error(error_msg)
            raise

    def run(self, data_path=None, output_path=None):
        if not data_path and self.params:
            data_path = self.params.get("data", {}).get("raw_data_file")
        if not output_path and self.params:
            output_path = self.params.get("featurization", {}).get("output_path")
        if not data_path:
            raise ValueError("No data path provided or found in params")

        df = self.load_data(data_path)
        df_transformed = self.fit_transform(df)
        if output_path:
            self.save_features(df_transformed, output_path)
        return df_transformed

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run feature engineering pipeline")
    parser.add_argument("--config", type=str, help="Path to params.yaml", default="params.yaml")
    args = parser.parse_args()
    engineer = FeatureEngineer(params_path=args.config)
    engineer.run()
