import json
import logging
import os
import re
import shutil
import warnings
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
import pandas as pd
from stats_transformer.featurization.feature_engineering import FeatureEngineer
from stats_transformer.featurization.data_merger import DataMerger
from stats_transformer.models.base import ModelBase
from stats_transformer.models.builder import ModelBuilder
from stats_transformer.reporting.models.regression_viz import RegressionVisualizer
from stats_transformer.reporting.eda.data_viz import DataVisualizer
from stats_transformer.reporting.eda.eda import EDAVisualizer
from stats_transformer.reporting.tables.pipeline_tables import PipelineTables
from stats_transformer.utils.config import Config, resolve_reporting_config
from stats_transformer.utils.run_context import RunContext

# Canonical research stages: data (ingest, resample, merge) -> features (transform)
# -> model (estimate, diagnostics, JSON summary) -> reporting (charts, tables).
STAGES = ("data", "features", "model", "reporting")

# Legacy stage names accepted by Pipeline.run(); each maps to a canonical stage.
_STAGE_ALIASES = {"resample": "data", "regression": "model", "visualization": "reporting"}

class Pipeline:
    """End-to-end research pipeline: data -> features -> model -> reporting stages."""

    def __init__(self, params_path: Optional[str] = None, transformations: Optional[List[Dict[str, Any]]] = None, entity_column: Optional[str] = None, date_column: str = "date", target: Optional[str] = None, features: Optional[List[str]] = None, add_entity_fixed_effects: bool = False, **kwargs: Any) -> None:
        self.logger = logging.getLogger(__name__)
        self.params_path = params_path
        self.entity_column = entity_column
        self.date_column = date_column
        self.target = target
        self.features = features
        self.transformations = transformations
        self.add_entity_fixed_effects = add_entity_fixed_effects
        self.kwargs = kwargs
        self.feature_engineer: Optional[FeatureEngineer] = None
        self.model: Optional[ModelBase] = None
        self.transformed_data: Optional[pd.DataFrame] = None
        self.model_results: Optional[Dict[str, Any]] = None
        self._config: Optional[Config] = None
        self._run_ctx = RunContext(lambda: self._get_config(), params_path)
        self._builder = ModelBuilder(self)
        self._tables = PipelineTables(self)

    # -- Full run ---------------------------------------------------------

    def run(self, stage: Optional[str] = None) -> Any:
        """Run the four canonical stages in order, or a single named stage."""
        if self.params_path:
            self._initialize_from_params()
            assert self.feature_engineer is not None  # set unconditionally by _initialize_from_params
        else:
            raise ValueError("Pipeline.run() requires a params_path")

        if stage is not None and stage in _STAGE_ALIASES:
            canonical = _STAGE_ALIASES[stage]
            warnings.warn(f'Pipeline stage "{stage}" is deprecated and will be removed in a future release; use "{canonical}".', DeprecationWarning, stacklevel=2)
            stage = canonical
        if stage is not None and stage != "eda" and stage not in STAGES:
            raise ValueError(f"Unknown stage '{stage}'. Valid values: {', '.join(STAGES)} (or 'eda'), or None for a full run")

        if stage is not None:
            return self._run_stage(stage)

        # Full run (stage is None): data -> features -> model -> reporting.
        # Configs without data.datasets feed raw_data_file straight into features.
        has_datasets = bool(self._get_config().to_dict().get("data", {}).get("datasets"))
        ctx = self._run_ctx
        ctx.run_dir = None
        ctx.full_run_active = True
        try:
            if has_datasets:
                self.run_data()
            transformed_data = self.run_features()
            self.run_model(transformed_data=transformed_data)
            self.run_reporting()
        finally:
            ctx.full_run_active = False
            ctx.write_manifest()
            ctx.run_dir = None
        return transformed_data

    def _run_stage(self, stage: str) -> Any:
        """Dispatch a validated single-stage run."""
        if stage == "data":
            return self.run_data()
        if stage == "features":
            return self.run_features()
        if stage == "eda":
            return self.run_eda()
        if stage == "model":
            return self.run_model()
        return self.run_reporting()

    # -- Stage 1: data ------------------------------------------------------

    def run_data(self) -> pd.DataFrame:
        """Data stage: ingest, resample to a common frequency, and merge."""
        paths = self._resolve_paths()
        assert self.feature_engineer is not None  # set by _resolve_paths via _initialize_from_params
        data_config = paths["params"].get("data", {})
        # Validate before begin_run(): a data stage without datasets must not
        # leave an empty run folder behind.
        datasets = data_config.get("datasets") or []
        if not datasets:
            raise ValueError("Data stage requires data.datasets in the config; this config only sets data.raw_data_file, which the features stage consumes directly (a full run skips the data stage automatically).")
        ctx = self._run_ctx
        run_dir = ctx.begin_run()
        # The canonical chaining location stays exactly as on main: the explicit
        # merge output_path when set, or the data/pipeline default. The run folder
        # also always receives a copy so each run is self-contained.
        output_path = paths["merged_output_path"]

        raw_dir = os.path.join(run_dir, "01_data", "raw")
        os.makedirs(raw_dir, exist_ok=True)
        resampled = []
        for ds_cfg in datasets:
            self.logger.info(f"Resampling dataset: {ds_cfg.get('name')}")
            df = self.feature_engineer.load_data(ds_cfg.get("path"))
            # Raw (pre-resample) source copy, named after the config's dataset key.
            raw_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(ds_cfg.get("name") or os.path.splitext(os.path.basename(str(ds_cfg.get("path"))))[0]))
            raw_path = os.path.join(raw_dir, f"{raw_name}.parquet")
            df.to_parquet(raw_path)
            ctx.files.append(raw_path)
            ctx.shapes.setdefault("raw", {})[raw_name] = list(df.shape)
            resampled_df = self.feature_engineer.resample_dataset(df, ds_cfg)
            resampled.append(resampled_df)
        dm = DataMerger(params_path=self.params_path)
        merge_config = data_config.get("merge", {})
        merge_on = merge_config.get("on", ["country", "date"])
        merge_how = merge_config.get("how", "outer")
        merged_df = resampled[0]
        for i, next_df in enumerate(resampled[1:], start=1):
            self.logger.info(f"Merging dataset {i + 1} of {len(resampled)}")
            merged_df = dm.merge(merged_df, next_df, on=merge_on, how=merge_how)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        merged_df.to_parquet(output_path)
        self.logger.info(f"Merged data saved to {output_path} with shape {merged_df.shape}")
        ctx.files.append(output_path)
        ctx.shapes["merged"] = list(merged_df.shape)
        run_copy = os.path.join(run_dir, "01_data", "merged.parquet")
        os.makedirs(os.path.dirname(run_copy), exist_ok=True)
        shutil.copyfile(output_path, run_copy)
        ctx.files.append(run_copy)
        ctx.end_stage("data")
        return merged_df

    # -- Stage 2: features --------------------------------------------------

    def run_features(self, data_path: Optional[str] = None) -> pd.DataFrame:
        """Features stage: apply transformations to the merged data."""
        paths = self._resolve_paths()
        data_path = data_path or paths["effective_data_path"]
        processed_path = paths["processed_path"]
        if not data_path:
            raise ValueError("No data path found for features stage")
        transformed_data = self.fit_transform(data_path, fit_model=False)
        if processed_path:
            transformed_data.to_csv(processed_path, index=False)
        ctx = self._run_ctx
        run_dir = ctx.begin_run()
        features_path = os.path.join(run_dir, "02_features", "features.parquet")
        os.makedirs(os.path.dirname(features_path), exist_ok=True)
        # Parquet (not the CSV above) keeps dtypes such as dates.
        transformed_data.to_parquet(features_path)
        if processed_path:
            ctx.files.append(processed_path)
        ctx.files.append(features_path)
        ctx.shapes["features"] = list(transformed_data.shape)
        ctx.end_stage("features")
        return transformed_data

    # -- Stage 3: model -----------------------------------------------------

    def run_model(self, data_path: Optional[str] = None, transformed_data: Optional[pd.DataFrame] = None) -> Optional[Dict[str, Any]]:
        """Model stage: fit the configured estimator and save its JSON summary."""
        paths = self._resolve_paths()
        processed_path = paths["processed_path"]
        summary_path = paths["summary_path"]

        if transformed_data is not None:
            # Full run: fit on the in-memory features, not the CSV round-trip (keeps dtypes such as dates)
            if self.model and self._builder.has_model_target():
                self.model.fit(transformed_data)
                self.model_results = self.model.get_model_metadata()
        elif processed_path and os.path.exists(processed_path):
            assert self.model is not None  # set unconditionally by _initialize_from_params
            transformed_data = pd.read_csv(processed_path)
            self.transformed_data = transformed_data
            self.model.fit(transformed_data)
            self.model_results = self.model.get_model_metadata()
        else:
            data_path = data_path or paths["effective_data_path"]
            self.fit_transform(data_path)

        # Model artefacts go to the run folder; the hardcoded reports/ default is gone.
        ctx = self._run_ctx
        run_dir = ctx.begin_run()
        model_dir = os.path.join(run_dir, "03_model")
        if self.model_results is not None:
            summary_run_path = os.path.join(model_dir, "model_summary.json")
            self.save_model_summary(summary_run_path)
            ctx.files.append(summary_run_path)
        if summary_path:
            self.save_model_summary(summary_path)
            ctx.files.append(summary_path)
        if self.model is not None:  # args-mode pipelines without target/features have no model
            model_df = self._tables.model_data()
            if model_df is not None:
                # The exact data the model was fitted on (df_clean restricted to the
                # columns the model uses); the descriptive-stats table reads the same.
                model_data_path = os.path.join(model_dir, "model_data.parquet")
                model_df.to_parquet(model_data_path)
                ctx.files.append(model_data_path)
                ctx.shapes["model_data"] = list(model_df.shape)
            try:
                model_path = os.path.join(model_dir, "model.joblib")
                self.model.save(model_path)
                ctx.files.append(model_path)
            except Exception as e:
                # An unpicklable attribute must not fail the run.
                self.logger.warning(f"Could not save fitted model {type(self.model).__name__}: {e}")
        ctx.end_stage("model")
        return self.model_results

    # -- Stage 4: reporting -------------------------------------------------

    def run_reporting(self) -> Dict[str, Any]:
        """Reporting stage: draw charts and tables from transformed data and model results."""
        paths = self._resolve_paths()
        processed_path = paths["processed_path"]
        summary_path = paths["summary_path"]

        if (self.transformed_data is None) and processed_path and os.path.exists(processed_path):
            self.transformed_data = pd.read_csv(processed_path)
        if (self.model_results is None) and summary_path and os.path.exists(summary_path):
            with open(summary_path, "r") as f:
                self.model_results = json.load(f)

        report_config = resolve_reporting_config(paths["params"])
        ctx = self._run_ctx
        run_dir = ctx.begin_run()
        # Charts live under the run folder's 04_reporting; an explicit
        # reporting.output_dir still receives a copy when one is configured.
        results = self.create_visualizations(output_dir=os.path.join(run_dir, "04_reporting"))
        explicit_output_dir = report_config.get("output_dir")
        chart_values: List[Any] = []
        for value in results.values():
            chart_values.extend(value if isinstance(value, list) else [value])
        for path in chart_values:
            if isinstance(path, str) and os.path.isfile(path):
                ctx.files.append(path)
                if explicit_output_dir:
                    os.makedirs(explicit_output_dir, exist_ok=True)
                    ctx.files.append(shutil.copyfile(path, os.path.join(explicit_output_dir, os.path.basename(path))))
        self._tables.generate_tables(report_config.get("tables"), run_dir)
        ctx.end_stage("reporting")
        return results

    def run_eda(self, data_path: Optional[str] = None) -> Any:
        """EDA: exploratory charts of the merged (or raw) data."""
        paths = self._resolve_paths()
        data_path = data_path or paths["effective_data_path"]
        if not data_path:
            raise ValueError("No data path found for eda stage")
        eda_viz = EDAVisualizer(params_path=self.params_path, output_dir=paths["report_output_dir"])
        return eda_viz.run(data_path=data_path)

    # -- Feature/model API --------------------------------------------------

    def fit_transform(self, data: Union[str, pd.DataFrame], fit_model: bool = True) -> pd.DataFrame:
        """Fit the feature engineer on raw data (and the model when requested) and return the transformed data."""
        self.logger.info("Starting pipeline fit_transform")
        if self.params_path:
            self._initialize_from_params()
        else:
            self._initialize_from_args()

        df = self._read_raw(data)

        if self.feature_engineer:
            self.transformed_data = self.feature_engineer.fit_transform(df)
        else:
            raise ValueError("No feature engineering component initialized")

        if fit_model and self.model and self._builder.has_model_target():
            self.model.fit(self.transformed_data)
            self.model_results = self.model.get_model_metadata()

        return self.transformed_data

    def transform(self, data: Union[str, pd.DataFrame]) -> Optional[pd.DataFrame]:
        """Apply the fitted feature engineering to raw data without refitting."""
        if self.params_path:
            self._initialize_from_params()
        else:
            self._initialize_from_args()

        df = self._read_raw(data)

        if self.feature_engineer:
            self.transformed_data = self.feature_engineer.transform(df)
        return self.transformed_data

    def predict(self, data: Optional[Union[str, pd.DataFrame]] = None) -> Any:
        """Predict with the fitted model on (default: the transformed training) data."""
        if not self.model or not hasattr(self.model, 'model') or self.model.model is None:
            raise ValueError("Model must be fitted before making predictions")

        data_to_predict: Optional[pd.DataFrame]
        if data is None:
            data_to_predict = self.transformed_data
        elif isinstance(data, str):
            data_to_predict = pd.read_csv(data) if data.endswith(".csv") else pd.read_parquet(data)
        else:
            data_to_predict = data.copy()

        if self.feature_engineer:
            data_to_predict = self.feature_engineer.transform(data_to_predict)

        if not hasattr(self.model, "predict"):
            raise NotImplementedError(f"{type(self.model).__name__} does not implement predict.")
        return self.model.predict(data_to_predict)

    def save_results(self, output_dir: str = "reports") -> Dict[str, str]:
        """Save transformed data and the model summary JSON to output_dir."""
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results: Dict[str, str] = {}

        if self.transformed_data is not None:
            data_path = os.path.join(output_dir, f"transformed_data_{timestamp}.csv")
            self.transformed_data.to_csv(data_path, index=False)
            results["transformed_data"] = data_path

        if self.model_results is not None:
            stable_path = os.path.join(output_dir, "model_summary.json")
            with open(stable_path, "w") as f:
                json.dump(self.model_results, f, indent=4)
            results["model_summary"] = stable_path

        return results

    def save_model_summary(self, output_path: str) -> None:
        """Write the model summary JSON to output_path (no-op without results)."""
        if self.model_results is None:
            return
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(self.model_results, f, indent=4)

    def create_visualizations(self, output_dir: str = "reports/visualizations", visualization_types: Optional[List[str]] = None, display_only: bool = False) -> Dict[str, Any]:
        """Create regression and time-series charts from the pipeline's results."""
        if not self.model_results and self.transformed_data is None:
            return {}

        report_config = {}
        if self.params_path and os.path.exists(self.params_path):
            report_config = self._get_config().get_reporting_config()

        results = {}
        if self.model_results:
            reg_viz = RegressionVisualizer(params_path=self.params_path, output_dir=output_dir, file_format=report_config.get("file_format", "png"), dpi=report_config.get("dpi", 300), style=report_config.get("style", "default"))
            viz_types = visualization_types or report_config.get("regression", {}).get("plots", ["coefficient_plot", "model_summary", "residuals"])
            for viz_type in viz_types:
                try:
                    if viz_type == "coefficient_plot":
                        results["coefficient_plot"] = reg_viz.create_coefficient_plot_from_json(self.model_results, subdir="regression", display_only=display_only)
                    elif viz_type == "model_summary":
                        results["model_summary"] = reg_viz.create_model_summary_plot(self.model_results, subdir="regression", display_only=display_only)
                except Exception as e:
                    self.logger.error(f"Error creating {viz_type} visualization: {e}")

        if self.transformed_data is not None:
            data_viz = DataVisualizer(params_path=self.params_path, output_dir=output_dir, file_format=report_config.get("file_format", "png"), dpi=report_config.get("dpi", 300), style=report_config.get("style", "default"))
            try:
                if self.entity_column and self.date_column:
                    results["time_series"] = data_viz.create_time_series_plot(self.transformed_data, entity_column=self.entity_column, date_column=self.date_column, display_only=display_only)
            except Exception as e:
                self.logger.error(f"Error creating data visualization: {e}")
        return results

    # -- Plumbing -----------------------------------------------------------

    def _get_config(self) -> Config:
        """Load and validate the config once, then cache it."""
        if self._config is None:
            self._config = Config(config_path=self.params_path)
            self._config.validate()
        return self._config

    def _initialize_from_params(self) -> None:
        """Initialize the feature engineer and model from the params file."""
        params = self._get_config().to_dict()

        if not self.target:
            self.target = params.get("model", {}).get("target_variable")

        if not self.features:
            self.features = params.get("model", {}).get("independent_variables", [])

        if not self.entity_column:
            self.entity_column = params.get("data", {}).get("featurization", {}).get("entity_column")

        self.feature_engineer = FeatureEngineer(params_path=self.params_path)

        model_type = params.get("model", {}).get("model_type", "ols").lower()
        entry = self._builder.resolve_registry_entry(model_type)
        self.model = self._builder.build_from_params(entry)

    def _initialize_from_args(self) -> None:
        """Initialize the feature engineer and model from constructor arguments."""
        if not self.entity_column:
            raise ValueError("entity_column must be specified when not using params_path")

        self.feature_engineer = FeatureEngineer(params_path=None, transformations=self.transformations, entity_column=self.entity_column, date_column=self.date_column)

        if self.target or self.features:
            model_type = self.kwargs.get("model_type", "ols").lower()
            entry = self._builder.resolve_registry_entry(model_type)
            self.model = self._builder.build_from_args(entry)

    def _resolve_paths(self) -> Dict[str, Any]:
        """Resolve config-derived paths used by the stage methods."""
        if not self.params_path:
            raise ValueError("Pipeline stages require a params_path")
        # Stage methods can be called directly, without going through run()
        if self.feature_engineer is None:
            self._initialize_from_params()
        params = self._get_config().to_dict()

        data_config = params.get("data", {})
        raw_data_file = data_config.get("raw_data_file")
        merged_output_path = data_config.get("merge", {}).get("output_path", "data/pipeline/resampled_merged.parquet")

        # Prefer merged output if it exists or if we are in a later stage
        effective_data_path = merged_output_path if os.path.exists(merged_output_path) else raw_data_file

        report_config = resolve_reporting_config(params)
        return {
            "params": params,
            "merged_output_path": merged_output_path,
            "effective_data_path": effective_data_path,
            "report_output_dir": report_config.get("output_dir", "reports/visualizations"),
            "processed_path": data_config.get("featurization", {}).get("output_path"),
            "summary_path": params.get("model", {}).get("summary_output_path"),
        }

    def _read_raw(self, data: Union[str, pd.DataFrame]) -> pd.DataFrame:
        """Read a raw data file; restore configured date/entity columns stored as index."""
        if isinstance(data, str):
            df = pd.read_csv(data) if data.endswith(".csv") else pd.read_parquet(data)
        else:
            df = data.copy()
        # Parquet files written from pandas often carry the date/entity columns
        # as (named) index levels, so columns the config declares (date_column,
        # entity_column) are invisible downstream until the index is reset.
        wanted = {getattr(self.feature_engineer, "date_column", None), getattr(self.feature_engineer, "entity_column", None)} - {None}
        named = df.index.names if isinstance(df.index, pd.MultiIndex) else [df.index.name]
        if any(n in wanted for n in named):
            df = df.reset_index()
        return df

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="params.yaml")
    parser.add_argument("--stage", type=str, choices=[*STAGES, "eda", *_STAGE_ALIASES], help="Run a single stage (data, features, model, reporting; default: run them all in order). Legacy names resample/regression/visualization are accepted but deprecated.")
    args = parser.parse_args()
    pipeline = Pipeline(params_path=args.config)
    pipeline.run(stage=args.stage)