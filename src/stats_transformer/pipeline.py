import logging
import os
import pandas as pd
from typing import Any, Dict, List, Optional, Union
from stats_transformer.featurization.feature_engineering import FeatureEngineer
from stats_transformer.featurization.data_merger import DataMerger
from stats_transformer.visualization.models.regression_viz import RegressionVisualizer
from stats_transformer.visualization.eda.data_viz import DataVisualizer
from stats_transformer.visualization.eda.eda import EDAVisualizer
from stats_transformer.models.base import ModelBase
from stats_transformer.models.registry import MODEL_REGISTRY, MODEL_TYPE_ALIASES
from stats_transformer.utils.config import Config

class Pipeline:

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

    def _get_config(self) -> Config:
        if self._config is None:
            self._config = Config(config_path=self.params_path)
            self._config.validate()
        return self._config

    def _initialize_from_params(self) -> None:
        params = self._get_config().to_dict()

        if not self.target:
            self.target = params.get("model", {}).get("target_variable")
        
        if not self.features:
            self.features = params.get("model", {}).get("independent_variables", [])
        
        if not self.entity_column:
            self.entity_column = params.get("data", {}).get("featurization", {}).get("entity_column")
        
        self.feature_engineer = FeatureEngineer(params_path=self.params_path)

        model_type = params.get("model", {}).get("model_type", "ols").lower()
        model_type = MODEL_TYPE_ALIASES.get(model_type, model_type)
        entry = self._resolve_registry_entry(model_type)
        self.model = self._build_from_params(entry)

    def _initialize_from_args(self) -> None:
        if not self.entity_column:
            raise ValueError("entity_column must be specified when not using params_path")

        self.feature_engineer = FeatureEngineer(params_path=None, transformations=self.transformations, entity_column=self.entity_column, date_column=self.date_column)

        if self.target or self.features:
            model_type = self.kwargs.get("model_type", "ols").lower()
            model_type = MODEL_TYPE_ALIASES.get(model_type, model_type)
            entry = self._resolve_registry_entry(model_type)
            self.model = self._build_from_args(entry)

    def _resolve_registry_entry(self, model_type: str) -> Dict[str, Any]:
        entry = MODEL_REGISTRY.get(model_type)
        if entry is None:
            valid = ", ".join(sorted(MODEL_REGISTRY))
            raise ValueError(f"Unknown model_type '{model_type}'. Valid values: {valid}")
        return entry

    def _build_from_params(self, entry: Dict[str, Any]) -> ModelBase:
        cls, kind = entry["cls"], entry["kind"]
        if kind == "single_equation":
            return cls(params_path=self.params_path, add_entity_fixed_effects=self.add_entity_fixed_effects)
        if kind == "panel":
            return cls(params_path=self.params_path, entity_column=self.entity_column)
        if kind == "iv":
            return cls(params_path=self.params_path)
        if kind == "panel_iv":
            return cls(params_path=self.params_path, entity_column=self.entity_column)
        if kind == "unsupervised":
            return cls(params_path=self.params_path, features=self.features)
        if kind == "svar_family":
            return cls(params_path=self.params_path)
        if kind == "lp_iv":
            return cls(params_path=self.params_path)
        if kind == "lp":
            model_params = self._get_config().to_dict().get("model", {})
            shock_var = model_params.get("shock_variable") or (self.features[0] if self.features else None)
            controls = [f for f in (self.features or []) if f != shock_var]
            return cls(params_path=self.params_path, target=self.target, shock_var=shock_var, controls=controls, horizon=model_params.get("horizon", 8), date_column=model_params.get("date_column"))
        raise ValueError(f"Unhandled registry kind '{kind}' for {cls.__name__}")

    def _build_from_args(self, entry: Dict[str, Any]) -> ModelBase:
        cls, kind = entry["cls"], entry["kind"]
        if kind == "single_equation":
            return cls(params_path=None, target=self.target, independent_variables=self.features, add_entity_fixed_effects=self.add_entity_fixed_effects, entity_column=self.entity_column)
        if kind == "panel":
            return cls(params_path=None, target=self.target, independent_variables=self.features, entity_column=self.entity_column)
        if kind == "iv":
            return cls(params_path=None, target=self.target, independent_variables=self.features, endogenous=self.kwargs.get("endogenous"), instruments=self.kwargs.get("instruments"))
        if kind == "panel_iv":
            return cls(params_path=None, target=self.target, independent_variables=self.features, endogenous=self.kwargs.get("endogenous"), instruments=self.kwargs.get("instruments"), entity_column=self.entity_column)
        if kind == "unsupervised":
            return cls(params_path=None, features=self.features)
        if kind == "svar_family":
            return cls(params_path=None, target_variables=[self.target] + (self.features or []))
        if kind == "lp_iv":
            return cls(params_path=None, target_variable=self.target, shock_variable=self.features[0] if self.features else None)
        if kind == "lp":
            shock_var = self.kwargs.get("shock_variable") or (self.features[0] if self.features else None)
            controls = [f for f in (self.features or []) if f != shock_var]
            return cls(params_path=None, target=self.target, shock_var=shock_var, controls=controls, horizon=self.kwargs.get("horizon", 8), date_column=self.kwargs.get("date_column"))
        raise ValueError(f"Unhandled registry kind '{kind}' for {cls.__name__}")

    def fit_transform(self, data: Union[str, pd.DataFrame], fit_model: bool = True) -> pd.DataFrame:
        self.logger.info("Starting pipeline fit_transform")
        if self.params_path:
            self._initialize_from_params()
        else:
            self._initialize_from_args()

        if isinstance(data, str):
            df = pd.read_csv(data) if data.endswith(".csv") else pd.read_parquet(data)
        else:
            df = data.copy()

        if self.feature_engineer:
            self.transformed_data = self.feature_engineer.fit_transform(df)
        else:
            raise ValueError("No feature engineering component initialized")

        if fit_model and self.model and self.target:
            self.model.fit(self.transformed_data)
            self.model_results = self.model.get_model_metadata()

        return self.transformed_data

    def transform(self, data: Union[str, pd.DataFrame]) -> Optional[pd.DataFrame]:
        if self.params_path:
            self._initialize_from_params()
        else:
            self._initialize_from_args()

        if isinstance(data, str):
            df = pd.read_csv(data) if data.endswith(".csv") else pd.read_parquet(data)
        else:
            df = data.copy()

        if self.feature_engineer:
            self.transformed_data = self.feature_engineer.transform(df)
        return self.transformed_data

    def predict(self, data: Optional[Union[str, pd.DataFrame]] = None) -> Any:
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
            raise NotImplementedError(f"{type(self.model).__name__} does not implement predict(); no model in stats_transformer currently does.")
        return self.model.predict(data_to_predict)

    def save_results(self, output_dir: str = "reports") -> Dict[str, str]:
        import json
        from datetime import datetime
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
        import json
        if self.model_results is None:
            return
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(self.model_results, f, indent=4)

    def create_visualizations(self, output_dir: str = "reports/visualizations", visualization_types: Optional[List[str]] = None, display_only: bool = False) -> Dict[str, Any]:
        if not self.model_results and not self.transformed_data:
            return {}
        
        viz_config = {}
        if self.params_path and os.path.exists(self.params_path):
            viz_config = self._get_config().get("visualization", {})
        
        results = {}
        if self.model_results:
            reg_viz = RegressionVisualizer(params_path=self.params_path, output_dir=output_dir, file_format=viz_config.get("file_format", "png"), dpi=viz_config.get("dpi", 300), style=viz_config.get("style", "default"))
            viz_types = visualization_types or viz_config.get("regression", {}).get("plots", ["coefficient_plot", "model_summary", "residuals"])
            for viz_type in viz_types:
                try:
                    if viz_type == "coefficient_plot":
                        results["coefficient_plot"] = reg_viz.create_coefficient_plot_from_json(self.model_results, subdir="regression", display_only=display_only)
                    elif viz_type == "model_summary":
                        results["model_summary"] = reg_viz.create_model_summary_plot(self.model_results, subdir="regression", display_only=display_only)
                except Exception as e:
                    self.logger.error(f"Error creating {viz_type} visualization: {e}")
        
        if self.transformed_data is not None:
            data_viz = DataVisualizer(params_path=self.params_path, output_dir=output_dir, file_format=viz_config.get("file_format", "png"), dpi=viz_config.get("dpi", 300), style=viz_config.get("style", "default"))
            try:
                if self.entity_column and self.date_column:
                    results["time_series"] = data_viz.create_time_series_plot(self.transformed_data, entity_column=self.entity_column, date_column=self.date_column, display_only=display_only)
            except Exception as e:
                self.logger.error(f"Error creating data visualization: {e}")
        return results

    def run(self, stage: Optional[str] = None) -> Any:
        if self.params_path:
            self._initialize_from_params()
            assert self.feature_engineer is not None  # set unconditionally by _initialize_from_params
            params = self._get_config().to_dict()

            data_config = params.get("data", {})
            raw_data_file = data_config.get("raw_data_file")
            merged_output_path = data_config.get("merge", {}).get("output_path", "data/pipeline/resampled_merged.parquet")
            
            # Prefer merged output if it exists or if we are in a later stage
            effective_data_path = merged_output_path if os.path.exists(merged_output_path) else raw_data_file
            
            viz_output_dir = params.get("visualization", {}).get("output_dir", "reports/visualizations")
            processed_path = data_config.get("featurization", {}).get("output_path")
            summary_path = params.get("model", {}).get("summary_output_path")
        else:
            raise ValueError("Pipeline.run() requires a params_path")

        valid_stages = {"resample", "features", "eda", "regression", "visualization"}
        if stage is not None and stage not in valid_stages:
            raise ValueError(f"Unknown stage '{stage}'. Valid values: {sorted(valid_stages)}, or None for a full run")

        if stage == "resample":
            data_config = params.get("data", {})
            datasets_config = data_config.get("datasets", [])
            resampled = []
            for ds_cfg in datasets_config:
                self.logger.info(f"Resampling dataset: {ds_cfg.get('name')}")
                df = self.feature_engineer.load_data(ds_cfg.get("path"))
                resampled_df = self.feature_engineer.resample_dataset(df, ds_cfg)
                resampled.append(resampled_df)
            dm = DataMerger(params_path=self.params_path)
            merge_config = data_config.get("merge", {})
            merge_on = merge_config.get("on", ["country", "date"])
            merge_how = merge_config.get("how", "outer")
            output_path = merge_config.get("output_path", "data/pipeline/resampled_merged.parquet")
            merged_df = resampled[0]
            for i, next_df in enumerate(resampled[1:], start=1):
                self.logger.info(f"Merging dataset {i + 1} of {len(resampled)}")
                merged_df = dm.merge(merged_df, next_df, on=merge_on, how=merge_how)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            merged_df.to_parquet(output_path)
            self.logger.info(f"Merged data saved to {output_path} with shape {merged_df.shape}")
            return merged_df

        if stage == "features":
            if not effective_data_path:
                raise ValueError("No data path found for features stage")
            transformed_data = self.fit_transform(effective_data_path, fit_model=False)
            if processed_path:
                transformed_data.to_csv(processed_path, index=False)
            return transformed_data
            
        elif stage == "eda":
            if not effective_data_path:
                raise ValueError("No data path found for eda stage")
            eda_viz = EDAVisualizer(params_path=self.params_path, output_dir=viz_output_dir)
            return eda_viz.run(data_path=effective_data_path)
        
        elif stage == "regression":
            if processed_path and os.path.exists(processed_path):
                assert self.model is not None  # set unconditionally by _initialize_from_params
                transformed_data = pd.read_csv(processed_path)
                self.transformed_data = transformed_data
                self.model.fit(transformed_data)
                self.model_results = self.model.get_model_metadata()
            else:
                transformed_data = self.fit_transform(effective_data_path)
            
            self.save_results(output_dir="reports")
            if summary_path:
                self.save_model_summary(summary_path)
            return self.model_results
            
        elif stage == "visualization":
            if (self.transformed_data is None) and processed_path and os.path.exists(processed_path):
                self.transformed_data = pd.read_csv(processed_path)
            if (self.model_results is None) and summary_path and os.path.exists(summary_path):
                import json
                with open(summary_path, "r") as f:
                    self.model_results = json.load(f)
            return self.create_visualizations(output_dir=viz_output_dir)
        
        else:
            # Full run (stage is None)
            data_config = params.get("data", {})
            datasets_config = data_config.get("datasets", [])
            resampled = []
            for ds_cfg in datasets_config:
                self.logger.info(f"Resampling dataset: {ds_cfg.get('name')}")
                df = self.feature_engineer.load_data(ds_cfg.get("path"))
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
            merged_df.to_parquet(merged_output_path)
            self.logger.info(f"Merged data saved to {merged_output_path}")
            transformed_data = self.fit_transform(merged_output_path)
            if processed_path:
                transformed_data.to_csv(processed_path, index=False)
            self.create_visualizations(output_dir=viz_output_dir)
            self.save_results(output_dir="reports")
            if summary_path:
                self.save_model_summary(summary_path)
            return transformed_data

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="params.yaml")
    parser.add_argument("--stage", type=str)
    args = parser.parse_args()
    pipeline = Pipeline(params_path=args.config)
    pipeline.run(stage=args.stage)
