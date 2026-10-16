"""LaTeX table generation for the pipeline's reporting stage."""

import logging
import os
from typing import Any, Dict, List, Optional

import pandas as pd

from stats_transformer.models.base import ModelBase
from stats_transformer.reporting.tables.descriptive_stats import DescriptiveStatsTable
from stats_transformer.reporting.tables.regression_table import RegressionTable
from stats_transformer.reporting.tables.table_generator import TableGenerator


class PipelineTables:
    """Generates the pipeline's descriptive-stats and regression tables; methods return written paths."""

    def __init__(self, logger: logging.Logger) -> None:
        self.logger = logger

    def generate_tables(
        self,
        model: Optional[ModelBase],
        transformed_data: Optional[pd.DataFrame],
        target: Optional[str],
        model_results: Optional[Dict[str, Any]],
        table_names: Optional[List[str]],
        tables_dir: str,
    ) -> List[str]:
        """Write the selected LaTeX tables into tables_dir and return the paths written."""
        if table_names is None:
            # Key absent -> generate both (cheap, broadly useful).
            table_names = ["descriptive_stats", "regression_table"]
        written: List[str] = []
        for name in table_names:
            if name == "descriptive_stats":
                table_path = self.descriptive_stats_table(model, transformed_data, target, tables_dir)
            elif name == "regression_table":
                table_path = self.regression_table(model_results, tables_dir)
            else:
                self.logger.warning(
                    f"Unknown reporting table '{name}'; expected 'descriptive_stats' or 'regression_table'."
                )
                continue
            if table_path:
                written.append(str(table_path))
        return written

    def model_data(self, model: Optional[ModelBase]) -> Optional[pd.DataFrame]:
        """The exact data the model was fitted on: df_clean restricted to the
        columns the model uses (entity/time stay in the index where the model
        indexed them there), rows with NaNs dropped. Shared by the
        model_data.parquet artefact (run_model) and the descriptive-stats table"""
        if model is None:
            return None
        df_clean = getattr(model, "df_clean", None)
        if df_clean is None:
            return None
        try:
            cols = [c for c in model._get_required_columns() if c in df_clean.columns]
        except Exception:
            cols = []
        return (df_clean[cols].dropna() if cols else df_clean.dropna())

    def descriptive_stats_table(
        self,
        model: Optional[ModelBase],
        transformed_data: Optional[pd.DataFrame],
        target: Optional[str],
        tables_dir: str,
    ) -> Optional[str]:
        """Descriptive stats of the data the model was fitted on (numeric columns only,
        dependent variable first, then regressors in model order)."""
        df = self.model_data(model)
        if df is None and model is not None and transformed_data is not None:
            # Reporting-only run (model never fitted): use the transformed data,
            # still restricted to the model's used columns.
            try:
                cols = [c for c in model._get_required_columns() if c in transformed_data.columns]
            except Exception:
                cols = []
            if cols:
                df = transformed_data[cols]
        if df is None:
            df = transformed_data
        if df is None:
            self.logger.info("No fitted data available; skipping descriptive_stats table.")
            return None
        if target and target in df.columns:
            # Most tables list the dependent variable first; _get_required_columns
            # puts it last (independent_variables first).
            df = df[[target] + [c for c in df.columns if c != target]]
        df = df.select_dtypes(include="number")
        table = DescriptiveStatsTable(table_generator=TableGenerator(output_dir=tables_dir))
        path = table.export_dataframe(df, "descriptive_stats")
        if path is None:
            self.logger.info("No numeric columns available; skipping descriptive_stats table.")
        return path

    def regression_table(
        self,
        model_results: Optional[Dict[str, Any]],
        tables_dir: str,
    ) -> Optional[str]:
        """Coefficient table from the fitted model's metadata (regression-family models only)."""
        if model_results is None:
            self.logger.info("No model results available; skipping regression_table table.")
            return None
        table = RegressionTable(table_generator=TableGenerator(output_dir=tables_dir))
        path = table.export(model_results, "regression_table")
        if path is None:
            self.logger.info("Model is not regression-family; skipping regression_table table.")
        return path