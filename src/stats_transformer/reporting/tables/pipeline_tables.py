"""LaTeX table generation for the pipeline's reporting stage."""

import logging
import os
from typing import TYPE_CHECKING, List, Optional

import pandas as pd

from stats_transformer.reporting.tables.descriptive_stats import DescriptiveStatsTable
from stats_transformer.reporting.tables.regression_table import RegressionTable
from stats_transformer.reporting.tables.table_generator import TableGenerator

if TYPE_CHECKING:
    from stats_transformer.pipeline import Pipeline


class PipelineTables:
    """Generates the pipeline's descriptive-stats and regression tables."""

    def __init__(self, pipeline: "Pipeline") -> None:
        self._p = pipeline
        self.logger = logging.getLogger(__name__)

    def generate_tables(self, table_names: Optional[List[str]], run_dir: str) -> None:
        """Write the selected LaTeX tables into <run_dir>/04_reporting/tables and register them."""
        if table_names is None:
            # Key absent -> generate both (cheap, broadly useful).
            table_names = ["descriptive_stats", "regression_table"]
        tables_dir = os.path.join(run_dir, "04_reporting", "tables")
        for name in table_names:
            if name == "descriptive_stats":
                table_path = self.descriptive_stats_table(tables_dir)
            elif name == "regression_table":
                table_path = self.regression_table(tables_dir)
            else:
                self.logger.warning(f"Unknown reporting table '{name}'; expected 'descriptive_stats' or 'regression_table'.")
                continue
            if table_path:
                self._p._run_ctx.files.append(str(table_path))

    def model_data(self) -> Optional[pd.DataFrame]:
        """The exact data the model was fitted on: df_clean restricted to the
        columns the model uses (entity/time stay in the index where the model
        indexed them there), rows with NaNs dropped. Shared by the
        model_data.parquet artefact (run_model) and the descriptive-stats table"""
        if self._p.model is None:
            return None
        df_clean = getattr(self._p.model, "df_clean", None)
        if df_clean is None:
            return None
        try:
            cols = [c for c in self._p.model._get_required_columns() if c in df_clean.columns]
        except Exception:
            cols = []
        return (df_clean[cols].dropna() if cols else df_clean.dropna())

    def descriptive_stats_table(self, tables_dir: str) -> Optional[str]:
        """Descriptive stats of the data the model was fitted on (numeric columns only,
        dependent variable first, then regressors in model order)."""
        df = self.model_data()
        if df is None and self._p.model is not None and self._p.transformed_data is not None:
            # Reporting-only run (model never fitted): use the transformed data,
            # still restricted to the model's used columns.
            try:
                cols = [c for c in self._p.model._get_required_columns() if c in self._p.transformed_data.columns]
            except Exception:
                cols = []
            if cols:
                df = self._p.transformed_data[cols]
        if df is None:
            df = self._p.transformed_data
        if df is None:
            self.logger.info("No fitted data available; skipping descriptive_stats table.")
            return None
        if self._p.target and self._p.target in df.columns:
            # Most tables list the dependent variable first; _get_required_columns
            # puts it last (independent_variables first).
            df = df[[self._p.target] + [c for c in df.columns if c != self._p.target]]
        df = df.select_dtypes(include="number")
        table = DescriptiveStatsTable(table_generator=TableGenerator(output_dir=tables_dir))
        path = table.export_dataframe(df, "descriptive_stats")
        if path is None:
            self.logger.info("No numeric columns available; skipping descriptive_stats table.")
        return path

    def regression_table(self, tables_dir: str) -> Optional[str]:
        """Coefficient table from the fitted model's metadata (regression-family models only)."""
        if self._p.model_results is None:
            self.logger.info("No model results available; skipping regression_table table.")
            return None
        table = RegressionTable(table_generator=TableGenerator(output_dir=tables_dir))
        path = table.export(self._p.model_results, "regression_table")
        if path is None:
            self.logger.info("Model is not regression-family; skipping regression_table table.")
        return path