from pathlib import Path

from stats_transformer.reporting.timeseries.adapters import BlanchardQuahResultAdapter, LocalProjectionsIVResultAdapter, LocalProjectionsResultAdapter, VARResultAdapter
from stats_transformer.reporting.timeseries.results import TimeSeriesReportRun
from stats_transformer.visualization.models.timeseries_viz import TimeSeriesVisualizer
from stats_transformer.visualization.tables.timeseries import TimeSeriesTableBuilder


class TimeSeriesReporter:

    def __init__(self, adapter, output_dir="reports", params_path=None, labels=None):
        self.adapter = adapter
        self.output_dir = Path(output_dir)
        self.labels = labels
        self.visualizer = TimeSeriesVisualizer(params_path=params_path, output_dir=self.output_dir / "figures")
        self.table_builder = TimeSeriesTableBuilder(output_dir=self.output_dir / "tables")
        self.report_data = None

    @classmethod
    def from_var(cls, model, horizons=20, output_dir="reports", params_path=None, labels=None):
        return cls(VARResultAdapter(model, horizons=horizons), output_dir=output_dir, params_path=params_path, labels=labels)

    @classmethod
    def from_blanchard_quah(cls, model, horizons=20, output_dir="reports", params_path=None, labels=None):
        return cls(BlanchardQuahResultAdapter(model, horizons=horizons), output_dir=output_dir, params_path=params_path, labels=labels)

    @classmethod
    def from_local_projections(cls, model, output_dir="reports", params_path=None, labels=None):
        return cls(LocalProjectionsResultAdapter(model), output_dir=output_dir, params_path=params_path, labels=labels)

    @classmethod
    def from_local_projections_iv(cls, model, output_dir="reports", params_path=None, labels=None):
        return cls(LocalProjectionsIVResultAdapter(model), output_dir=output_dir, params_path=params_path, labels=labels)

    def run(self, figure_outputs=None, table_formats=None, display_only=False):
        self.report_data = self.adapter.build()
        figures = self.visualizer.create_visualization(self.report_data, outputs=figure_outputs, labels=self.labels, display_only=display_only)
        tables = self.table_builder.export(self.report_data, formats=table_formats)
        return TimeSeriesReportRun(self.report_data, figures=figures, tables=tables)
