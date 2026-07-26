from pathlib import Path

from stats_transformer.visualization.tables.table_generator import TableGenerator


class TimeSeriesTableBuilder:

    def __init__(self, output_dir="reports/tables/timeseries"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.table_generator = TableGenerator(output_dir=self.output_dir)

    def export(self, report_data, formats=None):
        selected_formats = formats or ["csv", "latex", "excel"]
        exported = {}
        for name, table in report_data.tables():
            exported[name] = self._export_table(table, name, selected_formats)
        return exported

    def _export_table(self, table, name, formats):
        paths = {}
        if "csv" in formats:
            csv_path = self.output_dir / f"{name}.csv"
            table.to_csv(csv_path, index=False)
            paths["csv"] = csv_path
        if "latex" in formats:
            paths["latex"] = self.table_generator.export_latex(table, name)
        if "excel" in formats:
            paths["excel"] = self.table_generator.export_excel(table, name)
        return paths
