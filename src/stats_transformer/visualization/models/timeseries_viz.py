from stats_transformer.visualization.base import BaseVisualizer
from stats_transformer.visualization.charts.timeseries_reporting import HistoricalDecompositionChart, ImpulseResponseChart, VarianceDecompositionChart


class TimeSeriesVisualizer(BaseVisualizer):

    def create_visualization(self, report_data, outputs=None, labels=None, display_only=False):
        selected_outputs = outputs or ["irfs", "fevd", "historical_decomposition"]
        saved = {}
        if "irfs" in selected_outputs and not report_data.irfs.empty:
            saved["irfs"] = self.plot_irfs(report_data.irfs, labels=labels, display_only=display_only)
        if "fevd" in selected_outputs and not report_data.fevd.empty:
            saved["fevd"] = self.plot_fevd(report_data.fevd, labels=labels, display_only=display_only)
        if "historical_decomposition" in selected_outputs and not report_data.historical_decomposition.empty:
            saved["historical_decomposition"] = self.plot_historical_decomposition(report_data.historical_decomposition, labels=labels, display_only=display_only)
        return saved

    def plot_irfs(self, data, labels=None, display_only=False):
        paths = []
        chart = ImpulseResponseChart(style_path=self.viz_params.get("style", "timeseries"))
        for shock in data["shock"].drop_duplicates().tolist():
            figure, _ = chart.plot(data, shock=shock, labels=labels)
            filename = f"irf_{self._safe_name(shock)}"
            paths.append(self.save_figure(figure, filename, subdir="timeseries/irfs", display_only=display_only))
        return paths

    def plot_fevd(self, data, labels=None, display_only=False):
        chart = VarianceDecompositionChart(style_path=self.viz_params.get("style", "timeseries"))
        figure, _ = chart.plot(data, labels=labels)
        return self.save_figure(figure, "forecast_error_variance_decomposition", subdir="timeseries", display_only=display_only)

    def plot_historical_decomposition(self, data, labels=None, display_only=False):
        chart = HistoricalDecompositionChart(style_path=self.viz_params.get("style", "timeseries"))
        figure, _ = chart.plot(data, labels=labels)
        return self.save_figure(figure, "historical_decomposition", subdir="timeseries", display_only=display_only)

    def _safe_name(self, value):
        return str(value).strip().lower().replace(" ", "_").replace("/", "_")
