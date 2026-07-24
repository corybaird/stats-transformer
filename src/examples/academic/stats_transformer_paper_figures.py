from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from stats_transformer.data import load_example
from stats_transformer.models.regression.panel import PanelRegressionModel
from stats_transformer.models.regression.robust_ols import RobustOLSModel
from stats_transformer.models.timeseries.var import VARModel
from stats_transformer.visualization.charts.bar import CoefficientBarChart
from stats_transformer.visualization.charts.time_series import IRFPlot


class PaperFigureBuilder:

    def __init__(self):
        self.output_dir = Path("reports/overleaf/figures")

    def _save_figure(self, figure, filename):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        figure.set_layout_engine("none")
        figure.savefig(self.output_dir / filename, dpi=300)
        plt.close(figure)

    def _make_architecture(self):
        stages = [
            ("YAML", "data and\nmodel choices"),
            ("Resample", "frequency\nalignment"),
            ("Features", "lags, growth,\nrolling statistics"),
            ("Model", "OLS, panel,\nVAR, IV, PCA"),
            ("Visualize", "figures, tables,\nmodel metadata"),
        ]
        fig, ax = plt.subplots(figsize=(11, 2.5))
        ax.set_xlim(0, len(stages) * 2.1)
        ax.set_ylim(0, 2.3)
        ax.axis("off")
        for index, stage in enumerate(stages):
            x_position = index * 2.1 + 0.25
            box = plt.Rectangle((x_position, 0.55), 1.55, 1.15, facecolor="#edf3f7", edgecolor="#2f4858", linewidth=1.2)
            ax.add_patch(box)
            ax.text(x_position + 0.775, 1.35, stage[0], ha="center", va="center", fontsize=11, fontweight="bold", color="#102a43")
            ax.text(x_position + 0.775, 0.88, stage[1], ha="center", va="center", fontsize=8.5, color="#334e68")
            if index < len(stages) - 1:
                ax.annotate("", xy=(x_position + 1.95, 1.12), xytext=(x_position + 1.6, 1.12), arrowprops={"arrowstyle": "->", "color": "#5b6770", "linewidth": 1.2})
        fig.suptitle("A configuration-driven research workflow", fontsize=14, y=0.98)
        self._save_figure(fig, "pipeline_architecture.pdf")

    def _panel_data(self):
        data = load_example().sort_values(["country", "date"]).copy()
        data["gdp_growth"] = data.groupby("country")["gdp"].diff()
        data["gdp_growth_lag"] = data.groupby("country")["gdp_growth"].shift(1)
        return data.dropna(subset=["gdp_growth", "inflation", "gdp_growth_lag", "gdp"])

    def _make_panel_coefficients(self):
        data = self._panel_data()
        model = PanelRegressionModel(target="gdp_growth", independent_variables=["gdp_growth_lag", "gdp"], entity_column="country", time_column="date", entity_effects=True, time_effects=True)
        model.fit(data)
        coefficients = model.model.params.loc[["gdp_growth_lag", "gdp"]].tolist()
        standard_errors = model.model.std_errors.loc[["gdp_growth_lag", "gdp"]].tolist()
        p_values = model.model.pvalues.loc[["gdp_growth_lag", "gdp"]].tolist()
        chart = CoefficientBarChart()
        fig, _ = chart.plot(["gdp_growth_lag", "gdp"], coefficients, standard_errors, p_values, ylabel="Coefficient", title="Panel fixed-effects demonstration", footer="207 countries; entity and year effects")
        self._save_figure(fig, "panel_coefficients.pdf")

    def _make_mincer_coefficients(self):
        data = pd.read_csv("data/examples/regression/mincer_wage.csv")
        model = RobustOLSModel(params_path="src/examples/configs/mincer_wage.yaml")
        model.fit(data)
        variables = ["educ", "exper", "expersq", "tenure"]
        coefficients = model.model.params.loc[variables].tolist()
        standard_errors = model.model.bse.loc[variables].tolist()
        p_values = model.model.pvalues.loc[variables].tolist()
        chart = CoefficientBarChart()
        fig, _ = chart.plot(variables, coefficients, standard_errors, p_values, ylabel="Coefficient", title="Mincer wage equation", footer="HC3 robust standard errors; Wooldridge wage1 data")
        self._save_figure(fig, "mincer_coefficients.pdf")

    def _make_macro_irf(self):
        data = self._panel_data()
        annual = data.groupby("date", as_index=False)[["gdp_growth", "inflation"]].mean().dropna()
        model = VARModel(target_variables=["gdp_growth", "inflation"], date_column="date", maxlags=2)
        model.fit(annual)
        irf = model.model.irf(8)
        lower, upper = irf.errband_mc(orth=False, repl=200, signif=0.05)
        rows = []
        for response_index, response in enumerate(["gdp_growth", "inflation"]):
            for horizon in range(9):
                rows.append((horizon, response, irf.irfs[horizon, response_index, 1], lower[horizon, response_index, 1], upper[horizon, response_index, 1]))
        plot_data = pd.DataFrame(rows, columns=["horizon", "response", "coef", "lower", "upper"])
        chart = IRFPlot()
        fig, _ = chart.plot(plot_data, "horizon", "coef", "lower", "upper", "response", labels={"gdp_growth": "GDP growth", "inflation": "Inflation"}, title="Macro VAR impulse responses", ylabel="Response to inflation shock")
        fig.suptitle("Macro VAR impulse responses", y=0.97)
        fig.subplots_adjust(top=0.84)
        self._save_figure(fig, "macro_irf.pdf")

    def run(self):
        np.random.seed(42)
        self._make_architecture()
        self._make_mincer_coefficients()
        self._make_macro_irf()


if __name__ == "__main__":
    PaperFigureBuilder().run()
