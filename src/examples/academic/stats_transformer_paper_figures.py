from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from stats_transformer.data import load_example
from stats_transformer.models.regression.panel import PanelRegressionModel
from stats_transformer.models.regression.robust_ols import RobustOLSModel
from stats_transformer.models.timeseries.identification.blanchard_quah import BlanchardQuahModel
from stats_transformer.models.timeseries.reduced_form.var import VARModel
from stats_transformer.reporting.charts.bar import CoefficientBarChart, CoefficientForestPlot
from stats_transformer.reporting.charts.time_series import IRFPlot

REPO = Path(__file__).resolve().parents[3]


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
        data = pd.read_csv(REPO / "data/examples/regression/mincer_wage.csv")
        model = RobustOLSModel(params_path=str(REPO / "references/configs/regression/mincer_wage.yaml"))
        model.fit(data)
        variables = ["educ", "exper", "expersq", "tenure"]
        coefficients = model.model.params.loc[variables].tolist()
        standard_errors = model.model.bse.loc[variables].tolist()
        p_values = model.model.pvalues.loc[variables].tolist()
        chart = CoefficientForestPlot()
        fig, _ = chart.plot(variables, coefficients, standard_errors, p_values, ylabel="Coefficient", title="Mincer wage equation", footer="HC3 robust standard errors; Wooldridge wage1 data")
        self._save_figure(fig, "mincer_coefficients.pdf")

    def _make_grunfeld_coefficients(self):
        """Grunfeld two-way FE panel from the committed config, fitted the way
        the pipeline does (featurization here is only the column mapping year ->
        date, firm -> country). Numbers match the applications-table run and its
        model_summary.json."""
        config_path = REPO / "references/configs/regression/grunfeld_panel.yaml"
        spec = yaml.safe_load(config_path.read_text())
        data = pd.read_csv(REPO / spec["data"]["raw_data_file"])
        data = data.rename(columns={old: new for old, new in spec["data"]["featurization"].get("column_mapping", {}).items() if old in data.columns})
        model = PanelRegressionModel(params_path=str(config_path))
        model.fit(data)
        variables = spec["model"]["independent_variables"]
        coefficients = model.model.params.loc[variables].tolist()
        standard_errors = model.model.std_errors.loc[variables].tolist()
        p_values = model.model.pvalues.loc[variables].tolist()
        chart = CoefficientForestPlot()
        footer = f"Firm and year fixed effects; {int(model.model.nobs)} observations; unadjusted standard errors"
        fig, ax = chart.plot(variables, coefficients, standard_errors, p_values, ylabel="Coefficient", title="Grunfeld investment equation")
        # Footer below the axes: ForestPlot's in-axes footer collides with the
        # bottom (Capital) whisker.
        fig.tight_layout()
        fig.text(0.99, 0.005, footer, ha="right", va="bottom", fontsize=9, color="grey")
        self._save_figure(fig, "grunfeld_coefficients.pdf")

    def _make_macro_var_irf(self):
        """Orthogonalized (Cholesky, GDP first) responses of the three growth
        rates to a one-s.d. GDP-growth shock from the committed macro_var
        config (the data file itself holds percentage log-growth rates), with
        95% asymptotic bands from the VAR's orthogonalized stderr."""
        config_path = REPO / "references/configs/timeseries/reduced_form/macro_var.yaml"
        spec = yaml.safe_load(config_path.read_text())["model"]
        model = VARModel(target_variables=spec["target_variables"], maxlags=spec["maxlags"])
        model.fit(pd.read_csv(REPO / "data/examples/timeseries/macrodata.csv"))
        irf = model.model.irf(12)  # horizons 0..12 quarters
        response = irf.orth_irfs[:, :, 0]  # shock 0 = GDP growth (listed first)
        stderr = irf.stderr(orth=True)[:, :, 0]
        labels = {"realgdp": "Real GDP", "realcons": "Real consumption", "realinv": "Real investment"}
        rows = []
        for variable_index, variable in enumerate(spec["target_variables"]):
            for horizon in range(irf.orth_irfs.shape[0]):
                centre = float(response[horizon, variable_index])
                rows.append((horizon, variable, centre, centre - 1.96 * float(stderr[horizon, variable_index]), centre + 1.96 * float(stderr[horizon, variable_index])))
        plot_data = pd.DataFrame(rows, columns=["horizon", "facet", "coef", "lower", "upper"])
        chart = IRFPlot()
        fig, axes = chart.plot(plot_data, "horizon", "coef", "lower", "upper", "facet", labels=labels, ylabel="Response", ncols=3)
        for ax in axes[:3]:
            ax.set_xticks(range(0, 13, 3))
        # IRFPlot's suptitle sits at y=1.02 and gets clipped by _save_figure;
        # draw it inside the canvas instead.
        fig.suptitle("Responses to a GDP growth shock", size=18, y=1.0)
        fig.tight_layout(rect=(0, 0, 1, 0.955))
        self._save_figure(fig, "macro_var_irf.pdf")

    def _make_bq_irf(self):
        """Blanchard--Quah structural IRFs on the committed bq1989 config:
        responses of output level (cumulated y) and unemployment to the supply
        and demand shocks. y is quarterly GNP growth, so its level response is
        the cumulative sum of the growth response."""
        with open(REPO / "references/configs/timeseries/identification/blanchard_quah_1989.yaml") as stream:
            spec = yaml.safe_load(stream)["model"]
        model = BlanchardQuahModel(target_variables=spec["target_variables"], maxlags=spec["maxlags"])
        model.fit(pd.read_csv(REPO / "data/examples/timeseries/bq1989.csv"))
        irf = model.var_result.irf(40)  # ~40 quarters, horizon 0..40
        structural = irf.irfs @ model.B_0  # structural: variable i to shock j
        # BQ identification leaves each shock's sign free (the parity test pins
        # the MATLAB B_0, so the model stays untouched). For the figure only,
        # present both shocks as expansionary as in the paper: flip a shock's
        # column when its impact effect on output (y, variable 0, horizon 0)
        # is negative.
        structural = structural * np.where(structural[0, 0, :] >= 0, 1.0, -1.0)[None, None, :]
        variables = ["Output level", "Unemployment"]
        shocks = ["Supply shock", "Demand shock"]
        rows = []
        for variable_index in (0, 1):  # output level, unemployment
            variable = variables[variable_index]
            for shock_index, shock in enumerate(shocks):
                for horizon, response in enumerate(
                    np.cumsum(structural[:, variable_index, shock_index]) if variable_index == 0 else structural[:, variable_index, shock_index]
                ):
                    rows.append((horizon, (variable, shock), float(response), float(response), float(response)))
        plot_data = pd.DataFrame(rows, columns=["horizon", "facet", "coef", "lower", "upper"])
        labels = {(variable, shock): f"{variable}, {shock.lower()}" for variable in variables for shock in shocks}
        from stats_transformer.reporting.defaults.colors import PALETTE_CATEGORICAL

        chart = IRFPlot()
        # One colour per shock: group order alternates variables within a shock
        # column (output, unemployment, output, unemployment).
        colors = [PALETTE_CATEGORICAL[0], PALETTE_CATEGORICAL[1], PALETTE_CATEGORICAL[0], PALETTE_CATEGORICAL[1]]
        fig, axes = chart.plot(plot_data, "horizon", "coef", "lower", "upper", "facet", labels=labels, ylabel="Response", colors=colors)
        for ax in axes:
            ax.set_xticks(range(0, irf.irfs.shape[0] + 1, 8))  # 41 horizons: thin the ticks
        # IRFPlot's suptitle sits at y=1.02 and gets clipped by _save_figure;
        # draw it inside the canvas instead.
        fig.suptitle("Blanchard–Quah structural impulse responses", size=18, y=1.0)
        fig.tight_layout(rect=(0, 0, 1, 0.955))
        self._save_figure(fig, "bq_irf.pdf")

    def run(self):
        np.random.seed(42)
        self._make_architecture()
        self._make_mincer_coefficients()
        self._make_grunfeld_coefficients()
        self._make_macro_var_irf()
        self._make_bq_irf()


if __name__ == "__main__":
    PaperFigureBuilder().run()
