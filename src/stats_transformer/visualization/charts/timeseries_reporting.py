import matplotlib.pyplot as plt
import numpy as np

from stats_transformer.visualization.defaults.colors import get_color_palette
from stats_transformer.visualization.defaults.labels import get_readable_label
from stats_transformer.visualization.formatters.style import apply_style


def _create_panel_grid(panel_count, ncols, panel_width=5.5, panel_height=4.0):
    columns = min(ncols, panel_count)
    rows = (panel_count + columns - 1) // columns
    figure, axes = plt.subplots(rows, columns, figsize=(panel_width * columns, panel_height * rows), squeeze=False)
    return figure, axes.flatten()


def _hide_unused_axes(axes, used):
    for axis in axes[used:]:
        axis.set_visible(False)


def _label(value, labels):
    return labels.get(value, get_readable_label(value)) if labels else get_readable_label(value)


class ImpulseResponseChart:

    def __init__(self, style_path="timeseries"):
        self.style_path = style_path

    def plot(self, data, shock, labels=None, title=None, ylabel="Response", ncols=2, color=None):
        apply_style(self.style_path)
        selected = data[data["shock"] == shock].copy()
        responses = selected["response"].drop_duplicates().tolist()
        if not responses:
            raise ValueError(f"No impulse responses found for shock: {shock}")
        figure, axes = _create_panel_grid(len(responses), ncols)
        plot_color = color or get_color_palette("default", 1)[0]
        for index, response in enumerate(responses):
            axis = axes[index]
            response_data = selected[selected["response"] == response].sort_values("horizon")
            axis.plot(response_data["horizon"], response_data["estimate"], color=plot_color, linewidth=2)
            if {"lower", "upper"}.issubset(response_data.columns):
                axis.fill_between(response_data["horizon"], response_data["lower"], response_data["upper"], color=plot_color, alpha=0.2)
            axis.axhline(0, color="black", linewidth=0.8, linestyle="--")
            axis.set_title(_label(response, labels))
            axis.set_xlabel("Horizon")
            axis.set_ylabel(ylabel)
        _hide_unused_axes(axes, len(responses))
        figure.suptitle(title or f"Responses to {_label(shock, labels)} shock", fontsize=15)
        figure.tight_layout()
        return figure, axes


class VarianceDecompositionChart:

    def __init__(self, style_path="timeseries"):
        self.style_path = style_path

    def plot(self, data, labels=None, title="Forecast-error variance decomposition", ncols=2, colors=None):
        apply_style(self.style_path)
        responses = data["response"].drop_duplicates().tolist()
        shocks = data["shock"].drop_duplicates().tolist()
        if not responses:
            raise ValueError("No variance-decomposition results found")
        figure, axes = _create_panel_grid(len(responses), ncols)
        palette = colors or get_color_palette("default", len(shocks))
        for index, response in enumerate(responses):
            axis = axes[index]
            response_data = data[data["response"] == response]
            pivot = response_data.pivot(index="horizon", columns="shock", values="share").reindex(columns=shocks).fillna(0)
            axis.stackplot(pivot.index, *[pivot[shock].to_numpy() for shock in shocks], labels=[_label(shock, labels) for shock in shocks], colors=palette[:len(shocks)], alpha=0.85)
            axis.set_ylim(0, 1)
            axis.set_title(_label(response, labels))
            axis.set_xlabel("Horizon")
            axis.set_ylabel("Share")
        _hide_unused_axes(axes, len(responses))
        axes[0].legend(loc="best", fontsize=9)
        figure.suptitle(title, fontsize=15)
        figure.tight_layout()
        return figure, axes


class HistoricalDecompositionChart:

    def __init__(self, style_path="timeseries"):
        self.style_path = style_path

    def plot(self, data, labels=None, title="Historical decomposition", ncols=2, colors=None):
        apply_style(self.style_path)
        responses = data["response"].drop_duplicates().tolist()
        shocks = data["shock"].drop_duplicates().tolist()
        if not responses:
            raise ValueError("No historical-decomposition results found")
        figure, axes = _create_panel_grid(len(responses), ncols, panel_width=6.5)
        palette = colors or get_color_palette("default", len(shocks))
        for index, response in enumerate(responses):
            axis = axes[index]
            response_data = data[data["response"] == response]
            pivot = response_data.pivot(index="date", columns="shock", values="contribution").reindex(columns=shocks).fillna(0)
            positive_base = np.zeros(len(pivot))
            negative_base = np.zeros(len(pivot))
            for shock_index, shock in enumerate(shocks):
                values = pivot[shock].to_numpy()
                bottoms = np.where(values >= 0, positive_base, negative_base)
                axis.bar(pivot.index, values, bottom=bottoms, color=palette[shock_index], label=_label(shock, labels), width=1.0)
                positive_base += np.where(values >= 0, values, 0)
                negative_base += np.where(values < 0, values, 0)
            reconstruction = response_data.drop_duplicates("date").set_index("date")["reconstructed"].reindex(pivot.index)
            axis.plot(pivot.index, reconstruction, color="black", linewidth=1.2, label="Reconstructed component")
            axis.axhline(0, color="black", linewidth=0.7)
            axis.set_title(_label(response, labels))
            axis.set_xlabel("Date")
            axis.set_ylabel("Contribution")
        _hide_unused_axes(axes, len(responses))
        axes[0].legend(loc="best", fontsize=8)
        figure.suptitle(title, fontsize=15)
        figure.tight_layout()
        return figure, axes
