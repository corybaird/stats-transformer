import matplotlib.pyplot as plt
import pytest
from stats_transformer.visualization.utils.viz_utils import get_color_palette, create_grid_layout, configure_plot_aesthetics


def test_get_color_palette_colorblind():
    colors = get_color_palette(palette_type="colorblind", n_colors=5)
    assert len(colors) == 5
    assert colors[0] == "#377eb8"


def test_get_color_palette_sequential():
    colors = get_color_palette(palette_type="sequential", n_colors=4)
    assert len(colors) == 4


def test_get_color_palette_diverging():
    colors = get_color_palette(palette_type="diverging", n_colors=3)
    assert len(colors) == 3
    assert colors[0] == "#d73027"


def test_get_color_palette_default_falls_back_to_tableau():
    colors = get_color_palette(palette_type="unknown_type", n_colors=3)
    assert len(colors) == 3


def test_get_color_palette_wraps_around_when_n_exceeds_base_colors():
    colors = get_color_palette(palette_type="colorblind", n_colors=20)
    assert len(colors) == 20
    # base palette has 9 colors, so index 9 should repeat index 0
    assert colors[9] == colors[0]


def test_create_grid_layout_produces_correct_axes_count():
    fig, axes = create_grid_layout(n_plots=5, n_cols=2)
    try:
        assert len(axes) == 5
    finally:
        plt.close(fig)


def test_create_grid_layout_single_plot():
    fig, axes = create_grid_layout(n_plots=1, n_cols=2)
    try:
        assert len(axes) == 1
    finally:
        plt.close(fig)


def test_configure_plot_aesthetics_sets_title_and_labels():
    fig, ax = plt.subplots()
    try:
        configure_plot_aesthetics(ax, title="My Title", xlabel="X", ylabel="Y")
        assert ax.get_title() == "My Title"
        assert ax.get_xlabel() == "X"
        assert ax.get_ylabel() == "Y"
    finally:
        plt.close(fig)


def test_configure_plot_aesthetics_noop_fields_when_not_provided():
    fig, ax = plt.subplots()
    try:
        configure_plot_aesthetics(ax)
        assert ax.get_title() == ""
        assert ax.get_xlabel() == ""
    finally:
        plt.close(fig)


def test_configure_plot_aesthetics_adds_legend_when_requested():
    fig, ax = plt.subplots()
    try:
        ax.plot([1, 2], [3, 4], label="series")
        configure_plot_aesthetics(ax, legend=True)
        assert ax.get_legend() is not None
    finally:
        plt.close(fig)


def test_configure_plot_aesthetics_grid_false_does_not_raise():
    fig, ax = plt.subplots()
    try:
        configure_plot_aesthetics(ax, grid=False)
    finally:
        plt.close(fig)
