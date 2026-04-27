import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

def get_color_palette(palette_type="default", n_colors=10):
    if palette_type == "colorblind":
        base_colors = ['#377eb8', '#ff7f00', '#4daf4a', '#f781bf', '#a65628',
                       '#984ea3', '#999999', '#e41a1c', '#dede00']
    elif palette_type == "sequential":
        cmap = plt.cm.viridis
        base_colors = [cmap(i) for i in np.linspace(0, 1, 9)]
    elif palette_type == "diverging":
        base_colors = ['#d73027', '#f46d43', '#fdae61', '#fee090', '#ffffbf',
                       '#e0f3f8', '#abd9e9', '#74add1', '#4575b4']
    else:
        base_colors = list(mcolors.TABLEAU_COLORS.values())

    colors = []
    for i in range(n_colors):
        colors.append(base_colors[i % len(base_colors)])

    return colors

def create_grid_layout(n_plots, n_cols=2):
    n_rows = (n_plots + n_cols - 1) // n_cols
    fig = plt.figure(figsize=(6 * n_cols, 5 * n_rows))
    axes = []
    for i in range(n_plots):
        ax = fig.add_subplot(n_rows, n_cols, i + 1)
        axes.append(ax)
    fig.tight_layout(pad=3.0)
    return fig, axes

def configure_plot_aesthetics(ax, title=None, xlabel=None, ylabel=None, legend=False, grid=True):
    if title:
        ax.set_title(title, fontsize=12, fontweight='bold')
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=10)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=10)
    if legend:
        ax.legend(loc='best')
    if grid:
        ax.grid(True, linestyle='--', alpha=0.7)
