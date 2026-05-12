"""
Default color palettes and mappings for stats-transformer visualizations.
"""

SIGNIFICANCE_COLORS = {
    "significant": "#2196F3",
    "not_significant": "#90CAF9",
}

# Standard categorical palettes
PALETTE_COLORBLIND = [
    '#377eb8', '#ff7f00', '#4daf4a', '#f781bf', '#a65628',
    '#984ea3', '#999999', '#e41a1c', '#dede00'
]

PALETTE_CATEGORICAL = [
    '#2196F3', '#FF7043', '#66BB6A', '#AB47BC', '#FFC107', '#00ACC1'
]

PALETTE_DIVERGING = [
    '#E53935', '#FB8C00', '#43A047', '#3949AB', '#8E24AA', '#00ACC1'
]

def get_color_palette(palette_type="default", n_colors=10):
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt
    import numpy as np
    
    if palette_type == "colorblind":
        base_colors = PALETTE_COLORBLIND
    elif palette_type == "sequential":
        cmap = plt.cm.viridis
        base_colors = [cmap(i) for i in np.linspace(0, 1, 9)]
    elif palette_type == "diverging":
        base_colors = PALETTE_DIVERGING
    else:
        base_colors = list(mcolors.TABLEAU_COLORS.values())

    colors = []
    for i in range(n_colors):
        colors.append(base_colors[i % len(base_colors)])

    return colors
