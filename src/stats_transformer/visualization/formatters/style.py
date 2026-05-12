import os
import matplotlib.pyplot as plt
import logging
import importlib.resources

logger = logging.getLogger(__name__)

def resolve_style_path(style_name):
    """
    Resolves a style name or path to a valid matplotlib style path.
    First checks if it's a direct path, then checks the package's defaults/styles/.
    """
    if os.path.exists(style_name):
        return style_name
        
    if not style_name.endswith('.mplstyle'):
        style_name = f"{style_name}.mplstyle"
        
    try:
        # Check in the package's bundled styles
        pkg_path = importlib.resources.files('stats_transformer.visualization.defaults.styles') / style_name
        if pkg_path.is_file():
            return str(pkg_path)
    except Exception as e:
        logger.warning(f"Could not resolve packaged style {style_name}: {e}")
        
    return style_name

def apply_style(style_path="default"):
    """
    Applies a matplotlib style.
    """
    resolved_path = resolve_style_path(style_path)
    try:
        plt.style.use(resolved_path)
    except Exception as e:
        logger.warning(f"Could not apply style {style_path} (resolved to {resolved_path}): {e}. Using default.")
        plt.style.use("default")
        
    # Additional global config overrides if needed
    plt.rc('axes', grid=True)
    plt.rc('grid', linestyle='--', alpha=0.7)
    plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans', 'Helvetica', 'Verdana']})
