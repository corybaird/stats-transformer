import numpy as np

def format_metrics(metrics, digits=4):
    """Print a metrics dict as `key: value` lines with floats rounded to `digits`."""
    for key, value in metrics.items():
        if isinstance(value, dict):
            print(f"{key}:")
            format_metrics(value, digits)
        elif isinstance(value, np.ndarray):
            print(f"{key}: {np.array2string(value, precision=digits, floatmode='fixed')}")
        elif isinstance(value, (list, tuple)):
            print(f"{key}: {[round(float(v), digits) if isinstance(v, (float, np.floating)) else v for v in value]}")
        elif isinstance(value, (float, np.floating)):
            print(f"{key}: {round(float(value), digits)}")
        else:
            print(f"{key}: {value}")