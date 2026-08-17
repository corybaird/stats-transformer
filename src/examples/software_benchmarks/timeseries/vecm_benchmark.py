import numpy as np
import pandas as pd
from stats_transformer.models.timeseries.reduced_form.vecm import VECMModel
from src.examples.software_benchmarks.modules.base_comparator import BaseBenchmark
from src.examples.software_benchmarks.modules.stata_engine import StataEngine

DATA_PATH = "data/examples/timeseries/macrodata.csv"
TARGET_VARS = ["realgdp", "realcons", "realinv"]


class VECMBenchmark(BaseBenchmark):

    def run_stats_transformer(self, df):
        model = VECMModel(target_variables=TARGET_VARS, k_ar_diff=1, deterministic="ci")
        model.fit(df)
        if hasattr(model.model, "det_coef_coint") and model.model.det_coef_coint is not None and len(model.model.det_coef_coint) > 0:
            full_beta = np.vstack([model.model.beta, model.model.det_coef_coint])
        else:
            full_beta = model.model.beta
        beta_norm = np.asarray(full_beta / full_beta[0, 0]).flatten()
        return beta_norm

    def run_stata(self, df, stata_engine):
        if not stata_engine.is_ready:
            return None
        stata_engine.push_dataframe(df)
        stata_engine.run_command("capture drop t_time")
        stata_engine.run_command("gen t_time = _n")
        stata_engine.run_command("tsset t_time")
        stata_engine.run_command(f"vec {' '.join(TARGET_VARS)}, lags(2) rank(1)")
        stata_beta = stata_engine.run_command("matrix list e(beta)", "e(beta)")
        if stata_beta is None:
            return None
        stata_beta_norm = np.asarray(stata_beta).flatten() / stata_beta[0, 0]
        return stata_beta_norm

    def run(self):
        df = pd.read_csv(DATA_PATH)
        py_beta = self.run_stats_transformer(df)

        results = {}
        stata_engine = StataEngine()
        stata_beta = self.run_stata(df, stata_engine)
        if stata_beta is not None:
            results["stata_diff"] = float(np.max(np.abs(py_beta - stata_beta)))
            print(f"VECMBenchmark Stata max diff: {results['stata_diff']:.12g}")

        return results


if __name__ == "__main__":
    VECMBenchmark().run()
