import numpy as np
import pandas as pd
from stats_transformer.models.timeseries.reduced_form.var import VARModel
from src.examples.software_benchmarks.modules.base_comparator import BaseBenchmark
from src.examples.software_benchmarks.modules.stata_engine import StataEngine

DATA_PATH = "data/examples/timeseries/macrodata.csv"
TARGET_VARS = ["realgdp", "realcons", "realinv"]
LAGS = 2


class VARBenchmark(BaseBenchmark):

    def run_stats_transformer(self, df):
        model = VARModel(target_variables=TARGET_VARS, maxlags=LAGS)
        model.fit(df)
        params = model.model.params
        k = len(TARGET_VARS)
        py_ordered = []
        for eq_idx in range(k):
            for v_idx in range(k):
                for lag in range(1, LAGS + 1):
                    row_idx = (lag - 1) * k + v_idx + 1
                    py_ordered.append(params.iloc[row_idx, eq_idx])
            py_ordered.append(params.iloc[0, eq_idx])
        return np.array(py_ordered)

    def run_stata(self, df, stata_engine):
        if not stata_engine.is_ready:
            return None
        stata_engine.push_dataframe(df)
        stata_engine.run_command("capture drop t_time")
        stata_engine.run_command("gen t_time = _n")
        stata_engine.run_command("tsset t_time")
        stata_engine.run_command(f"var {' '.join(TARGET_VARS)}, lags(1/{LAGS})")
        stata_b = stata_engine.run_command("matrix list e(b)", "e(b)")
        if stata_b is None:
            return None
        stata_vec = np.asarray(stata_b).flatten()
        return stata_vec

    def run(self):
        df = pd.read_csv(DATA_PATH)
        py_coef = self.run_stats_transformer(df)

        results = {}
        stata_engine = StataEngine()
        stata_coef = self.run_stata(df, stata_engine)
        if stata_coef is not None:
            results["stata_diff"] = self.compare_results(py_coef, stata_coef)
            print(f"VARBenchmark Stata max diff: {results['stata_diff']:.12g}")

        return results


if __name__ == "__main__":
    VARBenchmark().run()
