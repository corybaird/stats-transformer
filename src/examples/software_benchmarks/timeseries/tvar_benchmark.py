import numpy as np
import pandas as pd
from stats_transformer.models.timeseries.nonlinear.tvar import TVARModel
from src.examples.software_benchmarks.modules.base_comparator import BaseBenchmark
from src.examples.software_benchmarks.modules.stata_engine import StataEngine

DATA_PATH = "data/examples/timeseries/macrodata.csv"


class TVARBenchmark(BaseBenchmark):

    def run_stats_transformer(self, df):
        model = TVARModel(target_variables=["realgdp"], threshold_variable="realcons", lags=1, delay=1, trim=0.15)
        model.fit(df)
        return float(model.gamma)

    def run_stata(self, df, stata_engine):
        if not stata_engine.is_ready:
            return None
        stata_engine.push_dataframe(df)
        stata_engine.run_command("capture drop t_time")
        stata_engine.run_command("gen t_time = _n")
        stata_engine.run_command("tsset t_time")
        stata_engine.run_command("capture drop L_realgdp L_realcons")
        stata_engine.run_command("gen L_realgdp = L.realgdp")
        stata_engine.run_command("gen L_realcons = L.realcons")
        stata_engine.run_command("threshold realgdp, regionvars(L_realgdp) threshvar(L_realcons) nthresholds(1) trim(30)")
        mat = stata_engine.run_command("matrix list e(thresholds)", "e(thresholds)")
        if mat is None:
            return None
        thresh_col = 1 if mat.shape[1] > 1 else 0
        return float(mat[0, thresh_col])

    def run(self):
        df = pd.read_csv(DATA_PATH)
        py_gamma = self.run_stats_transformer(df)

        results = {}
        stata_engine = StataEngine()
        stata_gamma = self.run_stata(df, stata_engine)
        if stata_gamma is not None:
            results["stata_diff"] = float(abs(py_gamma - stata_gamma))
            print(f"TVARBenchmark Stata gamma diff: {results['stata_diff']:.12g}")

        return results


if __name__ == "__main__":
    TVARBenchmark().run()
