import numpy as np
import pandas as pd
from stats_transformer.models.unsupervised.unsupervised import PCAModel
from src.examples.software_benchmarks.modules.base_comparator import BaseBenchmark
from src.examples.software_benchmarks.modules.r_engine import REngine
from src.examples.software_benchmarks.modules.stata_engine import StataEngine

DATA_PATH = "data/examples/regression/longley.csv"
FEATURE_VARS = ["GNPDEFL", "GNP", "UNEMP", "ARMED", "POP", "YEAR"]
N_COMPONENTS = 2


class PCABenchmark(BaseBenchmark):

    def run_stats_transformer(self, df):
        model = PCAModel(features=FEATURE_VARS, n_components=N_COMPONENTS)
        model.fit(df)
        return np.asarray(model.get_model_metrics()["explained_variance_ratio"])

    def run_r(self, df, r_engine):
        if not r_engine.is_ready:
            return None
        r_engine.push_dataframe(df)
        cols_str = ", ".join([f'"{col}"' for col in FEATURE_VARS])
        r_sdev = r_engine.run_command(f"pca_res <- prcomp(df[, c({cols_str})], scale.=TRUE)", "pca_res$sdev")
        r_ratios = (r_sdev**2) / np.sum(r_sdev**2)
        return r_ratios[:N_COMPONENTS]

    def run_stata(self, df, stata_engine):
        if not stata_engine.is_ready:
            return None
        stata_engine.push_dataframe(df)
        stata_ev = stata_engine.run_command(f"pca {' '.join(FEATURE_VARS)}", "e(Ev)")[0]
        stata_ratios = stata_ev / np.sum(stata_ev)
        return stata_ratios[:N_COMPONENTS]

    def run(self):
        df = pd.read_csv(DATA_PATH)
        py_ratios = self.run_stats_transformer(df)
        
        results = {}
        
        r_engine = REngine()
        r_ratios = self.run_r(df, r_engine)
        if r_ratios is not None:
            results["r_diff"] = self.compare_results(py_ratios, r_ratios)
            print(f"PCABenchmark R max diff: {results['r_diff']:.12g}")

        stata_engine = StataEngine()
        stata_ratios = self.run_stata(df, stata_engine)
        if stata_ratios is not None:
            results["stata_diff"] = self.compare_results(py_ratios, stata_ratios)
            print(f"PCABenchmark Stata max diff: {results['stata_diff']:.12g}")

        return results


if __name__ == "__main__":
    PCABenchmark().run()
