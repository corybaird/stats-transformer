import numpy as np
import pandas as pd
from stats_transformer.models.discrete.logit import LogitModel
from src.examples.software_benchmarks.modules.base_comparator import BaseBenchmark
from src.examples.software_benchmarks.modules.r_engine import REngine
from src.examples.software_benchmarks.modules.stata_engine import StataEngine

DATA_PATH = "data/examples/discrete/spector.csv"
TARGET_VAR = "GRADE"
INDEPENDENT_VARS = ["GPA", "TUCE", "PSI"]


class LogitBenchmark(BaseBenchmark):

    def run_stats_transformer(self, df):
        model = LogitModel(target=TARGET_VAR, independent_variables=INDEPENDENT_VARS)
        model.fit(df)
        return model.model.params.values

    def run_r(self, df, r_engine):
        if not r_engine.is_ready:
            return None
        r_engine.push_dataframe(df)
        formula = f"{TARGET_VAR} ~ " + " + ".join(INDEPENDENT_VARS)
        return r_engine.run_command(f"r_model <- glm({formula}, family=binomial, data=df)", "coef(r_model)")

    def run_stata(self, df, stata_engine):
        if not stata_engine.is_ready:
            return None
        stata_engine.push_dataframe(df)
        stata_coef = stata_engine.run_command(f"logit {TARGET_VAR} {' '.join(INDEPENDENT_VARS)}", "e(b)")[0]
        return np.concatenate([stata_coef[-1:], stata_coef[:-1]])

    def run(self):
        df = pd.read_csv(DATA_PATH)
        py_coef = self.run_stats_transformer(df)
        
        results = {}
        
        r_engine = REngine()
        r_coef = self.run_r(df, r_engine)
        if r_coef is not None:
            results["r_diff"] = self.compare_results(py_coef, r_coef)
            print(f"LogitBenchmark R max diff: {results['r_diff']:.12g}")

        stata_engine = StataEngine()
        stata_coef = self.run_stata(df, stata_engine)
        if stata_coef is not None:
            results["stata_diff"] = self.compare_results(py_coef, stata_coef)
            print(f"LogitBenchmark Stata max diff: {results['stata_diff']:.12g}")

        return results


if __name__ == "__main__":
    LogitBenchmark().run()
