from src.examples.software_benchmarks.regression.regression_benchmark import RegressionBenchmark
from src.examples.software_benchmarks.discrete.logit_benchmark import LogitBenchmark
from src.examples.software_benchmarks.unsupervised.pca_benchmark import PCABenchmark
from src.examples.software_benchmarks.timeseries.blanchard_quah_benchmark import BlanchardQuahBenchmark


class BenchmarkSuite:

    def __init__(self):
        self.benchmarks = {
            "RegressionModel (OLS)": RegressionBenchmark(),
            "LogitModel": LogitBenchmark(),
            "PCAModel": PCABenchmark(),
            "BlanchardQuahModel (SVAR)": BlanchardQuahBenchmark(),
        }

    def run(self):
        print("==================================================")
        print("  STATS-TRANSFORMER SOFTWARE BENCHMARK SUITE      ")
        print("==================================================")
        
        suite_results = {}
        for name, benchmark in self.benchmarks.items():
            print(f"\n--- Running {name} ---")
            suite_results[name] = benchmark.run()

        print("\n==================================================")
        print("              BENCHMARK SUMMARY RESULTS           ")
        print("==================================================")
        for name, res in suite_results.items():
            diffs = ", ".join([f"{k}: {v:.6g}" for k, v in res.items()]) if res else "No active engine"
            print(f"  {name:<30} -> {diffs}")
        print("==================================================")
        return suite_results


if __name__ == "__main__":
    BenchmarkSuite().run()
