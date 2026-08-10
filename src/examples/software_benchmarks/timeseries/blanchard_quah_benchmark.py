from pathlib import Path
import numpy as np
import pandas as pd
from stats_transformer.models.timeseries.identification.blanchard_quah import BlanchardQuahModel
from src.examples.software_benchmarks.modules.base_comparator import BaseBenchmark
from src.examples.software_benchmarks.modules.matlab_engine import MATLABEngine

DATA_PATH = "data/examples/matlab_examples/BQ1989_Data.xlsx"
MAX_LAGS = 8


class BlanchardQuahBenchmark(BaseBenchmark):

    def _read_data(self, sample_data_path):
        data = pd.read_excel(sample_data_path).iloc[1:].reset_index(drop=True)
        variables = [col for col in data.columns if col not in ["Date", "date", "year", "quarter", "Unnamed: 0"]][:2]
        data[variables] = data[variables].apply(pd.to_numeric, errors="coerce")
        return data[variables].dropna(), variables

    def run_stats_transformer(self, data, variables):
        python_model = BlanchardQuahModel(target_variables=variables, maxlags=MAX_LAGS)
        python_model.fit(data)
        return python_model.B_0

    def run_matlab(self, data, matlab_engine):
        if not matlab_engine.connect():
            return None
        import matlab
        options = matlab_engine.run_command("VARoption", nargout=1)
        options["ident"] = "long"
        options["inference"] = 0.0
        matlab_result = matlab_engine.run_command("VARmodel", matlab.double(data.values.tolist()), float(MAX_LAGS), 1.0, options, nargout=1)
        return np.asarray(matlab_result["B"])

    def run(self):
        data_path = Path(DATA_PATH)
        if not data_path.is_file():
            print(f"Data file not found: {data_path}")
            return {}

        data, variables = self._read_data(data_path)
        py_b0 = self.run_stats_transformer(data, variables)
        
        results = {}
        matlab_engine = MATLABEngine()
        matlab_b0 = self.run_matlab(data, matlab_engine)
        if matlab_b0 is not None:
            results["matlab_diff"] = self.compare_results(py_b0, matlab_b0)
            print(f"BlanchardQuahBenchmark MATLAB max diff: {results['matlab_diff']:.12g}")
            matlab_engine.close()

        return results


if __name__ == "__main__":
    BlanchardQuahBenchmark().run()
