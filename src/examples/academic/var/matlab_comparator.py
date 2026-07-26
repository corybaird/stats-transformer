import os
from pathlib import Path

import numpy as np
import pandas as pd

from stats_transformer.models.timeseries.identification.blanchard_quah import BlanchardQuahModel


class MATLABComparator:

    def __init__(self, toolbox_dir=None):
        configured_dir = toolbox_dir or os.environ.get("VAR_TOOLBOX_DIR")
        self.toolbox_dir = Path(configured_dir).expanduser() if configured_dir else None
        self.eng = None

    def connect_matlab_engine(self):
        if self.toolbox_dir is None or not self.toolbox_dir.is_dir():
            print("Set VAR_TOOLBOX_DIR to the local VAR-Toolbox directory before running this comparator.")
            return False

        try:
            import matlab.engine

            names = matlab.engine.find_matlab()
            if names:
                self.eng = matlab.engine.connect_matlab(names[0])
            else:
                self.eng = matlab.engine.start_matlab()
            self.eng.addpath(self.eng.genpath(str(self.toolbox_dir)), nargout=0)
            return True
        except ImportError:
            print("MATLAB Engine for Python is not installed in the active environment.")
            return False
        except Exception as error:
            print(f"Could not connect to MATLAB Engine: {error}")
            return False

    def _read_bq_data(self, sample_data_path):
        data = pd.read_excel(sample_data_path).iloc[1:].reset_index(drop=True)
        variables = [column for column in data.columns if column not in ["Date", "date", "year", "quarter", "Unnamed: 0"]][:2]
        data[variables] = data[variables].apply(pd.to_numeric, errors="coerce")
        return data[variables].dropna(), variables

    def verify_bq_model(self, data, variables, maxlags=8):
        import matlab

        python_model = BlanchardQuahModel(target_variables=variables, maxlags=maxlags)
        python_model.fit(data)
        options = self.eng.VARoption(nargout=1)
        options["ident"] = "long"
        options["inference"] = 0.0
        matlab_result = self.eng.VARmodel(matlab.double(data.values.tolist()), float(maxlags), 1.0, options, nargout=1)
        matlab_b0 = np.asarray(matlab_result["B"])
        maximum_absolute_difference = float(np.max(np.abs(python_model.B_0 - matlab_b0)))
        np.testing.assert_allclose(python_model.B_0, matlab_b0, rtol=1e-10, atol=1e-10)
        print(f"Blanchard-Quah MATLAB cross-verification passed with maximum absolute difference {maximum_absolute_difference:.12g}.")
        return maximum_absolute_difference

    def run(self, sample_data_path="data/examples/matlab_examples/BQ1989_Data.xlsx"):
        if not self.connect_matlab_engine():
            result = {"matlab_connected": False}
            print(result)
            return result

        try:
            data_path = Path(sample_data_path)
            if not data_path.is_file():
                raise FileNotFoundError(f"Blanchard-Quah sample data not found: {data_path}")
            data, variables = self._read_bq_data(data_path)
            maximum_absolute_difference = self.verify_bq_model(data, variables)
            result = {"matlab_connected": True, "maxlags": 8, "maximum_absolute_difference": maximum_absolute_difference, "observations": len(data), "variables": variables}
            print(result)
            return result
        finally:
            if self.eng is not None:
                self.eng.quit()
                self.eng = None


if __name__ == "__main__":
    MATLABComparator().run()
