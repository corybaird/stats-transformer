import os
import sys
import numpy as np
import pandas as pd
import matlab
from stats_transformer.models.timeseries.blanchard_quah import BlanchardQuahModel
from stats_transformer.models.timeseries.svar import SVARModel

class MATLABComparator:

    def __init__(self, matlab_benchmark_dir="references/matlab_benchmarks"):
        self.benchmark_dir = os.path.abspath(matlab_benchmark_dir)
        self.eng = None

    def connect_matlab_engine(self):
        try:
            import matlab.engine
            names = matlab.engine.find_matlab()
            if names:
                self.eng = matlab.engine.connect_matlab(names[0])
            else:
                self.eng = matlab.engine.start_matlab()
            self.eng.addpath(self.eng.genpath(self.benchmark_dir), nargout=0)
            return True
        except ImportError:
            print("MATLAB Engine for Python (matlab.engine) is not installed in the environment.")
            return False
        except Exception as e:
            print(f"Could not connect to MATLAB Engine: {e}")
            return False

    def verify_bq_model(self, df, vars_to_use):
        py_model = BlanchardQuahModel(target_variables=vars_to_use, maxlags=2)
        py_model.fit(df)
        py_b0 = py_model.B_0
        if self.eng:
            y_mat = matlab.double(df[vars_to_use].dropna().values.tolist())
            VARopt = self.eng.VARoption(nargout=1)
            VARopt['ident'] = 'long'
            res = self.eng.VARmodel(y_mat, float(2), float(1), VARopt, nargout=1)
            mat_b0 = np.array(res['B'])
            np.testing.assert_allclose(py_b0, mat_b0, rtol=1e-3, atol=1e-3)
            print("Blanchard-Quah MATLAB cross-verification PASSED!")
        else:
            print("Skipped MATLAB engine run (engine not active). Python model trained successfully.")

    def run(self, sample_data_path="data/examples/matlab_examples/BQ1989_Data.xlsx"):
        connected = self.connect_matlab_engine()
        if os.path.exists(sample_data_path):
            df = pd.read_excel(sample_data_path)
            df = df.iloc[1:].reset_index(drop=True)
            vars_to_use = [c for c in df.columns if c not in ['Date', 'date', 'year', 'quarter', 'Unnamed: 0']][:2]
            self.verify_bq_model(df, vars_to_use)
        return {"matlab_connected": connected}

if __name__ == "__main__":
    comparator = MATLABComparator()
    comparator.run()
