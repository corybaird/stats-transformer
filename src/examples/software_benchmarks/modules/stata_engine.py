import sys
from src.examples.software_benchmarks.modules.base_comparator import BaseSoftwareEngine


class StataEngine(BaseSoftwareEngine):

    def __init__(self, stata_path="/Applications/StataNow/utilities"):
        self.is_ready = False
        try:
            if stata_path not in sys.path:
                sys.path.append(stata_path)
            import pystata
            from pystata import config
            config.init("mp")
            import pystata.stata as stata
            import sfi
            self.stata = stata
            self.sfi = sfi
            self.is_ready = True
        except Exception as err:
            print(f"Stata Engine unavailable: {err}")

    def push_dataframe(self, df):
        if not self.is_ready:
            return False
        self.stata.pdataframe_to_data(df, force=True)
        return True

    def run_command(self, stata_cmd, matrix_name=None):
        if not self.is_ready:
            return None
        self.stata.run(stata_cmd)
        if matrix_name:
            import numpy as np
            return np.asarray(self.sfi.Matrix.get(matrix_name))
        return None
