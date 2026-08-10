from src.examples.software_benchmarks.modules.base_comparator import BaseSoftwareEngine


class REngine(BaseSoftwareEngine):

    def __init__(self):
        self.is_ready = False
        try:
            import rpy2.robjects as ro
            from rpy2.robjects import pandas2ri
            from rpy2.robjects.conversion import localconverter
            self.ro = ro
            self.pandas2ri = pandas2ri
            self.localconverter = localconverter
            self.is_ready = True
        except Exception as err:
            print(f"R Engine unavailable: {err}")

    def push_dataframe(self, df):
        if not self.is_ready:
            return False
        with self.localconverter(self.ro.default_converter + self.pandas2ri.converter):
            self.ro.globalenv['df'] = df
        return True

    def run_command(self, r_code, result_expr=None):
        if not self.is_ready:
            return None
        self.ro.r(r_code)
        if result_expr:
            import numpy as np
            return np.asarray(self.ro.r(result_expr))
        return None
