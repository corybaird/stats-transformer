from abc import ABC, abstractmethod
import numpy as np


class BaseSoftwareEngine(ABC):

    @abstractmethod
    def run_command(self, command_str, **kwargs):
        pass


class BaseBenchmark(ABC):

    def compare_results(self, py_coef, native_coef, rtol=1e-5, atol=1e-5):
        max_diff = float(np.max(np.abs(py_coef - native_coef)))
        np.testing.assert_allclose(py_coef, native_coef, rtol=rtol, atol=atol)
        return max_diff

    @abstractmethod
    def run(self):
        pass
