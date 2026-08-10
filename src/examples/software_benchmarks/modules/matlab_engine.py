import os
from pathlib import Path
from src.examples.software_benchmarks.modules.base_comparator import BaseSoftwareEngine


class MATLABEngine(BaseSoftwareEngine):

    def __init__(self, toolbox_dir=None):
        if toolbox_dir is None:
            default_dir = Path(__file__).parents[4] / "data" / "temp" / "VAR-Toolbox"
            env_dir = os.environ.get("VAR_TOOLBOX_DIR")
            if default_dir.is_dir():
                toolbox_dir = default_dir
            elif env_dir:
                toolbox_dir = Path(env_dir).expanduser()
            else:
                toolbox_dir = default_dir
        self.toolbox_dir = Path(toolbox_dir).resolve() if toolbox_dir else None
        self.eng = None

    def connect(self):
        if self.eng is not None:
            return True
        try:
            import matlab.engine
            names = matlab.engine.find_matlab()
            if names:
                self.eng = matlab.engine.connect_matlab(names[0])
            else:
                self.eng = matlab.engine.start_matlab()
            if self.toolbox_dir and self.toolbox_dir.is_dir():
                self.eng.addpath(self.eng.genpath(str(self.toolbox_dir)), nargout=0)
            return True
        except Exception as err:
            print(f"MATLAB Engine unavailable: {err}")
            return False

    def run_command(self, func_name, *args, **kwargs):
        if not self.connect():
            return None
        func = getattr(self.eng, func_name)
        return func(*args, **kwargs)

    def close(self):
        if self.eng is not None:
            self.eng.quit()
            self.eng = None
