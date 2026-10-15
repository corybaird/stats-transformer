import os
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SCRIPTS = [
    "src/examples/academic/nakamura_steinsson_pca.py",
    "src/examples/discrete/spector_logit.py",
    "src/examples/regression/grunfeld.py",
    "src/examples/regression/longley.py",
    "src/examples/regression/mincer_wage.py",
    "src/examples/regression/mroz_iv.py",
    "src/examples/regression/okuns_law.py",
    "src/examples/timeseries/macro_var.py",
]


def git_data_status():
    result = subprocess.run(
        ["git", "status", "--porcelain", "data/"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )
    return result.stdout


@pytest.mark.parametrize("script", SCRIPTS)
def test_example_script_does_not_modify_tracked_data(script):
    before = git_data_status()
    env = dict(os.environ, PYTHONPATH=str(PROJECT_ROOT))
    result = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / script)],
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert git_data_status() == before