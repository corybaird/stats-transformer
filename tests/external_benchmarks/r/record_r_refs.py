import json
import subprocess
import tempfile
import time
from pathlib import Path

import yaml

from tests.external_benchmarks.ref_common import RefCommon


class RRefRecorder:
    def __init__(self):
        self.here = Path(__file__).resolve().parent
        self.root = self.here.parents[2]
        self.cfg = yaml.safe_load((self.here / "cases.yaml").read_text())
        self.common = RefCommon(self.root)
        self.summary = []

    def run(self):
        with tempfile.TemporaryDirectory() as tmp:
            for case in self.cfg["cases"]:
                self._record_case(case, Path(tmp))
        for row in self.summary:
            print(row)

    def _docker(self, case, tmp):
        name = case["case"]
        out_in_container = "/out/" + name + ".json"
        cmd = ["docker", "run", "--rm", "-v", str(self.root) + ":/repo:ro", "-v", str(tmp) + ":/out", self.cfg["image"], "Rscript", "/repo/tests/external_benchmarks/r/drivers/r_ref_" + name + ".R", "/repo", out_in_container]
        if case.get("needs_csv"):
            self.common.xlsx_to_csv(case["data"], tmp / (name + "_input.csv"))
            cmd.append("/out/" + name + "_input.csv")
        elif case.get("input_from"):
            self.common.json_to_csv(case["input_from"], tmp / (name + "_input.csv"))
            cmd.append("/out/" + name + "_input.csv")
        return subprocess.run(cmd, capture_output=True, text=True)

    def _record_case(self, case, tmp):
        name = case["case"]
        start = time.time()
        out_path = self.here / (name + ".json")
        err_path = self.here / (name + ".error.txt")
        proc = self._docker(case, tmp)
        raw_path = tmp / (name + ".json")
        if proc.returncode != 0 or not raw_path.exists():
            err_path.write_text(proc.stdout + proc.stderr)
            self.summary.append((name, "error", proc.stderr[-200:], round(time.time() - start, 1)))
            return
        raw = json.loads(raw_path.read_text())
        record = {
            "case": name,
            "engine": {"product": "R", "version": raw["r_version"], "platform": raw["platform"], "image": self.cfg["image"], "packages": raw["packages"]},
            "data": self.common.json_data_block(case["input_from"]) if "input_from" in case else self.common.data_block(case["data"]),
            "settings": {"call": case["call"]},
            "outputs": raw["outputs"],
            "labels": raw["labels"],
            "layout": case["layout"],
            "driver": "tests/external_benchmarks/r/drivers/r_ref_" + name + ".R",
            "driver_sha256": self.common.sha(self.here / "drivers" / ("r_ref_" + name + ".R")),
            "recorded_at": self.common.now(),
            "host": self.common.host(),
        }
        out_path.write_text(json.dumps(record, indent=1))
        if err_path.exists():
            err_path.unlink()
        self.summary.append((name, "ok", {k: _shape(v) for k, v in raw["outputs"].items()}, round(time.time() - start, 1)))


def _shape(v):
    shape = []
    while isinstance(v, list):
        shape.append(len(v))
        v = v[0] if v else None
    return shape


if __name__ == "__main__":
    RRefRecorder().run()
