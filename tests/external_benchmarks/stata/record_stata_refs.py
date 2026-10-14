import json
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
import yaml

from tests.external_benchmarks.ref_common import RefCommon

STATA = "/Applications/StataNow/StataMP.app/Contents/MacOS/stata-mp"


class StataRefRecorder:
    def __init__(self):
        self.here = Path(__file__).resolve().parent
        self.root = self.here.parents[2]
        self.cfg = yaml.safe_load((self.here / "cases.yaml").read_text())
        self.common = RefCommon(self.root)
        self.summary = []

    def run(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            engine = self._engine_info(tmp)
            for case in self.cfg["cases"]:
                self._record_case(case, tmp, engine)
        for row in self.summary:
            print(row)

    def _stata(self, do_file, args, cwd):
        return subprocess.run([STATA, "-b", "do", str(do_file)] + [str(a) for a in args], cwd=cwd, capture_output=True, text=True)

    def _engine_info(self, tmp):
        do = tmp / "probe.do"
        do.write_text('file open fh using "' + str(tmp / "probe.txt") + '", write text replace\nfile write fh "`c(stata_version)'"'"'|`c(born_date)'"'"'|`c(edition_real)'"'"'|`c(os)'"'"'|`c(machine_type)'"'"'|`c(current_date)'"'"'" _n\nfile close fh\nexit, clear\n')
        self._stata(do, [], tmp)
        parts = (tmp / "probe.txt").read_text().strip().split("|")
        return {"product": "Stata", "version": parts[0], "born_date": parts[1], "edition": parts[2], "os": parts[3], "machine_type": parts[4]}

    def _load(self, path):
        rows = []
        for line in Path(path).read_text().strip().splitlines():
            rows.append([np.nan if t.strip() == "." else float(t) for t in line.split(",")])
        return rows

    def _labels(self, path):
        lab = Path(str(path) + ".labels")
        if not lab.exists():
            return {"rows": [], "cols": []}
        lines = lab.read_text().splitlines()
        return {"rows": lines[0].split() if lines else [], "cols": lines[1].split() if len(lines) > 1 else []}

    def _record_case(self, case, tmp, engine):
        name = case["case"]
        start = time.time()
        out_dir = tmp / ("out_" + name)
        out_dir.mkdir()
        args = [self.root, out_dir]
        if case.get("needs_csv"):
            csv = tmp / (name + "_input.csv")
            self.common.xlsx_to_csv(case["data"], csv)
            args.append(csv)
        elif case.get("input_from"):
            csv = tmp / (name + "_input.csv")
            self.common.json_to_csv(case["input_from"], csv)
            args.append(csv)
        do_file = self.here / "drivers" / ("st_ref_" + name + ".do")
        out_path = self.here / (name + ".json")
        err_path = self.here / (name + ".error.txt")
        self._stata(do_file, args, tmp)
        log = tmp / ("st_ref_" + name + ".log")
        missing = [f for f in case["outputs"].values() if not (out_dir / f).exists()]
        if missing:
            err_path.write_text("missing outputs: " + ", ".join(missing) + "\n" + (log.read_text()[-3000:] if log.exists() else ""))
            self.summary.append((name, "error", missing, round(time.time() - start, 1)))
            return
        outputs = {k: self._load(out_dir / f) for k, f in case["outputs"].items()}
        labels = {k: self._labels(out_dir / f) for k, f in case["outputs"].items()}
        record = {
            "case": name,
            "engine": engine,
            "data": self.common.json_data_block(case["input_from"]) if "input_from" in case else self.common.data_block(case["data"]),
            "settings": {"command": case["command"]},
            "outputs": outputs,
            "labels": labels,
            "layout": case["layout"],
            "driver": "tests/external_benchmarks/stata/drivers/st_ref_" + name + ".do",
            "driver_sha256": self.common.sha(do_file),
            "recorded_at": self.common.now(),
            "host": self.common.host(),
        }
        out_path.write_text(json.dumps(record, indent=1))
        if err_path.exists():
            err_path.unlink()
        self.summary.append((name, "ok", {k: [len(v), len(v[0])] for k, v in outputs.items()}, round(time.time() - start, 1)))


if __name__ == "__main__":
    StataRefRecorder().run()
