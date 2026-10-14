import hashlib
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


class MatlabRefRecorder:
    def __init__(self):
        self.here = Path(__file__).resolve().parent
        self.root = self.here.parents[2]
        self.cfg = yaml.safe_load((self.here / "cases.yaml").read_text())
        self.eng = None
        self.summary = []

    def run(self):
        import matlab.engine

        self.eng = matlab.engine.connect_matlab(matlab.engine.find_matlab()[0])
        self.eng.eval("addpath('" + str(self.here / "drivers") + "');", nargout=0)
        for case in self.cfg["cases"] + self.cfg.get("generic_cases", []):
            self._record_case(case)
        for row in self.summary:
            print(row)

    def _sha(self, path):
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()

    def _to_list(self, value):
        return np.asarray(value).tolist()

    def _engine_info(self):
        version = self.eng.version(nargout=1)
        release = self.eng.eval("version('-release')", nargout=1)
        return str(version), str(release)

    def _traced_call(self, name):
        self.eng.eval("profile clear; profile on;", nargout=0)
        try:
            raw = getattr(self.eng, "st_ref_" + name)(str(self.root), nargout=1)
        finally:
            self.eng.eval("profile off; st_info = profile('info'); st_fn = strjoin(string({st_info.FunctionTable.FileName}), '|'); st_root = matlabroot;", nargout=0)
        files = [f for f in str(self.eng.workspace["st_fn"]).split("|") if f]
        mroot = str(self.eng.workspace["st_root"])
        self.eng.eval("clear st_info st_fn st_root;", nargout=0)
        return raw, sorted(set(files)), mroot

    def _executed_dependencies(self, files, mroot):
        tb_root = self.root / self.cfg["toolbox"]["root"]
        tb_files = {}
        products = ["MATLAB"]
        core = {"matlab", "shared", "local", "compiler_sdk", "toolbox_cache"}
        tool_dirs = set()
        for f in files:
            fp = Path(f)
            if tb_root in fp.parents:
                tb_files[str(fp.relative_to(tb_root))] = self._sha(fp)
            elif (self.root / "tests" / "external_benchmarks" / "matlab" / "third_party") in fp.parents:
                tb_files[str(fp.relative_to(self.root))] = self._sha(fp)
            elif Path(mroot, "toolbox") in fp.parents:
                top = fp.relative_to(Path(mroot, "toolbox")).parts[0]
                if top not in core:
                    tool_dirs.add(top)
        for rel in self.cfg["toolbox"]["always_hash"]:
            tb_files.setdefault(rel, self._sha(tb_root / rel))
        for top in sorted(tool_dirs):
            try:
                self.eng.eval("st_v = ver('" + top + "'); st_vn = st_v(1).Name;", nargout=0)
                products.append(str(self.eng.workspace["st_vn"]))
            except Exception:
                products.append("toolbox/" + top)
        return dict(sorted(tb_files.items())), products

    def _record_case(self, case):
        name = case["case"]
        driver_path = self.here / "drivers" / ("st_ref_" + name + ".m")
        out_path = self.here / (name + ".json")
        err_path = self.here / (name + ".error.txt")
        start = time.time()
        try:
            raw, files, mroot = self._traced_call(name)
            shapes = {}
            outputs = {}
            for key, value in raw.items():
                arr = np.asarray(value)
                if hasattr(value, "size") and not np.isscalar(value):
                    assert tuple(value.size) == arr.shape, (key, value.size, arr.shape)
                shapes[key] = list(arr.shape)
                outputs[key] = self._to_list(value)
            record = self._build_record(case, driver_path, outputs, files, mroot)
            out_path.write_text(json.dumps(record, indent=1))
            if err_path.exists():
                err_path.unlink()
            self.summary.append((name, "ok", shapes, round(time.time() - start, 2)))
        except Exception as exc:
            err_path.write_text(repr(exc) + "\n")
            self.summary.append((name, "error", repr(exc)[:200], round(time.time() - start, 2)))

    def _build_record(self, case, driver_path, outputs, files, mroot):
        version, release = self._engine_info()
        files_sha, prods = self._executed_dependencies(files, mroot)
        record = {
            "case": case["case"],
            "engine": {"product": "MATLAB", "version": version, "release": release, "toolboxes": prods},
            "toolbox": {"name": self.cfg["toolbox"]["name"], "version": self.cfg["toolbox"]["version"], "files_sha256": files_sha},
        }
        if case["kind"] == "generic":
            record["data"] = {"files": [{"path": f, "sha256": self._sha(self.root / f)} for f in case["data"]["files"]]}
            settings = dict(case.get("settings", {}))
            for key in [k for k in outputs if k.startswith("set_")]:
                settings[key[4:]] = outputs.pop(key)
            record["settings"] = settings
            inputs = {k: outputs.pop(k) for k in case.get("inputs", []) if k in outputs}
            layout = dict(case["layout"])
        elif case["kind"] == "csv":
            data_path = self.root / case["data"]["path"]
            frame = pd.read_csv(data_path)
            record["data"] = {"path": case["data"]["path"], "sha256": self._sha(data_path), "sheet": None, "columns": case["columns"], "rows_used": [1, len(frame)]}
            record["settings"] = dict(case["settings"])
            inputs = {"X": outputs.pop("X"), "y": outputs.pop("y")}
            layout = dict(case["layout"])
        else:
            record.update(self._var_data_and_settings(case, outputs))
            inputs = {"ENDO": outputs.pop("ENDO"), "IV": outputs.pop("IV", None)}
            layout = {k: v for k, v in self.cfg["var_layout"].items() if k in outputs or k in inputs}
            self._check_inputs(case, inputs)
        record["inputs"] = inputs
        record["outputs"] = outputs
        record["layout"] = layout
        record["driver"] = str(driver_path.relative_to(self.root))
        record["driver_sha256"] = self._sha(driver_path)
        record["recorded_at"] = datetime.now(timezone.utc).isoformat()
        record["host"] = platform.platform() + " " + platform.machine()
        return record

    def _read_sheet(self, case):
        d = case["data"]
        return pd.read_excel(self.root / d["path"], sheet_name=d["sheet"], header=None)

    def _var_data_and_settings(self, case, outputs):
        sheet = self._read_sheet(case)
        d = case["data"]
        mnem = [str(m) for m in sheet.iloc[d["mnemonic_row"] - 1, 1:].tolist()]
        nobs = sheet.shape[0] - (d["first_data_row"] - 1)
        columns = case["endo"] or mnem
        if case["iv"]:
            columns = columns + case["iv"]
        data_path = self.root / d["path"]
        data = {"path": d["path"], "sha256": self._sha(data_path), "sheet": d["sheet"], "columns": columns, "rows_used": [1, nobs], "first_date": str(sheet.iloc[d["first_data_row"] - 1, 0]), "last_date": str(sheet.iloc[-1, 0])}
        settings = {"nlags": int(outputs.pop("set_nlags")), "const": int(outputs.pop("set_const")), "ident": case["ident"], "nsteps": int(outputs.pop("set_nsteps")), "inference": int(outputs.pop("set_inference"))}
        return {"data": data, "settings": settings}

    def _check_inputs(self, case, inputs):
        sheet = self._read_sheet(case)
        d = case["data"]
        mnem = [str(m) for m in sheet.iloc[d["mnemonic_row"] - 1, 1:].tolist()]
        body = sheet.iloc[d["first_data_row"] - 1:, 1:].astype(float).to_numpy()
        body[body == 123456789] = np.nan
        endo_cols = case["endo"] or mnem
        expected = body[:, [mnem.index(c) for c in endo_cols]]
        got = np.asarray(inputs["ENDO"], dtype=float)
        assert got.shape == expected.shape and np.allclose(got, expected, equal_nan=True, rtol=0, atol=0), "ENDO mismatch for " + case["case"]
        if case["iv"]:
            exp_iv = body[:, [mnem.index(c) for c in case["iv"]]]
            got_iv = np.asarray(inputs["IV"], dtype=float)
            assert np.allclose(got_iv, exp_iv, equal_nan=True, rtol=0, atol=0), "IV mismatch for " + case["case"]


if __name__ == "__main__":
    MatlabRefRecorder().run()
