import hashlib
import platform
from datetime import datetime, timezone
from pathlib import Path

import json

import numpy as np
import pandas as pd


class RefCommon:
    def __init__(self, root):
        self.root = Path(root)

    def sha(self, path):
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()

    def data_block(self, spec):
        path = self.root / spec["path"]
        if path.suffix == ".xlsx":
            frame = pd.read_excel(path, sheet_name=spec.get("sheet", "Sheet1"), header=None)
            nrows = frame.shape[0] - spec.get("first_data_row", 3) + 1
            columns = [str(c) for c in frame.iloc[spec.get("mnemonic_row", 2) - 1, 1:]]
        else:
            frame = pd.read_csv(path)
            nrows = len(frame)
            columns = [str(c) for c in frame.columns]
        return {"path": spec["path"], "sha256": self.sha(path), "sheet": spec.get("sheet"), "columns": columns, "nrows": int(nrows), "filter": spec.get("filter")}

    def host(self):
        return platform.platform() + " " + platform.machine()

    def now(self):
        return datetime.now(timezone.utc).isoformat()

    def xlsx_to_csv(self, spec, out_path):
        frame = pd.read_excel(self.root / spec["path"], sheet_name=spec.get("sheet", "Sheet1"), header=None)
        body = frame.iloc[spec.get("first_data_row", 3) - 1:, 1:].astype(float)
        body.columns = [str(c) for c in frame.iloc[spec.get("mnemonic_row", 2) - 1, 1:]]
        body.to_csv(out_path, index=False, float_format="%.17g")

    def json_to_csv(self, spec, out_path):
        src = json.loads((self.root / spec["json"]).read_text())
        cols = {}
        for name, key, idx in spec["columns"]:
            arr = np.asarray(src["inputs"][key], dtype=float)
            cols[name] = arr[:, idx] if arr.ndim == 2 else arr
        pd.DataFrame(cols).to_csv(out_path, index=False, float_format="%.17g")

    def json_data_block(self, spec):
        path = self.root / spec["json"]
        return {"source_json": spec["json"], "sha256": self.sha(path), "columns": [c[0] for c in spec["columns"]], "note": "input matrix copied from the MATLAB reference inputs so every program uses identical data"}
