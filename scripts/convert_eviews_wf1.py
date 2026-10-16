"""Convert an EViews/MicroTSP .wf1 workfile to parquet.

Follows the reverse-engineered format of the "New MicroTSP Workfile" layout
described by Allin Cottrell (https://users.wfu.edu/cottrell/eviews_format/):
a 24-byte identifier, a header holding the data frequency, sample start and
series count, a directory of 70-byte series records (name at record offset 22,
data-block position at offset 14), then per-series data blocks: a 22-byte
header (number of observations, start and end observation) followed by
little-endian float64 values. EViews NA (1e-37) becomes NaN. The constant C
and boilerplate RESID series are skipped.

Usage:
    uv run python scripts/convert_eviews_wf1.py in.wf1 out.parquet
    uv run python scripts/convert_eviews_wf1.py in.wf1 out.parquet --validate
"""

import argparse
import struct
import sys
from pathlib import Path

import numpy as np
import pandas as pd

EVIEWS_NA = 1e-37
HEADER_OFFSET = 24  # "New MicroTSP Workfile" identifier is 24 bytes
RECORD_SIZE = 70
# The 70-byte series records start 2 bytes after the header measured by the
# value at offset 80 (Cottrell's hlen + 26 relative to the file start).
RECORD_PADDING = 2


def read_wf1(path: Path) -> tuple[pd.DataFrame, dict]:
    """Read a .wf1 file and return (series as a DataFrame, header metadata)."""
    data = path.read_bytes()
    if not data.startswith(b"New MicroTSP Workfile"):
        raise ValueError(f"{path}: not a MicroTSP/EViews workfile")
    hlen = struct.unpack_from("<q", data, 80)[0]
    n_series = struct.unpack_from("<i", data, 114)[0] - 1
    freq = struct.unpack_from("<H", data, 124)[0]
    start_year = struct.unpack_from("<i", data, 128)[0]
    start_sub = struct.unpack_from("<i", data, 132)[0]
    n_obs = struct.unpack_from("<i", data, 140)[0]

    columns = {}
    for i in range(n_series):
        rec = data[hlen + HEADER_OFFSET + RECORD_PADDING + RECORD_SIZE * i : hlen + HEADER_OFFSET + RECORD_PADDING + RECORD_SIZE * (i + 1)]
        name = rec[22:54].split(b"\x00")[0].decode("ascii")
        code = struct.unpack_from("<h", rec, 62)[0]
        pos = struct.unpack_from("<i", rec, 14)[0]  # 4-byte stream position; the 4 bytes
        # after it are usually 0 but can be nonzero (greenbookquarterly PI_TARGET), so
        # the pointer must be read as <i, not as the 8-byte long Cottrell reports.
        if code != 44 or name == "RESID" or pos == 0:
            continue
        if not 0 < pos < len(data):
            raise ValueError(f"{path}: {name} points at an invalid data-block position ({pos})")
        n_block = struct.unpack_from("<i", data, pos)[0]
        values = np.frombuffer(data, dtype="<f8", count=n_block, offset=pos + 22).astype(float)
        values[values == EVIEWS_NA] = np.nan
        columns[name] = values

    if freq == 4:
        index = pd.period_range(f"{start_year}Q{start_sub or 1}", periods=n_obs, freq="Q").to_timestamp()
    else:
        index = range(1, n_obs + 1)
    df = pd.DataFrame(columns, index=index)
    return df, {"frequency": freq, "start_year": start_year, "start_subperiod": start_sub, "n_series": n_series, "n_obs": n_obs}


def validate(df: pd.DataFrame, meta: dict):
    """Print series count, sample span and NaN share per series."""
    print(f"series kept: {df.shape[1]} (n_series incl. C/RESID: {meta['n_series']})")
    print(f"observation count in header: {meta['n_obs']}, rows after dropping all-NaN: {df.shape[0]}")
    if meta["frequency"] == 4:
        print(f"sample: {df.index[0]} to {df.index[-1]} (quarterly)")
    spot = ["FFR", "EPIA", "EGY0"]
    for name in spot:
        if name in df.columns:
            s = df[name].dropna()
            print(f"  {name}: n={len(s)} first3={s.iloc[[0, 1, 2]].round(4).tolist()} last3={s.iloc[[-3, -2, -1]].round(4).tolist()}")
    for name in [c for c in df.columns if c not in spot and df[c].isna().all()]:
        print(f"  WARNING: all-NaN series {name}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args()
    df, meta = read_wf1(args.input)
    if args.validate:
        validate(df, meta)
    df = df.dropna(axis=0, how="all")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.output)
    print(f"wrote {args.output} ({df.shape[0]} rows, {df.shape[1]} series)")


if __name__ == "__main__":
    main()