# SPDX-FileCopyrightText: Copyright (c) 2026 Mikhail Yurasov
# SPDX-License-Identifier: Apache-2.0

"""Data-analysis suite: pandas vs polars vs cudf (where available).

Workloads:
  D1 csv_parse        -- read a synthetic CSV, sum a numeric column
  D2 group_aggregate  -- groupby + multi-agg on a synthetic frame
  D3 join             -- left-join 100M-row driving table with 10M-row dim
  D4 parquet_round    -- write + read a synthetic parquet (with snappy)
  D5 string_ops       -- regex extract + uppercase over a 5M-row string column

Sizes are deliberately big: the main pandas frame at 100M rows is
~2.1 GB in RAM; the polars and cudf copies bring the working set to
~5-6 GB during the join bench. This makes the suite sensitive to
memory bandwidth + compute, not just per-row Python overhead. Skips
gracefully on hosts that don't have the headroom (the runner records
the OOM and continues).

Quick mode (`--quick`) shrinks every workload by 10x so a smoke pass
runs in ~5 minutes on a modest CPU laptop. cudf paths run the same
logic against cudf for an apples-to-apples comparison; if cudf is
missing the bench records `ok=False` with a clear reason.
"""

from __future__ import annotations

import string
import tempfile
import time as _time
from pathlib import Path
from typing import Any

import numpy as np

from ..config import BenchConfig
from ..runner import Bench, register

# ---------- shared synthetic data builders (cached per process) ----------

_FRAMES: dict[str, Any] = {}


def _rng(seed: int = 7) -> np.random.Generator:
    return np.random.default_rng(seed)


def _build_pandas_frame(n_rows: int, n_groups: int = 1024) -> Any:
    import pandas as pd
    rng = _rng()
    return pd.DataFrame({
        "id": np.arange(n_rows, dtype=np.int64),
        "group": rng.integers(0, n_groups, n_rows, dtype=np.int32),
        "x": rng.standard_normal(n_rows).astype(np.float32),
        "y": rng.standard_normal(n_rows).astype(np.float32),
        "flag": rng.integers(0, 2, n_rows, dtype=np.int8),
    })


def _build_pandas_strings(n_rows: int) -> Any:
    """Synthetic strings shaped like 'abc-1234@host42.example' for regex
    extract. Keeps cardinality high so dictionary encoding can't trivialize
    the work."""
    import pandas as pd
    rng = _rng()
    letters = np.array(list(string.ascii_lowercase))
    head = ["".join(c) for c in rng.choice(letters, (n_rows, 4))]
    nums = rng.integers(0, 10000, n_rows, dtype=np.int32).astype("U")
    hosts = rng.integers(0, 1024, n_rows, dtype=np.int32).astype("U")
    s = pd.Series([f"{h}-{n}@host{x}.example" for h, n, x in zip(head, nums, hosts, strict=False)])
    return pd.DataFrame({"line": s})


def _build_csv(path: Path, df: Any) -> int:
    df.to_csv(path, index=False)
    return path.stat().st_size


def _sizes(quick: bool) -> dict[str, int]:
    # Full mode: main frame is ~2.1 GB pandas (100M rows x 21 bytes/row),
    # dim is ~80 MB, strings is ~400 MB. Polars and cudf each materialise
    # their own copy on demand, so peak working set during the join bench
    # is ~5-6 GB across the three frames.
    if quick:
        return {"main": 10_000_000, "dim": 1_000_000, "strings": 500_000}
    return {"main": 100_000_000, "dim": 10_000_000, "strings": 5_000_000}


# ---------- D1: CSV parse ----------

def _csv_path_for(cfg: BenchConfig) -> Path:
    s = _sizes(cfg.quick)["main"]
    cache = Path(tempfile.gettempdir()) / f"mishabench-csv-{s}.csv"
    if not cache.exists():
        df = _build_pandas_frame(s)
        _build_csv(cache, df)
    return cache


def bench_csv_pandas(cfg: BenchConfig):
    import pandas as pd
    p = _csv_path_for(cfg)
    mb = p.stat().st_size / 2**20
    t0 = _time.perf_counter()
    df = pd.read_csv(p)
    _ = float(df["x"].sum()) + float(df["y"].sum())
    sec = _time.perf_counter() - t0
    rate = mb / sec if sec > 0 else 0.0
    return ("throughput", round(rate, 2), "MiB/s parsed",
            {"rows": len(df), "file_mib": round(mb, 2), "seconds": round(sec, 4)})


def bench_csv_polars(cfg: BenchConfig):
    import polars as pl
    p = _csv_path_for(cfg)
    mb = p.stat().st_size / 2**20
    t0 = _time.perf_counter()
    df = pl.read_csv(p)
    _ = float(df["x"].sum()) + float(df["y"].sum())
    sec = _time.perf_counter() - t0
    rate = mb / sec if sec > 0 else 0.0
    return ("throughput", round(rate, 2), "MiB/s parsed",
            {"rows": df.height, "file_mib": round(mb, 2), "seconds": round(sec, 4)})


def bench_csv_cudf(cfg: BenchConfig):
    import cudf  # type: ignore[import-not-found]
    p = _csv_path_for(cfg)
    mb = p.stat().st_size / 2**20
    t0 = _time.perf_counter()
    df = cudf.read_csv(p)
    _ = float(df["x"].sum()) + float(df["y"].sum())
    sec = _time.perf_counter() - t0
    rate = mb / sec if sec > 0 else 0.0
    return ("throughput", round(rate, 2), "MiB/s parsed",
            {"rows": len(df), "file_mib": round(mb, 2), "seconds": round(sec, 4)})


# ---------- D2: group-by aggregate ----------

def _frame_pandas(cfg: BenchConfig):
    key = f"pd-{_sizes(cfg.quick)['main']}"
    if key not in _FRAMES:
        _FRAMES[key] = _build_pandas_frame(_sizes(cfg.quick)["main"])
    return _FRAMES[key]


def bench_groupby_pandas(cfg: BenchConfig):
    df = _frame_pandas(cfg)
    rows_m = len(df) / 1e6
    t0 = _time.perf_counter()
    out = df.groupby("group", sort=False).agg(
        x_sum=("x", "sum"),
        x_mean=("x", "mean"),
        y_max=("y", "max"),
        n=("id", "size"),
    )
    sec = _time.perf_counter() - t0
    rate = rows_m / sec if sec > 0 else 0.0
    return ("throughput", round(rate, 3), "M rows/s aggregated",
            {"groups_out": len(out), "rows_m": round(rows_m, 3), "seconds": round(sec, 4)})


def bench_groupby_polars(cfg: BenchConfig):
    import polars as pl
    # Frame conversion is excluded from the timed window: it's the equivalent
    # of pandas already having the frame in RAM. Time only the actual op.
    df = pl.from_pandas(_frame_pandas(cfg))
    rows_m = df.height / 1e6
    t0 = _time.perf_counter()
    out = df.group_by("group").agg(
        pl.col("x").sum().alias("x_sum"),
        pl.col("x").mean().alias("x_mean"),
        pl.col("y").max().alias("y_max"),
        pl.col("id").count().alias("n"),
    )
    sec = _time.perf_counter() - t0
    rate = rows_m / sec if sec > 0 else 0.0
    return ("throughput", round(rate, 3), "M rows/s aggregated",
            {"groups_out": out.height, "rows_m": round(rows_m, 3), "seconds": round(sec, 4)})


def bench_groupby_cudf(cfg: BenchConfig):
    import cudf  # type: ignore[import-not-found]
    # Host->device transfer is excluded from the timed window for the same
    # reason polars's from_pandas is excluded above: time the operation,
    # not the data setup. cudf operations are synchronous so perf_counter
    # accurately captures kernel time without manual cuda-sync.
    pdf = _frame_pandas(cfg)
    df = cudf.from_pandas(pdf)
    rows_m = len(df) / 1e6
    t0 = _time.perf_counter()
    out = df.groupby("group").agg({"x": ["sum", "mean"], "y": "max", "id": "count"})
    sec = _time.perf_counter() - t0
    rate = rows_m / sec if sec > 0 else 0.0
    return ("throughput", round(rate, 3), "M rows/s aggregated",
            {"groups_out": len(out), "rows_m": round(rows_m, 3), "seconds": round(sec, 4)})


# ---------- D3: join ----------

def _dim_pandas(cfg: BenchConfig):
    key = f"dim-{_sizes(cfg.quick)['dim']}"
    if key not in _FRAMES:
        import pandas as pd
        n = _sizes(cfg.quick)["dim"]
        _FRAMES[key] = pd.DataFrame({
            "id": np.arange(n, dtype=np.int64),
            "label": _rng(11).integers(0, 1000, n, dtype=np.int32),
        })
    return _FRAMES[key]


def bench_join_pandas(cfg: BenchConfig):
    left = _frame_pandas(cfg)[["id", "x"]]
    right = _dim_pandas(cfg)
    t0 = _time.perf_counter()
    merged = left.merge(right, on="id", how="left")
    sec = _time.perf_counter() - t0
    rows_m = len(merged) / 1e6
    rate = rows_m / sec if sec > 0 else 0.0
    return ("throughput", round(rate, 3), "M rows/s joined",
            {"left_rows": len(left), "right_rows": len(right),
             "rows_m": round(rows_m, 3), "seconds": round(sec, 4)})


def bench_join_polars(cfg: BenchConfig):
    import polars as pl
    left = pl.from_pandas(_frame_pandas(cfg).loc[:, ["id", "x"]])
    right = pl.from_pandas(_dim_pandas(cfg))
    t0 = _time.perf_counter()
    merged = left.join(right, on="id", how="left")
    sec = _time.perf_counter() - t0
    rows_m = merged.height / 1e6
    rate = rows_m / sec if sec > 0 else 0.0
    return ("throughput", round(rate, 3), "M rows/s joined",
            {"left_rows": left.height, "right_rows": right.height,
             "rows_m": round(rows_m, 3), "seconds": round(sec, 4)})


def bench_join_cudf(cfg: BenchConfig):
    import cudf  # type: ignore[import-not-found]
    left = cudf.from_pandas(_frame_pandas(cfg).loc[:, ["id", "x"]])
    right = cudf.from_pandas(_dim_pandas(cfg))
    t0 = _time.perf_counter()
    merged = left.merge(right, on="id", how="left")
    sec = _time.perf_counter() - t0
    rows_m = len(merged) / 1e6
    rate = rows_m / sec if sec > 0 else 0.0
    return ("throughput", round(rate, 3), "M rows/s joined",
            {"left_rows": len(left), "right_rows": len(right),
             "rows_m": round(rows_m, 3), "seconds": round(sec, 4)})


# ---------- D4: parquet round-trip ----------

def bench_parquet_pandas(cfg: BenchConfig):
    import pandas as pd
    df = _frame_pandas(cfg)
    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
        path = Path(f.name)
    try:
        t0 = _time.perf_counter()
        df.to_parquet(path, compression="snappy")
        df2 = pd.read_parquet(path)
        sec = _time.perf_counter() - t0
        size_mib = path.stat().st_size / 2**20
        # Round-trip throughput: file size both written and read once each,
        # so effective bytes processed = 2 * file_size. Reported as
        # "round-trip MiB/s" so a higher number is better and the metric
        # composes with the geomean cleanly.
        rate = (2 * size_mib) / sec if sec > 0 else 0.0
        return ("throughput", round(rate, 2), "MiB/s round-trip",
                {"rows": len(df2), "file_mib": round(size_mib, 2),
                 "seconds": round(sec, 4)})
    finally:
        path.unlink(missing_ok=True)


def bench_parquet_polars(cfg: BenchConfig):
    import polars as pl
    df = pl.from_pandas(_frame_pandas(cfg))
    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
        path = Path(f.name)
    try:
        t0 = _time.perf_counter()
        df.write_parquet(path, compression="snappy")
        df2 = pl.read_parquet(path)
        sec = _time.perf_counter() - t0
        size_mib = path.stat().st_size / 2**20
        rate = (2 * size_mib) / sec if sec > 0 else 0.0
        return ("throughput", round(rate, 2), "MiB/s round-trip",
                {"rows": df2.height, "file_mib": round(size_mib, 2),
                 "seconds": round(sec, 4)})
    finally:
        path.unlink(missing_ok=True)


# ---------- D5: string / regex ----------

def _strings_pandas(cfg: BenchConfig):
    key = f"str-{_sizes(cfg.quick)['strings']}"
    if key not in _FRAMES:
        _FRAMES[key] = _build_pandas_strings(_sizes(cfg.quick)["strings"])
    return _FRAMES[key]


def bench_regex_pandas(cfg: BenchConfig):
    df = _strings_pandas(cfg)
    rows_m = len(df) / 1e6
    t0 = _time.perf_counter()
    extracted = df["line"].str.extract(r"^([a-z]+)-(\d+)@host(\d+)\.")
    upper = df["line"].str.upper()
    sec = _time.perf_counter() - t0
    rate = rows_m / sec if sec > 0 else 0.0
    return ("throughput", round(rate, 3), "M rows/s regex+upper",
            {"first_unique": int(extracted[0].nunique()), "n_upper": len(upper),
             "rows_m": round(rows_m, 3), "seconds": round(sec, 4)})


def bench_regex_polars(cfg: BenchConfig):
    import polars as pl
    df = pl.from_pandas(_strings_pandas(cfg))
    rows_m = df.height / 1e6
    t0 = _time.perf_counter()
    extracted = df.select(
        pl.col("line").str.extract(r"^([a-z]+)-(\d+)@host(\d+)\.", 1).alias("g1"),
        pl.col("line").str.extract(r"^([a-z]+)-(\d+)@host(\d+)\.", 2).alias("g2"),
        pl.col("line").str.extract(r"^([a-z]+)-(\d+)@host(\d+)\.", 3).alias("g3"),
    )
    upper = df.select(pl.col("line").str.to_uppercase())
    sec = _time.perf_counter() - t0
    rate = rows_m / sec if sec > 0 else 0.0
    return ("throughput", round(rate, 3), "M rows/s regex+upper",
            {"first_unique": extracted["g1"].n_unique(), "n_upper": upper.height,
             "rows_m": round(rows_m, 3), "seconds": round(sec, 4)})


# ---------- registration ----------

# expected_seconds is conservative (CPU laptop estimate at full size,
# 100M-row main frame). The budget guard only uses these to *skip* a
# bench when not enough wall time remains. Tuned upward from the prior
# 5M-row workload sizes -- the new full-mode sizes are 20x larger.

register(Bench("data.csv.pandas", "data", "CSV parse", "cpu", bench_csv_pandas, expected_seconds=120))
register(Bench("data.csv.polars", "data", "CSV parse", "cpu", bench_csv_polars, expected_seconds=20))
register(Bench("data.csv.cudf",   "data", "CSV parse", "cuda", bench_csv_cudf, requires=("cuda", "cudf"), expected_seconds=20))

register(Bench("data.gb.pandas",  "data", "Group-by aggregate", "cpu", bench_groupby_pandas, expected_seconds=30))
register(Bench("data.gb.polars",  "data", "Group-by aggregate", "cpu", bench_groupby_polars, expected_seconds=15))
register(Bench("data.gb.cudf",    "data", "Group-by aggregate", "cuda", bench_groupby_cudf, requires=("cuda", "cudf"), expected_seconds=15))

register(Bench("data.join.pandas","data", "Left join", "cpu", bench_join_pandas, expected_seconds=120))
register(Bench("data.join.polars","data", "Left join", "cpu", bench_join_polars, expected_seconds=45))
register(Bench("data.join.cudf",  "data", "Left join", "cuda", bench_join_cudf, requires=("cuda", "cudf"), expected_seconds=30))

register(Bench("data.pq.pandas",  "data", "Parquet round-trip", "cpu", bench_parquet_pandas, expected_seconds=90))
register(Bench("data.pq.polars",  "data", "Parquet round-trip", "cpu", bench_parquet_polars, expected_seconds=30))

register(Bench("data.regex.pandas","data", "Regex + uppercase", "cpu", bench_regex_pandas, expected_seconds=120))
register(Bench("data.regex.polars","data", "Regex + uppercase", "cpu", bench_regex_polars, expected_seconds=15))
