# SPDX-FileCopyrightText: Copyright (c) 2026 Mikhail Yurasov
# SPDX-License-Identifier: Apache-2.0

"""Data-analysis suite: pandas vs polars vs cudf (where available).

Workloads:
  D1 csv_parse        -- read a synthetic CSV, sum a numeric column
  D2 group_aggregate  -- groupby + multi-agg on a synthetic frame
  D3 join             -- left-join 5M-row driving table with 500k-row dim
  D4 parquet_round    -- write + read a synthetic parquet (with snappy)
  D5 string_ops       -- regex extract + lower over a 1M-row string column

Sizes scale by 10x between `--quick` (smoke) and the default budget so
the suite stays under ~12 min on a modest CPU laptop. cudf paths run
the same logic against cudf for an apples-to-apples comparison; if
cudf is missing the bench records `ok=False` with a clear reason.
"""

from __future__ import annotations

import string
import tempfile
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
    if quick:
        return {"main": 500_000, "dim": 50_000, "strings": 100_000}
    return {"main": 5_000_000, "dim": 500_000, "strings": 1_000_000}


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
    df = pd.read_csv(p)
    rows = len(df)
    _ = float(df["x"].sum()) + float(df["y"].sum())
    mb = p.stat().st_size / 2**20
    return ("throughput", round(mb, 2), "MiB/s parsed (single shot)",
            {"rows": rows, "file_mib": round(mb, 2)})


def bench_csv_polars(cfg: BenchConfig):
    import polars as pl
    p = _csv_path_for(cfg)
    df = pl.read_csv(p)
    rows = df.height
    _ = float(df["x"].sum()) + float(df["y"].sum())
    mb = p.stat().st_size / 2**20
    return ("throughput", round(mb, 2), "MiB/s parsed (single shot)",
            {"rows": rows, "file_mib": round(mb, 2)})


def bench_csv_cudf(cfg: BenchConfig):
    import cudf  # type: ignore[import-not-found]
    p = _csv_path_for(cfg)
    df = cudf.read_csv(p)
    rows = len(df)
    _ = float(df["x"].sum()) + float(df["y"].sum())
    mb = p.stat().st_size / 2**20
    return ("throughput", round(mb, 2), "MiB/s parsed (single shot)",
            {"rows": rows, "file_mib": round(mb, 2)})


# ---------- D2: group-by aggregate ----------

def _frame_pandas(cfg: BenchConfig):
    key = f"pd-{_sizes(cfg.quick)['main']}"
    if key not in _FRAMES:
        _FRAMES[key] = _build_pandas_frame(_sizes(cfg.quick)["main"])
    return _FRAMES[key]


def bench_groupby_pandas(cfg: BenchConfig):
    df = _frame_pandas(cfg)
    out = df.groupby("group", sort=False).agg(
        x_sum=("x", "sum"),
        x_mean=("x", "mean"),
        y_max=("y", "max"),
        n=("id", "size"),
    )
    return ("throughput", round(len(df) / 1e6, 3), "M rows aggregated",
            {"groups_out": len(out)})


def bench_groupby_polars(cfg: BenchConfig):
    import polars as pl
    df = pl.from_pandas(_frame_pandas(cfg))
    out = df.group_by("group").agg(
        pl.col("x").sum().alias("x_sum"),
        pl.col("x").mean().alias("x_mean"),
        pl.col("y").max().alias("y_max"),
        pl.col("id").count().alias("n"),
    )
    return ("throughput", round(df.height / 1e6, 3), "M rows aggregated",
            {"groups_out": out.height})


def bench_groupby_cudf(cfg: BenchConfig):
    import cudf  # type: ignore[import-not-found]
    pdf = _frame_pandas(cfg)
    df = cudf.from_pandas(pdf)
    out = df.groupby("group").agg({"x": ["sum", "mean"], "y": "max", "id": "count"})
    return ("throughput", round(len(df) / 1e6, 3), "M rows aggregated",
            {"groups_out": len(out)})


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
    merged = left.merge(right, on="id", how="left")
    n = len(merged)
    return ("throughput", round(n / 1e6, 3), "M rows joined",
            {"left_rows": len(left), "right_rows": len(right)})


def bench_join_polars(cfg: BenchConfig):
    import polars as pl
    left = pl.from_pandas(_frame_pandas(cfg).loc[:, ["id", "x"]])
    right = pl.from_pandas(_dim_pandas(cfg))
    merged = left.join(right, on="id", how="left")
    return ("throughput", round(merged.height / 1e6, 3), "M rows joined",
            {"left_rows": left.height, "right_rows": right.height})


def bench_join_cudf(cfg: BenchConfig):
    import cudf  # type: ignore[import-not-found]
    left = cudf.from_pandas(_frame_pandas(cfg).loc[:, ["id", "x"]])
    right = cudf.from_pandas(_dim_pandas(cfg))
    merged = left.merge(right, on="id", how="left")
    return ("throughput", round(len(merged) / 1e6, 3), "M rows joined",
            {"left_rows": len(left), "right_rows": len(right)})


# ---------- D4: parquet round-trip ----------

def bench_parquet_pandas(cfg: BenchConfig):
    df = _frame_pandas(cfg)
    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
        path = Path(f.name)
    try:
        df.to_parquet(path, compression="snappy")
        df2 = __import__("pandas").read_parquet(path)
        size_mib = path.stat().st_size / 2**20
        return ("throughput", round(size_mib, 2), "MiB round-trip",
                {"rows": len(df2), "file_mib": round(size_mib, 2)})
    finally:
        path.unlink(missing_ok=True)


def bench_parquet_polars(cfg: BenchConfig):
    import polars as pl
    df = pl.from_pandas(_frame_pandas(cfg))
    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
        path = Path(f.name)
    try:
        df.write_parquet(path, compression="snappy")
        df2 = pl.read_parquet(path)
        size_mib = path.stat().st_size / 2**20
        return ("throughput", round(size_mib, 2), "MiB round-trip",
                {"rows": df2.height, "file_mib": round(size_mib, 2)})
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
    extracted = df["line"].str.extract(r"^([a-z]+)-(\d+)@host(\d+)\.")
    upper = df["line"].str.upper()
    return ("throughput", round(len(df) / 1e6, 3), "M rows regex+upper",
            {"first_unique": int(extracted[0].nunique()), "n_upper": len(upper)})


def bench_regex_polars(cfg: BenchConfig):
    import polars as pl
    df = pl.from_pandas(_strings_pandas(cfg))
    extracted = df.select(
        pl.col("line").str.extract(r"^([a-z]+)-(\d+)@host(\d+)\.", 1).alias("g1"),
        pl.col("line").str.extract(r"^([a-z]+)-(\d+)@host(\d+)\.", 2).alias("g2"),
        pl.col("line").str.extract(r"^([a-z]+)-(\d+)@host(\d+)\.", 3).alias("g3"),
    )
    upper = df.select(pl.col("line").str.to_uppercase())
    return ("throughput", round(df.height / 1e6, 3), "M rows regex+upper",
            {"first_unique": extracted["g1"].n_unique(), "n_upper": upper.height})


# ---------- registration ----------

# expected_seconds is conservative (CPU laptop estimate). If the bench
# finishes faster on a beefy box, that's fine; the budget guard only
# uses it to *skip* benches we know won't fit in the remaining budget.

register(Bench("data.csv.pandas", "data", "CSV parse", "cpu", bench_csv_pandas, expected_seconds=30))
register(Bench("data.csv.polars", "data", "CSV parse", "cpu", bench_csv_polars, expected_seconds=15))
register(Bench("data.csv.cudf",   "data", "CSV parse", "cuda", bench_csv_cudf, requires=("cuda", "cudf"), expected_seconds=15))

register(Bench("data.gb.pandas",  "data", "Group-by aggregate", "cpu", bench_groupby_pandas, expected_seconds=15))
register(Bench("data.gb.polars",  "data", "Group-by aggregate", "cpu", bench_groupby_polars, expected_seconds=10))
register(Bench("data.gb.cudf",    "data", "Group-by aggregate", "cuda", bench_groupby_cudf, requires=("cuda", "cudf"), expected_seconds=10))

register(Bench("data.join.pandas","data", "Left join", "cpu", bench_join_pandas, expected_seconds=20))
register(Bench("data.join.polars","data", "Left join", "cpu", bench_join_polars, expected_seconds=10))
register(Bench("data.join.cudf",  "data", "Left join", "cuda", bench_join_cudf, requires=("cuda", "cudf"), expected_seconds=10))

register(Bench("data.pq.pandas",  "data", "Parquet round-trip", "cpu", bench_parquet_pandas, expected_seconds=20))
register(Bench("data.pq.polars",  "data", "Parquet round-trip", "cpu", bench_parquet_polars, expected_seconds=15))

register(Bench("data.regex.pandas","data", "Regex + uppercase", "cpu", bench_regex_pandas, expected_seconds=30))
register(Bench("data.regex.polars","data", "Regex + uppercase", "cpu", bench_regex_polars, expected_seconds=15))
