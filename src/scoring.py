# SPDX-FileCopyrightText: Copyright (c) 2026 Mikhail Yurasov
# SPDX-License-Identifier: Apache-2.0

"""Score aggregation.

We don't have a published reference device, so scores are "absolute" in
the same sense as a SPEC composite: geometric mean of per-bench raw
values, scaled by a normalization constant chosen so a typical mid-range
laptop CPU lands near 1000. The constant is fixed across releases so
two runs are directly comparable; the absolute number is meaningful for
ratios (laptop vs DGX) even without a reference run.

The geomean choice (vs arithmetic mean) means a single bench cannot
dominate the score by being on a different scale than the others;
1.0x faster on a slow bench moves the score by the same proportion as
1.0x faster on a fast bench.

Per-suite, per-device sub-scores are reported alongside the device-total.
Points/watt = device_score / device_avg_watts (averaged across the
benches that ran on that device).
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field

from .runner import BenchResult

# Calibrated 2026-05 against a mid-range laptop (Apple M1, 16 GB) running
# the full suite (no LLM): geomean of raw values came out near 1.0; we
# scale by 1000 so the resulting score reads as "points". Fixed across
# releases so two runs are comparable.
SCORE_SCALE = 1000.0


def _device_class(device: str) -> str:
    """Bucket per-bench device strings into the three reporting classes.

    Anything starting with "cuda" -- including "cuda_multi" used by the
    multi-GPU benches -- buckets into "cuda" so the suite score reflects
    the host's overall CUDA capability. Anything starting with "cpu"
    (including "cpu_st" / "cpu_mt" used by the compute suite for
    single/multi-thread variants) buckets into "cpu"."""
    if device.startswith("cuda"):
        return "cuda"
    if device == "mps":
        return "mps"
    return "cpu"


def _geomean(values: list[float]) -> float | None:
    """Geometric mean. Drops non-positive values (a 0 or negative value
    would zero or invalidate the score; we'd rather report a partial
    score than no score at all)."""
    pos = [v for v in values if v is not None and v > 0]
    if not pos:
        return None
    return math.exp(sum(math.log(v) for v in pos) / len(pos))


@dataclass(slots=True)
class DeviceScore:
    device: str             # "cpu" / "cuda" / "mps"
    suite: str              # "data" / "cv" / "llm" / "total"
    score: float | None     # geomean(raw values) * SCORE_SCALE
    n_benches: int          # how many benches contributed
    avg_watts: float | None
    energy_j: float | None
    pts_per_watt: float | None
    estimated_power: bool


@dataclass(slots=True)
class ScoreReport:
    per_device_per_suite: dict[str, dict[str, DeviceScore]] = field(default_factory=dict)
    # NOTE: per_device_total is intentionally NOT populated. A geomean
    # across suites with wildly different units (M rows/s + img/s + tok/s
    # + GFLOPS) is a dimensionally meaningless number; keeping it would
    # invite users to compare it across hosts as if it were calibrated
    # against something. The per-suite scores ARE meaningful (geomean of
    # similar-unit benches within a suite) and that's the level we report.
    # The field stays on the dataclass for backward-compat with old code
    # that loads ScoreReport from JSON, but compute() leaves it empty.
    per_device_total: dict[str, DeviceScore] = field(default_factory=dict)


def compute(results: list[BenchResult]) -> ScoreReport:
    # Bucket: (device_class, suite) -> list of (raw_value, watts, est_flag, seconds)
    buckets: dict[tuple[str, str], list[tuple[float, float | None, bool, float]]] = defaultdict(list)
    for r in results:
        if not r.ok or not isinstance(r.value, (int, float)):
            continue
        dev = _device_class(r.device)
        watts = r.notes.get("avg_watts") if r.notes else None
        est = bool(r.notes.get("power_estimated")) if r.notes else False
        buckets[(dev, r.suite)].append((float(r.value), watts, est, r.seconds))

    rep = ScoreReport()

    for (dev, suite), rows in buckets.items():
        gm = _geomean([v for v, _, _, _ in rows])
        score = round(gm * SCORE_SCALE, 1) if gm is not None else None

        watt_vals = [w for _, w, _, _ in rows if w is not None and w > 0]
        avg_w = (sum(watt_vals) / len(watt_vals)) if watt_vals else None
        any_est = any(e for _, w, e, _ in rows if w is not None)

        # Energy: sum(watts * seconds) per bench. Honest aggregation that
        # weights long benches more than short ones.
        energy_j = sum(w * sec for _, w, _, sec in rows if w is not None) or None
        pts_per_watt = (score / avg_w) if (score is not None and avg_w and avg_w > 0) else None

        rep.per_device_per_suite.setdefault(dev, {})[suite] = DeviceScore(
            device=dev, suite=suite, score=score, n_benches=len(rows),
            avg_watts=round(avg_w, 2) if avg_w else None,
            energy_j=round(energy_j, 1) if energy_j else None,
            pts_per_watt=round(pts_per_watt, 2) if pts_per_watt else None,
            estimated_power=any_est,
        )

    # Per-device totals deliberately not computed -- see the docstring
    # on ScoreReport.per_device_total above.

    return rep
