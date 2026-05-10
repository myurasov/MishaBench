# SPDX-FileCopyrightText: Copyright (c) 2026 Mikhail Yurasov
# SPDX-License-Identifier: Apache-2.0

"""Runtime config: budget, suite gating, sizing knobs.

A single `BenchConfig` flows through the runner; suites read what they
need. `quick=True` shrinks every workload by 10x-100x so a smoke pass
runs in ~5 minutes on a CPU-only laptop.
"""

from __future__ import annotations

from dataclasses import dataclass, field

ALL_SUITES: tuple[str, ...] = ("data", "cv", "llm", "compute")


QUICK_BUDGET_S: float = 5 * 60.0    # 5-minute smoke run
FULL_BUDGET_S: float = 60 * 60.0    # 1-hour default
QUICK_SCALE: float = 0.1            # 10x smaller workloads in quick mode


@dataclass(slots=True)
class BenchConfig:
    suites: tuple[str, ...] = ALL_SUITES
    quick: bool = False

    # Hard ceiling on the whole run. The runner aborts the next benchmark
    # if exceeding this would push it over the wall-clock budget. Defaults
    # auto-shrink to QUICK_BUDGET_S when quick=True (see effective_budget).
    total_budget_s: float = FULL_BUDGET_S

    # Per-benchmark guard. A single bench cannot eat more than this. The
    # runner times out and records ok=false on overflow.
    per_bench_budget_s: float = 5 * 60.0

    # CUDA gating. Auto-detected; can be forced off via --no-cuda for
    # CPU-only baseline runs (handy when comparing two boxes).
    use_cuda: bool = True

    # MPS gating (Apple Silicon). Auto-detected; off for the LLM and CV
    # CUDA-only paths but used for the MPS-capable benches when CUDA is
    # absent.
    use_mps: bool = True

    # Where to write JSONL + the final HTML report.
    output_dir: str = "results"

    # HF model defaults. Kept small so the first-time download budget
    # stays under ~1 GB total. Override per-bench via env vars.
    llm_model: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    cv_classifier: str = "resnet50"  # torchvision name
    cv_feature_model: str = "facebook/dinov2-small"

    # Soft-fail mode: a single bench failure does not abort the run.
    continue_on_error: bool = True

    # Free-form labels recorded in the report header (e.g. "remote=mlbox").
    labels: dict[str, str] = field(default_factory=dict)

    @property
    def effective_budget_s(self) -> float:
        """Budget with --quick shrinkage applied. The CLI passes the raw
        flag in; downstream code reads this property so the rule is in
        one place."""
        if self.quick and self.total_budget_s == FULL_BUDGET_S:
            return QUICK_BUDGET_S
        return self.total_budget_s

    def expected_for(self, expected_seconds: float) -> float:
        """Per-bench expected time, post quick-mode scaling. Used by the
        budget guard in runner.run_all to decide whether to skip a bench
        when only a few seconds remain."""
        if self.quick:
            return max(2.0, expected_seconds * QUICK_SCALE)
        return expected_seconds


def parse_suites(s: str | None) -> tuple[str, ...]:
    """Parse a comma-separated `--suites` value. Empty / None -> all."""
    if not s:
        return ALL_SUITES
    out: list[str] = []
    for raw in s.split(","):
        name = raw.strip().lower()
        if not name:
            continue
        if name not in ALL_SUITES:
            raise ValueError(f"unknown suite: {name!r}; valid: {ALL_SUITES}")
        out.append(name)
    return tuple(out)
