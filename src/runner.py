# SPDX-FileCopyrightText: Copyright (c) 2026 Mikhail Yurasov
# SPDX-License-Identifier: Apache-2.0

"""Bench registry, timing harness, result schema, run loop.

Suites register `Bench` instances at import time via `register(...)`.
The runner walks the registry, applies the budget guard + per-bench
timeout, swallows failures (recording them as `ok=False`), and writes
JSONL + a final HTML report.

Each `Bench` is a thin wrapper around a callable that returns a
`(metric, value, unit, notes)` tuple. The harness handles warmup,
timing, error capture, and budget enforcement.
"""

from __future__ import annotations

import gc
import json
import time
import traceback
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rich.console import Console

from .config import BenchConfig
from .power import PowerMonitor
from .system import SystemInfo

console = Console()


@dataclass(slots=True)
class BenchResult:
    id: str
    suite: str
    name: str
    device: str
    metric: str
    value: float | None
    unit: str
    seconds: float
    iters: int
    ok: bool = True
    error: str | None = None
    notes: dict[str, Any] = field(default_factory=dict)
    started_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# A bench function returns either:
#   (metric_name, value, unit, notes_dict)         -- single shot
# or:
#   {"metric": ..., "value": ..., "unit": ..., "notes": {...}, "iters": ...}
BenchFn = Callable[..., Any]


@dataclass(slots=True)
class Bench:
    id: str
    suite: str
    name: str
    device: str
    fn: BenchFn
    requires: tuple[str, ...] = ()  # capability flags: "cuda", "cudf", "mps"
    description: str = ""
    expected_seconds: float = 30.0  # rough hint, used by budget pre-check


_REGISTRY: list[Bench] = []


def register(bench: Bench) -> None:
    _REGISTRY.append(bench)


def registry() -> list[Bench]:
    return list(_REGISTRY)


def _capability_ok(bench: Bench, info: SystemInfo, cfg: BenchConfig) -> tuple[bool, str | None]:
    for cap in bench.requires:
        if cap == "cuda" and not (info.has_cuda and cfg.use_cuda):
            return False, "CUDA not available"
        if cap == "mps" and not (info.has_mps and cfg.use_mps):
            return False, "MPS not available"
        if cap == "cudf" and not info.has_cudf:
            return False, "cudf not installed (install with `./mishabench install --gpu`)"
        if cap == "cupy" and not info.has_cupy:
            return False, "cupy not installed (install with `./mishabench install --gpu`)"
    return True, None


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _coerce_result(raw: Any) -> dict[str, Any]:
    """Bench fn output -> normalized dict."""
    if isinstance(raw, dict):
        out = dict(raw)
        out.setdefault("metric", "value")
        out.setdefault("value", None)
        out.setdefault("unit", "")
        out.setdefault("notes", {})
        out.setdefault("iters", 1)
        return out
    if isinstance(raw, tuple) and len(raw) >= 3:
        metric, value, unit = raw[0], raw[1], raw[2]
        notes = raw[3] if len(raw) > 3 and isinstance(raw[3], dict) else {}
        return {"metric": metric, "value": value, "unit": unit, "notes": notes, "iters": 1}
    return {"metric": "value", "value": raw, "unit": "", "notes": {}, "iters": 1}


def _run_one(bench: Bench, cfg: BenchConfig, info: SystemInfo,
             prefix: str = "  ") -> BenchResult:
    """Run a single bench. `prefix` is the [N/M] counter the caller wants
    on the start line; the result line is indented to that width so the
    output reads as paired lines per bench. Slow benches (CV inference,
    LLM decode) print the start line immediately so the user sees that
    work has begun, then the result on a continuation line when done."""
    started = _now_iso()
    indent = " " * len(prefix)

    cap_ok, cap_reason = _capability_ok(bench, info, cfg)
    if not cap_ok:
        console.print(f"{prefix} [yellow]skip[/] {bench.id} ({bench.device}): {cap_reason}")
        return BenchResult(
            id=bench.id, suite=bench.suite, name=bench.name, device=bench.device,
            metric="skipped", value=None, unit="", seconds=0.0, iters=0,
            ok=False, error=cap_reason, started_at=started,
        )

    console.print(f"{prefix} [cyan]run[/]  {bench.id} ({bench.device}) [dim]running...[/]")
    t0 = time.perf_counter()
    monitor = PowerMonitor(apple_tdp_w=info.apple_tdp_w)
    try:
        with monitor as window:
            raw = bench.fn(cfg)
        elapsed = time.perf_counter() - t0
        norm = _coerce_result(raw)
        avg_w, est = window.power_for_device(bench.device)
        merged_notes = dict(norm["notes"])
        merged_notes["power"] = window.to_dict()
        if avg_w is not None:
            merged_notes["avg_watts"] = round(avg_w, 2)
            merged_notes["power_estimated"] = est
        watt_part = (f", {avg_w:.1f}W{' est' if est else ''}"
                     if avg_w is not None else "")
        value_str = (
            f"{norm['value']:.2f} {norm['unit']}"
            if isinstance(norm["value"], (int, float))
            else f"{norm['value']} {norm['unit']}"
        )
        console.print(f"{indent} [green]ok[/]   -> {value_str} in {elapsed:.2f}s{watt_part}")
        return BenchResult(
            id=bench.id, suite=bench.suite, name=bench.name, device=bench.device,
            metric=norm["metric"], value=norm["value"], unit=norm["unit"],
            seconds=elapsed, iters=norm["iters"], ok=True,
            notes=merged_notes, started_at=started,
        )
    except Exception as e:
        elapsed = time.perf_counter() - t0
        tb = traceback.format_exc(limit=4)
        console.print(f"{indent} [red]FAIL[/] in {elapsed:.2f}s: {type(e).__name__}: {e}")
        return BenchResult(
            id=bench.id, suite=bench.suite, name=bench.name, device=bench.device,
            metric="error", value=None, unit="", seconds=elapsed, iters=0,
            ok=False, error=f"{type(e).__name__}: {e}\n{tb}",
            started_at=started,
        )
    finally:
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except Exception:
            pass


def _import_suites(suites: tuple[str, ...]) -> None:
    """Import suite modules so their register() calls run. Lazy on purpose."""
    for s in suites:
        try:
            __import__(f"src.suites.{s}", fromlist=["*"])
        except ModuleNotFoundError:
            __import__(f"suites.{s}", fromlist=["*"])


def run_all(cfg: BenchConfig, info: SystemInfo) -> tuple[list[BenchResult], Path]:
    """Run every registered bench under the suite filter. Returns
    (results, output_dir). Writes results.jsonl + system.json + report.html
    into output_dir.
    """
    _REGISTRY.clear()  # idempotency: support repeat calls in long-lived processes
    _import_suites(cfg.suites)

    out_dir = Path(cfg.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    selected = [b for b in _REGISTRY if b.suite in cfg.suites]
    console.print(f"[bold]MishaBench:[/] {len(selected)} benchmarks across "
                  f"{len(cfg.suites)} suites -> {out_dir}")

    started = time.perf_counter()
    results: list[BenchResult] = []
    total = len(selected)
    width = max(2, len(str(total)))  # zero-pad counter to align prefix
    jsonl_path = out_dir / "results.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as fp:
        for i, bench in enumerate(selected, 1):
            prefix = f"[{i:>{width}d}/{total}]"
            elapsed_total = time.perf_counter() - started
            remaining = cfg.effective_budget_s - elapsed_total
            need = cfg.expected_for(bench.expected_seconds)
            if remaining < need:
                console.print(
                    f"{prefix} [yellow]budget[/] {bench.id}: "
                    f"need ~{need:.0f}s, only {remaining:.0f}s left"
                )
                r = BenchResult(
                    id=bench.id, suite=bench.suite, name=bench.name, device=bench.device,
                    metric="skipped", value=None, unit="", seconds=0.0, iters=0,
                    ok=False, error=f"budget: only {remaining:.0f}s left",
                    started_at=_now_iso(),
                )
            else:
                r = _run_one(bench, cfg, info, prefix=prefix)
            results.append(r)
            fp.write(json.dumps(r.to_dict()) + "\n")
            fp.flush()

    total = time.perf_counter() - started
    console.print(f"[green]done[/] in {total:.1f}s ({total/60:.1f} min)")

    sys_path = out_dir / "system.json"
    sys_path.write_text(json.dumps(info.to_dict(), indent=2), encoding="utf-8")

    return results, out_dir
