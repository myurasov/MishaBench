# SPDX-FileCopyrightText: Copyright (c) 2026 Mikhail Yurasov
# SPDX-License-Identifier: Apache-2.0

"""Typer CLI entry point.

Subcommands:
  info                       -- print system probe (one-line + grid)
  list                       -- print every registered benchmark
  run                        -- run all benches (filtered by --suites);
                                writes JSONL + system.json + report.html
  report <results-dir>       -- regenerate report.html from a results dir
                                (handy after a partial / failed run)

The `--remote <host>` flag on `run` rsyncs the project to the host,
syncs a remote venv via uv, runs the bench there, and pulls the results
back to ./results/<host>-<runid>/.
"""

from __future__ import annotations

import time
from pathlib import Path

import typer
from rich.console import Console

from . import __version__
from .config import BenchConfig, parse_suites
from .remote import run_remote
from .report import load_results, write_report
from .runner import run_all
from .system import probe, short_summary

app = typer.Typer(
    add_completion=False,
    help=(
        "MishaBench v" + __version__ + " -- CPU + GPU benchmark for typical "
        "data, computer-vision, and LLM workloads. Local or --remote (SSH). "
        "Produces a self-contained HTML report."
    ),
)
console = Console()


@app.command()
def info() -> None:
    """Print system probe -- cpu, ram, gpu, cuda, libraries, power source."""
    sys = probe()
    console.print(short_summary(sys))
    if sys.apple_tdp_w is not None:
        console.print(f"  Apple chip TDP estimate: {sys.apple_tdp_w:.0f} W (will be used for points/watt)")
    if sys.has_rapl:
        console.print("  Intel RAPL: available (CPU power = real)")
    elif sys.rapl_status == "permission_denied":
        console.print("  [yellow]Intel RAPL: present but locked to root[/yellow]")
        if sys.rapl_hint:
            console.print(f"  hint: {sys.rapl_hint}")
    if sys.gpus:
        for g in sys.gpus:
            mem = f"{g.memory_mib} MiB" if g.memory_mib else "memory n/a (unified)"
            console.print(f"  GPU[{g.index}]: {g.name} -- {mem}, "
                          f"driver {g.driver}, sm_{g.compute_cap}")
    missing = [k for k, v in sys.libs.items() if v is None]
    if missing:
        console.print(f"  missing: {', '.join(missing)}")


@app.command("list")
def list_benches(
    suites: str = typer.Option(None, "--suites", "-s",
                               help="Comma-separated suites: data,cv,llm. Default: all."),
) -> None:
    """List every registered benchmark."""
    from . import runner as _runner
    cfg = BenchConfig(suites=parse_suites(suites))
    # Trigger registration via lazy import
    _runner._import_suites(cfg.suites)
    rows = [b for b in _runner.registry() if b.suite in cfg.suites]
    for b in rows:
        cap = ",".join(b.requires) or "-"
        console.print(f"  {b.id:30s}  suite={b.suite:5s}  device={b.device:6s}  "
                      f"requires={cap:14s}  ~{b.expected_seconds:.0f}s  {b.name}")
    console.print(f"\n  total: {len(rows)} benchmarks across {len(cfg.suites)} suites")


@app.command()
def run(
    suites: str = typer.Option(None, "--suites", "-s",
                               help="Comma-separated suites: data,cv,llm. Default: all."),
    quick: bool = typer.Option(False, "--quick", "-q",
                               help="5-minute smoke run. Shrinks workloads 10x and caps total budget."),
    no_cuda: bool = typer.Option(False, "--no-cuda",
                                 help="Force CPU-only (skip CUDA benches even if a GPU is present)."),
    no_mps: bool = typer.Option(False, "--no-mps",
                                help="Force MPS off (skip Apple-Silicon GPU benches)."),
    output: str = typer.Option("results", "--output", "-o",
                               help="Output directory for JSONL + report.html."),
    label: list[str] = typer.Option(None, "--label", "-l",
                                    help="key=value tag to embed in the report header. Repeatable."),
    remote: str = typer.Option(None, "--remote", "-r",
                               help="Run on a remote SSH host (alias from ~/.ssh/config). "
                                    "Project is rsynced; results are pulled back to ./results/<host>-<runid>/."),
    gpu_extra: bool = typer.Option(False, "--gpu",
                                   help="Install the gpu extra (cudf-cu12 + cupy). "
                                        "Local runs: handled by the wrapper before "
                                        "this CLI starts (idempotent). Remote runs: "
                                        "passed to the remote install step."),
) -> None:
    """Run benchmarks. Local by default; pass --remote <host> to drive a remote NVIDIA box."""

    if remote:
        out = run_remote(
            remote, suites=suites, quick=quick, no_cuda=no_cuda,
            gpu_extra=gpu_extra,
        )
        if out is None:
            raise typer.Exit(code=1)
        return

    labels: dict[str, str] = {}
    for kv in (label or []):
        if "=" in kv:
            k, v = kv.split("=", 1)
            labels[k.strip()] = v.strip()
        else:
            labels[kv.strip()] = "1"
    if quick:
        labels.setdefault("mode", "quick-5min")

    cfg = BenchConfig(
        suites=parse_suites(suites),
        quick=quick,
        use_cuda=not no_cuda,
        use_mps=not no_mps,
        output_dir=output,
        labels=labels,
    )

    sys_info = probe()
    console.print(short_summary(sys_info))

    started = time.perf_counter()
    results, out_dir = run_all(cfg, sys_info)
    total = time.perf_counter() - started
    report_path = write_report(results, sys_info, out_dir, total, labels=labels)

    failed = [r for r in results if not r.ok and not (r.error or "").startswith(("CUDA", "MPS", "cudf", "cupy", "budget"))]
    skipped = [r for r in results if not r.ok]
    console.print(
        f"  results.jsonl: {out_dir / 'results.jsonl'}\n"
        f"  system.json:   {out_dir / 'system.json'}\n"
        f"  report.html:   {report_path}  (open: file://{report_path.resolve()})"
    )
    console.print(f"  {len(results) - len(skipped)} ok, {len(failed)} failed, "
                  f"{len(skipped) - len(failed)} skipped")


@app.command()
def report(
    results_dir: str = typer.Argument(..., help="A results directory containing results.jsonl + system.json."),
) -> None:
    """Re-render report.html from an existing results directory."""
    rd = Path(results_dir)
    if not (rd / "results.jsonl").exists():
        raise typer.BadParameter(f"missing {rd / 'results.jsonl'}")
    rows, info = load_results(rd)
    total = sum(r.seconds for r in rows)
    p = write_report(rows, info, rd, total, labels={"rerendered": "1"})
    console.print(f"wrote {p}  (open: file://{p.resolve()})")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
