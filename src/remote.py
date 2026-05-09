# SPDX-FileCopyrightText: Copyright (c) 2026 Mikhail Yurasov
# SPDX-License-Identifier: Apache-2.0

"""--remote driver: rsync project, sync remote venv, run, fetch results.

Layout on the remote host:
    ~/mishabench/
        mishabench-tool/   <- rsynced source (this project, minus venv/results)
        results/           <- where bench output lands

Workflow on `mishabench run --remote <ssh-host> [opts]`:

  1. ssh <host> -- mkdir the layout
  2. rsync source tree -> ~/mishabench/mishabench-tool/
  3. ssh <host> -- ensure uv installed, `cd mishabench-tool && uv sync`
                   (with --extra gpu when the user passed --gpu)
  4. ssh <host> -- run `uv run python -m src run <opts> --output ~/mishabench/results/<runid>`
                   (output streamed to local terminal)
  5. rsync results back -> ./results/<remote-host>-<runid>/
  6. (optional) open the report locally
"""

from __future__ import annotations

import datetime as _dt
from pathlib import Path

from rich.console import Console

from . import _run

console = Console()

_RSYNC_EXCLUDES = [
    ".venv", "__pycache__", "*.pyc",
    ".pytest_cache", ".ruff_cache", ".mypy_cache",
    "*.egg-info", "dist", "build",
    "uv.lock",  # remote resolves its own
    ".DS_Store", ".git",
    "results",  # never push local results to the remote
    ".mishabench-cache", "*.parquet",
]

REMOTE_BASE_DEFAULT = "$HOME/mishabench"


def _local_project_root() -> Path:
    # __file__ -> src/remote.py; project root is one level up.
    return Path(__file__).resolve().parents[1]


def _runid() -> str:
    return _dt.datetime.now().strftime("%Y%m%d-%H%M%S")


def run_remote(host: str, *, suites: str | None = None, quick: bool = False,
               no_cuda: bool = False, gpu_extra: bool = False,
               remote_base: str = REMOTE_BASE_DEFAULT,
               local_results_root: Path | None = None) -> Path | None:
    """Run the suite on `host` and rsync the results back. Returns the
    local results directory on success, or None on failure."""

    project = _local_project_root()
    runid = _runid()
    remote_tool = f"{remote_base}/mishabench-tool"
    remote_results = f"{remote_base}/results/{runid}"
    local_dir = (local_results_root or (Path.cwd() / "results")) / f"{host}-{runid}"

    # 1. Ensure remote layout exists. ssh_run is fine here -- no progress
    #    to stream, just two mkdir's.
    console.print(f"[bold]>[/bold] mkdir on {host}")
    r = _run.ssh_run(host, f"""
        mkdir -p {remote_tool}
        mkdir -p {remote_results}
    """)
    if not r.ok:
        console.print(f"[red]ssh mkdir failed[/red]\n{r.stderr}")
        return None

    # 2. Rsync project sources -- streams rsync's per-file progress live
    #    so the user sees the upload as it happens.
    console.print(f"[bold]>[/bold] rsync {project.name}/ -> {host}:{remote_tool}/")
    rc = _run.rsync_to(host, project, remote_tool, exclude=_RSYNC_EXCLUDES)
    if rc != 0:
        console.print(f"[red]rsync failed (rc={rc})[/red]")
        return None

    # 3. Ensure uv + sync remote venv. Streamed -- the user sees uv's
    #    "Resolved N packages" / "Downloaded torch (700 MiB)" progress
    #    live, instead of waiting silently while gigabytes copy. With
    #    --gpu we also `uv pip install` the RAPIDS stack out-of-band
    #    against pypi.nvidia.com.
    extras = "--extra dev"
    gpu_step = (
        "uv pip install --extra-index-url https://pypi.nvidia.com cudf-cu12 cupy-cuda12x"
        if gpu_extra else "true"
    )
    console.print(f"[bold]>[/bold] uv sync on {host} ({extras}"
                  f"{' + gpu extras' if gpu_extra else ''}) -- live output:")
    rc = _run.ssh_run_stream(host, f"""
        export PATH="$HOME/.local/bin:$PATH"
        if ! command -v uv >/dev/null; then
          echo "==== Installing uv on remote ===="
          curl -LsSf https://astral.sh/uv/install.sh | sh
          export PATH="$HOME/.local/bin:$PATH"
        fi
        cd {remote_tool}
        uv sync {extras}
        {gpu_step}
        echo "==== uv sync complete ===="
    """)
    if rc != 0:
        console.print(f"[red]uv sync failed (rc={rc})[/red]")
        return None

    # 4. Build remote run command and stream output live
    flags: list[str] = []
    if suites:
        flags += ["--suites", suites]
    if quick:
        flags.append("--quick")
    if no_cuda:
        flags.append("--no-cuda")
    flags += ["--output", remote_results, "--label", f"remote={host}"]
    flag_str = " ".join(flags)

    console.print(f"[bold]>[/bold] running on {host}: mishabench run {flag_str}")
    rc = _run.ssh_stream(host, f"""
        export PATH="$HOME/.local/bin:$PATH"
        cd {remote_tool} && PYTHONPATH={remote_tool} uv run python -m src run {flag_str}
    """)
    if rc != 0:
        console.print(f"[red]remote bench exited rc={rc}[/red] -- "
                      "fetching whatever results were written so far")

    # 5. rsync results back regardless of bench exit code (partial results
    #    are still useful). Streamed too -- the result tarball is small
    #    but a slow link is a slow link.
    local_dir.mkdir(parents=True, exist_ok=True)
    console.print(f"[bold]>[/bold] rsync {host}:{remote_results}/ -> {local_dir}/")
    rc = _run.rsync_from(host, remote_results, local_dir)
    if rc != 0:
        console.print(f"[red]rsync from remote failed (rc={rc})[/red]")
        return None

    console.print(f"[green]done[/green] -- results in {local_dir}/")
    report = local_dir / "report.html"
    if report.exists():
        console.print(f"  open: file://{report.resolve()}")
    return local_dir
