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
    #    live, instead of waiting silently while gigabytes copy.
    #
    #    The script also auto-fixes the torch CUDA wheel: PyPI's default
    #    torch ships with cu130 these days; older drivers (e.g. 535.x ->
    #    CUDA 12.2) need cu126 or cu118 instead. We detect the mismatch
    #    via `torch.cuda.is_available()` after the initial sync and
    #    reinstall from the appropriate PyTorch index when needed.
    #
    #    With --gpu we also `uv pip install` the RAPIDS stack out-of-band
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

        # Auto-fix torch CUDA wheel if driver/wheel mismatch detected.
        # Skipped when MISHABENCH_TORCH_CUDA=cpu or the user explicitly
        # forces a particular index via the env var.
        TC="${{MISHABENCH_TORCH_CUDA:-auto}}"
        if [ "$TC" = "cpu" ]; then
          echo "==== MISHABENCH_TORCH_CUDA=cpu -- installing CPU-only torch ===="
          uv pip install --index-url https://download.pytorch.org/whl/cpu \\
            torch torchvision --reinstall
        elif [ "$TC" != "auto" ]; then
          echo "==== MISHABENCH_TORCH_CUDA=$TC -- forcing torch wheel ===="
          uv pip install --index-url "https://download.pytorch.org/whl/$TC" \\
            torch torchvision --reinstall
        elif command -v nvidia-smi >/dev/null 2>&1; then
          if ! .venv/bin/python -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
            DRV=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader,nounits 2>/dev/null | head -1 | cut -d. -f1 | tr -d ' ')
            if [ -n "$DRV" ]; then
              CU=""
              if [ "$DRV" -ge 545 ]; then
                echo "==== driver $DRV supports cu130 (default); skipping reinstall ===="
              elif [ "$DRV" -ge 525 ]; then
                CU=cu126
              elif [ "$DRV" -ge 470 ]; then
                CU=cu118
              else
                echo "==== driver $DRV too old for any current torch CUDA build ===="
              fi
              if [ -n "$CU" ]; then
                echo "==== driver $DRV detected; reinstalling torch from https://download.pytorch.org/whl/$CU ===="
                uv pip install --index-url "https://download.pytorch.org/whl/$CU" \\
                  torch torchvision --reinstall
                if .venv/bin/python -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
                  echo "==== torch.cuda now sees the GPU ===="
                else
                  echo "==== torch.cuda still unavailable; CUDA benches will skip ===="
                fi
              fi
            fi
          fi
        fi

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

    # Invoke .venv/bin/python directly, NOT `uv run python` -- the latter
    # triggers an implicit `uv sync` before every command, which reverts
    # our auto-fix's cu126 torch install back to the lockfile's cu130
    # exactly when the bench is about to start. Mirrors what the local
    # wrapper does (PYTHONPATH=$HERE exec $VENV_DIR/bin/python -m src ...).
    console.print(f"[bold]>[/bold] running on {host}: mishabench run {flag_str}")
    rc = _run.ssh_stream(host, f"""
        export PATH="$HOME/.local/bin:$PATH"
        cd {remote_tool} && PYTHONPATH={remote_tool} .venv/bin/python -m src run {flag_str}
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
