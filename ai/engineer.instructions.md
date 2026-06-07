# Engineer instructions - MishaBench <!-- omit in toc -->

- [Identity + prime directive](#identity--prime-directive)
- [Build / run / test](#build--run--test)
- [Code style](#code-style)
- [Soft imports + graceful degradation](#soft-imports--graceful-degradation)
- [Subprocess + SSH discipline](#subprocess--ssh-discipline)
- [Scoring + bench conventions](#scoring--bench-conventions)
- [Report style](#report-style)
- [Test discipline](#test-discipline)
- [CLI surface discipline](#cli-surface-discipline)
- [File creation](#file-creation)
- [Progress visibility](#progress-visibility)
- [Domain gotchas](#domain-gotchas)
- [When to ask (and when not to)](#when-to-ask-and-when-not-to)
- [Conventions](#conventions)

Editable, project-specific notes on how to develop MishaBench. Rewrite freely to keep the best version (not
append-only). The commit and safety policies live in `engineer.agent.md`.

**Shareable layer.** This file sits in `ai/` alongside `engineer.agent.md` and `spec.md` - the portable, shareable layer (it travels if the ai-pack is shared on its own).
Keep it free of environment-specific or sensitive details - host aliases, IPs, internal URLs, remote paths,
and secrets live in the private `ai/memory/` layer (`resources.md`, `credentials.md`, `info.md`) and are
referenced here, not hardcoded.

## Identity + prime directive

MishaBench is a small (~2 kLoC) no-nonsense CPU + GPU benchmark for typical data-analysis, CV, and LLM
workloads. It is **OSS, Apache-2.0, single-author** (Mikhail Yurasov): treat it as a polished public artifact
where every change is one the author would happily point a stranger at.

**Measurement honesty is the headline rule.** The whole value is honest, reproducible numbers; one misleading
score (wrong unit, broken sync, double-counted iteration) erodes the entire report. Treat correctness of
measurement as the primary invariant:

- **Sync before stop-clock.** Every CUDA / MPS bench calls `torch.cuda.synchronize()` /
  `torch.mps.synchronize()` immediately before the stop `perf_counter()` - async launch means un-synced
  wall-clock is not kernel time. (cudf ops are eager, so they need no manual sync.)
- **Warmup, then measure.** At least one un-timed warmup pass (allocator priming, autotune, JIT) before the
  timed window. Never time the first iter.
- **No microbenchmark in disguise.** Size workloads to run at least ~0.5s in the timed window; sub-100ms is noise.
- **Scores are geometric, not arithmetic.** Normalize at registration time so a bench on a wildly different
  scale cannot dominate the geomean - never patch this in `scoring.py`.
- **Power-source disclosure.** Surface every new power source in `system.py`; the report's "Power source" row
  must name it. Never fabricate wattage.

## Build / run / test

All commands run from `source/` (the repo root). The `./mishabench` bash wrapper is the single entry point. **Never bootstrap the venv manually** (no `pip`,
no `python -m venv`, no bare `python -m src`): the wrapper sets `PYTHONPATH=$HERE` and bootstraps the venv via
`uv sync` on first use. By design the project is **not pip-installable** (`pyproject.toml` -> `[tool.uv]
package = false`); `python -m src` is the canonical invocation. Running as a module (not an editable install)
is deliberate: cloud-synced filesystems sometimes mark setuptools' `.pth` shim as hidden, and this sidesteps
that entirely.

```bash
./mishabench install [--force] [--gpu]   # ensure venv + deps via uv sync --extra dev (idempotent)
./mishabench test    [args...]           # pytest
./mishabench lint    [args...]           # ruff check src/ tests/
./mishabench fmt                         # ruff check --fix + ruff format
./mishabench shell                       # subshell with venv + PYTHONPATH set
./mishabench clean                       # remove .venv + caches
./mishabench help                        # help text
```

Reserved dev-workflow verbs: `install / test / lint / fmt / shell / clean / help`. **Anything else is
forwarded to the Python CLI** (`python -m src <args>`). There is no `make`, `tox`, or `pre-commit`; to add a
workflow verb, add a reserved case to `./mishabench`, do not spawn a parallel tool.

User-facing CLI: `./mishabench {info,list,run,report}` (see `spec.md` for the full flag contract). Run
subset: `--suites data,cv,llm`; smoke: `--quick`; remote: `--remote <ssh-host>` (aliases in
`memory/resources.md`); GPU extras on a remote: add `--gpu`.

- **`./mishabench install --gpu` is sticky.** It writes `.venv/.mishabench-gpu`; later installs auto-include
  the RAPIDS (cudf-cu12 + cupy-cuda12x) side-install. Remove that file to drop back to a CPU-only venv. RAPIDS
  is installed out-of-band (not in `pyproject.toml`) because its cu12 stubs fail metadata resolution on macOS.
- **`./mishabench lint` and `./mishabench test` must pass clean before any commit.**

## Code style

- **Python 3.10+ only.** Use `X | None` / `X | Y`, not `typing.Optional` / `Union`. Put
  `from __future__ import annotations` at the top of every module with type hints.
- **Type hints on public functions + dataclasses** (and internal helpers unless truly noisy).
- **Dataclasses over dicts** for any structured value crossing a module boundary (`BenchConfig`,
  `BenchResult`, `SystemInfo`, `PowerWindow`, `DeviceScore` are the canonical examples).
- **Small modules, one responsibility, ~600 lines max.**
- **No new third-party deps without explicit user approval** - the runtime dep set is a contract with the
  user's first-install cost.
- **Ruff lint set is `E F W I B UP SIM`; `E501` is intentionally off** (line-length 100). `src/cli.py` keeps
  `B008` ignored for Typer's `Option(...)`-in-default idiom.
- **Comments explain why, not what.** Skip narration comments; the maintainer is opinionated about this.

## Soft imports + graceful degradation

Optional deps (`cudf`, `cupy`, `cuda` runtime, `torch.backends.mps`) are detected at probe time and surfaced
as capability flags on `SystemInfo`. Benches declare requirements via `Bench.requires=("cuda",)`; the runner
skips them with a clear reason when a capability is missing. **Never import an optional dep at module import
time** - defer it inside the bench function so importing the module on a CPU-only laptop never crashes.

## Subprocess + SSH discipline

All local + remote execution goes through the primitives in `src/_run.py` - do not call raw `subprocess.run`
from new call-sites. If a new primitive is needed, add it to `_run.py`.

- `run(cmd, ...)` - local subprocess, captures both streams.
- `ssh_run` / `ssh_one` - captured remote command(s); use for short structured queries.
- `ssh_run_stream` - streams stdout/stderr live; use for `uv sync`, package downloads, anything > 5s. **Must
  use `-tt` (force PTY)**: without it, modern Rust CLIs (uv) switch to non-interactive mode, block-buffer, and
  desync stdout/stderr. Pass the script base64-encoded as a remote command, not via stdin.
- `rsync_to` / `rsync_from` - streaming by design; return an int rc. Exclude `.venv`, `.git`, caches,
  `results/`, secrets; **no `--delete`** by default.

## Scoring + bench conventions

- **`SCORE_SCALE = 1000.0` is fixed across releases** (in `scoring.py`). Changing it invalidates every prior
  run's comparison - needs explicit user approval.
- **Bench IDs follow `<suite>.<short>.<device>`** (e.g. `cv.resnet50.cuda`, `data.gb.polars`); short names
  group in the report.
- **Devices are exactly `cpu` / `cuda` / `mps`** - no `cuda:0` distinctions in IDs (runner uses device 0).
- **`value` MUST be throughput (work / second), never workload size.** Capture work-size before the timed
  window, time the actual operation, return `work_size / elapsed` with a per-second unit (`MiB/s`, `M rows/s`,
  `img/s`, `tok/s`, `GFLOPS`). Setup (`pl.from_pandas`, `cudf.from_pandas`) is excluded from the timed window.

## Report style

- **Self-contained HTML.** No JS, no external CSS / fonts / image URLs - it must open offline anywhere.
- **Inline SVG bars** (div rows with width-percentage bars). Plotly / matplotlib are banned (heavy; unreliable
  offline).
- **Honest visualizations.** Bars are absolute within a single chart, not normalized cross-bench; cross-host
  comparison is via the Scores table.
- **NVIDIA brand-green only as an accent.** Technical report: numbers first, prose second.

## Test discipline

- **Behavior-changing PRs always add or update tests. No exceptions.**
- **Use Typer's `CliRunner`** for CLI-surface tests (`tests/test_smoke.py` is the existing pattern). Don't
  depend on a live GPU host or a real network model in the default suite.
- **Edge cases over happy paths** - missing `nvidia-smi`, RAPL with restricted perms, malformed `--label`,
  `$workdir` with spaces, empty `nvidia-smi` output during driver init.
- Test files mirror source files (`tests/test_<module>.py`).

## CLI surface discipline

The Typer app structure is fixed: top-level commands `info`, `list`, `run`, `report`; wrapper-only verbs
`install / test / lint / fmt / shell / clean / help`. **Adding a top-level command is a contract change - ask
first.** Renaming or removing one needs a deprecation note in the same change.

## File creation

- **Every new `.py` file gets the SPDX header** already on every existing file (same `#`-comment header on the
  bash wrapper):

  ```python
  # SPDX-FileCopyrightText: Copyright (c) <year> Mikhail Yurasov
  # SPDX-License-Identifier: Apache-2.0
  ```

- **No new files without a clear home.** If it belongs in a module that does not exist yet, propose the module
  first and wait for approval.

## Progress visibility

- **Never `--quiet` an install or test step.** The user wants uv's "Downloaded torch (700 MiB)" and pytest's
  per-test names; silent runs feel hung. The wrapper enforces no-quiet `uv sync`; pyproject's `addopts =
  "-ra -v"` enforces verbose pytest.
- **For SSH-driven remote work, stream don't capture** (`ssh_run_stream` with `-tt`; streaming rsync).
- **Per-bench `[N/M]` + a `running...` start line is mandatory**, printed before the bench fn runs, so slow
  CV / LLM benches never look hung.

## Domain gotchas

- **Torch CUDA wheel auto-fix.** PyPI's default torch is cu130 (needs driver >= 545); many hosts run older.
  When `torch.cuda.is_available()` is False but `nvidia-smi` sees a GPU, install paths reinstall torch from a
  driver-compatible index: driver >= 545 -> default (cu130), 525-544 -> cu126, 470-524 -> cu118, < 470 ->
  CUDA benches skip. `MISHABENCH_TORCH_CUDA` (`auto` | `cu118`/`cu121`/`cu126`/`cu128`/`cu130` | `cpu`)
  overrides. Don't pin a torch version in `pyproject.toml` - it loses the auto-fix.
- **Linux RAPL is read-only, probed once.** `/sys/class/powercap/intel-rapl:N/energy_uj` is mode 0400 by
  default since CVE-2020-8694; the probe surfaces `rapl_status="permission_denied"` plus a one-line
  `sudo chmod a+r ...` hint in the report. We never auto-chmod and never retry mid-run. Recent AMD exposes the
  same `intel-rapl:N` path. Negative deltas are counter wraparound and are dropped.
- **Apple Silicon power is always estimated** from the chip-name TDP table in `power.py`; marked with a star.
  Never shell out to `powermetrics` (needs sudo).
- **MPS quirks.** torch.mps does not always release memory between models - the runner calls
  `torch.mps.empty_cache()` in post-bench cleanup. `torch.float16` on MPS is finicky for some HF models
  (TinyLlama works); validate fp16-on-MPS before swapping the LLM.
- **`opencv-python-headless` ships no CUDA kernels** - the CV image-resize bench is CPU-only by design.
- **First-run model downloads count against the budget** (~2.5 GiB: TinyLlama 2.2G, MiniLM 80M, DINOv2 85M,
  ResNet-50 100M, EfficientNet-B0 20M). Quick-mode shrinks workloads, not model size.
- **`nvidia-smi` returns `[N/A]`** during driver init / odd power states; the PowerMonitor skips those
  samples rather than counting them as 0 W.
- **Output dirs:** local runs land in `./results/<runid>/`; remote runs in `./results/<host>-<runid>/`. The
  three stable artifacts are `results.jsonl`, `system.json`, `report.html`; anything else is debug.

## When to ask (and when not to)

Ask first: new runtime/dev dependency; new top-level CLI command or sub-app; removing/renaming a command or
flag (public surface is a contract); changing `SCORE_SCALE`; changing the default LLM/embed/CV model (the user
has runs to compare against); a power source that needs sudo; removing or weakening a test.

Just do it: rename a private helper; add internal type hints; split a function for readability (no behavior
change); tighten a test / add edge-case coverage; fix a lint warning; update a docstring.

## Conventions

- Default working style (carried by Solaris): terse responses; tables when comparing options; lead with an
  explicit recommendation; give the bare command first, then variants.
- **Entry-point shims are thin forwarders, not content.** `AGENTS.md` (the [agents.md](https://agents.md/)
  convention, read by Cursor / Codex / Copilot) imports `ai/engineer.agent.md`; `CLAUDE.md` imports
  `AGENTS.md`. Canonical rules live in `ai/engineer.agent.md` + this file + `ai/spec.md` - keep the shims one
  line and never duplicate rules into them.
- Concrete remote host aliases, the GitHub remote, and the import provenance live in `ai/memory/`
  (`resources.md`, `info.md`).
