# MishaBench - spec <!-- omit in toc -->

- [Goal](#goal)
- [Components](#components)
  - [Suites and benchmarks](#suites-and-benchmarks)
  - [Scoring](#scoring)
  - [Power monitoring](#power-monitoring)
  - [Output schema](#output-schema)
  - [Report layout](#report-layout)
  - [Remote runs](#remote-runs)
  - [CLI contract](#cli-contract)
- [Constraints](#constraints)
- [Open questions](#open-questions)

MishaBench is a cross-platform CPU + GPU benchmark for the workloads people actually run: typical
data-analysis, computer-vision, and LLM tasks. It runs locally or on a remote SSH host (`--remote <host>`),
completes in under an hour (or ~5 min in `--quick` mode), and emits a single self-contained HTML report.

> Current specification - the single source of truth for this project. Kept **self-sufficient**: it reads
> standalone and does not reference or depend on any other file. Updated through dialogue during planning and
> development. (README.md is explanatory; this file is normative.)

## Goal

Produce honest, reproducible CPU and GPU benchmark scores for typical data-analysis, computer-vision, and LLM
tasks, in under an hour, with a self-contained HTML report shareable as a single file. Three dimensions of
"honest":

- **Measurement honesty:** cuda/mps sync before `perf_counter`, warmup before measurement, geomean (not
  arithmetic) for composites, no fabricated wattage.
- **Disclosure:** the report states where every wattage figure came from (nvidia-smi, RAPL, Apple TDP
  estimate); estimates are marked with a star.
- **Reproducibility:** synthetic inputs (no dataset dependencies; only model weights, auto-cached), a fixed
  score-scale constant across releases, one offline-openable HTML report.

## Components

### Suites and benchmarks

Three suites, all default-on (`--suites data,cv,llm` is the explicit form). Variants per device where
applicable; unavailable variants are skipped with a clear reason.

- **`data`** (pandas / polars cpu; cudf cuda where installed): `csv_parse` (5M-row synthetic CSV, MiB/s),
  `group_aggregate` (5M rows, 1024 groups, multi-agg, M rows/s), `join` (5M left-join 500k dim, M rows/s),
  `parquet_round` (snappy write+read of 5M rows, MiB/s round-trip), `string_ops` (regex extract + uppercase
  over 1M strings, M rows/s). Quick-mode: 10x smaller.
- **`cv`** (cpu / cuda / mps for model paths): `resnet50` and `efficientnet_b0` forward-pass throughput
  (img/s; batch 4 cpu / 32 cuda / 16 mps), `conv2d_micro` (64->64 ch, 3x3, GFLOPS), `image_resize` (OpenCV
  1920x1080 -> 224x224, img/s, CPU-only), `dinov2_features` (facebook/dinov2-small, img/s). Quick-mode: 4x
  fewer iters, same shapes.
- **`llm`** (cpu / cuda / mps): `tokenize` (HF fast tokenizer, tok/s, CPU-only), `embed_minilm`
  (all-MiniLM-L6-v2, sentences/s), `prefill` (TinyLlama-1.1B seq=512, prefill tok/s), `decode` (TinyLlama
  decode 128 new tokens, decode tok/s). Default models overridable via `MISHABENCH_LLM_MODEL` /
  `MISHABENCH_EMBED_MODEL`. dtype: CPU fp32, CUDA fp16, MPS fp16. Quick-mode: shorter seqs, fewer iters.

Every bench captures work-size before the timed window and returns `work_size / elapsed` (throughput), not
workload size. Frame setup (`pl.from_pandas`, `cudf.from_pandas`) is excluded from the timed window.

### Scoring

- Per-bench raw value as reported (img/s, MiB/s, GFLOPS, tok/s, ...).
- Per `(suite, device)` score: geometric mean of the bucket's raw values, times `SCORE_SCALE = 1000.0`.
  Geomean drops non-positive values. **`SCORE_SCALE` is fixed across releases** (changing it invalidates
  cross-release comparison).
- Per-device total: geomean of that device's per-suite scores.
- Points-per-watt: `score / avg_watts`; the `estimated_power` flag propagates so the report can star
  estimate-derived cells.

### Power monitoring

- **NVIDIA:** `nvidia-smi --query-gpu=power.draw` polled at 2 Hz; GPU index 0 attributed to the bench, peak
  across GPUs tracked for the System section. `[N/A]` samples are skipped, not counted as 0 W.
- **Intel/AMD on Linux:** `/sys/class/powercap/intel-rapl:0/energy_uj` sampled at bench start/end;
  `delta_J / wall_s = avg W`. Mode 0400 by default (CVE-2020-8694) -> `rapl_status="permission_denied"` plus a
  one-time `sudo chmod a+r ...` hint in the report. Negative deltas (wraparound) dropped. No auto-chmod, no
  mid-run retry.
- **Apple Silicon:** chip-name TDP table in `power.APPLE_TDP_W`; marked `estimated=True` (star). No
  `powermetrics` (needs sudo).
- **None available:** field is `None`; report shows "-".

### Output schema

```
results/<runid>/
  results.jsonl   one JSON per line: {id, suite, name, device, metric, value, unit, seconds, iters, ok,
                  error, notes, started_at}; notes carries power {duration_s, cpu_avg_w, cpu_energy_j,
                  cpu_estimated, gpu_avg_w, gpu_energy_j, gpu_peak_w, n_samples}, avg_watts, power_estimated.
  system.json     SystemInfo dump (cpu, ram, gpus, libs, apple_tdp_w, has_rapl/cuda/mps/cudf/cupy).
  report.html     self-contained: inline CSS + inline SVG bars, no JS, no external assets.
```

### Report layout

Top to bottom: Header (project, hostname, timestamp, total wall time, label tags); Scores (device columns
CPU/CUDA/MPS, suite rows + Total; each cell score / avg W / pts/W; star = estimated); System (host, OS, CPU,
RAM, disk, python, GPUs, CUDA/driver, power-source disclosure, library inventory); per-suite bar charts
(absolute within a chart, not normalized); All-measurements table.

### Remote runs

`--remote <ssh-host>` (alias from `~/.ssh/config`, passwordless ssh required): ssh mkdir; rsync source to
`$HOME/mishabench/mishabench-tool/`; ensure `uv` (auto-install if missing); `uv sync --extra dev [--extra
gpu]`; `uv run python -m src run ...` streamed live; rsync results back to `./results/<host>-<runid>/`.
rsync excludes `.venv`, caches, `*.egg-info`, `dist`, `build`, `uv.lock` (remote resolves its own), `.git`,
`results/`, `.mishabench-cache/`, `*.parquet`.

### CLI contract

```
./mishabench info                       system probe one-liner + power-source disclosure
./mishabench list [--suites SUITES]     enumerate benches with expected runtimes
./mishabench run [opts]                 run + write report
    --suites data,cv,llm     comma list, default all      --no-cuda / --no-mps   skip device
    --quick / -q             5-min budget + 10x workloads  --output / -o DIR     output dir (default results)
    --label / -l KEY=VALUE   repeatable; appears in header --remote / -r HOST    ssh alias; rsync + run + fetch
    --gpu                    (with --remote) install gpu extra remotely
./mishabench report DIR                 regenerate report.html from a results dir
```

Wrapper-only verbs (handled by `./mishabench`, not forwarded): `install / test / lint / fmt / shell / clean /
help`.

## Constraints

- **Stack:** Python 3.10+, uv-managed, Typer CLI. Not pip-installable (`[tool.uv] package = false`); invoked
  as `python -m src` under `PYTHONPATH=<root>` via the `./mishabench` wrapper.
- **Deps:** runtime = typer, rich, pyyaml, numpy, pandas, polars, pyarrow, pillow, opencv-python-headless,
  torch, torchvision, transformers, sentence-transformers, tiktoken, huggingface-hub, psutil, scipy,
  threadpoolctl. Optional `gpu` extra (Linux x86_64): cudf-cu12, cupy-cuda12x (installed out-of-band by the
  wrapper, not in pyproject metadata). Dev = pytest, ruff. No new deps without approval.
- **Budgets:** default 60-min hard cap (`total_budget_s = 3600`); quick 5-min (`QUICK_BUDGET_S = 300`,
  `QUICK_SCALE = 0.1`); per-bench 5-min guard. The runner skips a bench whose expected cost would exceed the
  budget (recorded `ok=false, error="budget: ..."`).
- **Torch CUDA wheel selection:** auto-pick a driver-compatible wheel when the default cu130 cannot see the
  GPU (>= 545 default, 525-544 cu126, 470-524 cu118, < 470 skip); `MISHABENCH_TORCH_CUDA` overrides.
- **Non-goals:** timing-only, not accuracy (inputs are random tensors). A failing bench is recorded
  (`ok=false`) and does not abort the run. No multi-GPU id distinctions yet (device 0). No `powermetrics`.

## Open questions

- Per-precision FP pipe split (FP16/FP32/FP64) once a host exposes those metrics.
- Hard per-bench timeout (today the per-bench budget is cooperative via shape sizing).
- Wider AMD RAPL coverage (k10temp `hwmon` energy paths) for hosts where `intel-rapl:N` is absent.
- Additional power sources (only if sudo-free).
