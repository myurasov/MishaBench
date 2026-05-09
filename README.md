# MishaBench

A cross-platform CPU + GPU benchmark for the workloads people actually run: typical **data analysis**, **computer vision**, and **LLM** tasks. Designed to complete in under an hour on a modest CUDA-capable host (or about 5 minutes in `--quick` mode), produce a single self-contained HTML report, and be drivable locally **or** on a remote SSH host with one flag.

Copyright (c) 2026 Mikhail Yurasov. Licensed under the [Apache License 2.0](LICENSE).

## TL;DR

```bash
# install local venv + deps
./mishabench install

# everything, ~45 min on a CUDA box, ~30 min on Apple Silicon
./mishabench run

# 5-minute smoke run (10x smaller workloads, same suites)
./mishabench run --quick

# run it on a remote NVIDIA box (alias from ~/.ssh/config); results
# get rsync'd back to ./results/<host>-<runid>/
./mishabench run --remote mlbox-local --quick

# see system info before running
./mishabench info

# rebuild the HTML from a results directory (e.g. after a partial run)
./mishabench report results/2026-05-09-153022
```

The report shows **CPU and GPU scores as absolute numbers** per suite plus an overall total, and an **estimated points-per-watt** for power-efficiency comparison. NVIDIA wattage and Linux RAPL CPU wattage are real measurements; Apple Silicon power is a chip-name TDP estimate (the OS does not expose per-process power without sudo).

## What it measures

Three suites, each with CPU and -- where applicable -- CUDA and MPS variants. All workloads are synthetic so there's no dataset to download (only model weights for the CV / LLM suites, which auto-cache after the first run).

| Suite | Benchmarks |
|---|---|
| `data` | CSV parse, group-by aggregate, left join, parquet round-trip, regex + uppercase. Compares **pandas** vs **polars** on CPU and **cudf** on CUDA (when installed). |
| `cv`   | ResNet-50 inference, EfficientNet-B0 inference, Conv2D microbenchmark, image resize (OpenCV), DINOv2-small feature extraction. CPU / CUDA / MPS for the model paths. |
| `llm`  | HF tokenizer encode, MiniLM sentence-embedding throughput, TinyLlama-1.1B prefill, TinyLlama-1.1B autoregressive decode. CPU / CUDA / MPS. |

`./mishabench list` enumerates every registered benchmark with its expected runtime.

## Scoring + power efficiency

Each `(suite, device)` cell in the Scores section shows three numbers:

- **Score** -- geometric mean of the per-benchmark raw values, scaled by 1000. Higher is better. Geomean is used (not arithmetic mean) so a single bench at a different scale can't dominate the result.
- **Watts** -- average power draw measured during the benches that ran on that device.
- **pts/W** -- score / watts. The headline power-efficiency number.

Power source per platform:

| Platform | CPU | GPU | Status |
|---|---|---|---|
| Linux + Intel/AMD + NVIDIA | Intel RAPL via `/sys/class/powercap/intel-rapl:0/energy_uj` | `nvidia-smi --query-gpu=power.draw` polled at 2 Hz | **Real** |
| macOS + Apple Silicon | chip-name TDP table (M1=15 W ... M4 Max=90 W) | -- | **Estimated** (marked with a star in the report) |
| Anything else | n/a | n/a | "n/a" in the report rather than a fabricated number |

Sudo-free per-process power is not available on macOS, so we don't shell out to `powermetrics`. The chip-name table is a sustained-package TDP, so pts/W on Apple Silicon is a reasonable upper bound for a steady-state workload, but won't match a wattmeter at the wall.

## Install

```bash
# macOS / Linux laptop
brew install uv          # or: curl -LsSf https://astral.sh/uv/install.sh | sh

# clone + bootstrap
git clone https://github.com/myurasov/MishaBench.git mishabench
cd mishabench
./mishabench install

# Linux + NVIDIA + RAPIDS (cudf, cupy) -- optional, large
./mishabench install --gpu
```

Python 3.10+. `pip` is not used directly; everything goes through `uv`. The `./mishabench` wrapper auto-bootstraps the venv on every subcommand, so step 3 above is optional -- the first `./mishabench run` would do it anyway.

### Dependencies

Runtime: `typer`, `rich`, `pyyaml`, `numpy`, `pandas`, `polars`, `pyarrow`, `pillow`, `opencv-python-headless`, `torch`, `torchvision`, `transformers`, `sentence-transformers`, `tiktoken`, `huggingface-hub`, `psutil`.

Optional `--gpu` extra (Linux x86_64 only): `cudf-cu12`, `cupy-cuda12x`. The `data` suite's GPU paths run only when `cudf` is installed; everywhere else they're recorded as skipped with a clear reason.

Dev: `pytest`, `ruff`.

## Common runs

```bash
# Everything, default budget (60 min hard cap)
./mishabench run

# 5-minute smoke pass over every suite
./mishabench run --quick

# Just the LLM suite
./mishabench run --suites llm

# Just data + cv, force CPU only (compare two boxes' CPU baselines fairly)
./mishabench run --suites data,cv --no-cuda

# Tag this run with a label that ends up in the report header
./mishabench run --label "scenario=cooled-perf-mode" --label "host=desk-spark"
```

## Remote runs

`--remote <ssh-host>` rsyncs the project to `~/mishabench/mishabench-tool/` on the target, ensures `uv` is installed there, syncs the venv, runs the bench, and pulls the results back to `./results/<host>-<runid>/`.

```bash
# 5-min smoke on a remote NVIDIA box (alias from ~/.ssh/config)
./mishabench run --remote mlbox-local --quick

# Full run with the gpu extra (cudf + cupy) on the remote box
./mishabench run --remote mlbox-local --gpu

# Dell Spark from another machine
./mishabench run --remote myspark1-local --suites data,cv --quick
```

The remote driver only requires `ssh <host>` to work passwordlessly. uv installs itself if missing. The remote venv lives at `~/mishabench/mishabench-tool/.venv/` and is preserved across runs.

To open the report after a remote run: `open results/<host>-<runid>/report.html` on macOS, or `xdg-open` on Linux.

## Output

```
results/<runid>/
  results.jsonl   -- one line per benchmark, full BenchResult dataclass
  system.json     -- system probe at run-time (cpu / ram / gpus / libs)
  report.html     -- self-contained, no external assets, share-friendly
```

Re-render the report from a results directory at any time:

```bash
./mishabench report results/<runid>
```

## Dev workflow

```bash
./mishabench install          # ensure venv + deps
./mishabench test             # pytest
./mishabench lint             # ruff check
./mishabench fmt              # ruff check --fix + ruff format
./mishabench shell            # subshell with venv + PYTHONPATH set
./mishabench clean            # rm .venv + caches
```

## Project layout

```
mishabench/
  mishabench               -- bash wrapper (./mishabench install / test / lint / fmt / clean / *)
  pyproject.toml           -- uv-managed; declares optional `gpu` extra
  src/
    __init__.py
    __main__.py            -- python -m src
    cli.py                 -- typer app (info, list, run, report)
    _run.py                -- subprocess + ssh + rsync primitives
    config.py              -- BenchConfig, suite enum, --quick budget rule
    system.py              -- system probe (cpu, ram, gpu, cuda, libs, power source)
    power.py               -- PowerMonitor: nvidia-smi + RAPL + Apple TDP table
    runner.py              -- registry, timing harness, jsonl writer
    scoring.py             -- geomean -> per-(suite,device) score + pts/W
    report.py              -- self-contained HTML renderer (inline CSS + SVG)
    remote.py              -- --remote driver
    suites/
      data.py              -- pandas / polars / cudf benchmarks
      cv.py                -- torchvision + DINOv2 + OpenCV benchmarks
      llm.py               -- TinyLlama + MiniLM + tokenizer benchmarks
  tests/
    test_smoke.py
  ai/
    dev.agent.md           -- maintainer agent rules
    dev.memory.md          -- maintainer accumulated preferences
    spec.txt               -- canonical spec for what the bench measures
```

## Notes

- **First run downloads model weights** (~2.5 GiB total: TinyLlama 2.2 GiB + MiniLM 80 MiB + DINOv2 85 MiB + ResNet50 100 MiB + EfficientNet 20 MiB). The download counts against the wall-clock budget; the second run skips it.
- The bench is **timing-only**, not accuracy-only. Inputs are random tensors; we measure throughput / latency, not whether the model produces correct outputs.
- The report is **self-contained** -- inline CSS, inline SVG bars, zero JS, no external fonts or images. It opens in any browser without internet.
- A failing benchmark **does not abort** the rest of the run; it's recorded with `ok=false` and a short error message in the JSONL + report.
