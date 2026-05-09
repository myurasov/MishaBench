# dev.memory -- MishaBench

Accumulated maintainer preferences. Each entry is a hard rule unless explicitly overridden in the current turn.

## Workflow shortcuts

- **`./mishabench` does everything.** Never invoke `pip`, `python -m venv`, or call `python -m src` directly outside the wrapper -- the wrapper sets `PYTHONPATH` correctly and bootstraps the venv if needed.
- **`./mishabench install --gpu` is sticky.** It writes `.venv/.mishabench-gpu`; subsequent `./mishabench install` re-runs include the gpu extra automatically. Remove that file to drop back to a CPU-only venv.
- **Quick mode is the smoke pass.** `./mishabench run --quick` is the 5-minute version: 10x smaller workloads, total budget capped at 5 min. Use it for "did my change still work?" loops.

## Conventions

- **Score scale is `1000.0`, fixed across releases.** The constant is in `scoring.py`. Don't change it without explicit user approval -- it would invalidate every prior run's comparison.
- **Bench IDs follow `<suite>.<short>.<device>`.** E.g. `cv.resnet50.cuda`, `data.gb.polars`. Short names get reused for grouping in the report.
- **Devices are exactly `cpu` / `cuda` / `mps`.** No `cuda:0` / `cuda:1` distinctions in IDs (the runner picks device 0 by default). Multi-GPU is a future feature.
- **Apple Silicon power is always estimated.** Never shell out to `powermetrics` -- it requires sudo at run time, which the user shouldn't have to type to run a benchmark. The chip-name TDP table in `power.py` is the authoritative source on macOS.
- **Linux RAPL is read-only and probed once.** If the file isn't readable at probe time, CPU power is "n/a" for the whole run. Don't retry mid-run -- permissions don't change at runtime.

## Gotchas

- **First-run model downloads count against the budget.** TinyLlama (~2.2 GiB), MiniLM (~80 MiB), DINOv2 (~85 MiB), ResNet-50 (~100 MiB), EfficientNet-B0 (~20 MiB). On a slow link this can eat 10-15 minutes of the 60-min budget. Quick-mode runs the same downloads but uses smaller workloads -- the model size is the model size.
- **MPS quirks**: torch.mps doesn't always release memory between models. The runner calls `torch.mps.empty_cache()` (when available) in the post-bench cleanup; if you add a new MPS bench and see allocator OOMs across benches, that's where to look.
- **`opencv-python-headless` doesn't have CUDA support** in the wheels we install. The CV image-resize bench is CPU-only by design -- if you want CUDA OpenCV, that's a `cv2` build dance with the system CUDA toolkit, not a wheel install.
- **`torch.float16` on MPS is finicky for some HF models.** TinyLlama works; some other LLaMA variants don't. If you swap the LLM, validate it loads in fp16 on MPS first.
- **`uv sync` on the remote** uses whatever Python version it picks (3.10+). The bench's lockfile (if present) is excluded from rsync so the remote resolves freshly. This is intentional -- different host architectures need different torch wheels.
- **`nvidia-smi` returns `[N/A]`** when the GPU has just powered up or is in a weird power state. The PowerMonitor parses these lines as "skip this sample" rather than treating them as 0 W; without that, the avg watts would underestimate.

## Filing

- **Bench definitions live one per file under `src/suites/<name>.py`.** A new suite means a new file + adding the suite name to `config.ALL_SUITES`.
- **Per-suite README sections in the top-level README** stay synced with what's actually registered. If you add a benchmark, add a row to the table.
- **Test files mirror source files** (`tests/test_<module>.py`).

## Output

- **Default output dir is `./results/<runid>/`** for local runs.
- **Remote runs land in `./results/<host>-<runid>/`** so multiple remotes don't collide.
- **`results.jsonl`, `system.json`, and `report.html`** are the three artifacts. Anything else in the dir is debug; don't promise stability for it.

## Communication style

- **One- or two-sentence commit messages.** Subject + (when needed) one paragraph of *why*.
- **No emoji in code, comments, commit messages, or docs.** ASCII only.
- **No "we should" / "we could" / "TODO maybe" comments.** If it's worth doing, file an issue. If it's not, delete the comment.

## Progress visibility (always-on)

- **Never `--quiet` / `-q` an install or test step.** The user wants to see uv's "Downloaded torch (700 MiB)" and pytest's per-test names; silent runs feel hung. The wrapper enforces this for `uv sync` + `uv pip install`; pyproject's `addopts = "-ra -v"` enforces it for pytest.
- **For SSH-driven remote work, stream don't capture.** `ssh_run` captures (use it for short structured queries). `ssh_run_stream` lets stdout/stderr flow through to the user's terminal (use it for `uv sync`, package downloads, anything > 5 sec). Same rule for rsync: `rsync_to` / `rsync_from` are streaming by design and return an int rc, not a RunResult.
- **`ssh_run_stream` MUST use `-tt` (force PTY).** Without a PTY: (a) modern Rust CLIs like `uv` call `isatty()` and silently switch to non-interactive mode, sometimes emitting nothing at all; (b) pipe-mode stdio defaults to block-buffering, so even the lines that ARE emitted appear in chunks instead of streaming; (c) stdout and stderr arrive un-synchronized, so `echo done` prints before `uv`'s "Installed N packages" line. The `-tt` PTY fixes all three. Pass the script as a base64-encoded remote command (not via stdin) -- with a PTY, stdin echo would race with command interpretation.
- **Per-bench `[N/M]` + start line is mandatory.** Slow benches (CV inference, LLM decode) take 30 s+ on CPU; without the start line printed *before* the bench fn runs, the user can't tell the suite hasn't hung.
