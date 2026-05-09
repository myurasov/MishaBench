# AGENTS.md -- MishaBench

This file is the **universal entry-point for AI-enabled IDEs** (Cursor, Claude Code, OpenAI Codex, GitHub Copilot, and any other tool that respects the [`AGENTS.md`](https://agents.md/) convention). Read it on every turn; it is intentionally short and points at the canonical sources.

## What MishaBench is

A small (~2 kLoC) Python CLI that runs CPU + GPU benchmarks across three suites (`data`, `cv`, `llm`), measures power draw where possible, and produces a self-contained HTML report. Designed to complete in under an hour (or under 5 minutes in `--quick` mode), runnable locally or on a remote SSH host via `--remote <host>`.

Apache 2.0 licensed. User-facing surface:

```bash
# bring-up
./mishabench install [--force] [--gpu]

# usage
./mishabench info
./mishabench list
./mishabench run [--quick] [--suites data,cv,llm] [--no-cuda] [--no-mps]
                 [--label k=v ...] [--remote <ssh-host>] [--gpu]
./mishabench report <results-dir>

# dev workflow (handled by the wrapper itself, not forwarded)
./mishabench install | test | lint | fmt | shell | clean | help
```

## Read in order, on every turn

1. **[`ai/dev.agent.md`](ai/dev.agent.md)** -- the actual rules: who you are, how the maintainer wants the project built, the commit and test discipline, when to ask vs. just-do-it. **This is your primary instruction file.**
2. **[`ai/dev.memory.md`](ai/dev.memory.md)** -- accumulated maintainer preferences (workflow shortcuts, gotchas, conventions). Treat each entry as a hard rule unless overridden in the current turn.
3. **[`ai/spec.txt`](ai/spec.txt)** -- canonical specification of what MishaBench measures, the scoring rules, the power-source policy, and the report layout. Consult before adding, changing, or removing behavior.

## Bootstrap and run

Use the `./mishabench` script for everything. Never bootstrap the venv manually:

```bash
./mishabench install         # ensure venv + deps via uv sync --extra dev
./mishabench install --gpu   # also pull cudf-cu12 + cupy on Linux
./mishabench test            # pytest
./mishabench lint            # ruff check
./mishabench fmt             # ruff check --fix + ruff format
./mishabench <anything else> # forwarded to python -m src
```

Reserved dev-workflow names: `install / test / lint / fmt / shell / clean / help`. Anything else is forwarded to the bench Python CLI as-is. The wrapper sets `PYTHONPATH=$HERE` (the project root) and invokes `python -m src`, sidestepping editable installs entirely -- cloud-synced filesystems sometimes mark setuptools' `.pth` shim as hidden, and we want zero exposure to that.

## IDE-specific notes

- **Cursor** picks up `AGENTS.md` automatically (and any `.cursor/rules/*.mdc` files, none of which are present here).
- **Claude Code** picks up `CLAUDE.md` if present; absent that, it reads `AGENTS.md`. MishaBench ships only this file -- both work.
- **OpenAI Codex / Codex CLI** reads `AGENTS.md` per the published spec.
- **GitHub Copilot** reads `.github/copilot-instructions.md` if present; for MishaBench the canonical instructions live here, so link or import this file when configuring Copilot for the repo.

If you add another IDE-specific shim later, keep it as a thin forwarder to `ai/dev.agent.md` rather than duplicating content.
