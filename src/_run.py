# SPDX-FileCopyrightText: Copyright (c) 2026 Mikhail Yurasov
# SPDX-License-Identifier: Apache-2.0

"""Process / SSH primitives shared by the local runner and the --remote driver."""

from __future__ import annotations

import base64
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class RunResult:
    rc: int
    stdout: str
    stderr: str

    @property
    def ok(self) -> bool:
        return self.rc == 0


def run(cmd: list[str] | str, *, cwd: str | Path | None = None,
        check: bool = False, env: dict[str, str] | None = None) -> RunResult:
    """Local subprocess. Always captures both streams."""
    if isinstance(cmd, str):
        shell_cmd: list[str] | str = cmd
        shell = True
    else:
        shell_cmd = cmd
        shell = False
    proc = subprocess.run(
        shell_cmd,
        cwd=str(cwd) if cwd else None,
        shell=shell,
        capture_output=True,
        text=True,
        env=env,
    )
    if check and proc.returncode != 0:
        raise subprocess.CalledProcessError(
            proc.returncode, shell_cmd, output=proc.stdout, stderr=proc.stderr,
        )
    return RunResult(proc.returncode, proc.stdout, proc.stderr)


def ssh_run(host: str, script: str, *, env: dict[str, str] | None = None,
            check: bool = False) -> RunResult:
    """Run a multi-line bash script on `host` over SSH via stdin heredoc.

    Captures stdout / stderr. Use this for short, structured queries
    where the caller wants the output as a string (a status check, a
    metadata lookup). For long-running steps where the user wants live
    progress (uv sync over slow link, multi-GiB downloads), use
    `ssh_run_stream` instead.

    Passing the script over stdin is the robust form -- inline
    single-quoted SSH commands break on nested quotes or `$( )`.
    """
    env_prelude = ""
    if env:
        for k, v in env.items():
            env_prelude += f"export {k}={shlex.quote(v)}\n"
    full = "set -e\n" + env_prelude + script
    proc = subprocess.run(
        ["ssh", host, "bash", "-s"],
        input=full,
        capture_output=True,
        text=True,
    )
    if check and proc.returncode != 0:
        raise subprocess.CalledProcessError(
            proc.returncode, ["ssh", host, "bash", "-s"],
            output=proc.stdout, stderr=proc.stderr,
        )
    return RunResult(proc.returncode, proc.stdout, proc.stderr)


def ssh_run_stream(host: str, script: str, *,
                   env: dict[str, str] | None = None) -> int:
    """Like ssh_run, but stream stdout / stderr to the user's terminal in
    real time over a forced PTY. Returns the SSH exit code.

    Use for steps where progress visibility matters (uv sync, package
    downloads, anything that prints periodic status to stderr). The
    caller doesn't get the captured output -- the user sees it directly.

    Why `ssh -tt`:

      - `uv` (and most modern Rust CLIs) call `isatty(stdout)` and switch
        to non-interactive mode -- no progress bars, sometimes no status
        output at all -- when they think they're piped. Without a PTY
        the user just sees blank silence while gigabytes download.
      - Pipe-mode stdio defaults to block-buffering on both ends; even
        the lines uv *does* emit get stuck in a 4 KiB buffer and
        appear in chunks (or never, on a fast-finishing command).
      - `-tt` forces PTY allocation regardless of whether the local
        stdin is a TTY, so the remote process sees an interactive
        terminal and behaves as if the user were sitting at the host.

    The script is base64-encoded and passed as a remote-command argument
    rather than via stdin -- with `-tt`, stdin would be the PTY itself
    and feeding the script through it races with the PTY's own input
    echo (the script bytes get echoed back to the user's terminal
    before bash interprets them).
    """
    env_prelude = ""
    if env:
        for k, v in env.items():
            env_prelude += f"export {k}={shlex.quote(v)}\n"
    full = "set -e\n" + env_prelude + script
    b64 = base64.b64encode(full.encode("utf-8")).decode("ascii")
    proc = subprocess.run(
        ["ssh", "-tt", host, f"echo {b64} | base64 -d | bash"],
    )
    return proc.returncode


def ssh_one(host: str, cmd: str) -> RunResult:
    """Run a single oneshot command on `host` via SSH (no heredoc)."""
    return run(["ssh", host, cmd])


def ssh_stream(host: str, cmd: str) -> int:
    """Run a single command on `host` over SSH and stream output live.

    Used for long-running runs where the user wants progress as it happens
    (e.g. `mishabench run --remote ...`).
    """
    proc = subprocess.run(["ssh", host, cmd])
    return proc.returncode


def rsync_to(host: str, local_dir: str | Path, remote_dir: str,
             *, exclude: list[str] | None = None,
             delete: bool = True) -> int:
    """Rsync a local directory to a remote host over SSH. Streams rsync's
    own per-file output live to the user's terminal so the user sees
    progress for big trees / slow links. Returns the rsync exit code."""
    args = ["rsync", "-avh"]
    if delete:
        args.append("--delete")
    for pattern in (exclude or []):
        args += ["--exclude", pattern]
    args += [str(local_dir).rstrip("/") + "/", f"{host}:{remote_dir}/"]
    return subprocess.run(args).returncode


def rsync_from(host: str, remote_dir: str, local_dir: str | Path,
               *, exclude: list[str] | None = None) -> int:
    """Rsync a remote directory back to a local path. Streams output."""
    args = ["rsync", "-avh"]
    for pattern in (exclude or []):
        args += ["--exclude", pattern]
    args += [f"{host}:{remote_dir.rstrip('/')}/", str(local_dir).rstrip("/") + "/"]
    return subprocess.run(args).returncode
