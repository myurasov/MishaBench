# Contributing to MishaBench

Thanks for your interest in contributing! MishaBench is a personal open-source project, released under the [Apache License 2.0](LICENSE). This document explains the contribution workflow, the developer-certificate-of-origin (DCO) sign-off requirement, and the basic coding conventions.

## Table of Contents

- [Issue Tracking](#issue-tracking)
- [Pull Requests](#pull-requests)
- [Coding Guidelines](#coding-guidelines)
- [Signing Your Work (DCO)](#signing-your-work-dco)

## Issue Tracking

All enhancement, bug-fix, or change requests should start with a [GitHub Issue](https://github.com/myurasov/MishaBench/issues) so the design / scope can be discussed before code is written.

## Pull Requests

The developer workflow:

1. **Fork** the upstream repository: https://github.com/myurasov/MishaBench
2. **Branch** from `main` in your fork; one branch per logical change.
3. **Develop** locally:

   ```bash
   # rebuild the venv from a clean state
   ./mishabench install --force

   # pytest
   ./mishabench test

   # ruff check
   ./mishabench lint

   # ruff check --fix + ruff format
   ./mishabench fmt
   ```

4. **Sign off** every commit with `git commit -s` (see [Signing Your Work](#signing-your-work-dco)). Unsigned commits will not be accepted.
5. **Open a PR** from your fork's branch into `main` of the upstream repo. Use a descriptive title in the imperative mood (e.g. `Add cudf join benchmark`, not `added cudf join`).
6. **Reference the issue number** in the PR body if there's a corresponding issue (e.g. `Closes #42`).
7. While under review, prefix work-in-progress PRs with `[WIP]`.

A reviewer will look at the PR. Please respond to review comments promptly; PRs that go silent for > 30 days may be closed (and can always be reopened).

## Coding Guidelines

- Follow the existing style in the file you're editing. MishaBench uses `ruff` for both linting and formatting; running `./mishabench fmt` will auto-fix most issues.
- Internal package imports use **relative form** (`from . import config`, `from ._run import ssh_run`) -- keeps the package portable if the import name ever changes.
- Subprocess / SSH calls go through `_run.run` / `_run.ssh_run` / `_run.ssh_stream` / `_run.rsync_to`. Don't shell out via raw `subprocess.run` from new call-sites -- adding a primitive to `_run.py` is the right factoring.
- **Measurement honesty is the headline rule.** CUDA / MPS benches must call `torch.cuda.synchronize()` / `torch.mps.synchronize()` immediately before stopping the clock, and must run at least one un-timed warmup iteration before the timed window. A bench that produces a misleading number is worse than one that doesn't run at all.
- **Optional dependencies** (`cudf`, `cupy`, CUDA, MPS) are soft-imported inside the bench function -- never at module top. The runner skips a bench cleanly when its required capability is missing.
- The score scaling constant (`scoring.SCORE_SCALE = 1000.0`) is **fixed across releases** so two runs are directly comparable. Do not change it without explicit approval.
- Every new source file must include the SPDX license header (see existing files for the exact format):

  ```python
  # SPDX-FileCopyrightText: Copyright (c) <year> <Your Name>
  # SPDX-License-Identifier: Apache-2.0
  ```

  When making a substantial contribution to an existing file, you may add your copyright on a new line below the existing one -- multiple copyright holders per file are fine.

- Keep PRs focused. If you find unrelated bugs while working on something, file a separate issue / PR.
- Update tests under `tests/` whenever you add or change behavior.
- Update `README.md` and/or `ai/spec.md` whenever the user-facing surface or measurement contract changes.
- Update [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) in the same PR whenever you add, remove, or upgrade a runtime / dev dependency in `pyproject.toml`.
- Commit messages should be in imperative mood (`Add foo`, `Fix bar`, `Refactor baz`) and keep the subject ≤ 72 chars when practical. ASCII only.

## Signing Your Work (DCO)

I require every contributor to sign off on their commits. The sign-off certifies that the contribution is your original work, or you have rights to submit it under the project's license. This is the standard [Developer Certificate of Origin (DCO)](https://developercertificate.org/) -- the same one used by the Linux kernel, Docker, and many other open-source projects.

**To sign off on a commit**, use the `--signoff` (or `-s`) flag:

```bash
git commit -s -m "Add cool feature"
```

This appends a line to the commit message:

```
Signed-off-by: Your Name <your@email.com>
```

`Your Name` and `your@email.com` must match your `git config user.name` and `git config user.email`. Anonymous contributions (no real name, no real email) cannot be accepted.

PRs containing unsigned commits will be blocked until every commit is signed. To retroactively sign existing commits, use `git rebase --signoff <base>` (or `git commit --amend --signoff` for the most recent one).

### Full text of the DCO

```
Developer Certificate of Origin
Version 1.1

Copyright (C) 2004, 2006 The Linux Foundation and its contributors.
1 Letterman Drive
Suite D4700
San Francisco, CA, 94129

Everyone is permitted to copy and distribute verbatim copies of this
license document, but changing it is not allowed.


Developer's Certificate of Origin 1.1

By making a contribution to this project, I certify that:

(a) The contribution was created in whole or in part by me and I
    have the right to submit it under the open source license
    indicated in the file; or

(b) The contribution is based upon previous work that, to the best
    of my knowledge, is covered under an appropriate open source
    license and I have the right under that license to submit that
    work with modifications, whether created in whole or in part
    by me, under the same open source license (unless I am
    permitted to submit under a different license), as indicated
    in the file; or

(c) The contribution was provided directly to me by some other
    person who certified (a), (b) or (c) and I have not modified
    it.

(d) I understand and agree that this project and the contribution
    are public and that a record of the contribution (including all
    personal information I submit with it, including my sign-off) is
    maintained indefinitely and may be redistributed consistent with
    this project or the open source license(s) involved.
```
