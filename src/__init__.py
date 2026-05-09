# SPDX-FileCopyrightText: Copyright (c) 2026 Mikhail Yurasov
# SPDX-License-Identifier: Apache-2.0

"""MishaBench: cross-platform CPU + GPU benchmark for typical
data-analysis, computer-vision, and LLM workloads.

Designed to complete in under an hour on a modest CUDA-capable host
(or under a workday on a CPU-only laptop, with the LLM suite skipped).
Produces a single self-contained HTML report; no external assets.
Drives remote SSH targets via `mishabench run --remote <ssh-host>`.
"""

__version__ = "0.1.0"
