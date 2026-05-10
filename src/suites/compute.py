# SPDX-FileCopyrightText: Copyright (c) 2026 Mikhail Yurasov
# SPDX-License-Identifier: Apache-2.0

"""Compute suite: raw CPU primitives, single-thread and multi-thread.

The data / cv / llm suites are workload-flavoured: pandas dataframes,
torchvision models, HuggingFace LLMs. Useful, but their CPU paths
exercise BLAS / NumPy / torch all at once and the threading config is
implicit (whatever torch / openblas pick by default).

This suite isolates the underlying compute primitives:

  C1 matmul        4096x4096 fp32 matmul via numpy (BLAS-backed).
                   Two variants: multi-thread (mt, all cores) and
                   single-thread (st, BLAS limited to 1 thread). The
                   ratio is a clean per-core vs total-throughput signal.
  C2 fft           1D FFT of a 2^22 (4 Mi) point complex64 array via
                   scipy.fft. Same mt / st split. Memory-bandwidth +
                   FLOPS-bound; complements matmul which is FLOPS-only.
  C3 python        Pure-Python prime sieve up to N (single-threaded by
                   construction). Tests interpreter speed -- a number
                   that NumPy / torch / pandas all sit on top of.

Single-thread variants use `threadpoolctl.threadpool_limits(1)` to
clamp the BLAS / OpenMP threadpool around the timed window. Restored
on exit so subsequent benches see the original (parallel) limits.
"""

from __future__ import annotations

import time as _time

import numpy as np

from ..config import BenchConfig
from ..runner import Bench, register


def _matmul_size(quick: bool) -> int:
    return 1024 if quick else 4096


def _fft_size_log2(quick: bool) -> int:
    return 18 if quick else 22  # 2^18 = 262144 vs 2^22 = 4194304


def _python_n(quick: bool) -> int:
    return 200_000 if quick else 1_000_000


# ---- numpy matmul ----

def _bench_matmul(cfg: BenchConfig, threads: int | None):
    """numpy matmul throughput in GFLOPS. `threads` is None for 'use whatever
    BLAS picks by default' (multi-thread) or 1 for the single-thread variant."""
    from threadpoolctl import threadpool_limits
    n = _matmul_size(cfg.quick)
    rng = np.random.default_rng(11)
    a = rng.standard_normal((n, n), dtype=np.float32)
    b = rng.standard_normal((n, n), dtype=np.float32)
    # 2*N^3 FLOPs per matmul (one mul + one add per output element)
    flops_per = 2 * n * n * n
    iters = 1 if not cfg.quick else 1
    if threads is None:
        # warmup
        _ = a @ b
        t0 = _time.perf_counter()
        for _ in range(iters):
            _ = a @ b
        elapsed = _time.perf_counter() - t0
    else:
        with threadpool_limits(limits=threads, user_api="blas"):
            _ = a @ b  # warmup
            t0 = _time.perf_counter()
            for _ in range(iters):
                _ = a @ b
            elapsed = _time.perf_counter() - t0
    gflops = (flops_per * iters) / elapsed / 1e9
    return ("throughput", round(gflops, 2), "GFLOPS",
            {"n": n, "iters": iters, "threads": threads or "default",
             "seconds": round(elapsed, 4)})


def bench_matmul_mt(cfg: BenchConfig):
    return _bench_matmul(cfg, None)


def bench_matmul_st(cfg: BenchConfig):
    return _bench_matmul(cfg, 1)


# ---- scipy FFT ----

def _bench_fft(cfg: BenchConfig, threads: int | None):
    """1D FFT throughput in GFLOPS. FFT FLOP count = 5 * N * log2(N)."""
    from scipy import fft as scipy_fft
    from threadpoolctl import threadpool_limits
    log2n = _fft_size_log2(cfg.quick)
    n = 1 << log2n
    rng = np.random.default_rng(13)
    x = rng.standard_normal(n).astype(np.complex64)
    flops_per = 5 * n * log2n
    iters = 8 if cfg.quick else 16
    if threads is None:
        _ = scipy_fft.fft(x)  # warmup
        t0 = _time.perf_counter()
        for _ in range(iters):
            _ = scipy_fft.fft(x)
        elapsed = _time.perf_counter() - t0
    else:
        with threadpool_limits(limits=threads, user_api="blas"):
            _ = scipy_fft.fft(x)  # warmup
            t0 = _time.perf_counter()
            for _ in range(iters):
                _ = scipy_fft.fft(x)
            elapsed = _time.perf_counter() - t0
    gflops = (flops_per * iters) / elapsed / 1e9
    return ("throughput", round(gflops, 2), "GFLOPS",
            {"n": n, "log2n": log2n, "iters": iters,
             "threads": threads or "default", "seconds": round(elapsed, 4)})


def bench_fft_mt(cfg: BenchConfig):
    return _bench_fft(cfg, None)


def bench_fft_st(cfg: BenchConfig):
    return _bench_fft(cfg, 1)


# ---- pure-Python prime sieve ----

def bench_python_primes(cfg: BenchConfig):
    """Sieve of Eratosthenes up to N. Pure-Python loop work -- no NumPy,
    no torch, no BLAS. Tests interpreter execution speed (a single core
    by definition; the GIL doesn't share Python bytecode across threads).

    Reported as M-elements/s (sieve elements processed per second), so
    higher is better and it composes with the other compute benches'
    geomean cleanly."""
    n = _python_n(cfg.quick)
    t0 = _time.perf_counter()
    sieve = bytearray(b"\x01") * (n + 1)
    sieve[0] = sieve[1] = 0
    i = 2
    while i * i <= n:
        if sieve[i]:
            sieve[i * i::i] = bytearray(len(sieve[i * i::i]))
        i += 1
    n_primes = sum(sieve)
    elapsed = _time.perf_counter() - t0
    rate = (n / 1e6) / elapsed if elapsed > 0 else 0.0
    return ("throughput", round(rate, 3), "M elements/s sieved",
            {"n": n, "primes_found": n_primes, "seconds": round(elapsed, 4)})


# ---- registration ----

register(Bench("compute.matmul.cpu_mt", "compute", "Matmul 4096^2 (multi-thread)", "cpu_mt",
               bench_matmul_mt, expected_seconds=10))
register(Bench("compute.matmul.cpu_st", "compute", "Matmul 4096^2 (single-thread)", "cpu_st",
               bench_matmul_st, expected_seconds=60))

register(Bench("compute.fft.cpu_mt", "compute", "FFT 2^22 c64 (multi-thread)", "cpu_mt",
               bench_fft_mt, expected_seconds=10))
register(Bench("compute.fft.cpu_st", "compute", "FFT 2^22 c64 (single-thread)", "cpu_st",
               bench_fft_st, expected_seconds=20))

register(Bench("compute.python.cpu", "compute", "Python prime sieve to 1M", "cpu",
               bench_python_primes, expected_seconds=15))
