# SPDX-FileCopyrightText: Copyright (c) 2026 Mikhail Yurasov
# SPDX-License-Identifier: Apache-2.0

"""System probe: cpu, ram, gpus, driver, python, distro, library availability.

Output is a `SystemInfo` dataclass with everything the report needs in
its header, plus capability flags the runner uses to gate suites
(`has_cuda`, `has_mps`, `has_cudf`, ...).

Soft on every dependency -- a missing package returns None / False, not
an exception. The system probe must never crash a benchmark run.
"""

from __future__ import annotations

import importlib
import platform
import shutil
import socket
import sys
from dataclasses import asdict, dataclass, field
from typing import Any

from . import _run
from .power import detect_apple_tdp_w


@dataclass(slots=True)
class GpuInfo:
    index: int
    name: str
    # Optional because unified-memory devices (DGX Spark GB10, Jetson) report
    # `[N/A]` from `nvidia-smi --query-gpu=memory.total` -- they share VRAM
    # with system RAM and have no separate pool to query. We still want the
    # GPU listed; the report renders "memory n/a" instead of fabricating a
    # number. When torch.cuda is available, probe() backfills this from
    # `torch.cuda.get_device_properties().total_memory`.
    memory_mib: int | None
    driver: str
    compute_cap: str | None = None


@dataclass(slots=True)
class SystemInfo:
    hostname: str
    os_name: str
    os_version: str
    kernel: str
    arch: str
    distro: str | None
    cpu_model: str
    cpu_count_physical: int
    cpu_count_logical: int
    ram_total_gb: float
    ram_avail_gb: float
    disk_total_gb: float
    disk_free_gb: float
    python_version: str
    python_impl: str

    # GPUs
    gpus: list[GpuInfo] = field(default_factory=list)
    cuda_runtime: str | None = None
    nvidia_driver: str | None = None

    # Library availability + versions (only ones used by the suites)
    libs: dict[str, str | None] = field(default_factory=dict)

    # Capability flags
    has_cuda: bool = False
    has_mps: bool = False
    has_cudf: bool = False
    has_cupy: bool = False

    # Power-monitor capability + Apple Silicon TDP estimate
    apple_tdp_w: float | None = None  # set when CPU brand maps to APPLE_TDP_W
    has_rapl: bool = False  # /sys/class/powercap/intel-rapl:0/energy_uj readable

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["gpus"] = [asdict(g) for g in self.gpus]
        return d


def _try_import(modname: str) -> str | None:
    try:
        m = importlib.import_module(modname)
    except Exception:
        return None
    return getattr(m, "__version__", "?")


def _read_distro() -> str | None:
    if platform.system() != "Linux":
        return None
    try:
        with open("/etc/os-release", encoding="utf-8") as f:
            data = dict(
                line.strip().split("=", 1) for line in f if "=" in line
            )
    except Exception:
        return None
    name = data.get("PRETTY_NAME") or data.get("NAME")
    if name:
        return name.strip().strip('"')
    return None


def _read_cpu_model() -> str:
    if platform.system() == "Darwin":
        r = _run.run(["sysctl", "-n", "machdep.cpu.brand_string"])
        if r.ok and r.stdout.strip():
            return r.stdout.strip()
        return platform.processor() or "unknown"
    if platform.system() == "Linux":
        try:
            with open("/proc/cpuinfo", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("model name") and ":" in line:
                        return line.split(":", 1)[1].strip()
        except Exception:
            pass
    return platform.processor() or "unknown"


def _parse_int_or_none(s: str) -> int | None:
    """Parse an int from nvidia-smi CSV output. Returns None for `[N/A]`,
    `N/A`, or any field nvidia-smi declines to fill (notably memory.total
    on unified-memory devices like the DGX Spark GB10 and Jetson series)."""
    s = s.strip()
    if not s or s in {"[N/A]", "N/A"}:
        return None
    try:
        return int(s)
    except ValueError:
        return None


def _read_gpus() -> tuple[list[GpuInfo], str | None]:
    if not shutil.which("nvidia-smi"):
        return [], None
    fmt = "index,name,memory.total,driver_version,compute_cap"
    r = _run.run(["nvidia-smi", f"--query-gpu={fmt}", "--format=csv,noheader,nounits"])
    if not r.ok or not r.stdout.strip():
        return [], None
    gpus: list[GpuInfo] = []
    driver: str | None = None
    for line in r.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 4:
            continue
        try:
            idx = int(parts[0])
        except ValueError:
            continue
        gpus.append(GpuInfo(
            index=idx,
            name=parts[1] or "?",
            memory_mib=_parse_int_or_none(parts[2]),
            driver=parts[3],
            compute_cap=parts[4] if len(parts) > 4 else None,
        ))
        driver = parts[3]
    return gpus, driver


def _enrich_gpus_from_torch(gpus: list[GpuInfo],
                            driver: str | None) -> list[GpuInfo]:
    """When nvidia-smi parsing left holes -- or missed the GPU entirely --
    fall back to `torch.cuda.get_device_properties()`. Torch is what the
    bench actually uses, so its view is authoritative for the report.

    Two scenarios:
      - nvidia-smi missed the GPU (containerized / restricted hosts):
        synthesize the entry from torch alone.
      - nvidia-smi found the GPU but `memory.total` came back `[N/A]`
        (DGX Spark / Jetson unified memory): backfill memory_mib from
        torch's reported total_memory.
    """
    try:
        import torch
        if not torch.cuda.is_available():
            return gpus
        torch_count = torch.cuda.device_count()
    except Exception:
        return gpus

    if not gpus:
        for i in range(torch_count):
            try:
                p = torch.cuda.get_device_properties(i)
            except Exception:
                continue
            gpus.append(GpuInfo(
                index=i,
                name=p.name,
                memory_mib=p.total_memory // (2 ** 20),
                driver=driver or "?",
                compute_cap=f"{p.major}.{p.minor}",
            ))
        return gpus

    for g in gpus:
        if g.memory_mib is None and g.index < torch_count:
            try:
                p = torch.cuda.get_device_properties(g.index)
                g.memory_mib = p.total_memory // (2 ** 20)
                if not g.compute_cap:
                    g.compute_cap = f"{p.major}.{p.minor}"
            except Exception:
                pass
    return gpus


def _read_cuda_runtime() -> str | None:
    """Cuda runtime as reported by torch (preferred -- it's what the bench will use)."""
    try:
        import torch
        return torch.version.cuda
    except Exception:
        return None


def _detect_torch_caps() -> tuple[bool, bool]:
    try:
        import torch
        cuda = torch.cuda.is_available()
        mps = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
        return cuda, mps
    except Exception:
        return False, False


def _safe(fn, default):
    """Call psutil-like function; return `default` on any platform error."""
    try:
        v = fn()
        return v if v is not None else default
    except Exception:
        return default


def probe() -> SystemInfo:
    import os

    import psutil

    arch = platform.machine()
    sysname = platform.system()
    cpu_model = _read_cpu_model()
    distro = _read_distro()
    gpus, driver = _read_gpus()
    cuda_runtime = _read_cuda_runtime()
    has_cuda, has_mps = _detect_torch_caps()
    if has_cuda:
        gpus = _enrich_gpus_from_torch(gpus, driver)

    libs = {
        "numpy": _try_import("numpy"),
        "pandas": _try_import("pandas"),
        "polars": _try_import("polars"),
        "pyarrow": _try_import("pyarrow"),
        "torch": _try_import("torch"),
        "torchvision": _try_import("torchvision"),
        "transformers": _try_import("transformers"),
        "sentence_transformers": _try_import("sentence_transformers"),
        "tiktoken": _try_import("tiktoken"),
        "cv2": _try_import("cv2"),
        "PIL": _try_import("PIL"),
        "cudf": _try_import("cudf"),
        "cupy": _try_import("cupy"),
    }

    # psutil sysctls can be blocked on hardened macOS sandboxes / locked-
    # down containers. Fall back to os.cpu_count() and 0 for memory/disk
    # so the probe never crashes a bench run on an unusual host.
    cpu_logical = _safe(lambda: psutil.cpu_count(logical=True), os.cpu_count() or 0)
    cpu_physical = _safe(lambda: psutil.cpu_count(logical=False), cpu_logical)
    vm_total = _safe(lambda: psutil.virtual_memory().total, 0)
    vm_avail = _safe(lambda: psutil.virtual_memory().available, 0)
    du_total = _safe(lambda: psutil.disk_usage("/").total, 0)
    du_free = _safe(lambda: psutil.disk_usage("/").free, 0)

    apple_tdp = detect_apple_tdp_w(cpu_model) if sysname == "Darwin" else None
    has_rapl = False
    if sysname == "Linux":
        from .power import _rapl_package_path
        has_rapl = _rapl_package_path() is not None

    return SystemInfo(
        hostname=socket.gethostname(),
        os_name=sysname,
        os_version=platform.release(),
        kernel=platform.version(),
        arch=arch,
        distro=distro,
        cpu_model=cpu_model,
        cpu_count_physical=cpu_physical,
        cpu_count_logical=cpu_logical,
        ram_total_gb=round(vm_total / 2**30, 2),
        ram_avail_gb=round(vm_avail / 2**30, 2),
        disk_total_gb=round(du_total / 2**30, 1),
        disk_free_gb=round(du_free / 2**30, 1),
        python_version=platform.python_version(),
        python_impl=platform.python_implementation(),
        gpus=gpus,
        cuda_runtime=cuda_runtime,
        nvidia_driver=driver,
        libs=libs,
        has_cuda=has_cuda,
        has_mps=has_mps,
        has_cudf=libs["cudf"] is not None,
        has_cupy=libs["cupy"] is not None,
        apple_tdp_w=apple_tdp,
        has_rapl=has_rapl,
    )


def short_summary(info: SystemInfo) -> str:
    """One-line summary for terminal headers / log lines."""
    if info.gpus:
        gpu_part = ", ".join(
            f"{g.name} ({g.memory_mib} MiB)" if g.memory_mib else g.name
            for g in info.gpus
        )
    else:
        gpu_part = "no NVIDIA GPU"
    cuda_part = f"cuda={info.cuda_runtime}" if info.has_cuda else "cuda=off"
    return (
        f"{info.hostname} | {info.distro or info.os_name} {info.arch} | "
        f"{info.cpu_count_logical}t cpu, {info.ram_total_gb} GiB ram | "
        f"{gpu_part} | {cuda_part} | py{sys.version_info.major}.{sys.version_info.minor}"
    )
