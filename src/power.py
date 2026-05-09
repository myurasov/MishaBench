# SPDX-FileCopyrightText: Copyright (c) 2026 Mikhail Yurasov
# SPDX-License-Identifier: Apache-2.0

"""Power-draw sampling for points-per-watt scoring.

Three sources, each soft-detected at probe time:

  - **NVIDIA GPU**: `nvidia-smi --query-gpu=power.draw` polled at 2 Hz.
    Real measurements (W).
  - **Intel/AMD CPU on Linux**: Intel RAPL via
    `/sys/class/powercap/intel-rapl:0/energy_uj` (works for AMD too on
    recent kernels). Reads cumulative microjoule counter; bench window
    delta / wall_seconds = average watts. **Real** measurements when
    accessible without sudo (most desktops -- the file is world-readable
    on default Ubuntu / Debian / Fedora setups).
  - **Apple Silicon**: lookup table from chip name (M1 / M1 Pro / ... /
    M4 Max) to nominal package TDP. Marked `estimated=True`. Sudo-free
    `powermetrics` does not exist; we deliberately don't shell out for
    sudo at bench time -- the user shouldn't have to type a password to
    run a benchmark.

If no source is available for a given device class, the corresponding
field on `PowerWindow` is None and the report shows "n/a" instead of
fabricating a number.

The monitor is a thread that samples on a tick and stops cleanly via
the `with PowerMonitor(...) as window:` context manager.
"""

from __future__ import annotations

import shutil
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

# ---- Apple Silicon TDP lookup ----
# Sources: Apple's product pages + tech press teardowns; kept conservative.
# Whole-package (CPU + GPU + ANE + memory controller) sustained TDP.
APPLE_TDP_W: dict[str, float] = {
    "Apple M1": 15.0,
    "Apple M1 Pro": 30.0,
    "Apple M1 Max": 60.0,
    "Apple M1 Ultra": 100.0,
    "Apple M2": 15.0,
    "Apple M2 Pro": 35.0,
    "Apple M2 Max": 70.0,
    "Apple M2 Ultra": 140.0,
    "Apple M3": 18.0,
    "Apple M3 Pro": 40.0,
    "Apple M3 Max": 80.0,
    "Apple M4": 22.0,
    "Apple M4 Pro": 45.0,
    "Apple M4 Max": 90.0,
    "Apple M4 Ultra": 180.0,
}


def apple_chip_tdp(cpu_brand: str) -> float | None:
    """Best-effort match of `sysctl machdep.cpu.brand_string` against the
    TDP table. Returns None on miss."""
    if not cpu_brand:
        return None
    s = cpu_brand.strip()
    # Direct hits
    if s in APPLE_TDP_W:
        return APPLE_TDP_W[s]
    # Partial: prefer longer (more specific) matches first
    for name in sorted(APPLE_TDP_W.keys(), key=len, reverse=True):
        if name in s:
            return APPLE_TDP_W[name]
    if s.startswith("Apple "):
        return 15.0  # generic Apple Silicon fallback
    return None


# ---- RAPL (Intel + AMD on Linux) ----

_RAPL_ROOT = Path("/sys/class/powercap")


def _rapl_package_path() -> Path | None:
    """First package-domain RAPL energy file we can read."""
    if not _RAPL_ROOT.exists():
        return None
    for d in sorted(_RAPL_ROOT.glob("intel-rapl:*")):
        # Top-level intel-rapl:N is the CPU package; sub-domains
        # (intel-rapl:N:M) are uncore / dram / etc -- skip them here.
        if ":" in d.name and d.name.count(":") == 1:
            energy = d / "energy_uj"
            if energy.exists():
                try:
                    int(energy.read_text().strip())
                    return energy
                except (OSError, PermissionError, ValueError):
                    continue
    return None


def _read_rapl_uj(path: Path) -> int | None:
    try:
        return int(path.read_text().strip())
    except Exception:
        return None


# ---- NVIDIA GPU power via nvidia-smi ----

def _nvidia_smi_available() -> bool:
    return shutil.which("nvidia-smi") is not None


def _read_nvidia_power_w() -> list[float]:
    """Per-GPU instantaneous power draw in watts. Empty list on failure."""
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=power.draw",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=2.0,
        )
    except Exception:
        return []
    if r.returncode != 0:
        return []
    out: list[float] = []
    import contextlib
    for line in r.stdout.strip().splitlines():
        s = line.strip()
        if not s or s in {"[N/A]", "N/A"}:
            continue
        with contextlib.suppress(ValueError):
            out.append(float(s))
    return out


# ---- PowerWindow + PowerMonitor ----

@dataclass(slots=True)
class PowerSample:
    t: float
    cpu_w: float | None
    gpu_w: float | None  # primary GPU only (sum of all GPUs is in notes if needed)


@dataclass(slots=True)
class PowerWindow:
    samples: list[PowerSample] = field(default_factory=list)
    duration_s: float = 0.0

    cpu_avg_w: float | None = None
    cpu_energy_j: float | None = None
    cpu_estimated: bool = False

    gpu_avg_w: float | None = None
    gpu_energy_j: float | None = None
    gpu_peak_w: float | None = None

    def to_dict(self) -> dict:
        return {
            "duration_s": round(self.duration_s, 3),
            "cpu_avg_w": None if self.cpu_avg_w is None else round(self.cpu_avg_w, 2),
            "cpu_energy_j": None if self.cpu_energy_j is None else round(self.cpu_energy_j, 2),
            "cpu_estimated": self.cpu_estimated,
            "gpu_avg_w": None if self.gpu_avg_w is None else round(self.gpu_avg_w, 2),
            "gpu_energy_j": None if self.gpu_energy_j is None else round(self.gpu_energy_j, 2),
            "gpu_peak_w": None if self.gpu_peak_w is None else round(self.gpu_peak_w, 2),
            "n_samples": len(self.samples),
        }

    def power_for_device(self, device: str) -> tuple[float | None, bool]:
        """Return (avg_watts, estimated) for the given device class."""
        if device.startswith("cuda"):
            return self.gpu_avg_w, False
        # cpu / mps both attribute to package power on Apple; on Linux
        # MPS isn't a thing, so this branch is mac-only for mps.
        return self.cpu_avg_w, self.cpu_estimated


class PowerMonitor:
    """Background sampler. Use as `with PowerMonitor(...) as w: ...`.

    The monitor decides at construction time which probes are live (RAPL
    file readable, nvidia-smi present, Apple Silicon estimate). Probes
    that fail at construction stay dormant; missing data appears as None
    in the resulting `PowerWindow`.
    """

    def __init__(self, *, interval_s: float = 0.5,
                 apple_tdp_w: float | None = None) -> None:
        self.interval_s = interval_s
        self.window = PowerWindow()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._t_start: float = 0.0

        self._rapl_path = _rapl_package_path()
        self._rapl_t0_uj: int | None = None
        self._rapl_t0_t: float = 0.0
        self._has_nvidia = _nvidia_smi_available()
        self._apple_tdp_w = apple_tdp_w  # if set, used as a steady-state CPU power estimate

    # ---------- context-manager protocol ----------

    def __enter__(self) -> PowerWindow:
        self._t_start = time.perf_counter()
        if self._rapl_path is not None:
            self._rapl_t0_uj = _read_rapl_uj(self._rapl_path)
            self._rapl_t0_t = self._t_start
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="mishabench-power", daemon=True)
        self._thread.start()
        return self.window

    def __exit__(self, exc_type, exc, tb) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self.window.duration_s = time.perf_counter() - self._t_start
        self._finalize()

    # ---------- internals ----------

    def _loop(self) -> None:
        while not self._stop.is_set():
            cpu_w = self._sample_cpu_w()
            gpu_w = self._sample_gpu_w()
            self.window.samples.append(PowerSample(
                t=time.perf_counter() - self._t_start, cpu_w=cpu_w, gpu_w=gpu_w,
            ))
            # Sleep in small increments so stop is responsive
            self._stop.wait(self.interval_s)

    def _sample_cpu_w(self) -> float | None:
        """Instantaneous CPU power. Polled samples on RAPL are coarse
        because we'd need two reads to get a delta -- the final RAPL delta
        is computed in _finalize from the cumulative counter, which is
        the authoritative number. Here we just stub a snapshot."""
        if self._apple_tdp_w is not None:
            # Steady-state TDP estimate; honest about being an estimate
            # via PowerWindow.cpu_estimated = True.
            return self._apple_tdp_w
        return None

    def _sample_gpu_w(self) -> float | None:
        if not self._has_nvidia:
            return None
        per_gpu = _read_nvidia_power_w()
        if not per_gpu:
            return None
        # Primary GPU is index 0. Track peak across all GPUs separately.
        if self.window.gpu_peak_w is None:
            self.window.gpu_peak_w = max(per_gpu)
        else:
            self.window.gpu_peak_w = max(self.window.gpu_peak_w, *per_gpu)
        return per_gpu[0]

    def _finalize(self) -> None:
        s = self.window.samples
        # GPU avg from per-sample readings (real numbers)
        gpu_vals = [x.gpu_w for x in s if x.gpu_w is not None]
        if gpu_vals:
            self.window.gpu_avg_w = sum(gpu_vals) / len(gpu_vals)
            if self.window.duration_s > 0:
                self.window.gpu_energy_j = self.window.gpu_avg_w * self.window.duration_s

        # CPU: prefer RAPL delta (real); else Apple Silicon TDP estimate.
        if self._rapl_path is not None and self._rapl_t0_uj is not None:
            t1_uj = _read_rapl_uj(self._rapl_path)
            if t1_uj is not None:
                # RAPL counter wraps at MAX (varies per CPU; usually
                # 2^32 microjoules ~ 4 kJ). Rare in seconds-scale benches
                # but we treat negative deltas as wraparound and skip.
                d_uj = t1_uj - self._rapl_t0_uj
                d_t = time.perf_counter() - self._rapl_t0_t
                if d_uj >= 0 and d_t > 0:
                    energy_j = d_uj / 1e6
                    self.window.cpu_energy_j = energy_j
                    self.window.cpu_avg_w = energy_j / d_t
                    self.window.cpu_estimated = False
                    return
        if self._apple_tdp_w is not None:
            # Steady-state TDP -> we already filled samples with that
            # value. The "estimate" comes from the chip-name lookup.
            cpu_vals = [x.cpu_w for x in s if x.cpu_w is not None]
            if cpu_vals:
                self.window.cpu_avg_w = sum(cpu_vals) / len(cpu_vals)
                if self.window.duration_s > 0:
                    self.window.cpu_energy_j = self.window.cpu_avg_w * self.window.duration_s
                self.window.cpu_estimated = True


def detect_apple_tdp_w(cpu_brand: str | None) -> float | None:
    """Public re-export for the system probe to surface in info.libs."""
    if not cpu_brand:
        return None
    return apple_chip_tdp(cpu_brand)
