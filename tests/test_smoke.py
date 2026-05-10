# SPDX-FileCopyrightText: Copyright (c) 2026 Mikhail Yurasov
# SPDX-License-Identifier: Apache-2.0

"""Smoke tests. No live GPU host or network model required.

Strategy: import everything, exercise the system probe (it must never
crash on a probe-only call regardless of platform), exercise the CLI
help screens via Typer's CliRunner, and verify the report renderer
produces well-formed HTML from a hand-built BenchResult list.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

# ---- imports ----

def test_import_all_modules():
    import src.cli  # noqa: F401
    import src.config  # noqa: F401
    import src.power  # noqa: F401
    import src.remote  # noqa: F401
    import src.report  # noqa: F401
    import src.runner  # noqa: F401
    import src.scoring  # noqa: F401
    import src.suites.cv  # noqa: F401
    import src.suites.data  # noqa: F401
    import src.suites.llm  # noqa: F401
    import src.system  # noqa: F401


# ---- system probe ----

def test_probe_returns_systeminfo_with_required_fields():
    from src.system import probe
    info = probe()
    assert info.hostname
    assert info.cpu_count_logical >= 1
    assert info.ram_total_gb > 0
    # Capability flags are bool, never None
    assert isinstance(info.has_cuda, bool)
    assert isinstance(info.has_mps, bool)
    assert isinstance(info.has_cudf, bool)
    assert isinstance(info.has_rapl, bool)
    # apple_tdp_w is float | None; type-check by exclusion
    assert info.apple_tdp_w is None or isinstance(info.apple_tdp_w, float)


def test_short_summary_is_one_line():
    from src.system import probe, short_summary
    s = short_summary(probe())
    assert "\n" not in s
    assert "|" in s


# ---- power monitor ----

def test_power_monitor_exits_cleanly_when_no_sources_present():
    """Even on a host with neither nvidia-smi nor RAPL nor an Apple chip,
    the monitor must produce a usable PowerWindow (with all-None power
    fields)."""
    from src.power import PowerMonitor
    with PowerMonitor(interval_s=0.05) as window:
        # Spin briefly so the sampler thread runs at least once
        import time
        time.sleep(0.2)
    assert window.duration_s > 0
    # All None or all numeric -- never raise
    for field in ("cpu_avg_w", "gpu_avg_w"):
        v = getattr(window, field)
        assert v is None or isinstance(v, float)


# ---- registry / suites ----

def test_registry_populated_after_suite_import():
    import src.suites.compute  # noqa: F401
    import src.suites.cv  # noqa: F401
    import src.suites.data  # noqa: F401
    import src.suites.llm  # noqa: F401
    from src.runner import registry
    rows = registry()
    suites = {b.suite for b in rows}
    assert suites == {"data", "cv", "llm", "compute"}
    # 13 data + 25 cv + 16 llm + 5 compute ~= 59; lower bound here
    assert len(rows) >= 50


def test_every_bench_has_required_fields():
    from src.config import ALL_SUITES
    from src.runner import registry
    valid_devices = ("cpu", "cuda", "mps", "cuda_multi", "cpu_mt", "cpu_st")
    for b in registry():
        assert b.id and "." in b.id
        assert b.suite in ALL_SUITES
        assert b.device in valid_devices, f"{b.id} has unknown device {b.device}"
        assert b.expected_seconds > 0


# ---- config ----

def test_quick_budget_kicks_in():
    from src.config import FULL_BUDGET_S, QUICK_BUDGET_S, BenchConfig
    cfg = BenchConfig(quick=True)
    assert cfg.effective_budget_s == QUICK_BUDGET_S
    cfg2 = BenchConfig(quick=False)
    assert cfg2.effective_budget_s == FULL_BUDGET_S


def test_expected_for_scales_in_quick_mode():
    from src.config import BenchConfig
    cfg = BenchConfig(quick=True)
    assert cfg.expected_for(60.0) == 6.0
    assert cfg.expected_for(0.1) == 2.0  # floor at 2s
    cfg2 = BenchConfig(quick=False)
    assert cfg2.expected_for(60.0) == 60.0


def test_parse_suites_validates():
    from src.config import ALL_SUITES, parse_suites
    assert parse_suites(None) == ALL_SUITES
    assert parse_suites("data") == ("data",)
    assert parse_suites("cv,llm") == ("cv", "llm")
    with pytest.raises(ValueError):
        parse_suites("data,bogus")


# ---- scoring ----

def test_geomean_skips_non_positive():
    from src.scoring import _geomean
    assert _geomean([0.0, -1.0, 4.0]) == 4.0
    assert _geomean([1.0, 4.0]) == 2.0
    assert _geomean([]) is None


def test_compute_scores_basic_shape():
    from src.runner import BenchResult
    from src.scoring import compute
    rows = [
        BenchResult(id="a", suite="data", name="A", device="cpu",
                    metric="t", value=100.0, unit="x", seconds=1.0, iters=1,
                    notes={"avg_watts": 50.0, "power_estimated": False}),
        BenchResult(id="b", suite="data", name="B", device="cpu",
                    metric="t", value=400.0, unit="x", seconds=1.0, iters=1,
                    notes={"avg_watts": 50.0, "power_estimated": False}),
        BenchResult(id="c", suite="cv", name="C", device="cuda",
                    metric="t", value=2000.0, unit="x", seconds=1.0, iters=1,
                    notes={"avg_watts": 250.0, "power_estimated": False}),
    ]
    rep = compute(rows)
    cpu_data = rep.per_device_per_suite["cpu"]["data"]
    # geomean(100, 400) * 1000 = 200_000
    assert cpu_data.score == 200_000.0
    # per_device_total is intentionally NOT populated -- a geomean across
    # suites with different units is meaningless.
    assert rep.per_device_total == {}
    cuda_cv = rep.per_device_per_suite["cuda"]["cv"]
    assert cuda_cv.score == 2_000_000.0
    assert cuda_cv.avg_watts == 250.0
    assert cuda_cv.pts_per_watt == round(2_000_000.0 / 250.0, 2)


# ---- report renderer ----

def test_report_renders_well_formed_html(tmp_path: Path):
    from src.report import render
    from src.runner import BenchResult
    from src.system import SystemInfo
    info = SystemInfo(
        hostname="testhost", os_name="Linux", os_version="5.15", kernel="x",
        arch="x86_64", distro="Ubuntu", cpu_model="cpu", cpu_count_physical=8,
        cpu_count_logical=16, ram_total_gb=32.0, ram_avail_gb=20.0,
        disk_total_gb=500.0, disk_free_gb=300.0, python_version="3.12.1",
        python_impl="CPython", libs={"torch": "2.4"}, has_cuda=True,
    )
    rows = [
        BenchResult(id="data.gb.pandas", suite="data", name="Group-by",
                    device="cpu", metric="t", value=10.0, unit="M rows",
                    seconds=1.0, iters=1,
                    notes={"avg_watts": 50.0, "power_estimated": False}),
        BenchResult(id="cv.resnet50.cuda", suite="cv", name="ResNet-50",
                    device="cuda", metric="t", value=2000.0, unit="img/s",
                    seconds=1.0, iters=1,
                    notes={"avg_watts": 250.0, "power_estimated": False}),
    ]
    html = render(rows, info, total_seconds=2.0, labels={"mode": "test"})
    assert html.startswith("<!DOCTYPE html>")
    assert "MishaBench report" in html
    assert "testhost" in html
    assert "Group-by" in html
    assert "ResNet-50" in html
    assert "CUDA (NVIDIA)" in html


def test_unified_memory_gpu_renders_without_crashing():
    """DGX Spark GB10 (and Jetson) report `[N/A]` for memory.total because
    they have unified memory. The probe must keep the GPU listed and the
    report must render `memory n/a (unified)` instead of crashing."""
    from src.report import render
    from src.system import GpuInfo, SystemInfo, short_summary
    info = SystemInfo(
        hostname="myspark1", os_name="Linux", os_version="6.17", kernel="x",
        arch="aarch64", distro="Ubuntu 24.04.4 LTS", cpu_model="ARM Cortex",
        cpu_count_physical=20, cpu_count_logical=20, ram_total_gb=121.69,
        ram_avail_gb=100.0, disk_total_gb=1000.0, disk_free_gb=800.0,
        python_version="3.11.0", python_impl="CPython",
        libs={"torch": "2.11.0"}, has_cuda=True, cuda_runtime="13.0",
        nvidia_driver="580.142",
        gpus=[GpuInfo(index=0, name="NVIDIA GB10", memory_mib=None,
                      driver="580.142", compute_cap="12.1")],
    )
    s = short_summary(info)
    assert "NVIDIA GB10" in s
    # memory_mib was None -- summary shouldn't print "None MiB"
    assert "None" not in s

    html = render([], info, total_seconds=0.0)
    assert "NVIDIA GB10" in html
    assert "memory n/a (unified)" in html


def test_parse_na_helpers():
    """Direct test of the [N/A] tolerance in the nvidia-smi parser."""
    from src.system import _parse_int_or_none
    assert _parse_int_or_none("[N/A]") is None
    assert _parse_int_or_none("N/A") is None
    assert _parse_int_or_none("") is None
    assert _parse_int_or_none(" 4096 ") == 4096
    assert _parse_int_or_none("not a number") is None


def test_rapl_status_handles_missing_root():
    """On macOS / BSD / containers with /sys masked, rapl_status() must
    return ("missing", None, None) without crashing."""
    from src.power import rapl_status
    s, p, h = rapl_status()
    assert s in {"ok", "permission_denied", "missing"}
    if s == "missing":
        assert p is None
        assert h is None
    elif s == "permission_denied":
        assert p is not None
        assert h is not None
        assert "chmod" in h


def test_rapl_hint_renders_in_report():
    """When the probe reports permission_denied with a hint, the report's
    System section must include a 'Power hint' row with the hint text."""
    from src.report import render
    from src.system import SystemInfo
    hint = ("RAPL energy file present but not readable "
            "(/sys/class/powercap/intel-rapl:0/energy_uj); one-time fix: "
            "sudo chmod a+r /sys/class/powercap/intel-rapl*/energy_uj")
    info = SystemInfo(
        hostname="mlbox", os_name="Linux", os_version="5.15", kernel="x",
        arch="x86_64", distro="Ubuntu 20.04", cpu_model="Xeon",
        cpu_count_physical=20, cpu_count_logical=40, ram_total_gb=251.6,
        ram_avail_gb=200.0, disk_total_gb=2000.0, disk_free_gb=1500.0,
        python_version="3.12.1", python_impl="CPython",
        libs={"torch": "2.4"}, has_cuda=False, has_rapl=False,
        rapl_status="permission_denied", rapl_hint=hint,
    )
    html = render([], info, total_seconds=0.0)
    assert "Power hint" in html
    assert "chmod a+r" in html
    assert "root-only" in html  # the inline disclosure bit


# ---- CLI ----

def test_cli_help_exits_zero():
    from src.cli import app
    runner = CliRunner()
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "MishaBench" in result.output


def test_cli_info_runs():
    from src.cli import app
    runner = CliRunner()
    result = runner.invoke(app, ["info"])
    assert result.exit_code == 0


def test_cli_list_runs():
    from src.cli import app
    runner = CliRunner()
    result = runner.invoke(app, ["list", "--suites", "data"])
    assert result.exit_code == 0
    assert "data." in result.output
