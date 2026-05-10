# SPDX-FileCopyrightText: Copyright (c) 2026 Mikhail Yurasov
# SPDX-License-Identifier: Apache-2.0

"""HTML report renderer.

Single self-contained .html file -- inline CSS, inline SVG bar charts,
no external assets, no JS dependencies. Designed to be rsync'd back
from a remote host and opened locally without internet.

Layout:
  - Header (host, time, total runtime, label tags)
  - System summary (cpu / ram / gpus / cuda / libraries)
  - Per-suite section with grouped bar charts (CPU vs CUDA where the
    same workload was measured on both devices)
  - Detailed results table (all bench rows including skips)
  - Footer (version + reproducibility hint)
"""

from __future__ import annotations

import html
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from . import __version__
from .runner import BenchResult
from .scoring import ScoreReport
from .scoring import compute as compute_scores
from .system import SystemInfo

_CSS = """
:root {
  --fg: #1a1a1a; --muted: #6b7280; --bg: #ffffff; --panel: #f7f7f9;
  --accent: #76b900; --accent-soft: #e3f1c8;
  --cpu: #5b8def; --gpu: #76b900; --mps: #c084fc; --bad: #ef4444;
  --border: #e5e7eb;
}
* { box-sizing: border-box; }
body { margin: 0; font: 14px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI",
       Roboto, Oxygen-Sans, Ubuntu, sans-serif; color: var(--fg); background: var(--bg); }
.wrap { max-width: 1100px; margin: 0 auto; padding: 24px 28px; }
header { border-bottom: 4px solid var(--accent); padding-bottom: 14px; margin-bottom: 24px; }
header h1 { margin: 0 0 4px; font-size: 22px; letter-spacing: -.01em; }
header .sub { color: var(--muted); font-size: 13px; }
.tag { display: inline-block; background: var(--accent-soft); color: #2c4f00;
       border-radius: 999px; padding: 1px 10px; margin-right: 6px; font-size: 11px;
       font-weight: 600; }
section { margin: 28px 0; }
h2 { font-size: 18px; margin: 0 0 12px; padding-bottom: 6px; border-bottom: 1px solid var(--border); }
h3 { font-size: 14px; margin: 18px 0 8px; color: #444; }
.grid { display: grid; grid-template-columns: 200px 1fr; gap: 4px 12px; }
.grid div:nth-child(odd) { color: var(--muted); }
table { width: 100%; border-collapse: collapse; font-size: 13px; margin-top: 8px; }
th, td { text-align: left; padding: 6px 8px; border-bottom: 1px solid var(--border); }
th { background: var(--panel); font-weight: 600; }
tr.skip td { color: var(--muted); font-style: italic; }
tr.fail td { color: var(--bad); }
.dev-cpu { color: var(--cpu); font-weight: 600; }
.dev-cuda { color: var(--gpu); font-weight: 600; }
.dev-mps { color: var(--mps); font-weight: 600; }
.bar-row { display: grid; grid-template-columns: 220px 1fr 100px; gap: 8px;
            align-items: center; padding: 4px 0; }
.bar-name { font-size: 12px; color: var(--fg); white-space: nowrap;
            overflow: hidden; text-overflow: ellipsis; }
.bar { height: 16px; border-radius: 3px; background: var(--cpu); }
.bar.gpu { background: var(--gpu); }
.bar.mps { background: var(--mps); }
.bar.skip { background: #d1d5db; }
.value { font-variant-numeric: tabular-nums; font-size: 12px; color: #333; }
.legend { font-size: 12px; color: var(--muted); margin: 6px 0 14px; }
.legend span { margin-right: 12px; }
.swatch { display: inline-block; width: 10px; height: 10px; border-radius: 2px;
          vertical-align: middle; margin-right: 4px; }
/* Scores grid: device columns (CPU / CUDA / MPS) x suite rows (data / cv / llm / total) */
.scores { width: 100%; border-collapse: collapse; margin: 8px 0 4px; }
.scores th, .scores td { padding: 8px 10px; text-align: right; border-bottom: 1px solid var(--border); }
.scores th:first-child, .scores td:first-child { text-align: left; font-weight: 600; }
.scores thead th { background: var(--panel); font-weight: 700; }
.scores .total td { font-weight: 700; background: #fafafa; border-top: 2px solid var(--border); }
.scores .num { font-variant-numeric: tabular-nums; font-size: 15px; }
.scores .small { font-size: 11px; color: var(--muted); display: block; }
.scores .est { color: var(--muted); }
.scores .est::after { content: " *"; }
.note { font-size: 12px; color: var(--muted); margin-top: -4px; }
footer { margin-top: 40px; padding-top: 14px; border-top: 1px solid var(--border);
         color: var(--muted); font-size: 12px; }
code { background: #f3f4f6; padding: 1px 5px; border-radius: 3px; font-size: 12px; }
"""


_PAGE = """<!DOCTYPE html>
<html lang="en"><head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>MishaBench report -- {hostname}</title>
<style>{css}</style>
</head><body><div class="wrap">
{body}
</div></body></html>
"""


def _esc(x: Any) -> str:
    return html.escape(str(x), quote=True)


def _fmt(v: Any, unit: str = "") -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        if abs(v) >= 1000:
            return f"{v:,.0f} {unit}".strip()
        if abs(v) >= 10:
            return f"{v:.1f} {unit}".strip()
        return f"{v:.3f} {unit}".strip()
    return f"{v} {unit}".strip()


def _device_class(d: str) -> str:
    if d.startswith("cuda"):
        return "dev-cuda"
    if d == "mps":
        return "dev-mps"
    return "dev-cpu"


def _bar_color(d: str) -> str:
    if d.startswith("cuda"):
        return "gpu"
    if d == "mps":
        return "mps"
    return ""


def _system_grid(info: SystemInfo) -> str:
    rows: list[tuple[str, str]] = [
        ("Host", info.hostname),
        ("OS",  f"{info.distro or info.os_name} -- {info.os_version} ({info.arch})"),
        ("CPU", f"{info.cpu_model} ({info.cpu_count_physical}p / {info.cpu_count_logical}t)"),
        ("RAM", f"{info.ram_total_gb} GiB total ({info.ram_avail_gb} GiB free)"),
        ("Disk (/)", f"{info.disk_free_gb} GiB free of {info.disk_total_gb} GiB"),
        ("Python", f"{info.python_impl} {info.python_version}"),
    ]
    if info.gpus:
        for g in info.gpus:
            mem = f"{g.memory_mib} MiB" if g.memory_mib else "memory n/a (unified)"
            rows.append(
                (f"GPU [{g.index}]",
                 f"{g.name} -- {mem}, driver {g.driver}, "
                 f"sm_{g.compute_cap or '?'}")
            )
        rows.append(("CUDA runtime",
                     info.cuda_runtime or "(torch missing)"))
        rows.append(("Driver", info.nvidia_driver or "?"))
    else:
        rows.append(("GPU", "no NVIDIA GPU detected"))
    if info.has_mps:
        rows.append(("MPS", "available (Apple Silicon)"))

    # Power-monitor source disclosure -- the Scores section attributes
    # wattage to specific sources, and this row says what they are.
    power_sources: list[str] = []
    if info.gpus:
        power_sources.append("nvidia-smi (real)")
    if info.has_rapl:
        power_sources.append("Intel RAPL (real)")
    elif info.rapl_status == "permission_denied":
        power_sources.append("Intel RAPL (present, root-only -- see hint below)")
    if info.apple_tdp_w is not None:
        power_sources.append(f"Apple chip TDP table = {info.apple_tdp_w:.0f} W (estimated)")
    rows.append(("Power source", ", ".join(power_sources) if power_sources else "n/a"))
    if info.rapl_hint:
        rows.append(("Power hint", info.rapl_hint))

    libs_inline = ", ".join(
        f"{name}={ver}" for name, ver in sorted(info.libs.items()) if ver
    )
    rows.append(("Libraries", libs_inline or "(none)"))

    inner = "".join(f"<div>{_esc(k)}</div><div>{_esc(v)}</div>" for k, v in rows)
    return f'<div class="grid">{inner}</div>'


def _suite_charts(rows: list[BenchResult]) -> str:
    """Bars per suite. Higher metric = better in bench convention,
    but we render absolute values (not normalized) so reports are
    directly comparable across hosts."""
    if not rows:
        return "<p>(no measurements)</p>"
    valid = [r for r in rows if r.ok and isinstance(r.value, (int, float))]
    if not valid:
        return "<p>(all measurements skipped or failed)</p>"
    max_v = max(float(r.value) for r in valid)  # type: ignore[arg-type]
    if max_v <= 0:
        return "<p>(no positive measurements)</p>"

    out: list[str] = []
    for r in rows:
        if not (r.ok and isinstance(r.value, (int, float))):
            label = f"<span class='value'>{_esc(r.error or 'skipped')}</span>"
            out.append(
                f"<div class='bar-row'>"
                f"<div class='bar-name' title='{_esc(r.id)}'>"
                f"<span class='{_device_class(r.device)}'>{_esc(r.device)}</span> "
                f"{_esc(r.name)}</div>"
                f"<div><div class='bar skip' style='width:0'></div></div>"
                f"{label}</div>"
            )
            continue
        pct = max(2.0, 100.0 * float(r.value) / max_v)
        cls = _bar_color(r.device)
        out.append(
            f"<div class='bar-row'>"
            f"<div class='bar-name' title='{_esc(r.id)}'>"
            f"<span class='{_device_class(r.device)}'>{_esc(r.device)}</span> "
            f"{_esc(r.name)}</div>"
            f"<div><div class='bar {cls}' style='width:{pct:.1f}%'></div></div>"
            f"<div class='value'>{_fmt(r.value, r.unit)}</div>"
            f"</div>"
        )

    return "".join(out)


# Notes keys whose values are internal-shape (dict / large) and should
# not be inlined into the All measurements table. They're still in the
# JSONL for downstream processing.
_HIDDEN_NOTE_KEYS = {"power"}


def _detailed_table(rows: list[BenchResult]) -> str:
    head = (
        "<tr><th>id</th><th>name</th><th>device</th>"
        "<th>metric</th><th>value</th><th>seconds</th><th>watts</th>"
        "<th>iters</th><th>note</th></tr>"
    )
    body: list[str] = []
    for r in rows:
        cls = "" if r.ok else ("skip" if (r.error or "").startswith(("budget", "CUDA", "MPS", "cudf", "cupy")) else "fail")
        if r.error and not r.ok:
            note = r.error
        elif r.notes:
            visible = {k: v for k, v in r.notes.items() if k not in _HIDDEN_NOTE_KEYS}
            note = ", ".join(f"{k}={v}" for k, v in visible.items())
        else:
            note = ""

        watts = "-"
        if r.notes and r.notes.get("avg_watts") is not None:
            est = " *" if r.notes.get("power_estimated") else ""
            watts = f"{r.notes['avg_watts']:.1f}{est}"

        body.append(
            f"<tr class='{cls}'>"
            f"<td><code>{_esc(r.id)}</code></td>"
            f"<td>{_esc(r.name)}</td>"
            f"<td class='{_device_class(r.device)}'>{_esc(r.device)}</td>"
            f"<td>{_esc(r.metric)}</td>"
            f"<td>{_esc(_fmt(r.value, r.unit))}</td>"
            f"<td>{_esc(f'{r.seconds:.2f}')}</td>"
            f"<td>{_esc(watts)}</td>"
            f"<td>{_esc(r.iters)}</td>"
            f"<td>{_esc(note[:140])}</td>"
            f"</tr>"
        )
    return f"<table>{head}{''.join(body)}</table>"


def _legend() -> str:
    return (
        "<div class='legend'>"
        "<span><span class='swatch' style='background:var(--cpu)'></span>CPU</span>"
        "<span><span class='swatch' style='background:var(--gpu)'></span>CUDA</span>"
        "<span><span class='swatch' style='background:var(--mps)'></span>MPS</span>"
        "<span>Bars are absolute values within a chart; cross-host comparisons stay honest.</span>"
        "</div>"
    )


_DEVICE_LABEL = {"cpu": "CPU", "cuda": "CUDA (NVIDIA)", "mps": "MPS (Apple Silicon)"}
_DEVICE_ORDER = ("cpu", "cuda", "mps")
_SUITE_ORDER = ("data", "cv", "llm")


def _scores_section(scores: ScoreReport) -> str:
    """Top-level scores: rows are suites, columns are devices. Each cell
    shows the suite/device score with a tiny power line below when
    wattage is known.

    No grand-total row -- a geomean across suites whose units differ by
    orders of magnitude (M rows/s, img/s, tok/s, GFLOPS) is dimensionally
    meaningless. The per-suite scores ARE meaningful (geomean within a
    similar-unit family) and that's where the comparison stops."""
    # Discover devices from per_device_per_suite (per_device_total is no
    # longer populated; see scoring.py for why).
    devices = [d for d in _DEVICE_ORDER if d in scores.per_device_per_suite]
    if not devices:
        return "<p class='note'>(no scoreable measurements)</p>"

    head = "<tr><th>Suite</th>" + "".join(
        f"<th>{_esc(_DEVICE_LABEL[d])}</th>" for d in devices
    ) + "</tr>"

    body_rows: list[str] = []
    for suite in _SUITE_ORDER:
        if not any(suite in scores.per_device_per_suite.get(d, {}) for d in devices):
            continue  # suite didn't run -- skip the row entirely
        cells: list[str] = [f"<td>{_esc(suite)}</td>"]
        for d in devices:
            ds = scores.per_device_per_suite.get(d, {}).get(suite)
            if ds is None or ds.score is None:
                cells.append("<td class='num'>-</td>")
                continue
            est_cls = " est" if ds.estimated_power else ""
            sub = ""
            if ds.avg_watts is not None and ds.pts_per_watt is not None:
                sub = (f"<span class='small{est_cls}'>"
                       f"{ds.avg_watts:.1f} W -- {ds.pts_per_watt:.1f} pts/W"
                       f"</span>")
            cells.append(f"<td class='num'>{ds.score:,.0f}{sub}</td>")
        body_rows.append("<tr>" + "".join(cells) + "</tr>")

    has_estimated = any(
        s.estimated_power
        for d in scores.per_device_per_suite.values()
        for s in d.values()
    )
    note = ""
    if has_estimated:
        note = ("<p class='note'>* Power marked with a star is a chip-name TDP "
                "estimate (Apple Silicon, where sudo-free per-process power is "
                "not exposed). NVIDIA wattage and Linux RAPL CPU wattage are "
                "real, polled measurements.</p>")
    return (
        "<table class='scores'><thead>"
        + head
        + "</thead><tbody>"
        + "".join(body_rows)
        + "</tbody></table>"
        + note
        + "<p class='note'>Score = geomean of per-bench raw values within a "
          "(suite, device) cell, scaled by 1000. Higher is better. "
          "pts/W = score / avg watts. No cross-suite total is reported -- "
          "a geomean across suites with different units (rows/s vs img/s "
          "vs tok/s vs GFLOPS) would be dimensionally meaningless.</p>"
    )


def render(results: list[BenchResult], info: SystemInfo,
           total_seconds: float, labels: dict[str, str] | None = None) -> str:
    by_suite: dict[str, list[BenchResult]] = defaultdict(list)
    for r in results:
        by_suite[r.suite].append(r)

    scores = compute_scores(results)

    label_html = "".join(
        f"<span class='tag'>{_esc(k)}={_esc(v)}</span>"
        for k, v in (labels or {}).items()
    )

    suite_blocks: list[str] = []
    for suite_name in _SUITE_ORDER:
        if suite_name not in by_suite:
            continue
        rows = by_suite[suite_name]
        suite_blocks.append(
            f"<section><h2>Suite: {_esc(suite_name)}</h2>"
            f"{_legend()}"
            f"{_suite_charts(rows)}"
            f"</section>"
        )

    body = (
        f"<header>"
        f"<h1>MishaBench report</h1>"
        f"<div class='sub'>{_esc(info.hostname)} -- "
        f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} -- "
        f"total wall {total_seconds/60:.1f} min "
        f"({len(results)} benchmarks)</div>"
        f"<div style='margin-top:6px'>{label_html}</div>"
        f"</header>"

        f"<section><h2>Scores</h2>{_scores_section(scores)}</section>"

        f"<section><h2>System</h2>{_system_grid(info)}</section>"

        + "".join(suite_blocks)

        + f"<section><h2>All measurements</h2>{_detailed_table(results)}</section>"

        + f"<footer>MishaBench v{__version__}. "
          f"Re-render this report from the JSONL with: "
          f"<code>./mishabench report &lt;results-dir&gt;</code>. "
          f"</footer>"
    )

    return _PAGE.format(css=_CSS, hostname=_esc(info.hostname), body=body)


def write_report(results: list[BenchResult], info: SystemInfo,
                 out_dir: Path, total_seconds: float,
                 labels: dict[str, str] | None = None) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    html_text = render(results, info, total_seconds, labels=labels)
    p = out_dir / "report.html"
    p.write_text(html_text, encoding="utf-8")
    return p


def load_results(results_dir: Path) -> tuple[list[BenchResult], SystemInfo]:
    """Reload JSONL + system.json from a results directory (used by `report` subcommand)."""
    rows: list[BenchResult] = []
    with (results_dir / "results.jsonl").open(encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            rows.append(BenchResult(**d))
    info_d = json.loads((results_dir / "system.json").read_text(encoding="utf-8"))
    # Reconstruct GpuInfo objects
    from .system import GpuInfo
    info_d["gpus"] = [GpuInfo(**g) for g in info_d.get("gpus", [])]
    info = SystemInfo(**info_d)
    return rows, info
