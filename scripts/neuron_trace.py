#!/usr/bin/env python3
"""
neuron_trace.py — EEG-style time-series tracer for AWS NeuronCore utilization.

Wraps the `neuron-monitor` tool that ships with the Neuron SDK, parses its
JSON stream, and renders a stacked per-core plot (one row per NeuronCore,
shared time axis — like an EEG).

Usage
-----
Standalone capture (default 30 s @ 1 Hz, writes JSONL + PNG):

    python scripts/neuron_trace.py

Capture only, no plot:

    python scripts/neuron_trace.py --duration 60 --period 0.5 \\
        --output traces/run.jsonl --no-plot

Live rolling view (requires matplotlib with an interactive backend):

    python scripts/neuron_trace.py --live --window 30

Re-plot a previously captured JSONL:

    python scripts/neuron_trace.py --replot traces/run.jsonl

Dependencies
------------
- neuron-monitor on PATH (provided by the Neuron SDK)
- matplotlib (optional, only needed for plotting / --live)

This script intentionally has no dependency on the lqcd_neuron package so it
can be copied to any Inf2 / Trn1 host and run on its own.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator


# ---------------------------------------------------------------------------
# neuron-monitor wrangling
# ---------------------------------------------------------------------------

def _build_monitor_config(period_s: float) -> dict:
    """Minimal neuron-monitor config: per-core counters + system metrics."""
    period = f"{max(period_s, 0.1):.3f}s"
    return {
        "period": period,
        "neuron_runtimes": [
            {
                "tag_filter": ".*",
                "metrics": [
                    {"type": "neuroncore_counters"},
                    {"type": "memory_used"},
                ],
            }
        ],
        "system_metrics": [
            {"type": "vcpu_usage"},
            {"type": "memory_info"},
        ],
    }


def _spawn_monitor(period_s: float) -> subprocess.Popen:
    if shutil.which("neuron-monitor") is None:
        sys.exit(
            "error: `neuron-monitor` not found on PATH. "
            "Install the Neuron SDK or run this on an Inf2/Trn1 host."
        )

    cfg = _build_monitor_config(period_s)
    cfg_file = tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, prefix="neuron_trace_"
    )
    json.dump(cfg, cfg_file)
    cfg_file.flush()
    cfg_file.close()

    return subprocess.Popen(
        ["neuron-monitor", "-c", cfg_file.name],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )


# ---------------------------------------------------------------------------
# Sample extraction
# ---------------------------------------------------------------------------

@dataclass
class Sample:
    """One time-step of per-core utilization."""
    t: float                                 # seconds since capture start
    wall: str                                # ISO timestamp from neuron-monitor
    cores: dict[int, float] = field(default_factory=dict)  # core_id -> util [0,1]


def _extract_sample(record: dict, t0: float) -> Sample | None:
    """Pull per-NeuronCore utilization out of a neuron-monitor JSON record."""
    runtimes = record.get("neuron_runtime_data") or []
    cores: dict[int, float] = {}

    for rt in runtimes:
        report = rt.get("report") or {}
        nc = report.get("neuroncore_counters") or {}
        per_core = nc.get("neuroncores_in_use") or {}
        for core_id_str, info in per_core.items():
            try:
                cid = int(core_id_str)
            except ValueError:
                continue
            # neuron-monitor exposes utilization either as `neuroncore_utilization`
            # (percent) or as a counter delta — we accept either.
            util = info.get("neuroncore_utilization")
            if util is None:
                util = info.get("utilization", 0.0)
            try:
                util = float(util)
            except (TypeError, ValueError):
                util = 0.0
            # Normalize to [0, 1] regardless of whether SDK reports % or fraction.
            if util > 1.5:
                util /= 100.0
            cores[cid] = max(0.0, min(1.0, util))

    if not cores:
        return None

    wall = record.get("timestamp", "")
    return Sample(t=time.monotonic() - t0, wall=wall, cores=cores)


# ---------------------------------------------------------------------------
# Capture loop
# ---------------------------------------------------------------------------

def capture(
    duration_s: float,
    period_s: float,
    output: Path | None,
) -> list[Sample]:
    """Run neuron-monitor for `duration_s`, return collected samples."""
    proc = _spawn_monitor(period_s)
    samples: list[Sample] = []
    t0 = time.monotonic()
    out_fh = output.open("w") if output else None

    print(
        f"[neuron_trace] capturing for {duration_s:.1f}s @ {period_s:.2f}s "
        f"period (Ctrl-C to stop early)",
        file=sys.stderr,
    )

    try:
        assert proc.stdout is not None
        for line in proc.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue

            sample = _extract_sample(rec, t0)
            if sample is None:
                continue
            samples.append(sample)
            if out_fh is not None:
                out_fh.write(
                    json.dumps(
                        {"t": sample.t, "wall": sample.wall, "cores": sample.cores}
                    )
                    + "\n"
                )
                out_fh.flush()

            if time.monotonic() - t0 >= duration_s:
                break
    except KeyboardInterrupt:
        print("[neuron_trace] interrupted, finalizing...", file=sys.stderr)
    finally:
        if proc.poll() is None:
            proc.send_signal(signal.SIGINT)
            try:
                proc.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                proc.kill()
        if out_fh is not None:
            out_fh.close()

    print(f"[neuron_trace] captured {len(samples)} samples", file=sys.stderr)
    return samples


def load_jsonl(path: Path) -> list[Sample]:
    samples: list[Sample] = []
    with path.open() as fh:
        for line in fh:
            rec = json.loads(line)
            samples.append(
                Sample(
                    t=rec["t"],
                    wall=rec.get("wall", ""),
                    cores={int(k): float(v) for k, v in rec["cores"].items()},
                )
            )
    return samples


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _all_core_ids(samples: list[Sample]) -> list[int]:
    seen: set[int] = set()
    for s in samples:
        seen.update(s.cores.keys())
    return sorted(seen)


def plot_static(samples: list[Sample], out_png: Path, title: str) -> None:
    if not samples:
        print("[neuron_trace] no samples to plot", file=sys.stderr)
        return
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        sys.exit("error: matplotlib is required for plotting (pip install matplotlib)")

    cores = _all_core_ids(samples)
    n = len(cores)
    times = [s.t for s in samples]
    series = {c: [s.cores.get(c, 0.0) for s in samples] for c in cores}

    fig, axes = plt.subplots(
        n, 1, figsize=(11, max(1.2 * n, 3.0)), sharex=True, squeeze=False
    )
    axes = axes[:, 0]
    for ax, c in zip(axes, cores):
        ax.fill_between(times, 0, series[c], alpha=0.35, linewidth=0)
        ax.plot(times, series[c], linewidth=1.0)
        ax.set_ylim(0, 1.05)
        ax.set_yticks([0, 1])
        ax.set_ylabel(f"NC{c}", rotation=0, ha="right", va="center", labelpad=18)
        ax.grid(True, alpha=0.25)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    axes[-1].set_xlabel("time (s)")
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_png, dpi=140)
    print(f"[neuron_trace] wrote {out_png}", file=sys.stderr)


def live_view(period_s: float, window_s: float) -> None:
    """Rolling-window EEG-style live plot. Requires interactive matplotlib."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        sys.exit("error: matplotlib is required for --live (pip install matplotlib)")

    proc = _spawn_monitor(period_s)
    t0 = time.monotonic()
    samples: list[Sample] = []

    plt.ion()
    fig = plt.figure(figsize=(11, 6))
    fig.canvas.manager.set_window_title("neuron_trace — live")
    axes: list = []
    lines: dict[int, object] = {}
    fills: dict[int, object] = {}
    known_cores: list[int] = []

    def _rebuild_axes(cores: list[int]) -> None:
        nonlocal axes, lines, fills
        fig.clear()
        axes = []
        lines = {}
        fills = {}
        n = len(cores)
        for i, c in enumerate(cores):
            ax = fig.add_subplot(n, 1, i + 1)
            (line,) = ax.plot([], [], linewidth=1.0)
            ax.set_ylim(0, 1.05)
            ax.set_yticks([0, 1])
            ax.set_ylabel(f"NC{c}", rotation=0, ha="right", va="center", labelpad=18)
            ax.grid(True, alpha=0.25)
            for spine in ("top", "right"):
                ax.spines[spine].set_visible(False)
            if i < n - 1:
                ax.set_xticklabels([])
            else:
                ax.set_xlabel("time (s)")
            axes.append(ax)
            lines[c] = line
        fig.suptitle("NeuronCore utilization (live)", fontsize=11)
        fig.tight_layout(rect=[0, 0, 1, 0.97])

    print("[neuron_trace] live mode — close the window or Ctrl-C to stop", file=sys.stderr)
    try:
        assert proc.stdout is not None
        for line in proc.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            s = _extract_sample(rec, t0)
            if s is None:
                continue
            samples.append(s)

            cutoff = s.t - window_s
            samples = [x for x in samples if x.t >= cutoff]

            cores = _all_core_ids(samples)
            if cores != known_cores:
                known_cores = cores
                _rebuild_axes(cores)

            ts = [x.t for x in samples]
            for c, ln in lines.items():
                ys = [x.cores.get(c, 0.0) for x in samples]
                ln.set_data(ts, ys)
            for ax in axes:
                ax.set_xlim(max(0.0, s.t - window_s), max(window_s, s.t))
            fig.canvas.draw_idle()
            fig.canvas.flush_events()

            if not plt.fignum_exists(fig.number):
                break
    except KeyboardInterrupt:
        pass
    finally:
        if proc.poll() is None:
            proc.send_signal(signal.SIGINT)
            try:
                proc.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                proc.kill()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _default_output_paths() -> tuple[Path, Path]:
    out_dir = Path("traces")
    out_dir.mkdir(exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    return out_dir / f"neuron_{stamp}.jsonl", out_dir / f"neuron_{stamp}.png"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Record and plot per-NeuronCore utilization over time.",
    )
    p.add_argument("--duration", type=float, default=30.0,
                   help="Capture duration in seconds (default: 30).")
    p.add_argument("--period", type=float, default=1.0,
                   help="Sampling period in seconds (default: 1.0, min 0.1).")
    p.add_argument("--output", type=Path, default=None,
                   help="JSONL file to write (default: traces/neuron_<ts>.jsonl).")
    p.add_argument("--plot", type=Path, default=None,
                   help="PNG file to write (default: traces/neuron_<ts>.png).")
    p.add_argument("--no-plot", action="store_true",
                   help="Capture only, do not produce a PNG.")
    p.add_argument("--title", type=str, default=None,
                   help="Plot title (default: hostname + timestamp).")
    p.add_argument("--live", action="store_true",
                   help="Open a rolling-window live plot instead of capturing.")
    p.add_argument("--window", type=float, default=30.0,
                   help="Window size in seconds for --live (default: 30).")
    p.add_argument("--replot", type=Path, default=None,
                   help="Skip capture; replot an existing JSONL trace.")
    args = p.parse_args(argv)

    if args.live:
        live_view(period_s=args.period, window_s=args.window)
        return 0

    if args.replot is not None:
        samples = load_jsonl(args.replot)
        out_png = args.plot or args.replot.with_suffix(".png")
        title = args.title or f"{args.replot.name}"
        plot_static(samples, out_png, title)
        return 0

    out_jsonl, out_png_default = _default_output_paths()
    out_jsonl = args.output or out_jsonl
    out_png = args.plot or out_png_default

    samples = capture(
        duration_s=args.duration,
        period_s=args.period,
        output=out_jsonl,
    )

    if not args.no_plot:
        title = args.title or f"NeuronCore utilization — {os.uname().nodename} — {time.strftime('%Y-%m-%d %H:%M:%S')}"
        plot_static(samples, out_png, title)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
