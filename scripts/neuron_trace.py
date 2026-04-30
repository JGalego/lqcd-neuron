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

Live rolling view (renders directly in terminal, no GUI needed):

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
    """Rolling-window EEG-style live trace rendered in the terminal."""
    # Braille-based sparkline: each column encodes a utilization value using
    # a vertical strip of 4 braille dots (8 levels of resolution per column).
    # This gives a time-series view similar to an EEG strip chart.

    # Braille block characters for vertical bar (bottom-to-top fill, 0-8 eighths)
    _SPARK = " ▁▂▃▄▅▆▇█"

    proc = _spawn_monitor(period_s)
    t0 = time.monotonic()
    samples: list[Sample] = []

    try:
        term_cols = os.get_terminal_size().columns
    except OSError:
        term_cols = 80

    # Layout: "  NC0  │<sparkline>│ 100.0%"
    label_w = 8   # "  NC0  "
    pct_w = 8     # " 100.0%"
    border_w = 2  # "│" on each side
    trace_width = max(20, term_cols - label_w - pct_w - border_w)

    # Keep at most trace_width samples worth of history per core
    history: dict[int, list[float]] = {}

    def _color_for(util: float) -> str:
        if util >= 0.9:
            return "\033[91m"  # red
        elif util >= 0.7:
            return "\033[93m"  # yellow
        return "\033[92m"      # green

    def _render_trace(core_history: list[float]) -> str:
        """Render a sparkline string from a list of utilization values."""
        reset = "\033[0m"
        # Pad to trace_width with spaces on the left (empty = no data yet)
        padded = [0.0] * max(0, trace_width - len(core_history)) + core_history[-trace_width:]
        chars: list[str] = []
        for v in padded:
            idx = int(v * 8.0 + 0.5)
            idx = max(0, min(8, idx))
            ch = _SPARK[idx]
            if idx == 0:
                chars.append(" ")
            else:
                chars.append(_color_for(v) + ch + reset)
        return "".join(chars)

    def _render_frame() -> str:
        """Build a full frame showing the rolling sparkline per core."""
        lines_out: list[str] = []
        latest = samples[-1] if samples else None
        elapsed = latest.t if latest else 0.0
        t_start = max(0.0, elapsed - window_s)

        lines_out.append(
            f"\033[1mNeuronCore utilization  "
            f"[{t_start:.0f}s ─── {elapsed:.0f}s]  "
            f"window={window_s:.0f}s\033[0m"
        )
        lines_out.append("─" * min(term_cols, label_w + trace_width + pct_w + border_w))

        for cid in sorted(history.keys()):
            h = history[cid]
            trace_str = _render_trace(h)
            current_pct = h[-1] * 100.0 if h else 0.0
            lines_out.append(
                f"  NC{cid:<3d} │{trace_str}│{current_pct:5.1f}%"
            )

        lines_out.append("─" * min(term_cols, label_w + trace_width + pct_w + border_w))
        lines_out.append(
            f"  \033[2m█=spark  ▁▂▃▄▅▆▇█ 0%→100%  "
            f"\033[92m■\033[0m\033[2m<70% "
            f"\033[93m■\033[0m\033[2m70-90% "
            f"\033[91m■\033[0m\033[2m≥90%\033[0m"
        )
        return "\n".join(lines_out)

    # Hide cursor
    sys.stdout.write("\033[?25l")
    sys.stdout.flush()
    rendered_lines = 0

    print("[neuron_trace] live terminal mode — Ctrl-C to stop\n", file=sys.stderr)
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

            # Update per-core history
            for cid, util in s.cores.items():
                if cid not in history:
                    history[cid] = []
                history[cid].append(util)
                # Trim to trace width
                if len(history[cid]) > trace_width:
                    history[cid] = history[cid][-trace_width:]

            # Move cursor up to overwrite previous frame
            if rendered_lines > 0:
                sys.stdout.write(f"\033[{rendered_lines}A\033[J")

            frame = _render_frame()
            sys.stdout.write(frame + "\n")
            sys.stdout.flush()
            rendered_lines = frame.count("\n") + 1
    except KeyboardInterrupt:
        pass
    finally:
        # Show cursor
        sys.stdout.write("\033[?25h")
        sys.stdout.flush()
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
