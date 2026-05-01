"""
Example 4 — Dslash throughput benchmark on Neuron / CPU.

Measures sustained applications-per-second for WilsonDirac.forward() at
various lattice sizes, both with and without AoT Neuron compilation.

Run on CPU:
    python examples/bench_dslash.py

Run on Trn1 / Inf2 (torch-neuronx must be installed):
    python examples/bench_dslash.py --neuron

Output (example on Inf2):
    Lattice 4x4x4x4  | CPU   :   142 apps/s
    Lattice 4x4x4x4  | Neuron:  2840 apps/s  (×20)
    Lattice 8x4x4x4  | CPU   :    71 apps/s
    Lattice 8x4x4x4  | Neuron:  1420 apps/s  (×20)
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from typing import List, Tuple

import torch

from lqcd_neuron.core import ColorSpinorField, GaugeField, LatticeGeometry
from lqcd_neuron.dirac import WilsonDirac
from lqcd_neuron.neuron import NeuronCompiler, get_device, is_neuron_available


# ---------------------------------------------------------------------------
# Lattice sizes to benchmark
# ---------------------------------------------------------------------------

LATTICE_SIZES: List[Tuple[int, int, int, int]] = [
    # Microbenchmark / dispatch-overhead regime (toy volumes).
    (4, 4, 4, 4),
    (8, 4, 4, 4),
    (8, 8, 4, 4),
    (8, 8, 8, 4),
    # Sustained-throughput regime: launch cost is amortised, the
    # measurement reflects HBM bandwidth and MXU utilisation.
    (16, 16, 16, 16),   # V = 65,536  sites
    (24, 24, 24, 24),   # V = 331,776 sites
    (32, 32, 32, 32),   # V = 1,048,576 sites - may exceed single-core HBM
]

# Standard Wilson Dslash flop count (real flops per site, per RHS).
# 1320 = 8 directions x (3x3 colour matvec + 4x4 spin matvec + accumulate),
# matching the convention of Babich et al., SC'11 (arXiv:1109.2935).
FLOPS_PER_SITE = 1320

# Streaming bytes per site, per RHS, when the gauge field is baked into
# the .neff (no PCIe traffic for U): one spinor in + one spinor out =
# 2 * (Ns * Nc * 2 reals) = 2 * 24 = 48 reals = 192 B at FP32 (96 B at BF16).
BYTES_PER_SITE_FP32 = 192
BYTES_PER_SITE_BF16 = 96

WARMUP_ITERS = 5
BENCH_ITERS  = 50
BATCH_SIZE   = 8   # multi-RHS batch size for the batched Neuron column


def benchmark_one(
    matvec,
    psi: torch.Tensor,
    U: torch.Tensor,
    n: int = BENCH_ITERS,
    warmup: int = WARMUP_ITERS,
    label: str = "",
) -> float:
    """Return applications/second for ``matvec(psi, U)``."""
    with torch.inference_mode():
        for _ in range(warmup):
            _ = matvec(psi, U)

        torch.cuda.synchronize() if psi.is_cuda else None

        t0 = time.perf_counter()
        for _ in range(n):
            out = matvec(psi, U)
        elapsed = time.perf_counter() - t0
    return n / elapsed


def benchmark_batched(
    matvec,
    psi_batch: torch.Tensor,
    n: int = BENCH_ITERS,
    warmup: int = WARMUP_ITERS,
) -> float:
    """Return per-RHS applications/second for a batched ``matvec(psi)``.

    Each call applies the operator to ``psi_batch.shape[0]`` right-hand
    sides simultaneously, so we report ``n * B / elapsed`` for an
    apples-to-apples comparison with the single-RHS columns.
    """
    B = psi_batch.shape[0]
    with torch.inference_mode():
        for _ in range(warmup):
            _ = matvec(psi_batch)

        t0 = time.perf_counter()
        for _ in range(n):
            out = matvec(psi_batch)
        elapsed = time.perf_counter() - t0
    return n * B / elapsed


def derived_metrics(aps: float, shape: Tuple[int, int, int, int],
                    bf16: bool = False) -> Tuple[float, float]:
    """Return (GFLOP/s, GB/s) for a measured apps-per-second number.

    GFLOP/s uses the standard Wilson Dslash count of 1320 real flops/site.
    GB/s is the streaming spinor in+out traffic (gauge field baked in HBM).
    """
    V = shape[0] * shape[1] * shape[2] * shape[3]
    gflops = aps * V * FLOPS_PER_SITE / 1e9
    bytes_per_site = BYTES_PER_SITE_BF16 if bf16 else BYTES_PER_SITE_FP32
    gbps = aps * V * bytes_per_site / 1e9
    return gflops, gbps


def run(use_neuron: bool, mass: float = 0.1, fused: bool = True) -> None:
    dtype = torch.complex64
    nc    = 3

    if use_neuron and not is_neuron_available():
        print("WARNING: --neuron requested but no Neuron hardware detected. "
              "Running CPU comparison only.")
        use_neuron = False

    # Suppress noisy compilation logs — we only want the table
    logging.getLogger("lqcd_neuron.neuron.compiler").setLevel(logging.INFO)
    logging.getLogger("torch_neuronx").setLevel(logging.INFO)

    compiler = NeuronCompiler(dtype="bfloat16") if use_neuron else None
    num_cores = get_device().num_cores if use_neuron else 1
    show_multicore = use_neuron and num_cores > 1

    # ---- Collect results (compiler may emit messages here) ----
    results: list[dict] = []

    # Redirect stdout to suppress stray compiler messages during collection
    real_stdout = sys.stdout
    sys.stdout = open(os.devnull, "w")
    try:
        for shape in LATTICE_SIZES:
            T, Z, Y, X = shape
            geom   = LatticeGeometry(T=T, Z=Z, Y=Y, X=X)
            try:
                U      = GaugeField.random(geom, seed=0).tensor
                psi    = ColorSpinorField.gaussian(geom, seed=1).tensor
            except (RuntimeError, MemoryError) as e:
                # OOM at large volumes (e.g. 32^4 on a small host) is
                # expected; record and continue rather than aborting.
                results.append({
                    "label": f"{T}x{Z}x{Y}x{X}",
                    "cpu": float("nan"),
                    "skipped": f"alloc failed: {type(e).__name__}",
                })
                continue
            D      = WilsonDirac(mass=mass, nc=nc, dtype=dtype)

            cpu_aps = benchmark_one(D.forward, psi, U, label="CPU")

            entry: dict = {
                "label": f"{T}x{Z}x{Y}x{X}",
                "shape": shape,
                "cpu": cpu_aps,
            }

            if use_neuron:
                try:
                    D_n = compiler.compile_dslash(
                        D, shape, nc=nc, gauge_field=U, fused=fused,
                    )
                    entry["neuron"] = benchmark_one(D_n, psi, U, label="Neuron")

                    # The Batched/Multicore paths currently always use the
                    # fused kernel — skip them when running an unfused A/B
                    # comparison so the table reflects a single code path.
                    if not fused:
                        results.append(entry)
                        continue

                    D_b = compiler.compile_dslash_batched(
                        D, shape, batch_size=BATCH_SIZE, gauge_field=U, nc=nc
                    )
                    psi_batch = psi.unsqueeze(0).expand(BATCH_SIZE, *psi.shape).contiguous()
                    entry["batched"] = benchmark_batched(D_b, psi_batch)

                    if show_multicore:
                        mc_batch = num_cores * BATCH_SIZE
                        D_mc = compiler.compile_dslash_multicore(
                            D, shape, gauge_field=U, num_cores=num_cores,
                            per_core_batch_size=BATCH_SIZE, nc=nc,
                        )
                        psi_mc = psi.unsqueeze(0).expand(mc_batch, *psi.shape).contiguous()
                        entry["multicore"] = benchmark_batched(D_mc, psi_mc)
                except (RuntimeError, MemoryError) as e:
                    # Compile/HBM failure at large volumes; report CPU
                    # number only and continue.
                    entry["neuron_error"] = f"{type(e).__name__}: {e}"

            results.append(entry)
    finally:
        sys.stdout.close()
        sys.stdout = real_stdout

    # ---- Print table ----
    if use_neuron:
        cols = [
            f"{'Lattice':>16}",
            f"{'CPU':>14}",
            f"{'Neuron':>14}",
            f"{'Batched':>14}",
        ]
        if show_multicore:
            cols.append(f"{'Multicore':>14}")
        cols += [f"{'Speedup':>8}", f"{'GFLOP/s':>10}", f"{'GB/s':>8}"]
        header = "  ".join(cols)
        ruler  = "-" * len(header)
    else:
        header = f"{'Lattice':>16}  {'CPU':>14}  {'GFLOP/s':>10}  {'GB/s':>8}"
        ruler  = "-" * len(header)

    # Legend
    print()
    print("  Legend (all throughputs in Dslash applications/s):")
    print(f"    CPU       = CPU baseline, single RHS")
    if use_neuron:
        print(f"    Neuron    = 1 NeuronCore, single RHS")
        print(f"    Batched   = 1 NeuronCore, {BATCH_SIZE} RHS per call")
        if show_multicore:
            print(f"    Multicore = {num_cores} NeuronCores, {BATCH_SIZE} RHS each")
        print(f"    Speedup   = best Neuron / CPU")
        print(f"    GFLOP/s   = derived from best Neuron column "
              f"(1320 flops/site, Babich et al. 2011)")
        print(f"    GB/s      = streaming spinor in+out, BF16 "
              f"(gauge baked, no PCIe traffic for U)")
    else:
        print(f"    GFLOP/s   = derived from CPU column (1320 flops/site)")
        print(f"    GB/s      = streaming spinor in+out, FP32")
    print()
    print(header)
    print(ruler)

    for entry in results:
        if "skipped" in entry:
            print(f"{entry['label']:>16}  [skipped: {entry['skipped']}]",
                  flush=True)
            continue
        if entry.get("neuron_error") and use_neuron:
            print(f"{entry['label']:>16}  {entry['cpu']:>14.1f}"
                  f"  [neuron failed: {entry['neuron_error']}]",
                  flush=True)
            continue

        line = f"{entry['label']:>16}  {entry['cpu']:>14.1f}"
        shape = entry.get("shape")
        if use_neuron and "neuron" in entry:
            best = entry.get("multicore") or entry["batched"]
            speedup = best / entry["cpu"]
            gflops, gbps = derived_metrics(best, shape, bf16=True)
            line += f"  {entry['neuron']:>14.1f}  {entry['batched']:>14.1f}"
            if show_multicore:
                line += f"  {entry['multicore']:>14.1f}"
            line += f"  {speedup:>7.1f}x  {gflops:>10.2f}  {gbps:>8.2f}"
        elif not use_neuron:
            gflops, gbps = derived_metrics(entry["cpu"], shape, bf16=False)
            line += f"  {gflops:>10.2f}  {gbps:>8.2f}"
        print(line, flush=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dslash throughput benchmark")
    parser.add_argument(
        "--neuron",
        action="store_true",
        help="Compile kernels for Neuron and compare against CPU baseline.",
    )
    parser.add_argument(
        "--mass", type=float, default=0.1,
        help="Wilson bare quark mass (default: 0.1).",
    )
    parser.add_argument(
        "--no-fused",
        action="store_true",
        help=(
            "Disable fused (Ns*Nc)x(Ns*Nc) hopping kernels and bake only the "
            "raw gauge tensor.  Lower per-site working set; useful for "
            "diagnosing the fused-kernel SRAM-spill cliff at large lattices. "
            "Implies single-RHS only — Batched/Multicore columns are skipped."
        ),
    )
    args = parser.parse_args()
    run(use_neuron=args.neuron, mass=args.mass, fused=not args.no_fused)
