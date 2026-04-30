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
    (4, 4, 4, 4),
    (8, 4, 4, 4),
    (8, 8, 4, 4),
    (8, 8, 8, 4),
    #(16, 8, 8, 8),
    #(16, 16, 16, 16),
    #(32, 16, 16, 16),
]

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


def run(use_neuron: bool, mass: float = 0.1) -> None:
    dtype = torch.complex64
    nc    = 3

    if use_neuron and not is_neuron_available():
        print("WARNING: --neuron requested but no Neuron hardware detected. "
              "Running CPU comparison only.")
        use_neuron = False

    compiler = NeuronCompiler(dtype="bfloat16") if use_neuron else None
    num_cores = get_device().num_cores if use_neuron else 1
    show_multicore = use_neuron and num_cores > 1
    MC_COL_WIDTH = 22

    if use_neuron:
        cols = [
            f"{'Lattice':>16}",
            f"{'CPU (apps/s)':>14}",
            f"{'Neuron (apps/s)':>16}",
            f"{f'Neuron x{BATCH_SIZE} (apps/s)':>22}",
        ]
        if show_multicore:
            mc_label = f"Multicore x{num_cores} (apps/s)"
            cols.append(f"{mc_label:>{MC_COL_WIDTH}}")
        cols.append(f"{'Speedup':>8}")
        header = "  ".join(cols)
        ruler  = "-" * len(header)
    else:
        header = f"{'Lattice':>16}  {'CPU (apps/s)':>14}"
        ruler  = "-" * len(header)
    print("\n" + header)
    print(ruler)

    for shape in LATTICE_SIZES:
        T, Z, Y, X = shape
        geom   = LatticeGeometry(T=T, Z=Z, Y=Y, X=X)
        U      = GaugeField.random(geom, seed=0).tensor
        psi    = ColorSpinorField.gaussian(geom, seed=1).tensor
        D      = WilsonDirac(mass=mass, nc=nc, dtype=dtype)

        # CPU timing
        cpu_aps = benchmark_one(D.forward, psi, U, label="CPU")

        row = f"{T}x{Z}x{Y}x{X:>2}"
        line = f"{row:>16}  {cpu_aps:>14.1f}"

        if use_neuron:
            # Single-RHS Neuron path
            print(f"{row:>16}  compiling for Neuron …", end="\r", flush=True)
            D_n = compiler.compile_dslash(D, shape, nc=nc, gauge_field=U)
            neuron_aps = benchmark_one(D_n, psi, U, label="Neuron")

            # Multi-RHS Neuron path (single core)
            print(f"{row:>16}  compiling batched (B={BATCH_SIZE}) …",
                  end="\r", flush=True)
            D_b = compiler.compile_dslash_batched(
                D, shape, batch_size=BATCH_SIZE, gauge_field=U, nc=nc
            )
            psi_batch = psi.unsqueeze(0).expand(BATCH_SIZE, *psi.shape).contiguous()
            batched_aps = benchmark_batched(D_b, psi_batch)

            # Multi-core data-parallel path
            mc_aps = 0.0
            if show_multicore:
                # Each core handles BATCH_SIZE RHS; global batch = cores * BATCH_SIZE
                mc_batch = num_cores * BATCH_SIZE
                print(f"{row:>16}  compiling multicore ({num_cores} cores) …",
                      end="\r", flush=True)
                D_mc = compiler.compile_dslash_multicore(
                    D, shape, gauge_field=U, num_cores=num_cores,
                    per_core_batch_size=BATCH_SIZE, nc=nc,
                )
                psi_mc = psi.unsqueeze(0).expand(mc_batch, *psi.shape).contiguous()
                mc_aps = benchmark_batched(D_mc, psi_mc)

            speedup = (mc_aps if mc_aps > 0 else batched_aps) / cpu_aps
            line += f"  {neuron_aps:>16.1f}  {batched_aps:>22.1f}"
            if show_multicore:
                line += f"  {mc_aps:>{MC_COL_WIDTH}.1f}"
            line += f"  {speedup:>7.1f}x"

        print(line)


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
    args = parser.parse_args()
    run(use_neuron=args.neuron, mass=args.mass)
