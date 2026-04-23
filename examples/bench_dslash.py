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
from lqcd_neuron.neuron import NeuronCompiler, is_neuron_available


# ---------------------------------------------------------------------------
# Lattice sizes to benchmark
# ---------------------------------------------------------------------------

LATTICE_SIZES: List[Tuple[int, int, int, int]] = [
    (4, 4, 4, 4),
    (8, 4, 4, 4),
    (8, 8, 4, 4),
    (8, 8, 8, 4),
    (16, 8, 8, 8),
]

WARMUP_ITERS = 5
BENCH_ITERS  = 50


def benchmark_one(
    matvec,
    psi: torch.Tensor,
    U: torch.Tensor,
    n: int = BENCH_ITERS,
    warmup: int = WARMUP_ITERS,
    label: str = "",
) -> float:
    """Return applications/second for ``matvec(psi, U)``."""
    for _ in range(warmup):
        _ = matvec(psi, U)

    torch.cuda.synchronize() if psi.is_cuda else None

    t0 = time.perf_counter()
    for _ in range(n):
        out = matvec(psi, U)
    elapsed = time.perf_counter() - t0
    return n / elapsed


def run(use_neuron: bool, mass: float = 0.1) -> None:
    dtype = torch.complex64
    nc    = 3

    if use_neuron and not is_neuron_available():
        print("WARNING: --neuron requested but no Neuron hardware detected. "
              "Running CPU comparison only.")
        use_neuron = False

    compiler = NeuronCompiler(dtype="bfloat16") if use_neuron else None

    print(f"\n{'Lattice':>16}  {'CPU (apps/s)':>14}  "
          + (f"{'Neuron (apps/s)':>16}  {'Speedup':>8}" if use_neuron else ""))
    print("-" * (55 if use_neuron else 38))

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
            # AoT compile (first call is slow; result cached to disk)
            print(f"{row:>16}  compiling for Neuron …", end="\r", flush=True)
            D_n = compiler.compile_dslash(D, shape, nc=nc)
            neuron_aps = benchmark_one(D_n, psi, U, label="Neuron")
            speedup    = neuron_aps / cpu_aps
            line += f"  {neuron_aps:>16.1f}  {speedup:>7.1f}x"

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
