# LQCD-Neuron

A **QUDA-inspired** high-performance Python SDK for running Lattice QCD simulations on
[AWS Trainium (Trn1)](https://aws.amazon.com/ec2/instance-types/trn1/) and
[AWS Inferentia (Inf2)](https://aws.amazon.com/ec2/instance-types/inf2/) instances
using the [AWS Neuron SDK](https://awsdocs-neuron.readthedocs-hosted.com/).


## Overview

[QUDA](https://github.com/lattice/quda) is the de-facto standard library for
GPU-accelerated Lattice QCD.  It achieves high throughput by writing hand-tuned
CUDA kernels that run directly on NVIDIA GPU Stream Multiprocessors.

**This library takes a fundamentally different route.**  AWS Trainium and Inferentia
chips provide no general-purpose GPGPU interface - there is no CUDA, no HIP, no
OpenCL.  Their NeuronCores are purpose-built matrix-multiplication engines exposed
exclusively through the [AWS Neuron SDK](https://awsdocs-neuron.readthedocs-hosted.com/),
which integrates with PyTorch via an XLA backend (`torch-neuronx`).

LQCD-Neuron expresses every Lattice QCD primitive - the Wilson Dslash, the clover
term, iterative solvers, gauge observables - as a standard
`torch.nn.Module`.  The Neuron compiler (`neuronx-cc`) then lowers the XLA HLO
graph to a `.neff` binary that runs directly on the NeuronCores, with **no CUDA and
no custom kernel code**.

```
User Python code
      │
      ▼
 lqcd_neuron.dirac.WilsonDirac   (nn.Module)
      │  torch_neuronx.trace()
      ▼
 neuronx-cc compilation
      │  XLA HLO  →  NeuronCore ISA
      ▼
 .neff binary  →  NeuronCore-v2 (Trn1 / Inf2)
```


## Features

| Category             | Component                    | Description                                        |
|----------------------|------------------------------|----------------------------------------------------|
| **Fields**           | `GaugeField`                 | SU(Nc) link-variable tensor, cold/random/unitarize |
|                      | `ColorSpinorField`           | Quark spinor ψ, point source, Gaussian, arithmetic |
|                      | `LatticeGeometry`            | (T,Z,Y,X) hypercubic lattice descriptor            |
| **Dirac operators**  | `WilsonDslash`               | Pure hopping part, Neuron-traceable `nn.Module`    |
|                      | `WilsonDirac`                | Full M = (4+m)I + D_hop, plus M†, M†M              |
|                      | `CloverWilsonDirac`          | SW-improved Clover-Wilson operator                 |
| **Solvers**          | `ConjugateGradient`          | CG for Hermitian positive-definite M†M             |
|                      | `BiCGStab`                   | BiCGStab for general square systems                |
| **Observables**      | `plaquette`                  | Average Wilson plaquette P ∈ [0,1]                 |
|                      | `wilson_action`              | Wilson gauge action S_W                            |
|                      | `topological_charge`         | Clover-definition topological charge Q              |
|                      | `polyakov_loop`              | Spatially-averaged Polyakov loop ⟨L⟩               |
| **Neuron utilities** | `NeuronCompiler`             | `torch_neuronx.trace` wrapper; gauge-baking + fused kernels |
|                      | `compile_dslash_batched`     | Multi-RHS Dslash compilation for amortised dispatch |
|                      | `NeuronDevice`               | Hardware detection (Trn1 / Inf2 / CPU fallback)    |


## Quick Start

### Install

```bash
pip install lqcd-neuron            # CPU-only (dev / testing)

# On a Trn1 or Inf2 instance, also install the Neuron extras:
pip install lqcd-neuron[neuron]    # pulls in torch-neuronx, neuronx-cc
```

### Measure the plaquette

```python
from lqcd_neuron.core import LatticeGeometry, GaugeField
from lqcd_neuron.observables import plaquette

geom  = LatticeGeometry(T=8, Z=4, Y=4, X=4)
U     = GaugeField.cold(geom)
print(plaquette(U))   # → 1.0  (cold start)

U_hot = GaugeField.random(geom, seed=0)
print(plaquette(U_hot))  # → small positive number (disordered)
```

### Wilson quark propagator (CG inversion)

```python
from lqcd_neuron.core import ColorSpinorField, GaugeField, LatticeGeometry
from lqcd_neuron.dirac import WilsonDirac
from lqcd_neuron.solvers import ConjugateGradient

geom = LatticeGeometry(T=8, Z=4, Y=4, X=4)
U    = GaugeField.random(geom, seed=1)
b    = ColorSpinorField.point_source(geom, t=0, z=0, y=0, x=0)

D    = WilsonDirac(mass=0.1)
b_rhs = D.dagger(b.tensor, U.tensor)          # M† b

solver = ConjugateGradient(tol=1e-8, maxiter=1000, verbose=True)
x, info = solver.solve(lambda v: D.normal(v, U.tensor), b_rhs)

print(f"Converged in {info.iterations} iterations, |r|/|b| = {info.final_residual:.2e}")
```

### Compile for Neuron (Trn1 / Inf2)

```python
from lqcd_neuron.dirac import WilsonDirac
from lqcd_neuron.neuron import NeuronCompiler

D         = WilsonDirac(mass=0.1)
compiler  = NeuronCompiler(dtype="bfloat16")

# Recommended: pass `gauge_field=U` so the compiler can pre-fuse the
# spin/colour hopping kernels and bake U into the .neff as a buffer.
# Only the spinor crosses PCIe per call.
D_neuron  = compiler.compile_dslash(
    D, lattice_shape=(8, 4, 4, 4), nc=3, gauge_field=U,
)

# D_neuron(psi, U) now executes on NeuronCores; the second arg is
# accepted for API compatibility but ignored (U is already on-device).
out = D_neuron(psi, U)

# Multi-RHS: amortise per-call dispatch overhead across N right-hand sides
D_batched = compiler.compile_dslash_batched(
    D, lattice_shape=(8, 4, 4, 4), batch_size=8, gauge_field=U, nc=3,
)
out_batch = D_batched(psi_batch)   # psi_batch shape: (8, T, Z, Y, X, Ns, Nc)
```


## Performance

Wilson Dslash throughput on a single Inf2 NeuronCore-v2 (BF16) vs CPU (FP32),
50 iterations after warm-up.  Numbers in **applications/sec** (higher is better).

| Lattice    | CPU  | Neuron (1-RHS) | Neuron (8-RHS) | Speedup vs CPU |
|------------|-----:|---------------:|---------------:|---------------:|
| 4×4×4×4    |  818 |           1419 |       **8957** |     **10.9×** |
| 8×4×4×4    |  767 |            794 |       **4918** |      **6.4×** |
| 8×8×4×4    |  646 |            372 |       **2481** |      **3.8×** |
| 8×8×8×4    |  449 |            208 |       **1379** |      **3.1×** |

Three compile-time optimisations make this possible:

1. **Gauge baking** — when `gauge_field=U` is passed to `compile_dslash`, the
   gauge configuration is embedded in the `.neff` as a NeuronCore-resident
   buffer.  Only the spinor crosses PCIe per call.
2. **Fused spin-colour kernels** — each direction’s `(I ∓ γ_μ) ⊗ U(x,μ)`
   block is precomputed as a 12×12 matrix per site, replacing the per-call
   3×3 colour einsum + 4×4 spin einsum with a single matvec that maps cleanly
   onto the NeuronCore tensor engine.
3. **Multi-RHS batching** — `compile_dslash_batched(…, batch_size=B)` solves
   *B* spinors per call, amortising the ~1 ms per-call NeuronCore dispatch
   overhead.  This is the same pattern used by production multi-source LQCD
   propagator inverters.

Reproduce locally with:

```bash
make bench-neuron        # on an Inf2 / Trn1 instance
make connect-bench       # SSH into the OpenTofu-provisioned instance
```

### Profiling & monitoring

For an EEG-style per-NeuronCore utilization trace over time, run:

```bash
make trace               # 30 s capture, writes traces/neuron_<ts>.{jsonl,png}
make trace-live          # rolling-window live plot
```

See [scripts/neuron_trace.py](scripts/neuron_trace.py) &mdash; a standalone
wrapper around `neuron-monitor` that produces stacked per-core time-series
plots. Details in the [Quickstart](docs/quickstart.md#profiling--monitoring).


## Repository layout

```
lqcd-neuron/
├── src/lqcd_neuron/
│   ├── core/           # LatticeGeometry, GaugeField, ColorSpinorField
│   ├── dirac/          # gamma matrices, WilsonDirac, CloverWilsonDirac
│   ├── solvers/        # ConjugateGradient, BiCGStab
│   ├── blas/           # inner products, axpy, norms
│   ├── observables/    # plaquette, Wilson action, Polyakov loop, Q_top
│   ├── neuron/         # NeuronCompiler, NeuronDevice
│   └── params.py       # QUDA-inspired parameter dataclasses
├── examples/
│   ├── 01_plaquette.py
│   ├── 02_wilson_dslash.py
│   └── 03_cg_inversion.py
├── scripts/
│   ├── setup_inf2.sh       # Bootstrap a Trn1 / Inf2 instance
│   ├── connect_inf2.sh     # SSH helper for the provisioned instance
│   └── neuron_trace.py     # EEG-style NeuronCore utilization tracer
├── tests/
└── docs/
    ├── architecture.md   # Design overview
    ├── comparison.md     # QUDA vs LQCD-Neuron comparison table
    └── quickstart.md     # Step-by-step tutorial
```


## Documentation

- [Architecture overview](docs/architecture.md)
- [QUDA vs LQCD-Neuron comparison](docs/comparison.md)
- [Quickstart guide](docs/quickstart.md)


## Running the tests

```bash
pip install -e ".[dev]"
pytest
```


## License

Apache 2.0 - see [LICENSE](LICENSE).
