# Quickstart Guide

## Prerequisites

| Environment | Requirements |
|-------------|-------------|
| Development / CI | Python 3.9+, PyTorch ≥ 2.1 |
| AWS Trn1 / Inf2 | Above + `torch-neuronx ≥ 2.1`, `neuronx-cc ≥ 2.0` |

## Installation

```bash
# CPU-only (laptop, CI)
pip install lqcd-neuron

# With Neuron backend (Trn1 / Inf2 instance)
pip install lqcd-neuron[neuron]

# Editable install from source
git clone https://github.com/JGalego/lqcd-neuron
cd lqcd-neuron
pip install -e ".[dev]"
```

---

## Step 1 — Define a lattice

```python
from lqcd_neuron.core import LatticeGeometry

geom = LatticeGeometry(T=8, Z=4, Y=4, X=4)
print(geom.volume)        # 512
print(geom.gauge_shape)   # (8, 4, 4, 4, 4, 3, 3)
print(geom.spinor_shape)  # (8, 4, 4, 4, 4, 3)
```

`T, Z, Y, X` are the lattice extents in the temporal and three spatial directions.
`Nc=3` (SU(3)) and `Ns=4` (Dirac spinor) are the defaults.

---

## Step 2 — Create gauge and spinor fields

```python
import torch
from lqcd_neuron.core import GaugeField, ColorSpinorField

# Cold start — all links are identity matrices
U_cold = GaugeField.cold(geom)

# Hot start — random SU(3) gauge field sampled from Haar measure
U = GaugeField.random(geom, seed=42)

# Point source spinor δ(x) δ_{α0} δ_{c0}
b = ColorSpinorField.point_source(geom, t=0, z=0, y=0, x=0, spin=0, color=0)

# Gaussian random spinor
psi = ColorSpinorField.gaussian(geom, seed=7)
```

Cast to Neuron's preferred `bfloat16`:
```python
U_bf16 = U.to(dtype=torch.complex32)   # lossless bfloat16 complex
```

---

## Step 3 — Measure observables

```python
from lqcd_neuron.observables import plaquette, wilson_action, polyakov_loop

# Wilson plaquette  P ∈ [0, 1]
P = plaquette(U)
print(f"Plaquette: {P:.6f}")

# Wilson action  S_W = β Σ (1 - P_{μν}/Nc)
S = wilson_action(U, beta=6.0)
print(f"Wilson action: {S:.3f}")

# Polyakov loop  ⟨L⟩ — order parameter for deconfinement
L = polyakov_loop(U)
print(f"Polyakov loop: {L:.4f}")
```

---

## Step 4 — Apply the Wilson Dirac operator

```python
from lqcd_neuron.dirac import WilsonDirac

# D_W with bare mass m = 0.1
D = WilsonDirac(mass=0.1)

# D ψ
Dpsi = D(psi.tensor, U.tensor)

# M†M ψ  (for CG normal equations)
MtMpsi = D.normal(psi.tensor, U.tensor)
```

### Clover-Wilson (SW-improved)

```python
from lqcd_neuron.dirac import CloverWilsonDirac

D_clv = CloverWilsonDirac(mass=0.1, csw=1.0)
D_clv.set_gauge(U.tensor)   # pre-compute clover matrices
out = D_clv(psi.tensor, U.tensor)
```

---

## Step 5 — Solve for a quark propagator (CG)

```python
from lqcd_neuron.solvers import ConjugateGradient

# Build right-hand side: b̃ = D† b
b_rhs = D.dagger(b.tensor, U.tensor)

# Solve M†M x = b̃
solver = ConjugateGradient(tol=1e-8, maxiter=500, verbose=True)
x, info = solver.solve(lambda v: D.normal(v, U.tensor), b_rhs)

print(f"Converged: {info.converged}")
print(f"Iterations: {info.iterations}")
print(f"Final |r|/|b|: {info.final_residual:.2e}")
```

For non-Hermitian systems use `BiCGStab`:

```python
from lqcd_neuron.solvers import BiCGStab

solver = BiCGStab(tol=1e-8, maxiter=500)
x, info = solver.solve(lambda v: D(v, U.tensor), b.tensor)
```

---

## Step 6 — Compile for Neuron (Trn1 / Inf2)

```python
from lqcd_neuron.neuron import NeuronCompiler, is_neuron_available

if is_neuron_available():
    compiler = NeuronCompiler(dtype="bfloat16")

    # Recommended: pass `gauge_field=U` so the compiler can pre-fuse the
    # spin/colour hopping kernels and bake U into the .neff as a buffer.
    # The wrapper still accepts (psi, U) for API compatibility — U is
    # already on the NeuronCore so the second argument is ignored.
    D_neuron = compiler.compile_dslash(
        D, lattice_shape=geom.shape, nc=geom.nc, gauge_field=U.tensor,
    )

    Dpsi_neuron = D_neuron(psi.tensor, U.tensor)
```

The compiled module is a drop-in replacement for `D` — same signature, same output
shape.  The solver loop stays on the host:

```python
def matvec_neuron(v):
    return D.dagger(D_neuron(v, U.tensor), U.tensor)   # D†D_neuron

x_neuron, info = ConjugateGradient(tol=1e-8).solve(matvec_neuron, b_rhs)
```

### Multi-RHS (batched) inversion

For propagator calculations that need multiple right-hand sides, compile a
batched operator instead.  This amortises the fixed per-call NeuronCore
dispatch cost across all `B` spinors and is the highest-throughput path:

```python
import torch

B = 12   # e.g. one RHS per spin × colour source
D_batched = compiler.compile_dslash_batched(
    D, lattice_shape=geom.shape, batch_size=B,
    gauge_field=U.tensor, nc=geom.nc,
)

# psi_batch shape: (B, T, Z, Y, X, Ns, Nc)
psi_batch = torch.stack([psi.tensor for _ in range(B)], dim=0)
out_batch = D_batched(psi_batch)
```

### Multi-core (data-parallel) inversion

On instances with more than one NeuronCore (`inf2.xlarge` has 2,
`trn1.32xlarge` has 32), shard the batch across cores via
`torch_neuronx.DataParallel`:

```python
from lqcd_neuron.neuron import get_device

num_cores = get_device().num_cores      # auto-detected from /dev/neuron*
per_core_batch_size = 8
B_global = num_cores * per_core_batch_size

D_mc = compiler.compile_dslash_multicore(
    D, lattice_shape=geom.shape, gauge_field=U.tensor,
    num_cores=num_cores,                 # default: all detected cores
    per_core_batch_size=per_core_batch_size,
    nc=geom.nc,
)

psi_global = torch.stack([psi.tensor for _ in range(B_global)], dim=0)
out_global = D_mc(psi_global)            # shape: (B_global, T, Z, Y, X, Ns, Nc)
```

The gauge field is baked into each core's `.neff`, so only the spinor
shard crosses PCIe per core per call.  This stacks with multi-RHS
batching: each core still runs the fused 12×12 batched kernel on its
`per_core_batch_size` slice.

> **Note:** the `.neff` produced by the gauge-baked path is specific to the
> exact gauge configuration passed in.  Re-compile (cheap once warm) when `U`
> changes between solves.

---

## Step 7 — Run the test suite

```bash
pytest tests/ -v
```

All tests run on CPU without Neuron hardware.

---

## Profiling & monitoring

The Neuron SDK ships two stock observability tools:

- `neuron-top` &mdash; interactive `htop`-style snapshot of NeuronCore usage.
- `neuron-monitor` &mdash; JSON-stream metrics emitter, one record per period.

For an **EEG-style time-series view** (one row per NeuronCore, shared time
axis), this repo bundles a thin wrapper around `neuron-monitor`:
[scripts/neuron_trace.py](../scripts/neuron_trace.py).

```bash
# 30 s capture @ 1 Hz, writes traces/neuron_<ts>.{jsonl,png}
make trace

# Live rolling-window plot (requires interactive matplotlib backend)
make trace-live

# Custom capture
python scripts/neuron_trace.py --duration 60 --period 0.5 \
    --output traces/run.jsonl --plot traces/run.png

# Re-render an existing JSONL trace
python scripts/neuron_trace.py --replot traces/run.jsonl
```

The script has no dependency on `lqcd_neuron` itself &mdash; it is safe to copy
to any Inf2 / Trn1 host. It requires `neuron-monitor` on `PATH` (provided by
the Neuron SDK) and `matplotlib` for plotting.

For kernel-level timelines (closer to NVIDIA Nsight Systems), use
`neuron-profile` from the Neuron SDK and view the trace in Perfetto.

---

## Configuration reference

See [src/lqcd_neuron/params.py](../src/lqcd_neuron/params.py) for all parameter
dataclasses:

| Class | Key fields |
|-------|-----------|
| `GaugeParam` | `lattice_size`, `nc`, `precision`, `t_boundary`, `anisotropy` |
| `InvertParam` | `dslash_type`, `inv_type`, `mass`, `kappa`, `tol`, `maxiter` |
| `CloverParam` | `csw`, `precision` |
| `NeuronCompileParam` | `dtype`, `optimize_level`, `num_neuroncores` |
