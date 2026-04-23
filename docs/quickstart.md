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

    # Compile once — neuronx-cc runs and writes .neff to ~/.cache/lqcd-neuron/
    D_neuron = compiler.compile_dslash(D, lattice_shape=geom.shape, nc=geom.nc)

    # Subsequent calls hit the on-disk cache immediately
    Dpsi_neuron = D_neuron(psi.tensor, U.tensor)
```

The compiled module is a drop-in replacement for `D` — same signature, same output
shape.  The solver loop stays on the host:

```python
def matvec_neuron(v):
    return D.dagger(D_neuron(v, U.tensor), U.tensor)   # D†D_neuron

x_neuron, info = ConjugateGradient(tol=1e-8).solve(matvec_neuron, b_rhs)
```

---

## Step 7 — Run the test suite

```bash
pytest tests/ -v
```

All tests run on CPU without Neuron hardware.

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
