# Architecture

## Design goals

1. **Neuron-first**: every hot-path kernel is a `torch.nn.Module` that can be
   compiled to a `.neff` binary by `neuronx-cc` without modification.
2. **CPU fallback**: the library must work identically on CPU for development and
   testing.  No Neuron hardware is required to run the test suite.
3. **QUDA-inspired API**: parameter dataclasses (`GaugeParam`, `InvertParam`, …)
   and operator types mirror QUDA's naming but are idiomatic Python/PyTorch.
4. **Composable**: fields, operators, solvers, and observables are independent
   modules — mix and match.

---

## Tensor layout convention

All tensors follow the `(T, Z, Y, X, ...)` storage order, consistent with
numpy's row-major convention where the rightmost index is innermost (fastest-varying):

| Object | Shape | Notes |
|--------|-------|-------|
| Gauge field U | `(T, Z, Y, X, 4, Nc, Nc)` | dim 4 = direction μ = 0…3 |
| Spinor ψ | `(T, Z, Y, X, Ns, Nc)` | Ns=4 (Dirac), Nc=3 (SU(3)) |
| Field strength F̂_{μν} | `(T, Z, Y, X, 6, Nc, Nc)` | 6 pairs (01,02,03,12,13,23) |
| Clover matrix C | `(T, Z, Y, X, Ns, Nc, Ns, Nc)` | acts on spin⊗colour |

Direction-to-dimension map:
- μ = 0 ↔ lattice dim 0 (T), `torch.roll(x, ±1, 0)`
- μ = 1 ↔ lattice dim 1 (Z), `torch.roll(x, ±1, 1)`
- μ = 2 ↔ lattice dim 2 (Y), `torch.roll(x, ±1, 2)`
- μ = 3 ↔ lattice dim 3 (X), `torch.roll(x, ±1, 3)`

---

## Module hierarchy

```
lqcd_neuron/
├── params.py               GaugeParam, InvertParam, CloverParam, NeuronCompileParam
│
├── core/
│   ├── lattice.py          LatticeGeometry  — immutable shape descriptor
│   ├── gauge_field.py      GaugeField  — SU(Nc) tensor container
│   └── spinor_field.py     ColorSpinorField  — quark spinor container
│
├── dirac/
│   ├── gamma.py            Euclidean γ-matrices (DeGrand-Rossi basis)
│   ├── wilson.py           WilsonDslash, WilsonDirac — nn.Module
│   └── clover.py           compute_clover, CloverWilsonDirac — nn.Module
│
├── blas/
│   └── lattice_blas.py     inner, norm2, axpy, xpay … (JIT-scriptable)
│
├── solvers/
│   ├── cg.py               ConjugateGradient  — host loop, device matvec
│   └── bicgstab.py         BiCGStab  — host loop, device matvec
│
├── observables/
│   ├── plaquette.py        plaquette, wilson_action, topological_charge
│   └── polyakov.py         polyakov_loop, polyakov_loop_spatially_resolved
│
└── neuron/
    ├── device.py           NeuronDevice, is_neuron_available
    └── compiler.py         NeuronCompiler — torch_neuronx.trace wrapper
```

---

## The host-loop / device-kernel split

Iterative solvers require convergence checks at every iteration.  Because
`torch_neuronx.trace` only supports static graphs (no Python-level `while` loops),
the convergence loop runs on the **host** (Python) and only the expensive matvec
`D(ψ, U)` is compiled to Neuron:

```
Python host (CG loop):
  for k in range(maxiter):
      Ap   = D_neuron(p, U)    ←  crosses to NeuronCores here
      rho  = inner(r, r)       ←  scalar, runs on host (CPU)
      alpha = rho / inner(p,Ap)
      x  += alpha * p
      r  -= alpha * Ap
      if norm(r) < tol: break
```

This mirrors standard deep-learning training loops: the `model.forward()` runs on
the accelerator; the Python loop controls iteration, loss, and early stopping.

---

## Gamma matrices (DeGrand-Rossi / chiral basis)

All Dirac operators use the **Euclidean DeGrand-Rossi** basis, which satisfies:

$$\{\gamma_\mu, \gamma_\nu\} = 2\delta_{\mu\nu}\,I_4 \qquad \gamma_\mu^\dagger = \gamma_\mu$$

The four matrices are registered as non-trainable buffers on each Dirac `nn.Module`,
so they move automatically with `.to(device)` calls and remain visible to the Neuron
tracer.

---

## Wilson Dslash

$$
(D_W\psi)_x = (4+m)\psi_x
  - \tfrac{1}{2}\sum_{\mu=0}^{3}
    \bigl[(I-\gamma_\mu)\,U(x,\mu)\,\psi_{x+\hat\mu}
         + (I+\gamma_\mu)\,U^\dagger(x-\hat\mu,\mu)\,\psi_{x-\hat\mu}\bigr]
$$

Implemented as six `torch.roll` + `torch.einsum` calls (four directions × two hops).
The Python `for mu in range(4)` loop is **unrolled at trace time** into a static
computation graph — exactly what `neuronx-cc` requires.

---

## Clover (SW) improvement

The clover term corrects O(a) Wilson artefacts:

$$
D_{CW} = D_W + \frac{c_{SW}}{4}\sum_{\mu<\nu}\sigma_{\mu\nu}\hat{F}_{\mu\nu}(x)
$$

where $\hat{F}_{\mu\nu}$ is the 4-leaf clover estimate of the field strength and
$\sigma_{\mu\nu} = \tfrac{i}{2}[\gamma_\mu, \gamma_\nu]$.

**Pre-computation**: `CloverWilsonDirac.set_gauge(U)` builds the 12×12 Hermitian
clover matrix at every site from the gauge field.  Repeated solver calls reuse the
cached result — matching QUDA's `loadCloverQuda` pattern.

---

## Neuron compilation lifecycle

```
1. model = WilsonDirac(mass=0.1)        # pure Python nn.Module
2. compiler = NeuronCompiler()          # sets env flags for neuronx-cc
3. model_n = compiler.compile_dslash(   # calls torch_neuronx.trace(...)
       model, lattice_shape=(8,4,4,4))  #   → runs neuronx-cc offline
                                        #   → writes .neff to workdir cache
4. out = model_n(psi, U)               # runs .neff on NeuronCores instantly
```

The `NeuronCompiler` maintains an in-process cache keyed by
`(class_name, shape, dtype)` to avoid redundant compilations within a job.
