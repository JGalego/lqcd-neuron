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
|                      | `compile_dslash_multicore`   | Data-parallel Dslash sharded across all NeuronCores |
|                      | `compile_dslash_sharded`     | T-axis spatial sharding for V > 24⁴ (per-NEFF HLO budget) |
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

# Multi-core: shard a global batch of `num_cores * per_core_batch_size`
# RHS across all detected NeuronCores via torch_neuronx.DataParallel.
D_mc = compiler.compile_dslash_multicore(
    D, lattice_shape=(8, 4, 4, 4), gauge_field=U,
    num_cores=None,            # default: all detected NeuronCores
    per_core_batch_size=8, nc=3,
)
out_mc = D_mc(psi_global)      # psi_global shape: (num_cores * 8, T, Z, Y, X, Ns, Nc)
```


## Performance

Wilson Dslash throughput on an `inf2.8xlarge` (BF16) vs CPU (FP32),
50 iterations after warm-up.  Numbers in **applications/sec** (higher is better).
Speedup = best Neuron / CPU.

| Lattice      |    CPU |  Neuron |  Batched | Multicore | Speedup | GFLOP/s |  GB/s |
|--------------|-------:|--------:|---------:|----------:|--------:|--------:|------:|
| 4×4×4×4      |  804.6 |  1358.0 |   8521.5 | **10431.3** | **13.0×** |    3.52 |  0.26 |
| 8×4×4×4      |  769.5 |   785.1 |   4786.3 |  **7519.1** |  **9.8×** |    5.08 |  0.37 |
| 8×8×4×4      |  518.1 |   370.1 |   2511.1 |  **4229.5** |  **8.2×** |    5.72 |  0.42 |
| 8×8×8×4      |  384.0 |   207.6 |   1390.1 |  **2474.2** |  **6.4×** |    6.69 |  0.49 |
| 16×8×8×8     |  229.2 |    37.2 |    253.8 |      220.1 |   1.0× †|    2.38 |  0.17 |
| 16×16×16×16  |   59.6 |     1.2 |     24.4 |       15.8 |   0.3× †|    1.36 |  0.10 |

> **†** Fused kernels exceed the SRAM budget at this volume (default ~14.4 MiB
> on NeuronCore-v2, i.e. 60% of the 24 MiB SBUF); the compiler automatically
> falls back to the unfused baked-gauge path, which keeps gauge baking but
> drops the per-site 12×12 fusion in favour of the original 3×3 colour + 4×4
> spin einsums (≈12× smaller working set).  Override the trip point with
> `NeuronCompiler(sram_threshold_bytes=N)`, force the path with `fused=False`
> (or `make bench NEURON=1 NO_FUSED=1`), or hard-disable the silent
> downgrade with `NeuronCompiler(allow_fused_fallback=False)` to make any
> spill a `RuntimeError` instead.

> **Legend** — all throughputs in Dslash applications/sec (higher is better):
> - **CPU** — CPU baseline, single RHS (FP32)
> - **Neuron** — 1 NeuronCore, single RHS (BF16)
> - **Batched** — 1 NeuronCore, 8 RHS per call (BF16)
> - **Multicore** — 2 NeuronCores, 8 RHS each (BF16)
> - **Speedup** — best Neuron / CPU
> - **GFLOP/s** — derived from the Multicore column (1320 FLOP/site, Babich et al. 2011)
> - **GB/s** — streaming spinor in+out, BF16 (gauge baked; no PCIe traffic for U)

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
4. **Multi-core data-parallel sharding** — `compile_dslash_multicore(…)` wraps
   the compiled `.neff` in `torch_neuronx.DataParallel`, replicating it across
   every detected NeuronCore (e.g. all 2 cores on `inf2.xlarge`, 24 on
   `trn1.32xlarge`).  A global batch of `num_cores * per_core_batch_size`
   right-hand sides is split along dim 0 — combining inter-core parallelism
   with the intra-core multi-RHS path above.
5. **Spatial sharding (T-axis)** — `compile_dslash_sharded(…, num_shards=k)`
   splits the lattice along T into *k* slabs and compiles one NEFF per slab,
   bringing per-graph HLO instruction counts back under the `neuronx-cc`
   ~5M budget for $V > 24^4$ where the single-NEFF compile would otherwise
   abort with `[NCC_EVRF007]`.  Halos are gathered host-side under periodic
   BCs.  `compile_dslash` auto-routes through this path when the volume
   exceeds the per-NEFF cap.

Reproduce locally with:

```bash
make bench NEURON=1                       # default sweep (B = 8, 16, 32)
make bench NEURON=1 BATCH=1,8,32,64       # custom batch-size sweep
make bench NEURON=1 LATTICE=16x16x16x16   # restrict to one lattice
make connect-bench NEURON=1               # SSH into the instance and run there
```

The `--batch-sizes` flag emits one row per `(lattice, B)` so you can see where
dispatch overhead stops dominating and HBM bandwidth takes over.

### Fire-and-forget bench job (results emailed)

For unattended runs, `make bench-job` triggers the benchmark on AWS, archives
the full log to S3, and emails a summary (with a 7-day presigned download
link) via SNS.  Set `notification_email` in `infra/terraform.tfvars` and
`tofu apply` once to wire up the topic and confirm the subscription.

By default `infra/` provisions only the shared bench infra (VPC, IAM, SNS,
S3, launch template) — no always-on instance — and `make bench-job` runs in
ephemeral mode: a one-shot Inf2 that runs the benchmark, emails the result,
and self-terminates.  Set `skip_persistent_instance = false` in
`terraform.tfvars` if you want the long-running instance back for
interactive SSH/SSM work; `MODE=persistent` then becomes available.

```bash
make bench-job                                    # ephemeral one-shot Inf2 (default)
make bench-job MODE=persistent                    # SSM RunCommand on the long-running instance
make bench-job NEURON=1 LATTICE="16x16x16x16"     # restrict to one lattice
make bench-job NEURON=1 BATCH=1,8,32,64           # sweep batch sizes
make bench-job MODE=persistent WAIT=1             # block until the SSM command finishes
make bench-job MODE=ephemeral WALLCLOCK=480       # raise the 120-min wallclock kill switch (0 = disable)
```

Ephemeral runs schedule a `shutdown -h +120` at boot as a hard wallclock
kill switch.  Long sweeps that exceed two hours will see
`The system is going down for poweroff …` wall messages and get cut off;
bump or disable the timer with `WALLCLOCK=<minutes>` (or `WALLCLOCK=0`).
The instance still self-terminates at the end of the run via
`instance_initiated_shutdown_behavior=terminate`.

Partial results stream to S3 as each `(lattice, batch_size)` finishes, so
you don't have to wait for the full sweep to land in your inbox.  The
on-instance script writes a JSONL feed to
`s3://<bench-bucket>/runs/<run_id>/partial/results.jsonl` (one entry per
lattice/batch) and re-uploads `bench.log` every 20 s to
`s3://<bench-bucket>/runs/<run_id>/bench.log.partial`.  Two helpers:

```bash
make bench-runs                            # list runs in the bucket
make bench-tail RUN=<run_id>               # stream the partial JSONL
make bench-tail RUN=<run_id> LOG=1         # follow the live bench.log
```

### Profiling & monitoring

Use the Neuron SDK's built-in tools directly:

```bash
neuron-top                         # interactive htop-style snapshot
neuron-monitor --period 1          # JSON-stream metrics, one record/s
neuron-profile ...                 # kernel-level timelines (view in Perfetto)
```

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
│   └── connect_inf2.sh     # SSH helper for the provisioned instance
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
