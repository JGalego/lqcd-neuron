# QUDA vs LQCD-Neuron: A Detailed Comparison

This document compares the design, execution model, and capabilities of
[QUDA](https://github.com/lattice/quda) — the GPU library for Lattice QCD — against
**LQCD-Neuron**, which targets AWS Trainium (Trn1) and Inferentia (Inf2) via the
[AWS Neuron SDK](https://awsdocs-neuron.readthedocs-hosted.com/).

---

## The core constraint

AWS Trainium and Inferentia are **not general-purpose GPUs**.  They expose no CUDA
API, no HIP interface, and no OpenCL runtime.  NeuronCores are purpose-built matrix
engines; the only supported programming model is "compile a PyTorch/JAX *computation
graph* with neuronx-cc and run the resulting binary".

This makes a line-for-line translation of QUDA impossible.  Instead, LQCD-Neuron
re-expresses every Lattice QCD kernel as a **pure-tensor `torch.nn.Module`** that
the Neuron compiler can lower to NeuronCore-v2 instructions.

---

## Side-by-side comparison

| Dimension | QUDA | LQCD-Neuron |
|---|---|---|
| **Target hardware** | NVIDIA GPUs (Ampere, Hopper, Volta, …) | AWS Trainium Trn1, Inferentia Inf2 |
| **Accelerator type** | General-purpose GPU (GPGPU) | Purpose-built matrix/ML accelerator |
| **Compute interface** | CUDA runtime API + PTX/SASS | PyTorch/XLA → neuronx-cc → NeuronCore ISA |
| **Kernel authoring** | Hand-written CUDA C++ kernels (`.cuh`/`.cu`) | Standard `torch.nn.Module` (Python) |
| **JIT vs AoT** | CUDA JIT at first kernel launch; tuned kernels cached | Fully ahead-of-time: `torch_neuronx.trace` once → `.neff` binary |
| **Shape flexibility** | Fully dynamic (kernel re-launched per shape) | Static shapes at compile time; bucketing for dynamic shapes |
| **Mixed precision** | FP64, FP32, FP16, INT8, FP8 per-field; mixed in solver | FP32, BF16 (Trn1/Inf2 native); automatic mixed precision via AMP |
| **Preferred precision** | FP16 (Ampere/Hopper tensor cores) | BF16 (NeuronCore-v2 MXU) |
| **Auto-tuning** | Runtime kernel auto-tuner per lattice shape | `torch.compile` and neuronx-cc O1/O2/O3 flags |
| **Multi-device comms** | NCCL (GPU↔GPU NVLink/NVSwitch) + MPI | EFA (Elastic Fabric Adapter) + MPI; NeuronLink (on-chip) |
| **Multi-device API** | `QudaCommsParam`, `splitGrid` | `torch.distributed` with Neuron backend; `xm.mark_step()` |
| **Host language** | C / C++ library with Python (`pyQUDA`) and Fortran bindings | Pure Python (`torch.nn.Module`) |
| **Fermion formulations** | Wilson, Clover-Wilson, Staggered (HISQ/ASQTAD), Twisted Mass, Domain Wall (DWF/Möbius), Overlap | Wilson, Clover-Wilson *(Staggered, DWF planned)* |
| **Preconditioners** | Even-odd, symmetric/asymmetric, multigrid (MG) | Even-odd *(planned)*; MG not yet implemented |
| **Solvers** | CG, CGNE, CGNR, BiCGStab, GCR, GMRES-DR, CA-CG, Deflated CG, Multigrid | CG, BiCGStab |
| **Eigensolver** | Thick-restart Lanczos (TRLM), Block TRLM, IRAM | Not yet implemented |
| **Gauge actions** | Wilson, Symanzik, HISQ, Iwasaki | Wilson |
| **Gauge smearing** | APE, Stout, HYP, Wilson flow | Not yet implemented |
| **Topology / observables** | Plaquette, Polyakov loop, topological charge (clover + cooling) | Plaquette, Wilson action, Polyakov loop, topological charge (clover) |
| **Memory management** | Custom GPU allocator (`malloc_quda`), pinned host memory | PyTorch tensor allocator; XLA buffer recycling |
| **Boundary conditions** | Periodic, anti-periodic (per-dimension) | Anti-periodic (T), periodic (spatial) |
| **Communication** | QUDA internal ghost-exchange (`QudaComm`) | `torch.roll` with periodic boundaries (no halo exchange yet) |
| **Distributed memory** | Multi-GPU domain decomposition with Schwarz-alternating | Not yet implemented |
| **Profiling / autotuning** | `TuneKey` cache, NVTX ranges, QUDA timer | PyTorch profiler, `torch.autograd.profiler` |
| **License** | MIT | Apache 2.0 |
| **Language / LOC** | C++ / CUDA (~700 K LOC) | Python (~2 K LOC) |
| **Installation** | CMake build against CUDA toolkit (complex) | `pip install lqcd-neuron[neuron]` (simple) |

---

## Execution model deep-dive

### QUDA execution flow

```
Lattice QCD application (C++)
        │
        │  loadGaugeQuda() / invertQuda() / …
        ▼
  QUDA C++ layer
        │
        │  qudaLaunchKernel<KernelType>(…)
        ▼
  CUDA runtime kernel launch
        │
        │  grid × block threads on GPU SMs
        ▼
  PTX / SASS executing on NVIDIA CUDA cores
  (matrix ops on Tensor Cores for half, mixed precision)
```

Key properties:
- Kernels are custom CUDA C++ with explicit shared memory and register management.
- The auto-tuner (`tune_quda.h`) sweeps block sizes at runtime and caches the
  optimal configuration per kernel / lattice shape.
- Communication between multiple GPUs uses NCCL or NVSHMEM.

### LQCD-Neuron execution flow

```
User Python script
        │
        │  WilsonDirac(mass=0.1)
        ▼
  lqcd_neuron nn.Module (pure PyTorch tensors)
        │
        │  torch_neuronx.trace(model, example_inputs)
        ▼
  XLA HLO computation graph  (torch → xla lowering)
        │
        │  neuronx-cc compilation (ahead-of-time)
        ▼
  .neff binary (NeuronCore Executable File Format)
        │
        │  Neuron runtime loads .neff onto NeuronCores
        ▼
  NeuronCore-v2 MXU executing BF16/FP32 matrix ops
  (NeuronLink for on-chip data movement)
```

Key properties:
- No CUDA.  The entire program is a PyTorch graph compiled by `neuronx-cc`.
- The solver convergence loop runs on the **host**; only `D.forward()` crosses to
  the NeuronCores — the same pattern as a deep-learning training loop.
- Compilation happens once per unique `(model, shape, dtype)` tuple; the result
  is cached to disk as a `.neff` file.

---

## What cannot be ported directly

The following QUDA features rely on CUDA-specific mechanisms with no Neuron equivalent;
re-implementation would require fundamentally different approaches:

| QUDA feature | Why it can't be directly ported | Alternative path |
|---|---|---|
| Hand-written CUDA kernels | NeuronCores have no custom kernel interface | Re-express as `torch.einsum` / matmul |
| NCCL multi-GPU comms | NCCL is NVIDIA-specific | `torch.distributed` over EFA + MPI |
| NVSHMEM / peer-to-peer DRAM | NVIDIA-specific hardware feature | NeuronLink (intra-node) + EFA (inter-node) |
| Tensor Core (WMMA) intrinsics | CUDA-only | neuronx-cc auto-vectorises BF16 matmuls |
| Dynamic kernel auto-tuner | Requires runtime kernel re-compilation | One-time AoT compilation with O2/O3 flags |
| CUDA streams / async execution | CUDA-specific | XLA async dispatch; `xm.mark_step()` |
| FP8 / INT8 kernels | CUDA-only quantisation path | BF16 (Trn1/Inf2 native); INT8 via Neuron quantisation API *(planned)* |
| Multigrid preconditioner | Algorithm feasible; operator-projection heavy | Representable as matmul graph; not yet implemented |

---

## When to choose which library

| Scenario | Recommended |
|---|---|
| Maximum performance on NVIDIA GPUs with all fermion formulations | **QUDA** |
| Cost-optimised inference on AWS (propagator calculation) | **LQCD-Neuron on Inf2** |
| Training a ML surrogate / distillation model on gauge configurations | **LQCD-Neuron on Trn1** |
| Embedding lattice QCD inside a JAX/PyTorch ML pipeline | **LQCD-Neuron** |
| Multi-GPU production HMC on existing GPU cluster | **QUDA** |
| Exploring Lattice QCD on AWS without CUDA expertise | **LQCD-Neuron** |
