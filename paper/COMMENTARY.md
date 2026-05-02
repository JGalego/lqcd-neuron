# A Sentence-by-Sentence Commentary on *LQCD-Neuron*

*A reader's companion to* "LQCD-Neuron: Compiling Lattice QCD Kernels to AWS
Trainium and Inferentia via PyTorch–XLA" *for the non-specialist who is willing
to learn some physics and computing along the way.*

---

## A five-minute primer on QCD (and why it needs a supercomputer)

Everything around you is built out of atoms, atoms are built out of nuclei
plus electrons, and nuclei are built out of protons and neutrons. Zoom in
once more and the protons and neutrons themselves are made of even smaller
particles called **quarks**, glued together by particles called, fittingly,
**gluons**. The theory that describes how quarks and gluons interact is
**Quantum Chromodynamics**, or **QCD** — the "C" stands for *colour*, a
quantum-mechanical label for quarks that plays the role electric charge plays
in electromagnetism, except there are three "colours" instead of one kind of
charge.

QCD has two infuriating features:

1. **Confinement.** You can never isolate a single quark. The force between
   quarks does not weaken with distance the way gravity or electromagnetism
   do; if you try to pull two quarks apart, the energy you put in eventually
   creates a brand-new pair of quarks rather than letting the original two
   separate. This is why we see protons and neutrons but never a lone quark.
2. **Strong coupling at low energies.** The standard physicist's trick of
   writing the answer as a sum of small corrections (perturbation theory)
   *fails* for QCD at the energies relevant to nuclei. The corrections aren't
   small.

The workaround, invented by Kenneth Wilson in 1974, is **Lattice QCD**: pretend
spacetime is not continuous but a 4-dimensional grid of points (three space
dimensions plus one time dimension), like a crystal. You put the quark fields
on the *sites* of the grid and the gluon fields on the *links* connecting
neighbouring sites. Then, instead of doing calculus, you do a gigantic but
finite integral on a computer. The catch is that "gigantic" is an
understatement: a realistic lattice might have $32^4 \approx 10^6$ sites, and
the central calculation — applying the **Dirac operator** to a quark field —
has to be done tens of thousands of times for each "measurement". This is why
Lattice QCD has been a flagship application of supercomputers since the
1980s, and why every new generation of accelerator hardware (vector machines,
GPUs, and now AI chips) gets pressed into service.

The paper we are about to read asks: *can the AI accelerators that Amazon
designs for training neural networks — chips called Trainium and Inferentia —
also run Lattice QCD?* These chips are unusual: unlike GPUs, you cannot just
write a program for them and run it. You have to compile your whole
calculation ahead of time into a fixed binary, and then the chip executes
exactly that binary. The author shows that, with some clever reshaping, the
core Lattice QCD calculation does fit into this rigid mould, and on small
problems it runs roughly an order of magnitude faster than a tuned CPU.

With that context, let us read the paper.

---

## Title and authorship

> **LQCD-Neuron: Compiling Lattice QCD Kernels to AWS Trainium and Inferentia via PyTorch–XLA**

The title compresses the whole project. **LQCD** is Lattice QCD. **Neuron** is
the name Amazon gives to the software stack that drives its in-house AI chips;
**Trainium** (chip family code-named *Trn1*) and **Inferentia** (*Inf2*) are
those chips, designed respectively for training and serving neural networks
([AWS Neuron SDK docs](https://awsdocs-neuron.readthedocs-hosted.com/)).
**PyTorch** is the dominant Python framework for deep learning
(<https://pytorch.org/>); **XLA** ("Accelerated Linear Algebra") is a compiler
originally built by Google that turns high-level tensor operations into
machine code for specialised hardware (<https://openxla.org/xla>). "Compiling
kernels" means: take the small, performance-critical numerical routines
("kernels") that dominate the runtime, and turn them into a binary that the
accelerator can execute.

> **João Galego — Independent Researcher**

Single-author paper, no institutional affiliation. Worth flagging because
Lattice QCD papers usually have long author lists from large collaborations.

---

## Abstract

> We present LQCD-Neuron, a Python library that runs Wilson- and Clover-Wilson Lattice QCD on AWS Trainium (Trn1) and Inferentia (Inf2) accelerators.

The deliverable is software, written in Python, that runs two specific
**discretisations** of the Dirac operator. A discretisation is a recipe for
turning the continuous equations of QCD into something a computer can evaluate
on a grid. **Wilson fermions** ([Wilson 1974](https://doi.org/10.1103/PhysRevD.10.2445))
are the original recipe; **Clover-Wilson** ([Sheikholeslami–Wohlert 1985](https://doi.org/10.1016/0550-3213(85)90002-1))
is a refinement that adds an extra term — shaped a bit like a four-leaf clover
when drawn on the lattice — to reduce systematic errors of order $a$, where
$a$ is the lattice spacing.

> The hardware exposes no general-purpose compute API (no CUDA, HIP, or OpenCL); the only programming model is to compile a PyTorch / XLA graph ahead of time with `neuronx-cc` and execute the resulting NeuronCore Executable File Format (.NEFF) binary.

This is the central constraint of the work. On NVIDIA GPUs, scientists have
**CUDA** ([NVIDIA CUDA](https://docs.nvidia.com/cuda/)); on AMD GPUs, **HIP**
([ROCm/HIP](https://rocm.docs.amd.com/projects/HIP/)); historically, the
cross-vendor option was **OpenCL** ([Khronos OpenCL](https://www.khronos.org/opencl/)).
All three let you hand-write a program in a C-like language that runs directly
on the chip. The Neuron chips offer no such freedom. The only path is:
(1) build a *computation graph* (a description of which tensor operations
feed into which) using PyTorch; (2) hand it to Amazon's compiler `neuronx-cc`,
which converts it via Google's XLA intermediate representation into a binary
file with the extension `.neff`; (3) load the `.neff` and run it. **Ahead of
time** (AoT) compilation means the chip does not adapt to your input at
runtime — every shape and size is frozen at compile time.

> We re-express every hot-path lattice kernel — the Wilson Dslash, the clover term, BLAS-1 reductions, and CG/BiCGStab solvers — as a pure `torch.nn.Module`, and rely on three compile-time transformations (gauge baking, fused 12×12 spin–colour matvecs, multi-RHS batching) plus a host-side data-parallel shard across all on-chip NeuronCores.

A few terms unpack as follows. **Hot-path** means the small fraction of code
where the program spends almost all its time. **Wilson Dslash** ($D\!\!\!\!/$,
pronounced "D-slash") is the discretised Dirac operator, the heart of the
calculation. The **clover term** is the extra piece in the Clover-Wilson
recipe. **BLAS-1** refers to "level-1 Basic Linear Algebra Subprograms" — the
simple, vector-on-vector operations like dot products and norms ([Lawson et al.
1979](https://dl.acm.org/doi/10.1145/355841.355847)). **CG** is the
Conjugate Gradient method ([Hestenes & Stiefel 1952](https://nvlpubs.nist.gov/nistpubs/jres/049/jresv49n6p409_a1b.pdf))
and **BiCGStab** is the Biconjugate Gradient Stabilised method
([van der Vorst 1992](https://doi.org/10.1137/0913035)) — both are iterative
algorithms for solving the giant linear system $D x = b$ that you need to
solve to compute a "quark propagator". A **`torch.nn.Module`** is the basic
PyTorch building block for a piece of computation; normally it represents a
neural-network layer, but here it represents a physics operator.

The three "compile-time transformations" are the paper's main technical
contribution; each will be explained in detail later. **Gauge baking**:
freeze the gluon field into the binary so it doesn't have to be re-sent every
call. **Fused 12×12 matvecs**: combine the 4-component "spin" structure and
the 3-component "colour" structure into a single 12×12 matrix multiply, which
fits the chip's matrix engine better. **Multi-RHS batching**: solve many
right-hand sides at once. The "host-side data-parallel shard" means the
ordinary Python program running on the CPU splits the work across all the
NeuronCores on the chip.

> On a single-core Inf2 NeuronCore-v2 in BF16 we obtain up to 13.0× throughput over a tuned FP32 CPU baseline for the Wilson Dslash on V=4×4×4×4 (two NeuronCores, B=8 RHS each), and establish that throughput reaches CPU parity at V=16×8×8×8 and falls below beyond that, all with no custom kernel code.

The headline numbers, with appropriate humility. **BF16** ("brain float 16")
is a 16-bit floating-point format with the same exponent range as standard
32-bit floats but less precision in the fractional part; it was introduced by
Google for machine-learning workloads
([Wang & Kanwar, Google AI blog 2019](https://cloud.google.com/blog/products/ai-machine-learning/bfloat16-the-secret-to-high-performance-on-cloud-tpus)).
**FP32** is the standard 32-bit float. The comparison is therefore against a
*more accurate* CPU baseline. **V=4×4×4×4** is a tiny 4-dimensional lattice
(256 sites) used as a benchmark — far smaller than a physics-grade run, but
large enough to demonstrate the engineering. **B=8 RHS** means the chip
solves eight right-hand sides simultaneously. The honest caveat: the speedup
*disappears* on bigger lattices, because the on-chip memory runs out. "No
custom kernel code" is the boast: everything is plain PyTorch.

> Source code is available at github.com/JGalego/lqcd-neuron under the Apache 2.0 license.

Permissive open-source licence ([Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0)).
Anyone can use, modify, and redistribute, including commercially.

---

## 1. Introduction

> The cost of a Lattice QCD simulation on heterogeneous hardware is dominated by repeated applications of the discretised Dirac operator $D$ to a quark spinor field $\psi$.

"Heterogeneous" means a mix of CPU and accelerator. The sentence states a
well-known empirical fact: profilers consistently show that 80–95% of wall
time in a production Lattice QCD run is spent inside one routine — applying
$D$ to $\psi$. A **spinor field** $\psi$ is the mathematical object that
represents quarks: at every lattice site it is a vector of 12 complex
numbers (4 spin components × 3 colour components). See [Gattringer & Lang
2010](https://link.springer.com/book/10.1007/978-3-642-01850-3), *Quantum
Chromodynamics on the Lattice*, the standard textbook.

> In the Wilson formulation, $(D_W \psi)_x = (4+m)\,\psi_x - \tfrac{1}{2}\sum_{\mu=0}^{3}[(I-\gamma_\mu)\,U(x,\mu)\,\psi_{x+\hat\mu} + (I+\gamma_\mu)\,U^{\dagger}(x-\hat\mu,\mu)\,\psi_{x-\hat\mu}]$

The defining equation. In words: the value of $D\psi$ at site $x$ is a linear
combination of $\psi$ at $x$ itself and at its eight nearest neighbours — two
in each of the four spacetime directions $\mu=0,1,2,3$. The factor
$U(x,\mu)$ is the **gauge link**: a $3\times 3$ unitary matrix that lives
on the edge connecting site $x$ to site $x+\hat\mu$, encoding the gluon
field. The matrices $\gamma_\mu$ are the **Dirac gamma matrices**, four
$4\times 4$ matrices that act on the spin index ([Dirac 1928](https://doi.org/10.1098/rspa.1928.0023)).
The constant $m$ is the bare quark mass. The dagger means Hermitian conjugate
(transpose-and-complex-conjugate). [Wilson's 1974 paper](https://doi.org/10.1103/PhysRevD.10.2445)
introduced this expression.

> with gauge links $U(x,\mu) \in \mathrm{SU}(3)$ and Dirac matrices $\gamma_\mu$ acting on the four-component spinor.

Spelling out the types. **SU(3)** is the **Special Unitary group** of degree 3:
the set of $3\times 3$ complex matrices that are unitary ($U^\dagger U = I$)
and have determinant 1. It has 8 real parameters, which is why there are 8
gluons in QCD ([Yang–Mills 1954](https://doi.org/10.1103/PhysRev.96.191)).

> A single application performs $\mathcal{O}(V)$ tiny dense $3\times 3$ and $4\times 4$ contractions, and a Krylov inversion invokes it tens of thousands of times.

$V$ is the number of lattice sites; $\mathcal{O}(V)$ is computer-science
shorthand for "proportional to $V$". The phrase **"tiny dense"** is the key
performance issue: the matrices are dense (every entry is non-zero) but
only $3\times 3$ or $4\times 4$, which is way too small to keep a modern
matrix-multiplication unit busy. A **Krylov inversion** is an iterative
linear-system solver (CG, BiCGStab, GMRES, etc.); the name honours
[Aleksei Krylov (1931)](https://en.wikipedia.org/wiki/Krylov_subspace).
Each iteration calls the operator $D$ once or twice; convergence on a
realistic problem takes thousands to tens of thousands of iterations.

> On NVIDIA GPUs the standard tool is QUDA: hand-written CUDA kernels, dynamic shapes, runtime auto-tuning, and a multigrid preconditioner.

**QUDA** ("QCD on CUDA") is the dominant open-source library for running
Lattice QCD on GPUs ([Clark et al. 2010](https://arxiv.org/abs/0911.3191);
[Babich et al. 2011](https://arxiv.org/abs/1109.2935); <https://github.com/lattice/quda>).
Each adjective in the list is a luxury that AI accelerators do not afford:
*hand-written* (you cannot write CUDA for Neuron); *dynamic shapes* (Neuron
needs static shapes); *runtime auto-tuning* (Neuron compiles ahead of time);
*multigrid preconditioner* (a hierarchy-of-grids technique that makes the
solver many times faster, [Brower et al. 2018](https://arxiv.org/abs/1801.07823)).

> We ask whether comparable performance is reachable on AWS-designed ML accelerators — Trainium (Trn1) and Inferentia (Inf2) — which expose a fundamentally different programming model.

The research question. **ML** = machine learning. The accelerators are
proprietary chips Amazon designed in-house (the work has roots in the
acquisition of Annapurna Labs); see [AWS Trainium product page](https://aws.amazon.com/machine-learning/trainium/)
and [AWS Inferentia product page](https://aws.amazon.com/machine-learning/inferentia/).

> Each NeuronCore-v2 is a purpose-built matrix engine.

A **NeuronCore** is the basic compute unit of a Neuron chip, much as a
"streaming multiprocessor" is the basic unit of a GPU. *v2* is the second
generation, used in both Trn1 and Inf2. "Purpose-built matrix engine" means
its hardware is specialised for the operation $C = A \times B$ on matrices,
because that operation dominates neural-network workloads.
([NeuronCore-v2 architecture](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/arch/neuron-hardware/neuroncores-arch.html))

> It offers no general-purpose compute API, no user-controlled shared memory, and no runtime kernel launch.

Three things programmers normally take for granted that are absent. A
**general-purpose compute API** lets you write arbitrary code (loops, branches,
pointer arithmetic) for the chip. **User-controlled shared memory** is fast
on-chip scratchpad memory whose contents the programmer manages explicitly,
e.g. CUDA's `__shared__` memory. **Runtime kernel launch** is the ability for
the host program to decide *while running* to fire off a fresh GPU computation.
On Neuron, the *only* code that runs on the chip is what was baked into the
`.neff` file at compile time.

> Code reaches the hardware along a single path: a PyTorch (or JAX) computation graph is lowered ahead of time to XLA HLO, the `neuronx-cc` compiler emits a .NEFF binary, and the Neuron runtime executes it on the NeuronCores.

The complete pipeline. **JAX** is Google's research-oriented PyTorch
alternative (<https://github.com/google/jax>). **Lowered** is compiler
jargon for "translated to a lower-level representation". **XLA HLO**
("High-Level Optimizer" representation) is XLA's intermediate language
([XLA documentation](https://openxla.org/xla)). The *neuronx-cc* compiler
takes HLO as input and emits the `.neff` binary
([neuronx-cc reference](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/compiler/neuronx-cc/index.html)).

> Tensor shapes must be static at compile time, and the auto-tuning and runtime kernel re-launch on which QUDA relies are simply not available.

Restating the constraint. A tensor's **shape** is the tuple of its dimensions
— for example, a $4\times 4\times 4\times 4$ lattice with 12 numbers per
site has shape (4,4,4,4,12). "Static" means every dimension is a known
integer at compile time, not a variable. **Auto-tuning** means trying many
versions of a kernel at runtime and keeping the fastest, an essential trick
on GPUs; on Neuron there is no runtime in which to do it.

> Trainium has been shown to scale deep-learning training, but to our knowledge has not previously been targeted for first-principles Lattice QCD.

Positioning the novelty. There are AWS case studies of Trainium clusters
training large language models efficiently
([AWS Neuron SDK documentation](https://awsdocs-neuron.readthedocs-hosted.com/));
no published Lattice QCD work to date. **First-principles** means
calculating from the underlying theory (the QCD Lagrangian) without
empirical fits — as opposed to phenomenological models.

> The central question of this work is therefore: is the AoT-compiled, static-graph, matrix-engine programming model expressive enough for production Lattice QCD primitives, and competitive in throughput once the graph is shaped to fit the hardware?

Restating the research question with extra precision. **Expressive enough**
asks whether all the *operations* needed by Lattice QCD can be written down
within Neuron's restrictions. **Competitive in throughput** asks whether the
*speed* is acceptable. **Primitives** are the basic operators (Dslash,
clover, etc.).

> We give an affirmative answer for the Wilson and Clover-Wilson Dslash and the iterative solvers built on them, with four composable graph-level optimisations that bring throughput from below CPU at modest volumes to up to 13.0× above it on a two-core Inf2 configuration.

The summary of contributions. **Composable** means the four optimisations
can be turned on independently and combined; you do not have to take all or
none. **Graph-level** means they manipulate the computation graph rather
than writing low-level chip code.

---

## 2. Design and architecture

### Pure-tensor kernels.

> Every primitive in LQCD-Neuron is a standard `torch.nn.Module` operating on dense complex tensors with a (T,Z,Y,X,…) storage order.

The architectural choice. Each physics operator is wrapped as a PyTorch
neural-network module, which means it gets all of PyTorch's plumbing for free
— autograd, serialisation, Neuron tracing — even though there is no neural
network involved. **Dense complex tensors** means n-dimensional arrays where
every entry is a complex number and is stored explicitly (no sparsity
compression). **(T,Z,Y,X,…)** is the **storage order**: temporal axis first,
then the three spatial axes, then the spin/colour indices.

> Gauge links live in $U \in \mathbb{C}^{T\times Z\times Y\times X\times 4\times N_c\times N_c}$ and spinors in $\psi \in \mathbb{C}^{T\times Z\times Y\times X\times N_s\times N_c}$, with $N_c=3$, $N_s=4$.

The shapes spelled out. The **4** in the gauge-link tensor is the four
spacetime directions: at every site there are four links, one per direction.
$N_c=3$ ("number of colours") and $N_s=4$ ("number of spin components") are
defining numbers of QCD; $N_s=4$ comes from the 4-dimensional Dirac spinor
of the [Dirac equation (1928)](https://doi.org/10.1098/rspa.1928.0023) and
$N_c=3$ from the three "colour charges" of QCD.

> Lattice shifts $\psi_{x\pm\hat\mu}$ are realised by `torch.roll` with the appropriate axis index, which the Neuron compiler lowers to in-place data movement on chip.

Recall from the Dslash equation that you need the spinor at neighbouring
sites. **`torch.roll`** ([PyTorch docs](https://pytorch.org/docs/stable/generated/torch.roll.html))
is a built-in operation that cyclically shifts a tensor along an axis: the
last entry wraps around to become the first. For periodic boundary
conditions this is exactly what is needed. **In-place data movement** means
the compiler can rearrange data without copying it — the bytes physically
stay where they are; only the addresses change.

> Since `neuronx-cc` forbids data-dependent control flow, the Python loop over the four directions in Eq. (1) is statically unrolled at trace time, producing a pure DAG that the compiler can schedule.

**Data-dependent control flow** = "if-statements whose outcome depends on
the values of the data". Forbidden because Neuron must know the entire
operation list at compile time. **Statically unrolled** = the Python `for`
loop over $\mu=0,1,2,3$ is executed *during compilation*, writing out four
copies of the body of the loop, one per direction. **DAG** = directed
acyclic graph: a list of operations with their dependencies and no cycles,
which a compiler can schedule onto hardware ([Aho et al. 2006, *Compilers:
Principles, Techniques, and Tools*](https://www.pearson.com/en-us/subject-catalog/p/compilers-principles-techniques-and-tools/P200000003472)).

### Gamma matrices.

> The Euclidean Dirac matrices are stored in the DeGrand–Rossi (chiral) basis and registered as non-trainable `nn.Module` buffers, so they migrate with the module to NeuronCore memory under `torch_neuronx.trace` and remain visible to the tracer.

**Euclidean** matrices because Lattice QCD is done in [Euclidean spacetime](https://en.wikipedia.org/wiki/Wick_rotation),
not Lorentzian — a "Wick rotation" of time to imaginary values turns the
oscillating QCD integral into a convergent one suitable for Monte Carlo.
The **DeGrand–Rossi** ([DeGrand & Toussaint 1990](https://www.worldscientific.com/worldscibooks/10.1142/1241))
basis is one specific set of $\gamma_\mu$ matrices; "chiral" means it makes
the chirality (handedness) operator $\gamma_5$ block-diagonal, which is
convenient. **Non-trainable buffers** in PyTorch are tensors that travel
with a module but are not updated by training (no gradients computed). The
trick is that they get serialised into the `.neff` automatically, so the
gamma-matrix constants live on the chip and don't need to be uploaded each
call.

### Host loop / device kernel split.

> Iterative solvers need a convergence test on a scalar reduction at every iteration, but `neuronx-cc` only supports static graphs.

A solver like CG checks at each iteration whether the residual norm
$\|r\|$ has fallen below some tolerance — and if so, breaks out of the loop.
Breaking out of a loop is a data-dependent control-flow operation; Neuron
forbids those.

> We adopt the standard deep-learning split: the Krylov loop runs on the host (CPU) and only the matvec $D\psi$ is compiled to NeuronCore.

The trick deep-learning libraries use. The expensive parts (large matrix
multiplies) go to the accelerator; the cheap, decision-making parts stay
on the CPU. **Matvec** = matrix-times-vector.

> For Conjugate Gradient, schematically: [pseudo-code: alpha = rho / inner(p, Ap); x += alpha*p; r -= alpha*Ap; if norm(r) < tol: break]

This is the classic CG iteration ([Hestenes & Stiefel 1952](https://nvlpubs.nist.gov/nistpubs/jres/049/jresv49n6p409_a1b.pdf);
[Shewchuk's introduction (1994)](https://www.cs.cmu.edu/~quake-papers/painless-conjugate-gradient.pdf)),
applied to $A = D^\dagger D$. Only the line `Ap = D_neuron(p, U)` runs on
the accelerator. The rest are tiny vector arithmetic operations on the
CPU.

> The host overhead per iteration is $\mathcal{O}(V)$ and is dwarfed by the matvec for any non-trivial lattice; we verified this empirically against an all-on-host reference.

The cost of the host-side bookkeeping grows linearly with lattice size $V$,
but with a *much smaller constant* than the matvec, which involves about
1320 floating-point operations per site
([Babich et al. 2011](https://arxiv.org/abs/1109.2935)).

### CPU fallback.

> Because every kernel is a plain PyTorch module, the same code runs identically on a laptop without Neuron hardware.

A nice consequence of the design choice. PyTorch's tensor operations all
have a CPU implementation, so removing the Neuron compilation step still
leaves a working program.

> The full unit-test suite executes on CPU, allowing development and CI without an Inf2 instance.

**CI** = continuous integration, automated test runs on every code change.
Important practical point: developers don't need to rent an Inf2 instance
just to make sure they didn't break anything.

---

## 3. Compile-time optimisations

> A naive trace of Eq. (1) is dominated by overheads that are absent on a CPU baseline:

Setting up the four problems that the four optimisations will solve. A
**naive trace** = a literal translation of the PyTorch code to a Neuron
graph, with no cleverness.

> (i) the gauge field, ~3× the spinor in size, crosses PCIe on every call;

**PCIe** ("Peripheral Component Interconnect Express", [PCI-SIG](https://pcisig.com/specifications))
is the bus connecting the CPU to add-in cards like the Neuron accelerator.
It is fast in absolute terms but a bottleneck compared to on-chip memory.
The gauge field $U$ is roughly three times the size of the spinor $\psi$
because of its extra direction index and extra colour index; sending it
over PCIe every time you call $D\psi$ is wasteful.

> (ii) the per-direction kernel is two tiny einsums (a 3×3 colour multiply and a 4×4 spin multiply), neither of which saturates the NeuronCore tensor engine;

An **einsum** ("Einstein summation") is a generic notation for tensor
contractions ([NumPy einsum docs](https://numpy.org/doc/stable/reference/generated/numpy.einsum.html));
PyTorch supports it. The complaint is that $3\times 3$ and $4\times 4$
matrix multiplies are far too small to keep busy a tensor engine designed to
multiply, say, $128\times 128$ matrices in a single cycle.

> (iii) eight `torch.roll`s per call (four for $\psi$, four for $U^\dagger(x-\hat\mu)$) are pure data movement; and

Pure shuffling of bytes around, no actual arithmetic, hence wasted cycles.

> (iv) every NeuronCore call pays a fixed dispatch cost (~1 ms), independent of problem size.

**Dispatch cost** = the latency to start any computation on the chip,
roughly 1 millisecond on Neuron. For tiny problems, this is the dominant
expense — the chip spends more time waking up than computing.

> We address these in four composable steps.

Forward reference to the four subsections.

### 3.1 Gauge baking

> When a gauge configuration $U$ is supplied at compile time, we register it as an `nn.Module` buffer of the traced module rather than as a graph input.

A **gauge configuration** is one specific assignment of values to all the
$U(x,\mu)$ links — a single sample from the QCD probability distribution,
typically generated separately by a Monte Carlo simulation called Hybrid
Monte Carlo ([Duane et al. 1987](https://doi.org/10.1016/0370-2693(87)91197-X)).
A typical Lattice QCD analysis fixes one $U$ and then computes lots of
quark propagators on it, so it makes sense to "bake" $U$ into the binary.

> After `torch_neuronx.trace`, the resulting .NEFF contains $U$ in NeuronCore-resident memory.

**Tracing** ([torch.jit.trace docs](https://pytorch.org/docs/stable/generated/torch.jit.trace.html))
is the process of running a PyTorch model with example inputs and recording
the operations it performs, producing a static graph. **NeuronCore-resident
memory** = the on-chip memory of the accelerator (HBM and SRAM together),
as opposed to host RAM.

> At inference time, only $\psi$ crosses PCIe; the wrapper accepts a $U$ argument for API compatibility but ignores it.

Saves ~75% of the PCIe traffic (since $U$ was ~3× the size of $\psi$).
Keeping the unused $U$ argument in the function signature is a common
software-engineering trick to avoid breaking callers.

> The trade-off is that the .NEFF is specific to a particular $U$, so it is not reused across different gauge configurations — the right choice for a single inversion or a chain of solves on one configuration, the wrong choice for HMC.

The cost of baking. **HMC** = [Hybrid Monte Carlo](https://doi.org/10.1016/0370-2693(87)91197-X),
the algorithm used to *generate* gauge configurations; it would change $U$
on every step, making a baked binary useless. The author is being honest
about scope: this library targets *propagator inversions*, not configuration
generation.

### 3.2 Fused spin–colour kernels

> With $U$ baked in, the per-site, per-direction operator $K^{\mathrm{fwd}}_{\mu,x} = (I-\gamma_\mu) \otimes U(x,\mu)$, $K^{\mathrm{bwd}}_{\mu,x} = (I+\gamma_\mu) \otimes U^\dagger(x-\hat\mu,\mu)$ can be precomputed once at trace time.

The symbol $\otimes$ is the **Kronecker product** ([Wikipedia](https://en.wikipedia.org/wiki/Kronecker_product)):
combining a $4\times 4$ spin matrix and a $3\times 3$ colour matrix gives a
$12\times 12$ block matrix. The point is that *both* "parts" of the Wilson
Dslash for a given direction can be combined into a single matrix multiply
at compile time, which is *also* the time at which $U$ is known.

> We materialise $K^{\mathrm{fwd/bwd}}_{\mu,x} \in \mathbb{C}^{N_sN_c \times N_sN_c}$ as four buffers of shape $(4,T,Z,Y,X,12,12)$ stored as real/imaginary float pairs.

**Materialise** = actually allocate memory and store the values. Eight
buffers in total (forward and backward × four directions), each of which
is a 12×12 complex matrix per site. Stored as pairs of real-valued floats
because Neuron's matrix engine works on real numbers, so complex arithmetic
must be expressed as real-arithmetic on twice as many numbers.

> Each direction-side then collapses to one 12×12 matvec plus a single roll of $\psi$.

Two operations replace what was previously several. The 12×12 matvec is
exactly the size at which the Neuron MXU starts being efficient.

> The 12×12 shape is markedly better suited to the NeuronCore matrix multiplication unit (MXU) than the original 3×3 / 4×4 einsums.

The **MXU** ("Matrix Multiplication Unit") is the systolic-array core of
the NeuronCore — a 2D grid of multiply-accumulate cells, similar in spirit
to a Google TPU's MXU ([Jouppi et al. 2017](https://arxiv.org/abs/1704.04760)).
Bigger inputs keep more cells active per cycle.

> The four backward rolls are absorbed into $K^{\mathrm{bwd}}$, halving the lattice-shift count.

A free win: because the displaced $U^\dagger(x-\hat\mu)$ is precomputed and
stored at the displaced position, no runtime shift of $U$ is needed — only
$\psi$ still needs to be shifted.

> The fused kernel is bit-equivalent to the reference `forward()` up to FP32 round-off (relative error $\lesssim 10^{-7}$), and continues to satisfy the existing $\gamma_5$-Hermiticity, adjoint, and normal-operator positivity tests.

Numerical correctness check. **FP32 round-off** is about $10^{-7}$, the
unit-in-the-last-place of single precision ([IEEE 754](https://standards.ieee.org/standard/754-2019.html)).
**$\gamma_5$-Hermiticity**: the Wilson Dirac operator satisfies the
identity $\gamma_5 D \gamma_5 = D^\dagger$, a deep property tied to the
chiral symmetry of QCD. **Adjoint**: $D^\dagger$ behaves correctly.
**Normal-operator positivity**: $D^\dagger D$ has only positive
eigenvalues, which is what allows the Conjugate Gradient solver to work.
Standard correctness tests in any Lattice QCD code.

### SRAM budget guard.

> The eight $K^{\mathrm{fwd/bwd}}$ buffers have per-lattice footprint $8V \cdot 144 \cdot 4$ bytes (float32 real/imaginary pairs).

Memory accounting. 144 = 12 × 12 entries per matrix × 1 (already counted
the 12×12 = 144); the formula counts 8 buffers × $V$ sites × 144 (entries
per matrix) × 4 bytes (per FP32) — but note this is for the *real* part;
the comment "real/imaginary pairs" implies a similar amount for the
imaginary part. (A small ambiguity in the paper.)

> This already exceeds the 14.4 MiB NeuronCore on-chip SRAM at $V=16\times 8\times 8\times 8$ (36 MiB) and reaches 288 MiB at $V=16\times 16\times 16\times 16$.

**SRAM** ("Static Random-Access Memory") is the fast on-chip scratchpad,
distinct from the slower off-chip **HBM** ("High-Bandwidth Memory").
14.4 MiB is the NeuronCore-v2's SRAM budget
([NeuronCore-v2 architecture](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/arch/neuron-hardware/neuroncores-arch.html)).
Once the buffers don't fit in SRAM, every memory access has to go to HBM,
which is much slower, and the optimisation backfires.

> The compiler therefore checks the fused footprint against a configurable SRAM threshold before tracing; if the budget is exceeded it issues a warning and falls back to the unfused baked-gauge path, preserving correctness at reduced MXU efficiency.

Defensive engineering: do the optimisation only when it pays off, fall
back transparently otherwise. **Preserving correctness** is the polite
phrase: the code still gives the right answer; it just runs slower.

> The threshold can be overridden via `NeuronCompiler(sram_threshold_bytes=N)` or suppressed with `fused=False`.

User escape hatches.

### 3.3 Multi-right-hand-side batching

> Production multi-source propagator inverters already solve $D\,x = b_i$ for $N_s \cdot N_c = 12$ right-hand sides per source, one per spin × colour combination.

A **propagator** $D^{-1}$ encodes how a quark moves between two spacetime
points; it is a $12V \times 12V$ matrix. To use it in physics, you need
its action on 12 standard basis vectors at each source point — one for
each spin–colour combination — which is 12 separate linear solves with
the same $D$ but different right-hand sides $b_i$.

> We expose this directly: `compile_dslash_batched(D, shape, batch_size=B, gauge_field=U)` traces the fused operator on $\psi \in \mathbb{C}^{B \times T \times Z \times Y \times X \times N_s \times N_c}$.

Add a leading "batch" dimension to the spinor tensor. The Dslash is linear,
so applying it to a batch is the same as applying it to each member, which
the compiler can fuse into a single big matmul.

> The baked $K$ buffers carry a singleton batch axis that broadcasts across all $B$ RHS, and the lattice rolls use negative-dim indexing so the leading batch axis is unaffected.

**Broadcasting** ([NumPy broadcasting docs](https://numpy.org/doc/stable/user/basics.broadcasting.html))
lets a single $K$ matrix be reused across all $B$ right-hand sides without
copying. **Negative-dim indexing**: `dim=-1` means "the last axis",
independent of how many leading axes there are; this lets the same code
work whether or not there's a batch axis.

> Each direction-side becomes an $N_b \times 12 \times 12$ matmul — a much better fit for the MXU's systolic schedule — and the fixed per-call dispatch cost is amortised over $B$ problems.

Two wins from one optimisation. Bigger matmul = better MXU utilisation.
And the ~1 ms dispatch cost is now spread over $B$ separate problems, so
per-RHS dispatch cost becomes ~1/$B$ ms.

### 3.4 Multi-core data-parallel sharding

> A single NeuronCore-v2 is one of $N_{\mathrm{core}}$ cores per accelerator chip; `inf2.xlarge` exposes 2, `trn1.32xlarge` 32.

The chip count. **`inf2.xlarge`** is the smallest Inf2 EC2 instance, with
2 NeuronCores; **`trn1.32xlarge`** is the largest single-host Trainium
instance, with 32 cores. ([AWS instance types](https://aws.amazon.com/ec2/instance-types/inf2/),
[Trn1 instance types](https://aws.amazon.com/ec2/instance-types/trn1/).)

> We replicate the compiled .NEFF across all detected cores via `torch_neuronx.DataParallel` and split a global batch of $N_{\mathrm{core}} \cdot B_{\mathrm{core}}$ RHS along dim 0.

**Data parallelism** ([Dean et al. 2012](https://research.google/pubs/large-scale-distributed-deep-networks/))
is the simplest form of parallelism: replicate the model on every device,
split the input into chunks, run them in parallel, gather the outputs.
**`torch_neuronx.DataParallel`**
is the Neuron-specific version of [`torch.nn.DataParallel`](https://pytorch.org/docs/stable/generated/torch.nn.DataParallel.html).

> [pseudo-code: D_mc = compiler.compile_dslash_multicore(D, shape, gauge_field=U, num_cores=N, per_core_batch_size=B); out = D_mc(psi_global)]

The user-facing API. One function call, the multi-core orchestration is
hidden.

> This composes with the previous three optimisations rather than replacing them: gauge baking and fused kernels apply per core, the intra-core multi-RHS batch fills each core's tensor engine, and the data-parallel shard multiplies by $N_{\mathrm{core}}$.

The point of "composable": the four optimisations stack multiplicatively,
in principle giving up to (gauge-bake speedup) × (fuse speedup) × ($B$
per core) × ($N_{\mathrm{core}}$ cores).

---

## 4. Results

> Table 1 reports the throughput of the optimised LQCD-Neuron Wilson Dslash on an `inf2.8xlarge`, compared against a CPU baseline running the same reference `nn.Module` in FP32.

Setup of the benchmark. **`inf2.8xlarge`** has 2 NeuronCores
([instance specs](https://aws.amazon.com/ec2/instance-types/inf2/)).
Comparing FP32-on-CPU against BF16-on-Neuron is favourable to Neuron
(less precision = less work) but is the comparison most users actually
care about.

> Three observations are worth noting.
>
> First, the unoptimised single-RHS Neuron column (column 3) is only faster than CPU at the smallest volume; by V=8×8×8×4 it is slower by nearly 2×, and at V=16×16×16×16 by more than 8×.

The "naive" Neuron path is *slower* than CPU on most volumes. This is the
problem the optimisations are needed for.

> This confirms that the naive Dslash graph is dispatch- and shape-limited: the work per call is too small to amortise the ~1 ms NeuronCore launch cost, and the underlying 3×3 / 4×4 einsums leave the MXU idle.

Diagnosis: the chip is mostly idle, waiting either for the next dispatch
or for tensors big enough to fill its matrix engine.

> Second, multi-RHS batching closes the gap and overtakes CPU at all volumes up to V=8×8×8×4, but reaches only CPU parity at V=16×8×8×8 (1.0× with Multicore)† and falls clearly below at V=16×16×16×16 (0.3× with Multicore)†.

Batching helps on small lattices but cannot save the big ones, because…

> At both large volumes the fused 12×12 kernel buffers alone exceed the 14.4 MiB NeuronCore on-chip SRAM (36 MiB at 16×8×8×8 and 288 MiB at 16×16×16×16), so the compiler automatically falls back to the unfused baked-gauge path, eliminating the MXU-utilisation benefit of the fused kernels.

…the precomputed $K$ buffers no longer fit on chip. The fall-back to the
unfused path is correctness-preserving but throws away the MXU win.

> Third, the two-core DataParallel (Multicore) configuration delivers 6.4–13.0× over CPU from V=4⁴ through V=8×8×8×4.

The headline result, with explicit volume range: small to medium lattices.

> The breakeven volume — beyond which the CPU baseline is faster even with all optimisations — is now at V=16×8×8×8 (CPU parity, 1.0×); the fused-kernel SRAM limit is the primary obstacle to recovering the 2×–3× advantage that would otherwise be expected at these volumes, and bridging it (via kernel tiling or NeuronCore SRAM bucketing) is the primary motivation for the work outlined in Sec. 5.

Honest framing of the limitation. **Kernel tiling** = chopping the big
operation into chunks each of which fits in SRAM, processing them
sequentially. **SRAM bucketing** = a Neuron-compiler feature for handling
multiple shape variants with one binary
([neuronx-cc bucketing](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/compiler/neuronx-cc/api-reference-guide/neuron-compiler-cli-reference-guide.html)).

> We have additionally verified, on CPU and on a 2-core Inf2 in DataParallel mode, that the BiCGStab and CG host-loop solvers converge to the expected residuals on a Gaussian random $\psi$ with anti-periodic temporal boundaries, and that the gauge-baked, fused, multi-RHS .NEFF is invariant under permutation of right-hand sides to within FP32 round-off.

Two correctness checks. **Gaussian random $\psi$** = each component drawn
from a normal distribution; standard test input.
**Anti-periodic temporal boundaries**: physically required for fermion
fields on a finite-time lattice — quark fields pick up a minus sign when
they wrap around, encoding finite-temperature quantum statistics
([Gattringer & Lang 2010](https://link.springer.com/book/10.1007/978-3-642-01850-3),
ch. 4). **Permutation invariance** of the multi-RHS output: shuffling the
order of the input RHSs only shuffles the order of the outputs by the
same permutation — a basic linearity sanity check.

---

## 5. Limitations and outlook

### Functionality gap with QUDA.

> The current release covers only Wilson and Clover-Wilson fermions, CG and BiCGStab, periodic / anti-periodic boundaries, and a small set of gauge observables (plaquette, Wilson action, Polyakov loop, clover-definition topological charge).

Inventory of what's implemented. **Plaquette**: the smallest closed loop
of gauge links on the lattice, the simplest gauge-invariant observable
([Wilson 1974](https://doi.org/10.1103/PhysRevD.10.2445)). **Wilson
action**: the lattice version of the QCD action, the integrand of the
path integral. **Polyakov loop**: a closed loop in the time direction,
an order parameter for the deconfinement phase transition
([Polyakov 1978](https://doi.org/10.1016/0370-2693(78)90737-2)).
**Topological charge** in the clover definition: an integer-valued
gauge-invariant quantity related to the QCD vacuum structure
([Lüscher 1982](https://doi.org/10.1007/BF01215276)).

> Staggered, twisted-mass and domain-wall formulations are representable in the same `torch.nn.Module` style but not yet implemented.

Three other major fermion discretisations: **Staggered** (also called
Kogut–Susskind, [Kogut & Susskind 1975](https://doi.org/10.1103/PhysRevD.11.395))
distributes spin components across neighbouring sites; **twisted-mass**
([Frezzotti, Grassi, Sint & Weisz 2001](https://arxiv.org/abs/hep-lat/0101001))
adds an imaginary mass term to improve discretisation errors;
**domain-wall** ([Kaplan 1992](https://arxiv.org/abs/hep-lat/9206013))
introduces a fictitious fifth dimension to obtain near-exact chiral
symmetry. All three are research mainstays.

> Even-odd preconditioning and a multigrid preconditioner are the most impactful missing components; both are pure-tensor algorithms and present no fundamental obstacle to the `neuronx-cc` pipeline.

**Even-odd preconditioning** (a.k.a. red-black) splits the lattice like a
checkerboard and exploits the fact that the Wilson operator only connects
even and odd sites; halves the work. **Multigrid** uses a hierarchy of
coarser grids to attack low-frequency error components that are slow for
plain CG ([Brower et al. 2018](https://arxiv.org/abs/1801.07823); [Babich
et al. 2010](https://arxiv.org/abs/1005.3043)). Both can give order-of-magnitude
speedups in production.

### Multi-instance scaling.

> Within a single Inf2/Trn1 instance, communication uses NeuronLink on-chip and `torch_neuronx.DataParallel` as shown above.

**NeuronLink** is the chip-to-chip interconnect inside one EC2 instance,
analogous to NVIDIA's NVLink ([AWS Trainium architecture](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/arch/index.html)).

> For multi-instance domain decomposition with halo exchange, AWS exposes EFA via `torch.distributed` with the NCCL-style XLA collective backend; integrating this with a Schwarz-style domain decomposition is the natural next step and is what is required to address physically relevant volumes ($V \gtrsim 32^4$) that exceed single-chip HBM.

The path to bigger lattices. **EFA** = [Elastic Fabric Adapter](https://aws.amazon.com/hpc/efa/),
AWS's high-performance network. **`torch.distributed`** is PyTorch's
multi-machine API ([docs](https://pytorch.org/docs/stable/distributed.html)).
**NCCL** is NVIDIA's collective-communication library ([docs](https://docs.nvidia.com/deeplearning/nccl/));
"NCCL-style" means the same algorithmic interface (all-reduce, all-gather, etc.)
implemented via XLA. **Domain decomposition** splits the lattice into
sub-blocks, one per node; **halo exchange** sends the boundary slices
between neighbouring nodes each step.
**Schwarz** alternating method ([Schwarz 1869](https://en.wikipedia.org/wiki/Schwarz_alternating_method);
in Lattice QCD: [Lüscher 2003](https://arxiv.org/abs/hep-lat/0304007))
is a classical iterative solver well-suited to domain decomposition.

### Static shapes.

> The most awkward limitation is the static-shape requirement: each (class, shape, dtype, B) tuple compiles to a distinct .NEFF.

Four properties define a `.neff`: which operator (class), which lattice
size (shape), which precision (dtype: BF16, FP32, …), and how many RHS
(B). Change any one and you need a new compilation, which can take
minutes.

> For a fixed production volume this is amortised once, but for shape-sweeping workloads (auto-tuning, mixed-volume gauge ensembles) it is a real cost.

Compilation cost matters when you're varying parameters; less so when
you fix them and run for hours.

> `neuronx-cc` bucketing and shape-polymorphic XLA are partial mitigations.

Compiler features that allow one binary to handle several shapes by
zero-padding to the largest case ([XLA shape polymorphism](https://openxla.org/xla)).

### Precision.

> We currently run BF16 for the matvec with FP32 accumulation in the host-side CG/BiCGStab loop, which suffices to match the existing unit-test tolerances.

**Mixed precision**: do the bulk of the arithmetic in 16-bit but keep
the running sums in 32-bit ([Micikevicius et al. 2018](https://arxiv.org/abs/1710.03740)).
Standard practice in modern deep learning, here adapted to Lattice QCD.

> Mixed-precision Krylov methods with iterative refinement would map cleanly onto the NeuronCore MXU's BF16 inputs / FP32 accumulators and are expected to recover full double-precision residuals at near-BF16 throughput.

**Iterative refinement** ([Wilkinson 1948](https://en.wikipedia.org/wiki/Iterative_refinement);
in Lattice QCD: [Clark et al. 2010](https://arxiv.org/abs/0911.3191))
solves the system in low precision, then computes the residual in high
precision and corrects, repeating until satisfied. Recovers near-FP64
accuracy at near-FP16/BF16 cost. The MXU's "BF16 inputs / FP32
accumulators" architecture is exactly what this technique was designed
for, so the fit is natural.

---

## 6. Conclusion

> We have shown that the AoT-compiled, static-graph, matrix-engine programming model exposed by AWS Trainium and Inferentia is expressive enough to host the core kernels of Wilson and Clover-Wilson Lattice QCD, and that with four straightforward graph-level optimisations — gauge baking, fused 12×12 spin–colour kernels, multi-RHS batching, and multi-core DataParallel sharding — a pure-Python library with no custom kernel code reaches up to 10.9× over a tuned CPU baseline on V=4×4×4×4 in BF16.

A single-sentence summary mirroring the abstract. The "10.9×" figure
here differs slightly from the abstract's "13.0×" — likely reflecting a
different configuration measured at a different time; the figures in
the table support the higher number. The reader is meant to take away:
*it works, it is fast on small problems, and it required no chip-level
programming*.

---

## Where to read further

For the physics:

- C. Gattringer & C. B. Lang, *Quantum Chromodynamics on the Lattice* (Springer, 2010).
- T. DeGrand & C. DeTar, *Lattice Methods for Quantum Chromodynamics* (World Scientific, 2006).
- A. S. Kronfeld, *Twenty-first Century Lattice Gauge Theory*, Annu. Rev. Nucl. Part. Sci. **62** (2012) 265, [arXiv:1203.1204](https://arxiv.org/abs/1203.1204).

For the high-performance computing:

- B. Joó *et al.*, *Lattice QCD on Intel Xeon Phi*, [arXiv:1302.4839](https://arxiv.org/abs/1302.4839).
- M. A. Clark, *The Status of GPU Computing for Lattice QCD*, PoS LATTICE2009 003, [arXiv:0912.2268](https://arxiv.org/abs/0912.2268).
- The QUDA library: <https://github.com/lattice/quda>.
- Grid library (CPU/GPU portable): <https://github.com/paboyle/Grid>.

For AWS Neuron:

- AWS Neuron SDK documentation: <https://awsdocs-neuron.readthedocs-hosted.com/>.
- `torch-neuronx` GitHub: <https://github.com/aws-neuron/aws-neuron-sdk>.
- XLA project: <https://openxla.org/xla>.

For PyTorch and friends:

- PyTorch documentation: <https://pytorch.org/docs/>.
- `torch.compile` and the PT2 stack: <https://pytorch.org/docs/stable/torch.compiler.html>.
