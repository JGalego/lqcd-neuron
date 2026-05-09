"""
AWS Neuron ahead-of-time compilation utilities.

This module wraps ``torch_neuronx.trace`` (Trn1 / Inf2) to compile the hot
kernel inside each Lattice QCD operation — primarily the Dslash operator —
into a NeuronCore graph.

Why AoT compilation matters for Lattice QCD
--------------------------------------------
In QUDA, kernels are JIT-compiled by the CUDA runtime at first invocation
and then cached.  The Neuron SDK uses a *fully* ahead-of-time model:

1. You call ``torch_neuronx.trace(model, example_inputs)`` **once**.
2. The ``neuronx-cc`` compiler lowers the XLA HLO graph to a ``.neff``
   (Neuron Executable File Format) binary.
3. Subsequent calls to the returned ``ScriptModule`` execute the binary
   directly on the NeuronCores — no JIT overhead.

Constraints (and how we handle them)
-------------------------------------
• **Static shapes**: The compiled graph is tied to the shapes of
  ``example_inputs``.  We perform one compilation per unique lattice size.
  A simple cache keyed by ``(shape, dtype)`` avoids redundant compilations.

• **No Python control flow in the graph**: Solver loops (CG, BiCGStab)
  live on the host.  Only the ``forward()`` of each ``nn.Module`` is
  traced.  This is the same pattern as PyTorch training loops.

• **bfloat16 by default on Trn1**: The NeuronCores-v2 execute bfloat16
  matrix operations at peak throughput.  We offer a helper to cast float32
  models and inputs to bfloat16 before tracing.

Usage::

    from lqcd_neuron.dirac import WilsonDirac
    from lqcd_neuron.neuron import NeuronCompiler

    D = WilsonDirac(mass=0.1)
    compiler = NeuronCompiler()
    D_neuron = compiler.compile_dslash(D, lattice_shape=(8,4,4,4), nc=3)

    # D_neuron is a compiled ScriptModule; call it exactly like D.forward():
    out = D_neuron(psi, U)
"""

from __future__ import annotations

import os
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from .device import NeuronDevice, get_device, NeuronHardware

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# NeuronCore-v2 on-chip SRAM budget for the fused hopping kernels
# ---------------------------------------------------------------------------
# NeuronCore-v2 has 24 MiB SBUF + 2 MiB PSUM per core.  The four fused
# K_fwd/K_bwd buffers must all reside on-chip simultaneously for the kernel
# matvec to run at peak throughput.  Once they spill to HBM the operator
# becomes bandwidth-bound, which explains the measured speedup cliff:
#
#   V =  2 048 (8×8×8×4):  buffers ~9.4 MiB  → fits, 6.4× speedup
#   V =  8 192 (16×8×8×8): buffers ~37.7 MiB → spills, 2.2× speedup
#
# We budget 60% of SBUF for the kernels, leaving headroom for the spinor
# input/output and PSUM accumulate buffers.  compile_dslash auto-falls back
# to the unfused baked-gauge path when the fused buffers exceed this budget.
# Override with NeuronCompiler(sram_threshold_bytes=N).

_NC2_SRAM_BYTES = 24 * 1024 * 1024   # 24 MiB NeuronCore-v2 SBUF
_FUSED_SRAM_BUDGET = 0.60             # fraction of SBUF reserved for kernels


def _try_pin_to_neuron_core(
    compiled: nn.Module, core_id: int, total_cores: int,
) -> bool:
    """Best-effort pinning of *compiled* to a specific NeuronCore.

    Uses ``torch_neuronx.experimental.placement.set_neuron_cores``
    (Neuron SDK 2.x), falling back to ``torch_neuronx.set_neuron_cores``
    if the experimental path is unavailable.  Returns ``True`` when the
    NEFF was successfully pinned, ``False`` otherwise (older SDK, CPU-
    only environment, or any runtime error).

    The sharded Dslash wrapper uses this to spread its per-slab NEFFs
    across distinct NeuronCores so that the host-side ``ThreadPoolExecutor``
    fan-out actually overlaps compute on real hardware rather than
    serialising on core 0.  The total_cores hint lets the helper wrap
    around when there are more shards than visible cores (a benign
    serialisation onto a subset of cores).
    """
    try:
        from torch_neuronx.experimental import placement as _pl
        set_cores = _pl.set_neuron_cores
    except Exception:  # noqa: BLE001 — broad on purpose; multiple SDK paths
        try:
            import torch_neuronx as _tn
            set_cores = getattr(_tn, "set_neuron_cores", None)
            if set_cores is None:
                return False
        except Exception:
            return False
    try:
        target = core_id % max(1, total_cores)
        set_cores(compiled, start_nc=target, nc_count=1)
        return True
    except Exception as exc:  # noqa: BLE001
        logger.debug(
            "set_neuron_cores(start_nc=%d) failed for shard %d/%d: %s",
            core_id % max(1, total_cores), core_id, total_cores, exc,
        )
        return False


# ---------------------------------------------------------------------------
# Compile-path provenance
# ---------------------------------------------------------------------------
# Each ``compile_dslash*`` entry point stamps a small dict onto the module it
# returns describing exactly which code path was selected and why.  This lets
# benchmarks and integration tests record (and assert on) whether the fused
# kernel was used, whether the SRAM-spill auto-fallback fired, whether the
# T-axis sharded path was auto-routed, and the per-call batching / core
# layout.  See ``examples/bench_dslash.py`` for the standard consumer.
#
# Stable keys (forward-compatible additions allowed; consumers should
# tolerate missing keys):
#
#   kernel            : "fused" | "unfused" | "sharded" | "complex_in_graph"
#                       | "half_lattice_eo" | "cpu"
#   fused_kernel_mib  : float | None — size of the four K_fwd/K_bwd buffers
#   sram_budget_mib   : float | None — SRAM threshold used for the check
#   fused_fallback    : bool         — fused→unfused auto-downgrade fired
#   sharded_fallback  : bool         — unfused→sharded auto-route fired
#   num_shards        : int          — 1 unless sharded
#   T_local           : int          — full T unless sharded
#   batch_size        : int          — RHS per call (1 single-RHS)
#   num_cores         : int          — 1 unless multicore
#   lattice_shape     : tuple[int,int,int,int]

def _attach_compile_info(module: nn.Module, **info: Any) -> nn.Module:
    """Stamp ``lqcd_compile_info`` on *module*; safe if attribute is rejected."""
    try:
        existing = getattr(module, "lqcd_compile_info", None) or {}
        merged = {**existing, **info}
        # ``object.__setattr__`` bypasses nn.Module's parameter/buffer registry,
        # so the dict is stored as a plain Python attribute even on subclasses
        # with custom __setattr__.  ScriptModules reject new attributes — those
        # are wrapped in our own nn.Module shims, so this works in practice.
        object.__setattr__(module, "lqcd_compile_info", merged)
    except (AttributeError, RuntimeError):
        pass
    return module


def _fused_kernel_bytes(
    lattice_shape: Tuple[int, int, int, int],
    ns: int = 4,
    nc: int = 3,
    dtype: torch.dtype = torch.bfloat16,
) -> int:
    """Return bytes occupied by the four K_fwd/K_bwd fused-kernel buffers."""
    V = 1
    for d in lattice_shape:
        V *= d
    elem_bytes = 2 if dtype == torch.bfloat16 else 4
    # 4 tensors (re/im × fwd/bwd) × 4 directions × V sites × (Ns×Nc)² elements
    return 4 * 4 * V * (ns * nc) ** 2 * elem_bytes


class _NeuronPlaquetteAdapter(nn.Module):
    """Pure float32 plaquette kernel for Neuron.

    Accepts the gauge field as two real tensors (real and imaginary parts)
    and computes the average plaquette using
    :func:`~lqcd_neuron.observables.plaquette.plaquette_tensor_real`.
    This avoids the ``complex64`` dtype that ``neuronx-cc`` does not support.
    """

    def __init__(self, nc: int) -> None:
        super().__init__()
        self.nc = nc

    def forward(self, U_re: torch.Tensor, U_im: torch.Tensor) -> torch.Tensor:
        from ..observables.plaquette import plaquette_tensor_real

        return plaquette_tensor_real(U_re, U_im).mean() / self.nc


class _ComplexInputWrapper(nn.Module):
    """Host-side shim that splits a complex gauge tensor into real/imag parts.

    Wraps a compiled Neuron module that expects ``(U_re, U_im)`` so that
    callers can still pass a standard ``complex64`` gauge tensor.
    """

    def __init__(self, real_module: nn.Module, compute_dtype: torch.dtype = torch.float32) -> None:
        super().__init__()
        self._real_module = real_module
        self._compute_dtype = compute_dtype

    def forward(self, U: torch.Tensor) -> torch.Tensor:
        dt = self._compute_dtype
        return self._real_module(
            U.real.to(dt).contiguous(),
            U.imag.to(dt).contiguous(),
        )


class _ComplexDslashWrapper(nn.Module):
    """Host-side shim for the Dslash/Dirac real-arithmetic adapters.

    Splits ``complex64`` spinor and gauge tensors into real/imag halves,
    casts to the compute dtype used by the compiled kernel, and reassembles
    the result as a ``complex64`` spinor — preserving the standard
    ``forward(psi, U)`` interface expected by examples and solvers.

    The gauge field *U* is typically constant across solver iterations,
    so the split/cast result is cached and reused when the same tensor
    is passed again.  This avoids redundant host-side dtype conversions
    that otherwise scale with the lattice volume on every call.
    """

    def __init__(self, real_module: nn.Module, compute_dtype: torch.dtype = torch.float32) -> None:
        super().__init__()
        self._real_module = real_module
        self._compute_dtype = compute_dtype
        # Cached gauge-field conversion (avoids re-split + re-cast every call)
        self._cached_U_ptr: Optional[int] = None
        self._cached_U_re: Optional[torch.Tensor] = None
        self._cached_U_im: Optional[torch.Tensor] = None

    @torch.inference_mode()
    def forward(self, psi: torch.Tensor, U: torch.Tensor) -> torch.Tensor:
        dt = self._compute_dtype

        # Cache the gauge-field split+cast — U rarely changes between calls
        u_ptr = U.data_ptr()
        if u_ptr != self._cached_U_ptr:
            self._cached_U_re = U.real.to(dt).contiguous()
            self._cached_U_im = U.imag.to(dt).contiguous()
            self._cached_U_ptr = u_ptr

        r_re, r_im = self._real_module(
            psi.real.to(dt).contiguous(), psi.imag.to(dt).contiguous(),
            self._cached_U_re,            self._cached_U_im,
        )
        return torch.complex(r_re.float(), r_im.float())


class _BakedGaugeAdapter(nn.Module):
    """Wraps a Dslash adapter with the gauge field stored as on-device buffers.

    When compiled with ``torch_neuronx.trace``, model buffers live on the
    NeuronCore, so only the spinor field is transferred over PCIe per call.
    """

    def __init__(
        self,
        adapter: nn.Module,
        U_re: torch.Tensor,
        U_im: torch.Tensor,
    ) -> None:
        super().__init__()
        self.adapter = adapter
        self.register_buffer("U_re", U_re)
        self.register_buffer("U_im", U_im)

    def forward(
        self, psi_re: torch.Tensor, psi_im: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.adapter(psi_re, psi_im, self.U_re, self.U_im)


class _BakedGaugeDslashWrapper(nn.Module):
    """Host-side shim for Dslash with gauge field baked into the compiled model.

    Accepts the standard ``forward(psi, U)`` signature for API compatibility
    but ignores *U* — the gauge field is already on the NeuronCore.
    Only the spinor is split and transferred per call.
    """

    def __init__(self, real_module: nn.Module, compute_dtype: torch.dtype = torch.float32) -> None:
        super().__init__()
        self._real_module = real_module
        self._compute_dtype = compute_dtype
        # Pre-allocated spinor scratch buffers; lazily initialised on first
        # call so the wrapper can be constructed before any input is seen.
        # Avoids a per-call malloc + dtype cast that scales with lattice volume.
        self._buf_re: Optional[torch.Tensor] = None
        self._buf_im: Optional[torch.Tensor] = None

    @torch.inference_mode()
    def forward(self, psi: torch.Tensor, U: torch.Tensor = None) -> torch.Tensor:
        dt = self._compute_dtype
        real_shape = psi.real.shape
        if self._buf_re is None or self._buf_re.shape != real_shape:
            self._buf_re = torch.empty(real_shape, dtype=dt)
            self._buf_im = torch.empty(real_shape, dtype=dt)
        self._buf_re.copy_(psi.real)
        self._buf_im.copy_(psi.imag)
        r_re, r_im = self._real_module(self._buf_re, self._buf_im)
        return torch.complex(r_re.float(), r_im.float())


class _BakedGaugeBatchedAdapter(nn.Module):
    """Multi-RHS Dslash adapter with the gauge field baked as a buffer.

    Adds a leading singleton dim to *U_re/U_im* so a batched psi of shape
    ``(B, T, Z, Y, X, Ns, Nc)`` broadcasts cleanly through the underlying
    adapter's einsums.  The lattice rolls in the adapter use negative dim
    indices so the leading batch dim does not shift them.
    """

    def __init__(
        self,
        adapter: nn.Module,
        U_re: torch.Tensor,
        U_im: torch.Tensor,
    ) -> None:
        super().__init__()
        self.adapter = adapter
        # Singleton batch dim broadcasts across all right-hand sides.
        self.register_buffer("U_re", U_re.unsqueeze(0))
        self.register_buffer("U_im", U_im.unsqueeze(0))

    def forward(
        self, psi_re: torch.Tensor, psi_im: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.adapter(psi_re, psi_im, self.U_re, self.U_im)


class _BakedGaugeBatchedDslashWrapper(nn.Module):
    """Host-side shim for multi-RHS Dslash with the gauge field baked.

    Accepts a batched complex64 spinor ``psi`` of shape
    ``(B, T, Z, Y, X, Ns, Nc)``.  Returns a complex64 tensor of the same
    shape.  The gauge field is already on the NeuronCore.
    """

    def __init__(self, real_module: nn.Module, compute_dtype: torch.dtype = torch.float32) -> None:
        super().__init__()
        self._real_module = real_module
        self._compute_dtype = compute_dtype

    @torch.inference_mode()
    def forward(self, psi: torch.Tensor) -> torch.Tensor:
        dt = self._compute_dtype
        r_re, r_im = self._real_module(
            psi.real.to(dt).contiguous(),
            psi.imag.to(dt).contiguous(),
        )
        return torch.complex(r_re.float(), r_im.float())


# ---------------------------------------------------------------------------
# Fused spin-color hopping kernels
# ---------------------------------------------------------------------------
#
# When the gauge field is fixed, the per-site, per-direction operator
#
#     K_fwd[μ, x] = (I − γ_μ) ⊗ U(x, μ)            (4Nc × 4Nc)
#     K_bwd[μ, x] = (I + γ_μ) ⊗ U†(x − μ̂, μ)         (4Nc × 4Nc)
#
# can be precomputed once at compile time and baked into the model as a
# NeuronCore-resident buffer.  At runtime each Dslash call then performs
# only:
#
#   1. roll the (flattened, ns*nc-sized) spinor along each lattice axis
#   2. one (Ns*Nc) × (Ns*Nc) matrix-vector multiply per direction-side
#   3. sum the eight contributions
#
# This eliminates the four per-call backward-U rolls, fuses the spin
# projector and colour matvec into a single contraction, and presents a
# 12 × 12 matmul to the NeuronCore tensor engine — a much better fit than
# the original 4×4 spin and 3×3 colour einsums.
# ---------------------------------------------------------------------------


def _build_dslash_kernels(
    U: torch.Tensor,
    *,
    nc: int,
    ns: int,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pre-compute the fused spin-color hopping kernels for a fixed *U*.

    Returns ``(K_fwd_re, K_fwd_im, K_bwd_re, K_bwd_im)``, each of shape
    ``(4, T, Z, Y, X, ns*nc, ns*nc)`` and dtype *dtype*.
    """
    from ..dirac.gamma import degrand_rossi_gammas

    G  = degrand_rossi_gammas(dtype=torch.complex64)        # (4, 4, 4)
    I4 = torch.eye(4, dtype=torch.complex64)
    P_minus = torch.stack([I4 - G[mu] for mu in range(4)], dim=0)  # (4, 4, 4)
    P_plus  = torch.stack([I4 + G[mu] for mu in range(4)], dim=0)

    T, Z, Y, X = U.shape[:4]
    Uc = U.to(torch.complex64)

    K_fwd = torch.empty(4, T, Z, Y, X, ns * nc, ns * nc, dtype=torch.complex64)
    K_bwd = torch.empty_like(K_fwd)
    for mu in range(4):
        U_mu = Uc[..., mu, :, :]                            # (T,Z,Y,X, nc, nc)
        # U†(x − μ̂, μ) absorbs the per-call backward roll
        U_mu_bwd = torch.roll(U_mu, 1, dims=mu).conj().transpose(-1, -2)
        # Kronecker product P[s, s'] * U[c, c'] -> (T,Z,Y,X, ns, nc, ns, nc)
        # then flatten the (s, c) and (s', c') pairs into ns*nc.
        K_fwd[mu] = torch.einsum(
            "ab,...ij->...aibj", P_minus[mu], U_mu
        ).reshape(T, Z, Y, X, ns * nc, ns * nc)
        K_bwd[mu] = torch.einsum(
            "ab,...ij->...aibj", P_plus[mu], U_mu_bwd
        ).reshape(T, Z, Y, X, ns * nc, ns * nc)

    return (
        K_fwd.real.to(dtype).contiguous(),
        K_fwd.imag.to(dtype).contiguous(),
        K_bwd.real.to(dtype).contiguous(),
        K_bwd.imag.to(dtype).contiguous(),
    )


class _FusedDslashAdapter(nn.Module):
    """Wilson Dslash / Dirac with pre-fused spin-color kernels.

    Single-RHS path.  ``K_*`` buffers have shape
    ``(4, T, Z, Y, X, ns*nc, ns*nc)`` and are NeuronCore-resident after
    ``torch_neuronx.trace``.

    Args:
        diag: Diagonal coefficient ``(4 + mass)`` for the Dirac operator,
              or ``0.0`` for the bare Dslash hopping term.
        ns, nc: Spin and colour counts (used only for the final reshape).
    """

    def __init__(
        self,
        K_fwd_re: torch.Tensor, K_fwd_im: torch.Tensor,
        K_bwd_re: torch.Tensor, K_bwd_im: torch.Tensor,
        diag: float, ns: int, nc: int,
    ) -> None:
        super().__init__()
        self.register_buffer("K_fwd_re", K_fwd_re)
        self.register_buffer("K_fwd_im", K_fwd_im)
        self.register_buffer("K_bwd_re", K_bwd_re)
        self.register_buffer("K_bwd_im", K_bwd_im)
        self.diag = float(diag)
        self.ns = ns
        self.nc = nc

    def forward(
        self, psi_re: torch.Tensor, psi_im: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Flatten spin×colour into a single ns*nc dim so the fused 12×12
        # matvec maps straight onto the tensor engine.
        pr = psi_re.flatten(-2)        # (..., T, Z, Y, X, ns*nc)
        pi = psi_im.flatten(-2)

        out_re = self.diag * pr
        out_im = self.diag * pi

        for mu in range(4):
            # Lattice axes T,Z,Y,X live at positions -5..-2 of the flat tensor.
            ldim = mu - 5

            Kfr, Kfi = self.K_fwd_re[mu], self.K_fwd_im[mu]
            Kbr, Kbi = self.K_bwd_re[mu], self.K_bwd_im[mu]

            pf_re = torch.roll(pr, -1, dims=ldim)
            pf_im = torch.roll(pi, -1, dims=ldim)
            pb_re = torch.roll(pr,  1, dims=ldim)
            pb_im = torch.roll(pi,  1, dims=ldim)

            # Complex matvec K @ psi (real arithmetic):
            cf_re = (torch.einsum("...ij,...j->...i", Kfr, pf_re)
                   - torch.einsum("...ij,...j->...i", Kfi, pf_im))
            cf_im = (torch.einsum("...ij,...j->...i", Kfr, pf_im)
                   + torch.einsum("...ij,...j->...i", Kfi, pf_re))
            cb_re = (torch.einsum("...ij,...j->...i", Kbr, pb_re)
                   - torch.einsum("...ij,...j->...i", Kbi, pb_im))
            cb_im = (torch.einsum("...ij,...j->...i", Kbr, pb_im)
                   + torch.einsum("...ij,...j->...i", Kbi, pb_re))

            out_re = out_re - 0.5 * (cf_re + cb_re)
            out_im = out_im - 0.5 * (cf_im + cb_im)

        out_re = out_re.unflatten(-1, (self.ns, self.nc))
        out_im = out_im.unflatten(-1, (self.ns, self.nc))
        return out_re, out_im


class _FusedBatchedDslashAdapter(_FusedDslashAdapter):
    """Multi-RHS variant: K buffers carry a singleton leading batch dim.

    With ``K_*`` shaped ``(4, 1, T, Z, Y, X, ns*nc, ns*nc)``, indexing
    ``self.K_fwd_re[mu]`` yields ``(1, T, Z, Y, X, ns*nc, ns*nc)`` which
    broadcasts against a batched flat psi of shape
    ``(B, T, Z, Y, X, ns*nc)`` in the einsum.
    """

    def __init__(
        self,
        K_fwd_re: torch.Tensor, K_fwd_im: torch.Tensor,
        K_bwd_re: torch.Tensor, K_bwd_im: torch.Tensor,
        diag: float, ns: int, nc: int,
    ) -> None:
        # Insert a singleton batch dim AFTER the leading mu axis.
        super().__init__(
            K_fwd_re.unsqueeze(1), K_fwd_im.unsqueeze(1),
            K_bwd_re.unsqueeze(1), K_bwd_im.unsqueeze(1),
            diag=diag, ns=ns, nc=nc,
        )


class _FusedDslashWrapper(nn.Module):
    """Host-side shim around a compiled fused single-RHS adapter.

    Preserves the ``forward(psi, U=None)`` signature of the previous
    baked-gauge wrapper so existing callers and solvers keep working.
    The *U* argument is accepted but ignored — the gauge information is
    already encoded in the compiled NEFF.
    """

    def __init__(self, real_module: nn.Module, compute_dtype: torch.dtype = torch.float32) -> None:
        super().__init__()
        self._real_module = real_module
        self._compute_dtype = compute_dtype
        self._buf_re: Optional[torch.Tensor] = None
        self._buf_im: Optional[torch.Tensor] = None

    @torch.inference_mode()
    def forward(self, psi: torch.Tensor, U: torch.Tensor = None) -> torch.Tensor:
        dt = self._compute_dtype
        real_shape = psi.real.shape
        if self._buf_re is None or self._buf_re.shape != real_shape:
            self._buf_re = torch.empty(real_shape, dtype=dt)
            self._buf_im = torch.empty(real_shape, dtype=dt)
        self._buf_re.copy_(psi.real)
        self._buf_im.copy_(psi.imag)
        r_re, r_im = self._real_module(self._buf_re, self._buf_im)
        return torch.complex(r_re.float(), r_im.float())


class _FusedBatchedDslashWrapper(nn.Module):
    """Host-side shim around a compiled fused multi-RHS adapter."""

    def __init__(self, real_module: nn.Module, compute_dtype: torch.dtype = torch.float32) -> None:
        super().__init__()
        self._real_module = real_module
        self._compute_dtype = compute_dtype
        self._buf_re: Optional[torch.Tensor] = None
        self._buf_im: Optional[torch.Tensor] = None

    @torch.inference_mode()
    def forward(self, psi: torch.Tensor) -> torch.Tensor:
        dt = self._compute_dtype
        real_shape = psi.real.shape
        if self._buf_re is None or self._buf_re.shape != real_shape:
            self._buf_re = torch.empty(real_shape, dtype=dt)
            self._buf_im = torch.empty(real_shape, dtype=dt)
        self._buf_re.copy_(psi.real)
        self._buf_im.copy_(psi.imag)
        r_re, r_im = self._real_module(self._buf_re, self._buf_im)
        return torch.complex(r_re.float(), r_im.float())


class _HostLoopBatchedWrapper(nn.Module):
    """Fallback batched shim that loops a single-RHS module over the batch dim.

    Used by ``compile_dslash_batched`` when the fused multi-RHS NEFF would
    overflow NeuronCore SRAM or the per-NEFF HLO instruction budget.  The
    wrapped *single_rhs_module* is the output of ``compile_dslash`` (with
    its own auto fused→unfused→sharded fallbacks), so this preserves the
    batched API at the cost of dispatch-overhead amortisation.
    """

    def __init__(self, single_rhs_module: nn.Module) -> None:
        super().__init__()
        self._m = single_rhs_module

    @torch.inference_mode()
    def forward(self, psi: torch.Tensor) -> torch.Tensor:
        # psi: (B, T, Z, Y, X, Ns, Nc) complex.  The inner module bakes U.
        outs = [self._m(psi[b]) for b in range(psi.shape[0])]
        return torch.stack(outs, dim=0)


class _MultiCoreDslashWrapper(nn.Module):
    """Host-side shim for multi-core data-parallel Dslash execution.

    Splits a batched complex64 spinor across *num_cores* NeuronCores,
    runs each slice through the DataParallel-wrapped compiled model,
    and reassembles the results.

    ``forward(psi)`` accepts ``(B, T, Z, Y, X, Ns, Nc)`` where
    ``B == num_cores * per_core_batch_size``.
    """

    def __init__(
        self,
        parallel_module: nn.Module,
        compute_dtype: torch.dtype = torch.float32,
        num_cores: int = 1,
        per_core_batch_size: int = 1,
    ) -> None:
        super().__init__()
        self._parallel_module = parallel_module
        self._compute_dtype = compute_dtype
        self.num_cores = num_cores
        self.per_core_batch_size = per_core_batch_size
        self.global_batch_size = num_cores * per_core_batch_size
        self._buf_re: Optional[torch.Tensor] = None
        self._buf_im: Optional[torch.Tensor] = None

    @torch.inference_mode()
    def forward(self, psi: torch.Tensor) -> torch.Tensor:
        if psi.shape[0] != self.global_batch_size:
            raise ValueError(
                f"_MultiCoreDslashWrapper: psi.shape[0]={psi.shape[0]} but "
                f"expected num_cores * per_core_batch_size = "
                f"{self.num_cores} * {self.per_core_batch_size} = "
                f"{self.global_batch_size}"
            )
        dt = self._compute_dtype
        real_shape = psi.real.shape
        if self._buf_re is None or self._buf_re.shape != real_shape:
            self._buf_re = torch.empty(real_shape, dtype=dt)
            self._buf_im = torch.empty(real_shape, dtype=dt)
        self._buf_re.copy_(psi.real)
        self._buf_im.copy_(psi.imag)
        # DataParallel splits dim 0 across cores automatically
        r_re, r_im = self._parallel_module(self._buf_re, self._buf_im)
        return torch.complex(r_re.float(), r_im.float())


# ---------------------------------------------------------------------------
# Even-odd (checkerboard) preconditioning — half-lattice fused kernels
# ---------------------------------------------------------------------------
#
# The lattice is split into two V/2-site sublattices by parity:
#   even (p=0): (t+z+y+x) % 2 == 0
#   odd  (p=1): (t+z+y+x) % 2 == 1
#
# Each nearest-neighbour hop connects even ↔ odd sites, so the Dslash matrix
# is block off-diagonal in the (even, odd) basis:
#
#   D_hop = [ 0     D_eo ]
#            [ D_oe  0   ]
#
# Compiling D_oe and D_eo separately instead of the full D_hop halves the
# fused-kernel buffer footprint (V/2 sites instead of V), deferring the
# SRAM-spill cliff by one lattice doubling:
#
#   16×8×8×8 full  (~37.7 MiB) → spills SRAM (2.2× speedup measured)
#   16×8×8×8 half  (~18.9 MiB) → fits  in SRAM (expected ~6× speedup)
#
# Storage format
# --------------
# Half-lattice spinors / gauge fields are packed as (T, Z, Y, X//2, ...).
# For parity p at row (t, z, y), the x-coordinate of half-lattice index ix is:
#   x = 2*ix + (t+z+y+p) % 2
#
# Utility functions
# -----------------
#   pack_checkerboard(psi_full, parity)  → (T, Z, Y, X//2, Ns, Nc)
#   unpack_checkerboard(psi_half, parity, T, Z, Y, X) → (T, Z, Y, X, Ns, Nc)
# ---------------------------------------------------------------------------


def pack_checkerboard(psi_full: torch.Tensor, parity: int) -> torch.Tensor:
    """Pack a full-lattice spinor into half-lattice (T, Z, Y, X//2, Ns, Nc).

    Args:
        psi_full: Complex spinor of shape ``(T, Z, Y, X, Ns, Nc)``.
        parity:   0 for even sites ``(t+z+y+x)%2==0``,  1 for odd.

    Returns:
        Half-lattice spinor of shape ``(T, Z, Y, X//2, Ns, Nc)``.
    """
    T, Z, Y, X = psi_full.shape[:4]
    assert X % 2 == 0, "X must be even for even-odd decomposition"
    t = torch.arange(T, device=psi_full.device).view(T, 1, 1, 1)
    z = torch.arange(Z, device=psi_full.device).view(1, Z, 1, 1)
    y = torch.arange(Y, device=psi_full.device).view(1, 1, Y, 1)
    ix = torch.arange(X // 2, device=psi_full.device).view(1, 1, 1, X // 2)
    x = (2 * ix + (t + z + y + parity) % 2).expand(T, Z, Y, X // 2)
    return psi_full[
        t.expand(T, Z, Y, X // 2),
        z.expand(T, Z, Y, X // 2),
        y.expand(T, Z, Y, X // 2),
        x,
    ]


def unpack_checkerboard(
    psi_half: torch.Tensor,
    parity: int,
    T: int,
    Z: int,
    Y: int,
    X: int,
) -> torch.Tensor:
    """Unpack a half-lattice spinor back to full-lattice shape.

    Args:
        psi_half: Half-lattice spinor ``(T, Z, Y, X//2, Ns, Nc)``.
        parity:   0 (even) or 1 (odd).
        T, Z, Y, X: Full-lattice extents.

    Returns:
        ``(T, Z, Y, X, Ns, Nc)`` with zeros on the complementary parity sites.
    """
    Ns, Nc = psi_half.shape[-2], psi_half.shape[-1]
    psi_full = torch.zeros(
        T, Z, Y, X, Ns, Nc, dtype=psi_half.dtype, device=psi_half.device
    )
    t = torch.arange(T, device=psi_half.device).view(T, 1, 1, 1)
    z = torch.arange(Z, device=psi_half.device).view(1, Z, 1, 1)
    y = torch.arange(Y, device=psi_half.device).view(1, 1, Y, 1)
    ix = torch.arange(X // 2, device=psi_half.device).view(1, 1, 1, X // 2)
    x = (2 * ix + (t + z + y + parity) % 2).expand(T, Z, Y, X // 2)
    psi_full[
        t.expand(T, Z, Y, X // 2),
        z.expand(T, Z, Y, X // 2),
        y.expand(T, Z, Y, X // 2),
        x,
    ] = psi_half
    return psi_full


def _build_dslash_kernels_halfvol(
    U: torch.Tensor,
    out_parity: int,
    *,
    nc: int,
    ns: int,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pre-compute fused spin-color hopping kernels for a half-lattice.

    Only kernels at *out_parity* output sites are built (shape
    ``(4, T, Z, Y, X//2, ns*nc, ns*nc)``), halving on-chip memory vs the
    full-lattice ``_build_dslash_kernels``.

    Args:
        U:          Full-lattice gauge field ``(T, Z, Y, X, 4, Nc, Nc)`` (complex64).
        out_parity: Parity of **output** sites (0 = even, 1 = odd).
        nc, ns:     Colour and spin counts.
        dtype:      Element dtype of the returned tensors (e.g. bfloat16).

    Returns:
        ``(K_fwd_re, K_fwd_im, K_bwd_re, K_bwd_im)`` each of shape
        ``(4, T, Z, Y, X//2, ns*nc, ns*nc)`` and dtype *dtype*.
    """
    from ..dirac.gamma import degrand_rossi_gammas

    T, Z, Y, X = U.shape[:4]
    assert X % 2 == 0
    Uc = U.to(torch.complex64)

    G  = degrand_rossi_gammas(dtype=torch.complex64)
    I4 = torch.eye(4, dtype=torch.complex64)
    P_minus = [I4 - G[mu] for mu in range(4)]
    P_plus  = [I4 + G[mu] for mu in range(4)]

    # Coordinates of out_parity sites in the half-lattice --------------------
    t_idx = torch.arange(T).view(T, 1, 1, 1)
    z_idx = torch.arange(Z).view(1, Z, 1, 1)
    y_idx = torch.arange(Y).view(1, 1, Y, 1)
    ix    = torch.arange(X // 2).view(1, 1, 1, X // 2)
    x_out = (2 * ix + (t_idx + z_idx + y_idx + out_parity) % 2).expand(T, Z, Y, X // 2)

    T_e = t_idx.expand(T, Z, Y, X // 2)
    Z_e = z_idx.expand(T, Z, Y, X // 2)
    Y_e = y_idx.expand(T, Z, Y, X // 2)

    K_fwd = torch.zeros(4, T, Z, Y, X // 2, ns * nc, ns * nc, dtype=torch.complex64)
    K_bwd = torch.zeros_like(K_fwd)

    for mu in range(4):
        # U at the output site itself (for K_fwd = P_minus ⊗ U)
        U_mu_out = Uc[T_e, Z_e, Y_e, x_out, mu, :, :]  # (T,Z,Y,X//2,Nc,Nc)

        # U at the backward neighbour x_out - μ̂ (for K_bwd = P_plus ⊗ U†)
        if mu == 0:
            U_mu_bwd = Uc[(T_e - 1) % T, Z_e, Y_e, x_out, mu, :, :]
        elif mu == 1:
            U_mu_bwd = Uc[T_e, (Z_e - 1) % Z, Y_e, x_out, mu, :, :]
        elif mu == 2:
            U_mu_bwd = Uc[T_e, Z_e, (Y_e - 1) % Y, x_out, mu, :, :]
        else:  # mu == 3 (X)
            U_mu_bwd = Uc[T_e, Z_e, Y_e, (x_out - 1) % X, mu, :, :]

        K_fwd[mu] = torch.einsum(
            "ab,...ij->...aibj", P_minus[mu], U_mu_out
        ).reshape(T, Z, Y, X // 2, ns * nc, ns * nc)
        K_bwd[mu] = torch.einsum(
            "ab,...ij->...aibj", P_plus[mu], U_mu_bwd.conj().transpose(-1, -2)
        ).reshape(T, Z, Y, X // 2, ns * nc, ns * nc)

    return (
        K_fwd.real.to(dtype).contiguous(),
        K_fwd.imag.to(dtype).contiguous(),
        K_bwd.real.to(dtype).contiguous(),
        K_bwd.imag.to(dtype).contiguous(),
    )


class _HalfLatticeHopAdapter(nn.Module):
    """Wilson hop on a half-lattice (T, Z, Y, X//2, Ns, Nc) spinors.

    Applies D_{out_parity ← in_parity} using V/2-site fused kernels.
    T/Z/Y hops use standard ``torch.roll``; the X hop uses a staggered
    roll that is selected per (t, z, y) row via pre-baked boolean masks,
    keeping the forward pass Python-control-flow-free for ``torch_neuronx``
    tracing.

    Args:
        K_fwd_re/im: Forward kernels ``(4, T, Z, Y, X//2, ns*nc, ns*nc)``.
        K_bwd_re/im: Backward kernels (same shape).
        diag:        Diagonal coefficient (4+mass for Dirac, 0 for Dslash).
        ns, nc:      Spin / colour counts.
        out_parity:  Parity of output sites (0 or 1); determines X roll logic.
    """

    def __init__(
        self,
        K_fwd_re: torch.Tensor, K_fwd_im: torch.Tensor,
        K_bwd_re: torch.Tensor, K_bwd_im: torch.Tensor,
        diag: float, ns: int, nc: int,
        out_parity: int,
        lattice_shape: Tuple[int, int, int, int],
    ) -> None:
        super().__init__()
        self.register_buffer("K_fwd_re", K_fwd_re)
        self.register_buffer("K_fwd_im", K_fwd_im)
        self.register_buffer("K_bwd_re", K_bwd_re)
        self.register_buffer("K_bwd_im", K_bwd_im)
        self.diag = float(diag)
        self.ns = ns
        self.nc = nc

        # Pre-bake boolean masks for the staggered X-direction roll.
        # r_mask[t,z,y] = (t+z+y+out_parity) % 2
        # X fwd roll: apply roll(-1) when r_mask == 1  (i.e. in_parity=0 rows)
        # X bwd roll: apply roll(+1) when r_mask == 0
        T, Z, Y, X = lattice_shape
        t_idx = torch.arange(T).view(T, 1, 1, 1, 1)
        z_idx = torch.arange(Z).view(1, Z, 1, 1, 1)
        y_idx = torch.arange(Y).view(1, 1, Y, 1, 1)
        r = (t_idx + z_idx + y_idx + out_parity) % 2  # (T,Z,Y,1,1)
        # Expand over X//2 and Ns*Nc dims so torch.where broadcasts cleanly.
        r_expanded = r.expand(T, Z, Y, X // 2, ns * nc)
        self.register_buffer("_r_mask_fwd", (r_expanded == 1))   # bool
        self.register_buffer("_r_mask_bwd", (r_expanded == 0))   # bool

    def forward(
        self,
        psi_re: torch.Tensor,
        psi_im: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Flatten spin×colour (last two dims) into Ns*Nc.
        pr = psi_re.flatten(-2)   # (T, Z, Y, X//2, Ns*Nc)
        pi = psi_im.flatten(-2)

        out_re = self.diag * pr
        out_im = self.diag * pi

        for mu in range(4):
            Kfr, Kfi = self.K_fwd_re[mu], self.K_fwd_im[mu]
            Kbr, Kbi = self.K_bwd_re[mu], self.K_bwd_im[mu]

            # Lattice axes T,Z,Y,X//2 sit at dims -5..-2 of the flat tensor.
            ldim = mu - 5

            if mu < 3:
                # T / Z / Y: nearest neighbours are always on the complementary
                # parity and at the same ix coordinate → plain roll.
                pf_re = torch.roll(pr, -1, dims=ldim)
                pf_im = torch.roll(pi, -1, dims=ldim)
                pb_re = torch.roll(pr,  1, dims=ldim)
                pb_im = torch.roll(pi,  1, dims=ldim)
            else:
                # X (mu=3): roll amount depends on row parity (t+z+y+p)%2.
                # fwd: roll -1 when r==1, else no roll.
                # bwd: roll +1 when r==0, else no roll.
                pf_re = torch.where(self._r_mask_fwd, torch.roll(pr, -1, dims=ldim), pr)
                pf_im = torch.where(self._r_mask_fwd, torch.roll(pi, -1, dims=ldim), pi)
                pb_re = torch.where(self._r_mask_bwd, torch.roll(pr,  1, dims=ldim), pr)
                pb_im = torch.where(self._r_mask_bwd, torch.roll(pi,  1, dims=ldim), pi)

            cf_re = (torch.einsum("...ij,...j->...i", Kfr, pf_re)
                   - torch.einsum("...ij,...j->...i", Kfi, pf_im))
            cf_im = (torch.einsum("...ij,...j->...i", Kfr, pf_im)
                   + torch.einsum("...ij,...j->...i", Kfi, pf_re))
            cb_re = (torch.einsum("...ij,...j->...i", Kbr, pb_re)
                   - torch.einsum("...ij,...j->...i", Kbi, pb_im))
            cb_im = (torch.einsum("...ij,...j->...i", Kbr, pb_im)
                   + torch.einsum("...ij,...j->...i", Kbi, pb_re))

            out_re = out_re - 0.5 * (cf_re + cb_re)
            out_im = out_im - 0.5 * (cf_im + cb_im)

        return out_re.unflatten(-1, (self.ns, self.nc)), out_im.unflatten(-1, (self.ns, self.nc))


class _HalfLatticeDslashWrapper(nn.Module):
    """Host-side shim around a compiled half-lattice hop adapter.

    Accepts and returns **half-lattice** complex64 spinors of shape
    ``(T, Z, Y, X//2, Ns, Nc)``.  Use :func:`pack_checkerboard` /
    :func:`unpack_checkerboard` to convert to/from full-lattice tensors.
    """

    def __init__(self, real_module: nn.Module, compute_dtype: torch.dtype = torch.float32) -> None:
        super().__init__()
        self._real_module = real_module
        self._compute_dtype = compute_dtype
        self._buf_re: Optional[torch.Tensor] = None
        self._buf_im: Optional[torch.Tensor] = None

    @torch.inference_mode()
    def forward(self, psi_half: torch.Tensor) -> torch.Tensor:
        dt = self._compute_dtype
        real_shape = psi_half.real.shape
        if self._buf_re is None or self._buf_re.shape != real_shape:
            self._buf_re = torch.empty(real_shape, dtype=dt)
            self._buf_im = torch.empty(real_shape, dtype=dt)
        self._buf_re.copy_(psi_half.real)
        self._buf_im.copy_(psi_half.imag)
        r_re, r_im = self._real_module(self._buf_re, self._buf_im)
        return torch.complex(r_re.float(), r_im.float())


# ---------------------------------------------------------------------------
# Spatial sharding (T-axis domain decomposition)
# ---------------------------------------------------------------------------
#
# Large lattices (V >~ 24^4) overflow the per-NeuronCore HLO instruction
# budget of neuronx-cc, even with the unfused baked-gauge path:
#
#   [NCC_EVRF007] Instructions generated 33,776,000 exceeds typical limit
#   of 5,000,000.
#
# We bring the per-graph instruction count back into range by sharding the
# lattice along the T (slowest) axis into ``num_shards`` slabs of size
# ``T_local = T // num_shards``.  Each shard compiles to its own NEFF
# operating on a ``(T_local, Z, Y, X)`` sub-volume — the same per-graph cost
# as a ``(T_local * Z * Y * X)``-volume single-shard compile.
#
# Standard ``torch.roll`` handles the Z, Y, X axes locally (each shard owns
# all sites along those axes).  For the sharded T axis the boundary
# neighbours come from adjacent shards via host-side halo gather:
#
#   psi(local t = T_local-1).fwd  ←  psi(global t = (s+1)*T_local % T)
#   psi(local t = 0).bwd          ←  psi(global t = (s*T_local - 1) % T)
#
# The host wrapper assembles a one-slab halo on each side, the compiled
# shard adapter splices halos in via ``torch.cat`` (no roll on T), and the
# result is reassembled by concatenating shard outputs along T.
#
# Periodic BCs across the global T axis are preserved automatically because
# the host computes halo indices modulo T.
#
# Limitations of this initial implementation
# ------------------------------------------
# • Sequential per-shard dispatch on a single NeuronCore.  A multi-core
#   parallel variant (one shard per core) is a follow-up — it would require
#   per-shard NEFFs (different baked U) loaded into different cores.
# • Single-RHS only.  Multi-RHS sharded execution is also a follow-up.
# • Sharding along T only (no Z/Y/X cuts).  T is the natural slow axis and
#   keeps the spatial axes fully local, matching the QUDA convention.
# ---------------------------------------------------------------------------


def _shard_T_indices(T: int, num_shards: int) -> Tuple[int, int]:
    """Return (T_local, num_shards) after validating divisibility."""
    if T % num_shards != 0:
        raise ValueError(
            f"Spatial sharding requires T={T} divisible by num_shards="
            f"{num_shards}.  Try a power-of-2 num_shards that divides T."
        )
    return T // num_shards, num_shards


class _ShardedBakedGaugeAdapter(nn.Module):
    """Halo-aware Wilson hop on a T-sharded sub-volume.

    Bakes the local gauge slab and one extra slab of ``U(t-1, μ=0)`` (the
    backward-T link at the shard's left boundary) as NeuronCore-resident
    buffers.  Inputs are the local spinor plus two one-slab halos:

        psi_re/im     : (T_local, Z, Y, X, Ns, Nc)
        halo_l_re/im  : (1,       Z, Y, X, Ns, Nc)   — psi(global t0-1)
        halo_r_re/im  : (1,       Z, Y, X, Ns, Nc)   — psi(global t1)

    For Z/Y/X axes ``torch.roll`` works locally.  For the T axis, the
    forward/backward neighbours are spliced in via ``torch.cat`` so no
    wrap-around occurs at the shard boundary.
    """

    def __init__(
        self,
        U_local_re: torch.Tensor, U_local_im: torch.Tensor,
        U_tm1_mu0_re: torch.Tensor, U_tm1_mu0_im: torch.Tensor,
        diag: float,
        nc: int,
    ) -> None:
        super().__init__()
        self.register_buffer("U_local_re", U_local_re)
        self.register_buffer("U_local_im", U_local_im)
        self.register_buffer("U_tm1_mu0_re", U_tm1_mu0_re)
        self.register_buffer("U_tm1_mu0_im", U_tm1_mu0_im)
        self.diag = float(diag)
        self.nc = nc

        # Spin projectors — same construction as _NeuronWilsonDslashAdapter.
        from ..dirac.gamma import degrand_rossi_gammas
        G  = degrand_rossi_gammas(dtype=torch.complex64)
        I4 = torch.eye(4, dtype=torch.complex64)
        P_minus = torch.stack([I4 - G[mu] for mu in range(4)], dim=0)
        P_plus  = torch.stack([I4 + G[mu] for mu in range(4)], dim=0)
        self.register_buffer("P_minus_re", P_minus.real.float())
        self.register_buffer("P_minus_im", P_minus.imag.float())
        self.register_buffer("P_plus_re",  P_plus.real.float())
        self.register_buffer("P_plus_im",  P_plus.imag.float())

    @staticmethod
    def _color_mv(U_re, U_im, v_re, v_im):
        r_re = (torch.einsum("...ij,...sj->...si", U_re, v_re)
                - torch.einsum("...ij,...sj->...si", U_im, v_im))
        r_im = (torch.einsum("...ij,...sj->...si", U_re, v_im)
                + torch.einsum("...ij,...sj->...si", U_im, v_re))
        return r_re, r_im

    @staticmethod
    def _color_dag_mv(U_re, U_im, v_re, v_im):
        r_re = (torch.einsum("...ji,...sj->...si", U_re, v_re)
                + torch.einsum("...ji,...sj->...si", U_im, v_im))
        r_im = (torch.einsum("...ji,...sj->...si", U_re, v_im)
                - torch.einsum("...ji,...sj->...si", U_im, v_re))
        return r_re, r_im

    @staticmethod
    def _spin_mv(P_re, P_im, v_re, v_im):
        r_re = (torch.einsum("ij,...jk->...ik", P_re, v_re)
                - torch.einsum("ij,...jk->...ik", P_im, v_im))
        r_im = (torch.einsum("ij,...jk->...ik", P_re, v_im)
                + torch.einsum("ij,...jk->...ik", P_im, v_re))
        return r_re, r_im

    def forward(
        self,
        psi_re: torch.Tensor, psi_im: torch.Tensor,
        hl_re: torch.Tensor,  hl_im: torch.Tensor,
        hr_re: torch.Tensor,  hr_im: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        result_re = self.diag * psi_re
        result_im = self.diag * psi_im

        for mu in range(4):
            U_mu_re = self.U_local_re[..., mu, :, :]
            U_mu_im = self.U_local_im[..., mu, :, :]
            # Lattice axes T,Z,Y,X live at positions -6..-3 of psi.
            ldim = mu - 6

            if mu == 0:
                # T axis: splice halos instead of rolling.
                #   pf[t] = psi[t+1] for t<T_local-1, hr[0] for t=T_local-1
                #   pb[t] = psi[t-1] for t>0,        hl[0] for t=0
                pf_re = torch.cat([psi_re[1:], hr_re], dim=0)
                pf_im = torch.cat([psi_im[1:], hr_im], dim=0)
                pb_re = torch.cat([hl_re, psi_re[:-1]], dim=0)
                pb_im = torch.cat([hl_im, psi_im[:-1]], dim=0)
                # Backward link U†(t-1, μ=0) at output site t:
                #   for local t=0    → U_tm1_mu0 (baked from previous shard)
                #   for local t>0    → U_local at local t-1, μ=0
                Ub_re = torch.cat([self.U_tm1_mu0_re, U_mu_re[:-1]], dim=0)
                Ub_im = torch.cat([self.U_tm1_mu0_im, U_mu_im[:-1]], dim=0)
            else:
                pf_re = torch.roll(psi_re, -1, dims=ldim)
                pf_im = torch.roll(psi_im, -1, dims=ldim)
                pb_re = torch.roll(psi_re,  1, dims=ldim)
                pb_im = torch.roll(psi_im,  1, dims=ldim)
                Ub_re = torch.roll(U_mu_re, 1, dims=ldim)
                Ub_im = torch.roll(U_mu_im, 1, dims=ldim)

            # Forward hop:  − ½ (I − γ_μ) U(x,μ) ψ(x+μ̂)
            Upf_re, Upf_im = self._color_mv(U_mu_re, U_mu_im, pf_re, pf_im)
            cf_re,  cf_im  = self._spin_mv(
                self.P_minus_re[mu], self.P_minus_im[mu], Upf_re, Upf_im
            )
            # Backward hop: − ½ (I + γ_μ) U†(x−μ̂,μ) ψ(x−μ̂)
            Upb_re, Upb_im = self._color_dag_mv(Ub_re, Ub_im, pb_re, pb_im)
            cb_re,  cb_im  = self._spin_mv(
                self.P_plus_re[mu], self.P_plus_im[mu], Upb_re, Upb_im
            )

            result_re = result_re - 0.5 * (cf_re + cb_re)
            result_im = result_im - 0.5 * (cf_im + cb_im)

        return result_re, result_im


def _slice_shard_gauge(
    U_full: torch.Tensor,
    shard_idx: int,
    num_shards: int,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``(U_local_re, U_local_im, U_tm1_mu0_re, U_tm1_mu0_im)`` for shard.

    *U_local* covers the shard's interior global rows ``[t0, t1)``.  The
    extra ``U_tm1_mu0`` slab carries the μ=0 link at ``global t = t0-1``
    (modulo T) so the shard adapter can compute the backward-T hop at its
    leftmost local row without needing the previous shard's full gauge.
    """
    T_full = U_full.shape[0]
    T_local, _ = _shard_T_indices(T_full, num_shards)
    t0 = shard_idx * T_local
    t1 = t0 + T_local

    # Local U(t0..t1-1, *, *, *, μ, c, c).
    U_local = U_full[t0:t1].contiguous()
    # Extra μ=0 slab at global (t0 - 1) % T_full, with the slab dim retained.
    tm1 = (t0 - 1) % T_full
    U_tm1_mu0 = U_full[tm1:tm1 + 1, :, :, :, 0, :, :].contiguous()

    return (
        U_local.real.to(dtype).contiguous(),
        U_local.imag.to(dtype).contiguous(),
        U_tm1_mu0.real.to(dtype).contiguous(),
        U_tm1_mu0.imag.to(dtype).contiguous(),
    )


class _UnbakedShardedAdapter(nn.Module):
    """Wilson hop on a T-sharded sub-volume with gauge field as INPUT.

    Unlike :class:`_ShardedBakedGaugeAdapter` which bakes the gauge field
    into each shard's NEFF as registered buffers, this adapter takes the
    gauge slab as a *forward-time input*.  This allows **all shards to
    reuse a single compiled NEFF** — the adapter signature is identical
    for every shard; only the tensor *values* differ.  The NeuronCore
    never needs to swap instruction streams between shards, eliminating
    the catastrophic NEFF-reload overhead that dominates latency when
    the number of shards exceeds the number of available NeuronCores.

    The tradeoff is additional PCIe traffic per call (the gauge slab
    crosses the bus), but at ~6 MB per shard on a 24^4 lattice that is
    orders of magnitude cheaper than a full NEFF reload (~100 ms).

    Inputs:
        psi_re/im     : (T_local, Z, Y, X, Ns, Nc)     — local spinor
        U_local_re/im : (T_local, Z, Y, X, 4, Nc, Nc)  — local gauge field
        Utm1_re/im    : (1, Z, Y, X, Nc, Nc)            — U(t0-1, μ=0) for bwd-T
        hl_re/im      : (1, Z, Y, X, Ns, Nc)            — left halo
        hr_re/im      : (1, Z, Y, X, Ns, Nc)            — right halo
    """

    def __init__(self, diag: float, nc: int) -> None:
        super().__init__()
        self.diag = float(diag)
        self.nc = nc

        from ..dirac.gamma import degrand_rossi_gammas
        G  = degrand_rossi_gammas(dtype=torch.complex64)
        I4 = torch.eye(4, dtype=torch.complex64)
        P_minus = torch.stack([I4 - G[mu] for mu in range(4)], dim=0)
        P_plus  = torch.stack([I4 + G[mu] for mu in range(4)], dim=0)
        self.register_buffer("P_minus_re", P_minus.real.float())
        self.register_buffer("P_minus_im", P_minus.imag.float())
        self.register_buffer("P_plus_re",  P_plus.real.float())
        self.register_buffer("P_plus_im",  P_plus.imag.float())

    @staticmethod
    def _color_mv(U_re, U_im, v_re, v_im):
        r_re = (torch.einsum("...ij,...sj->...si", U_re, v_re)
                - torch.einsum("...ij,...sj->...si", U_im, v_im))
        r_im = (torch.einsum("...ij,...sj->...si", U_re, v_im)
                + torch.einsum("...ij,...sj->...si", U_im, v_re))
        return r_re, r_im

    @staticmethod
    def _color_dag_mv(U_re, U_im, v_re, v_im):
        r_re = (torch.einsum("...ji,...sj->...si", U_re, v_re)
                + torch.einsum("...ji,...sj->...si", U_im, v_im))
        r_im = (torch.einsum("...ji,...sj->...si", U_re, v_im)
                - torch.einsum("...ji,...sj->...si", U_im, v_re))
        return r_re, r_im

    @staticmethod
    def _spin_mv(P_re, P_im, v_re, v_im):
        r_re = (torch.einsum("ij,...jk->...ik", P_re, v_re)
                - torch.einsum("ij,...jk->...ik", P_im, v_im))
        r_im = (torch.einsum("ij,...jk->...ik", P_re, v_im)
                + torch.einsum("ij,...jk->...ik", P_im, v_re))
        return r_re, r_im

    def forward(
        self,
        psi_re: torch.Tensor, psi_im: torch.Tensor,
        U_local_re: torch.Tensor, U_local_im: torch.Tensor,
        Utm1_re: torch.Tensor, Utm1_im: torch.Tensor,
        hl_re: torch.Tensor,  hl_im: torch.Tensor,
        hr_re: torch.Tensor,  hr_im: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        result_re = self.diag * psi_re
        result_im = self.diag * psi_im

        for mu in range(4):
            U_mu_re = U_local_re[..., mu, :, :]
            U_mu_im = U_local_im[..., mu, :, :]
            ldim = mu - 6

            if mu == 0:
                pf_re = torch.cat([psi_re[1:], hr_re], dim=0)
                pf_im = torch.cat([psi_im[1:], hr_im], dim=0)
                pb_re = torch.cat([hl_re, psi_re[:-1]], dim=0)
                pb_im = torch.cat([hl_im, psi_im[:-1]], dim=0)
                Ub_re = torch.cat([Utm1_re, U_mu_re[:-1]], dim=0)
                Ub_im = torch.cat([Utm1_im, U_mu_im[:-1]], dim=0)
            else:
                pf_re = torch.roll(psi_re, -1, dims=ldim)
                pf_im = torch.roll(psi_im, -1, dims=ldim)
                pb_re = torch.roll(psi_re,  1, dims=ldim)
                pb_im = torch.roll(psi_im,  1, dims=ldim)
                Ub_re = torch.roll(U_mu_re, 1, dims=ldim)
                Ub_im = torch.roll(U_mu_im, 1, dims=ldim)

            Upf_re, Upf_im = self._color_mv(U_mu_re, U_mu_im, pf_re, pf_im)
            cf_re,  cf_im  = self._spin_mv(
                self.P_minus_re[mu], self.P_minus_im[mu], Upf_re, Upf_im
            )
            Upb_re, Upb_im = self._color_dag_mv(Ub_re, Ub_im, pb_re, pb_im)
            cb_re,  cb_im  = self._spin_mv(
                self.P_plus_re[mu], self.P_plus_im[mu], Upb_re, Upb_im
            )

            result_re = result_re - 0.5 * (cf_re + cb_re)
            result_im = result_im - 0.5 * (cf_im + cb_im)

        return result_re, result_im


class _BatchedUnbakedShardedAdapter(nn.Module):
    """All-shards-in-one-call Wilson hop with a leading shard batch dim.

    Processes **all** T-axis shards in a single NEFF invocation by adding
    a leading ``S`` (shard) dimension.  Instead of dispatching ``S``
    separate NEFF calls (each costing ~1-5 ms DMA + launch overhead),
    the host pre-assembles all shard inputs into a single contiguous
    tensor and invokes the compiled model once.

    All spatial roll / einsum operations naturally broadcast over the
    leading ``S`` dimension.  The T-axis halo splice uses the dim=1
    (T_local) axis, leaving ``S`` untouched.

    Inputs (all with leading shard dim ``S``):
        psi_re/im     : (S, T_local, Z, Y, X, Ns, Nc)
        U_local_re/im : (S, T_local, Z, Y, X, 4, Nc, Nc)
        Utm1_re/im    : (S, 1, Z, Y, X, Nc, Nc)
        hl_re/im      : (S, 1, Z, Y, X, Ns, Nc)
        hr_re/im      : (S, 1, Z, Y, X, Ns, Nc)
    """

    def __init__(self, diag: float, nc: int) -> None:
        super().__init__()
        self.diag = float(diag)
        self.nc = nc

        from ..dirac.gamma import degrand_rossi_gammas
        G  = degrand_rossi_gammas(dtype=torch.complex64)
        I4 = torch.eye(4, dtype=torch.complex64)
        P_minus = torch.stack([I4 - G[mu] for mu in range(4)], dim=0)
        P_plus  = torch.stack([I4 + G[mu] for mu in range(4)], dim=0)
        self.register_buffer("P_minus_re", P_minus.real.float())
        self.register_buffer("P_minus_im", P_minus.imag.float())
        self.register_buffer("P_plus_re",  P_plus.real.float())
        self.register_buffer("P_plus_im",  P_plus.imag.float())

    @staticmethod
    def _color_mv(U_re, U_im, v_re, v_im):
        r_re = (torch.einsum("...ij,...sj->...si", U_re, v_re)
                - torch.einsum("...ij,...sj->...si", U_im, v_im))
        r_im = (torch.einsum("...ij,...sj->...si", U_re, v_im)
                + torch.einsum("...ij,...sj->...si", U_im, v_re))
        return r_re, r_im

    @staticmethod
    def _color_dag_mv(U_re, U_im, v_re, v_im):
        r_re = (torch.einsum("...ji,...sj->...si", U_re, v_re)
                + torch.einsum("...ji,...sj->...si", U_im, v_im))
        r_im = (torch.einsum("...ji,...sj->...si", U_re, v_im)
                - torch.einsum("...ji,...sj->...si", U_im, v_re))
        return r_re, r_im

    @staticmethod
    def _spin_mv(P_re, P_im, v_re, v_im):
        r_re = (torch.einsum("ij,...jk->...ik", P_re, v_re)
                - torch.einsum("ij,...jk->...ik", P_im, v_im))
        r_im = (torch.einsum("ij,...jk->...ik", P_re, v_im)
                + torch.einsum("ij,...jk->...ik", P_im, v_re))
        return r_re, r_im

    def forward(
        self,
        psi_re: torch.Tensor, psi_im: torch.Tensor,
        U_local_re: torch.Tensor, U_local_im: torch.Tensor,
        Utm1_re: torch.Tensor, Utm1_im: torch.Tensor,
        hl_re: torch.Tensor,  hl_im: torch.Tensor,
        hr_re: torch.Tensor,  hr_im: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # psi: (S, T_local, Z, Y, X, Ns, Nc)
        # With the leading S dim, lattice axes sit at -6..-3 same as before,
        # and T_local is at dim 1.
        result_re = self.diag * psi_re
        result_im = self.diag * psi_im

        for mu in range(4):
            U_mu_re = U_local_re[..., mu, :, :]  # (S, T_l, Z, Y, X, Nc, Nc)
            U_mu_im = U_local_im[..., mu, :, :]

            # Lattice axes T,Z,Y,X in psi are at dims 1,2,3,4 (with S at 0).
            # With suffix (Ns,Nc) at -2,-1, they are also at -6,-5,-4,-3.
            # Use negative indexing for Z/Y/X to match the einsum pattern.
            ldim = mu - 6  # T→-6, Z→-5, Y→-4, X→-3

            if mu == 0:
                # T axis (dim -6 = dim 1): halo splice along dim=1.
                # psi[:,1:,...] + hr → forward neighbour
                pf_re = torch.cat([psi_re[:, 1:], hr_re], dim=1)
                pf_im = torch.cat([psi_im[:, 1:], hr_im], dim=1)
                pb_re = torch.cat([hl_re, psi_re[:, :-1]], dim=1)
                pb_im = torch.cat([hl_im, psi_im[:, :-1]], dim=1)
                Ub_re = torch.cat([Utm1_re, U_mu_re[:, :-1]], dim=1)
                Ub_im = torch.cat([Utm1_im, U_mu_im[:, :-1]], dim=1)
            else:
                # Z/Y/X: standard roll; dim is -5/-4/-3 which is correct
                # for (S, T_l, Z, Y, X, Ns, Nc) tensors.
                pf_re = torch.roll(psi_re, -1, dims=ldim)
                pf_im = torch.roll(psi_im, -1, dims=ldim)
                pb_re = torch.roll(psi_re,  1, dims=ldim)
                pb_im = torch.roll(psi_im,  1, dims=ldim)
                Ub_re = torch.roll(U_mu_re, 1, dims=ldim)
                Ub_im = torch.roll(U_mu_im, 1, dims=ldim)

            Upf_re, Upf_im = self._color_mv(U_mu_re, U_mu_im, pf_re, pf_im)
            cf_re,  cf_im  = self._spin_mv(
                self.P_minus_re[mu], self.P_minus_im[mu], Upf_re, Upf_im
            )
            Upb_re, Upb_im = self._color_dag_mv(Ub_re, Ub_im, pb_re, pb_im)
            cb_re,  cb_im  = self._spin_mv(
                self.P_plus_re[mu], self.P_plus_im[mu], Upb_re, Upb_im
            )

            result_re = result_re - 0.5 * (cf_re + cb_re)
            result_im = result_im - 0.5 * (cf_im + cb_im)

        return result_re, result_im


class _ShardedDslashWrapper(nn.Module):
    """Host orchestrator for T-axis sharded Dslash execution.

    Splits a full-lattice complex64 spinor along T into ``num_shards``
    slabs, gathers one-slab halos from neighbour shards under periodic
    boundary conditions, dispatches each shard through a compiled adapter,
    and concatenates the per-shard outputs along T.

    In **single-NEFF mode** (``gauge_field`` provided, ``shard_modules`` is
    a list of references to the *same* compiled NEFF or a small set of
    per-core copies), the gauge slab is passed as a forward-time input and
    no NEFF swapping occurs on the NeuronCores.  This is the fast path for
    large lattices where the number of shards exceeds the number of cores.

    In **legacy baked-gauge mode** (``gauge_field=None``), each entry in
    ``shard_modules`` is a distinct compiled NEFF with its own baked gauge
    slab.  The only inputs are the spinor and halos.

    Args:
        shard_modules:    List of compiled (or eager) shard adapters,
                          one per shard.  In single-NEFF mode, multiple
                          entries may point to the same underlying NEFF.
        num_shards:       Number of T-slabs.
        T_local:          Sites per slab (T // num_shards).
        compute_dtype:    Internal real dtype for the dispatched call.
        dispatch_workers: Maximum number of host threads used to fire the
                          per-shard NEFF calls concurrently.  Defaults to
                          ``num_shards`` (one thread per shard).  Set to 1
                          to force the legacy serial loop.
        gauge_field:      Full-lattice complex64 gauge tensor.  When
                          provided, activates single-NEFF mode: per-shard
                          gauge slices are computed once at construction and
                          passed as forward inputs.
    """

    def __init__(
        self,
        shard_modules,
        num_shards: int,
        T_local: int,
        compute_dtype: torch.dtype = torch.float32,
        dispatch_workers: Optional[int] = None,
        gauge_field: Optional[torch.Tensor] = None,
        batched_module: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        self._shard_modules = list(shard_modules)
        self.num_shards = num_shards
        self.T_local = T_local
        self._compute_dtype = compute_dtype
        self._dispatch_workers = dispatch_workers or num_shards
        self._executor: Optional["ThreadPoolExecutor"] = None

        # Batched single-NEFF mode: one NEFF call for all shards.
        self._batched_module = batched_module
        self._unbaked = gauge_field is not None

        # Pre-compute per-shard gauge slices (used by both batched and
        # per-shard unbaked paths).
        if self._unbaked:
            dt = compute_dtype
            self._gauge_slices: list = []
            T_full = gauge_field.shape[0]
            for s in range(num_shards):
                t0 = s * T_local
                t1 = t0 + T_local
                U_local = gauge_field[t0:t1].contiguous()
                tm1 = (t0 - 1) % T_full
                U_tm1_mu0 = gauge_field[tm1:tm1 + 1, :, :, :, 0, :, :].contiguous()
                self._gauge_slices.append((
                    U_local.real.to(dt).contiguous(),
                    U_local.imag.to(dt).contiguous(),
                    U_tm1_mu0.real.to(dt).contiguous(),
                    U_tm1_mu0.imag.to(dt).contiguous(),
                ))
            # Pre-stack gauge slices for the batched path.
            if batched_module is not None:
                self._U_stacked_re = torch.stack(
                    [g[0] for g in self._gauge_slices], dim=0
                )
                self._U_stacked_im = torch.stack(
                    [g[1] for g in self._gauge_slices], dim=0
                )
                self._Utm1_stacked_re = torch.stack(
                    [g[2] for g in self._gauge_slices], dim=0
                )
                self._Utm1_stacked_im = torch.stack(
                    [g[3] for g in self._gauge_slices], dim=0
                )

    def _get_executor(self):
        if self._executor is None and self._dispatch_workers > 1:
            from concurrent.futures import ThreadPoolExecutor
            self._executor = ThreadPoolExecutor(
                max_workers=self._dispatch_workers,
                thread_name_prefix="lqcd-shard",
            )
        return self._executor

    @torch.inference_mode()
    def forward(self, psi: torch.Tensor, U: torch.Tensor = None) -> torch.Tensor:
        del U
        dt = self._compute_dtype
        T = psi.shape[0]
        if T != self.num_shards * self.T_local:
            raise ValueError(
                f"_ShardedDslashWrapper: psi T-extent {T} does not match "
                f"num_shards * T_local = {self.num_shards} * {self.T_local}."
            )

        psi_re = psi.real.to(dt).contiguous()
        psi_im = psi.imag.to(dt).contiguous()

        # ------------------------------------------------------------------
        # Batched single-NEFF path: one dispatch for all shards.
        # ------------------------------------------------------------------
        if self._batched_module is not None and self._unbaked:
            S = self.num_shards
            Tl = self.T_local
            # Build (S, T_local, Z, Y, X, Ns, Nc) tensors.
            slab_re = psi_re.view(S, Tl, *psi_re.shape[1:])
            slab_im = psi_im.view(S, Tl, *psi_im.shape[1:])
            # Halos: left[s] = psi row (s*Tl - 1) % T, right[s] = (s*Tl + Tl) % T
            l_indices = [((s * Tl) - 1) % T for s in range(S)]
            r_indices = [((s * Tl) + Tl) % T for s in range(S)]
            hl_re = torch.stack([psi_re[i:i+1] for i in l_indices], dim=0)
            hl_im = torch.stack([psi_im[i:i+1] for i in l_indices], dim=0)
            hr_re = torch.stack([psi_re[i:i+1] for i in r_indices], dim=0)
            hr_im = torch.stack([psi_im[i:i+1] for i in r_indices], dim=0)

            r_re, r_im = self._batched_module(
                slab_re, slab_im,
                self._U_stacked_re, self._U_stacked_im,
                self._Utm1_stacked_re, self._Utm1_stacked_im,
                hl_re, hl_im, hr_re, hr_im,
            )
            # (S, T_local, ...) → (T, ...)
            out_re = r_re.reshape(T, *r_re.shape[2:]).float()
            out_im = r_im.reshape(T, *r_im.shape[2:]).float()
            return torch.complex(out_re, out_im)

        # ------------------------------------------------------------------
        # Per-shard fallback (legacy baked-gauge or per-shard unbaked).
        # ------------------------------------------------------------------

        shard_args = []
        for s in range(self.num_shards):
            t0 = s * self.T_local
            t1 = t0 + self.T_local
            local_re = psi_re[t0:t1].contiguous()
            local_im = psi_im[t0:t1].contiguous()

            l_idx = (t0 - 1) % T
            r_idx = t1 % T
            hl_re = psi_re[l_idx:l_idx + 1].contiguous()
            hl_im = psi_im[l_idx:l_idx + 1].contiguous()
            hr_re = psi_re[r_idx:r_idx + 1].contiguous()
            hr_im = psi_im[r_idx:r_idx + 1].contiguous()

            if self._unbaked:
                # Single-NEFF mode: pass pre-sliced gauge as input.
                U_l_re, U_l_im, Utm1_re, Utm1_im = self._gauge_slices[s]
                shard_args.append((
                    local_re, local_im,
                    U_l_re, U_l_im, Utm1_re, Utm1_im,
                    hl_re, hl_im, hr_re, hr_im,
                ))
            else:
                # Legacy baked-gauge mode.
                shard_args.append((local_re, local_im, hl_re, hl_im, hr_re, hr_im))

        executor = self._get_executor()
        if executor is None or self.num_shards == 1:
            results = [
                self._shard_modules[s](*shard_args[s])
                for s in range(self.num_shards)
            ]
        else:
            futures = [
                executor.submit(self._shard_modules[s], *shard_args[s])
                for s in range(self.num_shards)
            ]
            results = [f.result() for f in futures]

        out_re_shards = [r_re.float() for r_re, _ in results]
        out_im_shards = [r_im.float() for _, r_im in results]

        out_re = torch.cat(out_re_shards, dim=0)
        out_im = torch.cat(out_im_shards, dim=0)
        return torch.complex(out_re, out_im)


# Default per-shard volume cap.  Measured at --optlevel=1 the unfused
# baked-gauge Dslash emits ~32 HLO instructions per site, so the
# neuronx-cc ~5M HLO budget translates to ≈156k sites per NEFF.  Two
# data points pinning that ratio:
#
#   V = 1,048,576 (32^4)              → 33.78M HLO  (~32.2 insn/site)
#   V =   262,144 (8×32×32×32 shard)  →  8.44M HLO  (~32.2 insn/site)
#
# We deliberately stay well under the nominal 156k ceiling: at
# --optlevel=2 (our default), CSE / strength-reduction passes inflate
# the per-site instruction count, and graphs that fit at optlevel=1
# can still trip neuronx-cc exit code 70 ("instruction count exceeds
# typical limit") at higher optimisation levels.  Empirically V=82,944
# (24^4 split into N=4 slabs) hits this cliff, so we pick 80,000 to
# auto-route 24^4 to N=8 and 32^4 to N=16.  Override via
# ``compile_dslash_sharded(..., num_shards=N)`` for finer control;
# ``compile_dslash_sharded`` will also auto-retry with progressively
# more shards if a per-shard NEFF fails to compile.
_DEFAULT_SHARD_VOLUME_CAP = 80_000


def _auto_num_shards(lattice_shape: Tuple[int, int, int, int]) -> int:
    """Smallest power-of-2 num_shards along T that brings V_local ≤ cap.

    Returns 1 when the full lattice already fits the per-NEFF budget.
    """
    T, Z, Y, X = lattice_shape
    V = T * Z * Y * X
    if V <= _DEFAULT_SHARD_VOLUME_CAP:
        return 1
    n = 1
    while V // n > _DEFAULT_SHARD_VOLUME_CAP and n < T:
        n *= 2
        if T % n != 0:
            # Fall back to the largest divisor of T that is ≤ n.
            for cand in range(n, 1, -1):
                if T % cand == 0:
                    n = cand
                    break
            break
    return max(1, min(n, T))


class NeuronCompiler:
    """Compile ``nn.Module`` operators for execution on Neuron hardware.

    Args:
        workdir:        Directory for neuronx-cc intermediate artefacts.
                        Defaults to ``~/.cache/lqcd-neuron/neuronx``.
        dtype:          Data type for compilation (``'float32'`` or
                        ``'bfloat16'``).  Trn1/Inf2 prefer bfloat16.
        optimize_level: Compiler optimisation level (1–3).
        device:         Override the detected :class:`NeuronDevice`.
        sram_threshold_bytes:
                        Override the byte budget compared against the fused
                        kernel size to decide whether to auto-fall back to the
                        unfused baked-gauge path.  Defaults to
                        ``0.60 * 24 MiB`` on NeuronCore-v2.  Pass a very large
                        value (e.g. ``10**12``) to defer the check
                        indefinitely.
        allow_fused_fallback:
                        When ``True`` (default) ``compile_dslash(..., fused=True)``
                        silently downgrades to the unfused path if the fused
                        kernels exceed ``sram_threshold_bytes``.  When ``False``
                        that downgrade becomes a :class:`RuntimeError`, useful
                        when running A/B benchmarks that must isolate the
                        fused kernel.
    """

    def __init__(
        self,
        workdir: Optional[str] = None,
        dtype: str = "float32",
        optimize_level: int = 2,
        device: Optional[NeuronDevice] = None,
        sram_threshold_bytes: Optional[int] = None,
        allow_fused_fallback: bool = True,
    ) -> None:
        self.dtype = dtype
        self.optimize_level = optimize_level
        self._device = device or get_device()
        self._cache: Dict[Tuple, Any] = {}
        # Override the auto-detected SRAM budget for the fused-kernel fallback.
        # None means use the default fraction of _NC2_SRAM_BYTES.
        self.sram_threshold_bytes = sram_threshold_bytes
        # When False, raising the budget no longer matters: any attempt to
        # silently downgrade fused -> unfused becomes a hard RuntimeError so
        # users measuring fused-only throughput aren't surprised by a
        # backend swap they didn't request.
        self.allow_fused_fallback = allow_fused_fallback

        if workdir is None:
            workdir = str(Path.home() / ".cache" / "lqcd-neuron" / "neuronx")
        self.workdir = workdir
        os.makedirs(self.workdir, exist_ok=True)

    # ------------------------------------------------------------------
    # Dtype helpers
    # ------------------------------------------------------------------

    def _disk_cache_path(self, cache_key: Optional[str]) -> Optional[Path]:
        """Return the .pt path for *cache_key*, or None if key is falsy."""
        if not cache_key:
            return None
        # Sanitise key so it is safe as a filename
        safe = cache_key.replace(" ", "_").replace("(", "").replace(")", "").replace(",", "-")
        return Path(self.workdir) / f"{safe}.pt"

    @property
    def torch_dtype(self) -> torch.dtype:
        return torch.bfloat16 if self.dtype == "bfloat16" else torch.float32

    def _to_complex_dtype(self) -> torch.dtype:
        # torch.complex32 does not exist in PyTorch; always use complex64.
        return torch.complex64

    # ------------------------------------------------------------------
    # Core compilation entry point
    # ------------------------------------------------------------------

    def compile(
        self,
        model: nn.Module,
        example_inputs: Tuple[torch.Tensor, ...],
        cache_key: Optional[str] = None,
    ) -> nn.Module:
        """Compile *model* for Neuron given *example_inputs*.

        On non-Neuron hardware (CPU, CUDA) this returns the original
        *model* unchanged so the same calling code works everywhere.

        Args:
            model:          The ``nn.Module`` to compile.
            example_inputs: Tuple of example input tensors (matching shapes
                            and dtypes that the model will be called with at
                            runtime).
            cache_key:      Optional string key for the in-process cache.

        Returns:
            Compiled ``ScriptModule`` (Neuron) or the original module (CPU).
        """
        if cache_key and cache_key in self._cache:
            return self._cache[cache_key]

        # Check persistent on-disk cache first
        disk_path = self._disk_cache_path(cache_key) if cache_key else None
        if disk_path and disk_path.exists():
            logger.info("Loading cached Neuron module from %s", disk_path)
            compiled = torch.jit.load(str(disk_path))
            if cache_key:
                self._cache[cache_key] = compiled
            return compiled

        if not self._device.is_neuron:
            logger.info(
                "No Neuron hardware detected — returning PyTorch CPU module."
            )
            return model

        try:
            import torch_neuronx
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "torch-neuronx is required for Neuron compilation.  "
                "Install it with: pip install torch-neuronx"
            ) from exc

        logger.info(
            "Compiling %s for Neuron (dtype=%s, opt=%d) …",
            type(model).__name__,
            self.dtype,
            self.optimize_level,
        )

        # Set compiler environment flags
        os.environ.setdefault("NEURON_CC_FLAGS", f"--optlevel {self.optimize_level}")

        # Log the equivalent neuronx-cc invocation for transparency / debugging.
        # torch_neuronx.trace serialises the traced graph to a temporary HLO and
        # invokes `neuronx-cc compile <hlo> --framework XLA <NEURON_CC_FLAGS>`
        # under the hood.  We surface the flags and input signature here.
        def _sig(t: Any) -> str:
            if isinstance(t, torch.Tensor):
                return f"{tuple(t.shape)}:{str(t.dtype).replace('torch.', '')}"
            if isinstance(t, (list, tuple)):
                return "(" + ", ".join(_sig(x) for x in t) + ")"
            return type(t).__name__

        cc_flags = os.environ.get("NEURON_CC_FLAGS", "")
        logger.info(
            "neuronx-cc compile <traced-hlo> --framework XLA %s  "
            "# model=%s inputs=%s",
            cc_flags,
            type(model).__name__,
            _sig(example_inputs),
        )

        compiled = torch_neuronx.trace(model, example_inputs)

        if cache_key:
            self._cache[cache_key] = compiled
            if disk_path:
                torch.jit.save(compiled, str(disk_path))
                logger.info("Cached compiled module to %s", disk_path)

        logger.info("Compilation complete.")
        return compiled

    # ------------------------------------------------------------------
    # Convenience methods for common operators
    # ------------------------------------------------------------------

    def compile_dslash(
        self,
        dslash_module: nn.Module,
        lattice_shape: Tuple[int, int, int, int],
        nc: int = 3,
        ns: int = 4,
        gauge_field: Optional[torch.Tensor] = None,
        fused: bool = True,
        num_shards: Optional[int] = None,
    ) -> nn.Module:
        """Compile a Dslash / Dirac operator for a fixed lattice shape.

        Uses a pure float32 real-arithmetic adapter to work around the
        ``neuronx-cc`` restriction on complex dtypes (NCC_EVRF004).  The
        returned module preserves the standard ``forward(psi, U)`` interface
        with ``complex64`` tensors; the re/im split happens transparently
        inside a host-side wrapper.

        On non-Neuron hardware the original *dslash_module* is returned
        unchanged.

        Args:
            dslash_module: A ``WilsonDslash`` or ``WilsonDirac`` instance.
            lattice_shape: ``(T, Z, Y, X)`` lattice extents.
            nc:            Number of colours.
            ns:            Number of spin components.
            gauge_field:   Optional gauge tensor ``(T, Z, Y, X, 4, Nc, Nc)``
                           of ``complex64``.  When provided the gauge field is
                           baked into the compiled model as NeuronCore-resident
                           buffers so only the spinor crosses PCIe per call.
                           Recommended for benchmarks and iterative solvers
                           where *U* stays constant.
            fused:         When *True* (default) and *gauge_field* is provided,
                           pre-compute per-site, per-direction
                           ``(Ns*Nc) × (Ns*Nc)`` hopping kernels and bake them
                           into the NEFF.  When *False* and *gauge_field* is
                           provided, bake only the raw gauge tensor and keep
                           the spin/colour einsums in the graph.  Useful for
                           large lattices where the fused kernels overflow
                           NeuronCore on-chip memory.  Ignored when
                           *gauge_field* is *None*.
            num_shards:    Optional explicit T-axis shard count.  Forces
                           the sharded compile path even when the lattice
                           fits the single-NEFF budget.  When *None*
                           (default), sharding is auto-selected based on
                           ``_DEFAULT_SHARD_VOLUME_CAP`` (and
                           ``compile_dslash_sharded`` will further auto-
                           refine if a per-shard NEFF still fails to
                           compile).

        Returns:
            Module with ``forward(psi, U)`` accepting ``complex64`` tensors.
            When *gauge_field* was provided, *U* is accepted but ignored.
        """
        from ..dirac.wilson import (
            WilsonDirac,
            WilsonDslash,
            _NeuronWilsonDiracAdapter,
            _NeuronWilsonDslashAdapter,
        )

        if not self._device.is_neuron:
            logger.info(
                "No Neuron hardware detected — returning PyTorch CPU module."
            )
            return _attach_compile_info(
                dslash_module,
                kernel="cpu",
                lattice_shape=tuple(lattice_shape),
                batch_size=1,
                num_cores=1,
                num_shards=1,
                T_local=lattice_shape[0],
                fused_fallback=False,
                sharded_fallback=False,
            )

        if isinstance(dslash_module, WilsonDirac):
            adapter: nn.Module = _NeuronWilsonDiracAdapter(
                mass=dslash_module.mass, nc=nc
            )
        elif isinstance(dslash_module, WilsonDslash):
            adapter = _NeuronWilsonDslashAdapter(nc=nc)
        else:
            raise TypeError(
                f"compile_dslash: unsupported module type {type(dslash_module).__name__}. "
                "Only WilsonDslash and WilsonDirac are currently supported."
            )

        T, Z, Y, X = lattice_shape
        dt = self.torch_dtype
        # torch_neuronx.trace requires CPU tensors as example inputs regardless
        # of whether torch_xla is installed.  Using xm.xla_device() inputs would
        # put the traced model into XLA lazy-execution mode, so computations are
        # enqueued but never flushed to NeuronCores without an explicit
        # xm.mark_step() call — the root cause of 0% neuron-top utilisation.
        cpu = torch.device("cpu")

        if gauge_field is not None:
            # Track which auto-downgrades fired so the returned module's
            # compile_info reflects the *actual* path executed at runtime.
            fused_fallback = False
            sharded_fallback = False
            fused_kernel_mib: Optional[float] = None
            sram_budget_mib: Optional[float] = None

            # Honour an explicit num_shards request up-front: skip the
            # fused/unfused single-NEFF attempts entirely and go straight
            # to the sharded path.  This lets callers preempt the cap
            # heuristic when they know a finer split is needed.
            if num_shards is not None:
                logger.info(
                    "compile_dslash: explicit num_shards=%d requested — "
                    "routing through compile_dslash_sharded.",
                    num_shards,
                )
                sharded = self.compile_dslash_sharded(
                    dslash_module, lattice_shape, gauge_field,
                    num_shards=num_shards, nc=nc, ns=ns,
                )
                return _attach_compile_info(
                    sharded,
                    sharded_fallback=False,
                    fused_fallback=False,
                )

            # Auto-fallback: if the fused per-site (Ns×Nc)² kernels would
            # overflow NeuronCore SRAM, use the unfused baked-gauge path
            # instead.  The unfused path has ~12× smaller on-chip working set
            # (raw gauge links vs. full spin-colour fused matrices) and
            # outperforms the fused path once the latter causes HBM spill.
            if fused:
                sram_budget = (
                    self.sram_threshold_bytes
                    or int(_NC2_SRAM_BYTES * _FUSED_SRAM_BUDGET)
                )
                kb = _fused_kernel_bytes(lattice_shape, ns=ns, nc=nc, dtype=dt)
                fused_kernel_mib = kb / 1024**2
                sram_budget_mib = sram_budget / 1024**2
                if kb > sram_budget:
                    if not self.allow_fused_fallback:
                        raise RuntimeError(
                            f"compile_dslash: fused kernels "
                            f"({kb / 1024**2:.1f} MiB) exceed SRAM budget "
                            f"({sram_budget / 1024**2:.1f} MiB) for lattice "
                            f"{lattice_shape}, and "
                            "NeuronCompiler(allow_fused_fallback=False) was "
                            "set.  Either raise sram_threshold_bytes (e.g. "
                            "10**12 to disable the check), pass fused=False "
                            "explicitly, or re-enable allow_fused_fallback."
                        )
                    logger.warning(
                        "compile_dslash: fused kernels (%.1f MiB) exceed SRAM "
                        "budget (%.1f MiB) for lattice %s — auto-falling back "
                        "to unfused baked-gauge path. "
                        "Override with NeuronCompiler(sram_threshold_bytes=N), "
                        "pass fused=False explicitly, or set "
                        "allow_fused_fallback=False to raise instead.",
                        kb / 1024**2,
                        sram_budget / 1024**2,
                        lattice_shape,
                    )
                    fused = False
                    fused_fallback = True

            if not fused:
                # Bake the gauge field as raw (T,Z,Y,X,4,Nc,Nc) buffers but
                # keep the spin/colour einsums in the traced graph.  Working
                # set per call is ~12× smaller than the fused (Ns*Nc)² kernels,
                # so this path can outperform the fused one once the latter
                # overflows NeuronCore on-chip memory.
                #
                # For very large lattices (V > 24^4 ≈ 331k sites) even the
                # unfused single-NEFF graph blows past the neuronx-cc HLO
                # instruction budget (~5M instructions), so we hand off to
                # the T-axis sharded path which compiles one NEFF per slab.
                if T * Z * Y * X > _DEFAULT_SHARD_VOLUME_CAP:
                    auto_n = _auto_num_shards(lattice_shape)
                    logger.warning(
                        "compile_dslash: V=%d exceeds per-NEFF HLO budget "
                        "(cap=%d sites) — auto-routing through "
                        "compile_dslash_sharded with num_shards=%d "
                        "(T_local=%d).  Pass num_shards explicitly via "
                        "compile_dslash_sharded() to override.",
                        T * Z * Y * X, _DEFAULT_SHARD_VOLUME_CAP,
                        auto_n, T // auto_n,
                    )
                    sharded = self.compile_dslash_sharded(
                        dslash_module, lattice_shape, gauge_field,
                        num_shards=auto_n, nc=nc, ns=ns,
                    )
                    return _attach_compile_info(
                        sharded,
                        sharded_fallback=True,
                        fused_fallback=fused_fallback,
                        fused_kernel_mib=fused_kernel_mib,
                        sram_budget_mib=sram_budget_mib,
                    )

                adapter = adapter.to(dt)
                U_re = gauge_field.real.to(dt).contiguous()
                U_im = gauge_field.imag.to(dt).contiguous()
                baked = _BakedGaugeAdapter(adapter, U_re, U_im)
                psi_re = torch.zeros(T, Z, Y, X, ns, nc, dtype=dt, device=cpu)
                psi_im = torch.zeros_like(psi_re)
                # No cache key — NEFF embeds this specific gauge configuration.
                compiled = self.compile(baked, (psi_re, psi_im))
                return _attach_compile_info(
                    _BakedGaugeDslashWrapper(compiled, compute_dtype=dt),
                    kernel="unfused",
                    lattice_shape=tuple(lattice_shape),
                    fused_kernel_mib=fused_kernel_mib,
                    sram_budget_mib=sram_budget_mib,
                    fused_fallback=fused_fallback,
                    sharded_fallback=False,
                    num_shards=1,
                    T_local=T,
                    batch_size=1,
                    num_cores=1,
                )

            # Bake the gauge field into the compiled model as fused per-site,
            # per-direction (Ns*Nc)×(Ns*Nc) hopping kernels.  At runtime each
            # call only does a roll + one large matvec per direction-side; the
            # backward-U rolls and the spin/colour einsum split are absorbed
            # into the precomputed buffers.  Only the spinor crosses PCIe.
            if isinstance(dslash_module, WilsonDirac):
                diag = 4.0 + dslash_module.mass
            else:
                diag = 0.0
            K_fwd_re, K_fwd_im, K_bwd_re, K_bwd_im = _build_dslash_kernels(
                gauge_field, nc=nc, ns=ns, dtype=dt,
            )
            fused = _FusedDslashAdapter(
                K_fwd_re, K_fwd_im, K_bwd_re, K_bwd_im,
                diag=diag, ns=ns, nc=nc,
            ).to(dt)
            psi_re = torch.zeros(T, Z, Y, X, ns, nc, dtype=dt, device=cpu)
            psi_im = torch.zeros_like(psi_re)
            # No cache key — the NEFF embeds this specific gauge configuration.
            compiled = self.compile(fused, (psi_re, psi_im))
            return _attach_compile_info(
                _FusedDslashWrapper(compiled, compute_dtype=dt),
                kernel="fused",
                lattice_shape=tuple(lattice_shape),
                fused_kernel_mib=fused_kernel_mib,
                sram_budget_mib=sram_budget_mib,
                fused_fallback=False,
                sharded_fallback=False,
                num_shards=1,
                T_local=T,
                batch_size=1,
                num_cores=1,
            )

        adapter = adapter.to(dt)
        psi_re = torch.zeros(T, Z, Y, X, ns, nc, dtype=dt, device=cpu)
        psi_im = torch.zeros_like(psi_re)
        U_re   = torch.zeros(T, Z, Y, X, 4, nc, nc, dtype=dt, device=cpu)
        U_im   = torch.zeros_like(U_re)

        key = f"dslash_{type(dslash_module).__name__}_{lattice_shape}_{nc}_{dt}"
        compiled = self.compile(adapter, (psi_re, psi_im, U_re, U_im), cache_key=key)
        return _attach_compile_info(
            _ComplexDslashWrapper(compiled, compute_dtype=dt),
            kernel="complex_in_graph",
            lattice_shape=tuple(lattice_shape),
            fused_fallback=False,
            sharded_fallback=False,
            num_shards=1,
            T_local=T,
            batch_size=1,
            num_cores=1,
        )

    def compile_dslash_batched(
        self,
        dslash_module: nn.Module,
        lattice_shape: Tuple[int, int, int, int],
        batch_size: int,
        gauge_field: torch.Tensor,
        nc: int = 3,
        ns: int = 4,
    ) -> nn.Module:
        """Compile a multi-RHS Wilson Dslash / Dirac operator.

        Multi-RHS amortises the fixed NeuronCore dispatch overhead (~1 ms
        per call) across *batch_size* spinors and lets the tensor engine
        operate on larger contractions, dramatically improving throughput
        on the small-to-medium lattices typical for LQCD inversions.

        The gauge field is baked into the compiled model as a NeuronCore-
        resident buffer (broadcast across the batch dim), so only the
        batched spinor crosses PCIe per call.

        Args:
            dslash_module: A ``WilsonDslash`` or ``WilsonDirac`` instance.
            lattice_shape: ``(T, Z, Y, X)`` lattice extents.
            batch_size:    Number of right-hand sides per call.
            gauge_field:   Gauge tensor ``(T, Z, Y, X, 4, Nc, Nc)`` of
                           ``complex64``.  Required — the NEFF embeds this
                           specific configuration.
            nc:            Number of colours.
            ns:            Number of spin components.

        Returns:
            Module whose ``forward(psi)`` accepts a batched complex64
            spinor of shape ``(batch_size, T, Z, Y, X, Ns, Nc)``.
        """
        from ..dirac.wilson import (
            WilsonDirac,
            WilsonDslash,
            _NeuronWilsonDiracAdapter,
            _NeuronWilsonDslashAdapter,
        )

        if not self._device.is_neuron:
            logger.info(
                "No Neuron hardware detected — returning a host-side batched shim."
            )

            class _CpuBatched(nn.Module):
                def __init__(self, m: nn.Module, U_: torch.Tensor) -> None:
                    super().__init__()
                    self.m = m
                    self.U = U_

                def forward(self, psi: torch.Tensor) -> torch.Tensor:
                    return self.m(psi, self.U)

            return _CpuBatched(dslash_module, gauge_field)

        if isinstance(dslash_module, WilsonDirac):
            adapter: nn.Module = _NeuronWilsonDiracAdapter(
                mass=dslash_module.mass, nc=nc
            )
        elif isinstance(dslash_module, WilsonDslash):
            adapter = _NeuronWilsonDslashAdapter(nc=nc)
        else:
            raise TypeError(
                f"compile_dslash_batched: unsupported module type "
                f"{type(dslash_module).__name__}.  Only WilsonDslash and "
                f"WilsonDirac are currently supported."
            )

        T, Z, Y, X = lattice_shape
        dt = self.torch_dtype
        cpu = torch.device("cpu")

        # Guard rails — when the fused multi-RHS NEFF would overflow SRAM
        # or the per-NEFF HLO instruction budget, fall back to a host-side
        # loop over a single-RHS compile_dslash (which has its own auto
        # fused→unfused→sharded fallbacks).  This preserves the batched
        # API at the cost of per-RHS dispatch overhead.
        V = T * Z * Y * X
        sram_budget = (
            self.sram_threshold_bytes
            or int(_NC2_SRAM_BYTES * _FUSED_SRAM_BUDGET)
        )
        kb = _fused_kernel_bytes(lattice_shape, ns=ns, nc=nc, dtype=dt)
        needs_host_loop_reason: Optional[str] = None
        if kb > sram_budget:
            needs_host_loop_reason = (
                f"fused kernels ({kb / 1024**2:.1f} MiB) exceed SRAM budget "
                f"({sram_budget / 1024**2:.1f} MiB)"
            )
        elif V > _DEFAULT_SHARD_VOLUME_CAP:
            needs_host_loop_reason = (
                f"V={V} exceeds per-NEFF HLO instruction budget "
                f"(cap={_DEFAULT_SHARD_VOLUME_CAP} sites)"
            )

        if needs_host_loop_reason is not None:
            logger.warning(
                "compile_dslash_batched: %s for lattice %s — falling back "
                "to a host-side loop over compile_dslash (single-RHS, with "
                "auto fused→unfused→sharded fallbacks).  Per-RHS dispatch "
                "overhead is no longer amortised across the batch.",
                needs_host_loop_reason, lattice_shape,
            )
            single = self.compile_dslash(
                dslash_module, lattice_shape, nc=nc, ns=ns,
                gauge_field=gauge_field, fused=True,
            )
            single_info = getattr(single, "lqcd_compile_info", {}) or {}
            return _attach_compile_info(
                _HostLoopBatchedWrapper(single),
                kernel=single_info.get("kernel", "host_loop"),
                lattice_shape=tuple(lattice_shape),
                fused_fallback=bool(single_info.get("fused_fallback", False)),
                sharded_fallback=bool(single_info.get("sharded_fallback", False)),
                num_shards=int(single_info.get("num_shards", 1)),
                T_local=int(single_info.get("T_local", T)),
                batch_size=int(batch_size),
                num_cores=1,
                batched_host_loop=True,
            )

        if isinstance(dslash_module, WilsonDirac):
            diag = 4.0 + dslash_module.mass
        else:
            diag = 0.0
        K_fwd_re, K_fwd_im, K_bwd_re, K_bwd_im = _build_dslash_kernels(
            gauge_field, nc=nc, ns=ns, dtype=dt,
        )
        fused = _FusedBatchedDslashAdapter(
            K_fwd_re, K_fwd_im, K_bwd_re, K_bwd_im,
            diag=diag, ns=ns, nc=nc,
        ).to(dt)
        psi_re = torch.zeros(batch_size, T, Z, Y, X, ns, nc, dtype=dt, device=cpu)
        psi_im = torch.zeros_like(psi_re)
        # No cache key — the NEFF embeds this specific gauge configuration.
        compiled = self.compile(fused, (psi_re, psi_im))
        return _attach_compile_info(
            _FusedBatchedDslashWrapper(compiled, compute_dtype=dt),
            kernel="fused",
            lattice_shape=tuple(lattice_shape),
            fused_fallback=False,
            sharded_fallback=False,
            num_shards=1,
            T_local=T,
            batch_size=int(batch_size),
            num_cores=1,
        )

    def compile_dslash_eo(
        self,
        dslash_module: nn.Module,
        lattice_shape: Tuple[int, int, int, int],
        out_parity: int,
        gauge_field: torch.Tensor,
        nc: int = 3,
        ns: int = 4,
    ) -> nn.Module:
        """Compile a half-lattice even-odd Wilson hop D_{out ← in}.

        Builds fused hopping kernels only at *out_parity* output sites
        (V/2 sites instead of V), halving the NeuronCore on-chip working
        set.  This defers the SRAM-spill cliff by one lattice doubling:

        - ``16×8×8×8`` full-lattice fused kernels: ~37.7 MiB (spills SRAM)
        - ``16×8×8×8`` half-lattice fused kernels: ~18.9 MiB (fits in SRAM)

        The returned module accepts and returns **half-lattice** complex64
        spinors of shape ``(T, Z, Y, X//2, Ns, Nc)``.  Use
        :func:`pack_checkerboard` / :func:`unpack_checkerboard` to convert
        between full-lattice and half-lattice representations.

        Args:
            dslash_module: ``WilsonDslash`` or ``WilsonDirac`` instance
                           (only the ``mass`` attribute is used for the
                           diagonal; the actual graph is replaced by the
                           fused adapter).
            lattice_shape: ``(T, Z, Y, X)`` full-lattice extents.
                           *X* must be even.
            out_parity:    Parity of output sites: 0 (even) or 1 (odd).
                           ``out_parity=1`` gives D_oe (odd output, even
                           input); ``out_parity=0`` gives D_eo.
            gauge_field:   Full-lattice gauge tensor
                           ``(T, Z, Y, X, 4, Nc, Nc)`` of ``complex64``.
                           The gauge field is pre-processed at compile time
                           and baked into the NEFF.
            nc:            Number of colours.
            ns:            Number of spin components.

        Returns:
            Module with ``forward(psi_half)`` accepting a complex64
            half-lattice spinor ``(T, Z, Y, X//2, Ns, Nc)``.
        """
        from ..dirac.wilson import WilsonDirac, WilsonDslash

        if not self._device.is_neuron:
            logger.info(
                "No Neuron hardware detected — returning CPU even-odd shim."
            )
            from ..dirac.wilson import EvenOddWilsonDslash

            is_dirac = isinstance(dslash_module, WilsonDirac)
            mass = dslash_module.mass if is_dirac else 0.0
            return EvenOddWilsonDslash(
                mass=mass, nc=nc, dtype=torch.complex64
            ).hop(out_parity)

        T, Z, Y, X = lattice_shape
        assert X % 2 == 0, "X must be even for even-odd decomposition"
        dt = self.torch_dtype
        cpu = torch.device("cpu")

        if isinstance(dslash_module, WilsonDirac):
            diag = 4.0 + dslash_module.mass
        else:
            diag = 0.0

        K_fwd_re, K_fwd_im, K_bwd_re, K_bwd_im = _build_dslash_kernels_halfvol(
            gauge_field, out_parity=out_parity, nc=nc, ns=ns, dtype=dt,
        )
        adapter = _HalfLatticeHopAdapter(
            K_fwd_re, K_fwd_im, K_bwd_re, K_bwd_im,
            diag=diag, ns=ns, nc=nc,
            out_parity=out_parity,
            lattice_shape=lattice_shape,
        ).to(dt)
        psi_re = torch.zeros(T, Z, Y, X // 2, ns, nc, dtype=dt, device=cpu)
        psi_im = torch.zeros_like(psi_re)
        compiled = self.compile(adapter, (psi_re, psi_im))
        return _attach_compile_info(
            _HalfLatticeDslashWrapper(compiled, compute_dtype=dt),
            kernel="half_lattice_eo",
            lattice_shape=tuple(lattice_shape),
            out_parity=int(out_parity),
            fused_fallback=False,
            sharded_fallback=False,
            num_shards=1,
            T_local=T,
            batch_size=1,
            num_cores=1,
        )

    def compile_dslash_sharded(
        self,
        dslash_module: nn.Module,
        lattice_shape: Tuple[int, int, int, int],
        gauge_field: torch.Tensor,
        num_shards: Optional[int] = None,
        nc: int = 3,
        ns: int = 4,
    ) -> nn.Module:
        """Compile a Dslash with the lattice T-axis split into ``num_shards``.

        Each shard handles a ``(T // num_shards, Z, Y, X)`` sub-volume and
        compiles to its own NEFF.  Per-shard HLO instruction count scales
        with ``T_local`` rather than ``T``, restoring compilability for
        lattices that overflow the single-NEFF ``neuronx-cc`` budget
        (typically ``V > 24^4``).

        Halo exchange across the sharded T axis happens host-side via
        :class:`_ShardedDslashWrapper` under periodic boundary conditions.
        Z/Y/X axes stay local to each shard and use ordinary ``torch.roll``.

        On non-Neuron hardware the original *dslash_module* is returned
        unchanged — sharding is purely a Neuron compile-budget workaround
        and adds nothing on CPU.

        Args:
            dslash_module: ``WilsonDslash`` or ``WilsonDirac`` instance.
            lattice_shape: ``(T, Z, Y, X)`` full-lattice extents.
            gauge_field:   Full-lattice ``complex64`` tensor
                           ``(T, Z, Y, X, 4, Nc, Nc)``.  Each shard's NEFF
                           bakes the corresponding T-slab plus one extra
                           slab from the previous shard for the boundary
                           backward link.
            num_shards:    Number of T-slabs.  Must divide *T*.  When
                           ``None`` (default) chosen automatically so that
                           ``V_local ≤ 24^4`` (the empirical per-NEFF
                           compile budget for the unfused path).
            nc:            Number of colours.
            ns:            Number of spin components.

        Returns:
            Module with the standard ``forward(psi, U)`` signature.  *U*
            is accepted but ignored — the gauge field is already baked
            into each shard's NEFF.
        """
        from ..dirac.wilson import WilsonDirac, WilsonDslash

        T, Z, Y, X = lattice_shape

        if not self._device.is_neuron:
            logger.info(
                "No Neuron hardware detected — compile_dslash_sharded "
                "returning the original CPU module unchanged."
            )
            return dslash_module

        if num_shards is None:
            num_shards = _auto_num_shards(lattice_shape)
        T_local, num_shards = _shard_T_indices(T, num_shards)

        if isinstance(dslash_module, WilsonDirac):
            diag = 4.0 + dslash_module.mass
        elif isinstance(dslash_module, WilsonDslash):
            diag = 0.0
        else:
            raise TypeError(
                f"compile_dslash_sharded: unsupported module type "
                f"{type(dslash_module).__name__}.  Only WilsonDslash and "
                f"WilsonDirac are currently supported."
            )

        dt = self.torch_dtype
        cpu = torch.device("cpu")

        # Auto-retry policy.  neuronx-cc occasionally fails with exit
        # code 70 ("instructions generated exceeds typical limit") for
        # per-shard graphs that are nominally under the budget but trip
        # CSE/strength-reduction passes at higher --optlevel.  Rather
        # than surface a hard error, halve T_local (double num_shards)
        # and retry, up to T (one row per slab).
        max_retries = 0
        n = num_shards
        while n < T:
            n *= 2
            max_retries += 1
        retries_done = 0
        sharded_retry = False
        initial_num_shards = num_shards

        total_cores = max(1, self._device.num_cores)

        # -----------------------------------------------------------
        # Batched single-NEFF mode: compile ONE
        # _BatchedUnbakedShardedAdapter that processes all shards
        # in a single dispatch call.  The leading `S` (shard) dim
        # batches the per-shard work so there is exactly 1 PCIe
        # round-trip and 1 NeuronCore dispatch per full-lattice
        # Dslash application — no per-shard dispatch overhead.
        # -----------------------------------------------------------
        while True:
            logger.info(
                "compile_dslash_sharded: V=%d sharded along T into %d slabs of "
                "T_local=%d (V_local=%d).  Compiling 1 batched NEFF …",
                T * Z * Y * X, num_shards, T_local,
                T_local * Z * Y * X,
            )

            adapter = _BatchedUnbakedShardedAdapter(diag=diag, nc=nc).to(dt)
            S = num_shards
            psi_re = torch.zeros(S, T_local, Z, Y, X, ns, nc, dtype=dt, device=cpu)
            psi_im = torch.zeros_like(psi_re)
            U_l_re = torch.zeros(S, T_local, Z, Y, X, 4, nc, nc, dtype=dt, device=cpu)
            U_l_im = torch.zeros_like(U_l_re)
            Utm1_re = torch.zeros(S, 1, Z, Y, X, nc, nc, dtype=dt, device=cpu)
            Utm1_im = torch.zeros_like(Utm1_re)
            hl_re = torch.zeros(S, 1, Z, Y, X, ns, nc, dtype=dt, device=cpu)
            hl_im = torch.zeros_like(hl_re)
            hr_re = torch.zeros_like(hl_re)
            hr_im = torch.zeros_like(hl_re)

            try:
                compiled = self.compile(
                    adapter,
                    (psi_re, psi_im, U_l_re, U_l_im,
                     Utm1_re, Utm1_im, hl_re, hl_im, hr_re, hr_im),
                )
                break  # compilation succeeded
            except RuntimeError as exc:
                if retries_done >= max_retries:
                    raise
                next_n = num_shards * 2
                while next_n <= T and T % next_n != 0:
                    next_n += 1
                if next_n > T:
                    raise
                logger.warning(
                    "compile_dslash_sharded: batched compile failed at "
                    "num_shards=%d (T_local=%d) — %s.  Retrying with "
                    "num_shards=%d (T_local=%d).",
                    num_shards, T_local, exc, next_n, T // next_n,
                )
                T_local, num_shards = _shard_T_indices(T, next_n)
                retries_done += 1
                sharded_retry = True

        logger.info(
            "compile_dslash_sharded: batched single-NEFF compiled for "
            "%d shards — 1 dispatch per Dslash call (no per-shard overhead).",
            num_shards,
        )

        return _attach_compile_info(
            _ShardedDslashWrapper(
                [],  # no per-shard modules needed in batched mode
                num_shards=num_shards,
                T_local=T_local,
                compute_dtype=dt,
                gauge_field=gauge_field,
                batched_module=compiled,
            ),
            kernel="sharded",
            lattice_shape=tuple(lattice_shape),
            fused_fallback=False,
            sharded_fallback=False,
            sharded_retry=sharded_retry,
            initial_num_shards=int(initial_num_shards),
            num_shards=int(num_shards),
            T_local=int(T_local),
            batch_size=1,
            num_cores=1,
            single_neff=True,
            batched_shards=True,
        )

    def compile_observable(
        self,
        observable_module: nn.Module,
        lattice_shape: Tuple[int, int, int, int],
        nc: int = 3,
    ) -> nn.Module:
        """Compile a gauge observable (plaquette nn.Module) for a fixed shape.

        Args:
            observable_module: A plaquette or Polyakov-loop ``nn.Module``.
            lattice_shape:     ``(T, Z, Y, X)``.
            nc:                Number of colours.

        Returns:
            Compiled or original module.
        """
        # neuronx-cc rejects complex dtypes (NCC_EVRF004).  Split into real/imag
        # float32 tensors, matching the pattern used by compile_dslash and
        # compile_plaquette.  Callers that need a complex-input interface should
        # use compile_plaquette (which wraps _ComplexInputWrapper) instead.
        # See compile_dslash: example inputs must be CPU tensors for torch_neuronx.trace.
        T, Z, Y, X = lattice_shape
        dt = self.torch_dtype
        cpu = torch.device("cpu")

        observable_module = observable_module.to(dt)
        U_re = torch.zeros(T, Z, Y, X, 4, nc, nc, dtype=dt, device=cpu)
        U_im = torch.zeros_like(U_re)
        key = f"obs_{type(observable_module).__name__}_{lattice_shape}_{nc}_{dt}"
        return self.compile(observable_module, (U_re, U_im), cache_key=key)

    def compile_plaquette(
        self,
        lattice_shape: Tuple[int, int, int, int],
        nc: int = 3,
    ) -> nn.Module:
        """Compile the plaquette observable for Neuron hardware.

        Unlike :meth:`compile_observable`, this method uses a real-arithmetic
        implementation (:class:`_NeuronPlaquetteAdapter`) to avoid the
        ``complex64`` dtype restriction of ``neuronx-cc`` (NCC_EVRF004).

        On non-Neuron hardware the method returns a CPU-compatible wrapper
        with the same interface so calling code works unchanged everywhere.

        Args:
            lattice_shape: ``(T, Z, Y, X)`` lattice extents.
            nc:            Number of colours.

        Returns:
            Module that accepts a ``complex64`` gauge tensor of shape
            ``(T, Z, Y, X, 4, Nc, Nc)`` and returns the average plaquette
            as a real scalar tensor.
        """
        T, Z, Y, X = lattice_shape
        dt = self.torch_dtype
        adapter = _NeuronPlaquetteAdapter(nc).to(dt)

        if not self._device.is_neuron:
            logger.info(
                "No Neuron hardware detected — returning CPU plaquette module."
            )
            return _ComplexInputWrapper(adapter, compute_dtype=dt)

        # See compile_dslash: example inputs must be CPU tensors for torch_neuronx.trace.
        cpu = torch.device("cpu")
        U_re = torch.zeros(T, Z, Y, X, 4, nc, nc, dtype=dt, device=cpu)
        U_im = torch.zeros_like(U_re)
        key = f"plaquette_{lattice_shape}_{nc}_{dt}"
        compiled = self.compile(adapter, (U_re, U_im), cache_key=key)
        return _ComplexInputWrapper(compiled, compute_dtype=dt)

    # ------------------------------------------------------------------
    # Multi-core data-parallel compilation
    # ------------------------------------------------------------------

    def compile_multicore(
        self,
        model: nn.Module,
        example_inputs: Tuple[torch.Tensor, ...],
        num_cores: Optional[int] = None,
        cache_key: Optional[str] = None,
    ) -> nn.Module:
        """Compile *model* and replicate across multiple NeuronCores.

        Uses ``torch_neuronx.DataParallel`` to distribute input batches
        across *num_cores* NeuronCores.  Each core receives an equal slice
        of the leading (batch) dimension.

        On non-Neuron hardware, returns the original model unchanged.

        Args:
            model:          The ``nn.Module`` to compile.
            example_inputs: Tuple of example input tensors (single-core shapes).
            num_cores:      Number of NeuronCores to use.  Defaults to all
                            detected cores (``NeuronDevice.num_cores``).
            cache_key:      Optional string key for the in-process cache.

        Returns:
            A DataParallel-wrapped compiled module that splits input dim 0
            across *num_cores* NeuronCores.
        """
        if not self._device.is_neuron:
            logger.info(
                "No Neuron hardware detected — returning PyTorch CPU module."
            )
            return model

        if num_cores is None:
            num_cores = self._device.num_cores

        # Compile a single-core NEFF first
        compiled = self.compile(model, example_inputs, cache_key=cache_key)

        if num_cores <= 1:
            return compiled

        try:
            import torch_neuronx
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "torch-neuronx is required for multi-core compilation.  "
                "Install it with: pip install torch-neuronx"
            ) from exc

        device_ids = list(range(num_cores))
        logger.info(
            "Wrapping compiled model with DataParallel across %d cores.",
            num_cores,
        )
        return torch_neuronx.DataParallel(compiled, device_ids=device_ids, dim=0)

    def compile_dslash_multicore(
        self,
        dslash_module: nn.Module,
        lattice_shape: Tuple[int, int, int, int],
        gauge_field: torch.Tensor,
        num_cores: Optional[int] = None,
        per_core_batch_size: int = 1,
        nc: int = 3,
        ns: int = 4,
    ) -> nn.Module:
        """Compile a multi-core data-parallel Dslash operator.

        Distributes batched spinors across multiple NeuronCores for maximum
        throughput.  Each core processes ``per_core_batch_size`` right-hand
        sides concurrently; the host splits a global batch of
        ``num_cores * per_core_batch_size`` RHS along dim 0.

        The gauge field is baked into each core's compiled model, so only the
        spinor crosses PCIe per call.

        Args:
            dslash_module:        ``WilsonDslash`` or ``WilsonDirac`` instance.
            lattice_shape:        ``(T, Z, Y, X)`` lattice extents.
            gauge_field:          Gauge tensor ``(T,Z,Y,X,4,Nc,Nc)`` complex64.
            num_cores:            NeuronCores to use (default: all detected).
            per_core_batch_size:  RHS handled by each core per call.  Larger
                                  values amortise dispatch overhead and fill
                                  the tensor engine better.
            nc:                   Number of colours.
            ns:                   Number of spin components.

        Returns:
            A :class:`_MultiCoreDslashWrapper` whose ``forward(psi)`` accepts
            a batched complex64 spinor ``(B, T, Z, Y, X, Ns, Nc)`` with
            ``B == num_cores * per_core_batch_size``.
        """
        from ..dirac.wilson import WilsonDirac, WilsonDslash

        if num_cores is None:
            num_cores = self._device.num_cores

        if not self._device.is_neuron:
            logger.info(
                "No Neuron hardware detected — returning CPU batched shim."
            )

            class _CpuMulticore(nn.Module):
                def __init__(self, m: nn.Module, U_: torch.Tensor) -> None:
                    super().__init__()
                    self.m = m
                    self.U = U_

                def forward(self, psi: torch.Tensor) -> torch.Tensor:
                    return self.m(psi, self.U)

            return _CpuMulticore(dslash_module, gauge_field)

        T, Z, Y, X = lattice_shape
        dt = self.torch_dtype
        cpu = torch.device("cpu")

        if isinstance(dslash_module, WilsonDirac):
            diag = 4.0 + dslash_module.mass
        else:
            diag = 0.0

        # Guard rail: the fused multi-RHS adapter has the same SRAM and
        # HLO-budget overflow modes as compile_dslash_batched.  At
        # V > _DEFAULT_SHARD_VOLUME_CAP or kernels > SRAM the
        # _FusedBatchedDslashAdapter compile is guaranteed to fail with
        # NCC_EVRF007 / exit code 70.  Rather than crash and surface a
        # ``[batched failed]`` annotation in the bench table, fall back
        # to a host-loop wrapper around the single-RHS sharded module.
        # This mirrors the compile_dslash_batched fallback chain.
        kb = _fused_kernel_bytes(lattice_shape, ns=ns, nc=nc, dtype=dt)
        sram_budget = (
            self.sram_threshold_bytes
            or int(_NC2_SRAM_BYTES * _FUSED_SRAM_BUDGET)
        )
        V = T * Z * Y * X
        if kb > sram_budget or V > _DEFAULT_SHARD_VOLUME_CAP:
            logger.warning(
                "compile_dslash_multicore: fused kernels (%.1f MiB) exceed "
                "SRAM budget (%.1f MiB) or V=%d exceeds per-NEFF HLO budget "
                "(cap=%d sites) for lattice %s — falling back to a "
                "host-side loop over a sharded single-RHS NEFF (no "
                "DataParallel replication).  Per-RHS dispatch overhead is "
                "no longer amortised across the global batch.",
                kb / 1024**2, sram_budget / 1024**2,
                V, _DEFAULT_SHARD_VOLUME_CAP, lattice_shape,
            )
            single = self.compile_dslash(
                dslash_module, lattice_shape, nc=nc, ns=ns,
                gauge_field=gauge_field, fused=True,
            )
            return _attach_compile_info(
                _HostLoopBatchedWrapper(single),
                kernel=getattr(single, "lqcd_compile_info", {}).get(
                    "kernel", "sharded"
                ),
                lattice_shape=tuple(lattice_shape),
                fused_fallback=True,
                sharded_fallback=True,
                multicore_host_loop=True,
                num_shards=getattr(single, "lqcd_compile_info", {}).get(
                    "num_shards", 1
                ),
                T_local=getattr(single, "lqcd_compile_info", {}).get(
                    "T_local", T
                ),
                batch_size=int(num_cores * per_core_batch_size),
                num_cores=int(num_cores),
            )

        K_fwd_re, K_fwd_im, K_bwd_re, K_bwd_im = _build_dslash_kernels(
            gauge_field, nc=nc, ns=ns, dtype=dt,
        )
        fused = _FusedBatchedDslashAdapter(
            K_fwd_re, K_fwd_im, K_bwd_re, K_bwd_im,
            diag=diag, ns=ns, nc=nc,
        ).to(dt)

        # Each core's NEFF is compiled for per_core_batch_size RHS;
        # DataParallel splits the global dim-0 batch evenly across cores.
        psi_re = torch.zeros(per_core_batch_size, T, Z, Y, X, ns, nc, dtype=dt, device=cpu)
        psi_im = torch.zeros_like(psi_re)

        parallel = self.compile_multicore(fused, (psi_re, psi_im), num_cores=num_cores)
        return _attach_compile_info(
            _MultiCoreDslashWrapper(
                parallel,
                compute_dtype=dt,
                num_cores=num_cores,
                per_core_batch_size=per_core_batch_size,
            ),
            kernel="fused",
            lattice_shape=tuple(lattice_shape),
            fused_fallback=False,
            sharded_fallback=False,
            num_shards=1,
            T_local=T,
            batch_size=int(per_core_batch_size),
            num_cores=int(num_cores),
        )

    # ------------------------------------------------------------------
    # torch.compile backend (PyTorch 2.x alternative to trace)
    # ------------------------------------------------------------------

    @staticmethod
    def torch_compile(model: nn.Module, backend: str = "neuronx") -> nn.Module:
        """Wrap *model* with ``torch.compile(backend=backend)``.

        ``torch.compile`` provides a higher-level interface and supports
        dynamic shapes better than ``torch_neuronx.trace``.  Use this when
        the lattice size may vary between calls.

        Args:
            model:   The module to compile.
            backend: Compiler backend string.  ``'neuronx'`` for Trn1/Inf2,
                     ``'inductor'`` for GPU/CPU fallback.

        Returns:
            Compiled callable.
        """
        return torch.compile(model, backend=backend)
