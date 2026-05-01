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

    @torch.inference_mode()
    def forward(self, psi: torch.Tensor, U: torch.Tensor = None) -> torch.Tensor:
        dt = self._compute_dtype
        r_re, r_im = self._real_module(
            psi.real.to(dt).contiguous(),
            psi.imag.to(dt).contiguous(),
        )
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

    @torch.inference_mode()
    def forward(self, psi: torch.Tensor, U: torch.Tensor = None) -> torch.Tensor:
        dt = self._compute_dtype
        r_re, r_im = self._real_module(
            psi.real.to(dt).contiguous(),
            psi.imag.to(dt).contiguous(),
        )
        return torch.complex(r_re.float(), r_im.float())


class _FusedBatchedDslashWrapper(nn.Module):
    """Host-side shim around a compiled fused multi-RHS adapter."""

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
        # DataParallel splits dim 0 across cores automatically
        r_re, r_im = self._parallel_module(
            psi.real.to(dt).contiguous(),
            psi.imag.to(dt).contiguous(),
        )
        return torch.complex(r_re.float(), r_im.float())


class NeuronCompiler:
    """Compile ``nn.Module`` operators for execution on Neuron hardware.

    Args:
        workdir:        Directory for neuronx-cc intermediate artefacts.
                        Defaults to ``~/.cache/lqcd-neuron/neuronx``.
        dtype:          Data type for compilation (``'float32'`` or
                        ``'bfloat16'``).  Trn1/Inf2 prefer bfloat16.
        optimize_level: Compiler optimisation level (1–3).
        device:         Override the detected :class:`NeuronDevice`.
    """

    def __init__(
        self,
        workdir: Optional[str] = None,
        dtype: str = "float32",
        optimize_level: int = 2,
        device: Optional[NeuronDevice] = None,
    ) -> None:
        self.dtype = dtype
        self.optimize_level = optimize_level
        self._device = device or get_device()
        self._cache: Dict[Tuple, Any] = {}

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
            return dslash_module

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
            if not fused:
                # Bake the gauge field as raw (T,Z,Y,X,4,Nc,Nc) buffers but
                # keep the spin/colour einsums in the traced graph.  Working
                # set per call is ~12× smaller than the fused (Ns*Nc)² kernels,
                # so this path can outperform the fused one once the latter
                # overflows NeuronCore on-chip memory.
                adapter = adapter.to(dt)
                U_re = gauge_field.real.to(dt).contiguous()
                U_im = gauge_field.imag.to(dt).contiguous()
                baked = _BakedGaugeAdapter(adapter, U_re, U_im)
                psi_re = torch.zeros(T, Z, Y, X, ns, nc, dtype=dt, device=cpu)
                psi_im = torch.zeros_like(psi_re)
                # No cache key — NEFF embeds this specific gauge configuration.
                compiled = self.compile(baked, (psi_re, psi_im))
                return _BakedGaugeDslashWrapper(compiled, compute_dtype=dt)

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
            return _FusedDslashWrapper(compiled, compute_dtype=dt)

        adapter = adapter.to(dt)
        psi_re = torch.zeros(T, Z, Y, X, ns, nc, dtype=dt, device=cpu)
        psi_im = torch.zeros_like(psi_re)
        U_re   = torch.zeros(T, Z, Y, X, 4, nc, nc, dtype=dt, device=cpu)
        U_im   = torch.zeros_like(U_re)

        key = f"dslash_{type(dslash_module).__name__}_{lattice_shape}_{nc}_{dt}"
        compiled = self.compile(adapter, (psi_re, psi_im, U_re, U_im), cache_key=key)
        return _ComplexDslashWrapper(compiled, compute_dtype=dt)

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
        return _FusedBatchedDslashWrapper(compiled, compute_dtype=dt)

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
        return _MultiCoreDslashWrapper(
            parallel,
            compute_dtype=dt,
            num_cores=num_cores,
            per_core_batch_size=per_core_batch_size,
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
