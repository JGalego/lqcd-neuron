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
            # Bake the gauge field into the compiled model as NeuronCore-
            # resident buffers.  Only the spinor crosses PCIe per call.
            U_re = gauge_field.real.to(dt).contiguous()
            U_im = gauge_field.imag.to(dt).contiguous()
            baked = _BakedGaugeAdapter(adapter, U_re, U_im).to(dt)
            psi_re = torch.zeros(T, Z, Y, X, ns, nc, dtype=dt, device=cpu)
            psi_im = torch.zeros_like(psi_re)
            # No cache key — the NEFF embeds this specific gauge configuration.
            compiled = self.compile(baked, (psi_re, psi_im))
            return _BakedGaugeDslashWrapper(compiled, compute_dtype=dt)

        adapter = adapter.to(dt)
        psi_re = torch.zeros(T, Z, Y, X, ns, nc, dtype=dt, device=cpu)
        psi_im = torch.zeros_like(psi_re)
        U_re   = torch.zeros(T, Z, Y, X, 4, nc, nc, dtype=dt, device=cpu)
        U_im   = torch.zeros_like(U_re)

        key = f"dslash_{type(dslash_module).__name__}_{lattice_shape}_{nc}_{dt}"
        compiled = self.compile(adapter, (psi_re, psi_im, U_re, U_im), cache_key=key)
        return _ComplexDslashWrapper(compiled, compute_dtype=dt)

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
