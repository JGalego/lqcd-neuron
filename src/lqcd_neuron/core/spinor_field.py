"""
ColorSpinorField — quark spinor ψ(x) living on the lattice.

The backing tensor has shape (T, Z, Y, X, Ns, Nc), where Ns=4 for
Wilson-type fermions.  Complex arithmetic is done via torch.complex64 /
torch.complex128, making every operation JIT-traceable for Neuron.
"""

from __future__ import annotations

import math
from typing import Optional

import torch

from .lattice import LatticeGeometry


class ColorSpinorField:
    """Lattice quark field ψ_{α,a}(x).

    Args:
        tensor: Complex tensor of shape ``(T, Z, Y, X, Ns, Nc)``.
        geom:   Corresponding :class:`LatticeGeometry`.
    """

    def __init__(self, tensor: torch.Tensor, geom: LatticeGeometry) -> None:
        expected = geom.spinor_shape
        if tuple(tensor.shape) != expected:
            raise ValueError(
                f"Expected spinor tensor shape {expected}, got {tuple(tensor.shape)}"
            )
        self._tensor = tensor
        self._geom = geom

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------

    @classmethod
    def zeros(
        cls,
        geom: LatticeGeometry,
        dtype: torch.dtype = torch.complex64,
        device: Optional[torch.device] = None,
    ) -> "ColorSpinorField":
        """Return a zero spinor field."""
        return cls(torch.zeros(geom.spinor_shape, dtype=dtype, device=device), geom)

    @classmethod
    def gaussian(
        cls,
        geom: LatticeGeometry,
        dtype: torch.dtype = torch.complex64,
        device: Optional[torch.device] = None,
        seed: Optional[int] = None,
    ) -> "ColorSpinorField":
        """Return a Gaussian random spinor field, normalised to unit norm."""
        rng = torch.Generator(device=device)
        if seed is not None:
            rng.manual_seed(seed)
        shape = geom.spinor_shape
        real = torch.randn(shape, generator=rng, device=device)
        imag = torch.randn(shape, generator=rng, device=device)
        psi = torch.complex(real, imag).to(dtype)
        norm = psi.norm()
        return cls(psi / norm, geom)

    @classmethod
    def point_source(
        cls,
        geom: LatticeGeometry,
        t: int,
        z: int,
        y: int,
        x: int,
        spin: int = 0,
        color: int = 0,
        dtype: torch.dtype = torch.complex64,
        device: Optional[torch.device] = None,
    ) -> "ColorSpinorField":
        """Return a point-source spinor: δ(x-x₀) δ_{αα₀} δ_{cc₀}."""
        psi = torch.zeros(geom.spinor_shape, dtype=dtype, device=device)
        psi[t, z, y, x, spin, color] = 1.0
        return cls(psi, geom)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def tensor(self) -> torch.Tensor:
        return self._tensor

    @property
    def geom(self) -> LatticeGeometry:
        return self._geom

    @property
    def dtype(self) -> torch.dtype:
        return self._tensor.dtype

    @property
    def device(self) -> torch.device:
        return self._tensor.device

    # ------------------------------------------------------------------
    # Inner product and norms
    # ------------------------------------------------------------------

    def inner(self, other: "ColorSpinorField") -> torch.Tensor:
        """Hermitian inner product ⟨self|other⟩ = Σ_x ψ†(x) φ(x)."""
        return (self._tensor.conj() * other._tensor).sum()

    def norm2(self) -> torch.Tensor:
        """‖ψ‖² = ⟨ψ|ψ⟩, returned as a scalar tensor."""
        return (self._tensor.abs().pow(2)).sum()

    def norm(self) -> torch.Tensor:
        """‖ψ‖ = √‖ψ‖²."""
        return self.norm2().sqrt()

    # ------------------------------------------------------------------
    # Arithmetic operators (returns raw tensor, not a new field object)
    # ------------------------------------------------------------------

    def __add__(self, other: "ColorSpinorField") -> "ColorSpinorField":
        return ColorSpinorField(self._tensor + other._tensor, self._geom)

    def __sub__(self, other: "ColorSpinorField") -> "ColorSpinorField":
        return ColorSpinorField(self._tensor - other._tensor, self._geom)

    def __mul__(self, scalar) -> "ColorSpinorField":
        return ColorSpinorField(self._tensor * scalar, self._geom)

    __rmul__ = __mul__

    def to(
        self,
        device: torch.device = None,
        dtype: torch.dtype = None,
    ) -> "ColorSpinorField":
        t = self._tensor
        if device is not None:
            t = t.to(device=device)
        if dtype is not None:
            t = t.to(dtype=dtype)
        return ColorSpinorField(t, self._geom)

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"ColorSpinorField(geom={self._geom}, dtype={self.dtype}, "
            f"device={self.device})"
        )
