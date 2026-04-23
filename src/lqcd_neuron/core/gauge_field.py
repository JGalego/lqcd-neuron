"""
GaugeField — SU(Nc) link-variable container.

A gauge configuration is a rank-7 complex tensor of shape
    (T, Z, Y, X, 4, Nc, Nc)
where the fifth dimension indexes the four space-time directions
(μ = 0 → T, 1 → Z, 2 → Y, 3 → X) and the last two dimensions are the
Nc×Nc complex matrix at each site/direction.

All heavy tensor operations (staples, field-strength tensor, …) are
expressed as pure PyTorch calls so they can be compiled by the Neuron
SDK via torch_neuronx.trace or torch.compile(backend='neuronx').
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F

from .lattice import LatticeGeometry


class GaugeField:
    """SU(Nc) gauge field with helpers for common lattice operations.

    The backing tensor is always complex (torch.complex64 or complex128)
    and lives on the device provided at construction time.

    Args:
        tensor: Pre-allocated tensor of shape ``(T, Z, Y, X, 4, Nc, Nc)``.
        geom:   Matching :class:`LatticeGeometry` instance.

    Raises:
        ValueError: If *tensor* shape does not match *geom*.
    """

    def __init__(self, tensor: torch.Tensor, geom: LatticeGeometry) -> None:
        expected = geom.gauge_shape
        if tuple(tensor.shape) != expected:
            raise ValueError(
                f"Expected gauge tensor shape {expected}, got {tuple(tensor.shape)}"
            )
        if not tensor.is_floating_point() and not tensor.is_complex():
            raise TypeError("Gauge tensor must be floating-point or complex.")
        self._tensor = tensor
        self._geom = geom

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------

    @classmethod
    def cold(
        cls,
        geom: LatticeGeometry,
        dtype: torch.dtype = torch.complex64,
        device: Optional[torch.device] = None,
    ) -> "GaugeField":
        """Return a unity (cold-start) gauge configuration (all links = I)."""
        shape = geom.gauge_shape
        U = torch.zeros(shape, dtype=dtype, device=device)
        # Set diagonal to 1: identity SU(Nc) matrices
        eye = torch.eye(geom.nc, dtype=dtype, device=device)
        U[..., :, :] = eye  # broadcast over (T,Z,Y,X,4)
        return cls(U, geom)

    @classmethod
    def random(
        cls,
        geom: LatticeGeometry,
        dtype: torch.dtype = torch.complex64,
        device: Optional[torch.device] = None,
        seed: Optional[int] = None,
    ) -> "GaugeField":
        """Return a random (hot-start) SU(Nc) gauge field.

        Each link is drawn from the Haar measure via QR decomposition of a
        random complex Gaussian matrix, then the phase is fixed to obtain
        det = +1 (i.e. an element of SU(Nc)).
        """
        rng = torch.Generator(device=device)
        if seed is not None:
            rng.manual_seed(seed)
        T, Z, Y, X, ndirs, nc, _ = geom.gauge_shape
        n = T * Z * Y * X * ndirs
        # Random complex Gaussian: (n, nc, nc)
        real = torch.randn(n, nc, nc, generator=rng, device=device)
        imag = torch.randn(n, nc, nc, generator=rng, device=device)
        A = torch.complex(real, imag).to(dtype)
        # QR → Q is unitary; fix det to +1 for SU(nc)
        Q, R = torch.linalg.qr(A)  # Q: (n, nc, nc)
        # Make Q special-unitary: multiply by sign of diagonal of R
        d = torch.diagonal(R, dim1=-2, dim2=-1)  # (n, nc)
        phases = torch.sgn(d)  # real sign for complex numbers
        Q = Q * phases.unsqueeze(-2)
        # Fix determinant phase: det(Q) should have |det|=1; divide out common phase
        det = torch.linalg.det(Q)  # (n,)
        phase_correction = (det.abs() / det) ** (1.0 / nc)  # Nc-th root
        Q = Q / phase_correction.unsqueeze(-1).unsqueeze(-1)
        return cls(Q.reshape(geom.gauge_shape), geom)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def tensor(self) -> torch.Tensor:
        """The raw backing tensor of shape ``(T, Z, Y, X, 4, Nc, Nc)``."""
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
    # Arithmetic conveniences
    # ------------------------------------------------------------------

    def to(self, device: torch.device = None, dtype: torch.dtype = None) -> "GaugeField":
        """Move/cast the gauge field to a different device or dtype."""
        t = self._tensor
        if device is not None:
            t = t.to(device=device)
        if dtype is not None:
            t = t.to(dtype=dtype)
        return GaugeField(t, self._geom)

    def dagger(self) -> "GaugeField":
        """Return U† (conjugate-transpose of each link matrix)."""
        return GaugeField(self._tensor.conj().transpose(-1, -2), self._geom)

    def unitarize(self) -> "GaugeField":
        """Project each link back onto SU(Nc) via SVD-based polar decomposition."""
        U = self._tensor
        flat = U.reshape(-1, self._geom.nc, self._geom.nc)
        # Polar decomposition via SVD: A = U Σ V†  →  unitary part = U V†
        Usvd, _, Vh = torch.linalg.svd(flat)
        U_unitary = torch.einsum("...ij,...jk->...ik", Usvd, Vh)
        det = torch.linalg.det(U_unitary).unsqueeze(-1).unsqueeze(-1)
        phase = (det.abs() / det) ** (1.0 / self._geom.nc)
        U_su = (U_unitary / phase).reshape(U.shape)
        return GaugeField(U_su, self._geom)

    # ------------------------------------------------------------------
    # Lattice-shift helpers (used by Dslash and observables)
    # ------------------------------------------------------------------

    def shift(self, mu: int, forward: bool = True) -> "GaugeField":
        """Return the gauge field shifted by ±1 in direction μ.

        Args:
            mu:      Direction (0=T, 1=Z, 2=Y, 3=X).
            forward: True → U(x+μ̂), False → U(x-μ̂).
        """
        shift_amount = -1 if forward else 1
        return GaugeField(torch.roll(self._tensor, shift_amount, dims=mu), self._geom)

    def link(self, mu: int) -> torch.Tensor:
        """Return links in direction μ: shape (T, Z, Y, X, Nc, Nc)."""
        return self._tensor[..., mu, :, :]

    # ------------------------------------------------------------------
    # Gauge-field diagnostics
    # ------------------------------------------------------------------

    def norm2(self) -> float:
        """Frobenius norm squared, summed over all links."""
        return self._tensor.abs().pow(2).sum().item()

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"GaugeField(geom={self._geom}, dtype={self.dtype}, "
            f"device={self.device})"
        )
