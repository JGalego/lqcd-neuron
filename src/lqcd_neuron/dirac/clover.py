"""
Sheikholeslami-Wohlert (clover / SW) improvement term and the full
Clover-Wilson Dirac operator.

The clover term corrects O(a) artefacts in the Wilson formulation:

    D_CW = D_W + (c_SW / 4) Σ_{μ<ν} σ_{μν} F̂_{μν}

where:
  • D_W        — Wilson Dirac operator (see ``wilson.py``)
  • c_SW       — Sheikholeslami-Wohlert coefficient
  • σ_{μν}     — (i/2)[γ_μ, γ_ν]  (antisymmetric 4×4 spin matrix)
  • F̂_{μν}(x) — Hermitian field-strength tensor built from the
                 4-leaf clover plaquette

Field-strength tensor via the 4-leaf clover
-------------------------------------------
For each pair μ < ν define four elementary plaquettes meeting at x:

    Q_{μν}(x) = Σ_{(s,t) ∈ corners} [plaquette contribution]

The standard lattice field-strength is

    F̂_{μν}(x) = (1/8i) [ Q_{μν}(x) − Q_{μν}†(x) ]

which makes F̂ Hermitian (traceless only for SU(Nc) gauge theory).

The clover matrix at site x is the 12×12 Hermitian matrix acting on the
combined spin⊗colour space:

    C(x) = I₁₂ + (c_SW/4) Σ_{μ<ν} σ_{μν} ⊗ F̂_{μν}(x)

Tensor shapes
-------------
    U       : (T, Z, Y, X, 4, Nc, Nc)
    F_hat   : (T, Z, Y, X, 6, Nc, Nc)   — packed (01,02,03,12,13,23)
    clover  : (T, Z, Y, X, 4, 4, Nc, Nc) — C[..., α, β, a, b]
              or equivalently reshaped to (T,Z,Y,X, Ns*Nc, Ns*Nc)
    psi     : (T, Z, Y, X, 4, Nc)
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from .gamma import degrand_rossi_gammas, sigma_munu
from .wilson import WilsonDirac


# ---------------------------------------------------------------------------
# Four-leaf clover: field-strength tensor
# ---------------------------------------------------------------------------

_PAIRS = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]


def _plaquette_loop(U: torch.Tensor, mu: int, nu: int) -> torch.Tensor:
    """Compute the *standard* μ-ν plaquette at each site.

        P(x) = U(x,μ) U(x+μ̂,ν) U†(x+ν̂,μ) U†(x,ν)

    Args:
        U:   Gauge field ``(T,Z,Y,X, 4, Nc, Nc)``.
        mu:  First direction (0–3).
        nu:  Second direction (0–3).

    Returns:
        Tensor of shape ``(T,Z,Y,X, Nc, Nc)``.
    """
    U_mu = U[..., mu, :, :]                           # (…, Nc, Nc)
    U_nu = U[..., nu, :, :]
    U_mu_shift_nu = torch.roll(U_mu, -1, dims=nu)     # U(x+ν̂, μ)
    U_nu_shift_mu = torch.roll(U_nu, -1, dims=mu)     # U(x+μ̂, ν)

    # P = U_mu @ U_nu_shift_mu @ U_mu_shift_nu† @ U_nu†
    P = torch.einsum("...ij,...jk->...ik", U_mu, U_nu_shift_mu)
    P = torch.einsum("...ij,...kj->...ik", P, U_mu_shift_nu.conj())
    P = torch.einsum("...ij,...kj->...ik", P, U_nu.conj())
    return P


def compute_field_strength(U: torch.Tensor) -> torch.Tensor:
    """Compute the Hermitian field-strength tensor F̂_{μν} via the 4-leaf clover.

    For each pair (μ,ν) with μ < ν we sum the four plaquette-loops that share
    the corner at x:

        Q_{μν}(x) = P_{μν}(x)
                  + P_{νμ̄}(x)   [plaquette going -μ, +ν]
                  + P_{μ̄ν̄}(x)  [plaquette going -μ, -ν]
                  + P_{νμ̄}(x)  ... (all four corners)

    then antisymmetrise:

        F̂_{μν}(x) = (1/8i) [ Q_{μν}(x) − Q_{μν}†(x) ]

    Args:
        U: Gauge field ``(T, Z, Y, X, 4, Nc, Nc)``.

    Returns:
        F_hat of shape ``(T, Z, Y, X, 6, Nc, Nc)``, with the six (μ,ν)
        pairs ordered as (01, 02, 03, 12, 13, 23).
    """
    results = []
    for mu, nu in _PAIRS:
        # Leaf 1: standard plaquette P(x; +μ, +ν)
        Q = _plaquette_loop(U, mu, nu)

        # Leaf 2: plaquette P(x; +ν, −μ)
        U_shifted = torch.roll(U, 1, dims=mu)       # U at x−μ̂
        leaf2 = _plaquette_loop(U_shifted, nu, mu)
        # The leaf starts at x, goes +ν, -μ, -ν, +μ
        # → shift leaf2 back to align at x
        Q = Q + torch.roll(leaf2, -1, dims=nu)

        # Leaf 3: plaquette P(x; −μ, −ν)  ↔ plaquette P(x−μ̂−ν̂; +μ, +ν)†
        U_shifted2 = torch.roll(U_shifted, 1, dims=nu)   # U at x−μ̂−ν̂
        leaf3 = _plaquette_loop(U_shifted2, mu, nu)
        Q = Q + torch.roll(torch.roll(leaf3, 1, dims=mu), 1, dims=nu).conj().transpose(-1, -2)

        # Leaf 4: plaquette P(x; −ν, +μ)  ↔ shift variant
        U_shifted3 = torch.roll(U, 1, dims=nu)
        leaf4 = _plaquette_loop(U_shifted3, mu, nu)
        Q = Q + torch.roll(leaf4, 1, dims=nu).conj().transpose(-1, -2)

        # Antisymmetrise and normalise
        F = (Q - Q.conj().transpose(-1, -2)) / (8.0j)
        results.append(F)

    return torch.stack(results, dim=-3)  # (..., 6, Nc, Nc)


def compute_clover(
    U: torch.Tensor,
    csw: float,
    dtype: torch.dtype = torch.complex64,
) -> torch.Tensor:
    """Build the Clover matrix C(x) = I₁₂ + (c_SW/4) Σ_{μ<ν} σ_{μν} ⊗ F̂_{μν}.

    Args:
        U:    Gauge field ``(T, Z, Y, X, 4, Nc, Nc)``.
        csw:  SW coefficient.
        dtype: Target dtype.

    Returns:
        Clover matrix of shape ``(T, Z, Y, X, Ns=4, Nc, Ns=4, Nc)``.

        - Indices: ``C[..., α, a, β, b]``  where α,β are spin and a,b are colour.
        - Equivalent to the ``(Ns·Nc) × (Ns·Nc)`` Hermitian matrix at each site.
    """
    U_c = U.to(dtype=dtype)
    nc = U_c.shape[-1]
    ns = 4
    site_shape = U_c.shape[:4]         # (T, Z, Y, X)

    F_hat = compute_field_strength(U_c)  # (T,Z,Y,X, 6, Nc, Nc)
    sig = sigma_munu(dtype=dtype, device=str(U_c.device))  # (6, 4, 4)

    # Clover term = Σ_{pairs} σ_{μν} ⊗ F̂_{μν}
    # sig shape  : (6, ns, ns)
    # F_hat shape: (..., 6, nc, nc)
    # result[..., α, a, β, b] = Σ_p sig[p, α, β] * F_hat[..., p, a, b]
    clover_term = torch.einsum("pab,...pij->...aibj", sig, F_hat)
    # → shape (..., ns, nc, ns, nc)

    # Add identity
    I_spin  = torch.eye(ns, dtype=dtype, device=U_c.device)
    I_color = torch.eye(nc, dtype=dtype, device=U_c.device)
    # I₁₂ in the same index order (..., ns, nc, ns, nc)
    I12 = (I_spin.unsqueeze(-1).unsqueeze(-1) * I_color.unsqueeze(0).unsqueeze(0))
    # I_spin: (ns,ns) → unsqueeze → (ns,1,ns,1)
    # I_color:(nc,nc) → unsqueeze → (1,nc,1,nc)  ← broadcast
    I12 = I_spin.unsqueeze(1).unsqueeze(3) * I_color.unsqueeze(0).unsqueeze(2)
    # shape: (ns, nc, ns, nc) — broadcast over batch dims below
    I12 = I12.expand(*site_shape, ns, nc, ns, nc)

    return I12 + (csw / 4.0) * clover_term


# ---------------------------------------------------------------------------
# Clover-Wilson Dirac operator
# ---------------------------------------------------------------------------

class CloverWilsonDirac(nn.Module):
    r"""Clover-improved Wilson Dirac operator D_CW = D_W + c_SW·Δ_clover.

    The clover matrix C(x) is pre-computed from the gauge field before the
    first forward pass via :meth:`set_gauge`.  This mirrors QUDA's pattern
    of separating the expensive clover setup from repeated solves.

    Args:
        mass: Bare quark mass *m*.
        csw:  Sheikholeslami-Wohlert coefficient.
        nc:   Number of colours.
        dtype: Complex dtype for buffers.
    """

    def __init__(
        self,
        mass: float = 0.1,
        csw: float = 1.0,
        nc: int = 3,
        dtype: torch.dtype = torch.complex64,
    ) -> None:
        super().__init__()
        self.mass = mass
        self.csw = csw
        self.nc = nc
        self._dtype = dtype
        self.wilson = WilsonDirac(mass=mass, nc=nc, dtype=dtype)
        # Clover matrix is not a parameter — register as buffer (None until set)
        self.register_buffer("_clover", None)

    # ------------------------------------------------------------------

    def set_gauge(self, U: torch.Tensor) -> None:
        """Pre-compute and cache the clover matrix from gauge field *U*.

        Must be called before :meth:`forward`.

        Args:
            U: Gauge field ``(T, Z, Y, X, 4, Nc, Nc)``.
        """
        clv = compute_clover(U, self.csw, dtype=self._dtype)
        # Store as (T,Z,Y,X, Ns*Nc, Ns*Nc) for efficient matmul
        T, Z, Y, X, ns, nc, ns2, nc2 = clv.shape
        assert ns == ns2 and nc == nc2
        dim = ns * nc
        self._clover = clv.reshape(T, Z, Y, X, dim, dim)

    # ------------------------------------------------------------------

    def _apply_clover(self, psi: torch.Tensor) -> torch.Tensor:
        """Apply the cached clover matrix to spinor *psi*.

        Args:
            psi: Spinor ``(T, Z, Y, X, Ns, Nc)``.

        Returns:
            C·ψ, same shape.
        """
        if self._clover is None:
            raise RuntimeError("Call set_gauge(U) before forward().")

        T, Z, Y, X, ns, nc = psi.shape
        dim = ns * nc
        # Flatten spin-colour, apply clover matrix, unflatten
        psi_flat = psi.reshape(T, Z, Y, X, dim, 1)                    # (..., dim, 1)
        result_flat = torch.matmul(self._clover, psi_flat)             # (..., dim, 1)
        return result_flat.reshape(T, Z, Y, X, ns, nc)

    # ------------------------------------------------------------------

    def forward(self, psi: torch.Tensor, U: torch.Tensor) -> torch.Tensor:
        """Apply D_CW = D_W ψ + C ψ − I ψ  (the I comes from the identity in C).

        Equivalently ``D_CW ψ = D_hop ψ + C ψ`` because D_W already contains
        the ``(4+m)I`` diagonal and C contains ``I + clover_term``, so we
        compute ``D_hop ψ + C ψ`` (adding (4+m)ψ is folded into D_W forward).

        Args:
            psi: Spinor ``(T, Z, Y, X, 4, Nc)``.
            U:   Gauge field ``(T, Z, Y, X, 4, Nc, Nc)``.
        """
        return self.wilson.hop(psi, U) + self._apply_clover(psi)

    def normal(self, psi: torch.Tensor, U: torch.Tensor) -> torch.Tensor:
        """Apply D†D (for normal-equation CG)."""
        return self._dagger(self.forward(psi, U), U)

    def _dagger(self, psi: torch.Tensor, U: torch.Tensor) -> torch.Tensor:
        """Apply D†_CW.  Since C is Hermitian, D†_CW = D†_W + C†  = D†_W + C."""
        return self.wilson.dagger(psi, U) - (4.0 + self.mass) * psi + self._apply_clover(psi)
