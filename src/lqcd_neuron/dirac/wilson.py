"""
Wilson Dirac operator (D_W) and full Wilson fermion matrix (M = D_W + m).

The Wilson Dslash in Euclidean space is

    D_W ψ(x) = (4 + m) ψ(x)
               − ½ Σ_{μ=0}^{3} [ (I − γ_μ) U(x,μ) ψ(x+μ̂)
                                 + (I + γ_μ) U†(x−μ̂,μ) ψ(x−μ̂) ]

where U(x,μ) ∈ SU(Nc) are the link variables and γ_μ are the Euclidean
Dirac matrices in the DeGrand-Rossi (chiral) basis.

Both ``WilsonDslash`` (hopping part only, without the diagonal mass term) and
``WilsonDirac`` (full M = 1 + D_W) inherit from ``torch.nn.Module`` so that
they can be compiled for Neuron with ``torch_neuronx.trace`` or
``torch.compile(backend='neuronx')``.

Tensor-shape conventions
------------------------
    psi   : (T, Z, Y, X, Ns=4, Nc)   — spinor
    U     : (T, Z, Y, X, 4, Nc, Nc)   — gauge field
    output: (T, Z, Y, X, Ns=4, Nc)   — result spinor

Direction mapping (μ → lattice dimension to roll):
    μ = 0 → dim 0 (T)
    μ = 1 → dim 1 (Z)
    μ = 2 → dim 2 (Y)
    μ = 3 → dim 3 (X)
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from .gamma import degrand_rossi_gammas


class WilsonDslash(nn.Module):
    r"""Pure hopping part of the Wilson Dirac operator (without mass diagonal).

        out(x) = −½ Σ_μ [ (I−γ_μ) U(x,μ) ψ(x+μ̂) + (I+γ_μ) U†(x−μ̂,μ) ψ(x−μ̂) ]

    Registers the γ-matrices and projectors as non-trainable buffers so they
    travel automatically with the module during device/dtype casts and are
    visible to the ``torch_neuronx`` tracer.

    Args:
        nc:    Number of colours.
        dtype: Complex dtype.  Should match the input tensors.
    """

    def __init__(
        self,
        nc: int = 3,
        dtype: torch.dtype = torch.complex64,
    ) -> None:
        super().__init__()
        self.nc = nc

        G = degrand_rossi_gammas(dtype=dtype)  # (4, 4, 4)
        I4 = torch.eye(4, dtype=dtype)
        # (4, 4, 4): P_minus[mu] = I − γ_μ,  P_plus[mu] = I + γ_μ
        P_minus = torch.stack([I4 - G[mu] for mu in range(4)], dim=0)
        P_plus  = torch.stack([I4 + G[mu] for mu in range(4)], dim=0)

        self.register_buffer("P_minus", P_minus)
        self.register_buffer("P_plus",  P_plus)

    # ------------------------------------------------------------------
    # Forward pass — Neuron-traceable pure tensor graph
    # ------------------------------------------------------------------

    def forward(self, psi: torch.Tensor, U: torch.Tensor) -> torch.Tensor:
        """Apply the hopping part of the Wilson Dslash.

        Args:
            psi: Spinor field of shape ``(T, Z, Y, X, 4, Nc)``.
            U:   Gauge field  of shape ``(T, Z, Y, X, 4, Nc, Nc)``.

        Returns:
            Result spinor of the same shape as *psi*.
        """
        result = torch.zeros_like(psi)

        for mu in range(4):
            U_mu = U[..., mu, :, :]           # (T,Z,Y,X, Nc, Nc)

            # ---- Forward hop: contribution from ψ(x+μ̂) ---------------
            psi_fwd = torch.roll(psi, -1, dims=mu)
            # Colour matrix-vector multiply: ψ̃(x) = U(x,μ) ψ(x+μ̂)
            # U_mu[...,i,j] × psi_fwd[...,s,j] → (...)si
            Upsi_fwd = torch.einsum("...ij,...sj->...si", U_mu, psi_fwd)
            # Spin projector (I − γ_μ)
            contrib_fwd = torch.einsum("ij,...jk->...ik", self.P_minus[mu], Upsi_fwd)

            # ---- Backward hop: contribution from ψ(x−μ̂) --------------
            psi_bwd = torch.roll(psi, 1, dims=mu)
            # Back-shifted U† : U†(x−μ̂,μ) = conj-transpose of U(x−μ̂,μ)
            U_mu_bwd = torch.roll(U_mu, 1, dims=mu)
            # U†[...,i,j] = conj(U[...,j,i])  →  einsum index swap
            Upsi_bwd = torch.einsum("...ji,...sj->...si", U_mu_bwd.conj(), psi_bwd)
            contrib_bwd = torch.einsum("ij,...jk->...ik", self.P_plus[mu], Upsi_bwd)

            result = result - 0.5 * (contrib_fwd + contrib_bwd)

        return result


class WilsonDirac(nn.Module):
    r"""Full Wilson Dirac matrix M = (4 + m) I + D_hop.

    The Hermitian problem is usually solved in the normal-equation form
    M†M x = M†b, or via the even-odd preconditioned system.

    Args:
        mass:  Bare quark mass *m*.
        nc:    Number of colours.
        dtype: Complex dtype for internal buffers.
    """

    def __init__(
        self,
        mass: float = 0.1,
        nc: int = 3,
        dtype: torch.dtype = torch.complex64,
    ) -> None:
        super().__init__()
        self.mass = mass
        self.nc = nc
        self.hop = WilsonDslash(nc=nc, dtype=dtype)

    def forward(self, psi: torch.Tensor, U: torch.Tensor) -> torch.Tensor:
        """Apply M = (4+m)I + D_hop.

        Args:
            psi: Spinor ``(T, Z, Y, X, 4, Nc)``.
            U:   Gauge field ``(T, Z, Y, X, 4, Nc, Nc)``.

        Returns:
            M ψ, same shape as *psi*.
        """
        return (4.0 + self.mass) * psi + self.hop(psi, U)

    def dagger(self, psi: torch.Tensor, U: torch.Tensor) -> torch.Tensor:
        r"""Apply M† = (4+m)I + D_hop†.

        D_W† is obtained by reversing all hops and swapping projectors
        (γ_μ ↔ −γ_μ), which is equivalent to using −γ_μ in the projectors.
        In practice we negate the result of the hopping term and flip the
        P_minus ↔ P_plus assignment.

        This implementation creates a temporary negated hop module for clarity.
        """
        # D† ψ(x) = -½ Σ_μ [ (I+γ_μ) U(x,μ) ψ(x+μ̂) + (I-γ_μ) U†(x-μ̂,μ) ψ(x-μ̂) ]
        # i.e., swap P_minus and P_plus vs the forward pass.
        result = (4.0 + self.mass) * psi
        U_lat = U
        for mu in range(4):
            U_mu = U_lat[..., mu, :, :]
            psi_fwd = torch.roll(psi, -1, dims=mu)
            Upsi_fwd = torch.einsum("...ij,...sj->...si", U_mu, psi_fwd)
            # Dagger: swap P_minus ↔ P_plus
            contrib_fwd = torch.einsum(
                "ij,...jk->...ik", self.hop.P_plus[mu], Upsi_fwd
            )
            psi_bwd = torch.roll(psi, 1, dims=mu)
            U_mu_bwd = torch.roll(U_mu, 1, dims=mu)
            Upsi_bwd = torch.einsum("...ji,...sj->...si", U_mu_bwd.conj(), psi_bwd)
            contrib_bwd = torch.einsum(
                "ij,...jk->...ik", self.hop.P_minus[mu], Upsi_bwd
            )
            result = result - 0.5 * (contrib_fwd + contrib_bwd)
        return result

    def normal(self, psi: torch.Tensor, U: torch.Tensor) -> torch.Tensor:
        """Apply M†M (for use with CG on the normal equations)."""
        return self.dagger(self.forward(psi, U), U)
