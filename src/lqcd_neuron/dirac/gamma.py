"""
Euclidean γ-matrices in the DeGrand-Rossi (chiral) basis.

This is the basis used by QUDA and most modern lattice QCD codes.  The
matrices satisfy the Euclidean Clifford algebra

    {γ_μ, γ_ν} = 2 δ_{μν} I₄              (μ, ν = 0, 1, 2, 3)

with γ_μ Hermitian (γ_μ† = γ_μ) and γ₅ = γ₀γ₁γ₂γ₃.

Direction index convention matching the rest of the library:
    μ = 0 → T, 1 → Z, 2 → Y, 3 → X
"""

from __future__ import annotations

from functools import lru_cache
from typing import Optional

import torch


# ---------------------------------------------------------------------------
# DeGrand-Rossi (chiral) γ-matrices — complex128 "master" values
# ---------------------------------------------------------------------------

_GAMMA_VALUES = [
    # γ₀  (T direction)
    [
        [0, 0, 0, 1j],
        [0, 0, 1j, 0],
        [0, -1j, 0, 0],
        [-1j, 0, 0, 0],
    ],
    # γ₁  (Z direction)
    [
        [0, 0, 0, -1],
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [-1, 0, 0, 0],
    ],
    # γ₂  (Y direction)
    [
        [0, 0, 1j, 0],
        [0, 0, 0, -1j],
        [-1j, 0, 0, 0],
        [0, 1j, 0, 0],
    ],
    # γ₃  (X direction)
    [
        [0, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, 0],
    ],
]


@lru_cache(maxsize=8)
def degrand_rossi_gammas(
    dtype: torch.dtype = torch.complex64,
    device: Optional[str] = None,
) -> torch.Tensor:
    """Return the four Euclidean γ-matrices as a tensor of shape (4, 4, 4).

    Args:
        dtype:  Complex dtype (complex64 or complex128).
        device: Target device string, e.g. ``'cpu'`` or ``'cuda:0'``.

    Returns:
        Tensor ``G`` where ``G[mu]`` is the 4×4 matrix γ_μ.
    """
    dev = torch.device(device) if device else torch.device("cpu")
    G = torch.tensor(_GAMMA_VALUES, dtype=torch.complex128).to(dtype=dtype, device=dev)
    return G  # shape (4, 4, 4)


@lru_cache(maxsize=8)
def gamma5(
    dtype: torch.dtype = torch.complex64,
    device: Optional[str] = None,
) -> torch.Tensor:
    """Return γ₅ = γ₀γ₁γ₂γ₃  (shape 4×4).

    In the DeGrand-Rossi basis γ₅ = diag(−I₂, I₂) (up to a global phase).
    """
    G = degrand_rossi_gammas(dtype=dtype, device=device)
    # γ₅ = γ₀ γ₁ γ₂ γ₃
    g5 = G[0] @ G[1] @ G[2] @ G[3]
    return g5


@lru_cache(maxsize=8)
def sigma_munu(
    dtype: torch.dtype = torch.complex64,
    device: Optional[str] = None,
) -> torch.Tensor:
    """Return all σ_{μν} = (i/2) [γ_μ, γ_ν] for μ < ν.

    Returns:
        Tensor of shape ``(6, 4, 4)`` with the six independent σ_{μν} matrices
        in the order (01, 02, 03, 12, 13, 23).
    """
    G = degrand_rossi_gammas(dtype=dtype, device=device)
    I4 = torch.eye(4, dtype=dtype, device=G.device)
    pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    sigmas = []
    for mu, nu in pairs:
        commutator = G[mu] @ G[nu] - G[nu] @ G[mu]
        sigmas.append(0.5j * commutator)
    return torch.stack(sigmas, dim=0)  # (6, 4, 4)


def spin_project_plus(
    psi: torch.Tensor,
    G: torch.Tensor,
    mu: int,
) -> torch.Tensor:
    """Apply the forward spin projector P⁺_μ = (I + γ_μ) to ψ.

    Args:
        psi:  Spinor field (…, 4, Nc).
        G:    Gamma-matrix tensor of shape (4, 4, 4).
        mu:   Direction index.

    Returns:
        Tensor of shape matching *psi*.
    """
    I4 = torch.eye(4, dtype=G.dtype, device=G.device)
    P = I4 + G[mu]  # (4, 4)
    # einsum: P[a,b] * psi[..., b, c] → result[..., a, c]
    return torch.einsum("ab,...bc->...ac", P, psi)


def spin_project_minus(
    psi: torch.Tensor,
    G: torch.Tensor,
    mu: int,
) -> torch.Tensor:
    """Apply the backward spin projector P⁻_μ = (I − γ_μ) to ψ."""
    I4 = torch.eye(4, dtype=G.dtype, device=G.device)
    P = I4 - G[mu]
    return torch.einsum("ab,...bc->...ac", P, psi)
