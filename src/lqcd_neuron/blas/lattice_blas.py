"""
Lattice BLAS — inner products, norms, and vector operations on spinor fields.

All functions operate directly on plain ``torch.Tensor`` objects (not on
``ColorSpinorField`` wrappers) so they are JIT-scriptable and traceable by
``torch_neuronx``.

Conventions
-----------
  • Complex inner product: ⟨x|y⟩ = Σ_i x_i* y_i  (sesquilinear in first arg)
  • Norm²: ‖x‖² = ⟨x|x⟩  (real, non-negative)
  • axpy:  y += a·x
  • xpay:  x += a·y  (alias with swapped roles, convenient in CG)
"""

from __future__ import annotations

import torch


# ---------------------------------------------------------------------------
# Inner products and norms
# ---------------------------------------------------------------------------

def inner(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Hermitian inner product ⟨x|y⟩ = (x†·y) summed over all indices.

    Returns a complex scalar tensor.
    """
    return (x.conj() * y).sum()


def norm2(x: torch.Tensor) -> torch.Tensor:
    """Squared L² norm ‖x‖² = ⟨x|x⟩.  Returns a real scalar tensor."""
    return x.abs().pow(2).sum()


def norm(x: torch.Tensor) -> torch.Tensor:
    """L² norm ‖x‖.  Returns a real scalar tensor."""
    return norm2(x).sqrt()


# ---------------------------------------------------------------------------
# Level-1 BLAS
# ---------------------------------------------------------------------------

def axpy(a: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Return y + a·x  (does NOT modify y in-place)."""
    return y + a * x


def xpay(x: torch.Tensor, a: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Return x + a·y  (does NOT modify x in-place)."""
    return x + a * y


def axpby(
    a: torch.Tensor,
    x: torch.Tensor,
    b: torch.Tensor,
    y: torch.Tensor,
) -> torch.Tensor:
    """Return a·x + b·y."""
    return a * x + b * y


def caxpby(
    a: complex,
    x: torch.Tensor,
    b: complex,
    y: torch.Tensor,
) -> torch.Tensor:
    """Return a·x + b·y with Python complex scalars *a*, *b*."""
    return a * x + b * y


# ---------------------------------------------------------------------------
# Reduction helpers (multi-vector)
# ---------------------------------------------------------------------------

def dot_many(
    basis: torch.Tensor,
    v: torch.Tensor,
) -> torch.Tensor:
    """Compute ⟨basis_i | v⟩ for all vectors in *basis*.

    Args:
        basis: Tensor of shape ``(k, *field_shape)`` with *k* vectors.
        v:     Tensor of shape ``(*field_shape)``.

    Returns:
        Complex tensor of shape ``(k,)``.
    """
    # Flatten field dims for einsum efficiency
    k = basis.shape[0]
    flat_basis = basis.reshape(k, -1)
    flat_v = v.reshape(-1)
    return torch.mv(flat_basis.conj(), flat_v)
