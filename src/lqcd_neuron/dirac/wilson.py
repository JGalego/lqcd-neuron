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

from typing import Optional, Tuple

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
            # Negative dim indexing so an optional leading batch (multi-RHS)
            # dimension does not shift the lattice axes.  T,Z,Y,X always sit
            # at positions -6..-3 of psi and U_mu.
            psi_fwd = torch.roll(psi, -1, dims=mu - 6)
            # Colour matrix-vector multiply: ψ̃(x) = U(x,μ) ψ(x+μ̂)
            # U_mu[...,i,j] × psi_fwd[...,s,j] → (...)si
            Upsi_fwd = torch.einsum("...ij,...sj->...si", U_mu, psi_fwd)
            # Spin projector (I − γ_μ)
            contrib_fwd = torch.einsum("ij,...jk->...ik", self.P_minus[mu], Upsi_fwd)

            # ---- Backward hop: contribution from ψ(x−μ̂) --------------
            psi_bwd = torch.roll(psi, 1, dims=mu - 6)
            # Back-shifted U† : U†(x−μ̂,μ) = conj-transpose of U(x−μ̂,μ)
            U_mu_bwd = torch.roll(U_mu, 1, dims=mu - 6)
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
            psi_fwd = torch.roll(psi, -1, dims=mu - 6)
            Upsi_fwd = torch.einsum("...ij,...sj->...si", U_mu, psi_fwd)
            # Dagger: swap P_minus ↔ P_plus
            contrib_fwd = torch.einsum(
                "ij,...jk->...ik", self.hop.P_plus[mu], Upsi_fwd
            )
            psi_bwd = torch.roll(psi, 1, dims=mu - 6)
            U_mu_bwd = torch.roll(U_mu, 1, dims=mu - 6)
            Upsi_bwd = torch.einsum("...ji,...sj->...si", U_mu_bwd.conj(), psi_bwd)
            contrib_bwd = torch.einsum(
                "ij,...jk->...ik", self.hop.P_minus[mu], Upsi_bwd
            )
            result = result - 0.5 * (contrib_fwd + contrib_bwd)
        return result

    def normal(self, psi: torch.Tensor, U: torch.Tensor) -> torch.Tensor:
        """Apply M†M (for use with CG on the normal equations)."""
        return self.dagger(self.forward(psi, U), U)


# ---------------------------------------------------------------------------
# Real-arithmetic adapters for AWS Neuron (NCC_EVRF004 workaround)
# ---------------------------------------------------------------------------
# neuronx-cc does not support complex dtypes.  These adapters accept the real
# and imaginary parts of psi/U as separate float32 tensors, perform identical
# arithmetic using only real ops, and return (result_re, result_im).
# A _ComplexDslashWrapper in compiler.py splits/re-joins at the host boundary
# so callers keep the standard complex64 interface.
# ---------------------------------------------------------------------------

class _NeuronWilsonDslashAdapter(nn.Module):
    """Pure float32 Wilson hopping term for Neuron (NCC_EVRF004 workaround).

    Inputs: ``(psi_re, psi_im, U_re, U_im)`` — float32 tensors.
    Returns: ``(result_re, result_im)`` — float32 tensors.
    """

    def __init__(self, nc: int = 3) -> None:
        super().__init__()
        G  = degrand_rossi_gammas(dtype=torch.complex64)  # (4,4,4)
        I4 = torch.eye(4, dtype=torch.complex64)
        P_minus = torch.stack([I4 - G[mu] for mu in range(4)], dim=0)
        P_plus  = torch.stack([I4 + G[mu] for mu in range(4)], dim=0)
        self.register_buffer("P_minus_re", P_minus.real.float())
        self.register_buffer("P_minus_im", P_minus.imag.float())
        self.register_buffer("P_plus_re",  P_plus.real.float())
        self.register_buffer("P_plus_im",  P_plus.imag.float())

    @staticmethod
    def _color_mv(
        U_re: torch.Tensor, U_im: torch.Tensor,
        v_re: torch.Tensor, v_im: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Complex colour matmul: U @ v for each spin component.

        U: (..., Nc, Nc), v: (..., Ns, Nc) → (..., Ns, Nc).
        Equiv: ``einsum("...ij,...sj->...si", U, v)`` but float32.
        """
        r_re = (torch.einsum("...ij,...sj->...si", U_re, v_re)
                - torch.einsum("...ij,...sj->...si", U_im, v_im))
        r_im = (torch.einsum("...ij,...sj->...si", U_re, v_im)
                + torch.einsum("...ij,...sj->...si", U_im, v_re))
        return r_re, r_im

    @staticmethod
    def _color_dag_mv(
        U_re: torch.Tensor, U_im: torch.Tensor,
        v_re: torch.Tensor, v_im: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Complex colour matmul: U† @ v for each spin component.

        Equiv: ``einsum("...ji,...sj->...si", U.conj(), v)`` but float32.
        (U†)[i,j] = conj(U[j,i]) = U_re[j,i] − i U_im[j,i].
        """
        r_re = (torch.einsum("...ji,...sj->...si", U_re, v_re)
                + torch.einsum("...ji,...sj->...si", U_im, v_im))
        r_im = (torch.einsum("...ji,...sj->...si", U_re, v_im)
                - torch.einsum("...ji,...sj->...si", U_im, v_re))
        return r_re, r_im

    @staticmethod
    def _spin_mv(
        P_re: torch.Tensor, P_im: torch.Tensor,
        v_re: torch.Tensor, v_im: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Complex spin-projector matmul: P @ v.

        P_re/im: (4, 4) — spin; v_re/im: (..., 4, Nc) → (..., 4, Nc).
        Equiv: ``einsum("ij,...jk->...ik", P, v)`` but float32.
        """
        r_re = (torch.einsum("ij,...jk->...ik", P_re, v_re)
                - torch.einsum("ij,...jk->...ik", P_im, v_im))
        r_im = (torch.einsum("ij,...jk->...ik", P_re, v_im)
                + torch.einsum("ij,...jk->...ik", P_im, v_re))
        return r_re, r_im

    def forward(
        self,
        psi_re: torch.Tensor, psi_im: torch.Tensor,
        U_re: torch.Tensor, U_im: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        result_re = torch.zeros_like(psi_re)
        result_im = torch.zeros_like(psi_im)

        for mu in range(4):
            U_mu_re = U_re[..., mu, :, :]
            U_mu_im = U_im[..., mu, :, :]

            # Negative dim indexing: lattice axes T,Z,Y,X live at -6..-3
            # of both psi (..., T,Z,Y,X, Ns, Nc) and U_mu (..., T,Z,Y,X, Nc, Nc),
            # so an optional leading batch (multi-RHS) dim does not shift them.
            ldim = mu - 6

            # Forward hop: − ½ (I−γ_μ) U(x,μ) ψ(x+μ̂)
            pf_re = torch.roll(psi_re, -1, dims=ldim)
            pf_im = torch.roll(psi_im, -1, dims=ldim)
            Upf_re, Upf_im = self._color_mv(U_mu_re, U_mu_im, pf_re, pf_im)
            cf_re, cf_im = self._spin_mv(
                self.P_minus_re[mu], self.P_minus_im[mu], Upf_re, Upf_im
            )

            # Backward hop: − ½ (I+γ_μ) U†(x−μ̂,μ) ψ(x−μ̂)
            pb_re = torch.roll(psi_re, 1, dims=ldim)
            pb_im = torch.roll(psi_im, 1, dims=ldim)
            Ub_re = torch.roll(U_mu_re, 1, dims=ldim)
            Ub_im = torch.roll(U_mu_im, 1, dims=ldim)
            Upb_re, Upb_im = self._color_dag_mv(Ub_re, Ub_im, pb_re, pb_im)
            cb_re, cb_im = self._spin_mv(
                self.P_plus_re[mu], self.P_plus_im[mu], Upb_re, Upb_im
            )

            result_re = result_re - 0.5 * (cf_re + cb_re)
            result_im = result_im - 0.5 * (cf_im + cb_im)

        return result_re, result_im


# ---------------------------------------------------------------------------
# Even-odd (checkerboard) Wilson hop — CPU reference implementation
# ---------------------------------------------------------------------------

class EvenOddWilsonDslash(nn.Module):
    r"""Even-odd decomposition of the Wilson hopping operator.

    The full lattice is split into two V/2-site sublattices:
        even (p=0): (t+z+y+x) % 2 == 0
        odd  (p=1): (t+z+y+x) % 2 == 1

    The nearest-neighbour hop connects even ↔ odd sites exclusively, so::

        D_hop = [ 0     D_eo ]
                [ D_oe  0   ]

    This module provides :meth:`hop_oe` and :meth:`hop_eo` on **full-lattice**
    spinors and gauge fields (shape ``(T, Z, Y, X, Ns, Nc)`` and
    ``(T, Z, Y, X, 4, Nc, Nc)`` respectively).  Each method applies the
    hopping term only at the *output* parity sites and zeros the other half.

    For Neuron half-lattice compilation (which halves on-chip memory), use
    :meth:`NeuronCompiler.compile_dslash_eo` together with
    :func:`~lqcd_neuron.neuron.compiler.pack_checkerboard` /
    :func:`~lqcd_neuron.neuron.compiler.unpack_checkerboard`.

    Args:
        mass:  Bare quark mass.  Used only for the diagonal in the full Dirac
               form ``M = (4+m)I + D_hop``.  Pass 0.0 for the bare hop.
        nc:    Number of colours.
        dtype: Complex dtype for internal gamma-matrix buffers.
    """

    def __init__(
        self,
        mass: float = 0.0,
        nc: int = 3,
        dtype: torch.dtype = torch.complex64,
    ) -> None:
        super().__init__()
        self.mass = mass
        self.nc = nc

        G  = degrand_rossi_gammas(dtype=dtype)
        I4 = torch.eye(4, dtype=dtype)
        P_minus = torch.stack([I4 - G[mu] for mu in range(4)], dim=0)
        P_plus  = torch.stack([I4 + G[mu] for mu in range(4)], dim=0)
        self.register_buffer("P_minus", P_minus)
        self.register_buffer("P_plus",  P_plus)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parity_mask(
        T: int, Z: int, Y: int, X: int, parity: int, device: torch.device
    ) -> torch.Tensor:
        """Boolean mask of shape (T, Z, Y, X) — True at *parity* sites."""
        t = torch.arange(T, device=device).view(T, 1, 1, 1)
        z = torch.arange(Z, device=device).view(1, Z, 1, 1)
        y = torch.arange(Y, device=device).view(1, 1, Y, 1)
        x = torch.arange(X, device=device).view(1, 1, 1, X)
        return ((t + z + y + x) % 2) == parity

    def _hop(
        self,
        psi: torch.Tensor,
        U: torch.Tensor,
        in_parity: int,
        out_parity: int,
    ) -> torch.Tensor:
        """Apply the hopping term: output at *out_parity*, input at *in_parity*.

        Equivalent to zeroing psi at out-parity sites before the hop and
        zeroing the result at in-parity sites after.
        """
        T, Z, Y, X = psi.shape[:4]
        diag = (4.0 + self.mass) if out_parity == in_parity else 0.0
        result = diag * psi

        # Mask to zero out input on the wrong parity sites.
        in_mask = self._parity_mask(T, Z, Y, X, in_parity, psi.device)
        psi_in = psi * in_mask.view(T, Z, Y, X, 1, 1).to(psi.dtype)

        for mu in range(4):
            U_mu = U[..., mu, :, :]
            # Forward hop
            psi_fwd = torch.roll(psi_in, -1, dims=mu - 6)
            Upsi_fwd = torch.einsum("...ij,...sj->...si", U_mu, psi_fwd)
            contrib_fwd = torch.einsum("ij,...jk->...ik", self.P_minus[mu], Upsi_fwd)
            # Backward hop
            psi_bwd = torch.roll(psi_in,  1, dims=mu - 6)
            U_mu_bwd = torch.roll(U_mu, 1, dims=mu - 6)
            Upsi_bwd = torch.einsum("...ji,...sj->...si", U_mu_bwd.conj(), psi_bwd)
            contrib_bwd = torch.einsum("ij,...jk->...ik", self.P_plus[mu],  Upsi_bwd)
            result = result - 0.5 * (contrib_fwd + contrib_bwd)

        # Zero the result at in-parity sites (output only at out_parity).
        out_mask = self._parity_mask(T, Z, Y, X, out_parity, psi.device)
        return result * out_mask.view(T, Z, Y, X, 1, 1).to(psi.dtype)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def hop_oe(self, psi: torch.Tensor, U: torch.Tensor) -> torch.Tensor:
        """D_oe ψ: output at **odd** sites, input from **even** sites."""
        return self._hop(psi, U, in_parity=0, out_parity=1)

    def hop_eo(self, psi: torch.Tensor, U: torch.Tensor) -> torch.Tensor:
        """D_eo ψ: output at **even** sites, input from **odd** sites."""
        return self._hop(psi, U, in_parity=1, out_parity=0)

    def forward(self, psi: torch.Tensor, U: torch.Tensor) -> torch.Tensor:
        """Full D_hop = D_oe + D_eo (all sites).  Equivalent to WilsonDslash."""
        return self.hop_oe(psi, U) + self.hop_eo(psi, U)


class _NeuronWilsonDiracAdapter(nn.Module):
    """Pure float32 Wilson Dirac operator M = (4+m)I + D_hop for Neuron.

    Inputs: ``(psi_re, psi_im, U_re, U_im)`` — float32 tensors.
    Returns: ``(result_re, result_im)`` — float32 tensors.
    """

    def __init__(self, mass: float = 0.1, nc: int = 3) -> None:
        super().__init__()
        self.diag = 4.0 + mass
        self.hop  = _NeuronWilsonDslashAdapter(nc=nc)

    def forward(
        self,
        psi_re: torch.Tensor, psi_im: torch.Tensor,
        U_re: torch.Tensor, U_im: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        hop_re, hop_im = self.hop(psi_re, psi_im, U_re, U_im)
        return self.diag * psi_re + hop_re, self.diag * psi_im + hop_im
