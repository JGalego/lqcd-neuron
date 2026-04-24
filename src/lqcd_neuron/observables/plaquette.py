"""
Plaquette observable and related gauge actions.

The Wilson plaquette

    P = (1/(4 V Nc)) Σ_{x, μ<ν} Re Tr[ U(x,μ) U(x+μ̂,ν) U†(x+ν̂,μ) U†(x,ν) ]

equals 1 for a cold (identity) gauge configuration and decreases towards 0
as the gauge field becomes more disordered (hot / thermalized start).

The Wilson gauge action is

    S_W = β Σ_{x, μ<ν} [ 1 − (1/Nc) Re Tr P_{μν}(x) ]

so 1 − P is the per-plaquette contribution to S_W / β.

All functions accept raw tensors and are Neuron-traceable.
"""

from __future__ import annotations

from typing import Optional

import torch

from ..core.gauge_field import GaugeField


# ---------------------------------------------------------------------------
# Low-level plaquette loop (raw tensors)
# ---------------------------------------------------------------------------

def plaquette_tensor(U: torch.Tensor) -> torch.Tensor:
    """Compute the Re Tr of each elementary plaquette P_{μν}(x).

    Args:
        U: Gauge field tensor ``(T, Z, Y, X, 4, Nc, Nc)``.

    Returns:
        Real tensor of shape ``(T, Z, Y, X, 6)`` with the real trace of
        P_{μν}(x) for each of the 6 direction pairs (01, 02, 03, 12, 13, 23).
    """
    _PAIRS = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    results = []
    for mu, nu in _PAIRS:
        U_mu = U[..., mu, :, :]                    # (…, Nc, Nc)
        U_nu = U[..., nu, :, :]
        U_mu_xpnu = torch.roll(U_mu, -1, dims=nu)  # U(x+ν̂, μ)
        U_nu_xpmu = torch.roll(U_nu, -1, dims=mu)  # U(x+μ̂, ν)

        # P = U_mu @ U_nu_xpmu @ U_mu_xpnu† @ U_nu†
        P = torch.einsum("...ij,...jk->...ik", U_mu, U_nu_xpmu)
        P = torch.einsum("...ij,...kj->...ik", P, U_mu_xpnu.conj())
        P = torch.einsum("...ij,...kj->...ik", P, U_nu.conj())
        # Re Tr
        retr = torch.diagonal(P, dim1=-2, dim2=-1).real.sum(dim=-1)
        results.append(retr)

    return torch.stack(results, dim=-1)  # (T,Z,Y,X, 6)


def plaquette_tensor_real(
    U_re: torch.Tensor, U_im: torch.Tensor
) -> torch.Tensor:
    """Compute Re Tr of each elementary plaquette using pure real arithmetic.

    Equivalent to :func:`plaquette_tensor` but decomposes the SU(N) link
    matrices into real and imaginary parts so that no complex dtype appears
    in the computation graph.  Required for backends that do not support
    complex data types (e.g. AWS Neuron / ``neuronx-cc``).

    The identity used throughout is the real-part-of-trace shortcut: for the
    final matrix product ``C = B @ U_ν†`` we only compute ``C_re`` because
    ``Re Tr[C] = Tr[C_re]``.

    Args:
        U_re: Real part of the gauge field, shape ``(T, Z, Y, X, 4, Nc, Nc)``,
              dtype ``float32``.
        U_im: Imaginary part, same shape and dtype as *U_re*.

    Returns:
        Float tensor of shape ``(T, Z, Y, X, 6)`` — identical semantics to
        :func:`plaquette_tensor`.
    """
    _PAIRS = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    results = []
    for mu, nu in _PAIRS:
        Umr = U_re[..., mu, :, :];     Umi = U_im[..., mu, :, :]
        Unr = U_re[..., nu, :, :];     Uni = U_im[..., nu, :, :]
        Umr_xpnu = torch.roll(Umr, -1, dims=nu)
        Umi_xpnu = torch.roll(Umi, -1, dims=nu)
        Unr_xpmu = torch.roll(Unr, -1, dims=mu)
        Uni_xpmu = torch.roll(Uni, -1, dims=mu)

        # A = U_mu @ U_nu_xpmu  (regular complex matmul in real arithmetic)
        Ar = Umr @ Unr_xpmu - Umi @ Uni_xpmu
        Ai = Umr @ Uni_xpmu + Umi @ Unr_xpmu

        # B = A @ U_mu_xpnu†  (B[i,k] = sum_j A[i,j] * conj(U_mu_xpnu[k,j]))
        Ut_r = Umr_xpnu.transpose(-2, -1)
        Ut_i = Umi_xpnu.transpose(-2, -1)
        Br = Ar @ Ut_r + Ai @ Ut_i
        Bi = Ai @ Ut_r - Ar @ Ut_i

        # C_re only: C = B @ U_nu†  — Re Tr[C] = Tr[C_re]
        Cr = Br @ Unr.transpose(-2, -1) + Bi @ Uni.transpose(-2, -1)

        retr = torch.diagonal(Cr, dim1=-2, dim2=-1).sum(dim=-1)
        results.append(retr)

    return torch.stack(results, dim=-1)  # (T,Z,Y,X, 6)


def plaquette(U_or_field) -> float:
    """Compute the average plaquette  P ∈ [0, 1].

    Args:
        U_or_field: Either a raw gauge tensor ``(T,Z,Y,X,4,Nc,Nc)`` or a
                    :class:`~lqcd_neuron.core.GaugeField` instance.

    Returns:
        Average plaquette as a Python float.  For a cold (unit) gauge
        configuration this equals 1.0.
    """
    U = U_or_field.tensor if isinstance(U_or_field, GaugeField) else U_or_field
    nc = U.shape[-1]
    ptr = plaquette_tensor(U)           # (T,Z,Y,X,6)
    return ptr.mean().item() / nc       # normalise by Nc


def wilson_action(U_or_field, beta: float) -> float:
    """Compute the Wilson gauge action S_W = β Σ_{x,μ<ν}[1 − (1/Nc) Re Tr P].

    Args:
        U_or_field: Gauge tensor or :class:`GaugeField`.
        beta:       Inverse coupling β = 2 Nc / g².

    Returns:
        Total action S_W as a Python float.
    """
    U = U_or_field.tensor if isinstance(U_or_field, GaugeField) else U_or_field
    nc = U.shape[-1]
    ptr = plaquette_tensor(U)               # (T,Z,Y,X,6)
    return (beta * (1.0 - ptr / nc)).sum().item()


def topological_charge(U_or_field) -> float:
    """Approximate topological charge Q via the clover definition.

        Q = (1/32π²) Σ_{x} ε_{μνρσ} Tr[ F_{μν}(x) F_{ρσ}(x) ]

    Uses the 4-leaf clover estimate of F_{μν} already computed in
    :mod:`lqcd_neuron.dirac.clover`.

    Returns:
        Q as a Python float (not integer-quantised for finite lattice
        spacing; approaches an integer in the continuum limit).
    """
    from ..dirac.clover import compute_field_strength

    U = U_or_field.tensor if isinstance(U_or_field, GaugeField) else U_or_field
    F = compute_field_strength(U)  # (T,Z,Y,X, 6, Nc, Nc)

    # Map (01,02,03,12,13,23) ↔ indices 0-5
    # ε_{0123}=+1: F01*F23 + F02*F31 + F03*F12
    # Q = (1/32π²) Tr ε_{μνρσ} F_{μν} F_{ρσ}
    # = (2/16π²) Σ_{μ<ν, ρ<σ, ε=+1} Tr[ F_{μν} F_{ρσ} ]
    # Non-zero ε combinations: (01,23)→+1, (02,13)→-1, (03,12)→+1
    F01, F02, F03, F12, F13, F23 = [F[..., i, :, :] for i in range(6)]

    def tr_prod(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return torch.einsum("...ij,...ji->...", A, B).real

    q_density = (
        tr_prod(F01, F23)
        - tr_prod(F02, F13)
        + tr_prod(F03, F12)
    )
    import math
    prefactor = 1.0 / (8.0 * math.pi ** 2)
    return (prefactor * q_density).sum().item()
