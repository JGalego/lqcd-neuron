"""
Polyakov loop observable.

The Polyakov loop measures the free energy of a static quark.  For a
lattice with temporal extent T it is defined as

    L(x⃗) = (1/Nc) Tr[ Π_{t=0}^{T−1} U(t, x⃗, μ=0) ]

where μ=0 is the temporal direction (dim 0 in our storage convention).

The spatially averaged Polyakov loop

    <L> = (1/V₃) Σ_{x⃗} L(x⃗)

is an order parameter for the deconfinement phase transition; it
vanishes (by Z(Nc) symmetry) in the confined phase.
"""

from __future__ import annotations

import torch

from ..core.gauge_field import GaugeField


def polyakov_loop(U_or_field, mu: int = 0) -> torch.Tensor:
    """Compute the spatial average of the Polyakov loop in direction *mu*.

    Args:
        U_or_field: Raw gauge tensor ``(T, Z, Y, X, 4, Nc, Nc)`` or a
                    :class:`~lqcd_neuron.core.GaugeField`.
        mu:         Direction to wind around (0 = temporal T-direction).

    Returns:
        Complex scalar tensor: average Polyakov loop ⟨L⟩.
    """
    U = U_or_field.tensor if isinstance(U_or_field, GaugeField) else U_or_field
    T = U.shape[mu]
    nc = U.shape[-1]

    # Product of temporal links along the t-axis at each spatial site
    # Start with identity at each (x,y,z) site and accumulate product in t
    U_mu_0 = U.select(4, mu)  # (T, Z, Y, X, Nc, Nc) — links in dir mu

    # Initialize with the t=0 slice
    P = U_mu_0.select(mu, 0).clone()  # (Z, Y, X, Nc, Nc)  [or T-complement]

    # For loops in the T direction (mu=0), iterate over t slices
    for t in range(1, T):
        # Select the t-th temporal slice
        if mu == 0:
            P = torch.einsum("...ij,...jk->...ik", P, U_mu_0[t])
        elif mu == 1:
            P = torch.einsum("...ij,...jk->...ik", P, U_mu_0[:, t, :, :, :])
        elif mu == 2:
            P = torch.einsum("...ij,...jk->...ik", P, U_mu_0[:, :, t, :, :])
        else:
            P = torch.einsum("...ij,...jk->...ik", P, U_mu_0[:, :, :, t, :])

    # Trace over colour
    trace = torch.diagonal(P, dim1=-2, dim2=-1).sum(dim=-1)  # spatial vol
    return (trace / nc).mean()


def polyakov_loop_spatially_resolved(U_or_field, mu: int = 0) -> torch.Tensor:
    """Return the Polyakov loop L(x⃗) at every spatial site (not averaged).

    Args:
        U_or_field: Gauge tensor or :class:`GaugeField`.
        mu:         Winding direction.

    Returns:
        Complex tensor of shape ``(Z, Y, X)`` (if mu=0) with a complex
        scalar at each spatial site.
    """
    U = U_or_field.tensor if isinstance(U_or_field, GaugeField) else U_or_field
    T = U.shape[mu]
    nc = U.shape[-1]

    U_mu_0 = U.select(4, mu)   # (T, Z, Y, X, Nc, Nc)

    if mu == 0:
        P = U_mu_0[0].clone()
        for t in range(1, T):
            P = torch.einsum("...ij,...jk->...ik", P, U_mu_0[t])
    elif mu == 1:
        P = U_mu_0[:, 0].clone()
        for z in range(1, T):
            P = torch.einsum("...ij,...jk->...ik", P, U_mu_0[:, z])
    elif mu == 2:
        P = U_mu_0[:, :, 0].clone()
        for y in range(1, T):
            P = torch.einsum("...ij,...jk->...ik", P, U_mu_0[:, :, y])
    else:
        P = U_mu_0[:, :, :, 0].clone()
        for x in range(1, T):
            P = torch.einsum("...ij,...jk->...ik", P, U_mu_0[:, :, :, x])

    trace = torch.diagonal(P, dim1=-2, dim2=-1).sum(dim=-1)
    return trace / nc
