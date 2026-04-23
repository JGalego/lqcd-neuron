"""
BiConjugate Gradient Stabilised (BiCGStab) solver.

Solves  A x = b  for general (possibly non-Hermitian) square systems.
BiCGStab is slightly more expensive per iteration than CG but converges
for non-symmetric operators such as the un-preconditioned Wilson Dirac
matrix M (without the M†M normal-equation trick).

Reference: Van der Vorst, H.A. (1992). "Bi-CGSTAB: A fast and smoothly
converging variant of Bi-CG for the solution of nonsymmetric linear
systems." SIAM Journal on Scientific and Statistical Computing, 13(2).

Architecture note
-----------------
Same host-loop / device-kernel split as CG: the operator A is traced once
on Neuron; the three dot-products per iteration execute on the host.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Optional

import torch

from ..blas.lattice_blas import inner, norm2
from .cg import SolverInfo


class BiCGStab:
    """BiCGStab solver for general square linear systems.

    Args:
        tol:         Relative residual tolerance ‖rₙ‖/‖b‖.
        maxiter:     Maximum number of iterations.
        verbose:     Print residual history when True.
        print_every: Print frequency.
    """

    def __init__(
        self,
        tol: float = 1e-8,
        maxiter: int = 1000,
        verbose: bool = False,
        print_every: int = 50,
    ) -> None:
        self.tol = tol
        self.maxiter = maxiter
        self.verbose = verbose
        self.print_every = print_every

    def solve(
        self,
        A: Callable[[torch.Tensor], torch.Tensor],
        b: torch.Tensor,
        x0: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, SolverInfo]:
        """Solve A x = b via BiCGStab.

        Args:
            A:  Matrix-vector callable A(x).
            b:  Right-hand side.
            x0: Initial guess (default: zero).

        Returns:
            ``(x, info)``
        """
        b_norm2 = norm2(b).real.item()
        if b_norm2 == 0.0:
            return torch.zeros_like(b), SolverInfo(True, 0, 0.0, [0.0])

        b_norm = math.sqrt(b_norm2)
        tol_sq = (self.tol * b_norm) ** 2

        x = torch.zeros_like(b) if x0 is None else x0.clone()
        r = b - A(x) if x0 is not None else b.clone()
        r_hat = r.clone()  # arbitrary fixed shadow residual

        rho_prev = torch.ones(1, dtype=b.dtype, device=b.device)
        alpha    = torch.ones(1, dtype=b.dtype, device=b.device)
        omega    = torch.ones(1, dtype=b.dtype, device=b.device)
        v = torch.zeros_like(b)
        p = torch.zeros_like(b)

        r_norm2 = norm2(r).real.item()
        history = [math.sqrt(r_norm2) / b_norm]

        for k in range(1, self.maxiter + 1):
            rho = inner(r_hat, r)

            if rho.abs().item() < 1e-30:
                # Breakdown: restart from current x
                r = b - A(x)
                r_hat = r.clone()
                rho = inner(r_hat, r)

            beta = (rho / rho_prev) * (alpha / omega)
            p = r + beta * (p - omega * v)
            v = A(p)

            r_hat_v = inner(r_hat, v)
            if r_hat_v.abs().item() < 1e-30:
                break  # hard breakdown

            alpha = rho / r_hat_v
            s = r - alpha * v
            t = A(s)

            ts = inner(t, s)
            tt = inner(t, t)
            if tt.abs().item() < 1e-30:
                x = x + alpha * p
                break

            omega = ts / tt
            x = x + alpha * p + omega * s
            r = s - omega * t
            rho_prev = rho

            r_norm2 = norm2(r).real.item()
            rel_res = math.sqrt(r_norm2) / b_norm
            history.append(rel_res)

            if self.verbose and k % self.print_every == 0:
                print(f"  BiCGStab iter {k:5d}  |r|/|b| = {rel_res:.3e}")

            if r_norm2 < tol_sq:
                if self.verbose:
                    print(f"  BiCGStab converged at iter {k}, |r|/|b| = {rel_res:.3e}")
                return x, SolverInfo(True, k, rel_res, history)

        rel_res = math.sqrt(norm2(r).real.item()) / b_norm
        if self.verbose:
            print(f"  BiCGStab did NOT converge after {self.maxiter} iters, "
                  f"|r|/|b| = {rel_res:.3e}")
        return x, SolverInfo(False, self.maxiter, rel_res, history)
