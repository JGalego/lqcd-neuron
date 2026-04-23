"""
Conjugate Gradient (CG) solver for Hermitian positive-definite systems.

Solves  A x = b  where A is Hermitian positive-definite.  For Wilson
fermions A is typically M†M (normal equations) or the even-odd preconditioned
operator M_ee + M_eo M_oo^{−1} M_oe (not yet implemented here).

Architecture note for AWS Neuron
---------------------------------
The solver loop runs on the **host** (Python / CPU), while the operator
application ``A(x)`` is compiled for Neuron via ``torch_neuronx.trace``.
Each matrix-vector product crosses the host↔device boundary once per
iteration.  This mirrors how PyTorch training loops work: the loop logic
stays in Python; the expensive NN.forward() lives on the accelerator.

Usage::

    from lqcd_neuron.dirac import WilsonDirac
    from lqcd_neuron.solvers import ConjugateGradient

    D = WilsonDirac(mass=0.1)

    def matvec(x):
        return D.normal(x, U)   # M†M x

    solver = ConjugateGradient(tol=1e-8, maxiter=500)
    x, info = solver.solve(matvec, b)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Optional

import torch

from ..blas.lattice_blas import inner, norm2


@dataclass
class SolverInfo:
    """Statistics returned by every solver."""

    converged: bool
    iterations: int
    final_residual: float
    residual_history: list


class ConjugateGradient:
    """Iterative CG solver for Hermitian positive-definite systems.

    Args:
        tol:     Relative residual tolerance ‖r‖/‖b‖ at which to stop.
        maxiter: Maximum number of CG iterations.
        verbose: If True, print residual every *print_every* iterations.
        print_every: Print frequency when *verbose* is True.
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
        """Solve A x = b.

        Args:
            A:  Callable implementing the matrix-vector product A(x).
                Must produce a tensor of the same shape as *x*.
            b:  Right-hand side tensor.
            x0: Initial guess.  Defaults to the zero vector.

        Returns:
            ``(x, info)`` where *x* is the approximate solution and
            *info* is a :class:`SolverInfo` instance.
        """
        b_norm2 = norm2(b).real.item()
        if b_norm2 == 0.0:
            return torch.zeros_like(b), SolverInfo(True, 0, 0.0, [0.0])

        b_norm = math.sqrt(b_norm2)
        tol_sq = (self.tol * b_norm) ** 2

        x = torch.zeros_like(b) if x0 is None else x0.clone()
        r = b - A(x) if x0 is not None else b.clone()

        r_norm2 = norm2(r).real.item()
        p = r.clone()
        history = [math.sqrt(r_norm2) / b_norm]

        for k in range(1, self.maxiter + 1):
            Ap = A(p)
            pAp = inner(p, Ap).real.item()
            if abs(pAp) < 1e-30:
                break  # breakdown safeguard

            alpha = r_norm2 / pAp
            x = x + alpha * p
            r = r - alpha * Ap
            r_norm2_new = norm2(r).real.item()

            beta = r_norm2_new / r_norm2
            p = r + beta * p
            r_norm2 = r_norm2_new

            rel_res = math.sqrt(r_norm2) / b_norm
            history.append(rel_res)

            if self.verbose and k % self.print_every == 0:
                print(f"  CG iter {k:5d}  |r|/|b| = {rel_res:.3e}")

            if r_norm2 < tol_sq:
                if self.verbose:
                    print(f"  CG converged at iter {k}, |r|/|b| = {rel_res:.3e}")
                return x, SolverInfo(True, k, rel_res, history)

        rel_res = math.sqrt(r_norm2) / b_norm
        if self.verbose:
            print(f"  CG did NOT converge after {self.maxiter} iters, "
                  f"|r|/|b| = {rel_res:.3e}")
        return x, SolverInfo(False, self.maxiter, rel_res, history)
