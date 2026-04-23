"""Tests for CG and BiCGStab solvers."""

import math
import pytest
import torch

from lqcd_neuron.core import ColorSpinorField, GaugeField, LatticeGeometry
from lqcd_neuron.dirac import WilsonDirac
from lqcd_neuron.solvers import BiCGStab, ConjugateGradient


DTYPE = torch.complex64


@pytest.fixture
def geom():
    return LatticeGeometry(T=4, Z=4, Y=4, X=4)


@pytest.fixture
def U(geom):
    return GaugeField.random(geom, seed=99).tensor


# ---------------------------------------------------------------------------
# CG on a simple diagonal system
# ---------------------------------------------------------------------------

class TestConjugateGradient:
    def test_diagonal_system(self):
        """CG must solve A x = b exactly for a positive-definite diagonal A."""
        n = 48  # 4*4*3 spinor dofs on a 1-site lattice
        diag = (torch.rand(n) + 1.0).to(torch.complex64)  # eigenvalues in [1,2]
        b    = torch.randn(n, dtype=torch.complex64)
        x_exact = b / diag

        def matvec(x):
            return diag * x

        solver = ConjugateGradient(tol=1e-10, maxiter=300)
        x, info = solver.solve(matvec, b)
        assert info.converged
        assert torch.allclose(x, x_exact, atol=1e-6)

    def test_zero_rhs(self):
        """CG with b=0 should return immediately with x=0."""
        b = torch.zeros(48, dtype=torch.complex64)

        def matvec(x):
            return x

        solver = ConjugateGradient(tol=1e-8, maxiter=100)
        x, info = solver.solve(matvec, b)
        assert info.converged
        assert info.iterations == 0
        assert torch.all(x == 0)

    def test_initial_guess(self):
        """Providing a good initial guess should reduce the number of iterations."""
        n  = 48
        diag = (torch.rand(n) + 1.0).to(torch.complex64)
        b  = torch.randn(n, dtype=torch.complex64)
        x_exact = b / diag

        solver = ConjugateGradient(tol=1e-8, maxiter=300)
        # Solve from zero first to get baseline iteration count
        _, info_cold = solver.solve(lambda x: diag * x, b)

        # Solve with near-exact initial guess — should converge in fewer iters
        x_near = x_exact + 1e-4 * torch.randn_like(x_exact)
        _, info_warm = solver.solve(lambda x: diag * x, b, x0=x_near)
        assert info_warm.iterations <= info_cold.iterations

    def test_wilson_cg(self, geom, U):
        """CG should converge for a random Wilson configuration."""
        D    = WilsonDirac(mass=0.5, nc=geom.nc, dtype=DTYPE)
        b    = ColorSpinorField.gaussian(geom, seed=7, dtype=DTYPE).tensor
        b_rhs = D.dagger(b, U)

        solver = ConjugateGradient(tol=1e-6, maxiter=500)
        x, info = solver.solve(lambda v: D.normal(v, U), b_rhs)
        assert info.converged
        # Verify residual
        r = D.normal(x, U) - b_rhs
        rel_res = (r.abs().pow(2).sum() / b_rhs.abs().pow(2).sum()).sqrt().item()
        assert rel_res < 1e-5

    def test_residual_history_length(self):
        """History should have length = #iterations + 1 (initial residual)."""
        n    = 20
        diag = (torch.rand(n) + 0.5).to(torch.complex64)
        b    = torch.randn(n, dtype=torch.complex64)
        solver = ConjugateGradient(tol=1e-8, maxiter=100)
        _, info = solver.solve(lambda x: diag * x, b)
        assert len(info.residual_history) == info.iterations + 1


# ---------------------------------------------------------------------------
# BiCGStab
# ---------------------------------------------------------------------------

class TestBiCGStab:
    def test_diagonal_system(self):
        n    = 48
        diag = (torch.rand(n) + 1.0).to(torch.complex64)
        b    = torch.randn(n, dtype=torch.complex64)
        x_exact = b / diag

        solver = BiCGStab(tol=1e-9, maxiter=300)
        x, info = solver.solve(lambda v: diag * v, b)
        assert info.converged
        assert torch.allclose(x, x_exact, atol=1e-6)

    def test_zero_rhs(self):
        b = torch.zeros(48, dtype=torch.complex64)
        solver = BiCGStab(tol=1e-8, maxiter=100)
        x, info = solver.solve(lambda v: v, b)
        assert info.converged
        assert info.iterations == 0

    def test_wilson_bicgstab(self, geom, U):
        """BiCGStab should converge for Wilson M†M."""
        D     = WilsonDirac(mass=0.5, nc=geom.nc, dtype=DTYPE)
        b     = ColorSpinorField.gaussian(geom, seed=11, dtype=DTYPE).tensor
        b_rhs = D.dagger(b, U)

        solver = BiCGStab(tol=1e-6, maxiter=500)
        x, info = solver.solve(lambda v: D.normal(v, U), b_rhs)
        assert info.converged
