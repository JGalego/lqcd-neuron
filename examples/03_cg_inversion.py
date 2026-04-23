"""
Example 3 — Conjugate Gradient quark propagator.

Demonstrates:
  • Solving M†M x = M†b for a point source b using CG
  • Monitoring convergence via the solver info object
  • Residual verification after the solve
  • Clover-Wilson example with SW improvement

This pattern is the core loop of quark propagator generation (the most
compute-intensive step in most lattice QCD calculations).  On Trn1/Inf2 the
matrix-vector product M†M (which calls WilsonDirac.normal) is compiled to a
NeuronCore kernel; the CG loop itself runs on the host, just like a typical
PyTorch training loop.

Run:
    python examples/03_cg_inversion.py
"""

import time
import torch

from lqcd_neuron.core import ColorSpinorField, GaugeField, LatticeGeometry
from lqcd_neuron.dirac import WilsonDirac, CloverWilsonDirac
from lqcd_neuron.solvers import ConjugateGradient, BiCGStab
from lqcd_neuron.neuron import NeuronCompiler, is_neuron_available


def main() -> None:
    # ------------------------------------------------------------------
    # 1. Set up a small 4⁴ lattice for quick demonstration
    # ------------------------------------------------------------------
    geom  = LatticeGeometry(T=4, Z=4, Y=4, X=4)
    dtype = torch.complex64
    print(f"Lattice : {geom}  (volume = {geom.volume})")

    U   = GaugeField.random(geom, seed=12)
    b   = ColorSpinorField.point_source(geom, t=0, z=0, y=0, x=0,
                                         spin=0, color=0, dtype=dtype)

    # ------------------------------------------------------------------
    # 2. Wilson Dirac operator
    # ------------------------------------------------------------------
    mass = 0.05
    D    = WilsonDirac(mass=mass, nc=geom.nc, dtype=dtype)

    # Right-hand side: b̃ = M† b
    b_tilde = D.dagger(b.tensor, U.tensor)

    # ------------------------------------------------------------------
    # 3. Optional: compile M†M for Neuron
    # ------------------------------------------------------------------
    if is_neuron_available():
        print("Compiling WilsonDirac for Neuron …")
        compiler = NeuronCompiler()
        D_compiled = compiler.compile_dslash(D, geom.shape, nc=geom.nc)

        def matvec(x: torch.Tensor) -> torch.Tensor:
            Mx = D_compiled(x, U.tensor)
            return D.dagger(Mx, U.tensor)  # D† is not separately compiled here
    else:
        def matvec(x: torch.Tensor) -> torch.Tensor:
            return D.normal(x, U.tensor)

    # ------------------------------------------------------------------
    # 4. CG solve
    # ------------------------------------------------------------------
    solver = ConjugateGradient(tol=1e-8, maxiter=500, verbose=True, print_every=25)

    print(f"\n--- CG inversion (mass={mass}) ---")
    t0 = time.perf_counter()
    x_cg, info_cg = solver.solve(matvec, b_tilde)
    t1 = time.perf_counter()
    print(f"Converged : {info_cg.converged}")
    print(f"Iterations: {info_cg.iterations}")
    print(f"Final |r|/|b|: {info_cg.final_residual:.3e}")
    print(f"Wall time : {t1 - t0:.3f} s")

    # ------------------------------------------------------------------
    # 5. Verify residual: M†M x ≈ b̃
    # ------------------------------------------------------------------
    r_check = matvec(x_cg) - b_tilde
    true_res = (r_check.abs().pow(2).sum() / b_tilde.abs().pow(2).sum()
                ).sqrt().real.item()
    print(f"True ‖M†Mx−b̃‖/‖b̃‖: {true_res:.3e}")

    # ------------------------------------------------------------------
    # 6. BiCGStab comparison on the un-preconditioned Wilson system Mx=b
    # ------------------------------------------------------------------
    print(f"\n--- BiCGStab inversion (same problem via BiCGStab on M†M) ---")
    bicg = BiCGStab(tol=1e-8, maxiter=500, verbose=True, print_every=25)
    x_bicg, info_bicg = bicg.solve(matvec, b_tilde)
    print(f"Converged : {info_bicg.converged}")
    print(f"Iterations: {info_bicg.iterations}")
    print(f"Final |r|/|b|: {info_bicg.final_residual:.3e}")

    # ------------------------------------------------------------------
    # 7. Clover-Wilson solve
    # ------------------------------------------------------------------
    print(f"\n--- Clover-Wilson CG inversion (c_SW = 1.0) ---")
    D_clv = CloverWilsonDirac(mass=mass, csw=1.0, nc=geom.nc, dtype=dtype)
    D_clv.set_gauge(U.tensor)

    def matvec_clover(x: torch.Tensor) -> torch.Tensor:
        Mx  = D_clv(x, U.tensor)
        Mdx = D_clv._dagger(Mx, U.tensor)
        return Mdx

    b_clv   = D_clv._dagger(b.tensor, U.tensor)
    cg_clv  = ConjugateGradient(tol=1e-8, maxiter=500, verbose=True, print_every=25)
    x_clv, info_clv = cg_clv.solve(matvec_clover, b_clv)
    print(f"Converged : {info_clv.converged}")
    print(f"Iterations: {info_clv.iterations}")
    print(f"Final |r|/|b|: {info_clv.final_residual:.3e}")

    # ------------------------------------------------------------------
    # 8. Convergence plot (text-based sparkline)
    # ------------------------------------------------------------------
    hist = info_cg.residual_history
    import math
    print("\nCG residual history (every 5th iter):")
    for i, r in enumerate(hist[::5], 1):
        bar = "#" * max(1, int(-math.log10(max(r, 1e-16)) * 2))
        print(f"  iter {i*5:4d}: {r:.2e}  {bar}")


if __name__ == "__main__":
    main()
