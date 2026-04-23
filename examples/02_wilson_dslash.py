"""
Example 2 — Wilson Dslash operator.

Demonstrates:
  • Constructing WilsonDirac as an nn.Module
  • Applying it to a random spinor on a random gauge background
  • Verifying γ₅-Hermiticity: D† = γ₅ D γ₅
  • Neuron AoT compilation with torch_neuronx.trace

γ₅-Hermiticity check
---------------------
The Wilson Dirac operator satisfies

    D† = γ₅ D γ₅

where γ₅ is applied site-locally to the spinor.  We verify this
numerically as a self-consistency test of the implementation.

Run:
    python examples/02_wilson_dslash.py
"""

import torch
import torch.nn as nn

from lqcd_neuron.core import ColorSpinorField, GaugeField, LatticeGeometry
from lqcd_neuron.dirac import WilsonDirac, degrand_rossi_gammas, gamma5
from lqcd_neuron.neuron import NeuronCompiler, is_neuron_available


def apply_gamma5(psi: torch.Tensor, g5: torch.Tensor) -> torch.Tensor:
    """Apply γ₅ site-locally: out[...,α,c] = Σ_β γ₅[α,β] ψ[...,β,c]."""
    return torch.einsum("ab,...bc->...ac", g5, psi)


def check_gamma5_hermiticity(D: WilsonDirac, U: torch.Tensor, psi: torch.Tensor) -> float:
    """Verify ‖D†ψ − γ₅ D γ₅ ψ‖ / ‖ψ‖  (should be ~1e-6 or less)."""
    g5 = gamma5(dtype=psi.dtype)

    Dpsi        = D(psi, U)
    g5_Dpsi     = apply_gamma5(Dpsi, g5)          # γ₅ D ψ
    g5psi       = apply_gamma5(psi, g5)             # γ₅ ψ
    D_g5psi     = D(g5psi, U)                       # D γ₅ ψ
    Ddag_psi    = D.dagger(psi, U)                  # D† ψ

    # γ₅-Hermiticity: D† ψ = γ₅ D γ₅ ψ
    diff = Ddag_psi - apply_gamma5(D_g5psi, g5)
    rel_err = (diff.abs().pow(2).sum() / psi.abs().pow(2).sum()).sqrt().item()
    return rel_err


def main() -> None:
    geom = LatticeGeometry(T=4, Z=4, Y=4, X=4)
    dtype = torch.complex64
    print(f"Lattice : {geom}")

    # ------------------------------------------------------------------
    # 1. Random gauge field and spinor
    # ------------------------------------------------------------------
    U    = GaugeField.random(geom, seed=7)
    psi  = ColorSpinorField.gaussian(geom, seed=3)
    chi  = ColorSpinorField.gaussian(geom, seed=5)

    D = WilsonDirac(mass=0.1, nc=geom.nc, dtype=dtype)

    # ------------------------------------------------------------------
    # 2. Apply D and D†
    # ------------------------------------------------------------------
    Dpsi   = D(psi.tensor, U.tensor)
    Dagchi = D.dagger(chi.tensor, U.tensor)

    # Verify ⟨χ|Dψ⟩ ≈ ⟨D†χ|ψ⟩
    lhs = (chi.tensor.conj() * Dpsi).sum().item()
    rhs = (Dagchi.conj() * psi.tensor).sum().item()
    print(f"\n<χ|Dψ>          : {lhs:.6f}")
    print(f"<D†χ|ψ>         : {rhs:.6f}")
    print(f"Relative diff   : {abs(lhs - rhs) / (abs(lhs) + 1e-30):.2e}")

    # ------------------------------------------------------------------
    # 3. γ₅-Hermiticity
    # ------------------------------------------------------------------
    rel_err = check_gamma5_hermiticity(D, U.tensor, psi.tensor)
    print(f"\nγ₅-Hermiticity ‖D†ψ − γ₅Dγ₅ψ‖/‖ψ‖: {rel_err:.2e}")

    # ------------------------------------------------------------------
    # 4. Normal operator M†M
    # ------------------------------------------------------------------
    MtMpsi = D.normal(psi.tensor, U.tensor)
    # ⟨ψ|M†Mψ⟩ must be real and positive
    pMtMp  = (psi.tensor.conj() * MtMpsi).sum().item()
    print(f"\n<ψ|M†M|ψ> = {pMtMp.real:.6f} + {pMtMp.imag:.2e}i  (imag should be ~0)")

    # ------------------------------------------------------------------
    # 5. Optional Neuron compilation
    # ------------------------------------------------------------------
    if is_neuron_available():
        print("\nNeuron detected — compiling WilsonDirac …")
        compiler = NeuronCompiler()
        D_neuron = compiler.compile_dslash(D, geom.shape, nc=geom.nc)
        out = D_neuron(psi.tensor, U.tensor)
        diff = (out - Dpsi).abs().max().item()
        print(f"Max diff Neuron vs CPU: {diff:.2e}")
    else:
        print("\nNo Neuron hardware — example complete (CPU only).")


if __name__ == "__main__":
    main()
