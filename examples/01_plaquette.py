"""
Example 1 — Plaquette and gauge observables.

Demonstrates:
  • Creating a LatticeGeometry
  • Generating a cold (identity) and a random (hot) gauge configuration
  • Measuring the plaquette, Wilson action, and topological charge
  • Optional AoT Neuron compilation of a plaquette nn.Module

Run on CPU (no Neuron hardware required):

    python examples/01_plaquette.py

On a Trn1/Inf2 instance with torch-neuronx installed the script will
automatically detect the hardware and compile the observable kernel.
"""

import torch
import torch.nn as nn

from lqcd_neuron.core import GaugeField, LatticeGeometry
from lqcd_neuron.observables import plaquette, topological_charge, wilson_action
from lqcd_neuron.neuron import NeuronCompiler, is_neuron_available


# ---------------------------------------------------------------------------
# Wrap the plaquette as an nn.Module so it can be compiled for Neuron
# ---------------------------------------------------------------------------

class PlaquetteModule(nn.Module):
    """Pure-tensor plaquette measurement, JIT-traceable."""

    def forward(self, U: torch.Tensor) -> torch.Tensor:
        from lqcd_neuron.observables.plaquette import plaquette_tensor
        nc = U.shape[-1]
        return plaquette_tensor(U).mean() / nc


def main() -> None:
    # ------------------------------------------------------------------
    # 1. Define lattice geometry
    # ------------------------------------------------------------------
    geom = LatticeGeometry(T=8, Z=4, Y=4, X=4)
    print(f"Lattice: {geom}")
    print(f"Volume : {geom.volume} sites")

    # ------------------------------------------------------------------
    # 2. Cold start — every link is the identity matrix
    # ------------------------------------------------------------------
    U_cold = GaugeField.cold(geom)
    P_cold = plaquette(U_cold)
    print(f"\nCold configuration plaquette  : {P_cold:.10f}  (expected 1.0)")

    # ------------------------------------------------------------------
    # 3. Hot (random) start
    # ------------------------------------------------------------------
    U_hot = GaugeField.random(geom, seed=42)
    P_hot = plaquette(U_hot)
    print(f"Hot  configuration plaquette  : {P_hot:.6f}  (expected ≈ 0)")

    # ------------------------------------------------------------------
    # 4. Wilson action
    # ------------------------------------------------------------------
    beta = 6.0
    S_cold = wilson_action(U_cold, beta)
    S_hot  = wilson_action(U_hot,  beta)
    print(f"\nWilson action S_W (β={beta}):")
    print(f"  Cold: {S_cold:.6f}")
    print(f"  Hot : {S_hot:.6f}")

    # ------------------------------------------------------------------
    # 5. Topological charge (expect near 0 for random config)
    # ------------------------------------------------------------------
    Q = topological_charge(U_hot)
    print(f"\nTopological charge Q (hot, a≠0): {Q:.6f}")

    # ------------------------------------------------------------------
    # 6. Optional Neuron compilation
    # ------------------------------------------------------------------
    if is_neuron_available():
        print("\nNeuron hardware detected — compiling PlaquetteModule …")
        compiler = NeuronCompiler(dtype="bfloat16")
        plaq_mod = PlaquetteModule()
        plaq_neuron = compiler.compile_observable(plaq_mod, geom.shape)
        P_neuron = plaq_neuron(U_hot.tensor).item()
        print(f"Neuron plaquette (hot): {P_neuron:.6f}")
    else:
        print("\nNo Neuron hardware — skipping AoT compilation demo.")
        plaq_mod = PlaquetteModule()
        P_direct = plaq_mod(U_hot.tensor).item()
        print(f"CPU PlaquetteModule (hot): {P_direct:.6f}")


if __name__ == "__main__":
    main()
