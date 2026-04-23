"""
LQCD-Neuron: A QUDA-inspired Lattice QCD SDK for AWS Trainium and Inferentia.

This library expresses Lattice QCD computations as pure PyTorch tensor graphs,
enabling compilation and execution on AWS Neuron hardware (Trainium Trn1 and
Inferentia Inf2) via the AWS Neuron SDK (torch-neuronx / XLA).

Unlike QUDA, which relies on hand-written CUDA kernels launched via the CUDA
runtime, every operation here is a standard PyTorch nn.Module.  The Neuron
compiler (neuronx-cc) then lowers the XLA HLO graph to NeuronCore instructions
ahead-of-time, yielding deterministic, high-throughput execution without any
GPGPU interface.

Typical usage::

    from lqcd_neuron.core import LatticeGeometry, GaugeField, ColorSpinorField
    from lqcd_neuron.dirac import WilsonDslash
    from lqcd_neuron.solvers import ConjugateGradient
    from lqcd_neuron.observables import plaquette

    geom = LatticeGeometry(T=8, Z=4, Y=4, X=4)
    U = GaugeField.random(geom)
    print("Plaquette:", plaquette(U))

    D = WilsonDslash(mass=0.1)
    psi = ColorSpinorField.gaussian(geom)
    out = D(psi.tensor, U.tensor)
"""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("lqcd-neuron")
except PackageNotFoundError:
    __version__ = "0.1.0-dev"

__all__ = [
    "__version__",
]
