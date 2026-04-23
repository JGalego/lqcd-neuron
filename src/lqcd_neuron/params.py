"""
QUDA-inspired parameter dataclasses for configuring Lattice QCD operations.

These mirror the spirit of QUDA's QudaGaugeParam, QudaInvertParam, etc., but
are plain Python dataclasses instead of C structs—straightforward to
serialise, introspect, and pass across process boundaries.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, Tuple


# ---------------------------------------------------------------------------
# Enumerations  (mirrors enum_quda.h)
# ---------------------------------------------------------------------------

class Precision(Enum):
    FLOAT32 = "float32"
    FLOAT64 = "float64"
    BFLOAT16 = "bfloat16"    # Native Neuron precision
    FLOAT16  = "float16"     # Supported on Inf2


class DslashType(Enum):
    WILSON         = auto()
    CLOVER_WILSON  = auto()
    STAGGERED      = auto()   # Future
    DOMAIN_WALL    = auto()   # Future


class InverterType(Enum):
    CG       = auto()
    BICGSTAB = auto()
    GCR      = auto()   # Future


class TBoundary(Enum):
    PERIODIC       =  1
    ANTI_PERIODIC  = -1


class LinkType(Enum):
    SU3    = auto()
    GENERAL = auto()


class ResidualType(Enum):
    L2_RELATIVE  = auto()
    L2_ABSOLUTE  = auto()


# ---------------------------------------------------------------------------
# Parameter dataclasses
# ---------------------------------------------------------------------------

@dataclass
class GaugeParam:
    """Parameters that describe a gauge (link) field configuration.

    Attributes:
        lattice_size: (T, Z, Y, X) spatial extents.
        nc:           Number of colours (default 3 for SU(3)).
        precision:    Floating-point precision of the stored links.
        link_type:    SU3 (standard Wilson) or GENERAL.
        t_boundary:   Temporal boundary condition for fermion fields.
        anisotropy:   Anisotropy factor ξ = a_s / a_t (default 1.0).
        tadpole_coeff: Tadpole improvement coefficient u₀ (HISQ/ASQTAD).
    """

    lattice_size: Tuple[int, int, int, int] = (4, 4, 4, 4)
    nc: int = 3
    precision: Precision = Precision.FLOAT32
    link_type: LinkType = LinkType.SU3
    t_boundary: TBoundary = TBoundary.ANTI_PERIODIC
    anisotropy: float = 1.0
    tadpole_coeff: float = 1.0

    @property
    def volume(self) -> int:
        T, Z, Y, X = self.lattice_size
        return T * Z * Y * X


@dataclass
class CloverParam:
    """Parameters for the Sheikholeslami-Wohlert (clover) improvement term.

    Attributes:
        csw:          Clover coefficient c_SW.
        precision:    Precision used to compute / store the clover matrix.
        compute_clover: If True, recompute F_{μν} from the current gauge field.
    """

    csw: float = 1.0
    precision: Precision = Precision.FLOAT32
    compute_clover: bool = True


@dataclass
class InvertParam:
    """Parameters for a linear-system solver (fermion matrix inversion).

    Attributes:
        dslash_type:  Which Dirac operator to use.
        inv_type:     Which iterative solver to use.
        mass:         Bare quark mass (staggered).
        kappa:        Hopping parameter κ = 1/(8 + 2m) (Wilson).
        clover:       Clover improvement parameters (if dslash_type is CLOVER_WILSON).
        tol:          Relative residual tolerance ‖r‖/‖b‖.
        maxiter:      Maximum number of solver iterations.
        precision:    Working precision inside the solver.
        precision_sloppy: Sloppy (inner) precision for mixed-precision solvers.
        verbosity:    0 = silent, 1 = summary, 2 = per-iteration.
        residual_type: How convergence is assessed.
    """

    dslash_type: DslashType = DslashType.WILSON
    inv_type: InverterType = InverterType.CG
    mass: float = 0.1
    kappa: float = 0.125
    clover: CloverParam = field(default_factory=CloverParam)
    tol: float = 1e-8
    maxiter: int = 1000
    precision: Precision = Precision.FLOAT32
    precision_sloppy: Precision = Precision.FLOAT32
    verbosity: int = 0
    residual_type: ResidualType = ResidualType.L2_RELATIVE


@dataclass
class EigParam:
    """Parameters for an eigenvalue solver (Thick-Restart Lanczos etc.).

    Attributes:
        n_ev:    Number of desired eigenpairs.
        n_kr:    Krylov subspace size.
        tol:     Convergence tolerance for eigenvalue residuals.
        maxiter: Maximum Lanczos restarts.
    """

    n_ev: int = 16
    n_kr: int = 64
    tol: float = 1e-6
    maxiter: int = 500


@dataclass
class NeuronCompileParam:
    """Parameters controlling AWS Neuron ahead-of-time compilation.

    Attributes:
        compiler_workdir: Directory for neuronx-cc artefacts.
        dtype:            Data type passed to torch_neuronx.trace.
        optimize_level:   Compiler optimisation level (1–3).
        num_neuroncores:  Number of NeuronCores to partition across (data
                          parallel or tensor parallel).
        inline_weights:   Bake gauge-field values into the compiled graph
                          (faster, but requires recompilation per config).
    """

    compiler_workdir: Optional[str] = None
    dtype: str = "float32"
    optimize_level: int = 2
    num_neuroncores: int = 1
    inline_weights: bool = False
