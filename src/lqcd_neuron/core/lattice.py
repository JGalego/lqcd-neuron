"""
LatticeGeometry — encapsulates the (T, Z, Y, X) size of a 4-D hypercubic lattice.

Tensor layout convention (same throughout the whole library):
  dim 0 → T  (temporal)
  dim 1 → Z
  dim 2 → Y
  dim 3 → X  (innermost / fastest-varying)
  dim 4 → direction μ  (for gauge fields)
  dim 5 → spin         (for spinor fields)
  dim 6 → colour row   (for gauge / spinor fields)
  dim 7 → colour col   (for gauge fields)

Direction conventions:
  μ = 0 → T-direction (roll on dim 0)
  μ = 1 → Z-direction (roll on dim 1)
  μ = 2 → Y-direction (roll on dim 2)
  μ = 3 → X-direction (roll on dim 3)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class LatticeGeometry:
    """Immutable descriptor for a 4-D hypercubic Euclidean lattice.

    Args:
        T: Temporal extent (number of time-slices).
        Z: Spatial extent in z.
        Y: Spatial extent in y.
        X: Spatial extent in x.
        nc: Number of colours (default 3 for SU(3)).
        ns: Number of spin components (default 4 for Dirac spinors).

    Example::

        geom = LatticeGeometry(T=16, Z=8, Y=8, X=8)
        print(geom.volume)   # 8192
        print(geom.shape)    # (16, 8, 8, 8)
    """

    T: int
    Z: int
    Y: int
    X: int
    nc: int = 3
    ns: int = 4

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def shape(self) -> Tuple[int, int, int, int]:
        """Spatial extents as (T, Z, Y, X)."""
        return (self.T, self.Z, self.Y, self.X)

    @property
    def volume(self) -> int:
        """Total number of lattice sites."""
        return self.T * self.Z * self.Y * self.X

    @property
    def spatial_volume(self) -> int:
        """Number of sites on a single time-slice."""
        return self.Z * self.Y * self.X

    @property
    def gauge_shape(self):
        """Shape of a full SU(Nc) gauge tensor: (T, Z, Y, X, 4, Nc, Nc)."""
        return (self.T, self.Z, self.Y, self.X, 4, self.nc, self.nc)

    @property
    def spinor_shape(self):
        """Shape of a colour-spinor tensor: (T, Z, Y, X, Ns, Nc)."""
        return (self.T, self.Z, self.Y, self.X, self.ns, self.nc)

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"LatticeGeometry(T={self.T}, Z={self.Z}, Y={self.Y}, X={self.X}, "
            f"Nc={self.nc}, Ns={self.ns})"
        )
