"""Tests for gauge observables."""

import math
import pytest
import torch

from lqcd_neuron.core import GaugeField, LatticeGeometry
from lqcd_neuron.observables import (
    plaquette,
    plaquette_tensor,
    topological_charge,
    wilson_action,
    polyakov_loop,
)


@pytest.fixture
def geom():
    return LatticeGeometry(T=4, Z=4, Y=4, X=4)


@pytest.fixture
def U_cold(geom):
    return GaugeField.cold(geom)


@pytest.fixture
def U_random(geom):
    return GaugeField.random(geom, seed=77)


# ---------------------------------------------------------------------------
# Plaquette
# ---------------------------------------------------------------------------

class TestPlaquette:
    def test_cold_plaquette_is_one(self, U_cold):
        P = plaquette(U_cold)
        assert abs(P - 1.0) < 1e-6, f"Cold plaquette = {P}, expected 1.0"

    def test_random_plaquette_in_range(self, U_random):
        P = plaquette(U_random)
        assert 0.0 <= P <= 1.0, f"Plaquette out of [0,1]: {P}"

    def test_raw_tensor_input(self, U_cold):
        P = plaquette(U_cold.tensor)
        assert abs(P - 1.0) < 1e-6

    def test_plaquette_tensor_shape(self, geom, U_cold):
        T, Z, Y, X = geom.shape
        pt = plaquette_tensor(U_cold.tensor)
        assert pt.shape == (T, Z, Y, X, 6)

    def test_plaquette_tensor_cold_is_nc(self, geom, U_cold):
        nc = geom.nc
        pt = plaquette_tensor(U_cold.tensor)
        # Each Re Tr[I] = Nc for a cold configuration
        assert torch.allclose(
            pt, torch.full_like(pt, float(nc)), atol=1e-5
        )

    def test_plaquette_is_real(self, U_random):
        """Plaquette is a real number (imaginary part should vanish)."""
        U = U_random.tensor
        nc = U.shape[-1]
        pt = plaquette_tensor(U)  # This is already real (Re Tr)
        assert pt.dtype in (torch.float32, torch.float64)


# ---------------------------------------------------------------------------
# Wilson action
# ---------------------------------------------------------------------------

class TestWilsonAction:
    def test_cold_action_is_zero(self, U_cold):
        S = wilson_action(U_cold, beta=6.0)
        assert abs(S) < 1e-5, f"Cold action = {S}, expected 0"

    def test_action_is_positive(self, U_random):
        S = wilson_action(U_random, beta=6.0)
        assert S >= 0.0, f"Wilson action is negative: {S}"


# ---------------------------------------------------------------------------
# Topological charge
# ---------------------------------------------------------------------------

class TestTopologicalCharge:
    def test_cold_charge_is_zero(self, U_cold):
        Q = topological_charge(U_cold)
        assert abs(Q) < 1e-5, f"Cold topological charge = {Q}"

    def test_returns_float(self, U_random):
        Q = topological_charge(U_random)
        assert isinstance(Q, float)


# ---------------------------------------------------------------------------
# Polyakov loop
# ---------------------------------------------------------------------------

class TestPolyakovLoop:
    def test_returns_complex(self, U_random):
        L = polyakov_loop(U_random)
        assert L.is_complex()

    def test_modulus_at_most_one(self, U_random):
        """|⟨L⟩| ≤ 1 since each trace/Nc has |·| ≤ 1."""
        L = polyakov_loop(U_random)
        assert L.abs().item() <= 1.0 + 1e-5

    def test_cold_loop_is_one(self, U_cold):
        """For a cold (identity) gauge field the Polyakov loop = 1."""
        L = polyakov_loop(U_cold)
        assert abs(L.real.item() - 1.0) < 1e-5
        assert abs(L.imag.item())       < 1e-5
