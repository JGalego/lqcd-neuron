"""Tests for core field classes."""

import pytest
import torch

from lqcd_neuron.core import GaugeField, LatticeGeometry, ColorSpinorField


@pytest.fixture
def geom4():
    return LatticeGeometry(T=4, Z=4, Y=4, X=4)


@pytest.fixture
def geom_small():
    return LatticeGeometry(T=2, Z=2, Y=2, X=2)


# ---------------------------------------------------------------------------
# LatticeGeometry tests
# ---------------------------------------------------------------------------

class TestLatticeGeometry:
    def test_volume(self, geom4):
        assert geom4.volume == 4 * 4 * 4 * 4

    def test_spatial_volume(self, geom4):
        assert geom4.spatial_volume == 4 * 4 * 4

    def test_gauge_shape(self, geom4):
        assert geom4.gauge_shape == (4, 4, 4, 4, 4, 3, 3)

    def test_spinor_shape(self, geom4):
        assert geom4.spinor_shape == (4, 4, 4, 4, 4, 3)

    def test_shape_tuple(self, geom4):
        assert geom4.shape == (4, 4, 4, 4)


# ---------------------------------------------------------------------------
# GaugeField tests
# ---------------------------------------------------------------------------

class TestGaugeField:
    def test_cold_shape(self, geom4):
        U = GaugeField.cold(geom4)
        assert tuple(U.tensor.shape) == geom4.gauge_shape

    def test_cold_is_identity(self, geom4):
        U = GaugeField.cold(geom4)
        # Each link should be the 3x3 identity
        nc = geom4.nc
        eye = torch.eye(nc, dtype=torch.complex64)
        # Check one link at (0,0,0,0,0)
        link = U.tensor[0, 0, 0, 0, 0]
        assert torch.allclose(link, eye, atol=1e-6)

    def test_random_shape(self, geom4):
        U = GaugeField.random(geom4, seed=0)
        assert tuple(U.tensor.shape) == geom4.gauge_shape

    def test_random_is_unitary(self, geom4):
        U = GaugeField.random(geom4, seed=1)
        nc = geom4.nc
        links = U.tensor.reshape(-1, nc, nc)
        I = torch.eye(nc, dtype=U.dtype).unsqueeze(0).expand(links.shape[0], nc, nc)
        UUdag = torch.einsum("...ij,...kj->...ik", links, links.conj())
        assert torch.allclose(UUdag, I, atol=1e-5), "Links are not unitary"

    def test_determinant_is_one(self, geom4):
        U = GaugeField.random(geom4, seed=2)
        nc = geom4.nc
        links = U.tensor.reshape(-1, nc, nc)
        dets = torch.linalg.det(links)
        assert torch.allclose(dets.abs(), torch.ones_like(dets.abs()), atol=1e-5)

    def test_wrong_shape_raises(self, geom4):
        with pytest.raises(ValueError):
            GaugeField(torch.zeros(4, 4, 4, 4, 3, 3, 3), geom4)

    def test_dagger(self, geom4):
        U = GaugeField.random(geom4, seed=3)
        Udag = U.dagger()
        # U U† should equal identity for each link
        links = U.tensor.reshape(-1, geom4.nc, geom4.nc)
        dag   = Udag.tensor.reshape(-1, geom4.nc, geom4.nc)
        UUdag = torch.einsum("...ij,...jk->...ik", links, dag)
        I = torch.eye(geom4.nc, dtype=U.dtype)
        assert torch.allclose(UUdag, I.unsqueeze(0), atol=1e-5)

    def test_unitarize_cold(self, geom4):
        U = GaugeField.cold(geom4)
        U_unit = U.unitarize()
        nc = geom4.nc
        links = U_unit.tensor.reshape(-1, nc, nc)
        UUdag = torch.einsum("...ij,...kj->...ik", links, links.conj())
        I = torch.eye(nc, dtype=U.dtype)
        assert torch.allclose(UUdag, I.unsqueeze(0), atol=1e-5)


# ---------------------------------------------------------------------------
# ColorSpinorField tests
# ---------------------------------------------------------------------------

class TestColorSpinorField:
    def test_zeros_shape(self, geom4):
        psi = ColorSpinorField.zeros(geom4)
        assert tuple(psi.tensor.shape) == geom4.spinor_shape

    def test_gaussian_unit_norm(self, geom4):
        psi = ColorSpinorField.gaussian(geom4, seed=0)
        assert abs(psi.norm().item() - 1.0) < 1e-5

    def test_point_source(self, geom_small):
        psi = ColorSpinorField.point_source(geom_small, t=0, z=0, y=0, x=0, spin=1, color=2)
        assert psi.tensor.abs().sum().item() == pytest.approx(1.0, abs=1e-6)
        assert psi.tensor[0, 0, 0, 0, 1, 2].real.item() == pytest.approx(1.0)

    def test_inner_product_is_norm2(self, geom4):
        psi = ColorSpinorField.gaussian(geom4, seed=7)
        ip = psi.inner(psi).real.item()
        n2 = psi.norm2().item()
        assert abs(ip - n2) < 1e-5

    def test_addition(self, geom4):
        psi = ColorSpinorField.zeros(geom4)
        chi = ColorSpinorField.gaussian(geom4, seed=8)
        result = psi + chi
        assert torch.allclose(result.tensor, chi.tensor)

    def test_wrong_shape_raises(self, geom4):
        with pytest.raises(ValueError):
            ColorSpinorField(torch.zeros(4, 4, 4, 4, 3, 3), geom4)
