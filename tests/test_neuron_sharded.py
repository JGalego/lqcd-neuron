"""CPU correctness tests for T-axis spatial sharding of Wilson Dslash.

These tests validate the halo-aware sharded adapter and host orchestrator
against the eager full-volume Wilson Dslash without requiring any Neuron
hardware — the same nn.Modules execute in pure PyTorch.
"""

from __future__ import annotations

import pytest
import torch

from lqcd_neuron.core import ColorSpinorField, GaugeField, LatticeGeometry
from lqcd_neuron.dirac import WilsonDirac, WilsonDslash
from lqcd_neuron.neuron.compiler import (
    _BatchedUnbakedShardedAdapter,
    _ShardedBakedGaugeAdapter,
    _ShardedDslashWrapper,
    _auto_num_shards,
    _slice_shard_gauge,
    _shard_T_indices,
)


ATOL = 1e-4  # float32 matrix arithmetic tolerance


def _build_sharded(D, U, lattice_shape, num_shards, nc=3):
    """Build an eager (CPU) sharded wrapper equivalent to compile_dslash_sharded."""
    T, Z, Y, X = lattice_shape
    T_local, num_shards = _shard_T_indices(T, num_shards)
    diag = 4.0 + D.mass if isinstance(D, WilsonDirac) else 0.0

    shard_modules = []
    for s in range(num_shards):
        U_l_re, U_l_im, U_tm1_re, U_tm1_im = _slice_shard_gauge(
            U, s, num_shards, dtype=torch.float32,
        )
        shard_modules.append(
            _ShardedBakedGaugeAdapter(
                U_l_re, U_l_im, U_tm1_re, U_tm1_im,
                diag=diag, nc=nc,
            )
        )
    return _ShardedDslashWrapper(
        shard_modules, num_shards=num_shards, T_local=T_local,
        compute_dtype=torch.float32,
    )


@pytest.mark.parametrize("num_shards", [1, 2, 4])
@pytest.mark.parametrize("op_cls", [WilsonDslash, WilsonDirac])
def test_sharded_matches_full_volume(num_shards, op_cls):
    """Sharded wrapper output must equal the eager full-volume operator."""
    geom = LatticeGeometry(T=8, Z=4, Y=4, X=4)
    shape = (geom.T, geom.Z, geom.Y, geom.X)
    U = GaugeField.random(geom, seed=42).tensor
    psi = ColorSpinorField.gaussian(geom, seed=7).tensor

    if op_cls is WilsonDirac:
        D = WilsonDirac(mass=0.1, nc=3, dtype=torch.complex64)
    else:
        D = WilsonDslash(nc=3, dtype=torch.complex64)

    expected = D(psi, U)

    sharded = _build_sharded(D, U, shape, num_shards=num_shards, nc=3)
    got = sharded(psi)

    assert got.shape == expected.shape
    assert torch.allclose(got, expected, atol=ATOL), (
        f"max abs diff = {(got - expected).abs().max().item():.3e}"
    )


def test_sharded_periodic_boundary():
    """Halo gather must wrap periodically across the global T boundary."""
    # With num_shards == T, each slab is a single t-row, so every backward
    # link at local t=0 hits the U_tm1 slab and every forward neighbour
    # comes from the right halo — exercises the cat-splice paths fully.
    geom = LatticeGeometry(T=4, Z=4, Y=4, X=4)
    shape = (geom.T, geom.Z, geom.Y, geom.X)
    U = GaugeField.random(geom, seed=1).tensor
    psi = ColorSpinorField.gaussian(geom, seed=2).tensor

    D = WilsonDirac(mass=0.05, nc=3, dtype=torch.complex64)
    expected = D(psi, U)

    sharded = _build_sharded(D, U, shape, num_shards=geom.T, nc=3)
    got = sharded(psi)
    assert torch.allclose(got, expected, atol=ATOL)


def test_sharded_rejects_non_divisible_T():
    geom = LatticeGeometry(T=6, Z=4, Y=4, X=4)
    U = GaugeField.random(geom, seed=0).tensor
    with pytest.raises(ValueError, match="divisible"):
        _slice_shard_gauge(U, shard_idx=0, num_shards=4, dtype=torch.float32)


@pytest.mark.parametrize(
    "shape,expected",
    [
        ((4, 4, 4, 4), 1),         # V=256 well below cap
        ((16, 16, 16, 16), 1),     # V=65,536 below 80k cap → no shard needed
        ((24, 24, 24, 24), 8),     # V=331,776 → V_local=41,472 at n=8
        ((32, 32, 32, 32), 16),    # V=1,048,576 → V_local=65,536 at n=16
        ((48, 32, 32, 32), 24),    # V=2,359,296; T=48 not divisible by 32, falls
                                   # back to largest divisor ≤32 → 24 (V_local=98,304)
    ],
)
def test_auto_num_shards(shape, expected):
    assert _auto_num_shards(shape) == expected


@pytest.mark.parametrize("num_shards", [2, 4])
@pytest.mark.parametrize("op_cls", [WilsonDslash, WilsonDirac])
def test_batched_sharded_matches_full_volume(num_shards, op_cls):
    """Batched single-NEFF sharded path must equal the eager full-volume op."""
    geom = LatticeGeometry(T=8, Z=4, Y=4, X=4)
    shape = (geom.T, geom.Z, geom.Y, geom.X)
    U = GaugeField.random(geom, seed=42).tensor
    psi = ColorSpinorField.gaussian(geom, seed=7).tensor

    if op_cls is WilsonDirac:
        D = WilsonDirac(mass=0.1, nc=3, dtype=torch.complex64)
        diag = 4.0 + D.mass
    else:
        D = WilsonDslash(nc=3, dtype=torch.complex64)
        diag = 0.0

    expected = D(psi, U)

    T_local, num_shards = _shard_T_indices(geom.T, num_shards)
    batched_adapter = _BatchedUnbakedShardedAdapter(diag=diag, nc=3)
    wrapper = _ShardedDslashWrapper(
        [],
        num_shards=num_shards,
        T_local=T_local,
        compute_dtype=torch.float32,
        gauge_field=U,
        batched_module=batched_adapter,
    )
    got = wrapper(psi)

    assert got.shape == expected.shape
    assert torch.allclose(got, expected, atol=ATOL), (
        f"max abs diff = {(got - expected).abs().max().item():.3e}"
    )
