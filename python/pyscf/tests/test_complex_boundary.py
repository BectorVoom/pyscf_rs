"""Plan 20-07 — the complex, k-resolved NumPy boundary, through a live interpreter.

`crates/pyscf-py/tests/complex_roundtrip.rs` proves the pyo3-free core bitwise.
This file covers the numpy wrappers that core sits under
(`numpy_io::{to_ctensor, ctensor_to_pyarray, to_kmats, kmats_to_pylist,
to_kdms, kdms_to_pylist}`) via the private `_native._roundtrip_*` / `_planes`
hooks. The default `abi3-py310` feature turns on `pyo3/extension-module`, so no
Rust test binary can host an interpreter — hence the split.

Every comparison is BITWISE (`tobytes()` / `.view(np.uint64)`), never an
epsilon: the conversion is a pure element move, same implementation both ways.
"""
from __future__ import annotations

import numpy as np
import pytest

from pyscf import _native


def _rand_complex(shape, seed):
    rng = np.random.default_rng(seed)
    a = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    flat = a.reshape(-1)
    specials = [0.0, -0.0, 5e-324, -5e-324, 1.0 + np.finfo(float).eps, -1e308]
    for i, v in enumerate(specials[: flat.size]):
        flat[i] = complex(v, specials[-1 - i])
    return a


def _bits_equal(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    return (
        a.shape == b.shape
        and a.dtype == b.dtype == np.complex128
        and np.array_equal(a.real.view(np.uint64), b.real.view(np.uint64))
        and np.array_equal(a.imag.view(np.uint64), b.imag.view(np.uint64))
    )


STRIDE_FUZZ = {
    "a": lambda a: a,
    "a.T": lambda a: a.T,
    "a[::2]": lambda a: a[::2],
    "a[:,1:5]": lambda a: a[:, 1:5],
    "a[::-1,::3]": lambda a: a[::-1, ::3],
    "asfortranarray(a)": np.asfortranarray,
}


@pytest.mark.parametrize("order", ["C", "F"])
@pytest.mark.parametrize("view", list(STRIDE_FUZZ), ids=list(STRIDE_FUZZ))
def test_ctensor_round_trip_bitwise_any_strides(order, view):
    base = _rand_complex((6, 7), seed=1)
    a = STRIDE_FUZZ[view](base)
    out = _native._roundtrip_ctensor(a, order)
    assert _bits_equal(out, a), f"{view} {order}: round trip changed bits or transposed"
    if order == "F":
        assert out.flags.f_contiguous
    else:
        assert out.flags.c_contiguous


def test_planes_follow_the_requested_flattening_order():
    a = _rand_complex((3, 4, 2), seed=2)
    for order in ("C", "F"):
        re, im = _native._planes(a, order)
        want = a.reshape(-1, order=order)
        assert np.array_equal(re.view(np.uint64), want.real.copy().view(np.uint64))
        assert np.array_equal(im.view(np.uint64), want.imag.copy().view(np.uint64))
    # a transposed input reads by logical index: planes of a.T in C order are
    # a's planes in F order for 2-D — never a's raw memory.
    b = _rand_complex((5, 3), seed=3)
    re_t, _ = _native._planes(b.T, "C")
    re_f, _ = _native._planes(b, "F")
    assert np.array_equal(re_t.view(np.uint64), re_f.view(np.uint64))


def test_non_complex128_is_rejected_not_cast():
    with pytest.raises(TypeError):
        _native._roundtrip_ctensor(np.zeros((2, 2)), "C")
    with pytest.raises(TypeError):
        _native._roundtrip_ctensor(np.zeros((2, 2), dtype=np.complex64), "C")
    with pytest.raises(ValueError):
        _native._roundtrip_ctensor(np.zeros((2, 2), dtype=complex), "X")


def test_kmats_eight_blocks_round_trip_as_a_list_of_eight_arrays():
    blocks = [_rand_complex((4, 4), seed=10 + k) for k in range(8)]
    blocks[3] = blocks[3].T  # non-contiguous member
    out = _native._roundtrip_kmats(blocks, "C")
    assert isinstance(out, list) and len(out) == 8
    for k, (a, b) in enumerate(zip(blocks, out)):
        assert _bits_equal(a, b), f"k={k}"


def test_ksymm_blocks_with_different_shapes_round_trip():
    blocks = [_rand_complex((5, 3), seed=20), _rand_complex((5, 2), seed=21)]
    with pytest.raises(ValueError):
        np.stack(blocks)  # the stacked form cannot hold this data
    out = _native._roundtrip_kmats(tuple(blocks), "F")
    assert [o.shape for o in out] == [(5, 3), (5, 2)]
    for a, b in zip(blocks, out):
        assert _bits_equal(a, b)


def test_stacked_ndarray_is_rejected_for_kmats():
    stacked = _rand_complex((8, 4, 4), seed=30)
    with pytest.raises(TypeError, match="list"):
        _native._roundtrip_kmats(stacked, "C")
    # the documented spelling for an upstream (nkpts, nao, nao) array
    out = _native._roundtrip_kmats(list(stacked), "C")
    assert all(_bits_equal(a, b) for a, b in zip(stacked, out))


def test_kdms_spin_axis_is_a_nested_list():
    kdms = [[_rand_complex((3, 3), seed=40 + 4 * s + k) for k in range(4)] for s in range(2)]
    out = _native._roundtrip_kdms(kdms, "C")
    assert isinstance(out, list) and len(out) == 2
    for s in range(2):
        assert isinstance(out[s], list) and len(out[s]) == 4
        for k in range(4):
            assert _bits_equal(kdms[s][k], out[s][k]), f"set={s} k={k}"
    # a flat KMats list is not a KDms: each set must itself be a list
    with pytest.raises(TypeError):
        _native._roundtrip_kdms(kdms[0], "C")
