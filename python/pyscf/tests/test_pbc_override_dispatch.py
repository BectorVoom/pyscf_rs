"""Plan 20-11 — the k-point subclass-override contract (`KPyOverrideBridge`).

`crates/pyscf-py/src/pbc/kbridge.rs` implements `pyscf_pbc_scf::KOverrideHooks`
by dispatching every one of the eleven hooks through `slf.call_method1`, so a
Python subclass override is found by MRO. A hook the Python type does not
override runs the Rust default directly (no Python round trip); which hooks
are overridden is probed ONCE per `(type, native base)` and cached.

20-11 bound no driver; 20-12 added `KRHF` to `DRIVERS`. The contract is exercised now through a
PRIVATE harness, `pyscf._native.pbc.scf._KRHFBridgeSelftest(with_df)`: a
native base class whose eleven hook methods run the Rust `Krhf` defaults and
whose `kernel()` drives `pyscf_pbc_scf::kernel` through the bridge — exactly
the shape `KRHF` takes in 20-12. When `KRHF` lands, 20-12 adds it to
`DRIVERS` and every test below runs against both classes.

Fixture: He-fcc, all-electron, Bohr, mesh `[15]*3` (the 20-CONTEXT §2 cell)
but `cc-pvdz` (5 AO) on a 3×1×1 k-mesh. Both choices are load-bearing: with ONE
AO a row-/column-major transposition of the payload is invisible, and on a
TRIM mesh (any Γ-centred even mesh, e.g. 2×1×1) every k-matrix is real, so a
dropped or conjugated imaginary plane is invisible too (memory:
trim-meshes-make-phase-gates-vacuous). `test_fixture_is_not_vacuous` asserts
both preconditions.
"""

import numpy as np
import pytest

import pyscf._native.pbc.df as ndf
import pyscf._native.pbc.gto as ngto
import pyscf._native.pbc.scf as nscf

H_HE = 2.834589
CONV_TOL = 1e-10

HOOKS = [
    "get_ovlp",
    "get_hcore",
    "get_init_guess",
    "get_veff",
    "get_fock",
    "eig",
    "get_occ",
    "make_rdm1",
    "energy_elec",
    "energy_nuc",
    "get_grad",
]

# 20-12: the real `KRHF` runs every case too. It is constructed upstream-style
# (`KRHF(cell, kpts)`, then `mf.with_df = ...`) and configured by attribute, so
# `new_driver` / `run` / `run_reference` adapt the two shapes; the assertions are
# shared unchanged.
DRIVERS = [nscf._KRHFBridgeSelftest, nscf.KRHF]


def new_driver(cls, with_df):
    """An instance of `cls` (or a Python subclass of it) over `with_df`."""
    if issubclass(cls, nscf.KSCF):
        mf = cls(with_df.cell, with_df.kpts)
        mf.with_df = with_df
        return mf
    return cls(with_df)


def _result(mf):
    return {
        "e_tot": mf.e_tot,
        "e_elec": mf.e_elec,
        "e_coul": mf.e_coul,
        "e_nuc": mf.e_nuc,
        "converged": mf.converged,
        "cycles": mf.cycles,
        "mo_energy": mf.mo_energy,
        "mo_occ": mf.mo_occ,
        "overridden": mf._overridden_hooks,
    }


def run_reference(mf):
    """The Rust kernel with NO bridge at all (`Krhf::kernel`)."""
    if isinstance(mf, nscf.KSCF):
        mf.conv_tol = CONV_TOL
        mf._kernel_without_bridge()
        return _result(mf)
    return mf.kernel(conv_tol=CONV_TOL, use_bridge=False)


def fcc(h):
    return [[0.0, h, h], [h, 0.0, h], [h, h, 0.0]]


def bits(x):
    return np.asarray(x, dtype=np.float64).view(np.uint64)


@pytest.fixture(scope="module")
def with_df():
    cell = ngto.M(
        a=fcc(H_HE), atom=[("He", (0, 0, 0))], basis="cc-pvdz", unit="Bohr", mesh=[15, 15, 15]
    )
    kpts = cell.make_kpts([3, 1, 1])
    return ndf.FFTDF(cell, kpts)


@pytest.fixture(scope="module", params=DRIVERS, ids=lambda c: c.__name__)
def driver(request):
    return request.param


@pytest.fixture(scope="module")
def reference(driver, with_df):
    """The Rust kernel with NO bridge at all (`Krhf::kernel`)."""
    out = run_reference(new_driver(driver, with_df))
    assert out["converged"]
    return out


def run(mf):
    if isinstance(mf, nscf.KSCF):
        mf.conv_tol = CONV_TOL
        e = mf.kernel()
        assert e == mf.e_tot
        return _result(mf)
    return mf.kernel(conv_tol=CONV_TOL)


# ─── the fixture and the payload layout ──────────────────────────────────────


def test_fixture_is_not_vacuous(driver, with_df):
    mf = new_driver(driver, with_df)
    s = mf.get_ovlp()
    assert len(s) == 3 and s[0].shape == (5, 5)
    assert max(abs(x.imag).max() for x in s) > 1e-3, "all-real k-matrices: TRIM mesh"
    e, c = mf.eig(mf.get_hcore(), s)
    assert max(abs(x - x.T).max() for x in c) > 1e-3, "symmetric mo_coeff hides a transpose"


def test_kmats_payload_is_row_major_and_keeps_the_phase(driver, with_df):
    """`get_ovlp` through the codec == the independently laid-out 20-09 integral.

    Restated 2026-09-14 (20-18): since 20-19 D the SCF `get_ovlp` hook follows
    upstream `pbc/scf/hf.py:47-55` — `pbc_intor('int1e_ovlp', hermi=0)` at
    `precision * 1e-5` with `rcut = max(cell.rcut, estimate_rcut(precision * 1e-5))`
    — so the reference is laid out on a cell built at that precision (its
    auto-`rcut` is the larger of the two here: 15.85 vs 13.06 Bohr). Against the
    plain-precision integral the payload now differs by 1.01e-10, which is the
    precision rule, not the layout; against this reference it is 0.0. The
    tolerance is unchanged.
    """
    cell = with_df.cell
    tight = ngto.M(
        a=fcc(H_HE), atom=[("He", (0, 0, 0))], basis="cc-pvdz", unit="Bohr",
        mesh=[15, 15, 15], precision=cell.precision * 1e-5,
    )
    assert tight.rcut >= cell.rcut
    ref = tight.pbc_intor("int1e_ovlp", hermi=0, kpts=with_df.kpts)
    got = new_driver(driver, with_df).get_ovlp()
    for a, b in zip(got, ref):
        assert a.dtype == np.complex128 and a.shape == b.shape
        assert np.abs(a - b).max() < 1e-14
    # Γ is real symmetric; the other two blocks are complex Hermitian, where a
    # transpose (= conjugation) or a dropped imaginary plane would be caught.
    assert max(np.abs(a - b.T).max() for a, b in zip(got[1:], ref[1:])) > 1e-3


def test_mo_coeff_payload_is_column_major(driver, with_df):
    """`eig` output satisfies F c = S c e column by column, so c[:, i] is MO i."""
    mf = new_driver(driver, with_df)
    s = mf.get_ovlp()
    f = mf.get_hcore()
    e, c = mf.eig(f, s)
    for fk, sk, ek, ck in zip(f, s, e, c):
        assert ck.shape == (5, 5) and ek.shape == (5,)
        assert np.abs(fk @ ck - sk @ ck * ek).max() < 1e-10
        assert np.abs(fk @ ck.T - sk @ ck.T * ek).max() > 1e-3


# ─── the negative contract: nothing overridden → the Rust default, bitwise ───


def test_unsubclassed_driver_probes_no_override(driver, with_df, reference):
    out = run(new_driver(driver, with_df))
    assert out["overridden"] == []
    assert bits(out["e_tot"]) == bits(reference["e_tot"])


def test_subclass_overriding_nothing_is_bitwise_the_rust_default(driver, with_df, reference):
    class Plain(driver):
        pass

    class AlsoPlain(Plain):
        def unrelated(self):
            return 0

    for cls in (Plain, AlsoPlain):
        out = run(new_driver(cls, with_df))
        assert out["overridden"] == []
        assert out["converged"]
        assert bits(out["e_tot"]) == bits(reference["e_tot"]), (out["e_tot"], reference["e_tot"])
        assert out["cycles"] == reference["cycles"]


# ─── one test per hook: the override is reached, via MRO ─────────────────────


def counting_subclass(driver, hook, counter):
    def wrapper(self, *args, **kwargs):
        counter["n"] += 1
        return getattr(super(cls, self), hook)(*args, **kwargs)

    cls = type(f"Counted_{hook}", (driver,), {hook: wrapper})
    return cls


@pytest.mark.parametrize("hook", HOOKS)
def test_each_hook_override_is_dispatched(driver, with_df, reference, hook):
    counter = {"n": 0}
    cls = counting_subclass(driver, hook, counter)
    out = run(new_driver(cls, with_df))
    assert out["overridden"] == [hook]
    assert counter["n"] > 0, f"{hook} override never invoked"
    assert out["converged"]
    # A pass-through override round-trips the complex per-k payload both ways
    # (Rust -> numpy -> Rust); the conversions are element moves, so the
    # energy is the Rust default's to the bit.
    assert bits(out["e_tot"]) == bits(reference["e_tot"]), (hook, out["e_tot"], reference["e_tot"])


def test_override_found_through_an_intermediate_python_class(driver, with_df, reference):
    counter = {"n": 0}
    Mid = counting_subclass(driver, "get_veff", counter)

    class Leaf(Mid):
        pass

    out = run(new_driver(Leaf, with_df))
    assert out["overridden"] == ["get_veff"]
    assert counter["n"] >= out["cycles"]


def test_all_hooks_overridden_at_once_bitwise(driver, with_df, reference):
    counter = {"n": 0}
    body = {}
    for hook in HOOKS:

        def make(h):
            def wrapper(self, *args, **kwargs):
                counter["n"] += 1
                return getattr(super(Everything, self), h)(*args, **kwargs)

            return wrapper

        body[hook] = make(hook)
    Everything = type("Everything", (driver,), body)
    out = run(new_driver(Everything, with_df))
    assert sorted(out["overridden"]) == sorted(HOOKS)
    assert bits(out["e_tot"]) == bits(reference["e_tot"])
    assert counter["n"] >= 11


def test_get_veff_counted_per_cycle(driver, with_df):
    counter = {"n": 0}
    out = run(new_driver(counting_subclass(driver, "get_veff", counter), with_df))
    # once for the initial guess, once per cycle
    assert counter["n"] == out["cycles"] + 1


# ─── the override CHANGES the answer the way the physics says ────────────────


def test_get_hcore_shift_moves_energy_by_c_times_nelectron(driver, with_df, reference):
    c = 0.125

    class Shifted(driver):
        def get_hcore(self, cell=None):
            s = self.get_ovlp()
            return [h + c * sk for h, sk in zip(super().get_hcore(cell), s)]

    out = run(new_driver(Shifted, with_df))
    assert out["overridden"] == ["get_hcore"]
    # A uniform c*S shift leaves the density alone: dE = c * tr(D S) = c * N (He: 2).
    assert abs(out["e_tot"] - (reference["e_tot"] + 2 * c)) < 1e-8
    for e, e0 in zip(out["mo_energy"], reference["mo_energy"]):
        assert np.allclose(e, np.asarray(e0) + c, atol=1e-8)


def test_get_veff_shift_moves_energy_by_half_c_times_nelectron(driver, with_df, reference):
    c = 0.25
    counter = {"n": 0}

    class Shifted(driver):
        def get_veff(self, cell=None, dm_kpts=None):
            counter["n"] += 1
            s = self.get_ovlp()
            return [v + c * sk for v, sk in zip(super().get_veff(cell, dm_kpts), s)]

    out = run(new_driver(Shifted, with_df))
    assert counter["n"] > 0
    # e_coul = 1/2 tr(D V): dE = c * N / 2 = c for He.
    assert abs(out["e_tot"] - (reference["e_tot"] + c)) < 1e-8
    for e, e0 in zip(out["mo_energy"], reference["mo_energy"]):
        assert np.allclose(e, np.asarray(e0) + c, atol=1e-8)


def test_energy_nuc_override_enters_e_tot(driver, with_df, reference):
    class NoNuc(driver):
        def energy_nuc(self):
            return 0.0

    out = run(new_driver(NoNuc, with_df))
    assert out["e_nuc"] == 0.0
    assert bits(out["e_tot"]) == bits(out["e_elec"])
    assert bits(out["e_elec"]) == bits(reference["e_elec"])


def test_upstream_shaped_stacked_real_return_is_accepted(driver, with_df, reference):
    """Upstream returns `(nkpts, nao, nao)` ndarrays; real input widens exactly."""

    class Stacked(driver):
        def get_ovlp(self, cell=None):
            return np.asarray(super().get_ovlp(cell))

        def energy_elec(self, dm_kpts=None, h1e_kpts=None, vhf_kpts=None):
            e, ec = super().energy_elec(np.asarray(dm_kpts), h1e_kpts, vhf_kpts)
            return np.float64(e), np.float64(ec)

    out = run(new_driver(Stacked, with_df))
    assert bits(out["e_tot"]) == bits(reference["e_tot"])


def test_instance_attribute_override_is_seen(driver, with_df, reference):
    mf = new_driver(driver, with_df)
    calls = {"n": 0}

    def energy_nuc():
        calls["n"] += 1
        return 1.5

    mf.energy_nuc = energy_nuc
    out = run(mf)
    assert out["overridden"] == ["energy_nuc"]
    assert calls["n"] == 1
    assert bits(out["e_tot"]) == bits(out["e_elec"] + 1.5)


# ─── exceptions propagate as Python exceptions, never a panic ────────────────


class Boom(Exception):
    pass


@pytest.mark.parametrize("hook", HOOKS)
def test_exception_in_any_override_propagates_unchanged(driver, with_df, hook):
    def raiser(self, *args, **kwargs):
        raise Boom(f"boom from {hook}")

    cls = type(f"Raise_{hook}", (driver,), {hook: raiser})
    with pytest.raises(Boom, match=f"boom from {hook}"):
        run(new_driver(cls, with_df))


def test_exception_after_some_cycles_propagates(driver, with_df):
    state = {"n": 0}

    class LateFailure(driver):
        def get_grad(self, *args, **kwargs):
            state["n"] += 1
            if state["n"] == 2:
                raise KeyError("late")
            return super().get_grad(*args, **kwargs)

    with pytest.raises(KeyError, match="late"):
        run(new_driver(LateFailure, with_df))
    # get_grad is infallible in the trait: the bridge poisons itself so the
    # kernel stops at the next hook instead of running to max_cycle.
    assert state["n"] == 2


@pytest.mark.parametrize(
    "hook, bad",
    [
        ("get_hcore", lambda self, cell=None: [np.zeros((2, 2))] * 2),  # wrong nao
        ("get_ovlp", lambda self, cell=None: [np.eye(1)]),  # wrong nkpts
        ("get_veff", lambda self, cell=None, dm_kpts=None: "not a matrix list"),
        ("eig", lambda self, h, s: ([np.zeros(1)] * 2,)),  # not a pair
        ("get_occ", lambda self, e, c=None: [np.zeros(3)] * 2),  # wrong nmo
        ("energy_elec", lambda self, *a: 1.0),  # not a pair
        ("get_grad", lambda self, *a: "grad"),
    ],
)
def test_malformed_override_return_raises_not_panics(driver, with_df, hook, bad):
    cls = type(f"Bad_{hook}", (driver,), {hook: bad})
    with pytest.raises((TypeError, ValueError)):
        run(new_driver(cls, with_df))


# ─── the probe is cached per Python type ─────────────────────────────────────


def test_override_probe_runs_once_per_type(driver, with_df):
    counter = {"n": 0}
    cls = counting_subclass(driver, "energy_nuc", counter)
    before = nscf._kbridge_probe_count()
    run(new_driver(cls, with_df))
    after_first = nscf._kbridge_probe_count()
    assert after_first == before + 1
    run(new_driver(cls, with_df))
    run(new_driver(cls, with_df))
    assert nscf._kbridge_probe_count() == after_first
    assert counter["n"] == 3


def test_krhf_is_a_bridge_driver():
    assert nscf.KRHF in DRIVERS, "20-12: add nscf.KRHF to DRIVERS"
