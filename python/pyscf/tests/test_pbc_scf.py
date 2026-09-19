"""Plan 20-12 — the periodic SCF drivers (`pyscf._native.pbc.scf`).

Fixture: He-fcc, all-electron, Bohr (the 20-CONTEXT §2 cell). `sto-3g` (1 AO)
on a 2×2×2 mesh with the FFTDF mesh pinned to `[15]*3` is EXACTLY the fixture
of `crates/pyscf-pbc-scf/tests/krhf_bands_oracle.rs`, so its numbers are
reproduced through the binding; `cc-pvdz` (5 AO) is used where one AO would
make a check vacuous (smearing needs virtual orbitals; a basis projection
needs two different AO counts).

"Bitwise vs the Rust kernel" is established two ways (no Rust reference binary
is built from here — memory: gate-target-dir-lto-spelling):

1. same-implementation A/B, to the bit: `KRHF.kernel()` (through 20-11's
   `KPyOverrideBridge`) against `KRHF._kernel_without_bridge()`, which calls
   the concrete `Krhf::kernel` with no bridge and no Python;
2. against the value `krhf_bands_oracle.rs` PRINTS for the identical fixture
   and settings — `rust -2.807388116559753` (`{:.15}`, post-20-19-D,
   `target/p20-19-D/after/scf.log`; pre-D -2.807388116559963,
   `target/gate-20-12-bands.log`) — compared at that printed precision.

Upstream (vendored PySCF 2.12.1, subprocess, version asserted) is compared at
the measured floors (measurements/README.md): `e_tot` 2.18e-13 → gate 1e-12;
`mo_energy` 6.10e-11 and `get_bands` 1.68e-11 → gate 1e-9 (memory:
band-energies-are-never-bitwise-identical; 0/10 bitwise).
"""

import json
import os
import subprocess
import sys

import numpy as np
import pytest

import pyscf._native.pbc.df as ndf
import pyscf._native.pbc.gto as ngto
import pyscf._native.pbc.scf as nscf
import pyscf._native.pbc.symm as nsymm

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
H_HE = 2.834589
MESH = [15, 15, 15]
# `krhf_bands_oracle.rs` prints `rust {:.15}` for this fixture. Re-referenced
# 2026-09-14 (20-18) after 20-19 D's SCF overlap-precision change moved the bits:
# `target/p20-19-D/after/scf.log` "KRHF He 2x2x2: rust -2.807388116559753" (the
# pre-D print was -2.807388116559963, `target/gate-20-12-bands.log`); also
# `crates/pyscf-pbc-scf/tests/krhf_threads.rs` prints -2.80738811655975251e0.
RUST_PRINTED_E_TOT = "-2.807388116559753"
KBAND_SCALED = [[0.15, -0.07, 0.03], [-0.05, 0.11, 0.02]]


def fcc(h):
    return [[0.0, h, h], [h, 0.0, h], [h, h, 0.0]]


def helium(basis="sto-3g", **kw):
    return ngto.M(a=fcc(H_HE), atom=[("He", (0, 0, 0))], basis=basis, unit="Bohr", **kw)


def bits(a):
    a = np.ascontiguousarray(a)
    if np.iscomplexobj(a):
        a = a.view(np.float64)
    return np.asarray(a, dtype=np.float64).view(np.uint64)


def same_bits(xs, ys):
    xs = xs if isinstance(xs, list) else [xs]
    ys = ys if isinstance(ys, list) else [ys]
    return len(xs) == len(ys) and all(
        np.shape(x) == np.shape(y) and (bits(x) == bits(y)).all() for x, y in zip(xs, ys)
    )


def tight(mf):
    """`krhf_bands_oracle.rs::tight()` — and upstream's script settings."""
    mf.conv_tol = 1e-12
    mf.conv_tol_grad = 1e-8
    mf.max_cycle = 60
    return mf


def he_krhf(cls=nscf.KRHF, basis="sto-3g", nk=(2, 2, 2)):
    cell = helium(basis)
    kpts = cell.make_kpts(list(nk))
    mf = cls(cell, kpts)
    mf.with_df.mesh = MESH
    return tight(mf)


# ─── upstream oracle (one subprocess for every upstream number) ──────────────

UPSTREAM_SCRIPT = r"""
import json, sys
import numpy as np
from pyscf.pbc import gto, scf
from pyscf.pbc.scf import addons

h, mesh, kband_scaled, mo1 = json.loads(sys.argv[1])

def cell_of(basis):
    c = gto.Cell()
    c.a = [[0.0, h, h], [h, 0.0, h], [h, h, 0.0]]
    c.atom = [('He', (0.0, 0.0, 0.0))]
    c.basis = basis
    c.unit = 'Bohr'
    c.verbose = 0
    c.build()
    return c

def tight(mf):
    mf.with_df.mesh = mesh
    mf.conv_tol = 1e-12
    mf.conv_tol_grad = 1e-8
    mf.max_cycle = 60
    return mf

c = cell_of('sto-3g')
kpts = c.make_kpts([2, 2, 2])
mf = tight(scf.KRHF(c, kpts))
e = mf.kernel()
kband = c.get_abs_kpts(np.asarray(kband_scaled))
e_band, _ = mf.get_bands(kband)

cd = cell_of('cc-pvdz')
kd = cd.make_kpts([2, 2, 2])
sm = tight(addons.smearing_(scf.KRHF(cd, kd), sigma=0.1, method='fermi'))
sm.kernel()

k3 = c.make_kpts([3, 1, 1])
mo = [np.asarray(re) + 1j * np.asarray(im) for re, im in mo1]
proj = addons.project_mo_nr2nr(c, mo, cd, k3)

print(json.dumps({
    'version': __import__('pyscf').__version__,
    'e_tot': float(e), 'converged': bool(mf.converged),
    'e_mo': [np.asarray(x).tolist() for x in mf.mo_energy],
    'kband': np.asarray(kband).tolist(),
    'e_band': [np.asarray(x).tolist() for x in e_band],
    'smear': {'e_tot': float(sm.e_tot), 'e_free': float(sm.e_free),
              'e_zero': float(sm.e_zero), 'converged': bool(sm.converged)},
    'proj_re': [np.asarray(p).real.tolist() for p in proj],
    'proj_im': [np.asarray(p).imag.tolist() for p in proj],
}))
"""


def pseudo_mo(nk, nao, nmo, seed=7):
    rng = np.random.default_rng(seed)
    return [rng.standard_normal((nao, nmo)) + 1j * rng.standard_normal((nao, nmo)) for _ in range(nk)]


@pytest.fixture(scope="module")
def upstream():
    mo1 = [(m.real.tolist(), m.imag.tolist()) for m in pseudo_mo(3, 1, 1)]
    arg = json.dumps([H_HE, MESH, KBAND_SCALED, mo1])
    env = dict(os.environ, PYTHONPATH=REPO)
    proc = subprocess.run([sys.executable, "-c", UPSTREAM_SCRIPT, arg], cwd=REPO, env=env,
                          capture_output=True, text=True, check=False)
    assert proc.returncode == 0, proc.stderr
    up = json.loads([ln for ln in proc.stdout.splitlines() if ln.startswith("{")][-1])
    assert up["version"] == "2.12.1", "the oracle must be the VENDORED PySCF 2.12.1"
    assert up["converged"] and up["smear"]["converged"]
    return up


@pytest.fixture(scope="module")
def krhf_he():
    mf = he_krhf()
    e = mf.kernel()
    assert mf.converged
    return mf, e


# ─── Task 1: the identity of the drivers, and KRHF bitwise ───────────────────


def test_classes_mirror_upstream_hierarchy():
    assert issubclass(nscf.KRHF, nscf.KSCF) and issubclass(nscf.KUHF, nscf.KSCF)
    assert issubclass(nscf.KROHF, nscf.KRHF) and issubclass(nscf.KGHF, nscf.KSCF)
    assert issubclass(nscf.KsymAdaptedKRHF, nscf.KRHF)
    assert not issubclass(nscf.KUHF, nscf.KRHF)
    import pyscf.pbc.scf as oscf

    for name in ("KRHF", "KUHF", "KROHF", "KGHF", "KSCF", "KsymAdaptedKRHF"):
        assert getattr(oscf, name) is getattr(nscf, name)


def test_krhf_energy_is_the_rust_kernel_bitwise(krhf_he):
    mf, e = krhf_he
    assert e == mf.e_tot
    # (1) the concrete `Krhf::kernel`, no bridge, fresh object — to the bit.
    ref = he_krhf()
    e_ref = ref._kernel_without_bridge()
    assert bits(e) == bits(e_ref), (e, e_ref)
    assert mf.cycles == ref.cycles
    assert same_bits(mf.mo_energy, ref.mo_energy)
    assert same_bits(mf.mo_coeff, ref.mo_coeff)
    # (2) the value `krhf_bands_oracle.rs` prints for this fixture.
    assert f"{e:.15f}" == RUST_PRINTED_E_TOT, f"{e!r} vs rust {RUST_PRINTED_E_TOT}"
    assert mf._overridden_hooks == []


def test_krhf_energy_matches_upstream(krhf_he, upstream):
    mf, e = krhf_he
    de = abs(e - upstream["e_tot"])
    print(f"KRHF He 2x2x2 through the binding: |dE| = {de:e}")
    assert de < 1e-12, de


def test_config_attributes_round_trip():
    cell = helium()
    mf = nscf.KRHF(cell, cell.make_kpts([2, 2, 2]))
    # khf.py:483 — max(cell.precision * 10, 1e-8); precision defaults to 1e-8.
    assert mf.conv_tol == max(cell.precision * 10, 1e-8)
    assert mf.conv_tol_grad is None and mf.max_cycle == 50
    assert mf.diis is True and mf.diis_space == 8 and mf.diis_start_cycle == 1
    assert mf.init_guess == "minao" and mf.exxdiv == "ewald" and mf.chkfile is None
    for name, value in [("conv_tol", 3.5e-11), ("conv_tol_grad", 2e-7), ("max_cycle", 17),
                        ("diis_space", 6), ("damp", 0.25), ("level_shift", 0.1),
                        ("init_guess", "1e"), ("exxdiv", None), ("diis", False)]:
        setattr(mf, name, value)
        assert getattr(mf, name) == value
    with pytest.raises(NotImplementedError):
        mf.init_guess = "huckel"
    with pytest.raises(ValueError):
        mf.exxdiv = "bogus"
    assert mf.e_tot == 0.0 and mf.mo_coeff is None and mf.converged is False


def test_run_sets_attributes_and_returns_self():
    mf = he_krhf()
    out = mf.run(conv_tol=1e-11, max_cycle=40)
    assert out is mf and mf.conv_tol == 1e-11 and mf.max_cycle == 40 and mf.converged


def test_result_surface(krhf_he):
    mf, e = krhf_he
    assert len(mf.mo_coeff) == 8 and mf.mo_coeff[0].dtype == np.complex128
    assert mf.mo_coeff[0].shape == (1, 1) and mf.mo_energy[0].shape == (1,)
    assert all(o.tolist() == [2.0] for o in mf.mo_occ)
    assert bits(mf.e_tot) == bits(mf.e_elec + mf.e_nuc)
    assert bits(mf.e_nuc) == bits(mf.energy_nuc())
    assert len(mf.fermi) == 1 and mf.cycles >= 1
    # The hooks' defaults reproduce the stored state.
    dm = mf.make_rdm1()
    e1, _ = mf.energy_elec(dm)
    assert abs(e1 - mf.e_elec) < 1e-13
    g = mf.get_grad(mf.mo_coeff, mf.mo_occ)
    assert g.size == 0  # 1 AO, all occupied: no occupied-virtual block


def test_with_df_is_the_python_object_and_reassignment_is_seen():
    cell = helium()
    kpts = cell.make_kpts([2, 2, 2])
    mf = nscf.KRHF(cell, kpts)
    assert isinstance(mf.with_df, ndf.FFTDF) and mf.cell is cell
    assert np.array_equal(mf.kpts, kpts)
    gdf = ndf.GDF(cell, kpts)
    mf.with_df = gdf
    assert mf.with_df is gdf
    with pytest.raises(TypeError):
        mf.with_df = object()


# ─── Task 4: get_bands, gamma shims, chkfile, smearing ───────────────────────


def test_get_bands_matches_upstream(krhf_he, upstream):
    mf, _ = krhf_he
    kband = mf.cell.get_abs_kpts(np.asarray(KBAND_SCALED))
    assert np.abs(kband - np.asarray(upstream["kband"])).max() < 1e-14
    e_band, c_band = mf.get_bands(kband)
    assert len(e_band) == 2 and c_band[0].dtype == np.complex128
    d_mo = max(np.abs(np.asarray(a) - b).max() for a, b in zip(mf.mo_energy, upstream["e_mo"]))
    d_band = max(np.abs(np.asarray(a) - b).max() for a, b in zip(e_band, upstream["e_band"]))
    n_bitwise = sum(int(np.asarray(a)[0] == b[0]) for a, b in zip(e_band, upstream["e_band"]))
    print(f"mo_energy (on mesh) |d| = {d_mo:e}; get_bands (off mesh) |d| = {d_band:e}, "
          f"{n_bitwise}/2 bitwise")
    assert d_mo < 1e-9, d_mo
    assert d_band < 1e-9, d_band
    # A single (3,) k-point returns single arrays.
    e1, c1 = mf.get_bands(kband[0])
    assert e1.shape == (1,) and c1.shape == (1, 1)
    assert bits(e1) == bits(e_band[0])


def test_get_bands_on_the_scf_mesh_reproduce_mo_energy(krhf_he):
    mf, _ = krhf_he
    e_band, _ = mf.get_bands(mf.kpts)
    d = max(np.abs(a - b).max() for a, b in zip(e_band, mf.mo_energy))
    assert d < 1e-10, d


def test_gamma_shims_are_one_kpoint_drivers():
    import pyscf.pbc.scf as oscf

    cell = helium()
    g = oscf.RHF(cell)
    assert type(g) is nscf.KRHF and np.array_equal(g.kpts, np.zeros((1, 3)))
    g.with_df.mesh = MESH
    ref = nscf.KRHF(cell, np.zeros((1, 3)))
    ref.with_df.mesh = MESH
    assert bits(g.kernel()) == bits(ref._kernel_without_bridge())
    assert type(oscf.UHF(cell)) is nscf.KUHF and type(oscf.GHF(cell)) is nscf.KGHF
    assert type(oscf.ROHF(cell)) is nscf.KROHF
    assert type(oscf.RHF(cell, kpts=cell.make_kpts([2, 1, 1]))) is nscf.KRHF
    assert type(oscf.KHF(cell, cell.make_kpts([2, 1, 1]))) is nscf.KRHF


def test_chkfile_dump_load_round_trip(tmp_path, krhf_he):
    import h5py

    path = str(tmp_path / "krhf.chk")
    mf = he_krhf()
    mf.chkfile = path
    e = mf.kernel()
    cell, rec = nscf.load_scf(path)
    assert bits(rec["e_tot"]) == bits(e)
    assert np.array_equal(rec["kpts"], mf.kpts)
    assert same_bits(rec["mo_coeff"], mf.mo_coeff)
    assert same_bits(rec["mo_energy"], mf.mo_energy)
    assert same_bits(rec["mo_occ"], mf.mo_occ)
    assert isinstance(cell, ngto.Cell) and cell.nao_nr() == 1
    with h5py.File(path, "r") as f:
        assert f["scf/mo_coeff"].dtype == np.complex128
    # dump_chk to another path writes the same record.
    other = str(tmp_path / "again.chk")
    assert mf.dump_chk(other) is mf
    assert bits(nscf.load_scf(other)[1]["e_tot"]) == bits(e)
    # Restart: the stored density, and a converged restart.
    dm = mf.init_guess_by_chkfile()
    assert same_bits(dm, mf.make_rdm1())
    re = he_krhf()
    re.chkfile = path
    re.init_guess = "chkfile"
    e2 = re.kernel()
    assert re.converged and re.cycles <= 3 and abs(e2 - e) < 1e-11, (re.cycles, e2 - e)


def test_chkfile_restart_projects_across_a_basis_change(tmp_path):
    path = str(tmp_path / "sto3g.chk")
    small = he_krhf(basis="sto-3g")
    small.chkfile = path
    small.kernel()
    big = he_krhf(basis="cc-pvdz")
    dm = big.init_guess_by_chkfile(path)  # nao 1 -> 5: project_mo_nr2nr
    assert len(dm) == 8 and dm[0].shape == (5, 5)
    s = big.get_ovlp()
    ne = sum(np.einsum("ij,ji->", d, sk).real for d, sk in zip(dm, s)) / 8
    assert 0.0 < ne <= 2.0 + 1e-10, ne


def test_smearing_matches_upstream(upstream):
    mf = he_krhf(basis="cc-pvdz")
    assert nscf.smearing_(mf, sigma=0.1, method="fermi") is mf
    assert mf.sigma == 0.1 and mf.smearing_method == "fermi"
    mf.kernel()
    assert mf.converged
    up = upstream["smear"]
    d = {k: abs(getattr(mf, k) - up[k]) for k in ("e_tot", "e_free", "e_zero")}
    print("smearing KRHF He cc-pvdz 2x2x2 sigma=0.1:", {k: f"{v:e}" for k, v in d.items()})
    assert all(v < 1e-9 for v in d.values()), d
    assert mf.entropy > 0 and abs(mf.e_free - (mf.e_tot - 0.1 * mf.entropy)) < 1e-12
    # the smeared Rust kernel through the bridge == without it, to the bit
    ref = nscf.smearing_(he_krhf(basis="cc-pvdz"), sigma=0.1, method="fermi")
    assert bits(ref._kernel_without_bridge()) == bits(mf.e_tot)
    with pytest.raises(NotImplementedError):
        he_krhf(nscf.KROHF).smearing_(sigma=0.1)


# ─── Task 5: project_mo_nr2nr ────────────────────────────────────────────────


def test_project_mo_nr2nr_matches_upstream(upstream):
    c1, c2 = helium("sto-3g"), helium("cc-pvdz")
    k3 = c1.make_kpts([3, 1, 1])
    mo1 = pseudo_mo(3, 1, 1)
    got = nscf.project_mo_nr2nr(c1, mo1, c2, k3)
    assert len(got) == 3 and got[0].shape == (5, 1)
    d = max(max(np.abs(g.real - np.asarray(r)).max(), np.abs(g.imag - np.asarray(i)).max())
            for g, r, i in zip(got, upstream["proj_re"], upstream["proj_im"]))
    print(f"project_mo_nr2nr sto-3g -> cc-pvdz vs upstream: |d| = {d:e}")
    assert d < 1e-9, d
    one = nscf.project_mo_nr2nr(c1, mo1[0], c2)  # kpts=None: gamma, one array
    assert one.shape == (5, 1)


# ─── Task 6: the KPoints dispatch ────────────────────────────────────────────


@pytest.fixture(scope="module")
def he_symm():
    cell = helium(space_group_symmetry=True, symmorphic=True)
    # time_reversal_symmetry=False: with it, `little_cogroup_ops` indexes the
    # 2*nop column space and `use_ao_symmetry=True` refuses (upstream raises
    # IndexError) — khf_ksymm.rs::TIME_REVERSAL, 17-07.
    kp = cell.make_kpts([2, 2, 2], space_group_symmetry=True, time_reversal_symmetry=False)
    assert isinstance(kp, nsymm.KPoints) and kp.nkpts_ibz < kp.nkpts
    return cell, kp


@pytest.mark.parametrize("cls, upstream_file", [(nscf.KUHF, "kuhf_ksymm.py"),
                                                (nscf.KGHF, "kghf_ksymm.py")])
def test_kuhf_kghf_with_kpoints_raise(he_symm, cls, upstream_file):
    import pyscf.pbc.scf as oscf

    cell, kp = he_symm
    for call in (lambda: cls(cell, kpts=kp), lambda: cls(cell, kp),
                 lambda: getattr(oscf, cls.__name__)(cell, kpts=kp)):
        with pytest.raises(NotImplementedError, match=f"not implemented in pyscf-rs.*{upstream_file}"):
            call()
    with pytest.raises(NotImplementedError):
        oscf.UHF(cell, kpts=kp)


def test_unported_ksymm_modules_are_not_silently_served():
    import importlib

    import pyscf.pbc.scf as oscf

    for name in ("kuhf_ksymm", "kghf_ksymm"):
        mod = importlib.import_module(f"pyscf.pbc.scf.{name}")
        assert getattr(mod, "__file__", None) is None, "upstream file was loaded"
        assert getattr(oscf, name) is mod
        with pytest.raises(NotImplementedError, match="not implemented in pyscf-rs"):
            mod.KUHF if name == "kuhf_ksymm" else mod.KGHF
    ns = {}
    with pytest.raises(NotImplementedError):
        exec("from pyscf.pbc.scf.kuhf_ksymm import KUHF", ns)


def test_krhf_with_kpoints_runs_the_ksymm_driver(he_symm):
    cell, kp = he_symm
    mf = nscf.KRHF(cell, kpts=kp)
    assert type(mf) is nscf.KRHF and mf.kpts is kp and mf.use_ao_symmetry is True
    assert len(mf.with_df.kpts) == kp.nkpts
    tight(mf)
    e_ibz = mf.kernel()
    assert mf.converged and len(mf.mo_energy) == kp.nkpts_ibz
    full = tight(nscf.KRHF(cell, kp.kpts))
    e_bz = full.kernel()
    print(f"KRHF He ksymm vs full BZ: |dE| = {abs(e_ibz - e_bz):e} "
          f"({kp.nkpts_ibz} of {kp.nkpts} k-points)")
    assert abs(e_ibz - e_bz) < 1e-10
    explicit = tight(nscf.KsymAdaptedKRHF(cell, kp))
    assert bits(explicit.kernel()) == bits(e_ibz)
    with pytest.raises(TypeError):
        nscf.KsymAdaptedKRHF(cell, kp.kpts)


# ─── the other drivers on the closed-shell fixture ───────────────────────────


@pytest.mark.parametrize("cls", [nscf.KUHF, nscf.KROHF, nscf.KGHF])
def test_open_shell_drivers_reduce_to_krhf_on_a_closed_shell(krhf_he, cls):
    mf0, e0 = krhf_he
    mf = he_krhf(cls)
    e = mf.kernel()
    assert mf.converged and isinstance(mf, nscf.KSCF)
    print(f"{cls.__name__} He 2x2x2: |E - E_KRHF| = {abs(e - e0):e}")
    assert abs(e - e0) < 1e-10
    assert bits(he_krhf(cls)._kernel_without_bridge()) == bits(e)
    if cls is nscf.KUHF:
        assert len(mf.mo_energy) == 2 and len(mf.mo_energy[0]) == 8
        ss, mult = mf.spin_square()
        assert abs(ss) < 1e-10 and abs(mult - 1) < 1e-10
        kband = mf.cell.get_abs_kpts(np.asarray(KBAND_SCALED))
        (ea, eb), _ = mf.get_bands(kband)
        e_r, _ = mf0.get_bands(kband)
        assert max(np.abs(a - b).max() for a, b in zip(ea, e_r)) < 1e-9
    if cls is nscf.KGHF:
        assert mf.mo_coeff[0].shape == (2, 2)
