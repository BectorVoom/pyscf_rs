"""Plan 20-18 pre-task — the cheap S shims the unmodified `examples/pbc` scripts need.

1. `with_df.ao2mo(mo, kpts=kpt)` / `get_eri(kpts=kpt)` with ONE k-point broadcast
   to all four indices (upstream `_format_kpts`, `pbc/df/fft_ao2mo.py:430-439`),
   as `examples/pbc/22-k_points_mp2.py:67` spells it.
2. `pyscf.pbc.scf.addons` shim: native `smearing_` / `project_mo_nr2nr`, the rest
   announced upstream fallthrough (D-PBC-35).
3. `pyscf.pbc.dft.multigrid` shim: native `MultiGridNumInt` / `MultiGridNumInt2`.
4. `KRCCSD.ecc` (upstream `pyscf/cc/ccsd.py:990-992`).
5. `KRKS.entropy` (upstream `pyscf/pbc/scf/smearing.py:123-126`) and the `sigma`
   setter on KRKS/KRHF (`mf.sigma = .01; mf.kernel()`, `23-smearing.py:47-51`).
6. `pyscf.M(a=...)` → `pyscf.pbc.gto.M` (upstream `pyscf/__init__.py:106-112`) and
   `Cell.KRKS(...)`-style constructors (upstream `pyscf/pbc/gto/cell.py:1407-1511`),
   the spelling of `examples/pbc/23-smearing.py`.

Fixture: He-fcc, all-electron, Bohr (20-CONTEXT §2), FFTDF mesh pinned to 15³.
Upstream numbers come from ONE subprocess on the vendored 2.12.1 tree (version
asserted).
"""

import json
import os
import subprocess
import sys

import numpy as np
import pytest

import pyscf
import pyscf._native as native
import pyscf._native.pbc.cc as ncc
import pyscf._native.pbc.df as ndf
import pyscf._native.pbc.dft as ndft
import pyscf._native.pbc.gto as ngto
import pyscf._native.pbc.scf as nscf

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
H_HE = 2.834589
MESH = [15, 15, 15]
SIGMA = 0.1


def fcc(h):
    return [[0.0, h, h], [h, 0.0, h], [h, h, 0.0]]


def helium(basis="6-31g", **kw):
    return ngto.M(a=fcc(H_HE), atom=[("He", (0, 0, 0))], basis=basis, unit="Bohr", **kw)


def bits(a):
    a = np.ascontiguousarray(a)
    if np.iscomplexobj(a):
        a = a.view(np.float64)
    return np.asarray(a, dtype=np.float64).view(np.uint64)


def same_bits(x, y):
    return np.shape(x) == np.shape(y) and (bits(x) == bits(y)).all()


def mo_block(nao, seed=7):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((nao, nao)) + 1j * rng.standard_normal((nao, nao))


# ─── upstream oracle (one subprocess) ────────────────────────────────────────

UPSTREAM_SCRIPT = r"""
import json, sys
import numpy as np
import pyscf
from pyscf.pbc import gto, df

h, mesh, sigma, mo_re, mo_im = json.loads(sys.argv[1])
out = {'version': pyscf.__version__}

c = pyscf.M(a=[[0.0, h, h], [h, 0.0, h], [h, h, 0.0]], atom=[('He', (0, 0, 0))],
            basis='6-31g', unit='Bohr', verbose=0)
out['M_type'] = type(c).__module__ + '.' + type(c).__name__
kpts = c.make_kpts([1, 1, 2])
mo = np.asarray(mo_re) + 1j * np.asarray(mo_im)
mydf = df.FFTDF(c, kpts)
mydf.mesh = mesh
nmo = mo.shape[1]
eri = mydf.ao2mo(mo, kpts=kpts[1]).reshape([nmo] * 4)
out['ao2mo_re'] = eri.real.ravel().tolist()
out['ao2mo_im'] = eri.imag.ravel().tolist()

cd = pyscf.M(a=[[0.0, h, h], [h, 0.0, h], [h, h, 0.0]], atom=[('He', (0, 0, 0))],
             basis='cc-pvdz', unit='Bohr', verbose=0)
mf = cd.KRKS(xc='pbe', kpts=cd.make_kpts([1, 1, 2]))
mf.with_df.mesh = mesh
mf.grids.mesh = mesh
mf.conv_tol = 1e-12
mf.conv_tol_grad = 1e-8
mf.max_cycle = 60
mf = mf.smearing(sigma=sigma, method='fermi')
mf.kernel()
out['smear'] = {'e_tot': float(mf.e_tot), 'e_free': float(mf.e_free),
                'entropy': float(mf.entropy), 'converged': bool(mf.converged)}
mf.sigma = sigma / 2
mf.kernel()
out['smear_half'] = {'e_tot': float(mf.e_tot), 'e_free': float(mf.e_free),
                     'entropy': float(mf.entropy), 'converged': bool(mf.converged)}
print(json.dumps(out))
"""


@pytest.fixture(scope="module")
def upstream():
    mo = mo_block(helium().nao_nr())
    env = dict(os.environ, PYTHONPATH=REPO)
    arg = json.dumps([H_HE, MESH, SIGMA, mo.real.tolist(), mo.imag.tolist()])
    proc = subprocess.run([sys.executable, "-c", UPSTREAM_SCRIPT, arg], cwd=REPO, env=env,
                          capture_output=True, text=True, check=False)
    assert proc.returncode == 0, proc.stderr[-4000:]
    up = json.loads([ln for ln in proc.stdout.splitlines() if ln.startswith("{")][-1])
    assert up["version"] == "2.12.1", "the oracle must be the VENDORED PySCF 2.12.1"
    return up


# ─── 1. single-k-point ao2mo / get_eri ───────────────────────────────────────


def test_ao2mo_single_kpt_broadcasts_to_four(upstream):
    cell = helium()
    kpts = cell.make_kpts([1, 1, 2])
    k = kpts[1]
    assert np.abs(k).max() > 1e-3, "the broadcast must be checked away from gamma"
    mydf = ndf.FFTDF(cell, kpts)
    mydf.mesh = MESH
    mo = mo_block(cell.nao_nr())
    four = mydf.ao2mo(mo, kpts=np.vstack([k] * 4))
    for spelling in (k, k.reshape(1, 3), list(k)):
        assert same_bits(mydf.ao2mo(mo, kpts=spelling), four)
    assert same_bits(mydf.get_eri(kpts=k), mydf.get_eri(kpts=np.vstack([k] * 4)))
    with pytest.raises(ValueError):
        mydf.ao2mo(mo, kpts=kpts)  # two k-points: neither one nor four
    nmo = mo.shape[1]
    up = (np.asarray(upstream["ao2mo_re"]) + 1j * np.asarray(upstream["ao2mo_im"])).reshape(
        [nmo] * 4)
    d = np.abs(np.asarray(four).reshape([nmo] * 4) - up).max()
    print(f"ao2mo single-k vs upstream: {d:e}")
    assert d < 1e-10  # measured 2026-09-14: 8.41e-14


# ─── 2./3. overlay shims ─────────────────────────────────────────────────────


def test_scf_addons_shim():
    from pyscf.pbc import scf, which_impl
    from pyscf.pbc.scf import addons

    assert scf.addons is addons
    assert addons.smearing_ is nscf.smearing_
    assert addons.project_mo_nr2nr is nscf.project_mo_nr2nr
    assert which_impl("scf.addons") == "partial"
    assert which_impl("scf.addons.smearing_") == "native"
    assert which_impl("scf.addons.project_mo_nr2nr") == "native"
    assert which_impl("scf.addons.convert_to_uhf") == "upstream"
    assert not hasattr(addons, "__no_such_dunder__")


def test_dft_multigrid_shim():
    from pyscf.pbc import dft, which_impl
    from pyscf.pbc.dft import multigrid

    assert dft.multigrid is multigrid
    assert multigrid.MultiGridNumInt is ndft.MultiGridNumInt
    assert multigrid.MultiGridNumInt2 is ndft.MultiGridNumInt2
    assert which_impl("dft.multigrid") == "partial"
    assert which_impl("dft.multigrid.MultiGridNumInt2") == "native"
    assert which_impl("dft.multigrid.multigrid_pair") == "upstream"
    with pytest.raises(AttributeError):
        multigrid.no_such_name  # noqa: B018


# ─── 4. KRCCSD.ecc ───────────────────────────────────────────────────────────


def test_kccsd_ecc_is_e_corr():
    cell = helium()
    kpts = cell.make_kpts([1, 1, 2])
    mf = nscf.KRHF(cell, kpts, exxdiv=None)
    mf.with_df.mesh = MESH
    mf.conv_tol = 1e-10
    mf.kernel()
    mycc = ncc.KRCCSD(mf)
    assert mycc.ecc is None
    e_corr, _, _ = mycc.kernel()
    assert mycc.ecc == mycc.e_corr == e_corr
    assert mycc.e_corr < 0


# ─── 5./6. entropy, pyscf.M, Cell.KRKS — the 23-smearing.py spelling ─────────


def test_pyscf_M_dispatch():
    mol = pyscf.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g")
    assert type(mol) is native.gto.Mole
    cell = pyscf.M(a=fcc(H_HE), atom=[("He", (0, 0, 0))], basis="sto-3g", unit="Bohr")
    assert type(cell) is ngto.Cell
    assert cell.nao_nr() == 1


def test_cell_method_constructors():
    from pyscf.pbc import dft, scf

    cell = helium()
    kpts = cell.make_kpts([1, 1, 2])
    mf = cell.KRKS(xc="pbe", kpts=kpts, conv_tol=1e-9)
    assert type(mf) is dft.KRKS
    assert mf.cell is cell and mf.xc.lower() == "pbe" and mf.conv_tol == 1e-9
    assert np.array_equal(np.asarray(mf.kpts), kpts)
    assert type(cell.KRHF(kpts=kpts)) is scf.KRHF
    assert type(cell.KUKS(xc="lda")) is dft.KUKS
    with pytest.raises(AttributeError):
        cell.KRHF(1)  # cell.py:1502-1504: no positional arguments
    assert not hasattr(cell, "no_such_attribute")
    assert not hasattr(cell, "_private")
    assert not hasattr(cell, "get_hcore")


def test_krks_entropy_through_the_example_spelling(upstream):
    cell = pyscf.M(a=fcc(H_HE), atom=[("He", (0, 0, 0))], basis="cc-pvdz", unit="Bohr")
    mf = cell.KRKS(xc="pbe", kpts=cell.make_kpts([1, 1, 2]))
    assert mf.entropy is None
    mf.with_df.mesh = MESH
    mf.grids.mesh = MESH
    mf.conv_tol = 1e-12
    mf.conv_tol_grad = 1e-8
    mf.max_cycle = 60
    mf.kernel()
    assert mf.entropy is None  # no smearing
    mf = mf.smearing(sigma=SIGMA, method="fermi")
    mf.kernel()
    assert mf.converged
    assert mf.entropy == (mf.e_tot - mf.e_free) / SIGMA
    assert mf.entropy > 0
    up = upstream["smear"]
    assert up["converged"]
    d = {k: abs(getattr(mf, k) - up[k]) for k in ("e_tot", "e_free", "entropy")}
    print("KRKS smearing He cc-pvdz [1,1,2] vs upstream:", {k: f"{v:e}" for k, v in d.items()})
    # measured 2026-09-14: e_tot 1.38e-11, e_free 1.23e-11, entropy 1.58e-11
    assert d["e_tot"] < 1e-9 and d["e_free"] < 1e-9 and d["entropy"] < 1e-9

    # `mf.sigma = x` then `kernel()` (examples/pbc/23-smearing.py:47-51)
    mf.sigma = SIGMA / 2
    assert mf.sigma == SIGMA / 2 and mf.smearing_method == "fermi"
    mf.kernel()
    assert mf.converged and mf.entropy == (mf.e_tot - mf.e_free) / (SIGMA / 2)
    up = upstream["smear_half"]
    d = {k: abs(getattr(mf, k) - up[k]) for k in ("e_tot", "e_free", "entropy")}
    print("after mf.sigma = sigma/2 vs upstream:", {k: f"{v:e}" for k, v in d.items()})
    assert d["e_tot"] < 1e-9 and d["e_free"] < 1e-9 and d["entropy"] < 1e-9
    # sigma = 0 disables smearing (upstream get_occ falls back, smearing.py:151)
    mf.sigma = 0
    assert mf.sigma is None
    with pytest.raises(AttributeError):
        mf.sigma = 0.1  # no smearing configured any more


def test_sigma_setter_on_krhf():
    mf = nscf.KRHF(helium(), helium().make_kpts([1, 1, 2]))
    with pytest.raises(AttributeError):
        mf.sigma = 0.1
    mf.smearing_(sigma=0.1, method="gaussian")
    mf.sigma = 0.02
    assert mf.sigma == 0.02 and mf.smearing_method == "gaussian"
