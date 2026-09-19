"""Plan 20-15 — periodic post-SCF bindings: `pbc.mp`, `pbc.cc`, `pbc.ci`, `pbc.ao2mo`.

Every correlated driver consumes an ALREADY-CONVERGED 20-12 `KRHF`/`KUHF`/`KGHF`
plus its `with_df`; nothing re-runs SCF.

Fixture: He-fcc all-electron, Bohr (the 20-CONTEXT §2 cell) in `6-31g` — two AOs,
so `nocc = nvir = 1` per k-point, which is 16 measurements §4's all-electron
control (`m3_df_routes_he.out`, `[1,1,2]`, mesh `[15,15,15]`). `sto-3g` has
`nvir = 0` and cannot host CC. `exxdiv=None`, `KRHF conv_tol = 1e-10`, CC at the
Rust defaults `conv_tol 1e-9 / conv_tol_normt 1e-7` (the Phase-16 gate setting).

Three kinds of assertion, never mixed:

1. **Binding == Rust kernel, BITWISE.** `_rust_reference_kmp2` / `_rust_reference_krccsd`
   run `Krhf::kernel` → `Kmp2::kernel` / `Krccsd::ao2mo` + `kernel_with` + `(T)` on FRESH
   Rust builders of the same kind, with no Python object in between — the
   `krccsd_smoke.rs` sequence. The binding path (bridged KRHF → KMP2/KRCCSD) must
   reproduce `e_hf`, `e_corr` and `e_t` to the bit.
2. **vs upstream (vendored PySCF 2.12.1, one subprocess, version asserted),
   FFTDF ROUTE ONLY**, at the 16 measurements §1 gates: KMP2 `2e-6`
   (15-VERIFICATION), G1 KRCCSD `1e-7`, G6 EOM-IP/EA `1e-5`, G7 KCIS `1e-5`. The
   GDF route is asserted bitwise against Rust only: its `get_eri`/`ao2mo_7d`
   oracle gates currently FAIL (2.33e-10 / 8.2e-11 vs 1e-11, bisect pending,
   measurements/pbc-oracle-tiers.md) and are not papered over here.
3. **Port-internal identities**: G3 KGCCSD/KUCCSD vs KRCCSD `1e-8`; G4 (T) fast vs
   slow `1e-13` relative; G5 spin-orbital (T) vs RHF (T) `1e-9`; ksymm vs full BZ.

Every CC test id names its DF route (`fftdf` / `gdf`); no assertion compares
across routes (upstream's FFTDF/GDF pairs are 9.22e-4 Ha apart on diamond).

Amplitude indices are checked element-wise (memory: free-k-axis gathers are
shape-silent): `|t1|`, `|t2|` per `(ki, kj, ka)` against upstream — the moduli of
1×1×1×1 blocks are gauge-invariant — and the MP2 density matrices through the
gauge-invariant energy contraction of upstream's `pbc/mp/test/test_dm.py`.

Runtime: 28 cases in ~52 s on an idle box (He, <=8 k-points); no case exceeds
25 s, so none is `_high_cost`/`_slow`.
"""

import json
import os
import subprocess
import sys

import numpy as np
import pytest

import pyscf._native.pbc.ao2mo as nao2mo
import pyscf._native.pbc.cc as ncc
import pyscf._native.pbc.ci as nci
import pyscf._native.pbc.df as ndf
import pyscf._native.pbc.gto as ngto
import pyscf._native.pbc.mp as nmp
import pyscf._native.pbc.scf as nscf
import pyscf._native.pbc.symm as nsymm

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
H_HE = 2.834589
MESH = [15, 15, 15]
SCF_TOL = 1e-10
MAX_CYCLE = 50


def fcc(h):
    return [[0.0, h, h], [h, 0.0, h], [h, h, 0.0]]


def helium(basis="6-31g", **kw):
    return ngto.M(a=fcc(H_HE), atom=[("He", (0, 0, 0))], basis=basis, unit="Bohr", **kw)


def bits(x):
    return np.asarray(x, dtype=np.float64).view(np.uint64)


def converged(mf):
    mf.conv_tol = SCF_TOL
    mf.max_cycle = MAX_CYCLE
    mf.kernel()
    assert mf.converged
    return mf


def make_mf(route, cls=nscf.KRHF, basis="6-31g", nk=(1, 1, 2)):
    cell = helium(basis)
    kpts = cell.make_kpts(list(nk))
    mf = cls(cell, kpts, exxdiv=None)
    if route == "fftdf":
        mf.with_df.mesh = MESH
    elif route == "gdf":
        mf.with_df = ndf.GDF(cell, kpts)
    else:
        raise ValueError(route)
    return converged(mf)


# ─── upstream oracle (one subprocess, FFTDF route) ───────────────────────────

UPSTREAM_SCRIPT = r"""
import json, sys
import numpy as np
from pyscf.pbc import gto, scf, mp, cc
from pyscf.pbc.ci import kcis_rhf

h, mesh = json.loads(sys.argv[1])

def cell_of(basis):
    c = gto.Cell()
    c.a = [[0.0, h, h], [h, 0.0, h], [h, h, 0.0]]
    c.atom = [('He', (0.0, 0.0, 0.0))]
    c.basis = basis
    c.unit = 'Bohr'
    c.verbose = 0
    c.build()
    return c

def krhf(c, nk):
    mf = scf.KRHF(c, c.make_kpts(nk), exxdiv=None)
    mf.with_df.mesh = mesh
    mf.conv_tol = 1e-10
    mf.kernel()
    assert mf.converged
    return mf

out = {'version': __import__('pyscf').__version__}
c = cell_of('6-31g')
mf = krhf(c, [1, 1, 2])
out['e_hf'] = float(mf.e_tot)
m = mp.KMP2(mf)
m.kernel()
out['kmp2'] = float(m.e_corr)
mycc = cc.KRCCSD(mf)
mycc.conv_tol = 1e-9
mycc.conv_tol_normt = 1e-7
mycc.kernel()
out['krccsd'] = float(mycc.e_corr)
out['ccsd_t'] = float(mycc.ccsd_t())
out['abs_t1'] = np.abs(mycc.t1).ravel().tolist()
out['abs_t2'] = np.abs(mycc.t2).ravel().tolist()
eip, _ = mycc.ipccsd(nroots=1)
eea, _ = mycc.eaccsd(nroots=1)
out['ip'] = np.asarray(eip).real.tolist()
out['ea'] = np.asarray(eea).real.tolist()
e_cis, _ = kcis_rhf.KCIS(mf).kernel(nroots=1)
out['kcis'] = [np.asarray(x).real.tolist() for x in e_cis]
cd = cell_of('cc-pvdz')
mfd = krhf(cd, [1, 1, 2])
md = mp.KMP2(mfd, frozen=[4])
md.kernel()
out['kmp2_frozen4'] = float(md.e_corr)
from pyscf.pbc.ao2mo import eris
ca = cell_of('sto-3g')
ca.mesh = [9, 9, 9]
ca.build()
eye = [np.eye(1)] * 4
g = eris.get_mo_pairs_G(ca, eye[:2])[:, 0]
ig = eris.get_mo_pairs_invG(ca, eye[:2])[:, 0]
out['ao2mo'] = {
    'g_re': g.real.tolist(), 'g_im': g.imag.tolist(),
    'ig_re': ig.real.tolist(), 'ig_im': ig.imag.tolist(),
    'mo_eri': float(eris.get_mo_eri(ca, eye).real.ravel()[0]),
    'asm': float(eris.assemble_eri(ca, eris.get_mo_pairs_G(ca, eye[:2]),
                                   eris.get_mo_pairs_invG(ca, eye[:2])).real.ravel()[0]),
    'ao_eri': float(np.asarray(eris.get_ao_eri(ca)).real.ravel()[0]),
}
print(json.dumps(out))
"""


@pytest.fixture(scope="module")
def upstream():
    env = dict(os.environ, PYTHONPATH=REPO)
    proc = subprocess.run([sys.executable, "-c", UPSTREAM_SCRIPT, json.dumps([H_HE, MESH])],
                          cwd=REPO, env=env, capture_output=True, text=True, check=False)
    assert proc.returncode == 0, proc.stderr[-4000:]
    up = json.loads([ln for ln in proc.stdout.splitlines() if ln.startswith("{")][-1])
    assert up["version"] == "2.12.1", "the oracle must be the VENDORED PySCF 2.12.1"
    return up


@pytest.fixture(scope="module")
def mf_fftdf():
    return make_mf("fftdf")


@pytest.fixture(scope="module")
def mf_gdf():
    return make_mf("gdf")


@pytest.fixture(scope="module")
def cc_fftdf(mf_fftdf):
    mycc = ncc.KRCCSD(mf_fftdf)
    e, t1, t2 = mycc.kernel()
    assert mycc.converged
    return mycc


def reference(mf, fn):
    return fn(mf.with_df, None, SCF_TOL, None, MAX_CYCLE)


# ─── the identity contract ───────────────────────────────────────────────────


def test_overlay_names_are_the_native_objects():
    import pyscf.pbc.ao2mo as oao2mo
    import pyscf.pbc.cc as occ
    import pyscf.pbc.ci as oci
    import pyscf.pbc.mp as omp

    for ov, nat, names in [
        (omp, nmp, ["KMP2", "KRMP2", "KsymAdaptedKMP2", "KUMP2", "KMP2_stagger"]),
        (occ, ncc, ["KRCCSD", "KCCSD", "KUCCSD", "KGCCSD", "KsymAdaptedRCCSD", "EOMIP",
                    "EOMEA", "EOMEESinglet", "EOMEE"]),
        (oci, nci, ["KCIS", "CIS"]),
        (oao2mo, nao2mo, ["general", "get_mo_eri", "get_mo_pairs_G", "get_mo_pairs_invG",
                          "assemble_eri", "get_ao_pairs_G", "get_ao_eri"]),
    ]:
        for name in names:
            assert getattr(ov, name) is getattr(nat, name), name
    assert nmp.KRMP2 is nmp.KMP2 and ncc.KCCSD is ncc.KRCCSD and nci.CIS is nci.KCIS
    for cls in (ncc.KRCCSD, ncc.KUCCSD, ncc.KGCCSD, ncc.KsymAdaptedRCCSD):
        assert issubclass(cls, ncc._KCCSD)
    assert issubclass(nmp.KsymAdaptedKMP2, nmp.KMP2)


def test_unported_gamma_and_cisd_entry_points_raise(mf_fftdf):
    import pyscf.pbc.cc as occ
    import pyscf.pbc.ci as oci
    import pyscf.pbc.mp as omp

    for fn in (omp.RMP2, omp.UMP2, omp.GMP2, occ.RCCSD, occ.UCCSD, occ.GCCSD, oci.RCISD,
               oci.UCISD, oci.GCISD):
        with pytest.raises(NotImplementedError):
            fn(mf_fftdf)


def test_drivers_refuse_the_wrong_mean_field(mf_fftdf):
    with pytest.raises(TypeError):
        ncc.KUCCSD(mf_fftdf)
    with pytest.raises(TypeError):
        ncc.KGCCSD(mf_fftdf)
    with pytest.raises(TypeError):
        ncc.KsymAdaptedRCCSD(mf_fftdf)
    with pytest.raises(TypeError):
        nmp.KsymAdaptedKMP2(mf_fftdf)
    with pytest.raises(TypeError):
        ncc.KRCCSD(object())
    with pytest.raises(NotImplementedError):
        ncc.KRCCSD(mf_fftdf, frozen=1)
    cell = helium()
    fresh = nscf.KRHF(cell, cell.make_kpts([1, 1, 2]))
    with pytest.raises(ValueError):  # no kernel() yet
        nmp.KMP2(fresh)


# ─── KMP2 ────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("route", ["fftdf", "gdf"])
def test_kmp2_is_the_rust_kernel_bitwise(route, mf_fftdf, mf_gdf):
    mf = mf_fftdf if route == "fftdf" else mf_gdf
    mp = nmp.KMP2(mf)
    e_corr, t2 = mp.kernel()
    e_hf_ref, e_corr_ref = reference(mf, nmp._rust_reference_kmp2)
    print(f"KMP2 {route}: binding {e_corr!r} rust {e_corr_ref!r}")
    assert bits(mf.e_tot) == bits(e_hf_ref)
    assert bits(e_corr) == bits(e_corr_ref)
    assert mp.e_corr == e_corr and mp.e_tot == mp.e_hf + e_corr
    assert mp.e_corr_ss + mp.e_corr_os == pytest.approx(e_corr, abs=1e-15)
    assert t2.shape == (2, 2, 2, 1, 1, 1, 1) and t2.dtype == np.complex128
    assert mp.with_df_ints is (route == "gdf")


def test_kmp2_fftdf_matches_upstream(mf_fftdf, upstream):
    e_corr, _ = nmp.KMP2(mf_fftdf).kernel()
    d_hf = abs(mf_fftdf.e_tot - upstream["e_hf"])
    d = abs(e_corr - upstream["kmp2"])
    print(f"KMP2 fftdf vs upstream: |de_corr| {d:e} (mean field |dE| {d_hf:e})")
    assert d < 2e-6


def test_kmp2_frozen_specs(upstream):
    mf = make_mf("fftdf", basis="cc-pvdz")
    uniform = nmp.KMP2(mf, frozen=[4])
    e_u, _ = uniform.kernel()
    per_k = nmp.KMP2(mf, frozen=[[4], [4]])
    e_k, _ = per_k.kernel()
    assert bits(e_u) == bits(e_k)
    assert uniform.get_nocc() == 1 and uniform.get_nmo() == 4
    assert uniform.get_nmo(per_kpoint=True) == [4, 4]
    masks = uniform.get_frozen_mask()
    assert [m.tolist() for m in masks] == [[True] * 4 + [False]] * 2
    d = abs(e_u - upstream["kmp2_frozen4"])
    print(f"KMP2 fftdf cc-pvdz frozen=[4] vs upstream: |de_corr| {d:e}")
    assert d < 2e-6
    with pytest.raises(Exception):  # freezes the only occupied orbital
        nmp.KMP2(mf, frozen=1).kernel()


def test_kmp2_rdms_contract_to_the_mp2_energy(mf_fftdf):
    """`pbc/mp/test/test_dm.py::test_kmp2_contract_eri_dm` through the bindings.

    The contraction is gauge-invariant and sums every `(kp, kq, kr)` block, so a
    transposed or free-axis-swapped `make_rdm2` index cannot survive it.
    """
    mf = mf_fftdf
    mp = nmp.KMP2(mf)
    mp.kernel()
    cell = mf.cell
    kpts = np.asarray(mf.kpts)
    nk = len(kpts)
    mo = mf.mo_coeff
    hcore = mf.get_hcore()
    dm1 = mp.make_rdm1()
    dm2 = mp.make_rdm2()
    assert same_blocks(dm1, mp.make_rdm1(kind="padded"))
    assert np.array_equal(dm2, mp.make_rdm2(kind="padded"))
    kconserv = ngto.get_kconserv(cell, kpts)
    e = 0.0
    for k in range(nk):
        h = mo[k].conj().T @ hcore[k] @ mo[k]
        assert np.allclose(dm1[k], dm1[k].conj().T, atol=1e-14)
        e += np.einsum("pq,qp", dm1[k], h).real / nk
    # the electron count is conserved over the zone, not per k-point
    assert sum(np.trace(d).real for d in dm1) / nk == pytest.approx(2.0, abs=1e-12)
    for kp in range(nk):
        for kq in range(nk):
            for kr in range(nk):
                ks = kconserv[kp, kq, kr]
                eri = mf.with_df.ao2mo((mo[kp], mo[kq], mo[kr], mo[ks]),
                                       (kpts[kp], kpts[kq], kpts[kr], kpts[ks]),
                                       compact=False).reshape(2, 2, 2, 2) / nk
                e += np.einsum("pqrs,pqrs", dm2[kp, kq, kr], eri).real * 0.5 / nk
    e += cell.energy_nuc()
    print(f"rdm contraction {e!r} vs KMP2 e_tot {mp.e_tot!r}: {abs(e - mp.e_tot):e}")
    assert abs(e - mp.e_tot) < 1e-8


def same_blocks(xs, ys):
    return len(xs) == len(ys) and all(np.array_equal(x, y) for x, y in zip(xs, ys))


def test_kump2_is_bookkeeping_and_its_kernel_raises():
    mf = make_mf("fftdf", cls=nscf.KUHF)
    mp = nmp.KUMP2(mf)
    assert mp.get_nocc() == (1, 1) and mp.get_nmo() == (2, 2)
    assert mp.get_nocc(per_kpoint=True) == ([1, 1], [1, 1])
    assert nmp.KUMP2(mf, frozen=([1], [])).get_nmo() == (1, 2)
    with pytest.raises(NotImplementedError, match="kump2.py"):
        mp.kernel()


STAGGER_UPSTREAM = r"""
import json, sys
from pyscf.pbc import gto, scf
from pyscf.pbc.mp.kmp2_stagger import KMP2_stagger
h, mesh, flag = json.loads(sys.argv[1])
c = gto.Cell()
c.a = [[0.0, h, h], [h, 0.0, h], [h, h, 0.0]]
c.atom = [('He', (0.0, 0.0, 0.0))]
c.basis = '6-31g'
c.unit = 'Bohr'
c.verbose = 0
c.mesh = mesh
c.build()
mf = scf.KRHF(c, c.make_kpts([2, 2, 2]), exxdiv=None)
mf.conv_tol = 1e-10
mf.kernel()
print(json.dumps({'version': __import__('pyscf').__version__,
                  'e': float(KMP2_stagger(mf, flag_submesh=flag).kernel())}))
"""


def stagger_mf():
    """`cell.mesh` (not only `with_df.mesh`) is pinned: upstream's stagger kernel
    rebuilds `FFTDF(cell, kpts)` at `cell.mesh` (`kmp2_stagger.py:74`) while the
    Rust `Kmp2Stagger::integral_df` reuses the mean field's FFTDF when the k-points
    match — at `with_df.mesh = [15]*3` over a default-mesh cell the two differ by
    1.136e-3 (measured 2026-09-14; SUMMARY D5)."""
    cell = helium(mesh=MESH)
    return converged(nscf.KRHF(cell, cell.make_kpts([2, 2, 2]), exxdiv=None))


def stagger_upstream(flag):
    env = dict(os.environ, PYTHONPATH=REPO)
    proc = subprocess.run([sys.executable, "-c", STAGGER_UPSTREAM,
                           json.dumps([H_HE, MESH, flag])],
                          cwd=REPO, env=env, capture_output=True, text=True, check=False)
    assert proc.returncode == 0, proc.stderr[-4000:]
    up = json.loads([ln for ln in proc.stdout.splitlines() if ln.startswith("{")][-1])
    assert up["version"] == "2.12.1"
    return up["e"]


def test_kmp2_stagger_fftdf_submesh_matches_upstream():
    mf = stagger_mf()
    sub = nmp.KMP2_stagger(mf, flag_submesh=True)
    e_sub = sub.kernel()
    assert sub.e_corr == e_sub and sub.flag_submesh
    assert bits(nmp.KMP2_stagger(mf, flag_submesh=True).kernel()) == bits(e_sub)
    d = abs(e_sub - stagger_upstream(True))
    print(f"KMP2_stagger fftdf submesh He 2x2x2: {e_sub!r}, |d| vs upstream {d:e}")
    assert d < 2e-6
    odd = make_mf("fftdf", nk=(1, 1, 3))
    with pytest.raises(Exception, match="even"):
        nmp.KMP2_stagger(odd, flag_submesh=True).kernel()
    with pytest.raises(NotImplementedError):
        nmp.KMP2_stagger(mf, frozen=[0], flag_submesh=False)


def test_kmp2_stagger_fftdf_full_mesh_matches_upstream():
    """Measured 8.0 s for this case alone (binding + upstream subprocess) on an
    idle box, 2026-09-14; ~55 s under load average 30."""
    mf = stagger_mf()
    e_full = nmp.KMP2_stagger(mf, flag_submesh=False).kernel()
    d = abs(e_full - stagger_upstream(False))
    print(f"KMP2_stagger fftdf full mesh He 2x2x2: {e_full!r}, |d| vs upstream {d:e}")
    assert d < 2e-6


# ─── KRCCSD ──────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("route", ["fftdf", "gdf"])
def test_krccsd_is_the_rust_kernel_bitwise(route, mf_fftdf, mf_gdf, cc_fftdf):
    mf = mf_fftdf if route == "fftdf" else mf_gdf
    mycc = cc_fftdf if route == "fftdf" else ncc.KRCCSD(mf)
    if route == "gdf":
        mycc.kernel()
    e_t = mycc.ccsd_t()
    e_hf_ref, e_corr_ref, e_t_ref = reference(mf, ncc._rust_reference_krccsd)
    print(f"KRCCSD {route}: e_corr {mycc.e_corr!r} rust {e_corr_ref!r}; "
          f"(T) {e_t!r} rust {e_t_ref!r}")
    assert mycc.converged
    assert bits(mycc.e_hf) == bits(e_hf_ref)
    assert bits(mycc.e_corr) == bits(e_corr_ref)
    assert bits(e_t) == bits(e_t_ref)
    # determinism: a second driver on the same mean field, to the bit
    again = ncc.KRCCSD(mf)
    again.kernel()
    assert bits(again.e_corr) == bits(mycc.e_corr)
    assert np.array_equal(again.t2, mycc.t2) and np.array_equal(again.t1, mycc.t1)


def test_krccsd_fftdf_matches_upstream_g1(cc_fftdf, mf_fftdf, upstream):
    d_hf = abs(mf_fftdf.e_tot - upstream["e_hf"])
    d = abs(cc_fftdf.e_corr - upstream["krccsd"])
    print(f"G1 KRCCSD fftdf vs upstream: |de_corr| {d:e} (mean field |dE| {d_hf:e})")
    assert d < 1e-7


def test_krccsd_fftdf_amplitudes_elementwise_vs_upstream(cc_fftdf, upstream):
    """|t1|, |t2| per (ki, kj, ka) — gauge-invariant for 1x1(x1x1) blocks — so a
    swapped free k-axis (same shape) is caught element by element."""
    a1 = np.abs(cc_fftdf.t1).ravel()
    a2 = np.abs(cc_fftdf.t2).ravel()
    u1 = np.asarray(upstream["abs_t1"])
    u2 = np.asarray(upstream["abs_t2"])
    assert a2.shape == u2.shape
    d1, d2 = np.max(np.abs(a1 - u1)), np.max(np.abs(a2 - u2))
    print(f"KRCCSD fftdf |t1| max|d| {d1:e}; |t2| max|d| {d2:e}; |t2| spread "
          f"{a2.min():e}..{a2.max():e}")
    assert d1 < 1e-6 and d2 < 1e-6
    assert a2.max() - a2.min() > 1e-4, "t2 blocks too uniform for an index check"


def test_ccsd_t_fftdf_fast_vs_slow_g4(cc_fftdf, upstream):
    fast, slow = cc_fftdf.ccsd_t(), cc_fftdf._ccsd_t_slow()
    rel = abs(fast - slow) / abs(slow)
    print(f"G4 (T) fftdf fast {fast!r} slow {slow!r} rel {rel:e}; "
          f"vs upstream {abs(fast - upstream['ccsd_t']):e}")
    assert rel < 1e-13
    assert abs(fast - upstream["ccsd_t"]) < 1e-7


def test_eom_ip_ea_fftdf_match_upstream_g6(cc_fftdf, upstream):
    e_ip, v_ip = cc_fftdf.ipccsd(nroots=1)
    e_ea, v_ea = cc_fftdf.eaccsd(nroots=1)
    assert e_ip.shape == (2, 1) and len(v_ip) == 2 and v_ip[0].shape[0] == 1
    d_ip = np.max(np.abs(e_ip - np.asarray(upstream["ip"])))
    d_ea = np.max(np.abs(e_ea - np.asarray(upstream["ea"])))
    print(f"G6 fftdf IP {e_ip.ravel()} max|d| {d_ip:e}; EA {e_ea.ravel()} max|d| {d_ea:e}")
    assert d_ip < 1e-5 and d_ea < 1e-5
    eom = ncc.EOMIP(cc_fftdf)
    e, v = eom.kernel(nroots=1, kptlist=[1])
    assert np.array_equal(e, e_ip[1:]) and eom.e is e and eom.converged == [[True]]
    singlet = ncc.EOMEESinglet(cc_fftdf)
    e_ee, _ = singlet.kernel(nroots=1, kptlist=[0])
    print(f"EOMEESinglet fftdf kshift 0: {e_ee.ravel()}")
    assert e_ee.shape == (1, 1) and e_ee[0, 0] > 0
    with pytest.raises(NotImplementedError):
        ncc.EOMEE(cc_fftdf).kernel()


@pytest.mark.parametrize("partition", ["mp", "full"])
@pytest.mark.parametrize("family", ["KRCCSD", "KUCCSD", "KGCCSD"])
def test_eom_partition_raises_fftdf(partition, family, cc_fftdf, g_and_u_fftdf):
    mycc = cc_fftdf if family == "KRCCSD" else g_and_u_fftdf[family]
    with pytest.raises(NotImplementedError, match="eom_kccsd"):
        mycc.ipccsd(partition=partition)
    with pytest.raises(NotImplementedError, match="eom_kccsd"):
        mycc.eaccsd(partition=partition)
    eom = ncc.EOMEA(mycc)
    eom.partition = partition
    with pytest.raises(NotImplementedError, match="eom_kccsd"):
        eom.kernel()
    with pytest.raises(ValueError):
        mycc.ipccsd(partition="bogus")


def test_kcis_fftdf_matches_upstream_g7(mf_fftdf, upstream):
    cis = nci.KCIS(mf_fftdf)
    e, v = cis.kernel(nroots=1)
    assert v is None and e.shape == (2, 1)
    d = np.max(np.abs(e - np.asarray(upstream["kcis"])))
    print(f"G7 KCIS fftdf {e.ravel()} vs upstream {upstream['kcis']}: max|d| {d:e}")
    assert d < 1e-5
    dense = nci.KCIS(mf_fftdf)
    dense.davidson = False
    e_dense, _ = dense.kernel(nroots=1)
    assert np.max(np.abs(e_dense - e)) < 1e-6
    with pytest.raises(NotImplementedError):
        nci.KCIS(mf_fftdf, frozen=[0])


# ─── spin-orbital / unrestricted vs restricted (G3, G5), FFTDF ──────────────


@pytest.fixture(scope="module")
def g_and_u_fftdf():
    out = {}
    gmf = make_mf("fftdf", cls=nscf.KGHF)
    out["KGCCSD"] = ncc.KGCCSD(gmf)
    out["KGCCSD"].kernel()
    umf = make_mf("fftdf", cls=nscf.KUHF)
    out["KUCCSD"] = ncc.KUCCSD(umf)
    out["KUCCSD"].kernel()
    return out


def test_kgccsd_kuccsd_fftdf_equal_krccsd_on_a_closed_shell_g3_g5(cc_fftdf, g_and_u_fftdf):
    g, u = g_and_u_fftdf["KGCCSD"], g_and_u_fftdf["KUCCSD"]
    assert g.converged and u.converged
    dg = abs(g.e_corr - cc_fftdf.e_corr)
    du = abs(u.e_corr - cc_fftdf.e_corr)
    dt = abs(g.ccsd_t() - cc_fftdf.ccsd_t())
    print(f"G3 fftdf |KGCCSD-KRCCSD| {dg:e} |KUCCSD-KRCCSD| {du:e}; G5 |(T)_so-(T)_r| {dt:e}")
    assert dg < 1e-8 and du < 1e-8 and dt < 1e-9
    t1a, t1b = u.t1
    assert t1a.shape == (2, 1, 1) and len(u.t2) == 3
    e_ip_u, _ = u.ipccsd(nroots=1)
    e_ip_r, _ = cc_fftdf.ipccsd(nroots=1)
    assert np.max(np.abs(e_ip_u - e_ip_r)) < 1e-5
    with pytest.raises(NotImplementedError):
        u.ccsd_t()


# ─── k-point symmetry routes (FFTDF) ─────────────────────────────────────────


@pytest.fixture(scope="module")
def he_symm_mfs():
    """Upstream's own ksymm fixture geometry (`pbc/mp/test/test_ksym.py`,
    `cc/test/test_kccsd_ksymm.py`): He in a 2 Angstrom simple-cubic cell at
    `[2,2,2]` — in `6-31g`, because explicit shell lists are not bound (20-09) —
    with `cell.mesh = [15]*3`. Time reversal off (D-17-07-01).

    Not He-fcc: at `with_df.mesh = [15]*3` the fcc ksymm and full-BZ `KRHF`
    differ by 1.354e-6 (3.6e-14 at the default [99]^3 mesh, where `e_corr`
    still differs by 6.8e-9), measured 2026-09-14 (SUMMARY, observations) — a
    two-SCF gate on that cell would measure the mean fields."""
    cell = ngto.M(a=np.eye(3) * 2.0, atom=[("He", (1.0, 1.0, 1.0))], basis="6-31g",
                  mesh=[15, 15, 15], space_group_symmetry=True, symmorphic=True)
    kp = cell.make_kpts([2, 2, 2], space_group_symmetry=True, time_reversal_symmetry=False)
    assert isinstance(kp, nsymm.KPoints) and kp.nkpts_ibz == 4 and kp.nkpts == 8
    assert not kp.ops_outside_kmesh_subgroup().any()
    out = []
    for mf in (nscf.KRHF(cell, kpts=kp, exxdiv=None), nscf.KRHF(cell, kp.kpts, exxdiv=None)):
        mf.conv_tol = 1e-11
        mf.conv_tol_grad = 1e-9
        mf.kernel()
        assert mf.converged
        out.append(mf)
    return tuple(out)


def test_ksymm_kmp2_fftdf_vs_full_bz(he_symm_mfs):
    sym, full = he_symm_mfs
    e_sym, t2 = nmp.KMP2(sym).kernel()
    assert t2 is None  # the k-symmetric default is with_t2=False (kmp2_ksymm.py:28)
    e_cls, _ = nmp.KsymAdaptedKMP2(sym).kernel()
    assert bits(e_cls) == bits(e_sym)
    e_full, _ = nmp.KMP2(full).kernel()
    d = abs(e_sym - e_full)
    print(f"ksymm KMP2 fftdf vs full BZ: |de_corr| {d:e} "
          f"(SCF |dE| {abs(sym.e_tot - full.e_tot):e}); gate 5e-10 = kmp2_ksymm.rs E_CORR_TOL "
          f"(si floor 1.138e-10)")
    assert abs(sym.e_tot - full.e_tot) < 1e-12
    assert d < 5e-10
    dm_ibz = nmp.KMP2(sym).make_rdm1()
    assert len(dm_ibz) == len(sym.mo_energy)
    with pytest.raises(NotImplementedError):
        nmp.KsymAdaptedKMP2(sym).make_rdm2()


def test_ksymm_krccsd_fftdf_vs_full_bz(he_symm_mfs):
    sym, full = he_symm_mfs
    ks = ncc.KsymAdaptedRCCSD(sym)
    ks.kernel()
    fb = ncc.KRCCSD(full)
    fb.kernel()
    d = abs(ks.e_corr - fb.e_corr)
    print(f"ksymm KRCCSD fftdf vs full BZ: |de_corr| {d:e}, ao2mo transforms {ks._n_ao2mo} "
          f"(two SCFs; SCF |dE| {abs(sym.e_tot - full.e_tot):e})")
    # two SCFs, as kccsd_ksymm.rs's 1e-8; the one-mean-field 4.838e-12 (17-09) needs
    # the unfolded reference, which no Python driver exposes
    assert ks.converged and d < 1e-8
    assert ks.t2.shape == fb.t2.shape
    with pytest.raises(NotImplementedError):
        ks.ipccsd()
    with pytest.raises(TypeError, match="KsymAdaptedRCCSD"):
        ncc.KRCCSD(sym)


# ─── pbc.ao2mo wrappers ──────────────────────────────────────────────────────


def test_ao2mo_wrappers_match_upstream(upstream):
    """`pbc/ao2mo/eris.py` on He `sto-3g`, `cell.mesh = [9]*3`, Γ (upstream's
    values from the oracle subprocess). The pair arrays carry upstream's bare-FFT
    normalisation (SUMMARY D6), so upstream's own `get_mo_eri` recipe
    `assemble_eri(get_mo_pairs_G, get_mo_pairs_invG)` must reproduce `get_mo_eri`."""
    up = upstream["ao2mo"]
    cell = helium("sto-3g", mesh=[9, 9, 9])
    rng = np.random.default_rng(3)
    mo = rng.standard_normal((1, 1)) + 1j * rng.standard_normal((1, 1))
    mos = [mo, mo.conj(), mo, mo]
    assert np.array_equal(nao2mo.general(cell, mos, compact=True), nao2mo.get_mo_eri(cell, mos))
    eye = [np.eye(1, dtype=complex)] * 4
    ao = nao2mo.get_ao_eri(cell)
    mo_eri = nao2mo.get_mo_eri(cell, eye)
    fwd = nao2mo.get_mo_pairs_G(cell, eye[:2])
    inv = nao2mo.get_mo_pairs_invG(cell, eye[:2])
    assembled = nao2mo.assemble_eri(cell, fwd, inv)
    ao_pairs = nao2mo.get_ao_pairs_G(cell)
    assert ao.shape == (1, 1) and fwd.shape == (729, 1) and ao_pairs.shape == fwd.shape
    diffs = {
        "get_ao_eri": abs(ao[0, 0] - up["ao_eri"]),
        "get_mo_eri": abs(mo_eri[0, 0] - up["mo_eri"]),
        "assemble_eri": abs(assembled[0, 0] - up["asm"]),
        "get_mo_pairs_G": np.max(np.abs(fwd[:, 0] - up["g_re"] - 1j * np.asarray(up["g_im"]))),
        "get_mo_pairs_invG": np.max(np.abs(inv[:, 0] - up["ig_re"]
                                           - 1j * np.asarray(up["ig_im"]))),
        "get_ao_pairs_G": np.max(np.abs(ao_pairs[:, 0] - up["g_re"]
                                        - 1j * np.asarray(up["g_im"]))),
    }
    print("ao2mo vs upstream: " + ", ".join(f"{k} {v:e}" for k, v in diffs.items()))
    assert max(diffs.values()) < 1e-10
    assert abs(assembled[0, 0] - mo_eri[0, 0]) < 1e-12
