"""Plan 20-13 — periodic Kohn-Sham DFT (`pyscf._native.pbc.dft`).

Drivers: KRKS / KUKS / KROKS / KGKS, the ksymm adapters (KsymAdaptedKRKS,
KsymAdaptedKUKS, KsymAdaptedKRKSpU, KsymAdaptedKUKSpU), DFT+U (KRKSpU,
KUKSpU), the three numint backends (grid / multigrid v1 / multigrid v2), the
grids (UniformGrids / BeckeGrids), the refusals, and the Python dispatch.

"Bitwise vs the Rust kernel" (no Rust reference binary is built from here —
memory: gate-target-dir-lto-spelling) is established two ways:

1. same-implementation A/B, to the bit: ``KRKS.kernel()`` (through 20-11's
   ``KPyOverrideBridge``) against ``KRKS._kernel_without_bridge()``, which is
   the concrete ``pyscf_pbc_dft::krks::Krks::kernel`` with no bridge;
2. against the value ``crates/pyscf-pbc-dft/tests/gate.rs::
   krks_si_222_pbe_matches_upstream`` PRINTS for the identical fixture
   (``rust -7.785668903719571``, `target/p20-02-logs/
   dft__gate__krks_si_222_pbe_matches_upstream.log`, P4, 2026-09-14 14:49).

Upstream (vendored PySCF 2.12.1, subprocess, version asserted) is compared at
the measured floors of ``measurements/README.md``: KRKS Si PBE row 3
(6.45e-12, gate 1e-11). Everything else is Rust-vs-Rust or measured.
"""

import json
import os
import subprocess
import sys

import numpy as np
import pytest

import pyscf._native.pbc.df as ndf
import pyscf._native.pbc.dft as ndft
import pyscf._native.pbc.gto as ngto
import pyscf._native.pbc.symm as nsymm
from pyscf._native import PyscfRsRuntimeError  # type: ignore[attr-defined]

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
H_HE = 2.834589
H_SI, Q_SI = 5.1311, 2.55555  # `pyscf-pbc-dft/tests/common/mod.rs::silicon`
H_C, Q_C = 3.37032, 1.68516  # `common::diamond`
MESH_GATE = [31, 31, 31]  # gate.rs::MESH_GATE
MESH_HE = [15, 15, 15]
# gate.rs `krks_si_222_pbe_matches_upstream` prints `rust {:.15}`. Re-referenced
# 2026-09-14 (20-18) after 20-19 D moved the bits: `target/p20-19-D/after/dft.log`
# "KRKS Si 2x2x2 PBE rust -7.785668903725981" (pre-D print -7.785668903719571).
RUST_PRINTED_SI_PBE = "-7.785668903725981"


def fcc(h):
    return [[0.0, h, h], [h, 0.0, h], [h, h, 0.0]]


def helium(basis="sto-3g", **kw):
    return ngto.M(a=fcc(H_HE), atom=[("He", (0, 0, 0))], basis=basis, unit="Bohr", **kw)


def silicon(**kw):
    return ngto.M(a=fcc(H_SI), atom=[("Si", (0, 0, 0)), ("Si", (Q_SI, Q_SI, Q_SI))],
                  basis="gth-szv", pseudo="gth-pade", unit="Bohr", **kw)


def diamond(**kw):
    return ngto.M(a=fcc(H_C), atom=[("C", (0, 0, 0)), ("C", (Q_C, Q_C, Q_C))],
                  basis="gth-szv", pseudo="gth-pade", unit="Bohr", **kw)


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


def tight(mf, mesh=None):
    """`gate.rs::tight()`; pins BOTH meshes as the gate's oracle does."""
    if mesh is not None:
        mf.with_df.mesh = mesh
        mf.grids.mesh = mesh
    mf.conv_tol = 1e-12
    mf.conv_tol_grad = 1e-8
    mf.max_cycle = 60
    return mf


def he_ks(cls=ndft.KRKS, xc="pbe", nk=(2, 2, 2), **kw):
    cell = helium()
    return tight(cls(cell, cell.make_kpts(list(nk)), xc=xc, **kw), MESH_HE)


# ─── upstream oracle ─────────────────────────────────────────────────────────

UPSTREAM_SCRIPT = r"""
import json, sys
import numpy as np
from pyscf.pbc import gto, dft

h_si, q_si, mesh = json.loads(sys.argv[1])
c = gto.Cell()
c.a = [[0.0, h_si, h_si], [h_si, 0.0, h_si], [h_si, h_si, 0.0]]
c.atom = [('Si', (0.0, 0.0, 0.0)), ('Si', (q_si, q_si, q_si))]
c.basis = 'gth-szv'
c.pseudo = 'gth-pade'
c.unit = 'Bohr'
c.verbose = 0
c.build()
mf = dft.KRKS(c, c.make_kpts([2, 2, 2]))
mf.xc = 'pbe'
mf.with_df.mesh = mesh
mf.grids.mesh = mesh
mf.conv_tol = 1e-12
mf.conv_tol_grad = 1e-8
mf.max_cycle = 60
e = mf.kernel()

from pyscf.pbc.dft import krkspu
h_he, mesh_he = json.loads(sys.argv[2])
he = gto.Cell()
he.a = [[0.0, h_he, h_he], [h_he, 0.0, h_he], [h_he, h_he, 0.0]]
he.atom = [('He', (0.0, 0.0, 0.0))]
he.basis = 'sto-3g'
he.unit = 'Bohr'
he.verbose = 0
he.build()
pu = krkspu.KRKSpU(he, he.make_kpts([2, 2, 2]), xc='lda,vwn', U_idx=['He 1s'], U_val=[5.0])
pu.with_df.mesh = mesh_he
pu.grids.mesh = mesh_he
pu.conv_tol = 1e-12
pu.conv_tol_grad = 1e-8
pu.max_cycle = 60
e_pu = pu.kernel()
frac = np.array([np.eye(1) * 0.35] * 8, dtype=complex)
e_u_frac = float(pu.get_veff(he, frac).E_U.real)

from pyscf.pbc.dft import kukspu
uu = kukspu.KUKSpU(he, he.make_kpts([2, 2, 2]), xc='lda,vwn', U_idx=['He 1s'], U_val=[5.0])
uu.with_df.mesh = mesh_he
uu.grids.mesh = mesh_he
uu.init_guess_breaksym = 1
uu.conv_tol = 1e-12
uu.conv_tol_grad = 1e-8
uu.max_cycle = 60
e_uu = uu.kernel()
e_uu_conv = float(uu.get_veff(he, uu.make_rdm1()).E_U.real)
e_uu_frac = float(uu.get_veff(he, np.array([frac, frac])).E_U.real)
pol = np.array([[np.eye(1) * 0.5] * 8, [np.eye(1) * 0.2] * 8], dtype=complex)
e_uu_pol = float(uu.get_veff(he, pol).E_U.real)

print(json.dumps({'version': __import__('pyscf').__version__,
                  'xclib': mf._numint.libxc.__name__,
                  'e_tot': float(e), 'e_nuc': float(c.energy_nuc()),
                  'converged': bool(mf.converged),
                  'krkspu': {'e_tot': float(e_pu), 'converged': bool(pu.converged),
                             'E_U_frac': e_u_frac},
                  'kukspu': {'e_tot': float(e_uu), 'converged': bool(uu.converged),
                             'E_U_conv': e_uu_conv, 'E_U_frac': e_uu_frac,
                             'E_U_pol': e_uu_pol}}))
"""


@pytest.fixture(scope="module")
def upstream():
    arg = json.dumps([H_SI, Q_SI, MESH_GATE])
    arg_he = json.dumps([H_HE, MESH_HE])
    env = dict(os.environ, PYTHONPATH=REPO)
    proc = subprocess.run([sys.executable, "-c", UPSTREAM_SCRIPT, arg, arg_he], cwd=REPO, env=env,
                          capture_output=True, text=True, check=False)
    assert proc.returncode == 0, proc.stderr
    up = json.loads([ln for ln in proc.stdout.splitlines() if ln.startswith("{")][-1])
    assert up["version"] == "2.12.1", "the oracle must be the VENDORED PySCF 2.12.1"
    assert up["xclib"] == "pyscf.dft.libxc", "upstream must run its libxc default"
    assert up["converged"] and up["krkspu"]["converged"] and up["kukspu"]["converged"]
    return up


@pytest.fixture(scope="module")
def krks_si():
    cell = silicon()
    mf = tight(ndft.KRKS(cell, cell.make_kpts([2, 2, 2]), xc="pbe"), MESH_GATE)
    e = mf.kernel()
    assert mf.converged
    return mf, e


# ─── Task 3: the four KS drivers ─────────────────────────────────────────────


def test_classes_and_identity():
    import pyscf.pbc.dft as odft

    for name in ("KRKS", "KUKS", "KROKS", "KGKS", "KRKSpU", "KUKSpU", "KsymAdaptedKRKS",
                 "KsymAdaptedKUKS", "KsymAdaptedKRKSpU", "KsymAdaptedKUKSpU", "KohnShamDFT",
                 "UniformGrids", "BeckeGrids"):
        assert getattr(odft, name) is getattr(ndft, name), name
    for cls in (ndft.KRKS, ndft.KUKS, ndft.KROKS, ndft.KGKS):
        assert issubclass(cls, ndft.KohnShamDFT)
    assert issubclass(ndft.KsymAdaptedKRKS, ndft.KRKS)
    assert issubclass(ndft.KsymAdaptedKUKS, ndft.KUKS)
    assert issubclass(ndft.KRKSpU, ndft.KRKS) and issubclass(ndft.KUKSpU, ndft.KUKS)
    assert issubclass(ndft.KsymAdaptedKRKSpU, ndft.KsymAdaptedKRKS)
    assert issubclass(ndft.KsymAdaptedKUKSpU, ndft.KsymAdaptedKUKS)


def test_krks_si_pbe_is_the_rust_kernel_bitwise(krks_si):
    mf, e = krks_si
    assert e == mf.e_tot and mf.xc == "pbe"
    ref = silicon()
    ref = tight(ndft.KRKS(ref, ref.make_kpts([2, 2, 2]), xc="pbe"), MESH_GATE)
    e_ref = ref._kernel_without_bridge()
    assert bits(e) == bits(e_ref), (e, e_ref)
    assert mf.cycles == ref.cycles
    assert same_bits(mf.mo_energy, ref.mo_energy)
    assert same_bits(mf.mo_coeff, ref.mo_coeff)
    assert mf._overridden_hooks == []
    # gate.rs prints `rust {:.15}` for this fixture; one f64 ulp at |E| ~ 7.8 is
    # 8.9e-16, and the port is not bitwise reproducible run-to-run on every
    # comparison (measurements/README §3), so the printed value is compared
    # at that precision and the residual reported.
    printed = float(RUST_PRINTED_SI_PBE)
    print(f"KRKS Si PBE binding {e:.15f} vs gate.rs printed {RUST_PRINTED_SI_PBE}: "
          f"|d| = {abs(e - printed):e}")
    assert abs(e - printed) <= 5e-15


def test_krks_si_pbe_matches_upstream(krks_si, upstream):
    mf, e = krks_si
    assert abs(mf.e_nuc - upstream["e_nuc"]) < 1e-12
    de = abs(e - upstream["e_tot"])
    print(f"KRKS Si 2x2x2 PBE through the binding: |dE| = {de:e} (floor 6.45e-12)")
    assert de < 1e-11, de


def test_krks_attributes_and_hooks(krks_si):
    mf, _ = krks_si
    assert isinstance(mf.grids, ndft.UniformGrids) and list(mf.grids.mesh) == MESH_GATE
    assert isinstance(mf._numint, ndft.KNumInt)
    assert mf.exxdiv == "ewald" and mf.nlc == ""
    rho = mf.get_rho()
    assert rho.shape == (np.prod(MESH_GATE),)
    nelec = rho.sum() * mf.cell.vol / rho.size
    assert abs(nelec - 8.0) < 1e-6, nelec
    dm = mf.make_rdm1()
    v = mf.get_veff(mf.cell, dm)
    assert len(v) == 8 and v[0].shape == (8, 8)
    comps = mf._veff_components(dm)
    assert set(comps) >= {"vxc", "ecoul", "exc", "nelec", "E_U"} and comps["E_U"] is None
    assert same_bits(comps["vxc"], v)
    e_elec, _ = mf.energy_elec(dm)
    assert abs(e_elec + mf.energy_nuc() - mf.e_tot) < 1e-9
    # bands at the sampling k-points reproduce mo_energy (same Fock, same eig)
    e_band, c_band = mf.get_bands(np.asarray(mf.kpts))
    for a, b in zip(e_band, mf.mo_energy):
        assert np.max(np.abs(np.asarray(a) - np.asarray(b))) < 1e-9
    one_e, one_c = mf.get_bands(np.asarray(mf.kpts)[1])
    assert np.asarray(one_e).shape == (8,) and np.asarray(one_c).shape == (8, 8)


@pytest.mark.parametrize("cls", [ndft.KUKS, ndft.KROKS, ndft.KGKS])
def test_open_shell_drivers_reduce_to_krks_on_a_closed_shell(cls):
    mf0 = he_ks(ndft.KRKS)
    e0 = mf0.kernel()
    mf = he_ks(cls)
    e = mf.kernel()
    assert mf.converged
    print(f"{cls.__name__} He PBE 2x2x2: |E - E_KRKS| = {abs(e - e0):e}")
    assert abs(e - e0) < 1e-10
    assert bits(he_ks(cls)._kernel_without_bridge()) == bits(e)
    if cls is ndft.KUKS:
        assert len(mf.mo_energy) == 2 and len(mf.mo_energy[0]) == 8
        assert tuple(mf.nelec) == (8, 8)
        eb, _ = mf.get_bands(np.asarray(mf.kpts)[:2])
        assert len(eb) == 2 and len(eb[0]) == 2


def test_python_subclass_override_is_dispatched():
    calls = []

    class Counting(ndft.KRKS):
        def get_veff(self, cell=None, dm_kpts=None, *args, **kwargs):
            calls.append(1)
            return super().get_veff(cell, dm_kpts)

    cell = helium()
    mf = tight(Counting(cell, cell.make_kpts([2, 2, 2]), xc="lda,vwn"), MESH_HE)
    e = mf.kernel()
    assert calls and mf._overridden_hooks == ["get_veff"]
    ref = tight(ndft.KRKS(cell, cell.make_kpts([2, 2, 2]), xc="lda,vwn"), MESH_HE)
    assert bits(e) == bits(ref._kernel_without_bridge())


def test_smearing_and_chkfile(tmp_path):
    mf = he_ks(ndft.KRKS)
    assert mf.smearing_(sigma=0.05, method="fermi") is mf and mf.sigma == 0.05
    e = mf.kernel()
    assert mf.converged and mf.e_free is not None and mf.mu is not None
    assert bits(e) == bits(he_ks(ndft.KRKS).smearing_(sigma=0.05)._kernel_without_bridge())
    with pytest.raises(NotImplementedError):
        he_ks(ndft.KROKS).smearing_(sigma=0.05)
    chk = str(tmp_path / "krks.chk")
    mf2 = he_ks(ndft.KRKS)
    mf2.chkfile = chk
    mf2.kernel()
    import pyscf._native.pbc.scf as nscf

    _, rec = nscf.load_scf(chk)
    assert rec["e_tot"] == mf2.e_tot


# ─── Task 4: ksymm adapters and DFT+U ────────────────────────────────────────


def _he_symm(basis):
    cell = helium(basis, space_group_symmetry=True, symmorphic=True)
    # time_reversal_symmetry=False: khf_ksymm.rs::TIME_REVERSAL, 17-07 (as test_pbc_scf).
    kp = cell.make_kpts([2, 2, 2], space_group_symmetry=True, time_reversal_symmetry=False)
    assert isinstance(kp, nsymm.KPoints) and kp.nkpts_ibz < kp.nkpts
    return cell, kp


@pytest.fixture(scope="module")
def he_symm():
    return _he_symm("sto-3g")


@pytest.fixture(scope="module")
def si_symm():
    """Diamond-structure Si (`a0/4 = 2.56555`, the test_pbc_symm fixture —
    `common::silicon`'s 2.55555 detects only 12 ops, 20-14 D2). gth-szv
    carries p functions, which the symmetrized GGA quadrature needs (l = 1
    Wigner-D matrices)."""
    cell = ngto.M(a=fcc(H_SI), atom=[("Si", (0, 0, 0)), ("Si", (2.56555,) * 3)],
                  basis="gth-szv", pseudo="gth-pade", unit="Bohr",
                  space_group_symmetry=True, symmorphic=True)
    kp = cell.make_kpts([2, 2, 2], space_group_symmetry=True, time_reversal_symmetry=False)
    assert kp.nkpts_ibz < kp.nkpts
    return cell, kp


@pytest.mark.parametrize("cls, ksym", [(ndft.KRKS, ndft.KsymAdaptedKRKS),
                                       (ndft.KUKS, ndft.KsymAdaptedKUKS)])
def test_ksymm_ks_matches_full_bz(si_symm, cls, ksym):
    """Gate C (FFTDF) through the binding, at the cell's own symmetrized mesh.
    A PINNED coarse mesh is not a valid fixture: the IBZ and full-BZ arms then
    differ by quadrature aliasing (measured He sto-3g LDA: 8.3e-07 at 15^3,
    5.2e-07 at 16^3, 2.9e-09 at 21^3, 8.4e-14 at the default 43^3)."""
    cell, kp = si_symm
    mf = tight(cls(cell, kpts=kp, xc="pbe"))
    assert type(mf) is cls and mf.kpts is kp and mf.use_ao_symmetry is True
    assert len(mf.with_df.kpts) == kp.nkpts
    e_ibz = mf.kernel()
    assert mf.converged
    nch = 2 if cls is ndft.KUKS else 1
    mo = mf.mo_energy if nch == 1 else mf.mo_energy[0]
    assert len(mo) == kp.nkpts_ibz
    full = tight(cls(cell, kp.kpts, xc="pbe"))
    e_bz = full.kernel()
    print(f"{cls.__name__} Si PBE ksymm vs full BZ (FFTDF): |dE| = {abs(e_ibz - e_bz):e} "
          f"({kp.nkpts_ibz} of {kp.nkpts} k-points, mesh {list(cell.mesh)})")
    assert full.converged and abs(e_ibz - e_bz) < 1e-10
    explicit = tight(ksym(cell, kp, xc="pbe"))
    assert bits(explicit.kernel()) == bits(e_ibz)
    assert bits(tight(ksym(cell, kp, xc="pbe"))._kernel_without_bridge()) == bits(e_ibz)
    with pytest.raises(TypeError):
        ksym(cell, kp.kpts)


@pytest.mark.parametrize("cls", [ndft.KRKS, ndft.KUKS])
def test_ksymm_gga_on_an_s_only_basis_matches_full_bz(he_symm, cls):
    """20-13-FIX (D4): `KPoints::symmetrize_density_vec` indexed the l = 1
    Wigner-D matrices, which `Symmetry` builds only up to the basis' highest l;
    on an s-only basis the symmetrized GGA quadrature panicked and, with
    `panic = "abort"`, killed the interpreter. The l = 1 block is now built from
    the operation itself. Gate C at the cell's own mesh (Rust
    `ksymm_gga_s_only.rs`: KRKS 8.44e-14, KUKS 8.48e-14)."""
    cell, kp = he_symm
    mf = tight(cls(cell, kpts=kp, xc="pbe"))
    assert all(len(d) == 1 for d in kp.Dmats), "the fixture must be s-only"
    e_ibz = mf.kernel()
    full = tight(cls(cell, kp.kpts, xc="pbe"))
    e_bz = full.kernel()
    print(f"{cls.__name__} He sto-3g PBE ksymm vs full BZ: |dE| = {abs(e_ibz - e_bz):e} "
          f"(mesh {list(cell.mesh)})")
    assert mf.converged and full.converged
    assert abs(e_ibz - e_bz) < 1e-9
    ndft.KRKS(cell, kpts=kp, xc="lda,vwn")._veff_components([np.eye(1, dtype=complex)] * kp.nkpts_ibz)


def test_ksymm_gdf_route_gate_c(he_symm):
    """The GDF ksymm arm: 17-VERIFICATION row 11 was 1.432e-06 NOT MET; 20-04
    bisected it to the XC grid following the DF mesh, and 20-04-FIX made it MET
    at 2.1997e-10 (si, bound 1e-8). Asserted here at the SAME 1e-8 Gate-C bound,
    citing 20-04-FIX — not a loosened bound. CAVEAT: the 1-AO He fixture is
    weak (measured |dE| = 0.0, two cycles); the 2-AO He 6-31g run measured
    9.33e-15 but takes 125 s (99^3 XC grid), and Si GDF exceeded 600 s through
    the binding, so neither is in the suite. The grid-follows-cell.mesh
    property the 20-04 defect broke IS asserted directly."""
    cell, kp = he_symm
    mf = tight(ndft.KRKS(cell, kpts=kp, xc="lda,vwn"))
    mf.with_df = ndf.GDF(mf.with_df.cell, kp.kpts)
    full = tight(ndft.KRKS(cell, kp.kpts, xc="lda,vwn"))
    full.with_df = ndf.GDF(cell, kp.kpts)
    # both arms grid XC on cell.mesh (20-04-FIX)
    assert list(mf.grids.mesh) == list(full.grids.mesh) == list(cell.mesh)
    e_ibz, e_bz = mf.kernel(), full.kernel()
    print(f"KRKS He lda ksymm vs full BZ (GDF): |dE| = {abs(e_ibz - e_bz):e}")
    assert mf.converged and full.converged
    assert abs(e_ibz - e_bz) < 1e-8


@pytest.mark.parametrize("cls", [ndft.KRKSpU, ndft.KUKSpU])
def test_dft_u_e_u_over_the_ibz_matches_the_full_bz(he_symm, cls):
    """17-08 Gate: `E_U` with `weights_ibz` vs uniform full-BZ weights on a
    symmetric-by-construction FRACTIONAL density (a filled shell gives E_U=0).
    17-VERIFICATION: 6.939e-18. KUKSpU (20-13-FIX) on a polarised 0.5/0.2
    density."""
    cell, kp = he_symm
    nao = 1
    occ = (0.35,) if cls is ndft.KRKSpU else (0.5, 0.2)
    dm_ibz = [[np.eye(nao, dtype=complex) * o for _ in range(kp.nkpts_ibz)] for o in occ]
    dm_bz = [kp.transform_dm(d) for d in dm_ibz]
    if cls is ndft.KRKSpU:
        dm_ibz, dm_bz = dm_ibz[0], dm_bz[0]
    full = cls(cell, kp.kpts, xc="lda,vwn", U_idx=["He 1s"], U_val=[5.0])
    full.grids.mesh = MESH_HE
    ibz = cls(cell, kpts=kp, xc="lda,vwn", U_idx=["He 1s"], U_val=[5.0])
    ibz.grids.mesh = MESH_HE
    assert type(ibz) is cls and ibz.kpts is kp
    e_full = full._veff_components(dm_bz)["E_U"]
    e_ibz = ibz._veff_components(dm_ibz)["E_U"]
    de = abs(e_full - e_ibz)
    print(f"{cls.__name__} E_U full BZ = {e_full:.15f}, IBZ = {e_ibz:.15f}, |dE_U| = {de:e}")
    assert abs(e_full) > 1e-6
    assert de < 1e-10


def test_kukspu_closed_shell_e_u_equals_krkspu(he_symm):
    """20-13-FIX (D3): `kspu::add_vhubbard` now selects upstream's unrestricted
    expressions for a two-channel density (kukspu.py:96-97, Tr P - Tr P^2 and
    (1 - 2P) U/2). For a closed-shell density D = 2d the unrestricted E_U and
    per-spin potential equal the restricted ones exactly; the old per-channel
    restricted formula did not (measured 5.2916e-02 vs 4.1514e-02 upstream at
    0.35 per spin). Full BZ and IBZ."""
    cell, kp = he_symm
    he = helium()
    kpts = he.make_kpts([2, 2, 2])
    d = [np.eye(1, dtype=complex) * 0.7 for _ in range(8)]
    half = [x / 2 for x in d]
    r = ndft.KRKSpU(he, kpts, xc="lda,vwn", U_idx=["He 1s"], U_val=[5.0])
    u = ndft.KUKSpU(he, kpts, xc="lda,vwn", U_idx=["He 1s"], U_val=[5.0])
    for mf in (r, u):
        mf.grids.mesh = MESH_HE
    cr, cu = r._veff_components(d), u._veff_components([half, half])
    print(f"closed-shell E_U KRKSpU {cr['E_U']:.15f} KUKSpU {cu['E_U']:.15f}")
    assert abs(cr["E_U"]) > 1e-3
    assert abs(cr["E_U"] - cu["E_U"]) < 1e-14
    ibz = ndft.KUKSpU(cell, kpts=kp, xc="lda,vwn", U_idx=["He 1s"], U_val=[5.0])
    ibz.grids.mesh = MESH_HE
    h_ibz = [np.eye(1, dtype=complex) * 0.35 for _ in range(kp.nkpts_ibz)]
    rib = ndft.KRKSpU(cell, kpts=kp, xc="lda,vwn", U_idx=["He 1s"], U_val=[5.0])
    rib.grids.mesh = MESH_HE
    e_u_ibz = ibz._veff_components([h_ibz, h_ibz])["E_U"]
    e_r_ibz = rib._veff_components([2 * x for x in h_ibz])["E_U"]
    assert abs(e_u_ibz - e_r_ibz) < 1e-14


def test_kukspu_matches_upstream(upstream):
    """KUKSpU He sto-3g 2x2x2 mesh 15, U = 5 eV on 'He 1s' vs vendored 2.12.1
    (20-13-FIX). `e_tot` at the KRKSpU bound (1e-12); `E_U` at the converged
    density and at 0.35/0.35 and 0.5/0.2 per spin at the D7 local-orbital
    bound (1e-9). Rust gate `pyscf-pbc-dft/tests/kukspu.rs` measured 6.08e-14,
    5.62e-14, 3.45e-10, 3.10e-10."""
    mf = he_ks(ndft.KUKSpU, xc="lda,vwn", U_idx=["He 1s"], U_val=[5.0])
    e = mf.kernel()
    assert mf.converged
    assert bits(he_ks(ndft.KUKSpU, xc="lda,vwn", U_idx=["He 1s"],
                      U_val=[5.0])._kernel_without_bridge()) == bits(e)
    up = upstream["kukspu"]
    rows = [("e_tot", e, up["e_tot"], 1e-12),
            ("E_U conv", mf.e_u, up["E_U_conv"], 1e-9)]
    for label, occ, key in (("E_U 0.35/0.35", (0.35, 0.35), "E_U_frac"),
                            ("E_U 0.50/0.20", (0.5, 0.2), "E_U_pol")):
        dm = [[np.eye(1, dtype=complex) * o] * 8 for o in occ]
        rows.append((label, mf._veff_components(dm)["E_U"], up[key], 1e-9))
    for label, got, want, tol in rows:
        print(f"KUKSpU He {label}: {got:.15e} vs upstream {want:.15e}, |d| = {abs(got - want):e}")
    for label, got, want, tol in rows:
        assert abs(got - want) < tol, label


@pytest.mark.parametrize("cls, base", [(ndft.KRKSpU, ndft.KRKS), (ndft.KUKSpU, ndft.KUKS)])
def test_dft_u_scf(cls, base):
    mf = he_ks(cls, xc="lda,vwn", U_idx=["He 1s"], U_val=[5.0])
    assert list(mf.U_idx) == ["He 1s"] and list(mf.U_val) == [5.0] and mf.minao_ref == "MINAO"
    e = mf.kernel()
    assert mf.converged
    assert bits(he_ks(cls, xc="lda,vwn", U_idx=["He 1s"], U_val=[5.0])._kernel_without_bridge()) == bits(e)
    e0 = he_ks(base, xc="lda,vwn").kernel()
    # He 1s is a filled shell: E_U vanishes and so does the energy shift
    print(f"{cls.__name__} He: E_U = {mf.e_u:e}, |E - E_KS| = {abs(e - e0):e}")
    assert abs(mf.e_u) < 1e-12 and abs(e - e0) < 1e-9


def test_krkspu_matches_upstream(upstream):
    """KRKSpU He sto-3g 2x2x2 mesh 15, U = 5 eV on 'He 1s'. `e_tot` at the
    all-electron KRKS gate (row 3a: 9.81e-14, gate 1e-12). `E_U` on the
    0.35-occupied density has no prior floor: FIRST measurement, printed, and
    bounded at 1e-9 (the MINAO local orbitals are built by `zsolve_linear` +
    `zeigh_gen` here, `cho_solve` + `vec_lowdin` upstream)."""
    mf = he_ks(ndft.KRKSpU, xc="lda,vwn", U_idx=["He 1s"], U_val=[5.0])
    e = mf.kernel()
    up = upstream["krkspu"]
    de = abs(e - up["e_tot"])
    e_u = mf._veff_components([np.eye(1, dtype=complex) * 0.35] * 8)["E_U"]
    deu = abs(e_u - up["E_U_frac"])
    print(f"KRKSpU He vs upstream: |dE| = {de:e}; E_U(0.35) {e_u:.15f} vs "
          f"{up['E_U_frac']:.15f}, |dE_U| = {deu:e}")
    assert de < 1e-12
    assert deu < 1e-9


def test_dft_u_refuses_unported_site_specs():
    cell = helium()
    # integer lists index the LARGE basis upstream (mapped through AO labels,
    # rkspu.py:156-158); the Rust `USite::Indices` indexes the MINAO reference
    # basis. Refused rather than silently reinterpreted.
    for kw in ({"U_idx": [[0]], "U_val": [5.0]}, {"U_idx": ["1 He 1s"], "U_val": [5.0]},
               {"U_idx": ["He 1s"], "U_val": [5.0], "C_ao_lo": np.eye(1)},
               {"U_idx": ["He 1s"], "U_val": [5.0, 1.0]}):
        with pytest.raises((NotImplementedError, ValueError)):
            ndft.KRKSpU(cell, cell.make_kpts([1, 1, 1]), **kw)
    ok = ndft.KRKSpU(cell, cell.make_kpts([1, 1, 1]), U_idx=["He 1s"], U_val=[5.0],
                     C_ao_lo="minao")
    with pytest.raises(NotImplementedError):
        ok.U_idx = [[0]]


# ─── Task 5: numint backends and grids ───────────────────────────────────────


def _random_symmetric_dm(nao, seed=0xDEADBEEF):
    rng = np.random.default_rng(seed % (2**32))
    dm = rng.random((nao, nao)) * 0.2
    dm = 0.5 * (dm + dm.T) + np.eye(nao)
    return [dm.astype(complex)]


def test_numint_backends_selectable_and_v2_at_its_floor():
    """17-01 Gate E: v1 is algebraically exact against FFTDF/numint
    (1e-12...1e-14); v2 carries a DEFINITIONAL mesh-independent ~2e-8 (diamond)
    ... 2e-7 (si) floor against FFTDF (upstream's own two implementations). v2
    is asserted only at its own floor, never at v1's."""
    cell = diamond(mesh=[25, 25, 25])
    dm = _random_symmetric_dm(cell.nao_nr())
    grid = ndft.KRKS(cell, xc="lda,vwn")
    ref = grid._veff_components(dm)
    v1 = ndft.KRKS(cell, xc="lda,vwn").multigrid_numint()
    assert isinstance(v1._numint, ndft.MultiGridNumInt)
    v2 = ndft.KRKS(cell, xc="lda,vwn")
    v2._numint = ndft.MultiGridNumInt2(cell)
    c1, c2 = v1._veff_components(dm), v2._veff_components(dm)
    d1 = abs(c1["ecoul"] - ref["ecoul"])
    d2 = abs(c2["ecoul"] - ref["ecoul"])
    d12 = abs(c1["ecoul"] - c2["ecoul"])
    print(f"diamond ecoul: |v1-grid| = {d1:e}, |v2-grid| = {d2:e}, |v1-v2| = {d12:e}; "
          f"exc: |v1-grid| = {abs(c1['exc'] - ref['exc']):e}, "
          f"|v2-grid| = {abs(c2['exc'] - ref['exc']):e}")
    assert d1 < 1e-10
    assert d2 < 2e-7
    with pytest.raises(NotImplementedError):
        m = ndft.KRKS(cell, xc="lda,vwn").multigrid_numint(mesh=[31, 31, 31])
        m._veff_components(dm)


def test_multigrid_refusals():
    cell = helium()
    k8 = ndft.KRKS(cell, cell.make_kpts([2, 2, 2]), xc="lda,vwn").multigrid_numint()
    with pytest.raises(PyscfRsRuntimeError, match="gamma"):
        k8.kernel()
    hyb = ndft.KRKS(cell, xc="pbe0")
    hyb._numint = ndft.MultiGridNumInt2(cell)
    with pytest.raises(PyscfRsRuntimeError, match="hybrid"):
        hyb.kernel()
    with pytest.raises(NotImplementedError):
        ndft.KGKS(cell, xc="lda,vwn").multigrid_numint().kernel()
    with pytest.raises(TypeError):
        ndft.KRKS(cell).__setattr__("_numint", object())


def test_multigrid2_scf_converges_near_the_grid_energy():
    cell = silicon(mesh=[11, 11, 11])
    ref = ndft.KRKS(cell, xc="lda,vwn")
    ref.conv_tol, ref.conv_tol_grad, ref.max_cycle = 1e-10, 1e-7, 100
    e_grid = ref.kernel()
    mg = ndft.KRKS(cell, xc="lda,vwn")
    mg.conv_tol, mg.conv_tol_grad, mg.max_cycle = 1e-10, 1e-7, 100
    mg._numint = ndft.MultiGridNumInt2(cell)
    e_mg = mg.kernel()
    print(f"Si gamma mesh 11 KRKS: |E_v2 - E_grid| = {abs(e_mg - e_grid):e} "
          f"(multigrid_scf.rs gate 2e-3 at this mesh)")
    assert mg.converged and abs(e_mg - e_grid) < 2e-3


def test_grids_objects():
    cell = helium()
    g = ndft.UniformGrids(cell)
    assert list(g.mesh) == list(cell.mesh) and g.size == np.prod(cell.mesh)
    assert g.coords.shape == (g.size, 3)
    assert abs(g.weights.sum() - cell.vol) < 1e-10
    g.mesh = [9, 9, 9]
    assert g.size == 729 and g.coords.shape == (729, 3)
    b = ndft.BeckeGrids(cell)
    b.level = 0
    assert b.build() is b and b.size > 0 and b.coords.shape == (b.size, 3)
    mf = ndft.KRKS(cell, xc="lda,vwn")
    mf.grids = b
    assert mf.grids is b
    v_becke = mf.get_veff(cell, [np.eye(1, dtype=complex)])
    v_uni = ndft.KRKS(cell, xc="lda,vwn").get_veff(cell, [np.eye(1, dtype=complex)])
    assert np.isfinite(v_becke[0]).all() and not same_bits(v_becke, v_uni)
    with pytest.raises(TypeError):
        mf.grids = object()


# ─── Task 6: refusals and the Python dispatch ────────────────────────────────


def test_kgks_hybrid_refuses():
    mf = he_ks(ndft.KGKS, xc="pbe0")
    with pytest.raises(PyscfRsRuntimeError) as ei:
        mf.kernel()
    assert ei.value.args[1] == "NotYetImplemented" and "hybrid" in ei.value.args[0]


def test_kgks_collinear_refusals():
    cell = helium()
    kpts = cell.make_kpts([1, 1, 1])
    dm = [np.eye(2, dtype=complex) * 0.5]
    mf = ndft.KGKS(cell, kpts, xc="lda,vwn")
    assert mf.collinear == "col"
    mf.collinear = "mcol"
    with pytest.raises(PyscfRsRuntimeError, match="mcol") as ei:
        mf.get_veff(cell, dm)
    assert ei.value.args[1] == "NotYetImplemented"
    mf.collinear = "ncol"
    mf.xc = "pbe"
    with pytest.raises(PyscfRsRuntimeError, match="ncol") as ei:
        mf.get_veff(cell, dm)
    assert ei.value.args[1] == "NotYetImplemented"
    mf.xc = "lda,vwn"
    with pytest.raises(PyscfRsRuntimeError, match="non-collinear") as ei:
        mf.nr_fxc(dm)
    assert ei.value.args[1] == "NotYetImplemented"
    with pytest.raises(ValueError):
        mf.collinear = "bogus"
    with pytest.raises(AttributeError):
        ndft.KRKS(cell).collinear = "col"


def test_other_refusals(he_symm):
    cell, kp = he_symm
    for cls in (ndft.KROKS, ndft.KGKS):
        with pytest.raises(NotImplementedError, match="k-point symmetry"):
            cls(cell, kpts=kp)
    mf = ndft.KRKS(helium())
    with pytest.raises(NotImplementedError):
        mf.nlc = "vv10"
    mf.nlc = ""
    with pytest.raises(NotImplementedError):
        ndft.KROKS(helium()).get_bands(np.zeros(3))


def test_python_dispatch(he_symm):
    import pyscf.pbc.dft as odft
    import pyscf.pbc.scf as oscf

    cell, kp = he_symm
    he = helium()
    assert type(odft.KRKS(cell, kpts=kp)) is ndft.KRKS
    assert odft.KRKS(cell, kp)._is_ksymm and odft.KUKS(cell, kpts=kp)._is_ksymm
    assert odft.KRKSpU(cell, kpts=kp, U_idx=["He 1s"], U_val=[1.0])._is_ksymm
    assert odft.KUKSpU(cell, kp, U_idx=["He 1s"], U_val=[1.0])._is_ksymm
    assert not odft.KRKS(he, he.make_kpts([2, 2, 2]))._is_ksymm
    assert isinstance(odft.RKS(he), ndft.KRKS) and len(odft.RKS(he).with_df.kpts) == 1
    assert isinstance(odft.RKS(he, kpts=he.make_kpts([2, 1, 1])), ndft.KRKS)
    assert isinstance(odft.UKS(he), ndft.KUKS) and isinstance(odft.ROKS(he), ndft.KROKS)
    assert isinstance(odft.GKS(he), ndft.KGKS)
    assert isinstance(odft.KS(he), ndft.KRKS) and isinstance(odft.KKS(he), ndft.KRKS)
    li = ngto.M(a=np.eye(3) * 6.0, atom=[("Li", (0, 0, 0))], basis="sto-3g", unit="Bohr", spin=1)
    assert isinstance(odft.RKS(li), ndft.KROKS) and isinstance(odft.KS(li), ndft.KUKS)
    assert type(oscf.KRKS(he)) is ndft.KRKS
