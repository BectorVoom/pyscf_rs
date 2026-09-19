"""Plan 20-14 — `pbc.symm` (KPoints, Symmetry, SpaceGroup), `pbc.lib.kpts_helper`,
`pbc.tools` (fft family, get_coulG, madelung, ExxDiv, cutoff<->mesh).

Fixtures are the Bohr geometries of `test_pbc_cell.py` (diamond / silicon
`gth-szv`/`gth-pade`, He-fcc `sto-3g`).

**Gate A** is `.planning/phases/17-ksymm-multigrid/17-VERIFICATION.md` §3: the six
`nkpts_ibz` of a diamond-structure (Fd-3m) cell at a `[16,16,16]` mesh are
EXACTLY 145/145/245/408/816/2052, and the symmorphic Fm-3m control collapses
to 145/145/145/408/408/2052. The six configurations are the ones
`crates/pyscf-pbc-symm/tests/kpts_ibz.rs::six_configs` builds, reached here only
through the Python surface.

**The port deliberately differs from upstream k-symmetry** (17-VERIFICATION §6:
D-17-07-01, D-17-09-01, D-17-09-02, the `-1` index). Nothing below compares a
quantity those defects touch (`little_cogroup_ops`, the per-irrep eig, the
KCCSD T1 guard) against upstream; the IBZ bookkeeping that IS compared is not
affected by any of them.
"""

import json
import os
import subprocess
import sys
import textwrap

import numpy as np
import pytest

import pyscf._native as native
import pyscf._native.pbc.gto as ngto
import pyscf._native.pbc.lib as nlib
import pyscf._native.pbc.symm as nsymm
import pyscf._native.pbc.tools as ntools

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

H_C, Q_C = 3.37032, 1.68516  # diamond, fcc a0 = 6.74064 Bohr
# silicon, fcc a0 = 10.2622 Bohr, the second atom at a0/4. NOT test_pbc_cell.py's
# Q_SI = 2.55555 (a0/4 = 2.56555): that geometry is 0.01 Bohr off the diamond
# structure, so its space group is not Fd-3m (12 ops, and a 1795^3 symmetric mesh).
H_SI, Q_SI = 5.1311, 2.56555
H_HE = 2.834589  # He fcc, all-electron


def fcc(h):
    return [[0.0, h, h], [h, 0.0, h], [h, h, 0.0]]


def diamond(**kw):
    return ngto.M(a=fcc(H_C), atom=[("C", (0, 0, 0)), ("C", (Q_C, Q_C, Q_C))],
                  basis="gth-szv", pseudo="gth-pade", unit="Bohr", **kw)


def silicon(**kw):
    return ngto.M(a=fcc(H_SI), atom=[("Si", (0, 0, 0)), ("Si", (Q_SI, Q_SI, Q_SI))],
                  basis="gth-szv", pseudo="gth-pade", unit="Bohr", **kw)


def helium(**kw):
    return ngto.M(a=fcc(H_HE), atom=[("He", (0, 0, 0))], basis="sto-3g", unit="Bohr", **kw)


BUILD = {"diamond": diamond, "silicon": silicon, "helium": helium}


def bits(a):
    """Raw IEEE bits; a complex array is compared on both planes."""
    a = np.ascontiguousarray(a)
    if np.iscomplexobj(a):
        a = a.view(np.float64)
    return np.asarray(a, dtype=np.float64).view(np.uint64)


def assert_refusal(excinfo):
    err = excinfo.value
    assert isinstance(err, native.PyscfRsRuntimeError)
    assert err.args[1] == "NotYetImplemented", err.args


# ── upstream oracle (vendored PySCF 2.12.1, separate interpreter) ───────────

UPSTREAM_SCRIPT = textwrap.dedent(
    """
    import json, sys
    import numpy as np
    import pyscf
    from pyscf.pbc import gto, tools
    from pyscf.pbc.lib import kpts_helper
    out = {"version": pyscf.__version__}
    def mk(a, atom, basis, pseudo, **kw):
        c = gto.Cell()
        c.a = a; c.atom = [(s, x) for s, x in atom]; c.basis = basis
        c.pseudo = pseudo; c.unit = 'Bohr'; c.verbose = 0
        for k, v in kw.items():
            setattr(c, k, v)
        return c.build()
    for name, a, atom, basis, pseudo in json.loads(sys.argv[1]):
        d = {}
        cs = mk(a, atom, basis, pseudo, space_group_symmetry=True, symmorphic=False)
        d["mesh_sg"] = [int(x) for x in cs.mesh]
        kp = cs.make_kpts([4, 4, 4], space_group_symmetry=True, time_reversal_symmetry=True)
        d["nop"] = int(kp.nop)
        d["nkpts_ibz"] = int(kp.nkpts_ibz)
        d["ibz2bz"] = [int(x) for x in kp.ibz2bz]
        d["bz2ibz"] = [int(x) for x in kp.bz2ibz]
        d["weights_ibz"] = [float(x) for x in kp.weights_ibz]
        d["kpts_ibz"] = np.asarray(kp.kpts_ibz).ravel().tolist()
        c = mk(a, atom, basis, pseudo)
        lat = c.lattice_vectors()
        d["cutoff_to_mesh"] = [int(x) for x in tools.cutoff_to_mesh(lat, 50.0)]
        d["mesh_to_cutoff"] = [float(x) for x in tools.mesh_to_cutoff(lat, [15, 15, 15])]
        k222 = c.make_kpts([2, 2, 2])
        d["madelung"] = float(tools.madelung(c, k222))
        k333 = c.make_kpts([3, 3, 3])
        d["conj_pairs"] = [[int(i), None if j is None else int(j)]
                           for i, j in kpts_helper.group_by_conj_pairs(c, k333, return_kpts_pairs=False)]
        d["kk"] = [[np.asarray(k).tolist(), [int(x) for x in ki], [int(x) for x in kj], bool(sc)]
                   for k, ki, kj, sc in kpts_helper.kk_adapted_iter(c, k222)]
        d["kk_notrs"] = [[[int(x) for x in ki], [int(x) for x in kj], bool(sc)]
                         for k, ki, kj, sc in kpts_helper.kk_adapted_iter(c, k333, time_reversal_symmetry=False)]
        kc3 = kpts_helper.get_kconserv3(c, k222, [0, np.arange(8), 2, np.arange(8), 5])
        d["kconserv3"] = np.asarray(kc3).tolist()
        d["is_trim"] = [bool(x) for x in kpts_helper.is_trim(c, k333)]
        out[name] = d
    print(json.dumps(out))
    """
)

FIXTURES = [
    ("diamond", fcc(H_C), [["C", [0, 0, 0]], ["C", [Q_C, Q_C, Q_C]]], "gth-szv", "gth-pade"),
    ("silicon", fcc(H_SI), [["Si", [0, 0, 0]], ["Si", [Q_SI, Q_SI, Q_SI]]], "gth-szv", "gth-pade"),
]


@pytest.fixture(scope="module")
def upstream():
    env = dict(os.environ, PYTHONPATH=REPO)
    proc = subprocess.run([sys.executable, "-c", UPSTREAM_SCRIPT, json.dumps(FIXTURES)],
                          cwd=REPO, env=env, capture_output=True, text=True, check=False)
    assert proc.returncode == 0, proc.stderr
    line = [ln for ln in proc.stdout.splitlines() if ln.startswith("{")][-1]
    data = json.loads(line)
    assert data["version"] == "2.12.1", data["version"]
    return data


# ── Gate A — six EXACT integers through the Python surface ──────────────────

def six_configs(build):
    """`kpts_ibz.rs::six_configs`, spelled as an upstream script would."""
    cell = build(space_group_symmetry=True, symmorphic=False)
    cell_symm = build(space_group_symmetry=True, symmorphic=True)
    plain = build()
    km = [16, 16, 16]
    kg, kg_symm = cell.make_kpts(km), cell_symm.make_kpts(km)
    kng = cell.make_kpts(km, with_gamma_point=False)
    kng_symm = cell_symm.make_kpts(km, with_gamma_point=False)
    mk = nsymm.make_kpts
    got = [
        mk(cell, kg, space_group_symmetry=True, time_reversal_symmetry=False),
        mk(cell_symm, kg_symm, space_group_symmetry=True, time_reversal_symmetry=True),
        mk(cell_symm, kg_symm, space_group_symmetry=True, time_reversal_symmetry=False),
        mk(cell, kng, space_group_symmetry=True, time_reversal_symmetry=False),
        mk(cell_symm, kng_symm, space_group_symmetry=True, time_reversal_symmetry=False),
        mk(plain, kg, space_group_symmetry=False, time_reversal_symmetry=True),
    ]
    for kp in got:
        check_invariants(kp)
    return [kp.nkpts_ibz for kp in got]


def check_invariants(kp):
    nk, ni = kp.nkpts, kp.nkpts_ibz
    assert len(kp) == ni
    assert abs(kp.weights_ibz.sum() - 1.0) < 1e-15
    assert len(kp.stars) == ni and len(kp.bz2ibz) == nk and len(kp.ibz2bz) == ni
    for i, star in enumerate(kp.stars):
        assert kp.weights_ibz[i] == len(star) / nk
        assert kp.ibz2bz[i] in star
        assert (kp.bz2ibz[star] == i).all()
        assert (kp.stars_ops_bz[star] == kp.stars_ops[i]).all()
    assert sorted(np.concatenate(kp.stars).tolist()) == list(range(nk))


@pytest.mark.parametrize("name", ["silicon", "diamond"])
def test_gate_a_ibz_integers_are_exact(name):
    assert six_configs(BUILD[name]) == [145, 145, 245, 408, 816, 2052]


def test_gate_a_symmorphic_control_collapses():
    # He-fcc is Fm-3m (symmorphic): C == A and E == D.
    assert six_configs(helium) == [145, 145, 145, 408, 408, 2052]


def test_cell_make_kpts_with_symmetry_returns_kpoints_gate_a():
    cell = diamond(space_group_symmetry=True, symmorphic=False)
    kp = cell.make_kpts([16, 16, 16], space_group_symmetry=True)
    assert isinstance(kp, nsymm.KPoints)
    assert kp.nkpts == 4096 and kp.nkpts_ibz == 145
    # the free function is the same Rust call on the same k-mesh
    ref = nsymm.make_kpts(cell, cell.make_kpts([16, 16, 16]), space_group_symmetry=True)
    assert (bits(kp.kpts_ibz) == bits(ref.kpts_ibz)).all()
    assert (kp.bz2ibz == ref.bz2ibz).all()


def test_cell_make_kpts_symmetry_needs_a_symmetric_cell():
    # cell.py:876-881
    with pytest.raises(RuntimeError, match="space_group_symmetry"):
        diamond().make_kpts([2, 2, 2], space_group_symmetry=True)
    # time reversal alone needs no space group (cell.py:874)
    kp = diamond().make_kpts([2, 2, 2], time_reversal_symmetry=True)
    assert isinstance(kp, nsymm.KPoints)


# ── against upstream 2.12.1 (quantities no Phase-17 defect touches) ─────────

@pytest.mark.parametrize("name", ["diamond", "silicon"])
def test_symmetric_cell_build_matches_upstream(upstream, name):
    up = upstream[name]
    cell = BUILD[name](space_group_symmetry=True, symmorphic=False)
    # Cell.build carries the lattice symmetry and ENLARGES an auto mesh
    # (cell.py:1770-1772)
    assert list(cell.mesh) == up["mesh_sg"]
    kp = cell.make_kpts([4, 4, 4], space_group_symmetry=True, time_reversal_symmetry=True)
    assert kp.nop == up["nop"]
    assert kp.nkpts_ibz == up["nkpts_ibz"]
    assert kp.ibz2bz.tolist() == up["ibz2bz"]
    assert kp.bz2ibz.tolist() == up["bz2ibz"]
    np.testing.assert_allclose(kp.weights_ibz, up["weights_ibz"], rtol=0, atol=1e-15)
    np.testing.assert_allclose(kp.kpts_ibz.ravel(), up["kpts_ibz"], rtol=0, atol=1e-12)


# ── the KPoints object ──────────────────────────────────────────────────────

def test_kpoints_is_one_type_everywhere():
    import pyscf._native.pbc.lib.kpts as nkpts
    import pyscf.pbc.symm as osymm

    assert nkpts.KPoints is nsymm.KPoints
    assert nkpts.make_kpts is nsymm.make_kpts
    assert osymm.KPoints is nsymm.KPoints
    cell = diamond(space_group_symmetry=True)
    kp = nsymm.KPoints(cell, cell.make_kpts([2, 2, 2]))
    assert isinstance(kp, nsymm.KPoints) and isinstance(kp, nkpts.KPoints)
    assert kp.build(space_group_symmetry=True) is kp
    assert kp.cell is cell
    assert type(kp).__module__ == "pyscf._native.pbc.symm"

    class Sub(nsymm.KPoints):
        pass

    assert isinstance(Sub(cell, cell.make_kpts([2, 2, 2])), nsymm.KPoints)


def test_kpoints_unbuilt_is_identity_mapping():
    cell = diamond()
    k = cell.make_kpts([2, 2, 2])
    kp = nsymm.KPoints(cell, k)
    assert kp.nkpts == kp.nkpts_ibz == 8
    assert (bits(kp.kpts_ibz) == bits(k)).all()
    assert kp.ibz2bz.tolist() == list(range(8))


def test_kpoints_get_kconserv_is_the_kpts_helper_table():
    cell = diamond(space_group_symmetry=True)
    kp = cell.make_kpts([3, 3, 3], space_group_symmetry=True)
    kc = kp.get_kconserv()
    assert kc.shape == (27, 27, 27) and kc.dtype == np.int32
    assert (kc == nlib.kpts_helper.get_kconserv(cell, kp.kpts)).all()


def test_make_ktuples_ibz_weights_and_stars():
    cell = diamond(space_group_symmetry=True)
    kp = cell.make_kpts([2, 2, 2], space_group_symmetry=True)
    ibz2bz, w, bz2ibz, stars, stars_ops, stars_ops_bz = kp.make_ktuples_ibz(ntuple=2)
    assert len(bz2ibz) == 64 and abs(w.sum() - 1.0) < 1e-15
    assert sorted(np.concatenate(stars).tolist()) == list(range(64))
    with pytest.raises(NotImplementedError):
        kp.make_ktuples_ibz(kpts_scaled=kp.kpts_scaled, ntuple=2)


def test_make_k4_ibz_s1_s2_and_s4_refuses():
    cell = diamond(space_group_symmetry=True)
    kp = cell.make_kpts([2, 2, 2], space_group_symmetry=True)
    k4, w, bz2ibz = kp.make_k4_ibz(sym="s1")
    assert k4.shape[1] == 4 and abs(w.sum() - 1.0) < 1e-15 and len(bz2ibz) == 512
    assert len(kp.make_k4_ibz(sym="s1", return_ops=True)) == 6
    k4s2, ws2, _ = kp.make_k4_ibz(sym="s2")
    assert len(k4s2) <= len(k4) and abs(ws2.sum() - 1.0) < 1e-15
    # s4: upstream's own tree has no caller; the port refuses honestly
    with pytest.raises(native.PyscfRsRuntimeError) as e:
        kp.make_k4_ibz(sym="s4")
    assert_refusal(e)
    with pytest.raises(NotImplementedError):
        kp.make_k4_ibz(sym="s8")


def test_little_cogroups_refuses_under_time_reversal_d_17_07_01():
    # Diamond's symmorphic subset (Td) has no inversion, so time reversal is
    # KEPT and k2opk grows 2*nop columns; at Γ the little co-group then names
    # an op index >= nop. Upstream raises IndexError there (kpts.py:1091,
    # D-17-07-01); the port refuses with a typed error and is not "fixed".
    cell = diamond(space_group_symmetry=True, symmorphic=True)
    kp = cell.make_kpts([2, 2, 2], space_group_symmetry=True)
    copgs, idx = kp.little_cogroups()
    assert len(copgs) == 8 and len(idx) == 8
    assert copgs[0].shape[1:] == (3, 3) and copgs[0].dtype == np.int32
    outside = kp.ops_outside_kmesh_subgroup()
    assert outside.dtype == bool and not outside.any()  # [2,2,2] is cubic-closed
    kp_trs = cell.make_kpts([2, 2, 2], space_group_symmetry=True, time_reversal_symmetry=True)
    assert kp_trs.time_reversal
    with pytest.raises(native.PyscfRsRuntimeError) as e:
        kp_trs.little_cogroups()
    assert e.value.args[1] == "PbcSymm"


def test_ops_outside_kmesh_subgroup_detector_d_17_09_02():
    cell = silicon(space_group_symmetry=True)
    kp = cell.make_kpts([1, 1, 2], space_group_symmetry=True)
    out = kp.ops_outside_kmesh_subgroup()
    assert int(out.sum()) == 36 and len(out) == 48  # 17-VERIFICATION §6 item 3


def test_transforms_are_index_maps_and_identity_at_eye():
    cell = diamond(space_group_symmetry=True)
    kp = cell.make_kpts([2, 2, 2], space_group_symmetry=True)
    ni, nk, nao = kp.nkpts_ibz, kp.nkpts, cell.nao_nr()
    rng = np.random.default_rng(7)
    occ = [np.sort(rng.random(nao))[::-1] for _ in range(ni)]
    ene = [np.sort(rng.random(nao)) for _ in range(ni)]
    bz_occ = kp.transform_mo_occ(occ)
    bz_ene = kp.transform_mo_energy(ene)
    assert len(bz_occ) == len(bz_ene) == nk
    for k in range(nk):
        assert (bits(bz_occ[k]) == bits(occ[kp.bz2ibz[k]])).all()
        assert (bits(bz_ene[k]) == bits(ene[kp.bz2ibz[k]])).all()
    # UHF spelling: two spin channels
    u = kp.transform_mo_occ([occ, occ])
    assert len(u) == 2 and len(u[1]) == nk

    mo = [rng.random((nao, nao)) + 1j * rng.random((nao, nao)) for _ in range(ni)]
    bz_mo = kp.transform_mo_coeff(mo)
    assert len(bz_mo) == nk and bz_mo[0].shape == (nao, nao) and bz_mo[0].dtype == np.complex128
    for i, kbz in enumerate(kp.ibz2bz):
        # the IBZ representative is reached by the identity op
        assert (bits(bz_mo[kbz]) == bits(mo[i])).all()
    for k in range(nk):
        assert (bits(kp.transform_single_mo_coeff(mo, k)) == bits(bz_mo[k])).all()
    # a non-contiguous view reads by logical index
    mo_t = [np.ascontiguousarray(m.T).T for m in mo]
    assert all((bits(a) == bits(b)).all() for a, b in zip(kp.transform_mo_coeff(mo_t), bz_mo))
    with pytest.raises(native.PyscfRsRuntimeError):
        kp.transform_mo_occ(occ[:-1])


# ── Symmetry / SpaceGroup / geom ────────────────────────────────────────────

def test_space_group_and_symmetry():
    cell = diamond()
    sg = nsymm.SpaceGroup(cell).build(dump_info=False)
    assert sg.nop == 48 and len(sg.ops) == 48
    assert sg.groupname["point_group_symbol"] == "m-3m"
    assert sg.backend == "pyscf"
    with pytest.raises(NotImplementedError):
        sg.backend = "spglib"

    s = nsymm.Symmetry(cell).build(space_group_symmetry=True, symmorphic=False,
                                   check_mesh_symmetry=False)
    assert s._built and s.nop == 48 and s.has_inversion
    op0 = s.ops[0]
    assert op0.rot.dtype == np.int32 and op0.trans.shape == (3,)
    assert len(s.Dmats) == 48 and s.l_max == 1
    assert s.Dmats[0][1].shape == (3, 3)
    sym = nsymm.Symmetry(cell).build(space_group_symmetry=True, symmorphic=True)
    assert sym.nop == 24  # Fd-3m's symmorphic subset: Td
    assert all(op.trans_is_zero for op in sym.ops)
    rm = s.check_mesh_symmetry()
    rm_list, mesh1 = s.check_mesh_symmetry(return_mesh=True)
    assert rm == rm_list and len(mesh1) == 3
    assert nsymm.get_crystal_class(cell)[0] == "m-3m"
    assert s.spacegroup.nop == 48


def test_symmetry_without_space_group_is_identity():
    s = nsymm.Symmetry(helium()).build(space_group_symmetry=False)
    assert s.nop == 1 and s.ops[0].is_eye


# ── pbc.lib.kpts_helper ─────────────────────────────────────────────────────

def test_kpts_helper_scalars_and_identities():
    kh = nlib.kpts_helper
    assert kh.KPT_DIFF_TOL == 1e-6
    assert kh.is_gamma_point is kh.is_zero and kh.gamma_point is kh.is_zero
    assert kh.is_zero([1e-7, 0, 0]) and not kh.is_zero([1e-6, 0, 0])
    assert kh.get_kconserv is ngto.get_kconserv
    import pyscf._native.pbc.lib.kpts_helper as imported
    assert imported is kh
    k = helium().make_kpts([2, 2, 2])
    assert kh.member(k[3], k).tolist() == [3]
    kk = np.vstack([k, k[:3] + 1e-8])
    uk, idx, inv = kh.unique(kk)
    assert len(uk) == 8 and idx.tolist() == list(range(8))
    assert inv.tolist() == list(range(8)) + [0, 1, 2]
    assert kh.intersection(k, k[2:4]).tolist() == [2, 3]


@pytest.mark.parametrize("name", ["diamond", "silicon"])
def test_kpts_helper_matches_upstream(upstream, name):
    kh, up, cell = nlib.kpts_helper, upstream[name], BUILD[name]()
    k222, k333 = cell.make_kpts([2, 2, 2]), cell.make_kpts([3, 3, 3])
    pairs = kh.group_by_conj_pairs(cell, k333, return_kpts_pairs=False)
    assert [[i, j] for i, j in pairs] == up["conj_pairs"]
    idx_pairs, kpts_pairs = kh.group_by_conj_pairs(cell, k333)
    assert idx_pairs == pairs and len(kpts_pairs) == len(pairs)
    got = list(kh.kk_adapted_iter(cell, k222))
    assert len(got) == len(up["kk"])
    for (kpt, ki, kj, sc), (ukpt, uki, ukj, usc) in zip(got, up["kk"]):
        assert ki.tolist() == uki and kj.tolist() == ukj and sc == usc
        assert ki.dtype == np.int32
        np.testing.assert_allclose(kpt, ukpt, rtol=0, atol=1e-12)
    got = [(ki.tolist(), kj.tolist(), sc)
           for _, ki, kj, sc in kh.kk_adapted_iter(cell, k333, time_reversal_symmetry=False)]
    assert [list(g) for g in got] == up["kk_notrs"]
    with pytest.raises(NotImplementedError):
        list(kh.kk_adapted_iter(cell, k222, kk_idx=np.arange(4), time_reversal_symmetry=True))
    kc3 = kh.get_kconserv3(cell, k222, [0, np.arange(8), 2, np.arange(8), 5])
    assert kc3.tolist() == up["kconserv3"]
    assert kh.is_trim(cell, k333).tolist() == up["is_trim"]


# ── pbc.tools ───────────────────────────────────────────────────────────────

def test_exxdiv_is_one_object_and_get_coulg_accepts_it():
    assert ngto.ExxDiv is ntools.ExxDiv
    E = ntools.ExxDiv
    assert E.parse("ewald") == E.EWALD and E.parse("VCUT_WS") == E.VCUT_WS
    assert E.parse("none") is None
    assert str(E.VCUT_SPH) == "vcut_sph"
    assert ntools.get_coulG is ngto.get_coulG
    assert ntools.super_cell is ngto.super_cell
    cell = helium(mesh=[9, 9, 9])
    k = cell.make_kpts([2, 2, 2])
    by_str = ntools.get_coulG(cell, k=k[0], exx="ewald", kpts=k)
    by_enum = ntools.get_coulG(cell, k=k[0], exx=E.EWALD, kpts=k)
    assert (bits(by_str) == bits(by_enum)).all()
    # the Ewald probe charge lands at G + k = 0 only
    plain = ntools.get_coulG(cell, k=k[0])
    assert (bits(by_str) != bits(plain)).sum() == 1
    with pytest.raises(TypeError):
        ntools.get_coulG(cell, exx=1.5)


def _grid(mesh, nb=None, seed=3):
    rng = np.random.default_rng(seed)
    n = int(np.prod(mesh))
    shape = (n,) if nb is None else (nb, n)
    return rng.standard_normal(shape) + 1j * rng.standard_normal(shape)


@pytest.mark.parametrize("mesh", [[4, 4, 4], [5, 6, 7], [9, 9, 9]])
def test_fft_layout_matches_numpy_and_is_deterministic(mesh):
    f = _grid(mesh)
    g = ntools.fft(f, mesh)
    assert g.shape == f.shape and g.dtype == np.complex128
    ref = np.fft.fftn(f.reshape(mesh)).ravel()
    np.testing.assert_allclose(g, ref, rtol=0, atol=1e-11 * np.abs(ref).max())
    assert (bits(ntools.fft(f, mesh)) == bits(g)).all()
    back = ntools.ifft(g, mesh)
    np.testing.assert_allclose(back, np.fft.ifftn(g.reshape(mesh)).ravel(), rtol=0, atol=1e-13)


@pytest.mark.parametrize("mesh", [[4, 4, 4], [5, 6, 7], [9, 9, 9]])
def test_fft_ifft_round_trip(mesh):
    f = _grid(mesh, nb=3)
    back = ntools.ifft(ntools.fft(f, mesh), mesh)
    # recorded in 20-14-SUMMARY: NOT to_bits() in general (Rust's own
    # fft_accuracy.rs gates the same round trip at 1e-14)
    assert np.abs(back - f).max() < 1e-13
    # the batch is transformed row by row: stacking changes no bit
    g = ntools.fft(f, mesh)
    for b in range(3):
        assert (bits(ntools.fft(f[b], mesh)) == bits(g[b])).all()
    # a real input is widened exactly
    r = f.real.copy()
    assert (bits(ntools.fft(r, mesh)) == bits(ntools.fft(r + 0j, mesh))).all()


def test_fft_round_trip_is_bitwise_on_an_exact_input():
    # A constant field: fft puts n*c at G=0 and exact zeros elsewhere, ifft
    # divides back by mx*my*mz — every step is exact in binary64.
    mesh = [4, 4, 4]
    f = np.full(64, 2.0 + 1.0j)
    assert (bits(ntools.ifft(ntools.fft(f, mesh), mesh)) == bits(f)).all()


def _mul(f, w):
    """`fft.rs::scale_by`'s product, op for op (numpy's complex `*` may fuse)."""
    return (f.real * w.real - f.imag * w.imag) + 1j * (f.real * w.imag + f.imag * w.real)


def test_fftk_is_fft_of_the_phased_field():
    mesh = [5, 5, 5]
    f = _grid(mesh, nb=2)
    ph = np.exp(-1j * _grid(mesh, seed=9).real)
    assert (bits(ntools.fftk(f, mesh, ph)) == bits(ntools.fft(_mul(f, ph), mesh))).all()
    g = ntools.fft(f, mesh)
    assert (bits(ntools.ifftk(g, mesh, ph.conj())) == bits(_mul(ntools.ifft(g, mesh), ph.conj()))).all()
    np.testing.assert_allclose(ntools.fftk(f, mesh, ph), np.fft.fftn((f * ph).reshape(2, 5, 5, 5),
                               axes=(1, 2, 3)).reshape(2, -1), rtol=0, atol=1e-12)
    with pytest.raises(native.PyscfRsRuntimeError):
        ntools.fft(np.zeros(10, complex), mesh)


@pytest.mark.parametrize("name", ["diamond", "silicon"])
def test_mesh_cutoff_and_madelung_match_upstream(upstream, name):
    up, cell = upstream[name], BUILD[name]()
    a = cell.lattice_vectors()
    m = ntools.cutoff_to_mesh(a, 50.0)
    assert m.tolist() == up["cutoff_to_mesh"]
    np.testing.assert_allclose(ntools.mesh_to_cutoff(a, [15, 15, 15]), up["mesh_to_cutoff"],
                               rtol=1e-13, atol=0)
    mad = ntools.madelung(cell, cell.make_kpts([2, 2, 2]))
    assert abs(mad - up["madelung"]) <= 1e-10 * abs(up["madelung"]), (mad, up["madelung"])
