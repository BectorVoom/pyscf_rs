"""Plan 20-09 — the periodic Cell binding (`pyscf._native.pbc.gto`).

Every fixture is built in BOHR: `unit='Ang'` is CODATA-2014 in this port and
CODATA-2010 upstream, which moves the 8th digit of every lattice vector before
any integral is evaluated. The geometries are the ones
`crates/pyscf-pbc-scf/tests/common/mod.rs` gates against upstream.

"Bitwise vs Rust" here means: the binding returns exactly the bits the same Rust
function produces. For `make_kpts` that is checked against an independent
re-evaluation of `pyscf_pbc_gto::make_kpts`'s arithmetic in Python floats,
operation for operation (`i/n`, then `x*b0j + y*b1j + z*b2j` left to right —
IEEE-754 double arithmetic is the same in both languages, and the release build
has no FMA, `xtask check-no-fma`). Upstream numbers are compared at the floors
the Rust oracle tests use (`crates/pyscf-pbc-gto/tests/oracle_phase9.rs`:
`vol` 1e-12 relative, reciprocal vectors 1e-12).
"""

import json
import os
import pickle
import struct
import subprocess
import sys
import textwrap

import numpy as np
import pytest

import pyscf._native as native
import pyscf._native.pbc.gto as ngto

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

H_C, Q_C = 3.37032, 1.68516  # diamond, fcc a0 = 6.74064 Bohr
H_SI, Q_SI = 5.1311, 2.55555  # silicon, fcc a0 = 10.2622 Bohr
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


def bits(a):
    """Raw IEEE bits; a complex array is compared on both planes."""
    a = np.ascontiguousarray(a)
    if np.iscomplexobj(a):
        a = a.view(np.float64)
    return np.asarray(a, dtype=np.float64).view(np.uint64)


def assert_refusal(excinfo):
    """A Rust `NotYetImplemented` refusal reaching Python."""
    err = excinfo.value
    assert isinstance(err, native.PyscfRsRuntimeError)
    assert err.args[1] == "NotYetImplemented", err.args


# ── upstream oracle (vendored PySCF 2.12.1, separate interpreter) ───────────

UPSTREAM_SCRIPT = textwrap.dedent(
    """
    import json, sys
    import numpy as np
    import pyscf
    from pyscf.pbc import gto
    out = {"version": pyscf.__version__}
    for name, a, atom, basis, pseudo in json.loads(sys.argv[1]):
        cell = gto.Cell()
        cell.a = a
        cell.atom = [(s, x) for s, x in atom]
        cell.basis = basis
        cell.pseudo = pseudo
        cell.unit = 'Bohr'
        cell.verbose = 0
        cell.build()
        out[name] = {
            "nao": int(cell.nao_nr()),
            "charges": cell.atom_charges().tolist(),
            "vol": float(cell.vol),
            "b": cell.reciprocal_vectors().ravel().tolist(),
            "kpts": cell.make_kpts([2, 2, 2]).ravel().tolist(),
            "nelec": int(cell.tot_electrons()),
        }
    print(json.dumps(out))
    """
)

FIXTURES = [
    ("diamond", fcc(H_C), [["C", [0, 0, 0]], ["C", [Q_C, Q_C, Q_C]]], "gth-szv", "gth-pade"),
    ("silicon", fcc(H_SI), [["Si", [0, 0, 0]], ["Si", [Q_SI, Q_SI, Q_SI]]], "gth-szv", "gth-pade"),
    ("helium", fcc(H_HE), [["He", [0, 0, 0]]], "sto-3g", None),
]
BUILD = {"diamond": diamond, "silicon": silicon, "helium": helium}


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


# ── construction, Deref surface ─────────────────────────────────────────────

def test_cell_is_native_and_built():
    cell = diamond()
    assert type(cell) is ngto.Cell
    assert cell._built
    assert "pyscf._native.pbc.gto" == type(cell).__module__


def test_attribute_then_build_matches_M_bitwise():
    c = ngto.Cell()
    assert not c._built
    c.a = fcc(H_C)
    c.atom = [("C", (0, 0, 0)), ("C", (Q_C, Q_C, Q_C))]
    c.basis = "gth-szv"
    c.pseudo = "gth-pade"
    c.unit = "Bohr"
    assert c.build() is c
    ref = diamond()
    assert (bits(c.lattice_vectors()) == bits(ref.lattice_vectors())).all()
    assert (bits(c.make_kpts([2, 2, 2])) == bits(ref.make_kpts([2, 2, 2]))).all()
    assert c.mesh == ref.mesh


@pytest.mark.parametrize("name,nao", [("diamond", 8), ("silicon", 8), ("helium", 1)])
def test_nao_nr_resolves_through_the_mole_deref(name, nao):
    cell = BUILD[name]()
    # not a Cell method: reached through __getattr__ -> cell.mol (the Deref)
    assert "nao_nr" not in type(cell).__dict__
    assert cell.nao_nr() == nao
    assert cell.nao_nr() == cell.mol.nao_nr()
    assert type(cell.mol).__name__ == "Mole"
    assert cell.natm == cell.mol.natm


@pytest.mark.parametrize("name,charges", [("diamond", [4, 4]), ("silicon", [4, 4]), ("helium", [2])])
def test_atom_charges_are_gth_valence(name, charges):
    cell = BUILD[name]()
    q = cell.atom_charges()
    assert q.dtype == np.int32
    assert q.tolist() == charges
    # the molecular half was rewritten too, so the Deref'd answer agrees
    assert cell.mol.atom_charges().tolist() == charges
    assert cell.tot_electrons() == sum(charges)
    assert cell.tot_electrons(nkpts=8) == 8 * sum(charges)


def test_atom_pseudo():
    assert diamond().atom_pseudo(0)["nelec"] == [2, 2]
    assert helium().atom_pseudo(0) is None


@pytest.mark.parametrize("name", ["diamond", "silicon", "helium"])
def test_scalars_match_upstream(upstream, name):
    cell, up = BUILD[name](), upstream[name]
    assert cell.nao_nr() == up["nao"]
    assert cell.atom_charges().tolist() == up["charges"]
    assert cell.tot_electrons() == up["nelec"]
    assert abs(cell.vol - up["vol"]) <= abs(up["vol"]) * 1e-12
    np.testing.assert_allclose(cell.reciprocal_vectors().ravel(), up["b"], rtol=0, atol=1e-12)
    np.testing.assert_allclose(cell.make_kpts([2, 2, 2]).ravel(), up["kpts"], rtol=0, atol=1e-12)


# ── k-points: bitwise against the Rust arithmetic ───────────────────────────

def _rust_make_kpts(b, nks, wrap_around=False, with_gamma_point=True, scaled_center=None):
    """`pyscf_pbc_gto::make_kpts` (kpts_mesh.rs:47) + `vec_mat` (cell.rs:206),
    operation for operation, in Python floats."""
    axes = []
    for n in nks:
        if with_gamma_point or scaled_center is not None:
            ks = [i / n for i in range(n)]
        else:
            ks = [(i + 0.5) / n - 0.5 for i in range(n)]
        if wrap_around:
            ks = [k - 1.0 if k >= 0.5 else k for k in ks]
        axes.append(ks)
    c = scaled_center if scaled_center is not None else [0.0, 0.0, 0.0]
    out = []
    for x in axes[0]:
        for y in axes[1]:
            for z in axes[2]:
                v = [x + c[0], y + c[1], z + c[2]]
                out.append([v[0] * b[0][j] + v[1] * b[1][j] + v[2] * b[2][j] for j in range(3)])
    return np.array(out, dtype=np.float64)


@pytest.mark.parametrize("name", ["diamond", "silicon", "helium"])
@pytest.mark.parametrize("opts", [{}, {"wrap_around": True}, {"with_gamma_point": False},
                                  {"scaled_center": [0.1, -0.05, 0.25]}])
def test_make_kpts_bitwise_equals_the_rust_arithmetic(name, opts):
    cell = BUILD[name]()
    b = [[float(x) for x in row] for row in cell.reciprocal_vectors()]
    got = cell.make_kpts([2, 2, 2], **opts)
    want = _rust_make_kpts(b, [2, 2, 2], **opts)
    assert got.shape == (8, 3)
    assert (bits(got) == bits(want)).all()
    # the free function is the same Rust call
    assert (bits(ngto.make_kpts(cell, [2, 2, 2], **opts)) == bits(got)).all()


def test_abs_scaled_kpts_round_trip_shapes():
    cell = diamond()
    k = cell.make_kpts([2, 1, 3])
    s = cell.get_scaled_kpts(k)
    assert s.shape == (6, 3)
    np.testing.assert_allclose(cell.get_abs_kpts(s), k, rtol=0, atol=1e-14)
    assert cell.get_abs_kpts([0.5, 0.0, 0.0]).shape == (3,)


def test_kconserv_shape():
    cell = helium()
    kc = ngto.get_kconserv(cell, cell.make_kpts([2, 2, 2]))
    assert kc.shape == (8, 8, 8) and kc.dtype == np.int32
    assert sorted(kc[0, 0]) == list(range(8))


# ── hcore is NOT a Cell method ──────────────────────────────────────────────

def test_get_hcore_is_absent_and_points_at_the_df_object():
    cell = helium()
    assert hasattr(cell, "get_hcore") is False
    assert "get_hcore" not in dir(type(cell))
    assert "FFTDF" in ngto.Cell.__doc__ and "get_hcore" in ngto.Cell.__doc__


# ── evaluation surface ──────────────────────────────────────────────────────

def test_pbc_intor_gamma_real_and_kpts_list():
    cell = diamond()
    s = cell.pbc_intor("int1e_ovlp")
    assert s.shape == (8, 8) and s.dtype == np.float64
    assert (bits(cell.intor("int1e_ovlp")) == bits(s)).all()
    np.testing.assert_allclose(s, s.T, rtol=0, atol=1e-12)
    kpts = cell.make_kpts([2, 2, 2])
    sk = cell.pbc_intor("int1e_ovlp", hermi=1, kpts=kpts)
    assert isinstance(sk, list) and len(sk) == 8
    assert sk[3].shape == (8, 8) and sk[3].dtype == np.complex128
    again = cell.pbc_intor("int1e_ovlp", hermi=1, kpts=kpts)
    assert all((bits(a) == bits(b)).all() for a, b in zip(sk, again))
    ip = cell.pbc_intor("int1e_ipovlp")
    assert ip.shape == (3, 8, 8)


def test_eval_gto_shapes():
    cell = helium()
    coords = np.array([[0.1, 0.2, 0.3], [1.0, -0.5, 0.25], [2.0, 2.0, 2.0]])
    ao = cell.pbc_eval_gto("GTOval_sph", coords)
    assert ao.shape == (3, 1) and ao.dtype == np.float64
    kpts = cell.make_kpts([2, 1, 1])
    aok = cell.eval_gto("GTOval_sph_deriv1", coords, kpts=kpts)
    assert len(aok) == 2
    assert aok[0].shape == (4, 3, 1) and aok[1].dtype == np.complex128
    np.testing.assert_allclose(aok[0][0], ao, rtol=0, atol=1e-14)


def test_ewald_is_energy_nuc():
    cell = diamond()
    assert struct.pack("d", cell.energy_nuc()) == struct.pack("d", cell.ewald())
    assert np.isfinite(cell.energy_nuc())


# ── supercell, band path, serialisation ─────────────────────────────────────

def test_super_cell():
    cell = diamond()
    sc = ngto.super_cell(cell, [2, 1, 1])
    assert type(sc) is ngto.Cell
    assert sc.natm == 4 and sc.nao_nr() == 16
    assert sc.atom_charges().tolist() == [4, 4, 4, 4]
    assert abs(sc.vol - 2 * cell.vol) <= 2 * cell.vol * 1e-12
    assert sc.mesh == [2 * cell.mesh[0], cell.mesh[1], cell.mesh[2]]


def test_band_path_fcc():
    cell = silicon()
    assert ngto.detect_lattice(cell) == "fcc"
    path = ngto.band_path(cell, npoints=40)
    assert isinstance(path, ngto.KPath)
    assert path.kpts.shape == (len(path), 3) and len(path) > 0
    assert path.tick_labels[0] == "G"
    seg = ngto.band_path_from_segments(cell, [[("G", (0, 0, 0)), ("X", (0.5, 0, 0.5))]], 10)
    assert len(seg) > 0


def test_dumps_loads_pickle_round_trip_bitwise():
    cell = diamond()
    for other in (ngto.loads(cell.dumps()), pickle.loads(pickle.dumps(cell)),
                  ngto.unpack(ngto.pack(cell)), cell.copy()):
        assert (bits(other.lattice_vectors()) == bits(cell.lattice_vectors())).all()
        assert other.atom_charges().tolist() == [4, 4]
        assert other.nao_nr() == 8 and other.mesh == cell.mesh
        assert (bits(other.pbc_intor("int1e_ovlp")) == bits(cell.pbc_intor("int1e_ovlp"))).all()


def test_mesh_is_a_plain_attribute_after_build():
    cell = helium()
    cell.mesh = [15, 15, 15]
    assert cell._built and cell.mesh == [15, 15, 15]
    cell.basis = "sto-3g"  # a build input: the build is dropped until build()
    assert not cell._built
    with pytest.raises(ValueError, match="build"):
        cell.lattice_vectors()
    cell.build()
    assert cell.mesh == [15, 15, 15]


# ── the refusals raise ──────────────────────────────────────────────────────

def _slab(**kw):
    return ngto.M(a=[[4.0, 0, 0], [0, 4.0, 0], [0, 0, 12.0]], atom=[("He", (0, 0, 0))],
                  basis="sto-3g", unit="Bohr", dimension=2, mesh=[9, 9, 25], **kw)


def test_refusal_dimension_1_uniform_grid():
    # coulg.rs:180 (get_coulG, dimension = 1) sits behind Cell::build's own
    # refusal (cell.rs:563): a dimension-1 cell without inf_vacuum cannot be built.
    with pytest.raises(native.PyscfRsRuntimeError, match="dimension=1"):
        ngto.M(a=[[4.0, 0, 0], [0, 12.0, 0], [0, 0, 12.0]], atom=[("He", (0, 0, 0))],
               basis="sto-3g", unit="Bohr", dimension=1)


def test_refusal_vcut_sph_below_3d():
    with pytest.raises(native.PyscfRsRuntimeError) as e:
        ngto.get_coulG(_slab(), exx="vcut_sph")
    assert_refusal(e)


def test_refusal_vcut_ws_below_3d():
    with pytest.raises(native.PyscfRsRuntimeError):
        ngto.get_coulG(_slab(), exx="vcut_ws", kpts=[[0.0, 0.0, 0.0]])


def test_refusal_super_cell_with_space_group_symmetry():
    cell = diamond(space_group_symmetry=True)
    with pytest.raises(native.PyscfRsRuntimeError) as e:
        ngto.super_cell(cell, [2, 1, 1])
    assert_refusal(e)


def test_refusal_pbc_intor_outside_family_and_spinor():
    cell = helium()
    with pytest.raises(native.PyscfRsRuntimeError) as e:
        cell.pbc_intor("int1e_rinv")
    assert_refusal(e)
    with pytest.raises(native.PyscfRsRuntimeError) as e:
        cell.pbc_intor("int1e_ovlp_spinor")
    assert_refusal(e)


def test_make_kpts_with_symmetry_points_at_pbc_symm():
    with pytest.raises(NotImplementedError, match="symm"):
        diamond().make_kpts([2, 2, 2], space_group_symmetry=True)


def test_get_coulG_3d_runs():
    cell = helium(mesh=[9, 9, 9])
    v = ngto.get_coulG(cell)
    assert v.shape == (729,)


# ── extract_cell_from_pyany: an upstream-shaped Cell drives a native builder ─

class _UpstreamShapedCell:
    """Duck-types the attributes `bridge::extract_cell_from_pyany` reads off an
    upstream `pyscf.pbc.gto.Cell` (Bohr quantities after `build()`)."""

    def __init__(self):
        self._atom = [("He", [0.0, 0.0, 0.0])]
        self.basis = "sto-3g"
        self.pseudo = None
        self.mesh = [15, 15, 15]
        self.precision = 1e-8
        self.dimension = 3
        self.low_dim_ft_type = None
        self.ke_cutoff = None
        self.exp_to_discard = None
        self.charge = 0
        self.spin = 0
        self.cart = False

    def lattice_vectors(self):
        return np.array(fcc(H_HE))

    def atom_coords(self):
        return np.array([[0.0, 0.0, 0.0]])


def test_extract_cell_from_pyany_upstream_shape_and_json():
    import pyscf._native.pbc.df as ndf

    ref = helium(mesh=[15, 15, 15])
    kpts = ref.make_kpts([2, 1, 1])
    h_ref = ndf.FFTDF(ref, kpts).get_hcore(kpts)
    fake = _UpstreamShapedCell()
    df = ndf.FFTDF(fake, kpts)
    assert df.cell is fake
    for a, b in zip(df.get_hcore(kpts), h_ref):
        assert (bits(a) == bits(b)).all()
    df_json = ndf.FFTDF(ref.dumps(), kpts)
    for a, b in zip(df_json.get_hcore(kpts), h_ref):
        assert (bits(a) == bits(b)).all()
    with pytest.raises(TypeError):
        ndf.FFTDF(object(), kpts)
