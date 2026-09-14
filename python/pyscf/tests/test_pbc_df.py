"""Plan 20-10 — the periodic DF binding (`pyscf._native.pbc.df`).

One native base class (`PeriodicDf`) over `trait PeriodicDf`; `FFTDF`, `AFTDF`,
`GDF`, `MDF`, `RSDF` construct; `PWDF`/`DF`/`RSGDF` are the same type objects.

"Bitwise vs Rust": no Rust test prints `vj`/`vk` for these fixtures, and a Rust
reference binary (test or example) re-unifies features and rebuilds ~500 libxc
kernel crates (memory: gate-target-dir-lto-spelling). So the binding is pinned to
the Rust function's bits by (a) exact repeat determinism on fresh builders,
(b) route identity — the SAME Rust `get_jk` reached through the concrete
constructor, through `pyscf_pbc_df::density_fit` (a `Box<dyn PeriodicDf>`),
and through the driver handle (`extract_df` -> `SharedDf`) must agree to the
bit, as must every accepted `dm` spelling — and (c) `get_hcore` rebuilt in
numpy from `get_nuc` + `pbc_intor('int1e_kin')`, which is the Rust body's own
element-wise addition (`fftdf.rs:565`). Upstream is compared only where a
floor exists: `kpts_band` GDF J/K at the `band_kpoints.rs` gate, `< 2e-9`
(measurements/README.md §1 row 13).
"""

import json
import os
import subprocess
import sys
import textwrap

import numpy as np
import pytest

import pyscf._native as native
import pyscf._native.pbc.df as ndf
import pyscf._native.pbc.gto as ngto

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
H_HE = 2.834589


def fcc(h):
    return [[0.0, h, h], [h, 0.0, h], [h, h, 0.0]]


def helium(**kw):
    return ngto.M(a=fcc(H_HE), atom=[("He", (0, 0, 0))], basis="sto-3g", unit="Bohr", **kw)


def bits(a):
    a = np.ascontiguousarray(a)
    if np.iscomplexobj(a):
        a = a.view(np.float64)
    return np.asarray(a, dtype=np.float64).view(np.uint64)


def same_bits(xs, ys):
    xs = xs if isinstance(xs, list) else [xs]
    ys = ys if isinstance(ys, list) else [ys]
    assert len(xs) == len(ys)
    return all(x.shape == y.shape and (bits(x) == bits(y)).all() for x, y in zip(xs, ys))


def model_dm(nao, nkpts):
    """`band_kpoints.rs::model_dm` — real, Hermitian, deterministic."""
    out = []
    for k in range(nkpts):
        m = np.zeros((nao, nao))
        for p in range(nao):
            for q in range(nao):
                v = 0.3 / (1.0 + abs(p - q)) + (1.0 if p == q else 0.0)
                m[p, q] = v * (1.0 + 0.1 * k)
        out.append(m)
    return out


def assert_refusal(excinfo):
    err = excinfo.value
    assert isinstance(err, native.PyscfRsRuntimeError)
    assert err.args[1] == "NotYetImplemented", err.args


@pytest.fixture(scope="module")
def he15():
    cell = helium(mesh=[15, 15, 15])
    return cell, cell.make_kpts([2, 2, 2])


@pytest.fixture(scope="module")
def he_gdf_built(he15, tmp_path_factory):
    cell, kpts = he15
    path = str(tmp_path_factory.mktemp("cderi") / "he_gdf.h5")
    df = ndf.GDF(cell, kpts)
    df._cderi_to_save = path
    df.build()
    return df, path


# ── the class surface ───────────────────────────────────────────────────────

def test_aliases_are_the_same_objects():
    assert ndf.PWDF is ndf.AFTDF
    assert ndf.DF is ndf.GDF
    assert ndf.RSGDF is ndf.RSDF
    for cls in (ndf.FFTDF, ndf.AFTDF, ndf.GDF, ndf.MDF, ndf.RSDF):
        assert issubclass(cls, ndf.PeriodicDf)
        assert cls.__module__ == "pyscf._native.pbc.df"


def test_overlay_names_are_native():
    import pyscf.pbc.df as odf

    for name in ("FFTDF", "AFTDF", "GDF", "MDF", "RSDF", "PWDF", "DF"):
        assert getattr(odf, name) is getattr(ndf, name)


@pytest.mark.parametrize("cls,name", [(ndf.FFTDF, "FFTDF"), (ndf.AFTDF, "AFTDF"), (ndf.GDF, "GDF"),
                                      (ndf.MDF, "MDF"), (ndf.RSDF, "RSGDF")])
def test_construct_all_five(he15, cls, name):
    cell, kpts = he15
    df = cls(cell, kpts)
    assert isinstance(df, ndf.PeriodicDf) and type(df) is cls
    assert df.name == name
    assert df.cell is cell
    assert (bits(df.kpts) == bits(kpts)).all()
    assert cls.__name__ in repr(df)
    gamma = cls(cell)
    assert gamma.kpts.shape == (1, 3) and not gamma.kpts.any()


def test_density_fit_returns_the_matching_class(he15):
    cell, kpts = he15
    for kind, cls in (("FFTDF", ndf.FFTDF), ("AFTDF", ndf.AFTDF), ("GDF", ndf.GDF),
                      ("MDF", ndf.MDF), ("RSDF", ndf.RSDF), ("DF", ndf.GDF)):
        assert type(ndf.density_fit(cell, kpts, kind)) is cls


# ── get_jk: bitwise route identity + determinism ────────────────────────────

def test_fftdf_get_jk_bitwise_across_routes_and_input_forms(he15):
    cell, kpts = he15
    dm = model_dm(1, 8)
    df = ndf.FFTDF(cell, kpts)
    vj, vk = df.get_jk(dm, exxdiv="ewald")
    assert isinstance(vj, list) and len(vj) == 8 and vj[0].shape == (1, 1)
    assert vj[0].dtype == np.complex128

    # (a) a fresh builder, same call
    vj2, vk2 = ndf.FFTDF(cell, kpts).get_jk(dm, exxdiv="ewald")
    assert same_bits(vj, vj2) and same_bits(vk, vk2)
    # (b) pyscf_pbc_df::density_fit (Box<dyn PeriodicDf>)
    vj3, vk3 = ndf.density_fit(cell, kpts, "FFTDF").get_jk(dm, exxdiv="ewald")
    assert same_bits(vj, vj3) and same_bits(vk, vk3)
    # (c) the driver handle: extract_df -> SharedDf -> get_jk
    name, mesh, vj4, vk4 = ndf._driver_handle_get_jk(df, dm, "ewald")
    assert name == "FFTDF" and mesh == [15, 15, 15]
    assert same_bits(vj, vj4) and same_bits(vk, vk4)
    # every dm spelling reaches the same Rust call
    stacked = np.array(dm)
    for form in (stacked, tuple(dm), [np.asarray(d, dtype=complex) for d in dm]):
        a, b = df.get_jk(form, exxdiv="ewald")
        assert same_bits(vj, a) and same_bits(vk, b)
    nj, nk = df.get_jk([dm, dm], exxdiv="ewald")
    assert len(nj) == 2 and same_bits(vj, nj[1]) and same_bits(vk, nk[0])
    nj4, _ = df.get_jk(np.array([dm, dm]), exxdiv="ewald")
    assert same_bits(nj[0], nj4[0])
    # a half not asked for is None; kk_symmetry off equals the default
    vj_only, none = df.get_jk(dm, with_k=False, kk_symmetry=False)
    assert none is None and same_bits(vj, vj_only)


def test_gdf_get_jk_bitwise_across_routes(he15, he_gdf_built):
    cell, kpts = he15
    df, _ = he_gdf_built
    dm = model_dm(1, 8)
    vj, vk = df.get_jk(dm, exxdiv="ewald")
    vj2, vk2 = df.get_jk(dm, exxdiv="ewald")
    assert same_bits(vj, vj2) and same_bits(vk, vk2)
    name, _, vj3, vk3 = ndf._driver_handle_get_jk(df, dm, "ewald")
    assert name == "GDF" and same_bits(vj, vj3) and same_bits(vk, vk3)
    fresh = ndf.density_fit(cell, kpts, "GDF")
    vj4, vk4 = fresh.get_jk(dm, exxdiv="ewald")
    assert same_bits(vj, vj4) and same_bits(vk, vk4)


def test_with_df_is_assignable_after_construction(he15):
    """`mf.with_df = AFTDF(cell, kpts)` (PBC-MASTER-PLAN:1909): a driver keeps the
    Python object and re-fetches the builder at every call, so a reassignment
    or a mutation after assignment is what the next call sees."""
    cell, kpts = he15
    dm = model_dm(1, 8)

    class Holder:  # stands in for a 20-12 driver's `with_df` slot
        pass

    mf = Holder()
    mf.with_df = ndf.FFTDF(cell, kpts)
    assert ndf._driver_handle_get_jk(mf.with_df, dm)[0] == "FFTDF"
    mf.with_df = ndf.AFTDF(cell, kpts)
    name, mesh, vj, _ = ndf._driver_handle_get_jk(mf.with_df, dm)
    assert name == "AFTDF" and mesh == [15, 15, 15]
    mf.with_df.mesh = [11, 11, 11]
    name, mesh, _, _ = ndf._driver_handle_get_jk(mf.with_df, dm)
    assert mesh == [11, 11, 11]
    with pytest.raises(TypeError):
        ndf._driver_handle_get_jk(object(), dm)


def test_single_matrix_and_single_band_shapes(he15):
    cell, _ = he15
    df = ndf.FFTDF(cell)  # gamma
    vj, vk = df.get_jk(np.eye(1), exxdiv=None)
    assert isinstance(vj, np.ndarray) and vj.shape == (1, 1)
    vjb, _ = df.get_jk(np.eye(1), kpts_band=np.array([0.1, 0.0, 0.0]))
    assert isinstance(vjb, np.ndarray) and vjb.shape == (1, 1)


# ── get_hcore lives on the DF object ────────────────────────────────────────

def test_get_hcore_is_nuc_plus_kinetic_bitwise(he15):
    cell, kpts = he15
    df = ndf.FFTDF(cell, kpts)
    h = df.get_hcore(kpts)
    nuc = df.get_nuc(kpts)
    kin = cell.pbc_intor("int1e_kin", kpts=kpts)
    assert len(h) == 8
    for hk, vk_, tk in zip(h, nuc, kin):
        assert (bits(hk) == bits(vk_ + tk)).all()
    h0 = df.get_hcore()
    assert isinstance(h0, np.ndarray) and h0.shape == (1, 1)


def test_get_pp_on_a_pseudopotential_cell():
    q = 1.68516
    cell = ngto.M(a=fcc(3.37032), atom=[("C", (0, 0, 0)), ("C", (q, q, q))], basis="gth-szv",
                  pseudo="gth-pade", unit="Bohr", mesh=[11, 11, 11])
    df = ndf.FFTDF(cell)  # (AFTDF.get_pp here measured 244 s on a loaded box)
    pp = df.get_pp()
    assert pp.shape == (8, 8)
    np.testing.assert_allclose(pp, pp.conj().T, rtol=0, atol=1e-10)


# ── GDF persistence ─────────────────────────────────────────────────────────

def test_gdf_cderi_hdf5_round_trip(he15, he_gdf_built):
    cell, kpts = he15
    df, path = he_gdf_built
    assert os.path.isfile(path)
    assert df._cderi == path and df._cderi_to_save == path
    assert df.has_cderi()
    dm = model_dm(1, 8)
    vj, vk = df.get_jk(dm, exxdiv="ewald")

    again = ndf.GDF(cell, kpts)
    again._cderi = path  # upstream's `mydf._cderi = 'f.h5'`: read, never refit
    assert again._cderi == path
    vj2, vk2 = again.get_jk(dm, exxdiv="ewald")
    assert same_bits(vj, vj2) and same_bits(vk, vk2)
    assert again.get_naoaux() == df.get_naoaux() > 0
    blocks = again.sr_loop(compact=True)
    assert blocks and blocks[0][0].shape[0] > 0 and blocks[0][2] in (1, -1)


def test_ao2mo_surface_shapes(he15, he_gdf_built):
    cell, kpts = he15
    df, _ = he_gdf_built
    g4 = np.zeros((4, 3))
    eri = df.get_eri(g4)
    assert eri.shape == (1, 1) and eri.dtype == np.float64
    assert same_bits(df.get_eri(), eri)
    mo = np.eye(1)
    assert df.ao2mo(mo, g4).shape == (1, 1)
    e7 = df.ao2mo_7d([np.eye(1, dtype=complex)] * 8)
    assert e7.shape == (8, 8, 8, 1, 1, 1, 1)
    with pytest.raises(native.PyscfRsRuntimeError):
        ndf.FFTDF(cell, kpts).get_naoaux()


# ── upstream, where a floor exists ──────────────────────────────────────────

GDF_BAND_SCRIPT = textwrap.dedent(
    """
    import json, sys
    import numpy as np
    import pyscf
    from pyscf.pbc import gto as pgto
    from pyscf.pbc.df import df as pbcdf
    a, kpts, kband, dm = (json.loads(x) for x in sys.argv[1:5])
    cell = pgto.Cell()
    cell.a = a
    cell.atom = [('He', [0.0, 0.0, 0.0])]
    cell.basis = 'sto-3g'
    cell.unit = 'Bohr'
    cell.verbose = 0
    cell.build()
    kpts = np.array(kpts)
    mydf = pbcdf.GDF(cell, kpts)
    mydf.build()
    vj, vk = mydf.get_jk(np.array(dm), hermi=1, kpts=kpts, kpts_band=np.array(kband),
                         with_j=True, with_k=True, exxdiv='ewald')
    print(json.dumps({'version': pyscf.__version__,
                      'vj_re': vj.real.ravel().tolist(), 'vj_im': vj.imag.ravel().tolist(),
                      'vk_re': vk.real.ravel().tolist(), 'vk_im': vk.imag.ravel().tolist()}))
    """
)


def test_gdf_kpts_band_matches_upstream_at_the_band_kpoints_gate():
    """`crates/pyscf-pbc-df/tests/band_kpoints.rs` through the binding: He-fcc
    sto-3g (default mesh), 2x2x2 sampling + two band k-points, `< 2e-9`."""
    cell = helium()
    kpts = cell.make_kpts([2, 2, 2])
    kband = np.array([[0.15, -0.07, 0.03], [-0.05, 0.11, 0.02]])
    dm = model_dm(1, 8)
    vj, vk = ndf.GDF(cell, kpts).get_jk(dm, hermi=1, kpts=kpts, kpts_band=kband, exxdiv="ewald")
    assert len(vj) == 2

    env = dict(os.environ, PYTHONPATH=REPO)
    args = [json.dumps(fcc(H_HE)), json.dumps(kpts.tolist()), json.dumps(kband.tolist()),
            json.dumps([d.tolist() for d in dm])]
    proc = subprocess.run([sys.executable, "-c", GDF_BAND_SCRIPT, *args], cwd=REPO, env=env,
                          capture_output=True, text=True, check=False)
    assert proc.returncode == 0, proc.stderr
    up = json.loads([ln for ln in proc.stdout.splitlines() if ln.startswith("{")][-1])
    assert up["version"] == "2.12.1"
    got_j = np.array(vj).ravel()
    got_k = np.array(vk).ravel()
    dj = max(np.abs(got_j.real - up["vj_re"]).max(), np.abs(got_j.imag - up["vj_im"]).max())
    dk = max(np.abs(got_k.real - up["vk_re"]).max(), np.abs(got_k.imag - up["vk_im"]).max())
    print(f"GDF band through the binding: |dvj|={dj:e} |dvk|={dk:e}")
    assert dj < 2e-9, dj
    assert dk < 2e-9, dk


# ── the remaining refusals raise ────────────────────────────────────────────

@pytest.mark.parametrize("cls", [ndf.GDF, ndf.MDF])
def test_refusal_exp_to_discard(he15, cls):
    cell, kpts = he15
    df = cls(cell, kpts)
    df.exp_to_discard = 0.1
    assert df.exp_to_discard == 0.1
    with pytest.raises(native.PyscfRsRuntimeError) as e:
        df.build()
    assert_refusal(e)


def test_refusal_cartesian_fused_auxcell():
    cell = helium(cart=True, mesh=[15, 15, 15])
    with pytest.raises(native.PyscfRsRuntimeError) as e:
        ndf.GDF(cell).build()
    assert_refusal(e)


def test_gdf_get_jk_omega_is_a_real_kwarg(he15):
    """20-05 landed `GDF.get_jk(omega)` (`omega > 0`: AFTDF on an omega-derived
    mesh, `df.py:470-474`). The binding forwards it: the result is deterministic
    to the bit and is NOT the plain-Coulomb answer; `omega=0` IS plain Coulomb."""
    cell, kpts = he15
    dm = model_dm(1, 8)
    df = ndf.GDF(cell, kpts)
    vj_lr, vk_lr = df.get_jk(dm, omega=0.3, exxdiv="ewald")
    vj_lr2, vk_lr2 = ndf.GDF(cell, kpts).get_jk(dm, omega=0.3, exxdiv="ewald")
    assert same_bits(vj_lr, vj_lr2) and same_bits(vk_lr, vk_lr2)
    vj, vk = df.get_jk(dm, exxdiv="ewald")
    assert not same_bits(vj, vj_lr)
    vj0, vk0 = df.get_jk(dm, omega=0.0, exxdiv="ewald")
    assert same_bits(vj, vj0) and same_bits(vk, vk0)


def test_mdf_prefer_ccdf_defaults_to_upstream_false(he15):
    """Upstream `MDF._prefer_ccdf = False` (`pyscf/pbc/df/mdf.py:80`); the crate's
    `Mdf::new` says `true`, so the binding pins upstream's default."""
    cell, kpts = he15
    assert ndf.MDF(cell, kpts)._prefer_ccdf is False
    assert ndf.density_fit(cell, kpts, "MDF")._prefer_ccdf is False
    assert ndf.GDF(cell, kpts)._prefer_ccdf is False


@pytest.mark.parametrize("cls", [ndf.GDF, ndf.MDF])
def test_refusal_short_range_omega_with_prefer_ccdf(he15, cls):
    """The remaining named 20-05 refusal: `omega < 0` under `_prefer_ccdf = True`
    (`gdf/jk.rs`, `mdf/mdf_jk.rs` — upstream's `_CC*DFBuilder` on an attenuated
    cell is not ported). It must raise, never return plain Coulomb."""
    cell, kpts = he15
    df = cls(cell, kpts)
    df._prefer_ccdf = True
    with pytest.raises(native.PyscfRsRuntimeError) as e:
        df.get_jk(model_dm(1, 8), omega=-0.3)
    assert_refusal(e)


def test_gdf_mesh_setter_refuses_and_wrong_attrs_are_attribute_errors(he15):
    cell, kpts = he15
    with pytest.raises(NotImplementedError):
        ndf.GDF(cell, kpts).mesh = [9, 9, 9]
    assert not hasattr(ndf.FFTDF(cell, kpts), "auxbasis")
    assert not hasattr(ndf.FFTDF(cell, kpts), "_cderi")
