//! `FFTDF.get_nuc` BIT-identical to upstream PySCF 2.12.1 on He/STO-3G.
//!
//! Gated on `PYSCF_ORACLE_VENV` like the rest of the upstream layer:
//!
//! ```bash
//! PYSCF_ORACLE_VENV=1 cargo test -p pyscf-pbc-df --release --test get_nuc_bitexact -- --ignored
//! ```
//!
//! Every value crosses the process boundary as its raw `u64` bit pattern, so
//! no decimal round trip can move a last bit, and the assertions compare
//! `to_bits()`. The oracle runs with `OMP_NUM_THREADS=1`: upstream's `NPdgemm`
//! splits `lib.dot`'s K axis over threads and merges the partial sums in
//! `omp critical` order, so the multi-threaded upstream is not reproducible
//! even against itself (measured: the He imaginary parts change between runs).
//!
//! `cell._env` (the normalised contraction coefficients) is taken from
//! upstream: pyscf-rs's `Mole` normalisation is 1 ulp off upstream's on this
//! cell for reasons outside `get_nuc` (numpy's SVML `pow`, cephes `gamma`),
//! and the gate is on what `get_nuc` does with identical inputs.
//!
//! Besides the end result, each stage is checked on its own — grid
//! coordinates, `coulG`, `vneR`, the AO table — so that a regression names the
//! stage that moved.

mod common;

use common::{
    GATE, cell_args, diamond_all_electron, he_all_electron, he2_off_origin, oracle_python,
    run_python,
};
use pyscf_algebra::CTensor;
use pyscf_algebra::openblas_emu::{dgemm_nt, zgemm_nt};
use pyscf_core::Unit;
use pyscf_gto::{AtomInput, BasisInput, MoleBuildArgs};
use pyscf_pbc_df::{Fftdf, PeriodicDf, aor_loop_blocks};
use pyscf_pbc_gto::{
    ALattice, Cell, CellBuildArgs, eval_ao_kpts_upstream, get_coulg_at_gv, get_gv, get_si,
    make_kpts_default,
};

const MESH: [usize; 3] = [11, 11, 11];

const ORACLE_PY: &str = r#"
import os
os.environ['OMP_NUM_THREADS'] = '1'
import json, struct, sys
import numpy as np
from pyscf.pbc import gto, df, tools
from pyscf.pbc.dft import numint

def bits(x):
    x = np.ascontiguousarray(np.asarray(x, dtype=np.float64)).ravel()
    return [int(v) for v in x.view(np.uint64)]

a_json, xyz_json, sym_json, basis, nk_json, mesh_json = sys.argv[1:7]
c = gto.Cell()
c.a = json.loads(a_json)
c.atom = [(s, tuple(r)) for s, r in zip(json.loads(sym_json), json.loads(xyz_json))]
c.basis = basis
c.unit = 'Bohr'
c.verbose = 0
c.build()
kpts = c.make_kpts(json.loads(nk_json))
mesh = json.loads(mesh_json)
mydf = df.FFTDF(c, kpts)
mydf.mesh = mesh

nuc = np.asarray(mydf.get_nuc(kpts), dtype=np.complex128)
Gv = c.get_Gv(mesh)
coulG = tools.get_coulG(c, mesh=mesh, Gv=Gv)
vneR = tools.ifft(np.dot(-c.atom_charges(), c.get_SI(mesh=mesh)) * coulG, mesh).real
ao = numint.eval_ao_kpts(c, mydf.grids.coords, kpts)
out = {
    'version': __import__('pyscf').__version__,
    'nkpts': len(kpts), 'nao': int(c.nao_nr()),
    'env': bits(c._env),
    'nuc_re': bits(nuc.real), 'nuc_im': bits(nuc.imag),
    'coords': bits(mydf.grids.coords),
    'coulG': bits(coulG),
    'vneR': bits(vneR),
    # (nkpts, nao, ngrids) — the Rust layout
    'ao_re': bits([np.real(x).T for x in ao]),
    'ao_im': bits([np.imag(x).T for x in ao]),
}
print(json.dumps(out))
"#;

fn pull(v: &serde_json::Value, key: &str) -> Vec<u64> {
    v[key]
        .as_array()
        .unwrap_or_else(|| panic!("oracle payload has no {key}"))
        .iter()
        .map(|x| x.as_u64().expect("u64 bit pattern"))
        .collect()
}

/// Count of elements whose bits differ, plus the largest absolute gap.
fn bit_diff(got: &[f64], want: &[u64]) -> (usize, f64) {
    assert_eq!(got.len(), want.len(), "length mismatch");
    got.iter().zip(want).fold((0, 0.0_f64), |(n, w), (g, &b)| {
        let u = f64::from_bits(b);
        if g.to_bits() == b {
            (n, w)
        } else {
            (n + 1, w.max((g - u).abs()))
        }
    })
}

fn assert_bits(what: &str, got: &[f64], want: &[u64]) {
    let (n, w) = bit_diff(got, want);
    println!(
        "{what}: {n}/{} elements differ in the bits, max |delta| = {w:e}",
        got.len()
    );
    assert_eq!(
        n, 0,
        "{what} is not bit-identical to upstream ({n} elements, max {w:e})"
    );
}

#[test]
#[ignore = "T1: needs PYSCF_ORACLE_VENV + the vendored upstream PySCF"]
fn get_nuc_is_bit_identical_to_upstream_on_he() {
    let Some(py) = oracle_python() else {
        eprintln!("SKIP: {GATE} is not set");
        return;
    };
    let cell = he_all_electron();
    let kpts = make_kpts_default(&cell, [2, 2, 2]).expect("2x2x2 k-mesh");
    let args = cell_args(
        &cell,
        &[
            "sto-3g".to_string(),
            serde_json::to_string(&[2, 2, 2]).expect("json"),
            serde_json::to_string(&MESH.to_vec()).expect("json"),
        ],
    );
    let want = run_python(&py, ORACLE_PY, &args);
    assert_eq!(
        want["version"].as_str(),
        Some("2.12.1"),
        "vendored PySCF only"
    );

    // The one input this port cannot yet reproduce: upstream normalises the
    // contraction coefficients with numpy ARRAY arithmetic (`gto_norm`,
    // `_nomalize_contracted_ao`), and on an AVX-512 host numpy's `a**n1` is
    // SVML's, not glibc's; `scipy.special.gamma(1.5)` (cephes) is also 1 ulp
    // off the correctly rounded value. He/STO-3G's `_env` lands 1 ulp off in
    // two coefficients. That is `Mole` construction, shared by every
    // integral, not `get_nuc` — so pin upstream's `_env` here and gate the
    // `get_nuc` pipeline on identical inputs.
    let want_env = pull(&want, "env");
    let (n_env, w_env) = bit_diff(&cell.mol._env, &want_env);
    println!(
        "cell._env: {n_env}/{} entries differ from upstream (max {w_env:e}) — pinned below",
        want_env.len()
    );
    let mut cell = cell;
    cell.mol._env = want_env.iter().map(|&b| f64::from_bits(b)).collect();

    let df = Fftdf::with_mesh(cell.clone(), &kpts, MESH).expect("FFTDF");

    // Stage 1 — the uniform grid (numpy's `fftfreq` multiplies by 1/n).
    let coords: Vec<f64> = df.grids.coords.iter().flatten().copied().collect();
    assert_bits("grid coords", &coords, &pull(&want, "coords"));

    // Stage 2 — coulG (einsum's `(x^2 + z^2) + y^2`).
    let gv = get_gv(&cell, Some(MESH)).expect("Gv");
    let coulg = get_coulg_at_gv(&cell, MESH, &gv).expect("coulG");
    assert_bits("coulG", &coulg, &pull(&want, "coulG"));

    // Stage 3 — vneR through the pocketfft port. The SI is the separable
    // branch (`get_si(cell, None, Some(mesh), None)`), like `get_nuc` uses.
    let si = get_si(&cell, None, Some(MESH), None).expect("SI");
    let z = -(cell.atom_charges()[0] as f64);
    let re: Vec<f64> = (0..gv.len()).map(|g| z * si.re[g] * coulg[g]).collect();
    let im: Vec<f64> = (0..gv.len()).map(|g| z * si.im[g] * coulg[g]).collect();
    let vner = pyscf_pbc_tools::ifft_upstream(&CTensor::from_planes(re, im), MESH).expect("ifft");
    assert_bits("vneR", &vner.re, &pull(&want, "vneR"));

    // Stage 4 — the AO table (PBCeval_sph_iter's screen and image sum).
    let ao = eval_ao_kpts_upstream(&cell, &df.grids.coords, &kpts)
        .expect("eval")
        .expect("He/STO-3G is all-s");
    let ao_re: Vec<f64> = ao.kaos.iter().flat_map(|t| t.re.iter().copied()).collect();
    let ao_im: Vec<f64> = ao.kaos.iter().flat_map(|t| t.im.iter().copied()).collect();
    assert_bits("AO real part", &ao_re, &pull(&want, "ao_re"));
    assert_bits("AO imaginary part", &ao_im, &pull(&want, "ao_im"));

    // The result.
    let got = df.get_nuc(&kpts).expect("get_nuc");
    let nuc_re: Vec<f64> = got.iter().flat_map(|t| t.re.iter().copied()).collect();
    let nuc_im: Vec<f64> = got.iter().flat_map(|t| t.im.iter().copied()).collect();
    assert_bits("get_nuc real part", &nuc_re, &pull(&want, "nuc_re"));
    assert_bits("get_nuc imaginary part", &nuc_im, &pull(&want, "nuc_im"));
}

const BLAS_PY: &str = r#"
import ctypes, glob, json, os, sys
import numpy as np
from pyscf.lib import numpy_helper  # loads the wheel's libgfortran/libgomp first
# The OpenBLAS that `libnp_helper` (lib.dot) is linked against sits beside it.
here = os.path.dirname(numpy_helper._np_helper._name)
blas = ctypes.CDLL(glob.glob(os.path.join(here, 'libopenblas*.so'))[0])
blas.openblas_get_corename.restype = ctypes.c_char_p
I, D, P = ctypes.c_int, ctypes.c_double, ctypes.c_void_p
def r(x): return ctypes.byref(x)
rng = np.random.default_rng(20260918)
cases = []
for m, n, k in json.loads(sys.argv[1]):
    a = rng.standard_normal(m*k) + 1j*rng.standard_normal(m*k)
    b = rng.standard_normal(n*k) + 1j*rng.standard_normal(n*k)
    ar, br = np.ascontiguousarray(a.real), np.ascontiguousarray(b.real)
    c = np.zeros(m*n)
    blas.dgemm_(b'N', b'T', r(I(m)), r(I(n)), r(I(k)), r(D(1.0)), ar.ctypes.data_as(P), r(I(m)),
                br.ctypes.data_as(P), r(I(n)), r(D(0.0)), c.ctypes.data_as(P), r(I(m)))
    z = np.zeros(m*n, complex)
    one = np.array([1.0+0j]); zero = np.array([0j])
    blas.zgemm_(b'N', b'T', r(I(m)), r(I(n)), r(I(k)), one.ctypes.data_as(P), a.ctypes.data_as(P),
                r(I(m)), b.ctypes.data_as(P), r(I(n)), zero.ctypes.data_as(P), z.ctypes.data_as(P), r(I(m)))
    u = lambda x: [int(v) for v in np.ascontiguousarray(x).view(np.uint64)]
    cases.append({'m': m, 'n': n, 'k': k, 'a_re': u(a.real), 'a_im': u(a.imag), 'b_re': u(b.real),
                  'b_im': u(b.imag), 'd': u(c), 'z_re': u(z.real), 'z_im': u(z.imag)})
print(json.dumps({'core': blas.openblas_get_corename().decode(), 'cases': cases}))
"#;

/// The emulators against the real `dgemm_`/`zgemm_` of the OpenBLAS that
/// upstream links, on shapes that cover every register-tile tail class and
/// the K blocking (`GEMM_Q = 224`, halving of the last two blocks).
#[test]
#[ignore = "T1: needs PYSCF_ORACLE_VENV + the vendored upstream PySCF"]
fn openblas_emulation_matches_the_wheel_blas() {
    let Some(py) = oracle_python() else {
        eprintln!("SKIP: {GATE} is not set");
        return;
    };
    let shapes: Vec<[usize; 3]> = [1, 2, 3, 4, 5, 6, 7]
        .iter()
        .flat_map(|&m| [1, 2, 3, 5, 6].iter().map(move |&n| (m, n)))
        .flat_map(|(m, n)| [1, 5, 40, 225, 449, 1331].iter().map(move |&k| [m, n, k]))
        .collect();
    let want = run_python(
        &py,
        BLAS_PY,
        &[serde_json::to_string(&shapes).expect("json")],
    );
    let core = want["core"].as_str().unwrap_or("?");
    assert_eq!(
        core, "Barcelona",
        "the emulation models OpenBLAS's Barcelona kernels; this host runs {core}"
    );
    let f = |v: &serde_json::Value, key: &str| -> Vec<f64> {
        pull(v, key).into_iter().map(f64::from_bits).collect()
    };
    for case in want["cases"].as_array().expect("cases") {
        let dims = ["m", "n", "k"].map(|d| case[d].as_u64().expect("dim") as usize);
        let [m, n, k] = dims;
        let (a_re, a_im, b_re, b_im) = (
            f(case, "a_re"),
            f(case, "a_im"),
            f(case, "b_re"),
            f(case, "b_im"),
        );
        let mut d = vec![0.0; m * n];
        dgemm_nt(m, n, k, &a_re, &b_re, &mut d);
        let (nd, _) = bit_diff(&d, &pull(case, "d"));
        let (mut zr, mut zi) = (vec![0.0; m * n], vec![0.0; m * n]);
        zgemm_nt(m, n, k, &a_re, &a_im, &b_re, &b_im, &mut zr, &mut zi);
        let (nzr, _) = bit_diff(&zr, &pull(case, "z_re"));
        let (nzi, _) = bit_diff(&zi, &pull(case, "z_im"));
        assert_eq!(
            (nd, nzr, nzi),
            (0, 0, 0),
            "m={m} n={n} k={k}: dgemm/zgemm.re/zgemm.im differ in {nd}/{nzr}/{nzi} elements"
        );
    }
}

/// The `_ifftn_blas` GEMM shape against the wheel's OpenBLAS: `zgemm_('N','T',
/// M, N, K, alpha = 1/M, B, A, beta = 0, C)` with `M` an FFT axis length, `N`
/// the product of the other axes (up to 5000 here — the real `N` is
/// `ngrids/mx`), `K = M` (plus one `K > M` general case).
const BLAS_ALPHA_PY: &str = r#"
import ctypes, glob, json, os, sys
import numpy as np
from pyscf.lib import numpy_helper  # loads the wheel's libgfortran/libgomp first
here = os.path.dirname(numpy_helper._np_helper._name)
blas = ctypes.CDLL(glob.glob(os.path.join(here, 'libopenblas*.so'))[0])
blas.openblas_get_corename.restype = ctypes.c_char_p
I, P = ctypes.c_int, ctypes.c_void_p
def r(x): return ctypes.byref(x)
rng = np.random.default_rng(20260918)
cases = []
for m, n, k in json.loads(sys.argv[1]):
    # F-order (M,K)/(N,K) faces passed as C-order (K,M)/(K,N) buffers,
    # exactly how `_zgemm('T','N',...)` hands them to `NPzgemm`.
    a = np.ascontiguousarray(rng.standard_normal((k,m)) + 1j*rng.standard_normal((k,m)))
    b = np.ascontiguousarray(rng.standard_normal((k,n)) + 1j*rng.standard_normal((k,n)))
    z = np.zeros(m*n, complex)
    alpha = np.array([1.0/m+0j]); zero = np.array([0j])
    blas.zgemm_(b'N', b'T', r(I(m)), r(I(n)), r(I(k)), alpha.ctypes.data_as(P), a.ctypes.data_as(P),
                r(I(m)), b.ctypes.data_as(P), r(I(n)), zero.ctypes.data_as(P), z.ctypes.data_as(P), r(I(m)))
    u = lambda x: [int(v) for v in np.ascontiguousarray(x).ravel().view(np.uint64)]
    cases.append({'m': m, 'n': n, 'k': k, 'a_re': u(a.real), 'a_im': u(a.imag), 'b_re': u(b.real),
                  'b_im': u(b.imag), 'z_re': u(z.real), 'z_im': u(z.imag)})
print(json.dumps({'core': blas.openblas_get_corename().decode(), 'cases': cases}))
"#;

#[test]
#[ignore = "T1: needs PYSCF_ORACLE_VENV + the vendored upstream PySCF"]
fn openblas_zgemm_alpha_matches_the_wheel_blas() {
    use pyscf_algebra::openblas_emu::zgemm;
    let Some(py) = oracle_python() else {
        eprintln!("SKIP: {GATE} is not set");
        return;
    };
    let shapes: Vec<[usize; 3]> =
        vec![[17, 100, 17], [17, 2209, 17], [47, 5000, 47], [23, 437, 60]];
    let want = run_python(
        &py,
        BLAS_ALPHA_PY,
        &[serde_json::to_string(&shapes).expect("json")],
    );
    let core = want["core"].as_str().unwrap_or("?");
    assert_eq!(
        core, "Barcelona",
        "the emulation models OpenBLAS's Barcelona kernels; this host runs {core}"
    );
    let f = |v: &serde_json::Value, key: &str| -> Vec<f64> {
        pull(v, key).into_iter().map(f64::from_bits).collect()
    };
    for case in want["cases"].as_array().expect("cases") {
        let dims = ["m", "n", "k"].map(|d| case[d].as_u64().expect("dim") as usize);
        let [m, n, k] = dims;
        let (a_re, a_im, b_re, b_im) = (
            f(case, "a_re"),
            f(case, "a_im"),
            f(case, "b_re"),
            f(case, "b_im"),
        );
        let (mut zr, mut zi) = (vec![0.0; m * n], vec![0.0; m * n]);
        zgemm(
            'N',
            'T',
            m,
            n,
            k,
            (1.0 / m as f64, 0.0),
            &a_re,
            &a_im,
            m,
            &b_re,
            &b_im,
            n,
            (0.0, 0.0),
            &mut zr,
            &mut zi,
            m,
        );
        let (nzr, _) = bit_diff(&zr, &pull(case, "z_re"));
        let (nzi, _) = bit_diff(&zi, &pull(case, "z_im"));
        assert_eq!(
            (nzr, nzi),
            (0, 0),
            "m={m} n={n} k={k}: zgemm.re/zgemm.im differ in {nzr}/{nzi} elements"
        );
    }
}

/// F1: `get_nuc` over several `aoR_loop` blocks, bit-identical.
///
/// He/STO-3G, gamma only, mesh `[52, 52, 52]` (140 608 points). The plan
/// suggests `[45, 45, 45]`, but at the oracle-pinned `max_memory = 2000` that
/// mesh is a single block (`blksize = 91168 > 91125` points); `[52, 52, 52]`
/// is the smallest cubic mesh with two blocks (`blksize = 134400`, via the
/// 2400 cap). `52` is neither Bluestein-planned (first Bluestein length is
/// 89) nor in `_EXCLUDE`, so the FFT stays on the pocketfft route (F3/F4
/// untouched).
const MESH_MULTI: [usize; 3] = [52, 52, 52];

const ORACLE_MULTI_PY: &str = r#"
import os
os.environ['OMP_NUM_THREADS'] = '1'
import json, sys
import numpy as np
from pyscf.pbc import gto, df

def bits(x):
    x = np.ascontiguousarray(np.asarray(x, dtype=np.float64)).ravel()
    return [int(v) for v in x.view(np.uint64)]

a_json, xyz_json, sym_json, basis, nk_json, mesh_json = sys.argv[1:7]
c = gto.Cell()
c.a = json.loads(a_json)
c.atom = [(s, tuple(r)) for s, r in zip(json.loads(sym_json), json.loads(xyz_json))]
c.basis = basis
c.unit = 'Bohr'
c.verbose = 0
c.build()
kpts = c.make_kpts(json.loads(nk_json))
mesh = json.loads(mesh_json)
mydf = df.FFTDF(c, kpts)
mydf.mesh = mesh
mydf.max_memory = 0
nuc = np.asarray(mydf.get_nuc(kpts), dtype=np.complex128)
out = {
    'version': __import__('pyscf').__version__,
    'nkpts': len(kpts), 'nao': int(c.nao_nr()),
    'env': bits(c._env),
    'nuc_re': bits(nuc.real), 'nuc_im': bits(nuc.imag),
}
print(json.dumps(out))
"#;

#[test]
#[ignore = "T1: needs PYSCF_ORACLE_VENV + the vendored upstream PySCF"]
fn get_nuc_bit_identical_multi_block() {
    let Some(py) = oracle_python() else {
        eprintln!("SKIP: {GATE} is not set");
        return;
    };
    let cell = he_all_electron();
    let kpts = make_kpts_default(&cell, [1, 1, 1]).expect("gamma k-point");
    let nao = cell.mol.nao_nr;
    let ngrids = MESH_MULTI[0] * MESH_MULTI[1] * MESH_MULTI[2];
    let blocks = aor_loop_blocks(ngrids, nao, kpts.len(), 2000.0);
    assert!(
        blocks.len() >= 2,
        "mesh {MESH_MULTI:?} must split into >= 2 aoR blocks at max_memory=2000, got {blocks:?}"
    );

    let args = cell_args(
        &cell,
        &[
            "sto-3g".to_string(),
            serde_json::to_string(&[1, 1, 1]).expect("json"),
            serde_json::to_string(&MESH_MULTI.to_vec()).expect("json"),
        ],
    );
    let want = run_python(&py, ORACLE_MULTI_PY, &args);
    assert_eq!(
        want["version"].as_str(),
        Some("2.12.1"),
        "vendored PySCF only"
    );

    // Same `_env` pin as the baseline test (F6 removes it).
    let want_env = pull(&want, "env");
    let mut cell = cell;
    cell.mol._env = want_env.iter().map(|&b| f64::from_bits(b)).collect();

    let df = Fftdf::with_mesh(cell, &kpts, MESH_MULTI).expect("FFTDF");
    let got = df.get_nuc(&kpts).expect("get_nuc");
    let nuc_re: Vec<f64> = got.iter().flat_map(|t| t.re.iter().copied()).collect();
    let nuc_im: Vec<f64> = got.iter().flat_map(|t| t.im.iter().copied()).collect();
    assert_bits(
        "get_nuc multi-block real part",
        &nuc_re,
        &pull(&want, "nuc_re"),
    );
    assert_bits(
        "get_nuc multi-block imaginary part",
        &nuc_im,
        &pull(&want, "nuc_im"),
    );
}

/// F2: `get_nuc` on He/cc-pVTZ (has a d shell), spherical and Cartesian.
///
/// Mesh `[15, 15, 15]`, 2x2x2 k-points: `15` is neither Bluestein-planned
/// (first Bluestein length is 89) nor in `_EXCLUDE`, so the FFT stays on the
/// pocketfft route (F3/F4 untouched), and the grid is a single `aoR_loop`
/// block (F1 untouched). `_env` stays pinned (F6 removes it).
const MESH_D: [usize; 3] = [15, 15, 15];

const ORACLE_D_PY: &str = r#"
import os
os.environ['OMP_NUM_THREADS'] = '1'
import json, sys
import numpy as np
from pyscf.pbc import gto, df
from pyscf.pbc.dft import numint

def bits(x):
    x = np.ascontiguousarray(np.asarray(x, dtype=np.float64)).ravel()
    return [int(v) for v in x.view(np.uint64)]

a_json, xyz_json, sym_json, basis, cart_json, nk_json, mesh_json = sys.argv[1:8]
c = gto.Cell()
c.a = json.loads(a_json)
c.atom = [(s, tuple(r)) for s, r in zip(json.loads(sym_json), json.loads(xyz_json))]
c.basis = basis
c.cart = json.loads(cart_json)
c.unit = 'Bohr'
c.verbose = 0
c.build()
kpts = c.make_kpts(json.loads(nk_json))
mesh = json.loads(mesh_json)
mydf = df.FFTDF(c, kpts)
mydf.mesh = mesh
nuc = np.asarray(mydf.get_nuc(kpts), dtype=np.complex128)
ao = numint.eval_ao_kpts(c, mydf.grids.coords, kpts)
out = {
    'version': __import__('pyscf').__version__,
    'nkpts': len(kpts), 'nao': int(c.nao_nr()),
    'env': bits(c._env),
    'nuc_re': bits(nuc.real), 'nuc_im': bits(nuc.imag),
    'ao_re': bits([np.real(x).T for x in ao]),
    'ao_im': bits([np.imag(x).T for x in ao]),
}
print(json.dumps(out))
"#;

fn he_ccpvtz(cart: bool) -> Cell {
    let h = 2.834589;
    Cell::build(CellBuildArgs {
        mole: MoleBuildArgs {
            atom: AtomInput::Tuples(vec![("He".into(), [0.0, 0.0, 0.0])]),
            basis: BasisInput::Name("cc-pvtz".into()),
            cart,
            unit: Unit::Bohr,
            ..Default::default()
        },
        a: ALattice::Matrix([[0.0, h, h], [h, 0.0, h], [h, h, 0.0]]),
        ..Default::default()
    })
    .expect("He cc-pVTZ cell builds")
}

fn run_d_shell_case(cart: bool) {
    let Some(py) = oracle_python() else {
        eprintln!("SKIP: {GATE} is not set");
        return;
    };
    let cell = he_ccpvtz(cart);
    let kpts = make_kpts_default(&cell, [2, 2, 2]).expect("2x2x2 k-mesh");
    let args = cell_args(
        &cell,
        &[
            "cc-pvtz".to_string(),
            serde_json::to_string(&cart).expect("json"),
            serde_json::to_string(&[2, 2, 2]).expect("json"),
            serde_json::to_string(&MESH_D.to_vec()).expect("json"),
        ],
    );
    let want = run_python(&py, ORACLE_D_PY, &args);
    assert_eq!(
        want["version"].as_str(),
        Some("2.12.1"),
        "vendored PySCF only"
    );
    assert_eq!(
        want["nao"].as_u64(),
        Some(cell.mol.nao_nr as u64),
        "Rust and upstream disagree on nao (cart={cart})"
    );

    // Same `_env` pin as the baseline test (F6 removes it).
    let want_env = pull(&want, "env");
    let mut cell = cell;
    cell.mol._env = want_env.iter().map(|&b| f64::from_bits(b)).collect();

    let df = Fftdf::with_mesh(cell.clone(), &kpts, MESH_D).expect("FFTDF");

    // Stage assertion on the AO table (d shells through `c2s_ket_sph1`, or
    // straight Cartesians when `cart`).
    let ao = eval_ao_kpts_upstream(&cell, &df.grids.coords, &kpts)
        .expect("eval")
        .unwrap_or_else(|| panic!("He/cc-pVTZ (cart={cart}) must be ported"));
    let ao_re: Vec<f64> = ao.kaos.iter().flat_map(|t| t.re.iter().copied()).collect();
    let ao_im: Vec<f64> = ao.kaos.iter().flat_map(|t| t.im.iter().copied()).collect();
    assert_bits(
        &format!("AO real part (cart={cart})"),
        &ao_re,
        &pull(&want, "ao_re"),
    );
    assert_bits(
        &format!("AO imaginary part (cart={cart})"),
        &ao_im,
        &pull(&want, "ao_im"),
    );

    let got = df.get_nuc(&kpts).expect("get_nuc");
    let nuc_re: Vec<f64> = got.iter().flat_map(|t| t.re.iter().copied()).collect();
    let nuc_im: Vec<f64> = got.iter().flat_map(|t| t.im.iter().copied()).collect();
    assert_bits(
        &format!("get_nuc d-shell real part (cart={cart})"),
        &nuc_re,
        &pull(&want, "nuc_re"),
    );
    assert_bits(
        &format!("get_nuc d-shell imaginary part (cart={cart})"),
        &nuc_im,
        &pull(&want, "nuc_im"),
    );
}

#[test]
#[ignore = "T1: needs PYSCF_ORACLE_VENV + the vendored upstream PySCF"]
fn get_nuc_bit_identical_d_shells() {
    run_d_shell_case(false);
}

#[test]
#[ignore = "T1: needs PYSCF_ORACLE_VENV + the vendored upstream PySCF"]
fn get_nuc_bit_identical_d_shells_cart() {
    run_d_shell_case(true);
}

/// F3: `ifft_upstream` on Bluestein-planned meshes, bit-identical.
///
/// Random input is generated in Python and shipped as bits; the oracle
/// evaluates upstream's own `tools.ifft` (i.e. `scipy.fft.ifftn`, since none
/// of these meshes is all-`_EXCLUDE`) and ships the result bits.
/// `[89, 12, 15]` and `[101, 9, 10]` each carry a Bluestein axis (first
/// Bluestein length is 89); `[5, 6, 7]` is the `cfftp`-only control.
const ORACLE_FFT_PY: &str = r#"
import os
os.environ['OMP_NUM_THREADS'] = '1'
import json, sys
import numpy as np
from pyscf.pbc import tools

def bits(x):
    x = np.ascontiguousarray(np.asarray(x, dtype=np.float64)).ravel()
    return [int(v) for v in x.view(np.uint64)]

mesh = json.loads(sys.argv[1])
ngrids = int(np.prod(mesh))
rng = np.random.default_rng(20260918)
g = rng.standard_normal(ngrids) + 1j * rng.standard_normal(ngrids)
f = tools.ifft(g, mesh)
out = {
    'g_re': bits(g.real), 'g_im': bits(g.imag),
    'f_re': bits(f.real), 'f_im': bits(f.imag),
}
print(json.dumps(out))
"#;

#[test]
#[ignore = "T1: needs PYSCF_ORACLE_VENV + the vendored upstream PySCF"]
fn ifft_upstream_bit_identical_on_bluestein_meshes() {
    let Some(py) = oracle_python() else {
        eprintln!("SKIP: {GATE} is not set");
        return;
    };
    for mesh in [[89, 12, 15], [101, 9, 10], [5, 6, 7]] {
        let want = run_python(
            &py,
            ORACLE_FFT_PY,
            &[serde_json::to_string(&mesh.to_vec()).expect("json")],
        );
        let f =
            |key: &str| -> Vec<f64> { pull(&want, key).into_iter().map(f64::from_bits).collect() };
        let g = CTensor::from_planes(f("g_re"), f("g_im"));
        let got = pyscf_pbc_tools::ifft_upstream(&g, mesh).expect("ifft");
        assert_bits(&format!("ifft re {mesh:?}"), &got.re, &pull(&want, "f_re"));
        assert_bits(&format!("ifft im {mesh:?}"), &got.im, &pull(&want, "f_im"));
    }
}

/// F4: `ifft_upstream` on all-`_EXCLUDE` meshes (upstream's `_ifftn_blas`
/// GEMM route), bit-identical. Same harness as the Bluestein test — the
/// oracle's `tools.ifft` picks `_ifftn_blas` itself when every axis is in
/// `_EXCLUDE`.
#[test]
#[ignore = "T1: needs PYSCF_ORACLE_VENV + the vendored upstream PySCF"]
fn ifft_upstream_bit_identical_on_blas_meshes() {
    let Some(py) = oracle_python() else {
        eprintln!("SKIP: {GATE} is not set");
        return;
    };
    for mesh in [[17, 17, 17], [47, 47, 47], [17, 19, 23]] {
        let want = run_python(
            &py,
            ORACLE_FFT_PY,
            &[serde_json::to_string(&mesh.to_vec()).expect("json")],
        );
        let f =
            |key: &str| -> Vec<f64> { pull(&want, key).into_iter().map(f64::from_bits).collect() };
        let g = CTensor::from_planes(f("g_re"), f("g_im"));
        let got = pyscf_pbc_tools::ifft_upstream(&g, mesh).expect("ifft");
        assert_bits(&format!("ifft re {mesh:?}"), &got.re, &pull(&want, "f_re"));
        assert_bits(&format!("ifft im {mesh:?}"), &got.im, &pull(&want, "f_im"));
    }
}

/// F3: `get_nuc` on He/STO-3G with a Bluestein axis, bit-identical.
///
/// Mesh `[89, 16, 16]`, 2x2x2 k-points: `89` is Bluestein-planned but `16`
/// is not in `_EXCLUDE`, so upstream takes the `scipy.fft.ifftn` route (F4
/// untouched). `_env` stays pinned (F6 removes it).
const MESH_BLUE: [usize; 3] = [89, 16, 16];

const ORACLE_BLUE_PY: &str = r#"
import os
os.environ['OMP_NUM_THREADS'] = '1'
import json, sys
import numpy as np
from pyscf.pbc import gto, df

def bits(x):
    x = np.ascontiguousarray(np.asarray(x, dtype=np.float64)).ravel()
    return [int(v) for v in x.view(np.uint64)]

a_json, xyz_json, sym_json, basis, nk_json, mesh_json = sys.argv[1:7]
c = gto.Cell()
c.a = json.loads(a_json)
c.atom = [(s, tuple(r)) for s, r in zip(json.loads(sym_json), json.loads(xyz_json))]
c.basis = basis
c.unit = 'Bohr'
c.verbose = 0
c.build()
kpts = c.make_kpts(json.loads(nk_json))
mesh = json.loads(mesh_json)
mydf = df.FFTDF(c, kpts)
mydf.mesh = mesh
nuc = np.asarray(mydf.get_nuc(kpts), dtype=np.complex128)
out = {
    'version': __import__('pyscf').__version__,
    'env': bits(c._env),
    'nuc_re': bits(nuc.real), 'nuc_im': bits(nuc.imag),
}
print(json.dumps(out))
"#;

#[test]
#[ignore = "T1: needs PYSCF_ORACLE_VENV + the vendored upstream PySCF"]
fn get_nuc_bit_identical_bluestein_mesh() {
    let Some(py) = oracle_python() else {
        eprintln!("SKIP: {GATE} is not set");
        return;
    };
    let cell = he_all_electron();
    let kpts = make_kpts_default(&cell, [2, 2, 2]).expect("2x2x2 k-mesh");
    let args = cell_args(
        &cell,
        &[
            "sto-3g".to_string(),
            serde_json::to_string(&[2, 2, 2]).expect("json"),
            serde_json::to_string(&MESH_BLUE.to_vec()).expect("json"),
        ],
    );
    let want = run_python(&py, ORACLE_BLUE_PY, &args);
    assert_eq!(
        want["version"].as_str(),
        Some("2.12.1"),
        "vendored PySCF only"
    );

    // Same `_env` pin as the baseline test (F6 removes it).
    let want_env = pull(&want, "env");
    let mut cell = cell;
    cell.mol._env = want_env.iter().map(|&b| f64::from_bits(b)).collect();

    let df = Fftdf::with_mesh(cell, &kpts, MESH_BLUE).expect("FFTDF");
    let got = df.get_nuc(&kpts).expect("get_nuc");
    let nuc_re: Vec<f64> = got.iter().flat_map(|t| t.re.iter().copied()).collect();
    let nuc_im: Vec<f64> = got.iter().flat_map(|t| t.im.iter().copied()).collect();
    assert_bits(
        "get_nuc Bluestein-mesh real part",
        &nuc_re,
        &pull(&want, "nuc_re"),
    );
    assert_bits(
        "get_nuc Bluestein-mesh imaginary part",
        &nuc_im,
        &pull(&want, "nuc_im"),
    );
}

/// F4: `get_nuc` on He/STO-3G at mesh `[17, 17, 17]` (all-`_EXCLUDE`),
/// bit-identical. `_env` stays pinned (F6 removes it).
const MESH_BLAS: [usize; 3] = [17, 17, 17];

#[test]
#[ignore = "T1: needs PYSCF_ORACLE_VENV + the vendored upstream PySCF"]
fn get_nuc_bit_identical_blas_mesh() {
    let Some(py) = oracle_python() else {
        eprintln!("SKIP: {GATE} is not set");
        return;
    };
    let cell = he_all_electron();
    let kpts = make_kpts_default(&cell, [2, 2, 2]).expect("2x2x2 k-mesh");
    let args = cell_args(
        &cell,
        &[
            "sto-3g".to_string(),
            serde_json::to_string(&[2, 2, 2]).expect("json"),
            serde_json::to_string(&MESH_BLAS.to_vec()).expect("json"),
        ],
    );
    let want = run_python(&py, ORACLE_BLUE_PY, &args);
    assert_eq!(
        want["version"].as_str(),
        Some("2.12.1"),
        "vendored PySCF only"
    );

    // Same `_env` pin as the baseline test (F6 removes it).
    let want_env = pull(&want, "env");
    let mut cell = cell;
    cell.mol._env = want_env.iter().map(|&b| f64::from_bits(b)).collect();

    let df = Fftdf::with_mesh(cell, &kpts, MESH_BLAS).expect("FFTDF");
    let got = df.get_nuc(&kpts).expect("get_nuc");
    let nuc_re: Vec<f64> = got.iter().flat_map(|t| t.re.iter().copied()).collect();
    let nuc_im: Vec<f64> = got.iter().flat_map(|t| t.im.iter().copied()).collect();
    assert_bits(
        "get_nuc blas-mesh real part",
        &nuc_re,
        &pull(&want, "nuc_re"),
    );
    assert_bits(
        "get_nuc blas-mesh imaginary part",
        &nuc_im,
        &pull(&want, "nuc_im"),
    );
}

/// F5: `SI` and `rhoG` stage gates plus `get_nuc` on multi-atom cells.
///
/// The oracle evaluates upstream's own `cell.get_SI(mesh=mesh)` (the separable
/// branch — `fft.py:63` passes no `Gv`) and `numpy.dot(charge, SI)`, shipping
/// every stage as bits. Cases: diamond all-electron (`sto-3g`, 2 C, one off
/// origin, mesh 21) and He2 with one atom off origin (mesh 11).
const ORACLE_SI_PY: &str = r#"
import os
os.environ['OMP_NUM_THREADS'] = '1'
import json, sys
import numpy as np
from pyscf.pbc import gto, df

def bits(x):
    x = np.ascontiguousarray(np.asarray(x, dtype=np.float64)).ravel()
    return [int(v) for v in x.view(np.uint64)]

a_json, xyz_json, sym_json, basis, nk_json, mesh_json = sys.argv[1:7]
c = gto.Cell()
c.a = json.loads(a_json)
c.atom = [(s, tuple(r)) for s, r in zip(json.loads(sym_json), json.loads(xyz_json))]
c.basis = basis
c.unit = 'Bohr'
c.verbose = 0
c.build()
kpts = c.make_kpts(json.loads(nk_json))
mesh = json.loads(mesh_json)
mydf = df.FFTDF(c, kpts)
mydf.mesh = mesh
si = c.get_SI(mesh=mesh)
rhog = np.dot(-c.atom_charges(), si)
nuc = np.asarray(mydf.get_nuc(kpts), dtype=np.complex128)
out = {
    'version': __import__('pyscf').__version__,
    'natm': int(c.natm), 'nao': int(c.nao_nr()),
    'env': bits(c._env),
    'si_re': bits(si.real), 'si_im': bits(si.imag),
    'rhog_re': bits(rhog.real), 'rhog_im': bits(rhog.imag),
    'nuc_re': bits(nuc.real), 'nuc_im': bits(nuc.imag),
}
print(json.dumps(out))
"#;

fn run_multi_atom_case(
    make_cell: fn() -> Cell,
    basis: &str,
    nk: [usize; 3],
    mesh: [usize; 3],
    what: &str,
) {
    let Some(py) = oracle_python() else {
        eprintln!("SKIP: {GATE} is not set");
        return;
    };
    let cell = make_cell();
    let kpts = make_kpts_default(&cell, nk).expect("k-mesh");
    let args = cell_args(
        &cell,
        &[
            basis.to_string(),
            serde_json::to_string(&nk).expect("json"),
            serde_json::to_string(&mesh.to_vec()).expect("json"),
        ],
    );
    let want = run_python(&py, ORACLE_SI_PY, &args);
    assert_eq!(
        want["version"].as_str(),
        Some("2.12.1"),
        "vendored PySCF only"
    );
    assert_eq!(
        want["natm"].as_u64(),
        Some(cell.mol.natm as u64),
        "{what}: Rust and upstream disagree on natm"
    );

    // Same `_env` pin as the baseline test (F6 removes it).
    let want_env = pull(&want, "env");
    let mut cell = cell;
    cell.mol._env = want_env.iter().map(|&b| f64::from_bits(b)).collect();

    // Stage 1 — SI (separable branch).
    let si = get_si(&cell, None, Some(mesh), None).expect("SI");
    let si_re: Vec<f64> = si.re.clone();
    let si_im: Vec<f64> = si.im.clone();
    assert_bits(
        &format!("{what} SI real part"),
        &si_re,
        &pull(&want, "si_re"),
    );
    assert_bits(
        &format!("{what} SI imaginary part"),
        &si_im,
        &pull(&want, "si_im"),
    );

    // Stage 2 — rhoG = charge . SI (plain accumulation, like `get_nuc`).
    let charges = cell.atom_charges();
    let ngrids = mesh[0] * mesh[1] * mesh[2];
    let mut rho_re = vec![0.0_f64; ngrids];
    let mut rho_im = vec![0.0_f64; ngrids];
    for ia in 0..cell.mol.natm {
        let z = -(charges[ia] as f64);
        let base = ia * ngrids;
        for g in 0..ngrids {
            rho_re[g] += z * si.re[base + g];
            rho_im[g] += z * si.im[base + g];
        }
    }
    assert_bits(
        &format!("{what} rhoG real part"),
        &rho_re,
        &pull(&want, "rhog_re"),
    );
    assert_bits(
        &format!("{what} rhoG imaginary part"),
        &rho_im,
        &pull(&want, "rhog_im"),
    );

    // The result.
    let df = Fftdf::with_mesh(cell, &kpts, mesh).expect("FFTDF");
    let got = df.get_nuc(&kpts).expect("get_nuc");
    let nuc_re: Vec<f64> = got.iter().flat_map(|t| t.re.iter().copied()).collect();
    let nuc_im: Vec<f64> = got.iter().flat_map(|t| t.im.iter().copied()).collect();
    assert_bits(
        &format!("{what} get_nuc real part"),
        &nuc_re,
        &pull(&want, "nuc_re"),
    );
    assert_bits(
        &format!("{what} get_nuc imaginary part"),
        &nuc_im,
        &pull(&want, "nuc_im"),
    );
}

#[test]
#[ignore = "T1: needs PYSCF_ORACLE_VENV + the vendored upstream PySCF"]
fn get_nuc_bit_identical_diamond_all_electron() {
    run_multi_atom_case(
        diamond_all_electron,
        "sto-3g",
        [2, 2, 2],
        [21, 21, 21],
        "diamond-AE",
    );
}

#[test]
#[ignore = "T1: needs PYSCF_ORACLE_VENV + the vendored upstream PySCF"]
fn get_nuc_bit_identical_he2_off_origin() {
    run_multi_atom_case(
        he2_off_origin,
        "sto-3g",
        [2, 2, 2],
        [11, 11, 11],
        "He2-off-origin",
    );
}
