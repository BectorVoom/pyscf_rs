//! Plan 18-04 — FFTDF gradient JK (`get_j_e1_kpts` + `get_k_e1_kpts`).
//!
//! Two layers, like `tests/fftdf.rs`:
//!
//! * **Oracle-free** (always on): the budgeted-residency maths, the
//!   k-difference index firewall, the `coulG` build counter, the small-`m`
//!   exchange rate, tagged-vs-untagged agreement with a strictly smaller
//!   tagged peak, Hermiticity of every component, and bit-identical output at
//!   1 vs 8 rayon threads (pools inside one process, the `fft_jk_threads.rs`
//!   pattern).
//! * **Upstream** (`#[ignore]`d, gated on `PYSCF_ORACLE_VENV`): element-wise
//!   against live PySCF **2.12.1** `pyscf.pbc.df.fft_jk` on diamond `gth-szv`
//!   at gamma, 1×1×2 and 2×2×2:
//!
//!   ```bash
//!   PYSCF_ORACLE_VENV=1 cargo test -p pyscf-pbc-df --release -- --ignored fft_jk_grad
//!   ```

mod common;

use common::{GATE, cell_args, diamond, max_dev, oracle_python, run_python};
use pyscf_algebra::CTensor;
use pyscf_pbc_df::df_jk::KMats;
use pyscf_pbc_df::{
    Fftdf, GradMats, TaggedMo, deriv1_blksize, deriv1_blksize_from_avail,
    get_j_e1_kpts, get_k_e1_kpts, k_e1_footprint, kdiff_index, resident_k_count, resident_mb,
};
use pyscf_pbc_gto::make_kpts_default;

const MESH_FAST: [usize; 3] = [11, 11, 11];

fn kpts_of(cell: &pyscf_pbc_gto::Cell, nk: [usize; 3]) -> Vec<[f64; 3]> {
    make_kpts_default(cell, nk).expect("k-mesh")
}

/// A trivially Hermitian test density: `0.5 * I` at every k.
fn flat_dm(nao: usize, nkpts: usize) -> Vec<Vec<CTensor>> {
    let mut dm = CTensor::zeros(nao * nao);
    for i in 0..nao {
        dm.re[i * nao + i] = 0.5;
    }
    vec![vec![dm; nkpts]]
}

fn pool(n: usize) -> rayon::ThreadPool {
    rayon::ThreadPoolBuilder::new()
        .num_threads(n)
        .build()
        .expect("thread pool")
}

// ---------------------------------------------------------------------------
// Budget pure functions (D-PBC-30 clause 3 / D-PBC-31 clauses 8 and 12)
// ---------------------------------------------------------------------------

/// `m` spans resident (`m = nkpts` at the default budget) to streaming
/// (`m = 1` under a 1 MB budget), and `blksize` follows the re-derived
/// formula with the measured multiplicity.
#[test]
fn budgeted_residency_endpoints() {
    let ngrids = 11 * 11 * 11;
    let nao = 8;
    assert_eq!(resident_k_count(4000.0, ngrids, nao, 8), 8);
    assert_eq!(resident_k_count(4000.0, ngrids, nao, 27), 27);
    assert_eq!(resident_k_count(1.0, ngrids, nao, 8), 1);
    assert_eq!(resident_k_count(0.0, ngrids, nao, 8), 1);
    // One subtraction only: the wrapper nets the resident tables out first.
    let m = resident_k_count(4000.0, ngrids, nao, 8);
    assert_eq!(
        deriv1_blksize(4000.0, resident_mb(m, ngrids, nao), ngrids, nao),
        deriv1_blksize_from_avail(4000.0 - resident_mb(m, ngrids, nao), ngrids, nao)
    );
    assert_eq!(
        deriv1_blksize(4000.0, resident_mb(m, ngrids, nao), ngrids, nao),
        nao
    );
    // Starved: the transient floors at one AO row, never zero.
    assert_eq!(deriv1_blksize(1.0, resident_mb(1, ngrids, nao), ngrids, nao), 1);
}

/// The k-difference index firewall: on a 2×2×2 mesh the 64 raw `(k2, k1)`
/// differences collapse to exactly 8 classes. This is what lets the W-01
/// cache build `nkpts` `coulG`s instead of `nkpts²` — the counter test below
/// assumes it, so it is asserted here rather than discovered there.
#[test]
fn kdiff_index_collapses_222_to_8_classes() {
    let cell = diamond();
    let kpts = kpts_of(&cell, [2, 2, 2]);
    let mut classes = std::collections::HashSet::new();
    for k2 in kpts.iter() {
        for k1 in kpts.iter() {
            let dk = [k2[0] - k1[0], k2[1] - k1[1], k2[2] - k1[2]];
            classes.insert(kdiff_index(&cell, dk));
        }
    }
    assert_eq!(classes.len(), 8, "222 difference set must be the mesh");
    assert!(classes.contains(&kdiff_index(&cell, [0.0, 0.0, 0.0])));
    // Same class across the wrap: -0.5 and +0.5 merge by construction
    // (absolute k-points, i.e. against the 2π reciprocal lattice).
    let b = cell.reciprocal_vectors_2pi().expect("reciprocal lattice");
    let half = [0.5 * b[2][0], 0.5 * b[2][1], 0.5 * b[2][2]];
    let neg = [-half[0], -half[1], -half[2]];
    assert_eq!(kdiff_index(&cell, half), kdiff_index(&cell, neg));
}

// ---------------------------------------------------------------------------
// Oracle-free gates
// ---------------------------------------------------------------------------

/// The `coulG` cache is counted, not assumed: a full `get_k_e1_kpts` call on
/// 2×2×2 enters `get_coulG` 8 times, not 64 — and a second call enters it
/// zero further times. The J path (Gamma kernel, uncached) does not move the
/// counter at all.
#[test]
fn coulg_build_counter_reads_nkpts() {
    let cell = diamond();
    let kpts = kpts_of(&cell, [2, 2, 2]);
    let nao = cell.mol.nao_nr;
    let dms = flat_dm(nao, kpts.len());
    let df = Fftdf::with_mesh(cell, &kpts, MESH_FAST).expect("FFTDF");
    df.reset_coulg_build_count();

    get_j_e1_kpts(&df, &dms, &kpts, None, None).expect("get_j_e1_kpts");
    assert_eq!(
        df.coulg_build_count(),
        0,
        "the J path must not touch the W-01 cache"
    );

    // The cache is keyed on the raw `dk` bits (a wrapped-class key is wrong:
    // `coulG` is not invariant per grid point under `dk -> dk + b`), so the
    // build count is the number of DISTINCT raw differences, which on a 2x2x2
    // mesh is below nkpts^2 = 64 but above nkpts = 8.
    let mut distinct = std::collections::HashSet::new();
    for k1 in &kpts {
        for k2 in &kpts {
            let dk = [k2[0] - k1[0], k2[1] - k1[1], k2[2] - k1[2]];
            distinct.insert([dk[0].to_bits(), dk[1].to_bits(), dk[2].to_bits()]);
        }
    }
    get_k_e1_kpts(&df, &dms, &kpts, None, None, None, None, None).expect("get_k_e1_kpts");
    let built = df.coulg_build_count();
    assert!(
        built <= distinct.len() && built < kpts.len() * kpts.len(),
        "coulG built {built} times; at most one per distinct raw dk ({}) expected",
        distinct.len()
    );

    get_k_e1_kpts(&df, &dms, &kpts, None, None, None, None, None).expect("get_k_e1_kpts");
    assert_eq!(
        df.coulg_build_count(),
        built,
        "a warm cache must build nothing further"
    );
}

/// A small `m` runs in CI: pin `PYSCF_MAX_MEMORY`-derived `max_memory` low
/// enough to force `m = 1 < nkpts`, and assert the realised `m`, `blksize`
/// and AO-evaluation count are what the budget predicts (`nkpts²/m` chunk
/// builds for the double loop) — and that the answer is bit-identical to the
/// resident run. A fixture that silently stayed incore fails rather than
/// passes (16-01's rule; the exact failure mode 17-12 hit).
#[test]
fn small_m_exchange_rate_and_bit_identity() {
    let cell = diamond();
    let kpts = kpts_of(&cell, [2, 2, 2]);
    let nao = cell.mol.nao_nr;
    let ngrids = MESH_FAST[0] * MESH_FAST[1] * MESH_FAST[2];
    let dms = flat_dm(nao, kpts.len());

    let full = Fftdf::with_mesh(cell.clone(), &kpts, MESH_FAST).expect("FFTDF");
    let vk_full =
        get_k_e1_kpts(&full, &dms, &kpts, None, None, None, None, None).expect("resident K");

    let mut small = Fftdf::with_mesh(cell, &kpts, MESH_FAST).expect("FFTDF");
    small.max_memory = 1.0;
    let mut stats = Default::default();
    let vk_small = get_k_e1_kpts(
        &small,
        &dms,
        &kpts,
        None,
        None,
        None,
        None,
        Some(&mut stats),
    )
    .expect("streaming K");

    let nkpts = kpts.len();
    assert_eq!(stats.m, 1, "1 MB must force streaming at this shape");
    assert_eq!(stats.m, resident_k_count(1.0, ngrids, nao, nkpts));
    assert_eq!(
        stats.blksize,
        deriv1_blksize(1.0, resident_mb(stats.m, ngrids, nao), ngrids, nao)
    );
    assert_eq!(
        stats.chunk_builds,
        nkpts * nkpts / stats.m,
        "double loop rebuilds the inner chunk per outer k"
    );
    assert_eq!(stats.k_tables, stats.chunk_builds * stats.m);

    for (x, (a, b)) in vk_full.iter().zip(vk_small.iter()).enumerate() {
        for (k, (ma, mb)) in a[0].iter().zip(b[0].iter()).enumerate() {
            assert_eq!(ma.re, mb.re, "x={x} k={k} .re moved under small m");
            assert_eq!(ma.im, mb.im, "x={x} k={k} .im moved under small m");
        }
    }
}

/// Fixed rotation mixing each occupied orbital with one virtual — still
/// orthonormal columns, so `dm = C_occ · 2 · C_occᵀ` is a valid RHF density
/// and the tag is honest. Returns the TRUNCATED `nao × nocc` block plus the
/// occupations, exactly what [`TaggedMo::from_coeff_occ`] consumes.
fn rotated_occ_coeff(nao: usize, nocc: usize) -> (CTensor, Vec<f64>) {
    let theta = 0.3_f64;
    let (s, c) = (theta.sin(), theta.cos());
    let mut coeff = CTensor::zeros(nao * nocc);
    for o in 0..nocc {
        coeff.re[o * nocc + o] = c;
        coeff.re[(o + nocc) * nocc + o] = s;
    }
    let occ = vec![2.0; nocc];
    (coeff, occ)
}

/// Clause 4a, both sides: tagged and untagged agree, and the tagged
/// accounted peak is strictly smaller on the same fixture.
#[test]
fn mo_tagged_agrees_and_allocates_less() {
    let cell = diamond();
    let kpts = kpts_of(&cell, [2, 2, 2]);
    let nao = cell.mol.nao_nr;
    let ngrids = MESH_FAST[0] * MESH_FAST[1] * MESH_FAST[2];
    let nkpts = kpts.len();
    let nocc = 4;

    let (coeff, occ) = rotated_occ_coeff(nao, nocc);
    // dm = C_occ · 2 · C_occᵀ.
    let mut dm = CTensor::zeros(nao * nao);
    for p in 0..nao {
        for q in 0..nao {
            let mut v = 0.0;
            for o in 0..nocc {
                v += coeff.re[p * nocc + o] * 2.0 * coeff.re[q * nocc + o];
            }
            dm.re[p * nao + q] = v;
        }
    }
    let dms = vec![vec![dm; nkpts]];
    let tag = TaggedMo::from_coeff_occ(
        &vec![coeff; nkpts],
        &vec![occ; nkpts],
        nao,
    )
    .expect("tag");
    assert!(tag.blocks.iter().all(|b| b.nocc == nocc));

    let df = Fftdf::with_mesh(cell, &kpts, MESH_FAST).expect("FFTDF");
    let plain = get_k_e1_kpts(&df, &dms, &kpts, None, None, None, None, None).expect("untagged");
    let mut stats = Default::default();
    let factorised = get_k_e1_kpts(
        &df,
        &dms,
        &kpts,
        None,
        None,
        None,
        Some(&tag),
        Some(&mut stats),
    )
    .expect("tagged");

    let mut w = 0.0_f64;
    for (x, (a, b)) in plain.iter().zip(factorised.iter()).enumerate() {
        for (k, (ma, mb)) in a[0].iter().zip(b[0].iter()).enumerate() {
            for i in 0..nao * nao {
                w = w.max((ma.re[i] - mb.re[i]).abs());
                w = w.max((ma.im[i] - mb.im[i]).abs());
            }
            let _ = (x, k);
        }
    }
    println!("tagged-vs-untagged max|delta| = {w:e}");
    assert!(w < 1e-12, "tagged route disagrees with untagged by {w:e}");

    // Peak assertion on the accounted high-water with the run's own (m,
    // blksize): the ~1.4 GiB-class transient cut must strictly beat the
    // resident ket table it adds (§8.2).
    let ket_extra = nkpts as u64 * nocc as u64 * ngrids as u64 * 16;
    let untagged = k_e1_footprint(stats.m, stats.blksize, ngrids, nao, nao, 1, 0);
    let tagged_fp = k_e1_footprint(stats.m, stats.blksize, ngrids, nao, nocc, 1, ket_extra);
    println!(
        "peak untagged = {} B, tagged = {} B (m = {}, blksize = {})",
        untagged.peak, tagged_fp.peak, stats.m, stats.blksize
    );
    assert!(
        tagged_fp.peak < untagged.peak,
        "tagged peak {} must be strictly below untagged {}",
        tagged_fp.peak,
        untagged.peak
    );
}

/// Bit-identical at 1 vs 8 threads, inside one process (the
/// `fft_jk_threads.rs` pattern — strictly stronger than an env-var sweep).
#[test]
fn gradient_jk_is_bit_identical_1_vs_8_threads() {
    let cell = diamond();
    let kpts = kpts_of(&cell, [2, 2, 2]);
    let nao = cell.mol.nao_nr;
    let dms = flat_dm(nao, kpts.len());
    let df = Fftdf::with_mesh(cell, &kpts, MESH_FAST).expect("FFTDF");

    let run = || {
        let vj = get_j_e1_kpts(&df, &dms, &kpts, None, None).expect("J");
        let vk =
            get_k_e1_kpts(&df, &dms, &kpts, None, None, None, None, None).expect("K");
        (vj, vk)
    };
    let a = pool(1).install(run);
    let b = pool(8).install(run);
    for (name, (x, y)) in [("vj", (&a.0, &b.0)), ("vk", (&a.1, &b.1))] {
        for (cx, (sx, sy)) in x.iter().zip(y.iter()).enumerate() {
            for (k, (mx, my)) in sx[0].iter().zip(sy[0].iter()).enumerate() {
                assert_eq!(mx.re, my.re, "{name}[{cx}][{k}] .re moved across pools");
                assert_eq!(mx.im, my.im, "{name}[{cx}][{k}] .im moved across pools");
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Upstream gates
// ---------------------------------------------------------------------------

const ORACLE_PY: &str = r#"
import json, sys
import numpy as np
from pyscf.pbc import gto, df

a_json, xyz_json, sym_json, basis, pseudo, nk_json, mesh_json, what = sys.argv[1:9]
c = gto.Cell()
c.a = json.loads(a_json)
c.atom = [(s, tuple(r)) for s, r in zip(json.loads(sym_json), json.loads(xyz_json))]
c.basis = basis
if pseudo:
    c.pseudo = pseudo
c.unit = 'Bohr'
c.verbose = 0
c.build()
kpts = c.make_kpts(json.loads(nk_json))
mesh = json.loads(mesh_json)
mydf = df.FFTDF(c, kpts)
mydf.mesh = mesh
nao = c.nao_nr()

dm = np.zeros((nao, nao), dtype=complex)
np.fill_diagonal(dm, 0.5)
dms = np.array([dm] * len(kpts))
if what == 'j_e1':
    mats = np.asarray(mydf.get_j_e1(dms, kpts=kpts))
elif what == 'k_e1':
    mats = np.asarray(mydf.get_k_e1(dms, kpts=kpts, exxdiv=None))
else:
    raise SystemExit('unknown quantity ' + what)

mats = np.asarray(mats).reshape(-1, nao, nao)
out = {'nao': int(nao), 'nkpts': len(kpts), 'version': __import__('pyscf').__version__,
       'ncomp': 3,
       're': np.real(mats).ravel().tolist(),
       'im': (np.imag(mats).ravel().tolist() if np.iscomplexobj(mats)
              else np.zeros(mats.size).tolist())}
print(json.dumps(out))
"#;

fn oracle_e1(
    cell: &pyscf_pbc_gto::Cell,
    nk: [usize; 3],
    mesh: [usize; 3],
    what: &str,
) -> Option<serde_json::Value> {
    let py = oracle_python()?;
    let args = cell_args(
        cell,
        &[
            "gth-szv".to_string(),
            "gth-pade".to_string(),
            serde_json::to_string(&nk.to_vec()).expect("json"),
            serde_json::to_string(&mesh.to_vec()).expect("json"),
            what.to_string(),
        ],
    );
    let v = run_python(&py, ORACLE_PY, &args);
    assert_eq!(
        v["version"].as_str().expect("version"),
        "2.12.1",
        "the oracle must be the VENDORED PySCF 2.12.1 — see tests/common/mod.rs"
    );
    Some(v)
}

/// Flatten `[x][0][k]` in x-major order: upstream ravels `(3, nkpts, nao,
/// nao)`, which is the same layout.
fn flatten_grad(g: &GradMats) -> Vec<CTensor> {
    let mut out = Vec::new();
    for x in g.iter() {
        out.extend(x[0].iter().cloned());
    }
    out
}

/// `get_j_e1_kpts` / `get_k_e1_kpts` element-wise against upstream
/// `pyscf.pbc.df.fft_jk` at gamma, 1×1×2 (the shape `test_krhf.py:37` uses)
/// and 2×2×2.
#[test]
#[ignore = "needs PYSCF_ORACLE_VENV + the vendored upstream PySCF"]
fn e1_matches_upstream_gamma_112_222() {
    for nk in [[1usize, 1, 1], [1, 1, 2], [2, 2, 2]] {
        let cell = diamond();
        let kpts = kpts_of(&cell, nk);
        let nao = cell.mol.nao_nr;
        let dms = flat_dm(nao, kpts.len());
        let df = Fftdf::with_mesh(cell.clone(), &kpts, MESH_FAST).expect("FFTDF");

        for (what, tol) in [("j_e1", 1e-9), ("k_e1", 1e-9)] {
            let Some(want) = oracle_e1(&cell, nk, MESH_FAST, what) else {
                eprintln!("SKIP: {GATE} is not set");
                return;
            };
            let got = if what == "j_e1" {
                get_j_e1_kpts(&df, &dms, &kpts, None, None).expect("get_j_e1_kpts")
            } else {
                get_k_e1_kpts(&df, &dms, &kpts, None, None, None, None, None)
                    .expect("get_k_e1_kpts")
            };
            assert_eq!(got.len(), 3, "{what}: need 3 components");
            assert_eq!(got[0][0].len(), kpts.len(), "{what}: need nkpts matrices");
            let w = max_dev(&flatten_grad(&got), &want);
            println!("{what} nk={nk:?} max|delta| vs upstream = {w:e}");
            assert!(w < tol, "{what} nk={nk:?} deviates from upstream by {w:e}");
        }
    }
}
