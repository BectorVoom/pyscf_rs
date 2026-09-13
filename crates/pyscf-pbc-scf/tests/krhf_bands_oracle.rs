//! `Krhf::get_bands` against live upstream PySCF **2.12.1** — the band-energy
//! gate that `.planning/STATE.md` records as "written but unrecorded".
//!
//! # What this adds over the tests already here
//!
//! `krhf_bands.rs` / `kuhf_bands.rs` / `kuks_bands.rs` are ORACLE-FREE: they
//! assert internal invariants (bands on the SCF mesh reproduce the SCF
//! eigenvalues; `E(k) == E(-k)`). Those hold just as well for two
//! implementations that agree with each other and both disagree with upstream.
//! `pyscf-pbc-df/tests/band_kpoints.rs` DOES gate against upstream, but it
//! gates `GDF/MDF::get_jk(..., kpts_band=...)` on a MODEL density — deliberately
//! not a converged one, and not the eigenvalues.
//!
//! This file closes the remaining link: a converged KRHF, then the band
//! EIGENVALUES themselves, compared to upstream's `mf.get_bands(kpts_band)`.
//!
//! # The cheapest possible periodic fixture
//!
//! He on an fcc lattice in `sto-3g` is **one AO per k-point**, all-electron (no
//! pseudopotential, so `get_nuc` rather than the `ft_ao` `get_pp` expansion
//! whose plane-wave residual sets the few-1e-12 floor on the `gth-pade` gates —
//! see `kscf.rs::krhf_he_all_electron_matches_upstream`, which reaches 1e-12 on
//! exactly this cell). 2x2x2 sampling, mesh `[15,15,15]`. Seconds, not minutes.
//!
//! # Two preconditions, asserted before the band numbers are looked at
//!
//! A band comparison is vacuous unless the two sides are the same cell in the
//! same converged state. `e_nuc` proves the geometry; `e_tot` proves the
//! density. Both are checked FIRST, so a failure reports which one moved
//! rather than blaming the band code.
//!
//! ```bash
//! PYSCF_ORACLE_VENV=1 cargo test -p pyscf-pbc-scf --release \
//!     --test krhf_bands_oracle -- --ignored --nocapture
//! ```

mod common;

use common::{GATE, cell_args, he_all_electron, oracle_python, run_python};
use pyscf_pbc_df::Fftdf;
use pyscf_pbc_gto::make_kpts_default;
use pyscf_pbc_scf::{KScfConfig, Krhf};

/// The mesh at which `kscf.rs` reaches 1e-12 on this same all-electron cell.
const MESH: [usize; 3] = [15, 15, 15];
const NK: [usize; 3] = [2, 2, 2];

fn tight() -> KScfConfig {
    KScfConfig {
        conv_tol: 1e-12,
        conv_tol_grad: Some(1e-8),
        max_cycle: 60,
        ..KScfConfig::default()
    }
}

const BAND_PY: &str = r#"
import json, sys
import numpy as np
from pyscf.pbc import gto, scf

a_json, xyz_json, sym_json, basis, nk_json, mesh_json, kband_json = sys.argv[1:8]
c = gto.Cell()
c.a = json.loads(a_json)
c.atom = [(s, tuple(r)) for s, r in zip(json.loads(sym_json), json.loads(xyz_json))]
c.basis = basis
c.unit = 'Bohr'
c.verbose = 0
c.build()
kpts = c.make_kpts(json.loads(nk_json))
mf = scf.KRHF(c, kpts)
mf.with_df.mesh = json.loads(mesh_json)
mf.conv_tol = 1e-12
mf.conv_tol_grad = 1e-8
mf.max_cycle = 60
e = mf.kernel()

# The band k-points arrive as ABSOLUTE cartesian numbers computed on the Rust
# side, so the scaled->absolute conversion is not transcribed twice.
kband = np.asarray(json.loads(kband_json), dtype=float)
e_band, _ = mf.get_bands(kband)

print(json.dumps({
    'version': __import__('pyscf').__version__,
    'e_tot': float(e),
    'e_nuc': float(c.energy_nuc()),
    'converged': bool(mf.converged),
    'nao': int(c.nao_nr()),
    'e_band': [np.asarray(x).ravel().tolist() for x in e_band],
    'e_mo': [np.asarray(x).ravel().tolist() for x in mf.mo_energy],
}))
"#;

/// Largest deviation, and how many of the compared values are BITWISE
/// identical, between a stack of Rust eigenvalue blocks and upstream's.
fn compare(got: &[Vec<f64>], want: &serde_json::Value, label: &str) -> f64 {
    let want_blocks: Vec<Vec<f64>> = want
        .as_array()
        .unwrap_or_else(|| panic!("{label}: upstream payload is not an array"))
        .iter()
        .map(|b| {
            b.as_array()
                .expect("block")
                .iter()
                .map(|v| v.as_f64().expect("f64"))
                .collect()
        })
        .collect();

    assert_eq!(
        got.len(),
        want_blocks.len(),
        "{label}: block count {} != upstream {}",
        got.len(),
        want_blocks.len()
    );

    let mut worst = 0.0_f64;
    let mut total = 0usize;
    let mut bitwise = 0usize;
    for (b, (g_blk, w_blk)) in got.iter().zip(want_blocks.iter()).enumerate() {
        assert_eq!(
            g_blk.len(),
            w_blk.len(),
            "{label}: block {b} orbital count {} != upstream {}",
            g_blk.len(),
            w_blk.len()
        );
        for (g, w) in g_blk.iter().zip(w_blk.iter()) {
            total += 1;
            if g.to_bits() == w.to_bits() {
                bitwise += 1;
            }
            worst = worst.max((g - w).abs());
        }
    }
    println!("{label}: {bitwise}/{total} bitwise identical, worst |delta| = {worst:e}");
    worst
}

/// **THE BAND GATE.** A converged KRHF, then `get_bands` at two genuine
/// off-mesh band k-points, against upstream's `get_bands` on the same density.
#[test]
#[ignore = "needs PYSCF_ORACLE_VENV + the vendored upstream PySCF"]
fn krhf_get_bands_matches_upstream() {
    let Some(py) = oracle_python() else {
        eprintln!("SKIP: {GATE} is not set");
        return;
    };
    let cell = he_all_electron();

    // Genuinely off the 2x2x2 sampling mesh. Converted to absolute here and
    // handed to the oracle as literal numbers.
    let kband = cell
        .get_abs_kpts(&[[0.15, -0.07, 0.03], [-0.05, 0.11, 0.02]])
        .expect("scaled -> absolute k");

    let kpts = make_kpts_default(&cell, NK).expect("k-mesh");
    let df = Fftdf::with_mesh(cell.clone(), &kpts, MESH).expect("FFTDF");
    let mf = Krhf::from_df(Box::new(df));
    let scf = mf.kernel(&tight()).expect("KRHF");
    assert!(scf.converged, "the Rust fixture must converge");

    let (e_band, _) = mf.get_bands(&kband, &scf.dm).expect("get_bands");

    let kband_json: Vec<Vec<f64>> = kband.iter().map(|k| k.to_vec()).collect();
    let want = run_python(
        &py,
        BAND_PY,
        &cell_args(
            &cell,
            &[
                "sto-3g".to_string(),
                serde_json::to_string(&NK.to_vec()).expect("json"),
                serde_json::to_string(&MESH.to_vec()).expect("json"),
                serde_json::to_string(&kband_json).expect("json"),
            ],
        ),
    );

    assert_eq!(
        want["version"].as_str(),
        Some("2.12.1"),
        "the oracle must be the VENDORED PySCF 2.12.1 — see tests/common/mod.rs"
    );
    assert!(
        want["converged"].as_bool().unwrap_or(false),
        "upstream did not converge"
    );

    // ---- Precondition 1: the same cell. ----
    let e_nuc_ref = want["e_nuc"].as_f64().expect("e_nuc");
    assert!(
        (scf.e_nuc - e_nuc_ref).abs() < 1e-12,
        "e_nuc {} != upstream {e_nuc_ref} — the two runs are not the same cell",
        scf.e_nuc
    );

    // ---- Precondition 2: the same converged density. ----
    let e_tot_ref = want["e_tot"].as_f64().expect("e_tot");
    let de = (scf.e_tot - e_tot_ref).abs();
    println!(
        "KRHF He 2x2x2: rust {:.15}  upstream {:.15}  delta {de:e}",
        scf.e_tot, e_tot_ref
    );
    assert!(
        de < 1e-12,
        "the densities differ (|dE_tot| = {de:e}); a band comparison would be \
         measuring density drift, not the band code"
    );

    // ---- The band eigenvalues themselves. ----
    let d_mo = compare(&scf.mo_energy, &want["e_mo"], "KRHF mo_energy (on mesh)");
    let d_band = compare(&e_band, &want["e_band"], "KRHF get_bands (off mesh)");

    // Upstream's `get_bands` chains get_hcore + the `kpts_band` J/K route over
    // a plane-wave lattice sum, then a per-k eigendecomposition. The same
    // order of cross-implementation agreement `band_kpoints.rs` measures for
    // the J/K half alone (~1.4e-9) is the most that can be expected here; the
    // assertion is deliberately the measured band of that chain, not a guess.
    assert!(
        d_mo < 1e-9,
        "SCF eigenvalues diverge from upstream: {d_mo:e}"
    );
    assert!(
        d_band < 1e-9,
        "band eigenvalues diverge from upstream: {d_band:e}"
    );
}
