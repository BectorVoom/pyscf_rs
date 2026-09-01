//! D-PBC-23's `exclude_dd_block` — the Ha-level SCF acceptance numbers. Plan
//! 17-10 Task 3.
//!
//! `crates/pyscf-pbc-df/tests/exclude_dd_block.rs` gates the CDERI-tensor
//! seam (He-fcc's bit-identical-by-construction zero, diamond's
//! must-differ). This file gates the number that seam is FOR: the `KRHF`
//! energy difference D-PBC-23 priced, reproducing `measurements/ddblock.py`'s
//! own methodology exactly — `KRHF`, `conv_tol=1e-11`, `exxdiv='ewald'`,
//! `_CCGDFBuilder` (`prefer_ccdf = true`) with `exclude_dd_block` flipped.
//!
//! **Run in `--release`.** A debug build of one diamond `KRHF` cycle already
//! takes minutes on this workspace's PRE-EXISTING (unrelated to this plan)
//! real-space `aux_e2` lattice sum
//! (`crates/pyscf-pbc-df/tests/gdf_builder.rs`'s own
//! `cderi_fingerprint_matches_upstream_diamond`, marked `#[ignore]` for the
//! same reason); this plan's correction adds a second, smaller real-space
//! pass plus an FFT pass on top of it, and BOTH `exclude_dd_block` values are
//! built here.
//!
//! ```text
//! PYSCF_ORACLE_VENV=1 cargo test --release -p pyscf-pbc-scf \
//!     --test exclude_dd_block_energy -- --ignored --nocapture
//! ```

mod common;

use common::diamond;
use pyscf_pbc_df::Gdf;
use pyscf_pbc_scf::krhf::Krhf;
use pyscf_pbc_scf::types::KScfConfig;

fn cfg(cell: &pyscf_pbc_gto::Cell) -> KScfConfig {
    let mut c = KScfConfig::for_cell(cell);
    c.conv_tol = 1e-11;
    c.max_cycle = 60;
    c
}

/// `(E(exclude_dd_block=true), E(exclude_dd_block=false))` on `_CCGDFBuilder`
/// (`prefer_ccdf = true`, matching D-PBC-23's own `measurements/ddblock.py`).
fn dd_block_energies(cell: pyscf_pbc_gto::Cell, kmesh: [usize; 3]) -> (f64, f64, bool, bool) {
    let kpts = pyscf_pbc_gto::kpts_mesh::make_kpts(&cell, kmesh, false, true, None)
        .expect("kpts");

    let mut d_true = Gdf::new(cell.clone(), &kpts);
    d_true.prefer_ccdf = true;
    d_true.exclude_dd_block = Some(true);
    let a = Krhf::from_df(Box::new(d_true));
    let ra = a.kernel(&cfg(&cell)).expect("scf exclude_dd_block=true");

    let mut d_false = Gdf::new(cell.clone(), &kpts);
    d_false.prefer_ccdf = true;
    d_false.exclude_dd_block = Some(false);
    let b = Krhf::from_df(Box::new(d_false));
    let rb = b.kernel(&cfg(&cell)).expect("scf exclude_dd_block=false");

    (ra.e_tot, rb.e_tot, ra.converged, rb.converged)
}

/// **Oracle target (D-PBC-23 / `measurements/ddblock.py`): 2.900e-08 Ha.**
/// Independently re-derived against the vendored PySCF 2.12.1 oracle for
/// this plan: `2.9002556800605817e-08` (`scratchpad/ddblock_energy.py`,
/// same `_CCGDFBuilder`/`make_j3c(aosym='s2')`/`KRHF` recipe this test uses).
#[test]
#[ignore = "slow — run with --release; see this file's module docs"]
fn diamond_gamma_matches_upstream() {
    let (e_true, e_false, conv_true, conv_false) = dd_block_energies(diamond(), [1, 1, 1]);
    assert!(conv_true && conv_false, "both SCFs must converge");
    let diff = (e_true - e_false).abs();
    eprintln!("gamma: E(true)={e_true:.14} E(false)={e_false:.14} |dE|={diff:e}");
    assert!(
        (diff - 2.900e-8).abs() < 2e-9,
        "diamond gamma |dE| = {diff:e}, want ~2.900e-8 (D-PBC-23)"
    );
}

/// **Oracle target (D-PBC-23 / `measurements/ddblock.py`): 1.835e-08 Ha.**
#[test]
#[ignore = "slow — run with --release; see this file's module docs"]
fn diamond_2x2x2_matches_upstream() {
    let (e_true, e_false, conv_true, conv_false) = dd_block_energies(diamond(), [2, 2, 2]);
    assert!(conv_true && conv_false, "both SCFs must converge");
    let diff = (e_true - e_false).abs();
    eprintln!("2x2x2: E(true)={e_true:.14} E(false)={e_false:.14} |dE|={diff:e}");
    assert!(
        (diff - 1.835e-8).abs() < 2e-9,
        "diamond 2x2x2 |dE| = {diff:e}, want ~1.835e-8 (D-PBC-23)"
    );
}

/// **He-fcc: exactly 0**, the strongest of the three targets — restated here
/// at the SCF level (the tensor-level bit-identity is
/// `pyscf-pbc-df/tests/exclude_dd_block.rs`). Fast enough to run by default
/// (no `#[ignore]`): He-fcc's real-space `aux_e2` is not the "minutes" cost
/// diamond's is.
#[test]
fn he_fcc_2x2x2_is_bit_identical() {
    let (e_true, e_false, conv_true, conv_false) =
        dd_block_energies(common::he_all_electron(), [2, 2, 2]);
    assert!(conv_true && conv_false, "both SCFs must converge");
    assert_eq!(
        e_true.to_bits(),
        e_false.to_bits(),
        "He-fcc has no smooth shell (D-PBC-23) — exclude_dd_block must not move the energy \
         at all, not merely by a small amount: {e_true:.17e} vs {e_false:.17e}"
    );
}
