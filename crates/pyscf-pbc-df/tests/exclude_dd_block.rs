//! D-PBC-23's `exclude_dd_block` — plan 17-10 Task 3.
//!
//! The Ha-level energy gates (1.835e-8 diamond 2x2x2, 2.900e-8 diamond gamma,
//! exactly 0 on He-fcc) are SCF-level and live in
//! `crates/pyscf-pbc-scf/tests/exclude_dd_block_energy.rs`, where a `KRHF`
//! driver is available. This file gates the seam this crate owns: the
//! `CcGdfBuilder`/`RsGdfBuilder` `exclude_dd_block` flag builds cleanly both
//! ways, and He-fcc/`sto-3g` under `_CCGDFBuilder`'s OWN `ke_cut_threshold`
//! (D-PBC-23's all-electron control — no smooth shell there) produces a
//! BIT-IDENTICAL `cderi` regardless of the flag. That identity is the
//! strongest form of "cost is exactly 0": it is not a tolerance, it is
//! `re.to_bits() == re.to_bits()`. The range-separated builder derives its
//! OWN threshold from a different formula (`_guess_omega`, not `_guess_eta`)
//! and genuinely splits He-fcc's single shell at some k-meshes — see that
//! test's own doc comment for the measured detail.

mod common;

use pyscf_pbc_df::gdf_builder::CcGdfBuilder;
use pyscf_pbc_df::incore::Aosym;
use pyscf_pbc_df::rsdf_builder::RsGdfBuilder;

fn kpts(cell: &pyscf_pbc_gto::Cell, km: [usize; 3]) -> Vec<[f64; 3]> {
    pyscf_pbc_gto::kpts_mesh::make_kpts(cell, km, false, true, None).expect("kpts")
}

fn bits(t: &pyscf_algebra::CTensor) -> (Vec<u64>, Vec<u64>) {
    (
        t.re.iter().map(|v| v.to_bits()).collect(),
        t.im.iter().map(|v| v.to_bits()).collect(),
    )
}

/// **He-fcc has no smooth shell (D-PBC-23) — the two routes must be
/// BIT-IDENTICAL, not merely close.** Any implementation of `exclude_dd_block`
/// that perturbs a cell with no smooth basis is wrong by construction; this is
/// the strongest of the plan's three acceptance numbers because it has no
/// tolerance to hide behind.
#[test]
fn he_fcc_gdf_cderi_is_bit_identical_either_way() {
    let cell = common::he_all_electron();
    let k = kpts(&cell, [2, 2, 2]);

    let mut b_true = CcGdfBuilder::new(cell.clone(), &k);
    b_true.exclude_dd_block = true;
    b_true.build().expect("build true");
    let c_true = b_true.make_j3c(Aosym::S1, false).expect("j3c true");

    let mut b_false = CcGdfBuilder::new(cell, &k);
    b_false.exclude_dd_block = false;
    b_false.build().expect("build false");
    let c_false = b_false.make_j3c(Aosym::S1, false).expect("j3c false");

    assert_eq!(c_true.blocks.len(), c_false.blocks.len());
    for (key, block_true) in &c_true.blocks {
        let block_false = c_false.blocks.get(key).unwrap_or_else(|| {
            panic!("block {key} present with exclude_dd_block=true but not false")
        });
        assert_eq!(
            bits(&block_true.data),
            bits(&block_false.data),
            "block {key}: dd=true and dd=false must be bit-identical on He-fcc \
             (no smooth shell to route through the FFT)"
        );
    }
}

/// The RANGE-SEPARATED builder's `exclude_dd_block` seam builds cleanly both
/// ways on He-fcc.
///
/// **Not a bit-identical claim here.** D-PBC-23's "He-fcc has no smooth
/// shell" is measured against `_CCGDFBuilder`'s own `ke_cut_threshold`
/// (`_guess_eta`'s mesh); [`RsGdfBuilder`] derives its threshold from
/// `_guess_omega` instead, a DIFFERENT formula, and at `kmesh=[1,1,1]` it
/// genuinely splits He-fcc's single `sto-3g` shell into LOCAL + SMOOTH
/// (`bas_type = [1, 2]`, confirmed by direct inspection) — so the two routes
/// are EXPECTED to differ here, and asserting bit-identity would be testing
/// the wrong invariant. The GDF/CompensatedCharge route above is where
/// D-PBC-23's all-electron control actually applies; this test only confirms
/// the seam is live on the range-separated builder too.
#[test]
fn he_fcc_rsdf_dd_block_seam_is_live() {
    let cell = common::he_all_electron();
    let k = kpts(&cell, [1, 1, 1]);

    let mut b_true = RsGdfBuilder::new(cell.clone(), &k);
    b_true.exclude_dd_block = true;
    b_true.build().expect("build true");
    let rs = b_true.rs_cell.as_ref().expect("rs_cell built when exclude_dd_block");
    assert_eq!(rs.bas_type, vec![1, 2], "He-fcc DOES split under RSGDF's own threshold");
    let c_true = b_true.make_j3c(Aosym::S1, false).expect("j3c true");
    assert!(!c_true.blocks.is_empty());

    let mut b_false = RsGdfBuilder::new(cell, &k);
    b_false.exclude_dd_block = false;
    b_false.build().expect("build false");
    let c_false = b_false.make_j3c(Aosym::S1, false).expect("j3c false");
    assert!(!c_false.blocks.is_empty());
}

/// Both flag values build and produce a well-formed `Cderi` on a system that
/// DOES split (diamond) — the structural half of the gate. The Ha-level
/// energy numbers are the oracle test in `pyscf-pbc-scf`; this just proves
/// the seam is live and does not panic or silently no-op.
#[test]
#[ignore = "slow — diamond's real-space aux_e2 is minutes in a debug build \
            (see tests/gdf_builder.rs's cderi_fingerprint_matches_upstream_diamond), \
            and this test runs it twice (dd=true adds a THIRD, smaller pass)"]
fn diamond_gdf_both_routes_produce_a_cderi() {
    let cell = common::diamond();
    let k = kpts(&cell, [1, 1, 1]);

    let mut b_true = CcGdfBuilder::new(cell.clone(), &k);
    b_true.exclude_dd_block = true;
    b_true.build().expect("build true");
    // Confirm the SMOOTH cell that carries the correction is genuinely
    // non-empty here — the point of testing diamond rather than He-fcc.
    let rs = b_true.rs_cell.as_ref().expect("rs_cell built when exclude_dd_block");
    let smooth = rs.smooth_basis_cell().expect("smooth_basis_cell");
    assert!(smooth.mol.nao_nr > 0, "diamond must have a live smooth block");
    let c_true = b_true.make_j3c(Aosym::S1, false).expect("j3c true");

    let mut b_false = CcGdfBuilder::new(cell, &k);
    b_false.exclude_dd_block = false;
    b_false.build().expect("build false");
    let c_false = b_false.make_j3c(Aosym::S1, false).expect("j3c false");

    assert_eq!(c_true.blocks.len(), c_false.blocks.len());
    // The two routes must DIFFER here (unlike He-fcc) — a silent no-op on a
    // cell that has a smooth block would be as wrong as an incorrect
    // correction, just quieter.
    let mut any_diff = false;
    for (key, bt) in &c_true.blocks {
        let bf = &c_false.blocks[key];
        if bt.data.re != bf.data.re || bt.data.im != bf.data.im {
            any_diff = true;
        }
    }
    assert!(any_diff, "exclude_dd_block must change diamond's cderi (D-PBC-23)");
}
