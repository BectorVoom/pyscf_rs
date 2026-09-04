//! `ft_ao::ExtendedMole` — plan 17-10 Task 2.
//!
//! Gated BOTH ways, per the plan: agreement with upstream's per-shell-pair
//! radii (`gdf_builder.estimate_rcut`, ~1e-9) AND no regression against this
//! port's own flattened-maximum result (`gdf_builder::eta::estimate_rcut`,
//! already gated since plan 14-02 at 16.729034885581783 on diamond).

mod common;

use pyscf_pbc_df::ft_ao::rs_cell::RCUT_THRESHOLD;
use pyscf_pbc_df::ft_ao::{ExtendedMole, RsCell};
use pyscf_pbc_df::gdf_builder::{estimate_rcut, estimate_rcut_per_shell, fuse_auxcell};

fn kpts(cell: &pyscf_pbc_gto::Cell, km: [usize; 3]) -> Vec<[f64; 3]> {
    pyscf_pbc_gto::kpts_mesh::make_kpts(cell, km, false, true, None).expect("kpts")
}

/// **Oracle (offline-recorded — `PYSCF_ORACLE_VENV` reproduces it live).**
/// Diamond/`gth-szv` 2x2x2, `_CCGDFBuilder`'s own `ke_cutoff`
/// (`21.721883440437864`). Upstream's `gdf_builder.estimate_rcut(rs_cell,
/// fused_cell, exclude_dd_block=False)`:
/// `[11.443289749179039, 15.929321195778803, 11.831713483884991,
///   16.729034885581783]` repeated once per carbon atom (identical shells).
#[test]
fn estimate_rcut_per_shell_matches_upstream_diamond_false() {
    let cell = common::diamond();
    let k = kpts(&cell, [2, 2, 2]);
    let ke_cutoff = 21.721_883_440_437_864;

    let rs = RsCell::from_cell(&cell, Some(ke_cutoff), Some(RCUT_THRESHOLD), false)
        .expect("decontracts");
    // eta for this system (measured, `gdf_builder.rs`'s own doc comment).
    let fused = fuse_auxcell(&cell, None, 0.464_883_124_929_945_55).expect("fuse");

    let rcut = estimate_rcut_per_shell(&rs, &fused.fused.cell, None, false);
    let want = [
        11.443_289_749_179_039,
        15.929_321_195_778_803,
        11.831_713_483_884_991,
        16.729_034_885_581_783,
        11.443_289_749_179_039,
        15.929_321_195_778_803,
        11.831_713_483_884_991,
        16.729_034_885_581_783,
    ];
    assert_eq!(rcut.len(), want.len());
    for (got, want) in rcut.iter().zip(want.iter()) {
        assert!(
            (got - want).abs() < 1e-9,
            "estimate_rcut_per_shell diverges from upstream: got {got:e}, want {want:e}"
        );
    }

    // No regression: the max of the per-shell array reproduces this port's
    // OWN already-gated flattened scalar exactly.
    let scalar = estimate_rcut(&cell, &fused.fused.cell, None);
    let max_of_array = rcut.iter().cloned().fold(0.0_f64, f64::max);
    assert!(
        (scalar - max_of_array).abs() < 1e-12,
        "estimate_rcut_per_shell's max ({max_of_array:e}) regresses vs the flattened \
         scalar estimate_rcut ({scalar:e})"
    );
    let k = k.len();
    let _ = k;
}

/// **Oracle.** `exclude_dd_block=True` half — the SMOOTH shells' radius is
/// overridden to the single most diffuse COMPACT shell's value, per
/// `gdf_builder.py:975-1006`. Upstream: `[..., 15.979819339486047, ...,
/// 16.77819497058717, ...]` (only positions 1 and 3 — the SMOOTH shells —
/// move; 0 and 2 — LOCAL — are untouched).
#[test]
fn estimate_rcut_per_shell_matches_upstream_diamond_true() {
    let cell = common::diamond();
    let ke_cutoff = 21.721_883_440_437_864;
    let rs = RsCell::from_cell(&cell, Some(ke_cutoff), Some(RCUT_THRESHOLD), false)
        .expect("decontracts");
    let fused = fuse_auxcell(&cell, None, 0.464_883_124_929_945_55).expect("fuse");

    let rcut = estimate_rcut_per_shell(&rs, &fused.fused.cell, None, true);
    let want = [
        11.443_289_749_179_039,
        15.979_819_339_486_047,
        11.831_713_483_884_991,
        16.778_194_970_587_17,
        11.443_289_749_179_039,
        15.979_819_339_486_047,
        11.831_713_483_884_991,
        16.778_194_970_587_17,
    ];
    for (got, want) in rcut.iter().zip(want.iter()) {
        assert!(
            (got - want).abs() < 1e-9,
            "estimate_rcut_per_shell(exclude_dd_block=true) diverges: got {got:e}, want {want:e}"
        );
    }
    // The LOCAL shells (0, 2) must be UNCHANGED from the false route.
    let rcut_false = estimate_rcut_per_shell(&rs, &fused.fused.cell, None, false);
    assert!((rcut[0] - rcut_false[0]).abs() < 1e-15);
    assert!((rcut[2] - rcut_false[2]).abs() < 1e-15);
}

/// **Oracle.** `ExtendedMole::from_cell` + `strip_basis` on the SAME system:
/// upstream's surviving `(bvk, shell, image)` triple count after
/// `strip_basis(rcut)` is **1450** (`bvk_ncells=8, nimgs=201` before
/// stripping, `12864` raw triples).
#[test]
fn strip_basis_surviving_count_matches_upstream_diamond() {
    let cell = common::diamond();
    let ke_cutoff = 21.721_883_440_437_864;
    let rs = RsCell::from_cell(&cell, Some(ke_cutoff), Some(RCUT_THRESHOLD), false)
        .expect("decontracts");
    let fused = fuse_auxcell(&cell, None, 0.464_883_124_929_945_55).expect("fuse");
    let rcut = estimate_rcut_per_shell(&rs, &fused.fused.cell, None, false);
    let rcut_max = rcut.iter().cloned().fold(0.0_f64, f64::max);

    let mut supmol =
        ExtendedMole::from_cell(&rs, [2, 2, 2], Some(rcut_max)).expect("ExtendedMole::from_cell");
    assert_eq!(
        supmol.bas_mask.len(),
        12864,
        "raw (bvk,shell,image) triple count"
    );
    assert_eq!(supmol.ls.len(), 201, "nimgs");

    supmol.strip_basis(&rcut);
    let surviving = supmol.bas_mask.iter().filter(|&&b| b).count();
    assert_eq!(surviving, 1450, "surviving triples after strip_basis");
}
