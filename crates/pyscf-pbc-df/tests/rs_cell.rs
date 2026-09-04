//! `ft_ao::rs_cell::RsCell` — plan 17-10 Task 1.

mod common;

use pyscf_pbc_df::ft_ao::ft_aopair_kpt_with_images;
use pyscf_pbc_df::ft_ao::rs_cell::{LOCAL_BASIS, RCUT_THRESHOLD, RsCell, SMOOTH_BASIS};

// ---------------------------------------------------------------------------
// He-fcc / sto-3g — D-PBC-23's all-electron control: no smooth shell.
// ---------------------------------------------------------------------------

/// He-fcc/`sto-3g` decontracts to `bas_type = [1]` (all `LOCAL_BASIS`) — no
/// shell is ever split, matching D-PBC-23's stated fixture exactly.
#[test]
fn he_fcc_has_no_smooth_shell() {
    let cell = common::he_all_electron();
    // `ke_cutoff` matches what `gdf_builder._guess_eta` picks for a 2x2x2
    // mesh on this system (`measurements`/oracle: 19.65348325887675). The
    // KECUT/RCUT thresholds are the module defaults, exactly as
    // `gdf_builder.py:127` calls `_RangeSeparatedCell.from_cell`.
    let rs = RsCell::from_cell(
        &cell,
        Some(19.653_483_258_876_75),
        Some(RCUT_THRESHOLD),
        false,
    )
    .expect("He-fcc decontracts");
    assert_eq!(
        rs.cell.mol.nbas, cell.mol.nbas,
        "no shell splits for He-fcc"
    );
    assert_eq!(rs.bas_type, vec![LOCAL_BASIS]);
    assert_eq!(rs.bas_map, vec![0]);
    assert_eq!(rs.sh_loc, vec![0, 1]);

    // The "exclude_dd_block cost is exactly 0" claim (D-PBC-23) requires
    // there be no SMOOTH shell to route through the FFT at all.
    let n_smooth = rs.bas_type.iter().filter(|&&t| t == SMOOTH_BASIS).count();
    assert_eq!(n_smooth, 0, "He-fcc must have zero smooth shells");
}

// ---------------------------------------------------------------------------
// Diamond / gth-szv — D-PBC-23's flagship system: the split IS live.
// ---------------------------------------------------------------------------

/// `rs_cell.nbas == 8` where `cell.nbas == 4`, `bas_type == [1,2,1,2,1,2,1,2]`
/// — reproduces D-PBC-23's exact numbers (`14-CONTEXT.md` and
/// `measurements/ddblock.py`'s system), against the SAME `ke_cutoff`
/// `gdf_builder._guess_eta` picks on diamond 2x2x2 (oracle-measured:
/// `21.721883440437864`).
#[test]
fn diamond_gth_szv_splits_every_shell_into_local_and_smooth() {
    let cell = common::diamond();
    let rs = RsCell::from_cell(
        &cell,
        Some(21.721_883_440_437_864),
        Some(RCUT_THRESHOLD),
        false,
    )
    .expect("diamond decontracts");
    assert_eq!(cell.mol.nbas, 4);
    assert_eq!(rs.cell.mol.nbas, 8);
    assert_eq!(rs.bas_type, vec![1, 2, 1, 2, 1, 2, 1, 2]);
    assert_eq!(rs.bas_map, vec![0, 0, 1, 1, 2, 2, 3, 3]);
    assert_eq!(rs.sh_loc, vec![0, 2, 4, 6, 8]);
}

// ---------------------------------------------------------------------------
// The permutation identity — bit-exact by construction (no shared floating-
// point summation is involved: `_env` is reordered, never re-summed).
// ---------------------------------------------------------------------------

/// For every original shell, the MULTISET of `(exponent, coefficient-row)`
/// primitive records across its decontracted children is a PERMUTATION of the
/// original shell's own primitive records — bit-for-bit, since the values are
/// copied verbatim (module docs: no renormalisation ever runs). This is the
/// structural half of "recontract(decontract(x)) == x": the bookkeeping that
/// makes the physical (floating-point) identity below hold at all.
#[test]
fn decontraction_is_a_bit_exact_permutation_of_primitives() {
    for (label, cell, ke) in [
        ("he_fcc", common::he_all_electron(), 19.653_483_258_876_75),
        ("diamond", common::diamond(), 21.721_883_440_437_864),
    ] {
        let rs =
            RsCell::from_cell(&cell, Some(ke), Some(RCUT_THRESHOLD), false).expect("decontracts");
        use pyscf_core::raw_layout::{ANG_OF, BAS_SLOTS, NCTR_OF, NPRIM_OF, PTR_COEFF, PTR_EXP};

        for ib in 0..cell.mol.nbas {
            let row = &cell.mol._bas[ib * BAS_SLOTS..ib * BAS_SLOTS + BAS_SLOTS];
            let nprim = row[NPRIM_OF] as usize;
            let nctr = row[NCTR_OF] as usize;
            let pe = row[PTR_EXP] as usize;
            let pc = row[PTR_COEFF] as usize;
            let mut orig: Vec<(u64, Vec<u64>)> = (0..nprim)
                .map(|p| {
                    let e = cell.mol._env[pe + p].to_bits();
                    let c: Vec<u64> = (0..nctr)
                        .map(|ic| cell.mol._env[pc + ic * nprim + p].to_bits())
                        .collect();
                    (e, c)
                })
                .collect();
            orig.sort();

            let mut got: Vec<(u64, Vec<u64>)> = Vec::new();
            for (child, &orig_id) in rs.bas_map.iter().enumerate() {
                if orig_id as usize != ib {
                    continue;
                }
                let crow = &rs.cell.mol._bas[child * BAS_SLOTS..child * BAS_SLOTS + BAS_SLOTS];
                assert_eq!(crow[ANG_OF], row[ANG_OF], "{label}: l preserved");
                assert_eq!(crow[NCTR_OF], row[NCTR_OF], "{label}: nctr preserved");
                let cnprim = crow[NPRIM_OF] as usize;
                let cpe = crow[PTR_EXP] as usize;
                let cpc = crow[PTR_COEFF] as usize;
                for p in 0..cnprim {
                    let e = rs.cell.mol._env[cpe + p].to_bits();
                    let c: Vec<u64> = (0..nctr)
                        .map(|ic| rs.cell.mol._env[cpc + ic * cnprim + p].to_bits())
                        .collect();
                    got.push((e, c));
                }
            }
            got.sort();
            assert_eq!(
                got, orig,
                "{label} shell {ib}: decontraction is not a bit-exact permutation"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// The physical identity — `ft_aopair` at G=0 (== the overlap integral) over
// the RS cell, recontracted, equals the direct lattice sum over the
// reference cell, WITH ALL SCREENS DISABLED (`ft_aopair_kpt_with_images`,
// which is exactly `FtScreen::None`). D-PBC-21's "numerically transparent"
// claim, tested — not merely asserted.
// ---------------------------------------------------------------------------

/// This is the Task-1 gate in the plan's own words: "`ft_aopair` evaluated
/// over the RS cell and recontracted equals the current direct-lattice-sum
/// result to 1e-13 with all screens disabled". He-fcc has no split at all, so
/// its residual is exactly the summation-order-independent case (a single
/// group per shell); diamond genuinely splits every shell into LOCAL +
/// SMOOTH, and the residual there is the real test of D-PBC-21's premise.
///
/// **Both land at 1e-13 or tighter** — see `17-10-SUMMARY.md` for the
/// verify-or-refute write-up this task requires.
#[test]
fn recontracted_ft_aopair_matches_direct_lattice_sum_with_screens_off() {
    let gv = [[0.0_f64; 3]];
    let q = [0.0_f64; 3];
    let kpt = [0.0_f64; 3];

    for (label, cell, ke) in [
        ("he_fcc", common::he_all_electron(), 19.653_483_258_876_75),
        ("diamond", common::diamond(), 21.721_883_440_437_864),
    ] {
        let rs =
            RsCell::from_cell(&cell, Some(ke), Some(RCUT_THRESHOLD), false).expect("decontracts");

        // Same image list on both sides — `rs.cell.rcut` is inherited
        // UNCHANGED from `cell.rcut` (module docs), so this is a fair,
        // identically-converged comparison, exactly as `ft_aopair_kpt_with_images`
        // (Gate 1c's own mechanism) intends.
        let ls = pyscf_pbc_gto::lattice::get_lattice_ls_default(&cell).expect("Ls");

        let out_ref = ft_aopair_kpt_with_images(&cell, &gv, q, kpt, &ls).expect("ref ft_aopair");
        let out_rs = ft_aopair_kpt_with_images(&rs.cell, &gv, q, kpt, &ls).expect("rs ft_aopair");

        let recon = rs.recontract2d(&out_rs.at(0).re);
        let want = &out_ref.at(0).re;
        assert_eq!(recon.len(), want.len());
        let maxdiff = recon
            .iter()
            .zip(want.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        assert!(
            maxdiff <= 1e-13,
            "{label}: recontracted ft_aopair(G=0) differs from the direct sum by {maxdiff:e} (tol 1e-13)"
        );
    }
}

// ---------------------------------------------------------------------------
// `_reverse_bas_map` == `sh_loc` — upstream's own internal debug assertion
// (`ft_ao.py:369-370`).
// ---------------------------------------------------------------------------

#[test]
fn reverse_bas_map_matches_sh_loc() {
    for (cell, ke) in [
        (common::he_all_electron(), 19.653_483_258_876_75),
        (common::diamond(), 21.721_883_440_437_864),
    ] {
        let rs =
            RsCell::from_cell(&cell, Some(ke), Some(RCUT_THRESHOLD), false).expect("decontracts");
        assert_eq!(RsCell::reverse_bas_map(&rs.bas_map), rs.sh_loc);
    }
}

// ---------------------------------------------------------------------------
// The trivial (`ke_cut_threshold = None`) short-circuit.
// ---------------------------------------------------------------------------

#[test]
fn no_threshold_is_a_trivial_wrap() {
    let cell = common::diamond();
    let rs = RsCell::from_cell(&cell, None, None, false).expect("trivial wrap");
    assert_eq!(rs.cell.mol.nbas, cell.mol.nbas);
    assert!(rs.bas_type.iter().all(|&t| t == LOCAL_BASIS));
    assert_eq!(rs.bas_map, (0..cell.mol.nbas as i32).collect::<Vec<_>>());
    assert_eq!(rs.sh_loc, (0..=cell.mol.nbas as i32).collect::<Vec<_>>());
}

// ---------------------------------------------------------------------------
// `smooth_basis_cell` / `compact_basis_cell` — the split-basis view cells.
// ---------------------------------------------------------------------------

#[test]
fn smooth_and_compact_cells_partition_the_decontracted_shells() {
    let cell = common::diamond();
    let rs = RsCell::from_cell(
        &cell,
        Some(21.721_883_440_437_864),
        Some(RCUT_THRESHOLD),
        false,
    )
    .expect("decontracts");

    let smooth = rs.smooth_basis_cell().expect("smooth_basis_cell");
    let compact = rs.compact_basis_cell().expect("compact_basis_cell");

    assert_eq!(
        smooth.mol.nbas, 4,
        "one SMOOTH shell per original shell here"
    );
    assert_eq!(
        compact.cell.mol.nbas, 4,
        "one LOCAL shell per original shell here"
    );
    assert_eq!(smooth.mol.nbas + compact.cell.mol.nbas, rs.cell.mol.nbas);
    assert!(compact.bas_type.iter().all(|&t| t != SMOOTH_BASIS));
}

/// He-fcc has no smooth shell at all, so `smooth_basis_cell` must come back
/// empty and `compact_basis_cell` must reproduce `rs_cell` unchanged.
#[test]
fn he_fcc_smooth_basis_cell_is_empty() {
    let cell = common::he_all_electron();
    let rs = RsCell::from_cell(
        &cell,
        Some(19.653_483_258_876_75),
        Some(RCUT_THRESHOLD),
        false,
    )
    .expect("decontracts");
    let smooth = rs.smooth_basis_cell().expect("smooth_basis_cell");
    assert_eq!(smooth.mol.nbas, 0);
    let compact = rs.compact_basis_cell().expect("compact_basis_cell");
    assert_eq!(compact.cell.mol.nbas, rs.cell.mol.nbas);
}

// ---------------------------------------------------------------------------
// `get_ao_type` — one tag per AO.
// ---------------------------------------------------------------------------

#[test]
fn get_ao_type_tags_every_ao() {
    let cell = common::diamond();
    let rs = RsCell::from_cell(
        &cell,
        Some(21.721_883_440_437_864),
        Some(RCUT_THRESHOLD),
        false,
    )
    .expect("decontracts");
    let tags = rs.get_ao_type();
    assert_eq!(tags.len(), rs.cell.mol.nao_nr);
    let n_smooth_ao = tags.iter().filter(|&&t| t == SMOOTH_BASIS).count();
    assert!(n_smooth_ao > 0, "diamond has smooth AOs");
}
