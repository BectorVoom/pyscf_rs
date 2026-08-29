//! `_CCNucBuilder` — the compensated nuclear-attraction builder
//! (`gdf_builder.py:497-727`), plus `get_nuc` / `get_pp`
//! (`rsdf_builder.py:1013-1038`). Plan 14-03, Task 2.
//!
//! # Its `eta` is NOT `_CCGDFBuilder`'s
//!
//! `_CCGDFBuilder` runs `_guess_eta(auxcell, kpts, mesh)`; this one uses
//! `eta = max(0.5 / (0.5 + nkpts^(1/9)), ETA_MIN)` and then derives the mesh
//! from it (`gdf_builder.py:517-546`). Two different knobs with the same name,
//! ported separately and tested separately.
//!
//! # `get_pp` part 2 goes through the Phase-10 route, deliberately
//!
//! Phase 13 measured that upstream's own `aft._IntPPBuilder.get_pp_loc_part2`
//! and `pseudo.pp_int.get_pp_loc_part2` **disagree with each other by
//! 1.7933e-9** in PySCF 2.12.1, and that `fft.get_pp` agrees with the `pp_int`
//! route. This port uses `pp_int` — via
//! `pyscf_pbc_gto::pseudo::vloc_part2::get_pp_loc_part2` — for the same reason
//! plan 13-04 did, and `tests/gdf.rs` asserts the substitution so the
//! attribution cannot rot.
//!
//! Phase-13 defect #4 also applies: `get_pp_loc_part2` and `get_pp_nl` return
//! **F-order**; they are transposed before being added.

use pyscf_algebra::CTensor;
use pyscf_pbc_gto::Cell;

use crate::aftdf::Aftdf;
use crate::error::PbcDfError;
use crate::gdf_builder::eta::{ETA_MIN, estimate_eta_for_ke_cutoff, estimate_ke_cutoff_for_eta};

/// `_CCNucBuilder.build(eta)`'s `eta` — `gdf_builder.py:525-527`.
pub fn nuc_eta(nkpts: usize) -> f64 {
    (0.5 / (0.5 + (nkpts.max(1) as f64).powf(1.0 / 9.0))).max(ETA_MIN)
}

/// The `(eta, mesh, ke_cutoff)` `_CCNucBuilder` settles on —
/// `gdf_builder.py:517-546`, the `dimension > 0` branch.
///
/// # Errors
/// Propagates `cutoff_to_mesh` / `mesh_to_cutoff`.
pub fn nuc_eta_mesh(cell: &Cell, nkpts: usize) -> Result<(f64, [usize; 3], f64), PbcDfError> {
    let eta0 = nuc_eta(nkpts);
    let ke0 = estimate_ke_cutoff_for_eta(cell, eta0, None);
    let mesh = cell.cutoff_to_mesh(ke0)?;
    let a = cell.lattice_vectors();
    let dim = (cell.dimension as usize).max(1);
    let ke_cutoff = pyscf_pbc_tools::mesh::mesh_to_cutoff(&a, mesh)?[..dim]
        .iter()
        .copied()
        .fold(f64::INFINITY, f64::min);
    let eta = estimate_eta_for_ke_cutoff(cell, ke_cutoff, None);
    Ok((eta, mesh, ke_cutoff))
}

/// `get_nuc(nuc_builder)` — `rsdf_builder.py:1013-1019`, i.e.
/// `get_pp_loc_part1(with_pseudo = False)`.
///
/// # This delegates to AFTDF at the CELL's mesh, and that choice is measured
///
/// `_CCNucBuilder` splits `get_pp_loc_part1` into a real-space `_int_nuc_vloc`
/// against the compensating nuclear charge plus a reciprocal-space remainder,
/// and that split is what lets it run on the tiny compensated mesh —
/// `[9,9,9]` on He-fcc against the cell's `[43,43,43]`.
///
/// **Evaluating the WHOLE nuclear attraction on the compensated mesh is wrong,
/// and plan 14-04 measured how wrong.** The mesh resolves the MODEL CHARGE, not
/// the nuclear density:
///
/// | mesh | `v_nuc[0,0]`, He-fcc 2×2×2 |
/// |---|---|
/// | `[9,9,9]` (`_CCNucBuilder`'s) | −1.835938176640 |
/// | `[15,15,15]` | −1.871405034120 |
/// | `[21,21,21]` | −1.872891481488 |
/// | `[31,31,31]` | −1.872934360277 |
/// | `[43,43,43]` (the cell's) | −1.872934388301 |
///
/// The `[9,9,9]` value is off by 3.7e-2 per element, which took the converged
/// `KRHF` 0.0743 Ha away from upstream. So this uses the CELL's mesh, where
/// AFTDF's `get_nuc` is oracle-gated to 2.755e-12 (`13-VERIFICATION.md`).
///
/// The cost of not porting the split is therefore **performance, not accuracy**:
/// a converged `get_nuc` at `[43,43,43]` instead of a split one at `[9,9,9]`.
/// Porting `_int_nuc_vloc` is a named carry-over.
///
/// # Errors
/// Propagates the AFTDF build and `ft_loop`.
pub fn get_nuc(cell: &Cell, kpts: &[[f64; 3]]) -> Result<Vec<CTensor>, PbcDfError> {
    let aft = Aftdf::new(cell.clone(), kpts)?;
    crate::aftdf::get_nuc(&aft, kpts)
}

/// `get_pp(nuc_builder)` — `rsdf_builder.py:1021-1038`:
/// `get_pp_loc_part1() + get_pp_loc_part2() + get_pp_nl()`.
///
/// Same mesh choice, and the same reason — see [`get_nuc`].
///
/// Part 2 is the k-resolved `pp_int::get_pp_loc_part2_kpts` (upstream's
/// `aft._IntPPBuilder`), which Phase 14 ported after plan 14-03 found Phase
/// 10's gamma-only version blocking every k-point pseudopotential path.
///
/// # Errors
/// Propagates each of the three parts.
pub fn get_pp(cell: &Cell, kpts: &[[f64; 3]]) -> Result<Vec<CTensor>, PbcDfError> {
    let aft = Aftdf::new(cell.clone(), kpts)?;
    crate::aftdf::get_pp(&aft, kpts)
}
