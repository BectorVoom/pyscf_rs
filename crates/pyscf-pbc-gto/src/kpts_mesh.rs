//! k-point meshes — `make_kpts` (`pyscf/pbc/gto/cell.py:827-884`) and the
//! `Cell`-taking wrappers over [`pyscf_pbc_lib::kpts_helper`].
//!
//! The momentum-conservation tables themselves live in `pyscf-pbc-lib`, the
//! bottom of the periodic DAG, because they read nothing off a `Cell` but its
//! lattice vectors. Same split plan 09-04 and 09-06 used.

use crate::cell::Cell;
use pyscf_core::PyscfRsError;
use pyscf_pbc_lib::kpts_helper as kh;
pub use pyscf_pbc_lib::kpts_helper::{
    KCONSERV_TOL, KPT_DIFF_TOL, KIdx, Kconserv, Kconserv3, UniqueKpts, gamma_point, intersection,
    is_gamma_point, is_zero, member, round_to_fbz, unique,
};

/// Upstream `WRAP_AROUND` (`cell.py:42`) — `make_kpts`'s default.
pub const WRAP_AROUND: bool = false;
/// Upstream `WITH_GAMMA` (`cell.py:43`) — `make_kpts`'s default.
pub const WITH_GAMMA: bool = true;

/// Generate a Monkhorst-Pack k-point mesh. Ports `make_kpts`
/// (`cell.py:827-884`).
///
/// Returns ABSOLUTE k-points in 1/Bohr, C-order over the three axes (last index
/// fastest, matching `lib.cartesian_prod`).
///
/// * `with_gamma_point` — `true` puts the grid at `arange(n)/n` (Gamma is the
///   zeroth point); `false` shifts it to `(arange(n) + 0.5)/n - 0.5`. Passing a
///   `scaled_center` forces the `arange(n)/n` grid regardless
///   (`cell.py:862`), so the center IS the zeroth point.
/// * `wrap_around` — folds each AXIS grid into `[-0.5, 0.5)` (`ks[ks>=.5] -= 1`)
///   BEFORE the cartesian product and BEFORE `scaled_center` is added.
/// * `scaled_center` — added to every point after the fold; in units of the
///   reciprocal lattice.
///
/// # Note on the plan text
/// PBC-MASTER-PLAN §8.1 plan 09-07 step 1 describes `wrap_around` as
/// `round_to_fbz(scaled_kpts)` applied to the finished product. Upstream does
/// something different and observably so — it folds each axis independently,
/// before the product and before the center shift, and it does no rounding or
/// `cleanse`. RULE 2 makes `cell.py:866-867` authoritative; this is a port of
/// that.
///
/// # Errors
/// * [`PyscfRsError::NotYetImplemented`] `{ phase: 17 }` for
///   `space_group_symmetry` / `time_reversal_symmetry`, which return a
///   `KPoints` object upstream (D-PBC-15).
/// * [`pyscf_core::CoreError::InvalidMolecule`] if the lattice is singular, or
///   if any `nks` entry is zero (upstream divides by it).
pub fn make_kpts(
    cell: &Cell,
    nks: [usize; 3],
    wrap_around: bool,
    with_gamma_point: bool,
    scaled_center: Option<[f64; 3]>,
) -> Result<Vec<[f64; 3]>, PyscfRsError> {
    if nks.contains(&0) {
        return Err(PyscfRsError::Core(
            pyscf_core::CoreError::InvalidMolecule(format!(
                "make_kpts: nks {nks:?} has a zero axis"
            )),
        ));
    }
    // cell.py:860-868 — one 1-D grid per axis.
    let mut ks_each_axis: [Vec<f64>; 3] = [Vec::new(), Vec::new(), Vec::new()];
    for (axis, ks) in ks_each_axis.iter_mut().enumerate() {
        let n = nks[axis];
        *ks = if with_gamma_point || scaled_center.is_some() {
            // ks = np.arange(n, dtype=float) / n
            (0..n).map(|i| i as f64 / n as f64).collect()
        } else {
            // ks = (np.arange(n)+.5)/n-.5
            (0..n).map(|i| (i as f64 + 0.5) / n as f64 - 0.5).collect()
        };
        if wrap_around {
            // ks[ks>=.5] -= 1
            for k in ks.iter_mut() {
                if *k >= 0.5 {
                    *k -= 1.0;
                }
            }
        }
    }
    // cell.py:869-872 — cartesian_prod, then the (possibly zero) center shift.
    let center = scaled_center.unwrap_or([0.0, 0.0, 0.0]);
    let mut scaled_kpts = Vec::with_capacity(nks[0] * nks[1] * nks[2]);
    for &x in &ks_each_axis[0] {
        for &y in &ks_each_axis[1] {
            for &z in &ks_each_axis[2] {
                scaled_kpts.push([x + center[0], y + center[1], z + center[2]]);
            }
        }
    }
    // cell.py:873
    cell.get_abs_kpts(&scaled_kpts)
}

/// [`make_kpts`] with upstream's defaults: `wrap_around = WRAP_AROUND`,
/// `with_gamma_point = WITH_GAMMA`, `scaled_center = None`.
///
/// # Errors
/// As [`make_kpts`].
pub fn make_kpts_default(cell: &Cell, nks: [usize; 3]) -> Result<Vec<[f64; 3]>, PyscfRsError> {
    make_kpts(cell, nks, WRAP_AROUND, WITH_GAMMA, None)
}

/// `make_kpts` with k-point symmetry — `cell.py:874-883`.
///
/// # Errors
/// Always [`PyscfRsError::NotYetImplemented`] `{ phase: 17 }`: upstream returns
/// a `pbc.lib.kpts.KPoints` object here, and k-point symmetry is a Phase 17
/// add-on layer (D-PBC-15), never a fork of the Phase 11/12 drivers.
pub fn make_kpts_with_symmetry(
    _cell: &Cell,
    _nks: [usize; 3],
    _space_group_symmetry: bool,
    _time_reversal_symmetry: bool,
) -> Result<Vec<[f64; 3]>, PyscfRsError> {
    Err(PyscfRsError::NotYetImplemented {
        phase: 17,
        what: "make_kpts with space_group_symmetry / time_reversal_symmetry \
               returns a KPoints object (cell.py:874-883)",
    })
}

/// `get_kconserv(cell, kpts)` — `kpts_helper.py:291-325`. See
/// [`pyscf_pbc_lib::kpts_helper::get_kconserv`] for the algorithm and for the
/// one deviation (the `k2gamma` shortcut is not ported).
pub fn get_kconserv(cell: &Cell, kpts: &[[f64; 3]]) -> Kconserv {
    kh::get_kconserv(&cell.lattice_vectors(), kpts)
}

/// `get_kconserv3(cell, kpts, kijkab)` — `kpts_helper.py:409-439`.
///
/// # Panics
/// If `kijkab` does not have exactly 5 entries, or an index is out of range.
pub fn get_kconserv3(cell: &Cell, kpts: &[[f64; 3]], kijkab: &[KIdx]) -> Kconserv3 {
    kh::get_kconserv3(&cell.lattice_vectors(), kpts, kijkab)
}

impl Cell {
    /// `cell.make_kpts(nks)` with upstream's defaults. See [`make_kpts`].
    ///
    /// # Errors
    /// As [`make_kpts`].
    pub fn make_kpts(&self, nks: [usize; 3]) -> Result<Vec<[f64; 3]>, PyscfRsError> {
        make_kpts_default(self, nks)
    }
}
