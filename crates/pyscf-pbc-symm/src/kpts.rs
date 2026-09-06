//! Port of `pyscf/pbc/lib/kpts.py` (1223 l) — `KPoints`, the IBZ fold and
//! the IBZ -> BZ unfolds (`17-05-PLAN.md`).
//!
//! # Why this is not in `pyscf-pbc-lib` (D-PBC-25 / 17-CONTEXT §4)
//!
//! Upstream declares `class KPoints(symm.Symmetry, lib.StreamObject)`
//! (`kpts.py:847`). Mirroring the file name would put `KPoints` in
//! `pyscf-pbc-lib`, but that crate sits BELOW `pyscf-pbc-symm` in the
//! periodic DAG and cannot see either [`Symmetry`] or `Cell`. The file-name
//! mirror is therefore broken deliberately: `KPoints` lives here, next to
//! [`Symmetry`], and holds one by **composition**, not inheritance.
//!
//! # `KPoints` never owns a `Cell` (17-CONTEXT §3.9)
//!
//! Upstream's `KPoints` stores `self.cell` and `cell.py:1576-1579` then
//! deletes the back-references purely to break a Python refcount cycle. This
//! port follows [`Symmetry::build`]'s established discipline instead: every
//! method that needs a cell takes a BORROWED `&Cell` for the duration of the
//! call and stores nothing. A cloned `Cell` would silently desynchronise on
//! the next `cell.build()`.
//!
//! # Op enumeration order is observable (17-CONTEXT §3.6)
//!
//! `stars_ops`, `stars_ops_bz` and upstream's pinned `finger(kpts_ibz)`
//! values all depend on the order [`crate::geom::search_point_group_ops`]
//! appends survivors in. Nothing in this module re-sorts an op list.

use std::sync::{Arc, Mutex, OnceLock};

use num_complex::Complex64;
use rayon::prelude::*;

use pyscf_pbc_gto::Cell;
use pyscf_pbc_lib::kpts_helper::{KPT_DIFF_TOL, round_to_fbz};

use crate::error::PbcSymmError;
use crate::group::{PgElement, PointGroup, Representation};
use crate::space_group::SPGElement;
use crate::symmetry::{DmatSet, Symmetry};

/// Sentinel used by upstream's `bz2bz_ks` for "this op maps this k-point
/// outside the mesh". Upstream stores it as `-1` in an integer array; this
/// port keeps the same `i64` representation rather than an `Option<usize>`
/// so the `-1`-indexing tricks of `make_kpts_ibz` (`kpts.py:65-74`) port
/// literally.
pub const NO_MAP: i64 = -1;

// ---------------------------------------------------------------------
// Task 1 — map_k_points_fast / map_kpts_tuples (kpts.py:305-373)
// ---------------------------------------------------------------------

/// `kpts.py:305-325` — `map_k_points_fast`. `bz2bz_ks[k1][s] == k2` iff
/// `ops[s] @ kpts_scaled[k1] == kpts_scaled[k2] + K` for a reciprocal
/// lattice vector `K`; [`NO_MAP`] where there is no such `k2`.
///
/// This routine is modified from GPAW (upstream's own note). The rotations
/// are given in the RECIPROCAL-lattice basis (`op.a2b(cell).rot`), so they
/// are integer-valued and act directly on scaled k-points.
///
/// Reuses [`pyscf_pbc_lib::kpts_helper::round_to_fbz`] — the `wrap_around =
/// false` fold to `[0, 1)`, exactly upstream's default (`kpts.py:355`).
pub fn map_k_points_fast(
    kpts_scaled: &[[f64; 3]],
    ops: &[[[f64; 3]; 3]],
    tol: f64,
) -> Vec<Vec<i64>> {
    let nkpts = kpts_scaled.len();
    let nop = ops.len();

    // Each op owns one COLUMN of `bz2bz_ks` and touches nothing else, so the
    // op loop parallelises with no accumulation ordering to protect. This is
    // where the fold's wall clock actually goes: one `round_to_fbz`
    // (an argsort of `2 * nkpts` values per coordinate) plus one lexsort per
    // op — at Gate A's `nkpts = 4096`, `nop = 96` that dominates the
    // `O(nkpts x nop)` star search by an order of magnitude. Columns are
    // collected in op order, so the result is independent of the worker count.
    let columns: Vec<Vec<i64>> = ops
        .par_iter()
        .map(|op| {
            // op_kpts_scaled = np.einsum('kix,xy->kiy', kpts_scaled, op.T)
            //               == op @ k  for each k (see `space_group::matvec3`).
            let mut k_opk: Vec<[f64; 3]> = Vec::with_capacity(2 * nkpts);
            k_opk.extend_from_slice(kpts_scaled);
            for k in kpts_scaled {
                k_opk.push(std::array::from_fn(|i| {
                    op[i][0] * k[0] + op[i][1] * k[1] + op[i][2] * k[2]
                }));
            }
            // k_opk = round_to_fbz(k_opk, tol=tol) — cleanse runs over the
            // WHOLE concatenated array, which is what lets the original and
            // the rotated copy land on bit-identical values.
            let k_opk = round_to_fbz(&k_opk, false, tol);
            let mut col = vec![NO_MAP; nkpts];
            for (k2, k1) in equivalent_pairs_width3(&k_opk, nkpts) {
                col[k2] = k1 as i64;
            }
            col
        })
        .collect();

    // Transpose to upstream's `(nkpts, nop)` layout.
    let mut bz2bz_ks = vec![vec![NO_MAP; nop]; nkpts];
    for (s, col) in columns.iter().enumerate() {
        for (k, &v) in col.iter().enumerate() {
            bz2bz_ks[k][s] = v;
        }
    }
    bz2bz_ks
}

/// The `lexsort` + `diff == 0` + `maps` block of `map_kpts_tuples`
/// (`kpts.py:363-372`) for the width-3 (`ntuple = 1`) case.
///
/// Yields `(k2 - nkpts, k1)` pairs: `k1 < nkpts` indexes the original block
/// and `k1 + nkpts` … the rotated block, exactly as upstream's two asserts
/// (`:368-369`) require.
fn equivalent_pairs_width3(k_opk: &[[f64; 3]], nkpts: usize) -> Vec<(usize, usize)> {
    // np.lexsort(k_opk.T) sorts by the LAST column first (slowest key is the
    // last row of `k_opk.T`, i.e. column 2 of `k_opk`)... no: lexsort's LAST
    // key is primary. `k_opk.T` has rows = columns of `k_opk`, so the primary
    // key is column 2, then column 1, then column 0.
    let mut order: Vec<usize> = (0..k_opk.len()).collect();
    order.sort_by(|&a, &b| {
        for c in [2usize, 1, 0] {
            match k_opk[a][c].partial_cmp(&k_opk[b][c]) {
                Some(std::cmp::Ordering::Equal) | None => {}
                Some(o) => return o,
            }
        }
        std::cmp::Ordering::Equal
    });
    let mut out = Vec::new();
    for w in order.windows(2) {
        let (a, b) = (w[0], w[1]);
        if k_opk[a] == k_opk[b] {
            debug_assert!(
                a < nkpts,
                "map_k_points_fast: maps[0] must index the original block"
            );
            debug_assert!(
                b >= nkpts,
                "map_k_points_fast: maps[1] must index the rotated block"
            );
            out.push((b - nkpts, a));
        }
    }
    out
}

// ---------------------------------------------------------------------
// Task 1 — make_kpts_ibz (kpts.py:39-114)
// ---------------------------------------------------------------------

/// `kpts.py:39-114` — `make_kpts_ibz`, as a free function taking the
/// `KPoints` it fills (upstream's own note: "This function modifies the
/// `kpts` object").
///
/// # Errors
/// Propagates [`SPGElement::a2b`] (a singular reciprocal lattice, or a
/// rotation that is not integer in the reciprocal basis).
pub fn make_kpts_ibz(kpts: &mut KPoints, cell: &Cell, tol: f64) -> Result<(), PbcSymmError> {
    let nkpts = kpts.nkpts();
    let nop = kpts.nop();

    // op_rot = np.asarray([op.a2b(cell).rot for op in kpts.ops])
    let mut op_rot: Vec<[[f64; 3]; 3]> = Vec::with_capacity(nop);
    for op in kpts.ops() {
        op_rot.push(op.a2b(cell)?.rot);
    }
    // if kpts.time_reversal: op_rot = np.concatenate([op_rot, -op_rot])
    if kpts.time_reversal {
        let neg: Vec<[[f64; 3]; 3]> = op_rot
            .iter()
            .map(|m| std::array::from_fn(|i| std::array::from_fn(|j| -m[i][j])))
            .collect();
        op_rot.extend(neg);
    }
    let nop_tot = op_rot.len();

    let mut bz2bz_ks = map_k_points_fast(&kpts.kpts_scaled, &op_rot, tol);
    // kpts.k2opk = bz2bz_ks.copy()  — taken BEFORE the column wipe below.
    kpts.k2opk = bz2bz_ks.clone();

    // if -1 in bz2bz_ks:
    //     bz2bz_ks[:, np.unique(np.where(bz2bz_ks == -1)[1])] = -1
    let mut col_has_no_map = vec![false; nop_tot];
    for row in &bz2bz_ks {
        for (io, v) in row.iter().enumerate() {
            if *v == NO_MAP {
                col_has_no_map[io] = true;
            }
        }
    }
    if col_has_no_map.iter().any(|b| *b) {
        tracing::warn!("k-points have lower symmetry than lattice.");
        for row in bz2bz_ks.iter_mut() {
            for (io, v) in row.iter_mut().enumerate() {
                if col_has_no_map[io] {
                    *v = NO_MAP;
                }
            }
        }
    }

    // bz2bz_k = -np.ones(nkpts+1, dtype=int)   — the extra slot absorbs the
    // `-1` entries of `bz2bz_ks[k]` (numpy's negative indexing writes there).
    let mut bz2bz_k = vec![NO_MAP; nkpts + 1];
    let mut ibz2bz_k: Vec<usize> = Vec::new();
    for k in (0..nkpts).rev() {
        if bz2bz_k[k] == NO_MAP {
            for &t in &bz2bz_ks[k] {
                let slot = if t == NO_MAP { nkpts } else { t as usize };
                bz2bz_k[slot] = k as i64;
            }
            ibz2bz_k.push(k);
        }
    }
    ibz2bz_k.reverse();
    bz2bz_k.truncate(nkpts);

    // bz2ibz_k[ibz2bz_k] = arange(nkpts_ibz); bz2ibz_k = bz2ibz_k[bz2bz_k]
    let mut ibz_index_of_bz = vec![0usize; nkpts];
    for (i, &k) in ibz2bz_k.iter().enumerate() {
        ibz_index_of_bz[k] = i;
    }
    let bz2ibz_k: Vec<usize> = bz2bz_k
        .iter()
        .map(|&b| {
            debug_assert!(b >= 0, "make_kpts_ibz: bz2bz_k is not total");
            ibz_index_of_bz[b as usize]
        })
        .collect();

    kpts.bz2ibz = bz2ibz_k;
    kpts.ibz2bz = ibz2bz_k;
    // weights_ibz = np.bincount(bz2ibz_k) * (1.0/nkpts)
    let nkpts_ibz = kpts.ibz2bz.len();
    let mut counts = vec![0usize; nkpts_ibz];
    for &i in &kpts.bz2ibz {
        counts[i] += 1;
    }
    kpts.weights_ibz = counts.iter().map(|&c| c as f64 / nkpts as f64).collect();
    kpts.kpts_scaled_ibz = kpts.ibz2bz.iter().map(|&k| kpts.kpts_scaled[k]).collect();
    // S-09 (optimisation session 3): the IBZ points ARE full-BZ points
    // (`kpts_ibz[i] == kpts[ibz2bz[i]]` by definition), so take them as
    // copies. Upstream (`kpts.py:74`) re-derives them through
    // `cell.get_abs_kpts(kpts_scaled_ibz)` — an abs→scaled→abs round trip
    // that is not a bitwise identity, so the IBZ list was never a BITWISE
    // subset of the sampling list, and the two band-table reuses
    // (`KNumInt::band_subset_map`, `Fftdf::ao_kpts`) refused on every
    // k-symmetric driver — measured: 4 cold AO tables per ksymm SCF instead
    // of 2. The copy moves each IBZ k-vector by at most a few ulps; every
    // ksymm gate is 1e-11 or looser (GATE C) or a same-binary bit-identity,
    // and the star/little-group bookkeeping below uses the SCALED points.
    // `PYSCF_PBC_KPTS_IBZ_ROUNDTRIP=1` restores upstream's derivation.
    let roundtrip = std::env::var("PYSCF_PBC_KPTS_IBZ_ROUNDTRIP").is_ok_and(|v| v == "1");
    kpts.kpts_ibz = if roundtrip {
        cell.get_abs_kpts(&kpts.kpts_scaled_ibz)
            .map_err(PbcSymmError::Core)?
    } else {
        kpts.ibz2bz.iter().map(|&k| kpts.kpts[k]).collect()
    };
    kpts.set_nkpts_ibz(kpts.kpts_ibz.len());

    // ---- the star-op search (kpts.py:83-99) -------------------------
    // O(nkpts x nop) and every BZ k-point is independent of every other:
    // parallelise the OUTER loop (17-05-PLAN.md Task 1 "Speed"). There is no
    // accumulation here, so no `oracle_sum` ordering to protect — unlike
    // Task 4's density sum.
    let kpts_scaled = &kpts.kpts_scaled;
    let kpts_scaled_ibz = &kpts.kpts_scaled_ibz;
    let bz2ibz = &kpts.bz2ibz;
    let found: Vec<(usize, usize)> = (0..nkpts)
        .into_par_iter()
        .map(|k| {
            let bz_k_scaled = kpts_scaled[k];
            let ibz_k_scaled = kpts_scaled_ibz[bz2ibz[k]];
            for (io, op) in op_rot.iter().enumerate() {
                if col_has_no_map[io] {
                    // This rotation is not in the subgroup that the k-mesh
                    // belongs to; only happens when the k-mesh has lower
                    // symmetry than the lattice.
                    continue;
                }
                // diff = bz_k_scaled - np.dot(ibz_k_scaled, op.T)
                let mut ok = true;
                for i in 0..3 {
                    let rot = op[i][0] * ibz_k_scaled[0]
                        + op[i][1] * ibz_k_scaled[1]
                        + op[i][2] * ibz_k_scaled[2];
                    let d = bz_k_scaled[i] - rot;
                    // diff = diff - diff.round()  (NumPy rounds halves to even)
                    let d = d - d.round_ties_even();
                    if d.abs() >= tol {
                        ok = false;
                        break;
                    }
                }
                if ok {
                    return (io / nop, io % nop);
                }
            }
            (0, 0)
        })
        .collect();
    for (k, (tr, iop)) in found.into_iter().enumerate() {
        kpts.time_reversal_symm_bz[k] = tr;
        kpts.stars_ops_bz[k] = iop;
    }

    // ---- stars (kpts.py:101-104) ------------------------------------
    kpts.stars = vec![Vec::new(); nkpts_ibz];
    for (k, &i) in kpts.bz2ibz.iter().enumerate() {
        kpts.stars[i].push(k);
    }
    kpts.stars_ops = kpts
        .stars
        .iter()
        .map(|star| star.iter().map(|&k| kpts.stars_ops_bz[k]).collect())
        .collect();

    // ---- little co-group ops (kpts.py:106-113) ----------------------
    let mut little_cogroup_ops = Vec::with_capacity(nkpts_ibz);
    for ki_ibz in 0..nkpts_ibz {
        let ki = kpts.ibz2bz[ki_ibz];
        let ops_id: Vec<usize> = kpts.k2opk[ki]
            .iter()
            .enumerate()
            .filter(|&(_, &v)| v == ki as i64)
            .map(|(i, _)| i)
            .collect();
        little_cogroup_ops.push(ops_id);
    }
    kpts.little_cogroup_ops = little_cogroup_ops;

    Ok(())
}

// ---------------------------------------------------------------------
// KPoints
// ---------------------------------------------------------------------

/// `kpts.py:847-1123` — `KPoints`. See the module doc for the two structural
/// deviations from upstream (composition instead of inheritance; no stored
/// `Cell`).
#[derive(Debug, Clone, Default)]
pub struct KPoints {
    /// D-PBC-25 point 2 — composition, not inheritance.
    pub symmetry: Symmetry,
    /// `kpts.py:980` — whether time-reversal symmetry is folded in. Set by
    /// [`KPoints::build`] to `time_reversal_symmetry && !has_inversion`.
    pub time_reversal: bool,

    /// Absolute k-points in the full BZ, 1/Bohr.
    pub kpts: Vec<[f64; 3]>,
    /// Scaled (fractional) k-points in the full BZ.
    pub kpts_scaled: Vec<[f64; 3]>,
    /// `1/nkpts` for every full-BZ k-point.
    pub weights: Vec<f64>,

    /// Absolute k-points in the IBZ.
    pub kpts_ibz: Vec<[f64; 3]>,
    /// Scaled k-points in the IBZ.
    pub kpts_scaled_ibz: Vec<[f64; 3]>,
    /// IBZ weights; `weights_ibz[i] == stars[i].len() / nkpts` and they sum
    /// to 1. **17-CONTEXT §3.5: this, not `1/nkpts`, is what `energy_elec`
    /// weights by.**
    pub weights_ibz: Vec<f64>,

    /// `ibz2bz[i]` — the full-BZ index of IBZ point `i`.
    pub ibz2bz: Vec<usize>,
    /// `bz2ibz[k]` — the IBZ index of full-BZ point `k`. Total by
    /// construction.
    pub bz2ibz: Vec<usize>,

    /// `k2opk[k][s]` — the full-BZ index of `ops[s] @ kpts[k]`, or
    /// [`NO_MAP`]. `nop * (time_reversal + 1)` columns. Taken BEFORE
    /// `make_kpts_ibz`'s column wipe, so it may retain partial `-1`s.
    pub k2opk: Vec<Vec<i64>>,

    /// `stars[i]` — the full-BZ indices symmetry-equivalent to IBZ point `i`.
    pub stars: Vec<Vec<usize>>,
    /// `stars_ops[i][j]` — the op index taking `kpts_ibz[i]` to
    /// `kpts[stars[i][j]]`. Always equals `stars_ops_bz[stars[i][j]]`.
    pub stars_ops: Vec<Vec<usize>>,
    /// `stars_ops_bz[k]` — the same op index, keyed by full-BZ index.
    pub stars_ops_bz: Vec<usize>,
    /// `time_reversal_symm_bz[k]` — 1 when `k` is reached from its IBZ
    /// representative only with an additional time reversal.
    pub time_reversal_symm_bz: Vec<usize>,
    /// `little_cogroup_ops[i]` — indices into `k2opk`'s column space of the
    /// ops that leave `kpts_ibz[i]` invariant. Consumed by 17-04's
    /// [`crate::basis::symm_adapted_basis`].
    pub little_cogroup_ops: Vec<Vec<usize>>,

    _nkpts: usize,
    _nkpts_ibz: usize,

    /// Lazily built caches (`kpts.py:1005-1009`'s five private fields).
    /// `OnceLock`, not `OnceCell`, so `&KPoints` stays `Sync` and the
    /// rayon loops of Tasks 1/3/4 can borrow it.
    pub(crate) addition_table: OnceLock<Vec<i32>>,
    pub(crate) inverse_table: OnceLock<Vec<i32>>,
    #[allow(clippy::type_complexity)]
    pub(crate) copgs: OnceLock<(Vec<PointGroup>, Vec<Vec<usize>>)>,
    /// S-06: source-grid permutations, keyed by `(IBZ index, mesh)`.  Clones
    /// share the lazily populated cache; entries are immutable once inserted.
    grid_permutations:
        Arc<OnceLock<Mutex<std::collections::HashMap<(usize, [usize; 3]), Arc<Vec<u32>>>>>>,
}

impl KPoints {
    /// `kpts.py:975-1009` — `__init__`. `kpts` are ABSOLUTE k-points
    /// (1/Bohr) in the full BZ.
    pub fn new(kpts: Vec<[f64; 3]>) -> Self {
        let nkpts = kpts.len();
        Self {
            symmetry: Symmetry::default(),
            time_reversal: false,
            kpts_ibz: kpts.clone(),
            kpts,
            kpts_scaled_ibz: Vec::new(),
            kpts_scaled: Vec::new(),
            weights: vec![1.0 / nkpts as f64; nkpts],
            weights_ibz: vec![1.0 / nkpts as f64; nkpts],
            ibz2bz: (0..nkpts).collect(),
            bz2ibz: (0..nkpts).collect(),
            k2opk: Vec::new(),
            stars: Vec::new(),
            stars_ops: Vec::new(),
            stars_ops_bz: vec![0; nkpts],
            time_reversal_symm_bz: vec![0; nkpts],
            little_cogroup_ops: Vec::new(),
            _nkpts: nkpts,
            _nkpts_ibz: nkpts,
            addition_table: OnceLock::new(),
            inverse_table: OnceLock::new(),
            copgs: OnceLock::new(),
            grid_permutations: Arc::new(OnceLock::new()),
        }
    }

    /// `kpts.py:1044-1047` — `nkpts`.
    pub fn nkpts(&self) -> usize {
        self._nkpts
    }

    /// `kpts.py:1052-1055` — `nkpts_ibz`.
    pub fn nkpts_ibz(&self) -> usize {
        self._nkpts_ibz
    }

    pub(crate) fn set_nkpts_ibz(&mut self, n: usize) {
        self._nkpts_ibz = n;
    }

    /// The number of space-group operations — `Symmetry.nop`, re-exposed so
    /// callers do not reach through `.symmetry` (D-PBC-25 point 2).
    pub fn nop(&self) -> usize {
        self.symmetry.nop
    }

    /// The space-group operations — `Symmetry.ops`, re-exposed.
    pub fn ops(&self) -> &[SPGElement] {
        &self.symmetry.ops
    }

    /// The Wigner-D matrices — `Symmetry.dmats`, re-exposed.
    pub fn dmats(&self) -> &[DmatSet] {
        &self.symmetry.dmats
    }

    /// `Symmetry.has_inversion`, re-exposed.
    pub fn has_inversion(&self) -> bool {
        self.symmetry.has_inversion
    }

    /// `kpts.py:1017-1033` — `build`.
    ///
    /// # Errors
    /// Propagates [`Symmetry::build`] and [`make_kpts_ibz`].
    pub fn build(
        &mut self,
        cell: &Cell,
        space_group_symmetry: bool,
        time_reversal_symmetry: bool,
        symmorphic: bool,
        check_mesh_symmetry: bool,
    ) -> Result<(), PbcSymmError> {
        // kpts.py:1018-1021 — if the cell already carries a lattice
        // symmetry, adopt it rather than re-running the space-group search
        // (upstream: `self.__dict__.update(_lattice_symm.__dict__)`). This
        // also inherits the `check_mesh_symmetry` decision `Cell.build` made
        // (`cell.py:1771-1772`).
        if let (true, Some(ls)) = (space_group_symmetry, cell.lattice_symmetry.as_ref()) {
            self.symmetry = Symmetry::from_lattice_symmetry(ls);
        }
        if !self.symmetry.built {
            self.symmetry =
                Symmetry::build(cell, space_group_symmetry, symmorphic, check_mesh_symmetry)?;
        }
        // self.time_reversal = time_reversal_symmetry and not self.has_inversion
        self.time_reversal = time_reversal_symmetry && !self.has_inversion();
        self.kpts_scaled = cell.get_scaled_kpts(&self.kpts);
        self.kpts_scaled_ibz = self.kpts_scaled.clone();
        make_kpts_ibz(self, cell, KPT_DIFF_TOL)?;
        self.dump_info();
        Ok(())
    }

    /// `kpts.py:1035-1042` — `dump_info`, at `tracing::info!`.
    pub fn dump_info(&self) {
        tracing::info!(time_reversal = self.time_reversal, "time reversal");
        for k in 0..self.nkpts_ibz() {
            let kk = self.kpts_scaled_ibz[k];
            tracing::info!(
                "{k:3}: {:9.6}, {:9.6}, {:9.6}    {}/{}",
                kk[0],
                kk[1],
                kk[2],
                (self.weights_ibz[k] * self.nkpts() as f64).floor(),
                self.nkpts()
            );
        }
    }
}

/// `kpts.py:804-845` — `make_kpts(cell, kpts, space_group_symmetry,
/// time_reversal_symmetry)`, the wrapper that builds a [`KPoints`].
///
/// **This is where `cell.make_kpts(..., space_group_symmetry=True)` moved
/// to** (17-05-PLAN.md Task 6). `Cell` lives in `pyscf-pbc-gto`, which sits
/// below this crate, so it cannot return a `KPoints` without inverting
/// D-PBC-25.
///
/// `symmorphic` and `check_mesh_symmetry` come off the `Cell`
/// (`cell.symmorphic`; upstream's `Symmetry.build` default for the mesh
/// check is `True`).
///
/// # Errors
/// As [`KPoints::build`].
pub fn make_kpts(
    cell: &Cell,
    kpts: &[[f64; 3]],
    space_group_symmetry: bool,
    time_reversal_symmetry: bool,
) -> Result<KPoints, PbcSymmError> {
    let mut kpts_symm = KPoints::new(kpts.to_vec());
    kpts_symm.build(
        cell,
        space_group_symmetry,
        time_reversal_symmetry,
        cell.symmorphic,
        true,
    )?;
    Ok(kpts_symm)
}

// ---------------------------------------------------------------------
// Task 5 — the k-tuple machinery (kpts.py:1017-1123, :116-303, :1174-1223)
// ---------------------------------------------------------------------

/// The return of [`KPoints::make_ktuples_ibz`] — upstream's six-tuple
/// (`kpts.py:116-199`).
#[derive(Debug, Clone)]
pub struct KtuplesIbz {
    /// `ibz2bz[i]` — the flat tuple index of IBZ tuple `i`.
    pub ibz2bz: Vec<usize>,
    /// `weight_ibz[i] == |stars[i]| / nkpts^ntuple`; sums to 1.
    pub weight_ibz: Vec<f64>,
    /// `bz2ibz[t]` — the IBZ tuple index of flat tuple `t`.
    pub bz2ibz: Vec<usize>,
    /// `stars[i]` — the flat tuple indices equivalent to IBZ tuple `i`,
    /// ASCENDING (upstream's `np.unique`).
    pub stars: Vec<Vec<usize>>,
    /// `stars_ops[i][j]` — the op index (a column of `k2opk`) taking IBZ
    /// tuple `i` to `stars[i][j]`.
    pub stars_ops: Vec<Vec<usize>>,
    /// The same, keyed by flat tuple index.
    pub stars_ops_bz: Vec<usize>,
}

/// The return of [`KPoints::make_k4_ibz`] with `sym = "s1"`
/// (`kpts.py:205-217`).
#[derive(Debug, Clone)]
pub struct K4Ibz {
    /// `k4[i] = [ki, kj, ka, kb]` — physicist's notation, `kb` from
    /// momentum conservation.
    pub k4: Vec<[usize; 4]>,
    pub weight: Vec<f64>,
    pub bz2ibz: Vec<usize>,
    pub ibz2bz: Vec<usize>,
    pub stars_ops: Vec<Vec<usize>>,
    pub stars_ops_bz: Vec<usize>,
}

impl KPoints {
    fn density_grid_permutation(
        &self,
        ibz_k_idx: usize,
        mesh: [usize; 3],
    ) -> Result<Arc<Vec<u32>>, PbcSymmError> {
        let cache = self
            .grid_permutations
            .get_or_init(|| Mutex::new(std::collections::HashMap::new()));
        if let Some(found) = cache
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .get(&(ibz_k_idx, mesh))
            .cloned()
        {
            return Ok(found);
        }
        let ngrids = mesh[0] * mesh[1] * mesh[2];
        assert!(
            u32::try_from(ngrids).is_ok(),
            "symmetry grid exceeds u32 index range"
        );
        let ops = star_grid_ops(self, ibz_k_idx, mesh)?;
        let mut permutation = Vec::with_capacity(ops.len() * ngrids);
        for entry in &ops {
            for g in 0..ngrids {
                let src = match entry {
                    None => g,
                    Some((rot, ft)) => rotated_grid_index(rot, *ft, mesh, grid_xyz(g, mesh)),
                };
                permutation.push(src as u32);
            }
        }
        let permutation = Arc::new(permutation);
        cache
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .insert((ibz_k_idx, mesh), permutation.clone());
        Ok(permutation)
    }

    /// Number of lazily materialized `(IBZ point, mesh)` grid permutations.
    ///
    /// This is intentionally a narrow diagnostic surface for profilers and
    /// integration tests; density consumers never need to inspect the cached
    /// index arrays themselves.
    pub fn density_grid_cache_len(&self) -> usize {
        self.grid_permutations
            .get()
            .map(|cache| {
                cache
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .len()
            })
            .unwrap_or(0)
    }

    // -----------------------------------------------------------------
    // index <-> tuple (kpts.py:1035-1046)
    // -----------------------------------------------------------------

    /// `kpts.py:1043-1044` — `ktuple_to_index`, i.e.
    /// `lib.inv_base_repr_int(kk, nkpts)`: read `kk` as a base-`nkpts`
    /// numeral, MOST significant digit first.
    pub fn ktuple_to_index(&self, kk: &[usize]) -> usize {
        let base = self.nkpts();
        let nd = kk.len();
        kk.iter()
            .enumerate()
            .map(|(i, &d)| d * base.pow((nd - i - 1) as u32))
            .sum()
    }

    /// `kpts.py:1046-1047` — `index_to_ktuple`, i.e.
    /// `lib.base_repr_int(k, nkpts, ntuple)`, zero-padded on the LEFT to
    /// `ntuple` digits.
    pub fn index_to_ktuple(&self, k: usize, ntuple: usize) -> Vec<usize> {
        let base = self.nkpts();
        let mut out = vec![0usize; ntuple];
        let mut num = k;
        for i in (0..ntuple).rev() {
            let p = base.pow(i as u32);
            out[ntuple - 1 - i] = num / p;
            num -= out[ntuple - 1 - i] * p;
        }
        out
    }

    /// `kpts.py:1035-1041` — `loop_ktuples`.
    pub fn loop_ktuples<'a>(
        &'a self,
        ibz2bz: &'a [usize],
        ntuple: usize,
    ) -> impl Iterator<Item = Vec<usize>> + 'a {
        ibz2bz.iter().map(move |&k| self.index_to_ktuple(k, ntuple))
    }

    // -----------------------------------------------------------------
    // addition / inverse tables (kpts.py:1049-1074)
    // -----------------------------------------------------------------

    /// `kpts.py:1049-1063` — `addition_table`. `table[i * nkpts + j]` is the
    /// index of `k_i + k_j` modulo a reciprocal lattice vector. Lazily built
    /// and cached.
    ///
    /// **Deviation, recorded:** upstream materialises the full
    /// `(nkpts, nkpts, nkpts, 3)` difference tensor (`:1052-1053`), which is
    /// `O(nkpts^3)` memory — 400 GB at Gate A's `nkpts = 4096`. This port
    /// computes the same table one ROW at a time: fold
    /// `[kpts_scaled ; k_i + kpts_scaled]` with
    /// [`pyscf_pbc_lib::kpts_helper::round_to_fbz`] and match the two halves
    /// by exact equality — the identical construction
    /// [`map_k_points_fast`] already relies on, and `O(nkpts)` memory. The
    /// only observable difference is which set of values `cleanse` clusters
    /// together, and folding each row alongside the reference set is the
    /// stronger grouping.
    ///
    /// # Panics
    /// If the k-mesh is not closed under addition (upstream's
    /// `assert (table > -1).all()`, `:1060`).
    pub fn addition_table(&self) -> &[i32] {
        self.addition_table.get_or_init(|| {
            let nk = self.nkpts();
            let mut table = vec![-1i32; nk * nk];
            for i in 0..nk {
                let ki = self.kpts_scaled[i];
                let mut rows: Vec<[f64; 3]> = Vec::with_capacity(2 * nk);
                rows.extend_from_slice(&self.kpts_scaled);
                for kj in &self.kpts_scaled {
                    rows.push([ki[0] + kj[0], ki[1] + kj[1], ki[2] + kj[2]]);
                }
                let folded = round_to_fbz(&rows, false, KPT_DIFF_TOL);
                let lookup: std::collections::HashMap<[u64; 3], usize> =
                    (0..nk).map(|m| (bits3(&folded[m]), m)).collect();
                for j in 0..nk {
                    let m = lookup.get(&bits3(&folded[nk + j])).copied();
                    table[i * nk + j] =
                        m.expect("addition_table: k-mesh is not closed under addition") as i32;
                }
            }
            table
        })
    }

    /// `kpts.py:1065-1074` — `inverse_table`. `table[i]` is the index of
    /// `-k_i` modulo a reciprocal lattice vector. Lazily built and cached.
    ///
    /// # Panics
    /// If the k-mesh is not closed under negation (upstream's
    /// `assert (table > -1).all()`, `:1073`).
    pub fn inverse_table(&self) -> &[i32] {
        self.inverse_table.get_or_init(|| {
            let nk = self.nkpts();
            let mut rows: Vec<[f64; 3]> = Vec::with_capacity(2 * nk);
            rows.extend_from_slice(&self.kpts_scaled);
            for k in &self.kpts_scaled {
                rows.push([-k[0], -k[1], -k[2]]);
            }
            let folded = round_to_fbz(&rows, false, KPT_DIFF_TOL);
            let lookup: std::collections::HashMap<[u64; 3], usize> =
                (0..nk).map(|m| (bits3(&folded[m]), m)).collect();
            (0..nk)
                .map(|i| {
                    lookup
                        .get(&bits3(&folded[nk + i]))
                        .copied()
                        .expect("inverse_table: k-mesh is not closed under negation")
                        as i32
                })
                .collect()
        })
    }

    /// `kpts.py:1076-1082` — `get_kconserv`.
    ///
    /// **Delegates** to the already-shipped
    /// [`pyscf_pbc_lib::kpts_helper::get_kconserv`] (15-CONTEXT §1.1 ruled
    /// that it ships; `kpts_helper.rs:282`). Upstream's own version here is
    /// only a faster route to the SAME table —
    /// `add_tab[add_tab[:, inv_tab], :][i][j][m]` is the index of
    /// `k_i - k_j + k_m`, which is exactly `kconserv[i, j, m]`
    /// (`kpts_helper.py:291-325`). Re-porting it would give the workspace
    /// two `kconserv`s that can drift, which 17-05-PLAN.md Task 5 forbids.
    pub fn get_kconserv(&self, cell: &Cell) -> pyscf_pbc_lib::kpts_helper::Kconserv {
        pyscf_pbc_lib::kpts_helper::get_kconserv(&cell.lattice_vectors(), &self.kpts)
    }

    // -----------------------------------------------------------------
    // make_gdf_kptij_lst_jk (kpts.py:1017-1033)
    // -----------------------------------------------------------------

    /// `kpts.py:1017-1033` — `make_gdf_kptij_lst_jk`: the GDF k-point-pair
    /// list for `get_jk`. All `(k, k)` diagonal pairs, then every
    /// `(k_ibz, k_bz)` pair whose `k_bz` is not already the `k_ibz` itself.
    pub fn make_gdf_kptij_lst_jk(&self) -> Vec<([f64; 3], [f64; 3])> {
        let mut kptij_lst: Vec<([f64; 3], [f64; 3])> = (0..self.nkpts())
            .map(|i| (self.kpts[i], self.kpts[i]))
            .collect();
        for i in 0..self.nkpts_ibz() {
            let ki = self.kpts_ibz[i];
            let where_ = pyscf_pbc_lib::kpts_helper::member(&ki, &self.kpts);
            for j in 0..self.nkpts() {
                if !where_.contains(&j) {
                    kptij_lst.push((ki, self.kpts[j]));
                }
            }
        }
        kptij_lst
    }

    // -----------------------------------------------------------------
    // little co-groups (kpts.py:1084-1108)
    // -----------------------------------------------------------------

    /// `kpts.py:1084-1100` — `little_cogroups`. Returns
    /// `(copgs, indices)`: the little co-group of EVERY full-BZ k-point
    /// (conjugated from its IBZ representative's by `stars_ops_bz[ki]`), and
    /// the permutation `indices[ki]` that sorts the conjugated elements.
    ///
    /// This is what 17-04's [`crate::basis::symm_adapted_basis`] consumes;
    /// it takes `little_cogroups` / `little_cogroup_rep` as INPUT parameters
    /// precisely so this call ordering works (they come AFTER `basis.py` in
    /// upstream's file order but BEFORE it in the call order).
    ///
    /// # Errors
    /// * [`PbcSymmError::KptsSymmInputMismatch`] if a `little_cogroup_ops`
    ///   entry indexes past `nop`. That is reachable only with
    ///   `time_reversal = true`, where `k2opk` has `2 * nop` columns while
    ///   `ops` has `nop` entries — **upstream raises `IndexError` on the
    ///   same input** (`kpts.py:1091`: `self.ops[i] for i in ops_ibz`).
    ///   Recorded rather than silently folded with `i % nop`, which would
    ///   invent a different group.
    /// * [`PbcSymmError::NotAGroup`] if the conjugated elements do not form
    ///   a group.
    pub fn little_cogroups(&self) -> Result<(Vec<PointGroup>, Vec<Vec<usize>>), PbcSymmError> {
        if let Some(cached) = self.copgs.get() {
            return Ok(cached.clone());
        }
        let nop = self.nop();
        let mut copgs = Vec::with_capacity(self.nkpts());
        let mut indices = Vec::with_capacity(self.nkpts());
        for ki in 0..self.nkpts() {
            let ki_ibz = self.bz2ibz[ki];
            let ops_ibz = &self.little_cogroup_ops[ki_ibz];
            for &i in ops_ibz {
                if i >= nop {
                    return Err(PbcSymmError::KptsSymmInputMismatch(format!(
                        "little_cogroups: little_cogroup_ops[{ki_ibz}] contains op index {i} \
                         but there are only {nop} space-group operations (this happens only \
                         with time_reversal = true, where k2opk has 2*nop columns; upstream \
                         raises IndexError here, kpts.py:1091)"
                    )));
                }
            }
            // elements = np.sort([PGElement(self.ops[i].rot) for i in ops_ibz])
            let mut elements: Vec<PgElement> = ops_ibz
                .iter()
                .map(|&i| pg_element(&self.ops()[i]))
                .collect();
            elements.sort();
            // op_i = PGElement(self.ops[stars_ops_bz[ki]].rot)
            let op_i = pg_element(&self.ops()[self.stars_ops_bz[ki]]);
            let op_i_inv = op_i.inv();
            // elements_i = [op_i @ g @ op_i.inv() for g in elements]
            let elements_i: Vec<PgElement> = elements
                .iter()
                .map(|g| op_i.compose(g).compose(&op_i_inv))
                .collect();
            // idx = np.argsort(elements_i)
            let mut idx: Vec<usize> = (0..elements_i.len()).collect();
            idx.sort_by(|&a, &b| elements_i[a].cmp(&elements_i[b]));
            let sorted: Vec<PgElement> = idx.iter().map(|&i| elements_i[i]).collect();
            indices.push(idx);
            copgs.push(PointGroup::new(sorted)?);
        }
        let out = (copgs, indices);
        let _ = self.copgs.set(out.clone());
        Ok(out)
    }

    /// `kpts.py:1102-1108` — `little_cogroup_rep`: the representation of
    /// irrep `ir` of IBZ point `ki`'s little co-group, transported to the
    /// full-BZ point `ki`.
    ///
    /// # Errors
    /// As [`KPoints::little_cogroups`], plus
    /// [`crate::group::Representation::from_chi`].
    pub fn little_cogroup_rep(&self, ki: usize, ir: usize) -> Result<Representation, PbcSymmError> {
        let (copgs, indices) = self.little_cogroups()?;
        let ki_ibz = self.bz2ibz[ki];
        let pg_ibz = &copgs[self.ibz2bz[ki_ibz]];
        let chi = pg_ibz.get_irrep_chi(ir);
        let chi_ki: Vec<crate::group::Complex64> = indices[ki].iter().map(|&m| chi[m]).collect();
        Representation::from_chi(copgs[ki].clone(), chi_ki)
    }

    // -----------------------------------------------------------------
    // make_ktuples_ibz / make_k4_ibz (kpts.py:116-217)
    // -----------------------------------------------------------------

    /// `kpts.py:116-199` — `make_ktuples_ibz`, the `kpts_scaled is None`
    /// branch (`:150-163`): symmetry relations among ALL `nkpts^ntuple`
    /// k-point tuples, derived from the already-built `k2opk`.
    ///
    /// The `kpts_scaled is not None` branch (`:143-149`) needs
    /// `map_kpts_tuples` at `ntuple > 1`; nothing in this workspace reaches
    /// it (`make_k4_ibz` calls this with `kpts_scaled = None`), so it is not
    /// ported — see the module SUMMARY.
    ///
    /// # Panics
    /// If `nkpts^ntuple` overflows `usize`.
    pub fn make_ktuples_ibz(&self, ntuple: usize) -> KtuplesIbz {
        let nkpts = self.nkpts();
        let nop = self.k2opk.first().map_or(0, |r| r.len());
        let nktuple = nkpts
            .checked_pow(ntuple as u32)
            .expect("nkpts^ntuple overflows usize");

        // kt2opkt[t][iop] — the flat index of op(t), or NO_MAP.
        let mut kt2opkt = vec![vec![NO_MAP; nop]; nktuple];
        for iop in 0..nop {
            let col: Vec<i64> = (0..nkpts).map(|k| self.k2opk[k][iop]).collect();
            if col.contains(&NO_MAP) {
                continue; // the whole column stays NO_MAP
            }
            // tmp = lib.cartesian_prod([col] * ntuple); then
            // inv_base_repr_int(tmp, nkpts) — done in one pass, LAST axis
            // varying fastest (C order), so the flat index `t` decomposes
            // exactly as `index_to_ktuple` does.
            for (t, slot) in kt2opkt.iter_mut().enumerate() {
                let digits = self.index_to_ktuple(t, ntuple);
                let mut out = 0usize;
                for (i, &d) in digits.iter().enumerate() {
                    out += (col[d] as usize) * nkpts.pow((ntuple - i - 1) as u32);
                }
                slot[iop] = out as i64;
            }
        }

        let mut bz2bz = vec![NO_MAP; nktuple + 1];
        let mut ibz2bz: Vec<usize> = Vec::new();
        let mut stars: Vec<Vec<usize>> = Vec::new();
        let mut stars_ops: Vec<Vec<usize>> = Vec::new();
        let mut stars_ops_bz = vec![0usize; nktuple];
        for k in (0..nktuple).rev() {
            if bz2bz[k] != NO_MAP {
                continue;
            }
            for &t in &kt2opkt[k] {
                let slot = if t == NO_MAP { nktuple } else { t as usize };
                bz2bz[slot] = k as i64;
            }
            // np.unique(kt2opkt[k], return_index=True) — sorted unique values
            // with the index of each one's FIRST occurrence (numpy uses a
            // stable sort whenever return_index is set).
            let mut seen: std::collections::BTreeMap<i64, usize> =
                std::collections::BTreeMap::new();
            for (iop, &t) in kt2opkt[k].iter().enumerate() {
                seen.entry(t).or_insert(iop);
            }
            let mut k_idx: Vec<usize> = Vec::new();
            let mut op_idx: Vec<usize> = Vec::new();
            for (&t, &iop) in seen.iter() {
                if t == NO_MAP {
                    continue; // upstream drops the leading -1 (`:180-182`)
                }
                k_idx.push(t as usize);
                op_idx.push(iop);
            }
            for (&t, &iop) in k_idx.iter().zip(op_idx.iter()) {
                stars_ops_bz[t] = iop;
            }
            stars.push(k_idx);
            stars_ops.push(op_idx);
            ibz2bz.push(k);
        }
        ibz2bz.reverse();
        stars.reverse();
        stars_ops.reverse();
        bz2bz.truncate(nktuple);

        let mut ibz_index_of = vec![0usize; nktuple];
        for (i, &t) in ibz2bz.iter().enumerate() {
            ibz_index_of[t] = i;
        }
        let bz2ibz: Vec<usize> = bz2bz.iter().map(|&b| ibz_index_of[b as usize]).collect();
        let mut counts = vec![0usize; ibz2bz.len()];
        for &i in &bz2ibz {
            counts[i] += 1;
        }
        let weight_ibz = counts.iter().map(|&c| c as f64 / nktuple as f64).collect();

        KtuplesIbz {
            ibz2bz,
            weight_ibz,
            bz2ibz,
            stars,
            stars_ops,
            stars_ops_bz,
        }
    }

    /// `kpts.py:205-217` — `make_k4_ibz`, `sym = "s1"` (physicist's
    /// notation). `kb` comes from momentum conservation:
    /// `kb = kconserv[ki, ka, kj]`.
    ///
    /// # Errors
    /// [`PbcSymmError::UnsupportedK4Symmetry`] for `"s2"` / `"s4"`
    /// (`kpts.py:218-300`) — deferred to 17-09 (`kccsd_rhf_ksymm`), their
    /// only consumer — and for anything else (upstream's own
    /// `raise NotImplementedError("Unsupported symmetry.")`, `:301`).
    pub fn make_k4_ibz(&self, cell: &Cell, sym: &str) -> Result<K4Ibz, PbcSymmError> {
        if sym != "s1" {
            return Err(PbcSymmError::UnsupportedK4Symmetry(sym.to_string()));
        }
        let t = self.make_ktuples_ibz(3);
        let kconserv = self.get_kconserv(cell);
        let k4: Vec<[usize; 4]> = t
            .ibz2bz
            .iter()
            .map(|&idx| {
                let kija = self.index_to_ktuple(idx, 3);
                let kb = kconserv.get(kija[0], kija[2], kija[1]) as usize;
                [kija[0], kija[1], kija[2], kb]
            })
            .collect();
        Ok(K4Ibz {
            k4,
            weight: t.weight_ibz,
            bz2ibz: t.bz2ibz,
            ibz2bz: t.ibz2bz,
            stars_ops: t.stars_ops,
            stars_ops_bz: t.stars_ops_bz,
        })
    }
}

/// `PGElement(op.rot)` — the point-group element of a space-group op's
/// rotation. The rotation is always exactly integer in the lattice basis
/// (see [`SPGElement`]'s doc).
fn pg_element(op: &SPGElement) -> PgElement {
    PgElement::new(std::array::from_fn(|i| {
        std::array::from_fn(|j| op.rot[i][j].round() as i32)
    }))
}

/// Exact bit key of a folded scaled k-point. `round_to_fbz` produces
/// bit-identical values for points that are the same, which is what
/// [`map_k_points_fast`] already relies on; `-0.0` is normalised to `0.0`
/// so it hashes with it.
fn bits3(k: &[f64; 3]) -> [u64; 3] {
    std::array::from_fn(|i| (if k[i] == 0.0 { 0.0 } else { k[i] }).to_bits())
}

// ---------------------------------------------------------------------
// Task 5 — KQuartets (kpts.py:1174-1223)
// ---------------------------------------------------------------------

/// `kpts.py:1174-1223` — `KQuartets`: the symmetry relations between
/// k-quartets. Consumed by 17-09 (`kmp2_ksymm`, `kccsd_rhf_ksymm`).
///
/// Upstream holds a reference to its `KPoints`; this port takes one by
/// reference at every call instead, for the same reason [`KPoints`] does not
/// hold a `Cell` (17-CONTEXT §3.9).
#[derive(Debug, Clone)]
pub struct KQuartets {
    pub kqrts_ibz: Vec<[usize; 4]>,
    pub weights_ibz: Vec<f64>,
    pub ibz2bz: Vec<usize>,
    pub bz2ibz: Vec<usize>,
    pub stars_ops: Vec<Vec<usize>>,
    pub stars_ops_bz: Vec<usize>,
    kqrts_stab: Option<Vec<Vec<[usize; 4]>>>,
    ops_stab: Option<Vec<Vec<usize>>>,
}

impl KQuartets {
    /// `kpts.py:1188-1210` — `__init__` + `build`.
    ///
    /// # Errors
    /// As [`KPoints::make_k4_ibz`].
    pub fn build(kpts: &KPoints, cell: &Cell) -> Result<Self, PbcSymmError> {
        let k4 = kpts.make_k4_ibz(cell, "s1")?;
        Ok(Self {
            kqrts_ibz: k4.k4,
            weights_ibz: k4.weight,
            ibz2bz: k4.ibz2bz,
            bz2ibz: k4.bz2ibz,
            stars_ops: k4.stars_ops,
            stars_ops_bz: k4.stars_ops_bz,
            kqrts_stab: None,
            ops_stab: None,
        })
    }

    /// `kpts.py:1212-1222` — `cache_stabilizer`.
    pub fn cache_stabilizer(&mut self, kpts: &KPoints) {
        let mut kqrts_stab = Vec::with_capacity(self.kqrts_ibz.len());
        let mut ops_stab = Vec::with_capacity(self.kqrts_ibz.len());
        for (i, kq) in self.kqrts_ibz.iter().enumerate() {
            let op_group = &self.stars_ops[i];
            // idx = np.where(kpts.k2opk[kq[0], op_group] == kq[0])[0]
            let op_group_small: Vec<usize> = op_group
                .iter()
                .copied()
                .filter(|&iop| kpts.k2opk[kq[0]][iop] == kq[0] as i64)
                .collect();
            // klcd = kpts.k2opk[kq[:,None], op_group_small].T
            let klcd: Vec<[usize; 4]> = op_group_small
                .iter()
                .map(|&iop| std::array::from_fn(|d| kpts.k2opk[kq[d]][iop] as usize))
                .collect();
            kqrts_stab.push(klcd);
            ops_stab.push(op_group_small);
        }
        self.kqrts_stab = Some(kqrts_stab);
        self.ops_stab = Some(ops_stab);
    }

    /// `kpts.py:1224-1227` — `loop_stabilizer`.
    ///
    /// # Panics
    /// If [`KQuartets::cache_stabilizer`] has not been called (upstream
    /// calls it lazily; this port cannot, because it would need `&mut self`
    /// behind an iterator).
    pub fn loop_stabilizer(&self, index: usize) -> impl Iterator<Item = ([usize; 4], usize)> + '_ {
        let k = self
            .kqrts_stab
            .as_ref()
            .expect("call cache_stabilizer first");
        let o = self.ops_stab.as_ref().expect("call cache_stabilizer first");
        k[index].iter().copied().zip(o[index].iter().copied())
    }
}

// ---------------------------------------------------------------------
// Task 3 — the unfolds (kpts.py:449-802)
// ---------------------------------------------------------------------
//
// Each of these delegates the PER-OP work to `crate::symmetry`'s transforms
// (17-03); this layer is the loop over stars plus the phase bookkeeping.
// There is exactly one AO-rotation assembly in this crate
// (`symmetry::get_rotation_mat`) and everything here goes through it —
// 17-CONTEXT §3.2.
//
// LAYOUT: every complex matrix here is ROW-MAJOR, matching
// `crate::symmetry`'s transforms. `mo_coeff` as stored by
// `pyscf-pbc-scf` is COLUMN-MAJOR (`types.rs:119`); converting is the
// CALLER's job, done once at the boundary, exactly as
// `tests/symmetry.rs` already does. Mixing the two is the 14-05 defect
// shape (17-CONTEXT §3.2), so it is stated here rather than left implicit.
//
// SPEED: each unfold writes one BZ k-point per output slot and every slot
// belongs to exactly one star, so the outer loop is a rayon `par_iter`
// `map`/`collect` — disjoint by construction, with no reduction and hence
// no `oracle_sum` ordering to protect (unlike `symmetrize_density`).
// `tests/kpts_transform.rs` proves the disjointness in code with a
// 1-vs-8-thread bit-identity test rather than only on paper.

impl KPoints {
    /// `kpts.py:449-491` — `transform_mo_coeff` (RHF/RKS shape). `mo_coeff_ibz[i]`
    /// is the ROW-MAJOR `nao x nmo` matrix at IBZ point `i`; the result has
    /// one such matrix per full-BZ k-point.
    ///
    /// # Errors
    /// [`PbcSymmError::KptsSymmInputMismatch`] if the input does not have
    /// `nkpts_ibz` entries (upstream's `raise KeyError`, `:483-485`), plus
    /// [`crate::symmetry::transform_mo_coeff`].
    pub fn transform_mo_coeff(
        &self,
        cell: &Cell,
        mo_coeff_ibz: &[Vec<Complex64>],
        nao: usize,
        nmo: usize,
    ) -> Result<Vec<Vec<Complex64>>, PbcSymmError> {
        self.check_ibz_len(mo_coeff_ibz.len(), "mo_coeff")?;
        (0..self.nkpts())
            .into_par_iter()
            .map(|k| self.transform_mo_coeff_k(cell, mo_coeff_ibz, nao, nmo, k))
            .collect()
    }

    /// `kpts.py:494-526` — `transform_mo_coeff_k` (upstream's
    /// `transform_single_mo_coeff`): the MO coefficients of ONE full-BZ
    /// k-point.
    ///
    /// # Errors
    /// As [`crate::symmetry::transform_mo_coeff`].
    pub fn transform_mo_coeff_k(
        &self,
        cell: &Cell,
        mo_coeff_ibz: &[Vec<Complex64>],
        nao: usize,
        nmo: usize,
        k: usize,
    ) -> Result<Vec<Complex64>, PbcSymmError> {
        let ibz_k_idx = self.bz2ibz[k];
        let ibz_k_scaled = self.kpts_scaled_ibz[ibz_k_idx];
        let iop = self.stars_ops_bz[k];
        let op = &self.ops()[iop];
        let mo_ibz = &mo_coeff_ibz[ibz_k_idx];

        let mut mo_bz = if op.is_eye() {
            mo_ibz.clone()
        } else {
            crate::symmetry::transform_mo_coeff(
                cell,
                ibz_k_scaled,
                mo_ibz,
                nao,
                nmo,
                op,
                &self.dmats()[iop],
            )?
        };
        if self.time_reversal_symm_bz[k] == 1 {
            for v in mo_bz.iter_mut() {
                *v = v.conj();
            }
        }
        Ok(mo_bz)
    }

    /// `kpts.py:528-554` — `transform_mo_occ`. Pure index mapping: the
    /// occupations of a BZ point are those of its IBZ representative.
    ///
    /// # Errors
    /// [`PbcSymmError::KptsSymmInputMismatch`] on a wrong input length.
    pub fn transform_mo_occ(&self, mo_occ_ibz: &[Vec<f64>]) -> Result<Vec<Vec<f64>>, PbcSymmError> {
        self.check_ibz_len(mo_occ_ibz.len(), "mo_occ")?;
        Ok((0..self.nkpts())
            .map(|k| mo_occ_ibz[self.bz2ibz[k]].clone())
            .collect())
    }

    /// `kpts.py:644-661` — `transform_mo_energy`. Pure index mapping, like
    /// [`KPoints::transform_mo_occ`].
    ///
    /// # Errors
    /// [`PbcSymmError::KptsSymmInputMismatch`] on a wrong input length.
    pub fn transform_mo_energy(
        &self,
        mo_energy_ibz: &[Vec<f64>],
    ) -> Result<Vec<Vec<f64>>, PbcSymmError> {
        self.check_ibz_len(mo_energy_ibz.len(), "mo_energy")?;
        Ok((0..self.nkpts())
            .map(|k| mo_energy_ibz[self.bz2ibz[k]].clone())
            .collect())
    }

    /// `kpts.py:556-620` — `transform_dm`. `dm_ibz[i]` is the ROW-MAJOR
    /// `nao x nao` density matrix at IBZ point `i`.
    ///
    /// Upstream's `mo_coeff`/`mo_occ` tag-array passthrough (`:591-593`,
    /// `:618-619`) is not modelled: this port has no `lib.tag_array`, and
    /// the two transforms it would call are public here for a caller that
    /// wants them.
    ///
    /// # Errors
    /// [`PbcSymmError::KptsSymmInputMismatch`] on a wrong input length, plus
    /// [`crate::symmetry::transform_dm`].
    pub fn transform_dm(
        &self,
        cell: &Cell,
        dm_ibz: &[Vec<Complex64>],
        nao: usize,
    ) -> Result<Vec<Vec<Complex64>>, PbcSymmError> {
        self.check_ibz_len(dm_ibz.len(), "density matrix")?;
        self.sandwich_unfold(cell, dm_ibz, nao, crate::symmetry::transform_dm)
    }

    /// `kpts.py:663-714` — `transform_1e_operator` (upstream's
    /// `transform_fock`). Identical sandwich to [`KPoints::transform_dm`];
    /// kept as its own entry point because upstream does, and because
    /// `khf_ksymm` calls them at different points of the cycle.
    ///
    /// # Errors
    /// As [`KPoints::transform_dm`].
    pub fn transform_1e_operator(
        &self,
        cell: &Cell,
        fock_ibz: &[Vec<Complex64>],
        nao: usize,
    ) -> Result<Vec<Vec<Complex64>>, PbcSymmError> {
        self.check_ibz_len(fock_ibz.len(), "1-electron operator")?;
        self.sandwich_unfold(cell, fock_ibz, nao, crate::symmetry::transform_1e_operator)
    }

    /// `kpts.py:713` — `transform_fock`, upstream's alias of
    /// [`KPoints::transform_1e_operator`].
    ///
    /// # Errors
    /// As [`KPoints::transform_1e_operator`].
    pub fn transform_fock(
        &self,
        cell: &Cell,
        fock_ibz: &[Vec<Complex64>],
        nao: usize,
    ) -> Result<Vec<Vec<Complex64>>, PbcSymmError> {
        self.transform_1e_operator(cell, fock_ibz, nao)
    }

    /// The shared body of [`KPoints::transform_dm`] and
    /// [`KPoints::transform_1e_operator`] — both are `R M R^H` followed by
    /// an optional conjugation, differing only in which 17-03 entry point
    /// they name.
    fn sandwich_unfold<F>(
        &self,
        cell: &Cell,
        m_ibz: &[Vec<Complex64>],
        nao: usize,
        per_op: F,
    ) -> Result<Vec<Vec<Complex64>>, PbcSymmError>
    where
        F: Fn(
                &Cell,
                [f64; 3],
                &[Complex64],
                usize,
                &SPGElement,
                &DmatSet,
            ) -> Result<Vec<Complex64>, PbcSymmError>
            + Sync,
    {
        (0..self.nkpts())
            .into_par_iter()
            .map(|k| {
                let ibz_k_idx = self.bz2ibz[k];
                let ibz_kpt_scaled = self.kpts_scaled_ibz[ibz_k_idx];
                let iop = self.stars_ops_bz[k];
                let op = &self.ops()[iop];
                let mut out = if op.is_eye() {
                    m_ibz[ibz_k_idx].clone()
                } else {
                    per_op(
                        cell,
                        ibz_kpt_scaled,
                        &m_ibz[ibz_k_idx],
                        nao,
                        op,
                        &self.dmats()[iop],
                    )?
                };
                if self.time_reversal_symm_bz[k] == 1 {
                    for v in out.iter_mut() {
                        *v = v.conj();
                    }
                }
                Ok(out)
            })
            .collect()
    }

    /// `kpts.py:622-642` — `dm_at_ref_cell`: the reference-cell density
    /// matrix, `sum_k dm_bz[k] / nkpts`.
    ///
    /// The `nkpts x nao^2` accumulation goes through
    /// [`pyscf_algebra::oracle_sum`] (D-PBC-17), so the answer does not
    /// depend on the thread count or on the star order.
    ///
    /// Emits upstream's `logger.warn` (`:639-641`) as a `tracing::warn!`
    /// when the imaginary part is not negligible.
    ///
    /// # Errors
    /// As [`KPoints::transform_dm`].
    pub fn dm_at_ref_cell(
        &self,
        cell: &Cell,
        dm_ibz: &[Vec<Complex64>],
        nao: usize,
    ) -> Result<Vec<Complex64>, PbcSymmError> {
        let dm_bz = self.transform_dm(cell, dm_ibz, nao)?;
        let nkpts = self.nkpts() as f64;
        let dm0: Vec<Complex64> = (0..nao * nao)
            .into_par_iter()
            .map(|i| {
                let re: Vec<f64> = dm_bz.iter().map(|m| m[i].re).collect();
                let im: Vec<f64> = dm_bz.iter().map(|m| m[i].im).collect();
                Complex64::new(
                    pyscf_algebra::oracle_sum(&re) / nkpts,
                    pyscf_algebra::oracle_sum(&im) / nkpts,
                )
            })
            .collect();
        let max_im = dm0.iter().map(|v| v.im.abs()).fold(0.0_f64, f64::max);
        if max_im > 1e-10 {
            tracing::warn!(
                "Imaginary density matrix found at reference cell: abs(dm0.imag).max() = {max_im:e}"
            );
        }
        Ok(dm0)
    }

    /// `kpts.py:717-755` — `check_mo_occ_symmetry`. Verifies the full-BZ MO
    /// occupations really are constant across each star, then returns the
    /// IBZ slice.
    ///
    /// # Errors
    /// [`PbcSymmError::SymmetryBrokenOccupation`] — upstream's
    /// `raise RuntimeError("Symmetry broken solution found. ...")`.
    pub fn check_mo_occ_symmetry(
        &self,
        mo_occ: &[Vec<f64>],
        tol: f64,
    ) -> Result<Vec<Vec<f64>>, PbcSymmError> {
        for bz_k in &self.stars {
            for i in 0..bz_k.len() {
                for j in (i + 1)..bz_k.len() {
                    let (a, b) = (&mo_occ[bz_k[i]], &mo_occ[bz_k[j]]);
                    if a.len() != b.len()
                        || a.iter().zip(b.iter()).any(|(x, y)| (x - y).abs() >= tol)
                    {
                        return Err(PbcSymmError::SymmetryBrokenOccupation(bz_k[i], bz_k[j]));
                    }
                }
            }
        }
        Ok((0..self.nkpts_ibz())
            .map(|k| mo_occ[self.ibz2bz[k]].clone())
            .collect())
    }

    /// `kpts.py:757-802` — `get_rotation_mat_for_mos`: the matrices that
    /// rotate `MO[k1]` into `MO[k2]`, one per requested operation.
    ///
    /// `mo_coeff[k]` and `ovlp[k]` are ROW-MAJOR `nao x nmo` / `nao x nao`.
    /// `ops_id = None` means "all `nop` rotations".
    ///
    /// # Errors
    /// [`PbcSymmError::KptsSymmInputMismatch`] if `k1` and `k2` (or
    /// `ops_id`) have different lengths, plus
    /// [`crate::symmetry::get_rotation_mat`].
    // Nine parameters, mirroring upstream's `get_rotation_mat_for_mos(kpts,
    // mo_coeff, ovlp, k1, k2, ops_id)` (`kpts.py:757`) plus the `cell` this
    // port passes explicitly (17-CONTEXT §3.9) and the `nao`/`nmo` that a
    // flat `Vec<Complex64>` cannot carry the way a NumPy array can. Grouping
    // them into a struct would obscure the correspondence RULE 2 asks for.
    #[allow(clippy::too_many_arguments)]
    pub fn get_rotation_mat_for_mos(
        &self,
        cell: &Cell,
        mo_coeff: &[Vec<Complex64>],
        ovlp: &[Vec<Complex64>],
        nao: usize,
        nmo: usize,
        k1: &[usize],
        k2: &[usize],
        ops_id: Option<&[Vec<usize>]>,
    ) -> Result<Vec<Vec<Vec<Complex64>>>, PbcSymmError> {
        if k1.len() != k2.len() || ops_id.is_some_and(|o| o.len() != k1.len()) {
            return Err(PbcSymmError::KptsSymmInputMismatch(format!(
                "get_rotation_mat_for_mos: len(k1) = {}, len(k2) = {}, len(ops_id) = {:?}",
                k1.len(),
                k2.len(),
                ops_id.map(|o| o.len())
            )));
        }
        let all_ops: Vec<usize> = (0..self.nop()).collect();
        (0..k1.len())
            .into_par_iter()
            .map(|k| {
                let (k_orig, k_target) = (k1[k], k2[k]);
                let ids: &[usize] = match ops_id {
                    Some(o) => &o[k],
                    None => &all_ops,
                };
                ids.iter()
                    .map(|&iop| {
                        let mat_ao = crate::symmetry::get_rotation_mat(
                            cell,
                            self.kpts_scaled[k_orig],
                            nao,
                            &self.ops()[iop],
                            &self.dmats()[iop],
                            false,
                            crate::space_group::SYMPREC,
                        )?;
                        // reduce(dot, (C[k_orig]^H, S[k_orig], R^H, C[k_target]))
                        let ch = conj_transpose(&mo_coeff[k_orig], nao, nmo);
                        let t1 = cmatmul(&ch, nmo, nao, &ovlp[k_orig], nao, nao);
                        let rh = conj_transpose(&mat_ao, nao, nao);
                        let t2 = cmatmul(&t1, nmo, nao, &rh, nao, nao);
                        Ok(cmatmul(&t2, nmo, nao, &mo_coeff[k_target], nao, nmo))
                    })
                    .collect::<Result<Vec<_>, PbcSymmError>>()
            })
            .collect()
    }

    fn check_ibz_len(&self, got: usize, what: &str) -> Result<(), PbcSymmError> {
        if got != self.nkpts_ibz() {
            return Err(PbcSymmError::KptsSymmInputMismatch(format!(
                "shape of {what} does not match the number of IBZ k-points: \
                 {got} vs {}",
                self.nkpts_ibz()
            )));
        }
        Ok(())
    }
}

/// Row-major `m x n` complex conjugate transpose -> `n x m`.
fn conj_transpose(a: &[Complex64], m: usize, n: usize) -> Vec<Complex64> {
    let mut out = vec![Complex64::new(0.0, 0.0); m * n];
    for i in 0..m {
        for j in 0..n {
            out[j * m + i] = a[i * n + j].conj();
        }
    }
    out
}

/// Row-major complex `(m x k) @ (k x n)`. A local, host-only helper: this
/// crate never crosses the algebra wall (ALG-06/RULE 8) and the matrices
/// here are `nao`-sized, so a device `gemm` would cost more than it saves.
fn cmatmul(
    a: &[Complex64],
    m: usize,
    k: usize,
    b: &[Complex64],
    kb: usize,
    n: usize,
) -> Vec<Complex64> {
    debug_assert_eq!(k, kb);
    let mut out = vec![Complex64::new(0.0, 0.0); m * n];
    for i in 0..m {
        for p in 0..k {
            let aip = a[i * k + p];
            if aip == Complex64::new(0.0, 0.0) {
                continue;
            }
            for j in 0..n {
                out[i * n + j] += aip * b[p * n + j];
            }
        }
    }
    out
}

/// `kpts.py:1127-1173` — `MORotationMatrix`: the cache of
/// [`KPoints::get_rotation_mat_for_mos`] over every `(k, op)` pair, split
/// into the occupied-occupied and virtual-virtual blocks.
///
/// Built LAZILY ([`MORotationMatrix::build`]) because `kmp2_ksymm` and
/// `kccsd_rhf_ksymm` read it per k-tuple and neither wants to pay for it
/// unless it runs.
#[derive(Debug, Clone)]
pub struct MORotationMatrix {
    pub nocc: usize,
    pub nmo: usize,
    /// `oo[ki][iop]` — the `nocc x nocc` rotation, ROW-MAJOR.
    pub oo: Option<Vec<Vec<Vec<Complex64>>>>,
    /// `vv[ki][iop]` — the `nvir x nvir` rotation, ROW-MAJOR.
    pub vv: Option<Vec<Vec<Vec<Complex64>>>>,
}

impl MORotationMatrix {
    /// `kpts.py:1131-1140` — `__init__`.
    ///
    /// # Panics
    /// If `nmo < nocc` (upstream's `assert`).
    pub fn new(nocc: usize, nmo: usize) -> Self {
        assert!(nmo >= nocc, "MORotationMatrix: nmo ({nmo}) < nocc ({nocc})");
        Self {
            nocc,
            nmo,
            oo: None,
            vv: None,
        }
    }

    /// `kpts.py:1142-1173` — `build`. `mo_coeff[k]` / `ovlp[k]` are
    /// ROW-MAJOR `nao x nmo` / `nao x nao`.
    ///
    /// # Errors
    /// As [`KPoints::get_rotation_mat_for_mos`].
    pub fn build(
        &mut self,
        kpts: &KPoints,
        cell: &Cell,
        mo_coeff: &[Vec<Complex64>],
        ovlp: &[Vec<Complex64>],
        nao: usize,
    ) -> Result<(), PbcSymmError> {
        let (nocc, nmo) = (self.nocc, self.nmo);
        let nvir = nmo - nocc;
        // orb_occ = [mo[:, :nocc] for mo in mo_coeff]; orb_vir = [mo[:, nocc:]]
        let orb_occ: Vec<Vec<Complex64>> = mo_coeff
            .iter()
            .map(|m| column_slice(m, nao, nmo, 0, nocc))
            .collect();
        let orb_vir: Vec<Vec<Complex64>> = mo_coeff
            .iter()
            .map(|m| column_slice(m, nao, nmo, nocc, nmo))
            .collect();

        let nkpts = kpts.nkpts();
        let nop = kpts.nop();
        let all_ops: Vec<usize> = (0..nop).collect();
        let mut oo = Vec::with_capacity(nkpts);
        let mut vv = Vec::with_capacity(nkpts);
        for ki in 0..nkpts {
            let k1 = vec![ki; nop];
            let k2: Vec<usize> = (0..nop).map(|iop| kpts.k2opk[ki][iop] as usize).collect();
            // upstream passes `ops_id = np.arange(nop)` (`kpts.py:1164`), and
            // `get_rotation_mat_for_mos` then does
            // `ids = np.asarray(ops_id[k]).reshape(-1)` (`:793`) — so pair
            // `iop` uses EXACTLY ONE op, `iop`, and the result reshapes to
            // `(nop, n, n)`. Passing `None` here would compute `nop x nop`
            // matrices and throw all but the diagonal away.
            let ops: Vec<Vec<usize>> = all_ops.iter().map(|&iop| vec![iop]).collect();
            let rot_oo = kpts.get_rotation_mat_for_mos(
                cell,
                &orb_occ,
                ovlp,
                nao,
                nocc,
                &k1,
                &k2,
                Some(&ops),
            )?;
            let rot_vv = kpts.get_rotation_mat_for_mos(
                cell,
                &orb_vir,
                ovlp,
                nao,
                nvir,
                &k1,
                &k2,
                Some(&ops),
            )?;
            oo.push(rot_oo.into_iter().map(|mut v| v.remove(0)).collect());
            vv.push(rot_vv.into_iter().map(|mut v| v.remove(0)).collect());
        }
        self.oo = Some(oo);
        self.vv = Some(vv);
        Ok(())
    }
}

/// `mo[:, lo:hi]` for a ROW-MAJOR `nrows x ncols` matrix.
fn column_slice(
    m: &[Complex64],
    nrows: usize,
    ncols: usize,
    lo: usize,
    hi: usize,
) -> Vec<Complex64> {
    let w = hi - lo;
    let mut out = vec![Complex64::new(0.0, 0.0); nrows * w];
    for r in 0..nrows {
        out[r * w..(r + 1) * w].copy_from_slice(&m[r * ncols + lo..r * ncols + hi]);
    }
    out
}

// ---------------------------------------------------------------------
// Task 4 — symmetrize_density / symmetrize_wavefunction (kpts.py:377-448)
// ---------------------------------------------------------------------
//
// # D-PBC-17 from the FIRST version (17-CONTEXT §3.8)
//
// This is an `nkpts x ngrids` accumulation — the same shape D-PBC-17
// governs and that 15-CONTEXT ruled on for KMP2. The per-grid-point sum
// over the star's operations goes through
// [`pyscf_algebra::oracle_sum`]'s fixed-shape pairwise tree, so the result
// is bit-identical at any thread count, and the outer loop is parallelised
// over GRID POINTS (disjoint writes), not over operations. It ships that
// way here rather than as a retrofit; `tests/kpts_transform.rs` carries the
// §9.3 1-vs-8-thread bit-identity test from day one.
//
// # The rotated index must land EXACTLY on a mesh point
//
// Upstream's C kernel (`pyscf/lib/pbc/symmetry.c:25-48`) computes the
// fractional-translation offset as `(int)(ft[0] * nx)` — a TRUNCATION. That
// is silently wrong whenever `ft * n` is not exactly representable (e.g.
// `ft = 1/3`, `n = 3`, where `0.3333333333333333 * 3 = 0.9999999999999999`
// truncates to 0 instead of 1). `check_mesh_symmetry` (`symmetry.py:96`,
// 17-03 Task 4) is what guarantees `ft * n` IS an integer; this port
// ASSERTS that and fails with [`PbcSymmError::MeshNotSymmetric`] rather
// than rounding silently, because a silent round is a wrong density
// (17-05-PLAN.md Task 4).

/// The `[x_p, y_p, z_p]` source index for grid point `(x, y, z)` under the
/// integer rotation `op` and integer translation offset `ft` — upstream's
/// `symmetry.c:16-20`, with C's `((v % n) + n) % n` non-negative modulo.
#[inline]
fn rotated_grid_index(op: &[[i32; 3]; 3], ft: [i64; 3], mesh: [usize; 3], xyz: [i64; 3]) -> usize {
    let mut p = [0usize; 3];
    for (i, pi) in p.iter_mut().enumerate() {
        let n = mesh[i] as i64;
        let v =
            op[i][0] as i64 * xyz[0] + op[i][1] as i64 * xyz[1] + op[i][2] as i64 * xyz[2] + ft[i];
        *pi = (((v % n) + n) % n) as usize;
    }
    (p[0] * mesh[1] + p[1]) * mesh[2] + p[2]
}

/// `ft * mesh` as an exact integer vector, or
/// [`PbcSymmError::MeshNotSymmetric`]. See this section's module comment.
///
/// `pub` so `tests/kpts_transform.rs` can pin the fail-loudly behaviour
/// DIRECTLY. It cannot reach it through [`KPoints::symmetrize_density`] on
/// the §9.2 fixtures: `SPGElement`'s ordering puts zero-translation ops
/// first (`hash_key = trans*3^9 + rot`), and `make_kpts_ibz`'s star search
/// `break`s at the first op that maps the IBZ point onto the BZ point — so
/// on a cell whose symmorphic subgroup already covers every star,
/// `stars_ops` never names a non-symmorphic op. That is a property of the
/// fixtures, not a guarantee, which is exactly why the check must still be
/// tested.
pub fn ft_offsets(iop: usize, trans: [f64; 3], mesh: [usize; 3]) -> Result<[i64; 3], PbcSymmError> {
    let mut out = [0i64; 3];
    for i in 0..3 {
        let v = trans[i] * mesh[i] as f64;
        if (v - v.round()).abs() > 1e-9 {
            return Err(PbcSymmError::MeshNotSymmetric(iop, i, v));
        }
        out[i] = v.round() as i64;
    }
    Ok(out)
}

/// One star operation's action on the real-space grid: an integer rotation
/// in the direct-lattice basis plus an integer translation offset in mesh
/// units. `None` is the identity (upstream's `if op.is_eye` fast path,
/// `kpts.py:430`).
type GridOp = Option<([[i32; 3]; 3], [i64; 3])>;

/// The [`GridOp`] of every op in IBZ point `ibz_k_idx`'s star, in upstream's
/// order — `inv_op = op.inv()` (`kpts.py:432`), NOT `op` itself.
fn star_grid_ops(
    kpts: &KPoints,
    ibz_k_idx: usize,
    mesh: [usize; 3],
) -> Result<Vec<GridOp>, PbcSymmError> {
    let mut out = Vec::with_capacity(kpts.stars_ops[ibz_k_idx].len());
    for &iop in &kpts.stars_ops[ibz_k_idx] {
        let op = &kpts.ops()[iop];
        if op.is_eye() {
            // `if op.is_eye: rhoR += rhoR_k` — the identity contribution.
            out.push(None);
            continue;
        }
        let inv_op = op.inv()?;
        let rot: [[i32; 3]; 3] =
            std::array::from_fn(|i| std::array::from_fn(|j| inv_op.rot[i][j].round() as i32));
        let ft = if inv_op.trans_is_zero() {
            [0i64; 3]
        } else {
            ft_offsets(iop, inv_op.trans, mesh)?
        };
        out.push(Some((rot, ft)));
    }
    Ok(out)
}

impl KPoints {
    /// `kpts.py:377-414` — `symmetrize_density`: unfold ONE IBZ k-point's
    /// real-space density over its whole star and accumulate.
    ///
    /// `rho_k` is the real-space density on the `mesh` grid, C-ordered
    /// (`x` slowest, `z` fastest) — `ngrids = mesh[0] * mesh[1] * mesh[2]`.
    ///
    /// The star sum goes through [`pyscf_algebra::oracle_sum`] (D-PBC-17);
    /// see this module's Task 4 section.
    ///
    /// # Errors
    /// [`PbcSymmError::MeshNotSymmetric`] if a fractional translation does
    /// not land on a mesh point, and a singular rotation from
    /// [`SPGElement::inv`].
    ///
    /// # Panics
    /// If `rho_k.len() != mesh[0] * mesh[1] * mesh[2]`.
    pub fn symmetrize_density(
        &self,
        rho_k: &[f64],
        ibz_k_idx: usize,
        mesh: [usize; 3],
    ) -> Result<Vec<f64>, PbcSymmError> {
        let ngrids = mesh[0] * mesh[1] * mesh[2];
        assert_eq!(
            rho_k.len(),
            ngrids,
            "symmetrize_density: rho_k does not match the mesh"
        );
        let permutation = self.density_grid_permutation(ibz_k_idx, mesh)?;
        let nops = permutation.len() / ngrids;
        const CHUNK: usize = 4096;
        let mut out = vec![0.0; ngrids];
        out.par_chunks_mut(CHUNK)
            .enumerate()
            .for_each(|(chunk, dst)| {
                let mut terms = vec![0.0; nops];
                let g0 = chunk * CHUNK;
                for (local, value) in dst.iter_mut().enumerate() {
                    let g = g0 + local;
                    for op in 0..nops {
                        terms[op] = rho_k[permutation[op * ngrids + g] as usize];
                    }
                    *value = pyscf_algebra::oracle_sum(&terms);
                }
            });
        Ok(out)
    }

    /// [`KPoints::symmetrize_density`] for a COMPLEX density, in the planar
    /// `(re, im)` split (RULE 8) — upstream's `libpbc.symmetrize_complex`
    /// branch (`kpts.py:396-397`). Real and imaginary parts each go through
    /// their own [`pyscf_algebra::oracle_sum`].
    ///
    /// # Errors
    /// As [`KPoints::symmetrize_density`].
    ///
    /// # Panics
    /// If `re` and `im` do not both match the mesh.
    pub fn symmetrize_density_complex(
        &self,
        re: &[f64],
        im: &[f64],
        ibz_k_idx: usize,
        mesh: [usize; 3],
    ) -> Result<(Vec<f64>, Vec<f64>), PbcSymmError> {
        let ngrids = mesh[0] * mesh[1] * mesh[2];
        assert_eq!(re.len(), ngrids);
        assert_eq!(im.len(), ngrids);
        let permutation = self.density_grid_permutation(ibz_k_idx, mesh)?;
        let nops = permutation.len() / ngrids;
        const CHUNK: usize = 4096;
        let mut out_re = vec![0.0; ngrids];
        let mut out_im = vec![0.0; ngrids];
        out_re
            .par_chunks_mut(CHUNK)
            .zip(out_im.par_chunks_mut(CHUNK))
            .enumerate()
            .for_each(|(chunk, (dst_re, dst_im))| {
                let mut tr = vec![0.0; nops];
                let mut ti = vec![0.0; nops];
                let g0 = chunk * CHUNK;
                for local in 0..dst_re.len() {
                    let g = g0 + local;
                    for op in 0..nops {
                        let src = permutation[op * ngrids + g] as usize;
                        tr[op] = re[src];
                        ti[op] = im[src];
                    }
                    dst_re[local] = pyscf_algebra::oracle_sum(&tr);
                    dst_im[local] = pyscf_algebra::oracle_sum(&ti);
                }
            });
        Ok((out_re, out_im))
    }

    /// Symmetrize the Cartesian vector part of an LDA/GGA density.
    ///
    /// Grid points use the same cached inverse-operation permutation as
    /// [`Self::symmetrize_density`].  Vector values are then rotated by the
    /// operation's `l = 1` Wigner-D matrix, whose rows and columns are in
    /// `(x, y, z)` order.  Each output component keeps the star-operation
    /// order and [`pyscf_algebra::oracle_sum`] reduction used by the scalar
    /// path, making the result independent of the Rayon worker count.
    ///
    /// # Panics
    /// If any component does not match the mesh.
    pub fn symmetrize_density_vec(
        &self,
        rho: [&[f64]; 3],
        ibz_k_idx: usize,
        mesh: [usize; 3],
    ) -> Result<[Vec<f64>; 3], PbcSymmError> {
        let ngrids = mesh[0] * mesh[1] * mesh[2];
        for component in rho {
            assert_eq!(component.len(), ngrids);
        }
        let permutation = self.density_grid_permutation(ibz_k_idx, mesh)?;
        let star = &self.stars_ops[ibz_k_idx];
        debug_assert_eq!(permutation.len(), star.len() * ngrids);

        const CHUNK: usize = 4096;
        let mut out: [Vec<f64>; 3] = std::array::from_fn(|_| vec![0.0; ngrids]);
        let (x, tail) = out.split_at_mut(1);
        let (y, z) = tail.split_at_mut(1);
        x[0].par_chunks_mut(CHUNK)
            .zip(y[0].par_chunks_mut(CHUNK))
            .zip(z[0].par_chunks_mut(CHUNK))
            .enumerate()
            .for_each(|(chunk, ((dst_x, dst_y), dst_z))| {
                let mut terms: [Vec<f64>; 3] = std::array::from_fn(|_| vec![0.0; star.len()]);
                let g0 = chunk * CHUNK;
                for local in 0..dst_x.len() {
                    let g = g0 + local;
                    for (op_pos, &iop) in star.iter().enumerate() {
                        let src = permutation[op_pos * ngrids + g] as usize;
                        let d = &self.dmats()[iop][1];
                        for row in 0..3 {
                            terms[row][op_pos] = d[row][0] * rho[0][src]
                                + d[row][1] * rho[1][src]
                                + d[row][2] * rho[2][src];
                        }
                    }
                    dst_x[local] = pyscf_algebra::oracle_sum(&terms[0]);
                    dst_y[local] = pyscf_algebra::oracle_sum(&terms[1]);
                    dst_z[local] = pyscf_algebra::oracle_sum(&terms[2]);
                }
            });
        Ok(out)
    }

    /// `kpts.py:416-448` — `symmetrize_wavefunction`.
    ///
    /// **Upstream refuses this path outright**: its very first statement is
    /// `raise RuntimeError('need verification')` (`:415`), so every line
    /// below it is dead code that has never run. RULE 2 makes that
    /// authoritative — this port refuses identically rather than shipping
    /// an algorithm upstream itself will not vouch for.
    ///
    /// # Errors
    /// Always [`PbcSymmError::SymmetrizeWavefunctionUnverified`].
    pub fn symmetrize_wavefunction(&self) -> Result<(), PbcSymmError> {
        Err(PbcSymmError::SymmetrizeWavefunctionUnverified)
    }
}

/// The `(x, y, z)` mesh coordinates of C-ordered flat index `g`.
#[inline]
fn grid_xyz(g: usize, mesh: [usize; 3]) -> [i64; 3] {
    let z = g % mesh[2];
    let rest = g / mesh[2];
    let y = rest % mesh[1];
    let x = rest / mesh[1];
    [x as i64, y as i64, z as i64]
}
