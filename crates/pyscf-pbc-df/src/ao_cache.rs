//! The budgeted k-point AO table shared by the FFTDF gradient JK (plan 18-04)
//! and `hcore_generator` (plan 18-05) — D-PBC-30 clause 3 as corrected by
//! D-PBC-31 clauses 8 and 12.
//!
//! `get_k_e1_kpts` (`pyscf/pbc/df/fft_jk.py:346-350`) builds `ao1_kpts` for
//! **all** k-points at `deriv = 1` before the loop — `nkpts · 4 · ngrids · nao`
//! complex, 1.02 GiB at diamond `gth-dzvp` 3×3×3 — while `blksize` (`:363`)
//! guards only the inner `rho1`. The grid cannot be blocked (the FFT at `:392`
//! needs it whole), so the only lever is how many k-points are resident.
//!
//! One parameter, not two branches (clause 8): hold **`m`** k-point tables,
//! `m` chosen from `PYSCF_MAX_MEMORY` (the `aftdf.rs:84-85` precedent), and
//! tile the inner `k1` loop in chunks of `m`. Peak `(m + 1) · 4 · ngrids · nao`
//! complex, cost `nkpts²/m` chunk builds. `m = nkpts` **is** the resident case
//! and `m = 1` the streaming one — Part I's two branches are the endpoints of
//! this one number, not two code paths.
//!
//! `blksize` is re-derived, not transcribed (clause 12). Upstream's `:363` is
//! `(max_memory-mem_now)*1e6/16/4/3/ngrids/nao`: the `16` is bytes per
//! complex and the `3` the component axis (both kept); the **`4` is buffer
//! multiplicity and is deliberate** — `:389-394` holds `rho1`+`vG` across the
//! forward transform and `vG`+`vR` across the inverse — but its value is the
//! one 18-01 Task 5 measured (**2**, in-place transform; see
//! [`BUFFER_MULTIPLICITY`]), not the literal `4`; and the **doubled `mem_now`
//! is the defect** (`:362` subtracts it, `:363` subtracts it again —
//! `18-CONTEXT §3.7`). Only one subtraction survives here.
//!
//! No CubeCL kernels live here (ALG-06): the tables are built by the existing
//! `eval_ao_kpts` evaluators in `pyscf-kernels`, and everything else is
//! host-side contractions plus the FFT, exactly like `fft_jk.rs`.

use std::sync::atomic::{AtomicUsize, Ordering};

use pyscf_algebra::CTensor;
use pyscf_pbc_gto::{Cell, eval_ao_kpts};

use crate::error::PbcDfError;

/// `deriv = 1` components: the value plus the three Cartesian derivatives —
/// upstream's `ao.transpose(0,2,1)` in `fft_jk.py:346-350`, i.e. `ao1T[0]` is
/// the value table and `ao1T[1:]` the gradient bra.
pub const DERIV1_COMPONENTS: usize = 4;

/// Buffer multiplicity of the inner `rho1` transient: **2**, the integer 18-01
/// Task 5 measured (`measurements/sizings.md` §2d — blksize-slope 2.2
/// rho-units, i.e. integer 2 plus ~0.2 units FFT/transpose workspace). The
/// transform runs effectively in place; there is no 3–4× blowup, so the
/// re-derived [`deriv1_blksize_from_avail`] keeps `/2`, not upstream's `/4`.
pub const BUFFER_MULTIPLICITY: f64 = 2.0;

/// Bytes of one deriv-1 k-point table: 4 components × `ngrids` × `nao`
/// complex entries at 16 B each.
pub fn deriv1_table_bytes(ngrids: usize, nao: usize) -> u64 {
    DERIV1_COMPONENTS as u64 * ngrids as u64 * nao as u64 * 16
}

/// MB residency of an `m`-table — the `mem_now` the gradient loop reports to
/// the `blksize` sizing, so the inner transient is sized from what is left
/// after the resident tables, not from the whole budget.
pub fn resident_mb(m: usize, ngrids: usize, nao: usize) -> f64 {
    m as f64 * deriv1_table_bytes(ngrids, nao) as f64 / 1e6
}

/// How many deriv-1 k-point tables fit in `max_memory_mb`, clamped to
/// `[1, nkpts]`. `m = nkpts` is the resident endpoint, `m = 1` the streaming
/// one (D-PBC-31 clause 8).
pub fn resident_k_count(max_memory_mb: f64, ngrids: usize, nao: usize, nkpts: usize) -> usize {
    let nkpts = nkpts.max(1);
    let per_mb = deriv1_table_bytes(ngrids, nao) as f64 / 1e6;
    if !(per_mb > 0.0) {
        return nkpts;
    }
    ((max_memory_mb / per_mb).floor() as usize).clamp(1, nkpts)
}

/// `blksize` from AVAILABLE MB — the budget already net of the resident
/// tables. Upstream's `:363` shape with the measured multiplicity:
/// `avail·1e6/16/2/3/ngrids/nao`, clamped to `[1, nao]`. Under MO tagging the
/// transient is smaller (`naoj = nocc`), so sizing by the full `nao` is
/// conservative there — it can only undershoot, never overshoot.
pub fn deriv1_blksize_from_avail(avail_mb: f64, ngrids: usize, nao: usize) -> usize {
    let blk =
        (avail_mb.max(0.0) * 1e6 / 16.0 / BUFFER_MULTIPLICITY / 3.0 / ngrids as f64 / nao as f64)
            as usize;
    nao.min(blk.max(1))
}

/// [`deriv1_blksize_from_avail`] from the gross budget minus the resident
/// tables. Exactly ONE `mem_now` subtraction — upstream's `:362`–`:363`
/// subtracts it twice, which is the defect this drops.
pub fn deriv1_blksize(max_memory_mb: f64, mem_now_mb: f64, ngrids: usize, nao: usize) -> usize {
    deriv1_blksize_from_avail(max_memory_mb - mem_now_mb, ngrids, nao)
}

/// The k-difference index (D-PBC-31 clause 9): `dk` at fixed fractional
/// coordinates, wrapped into `[0, 1)` and quantised at 1e-9.
///
/// On a Monkhorst–Pack mesh the set `{k2 − k1}` **is** the mesh, so there are
/// `nkpts` distinct classes against `nkpts²` pairs — but only after wrapping:
/// the raw Cartesian differences span `±0.5`-style duplicates (`-0.5` and
/// `+0.5` are different bit patterns for the same class). The fractional
/// coordinates are the port's own `get_scaled_kpts` (absolute k-points are
/// `scaled · B(2π)`), and the 1e-9 quantum absorbs the `~1e-16` conversion
/// noise while genuine classes sit `≥ 1/nk ≫ 1e-9` apart. A quantum that
/// lands exactly on `1_000_000_000` folds to `0` (`1 ≡ 0 mod 1`), so the
/// diagonal can never miss itself.
///
/// This keys the **coulG** side of the W-01 cache only. `coulG` enters purely
/// through the wrapped `G + k` grid, so it is genuinely class-invariant; the
/// `expmikr` phase table depends on the unwrapped representative and stays
/// keyed on the raw `dk` bits.
pub fn kdiff_index(cell: &Cell, dk: [f64; 3]) -> [i64; 3] {
    let scaled = cell.get_scaled_kpts(&[dk]);
    let mut idx = [0i64; 3];
    for j in 0..3 {
        let w = scaled[0][j].rem_euclid(1.0);
        let mut r = (w * 1e9).round() as i64;
        if r == 1_000_000_000 {
            r = 0;
        }
        idx[j] = r;
    }
    idx
}

/// Counts deriv-1 table builds: one `builds` per `eval_ao_kpts` call, one
/// `kpoints` per k-point table inside it. The 18-04 CI test asserts the
/// clause-8 exchange rate against these (`nkpts²/m` chunk builds for the
/// double loop), which is what keeps a fixture that silently stayed incore
/// from passing.
#[derive(Debug, Default)]
pub struct AoEvalCount {
    builds: AtomicUsize,
    kpoints: AtomicUsize,
}

impl AoEvalCount {
    /// Number of `eval_ao_kpts` calls issued through [`eval_deriv1_chunk`].
    pub fn builds(&self) -> usize {
        self.builds.load(Ordering::Relaxed)
    }

    /// Total k-point tables built.
    pub fn kpoints(&self) -> usize {
        self.kpoints.load(Ordering::Relaxed)
    }
}

/// One k-point's deriv-1 table. `kao` is `4·ngrids·nao` complex, F-order per
/// component — component `c`, AO `p`, grid `g` sits at `(c·nao + p)·ngrids +
/// g`, i.e. every component block reads as `(nao, ngrids)` row-major, exactly
/// upstream's `ao1T[c]`.
#[derive(Debug)]
pub struct Deriv1Table {
    /// Index into the caller's k-point list (chunks need not be contiguous).
    pub kidx: usize,
    /// The `GTOval_sph_deriv1` table at that k-point.
    pub kao: CTensor,
}

/// Evaluate the deriv-1 AO tables for the k-point subset `idxs` in ONE
/// `eval_ao_kpts` call over the full grid — the grid cannot be blocked (the
/// FFT needs it whole), so tiling is over k-points only — and count it on
/// `counter` when given.
///
/// # Errors
/// Propagates [`eval_ao_kpts`].
pub fn eval_deriv1_chunk(
    cell: &Cell,
    coords: &[[f64; 3]],
    kpts: &[[f64; 3]],
    idxs: &[usize],
    counter: Option<&AoEvalCount>,
) -> Result<Vec<Deriv1Table>, PbcDfError> {
    let sub: Vec<[f64; 3]> = idxs.iter().map(|&i| kpts[i]).collect();
    let out = eval_ao_kpts(cell, "GTOval_sph_deriv1", coords, &sub)?;
    debug_assert_eq!(
        out.comp, DERIV1_COMPONENTS,
        "GTOval_sph_deriv1 must carry value + 3 derivatives"
    );
    if let Some(c) = counter {
        c.builds.fetch_add(1, Ordering::Relaxed);
        c.kpoints.fetch_add(idxs.len(), Ordering::Relaxed);
    }
    Ok(idxs
        .iter()
        .zip(out.kaos)
        .map(|(&kidx, kao)| Deriv1Table { kidx, kao })
        .collect())
}

/// Accounted high-water of one `get_k_e1_kpts` run, in bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KGradFootprint {
    /// `m` resident deriv-1 tables plus the MO-tagged ket table when present
    /// (`ket_extra_bytes`).
    pub resident: u64,
    /// One inner block at the measured multiplicity
    /// (`2·3·blksize·naoj·ngrids·16`, §8.1) plus the hoisted `vR_dm`
    /// (`3·nset·nao·ngrids·16`).
    pub transient: u64,
    /// `resident + transient`.
    pub peak: u64,
}

/// [`KGradFootprint`] from a run's realised `(m, blksize, naoj)`. `naoj` is
/// `nao` untagged and the occupied count tagged; `ket_extra_bytes` is `0`
/// untagged and the `nkpts·nocc·ngrids·16` MO table (§8.2) tagged. The 18-04
/// Task 3 gate asserts the tagged peak is strictly below the untagged one on
/// the same fixture — an optimisation with no test that it *was* taken is one
/// that silently stops being taken.
pub fn k_e1_footprint(
    m: usize,
    blksize: usize,
    ngrids: usize,
    nao: usize,
    naoj: usize,
    nset: usize,
    ket_extra_bytes: u64,
) -> KGradFootprint {
    let resident = m as u64 * deriv1_table_bytes(ngrids, nao) + ket_extra_bytes;
    let rho1 = BUFFER_MULTIPLICITY as u64 * 3 * blksize as u64 * naoj as u64 * ngrids as u64 * 16;
    let vrdm = 3 * nset as u64 * nao as u64 * ngrids as u64 * 16;
    let transient = rho1 + vrdm;
    KGradFootprint {
        resident,
        transient,
        peak: resident + transient,
    }
}
