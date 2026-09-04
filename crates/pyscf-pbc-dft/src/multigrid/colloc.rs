//! Multigrid density / potential collocation — plan 17-11 Tasks 2/3.
//!
//! Two directions, both built on ONE kernel call per level
//! (`pyscf_kernels::multigrid_collocate::collocate`, evaluating every pshell
//! this level needs — `level.dense ∪ level.sparse` — on the level's own
//! mesh):
//!
//! * [`level_rho`] — `NUMINT_fill`/`eval_rho` (`multigrid.py:200-365`):
//!   contract a decontracted density matrix against the collocated values to
//!   get `rho(r)` on this level's mesh. Every grid point's accumulation is a
//!   FIXED-order [`oracle_sum`] over the pair-term list (D-PBC-17 shape).
//! * [`level_pass2`] — `NUMINT_fill2c`/`eval_mat`
//!   (`multigrid.py:77-199`, the "pass2" `_get_j_pass2` calls into): contract
//!   a real-space weight field against the collocated values to get a
//!   decontracted potential matrix contribution. Every matrix entry's
//!   accumulation over the grid is an [`oracle_sum`], matching
//!   `crate::numint::vxc_mat_one`'s established idiom.
//!
//! Both only ever touch the pshells `level.dense ∪ level.sparse` — the
//! `Part A` (`dense x (dense∪sparse)`) / `Part B` (`sparse x dense`) split
//! documented in `tasks.rs`'s module doc reproduces upstream's dense/sparse
//! coverage (every pair computed exactly once, at the finer of its two
//! pshells' levels) without upstream's separate `h_coeff`/`l_coeff`/`t_cell`
//! bookkeeping.

use pyscf_algebra::oracle_sum;
use pyscf_kernels::multigrid_collocate::{PshellGridTable, collocate};
use pyscf_pbc_gto::Cell;
use rayon::prelude::*;

use crate::error::PbcDftError;
use crate::multigrid::tasks::{Decontracted, GridLevel, pshell_cart_powers};

/// One level's collocated Cartesian primitive values, plus enough bookkeeping
/// to find any pshell's slot range in the flat `(slot, grid)` buffer.
pub struct LevelValues {
    /// `ids[local] = global pshell index`, `dense` first then `sparse`.
    pub ids: Vec<usize>,
    pub dense_count: usize,
    /// `slot0[local]..slot0[local+1]` is that pshell's Cartesian slot range.
    pub slot0: Vec<usize>,
    /// `(n_slots, ngrids)` row-major flat values.
    pub values: Vec<f64>,
    pub ngrids: usize,
    pub mesh: [usize; 3],
    /// Grid coordinates and periodic image centres retained only for the
    /// opt-in pass2 radius screen.
    pub coords: Vec<[f64; 3]>,
    pub pshell_rec0: Vec<usize>,
    pub pshell_nrec: Vec<usize>,
    pub rec_center: Vec<[f64; 3]>,
    pub rcut: Vec<f64>,
}

/// Collocate every pshell `level.dense ∪ level.sparse` needs on `level`'s own
/// uniform mesh, gamma point (no Bloch phase — see the crate-level scope
/// note in `numint.rs`).
///
/// # Errors
/// [`PbcDftError`] from grid construction or the kernel's shape checks.
pub fn collocate_level(
    cell: &Cell,
    decon: &Decontracted,
    level: &GridLevel,
) -> Result<LevelValues, PbcDftError> {
    let grids = crate::gen_grid::PeriodicGrids::uniform(cell, Some(level.mesh))?;
    let coords = grids.coords()?;
    let ngrids = coords.len();
    let mut coords_flat = Vec::with_capacity(ngrids * 3);
    for c in coords {
        coords_flat.push(c[0]);
        coords_flat.push(c[1]);
        coords_flat.push(c[2]);
    }

    let mut ids = level.dense.clone();
    ids.extend_from_slice(&level.sparse);
    let dense_count = level.dense.len();

    // Per-pshell image list: the lattice sum out to that pshell's OWN rcut
    // (D-PBC-08's screening idea, at pshell granularity).
    //
    // `discard = false` is LOAD-BEARING, not a default left alone.
    // `get_lattice_ls`'s `discard = true` path drops images "that cannot
    // reach any [atom] pair" (`pyscf_pbc_tools::lattice::get_lattice_ls`'s
    // own doc) — a screen upstream integral drivers use because THEY sum
    // over ATOM-PAIR shell blocks. This call sums a SINGLE pshell's own
    // periodic images against ITSELF (the definition of collocating one
    // Gaussian onto a periodic mesh), which is exactly the case the
    // atom-pair discard heuristic does not recognise as needed — with
    // `discard = true` an off-origin atom (e.g. diamond's second C, at
    // a/4·(1,1,1)) silently lost images whose absence broke
    // `∫ rho dr = Tr(dm.S)` by ~1e-4..1e-3, while an atom AT the origin
    // (where the discarded images happened to be genuinely negligible)
    // was unaffected — this is exactly the failure mode that made the bug
    // easy to miss with an origin-only fixture. Measured and fixed while
    // writing `crates/pyscf-pbc-dft/tests/multigrid.rs`'s
    // `int_rho_matches_tr_dm_s` gate; see that test's isolation trail in
    // the plan's SUMMARY for the debugging steps that found it.
    let mut slot_pow = Vec::new();
    let mut slot_pshell = Vec::new();
    let mut pshell_rec0 = Vec::new();
    let mut pshell_nrec = Vec::new();
    let mut pshell_alpha = Vec::new();
    let mut pshell_coef = Vec::new();
    let mut rec_center = Vec::new();
    let mut screen_center = Vec::new();
    let mut screen_rec0 = Vec::new();
    let mut screen_nrec = Vec::new();
    let mut screen_rcut = Vec::new();
    let mut slot0 = vec![0usize; ids.len() + 1];

    for (local, &pid) in ids.iter().enumerate() {
        let p = &decon.pshells[pid];
        let ls = pyscf_pbc_gto::lattice::get_lattice_ls(cell, Some(p.rcut.max(1e-6)), None, false)?;
        let rec0 = (rec_center.len() / 3) as u32;
        screen_rec0.push(screen_center.len());
        for l in &ls {
            let c = [p.center[0] + l[0], p.center[1] + l[1], p.center[2] + l[2]];
            rec_center.extend_from_slice(&c);
            screen_center.push(c);
        }
        screen_nrec.push(ls.len());
        screen_rcut.push(p.rcut);
        pshell_rec0.push(rec0);
        pshell_nrec.push(ls.len() as u32);
        pshell_alpha.push(p.alpha);
        // The E matrix already carries the contraction coefficient and
        // `common_fac_sp`; the kernel's own `pshell_coef` here is just 1.0
        // per raw Cartesian primitive (`x^ix y^iy z^iz exp(-alpha r^2)`), so
        // the SAME collocated values can be re-scaled per (bra,ket) AO pair
        // through `E`'s sandwich rather than baked once per pshell.
        pshell_coef.push(1.0);

        let powers = pshell_cart_powers(p.l);
        for &(ix, iy, iz) in &powers {
            slot_pow.push(ix);
            slot_pow.push(iy);
            slot_pow.push(iz);
            slot_pshell.push(local as u32);
        }
        slot0[local + 1] = slot0[local] + powers.len();
    }

    let table = PshellGridTable {
        coords: coords_flat,
        slot_pow,
        slot_pshell,
        pshell_rec0,
        pshell_nrec,
        pshell_alpha,
        pshell_coef,
        rec_center,
    };
    let client = pyscf_algebra::select_backend()
        .map_err(|e| {
            PbcDftError::Core(pyscf_core::PyscfRsError::Core(
                pyscf_core::CoreError::InvalidMolecule(format!(
                    "multigrid collocate: backend selection failed: {e}"
                )),
            ))
        })?
        .client;
    let values = collocate(&client, &table).map_err(|e| {
        PbcDftError::Core(pyscf_core::PyscfRsError::Core(
            pyscf_core::CoreError::InvalidMolecule(format!("multigrid collocate: {e}")),
        ))
    })?;

    Ok(LevelValues {
        ids,
        dense_count,
        slot0,
        values,
        ngrids,
        mesh: level.mesh,
        coords: coords.to_vec(),
        pshell_rec0: screen_rec0,
        pshell_nrec: screen_nrec,
        rec_center: screen_center,
        rcut: screen_rcut,
    })
}

/// One flat pair-term: `coeff * values[slot_i,g] * values[slot_j,g]`,
/// `coeff` already carrying `E`'s two contraction/cart2sph factors and the
/// (possibly non-symmetric) density- or identity-matrix entry.
struct Term {
    slot_i: usize,
    slot_j: usize,
    coeff: f64,
}

/// Build the flat pair-term list for `level`'s Part A (`dense x
/// (dense∪sparse)`) + Part B (`sparse x dense`) coverage, weighting each
/// (global cart row, global cart col) entry by `mat[row * nao_p + col]`.
fn pair_terms(lv: &LevelValues, decon: &Decontracted, mat: &[f64]) -> Vec<Term> {
    let mut terms = Vec::new();
    let mut push_block = |i_range: std::ops::Range<usize>, j_range: std::ops::Range<usize>| {
        for i in i_range.clone() {
            let pi = lv.ids[i];
            let (si0, si1) = (lv.slot0[i], lv.slot0[i + 1]);
            let ci0 = decon.pshells[pi].cart_ao0;
            for j in j_range.clone() {
                let pj = lv.ids[j];
                let (sj0, sj1) = (lv.slot0[j], lv.slot0[j + 1]);
                let cj0 = decon.pshells[pj].cart_ao0;
                for (si, ci) in (si0..si1).zip(ci0..ci0 + (si1 - si0)) {
                    for (sj, cj) in (sj0..sj1).zip(cj0..cj0 + (sj1 - sj0)) {
                        let coeff = mat[ci * decon.nao_p + cj];
                        if coeff != 0.0 {
                            terms.push(Term {
                                slot_i: si,
                                slot_j: sj,
                                coeff,
                            });
                        }
                    }
                }
            }
        }
    };
    push_block(0..lv.dense_count, 0..lv.ids.len());
    push_block(lv.dense_count..lv.ids.len(), 0..lv.dense_count);
    terms
}

/// `NUMINT_fill`/`eval_rho` at this level — Task 2/3's forward direction.
///
/// `dm_p` is the FULL decontracted density matrix (`nao_p x nao_p`,
/// row-major); only the entries this level's Part A/B blocks touch are read.
/// Returns `rho(r)` on `lv`'s mesh — one [`oracle_sum`] per grid point, over
/// the FIXED-order pair-term list (D-PBC-17 shape; the term list itself does
/// not depend on which grid points a parallel worker owns, only on the level,
/// so this is bit-identical regardless of how the grid axis is chunked).
pub fn level_rho(lv: &LevelValues, decon: &Decontracted, dm_p: &[f64]) -> Vec<f64> {
    let terms = pair_terms(lv, decon, dm_p);
    let ngrids = lv.ngrids;
    let mut rho = vec![0.0f64; ngrids];
    if terms.is_empty() {
        return rho;
    }
    // W-06-style split: disjoint grid chunks across workers, each computing
    // its own points' FULL fixed-order term list — see the module doc.
    //
    // M-04 step 1: the term buffer is allocated ONCE PER CHUNK instead of once
    // per grid point. It used to be a `vec![0.0; terms.len()]` inside a
    // `par_iter_mut().enumerate()` over every point — `ngrids` heap
    // allocations per level per call (15 625 at `25^3`, 42 875 at `35^3`),
    // where upstream's per-point work allocates nothing at all.
    //
    // BIT-EXACT: every element of `buf` is assigned before it is read, the
    // term list and its order are unchanged, each point still reduces its own
    // full list with the same `oracle_sum`, and the chunking is over DISJOINT
    // outputs so no reduction axis is touched. `CHUNK` therefore cannot appear
    // in any result — it only decides how many allocations happen.
    use rayon::prelude::*;
    const CHUNK: usize = 512;
    rho.par_chunks_mut(CHUNK).enumerate().for_each(|(c, out)| {
        let g0 = c * CHUNK;
        let mut buf = vec![0.0f64; terms.len()];
        for (i, slot) in out.iter_mut().enumerate() {
            let g = g0 + i;
            for (k, t) in terms.iter().enumerate() {
                buf[k] =
                    t.coeff * lv.values[t.slot_i * ngrids + g] * lv.values[t.slot_j * ngrids + g];
            }
            *slot = oracle_sum(&buf);
        }
    });
    rho
}

/// `NUMINT_fill2c`/`eval_mat` at this level — Task 2/3's reverse direction
/// ("pass2"). `weight` is a real-space field on `lv`'s mesh (already carrying
/// the quadrature/cell volume weight, upstream's `wv`). Accumulates into
/// `v_p` (`nao_p x nao_p`, row-major) — ADDS, does not overwrite, so a
/// caller sums every level's contribution into one matrix, matching
/// `_get_j_pass2`'s per-task accumulation.
///
/// Every `(row, col)` entry's grid reduction is an [`oracle_sum`] over the
/// full grid, matching `crate::numint::vxc_mat_one`.
pub fn level_pass2(lv: &LevelValues, decon: &Decontracted, weight: &[f64], v_p: &mut [f64]) {
    debug_assert_eq!(weight.len(), lv.ngrids);
    let ngrids = lv.ngrids;
    // M-04 step 2: ONE `ngrids` buffer for the whole sweep instead of a fresh
    // `vec![0.0; ngrids]` per `(ci, cj)` matrix entry — `nao_p^2` allocations
    // of `ngrids` doubles each, per level, per call.
    //
    // BIT-EXACT: every element is assigned before it is read on every entry,
    // and the `oracle_sum` it feeds sees the identical values in the identical
    // order.
    let mut entries = Vec::new();
    let mut append_block = |i_range: std::ops::Range<usize>, j_range: std::ops::Range<usize>| {
        for i in i_range.clone() {
            let pi = lv.ids[i];
            let (si0, si1) = (lv.slot0[i], lv.slot0[i + 1]);
            let ci0 = decon.pshells[pi].cart_ao0;
            for j in j_range.clone() {
                let pj = lv.ids[j];
                let (sj0, sj1) = (lv.slot0[j], lv.slot0[j + 1]);
                let cj0 = decon.pshells[pj].cart_ao0;
                for (si, ci) in (si0..si1).zip(ci0..ci0 + (si1 - si0)) {
                    for (sj, cj) in (sj0..sj1).zip(cj0..cj0 + (sj1 - sj0)) {
                        entries.push((ci * decon.nao_p + cj, si, sj, i, j));
                    }
                }
            }
        }
    };
    append_block(0..lv.dense_count, 0..lv.ids.len());
    append_block(lv.dense_count..lv.ids.len(), 0..lv.dense_count);

    let screen = std::env::var("PYSCF_PBC_MULTIGRID_PASS2_SCREEN")
        .is_ok_and(|v| matches!(v.as_str(), "1" | "true" | "on" | "yes"));
    let reach = screen.then(|| {
        (0..lv.ids.len())
            .map(|p| {
                let r0 = lv.pshell_rec0[p];
                let r1 = r0 + lv.pshell_nrec[p];
                let cutoff2 = lv.rcut[p] * lv.rcut[p];
                lv.coords
                    .iter()
                    .map(|g| {
                        lv.rec_center[r0..r1].iter().any(|c| {
                            let dx = g[0] - c[0];
                            let dy = g[1] - c[1];
                            let dz = g[2] - c[2];
                            dx * dx + dy * dy + dz * dz <= cutoff2
                        })
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>()
    });

    // M-09: indexed parallel collection preserves the entry order above;
    // `map_init` gives each Rayon worker one reusable grid buffer. Each
    // entry's `oracle_sum` therefore sees exactly the old sequence, and the
    // final `v_p +=` operations happen serially in the old entry order.
    let values: Vec<(usize, f64)> = entries
        .into_par_iter()
        .map_init(
            || vec![0.0f64; ngrids],
            |buf, (idx, si, sj, pi, pj)| {
                for g in 0..ngrids {
                    buf[g] = if reach.as_ref().is_none_or(|mask| mask[pi][g] && mask[pj][g]) {
                        weight[g] * lv.values[si * ngrids + g] * lv.values[sj * ngrids + g]
                    } else {
                        0.0
                    };
                }
                (idx, oracle_sum(buf))
            },
        )
        .collect();
    for (idx, value) in values {
        v_p[idx] += value;
    }
}

/// `dm_p = E . dm . E^T` — contract a contracted-AO density matrix (`nao x
/// nao`, row-major) into the decontracted Cartesian pshell basis.
pub fn expand_dm(decon: &Decontracted, dm: &[f64]) -> Vec<f64> {
    let (nao_p, nao) = (decon.nao_p, decon.nao);
    // tmp = dm . E^T   (nao x nao_p)
    let mut tmp = vec![0.0f64; nao * nao_p];
    for i in 0..nao {
        for k in 0..nao {
            let d = dm[i * nao + k];
            if d == 0.0 {
                continue;
            }
            for p in 0..nao_p {
                tmp[i * nao_p + p] += d * decon.expand[p * nao + k];
            }
        }
    }
    // dm_p = E . tmp   (nao_p x nao_p)
    let mut dm_p = vec![0.0f64; nao_p * nao_p];
    for p in 0..nao_p {
        for i in 0..nao {
            let e = decon.expand[p * nao + i];
            if e == 0.0 {
                continue;
            }
            for q in 0..nao_p {
                dm_p[p * nao_p + q] += e * tmp[i * nao_p + q];
            }
        }
    }
    dm_p
}

/// `v = E^T . v_p . E` — contract a decontracted potential matrix
/// (`nao_p x nao_p`) back into the contracted AO basis (`nao x nao`).
pub fn contract_v(decon: &Decontracted, v_p: &[f64]) -> Vec<f64> {
    let (nao_p, nao) = (decon.nao_p, decon.nao);
    // tmp = v_p . E   (nao_p x nao)
    let mut tmp = vec![0.0f64; nao_p * nao];
    for p in 0..nao_p {
        for q in 0..nao_p {
            let w = v_p[p * nao_p + q];
            if w == 0.0 {
                continue;
            }
            for j in 0..nao {
                tmp[p * nao + j] += w * decon.expand[q * nao + j];
            }
        }
    }
    // v = E^T . tmp   (nao x nao)
    let mut v = vec![0.0f64; nao * nao];
    for p in 0..nao_p {
        for i in 0..nao {
            let e = decon.expand[p * nao + i];
            if e == 0.0 {
                continue;
            }
            for j in 0..nao {
                v[i * nao + j] += e * tmp[p * nao + j];
            }
        }
    }
    v
}
