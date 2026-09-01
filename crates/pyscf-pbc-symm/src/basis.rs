//! Port of `pyscf/pbc/symm/basis.py` (161 l) — symmetry-adapted crystalline
//! AO bases for symmorphic (and non-symmorphic) space groups (`17-04-PLAN.md`).
//!
//! # Why this plan exists (17-CONTEXT §1.2)
//!
//! `PBC-MASTER-PLAN.md §8.9`'s table never mentions `basis.py`, but
//! `khf_ksymm.eig` (`khf_ksymm.py:104-119`) reads `cell.symm_orb` /
//! `cell.irrep_id` directly, and `ksymm_scf_common_init` (`khf_ksymm.py:142`)
//! defaults `use_ao_symmetry = True` — this is the DEFAULT SCF branch, not an
//! opt-in one. This module is the missing producer.
//!
//! # The little co-group is an INPUT, not something this module derives
//!
//! `pyscf.pbc.lib.kpts.KPoints` — the IBZ-folding machinery that computes
//! `little_cogroup_ops` (`kpts.py:1084-1126`) — is plan 17-05, not this one.
//! [`symm_adapted_basis`] and [`build_symmetry`] therefore take exactly the
//! four fields `basis.py:109-130` reads off `kpts` as a plain
//! [`SymmAdaptedBasisInput`] struct (`kpts_scaled_ibz`, `little_cogroup_ops`,
//! `ops`, `dmats` — the last two are `Symmetry::ops` / `Symmetry::dmats`,
//! since `KPoints` inherits from `Symmetry` upstream, `kpts.py: class
//! KPoints(Symmetry)`). When 17-05 lands the real `KPoints`, the expected
//! adaptation is reading these same four fields off it at the call site —
//! the algorithm here does not change.
//!
//! # The per-op phase MUST be threaded through the projector
//!
//! `_get_phase` (`crate::symmetry::get_phase`, `symmetry.py:226`) returns a
//! per-atom phase for each operation. A caller that drops it (e.g. calls
//! `get_phase(..., ignore_phase = true, ...)` or forgets to multiply it in)
//! still gets an ORTHONORMAL-looking basis — orthonormality never depends on
//! getting the phase right, only on the projector being applied consistently
//! — but it is the WRONG basis, and an SCF built on it converges to a
//! different (wrong) state. [`symm_adapted_basis_at_k`] is the ONE place
//! this crate builds the projector, and it always calls
//! [`crate::symmetry::get_phase`] with `ignore_phase = false`.
//!
//! # `Cell::symm_orb` / `Cell::irrep_id` are FLATTENED, not a direct mirror
//!
//! Upstream stores, per IBZ k-point, a Python LIST of per-irrep arrays
//! (`sos`) alongside a parallel list of irrep ids (`irrep_ids`) — `hf_symm.eig`
//! (`pyscf/scf/hf_symm.py:296-336`) iterates `zip(irrep_id[k], symm_orb[k])`
//! and solves one generalized eigenproblem per block. This port instead
//! stores, per k-point, ONE `nao x nao` [`pyscf_algebra::CTensor`] with
//! columns from every surviving irrep concatenated in discovery order, plus
//! ONE irrep id per COLUMN (not per block) in [`pyscf_pbc_gto::Cell::irrep_id`].
//! The two representations carry identical information: a block's column
//! range is exactly the maximal run of equal `irrep_id[k][c]`, since
//! [`symm_adapted_basis_at_k`] appends each irrep's columns as one
//! contiguous group and every irrep index appears in the list at most once.
//! This flattened shape needs no separate per-block width array and is
//! self-sufficient for whatever 17-07's `eig` does with it (re-derive block
//! boundaries with a single linear scan over `irrep_id[k]`).
//!
//! `so`/`symm_orb` matrices are `nao x ncol`, **COLUMN-MAJOR** (F-order) —
//! matching `pyscf-pbc-scf`'s `mo_coeff` convention (`symmetry.rs`'s module
//! doc), since `symm_orb` plays the same "AO -> reduced basis" role
//! `mo_coeff` plays for "AO -> MO". This is a DELIBERATE divergence from
//! `crate::symmetry`'s own row-major convention for SQUARE `nao x nao`
//! rotation/sandwich matrices — `symm_orb` is not square and is consumed
//! downstream as a coefficient matrix, not a rotation.

use num_complex::Complex64;

use pyscf_algebra::CTensor;
use pyscf_pbc_gto::Cell;

use crate::error::PbcSymmError;
use crate::group::{PgElement, PointGroup};
use crate::space_group::{SPGElement, SYMPREC};
use crate::symmetry::{DmatSet, aoslice_by_atom, bas_angular, bas_nctr, get_phase};

/// `basis.py:26` / `:93` — the rank-detection / Gram-Schmidt drop tolerance.
/// A NAMED constant per 17-04-PLAN.md Task 1 (upstream repeats the literal
/// `1e-9` as a default argument in both functions; this port has one source
/// of truth for it).
pub const TOL: f64 = 1e-9;

// ---------------------------------------------------------------------
// Task 1 — _symm_adapted_basis (basis.py:26-91)
// ---------------------------------------------------------------------

/// One surviving irrep block from [`symm_adapted_basis_at_k`]: `so` is
/// `nao x ncol`, COLUMN-MAJOR (see the module doc). Upstream appends a block
/// even when Gram-Schmidt reduces it to zero columns (`basis.py:85-89`: the
/// `if so.shape[-1] > 0:` guard is on the PRE-Gram-Schmidt column count, not
/// the post one) — ported faithfully so `irrep_id` lists stay bit-comparable
/// against upstream (17-04-PLAN.md Task 4's tier-2 oracle test), so `ncol`
/// here can legitimately be `0`.
#[derive(Debug, Clone)]
pub struct IrrepBlock {
    pub irrep_id: i32,
    pub so: CTensor,
    pub ncol: usize,
}

/// `basis.py:26-91` — `_symm_adapted_basis`: project `cell`'s AO basis onto
/// every irrep of the little co-group `pg` at one IBZ k-point (`kpt_scaled`,
/// SCALED/fractional coordinates), and Gram-Schmidt-orthonormalize each
/// surviving block.
///
/// `spg_ops`/`dmats` must be indexed IN LOCKSTEP with `pg.elements` (same
/// order, same length `pg.order()`) — [`symm_adapted_basis`] (Task 3)
/// guarantees this by construction; a caller assembling `pg` by hand must
/// preserve it too.
///
/// # A provably inert simplification (RULE 2 note)
///
/// Upstream first partitions `0..cell.natm` into symmetry orbits
/// (`eql_atom_ids`, `basis.py:37-43`, via a `unique`/`sort`/`unique` dance
/// over `atm_maps`) purely to choose the ITERATION ORDER of the `for iatm`
/// loop below. The loop body writes only to `cbase[:, ..,
/// icol..icol+shell_width]`, a column range that depends solely on `iatm`
/// itself (via `aoslice_by_atom`) and is DISJOINT across atoms — so the
/// final `cbase` does not depend on which order the atoms are visited in,
/// only on every atom being visited exactly once. `eql_atom_ids`'s columns
/// are, by construction, a partition of `0..natm` (each atom lies in exactly
/// one group-orbit), so `for iatm in 0..cell.natm` visits the same atoms,
/// same count, and produces a BIT-IDENTICAL `cbase`. This port therefore
/// skips reconstructing `eql_atom_ids` — not an algorithm change, a removal
/// of scaffolding whose only effect (loop order) is provably unobservable.
///
/// # Errors
/// * Propagates [`get_phase`] (a malformed little co-group whose ops do not
///   actually fix `kpt_scaled`, or a cell whose atoms are not related as the
///   op claims).
/// * [`PbcSymmError::IncompleteBasis`] if the surviving columns across every
///   irrep do not sum to `cell.nao` (`basis.py:90`'s `assert`).
pub fn symm_adapted_basis_at_k(
    cell: &Cell,
    kpt_scaled: [f64; 3],
    pg: &PointGroup,
    spg_ops: &[SPGElement],
    dmats: &[DmatSet],
    tol: f64,
) -> Result<Vec<IrrepBlock>, PbcSymmError> {
    let chartab = pg.character_table(true); // [nirrep][order], per-ELEMENT columns.
    let nirrep = chartab.len();
    let order = pg.order();
    debug_assert_eq!(spg_ops.len(), order, "spg_ops must match pg's element order");
    debug_assert_eq!(dmats.len(), order, "dmats must match pg's element order");
    let nao = cell.nao_nr;
    let natm = cell.natm;

    // atm_maps[iop][iatm], phases[iop][iatm] — basis.py:31-36.
    let mut atm_maps: Vec<Vec<usize>> = Vec::with_capacity(order);
    let mut phases: Vec<Vec<Complex64>> = Vec::with_capacity(order);
    for op in spg_ops {
        // ignore_phase = false: see the module doc — this is the ONE call
        // site that must never drop the phase.
        let (atm_map, phase) = get_phase(cell, op, kpt_scaled, false, SYMPREC)?;
        atm_maps.push(atm_map);
        phases.push(phase);
    }

    let aoslice = aoslice_by_atom(cell);

    // cbase[ir] : nao x nao, ROW-MAJOR complex (an internal accumulator —
    // only the FINAL per-irrep column extraction below is column-major).
    let mut cbase: Vec<Vec<Complex64>> = vec![vec![Complex64::new(0.0, 0.0); nao * nao]; nirrep];

    // basis.py:47-75, minus the eql_atom_ids indirection — see this
    // function's doc.
    for iatm in 0..natm {
        let op_relate_idx: Vec<usize> = (0..order).map(|iop| atm_maps[iop][iatm]).collect();
        let ao_loc: Vec<usize> = op_relate_idx.iter().map(|&j| aoslice[j][2]).collect();

        let b0 = aoslice[iatm][0];
        let b1 = aoslice[iatm][1];
        let mut ioff = 0usize;
        let mut icol = aoslice[iatm][2];
        for ib in b0..b1 {
            let nctr = bas_nctr(cell, ib);
            let l = bas_angular(cell, ib);
            let degen = if cell.cart { (l + 1) * (l + 2) / 2 } else { 2 * l + 1 };

            for n in 0..degen {
                for iop in 0..order {
                    let dmat = &dmats[iop][l]; // degen x degen
                    let phase_val = phases[iop][iatm];
                    // basis.py:70-73: `idx` starts at `ao_loc[iop] + ioff`
                    // for THIS (n, iop) pair and advances by `degen` per
                    // contraction — a fresh running offset each time, not
                    // shared across `iop`.
                    let mut idx = ao_loc[iop] + ioff;
                    for ictr in 0..nctr {
                        let col = icol + n + ictr * degen;
                        for row in 0..degen {
                            let d = dmat[row][n];
                            for ir in 0..nirrep {
                                let fac = (chartab[ir][0] / (order as f64))
                                    * chartab[ir][iop].conj()
                                    * phase_val;
                                cbase[ir][(idx + row) * nao + col] += fac * d;
                            }
                        }
                        idx += degen;
                    }
                }
            }
            ioff += degen * nctr;
            icol += degen * nctr;
        }
    }

    // basis.py:77-91.
    let mut blocks = Vec::with_capacity(nirrep);
    let mut nso = 0usize;
    for ir in 0..nirrep {
        let mut cols: Vec<usize> = Vec::new();
        for col in 0..nao {
            let mut s = 0.0;
            for row in 0..nao {
                s += cbase[ir][row * nao + col].norm();
            }
            if s > tol {
                cols.push(col);
            }
        }
        let ncol = cols.len();
        if ncol == 0 {
            continue;
        }
        let mut re = vec![0.0_f64; nao * ncol];
        let mut im = vec![0.0_f64; nao * ncol];
        let mut im_sum = 0.0_f64;
        for (c, &col) in cols.iter().enumerate() {
            for row in 0..nao {
                let v = cbase[ir][row * nao + col];
                re[c * nao + row] = v.re;
                im[c * nao + row] = v.im;
                im_sum += v.im.abs();
            }
        }
        if im_sum < tol {
            im.iter_mut().for_each(|x| *x = 0.0);
        }
        let so = CTensor { re, im };
        let (so, ncol2) = gram_schmidt(&so, nao, ncol, tol);
        nso += ncol2;
        blocks.push(IrrepBlock { irrep_id: ir as i32, so, ncol: ncol2 });
    }

    if nso != nao {
        return Err(PbcSymmError::IncompleteBasis { expected: nao, got: nso });
    }
    Ok(blocks)
}

// ---------------------------------------------------------------------
// Task 2 — _gram_schmidt (basis.py:93-108)
// ---------------------------------------------------------------------

/// `basis.py:93-108` — `_gram_schmidt`: ordinary modified Gram-Schmidt over
/// the `ncol` columns of `v` (`nao x ncol`, COLUMN-MAJOR), dropping any
/// column whose residual norm falls below `tol` — INCLUDING the very first
/// one if `v`'s first column happens to be zero (upstream never guards
/// `k == 0`; this port doesn't either, matching `basis.py:96` exactly).
/// Preserves the exact input column order (17-04-PLAN.md Task 2): no
/// `HashSet`/`HashMap` anywhere in this path.
///
/// Returns the orthonormalized, possibly-narrower `(CTensor, ncol)`.
pub fn gram_schmidt(v: &CTensor, nao: usize, ncol: usize, tol: f64) -> (CTensor, usize) {
    if ncol == 0 {
        return (CTensor { re: Vec::new(), im: Vec::new() }, 0);
    }

    let get_col = |t: &CTensor, k: usize| -> Vec<Complex64> {
        (0..nao).map(|row| Complex64::new(t.re[k * nao + row], t.im[k * nao + row])).collect()
    };

    let mut u_cols: Vec<Vec<Complex64>> = vec![vec![Complex64::new(0.0, 0.0); nao]; ncol];

    let v0 = get_col(v, 0);
    let norm0 = v0.iter().map(|c| c.norm_sqr()).sum::<f64>().sqrt();
    for row in 0..nao {
        u_cols[0][row] = v0[row] / norm0;
    }

    for k in 1..ncol {
        let mut uk = get_col(v, k);
        for j in 0..k {
            let mut dot = Complex64::new(0.0, 0.0);
            for row in 0..nao {
                dot += u_cols[j][row].conj() * uk[row];
            }
            for row in 0..nao {
                uk[row] -= dot * u_cols[j][row];
            }
        }
        let norm = uk.iter().map(|c| c.norm_sqr()).sum::<f64>().sqrt();
        if norm < tol {
            continue; // u_cols[k] stays all-zero, matching basis.py:102-103.
        }
        for row in 0..nao {
            u_cols[k][row] = uk[row] / norm;
        }
    }

    // basis.py:105-106: drop any (near-)all-zero column, including a
    // skipped one.
    let keep: Vec<usize> = (0..ncol)
        .filter(|&k| u_cols[k].iter().map(|c| c.norm()).sum::<f64>() > tol)
        .collect();
    let ncol2 = keep.len();
    let mut re = vec![0.0_f64; nao * ncol2];
    let mut im = vec![0.0_f64; nao * ncol2];
    for (c, &k) in keep.iter().enumerate() {
        for row in 0..nao {
            re[c * nao + row] = u_cols[k][row].re;
            im[c * nao + row] = u_cols[k][row].im;
        }
    }
    (CTensor { re, im }, ncol2)
}

// ---------------------------------------------------------------------
// Task 3 — symm_adapted_basis / Cell::build_symmetry (basis.py:109-130,
// cell.py:1515-1527)
// ---------------------------------------------------------------------

/// `symm_adapted_basis`/[`build_symmetry`]'s primitive input — see the
/// module doc for why this stands in for `KPoints` until 17-05.
///
/// * `kpts_scaled_ibz[i]` — SCALED (fractional) coordinates of IBZ k-point
///   `i` (`kpts.kpts_scaled_ibz`).
/// * `little_cogroup_ops[i]` — indices into `ops`/`dmats` of the operations
///   that fix IBZ k-point `i` modulo a reciprocal lattice vector
///   (`kpts.little_cogroup_ops`, `kpts.py:1084-1126`; the same subset a
///   direct per-k-point `op.a2b(cell).dot_rot(k) == k (mod 1)` check
///   produces — 17-04-PLAN.md's own note on how to build this without the
///   full `KPoints` machinery).
/// * `ops` — the full space-group operator list (`kpts.ops` ==
///   `Symmetry::ops`).
/// * `dmats` — Wigner-D matrices for every op in `ops`, same order
///   (`kpts.Dmats` == `Symmetry::dmats`).
#[derive(Debug, Clone)]
pub struct SymmAdaptedBasisInput {
    pub kpts_scaled_ibz: Vec<[f64; 3]>,
    pub little_cogroup_ops: Vec<Vec<usize>>,
    pub ops: Vec<SPGElement>,
    pub dmats: Vec<DmatSet>,
}

/// `basis.py:109-130` — `symm_adapted_basis`: [`symm_adapted_basis_at_k`]
/// looped over every IBZ k-point, with the little co-group (subset of
/// `ops`/`dmats`, SORTED by [`PgElement`]'s order — `basis.py:118-125`)
/// assembled at each one.
///
/// # Errors
/// * [`PbcSymmError::KptsSymmInputMismatch`] if `kpts_scaled_ibz` and
///   `little_cogroup_ops` have different lengths, `ops`/`dmats` have
///   different lengths, or some `little_cogroup_ops[i]` names an op index
///   outside `0..ops.len()`.
/// * Propagates [`symm_adapted_basis_at_k`].
pub fn symm_adapted_basis(
    cell: &Cell,
    kpts_scaled_ibz: &[[f64; 3]],
    little_cogroup_ops: &[Vec<usize>],
    ops: &[SPGElement],
    dmats: &[DmatSet],
    tol: f64,
) -> Result<Vec<Vec<IrrepBlock>>, PbcSymmError> {
    if kpts_scaled_ibz.len() != little_cogroup_ops.len() {
        return Err(PbcSymmError::KptsSymmInputMismatch(format!(
            "kpts_scaled_ibz has {} entries but little_cogroup_ops has {}",
            kpts_scaled_ibz.len(),
            little_cogroup_ops.len()
        )));
    }
    if ops.len() != dmats.len() {
        return Err(PbcSymmError::KptsSymmInputMismatch(format!(
            "ops has {} entries but dmats has {}",
            ops.len(),
            dmats.len()
        )));
    }

    let mut sos_ks = Vec::with_capacity(kpts_scaled_ibz.len());
    for (i, ops_idx) in little_cogroup_ops.iter().enumerate() {
        let kpt_scaled = kpts_scaled_ibz[i];

        // basis.py:115-125: elements, sorted; spg_ops/Dmats_small carried
        // along in LOCKSTEP with the same permutation.
        let mut triples: Vec<(PgElement, usize)> = Vec::with_capacity(ops_idx.len());
        for &iop in ops_idx {
            let op = ops.get(iop).ok_or_else(|| {
                PbcSymmError::KptsSymmInputMismatch(format!(
                    "little_cogroup_ops[{i}] references op index {iop}, but ops has {} entries",
                    ops.len()
                ))
            })?;
            let rot: [[i32; 3]; 3] =
                std::array::from_fn(|r| std::array::from_fn(|c| op.rot[r][c].round() as i32));
            triples.push((PgElement::new(rot), iop));
        }
        triples.sort_by(|a, b| a.0.cmp(&b.0));

        let elements: Vec<PgElement> = triples.iter().map(|(e, _)| *e).collect();
        let spg_ops: Vec<SPGElement> = triples.iter().map(|&(_, iop)| ops[iop]).collect();
        let dmats_small: Vec<DmatSet> = triples.iter().map(|&(_, iop)| dmats[iop].clone()).collect();

        let pg = PointGroup::new(elements)?;
        let blocks = symm_adapted_basis_at_k(cell, kpt_scaled, &pg, &spg_ops, &dmats_small, tol)?;
        sos_ks.push(blocks);
    }
    Ok(sos_ks)
}

/// `cell.py:1515-1527` — `Cell._build_symmetry`, the `kpts is not None`
/// branch. A FREE FUNCTION rather than a `Cell` method for the same
/// D-PBC-25 layering reason [`crate::symmetry::build_lattice_symmetry`]
/// documents: `Cell` lives in `pyscf-pbc-gto`, below this crate, so it
/// cannot call [`symm_adapted_basis`] itself.
///
/// Upstream's `else: raise RuntimeError('Symmetry information not found in
/// kpts. kpts needs to be initialized as a KPoints object.')` branch
/// (`cell.py:1526-1527`, reached when `kpts` is neither `None` nor a
/// `KPoints`) has no direct analogue: [`SymmAdaptedBasisInput`] IS the
/// well-typed "this came from a real k-point-symmetry object" shape, so
/// there is no ill-typed value a caller could pass instead. What that guard
/// protects against — silently symmetrizing nothing — is preserved as
/// explicit validation of the input's internal consistency
/// ([`symm_adapted_basis`]'s [`PbcSymmError::KptsSymmInputMismatch`]) and of
/// the completeness identity `_symm_adapted_basis` itself asserts
/// ([`PbcSymmError::IncompleteBasis`], `basis.py:90`, ported as a `Result`
/// rather than a panic).
///
/// On success, sets [`pyscf_pbc_gto::Cell::symm_orb`] /
/// [`pyscf_pbc_gto::Cell::irrep_id`] — see [`Cell::symm_orb`]'s doc and this
/// module's top doc for the flattened (per-column, not per-irrep-block)
/// shape.
///
/// # Errors
/// As [`symm_adapted_basis`].
pub fn build_symmetry(cell: &mut Cell, input: &SymmAdaptedBasisInput) -> Result<(), PbcSymmError> {
    let sos_ks = symm_adapted_basis(
        cell,
        &input.kpts_scaled_ibz,
        &input.little_cogroup_ops,
        &input.ops,
        &input.dmats,
        TOL,
    )?;

    let nao = cell.nao_nr;
    let mut symm_orb = Vec::with_capacity(sos_ks.len());
    let mut irrep_id = Vec::with_capacity(sos_ks.len());
    for blocks in &sos_ks {
        let ncol_total: usize = blocks.iter().map(|b| b.ncol).sum();
        let mut re = vec![0.0_f64; nao * ncol_total];
        let mut im = vec![0.0_f64; nao * ncol_total];
        let mut ids = Vec::with_capacity(ncol_total);
        let mut col_off = 0usize;
        for b in blocks {
            for c in 0..b.ncol {
                for row in 0..nao {
                    re[(col_off + c) * nao + row] = b.so.re[c * nao + row];
                    im[(col_off + c) * nao + row] = b.so.im[c * nao + row];
                }
                ids.push(b.irrep_id);
            }
            col_off += b.ncol;
        }
        symm_orb.push(CTensor { re, im });
        irrep_id.push(ids);
    }

    cell.symm_orb = Some(symm_orb);
    cell.irrep_id = Some(irrep_id);
    Ok(())
}
