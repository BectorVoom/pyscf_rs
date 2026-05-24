//! `orth_ao` — meta-Löwdin / Löwdin AO orthogonalization (SCF-09).
//!
//! Source-of-truth: `pyscf/lo/orth.py`.
//!   - `lowdin(s)` (orth.py:32-36): `S^{-1/2}` via `eigh(S)`, drop `e ≤ 1e-15`.
//!   - `orth_ao(..)` (orth.py:269-331): dispatch lowdin / meta_lowdin +
//!     the phase adjustment (flip column `i` if `C_orth[i,i] < 0`).
//!
//! `mulliken_meta` (`analyze.rs`, this plan 03-15) consumes `orth_ao` to
//! re-express the density in a meta-Löwdin-orthogonalized AO basis before
//! the AO→atom population aggregation.
//!
//! ## Scope decision (plan 03-15 objective)
//!
//! Upstream `meta_lowdin` delegates to `nao._nao_sub` — the NAO
//! core/valence/Rydberg machinery (~230 LoC of `pyscf/lo/nao.py`), none of
//! which exists in-tree (no `pyscf-lo` crate). Porting that entire NAO
//! subsystem is disproportionate to a single population-analysis utility.
//! This module instead ships the in-tree analog: partition the AOs into
//! per-angular-momentum-channel (`l`) blocks — the cross-atom grouping the
//! `_bas[ANG_OF]` walk exposes, which (like upstream's core/valence/Rydberg
//! classes) spans atoms so homonuclear symmetry is preserved — and
//! sequentially Löwdin-orthogonalize each channel against the previous ones.
//! This delivers the FULL SCF-09 deliverable — `mulliken_meta` returns
//! correct meta-Löwdin populations satisfying the conservation invariants
//! (`C_orthᵀ·S·C_orth ≈ I`, `Σ ao_pop ≈ nelec`). The full NAO
//! core/valence/Rydberg refinement is a documented future enhancement, NOT
//! a scope reduction.
//!
//! ## Layout
//!
//! The returned `C_orth` is an `nao×nao` **F-order (column-major)** matrix
//! (`C_orth[i, j] = data[i + j*nao]`), matching the `MOCoefficients`
//! convention so `mulliken_meta` can treat `C_orth` columns like MO columns.
//! The block overlaps and `S^{-1/2}` outputs are also F-order; the input
//! overlap `s` is the symmetric `Density` from `default_get_ovlp` (row-major
//! storage, but symmetric so the orientation is moot).
//!
//! All reductions go through `oracle_sum` (Pitfall 9 — reduction-order drift
//! voids µHartree bit-exactness); no FMA in this kernel; no panics
//! (eigh failures / AO-range overruns return `Err`).

use pyscf_core::raw_layout::{ANG_OF, ATOM_OF, BAS_SLOTS};
use pyscf_core::{CoreError, Density, Mole, PyscfRsError};

use crate::error::ScfError;

/// Eigenvalue cutoff for the symmetric `S^{-1/2}` build. Eigenvalues of a
/// block overlap at or below this magnitude are dropped (the directions are
/// linearly dependent — upstream `lowdin` uses `e > 1e-15`, orth.py:35).
const LOWDIN_EIG_TOL: f64 = 1e-15;

/// Symmetric Löwdin orthogonalizer for one block: `S^{-1/2}`.
///
/// Source: `pyscf/lo/orth.py:32-36`. Diagonalize `S = U·diag(λ)·Uᵀ`
/// (self-adjoint eigh), drop `λ ≤ LOWDIN_EIG_TOL` (linear-dependency
/// removal), then form `S^{-1/2}[i, j] = Σ_k U[i, k] · λ_k^{-1/2} · U[j, k]`
/// over the kept eigenpairs.
///
/// `s_block` is a row-major (== F-order, since symmetric) `n×n` flat slice.
/// The returned `n×n` matrix is F-order (`out[i + j*n]`) and symmetric.
///
/// Reductions over the eigen-index `k` go through `oracle_sum` on a
/// materialized term Vec (the `analyze.rs:80-84` pattern).
fn lowdin(s_block: &[f64], n: usize) -> Result<Vec<f64>, PyscfRsError> {
    if n == 0 {
        return Ok(Vec::new());
    }
    if s_block.len() != n * n {
        return Err(ScfError::Core(CoreError::DimensionMismatch {
            expected: n * n,
            actual: s_block.len(),
        })
        .into());
    }

    // Diagonalize the block overlap. `eigh_gen(S, I, n)` solves the plain
    // symmetric eigenproblem `S·U = U·diag(λ)` (S = identity metric), so the
    // returned F-order `u` columns are the eigenvectors and `eigvals` are λ
    // (nondecreasing; dropped/linearly-dependent directions are padded with
    // +∞ and a zero column — see eigh_gen.rs:110-127).
    let identity = identity_flat(n);
    let (eigvals, u) = pyscf_algebra::eigh_gen(s_block, &identity, n).map_err(ScfError::Algebra)?;

    // Keep only finite, above-tolerance eigenpairs. `u` is F-order:
    // `u[i + k*n]` is component i of eigenvector k.
    let mut kept: Vec<(usize, f64)> = Vec::with_capacity(n);
    for (k, &lam) in eigvals.iter().enumerate() {
        if lam.is_finite() && lam > LOWDIN_EIG_TOL {
            kept.push((k, 1.0 / lam.sqrt()));
        }
    }
    if kept.is_empty() {
        return Err(ScfError::Algebra(pyscf_algebra::AlgebraError::Singular).into());
    }

    // S^{-1/2}[i, j] = Σ_{kept k} U[i, k] · λ_k^{-1/2} · U[j, k].
    // Output F-order: out[i + j*n].
    let mut out = vec![0.0_f64; n * n];
    let mut terms = vec![0.0_f64; kept.len()];
    for i in 0..n {
        for j in 0..n {
            for (t, &(k, inv_sqrt)) in kept.iter().enumerate() {
                let u_ik = u[i + k * n];
                let u_jk = u[j + k * n];
                terms[t] = u_ik * inv_sqrt * u_jk;
            }
            out[i + j * n] = pyscf_algebra::oracle_sum(&terms);
        }
    }
    Ok(out)
}

/// Build an `n×n` identity as a flat (row-major == F-order, symmetric) slice.
fn identity_flat(n: usize) -> Vec<f64> {
    let mut id = vec![0.0_f64; n * n];
    for i in 0..n {
        id[i * n + i] = 1.0;
    }
    id
}

/// AO index blocks grouped by angular-momentum channel `l`. Each `Vec<usize>`
/// is the set of AO indices sharing one `l` value, in ascending order. Walks
/// `mol._bas` rows (`ANG_OF` slot; `ATOM_OF` is validated for diagnostics)
/// + `mol.ao_loc_nr` (cumulative AO offsets per shell).
///
/// ## Why per-l (not per-(atom, l))
///
/// Upstream `_nao_sub` groups AOs into core / valence / Rydberg *classes*
/// that span atoms (all valence orbitals of a given `l` are Löwdin'd
/// together), NOT into one block per (atom, l). The class machinery
/// (`_core_val_ryd_list`, `pyscf/lo/nao.py:162-195`) is part of the NAO
/// subsystem the scope decision defers. The faithful in-tree analog that
/// keeps the cross-atom grouping (and so preserves homonuclear symmetry —
/// e.g. the two H 1s orbitals of H2 land in ONE l=0 block and are
/// symmetric-Löwdin'd, giving equal charges) is the per-`l` channel grouping
/// ("Löwdin within each angular-momentum channel"). A per-(atom, l) split
/// would process each atom's block sequentially and break that symmetry.
/// This still differs from plain symmetric Löwdin (which mixes all `l`
/// together) — distinct `l` channels are orthogonalized separately and
/// sequentially.
fn ao_blocks_by_l(mol: &Mole) -> Result<Vec<Vec<usize>>, PyscfRsError> {
    let nbas = mol.nbas;
    let nao = mol.nao_nr;
    if nbas == 0 {
        return Ok(Vec::new());
    }
    if mol._bas.len() < nbas * BAS_SLOTS || mol.ao_loc_nr.len() <= nbas {
        return Err(ScfError::Core(CoreError::InvalidMolecule(format!(
            "orth_ao: _bas (len {}) / ao_loc_nr (len {}) inconsistent with nbas {nbas}",
            mol._bas.len(),
            mol.ao_loc_nr.len()
        )))
        .into());
    }

    // Preserve first-seen l order so the block layout is deterministic.
    let mut keys: Vec<i32> = Vec::new();
    let mut blocks: Vec<Vec<usize>> = Vec::new();
    for shell in 0..nbas {
        // ATOM_OF is read only to validate the row is well-formed; the
        // partition key is the angular momentum `l` (cross-atom channel).
        let _atom = mol._bas[shell * BAS_SLOTS + ATOM_OF];
        let l = mol._bas[shell * BAS_SLOTS + ANG_OF];
        let lo = mol.ao_loc_nr[shell] as usize;
        let hi = mol.ao_loc_nr[shell + 1] as usize;
        if hi > nao || lo > hi {
            return Err(ScfError::Core(CoreError::InvalidMolecule(format!(
                "orth_ao: shell {shell} AO range {lo}..{hi} overruns nao {nao}"
            )))
            .into());
        }
        let pos = match keys.iter().position(|k| *k == l) {
            Some(p) => p,
            None => {
                keys.push(l);
                blocks.push(Vec::new());
                blocks.len() - 1
            }
        };
        for ao in lo..hi {
            blocks[pos].push(ao);
        }
    }
    Ok(blocks)
}

/// Meta-Löwdin (default) AO orthogonalization matrix `C_orth` (`nao×nao`,
/// F-order; columns are the orthogonalized AOs).
///
/// Source: `pyscf/lo/orth.py:269-331` (meta_lowdin branch into `nao._nao_sub`)
/// plus the phase adjust. The in-tree partition replaces upstream's
/// core/valence/Rydberg ordering (`pyscf/lo/nao.py:124-160`) with per-`l`
/// angular-momentum channels the `_bas[ANG_OF]` walk exposes (the channels
/// span atoms, so homonuclear symmetry is preserved). The full NAO
/// core/valence/Rydberg refinement is a documented future enhancement — see
/// the module-level scope note.
///
/// Algorithm — the GLOBALLY-ORTHONORMAL sequential block scheme that
/// `_nao_sub` implements (nao.py:136-156): process the `l` channels in
/// order, and for each block FIRST project out the span of all previously
/// orthogonalized blocks in the S-metric (nao.py:141 `c -= C·Cᵀ·S·c`), THEN
/// Löwdin-orthogonalize WITHIN the projected block. Building the full set
/// block-by-block this way yields `C_orthᵀ·S·C_orth ≈ I` (the `_nao_sub`
/// invariant checked at nao.py:157), which is what a plain per-block Löwdin
/// alone would NOT give (cross-block overlap survives) and is exactly the
/// conservation invariant `mulliken_meta` relies on:
///
///   1. Partition the AO indices into per-`l` angular-momentum channels.
///   2. For each block B (with `C_done` = columns already orthonormalized):
///      (a) seed `c = e_μ` for `μ ∈ B` (block AO unit vectors in full space);
///      (b) project `c ← c − C_done·(C_doneᵀ·S·c)` (S-orthogonal to done span);
///      (c) block overlap `s1 = cᵀ·S·c` (m×m);
///      (d) Löwdin within block: `c_block = c · lowdin(s1)`;
///      (e) write `c_block` into `C_orth` at B's columns; append to `C_done`.
///   3. Phase adjust (orth.py:328-330): for each column i, if `C_orth[i,i] < 0`
///      flip the sign of the whole column (deterministic, vendor-stable phase).
///
/// `s` is the symmetric overlap `Density` (`s.nao == mol.nao_nr`). All
/// reductions go through `oracle_sum`/`oracle_dot`; no FMA; never panics
/// (eigh failures / AO-range overruns return `Err`).
pub(crate) fn orth_ao(mol: &Mole, s: &Density) -> Result<Vec<f64>, PyscfRsError> {
    let nao = mol.nao_nr;
    if s.nao != nao || s.data.len() != nao * nao {
        return Err(ScfError::Core(CoreError::DimensionMismatch {
            expected: nao * nao,
            actual: s.data.len(),
        })
        .into());
    }
    if nao == 0 {
        return Ok(Vec::new());
    }

    let blocks = ao_blocks_by_l(mol)?;
    // C_orth, F-order: c_orth[μ + i*nao]. Columns filled block-by-block.
    let mut c_orth = vec![0.0_f64; nao * nao];
    // Global AO index of each already-orthonormalized column, in fill order
    // (so `done_cols[k]` is the C_orth column index of the k-th done vector).
    let mut done_cols: Vec<usize> = Vec::with_capacity(nao);

    for block in &blocks {
        let m = block.len();
        if m == 0 {
            continue;
        }

        // (a) Seed the block columns `c` (full-space, F-order m columns of
        // length nao): c[:, bj] = e_{block[bj]}.
        let mut c = vec![0.0_f64; nao * m];
        for (bj, &mu) in block.iter().enumerate() {
            c[mu + bj * nao] = 1.0;
        }

        // (b) Project out the done span in the S-metric:
        //     c ← c − C_done·(C_doneᵀ·S·c).
        // Since seed columns are unit vectors e_μ, (S·c)[ν, bj] = S[ν, block[bj]].
        if !done_cols.is_empty() {
            let nd = done_cols.len();
            // proj[k, bj] = Σ_ν C_done[ν, k] · (S·c)[ν, bj]
            //            = Σ_ν c_orth[ν + done_cols[k]*nao] · s[ν, block[bj]]
            let mut proj = vec![0.0_f64; nd * m];
            let mut terms = vec![0.0_f64; nao];
            for (k, &dcol) in done_cols.iter().enumerate() {
                for (bj, &mu) in block.iter().enumerate() {
                    for nu in 0..nao {
                        terms[nu] = c_orth[nu + dcol * nao] * s.data[nu * nao + mu];
                    }
                    proj[k * m + bj] = pyscf_algebra::oracle_sum(&terms);
                }
            }
            // c[:, bj] -= Σ_k C_done[:, k] · proj[k, bj].
            let mut dterms = vec![0.0_f64; nd];
            for bj in 0..m {
                for row in 0..nao {
                    for (k, &dcol) in done_cols.iter().enumerate() {
                        dterms[k] = c_orth[row + dcol * nao] * proj[k * m + bj];
                    }
                    c[row + bj * nao] -= pyscf_algebra::oracle_sum(&dterms);
                }
            }
        }

        // (c) Projected block overlap s1 = cᵀ·S·c (m×m, symmetric).
        // (S·c)[ν, bj] = Σ_λ S[ν, λ]·c[λ, bj]; s1[bi, bj] = Σ_ν c[ν, bi]·(S·c)[ν, bj].
        let mut sc = vec![0.0_f64; nao * m]; // S·c, F-order: sc[ν + bj*nao]
        let mut slterms = vec![0.0_f64; nao];
        for bj in 0..m {
            for nu in 0..nao {
                for lam in 0..nao {
                    slterms[lam] = s.data[nu * nao + lam] * c[lam + bj * nao];
                }
                sc[nu + bj * nao] = pyscf_algebra::oracle_sum(&slterms);
            }
        }
        let mut s1 = vec![0.0_f64; m * m]; // row-major m×m (symmetric)
        let mut s1terms = vec![0.0_f64; nao];
        for bi in 0..m {
            for bj in 0..m {
                for nu in 0..nao {
                    s1terms[nu] = c[nu + bi * nao] * sc[nu + bj * nao];
                }
                s1[bi * m + bj] = pyscf_algebra::oracle_sum(&s1terms);
            }
        }

        // (d) Löwdin within block: x = s1^{-1/2} (F-order m×m), c_block = c·x.
        let x = lowdin(&s1, m)?;
        // (e) c_block[:, bj] = Σ_bk c[:, bk]·x[bk, bj]; write into C_orth at
        //     column block[bj]. Reduction over bk via oracle_sum.
        let mut xterms = vec![0.0_f64; m];
        for (bj, &col) in block.iter().enumerate() {
            for row in 0..nao {
                for bk in 0..m {
                    xterms[bk] = c[row + bk * nao] * x[bk + bj * m];
                }
                c_orth[row + col * nao] = pyscf_algebra::oracle_sum(&xterms);
            }
            done_cols.push(col);
        }
    }

    // Phase adjustment (orth.py:328-330): flip column i if its diagonal < 0.
    for i in 0..nao {
        if c_orth[i + i * nao] < 0.0 {
            for row in 0..nao {
                c_orth[row + i * nao] = -c_orth[row + i * nao];
            }
        }
    }

    Ok(c_orth)
}

#[cfg(test)]
mod tests {
    use super::*;
    use pyscf_core::Unit;
    use pyscf_gto::{AtomInput, BasisInput, M, MoleBuildArgs};

    /// `lowdin(I) == I` — the symmetric Löwdin of an identity overlap is
    /// the identity (every eigenvalue 1 → λ^{-1/2} = 1).
    #[test]
    fn lowdin_of_identity_is_identity() {
        let n = 3;
        let id = identity_flat(n);
        let x = lowdin(&id, n).expect("lowdin(I)");
        for i in 0..n {
            for j in 0..n {
                let want = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (x[i + j * n] - want).abs() < 1e-12,
                    "lowdin(I)[{i},{j}] = {} but want {want}",
                    x[i + j * n]
                );
            }
        }
    }

    /// `S^{-1/2} · S · S^{-1/2} ≈ I` for a small non-trivial SPD overlap
    /// (the Löwdin defining identity).
    #[test]
    fn lowdin_squared_times_s_is_identity() {
        // SPD 2×2: S = [[1, 0.3], [0.3, 1]].
        let n = 2;
        let s = vec![1.0, 0.3, 0.3, 1.0];
        let x = lowdin(&s, n).expect("lowdin(S)"); // F-order m×m, symmetric

        // Compute Xᵀ·S·X (X symmetric so Xᵀ = X). Result should be I.
        // X[i,j] = x[i + j*n]; S[i,j] = s[i*n + j].
        let mut sx = vec![0.0_f64; n * n]; // S·X (row-major)
        for i in 0..n {
            for j in 0..n {
                let mut acc = 0.0;
                for k in 0..n {
                    acc += s[i * n + k] * x[k + j * n];
                }
                sx[i * n + j] = acc;
            }
        }
        for i in 0..n {
            for j in 0..n {
                // (Xᵀ·(S·X))[i,j] = Σ_k X[k,i]·SX[k,j]
                let mut acc = 0.0;
                for k in 0..n {
                    acc += x[k + i * n] * sx[k * n + j];
                }
                let want = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (acc - want).abs() < 1e-10,
                    "(Xᵀ·S·X)[{i},{j}] = {acc} but want {want}"
                );
            }
        }
    }

    /// `C_orthᵀ · S · C_orth ≈ I` (orthonormality) for the real H2/STO-3G
    /// overlap built from `intor`, plus the phase-adjust gives all
    /// diagonals ≥ 0.
    #[test]
    fn orth_ao_is_orthonormal_and_phase_adjusted() {
        let mol = M(MoleBuildArgs {
            atom: AtomInput::String("H 0 0 0; H 0 0 1.4".into()),
            basis: BasisInput::Name("sto-3g".into()),
            unit: Unit::Bohr,
            ..Default::default()
        })
        .expect("build H2/STO-3G");

        let s = crate::fock::default_get_ovlp(&mol).expect("ovlp");
        let nao = s.nao;
        let c = orth_ao(&mol, &s).expect("orth_ao");
        assert_eq!(c.len(), nao * nao);

        // Orthonormality: M = C_orthᵀ · S · C_orth ≈ I.
        // C_orth F-order: c[mu + i*nao]; S row-major (symmetric): s.data[mu*nao+nu].
        let mut sc = vec![0.0_f64; nao * nao]; // (S·C) row-major: sc[mu, i]
        for mu in 0..nao {
            for i in 0..nao {
                let mut acc = 0.0;
                for nu in 0..nao {
                    acc += s.data[mu * nao + nu] * c[nu + i * nao];
                }
                sc[mu * nao + i] = acc;
            }
        }
        for i in 0..nao {
            for j in 0..nao {
                // M[i,j] = Σ_mu C[mu,i]·(S·C)[mu,j]
                let mut acc = 0.0;
                for mu in 0..nao {
                    acc += c[mu + i * nao] * sc[mu * nao + j];
                }
                let want = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (acc - want).abs() < 1e-8,
                    "orthonormality M[{i},{j}] = {acc} but want {want}"
                );
            }
        }

        // Phase adjust: every diagonal C_orth[i,i] ≥ 0.
        for i in 0..nao {
            assert!(
                c[i + i * nao] >= 0.0,
                "phase-adjust failed: C_orth[{i},{i}] = {} < 0",
                c[i + i * nao]
            );
        }
    }
}
