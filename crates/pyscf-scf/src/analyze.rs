//! `analyze` / `mulliken_pop` / `mulliken_meta` / `dip_moment` — SCF-09.
//!
//! Source: `pyscf/scf/hf.py:1199-1450`.
//!
//! Plan 03-11 ships real bodies for:
//!   - `analyze(rhf)` — log summary + dispatch to mulliken_pop + dip_moment.
//!   - `mulliken_pop(rhf)` — closed-shell Mulliken population analysis.
//!   - `dip_moment(rhf)` — electric dipole moment (uses int1e_r).
//!
//! Plan 03-15 ships the real `mulliken_meta(rhf)` body — meta-Löwdin
//! population analysis (`pyscf/scf/hf.py:1301-1340`). It re-expresses the
//! density in a meta-Löwdin-orthogonalized AO basis (built by
//! `crate::orth::orth_ao`) then reuses the same AO→atom aggregation as
//! `mulliken_pop` (the shared `aggregate_pop_to_charges` helper) with the
//! overlap = identity. The per-`l`-channel meta-Löwdin satisfies the
//! conservation invariants (`Σ ao_pop ≈ nelec`, `Σ chg ≈ molecule charge`);
//! the full NAO core/valence/Rydberg byte-identity is a documented future
//! enhancement (SCF-09 `[~]`).
//!
//! AO→atom mapping: walks `mol._bas` rows (each shell carries an
//! `ATOM_OF` slot per libcint's BAS_SLOTS layout) plus `mol.ao_loc_nr`
//! (cumulative AO offsets per shell) to aggregate per-AO populations
//! onto atoms. This is the same algorithm upstream uses at
//! `pyscf/gto/mole.py:aoslice_by_atom`.

use pyscf_core::PyscfRsError;
use pyscf_core::raw_layout::{ATOM_OF, BAS_SLOTS};

use crate::RHF;
use crate::error::ScfError;

/// Mulliken population-analysis result. Consumed by oracle Arms 7 + 8
/// (plan 03-08).
pub struct MullikenResult {
    /// Per-atom partial charge (Z_A - Σ_{μ ∈ A} pop[μ]).
    pub atom_charges: Vec<f64>,
    /// Per-AO Mulliken population pop[μ] = (D·S)[μ, μ].
    pub ao_populations: Vec<f64>,
}

/// `analyze(rhf)` — log a summary line then dispatch to Mulliken + dipole.
///
/// Source: `pyscf/scf/hf.py:1248-1260`.
pub fn analyze(rhf: &RHF) -> Result<(), PyscfRsError> {
    tracing::info!(
        target: "pyscf_scf::analyze",
        converged = rhf.converged,
        cycles = rhf.cycles,
        e_tot = rhf.e_tot,
        "SCF analysis"
    );
    let _ = mulliken_pop(rhf)?;
    let _ = dip_moment(rhf)?;
    Ok(())
}

/// Closed-shell Mulliken population analysis.
///
/// Algorithm (`pyscf/scf/hf.py:1262-1300`):
///   1. Build D and S.
///   2. AO populations: `pop[μ] = (D · S)[μ, μ]` (reduction via oracle_sum).
///   3. Atom charges: `chg[A] = Z_A - Σ_{μ ∈ A} pop[μ]`.
pub fn mulliken_pop(rhf: &RHF) -> Result<MullikenResult, PyscfRsError> {
    let mo = rhf.mo_coeff.as_ref().ok_or_else(|| {
        ScfError::Core(pyscf_core::CoreError::InvalidMolecule(
            "mulliken_pop: kernel not run yet — mo_coeff is None".into(),
        ))
    })?;
    let dm = crate::rdm::default_make_rdm1(mo)?;
    let s = crate::fock::default_get_ovlp(&rhf.mol)?;
    let nao = dm.nao;
    if s.nao != nao {
        return Err(ScfError::Core(pyscf_core::CoreError::DimensionMismatch {
            expected: nao,
            actual: s.nao,
        })
        .into());
    }

    // AO populations: pop[μ] = Σ_ν D[μ, ν] · S[ν, μ].
    // Both D and S are row-major nao×nao.
    let mut ao_pop = vec![0.0_f64; nao];
    let mut terms = vec![0.0_f64; nao];
    // Flat-array reduction: mu/nu drive row-major D/S offsets + buffers.
    #[allow(clippy::needless_range_loop)]
    for mu in 0..nao {
        for nu in 0..nao {
            terms[nu] = dm.data[mu * nao + nu] * s.data[nu * nao + mu];
        }
        ao_pop[mu] = pyscf_algebra::oracle_sum(&terms);
    }

    // AO→atom aggregation + charges (shared with mulliken_meta).
    aggregate_pop_to_charges(rhf, ao_pop)
}

/// Shared AO→atom aggregator + charge builder.
///
/// Both `mulliken_pop` (`pop[μ] = (D·S)[μ,μ]`) and `mulliken_meta`
/// (`pop[μ] = D'[μ,μ]`, basis orthonormal so `S = I`) produce a per-AO
/// population vector; the downstream aggregation is identical. Extracting it
/// here gives a SINGLE oracle-reduction site to audit (threat T-03-15-NUM)
/// and avoids duplicating the ~40-line `_bas[ATOM_OF]` / `ao_loc_nr` walk.
///
/// Aggregates `ao_pop` onto atoms via `mol._bas[shell, ATOM_OF]` +
/// `mol.ao_loc_nr` (cumulative AO offsets per shell, `oracle_sum` per shell),
/// then `chg[A] = Z_A − pop[A]` from `rhf.mol.atom_charges()`. AO-range /
/// atom-index overruns return `Err` (never panics — T-03-15-PANIC).
fn aggregate_pop_to_charges(rhf: &RHF, ao_pop: Vec<f64>) -> Result<MullikenResult, PyscfRsError> {
    let natm = rhf.mol.natm;
    let nbas = rhf.mol.nbas;
    let nao = ao_pop.len();
    let mut atom_pop = vec![0.0_f64; natm];
    if !rhf.mol._bas.is_empty() && rhf.mol.ao_loc_nr.len() > nbas {
        // Standard path (post mol.build()): ao_loc_nr is `nbas+1` long.
        for shell in 0..nbas {
            let atom = rhf.mol._bas[shell * BAS_SLOTS + ATOM_OF] as usize;
            if atom >= natm {
                return Err(
                    ScfError::Core(pyscf_core::CoreError::InvalidMolecule(format!(
                        "mulliken: _bas[{shell}, ATOM_OF] = {atom} but natm = {natm}"
                    )))
                    .into(),
                );
            }
            let lo = rhf.mol.ao_loc_nr[shell] as usize;
            let hi = rhf.mol.ao_loc_nr[shell + 1] as usize;
            if hi > nao || lo > hi {
                return Err(
                    ScfError::Core(pyscf_core::CoreError::InvalidMolecule(format!(
                        "mulliken: shell {shell} AO range {lo}..{hi} overruns nao {nao}"
                    )))
                    .into(),
                );
            }
            let pop_shell = pyscf_algebra::oracle_sum(&ao_pop[lo..hi]);
            atom_pop[atom] += pop_shell;
        }
    }

    // Atom charges: chg[A] = Z_A - pop[A].
    let charges = rhf.mol.atom_charges();
    let mut atom_charges = vec![0.0_f64; natm];
    for a in 0..natm {
        let z = if a < charges.len() {
            charges[a] as f64
        } else {
            0.0
        };
        atom_charges[a] = z - atom_pop[a];
    }

    Ok(MullikenResult {
        atom_charges,
        ao_populations: ao_pop,
    })
}

/// Meta-Löwdin Mulliken population analysis (SCF-09).
///
/// Source: `pyscf/scf/hf.py:1301-1340`. Re-expresses the density in a
/// meta-Löwdin-orthogonalized AO basis (built by `crate::orth::orth_ao`),
/// then runs the SAME AO→atom aggregation `mulliken_pop` does but with the
/// overlap = identity (the basis is now orthonormal):
///
///   1. `dm`/`s` as in `mulliken_pop`.
///   2. `C_orth = orth_ao(mol, s)` (nao×nao, F-order; columns are the
///      meta-Löwdin AOs).
///   3. `c_inv = C_orthᵀ · S` (nao×nao); `D' = c_inv · D · c_invᵀ` — the
///      density in the orthogonal basis.
///   4. `pop[μ] = D'[μ, μ]` (diagonal, since S = I); aggregate onto atoms and
///      form charges via the shared `aggregate_pop_to_charges`.
///
/// Because `C_orth` is a complete S-orthonormalization
/// (`C_orthᵀ·S·C_orth = I`), the transform preserves the electron count:
/// `Σ pop = Tr(D') = Tr(c_invᵀ·c_inv·D) = Tr(S·D) = nelec` — the conservation
/// invariant the integration test asserts.
///
/// All matrix-product reductions go through `oracle_sum` on materialized term
/// Vecs (threat T-03-15-NUM); no FMA; never panics (missing-mo_coeff guard +
/// dimension checks return `Err`).
pub fn mulliken_meta(rhf: &RHF) -> Result<MullikenResult, PyscfRsError> {
    let mo = rhf.mo_coeff.as_ref().ok_or_else(|| {
        ScfError::Core(pyscf_core::CoreError::InvalidMolecule(
            "mulliken_meta: kernel not run yet — mo_coeff is None".into(),
        ))
    })?;
    let dm = crate::rdm::default_make_rdm1(mo)?;
    let s = crate::fock::default_get_ovlp(&rhf.mol)?;
    let nao = dm.nao;
    if s.nao != nao {
        return Err(ScfError::Core(pyscf_core::CoreError::DimensionMismatch {
            expected: nao,
            actual: s.nao,
        })
        .into());
    }

    // (2) C_orth: nao×nao F-order — c_orth[μ + i*nao].
    let c_orth = crate::orth::orth_ao(&rhf.mol, &s)?;
    if c_orth.len() != nao * nao {
        return Err(ScfError::Core(pyscf_core::CoreError::DimensionMismatch {
            expected: nao * nao,
            actual: c_orth.len(),
        })
        .into());
    }

    // (3a) c_inv = C_orthᵀ · S  (nao×nao).
    //   c_inv[i, ν] = Σ_μ C_orth[μ, i] · S[μ, ν]
    //              = Σ_μ c_orth[μ + i*nao] · s.data[μ*nao + ν]
    // Store c_inv row-major: c_inv[i*nao + ν].
    let mut c_inv = vec![0.0_f64; nao * nao];
    let mut terms = vec![0.0_f64; nao];
    for i in 0..nao {
        for nu in 0..nao {
            for mu in 0..nao {
                terms[mu] = c_orth[mu + i * nao] * s.data[mu * nao + nu];
            }
            c_inv[i * nao + nu] = pyscf_algebra::oracle_sum(&terms);
        }
    }

    // (3b) D' = c_inv · D · c_invᵀ. Build M = c_inv · D first (row-major),
    //   M[i, σ] = Σ_λ c_inv[i, λ] · D[λ, σ]
    //          = Σ_λ c_inv[i*nao + λ] · dm.data[λ*nao + σ]
    // then D'[i, j] = Σ_σ M[i, σ] · c_invᵀ[σ, j] = Σ_σ M[i, σ] · c_inv[j, σ].
    let mut m = vec![0.0_f64; nao * nao];
    for i in 0..nao {
        for sigma in 0..nao {
            for lam in 0..nao {
                terms[lam] = c_inv[i * nao + lam] * dm.data[lam * nao + sigma];
            }
            m[i * nao + sigma] = pyscf_algebra::oracle_sum(&terms);
        }
    }

    // (4) AO populations in the orthogonal basis are the diagonal of D':
    //   pop[i] = D'[i, i] = Σ_σ M[i, σ] · c_inv[i*nao + σ].
    let mut ao_pop = vec![0.0_f64; nao];
    for i in 0..nao {
        for sigma in 0..nao {
            terms[sigma] = m[i * nao + sigma] * c_inv[i * nao + sigma];
        }
        ao_pop[i] = pyscf_algebra::oracle_sum(&terms);
    }

    // AO→atom aggregation + charges (shared with mulliken_pop, S = I here).
    aggregate_pop_to_charges(rhf, ao_pop)
}

/// Electric dipole moment in atomic units (e · Bohr).
///
/// `dip[k] = -Σ_{μν} D[μν] · r[k, μ, ν] + Σ_A Z_A · r_A[k]` for k ∈ {0,1,2}.
///
/// `int1e_r` returns a `(3, nao, nao)` component-leading F-order tensor
/// per `pyscf-gto::layout_table` (`int1e_r_sph` is
/// `ComponentLeadingFOrder { components: 3 }`).
pub fn dip_moment(rhf: &RHF) -> Result<[f64; 3], PyscfRsError> {
    let mo = rhf.mo_coeff.as_ref().ok_or_else(|| {
        ScfError::Core(pyscf_core::CoreError::InvalidMolecule(
            "dip_moment: kernel not run yet — mo_coeff is None".into(),
        ))
    })?;
    let dm = crate::rdm::default_make_rdm1(mo)?;
    let nao = dm.nao;
    let r = pyscf_gto::intor(&rhf.mol, "int1e_r")?;
    let expected = 3 * nao * nao;
    if r.values.len() != expected {
        return Err(ScfError::Core(pyscf_core::CoreError::DimensionMismatch {
            expected,
            actual: r.values.len(),
        })
        .into());
    }

    // Component-leading F-order index for (comp, mu, nu):
    //   r.values[comp + mu * 3 + nu * 3 * nao]
    // (See pyscf-gto::intor::stitch_arity2_block — `out_idx = comp + (oi+ii) *
    //  components + (oj+jj) * components * nao`.)
    let mut dip = [0.0_f64; 3];
    let mut terms = vec![0.0_f64; nao * nao];
    // Flat-array reduction: k/mu/nu drive the component-leading r offsets.
    #[allow(clippy::needless_range_loop)]
    for k in 0..3 {
        for mu in 0..nao {
            for nu in 0..nao {
                let r_idx = k + mu * 3 + nu * 3 * nao;
                terms[mu * nao + nu] = dm.data[mu * nao + nu] * r.values[r_idx];
            }
        }
        let e = pyscf_algebra::oracle_sum(&terms);
        dip[k] = -e;
    }

    // Nuclear contribution: Σ_A Z_A · r_A.
    let charges = rhf.mol.atom_charges();
    for a in 0..rhf.mol.natm {
        let z = if a < charges.len() {
            charges[a] as f64
        } else {
            0.0
        };
        let coords = if a < rhf.mol._atom.len() {
            rhf.mol._atom[a].1
        } else {
            [0.0, 0.0, 0.0]
        };
        for (k, item) in dip.iter_mut().enumerate() {
            *item += z * coords[k];
        }
    }

    Ok(dip)
}
