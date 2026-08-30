//! DFT-05: range-coulomb `env[8]` (`PTR_RANGE_OMEGA`) set/restore on the
//! `mol.intor` path — the `mol.with_range_coulomb(omega)`-equivalent.
//!
//! Source-of-truth references:
//!   - `pyscf/gto/mole.py:with_range_coulomb` — the env-slot set/restore
//!     context manager: `mol._env[PTR_RANGE_OMEGA] = omega` on enter,
//!     restore the prior value on exit.
//!   - `pyscf/dft/rks.py:108-129` — the `get_veff` RSH branch that toggles
//!     `+omega` (long-range `erf(ωr)/r`) / `-omega` (short-range complement)
//!     around the standard `int2e`/`get_k` call.
//!   - 04-RESEARCH.md Pitfall 1 + Pattern 4 + Open Question A5.
//!
//! ## RESEARCH CORRECTION (Pitfall 1 — honored here)
//! Range-separated hybrids do **NOT** use distinct `int2e_lr_*`/`int2e_sr_*`
//! integral symbols (those do not exist in cintx/libcint). The mechanism is:
//! set `PTR_RANGE_OMEGA = env[8]` (positive ⇒ long-range `erf(ωr)/r`,
//! negative ⇒ short-range complement), call the **standard** `int2e`, then
//! restore `env[8]`. cintx reads `env[8]` (verified: `cintx-compat::raw`
//! documents `PTR_RANGE_OMEGA = 8`). This module therefore wraps a standard
//! [`crate::intor`] call between a guarded env-slot mutation; it introduces
//! NO new intor symbol.
//!
//! ## Open Question A5 — CLOSED, and the fix was not the env slot
//!
//! The original finding was right: cintx *reads* `env[8]` on its
//! `cintx-compat::raw` path, but its **safe API** takes its parameters from
//! `ExecutionOptions`, and `intor` builds cintx's `BasisSet` from
//! `mol._atom`/`_basis` — it never hands cintx the caller's `_env`. So writing
//! `mol._env[PTR_RANGE_OMEGA]` and calling `intor` set a slot nobody read: the
//! call returned the FULL-RANGE operator under a range-separated request. It
//! ran, it converged, and it was silently a different method.
//!
//! **D-PBC-24 closed the capability** — `ExecutionOptions::range_omega` /
//! `SessionBuilder::with_range_omega`, honoured end to end for `int2e`,
//! `int3c2e` and `int2c2e` and gated against vendored libcint 6.1.3 at 3.4e-14.
//! [`intor_with_omega`] therefore threads ω through
//! [`crate::intor_with_options`] as well as setting the env slot, and the
//! numerical RSH ERI is live rather than CI-gated.
//!
//! The [`OmegaGuard`] stays. `mol._env[8]` is the compat surface this port
//! promises (`with_range_coulomb`'s observable effect, asserted by
//! `tests/range_coulomb_env.rs`), and any consumer that reaches cintx through
//! `cintx-compat::raw` rather than the safe API still reads it. Both halves are
//! set from the same argument, so they cannot disagree.
//!
//! ## Fail closed, not full range
//!
//! cintx's safe API refuses a non-zero ω on any operator it has not implemented
//! range separation for, rather than evaluating the full-range kernel. So
//! `intor_with_omega(mol, "int1e_ovlp", -0.8)` is now an ERROR where it used to
//! return full-range overlaps. That is the intended contract: upstream's
//! `with_range_coulomb` block is harmless around a 1e integral only because
//! libcint's `g1e.c` ignores the slot, and a caller who cannot tell the two
//! apart is one refactor away from a wrong number. `omega == 0.0` is the full
//! Coulomb operator and is accepted everywhere.

use pyscf_core::{Density, Mole, PyscfRsError};

use cintx_runtime::ExecutionOptions;

use crate::intor::{IntorOutput, intor_with_options};

/// libcint global-param slot for the range-separation omega
/// (`PTR_RANGE_OMEGA = 8`). Defined locally because `pyscf_core::raw_layout`
/// re-exports `PTR_ZETA`/`PTR_EXP`/… but NOT `PTR_RANGE_OMEGA` (cintx-compat
/// documents the constant at `raw.rs:35` but does not surface it through the
/// `pyscf_core::raw_layout` façade). Matches `cintx-compat::raw`'s value.
pub const PTR_RANGE_OMEGA: usize = 8;

/// RAII guard that sets `mol._env[PTR_RANGE_OMEGA] = omega` on construction
/// and **restores the prior value on drop** — even if the wrapped operation
/// panics or returns `Err` (T-04-07a mitigation: no omega leak across intor
/// calls). Mirrors the `mol.with_range_coulomb(omega)` context manager.
///
/// Holds `&mut Mole` (not just `&mut _env`) so it can be constructed,
/// surrender a shared `&Mole` for the wrapped `intor` call via
/// [`OmegaGuard::mol`], and still restore `_env[8]` on drop. The `_env` slot
/// is reserved by `make_env` (the first `PTR_ENV_START = 20` slots are
/// libcint-global zeros), so index 8 is always present after a `build()`.
struct OmegaGuard<'m> {
    mol: &'m mut Mole,
    prior: f64,
}

impl<'m> OmegaGuard<'m> {
    /// Set `mol._env[PTR_RANGE_OMEGA] = omega`, remembering the prior value
    /// so [`Drop`] can restore it. Returns an error if `_env` is too short to
    /// hold the global-param slots (an unbuilt / malformed Mole).
    fn new(mol: &'m mut Mole, omega: f64) -> Result<Self, PyscfRsError> {
        if mol._env.len() <= PTR_RANGE_OMEGA {
            return Err(PyscfRsError::Core(pyscf_core::CoreError::InvalidMolecule(
                format!(
                    "with_range_coulomb: _env has {} slots, need > {PTR_RANGE_OMEGA} \
                     (Mole not built — call mol.build() / pyscf_gto::M first)",
                    mol._env.len()
                ),
            )));
        }
        let prior = mol._env[PTR_RANGE_OMEGA];
        mol._env[PTR_RANGE_OMEGA] = omega;
        Ok(Self { mol, prior })
    }

    /// A shared borrow of the guarded Mole (env[8] already set to omega) for
    /// the wrapped `intor` call. The guard outlives the returned reference,
    /// so `_env[8]` is restored after the caller finishes using it.
    fn mol(&self) -> &Mole {
        self.mol
    }
}

impl Drop for OmegaGuard<'_> {
    fn drop(&mut self) {
        // Restore the prior omega — runs on the normal path AND on unwind /
        // early `?` return (T-04-07a: env[8] never leaks across calls).
        self.mol._env[PTR_RANGE_OMEGA] = self.prior;
    }
}

/// Run a named integral with the range-separation omega set in `env[8]`,
/// restoring the prior value afterwards (`mol.with_range_coulomb(omega)`
/// equivalent). `omega > 0` selects the long-range `erf(ωr)/r` operator,
/// `omega < 0` the short-range complement, `omega == 0` the standard ERI.
///
/// This takes `&mut Mole` because the omega lives in the Mole's own `_env`
/// flat array (D-03); the guard restores it on return. After restore, the
/// Mole's `_env[8]` is byte-identical to its value on entry, so subsequent
/// (non-ranged) intor calls are unaffected (tested in `range_coulomb_env`).
///
/// ω reaches the integral through cintx's [`ExecutionOptions::range_omega`],
/// NOT through the env slot the guard writes: `intor` builds cintx's
/// `BasisSet` from `mol._atom`/`_basis` and never hands cintx this `_env`, so
/// the slot alone would have been inert (see the module docs). Both are set
/// from the same `omega` argument, so the compat surface and the evaluated
/// operator cannot disagree.
///
/// # Errors
/// As [`crate::intor`], plus a typed refusal from cintx when `omega != 0.0` is
/// asked of an operator with no range-separated kernel — a refusal, never a
/// full-range substitution.
pub fn intor_with_omega(
    mol: &mut Mole,
    name: &str,
    omega: f64,
) -> Result<IntorOutput, PyscfRsError> {
    let guard = OmegaGuard::new(mol, omega)?;
    // Standard intor — NOT an `int2e_lr_*`/`int2e_sr_*` symbol (Pitfall 1).
    // `guard.mol()` is a shared borrow of the omega-set Mole; the guard
    // restores env[8] when it drops at the end of this scope, including on
    // the `?`-propagated error path below.
    intor_with_options(
        guard.mol(),
        name,
        ExecutionOptions {
            range_omega: Some(omega),
            ..ExecutionOptions::default()
        },
    )
    // guard drops here → env[8] restored.
}

/// `get_k_with_omega(mol, dm, omega)` — the range-separated exact-exchange
/// K matrix used by the `get_veff` RSH branch (`rks.py:108-129`). Sets
/// `env[8] = omega` around the standard `int2e`/K build and restores it.
///
/// `omega > 0` ⇒ long-range K (`erf(ωr)/r`); `omega < 0` ⇒ short-range K.
///
/// Implementation: route through the SCF exact-exchange K builder
/// (`pyscf_scf::default_get_jk`, which calls `intor(mol, "int2e")`) inside
/// the omega guard, returning only the K matrix. We cannot call the SCF
/// crate from `pyscf-gto` (it sits below SCF in the dep graph), so this
/// helper builds K from the ranged `int2e` directly — the same 8-fold-
/// symmetric `K[μν] = Σ eri[μλ,νσ] D[λσ]` contraction `fock.rs:108-130`
/// uses, but on the omega-screened ERIs.
///
/// Returns the K matrix as a [`Density`]. Since D-PBC-24 the ERIs it contracts
/// really are the ranged ones — before that this built a full-range K under a
/// range-separated name.
pub fn get_k_with_omega(mol: &mut Mole, dm: &Density, omega: f64) -> Result<Density, PyscfRsError> {
    let nao = mol.nao_nr;
    if dm.nao != nao {
        return Err(PyscfRsError::Core(
            pyscf_core::CoreError::DimensionMismatch {
                expected: nao,
                actual: dm.nao,
            },
        ));
    }
    // Ranged ERIs: standard int2e with env[8] = omega (set/restored by the
    // guard inside intor_with_omega). Pitfall 1: NOT int2e_lr_/int2e_sr_.
    let eri = intor_with_omega(mol, "int2e", omega)?;
    let expected = nao * nao * nao * nao;
    if eri.values.len() != expected {
        return Err(PyscfRsError::Core(
            pyscf_core::CoreError::DimensionMismatch {
                expected,
                actual: eri.values.len(),
            },
        ));
    }
    // K[μν] = Σ_{λσ} eri[μ,λ,ν,σ] · D[λ,σ] — the same 8-fold-symmetric
    // contraction as fock.rs:108-130, on the omega-screened ERIs.
    let n2 = nao * nao;
    let n3 = n2 * nao;
    let mut k_data = vec![0.0_f64; nao * nao];
    let mut k_terms = vec![0.0_f64; nao * nao];
    for mu in 0..nao {
        for nu in 0..nao {
            for lambda in 0..nao {
                for sigma in 0..nao {
                    let i_k = mu + lambda * nao + nu * n2 + sigma * n3;
                    let d_ls = dm.data[lambda * nao + sigma];
                    k_terms[lambda * nao + sigma] = eri.values[i_k] * d_ls;
                }
            }
            // Reduction over (λ, σ) — Pitfall 9 (oracle_sum, bit-exact).
            k_data[mu * nao + nu] = pyscf_algebra::oracle_sum(&k_terms);
        }
    }
    Ok(Density { nao, data: k_data })
}

#[cfg(test)]
mod tests {
    use super::*;
    use pyscf_core::Unit;

    fn h2_mol() -> Mole {
        crate::M(crate::MoleBuildArgs {
            atom: crate::AtomInput::String("H 0 0 0; H 0 0 1.4".into()),
            basis: crate::BasisInput::Name("sto-3g".into()),
            unit: Unit::Bohr,
            max_memory: 4000.0,
            axes: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            ..Default::default()
        })
        .expect("H2/STO-3G build")
    }

    /// The guard sets env[8] and restores the prior value on drop — including
    /// on the early-return / panic path (T-04-07a). We assert via a scoped
    /// block that env[8] is `omega` inside and `prior` after.
    #[test]
    fn omega_guard_sets_and_restores() {
        let mut mol = h2_mol();
        let prior = mol._env[PTR_RANGE_OMEGA];
        {
            let g = OmegaGuard::new(&mut mol, 0.33).expect("guard");
            assert_eq!(
                g.mol()._env[PTR_RANGE_OMEGA],
                0.33,
                "env[8] set inside guard"
            );
        }
        assert_eq!(
            mol._env[PTR_RANGE_OMEGA], prior,
            "env[8] restored to prior value after guard drops"
        );
    }

    /// Guard restores env[8] even when the wrapped operation errors
    /// (the `?`-propagation path). We force an error by asking for an
    /// unknown intor name and confirm env[8] is back to its prior value.
    #[test]
    fn omega_restored_on_error_path() {
        let mut mol = h2_mol();
        let prior = mol._env[PTR_RANGE_OMEGA];
        let res = intor_with_omega(&mut mol, "int_does_not_exist", 0.5);
        assert!(res.is_err(), "unknown intor name must error");
        assert_eq!(
            mol._env[PTR_RANGE_OMEGA], prior,
            "env[8] restored after the error path (no leak — T-04-07a)"
        );
    }

    /// A short `_env` (unbuilt Mole) is rejected rather than panicking on the
    /// out-of-bounds index.
    #[test]
    fn omega_guard_rejects_short_env() {
        let mut mol = h2_mol();
        mol._env = vec![0.0_f64; 4]; // < PTR_RANGE_OMEGA + 1 (simulate unbuilt)
        let res = OmegaGuard::new(&mut mol, 0.1);
        assert!(res.is_err(), "short _env must be rejected, not indexed OOB");
    }
}
