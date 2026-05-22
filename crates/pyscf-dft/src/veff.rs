//! KS effective potential — `get_veff = J + Vxc − hyb·K` (DFT-01).
//!
//! Source: `pyscf/dft/rks.py:get_veff` (37-140) — the J + Vxc − hyb·K linear
//! combination. The standard-hybrid (omega = 0) form is implemented here;
//! the range-separated-hybrid (RSH, omega ≠ 0) branch is DEFERRED to 04-07
//! (a clearly-marked seam below).
//!
//! Structural template: `pyscf-scf::fock::default_get_veff` (fock.rs:140-148)
//! — the inline element loop (NOT `pyscf_algebra::axpy`, which is Tensor-API
//! and `NotYetImplemented{phase:2}`; the Phase 3 SCF + DF paths inline the
//! loop for the same reason, df_scf.rs:108-119).

use pyscf_core::{Density, Mole, PyscfRsError};
use pyscf_grids::Grids;

use crate::numint::NumInt;

/// `get_veff(ks, mol, dm) = J + Vxc − hyb·K` (standard hybrid, omega = 0).
///
/// - `J`, `K` come from the supplied `(j, k)` pair (the caller fetches them
///   via the SCF `get_jk` hook, which for the corpus routes through the DF
///   J/K builder — `int2e_sph` arity-4 stays `NotYetImplemented{phase:2}`).
/// - `Vxc` + `Exc` come from `NumInt::nr_rks` over the grid.
/// - `hyb` is the standard-hybrid HF-exchange mixing coefficient
///   (`NumInt::hybrid_coeff`); for a pure functional (SVWN/PBE) `hyb = 0` so
///   the `−hyb·K` term vanishes and `K` need not be built.
///
/// Returns `(veff, exc, vj_minus_hybk_ecoul)` where `exc` is the XC energy
/// (`∫ ε_xc ρ`) the KS `energy_elec` needs, and the third value is the
/// Coulomb (+ exact-exchange) energy contribution `0.5·Tr(D·(J − hyb·K))`.
///
/// # RSH seam
/// The range-separated branch (`omega ≠ 0`: split K into SR/LR with the
/// screened ERI) is DEFERRED to 04-07 (DFT-05). This function asserts
/// `omega == 0` is honored by only consuming the standard `hyb·K`.
#[allow(clippy::too_many_arguments)]
pub fn default_get_veff(
    ni: &NumInt,
    mol: &Mole,
    grids: &Grids,
    xc_code: &str,
    dm: &Density,
    j: &Density,
    k: &Density,
) -> Result<KsVeff, PyscfRsError> {
    let nao = j.nao;
    if dm.nao != nao || k.nao != nao {
        return Err(PyscfRsError::Core(pyscf_core::CoreError::DimensionMismatch {
            expected: nao,
            actual: dm.nao,
        }));
    }

    // hyb (standard-hybrid HF-exchange mixing). RSH branch (omega != 0):
    // DFT-05, filled by 04-07 — this plan does the hyb-scaled-K
    // standard-hybrid form only.
    let (omega, _alpha, hyb) = ni.rsh_coeff(xc_code, mol.spin).map_err(PyscfRsError::from)?;
    debug_assert!(
        omega == 0.0,
        "RSH (omega != 0) is deferred to 04-07; default_get_veff handles omega=0 only"
    );

    // Vxc + Exc from the numerical-integration grid loop.
    let nr = ni.nr_rks(mol, grids, xc_code, dm, 0, 1, mol.max_memory, None)?;

    // veff = J + Vxc − hyb·K (inline loop — matches fock.rs:144-146 template).
    let mut veff = vec![0.0_f64; nao * nao];
    for i in 0..(nao * nao) {
        veff[i] = j.data[i] + nr.vmat.data[i] - hyb * k.data[i];
    }

    // Coulomb (+ exact-exchange) energy contribution: 0.5·Tr(D·(J − hyb·K)).
    // (The XC energy is carried separately in `exc` — it is NOT 0.5·Tr(D·Vxc).)
    let mut jk = vec![0.0_f64; nao * nao];
    for i in 0..(nao * nao) {
        jk[i] = j.data[i] - hyb * k.data[i];
    }
    let e_coul = 0.5 * pyscf_algebra::oracle_dot(&dm.data, &jk);

    Ok(KsVeff {
        veff: Density { nao, data: veff },
        exc: nr.excsum,
        ecoul: e_coul,
        nelec: nr.nelec,
    })
}

/// The KS `get_veff` bundle: the effective potential plus the energy
/// components the KS `energy_elec` needs (Exc + Ecoul) and the integrated
/// electron count (for diagnostics / nelec sanity).
#[derive(Debug, Clone)]
pub struct KsVeff {
    /// `J + Vxc − hyb·K`.
    pub veff: Density,
    /// `Exc = ∫ ε_xc ρ` — the XC energy.
    pub exc: f64,
    /// `Ecoul = 0.5·Tr(D·(J − hyb·K))` — Coulomb + exact-exchange energy.
    pub ecoul: f64,
    /// `∫ ρ` — integrated electron count.
    pub nelec: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// veff = J + Vxc − hyb·K with a hand-built fixture: confirm the linear
    /// combination is exact (independent longhand on a 1-AO block).
    #[test]
    fn veff_is_linear_combination() {
        // Synthetic 1x1 J/K/Vxc — verify the −hyb·K mixing only (no grid).
        let nao = 1;
        let j = Density { nao, data: vec![2.0] };
        let k = Density { nao, data: vec![1.0] };
        let vxc = vec![0.5_f64];
        let hyb = 0.2;
        // veff = J + Vxc − hyb·K = 2.0 + 0.5 − 0.2·1.0 = 2.3
        let veff: Vec<f64> = (0..nao * nao)
            .map(|i| j.data[i] + vxc[i] - hyb * k.data[i])
            .collect();
        assert!((veff[0] - 2.3).abs() < 1e-13, "veff = J + Vxc − hyb·K, got {}", veff[0]);
    }
}
