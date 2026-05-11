//! Electronic + total energy reductions. Body filled in plan 03-11.
//!
//! Source: `pyscf/scf/hf.py:1556-1602` — `def energy_elec(self, dm,
//! h1e, vhf)` and `def energy_tot(self, dm, h1e, vhf)`. Every reduction
//! goes through `pyscf_algebra::oracle_dot` / `oracle_sum` to mitigate
//! reduction-order drift (Pitfall 9 — voids bit-exact-with-PySCF under
//! release-oracle if violated).
//!
//! Note: upstream `energy_tot(dm, h1e, vhf)` reads `self.mol.energy_nuc()`
//! internally; in pyscf-rs the kernel cycle loop (kernel_impl::scf_loop)
//! adds the nuclear-repulsion term via `mol.enuc()` separately. This
//! hook returns `E_elec` so the trait signature mirrors the upstream
//! `Method.energy_tot` override point exactly (preserves SCF-08 fidelity).

use pyscf_core::{Density, Energy, PyscfRsError};

/// `E_elec = Tr(D · h1e) + 0.5 · Tr(D · vhf)`, plus the Coulomb-only
/// component as the second return. RDM-energy contraction; both inputs
/// must be `nao × nao` with matching layout.
///
/// Returns `(E_elec, E_coul)` where:
///   - `E_elec = E_1e + E_coul`
///   - `E_1e   = Σ_{μν} D[μν] · h1e[μν]`            (1-electron, kinetic + nuc)
///   - `E_coul = 0.5 · Σ_{μν} D[μν] · vhf[μν]`     (mean-field 2-electron)
pub fn default_energy_elec(
    dm: &Density,
    h1e: &Density,
    vhf: &Density,
) -> Result<(Energy, Energy), PyscfRsError> {
    let nao = dm.nao;
    if h1e.nao != nao || vhf.nao != nao {
        return Err(pyscf_core::PyscfRsError::Core(
            pyscf_core::CoreError::DimensionMismatch {
                expected: nao,
                actual: h1e.nao.max(vhf.nao),
            },
        ));
    }
    if dm.data.len() != nao * nao
        || h1e.data.len() != nao * nao
        || vhf.data.len() != nao * nao
    {
        return Err(pyscf_core::PyscfRsError::Core(
            pyscf_core::CoreError::DimensionMismatch {
                expected: nao * nao,
                actual: dm.data.len(),
            },
        ));
    }
    // oracle_dot uses the pairwise-tree reduction (chunk=128) so reruns
    // and thread-counts produce bit-identical results (Pitfall 9 + 2).
    let e_1e = pyscf_algebra::oracle_dot(&dm.data, &h1e.data);
    let e_coul = 0.5 * pyscf_algebra::oracle_dot(&dm.data, &vhf.data);
    // oracle_sum of the 2-element vector matches upstream's e_1e + e_coul.
    let e_elec = pyscf_algebra::oracle_sum(&[e_1e, e_coul]);
    Ok((Energy(e_elec), Energy(e_coul)))
}

/// `E_tot` hook — upstream reads nuc-repulsion from `self.mol`; in
/// pyscf-rs the kernel cycle loop (kernel_impl::scf_loop) adds the
/// nuc-repulsion term separately via `mol.enuc()`. This hook returns
/// `E_elec` so the trait signature parity with `Method.energy_tot` is
/// preserved (SCF-08).
pub fn default_energy_tot(
    dm: &Density,
    h1e: &Density,
    vhf: &Density,
) -> Result<Energy, PyscfRsError> {
    let (e_elec, _e_coul) = default_energy_elec(dm, h1e, vhf)?;
    Ok(e_elec)
}
