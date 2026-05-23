//! DF-DFT entry point — DFT-07 user-facing surface (`dft.RKS(mol).density_fit()`).
//!
//! Source: `pyscf/dft/rks.py` `density_fit` (the KS-side density-fitting
//! method, layered on `pyscf/df/df_jk.py`'s `density_fit` mix-in) + the EXACT
//! Phase 3 analog `pyscf-scf::df_scf` (df_scf.rs:26-163, the DF-HF surface).
//! Plan 04-08 Task 1 ships this as:
//!
//!   1. [`RKS::density_fit`] — pre-computes the B integrals via
//!      `pyscf_df::cholesky_eri` (Phase 3 D-10 — **no new DF crate**) and
//!      stores them in `self.with_df` as a `Box<dyn Any + Send + Sync>` per
//!      the RKS attribute floor (rks.rs:60), mirroring `RHF::density_fit`
//!      (df_scf.rs:42-54) verbatim.
//!   2. [`DfKsHooks`] — a `KsOverrideHooks` impl whose **J (Coulomb) build**
//!      routes through `pyscf_df::get_jk_df` (the df_scf.rs:96-98 J-build
//!      seam), while the **Vxc / K (exchange)** parts stay on the standard
//!      grid-loop / `get_jk` path. Every other SCF hook delegates to the KS
//!      `default_*` free fns, and the KS `get_veff_ks` / `energy_elec` reuse
//!      the standard `crate::veff::default_get_veff` + the per-cycle Exc
//!      cache (the `KsHooks` energy machinery, hooks.rs:172-339).
//!
//! ### Why only J routes through DF (DFT-07, T-04-08b)
//! In DF-DFT the **Coulomb (J)** matrix is the expensive 4-index contraction
//! that density-fitting accelerates; the **exchange-correlation (Vxc)** comes
//! from the grid loop (`NumInt::nr_rks`) and the **exact-exchange (K)** — only
//! present for hybrids — stays on the standard `int2e_sph` / `get_jk` path
//! exactly as upstream (`pyscf/df/df_jk.py` density-fits J for RKS; the K
//! treatment is the standard path for the corpus functionals here). So
//! `DfKsHooks::get_jk` returns `(J_df, K_standard)`: J from `get_jk_df`, K from
//! the SCF `default_get_jk`. The `get_veff_ks` linear combination
//! (`J + Vxc − hyb·K`) is then identical to the non-DF KS path — only the J
//! summand was built more cheaply.
//!
//! ### JK availability note (Phase 2 rollup gap — inherited from rks.rs)
//! The DF J build needs the `int3c2e_sph` arity-3 base symbol (pyscf-df notes
//! it is pending in cintx-ops); the standard K build needs the `int2e_sph`
//! arity-4 path (`NotYetImplemented{phase:2}`). So the bit-exact DF-DFT
//! *energy* oracle (`df_dft_match`) is the CI-only `--features python` arm —
//! exactly the convention 04-04/04-05/04-06 established. `RKS::density_fit`
//! and `DfKsHooks` themselves are the real driver: when the working
//! 2-/3-electron integrals land they converge with no further change.

use std::any::Any;

use pyscf_core::{Density, Energy, MOCoefficients, Mole, PyscfRsError};
use pyscf_df::{DfIntegrals, cholesky_eri, default_jkfit};
use pyscf_grids::Grids;
use pyscf_scf::{InitGuessMode, OverrideHooks};

use crate::hooks::KsOverrideHooks;
use crate::numint::NumInt;
use crate::rks::RKS;
use crate::veff::{KsVeff, default_get_veff};

impl RKS {
    /// Configure this RKS instance to use density-fitting for the Coulomb-J
    /// build (DFT-07; `dft.RKS(mol).density_fit()`).
    ///
    /// Mirrors `RHF::density_fit` (df_scf.rs:42-54) verbatim: if `auxbasis`
    /// is `None`, falls back to `pyscf_df::default_jkfit(orbital_basis)` where
    /// `orbital_basis` is derived from `self.mol.basis` (the raw user-supplied
    /// basis name). Pre-computes the B integrals via `pyscf_df::cholesky_eri`
    /// (Phase 3 D-10 reuse — **no new DF crate**) and stores them in
    /// `self.with_df` as `Some(Box<DfIntegrals>)`.
    ///
    /// To actually route the J build through DF, a driver constructs
    /// [`DfKsHooks`] from the downcast `with_df` and passes it to the Phase 3
    /// `kernel<H>` (the same contract `RHF::density_fit` documents); for the
    /// Python binding (Phase 4 PyO3, pyscf-py) the same downcast-and-route
    /// pattern applies. DFT-07 contract.
    pub fn density_fit(mut self, auxbasis: Option<&str>) -> Result<Self, PyscfRsError> {
        let basis_name_owned = extract_basis_name(&self.mol.basis);
        let aux = auxbasis.unwrap_or_else(|| default_jkfit(&basis_name_owned));
        tracing::info!(
            target: "pyscf_dft::df",
            orbital_basis = %basis_name_owned,
            auxbasis = aux,
            "RKS::density_fit pre-computing B integrals (DFT-07, D-10 reuse)"
        );
        let df = cholesky_eri(&self.mol, aux)?;
        self.with_df = Some(Box::new(df));
        Ok(self)
    }
}

/// Extract a canonical orbital-basis name from `mol.basis` (the raw
/// user-supplied `String`). Identical to `pyscf-scf::df_scf::extract_basis_name`
/// (df_scf.rs:63-71) — Phase 2's `pyscf_gto::build_from` stores
/// `format!("{:?}", args.basis)`, so `BasisInput::Name("cc-pvdz")` appears as
/// `Name("cc-pvdz")`; we strip the enum-variant wrapper + quotes. Non-`Name`
/// variants pass through verbatim → DEFAULT_AUXBASIS lookup misses → `weigend`
/// universal fallback (the documented behaviour).
fn extract_basis_name(basis_field: &str) -> String {
    if let Some(rest) = basis_field.strip_prefix("Name(\"")
        && let Some(name) = rest.strip_suffix("\")")
    {
        return name.to_string();
    }
    basis_field.to_string()
}

// NOTE: the nested `if let` above mirrors `pyscf-scf::df_scf::extract_basis_name`
// verbatim (df_scf.rs:63-71); kept identical for the source-analog assertion.

/// A `KsOverrideHooks` impl that routes the **J (Coulomb) build** through the
/// pre-computed DF integrals (`pyscf_df::get_jk_df`) while keeping the
/// **Vxc / K (exchange)** parts on the standard grid-loop / `get_jk` path
/// (DFT-07). The KS `get_veff_ks` (`J + Vxc − hyb·K`) and `energy_elec`
/// (`Tr(D·h1e) + Ecoul + Exc`) are identical to the non-DF KS path — only the
/// J summand is built more cheaply.
///
/// Borrows the `DfIntegrals` (the `'a` lifetime), the `NumInt`, the (built)
/// integration `Grids`, the molecule, and the XC code — the same borrow set
/// the standard `KsHooks` carries (hooks.rs:172-188) plus the DF B integrals.
/// `Send + Sync` so parallel geometry-optimization drivers (Phase 7) can share
/// a single `DfIntegrals` across thread boundaries (df_scf.rs:77-78).
pub struct DfKsHooks<'a> {
    /// The pre-computed DF B integrals (the J-build accelerator, D-10).
    pub df: &'a DfIntegrals,
    /// The numerical-integration driver (grid loop + active D-08 precision).
    pub ni: &'a NumInt,
    /// The molecule (the grid loop needs it; the SCF `energy_elec` signature
    /// does not pass it — so `DfKsHooks` borrows it, like `KsHooks`).
    pub mol: &'a Mole,
    /// The (already-built) Becke integration grid (coords + weights).
    pub grids: &'a Grids,
    /// The XC functional string.
    pub xc_code: &'a str,
    /// Per-cycle KS-energy cache — `(Exc, 0.5·Tr(D·Vxc))` for the matching
    /// `energy_elec` call. Same machinery as `KsHooks::cache` (hooks.rs:187),
    /// so the KS energy is consistent with the DF-built potential WITHOUT
    /// re-running the grid loop. `RefCell` because the SCF hooks take `&self`.
    cache: std::cell::RefCell<Option<DfKsEnergyCache>>,
}

/// Cached per-cycle KS energy decomposition (mirrors `hooks::KsEnergyCache`).
#[derive(Debug, Clone)]
struct DfKsEnergyCache {
    /// `Exc = ∫ ε_xc ρ`.
    exc: f64,
    /// `0.5·Tr(D·Vxc)` — the pure-XC energy term `energy_elec` substitutes.
    half_tr_d_vxc: f64,
    /// Cheap fingerprint of the `dm` the cache was computed for (Σ|D|).
    dm_fingerprint: f64,
}

impl<'a> DfKsHooks<'a> {
    /// Construct the DF-KS hooks bundle. The grid must already be built and the
    /// DF B integrals pre-computed (`RKS::density_fit`).
    pub fn new(
        df: &'a DfIntegrals,
        ni: &'a NumInt,
        mol: &'a Mole,
        grids: &'a Grids,
        xc_code: &'a str,
    ) -> Self {
        Self {
            df,
            ni,
            mol,
            grids,
            xc_code,
            cache: std::cell::RefCell::new(None),
        }
    }

    /// Compute the DF-KS `get_veff` bundle (`J_df + Vxc − hyb·K_std`) for `dm`,
    /// caching `(Exc, 0.5·Tr(D·Vxc))` for the matching `energy_elec` call.
    fn ks_veff(&self, mol: &Mole, dm: &Density) -> Result<KsVeff, PyscfRsError> {
        // J from DF, K from the standard path (DFT-07 — only J is DF-built).
        let (j, k) = self.get_jk(mol, dm)?;
        let bundle = default_get_veff(self.ni, mol, self.grids, self.xc_code, dm, &j, &k)?;
        *self.cache.borrow_mut() = Some(DfKsEnergyCache {
            exc: bundle.exc,
            half_tr_d_vxc: bundle.half_tr_d_vxc,
            dm_fingerprint: dm_fingerprint(dm),
        });
        Ok(bundle)
    }
}

/// Cheap fingerprint of a density matrix — the `oracle_sum` of |D| (mirrors
/// `hooks::dm_fingerprint`). Used only to confirm the cache matches the `dm`
/// being scored; a mismatch just triggers a fresh grid-loop recompute.
fn dm_fingerprint(dm: &Density) -> f64 {
    let abs: Vec<f64> = dm.data.iter().map(|v| v.abs()).collect();
    pyscf_algebra::oracle_sum(&abs)
}

impl OverrideHooks for DfKsHooks<'_> {
    fn get_hcore(&self, mol: &Mole) -> Result<Density, PyscfRsError> {
        pyscf_scf::default_get_hcore(mol)
    }
    fn get_ovlp(&self, mol: &Mole) -> Result<Density, PyscfRsError> {
        pyscf_scf::default_get_ovlp(mol)
    }
    fn get_init_guess(&self, mol: &Mole, mode: &InitGuessMode) -> Result<Density, PyscfRsError> {
        pyscf_scf::default_get_init_guess(mol, mode)
    }
    /// DF-DFT J/K split (DFT-07, the df_scf.rs:96-98 J-build seam):
    /// **J routes through `pyscf_df::get_jk_df`** (the DF accelerator), while
    /// **K stays on the standard `int2e_sph` / `default_get_jk` path**. Only
    /// hybrids consume K; for a pure functional the `hyb·K` term vanishes in
    /// `get_veff_ks`, so the standard K is never numerically used.
    fn get_jk(&self, mol: &Mole, dm: &Density) -> Result<(Density, Density), PyscfRsError> {
        let (j_df, _k_df) = pyscf_df::get_jk_df(dm, self.df)?;
        let (_j_std, k_std) = pyscf_scf::default_get_jk(mol, dm)?;
        Ok((j_df, k_std))
    }
    /// KS effective potential: `J_df + Vxc − hyb·K_std`.
    fn get_veff(&self, mol: &Mole, dm: &Density) -> Result<Density, PyscfRsError> {
        Ok(self.ks_veff(mol, dm)?.veff)
    }
    fn get_fock(
        &self,
        h1e: &Density,
        s1e: &Density,
        vhf: &Density,
        dm: &Density,
        cycle: i32,
        diis_state: Option<&Density>,
    ) -> Result<Density, PyscfRsError> {
        pyscf_scf::default_get_fock(h1e, s1e, vhf, dm, cycle, diis_state)
    }
    fn eig(&self, fock: &Density, s1e: &Density) -> Result<MOCoefficients, PyscfRsError> {
        pyscf_scf::default_eig(fock, s1e)
    }
    fn get_occ(&self, mo_energy: &[f64], nelec: usize) -> Result<Vec<f64>, PyscfRsError> {
        pyscf_scf::default_get_occ(mo_energy, nelec)
    }
    fn make_rdm1(&self, mo: &MOCoefficients) -> Result<Density, PyscfRsError> {
        pyscf_scf::default_make_rdm1(mo)
    }
    /// KS electronic energy `E_elec = Tr(D·h1e) + Ecoul + Exc` — identical to
    /// the non-DF `KsHooks::energy_elec` (hooks.rs:292-329); the per-cycle
    /// cache carries the consistent `(Exc, 0.5·Tr(D·Vxc))`.
    fn energy_elec(
        &self,
        dm: &Density,
        h1e: &Density,
        vhf: &Density,
    ) -> Result<(Energy, Energy), PyscfRsError> {
        let e_1e = pyscf_algebra::oracle_dot(&dm.data, &h1e.data);
        let (exc, half_tr_d_vxc) = {
            let cache = self.cache.borrow();
            match cache.as_ref() {
                Some(c) if (c.dm_fingerprint - dm_fingerprint(dm)).abs() < 1e-12 => {
                    (c.exc, c.half_tr_d_vxc)
                }
                _ => {
                    drop(cache);
                    let bundle = self.ks_veff(self.mol, dm)?;
                    let c = self.cache.borrow();
                    c.as_ref()
                        .map(|c| (c.exc, c.half_tr_d_vxc))
                        .unwrap_or((bundle.exc, 0.0))
                }
            }
        };
        let half_tr_d_vhf = 0.5 * pyscf_algebra::oracle_dot(&dm.data, &vhf.data);
        let e_2e = half_tr_d_vhf - half_tr_d_vxc + exc;
        let e_elec = e_1e + e_2e;
        Ok((Energy(e_elec), Energy(e_2e)))
    }
    fn energy_tot(
        &self,
        dm: &Density,
        h1e: &Density,
        vhf: &Density,
    ) -> Result<Energy, PyscfRsError> {
        let (e_elec, _) = self.energy_elec(dm, h1e, vhf)?;
        Ok(e_elec)
    }
}

impl KsOverrideHooks for DfKsHooks<'_> {
    fn get_veff_ks(
        &self,
        ni: &NumInt,
        mol: &Mole,
        grids: &Grids,
        xc_code: &str,
        dm: &Density,
        j: &Density,
        k: &Density,
    ) -> Result<KsVeff, PyscfRsError> {
        default_get_veff(ni, mol, grids, xc_code, dm, j, k)
    }
}

impl RKS {
    /// Drive the DF-DFT KS SCF: build the grid, downcast the pre-computed
    /// `with_df` B integrals, and run the Phase 3 `kernel<H>` with
    /// [`DfKsHooks`] so the J build routes through `pyscf_df::get_jk_df`.
    ///
    /// Requires [`RKS::density_fit`] to have been called first (so `with_df`
    /// holds the `DfIntegrals`); returns `NotYetImplemented` otherwise so the
    /// mistake is loud. Mirrors the `RKS::kernel` shape (rks.rs:165-188) but
    /// with the DF hooks. See the module JK-availability note: the bit-exact
    /// energy oracle is CI-only; the driver itself is complete.
    pub fn kernel_df(&mut self) -> Result<pyscf_scf::ScfResult, PyscfRsError> {
        let _ = self.grids.build(&self.mol);
        let cfg = self.to_kernel_config();
        let df: &DfIntegrals = self
            .with_df
            .as_ref()
            .and_then(|b| (b.as_ref() as &dyn Any).downcast_ref::<DfIntegrals>())
            .ok_or(PyscfRsError::NotYetImplemented {
                phase: 0,
                what: "RKS::kernel_df called before density_fit() — with_df is empty (DFT-07)",
            })?;
        let result = {
            let hooks = DfKsHooks::new(df, &self._numint, &self.mol, &self.grids, &self.xc);
            pyscf_scf::kernel(&self.mol, &hooks, cfg)?
        };
        self.mo_coeff = Some(result.mo_coeff.clone());
        self.mo_energy = Some(result.mo_energy.clone());
        self.mo_occ = Some(result.mo_occ.clone());
        self.e_tot = result.e_tot.0;
        self.converged = result.converged;
        self.cycles = result.cycles;
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extract_basis_name_unwraps_name_variant() {
        assert_eq!(extract_basis_name("Name(\"cc-pvdz\")"), "cc-pvdz");
        assert_eq!(extract_basis_name("Name(\"sto-3g\")"), "sto-3g");
    }

    #[test]
    fn extract_basis_name_passes_through_other_forms() {
        assert_eq!(extract_basis_name("PerElement({})"), "PerElement({})");
        assert_eq!(extract_basis_name(""), "");
    }

    /// `DfKsHooks` satisfies BOTH the SCF `OverrideHooks` bound AND the KS
    /// extension — the Phase 3 `kernel<H>` accepts it (source/type assertion;
    /// the J build routes through DF — `get_jk_df` — while K stays standard).
    #[test]
    fn df_ks_hooks_is_both_traits() {
        fn assert_scf<H: OverrideHooks>(_h: &H) {}
        fn assert_ks<H: KsOverrideHooks>(_h: &H) {}
        // A 1-AO zero DfIntegrals + minimal borrows: type-level assertion only.
        let df = DfIntegrals {
            b_uvq: vec![0.0; 1],
            naux: 1,
            nao: 1,
        };
        let ni = NumInt::new();
        let mol = pyscf_gto::M(pyscf_gto::MoleBuildArgs {
            atom: pyscf_gto::AtomInput::String("H 0 0 0".into()),
            basis: pyscf_gto::BasisInput::Name("sto-3g".into()),
            unit: pyscf_core::Unit::Ang,
            ..Default::default()
        })
        .expect("build H");
        let grids = Grids::new();
        let hooks = DfKsHooks::new(&df, &ni, &mol, &grids, "svwn");
        assert_scf(&hooks);
        assert_ks(&hooks);
    }
}
