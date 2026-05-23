//! `KsOverrideHooks` — the KS extension of the SCF `OverrideHooks` surface
//! (DFT-08).
//!
//! Source: `pyscf/dft/rks.py` (the KS `get_veff` + `define_xc_` override
//! points layered on `scf/hf.py`'s `Method`). `KsOverrideHooks` EXTENDS
//! `pyscf_scf::OverrideHooks` (hooks.rs:13-131) with two KS-specific seams:
//!
//!   - `get_veff_ks` — the KS effective potential `J + Vxc − hyb·K`
//!     (the SCF base trait's `get_veff` is `J − 0.5·K`; KS replaces the
//!     exchange-correlation half with the grid-integrated `Vxc`).
//!   - `define_xc_` — the runtime XC-functional (re)definition seam
//!     (`numint.define_xc_`).
//!
//! `NoKsOverrides` is the default-impl: it delegates the inherited SCF hooks
//! to the `pyscf_scf::default_*` free fns (hooks.rs:63-131 pattern) and the
//! KS hooks to `crate::veff::default_get_veff` / the parser.
//!
//! `pyscf-dft` stays **pyo3-free** — the Python surface lives in pyscf-py.

use pyscf_core::{Density, Energy, MOCoefficients, Mole, PyscfRsError};
use pyscf_grids::Grids;
use pyscf_scf::{InitGuessMode, OverrideHooks};

use crate::numint::NumInt;
use crate::parser::{XcSpec, xcfun};
use crate::veff::{KsVeff, default_get_veff};

/// The KS override surface — extends `pyscf_scf::OverrideHooks` (a supertrait
/// bound) with the two KS-specific seams (DFT-08).
///
/// Implementors get the full 11-method SCF hook surface PLUS:
///   - [`KsOverrideHooks::get_veff_ks`] — `J + Vxc − hyb·K`.
///   - [`KsOverrideHooks::define_xc_`]  — runtime XC (re)definition.
pub trait KsOverrideHooks: OverrideHooks {
    /// KS effective potential `J + Vxc − hyb·K` (standard hybrid, omega = 0;
    /// RSH branch deferred to 04-07). `j`/`k` are the Coulomb / exact-exchange
    /// matrices the caller built via the SCF `get_jk` hook; `grids` is the
    /// (already-built) integration grid; `xc_code` is the functional string.
    #[allow(clippy::too_many_arguments)]
    fn get_veff_ks(
        &self,
        ni: &NumInt,
        mol: &Mole,
        grids: &Grids,
        xc_code: &str,
        dm: &Density,
        j: &Density,
        k: &Density,
    ) -> Result<KsVeff, PyscfRsError>;

    /// `define_xc_` — (re)define the active XC functional from a description.
    ///
    /// Source: `pyscf/dft/numint.py:define_xc_`. Two upstream forms:
    ///   - **string-recombination** (`'0.5*B3LYP + 0.5*wB97'`): feeds the
    ///     parser and returns the resolved [`XcSpec`].
    ///   - **callable** (a user Python/Rust closure): returns
    ///     `NotYetImplemented{deferred}` (D-02 — the callable-functional path
    ///     is a v1.x deferred item, per the project Deferred Items table).
    fn define_xc_(&self, description: &str) -> Result<XcSpec, PyscfRsError> {
        // String-recombination form — route through the (default xcfun) parser.
        xcfun::parse_xc(description).map_err(PyscfRsError::from)
    }
}

/// The default KS hook impl: inherited SCF hooks delegate to the
/// `pyscf_scf::default_*` free fns; the KS hooks delegate to
/// `crate::veff::default_get_veff` and the parser.
///
/// NOTE: the inherited `OverrideHooks::get_veff` (the SCF `J − 0.5·K` form)
/// is delegated to the SCF default for trait-completeness, but a KS SCF
/// driver calls [`KsOverrideHooks::get_veff_ks`] (not the inherited
/// `get_veff`) so the XC term is included — see `crate::rks::RKS::kernel`.
pub struct NoKsOverrides;

impl OverrideHooks for NoKsOverrides {
    fn get_hcore(&self, mol: &Mole) -> Result<Density, PyscfRsError> {
        pyscf_scf::default_get_hcore(mol)
    }
    fn get_ovlp(&self, mol: &Mole) -> Result<Density, PyscfRsError> {
        pyscf_scf::default_get_ovlp(mol)
    }
    fn get_init_guess(&self, mol: &Mole, mode: &InitGuessMode) -> Result<Density, PyscfRsError> {
        pyscf_scf::default_get_init_guess(mol, mode)
    }
    fn get_jk(&self, mol: &Mole, dm: &Density) -> Result<(Density, Density), PyscfRsError> {
        pyscf_scf::default_get_jk(mol, dm)
    }
    fn get_veff(&self, mol: &Mole, dm: &Density) -> Result<Density, PyscfRsError> {
        // Inherited SCF form (J − 0.5·K). KS drivers use `get_veff_ks`.
        pyscf_scf::default_get_veff(mol, dm)
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
    fn energy_elec(
        &self,
        dm: &Density,
        h1e: &Density,
        vhf: &Density,
    ) -> Result<(Energy, Energy), PyscfRsError> {
        pyscf_scf::default_energy_elec(dm, h1e, vhf)
    }
    fn energy_tot(
        &self,
        dm: &Density,
        h1e: &Density,
        vhf: &Density,
    ) -> Result<Energy, PyscfRsError> {
        pyscf_scf::default_energy_tot(dm, h1e, vhf)
    }
}

impl KsOverrideHooks for NoKsOverrides {
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

impl NoKsOverrides {
    /// The callable-functional form of `define_xc_` (D-02 deferred). Surfaced
    /// as a distinct method so the source-assertion test can confirm it
    /// returns `NotYetImplemented{deferred}` without a real callable type.
    pub fn define_xc_callable(&self) -> Result<XcSpec, PyscfRsError> {
        Err(PyscfRsError::NotYetImplemented {
            phase: 0,
            what: "define_xc_(callable): user-defined XC functions are a v1.x deferred item (D-02)",
        })
    }
}

/// The concrete KS `OverrideHooks` impl the `RKS::kernel` driver passes to
/// the Phase 3 generic `kernel<H>`. It borrows the `NumInt`, the (built)
/// integration `Grids`, and the XC code, and:
///   - overrides `get_veff` to return `J + Vxc − hyb·K` (the KS Fock term);
///   - overrides `energy_elec` to return `Tr(D·h1e) + Ecoul + Exc` (the KS
///     energy — Exc is the grid integral `∫ ε_xc ρ`, NOT `0.5·Tr(D·Vxc)`).
///
/// All other hooks delegate to the `pyscf_scf::default_*` free fns. The J/K
/// build routes through the SCF `default_get_jk` (which, for the corpus,
/// surfaces the Phase 2 `int2e_sph` rollup gap — see `rks.rs` module note;
/// the energy oracle is the CI-only `--features python` arm).
pub struct KsHooks<'a> {
    /// The numerical-integration driver (grid loop + active D-08 precision).
    pub ni: &'a NumInt,
    /// The molecule (needed by the grid loop, which `energy_elec`'s SCF-trait
    /// signature does not pass — so KsHooks borrows it).
    pub mol: &'a Mole,
    /// The (already-built) Becke integration grid (coords + weights).
    pub grids: &'a Grids,
    /// The XC functional string.
    pub xc_code: &'a str,
    /// Per-cycle XC-energy cache. `get_veff(dm)` runs the grid loop and caches
    /// `(Exc, 0.5·Tr(D·Vxc))`; `energy_elec(dm, …)` — called immediately after
    /// in the same SCF iteration (kernel_impl.rs:128-131) — reads it back so
    /// the energy is consistent with the potential WITHOUT re-running the grid
    /// loop. `RefCell` because the SCF hooks take `&self`.
    cache: std::cell::RefCell<Option<KsEnergyCache>>,
}

/// Cached per-cycle KS energy decomposition (see `KsHooks::cache`).
#[derive(Debug, Clone)]
struct KsEnergyCache {
    /// `Exc = ∫ ε_xc ρ`.
    exc: f64,
    /// `0.5·Tr(D·Vxc)` — the part of `0.5·Tr(D·vhf)` that must be replaced by
    /// `Exc` to get the correct KS 2e energy.
    half_tr_d_vxc: f64,
    /// Injective content fingerprint of the `dm` the cache was computed for
    /// (a `u64` hash of the raw `f64` bits — CR-04). Cache hits require exact
    /// equality, so a changed density NEVER returns a stale `Exc`.
    dm_fingerprint: u64,
}

impl<'a> KsHooks<'a> {
    /// Construct the KS hooks bundle. The grid must already be built.
    pub fn new(ni: &'a NumInt, mol: &'a Mole, grids: &'a Grids, xc_code: &'a str) -> Self {
        Self {
            ni,
            mol,
            grids,
            xc_code,
            cache: std::cell::RefCell::new(None),
        }
    }

    /// Compute the KS `get_veff` bundle (`J + Vxc − hyb·K` + Exc + Ecoul) for
    /// `dm`, caching `(Exc, 0.5·Tr(D·Vxc))` for the matching `energy_elec`
    /// call.
    fn ks_veff(&self, mol: &Mole, dm: &Density) -> Result<KsVeff, PyscfRsError> {
        let (j, k) = self.get_jk(mol, dm)?;
        let bundle = default_get_veff(self.ni, mol, self.grids, self.xc_code, dm, &j, &k)?;
        // The energy term `energy_elec` subtracts from 0.5·Tr(D·vhf)
        // (replacing it by Exc) is 0.5·Tr(D·Vxc). `default_get_veff` computes
        // it from the GENUINE Vxc (nr.vmat) and surfaces it as
        // `bundle.half_tr_d_vxc`. This is correct for both standard-hybrid
        // and RSH veff — we no longer reconstruct Vxc as `veff − J + hyb·K`
        // (which would be wrong once vk carries the RSH long-range term).
        *self.cache.borrow_mut() = Some(KsEnergyCache {
            exc: bundle.exc,
            half_tr_d_vxc: bundle.half_tr_d_vxc,
            dm_fingerprint: dm_fingerprint(dm),
        });
        Ok(bundle)
    }
}

/// Injective content fingerprint of a density matrix (CR-04). Hashes each
/// element's raw `f64` bit pattern (`v.to_bits()`) into a `u64` via the stdlib
/// `DefaultHasher` — no new crate dependency.
///
/// The old `Σ|D|` (L1-norm) key was NON-INJECTIVE: two distinct density
/// matrices with the same L1 norm collide, so at µHartree convergence
/// (where the 1µH bit-exact gate operates) a genuine step-to-step change in
/// `Exc` could be masked by an unchanged `Σ|D|`, returning a STALE XC energy.
/// Hashing the bit pattern makes distinct densities produce distinct keys, so
/// the cache-hit guard (`==`) only fires for a genuinely identical `dm`; any
/// difference triggers a fresh grid-loop recompute (the safety-net fallback).
///
/// Hashing the bits (not the value) is deliberate: bit-exact comparison means
/// `-0.0 != 0.0` and distinct NaN payloads hash differently — the cache only
/// reuses an `Exc` when the density is byte-for-byte the same one it scored.
fn dm_fingerprint(dm: &Density) -> u64 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    let mut h = DefaultHasher::new();
    for &v in &dm.data {
        v.to_bits().hash(&mut h);
    }
    h.finish()
}

impl OverrideHooks for KsHooks<'_> {
    fn get_hcore(&self, mol: &Mole) -> Result<Density, PyscfRsError> {
        pyscf_scf::default_get_hcore(mol)
    }
    fn get_ovlp(&self, mol: &Mole) -> Result<Density, PyscfRsError> {
        pyscf_scf::default_get_ovlp(mol)
    }
    fn get_init_guess(&self, mol: &Mole, mode: &InitGuessMode) -> Result<Density, PyscfRsError> {
        pyscf_scf::default_get_init_guess(mol, mode)
    }
    fn get_jk(&self, mol: &Mole, dm: &Density) -> Result<(Density, Density), PyscfRsError> {
        pyscf_scf::default_get_jk(mol, dm)
    }
    /// KS effective potential: `J + Vxc − hyb·K`.
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
    /// KS electronic energy: `E_elec = Tr(D·h1e) + Ecoul + Exc`.
    ///
    /// The SCF default `energy_elec` computes `Tr(D·h1e) + 0.5·Tr(D·vhf)`,
    /// which is WRONG for KS because `vhf` carries `Vxc` (whose energy is the
    /// grid integral `Exc`, not `0.5·Tr(D·Vxc)`). So KS recomputes the J/K +
    /// grid Exc here (matching upstream `rks.energy_elec`, rks.py:226). The
    /// passed `vhf` is ignored — Exc/Ecoul come from the consistent
    /// `ks_veff` recomputation over the same `dm`.
    fn energy_elec(
        &self,
        dm: &Density,
        h1e: &Density,
        vhf: &Density,
    ) -> Result<(Energy, Energy), PyscfRsError> {
        // E_1e = Tr(D·h1e) (oracle_dot — Pitfall 9).
        let e_1e = pyscf_algebra::oracle_dot(&dm.data, &h1e.data);

        // The SCF cycle calls get_veff(mol, dm) immediately before
        // energy_elec(dm, h1e, vhf) in the same iteration (kernel_impl.rs:
        // 128-131), so the cache holds (Exc, 0.5·Tr(D·Vxc)) for THIS dm.
        // The KS 2e energy is Ecoul + Exc; since vhf = J + Vxc − hyb·K,
        //   0.5·Tr(D·vhf) = Ecoul + 0.5·Tr(D·Vxc),  Ecoul = 0.5·Tr(D·(J−hyb·K))
        // so  E_2e = 0.5·Tr(D·vhf) − 0.5·Tr(D·Vxc) + Exc.
        let (exc, half_tr_d_vxc) = {
            let cache = self.cache.borrow();
            match cache.as_ref() {
                // CR-04: exact `u64` equality on the injective content hash —
                // NOT a float approximation. Any change to the density (even
                // below µHartree) misses the cache and recomputes the grid loop.
                Some(c) if c.dm_fingerprint == dm_fingerprint(dm) => {
                    (c.exc, c.half_tr_d_vxc)
                }
                // Cache miss (cold call / dm changed): recompute the grid loop.
                // (mol is borrowed on KsHooks for exactly this fallback.)
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

impl KsOverrideHooks for KsHooks<'_> {
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

#[cfg(test)]
mod tests {
    use super::*;

    /// `define_xc_` string form feeds the parser and returns a spec.
    #[test]
    fn define_xc_string_form_parses() {
        let hooks = NoKsOverrides;
        let spec = hooks.define_xc_("b3lyp").expect("string form parses");
        // xcfun b3lyp → hyb=0.2.
        assert!((spec.hyb().0 - 0.2).abs() < 1e-9);
    }

    /// `define_xc_` callable form returns NotYetImplemented{deferred} (D-02).
    #[test]
    fn define_xc_callable_form_deferred() {
        let hooks = NoKsOverrides;
        let err = hooks.define_xc_callable().unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("deferred")
                || msg.contains("not yet implemented")
                || msg.contains("NotYetImplemented"),
            "callable define_xc_ must be deferred (D-02), got: {msg}"
        );
    }

    /// NoKsOverrides satisfies BOTH the SCF OverrideHooks bound AND the KS
    /// extension — the SCF kernel accepts it.
    #[test]
    fn no_ks_overrides_is_both_traits() {
        fn assert_scf<H: OverrideHooks>(_h: &H) {}
        fn assert_ks<H: KsOverrideHooks>(_h: &H) {}
        let h = NoKsOverrides;
        assert_scf(&h);
        assert_ks(&h);
    }

    /// CR-04: the KS energy-cache fingerprint MUST be injective. The old
    /// `Σ|D|` (L1-norm) fingerprint is NOT — two density matrices with the
    /// same L1 norm but different entries collide, so at µHartree convergence
    /// `energy_elec` could return a stale `Exc` from the prior SCF cycle.
    ///
    /// `a` and `b` below both have `Σ|D| = 4` but differ entry-by-entry, so a
    /// correct (injective) fingerprint MUST distinguish them.
    #[test]
    fn dm_fingerprint_is_injective() {
        let a = Density {
            nao: 2,
            data: vec![1.0, -1.0, 1.0, -1.0],
        };
        let b = Density {
            nao: 2,
            data: vec![2.0, -2.0, 0.0, 0.0],
        };

        // Sanity: the OLD non-injective L1-norm key would treat these as equal.
        let l1_a: f64 = a.data.iter().map(|v| v.abs()).sum();
        let l1_b: f64 = b.data.iter().map(|v| v.abs()).sum();
        assert!(
            (l1_a - l1_b).abs() < 1e-12,
            "fixture must share an L1 norm (Σ|D|) so the test exercises the collision"
        );

        // Injective fingerprint: distinct data ⇒ distinct fingerprint.
        assert_ne!(
            dm_fingerprint(&a),
            dm_fingerprint(&b),
            "two dm with the same Σ|D| but different data must NOT collide"
        );

        // Deterministic: same data ⇒ same fingerprint (cache-hit correctness).
        assert_eq!(
            dm_fingerprint(&a),
            dm_fingerprint(&a),
            "fingerprint must be deterministic for an identical dm"
        );
    }
}
