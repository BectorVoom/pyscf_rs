//! `KRKS` restricted to the irreducible Brillouin zone — `krks_ksymm.py`
//! (144 l), plan 17-08 Task 2.
//!
//! # Why this file is thin
//!
//! Upstream says it in its own class declaration:
//!
//! ```text
//! class KsymAdaptedKRKS(krks.KRKS, khf_ksymm.KRHF)      # krks_ksymm.py:88
//! ```
//!
//! The DFT half is inherited unchanged; only `get_veff` (`:29-87`),
//! `energy_elec` (`:118-138`) and `to_hf` (`:139`) are overridden. This port
//! has no inheritance, so it **shares the functions instead of copying them**:
//! [`eig_symm_adapted`] and [`ksymm_get_occ_restricted`] are the very ones
//! [`pyscf_pbc_scf::KsymAdaptedKrhf`] uses, made `pub` by 17-07 for exactly
//! this.
//!
//! # The mechanism, and the one thing that makes it work
//!
//! Upstream sets **`kpts_band = kpts.kpts_ibz`** when no band k-points are
//! given (`krks_ksymm.py:41-42`). That single line is what reconciles a
//! full-BZ density with an IBZ-length potential: `nr_rks` evaluates `rho` over
//! the whole zone (the Group-A unfold — see `17-08-FINDING-numint.md`) but
//! builds the potential matrices **at the band k-points**, which are the IBZ
//! set. The J/K half is handed the same `kpts_band`, so both halves return
//! `nkpts_ibz` matrices and the [`KOverrideHooks`] contract is satisfied
//! without anything being folded by hand.
//!
//! # Every weighted sum in this module, named (17-CONTEXT §3.5)
//!
//! As in `khf_ksymm.rs`, written out rather than left to a diff — 15-CONTEXT
//! §3 recorded the KMP2 trap where `1/nkpts` appeared at three sites and was
//! two distinct divisions.
//!
//! | quantity | upstream | weight |
//! |---|---|---|
//! | `ecoul` | `krks_ksymm.py:76` | **`weights_ibz`** (`einsum('K,Kij,Kji', weight, dm, vj) * .5`) |
//! | `exc`'s hybrid correction | `:80` | **`weights_ibz`** (`... * .25`) |
//! | `exc` from the quadrature | `:53` | **none** — `nr_rks` already returns the integrated value |
//! | `energy_elec`'s `e1` | `:130` | **`weights_ibz`** |
//! | `nelectron` | inherited `khf_ksymm.py:38` | **`nkpts`, NOT `nkpts_ibz`** |
//!
//! `weights_ibz` sums to 1, so none of these carries a further `1/nkpts`.
//! Non-symmetric `krks.rs` uses `weight = 1.0 / nkpts` at the same sites; that
//! is the *same* quantity only when every star has one member.

use std::cell::Cell as StdCell;

use pyscf_algebra::CTensor;
use pyscf_core::PyscfRsError;
use pyscf_pbc_df::{Fftdf, PeriodicDf};
use pyscf_pbc_gto::{Cell, ExxDiv};
use pyscf_pbc_scf::khf_ksymm::{eig_symm_adapted, ksymm_get_occ_restricted};
use pyscf_pbc_scf::khooks::KOverrideHooks;
use pyscf_pbc_scf::types::{KDms, KInitGuess, KMats, KScfConfig, KScfResult};
use pyscf_pbc_symm::kpts::KPoints;

use crate::error::PbcDftError;
use crate::gen_grid::PeriodicGrids;
use crate::krks::{KsEnergyTags, unwrap_err};
use crate::numint::KNumInt;
use crate::veff::{add_assign, get_jk, sub_scaled};
use crate::xc::err;

/// Restricted periodic Kohn-Sham over an irreducible k-point set.
///
/// `krks_ksymm.py:88` — upstream's `KsymAdaptedKRKS`.
#[derive(Debug)]
pub struct KsymAdaptedKrks {
    /// The density-fitting object, built over the **full BZ** — the same
    /// arrangement `KsymAdaptedKrhf` uses, and for the same reason: every DF
    /// entry point takes its k-points explicitly, so one object serves the
    /// full-BZ density and the IBZ-length potential.
    pub with_df: Box<dyn PeriodicDf>,
    /// The k-point symmetry, by composition (D-PBC-25).
    pub kpts: KPoints,
    /// Materialised so [`KOverrideHooks::kpts`] has a slice to borrow.
    kpts_ibz: Vec<[f64; 3]>,
    /// The XC functional string.
    pub xc: String,
    /// Exchange divergence treatment; upstream's default is Ewald.
    pub exxdiv: Option<ExxDiv>,
    /// The integration grid.
    pub grids: PeriodicGrids,
    /// The numerical-integration driver, built with
    /// [`KNumInt::with_symmetry`] so its seven `KPoints` branches are live.
    pub ni: KNumInt,
    /// `ksymm_scf_common_init` (`khf_ksymm.py:142`) defaults this to **true**.
    pub use_ao_symmetry: bool,
    pub(crate) tags: StdCell<Option<KsEnergyTags>>,
}

impl KsymAdaptedKrks {
    /// Build with the default `FFTDF` over the full BZ and the uniform grid on
    /// `cell.mesh`.
    ///
    /// # Errors
    /// Propagates the `FFTDF` and grid construction.
    pub fn new(cell: Cell, kpts: KPoints, xc: &str) -> Result<Self, PbcDftError> {
        let grids = PeriodicGrids::uniform(&cell, Some(cell.mesh))?;
        let with_df = Fftdf::new(cell, &kpts.kpts)
            .map_err(|e| err(format!("KRKS/ksymm: FFTDF construction failed: {e}")))?;
        Ok(Self::from_df(Box::new(with_df), kpts, xc, grids))
    }

    /// Build over an explicitly configured full-BZ density-fitting object.
    pub fn from_df(
        with_df: Box<dyn PeriodicDf>,
        kpts: KPoints,
        xc: &str,
        grids: PeriodicGrids,
    ) -> Self {
        let kpts_ibz = kpts.kpts_ibz.clone();
        let ni = KNumInt::with_symmetry(&kpts);
        Self {
            with_df,
            kpts,
            kpts_ibz,
            xc: xc.to_string(),
            exxdiv: Some(ExxDiv::Ewald),
            grids,
            ni,
            use_ao_symmetry: true,
            tags: StdCell::new(None),
        }
    }

    /// Electrons in the whole BZ supercell — inherited `khf_ksymm.py:38`.
    ///
    /// **`nkpts`, not `nkpts_ibz`**: the Fermi level is a full-BZ quantity
    /// (17-CONTEXT §3.4).
    pub fn nelectron(&self) -> usize {
        self.cell().tot_electrons(self.kpts.nkpts())
    }

    /// The full-BZ k-points.
    pub fn kpts_bz(&self) -> &[[f64; 3]] {
        &self.kpts.kpts
    }

    /// Run the SCF over the IBZ.
    ///
    /// # Errors
    /// Propagates every hook and the driver.
    pub fn kernel(&self, cfg: &KScfConfig) -> Result<KScfResult, PyscfRsError> {
        pyscf_pbc_scf::kernel(self, cfg)
    }

    /// `krks_ksymm.py:29-87` — the KS effective potential over the IBZ, plus
    /// the energy tags the driver's `energy_elec` reads back.
    ///
    /// # Errors
    /// Propagates the quadrature, the density fitting and the unfold.
    pub fn get_veff_tagged(
        &self,
        dms: &KDms,
        kpts_band: Option<&[[f64; 3]]>,
    ) -> Result<(KDms, KsEnergyTags), PbcDftError> {
        let cell = self.cell();
        let nao = cell.mol.nao_nr;
        let ground_state = kpts_band.is_none();
        // `krks_ksymm.py:41-42` — THE line that makes the shapes work.
        let band: Vec<[f64; 3]> = kpts_band.map_or_else(|| self.kpts_ibz.clone(), <[_]>::to_vec);

        // S-01: the density is unfolded to the full BZ ONCE, here, and the
        // full-BZ stack is then handed to BOTH halves.
        //
        // It used to be unfolded twice per cycle: `nr_rks` calls
        // `unfold_kdms` itself (Group A, `numint.py:328-331`) and the J/K half
        // called it again a few lines below. Each call is `nkpts` `R·M·Rᴴ`
        // sandwiches plus two format conversions across the
        // `CTensor`/`Complex64` seam, so half of that was pure duplication.
        //
        // BIT-EXACT: `nr_rks`'s own `unfold_kdms` on this already-full-BZ
        // stack returns `Cow::Borrowed` unchanged (the guard `unfold_dms` has
        // always carried, asserted by
        // `tests/krks_ksymm.rs::unfold_is_a_bit_exact_no_op_on_full_bz_input`),
        // so the quadrature sees exactly the density it saw before.
        let dm_bz = self.ni.unfold_kdms(cell, dms, nao)?;

        // The XC half. `self.ni` is a `KSet::Ibz` numint, so the density is
        // evaluated over the full BZ (Group A) and the potential is built at
        // `band` — the IBZ points.
        let nr = self
            .ni
            .nr_rks(cell, &self.grids, &self.xc, &dm_bz, 1, Some(&band))?;
        let mut vxc = nr.vmat;
        let mut exc = nr.excsum[0];

        // The J/K half. The DF layer is handed the FULL BZ k-points and a
        // full-BZ density, with `kpts_band` selecting the IBZ output — so it
        // still knows nothing about symmetry (D-PBC-15), it only ever sees two
        // plain k-point lists.
        let jk = get_jk(
            self.with_df.as_ref(),
            &self.xc,
            &dm_bz,
            1,
            self.kpts_bz(),
            Some(&band),
            self.exxdiv,
            true,
        )?;
        let vj = jk
            .vj
            .ok_or_else(|| err("KRKS/ksymm: the density-fitting object returned no vj"))?;
        add_assign(&mut vxc, &vj);

        // `krks_ksymm.py:76` — ecoul = einsum('K,Kij,Kji', weights_ibz, dm, vj) * .5
        let ecoul = if ground_state {
            0.5 * self.weighted_trace(dms, &vj, nao)
        } else {
            0.0
        };

        if let Some(vk) = jk.vk.as_ref() {
            // `:79` — vxc -= .5 * vk
            sub_scaled(&mut vxc, 0.5, vk);
            if ground_state {
                // `:81` — exc -= einsum('K,Kij,Kji', weights_ibz, dm, vk).real * .25
                exc -= 0.25 * self.weighted_trace(dms, vk, nao);
            }
        }

        Ok((
            vxc,
            KsEnergyTags {
                ecoul,
                exc,
                nelec: nr.nelec[0],
            },
        ))
    }

    /// `Re einsum('K,Kij,Kji', weights_ibz, dm, v)` — the **`weights_ibz`**
    /// contraction, not `1/nkpts`.
    ///
    /// Writing `1/nkpts` here would silently drop every star multiplicity, and
    /// would be invisible on any cell whose stars all have the same size.
    ///
    /// # P-02 — this used to be a hand-rolled nest, and its doc was wrong
    ///
    /// The body was two naive running sums (`nao^2` products, then
    /// `nkpts_ibz` weighted partials) and the doc comment claimed "the
    /// accumulation is ordered (D-PBC-17), so the result is bit-identical
    /// under any thread count". The conclusion held but the premise did not:
    /// the fold was *serial*, which makes it thread-independent, and serial
    /// is not what D-PBC-17 asks for — it asks for the ordered primitive,
    /// whose error bound is `O(log2 n · eps)` rather than `O(n · eps)`. The
    /// distinction matters here because this one function feeds `ecoul`, the
    /// hybrid `exc` correction and `energy_elec`'s `e1` for every
    /// k-symmetric KS driver.
    ///
    /// It now delegates to [`crate::veff::weighted_trace_dm_v`], which is
    /// ordered in both axes. Bit-exact at every cell this repository gates on
    /// — see that function's own note.
    fn weighted_trace(&self, dms: &KDms, v: &[KMats], nao: usize) -> f64 {
        crate::veff::weighted_trace_dm_v(dms, v, &self.kpts.weights_ibz, nao)
    }

    /// [`KsymAdaptedKrks::weighted_trace`] against ONE shared matrix stack —
    /// used by `energy_elec`, where the same `h1e` is traced against the
    /// single density channel and materialising `&[h1e.to_vec()]` was a full
    /// `nkpts_ibz x nao^2` complex clone per call.
    fn weighted_trace_shared(&self, dms: &KDms, v: &KMats, nao: usize) -> f64 {
        crate::veff::weighted_trace_dm_v_shared(dms, v, &self.kpts.weights_ibz, nao)
    }
}

impl KOverrideHooks for KsymAdaptedKrks {
    fn cell(&self) -> &Cell {
        self.with_df.cell()
    }

    /// The IBZ set — the whole indirection, exactly as in `khf_ksymm.rs`.
    fn kpts(&self) -> &[[f64; 3]] {
        &self.kpts_ibz
    }

    fn nset(&self) -> usize {
        1
    }

    fn get_ovlp(&self) -> Result<KMats, PyscfRsError> {
        let nao = self.cell().mol.nao_nr;
        Ok(pyscf_pbc_scf::krhf::to_row_major(
            pyscf_pbc_gto::get_ovlp(self.cell(), self.kpts())?,
            nao,
        ))
    }

    fn get_hcore(&self) -> Result<KMats, PyscfRsError> {
        pyscf_pbc_df::get_hcore(self.with_df.as_ref(), self.kpts())
            .map_err(|e| unwrap_err(err(format!("KRKS/ksymm: get_hcore: {e}"))))
    }

    fn get_init_guess(&self, mode: &KInitGuess, s1e: &KMats) -> Result<KDms, PyscfRsError> {
        pyscf_pbc_scf::init_guess::get_init_guess(
            self.cell(),
            self.kpts_ibz.len(),
            1,
            mode,
            s1e,
            &[self.nelectron() as f64],
            0,
        )
    }

    fn get_veff(&self, dms: &KDms) -> Result<KDms, PyscfRsError> {
        let (v, tags) = self.get_veff_tagged(dms, None).map_err(unwrap_err)?;
        self.tags.set(Some(tags));
        Ok(v)
    }

    fn eig(
        &self,
        fock: &KDms,
        s1e: &KMats,
    ) -> Result<(Vec<Vec<f64>>, Vec<CTensor>), PyscfRsError> {
        if !self.use_ao_symmetry {
            return pyscf_pbc_scf::krhf::eig_channel(&fock[0], s1e, self.nao());
        }
        let cell = self.cell();
        let need = |what: &str| {
            PyscfRsError::Core(pyscf_core::CoreError::InvalidMolecule(format!(
                "KRKS/ksymm: use_ao_symmetry = true needs cell.{what}, which is unset. \
                 Call Cell::build_symmetry(&kpts) before the SCF, or set \
                 use_ao_symmetry = false."
            )))
        };
        let so = cell.symm_orb.as_deref().ok_or_else(|| need("symm_orb"))?;
        let ids = cell.irrep_id.as_deref().ok_or_else(|| need("irrep_id"))?;
        let nao = self.nao();
        let mut es = Vec::with_capacity(fock[0].len());
        let mut cs = Vec::with_capacity(fock[0].len());
        for (k, f) in fock[0].iter().enumerate() {
            let (e, c) = eig_symm_adapted(f, &s1e[k], &so[k], &ids[k], nao).map_err(|err| {
                PyscfRsError::Core(pyscf_core::CoreError::InvalidMolecule(format!(
                    "KRKS/ksymm: symmetry-adapted eig failed at IBZ k = {k}: {err}"
                )))
            })?;
            es.push(e);
            cs.push(c);
        }
        Ok((es, cs))
    }

    fn get_occ(
        &self,
        mo_energy: &[Vec<f64>],
    ) -> Result<(Vec<Vec<f64>>, Vec<f64>), PyscfRsError> {
        // ONE Fermi level over the UNFOLDED BZ (17-CONTEXT §3.4). Shared with
        // `KsymAdaptedKrhf`, which upstream reaches by inheritance.
        ksymm_get_occ_restricted(&self.kpts, mo_energy, self.nelectron())
    }

    fn make_rdm1(
        &self,
        mo_coeff: &[CTensor],
        mo_occ: &[Vec<f64>],
    ) -> Result<KDms, PyscfRsError> {
        Ok(vec![pyscf_pbc_scf::krdm::make_rdm1(
            mo_coeff,
            mo_occ,
            self.nao(),
        )])
    }

    /// `krks_ksymm.py:118-138` — `e1` is a **`weights_ibz`** sum, and the
    /// Coulomb/XC parts come from the tags `get_veff` left behind.
    fn energy_elec(
        &self,
        dms: &KDms,
        h1e: &KMats,
        vhf: &KDms,
    ) -> Result<(f64, f64), PyscfRsError> {
        let nao = self.nao();
        let tags = self.tags.get().ok_or_else(|| {
            PyscfRsError::Core(pyscf_core::CoreError::InvalidMolecule(
                "KRKS/ksymm: energy_elec was called before get_veff produced its \
                 energy tags"
                    .to_string(),
            ))
        })?;
        let _ = vhf;
        // P-02: `&[h1e.to_vec()]` cloned the whole one-electron k-stack on
        // every cycle to satisfy a `v[s][k]` index shape. Bit-exact to share.
        let e1 = self.weighted_trace_shared(dms, h1e, nao);
        Ok((e1 + tags.ecoul + tags.exc, tags.ecoul))
    }
}

// =====================================================================
// Task 4 — DFT+U over an IBZ k-set (`krkspu_ksymm.py`, 72 l)
// =====================================================================

/// Restricted periodic Kohn-Sham with a Hubbard `U`, over an irreducible
/// k-point set — `krkspu_ksymm.py:56` (`KsymAdaptedKRKSpU`).
///
/// # The plan's premise for this task was wrong — D-17-08-02
///
/// 17-08-PLAN.md Task 4 said "the local projectors `C_ao_lo` must be rotated
/// with the space group when the density is unfolded". **They are not, and
/// they must not be.** Upstream's whole ksymm DFT+U is
/// `krks_ksymm.get_veff` followed by the SHARED `krkspu._add_Vhubbard`, whose
/// only two symmetry-aware lines are `kpts = kpts.kpts_ibz` (`krkspu.py:77`)
/// and `weight = weights_ibz` (`:93`).
///
/// `make_minao_lo` is then called with the IBZ k-points, so the projectors are
/// built **directly at the IBZ points**, where they are already correct. The
/// Hubbard term never unfolds anything, so there is nothing to rotate. Full
/// evidence in `17-08-FINDING-numint.md`.
#[derive(Debug)]
pub struct KsymAdaptedKrkspu {
    /// The underlying k-symmetric KS object; owns the cell, the `KPoints` and
    /// the grid.
    pub ks: KsymAdaptedKrks,
    /// The Hubbard configuration.
    pub u: crate::kspu::HubbardU,
    e_u: StdCell<f64>,
}

impl KsymAdaptedKrkspu {
    /// Wrap a [`KsymAdaptedKrks`] with a Hubbard `U`.
    pub fn new(ks: KsymAdaptedKrks, u: crate::kspu::HubbardU) -> Self {
        Self {
            ks,
            u,
            e_u: StdCell::new(0.0),
        }
    }

    /// `E_U` of the last `get_veff`.
    pub fn e_u(&self) -> f64 {
        self.e_u.get()
    }

    /// Run the SCF over the IBZ.
    ///
    /// # Errors
    /// Propagates every hook and the driver.
    pub fn kernel(&self, cfg: &KScfConfig) -> Result<KScfResult, PyscfRsError> {
        pyscf_pbc_scf::kernel(self, cfg)
    }
}

impl KOverrideHooks for KsymAdaptedKrkspu {
    fn cell(&self) -> &Cell {
        self.ks.cell()
    }
    fn kpts(&self) -> &[[f64; 3]] {
        self.ks.kpts()
    }
    fn nset(&self) -> usize {
        1
    }
    fn get_ovlp(&self) -> Result<KMats, PyscfRsError> {
        self.ks.get_ovlp()
    }
    fn get_hcore(&self) -> Result<KMats, PyscfRsError> {
        self.ks.get_hcore()
    }
    fn get_init_guess(&self, mode: &KInitGuess, s1e: &KMats) -> Result<KDms, PyscfRsError> {
        self.ks.get_init_guess(mode, s1e)
    }

    /// `krkspu_ksymm.py:26-52` — the KS potential, then the Hubbard term.
    fn get_veff(&self, dms: &KDms) -> Result<KDms, PyscfRsError> {
        let (mut v, tags) = self.ks.get_veff_tagged(dms, None).map_err(unwrap_err)?;
        // `krkspu.py:77, :93` — the IBZ k-points and `weights_ibz`, which is
        // the entirety of what makes the Hubbard term symmetry-aware.
        let e_u = crate::kspu::add_vhubbard_weighted(
            &mut v,
            self.cell(),
            self.ks.kpts(),
            dms,
            &self.u,
            &self.ks.kpts.weights_ibz,
        )
        .map_err(unwrap_err)?;
        self.e_u.set(e_u);
        self.ks.tags.set(Some(tags));
        Ok(v)
    }

    fn eig(
        &self,
        fock: &KDms,
        s1e: &KMats,
    ) -> Result<(Vec<Vec<f64>>, Vec<CTensor>), PyscfRsError> {
        self.ks.eig(fock, s1e)
    }
    fn get_occ(
        &self,
        mo_energy: &[Vec<f64>],
    ) -> Result<(Vec<Vec<f64>>, Vec<f64>), PyscfRsError> {
        self.ks.get_occ(mo_energy)
    }
    fn make_rdm1(
        &self,
        mo_coeff: &[CTensor],
        mo_occ: &[Vec<f64>],
    ) -> Result<KDms, PyscfRsError> {
        self.ks.make_rdm1(mo_coeff, mo_occ)
    }

    /// `krkspu.energy_elec` (`krkspu.py:145-160`), which is itself
    /// `weights_ibz`-aware — the ksymm class reuses it unchanged.
    fn energy_elec(
        &self,
        dms: &KDms,
        h1e: &KMats,
        vhf: &KDms,
    ) -> Result<(f64, f64), PyscfRsError> {
        let (e, ecoul) = self.ks.energy_elec(dms, h1e, vhf)?;
        Ok((e + self.e_u.get(), ecoul))
    }
}

// =====================================================================
// Task 3 — `KUKS` over an IBZ k-set (`kuks_ksymm.py`, 147 l)
// =====================================================================

/// Unrestricted periodic Kohn-Sham over an irreducible k-point set.
///
/// `kuks_ksymm.py` — upstream's `KsymAdaptedKUKS`, declared
/// `class KsymAdaptedKUKS(kuks.KUKS, kuhf_ksymm.KUHF)`: the DFT half inherited
/// from `KUKS`, the k-symmetry half from `KUHF`.
///
/// This port has neither inheritance nor (yet) a `KsymAdaptedKuhf` — 17-07
/// Task 5 has not shipped one. So the k-symmetry half is taken from the two
/// **shared** helpers 17-07 made `pub`
/// ([`eig_symm_adapted`], [`ksymm_get_occ_unrestricted`]), and this type
/// implements [`KOverrideHooks`] directly with `nset() == 2`. When
/// `KsymAdaptedKuhf` lands it should take over `get_occ`/`eig` here rather
/// than duplicating them, exactly as `KsymAdaptedKrks` and `KsymAdaptedKrhf`
/// already share theirs.
///
/// `KROKS` and `KGKS` have **no** upstream `*_ksymm` module, matching KROHF's
/// absence in 17-07 Task 5. They are not invented here.
#[derive(Debug)]
pub struct KsymAdaptedKuks {
    /// The density-fitting object, over the **full BZ**.
    pub with_df: Box<dyn PeriodicDf>,
    /// The k-point symmetry, by composition (D-PBC-25).
    pub kpts: KPoints,
    kpts_ibz: Vec<[f64; 3]>,
    /// The XC functional string.
    pub xc: String,
    /// Exchange divergence treatment.
    pub exxdiv: Option<ExxDiv>,
    /// The integration grid.
    pub grids: PeriodicGrids,
    /// The numerical-integration driver, `KSet::Ibz`.
    pub ni: KNumInt,
    /// Explicit `(nalpha, nbeta)` over the FULL BZ; `None` derives it from the
    /// cell's charge and spin.
    pub nelec: Option<(usize, usize)>,
    /// Upstream's default is `true`.
    pub use_ao_symmetry: bool,
    /// `init_guess_breaksym` — `kuhf.py:417`, inherited by
    /// `KsymAdaptedKUHF`/`KsymAdaptedKUKS`. Upstream's default is `1`.
    pub init_guess_breaksym: i32,
    tags: StdCell<Option<KsEnergyTags>>,
}

impl KsymAdaptedKuks {
    /// Build with the default `FFTDF` over the full BZ and the uniform grid.
    ///
    /// # Errors
    /// Propagates the `FFTDF` and grid construction.
    pub fn new(cell: Cell, kpts: KPoints, xc: &str) -> Result<Self, PbcDftError> {
        let grids = PeriodicGrids::uniform(&cell, Some(cell.mesh))?;
        let with_df = Fftdf::new(cell, &kpts.kpts)
            .map_err(|e| err(format!("KUKS/ksymm: FFTDF construction failed: {e}")))?;
        let kpts_ibz = kpts.kpts_ibz.clone();
        let ni = KNumInt::with_symmetry(&kpts);
        Ok(Self {
            with_df: Box::new(with_df),
            kpts,
            kpts_ibz,
            xc: xc.to_string(),
            exxdiv: Some(ExxDiv::Ewald),
            grids,
            ni,
            nelec: None,
            use_ao_symmetry: true,
            init_guess_breaksym: 1,
            tags: StdCell::new(None),
        })
    }

    /// `(nalpha, nbeta)` over the **whole BZ** — the same full-zone count
    /// `khf_ksymm.py:38` uses, not an IBZ one.
    ///
    /// # Errors
    /// When the cell's electron count and spin are inconsistent.
    pub fn nelec(&self) -> Result<(usize, usize), PyscfRsError> {
        if let Some(n) = self.nelec {
            return Ok(n);
        }
        let cell = self.cell();
        let ne = cell.tot_electrons(self.kpts.nkpts()) as i64;
        let spin = cell.mol.spin as i64;
        let nalpha = (ne + spin) / 2;
        let nbeta = nalpha - spin;
        if nalpha + nbeta != ne || nalpha < 0 || nbeta < 0 {
            return Err(PyscfRsError::Core(pyscf_core::CoreError::InvalidMolecule(
                format!("KUKS/ksymm: inconsistent electron count {ne} and spin {spin}"),
            )));
        }
        Ok((nalpha as usize, nbeta as usize))
    }

    /// The full-BZ k-points.
    pub fn kpts_bz(&self) -> &[[f64; 3]] {
        &self.kpts.kpts
    }

    /// Hand the energy tags back, for the DFT+U wrapper that computes
    /// `get_veff` through this type and then adds its own term.
    pub(crate) fn set_tags(&self, tags: KsEnergyTags) {
        self.tags.set(Some(tags));
    }

    /// Run the SCF over the IBZ.
    ///
    /// # Errors
    /// Propagates every hook and the driver.
    pub fn kernel(&self, cfg: &KScfConfig) -> Result<KScfResult, PyscfRsError> {
        pyscf_pbc_scf::kernel(self, cfg)
    }

    /// `kuks_ksymm.py`'s `get_veff` — the two-channel analogue of
    /// [`KsymAdaptedKrks::get_veff_tagged`].
    ///
    /// It mirrors [`crate::kuks::Kuks::veff_from_parts`] rather than calling
    /// it, for one reason: that function derives `nkpts` from
    /// `with_df.kpts().len()` and forms `weight = 1.0 / nkpts`. Here the DF is
    /// over the FULL BZ while the weights are `weights_ibz`, so the two
    /// quantities come apart and the shared body cannot serve both. The
    /// duplication is therefore load-bearing, and this comment is why.
    ///
    /// # Errors
    /// Propagates the quadrature, the density fitting and the unfold.
    pub fn get_veff_tagged(
        &self,
        dms: &KDms,
        kpts_band: Option<&[[f64; 3]]>,
    ) -> Result<(KDms, KsEnergyTags), PbcDftError> {
        let cell = self.cell();
        let nao = cell.mol.nao_nr;
        let ground_state = kpts_band.is_none();
        // `krks_ksymm.py:41-42`, applied to the unrestricted case.
        let band: Vec<[f64; 3]> = kpts_band.map_or_else(|| self.kpts_ibz.clone(), <[_]>::to_vec);

        // S-01: ONE unfold per cycle, shared by both halves — the
        // unrestricted case unfolded FOUR times (once per spin inside
        // `nr_uks`, then both again for J/K) where two suffice. Bit-exact for
        // the reason `KsymAdaptedKrks::get_veff_tagged` records.
        let dm_bz = self.ni.unfold_kdms(cell, dms, nao)?;

        // `kuks.py:59-60` — one density SET per spin.
        let sets: [KDms; 2] = [vec![dm_bz[0].clone()], vec![dm_bz[1].clone()]];
        let nr = self
            .ni
            .nr_uks(cell, &self.grids, &self.xc, &sets, 1, Some(&band))?;
        // P-02 step 3 (U-06 step 3, applied to the k-symmetric twin): `nr` is
        // an OWNED `NrKUksResult`, so take the two `vmat` stacks out of it
        // rather than cloning `nkpts_ibz x nao^2` complex twice per cycle.
        // `nr.excsum` is read FIRST because the move invalidates that field.
        let mut exc = nr.excsum[0];
        let mut nr_vmat = nr.vmat;
        let vmat_b = nr_vmat[1].swap_remove(0);
        let vmat_a = nr_vmat[0].swap_remove(0);
        let mut vxc: KDms = vec![vmat_a, vmat_b];

        let jk = get_jk(
            self.with_df.as_ref(),
            &self.xc,
            &dm_bz,
            1,
            self.kpts_bz(),
            Some(&band),
            self.exxdiv,
            true,
        )?;
        let vj = jk
            .vj
            .ok_or_else(|| err("KUKS/ksymm: the density-fitting object returned no vj"))?;
        // `kuks.py:82` — ONE Coulomb matrix for both spins.
        let nband = vj[0].len();
        let jtot: KMats = (0..nband)
            .map(|k| {
                let mut m = vj[0][k].clone();
                for i in 0..m.re.len() {
                    m.re[i] += vj[1][k].re[i];
                    m.im[i] += vj[1][k].im[i];
                }
                m
            })
            .collect();
        for set in vxc.iter_mut() {
            for (k, m) in set.iter_mut().enumerate() {
                for i in 0..m.re.len() {
                    m.re[i] += jtot[k].re[i];
                    m.im[i] += jtot[k].im[i];
                }
            }
        }

        // `kuks.py:87` with `weights_ibz` in place of `1/nkpts`.
        let ecoul = if ground_state {
            // P-02 (the U-06 change, applied to the k-symmetric twin): trace
            // the 2-set density against ONE shared Coulomb stack instead of
            // cloning `jtot` into a two-set stack. Bit-exact.
            0.5 * self.weighted_trace_uks_shared(dms, &jtot, nao)
        } else {
            0.0
        };

        if let Some(vk) = jk.vk.as_ref() {
            // `kuks.py:89` — the FULL exchange, per spin.
            sub_scaled(&mut vxc, 1.0, vk);
            if ground_state {
                exc -= 0.5 * self.weighted_trace_uks(dms, vk, nao);
            }
        }

        Ok((
            vxc,
            KsEnergyTags {
                ecoul,
                exc,
                nelec: nr.nelec[0].0 + nr.nelec[0].1,
            },
        ))
    }

    /// `Re Σ_spin Σ_k weights_ibz[k] Tr(dm[spin][k] v[spin][k])`.
    ///
    /// The unrestricted analogue of [`KsymAdaptedKrks::weighted_trace`], and
    /// the same warning applies: `1/nkpts` here would silently drop every star
    /// multiplicity. P-02 moved the body to
    /// [`crate::veff::weighted_trace_dm_v`]; the partials are pushed in the
    /// same `(spin, k)` order the nest used, so this is bit-exact at every
    /// gated cell.
    fn weighted_trace_uks(&self, dms: &KDms, v: &[KMats], nao: usize) -> f64 {
        crate::veff::weighted_trace_dm_v(dms, v, &self.kpts.weights_ibz, nao)
    }

    /// [`KsymAdaptedKuks::weighted_trace_uks`] against ONE shared matrix
    /// stack — both spin channels traced against the same `v`.
    ///
    /// # P-02 — the two clones U-06 deleted from `kuks.rs`, one file over
    ///
    /// `get_veff_tagged` traced the spin-summed Coulomb matrix through
    /// `vec![jtot.clone(), jtot.clone()]` and `energy_elec` traced the
    /// one-electron matrix through `vec![h1e.to_vec(), h1e.to_vec()]` — two
    /// full `nkpts_ibz x nao^2` complex stacks built and dropped on every
    /// cycle, purely to satisfy a `v[s][k]` index shape. Bit-exact to remove:
    /// the operands are numerically identical and the partial order is
    /// unchanged.
    fn weighted_trace_uks_shared(&self, dms: &KDms, v: &KMats, nao: usize) -> f64 {
        crate::veff::weighted_trace_dm_v_shared(dms, v, &self.kpts.weights_ibz, nao)
    }
}

impl KOverrideHooks for KsymAdaptedKuks {
    fn cell(&self) -> &Cell {
        self.with_df.cell()
    }
    fn kpts(&self) -> &[[f64; 3]] {
        &self.kpts_ibz
    }
    fn nset(&self) -> usize {
        2
    }

    fn get_ovlp(&self) -> Result<KMats, PyscfRsError> {
        let nao = self.cell().mol.nao_nr;
        Ok(pyscf_pbc_scf::krhf::to_row_major(
            pyscf_pbc_gto::get_ovlp(self.cell(), self.kpts())?,
            nao,
        ))
    }

    fn get_hcore(&self) -> Result<KMats, PyscfRsError> {
        pyscf_pbc_df::get_hcore(self.with_df.as_ref(), self.kpts())
            .map_err(|e| unwrap_err(err(format!("KUKS/ksymm: get_hcore: {e}"))))
    }

    fn get_init_guess(&self, mode: &KInitGuess, s1e: &KMats) -> Result<KDms, PyscfRsError> {
        let (na, nb) = self.nelec()?;
        pyscf_pbc_scf::init_guess::get_init_guess(
            self.cell(),
            self.kpts_ibz.len(),
            2,
            mode,
            s1e,
            &[na as f64, nb as f64],
            self.init_guess_breaksym,
        )
    }

    fn get_veff(&self, dms: &KDms) -> Result<KDms, PyscfRsError> {
        let (v, tags) = self.get_veff_tagged(dms, None).map_err(unwrap_err)?;
        self.tags.set(Some(tags));
        Ok(v)
    }

    fn eig(
        &self,
        fock: &KDms,
        s1e: &KMats,
    ) -> Result<(Vec<Vec<f64>>, Vec<CTensor>), PyscfRsError> {
        let nao = self.nao();
        if !self.use_ao_symmetry {
            let mut es = Vec::new();
            let mut cs = Vec::new();
            for set in fock.iter() {
                let (e, c) = pyscf_pbc_scf::krhf::eig_channel(set, s1e, nao)?;
                es.extend(e);
                cs.extend(c);
            }
            return Ok((es, cs));
        }
        let cell = self.cell();
        let need = |what: &str| {
            PyscfRsError::Core(pyscf_core::CoreError::InvalidMolecule(format!(
                "KUKS/ksymm: use_ao_symmetry = true needs cell.{what}, which is unset. \
                 Call Cell::build_symmetry(&kpts) before the SCF, or set \
                 use_ao_symmetry = false."
            )))
        };
        let so = cell.symm_orb.as_deref().ok_or_else(|| need("symm_orb"))?;
        let ids = cell.irrep_id.as_deref().ok_or_else(|| need("irrep_id"))?;
        let mut es = Vec::new();
        let mut cs = Vec::new();
        for (spin, set) in fock.iter().enumerate() {
            for (k, f) in set.iter().enumerate() {
                let (e, c) =
                    eig_symm_adapted(f, &s1e[k], &so[k], &ids[k], nao).map_err(|err| {
                        PyscfRsError::Core(pyscf_core::CoreError::InvalidMolecule(format!(
                            "KUKS/ksymm: symmetry-adapted eig failed at spin {spin}, \
                             IBZ k = {k}: {err}"
                        )))
                    })?;
                es.push(e);
                cs.push(c);
            }
        }
        Ok((es, cs))
    }

    fn get_occ(
        &self,
        mo_energy: &[Vec<f64>],
    ) -> Result<(Vec<Vec<f64>>, Vec<f64>), PyscfRsError> {
        // TWO Fermi levels, each over the UNFOLDED BZ (17-CONTEXT §3.4).
        pyscf_pbc_scf::khf_ksymm::ksymm_get_occ_unrestricted(
            &self.kpts,
            mo_energy,
            self.nelec()?,
        )
    }

    fn make_rdm1(
        &self,
        mo_coeff: &[CTensor],
        mo_occ: &[Vec<f64>],
    ) -> Result<KDms, PyscfRsError> {
        let nao = self.nao();
        let nk = self.kpts_ibz.len();
        Ok(vec![
            pyscf_pbc_scf::krdm::make_rdm1(&mo_coeff[..nk], &mo_occ[..nk], nao),
            pyscf_pbc_scf::krdm::make_rdm1(&mo_coeff[nk..], &mo_occ[nk..], nao),
        ])
    }

    fn energy_elec(
        &self,
        dms: &KDms,
        h1e: &KMats,
        vhf: &KDms,
    ) -> Result<(f64, f64), PyscfRsError> {
        let nao = self.nao();
        let tags = self.tags.get().ok_or_else(|| {
            PyscfRsError::Core(pyscf_core::CoreError::InvalidMolecule(
                "KUKS/ksymm: energy_elec was called before get_veff produced its \
                 energy tags"
                    .to_string(),
            ))
        })?;
        let _ = vhf;
        // `e1` sums BOTH spin channels against the same one-electron matrix —
        // P-02: against it directly, not against two clones of it.
        let e1 = self.weighted_trace_uks_shared(dms, h1e, nao);
        Ok((e1 + tags.ecoul + tags.exc, tags.ecoul))
    }
}

/// Unrestricted periodic Kohn-Sham with a Hubbard `U`, over an irreducible
/// k-point set — `kukspu_ksymm.py:41` (`KsymAdaptedKUKSpU`).
///
/// The unrestricted twin of [`KsymAdaptedKrkspu`], and upstream's file has the
/// same shape: `kuks_ksymm.get_veff` followed by the SHARED
/// `kukspu._add_Vhubbard`, whose symmetry handling is again exactly
/// `kpts = kpts.kpts_ibz` (`kukspu.py:59`) and
/// `weight = weights_ibz` (`:78`). **No projector rotation** — D-17-08-02
/// applies unchanged to the two-channel case.
///
/// One upstream detail worth naming: `kukspu.py:68-70` applies **the same
/// `C_ao_lo` to both spins** (`if C_ao_lo[0][0].ndim != 2: C_ao_lo =
/// [C_ao_lo, C_ao_lo]`). The projectors are spin-independent, which is why
/// [`crate::kspu::add_vhubbard_weighted`] needs no spin-aware change here —
/// it already loops over whatever density channels it is handed.
#[derive(Debug)]
pub struct KsymAdaptedKukspu {
    /// The underlying k-symmetric unrestricted KS object.
    pub ks: KsymAdaptedKuks,
    /// The Hubbard configuration.
    pub u: crate::kspu::HubbardU,
    e_u: StdCell<f64>,
}

impl KsymAdaptedKukspu {
    /// Wrap a [`KsymAdaptedKuks`] with a Hubbard `U`.
    pub fn new(ks: KsymAdaptedKuks, u: crate::kspu::HubbardU) -> Self {
        Self {
            ks,
            u,
            e_u: StdCell::new(0.0),
        }
    }

    /// `E_U` of the last `get_veff`.
    pub fn e_u(&self) -> f64 {
        self.e_u.get()
    }

    /// Run the SCF over the IBZ.
    ///
    /// # Errors
    /// Propagates every hook and the driver.
    pub fn kernel(&self, cfg: &KScfConfig) -> Result<KScfResult, PyscfRsError> {
        pyscf_pbc_scf::kernel(self, cfg)
    }
}

impl KOverrideHooks for KsymAdaptedKukspu {
    fn cell(&self) -> &Cell {
        self.ks.cell()
    }
    fn kpts(&self) -> &[[f64; 3]] {
        self.ks.kpts()
    }
    fn nset(&self) -> usize {
        2
    }
    fn get_ovlp(&self) -> Result<KMats, PyscfRsError> {
        self.ks.get_ovlp()
    }
    fn get_hcore(&self) -> Result<KMats, PyscfRsError> {
        self.ks.get_hcore()
    }
    fn get_init_guess(&self, mode: &KInitGuess, s1e: &KMats) -> Result<KDms, PyscfRsError> {
        self.ks.get_init_guess(mode, s1e)
    }

    /// `kukspu_ksymm.py:25-39` — the KS potential, then the Hubbard term at the
    /// IBZ points with `weights_ibz`.
    fn get_veff(&self, dms: &KDms) -> Result<KDms, PyscfRsError> {
        let (mut v, tags) = self.ks.get_veff_tagged(dms, None).map_err(unwrap_err)?;
        let e_u = crate::kspu::add_vhubbard_weighted(
            &mut v,
            self.cell(),
            self.ks.kpts(),
            dms,
            &self.u,
            &self.ks.kpts.weights_ibz,
        )
        .map_err(unwrap_err)?;
        self.e_u.set(e_u);
        self.ks.set_tags(tags);
        Ok(v)
    }

    fn eig(
        &self,
        fock: &KDms,
        s1e: &KMats,
    ) -> Result<(Vec<Vec<f64>>, Vec<CTensor>), PyscfRsError> {
        self.ks.eig(fock, s1e)
    }
    fn get_occ(
        &self,
        mo_energy: &[Vec<f64>],
    ) -> Result<(Vec<Vec<f64>>, Vec<f64>), PyscfRsError> {
        self.ks.get_occ(mo_energy)
    }
    fn make_rdm1(
        &self,
        mo_coeff: &[CTensor],
        mo_occ: &[Vec<f64>],
    ) -> Result<KDms, PyscfRsError> {
        self.ks.make_rdm1(mo_coeff, mo_occ)
    }

    /// `kukspu.energy_elec` (`kukspu.py:125-140`), itself `weights_ibz`-aware.
    fn energy_elec(
        &self,
        dms: &KDms,
        h1e: &KMats,
        vhf: &KDms,
    ) -> Result<(f64, f64), PyscfRsError> {
        let (e, ecoul) = self.ks.energy_elec(dms, h1e, vhf)?;
        Ok((e + self.e_u.get(), ecoul))
    }
}
