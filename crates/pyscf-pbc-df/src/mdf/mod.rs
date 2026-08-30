//! `MDF` — mixed density fitting (`pyscf/pbc/df/mdf.py:49-236`), plan 14-06.
//!
//! # What MDF buys, in one table
//!
//! `|E_KRHF(builder) - E_KRHF(FFTDF, mesh 31)|`, upstream, diamond `gth-szv`
//! 2x2x2 (`measurements/mdfladder.out`, `builders.out`):
//!
//! ```text
//! GDF                        1.222e-03    the DF fitting error
//! MDF, mesh 7                1.124e-06
//! MDF, mesh 21               3.241e-10    the plateau
//! ```
//!
//! Three to six orders. GDF **fits** the Coulomb integrals in a Gaussian
//! auxiliary basis and stops; MDF carries the residual the Gaussians cannot fit
//! on a plane-wave grid, so it converges to the exact builder as that grid
//! rises. That is the only reason a cross-builder gate against FFTDF is
//! meaningful at all — see `14-CONTEXT.md` § "The gates the roadmap gets
//! wrong".
//!
//! # Composition, not a second implementation
//!
//! `MDF` is `GDF` plus the AFT residual, and **both halves are already
//! shipped**:
//!
//! * the Gaussian half is 14-02's `make_j3c` driven with
//!   [`crate::gdf_builder::j3c::Scheme::Mixed`] and 14-04's `df_jk` /
//!   14-05's `df_ao2mo` contractions, reached through an inner [`Gdf`] holding
//!   the MDF `cderi`;
//! * the plane-wave half is Phase 13's `aft_jk` / `pbc_ao2mo`, reached through
//!   an inner [`Aftdf`] at MDF's mesh with
//!   [`Aftdf::mdf_pw_edge_screen`] set.
//!
//! Nothing here re-implements a contraction. Upstream's `mdf_jk` and
//! `mdf_ao2mo` are exactly this thin too (149 + 176 lines, mostly `+=`).
//!
//! # Both builders ship
//!
//! `MDF._prefer_ccdf = False` (`mdf.py:79`), so `df.MDF()` runs
//! `_RSMDFBuilder`. 14-06 shipped `_CCMDFBuilder` alone, mirroring 14-02's
//! choice for GDF, because the range-separated route was blocked on cintx.
//! D-PBC-24 lifted that and plan 14-07 7b/7c ported `_RSGDFBuilder`;
//! `_RSMDFBuilder` is that builder with `mixed` set
//! ([`crate::rsdf_builder::RsGdfBuilder::new_mixed`]), since `mdf.py:238-353`
//! is a subclass overriding exactly three methods.
//!
//! [`Mdf::prefer_ccdf`] still defaults to `true` here, as [`Gdf::prefer_ccdf`]
//! does — flipping either moves a committed reference energy and plan 14-07
//! Task 7d requires that be its own cited edit.
//! **`measurements/mdfladder.out` was recorded on the RS route** and is now
//! reachable; `mdfladder_cc.py` records the CC ladder, which is what
//! `tests/mdf.rs` asserts against today.

pub mod builder;
pub mod mdf_ao2mo;
pub mod mdf_jk;

use pyscf_algebra::CTensor;
use pyscf_pbc_gto::Cell;

use crate::aftdf::Aftdf;
use crate::df_jk::KMats;
use crate::error::PbcDfError;
use crate::gdf::Gdf;
use crate::gdf_builder::CcGdfBuilder;
use crate::gdf_builder::j3c::{Cderi, Scheme, make_j3c_scheme};
use crate::incore::Aosym;
use crate::traits::{JkOpts, JkResult, PeriodicDf};

/// `MDF` — `mdf.py:49-236`.
#[derive(Debug)]
pub struct Mdf {
    /// The cell.
    pub cell: Cell,
    /// The sampling k-points.
    pub kpts: Vec<[f64; 3]>,
    /// Auxiliary basis name; `None` runs `make_auxbasis`.
    pub auxbasis: Option<String>,
    /// The plane-wave mesh the residual is carried on. `None` takes the
    /// builder's own estimate — `[11,11,11]` on diamond 2x2x2, `[9,9,9]` on
    /// He-fcc 2x2x2 (measured; **`14-06-PLAN.md`'s `[7,7,7]` is wrong** — 7 is
    /// simply `mdfladder.py`'s lowest rung, not a default).
    pub mesh: Option<[usize; 3]>,
    /// How the `(mu, nu)` index is packed on disk.
    pub aosym: Aosym,
    /// `j_only` — build only the diagonal `(k, k)` pairs.
    pub j_only: bool,
    /// See the module docs. `false` selects `_RSMDFBuilder`, upstream's
    /// default; `true` (this port's default) selects `_CCMDFBuilder`.
    pub prefer_ccdf: bool,
    /// `exp_to_discard` — refused when set, as [`Gdf`] does.
    pub exp_to_discard: Option<f64>,
    inner: std::sync::OnceLock<Gdf>,
    aft: std::sync::OnceLock<Aftdf>,
    resolved_mesh: std::sync::OnceLock<[usize; 3]>,
}

impl Mdf {
    /// An `MDF` on `cell` at `kpts`, with upstream's defaults.
    pub fn new(cell: Cell, kpts: &[[f64; 3]]) -> Self {
        Self {
            cell,
            kpts: if kpts.is_empty() {
                vec![[0.0; 3]]
            } else {
                kpts.to_vec()
            },
            auxbasis: None,
            mesh: None,
            aosym: Aosym::S2,
            j_only: false,
            prefer_ccdf: true,
            exp_to_discard: None,
            inner: std::sync::OnceLock::new(),
            aft: std::sync::OnceLock::new(),
            resolved_mesh: std::sync::OnceLock::new(),
        }
    }

    fn refuse_unsupported(&self) -> Result<(), PbcDfError> {
        // `prefer_ccdf = false` is no longer refused: plan 14-07 sub-tasks
        // 7b/7c shipped `_RSGDFBuilder` on D-PBC-24's cintx `range_omega`, and
        // `_RSMDFBuilder` is that builder with `mixed` set (`mdf.py:238-353`
        // is a subclass overriding three methods). `measurements/mdfladder.out`
        // — recorded on THIS route — is now reachable.
        if self.exp_to_discard.is_some() {
            return Err(PbcDfError::Core(
                pyscf_core::PyscfRsError::NotYetImplemented {
                    phase: 14,
                    what: "MDF.exp_to_discard — make_modrho_basis(drop_eta) changes \
                           naux silently (mdf.py:74-77)",
                },
            ));
        }
        Ok(())
    }

    /// The range-separated builder MDF's Gaussian half runs on when
    /// [`Mdf::prefer_ccdf`] is `false` — upstream's default — and the mesh it
    /// picked.
    ///
    /// # Errors
    /// Propagates the auxiliary-cell build, and refuses the unsupported flags.
    fn rs_builder(&self) -> Result<(crate::rsdf_builder::RsGdfBuilder, [usize; 3]), PbcDfError> {
        self.refuse_unsupported()?;
        let mut b = crate::rsdf_builder::RsGdfBuilder::new_mixed(self.cell.clone(), &self.kpts);
        b.auxbasis = self.auxbasis.clone();
        b.mesh = self.mesh;
        b.build()?;
        let guess = b.mesh.unwrap_or([1, 1, 1]);
        Ok((b, self.mesh.unwrap_or(guess)))
    }

    /// The compensated builder MDF's Gaussian half runs on, and the mesh it
    /// picked.
    ///
    /// # Errors
    /// Propagates the auxiliary-cell build, and refuses the unsupported flags.
    fn cc_builder(&self) -> Result<(CcGdfBuilder, [usize; 3]), PbcDfError> {
        self.refuse_unsupported()?;
        let mut b = CcGdfBuilder::new(self.cell.clone(), &self.kpts);
        b.auxbasis = self.auxbasis.clone();
        // `mdf.py:362-365` — never Cholesky. See `builder`'s module docs.
        b.j2c_eig_always = true;
        b.build()?;
        let guess = b.eta.map_or([1, 1, 1], |e| e.mesh);
        Ok((b, self.mesh.unwrap_or(guess)))
    }

    /// The plane-wave mesh actually in use.
    ///
    /// # Errors
    /// As [`Mdf::build`].
    pub fn resolved_mesh(&self) -> Result<[usize; 3], PbcDfError> {
        if let Some(m) = self.resolved_mesh.get() {
            return Ok(*m);
        }
        if let Some(m) = self.mesh {
            let _ = self.resolved_mesh.set(m);
            return Ok(m);
        }
        let m = if self.prefer_ccdf {
            self.cc_builder()?.1
        } else {
            self.rs_builder()?.1
        };
        let _ = self.resolved_mesh.set(m);
        Ok(m)
    }

    /// The Gaussian half: a [`Gdf`] carrying the MDF `cderi`.
    ///
    /// # Errors
    /// Propagates every stage of the 3-centre build.
    pub fn gdf(&self) -> Result<&Gdf, PbcDfError> {
        if let Some(g) = self.inner.get() {
            return Ok(g);
        }
        let cderi: Cderi = if self.prefer_ccdf {
            let (b, mesh) = self.cc_builder()?;
            let _ = self.resolved_mesh.set(mesh);
            let (Some(fused), Some(_eta)) = (b.fused.as_ref(), b.eta) else {
                return Err(PbcDfError::Core(pyscf_core::PyscfRsError::Core(
                    pyscf_core::CoreError::InvalidMolecule("MDF: builder did not build".into()),
                )));
            };
            let rcut = crate::gdf_builder::estimate_rcut(&self.cell, &fused.fused.cell, None);
            make_j3c_scheme(
                &self.cell,
                fused,
                &self.kpts,
                self.aosym,
                mesh,
                self.j_only,
                true,
                Some(rcut),
                Scheme::Mixed,
            )?
        } else {
            // `_RSMDFBuilder` — the mesh and rcut are its own, and
            // `make_j3c` already knows the scheme from `mixed`.
            let (b, mesh) = self.rs_builder()?;
            let _ = self.resolved_mesh.set(mesh);
            b.make_j3c(self.aosym, self.j_only)?
        };
        let _ = self.inner.set(Gdf::with_cderi(self.cell.clone(), cderi));
        self.inner.get().ok_or_else(|| {
            PbcDfError::Core(pyscf_core::PyscfRsError::Core(
                pyscf_core::CoreError::InvalidMolecule("MDF: inner GDF init raced".into()),
            ))
        })
    }

    /// The plane-wave half: an [`Aftdf`] at MDF's mesh, with MDF's edge screen.
    ///
    /// # Errors
    /// Propagates the AFTDF build.
    pub fn aftdf(&self) -> Result<&Aftdf, PbcDfError> {
        if let Some(a) = self.aft.get() {
            return Ok(a);
        }
        let mesh = self.resolved_mesh()?;
        let mut a = Aftdf::with_mesh(self.cell.clone(), &self.kpts, mesh)?;
        a.mdf_pw_edge_screen = true;
        let _ = self.aft.set(a);
        self.aft.get().ok_or_else(|| {
            PbcDfError::Core(pyscf_core::PyscfRsError::Core(
                pyscf_core::CoreError::InvalidMolecule("MDF: inner AFTDF init raced".into()),
            ))
        })
    }

    /// `build(j_only, with_j3c, kpts_band)` — `mdf.py:100-113`. Eager.
    ///
    /// # Errors
    /// As [`Mdf::gdf`].
    pub fn build(&mut self) -> Result<(), PbcDfError> {
        self.gdf()?;
        self.aftdf()?;
        let mesh = self.resolved_mesh()?;
        // `mdf.py:105-112` — an even axis produces plane waves without a
        // `-G` counterpart (`np.fft.fftfreq`), which breaks the Hermiticity of
        // J and K. Upstream warns rather than refusing; so does this.
        if mesh
            .iter()
            .take(self.cell.dimension as usize)
            .any(|m| m % 2 == 0)
        {
            tracing::warn!(
                "MDF with an even number in mesh {mesh:?} may have significant errors: \
                 the plane waves have no -G counterpart, which breaks the hermiticity \
                 of J and K"
            );
        }
        Ok(())
    }

    /// `get_naoaux()` — `mdf.py:235-236`. The Gaussian rank ONLY; upstream adds
    /// `AFTDF.get_naoaux()` (the grid size) on top, which is a plane-wave count
    /// and not a `cderi` rank.
    ///
    /// # Errors
    /// As [`Mdf::gdf`].
    pub fn get_naoaux(&self) -> Result<usize, PbcDfError> {
        self.gdf()?.get_naoaux()
    }

    /// `sr_loop` on the Gaussian half.
    ///
    /// # Errors
    /// As [`crate::gdf::sr_loop`].
    pub fn sr_loop(
        &self,
        ki: usize,
        kj: usize,
        compact: bool,
    ) -> Result<Vec<crate::gdf::SrBlock>, PbcDfError> {
        self.gdf()?.sr_loop(ki, kj, compact)
    }
}

impl PeriodicDf for Mdf {
    fn cell(&self) -> &Cell {
        &self.cell
    }
    fn mesh(&self) -> [usize; 3] {
        self.resolved_mesh().unwrap_or([1, 1, 1])
    }
    fn kpts(&self) -> &[[f64; 3]] {
        &self.kpts
    }
    fn build(&mut self) -> Result<(), PbcDfError> {
        Mdf::build(self)
    }
    fn get_nuc(&self, kpts: &[[f64; 3]]) -> Result<Vec<CTensor>, PbcDfError> {
        // `mdf.py:174-175` — `get_pp = df.GDF.get_pp`, `get_nuc = df.GDF.get_nuc`.
        crate::gdf::nuc::get_nuc(&self.cell, kpts)
    }
    fn get_pp(&self, kpts: &[[f64; 3]]) -> Result<Vec<CTensor>, PbcDfError> {
        crate::gdf::nuc::get_pp(&self.cell, kpts)
    }
    fn name(&self) -> &'static str {
        "MDF"
    }
    fn get_jk(
        &self,
        dms: &[KMats],
        kpts: &[[f64; 3]],
        opts: JkOpts<'_>,
    ) -> Result<JkResult, PbcDfError> {
        mdf_jk::get_jk(self, dms, kpts, opts)
    }
}
