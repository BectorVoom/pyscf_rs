//! `GDF` — Gaussian density fitting (`pyscf/pbc/df/df.py:125-611`), plan 14-03.
//!
//! # What GDF is for
//!
//! FFTDF and AFTDF evaluate the periodic Coulomb integrals *exactly*, at a cost
//! set by the plane-wave mesh. GDF **fits** them in a Gaussian auxiliary basis:
//! the answer changes — by **1.222e-03 Ha** on diamond `gth-szv` 2×2×2, the DF
//! fitting error, which is a property of the auxiliary basis and not of any
//! implementation — but the integrals become a small on-disk tensor that every
//! later contraction reads instead of a grid.
//!
//! Upstream's own numbers on that system: 6.4 s for a GDF-driven `KRHF` against
//! 30.0 s for FFTDF and 450.6 s for AFTDF, with `_cderi` at 3.86 MiB against
//! FFTDF's 62.5 MiB AO table (`measurements/README.md`). That is the whole
//! case for the phase.
//!
//! # Which builder this drives
//!
//! Upstream's `GDF._prefer_ccdf = False`, so `df.GDF()` runs
//! `rsdf_builder._RSGDFBuilder`. **This port ships `_CCGDFBuilder` first**
//! (plan 14-02) because it is the self-contained one, so [`Gdf::prefer_ccdf`]
//! defaults to `true` here and plan 14-07 flips it. The two routes are not
//! interchangeable: upstream's own disagree by **5.960e-07** on diamond 2×2×2,
//! **4.502e-06** at gamma and **5.222e-10** on He-fcc (`measurements/ccdf.py`),
//! which is why the flip is a recorded decision rather than a default.

pub mod cderi_store;
pub mod jk;
pub mod nuc;

use pyscf_algebra::CTensor;
use pyscf_pbc_gto::Cell;

use crate::df_jk::KMats;
use crate::error::PbcDfError;
use crate::gdf_builder::{CcGdfBuilder, j3c::Cderi};
use crate::incore::Aosym;
use crate::traits::{JkOpts, JkResult, PeriodicDf};

pub use cderi_store::{CderiFile, SrBlock, get_naoaux, sr_loop};

/// `GDF` — `df.py:125-611`.
#[derive(Debug)]
pub struct Gdf {
    /// The cell.
    pub cell: Cell,
    /// The sampling k-points.
    pub kpts: Vec<[f64; 3]>,
    /// Auxiliary basis name; `None` runs `make_auxbasis`.
    pub auxbasis: Option<String>,
    /// `exp_to_discard` — refused when set (plan 14-01).
    pub exp_to_discard: Option<f64>,
    /// How the `(mu, nu)` index is packed on disk.
    pub aosym: Aosym,
    /// `j_only` — build only the diagonal `(k, k)` pairs. J needs no more.
    pub j_only: bool,
    /// See the module docs. `true` here until plan 14-07 ships the RS route.
    pub prefer_ccdf: bool,
    /// Force the eigenvalue-decomposed metric.
    pub j2c_eig_always: bool,
    /// Where to keep `_cderi`. `None` keeps it in memory only.
    pub cderi_to_save: Option<std::path::PathBuf>,
    /// The fitted tensor. Built LAZILY on first use, matching upstream's
    /// `if mydf._cderi is None: mydf.build(...)` at the head of `get_j_kpts`
    /// and `get_k_kpts` (`df_jk.py:86-92`, `:292-299`) — the k-point SCF
    /// drivers hand the builder straight to `get_jk` and never call `build`.
    cderi: std::sync::OnceLock<Cderi>,
    /// The file handle, when `cderi_to_save` was set.
    file: std::sync::Mutex<Option<CderiFile>>,
    /// The compensated builder, retained for its `eta` / `mesh` / fused cell.
    /// Built lazily too, and much cheaper than `cderi` — `mesh()` needs only
    /// this.
    builder: std::sync::OnceLock<CcGdfBuilder>,
}

impl Gdf {
    /// A `GDF` on `cell` at `kpts`, with upstream's defaults.
    pub fn new(cell: Cell, kpts: &[[f64; 3]]) -> Self {
        Self {
            cell,
            kpts: if kpts.is_empty() {
                vec![[0.0; 3]]
            } else {
                kpts.to_vec()
            },
            auxbasis: None,
            exp_to_discard: None,
            aosym: Aosym::S2,
            j_only: false,
            prefer_ccdf: true,
            j2c_eig_always: false,
            cderi_to_save: None,
            cderi: std::sync::OnceLock::new(),
            file: std::sync::Mutex::new(None),
            builder: std::sync::OnceLock::new(),
        }
    }

    /// A `GDF` whose `cderi` is ALREADY built — upstream's
    /// `mydf._cderi = <path or array>`, which `df.py:253-289` honours by
    /// skipping the build entirely.
    ///
    /// The k-points come from the store, so a file written by this port (or by
    /// PySCF, through [`CderiFile::load`]) can drive `get_jk`, `sr_loop` and
    /// [`crate::df_ao2mo`] without re-running `make_j3c`. It is also what lets
    /// `tests/df_ao2mo.rs` exercise the contraction algebra on a synthetic
    /// tensor in milliseconds instead of the minutes a real build costs.
    pub fn with_cderi(cell: Cell, cderi: Cderi) -> Self {
        let kpts = cderi.kpts.clone();
        let aosym = cderi.aosym;
        let mut df = Self::new(cell, &kpts);
        df.aosym = aosym;
        let _ = df.cderi.set(cderi);
        df
    }

    /// A `GDF` reading an existing `_cderi` file.
    ///
    /// # Errors
    /// Propagates [`CderiFile::load`].
    pub fn load_cderi(cell: Cell, path: impl AsRef<std::path::Path>) -> Result<Self, PbcDfError> {
        Ok(Self::with_cderi(cell, CderiFile::load(path)?))
    }

    /// The compensated builder, built on first use. Cheap — it stops at the
    /// fused auxiliary cell and does no 3-centre work.
    ///
    /// # Errors
    /// As [`Gdf::build`].
    pub fn builder(&self) -> Result<&CcGdfBuilder, PbcDfError> {
        if let Some(b) = self.builder.get() {
            return Ok(b);
        }
        self.refuse_unsupported()?;
        let mut b = CcGdfBuilder::new(self.cell.clone(), &self.kpts);
        b.auxbasis = self.auxbasis.clone();
        b.j2c_eig_always = self.j2c_eig_always;
        b.build()?;
        let _ = self.builder.set(b);
        self.builder.get().ok_or_else(|| {
            PbcDfError::Core(pyscf_core::PyscfRsError::Core(
                pyscf_core::CoreError::InvalidMolecule("GDF: builder init raced".into()),
            ))
        })
    }

    /// The fitted tensor, built on first use.
    ///
    /// # Errors
    /// As [`Gdf::build`].
    pub fn cderi(&self) -> Result<&Cderi, PbcDfError> {
        if let Some(c) = self.cderi.get() {
            return Ok(c);
        }
        let b = self.builder()?;
        let c = b.make_j3c(self.aosym, self.j_only)?;
        if let Some(p) = self.cderi_to_save.clone() {
            let mut f = CderiFile::save(&c, &p, true)?;
            f.keep();
            if let Ok(mut slot) = self.file.lock() {
                *slot = Some(f);
            }
        }
        let _ = self.cderi.set(c);
        self.cderi.get().ok_or_else(|| {
            PbcDfError::Core(pyscf_core::PyscfRsError::Core(
                pyscf_core::CoreError::InvalidMolecule("GDF: cderi init raced".into()),
            ))
        })
    }

    /// The `_cderi` file path, if one was written.
    pub fn cderi_path(&self) -> Option<std::path::PathBuf> {
        self.file
            .lock()
            .ok()
            .and_then(|s| s.as_ref().map(|f| f.path().to_path_buf()))
    }

    fn refuse_unsupported(&self) -> Result<(), PbcDfError> {
        if !self.prefer_ccdf {
            return Err(PbcDfError::Core(
                pyscf_core::PyscfRsError::NotYetImplemented {
                    phase: 14,
                    what: "GDF via rsdf_builder._RSGDFBuilder — plan 14-07. It is \
                           upstream's DEFAULT route and differs from the compensated \
                           one by up to 4.502e-06 (measurements/ccdf.py)",
                },
            ));
        }
        if self.exp_to_discard.is_some() {
            return Err(PbcDfError::Core(
                pyscf_core::PyscfRsError::NotYetImplemented {
                    phase: 14,
                    what: "GDF.exp_to_discard — make_modrho_basis(drop_eta) changes naux \
                           silently (df.py:88-92)",
                },
            ));
        }
        Ok(())
    }

    /// `build(j_only, with_j3c, kpts_band)` — `df.py:253-289`. Eager; the
    /// lazy path is [`Gdf::cderi`].
    ///
    /// # Errors
    /// [`PyscfRsError::NotYetImplemented`] when `prefer_ccdf` is `false`
    /// (plan 14-07 owns the range-separated route) or `exp_to_discard` is set,
    /// and propagates the builder.
    pub fn build(&mut self) -> Result<(), PbcDfError> {
        self.cderi()?;
        Ok(())
    }

    /// `sr_loop(kpti_kptj, ..., compact)` — `df.py:338-400`.
    ///
    /// # Errors
    /// As [`Gdf::cderi`], and when the pair was not built (`j_only`).
    pub fn sr_loop(&self, ki: usize, kj: usize, compact: bool) -> Result<Vec<SrBlock>, PbcDfError> {
        sr_loop(self.cderi()?, ki, kj, self.cell.mol.nao_nr, compact)
    }

    /// `get_naoaux()` — `df.py:568-611`.
    ///
    /// # Errors
    /// As [`Gdf::cderi`] and [`Cderi::naoaux`].
    pub fn get_naoaux(&self) -> Result<usize, PbcDfError> {
        get_naoaux(self.cderi()?)
    }
}

impl PeriodicDf for Gdf {
    fn cell(&self) -> &Cell {
        &self.cell
    }
    fn mesh(&self) -> [usize; 3] {
        // GDF's own mesh is the compensating-charge one — TINY next to FFTDF's
        // ([11,11,11] against [47,47,47] on diamond), because it only has to
        // resolve the model charge, not the density.
        self.builder()
            .ok()
            .and_then(|b| b.eta)
            .map_or([1, 1, 1], |e| e.mesh)
    }
    fn kpts(&self) -> &[[f64; 3]] {
        &self.kpts
    }
    fn build(&mut self) -> Result<(), PbcDfError> {
        Gdf::build(self)
    }
    fn get_nuc(&self, kpts: &[[f64; 3]]) -> Result<Vec<CTensor>, PbcDfError> {
        nuc::get_nuc(&self.cell, kpts)
    }
    fn get_pp(&self, kpts: &[[f64; 3]]) -> Result<Vec<CTensor>, PbcDfError> {
        nuc::get_pp(&self.cell, kpts)
    }
    fn name(&self) -> &'static str {
        "GDF"
    }
    fn get_jk(
        &self,
        dms: &[KMats],
        kpts: &[[f64; 3]],
        opts: JkOpts<'_>,
    ) -> Result<JkResult, PbcDfError> {
        jk::get_jk(self, dms, kpts, opts)
    }
}
