//! `density_fit(mf, auxbasis, mesh, with_df)` — the four `mf.density_fit()`
//! shims, in one place. Plan 14-08, Task 3.
//!
//! Upstream has four near-identical copies: `df_jk.density_fit` (`df_jk.py`),
//! `mdf_jk.density_fit` (`mdf_jk.py:24-66`), `rsdf_jk.density_fit`
//! (`rsdf_jk.py`) and `aft_jk.density_fit`. Each builds one `with_df` object,
//! copies `max_memory` / `stdout` / `verbose` off the mean-field object, sets
//! `auxbasis` (and, for MDF, `mesh`), and returns a copy of `mf` with
//! `mf.with_df` swapped and `mf._eri` cleared. Plan 14-08 Task 3 asks for all
//! four to land together and share one implementation; this is it.
//!
//! # The shape is different here, and deliberately
//!
//! Upstream mutates a mean-field object because `with_df` is a mutable
//! attribute on it. This port's drivers take the builder at construction —
//! `Krhf::from_df(Box<dyn PeriodicDf>)` (D-PBC-22) — so the shim's job is to
//! *produce the builder*, not to patch a driver. The verbosity/memory copying
//! has no analogue: those live on the cell here.
//!
//! ```ignore
//! let df = density_fit(cell, &kpts, DfKind::Mdf, DfOpts::default())?;
//! let mf = Krhf::from_df(df);
//! ```

use pyscf_pbc_gto::Cell;

use crate::error::PbcDfError;
use crate::traits::PeriodicDf;

/// Which builder `density_fit` should produce.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DfKind {
    /// `fft.FFTDF` — exact, plane-wave, no fitting (Phase 11).
    Fftdf,
    /// `aft.AFTDF` — exact, analytic Fourier transform (Phase 13).
    Aftdf,
    /// `df.GDF` — Gaussian density fitting (plans 14-01 … 14-04).
    Gdf,
    /// `mdf.MDF` — GDF plus the plane-wave residual (plan 14-06).
    Mdf,
    /// `rsdf.RSGDF` — range-separated (plan 14-07 7b/7c); see [`crate::rsdf`].
    Rsdf,
}

impl DfKind {
    /// The name the produced builder reports.
    pub fn name(self) -> &'static str {
        match self {
            Self::Fftdf => "FFTDF",
            Self::Aftdf => "AFTDF",
            Self::Gdf => "GDF",
            Self::Mdf => "MDF",
            Self::Rsdf => "RSDF",
        }
    }
}

/// The two knobs upstream's shims accept.
#[derive(Debug, Clone, Default)]
pub struct DfOpts {
    /// `auxbasis` — `None` runs `make_auxbasis`. Ignored by the two exact
    /// builders, which have no auxiliary basis.
    pub auxbasis: Option<String>,
    /// `mesh` — the plane-wave mesh. Meaningful for FFTDF, AFTDF and MDF;
    /// ignored by GDF, whose own mesh is the compensating-charge one.
    pub mesh: Option<[usize; 3]>,
}

/// Build one `with_df` object.
///
/// # Errors
/// Propagates the builder's construction.
pub fn density_fit(
    cell: Cell,
    kpts: &[[f64; 3]],
    kind: DfKind,
    opts: DfOpts,
) -> Result<Box<dyn PeriodicDf>, PbcDfError> {
    Ok(match kind {
        DfKind::Fftdf => match opts.mesh {
            Some(m) => Box::new(crate::Fftdf::with_mesh(cell, kpts, m)?),
            None => Box::new(crate::Fftdf::new(cell, kpts)?),
        },
        DfKind::Aftdf => match opts.mesh {
            Some(m) => Box::new(crate::Aftdf::with_mesh(cell, kpts, m)?),
            None => Box::new(crate::Aftdf::new(cell, kpts)?),
        },
        DfKind::Gdf => {
            let mut d = crate::Gdf::new(cell, kpts);
            d.auxbasis = opts.auxbasis;
            Box::new(d)
        }
        DfKind::Mdf => {
            let mut d = crate::Mdf::new(cell, kpts);
            d.auxbasis = opts.auxbasis;
            d.mesh = opts.mesh;
            Box::new(d)
        }
        DfKind::Rsdf => {
            let mut d = crate::Rsdf::new(cell, kpts);
            d.gdf.auxbasis = opts.auxbasis;
            Box::new(d)
        }
    })
}
