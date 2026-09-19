//! `get_ovlp` and `get_hcore` — the k-resolved one-electron matrices a periodic
//! SCF driver consumes (plan 10-07).
//!
//! Ports `pyscf/pbc/scf/scfint.py:37-71` (`get_hcore`, `get_ovlp`, `get_t`) and
//! the `get_pp` assembly of `pyscf/pbc/gto/pseudo/pp_int.py`.
//!
//! ```text
//! S^k = pbc_intor("int1e_ovlp", k)
//! T^k = pbc_intor("int1e_kin",  k)
//! H^k = T^k + V^k,   V^k = get_pp(k)      for a pseudopotential cell
//!                    V^k = get_nuc(k)     for an all-electron cell  (Phase 11)
//! ```
//!
//! # What Phase 10 can finish
//!
//! `get_pp = V_loc,1 + V_loc,2 + V_nl` (`pp_int.py`), and only two of the three
//! terms are Phase-10 work:
//!
//! | term | status |
//! |---|---|
//! | `V_nl` | complete, k-resolved — [`crate::pseudo::get_pp_nl`] |
//! | `V_loc,2` | complete at GAMMA — [`crate::pseudo::get_pp_loc_part2`] |
//! | `V_loc,1` | **Phase 11** — upstream's `pp_int.get_pp_loc_part1` raises `NotImplementedError` and defers to FFTDF (`ifft(vlocG · SI)`) or AFTDF (`ft_aopair`). The G-space factor it needs, [`crate::pseudo::get_gth_vlocg_part1`], IS finished here. |
//!
//! So [`get_hcore`] returns `NotYetImplemented { phase: 11 }` — it cannot honestly
//! do otherwise — while [`get_hcore_parts`] hands back every piece Phase 10 owns
//! so a caller (and Phase 11's FFTDF) can assemble the rest. The all-electron
//! branch is the same story with `get_nuc` in place of `V_loc,1`.

use crate::cell::Cell;
use crate::pbc_intor::{PbcIntorOpts, pbc_intor};
use pyscf_algebra::CTensor;
use pyscf_core::PyscfRsError;

/// `get_ovlp(cell, kpts)` — `scfint.py:64-71`.
///
/// One `nao x nao` F-order `CTensor` per k-point. An empty `kpts` means the
/// single gamma point.
///
/// Upstream passes `hermi=1` so only the lower triangle is evaluated and the
/// rest mirrored; this port does the same, which is both faster and exactly
/// Hermitian by construction.
///
/// **Not the SCF overlap.** Every periodic SCF driver consumes
/// [`get_ovlp_scf`] (`pbc/scf/hf.py:get_ovlp`, tightened precision). This one
/// is the plain `cell.pbc_intor('int1e_ovlp', hermi=1)` that `krkspu.py`,
/// `scf/addons.py` and `df_jk.py` call directly.
///
/// # Errors
/// As [`crate::pbc_intor::intor_cross`].
pub fn get_ovlp(cell: &Cell, kpts: &[[f64; 3]]) -> Result<Vec<CTensor>, PyscfRsError> {
    Ok(pbc_intor(
        cell,
        "int1e_ovlp",
        kpts,
        PbcIntorOpts {
            comp: None,
            hermi: 1,
            screen: cell.use_loose_rcut,
            omega: None,
        },
    )?
    .kmats)
}

/// The factor upstream's SCF overlap tightens `cell.precision` by —
/// `precision = cell.precision * 1e-5` (`pbc/scf/hf.py:50`).
pub const SCF_OVLP_PRECISION_FACTOR: f64 = 1e-5;

/// `get_ovlp(cell, kpt)` — `pbc/scf/hf.py:47-76`. **The overlap every periodic
/// SCF consumes** (`SCF.get_ovlp` `:645-648`, `khf.get_ovlp` / `KSCF.get_ovlp`
/// `khf.py:52-63,457`, and through them `KsymAdaptedKSCF`, `KGHF`, the KS
/// classes and `get_bands`' `s1e`).
///
/// It is NOT [`get_ovlp`] (`scfint.get_ovlp`, plain `cell.precision`). Upstream
/// evaluates the lattice sum as
///
/// ```text
/// precision = cell.precision * 1e-5
/// rcut      = max(cell.rcut, estimate_rcut(cell, precision))
/// with temporary_env(cell, rcut=rcut, precision=precision):
///     s = cell.pbc_intor('int1e_ovlp', hermi=0, kpts=kpt, pbcopt=NULL)
/// ```
///
/// which this ports line by line:
///
/// * `rcut` widens the image list `Ls` (`cell.py:222-223`).
/// * `precision` reaches the lattice sum only through the neighbor list of the
///   `use_loose_rcut` route (`_intor_cross_screened` -> `rcut_by_shells(precision)`,
///   `neighborlist.py:87-88`); the plain `intor_cross` route never reads it.
/// * `hermi=0` — the full `s1` fill, NOT mirrored, so `S` is Hermitian only to
///   rounding, exactly as upstream's is.
/// * `pbcopt=lib.c_null_ptr()` is swallowed by `**kwargs` in 2.12.1's
///   `intor_cross`, so it changes nothing and has no counterpart here.
///
/// The upstream Hermiticity check (`:57-61`) is a `tracing::warn!`; the
/// `verbose >= DEBUG` condition-number warning (`:63-73`) is diagnostics only
/// and is not ported.
///
/// On a near-singular overlap this matters: `examples/pbc/23-smearing.py`'s Al
/// cell has a Γ overlap with λ_min ≈ 3e-9, and the plain-precision sum differs
/// by 2e-9 — enough to move a fractionally occupied band by 0.2 Ha and the
/// σ = 0.1 free energy by 5e-3 Ha (20-18-PRE-SUMMARY, 20-19 item D).
///
/// # Errors
/// As [`crate::pbc_intor::intor_cross`] and [`crate::lattice::get_lattice_ls`].
pub fn get_ovlp_scf(cell: &Cell, kpts: &[[f64; 3]]) -> Result<Vec<CTensor>, PyscfRsError> {
    let precision = cell.precision * SCF_OVLP_PRECISION_FACTOR;
    // `max(cell.rcut, gto.estimate_rcut(cell, precision))` — Python's `max`
    // keeps the FIRST argument on a tie.
    let cell_rcut = cell.try_rcut()?;
    let est = crate::cutoff::estimate_rcut(cell, precision);
    let rcut = if est > cell_rcut { est } else { cell_rcut };
    let ls = crate::lattice::get_lattice_ls(cell, Some(rcut), None, true)?;
    // `Cell.pbc_intor` picks the screened route on `use_loose_rcut`
    // (`cell.py:2033-2038`); there the neighbor list is built under the
    // temporary precision.
    let nl = if cell.use_loose_rcut {
        Some(crate::neighborlist::build_neighbor_list(
            cell,
            None,
            &ls,
            None,
            None,
            0,
            Some(precision),
        )?)
    } else {
        None
    };
    let out = crate::pbc_intor::intor_cross_with_images(
        "int1e_ovlp",
        cell,
        cell,
        kpts,
        PbcIntorOpts {
            comp: None,
            hermi: 0,
            screen: false,
            omega: None,
        },
        &ls,
        nl.as_ref(),
    )?;

    // hf.py:56-61 — `abs(s - s^H).max()`, warned above cell.precision and 1e-12.
    let n = out.ni;
    let mut hermi_error = 0.0_f64;
    for s in &out.kmats {
        for j in 0..n {
            for i in 0..n {
                let p = i + j * n;
                let q = j + i * n;
                let dre = s.re[p] - s.re[q];
                let dim = s.im[p] + s.im[q];
                hermi_error = hermi_error.max((dre * dre + dim * dim).sqrt());
            }
        }
    }
    if hermi_error > cell.precision && hermi_error > 1e-12 {
        tracing::warn!(
            "{hermi_error:.4e} error found in overlap integrals. cell.precision or \
             cell.rcut can be adjusted to improve accuracy."
        );
    }
    Ok(out.kmats)
}

/// `get_t(cell, kpts)` — `scfint.py:57-62`. The kinetic-energy matrix.
///
/// # Errors
/// As [`crate::pbc_intor::intor_cross`].
pub fn get_t(cell: &Cell, kpts: &[[f64; 3]]) -> Result<Vec<CTensor>, PyscfRsError> {
    Ok(pbc_intor(
        cell,
        "int1e_kin",
        kpts,
        PbcIntorOpts {
            comp: None,
            hermi: 1,
            screen: cell.use_loose_rcut,
            omega: None,
        },
    )?
    .kmats)
}

/// Every piece of `hcore` Phase 10 owns, so a caller can assemble the rest.
///
/// See the module docs for why the assembly cannot be completed here.
#[derive(Debug, Clone, PartialEq)]
pub struct HcoreParts {
    /// `T^k` — the kinetic-energy matrix, one per k-point.
    pub kinetic: Vec<CTensor>,
    /// `V_nl^k` — the GTH non-local pseudopotential, one per k-point.
    /// All-zero for an all-electron cell.
    pub vnl: Vec<CTensor>,
    /// `V_loc,2` — the short-range local pseudopotential, real and
    /// k-INDEPENDENT (gamma only). `None` when the cell has no
    /// pseudopotential, or when the k-points requested are not gamma.
    pub vloc_part2: Option<Vec<f64>>,
    /// Whether the missing term is the pseudopotential's `V_loc,1`
    /// (`true`) or the all-electron `get_nuc` (`false`). Both are Phase 11.
    pub pseudo: bool,
}

impl HcoreParts {
    /// `T^k + V_nl^k + V_loc,2` — everything Phase 10 can assemble.
    ///
    /// This is NOT `hcore`: the long-range local term is missing. It is exposed
    /// so Phase 11's FFTDF can add `ifft(vlocG_part1 · SI)` (or `get_nuc`) and
    /// be done, and so tests can check the Hermiticity of what does exist.
    pub fn partial_hcore(&self) -> Vec<CTensor> {
        let mut out = self.kinetic.clone();
        for (k, m) in out.iter_mut().enumerate() {
            for (p, v) in self.vnl[k].re.iter().enumerate() {
                m.re[p] += v;
            }
            for (p, v) in self.vnl[k].im.iter().enumerate() {
                m.im[p] += v;
            }
            if let Some(v2) = self.vloc_part2.as_ref() {
                for (p, v) in v2.iter().enumerate() {
                    m.re[p] += v;
                }
            }
        }
        out
    }
}

/// Assemble everything Phase 10 owns of `hcore` — see [`HcoreParts`].
///
/// `V_loc,2` is only computed when every requested k-point is gamma, because
/// its k-resolved form needs `ft_ao` (Phase 13); otherwise the field is `None`
/// and the caller is told by [`HcoreParts::vloc_part2`] being absent.
///
/// # Errors
/// As [`crate::pbc_intor::intor_cross`] and [`crate::pseudo::get_pp_nl`].
pub fn get_hcore_parts(cell: &Cell, kpts: &[[f64; 3]]) -> Result<HcoreParts, PyscfRsError> {
    let owned_gamma = [[0.0_f64; 3]];
    let kpts: &[[f64; 3]] = if kpts.is_empty() { &owned_gamma } else { kpts };

    let kinetic = get_t(cell, kpts)?;
    let pseudo = cell.pseudo.is_some();
    let vnl = if pseudo {
        crate::pseudo::get_pp_nl(cell, kpts)?
    } else {
        vec![CTensor::zeros(cell.mol.nao_nr * cell.mol.nao_nr); kpts.len()]
    };
    let all_gamma = kpts.iter().all(crate::pbc_intor::is_gamma);
    let vloc_part2 = if pseudo && all_gamma {
        Some(crate::pseudo::get_pp_loc_part2_gamma(cell)?)
    } else {
        None
    };

    Ok(HcoreParts {
        kinetic,
        vnl,
        vloc_part2,
        pseudo,
    })
}

/// `get_hcore(cell, kpts)` — `scfint.py:37-55`.
///
/// # This function is a SIGNPOST, not a stub
///
/// Phase 11 LANDED the missing term, but it cannot land it here: the long-range
/// local pseudopotential is `ifft(vlocG * SI)` and the all-electron nuclear
/// attraction is `get_nuc`, both of which need the FFT box and the uniform-grid
/// AO evaluation — i.e. a density-fitting object. `pyscf-pbc-df` depends on
/// this crate, so the assembled `hcore` lives THERE:
///
/// ```ignore
/// let df = pyscf_pbc_df::Fftdf::new(cell, &kpts)?;
/// let h  = pyscf_pbc_df::get_hcore(&df, &kpts)?;   // T + V_pp (or T + V_ne)
/// ```
///
/// which is what `pyscf_pbc_scf::Krhf` and friends call. [`get_hcore_parts`]
/// remains the way to reach the Phase-10 half (`T + V_nl + V_loc,2`) without a
/// density-fitting object.
///
/// # Errors
/// ALWAYS [`PyscfRsError::NotYetImplemented`], pointing at the function above.
pub fn get_hcore(cell: &Cell, kpts: &[[f64; 3]]) -> Result<Vec<CTensor>, PyscfRsError> {
    let _ = (cell, kpts);
    Err(PyscfRsError::NotYetImplemented {
        phase: 11,
        what: "get_hcore is assembled by the density-fitting object, because its \
               missing term (get_pp_loc_part1 for a pseudopotential cell, get_nuc for \
               an all-electron one) needs the FFT box: call \
               pyscf_pbc_df::get_hcore(&Fftdf::new(cell, kpts)?, kpts). Use \
               pyscf_pbc_gto::hcore::get_hcore_parts for the T + V_nl + V_loc,2 half",
    })
}

impl Cell {
    /// `cell`-side alias for [`get_ovlp`].
    ///
    /// # Errors
    /// As [`get_ovlp`].
    pub fn get_ovlp(&self, kpts: &[[f64; 3]]) -> Result<Vec<CTensor>, PyscfRsError> {
        get_ovlp(self, kpts)
    }
}
