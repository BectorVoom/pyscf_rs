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
        },
    )?
    .kmats)
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
/// # Errors
/// ALWAYS [`PyscfRsError::NotYetImplemented`] `{ phase: 11 }` today, for the
/// reason spelled out in the module docs: the long-range half of the local
/// pseudopotential (and, for an all-electron cell, `get_nuc`) is an FFT/AFT
/// quantity that Phase 11 owns. Use [`get_hcore_parts`] for the Phase-10 half.
///
/// The signature is final so plan 11-xx only has to fill the body.
pub fn get_hcore(cell: &Cell, kpts: &[[f64; 3]]) -> Result<Vec<CTensor>, PyscfRsError> {
    let _ = (cell, kpts);
    Err(PyscfRsError::NotYetImplemented {
        phase: 11,
        what: "get_hcore needs the long-range term — get_pp_loc_part1 for a \
               pseudopotential cell (ifft(vlocG_part1 * SI), FFTDF) or get_nuc for an \
               all-electron one. Everything else is ready: use \
               pyscf_pbc_gto::hcore::get_hcore_parts",
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
