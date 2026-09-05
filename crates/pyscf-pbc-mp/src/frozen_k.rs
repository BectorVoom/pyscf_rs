//! K-resolved frozen-orbital bookkeeping (`pyscf/pbc/mp/kmp2.py:376-552`).

use std::collections::HashSet;

use pyscf_mp2::Frozen;

use crate::PbcMpError;

#[derive(Debug, Clone, PartialEq)]
pub enum FrozenK {
    Uniform(Frozen),
    PerKpt(Vec<Vec<usize>>),
}

impl Default for FrozenK {
    fn default() -> Self {
        Self::Uniform(Frozen::None)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum KCount {
    PerKpoint(Vec<usize>),
    Dense(usize),
}

impl KCount {
    pub fn per_kpoint(self) -> Vec<usize> {
        match self {
            Self::PerKpoint(v) => v,
            Self::Dense(v) => vec![v],
        }
    }
}

fn list_for(
    frozen: &FrozenK,
    k: usize,
    nk: usize,
) -> Result<Option<&[usize]>, PbcMpError> {
    match frozen {
        FrozenK::Uniform(Frozen::None) => Ok(None),
        FrozenK::Uniform(Frozen::List(v)) => Ok(Some(v)),
        FrozenK::PerKpt(v) => {
            if v.len() != nk {
                return Err(PbcMpError::FrozenKpointCount {
                    expected: nk,
                    got: v.len(),
                });
            }
            Ok(Some(&v[k]))
        }
        FrozenK::Uniform(Frozen::Count(_)) => Ok(None),
        FrozenK::Uniform(Frozen::Auto | Frozen::Window { .. }) => {
            Err(PbcMpError::UnsupportedFrozen)
        }
    }
}

/// `_frozen_sanity_check` (`kmp2.py:376-399`).
pub fn frozen_sanity_check(frozen: &[usize], mo_occ: &[f64], kpt: usize) -> Result<(), PbcMpError> {
    if !mo_occ.iter().any(|&x| x > 0.0) {
        return Err(PbcMpError::InvalidFrozen {
            kpt,
            reason: "No occupied orbitals?".into(),
        });
    }
    let unique: HashSet<_> = frozen.iter().copied().collect();
    if unique.len() != frozen.len() {
        return Err(PbcMpError::InvalidFrozen {
            kpt,
            reason: "Frozen orbital list contains duplicates!".into(),
        });
    }
    if frozen.iter().any(|&i| i >= mo_occ.len()) {
        return Err(PbcMpError::InvalidFrozen {
            kpt,
            reason: "Freezing orbital not in MO list!".into(),
        });
    }
    Ok(())
}

/// `get_nocc` (`kmp2.py:401-458`). Fractional occupations are refused before
/// any frozen-orbital handling, matching upstream.
pub fn get_nocc(
    mo_occ: &[Vec<f64>],
    frozen: &FrozenK,
    per_kpoint: bool,
) -> Result<KCount, PbcMpError> {
    for (k, occ) in mo_occ.iter().enumerate() {
        if occ.iter().any(|x| x.fract() != 0.0) {
            return Err(PbcMpError::FractionalOccupation { kpt: k });
        }
    }
    let nk = mo_occ.len();
    let mut out = Vec::with_capacity(nk);
    for (k, occ) in mo_occ.iter().enumerate() {
        let occupied = occ.iter().filter(|&&x| x > 0.0).count();
        let frozen_occ = match frozen {
            FrozenK::Uniform(Frozen::Count(n)) => *n,
            _ => match list_for(frozen, k, nk)? {
                None => 0,
                Some(list) => {
                    frozen_sanity_check(list, occ, k)?;
                    let max_occ = occ.iter().rposition(|&x| x > 0.0).ok_or_else(|| {
                        PbcMpError::InvalidFrozen {
                            kpt: k,
                            reason: "No occupied orbitals?".into(),
                        }
                    })?;
                    list.iter().filter(|&&i| i <= max_occ).count()
                }
            },
        };
        out.push(
            occupied
                .checked_sub(frozen_occ)
                .ok_or_else(|| PbcMpError::InvalidFrozen {
                    kpt: k,
                    reason: "Must have occupied orbitals!".into(),
                })?,
        );
    }
    if !out.iter().any(|&n| n > 0) {
        return Err(PbcMpError::InvalidFrozen {
            kpt: 0,
            reason: "Must have occupied orbitals!".into(),
        });
    }
    Ok(if per_kpoint {
        KCount::PerKpoint(out)
    } else {
        KCount::Dense(*out.iter().max().unwrap_or(&0))
    })
}

/// `get_nmo` (`kmp2.py:461-514`). The dense dimension is
/// `max(nocc) + max(nmo - nocc)`, not `max(nmo)`.
pub fn get_nmo(
    mo_occ: &[Vec<f64>],
    frozen: &FrozenK,
    per_kpoint: bool,
) -> Result<KCount, PbcMpError> {
    let nk = mo_occ.len();
    let mut out = Vec::with_capacity(nk);
    for (k, occ) in mo_occ.iter().enumerate() {
        let removed = match frozen {
            FrozenK::Uniform(Frozen::None) => 0,
            FrozenK::Uniform(Frozen::Count(n)) => *n,
            FrozenK::Uniform(Frozen::List(v)) => {
                frozen_sanity_check(v, occ, k)?;
                v.len()
            }
            FrozenK::PerKpt(_) => {
                let list = list_for(frozen, k, nk)?.unwrap_or_default();
                frozen_sanity_check(list, occ, k)?;
                list.len()
            }
            FrozenK::Uniform(Frozen::Auto | Frozen::Window { .. }) => {
                return Err(PbcMpError::UnsupportedFrozen);
            }
        };
        let n = occ
            .len()
            .checked_sub(removed)
            .ok_or_else(|| PbcMpError::InvalidFrozen {
                kpt: k,
                reason: "Must have a positive number of orbitals!".into(),
            })?;
        if n == 0 {
            return Err(PbcMpError::InvalidFrozen {
                kpt: k,
                reason: "Must have a positive number of orbitals!".into(),
            });
        }
        out.push(n);
    }
    if per_kpoint {
        return Ok(KCount::PerKpoint(out));
    }
    let nocc = match get_nocc(mo_occ, frozen, true)? {
        KCount::PerKpoint(v) => v,
        KCount::Dense(_) => unreachable!(),
    };
    let dense = nocc.iter().copied().max().unwrap_or(0)
        + out.iter().zip(&nocc).map(|(m, o)| m - o).max().unwrap_or(0);
    Ok(KCount::Dense(dense))
}

/// `get_frozen_mask` (`kmp2.py:517-552`); `true` means active.
pub fn get_frozen_mask(
    mo_occ: &[Vec<f64>],
    frozen: &FrozenK,
) -> Result<Vec<Vec<bool>>, PbcMpError> {
    let nk = mo_occ.len();
    let mut masks: Vec<Vec<bool>> = mo_occ.iter().map(|o| vec![true; o.len()]).collect();
    for k in 0..nk {
        match frozen {
            FrozenK::Uniform(Frozen::None) => {}
            FrozenK::Uniform(Frozen::Count(n)) => {
                if *n > masks[k].len() {
                    return Err(PbcMpError::InvalidFrozen {
                        kpt: k,
                        reason: "Freezing orbital not in MO list!".into(),
                    });
                }
                masks[k][..*n].fill(false);
            }
            FrozenK::Uniform(Frozen::List(v)) => {
                frozen_sanity_check(v, &mo_occ[k], k)?;
                for &i in v {
                    masks[k][i] = false;
                }
            }
            FrozenK::PerKpt(_) => {
                let list = list_for(frozen, k, nk)?.unwrap_or_default();
                frozen_sanity_check(list, &mo_occ[k], k)?;
                for &i in list {
                    masks[k][i] = false;
                }
            }
            FrozenK::Uniform(Frozen::Auto | Frozen::Window { .. }) => {
                return Err(PbcMpError::UnsupportedFrozen);
            }
        }
    }
    Ok(masks)
}
