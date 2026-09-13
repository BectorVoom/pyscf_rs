//! k-point ADC AO→MO transform (`pbc/adc/kadc_ao2mo.py`, 294 l).
//!
//! Ports `transform_integrals_incore`'s block structure over the SHIPPED
//! transform: the 4-index MO block for each k-triple comes from
//! [`pyscf_pbc_ao2mo::general`] (via [`ao2mo_block`]) — no new 4-index
//! transform is written here (19-14 Task 1). What IS ADC-specific, and what
//! this module owns, is everything downstream of that call: the six-way
//! occupied/virtual slicing (`oooo`/`oovv`/`ovoo`/`ovov`/`ovvv`/`ovvo`), the
//! `/nkpts` normalization, and the momentum-conservation indexing.
//!
//! All blocks are [`CTensor`] (planar, matching `mo_coeff` dtype — real
//! fixtures carry zero `im`, never a truncation) in row-major `[ki][kj][ka]`
//! order. Accumulation targets are allocated with [`CTensor::zeros`] and
//! explicitly re-zeroed before reuse — the complex arena recycles dirty,
//! over-sized buffers, so nothing trusts a "fresh" buffer (19-14 Task 4).
//!
//! The shape-silent trap: `oovv[:,km,ke]` and `oovv[km,:,ke]` have identical
//! shapes, so every gather goes through the named [`KadcEris`] accessors and
//! the `kadc_base` test pins them element-wise on synthetic amplitudes with
//! distinguishable values per k-index (19-14 Task 3).

use pyscf_algebra::CTensor;
use pyscf_pbc_df::MoCoeff;
use pyscf_pbc_gto::Cell;

use crate::error::PbcAdcError;

/// k-resolved ADC ERI blocks (incore).
///
/// Each block is row-major over `[ki][kj][ka]` followed by its MO axes:
/// `oooo`: `[i,j,k,l]`, `oovv`: `[i,j,a,b]`, `ovoo`: `[i,a,j,k]`,
/// `ovov`: `[i,a,j,b]`, `ovvv`: `[i,a,b,c]`, `ovvo`: `[i,a,b,j]`.
#[derive(Debug, Clone)]
pub struct KadcEris {
    /// Number of k-points.
    pub nkpts: usize,
    /// Occupied count per k-point.
    pub nocc: usize,
    /// Virtual count per k-point.
    pub nvir: usize,
    /// `[ki][kj][ka][i,j,k,l]`.
    pub oooo: CTensor,
    /// `[ki][kj][ka][i,j,a,b]`.
    pub oovv: CTensor,
    /// `[ki][kj][ka][i,a,j,k]`.
    pub ovoo: CTensor,
    /// `[ki][kj][ka][i,a,j,b]`.
    pub ovov: CTensor,
    /// `[ki][kj][ka][i,a,b,c]`.
    pub ovvv: CTensor,
    /// `[ki][kj][ka][i,a,b,j]`.
    pub ovvo: CTensor,
}

impl KadcEris {
    /// Assemble from pre-sliced blocks (outcore/DF routes, fixtures).
    ///
    /// Each block is row-major over `[ki][kj][ka]` + its MO axes (the same
    /// layout [`build_incore`] produces). Lengths are asserted — a truncated
    /// backend must fail here, never feed a short block into a manifold.
    #[allow(clippy::too_many_arguments)]
    pub fn from_blocks(
        nkpts: usize,
        nocc: usize,
        nvir: usize,
        oooo: CTensor,
        oovv: CTensor,
        ovoo: CTensor,
        ovov: CTensor,
        ovvv: CTensor,
        ovvo: CTensor,
    ) -> Result<Self, PbcAdcError> {
        let nk3 = nkpts * nkpts * nkpts;
        let want = [
            nk3 * nocc * nocc * nocc * nocc,
            nk3 * nocc * nocc * nvir * nvir,
            nk3 * nocc * nvir * nocc * nocc,
            nk3 * nocc * nvir * nocc * nvir,
            nk3 * nocc * nvir * nvir * nvir,
            nk3 * nocc * nvir * nvir * nocc,
        ];
        let got = [oooo.re.len(), oovv.re.len(), ovoo.re.len(), ovov.re.len(), ovvv.re.len(), ovvo.re.len()];
        for (w, g) in want.iter().zip(got.iter()) {
            if w != g {
                return Err(PbcAdcError::ShapeMismatch { expected: *w, got: *g });
            }
        }
        Ok(Self { nkpts, nocc, nvir, oooo, oovv, ovoo, ovov, ovvv, ovvo })
    }
    /// Flat offset of triple `(ki, kj, ka)` in a block with trailing size `t`.
    fn triple_offset(&self, ki: usize, kj: usize, ka: usize, t: usize) -> usize {
        ((ki * self.nkpts + kj) * self.nkpts + ka) * t
    }

    /// `ovov[ki,kj,ka]` slice (length `nocc·nvir·nocc·nvir`, `[i,a,j,b]`).
    pub fn ovov_at(&self, ki: usize, kj: usize, ka: usize) -> (&[f64], &[f64]) {
        let t = self.nocc * self.nvir * self.nocc * self.nvir;
        let o = self.triple_offset(ki, kj, ka, t);
        (&self.ovov.re[o..o + t], &self.ovov.im[o..o + t])
    }

    /// `oovv[ki,kj,ka]` slice (length `nocc·nocc·nvir·nvir`, `[i,j,a,b]`).
    pub fn oovv_at(&self, ki: usize, kj: usize, ka: usize) -> (&[f64], &[f64]) {
        let t = self.nocc * self.nocc * self.nvir * self.nvir;
        let o = self.triple_offset(ki, kj, ka, t);
        (&self.oovv.re[o..o + t], &self.oovv.im[o..o + t])
    }
}

/// Fetch one full 4-index MO block `(mo[ki], mo[kj], mo[kk], mo[kl])` as a
/// row-major `[p][q][r][s]` [`CTensor`] of length `nmo⁴`.
///
/// The production implementation is [`ao2mo_block`]; tests substitute a
/// synthetic provider with distinguishable values per index.
pub type MoBlockFetch<'a> =
    dyn Fn(usize, usize, usize, usize) -> Result<CTensor, PbcAdcError> + 'a;

/// Build the six incore blocks (`transform_integrals_incore`).
///
/// For each `(ikp, ikq, ikr)`: `iks = kconserv[ikp,ikq,ikr]`,
/// `blk = fetch(ikp,ikq,ikr,iks)` (full `nmo⁴`, `nmo = nocc + nvir`), sliced
/// into the six blocks with the `/nkpts` normalization. `kconserv` is flat
/// `nkpts³` with `[p][q][r]` at `(p·nk + q)·nk + r`.
pub fn build_incore(
    nkpts: usize,
    nocc: usize,
    nvir: usize,
    kconserv: &[usize],
    fetch: &MoBlockFetch<'_>,
) -> Result<KadcEris, PbcAdcError> {
    if kconserv.len() != nkpts * nkpts * nkpts {
        return Err(PbcAdcError::ShapeMismatch {
            expected: nkpts * nkpts * nkpts,
            got: kconserv.len(),
        });
    }
    let nmo = nocc + nvir;
    let nk3 = nkpts * nkpts * nkpts;
    let inv_nk = 1.0 / nkpts as f64;
    let mut eris = KadcEris {
        nkpts,
        nocc,
        nvir,
        oooo: CTensor::zeros(nk3 * nocc * nocc * nocc * nocc),
        oovv: CTensor::zeros(nk3 * nocc * nocc * nvir * nvir),
        ovoo: CTensor::zeros(nk3 * nocc * nvir * nocc * nocc),
        ovov: CTensor::zeros(nk3 * nocc * nvir * nocc * nvir),
        ovvv: CTensor::zeros(nk3 * nocc * nvir * nvir * nvir),
        ovvo: CTensor::zeros(nk3 * nocc * nvir * nvir * nocc),
    };
    // Explicitly zeroed (arena-recycling discipline — zeros() already zeroes,
    // but the re-zero is the documented habit, not an optimization).
    for b in [
        &mut eris.oooo,
        &mut eris.oovv,
        &mut eris.ovoo,
        &mut eris.ovov,
        &mut eris.ovvv,
        &mut eris.ovvo,
    ] {
        b.re.fill(0.0);
        b.im.fill(0.0);
    }
    for ikp in 0..nkpts {
        for ikq in 0..nkpts {
            for ikr in 0..nkpts {
                let iks = kconserv[(ikp * nkpts + ikq) * nkpts + ikr];
                if iks >= nkpts {
                    return Err(PbcAdcError::ShapeMismatch { expected: nkpts, got: iks });
                }
                let blk = fetch(ikp, ikq, ikr, iks)?;
                if blk.re.len() != nmo * nmo * nmo * nmo || blk.im.len() != nmo * nmo * nmo * nmo {
                    return Err(PbcAdcError::ShapeMismatch {
                        expected: nmo * nmo * nmo * nmo,
                        got: blk.re.len().min(blk.im.len()),
                    });
                }
                let at = |p: usize, q: usize, r: usize, s: usize| -> (f64, f64) {
                    let o = ((p * nmo + q) * nmo + r) * nmo + s;
                    (blk.re[o], blk.im[o])
                };
                let base = ((ikp * nkpts + ikq) * nkpts + ikr) as usize;
                for i in 0..nocc {
                    for j in 0..nocc {
                        for k in 0..nocc {
                            for l in 0..nocc {
                                let o = (base * nocc * nocc * nocc + i * nocc * nocc + j * nocc + k) * nocc + l;
                                let (re, im) = at(i, j, k, l);
                                eris.oooo.re[o] = re * inv_nk;
                                eris.oooo.im[o] = im * inv_nk;
                            }
                        }
                        for a in 0..nvir {
                            for b in 0..nvir {
                                let o = (base * nocc * nocc * nvir + i * nocc * nvir + j * nvir + a) * nvir + b;
                                let (re, im) = at(i, j, nocc + a, nocc + b);
                                eris.oovv.re[o] = re * inv_nk;
                                eris.oovv.im[o] = im * inv_nk;
                            }
                            for jj in 0..nocc {
                                let o = (base * nocc * nvir * nocc + i * nvir * nocc + a * nocc + jj) * nocc;
                                for kk in 0..nocc {
                                    let (re, im) = at(i, nocc + a, jj, kk);
                                    eris.ovoo.re[o + kk] = re * inv_nk;
                                    eris.ovoo.im[o + kk] = im * inv_nk;
                                }
                            }
                            for jj in 0..nocc {
                                for b in 0..nvir {
                                    let o = (base * nocc * nvir * nocc + i * nvir * nocc + a * nocc + jj) * nvir + b;
                                    let (re, im) = at(i, nocc + a, jj, nocc + b);
                                    eris.ovov.re[o] = re * inv_nk;
                                    eris.ovov.im[o] = im * inv_nk;
                                }
                            }
                            for b in 0..nvir {
                                for c in 0..nvir {
                                    let o = (base * nocc * nvir * nvir + i * nvir * nvir + a * nvir + b) * nvir + c;
                                    let (re, im) = at(i, nocc + a, nocc + b, nocc + c);
                                    eris.ovvv.re[o] = re * inv_nk;
                                    eris.ovvv.im[o] = im * inv_nk;
                                }
                                for jj in 0..nocc {
                                    let o = (base * nocc * nvir * nvir + i * nvir * nvir + a * nvir + b) * nocc + jj;
                                    let (re, im) = at(i, nocc + a, nocc + b, jj);
                                    eris.ovvo.re[o] = re * inv_nk;
                                    eris.ovvo.im[o] = im * inv_nk;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    Ok(eris)
}

/// Production 4-index MO block over the shipped transform.
///
/// Calls [`pyscf_pbc_ao2mo::general`] with the four per-k MO sets at the four
/// k-vectors and returns the full `nmo⁴` `[p][q][r][s]` block (`Eri.data` is
/// row-major `[(p,q),(r,s)]`, which is the same flat order). Asserts the
/// `nmo⁴` length — a packed or truncated backend must fail here, never feed a
/// short block into the slicer.
pub fn ao2mo_block(
    cell: &Cell,
    mos: [&MoCoeff; 4],
    kpts: [[f64; 3]; 4],
) -> Result<CTensor, PbcAdcError> {
    let nmo = mos[0].nmo;
    let eri = pyscf_pbc_ao2mo::general(cell, mos, Some(kpts), false).map_err(|e| {
        PbcAdcError::Core(pyscf_core::PyscfRsError::Core(
            pyscf_core::CoreError::InvalidMolecule(format!("adc ao2mo_block: {e}")),
        ))
    })?;
    if eri.data.re.len() != nmo * nmo * nmo * nmo {
        return Err(PbcAdcError::ShapeMismatch {
            expected: nmo * nmo * nmo * nmo,
            got: eri.data.re.len(),
        });
    }
    Ok(eri.data)
}
