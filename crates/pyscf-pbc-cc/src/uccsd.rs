//! `uccsd` — the MOLECULAR, complex-capable unrestricted CCSD
//! (`pyscf/cc/uccsd.py`).
//!
//! # Why this lives here
//!
//! Same reason as [`crate::rccsd`] and [`crate::gccsd`]: `crates/pyscf-ccsd`'s
//! `uccsd` is `f64` by construction, `pbc/cc/ccsd.py:61`'s `UCCSD` needs a
//! complex one, and the planar-complex substrate is in this crate.
//!
//! # This is not [`crate::kccsd_uhf`] at `nkpts = 1`
//!
//! It would give the same numbers — `oracle_gamma_ug.rs` measures that — but
//! `pbc/cc/kccsd_uhf.py` replaces `update_amps` wholesale, and the two are not
//! even the same shape of expression: the k-point form threads `kconserv`
//! through every contraction, and this one is written around the six `w*`
//! ring intermediates and the `P(ij)P(ab)` antisymmetrisation at the end.
//!
//! # The blocking is collapsed to ONE block, and that is not an approximation
//!
//! Upstream loops `for p0, p1 in lib.prange(0, nocc, blksize)` over four of
//! the `ovvv`-like blocks (`:85-135`), with `blksize` chosen from
//! `cc.max_memory`. When memory allows, `prange` yields the single range
//! `(0, nocc)` and the loop body runs once — which is what this port does.
//! The expressions are upstream's, unchanged; what is dropped is the memory
//! schedule, not a term. `eris.get_ovvv(slice(p0,p1))` is then just
//! `eris.ovvv`, because `_make_eris_incore` skips its `pack_tril` packing when
//! `ao2mofn` is callable (`:936`), and for the Γ shim it always is.
//!
//! # 25 blocks, and lowercase/uppercase is the spin
//!
//! `_ChemistsERIs` (`:773-794`) carries seven alpha-alpha blocks, seven
//! beta-beta, seven alpha-beta and four beta-alpha, all in CHEMISTS' notation
//! `(pq|rs)`. Lowercase is alpha and uppercase is beta throughout, upstream's
//! own convention, kept here so a reader can match a field to its line.

use crate::error::PbcCcError;
use crate::kintermediates_uhf::{UT1, UT2};
use crate::zarr::{ZArr, einsum, einsum_scaled};

fn shape(m: impl Into<String>) -> PbcCcError {
    PbcCcError::Shape(m.into())
}

/// `_ChemistsERIs` (`uccsd.py:773-794`) — the 25 chemists' blocks, complex.
#[derive(Debug, Clone)]
pub struct ChemistsErisU {
    pub nocca: usize,
    pub noccb: usize,
    pub nvira: usize,
    pub nvirb: usize,
    /// `[nmoa, nmoa]`.
    pub focka: ZArr,
    /// `[nmob, nmob]`.
    pub fockb: ZArr,
    /// `(mo_energy_a, mo_energy_b)`, with the Γ shim's Madelung re-add applied
    /// to each occupied block (`pbc/cc/ccsd.py:89-91`).
    pub mo_energy: (Vec<f64>, Vec<f64>),

    // alpha-alpha
    pub oooo: ZArr,
    pub ovoo: ZArr,
    pub ovov: ZArr,
    pub oovv: ZArr,
    pub ovvo: ZArr,
    pub ovvv: ZArr,
    pub vvvv: ZArr,
    // beta-beta
    pub oooo_bb: ZArr,
    pub ovoo_bb: ZArr,
    pub ovov_bb: ZArr,
    pub oovv_bb: ZArr,
    pub ovvo_bb: ZArr,
    pub ovvv_bb: ZArr,
    pub vvvv_bb: ZArr,
    // alpha-beta
    pub oo_oo: ZArr,
    pub ov_oo: ZArr,
    pub ov_ov: ZArr,
    pub oo_vv: ZArr,
    pub ov_vo: ZArr,
    pub ov_vv: ZArr,
    pub vv_vv: ZArr,
    // beta-alpha
    pub ovoo_ba: ZArr,
    pub oovv_ba: ZArr,
    pub ovvo_ba: ZArr,
    pub ovvv_ba: ZArr,
}

impl ChemistsErisU {
    /// `_make_eris_incore(mycc, mo_coeff, ao2mofn)` (`:873-951`) from the
    /// three full chemists' tensors it builds.
    ///
    /// `eri_aa` is `[nmoa]⁴`, `eri_bb` is `[nmob]⁴`, `eri_ab` is
    /// `[nmoa, nmoa, nmob, nmob]`, and `eri_ba = eri_ab.transpose(2,3,0,1)`
    /// (`:895`).
    ///
    /// **The `pack_tril` compaction at `:936-951` is skipped**, exactly as
    /// upstream skips it: it is inside `if not callable(ao2mofn)`, and the
    /// periodic transform is always a callable.
    ///
    /// # Errors
    /// [`PbcCcError::Shape`] on any tensor whose length is not the product of
    /// its declared dimensions.
    #[allow(clippy::too_many_arguments)]
    pub fn from_full_chemists(
        eri_aa: &ZArr,
        eri_bb: &ZArr,
        eri_ab: &ZArr,
        focka: ZArr,
        fockb: ZArr,
        mo_energy: (Vec<f64>, Vec<f64>),
        nocc: (usize, usize),
    ) -> Result<Self, PbcCcError> {
        let (nocca, noccb) = nocc;
        let nmoa = mo_energy.0.len();
        let nmob = mo_energy.1.len();
        let (nvira, nvirb) = (nmoa - nocca, nmob - noccb);
        let want = |n: usize, e: &ZArr, tag: &str| -> Result<(), PbcCcError> {
            if e.len() == n {
                Ok(())
            } else {
                Err(shape(format!(
                    "uccsd::_make_eris_incore: {tag} has {} elements, expected {n}",
                    e.len()
                )))
            }
        };
        want(nmoa.pow(4), eri_aa, "eri_aa")?;
        want(nmob.pow(4), eri_bb, "eri_bb")?;
        want(nmoa * nmoa * nmob * nmob, eri_ab, "eri_ab")?;

        let aa = eri_aa.reshape(&[nmoa, nmoa, nmoa, nmoa])?;
        let bb = eri_bb.reshape(&[nmob, nmob, nmob, nmob])?;
        let ab = eri_ab.reshape(&[nmoa, nmoa, nmob, nmob])?;
        // `:895` — `eri_ba = eri_ab.reshape(...).transpose(2,3,0,1)`.
        let ba = ab.transpose(&[2, 3, 0, 1])?;

        let (oa, va) = ((0, nocca), (nocca, nmoa));
        let (ob, vb) = ((0, noccb), (noccb, nmob));
        Ok(Self {
            nocca,
            noccb,
            nvira,
            nvirb,
            focka,
            fockb,
            mo_energy,
            oooo: aa.slice_axes(&[oa, oa, oa, oa])?,
            ovoo: aa.slice_axes(&[oa, va, oa, oa])?,
            ovov: aa.slice_axes(&[oa, va, oa, va])?,
            oovv: aa.slice_axes(&[oa, oa, va, va])?,
            ovvo: aa.slice_axes(&[oa, va, va, oa])?,
            ovvv: aa.slice_axes(&[oa, va, va, va])?,
            vvvv: aa.slice_axes(&[va, va, va, va])?,
            oooo_bb: bb.slice_axes(&[ob, ob, ob, ob])?,
            ovoo_bb: bb.slice_axes(&[ob, vb, ob, ob])?,
            ovov_bb: bb.slice_axes(&[ob, vb, ob, vb])?,
            oovv_bb: bb.slice_axes(&[ob, ob, vb, vb])?,
            ovvo_bb: bb.slice_axes(&[ob, vb, vb, ob])?,
            ovvv_bb: bb.slice_axes(&[ob, vb, vb, vb])?,
            vvvv_bb: bb.slice_axes(&[vb, vb, vb, vb])?,
            oo_oo: ab.slice_axes(&[oa, oa, ob, ob])?,
            ov_oo: ab.slice_axes(&[oa, va, ob, ob])?,
            ov_ov: ab.slice_axes(&[oa, va, ob, vb])?,
            oo_vv: ab.slice_axes(&[oa, oa, vb, vb])?,
            ov_vo: ab.slice_axes(&[oa, va, vb, ob])?,
            ov_vv: ab.slice_axes(&[oa, va, vb, vb])?,
            vv_vv: ab.slice_axes(&[va, va, vb, vb])?,
            ovoo_ba: ba.slice_axes(&[ob, vb, oa, oa])?,
            oovv_ba: ba.slice_axes(&[ob, ob, va, va])?,
            ovvo_ba: ba.slice_axes(&[ob, vb, va, oa])?,
            ovvv_ba: ba.slice_axes(&[ob, vb, va, va])?,
        })
    }

    fn blk(&self, f: &ZArr, r: [(usize, usize); 2]) -> Result<ZArr, PbcCcError> {
        f.slice_axes(&r)
    }

    /// `eris.focka[:nocca,nocca:]`.
    ///
    /// # Errors
    /// Propagates the slice.
    pub fn fova(&self) -> Result<ZArr, PbcCcError> {
        let n = self.nocca + self.nvira;
        self.blk(&self.focka, [(0, self.nocca), (self.nocca, n)])
    }
    /// `eris.fockb[:noccb,noccb:]`.
    ///
    /// # Errors
    /// Propagates the slice.
    pub fn fovb(&self) -> Result<ZArr, PbcCcError> {
        let n = self.noccb + self.nvirb;
        self.blk(&self.fockb, [(0, self.noccb), (self.noccb, n)])
    }
    /// `eris.focka[:nocca,:nocca]`.
    ///
    /// # Errors
    /// Propagates the slice.
    pub fn fooa(&self) -> Result<ZArr, PbcCcError> {
        self.blk(&self.focka, [(0, self.nocca), (0, self.nocca)])
    }
    /// `eris.fockb[:noccb,:noccb]`.
    ///
    /// # Errors
    /// Propagates the slice.
    pub fn foob(&self) -> Result<ZArr, PbcCcError> {
        self.blk(&self.fockb, [(0, self.noccb), (0, self.noccb)])
    }
    /// `eris.focka[nocca:,nocca:]`.
    ///
    /// # Errors
    /// Propagates the slice.
    pub fn fvva(&self) -> Result<ZArr, PbcCcError> {
        let n = self.nocca + self.nvira;
        self.blk(&self.focka, [(self.nocca, n), (self.nocca, n)])
    }
    /// `eris.fockb[noccb:,noccb:]`.
    ///
    /// # Errors
    /// Propagates the slice.
    pub fn fvvb(&self) -> Result<ZArr, PbcCcError> {
        let n = self.noccb + self.nvirb;
        self.blk(&self.fockb, [(self.noccb, n), (self.noccb, n)])
    }
}

/// `make_tau_aa(t2aa, t1a, r1a, fac)` — `uccsd.py:1192-1198`.
///
/// # Errors
/// Propagates every contraction.
pub fn make_tau_aa(t2aa: &ZArr, t1a: &ZArr, r1a: &ZArr, fac: f64) -> Result<ZArr, PbcCcError> {
    let p = einsum("ia,jb->ijab", &[t1a, r1a])?;
    let mut tau = p.clone();
    // `:1194` — `-= einsum('ia,jb->jiab', …)`, i.e. minus the `ij` transpose.
    tau.sub_assign(&p.transpose(&[1, 0, 2, 3])?)?;
    let sw = tau.transpose(&[0, 1, 3, 2])?;
    tau.sub_assign(&sw)?;
    tau.scale(fac * 0.5);
    tau.add_assign(t2aa)?;
    Ok(tau)
}

/// `make_tau_ab(t2ab, t1, r1, fac)` — `:1200-1207`.
///
/// # Errors
/// Propagates every contraction.
pub fn make_tau_ab(t2ab: &ZArr, t1: &UT1, r1: &UT1, fac: f64) -> Result<ZArr, PbcCcError> {
    let mut tau = einsum("ia,jb->ijab", &[&t1.0, &r1.1])?;
    tau.add_assign(&einsum("ia,jb->ijab", &[&r1.0, &t1.1])?)?;
    tau.scale(fac * 0.5);
    tau.add_assign(t2ab)?;
    Ok(tau)
}

/// `make_tau(t2, t1, r1, fac)` — `:1184-1190`.
///
/// # Errors
/// Propagates every contraction.
pub fn make_tau(t2: &UT2, t1: &UT1, r1: &UT1, fac: f64) -> Result<UT2, PbcCcError> {
    Ok((
        make_tau_aa(&t2.0, &t1.0, &r1.0, fac)?,
        make_tau_ab(&t2.1, t1, r1, fac)?,
        make_tau_aa(&t2.2, &t1.1, &r1.1, fac)?,
    ))
}

/// `_add_vvvv(mycc, None, tau, eris)` — the INCORE, non-`direct`,
/// `t2sym = None` path (`:531-534`).
///
/// # The same-spin blocks are NOT a plain `einsum`, and that is measurable
///
/// `uccsd.py:57-59` writes the contraction out as a comment beside the call
/// it replaced —
///
/// ```text
/// u2aa += lib.einsum('ijef,aebf->ijab', tauaa, eris_vvvv) * .5
/// ```
///
/// — but the code does something else for `aa` and `bb`: it contracts only the
/// LOWER TRIANGLE `tauaa[tril_indices(nocca)]` and then unpacks with
/// `_unpack_t2_tril(..., 'jiba')` (`ccsd.py:641-654`), which is
///
/// ```text
/// t2[idy,idx] = t2tril.transpose(0,2,1)   # written FIRST
/// t2[idx,idy] = t2tril                    # written SECOND
/// ```
///
/// so the result is forced to satisfy `u2[j,i,b,a] = u2[i,j,a,b]`, and on the
/// DIAGONAL `i == j` the second write wins. For a physical `tau` — which is
/// antisymmetric in `(ij)` and in `(ab)`, hence already symmetric under the
/// simultaneous swap — the two agree exactly, and the comment is true. For a
/// synthetic `tau` they do not: measured, `1.3e-2` apart on the gate's own
/// random amplitudes, against `0` for the tril form. This ports the CODE.
///
/// `ab` has no such structure and IS the plain contraction (`:533`) — measured
/// at `0.0`.
///
/// The `* .5` on the same-spin blocks is applied by the CALLER (`:63-64`), not
/// here — this returns what `_add_vvvv` returns.
///
/// # Errors
/// Propagates every contraction.
pub fn add_vvvv(tau: &UT2, eris: &ChemistsErisU) -> Result<UT2, PbcCcError> {
    let same = |t: &ZArr, vvvv: &ZArr, nocc: usize, nvir: usize| -> Result<ZArr, PbcCcError> {
        let x = einsum("ijef,aebf->ijab", &[t, vvvv])?;
        let mut out = ZArr::zeros(&[nocc, nocc, nvir, nvir]);
        let at = |i: usize, j: usize, a: usize, b: usize| ((i * nocc + j) * nvir + a) * nvir + b;
        for i in 0..nocc {
            for j in 0..=i {
                for a in 0..nvir {
                    for b in 0..nvir {
                        // `:645` — the transposed write, into `(j, i, b, a)`.
                        let src = at(i, j, a, b);
                        let dst = at(j, i, b, a);
                        out.data_mut().re[dst] = x.data().re[src];
                        out.data_mut().im[dst] = x.data().im[src];
                    }
                }
            }
        }
        for i in 0..nocc {
            for j in 0..=i {
                for a in 0..nvir {
                    for b in 0..nvir {
                        // `:646` — the straight write, SECOND, so it wins on
                        // the `i == j` diagonal.
                        let f = at(i, j, a, b);
                        out.data_mut().re[f] = x.data().re[f];
                        out.data_mut().im[f] = x.data().im[f];
                    }
                }
            }
        }
        Ok(out)
    };
    Ok((
        same(&tau.0, &eris.vvvv, eris.nocca, eris.nvira)?,
        einsum("ijef,aebf->ijab", &[&tau.1, &eris.vv_vv])?,
        same(&tau.2, &eris.vvvv_bb, eris.noccb, eris.nvirb)?,
    ))
}

/// `energy(cc, t1, t2, eris)` — `uccsd.py:773-802`.
///
/// Returns `(re, im)`; `:800` warns above `|Im| > 1e-4` and returns `e.real`.
///
/// # Errors
/// Propagates every contraction.
pub fn energy(t1: &UT1, t2: &UT2, eris: &ChemistsErisU) -> Result<(f64, f64), PbcCcError> {
    let (ovov, ovov_bb, ov_ov) = (&eris.ovov, &eris.ovov_bb, &eris.ov_ov);
    let mut e = einsum("ia,ia->", &[&eris.fova()?, &t1.0])?;
    e.add_assign(&einsum("ia,ia->", &[&eris.fovb()?, &t1.1])?)?;
    e.add_assign(&einsum_scaled("ijab,iajb->", &[&t2.0, ovov], 0.25)?)?;
    e.sub_assign(&einsum_scaled("ijab,ibja->", &[&t2.0, ovov], 0.25)?)?;
    e.add_assign(&einsum_scaled("ijab,iajb->", &[&t2.2, ovov_bb], 0.25)?)?;
    e.sub_assign(&einsum_scaled("ijab,ibja->", &[&t2.2, ovov_bb], 0.25)?)?;
    e.add_assign(&einsum("ijab,iajb->", &[&t2.1, ov_ov])?)?;
    e.add_assign(&einsum_scaled("ia,jb,iajb->", &[&t1.0, &t1.0, ovov], 0.5)?)?;
    e.sub_assign(&einsum_scaled("ia,jb,ibja->", &[&t1.0, &t1.0, ovov], 0.5)?)?;
    e.add_assign(&einsum_scaled(
        "ia,jb,iajb->",
        &[&t1.1, &t1.1, ovov_bb],
        0.5,
    )?)?;
    e.sub_assign(&einsum_scaled(
        "ia,jb,ibja->",
        &[&t1.1, &t1.1, ovov_bb],
        0.5,
    )?)?;
    e.add_assign(&einsum("ia,jb,iajb->", &[&t1.0, &t1.1, ov_ov])?)?;
    e.at(&[])
}

/// The `e_i − e_a` table for one spin, row-major `[nocc, nvir]`.
fn eia(mo_e: &[f64], nocc: usize, nvir: usize, level_shift: f64) -> Vec<f64> {
    (0..nocc)
        .flat_map(|i| (0..nvir).map(move |a| (i, a)))
        .map(|(i, a)| mo_e[i] - (mo_e[nocc + a] + level_shift))
        .collect()
}

/// Divide `x[i,j,a,b]` by `eia_1[i,a] + eia_2[j,b]` in place.
fn divide_ijab(x: &mut ZArr, e1: &[f64], e2: &[f64], dims: [usize; 4]) {
    let [ni, nj, na, nb] = dims;
    let mut f = 0;
    for i in 0..ni {
        for j in 0..nj {
            for a in 0..na {
                for b in 0..nb {
                    let d = e1[i * na + a] + e2[j * nb + b];
                    x.data_mut().re[f] /= d;
                    x.data_mut().im[f] /= d;
                    f += 1;
                }
            }
        }
    }
}

/// `UCCSD.init_amps(eris)` — `uccsd.py:556-592`.
///
/// Returns `(emp2, (t1a,t1b), (t2aa,t2ab,t2bb))`.
///
/// # Errors
/// Propagates every contraction.
pub fn init_amps(eris: &ChemistsErisU) -> Result<(f64, UT1, UT2), PbcCcError> {
    let (na, nb) = (eris.nocca, eris.noccb);
    let (va, vb) = (eris.nvira, eris.nvirb);
    let ea = eia(&eris.mo_energy.0, na, va, 0.0);
    let eb = eia(&eris.mo_energy.1, nb, vb, 0.0);

    // `:574-575` — the CONJUGATE of `fov`, unlike the restricted `init_amps`.
    let mut t1a = eris.fova()?.conj();
    for (f, d) in ea.iter().enumerate() {
        t1a.data_mut().re[f] /= d;
        t1a.data_mut().im[f] /= d;
    }
    let mut t1b = eris.fovb()?.conj();
    for (f, d) in eb.iter().enumerate() {
        t1b.data_mut().re[f] /= d;
        t1b.data_mut().im[f] /= d;
    }

    let mut t2aa = eris.ovov.transpose(&[0, 2, 1, 3])?.conj();
    divide_ijab(&mut t2aa, &ea, &ea, [na, na, va, va]);
    let mut t2ab = eris.ov_ov.transpose(&[0, 2, 1, 3])?.conj();
    divide_ijab(&mut t2ab, &ea, &eb, [na, nb, va, vb]);
    let mut t2bb = eris.ovov_bb.transpose(&[0, 2, 1, 3])?.conj();
    divide_ijab(&mut t2bb, &eb, &eb, [nb, nb, vb, vb]);
    // `:584-585` — antisymmetrise the same-spin blocks.
    let sw = t2aa.transpose(&[0, 1, 3, 2])?;
    t2aa.sub_assign(&sw)?;
    let sw = t2bb.transpose(&[0, 1, 3, 2])?;
    t2bb.sub_assign(&sw)?;

    let mut e = einsum("ijab,iajb->", &[&t2ab, &eris.ov_ov])?;
    e.add_assign(&einsum_scaled("ijab,iajb->", &[&t2aa, &eris.ovov], 0.25)?)?;
    e.sub_assign(&einsum_scaled("ijab,ibja->", &[&t2aa, &eris.ovov], 0.25)?)?;
    e.add_assign(&einsum_scaled(
        "ijab,iajb->",
        &[&t2bb, &eris.ovov_bb],
        0.25,
    )?)?;
    e.sub_assign(&einsum_scaled(
        "ijab,ibja->",
        &[&t2bb, &eris.ovov_bb],
        0.25,
    )?)?;
    Ok((e.at(&[])?.0, (t1a, t1b), (t2aa, t2ab, t2bb)))
}

/// `update_amps(cc, t1, t2, eris)` — `uccsd.py:41-341`, with the four
/// `lib.prange` loops collapsed to one block. See the module doc.
///
/// # Errors
/// Propagates every contraction.
#[allow(clippy::too_many_lines)]
pub fn update_amps(
    t1: &UT1,
    t2: &UT2,
    eris: &ChemistsErisU,
    level_shift: f64,
) -> Result<(UT1, UT2), PbcCcError> {
    let (t1a, t1b) = (&t1.0, &t1.1);
    let (t2aa, t2ab, t2bb) = (&t2.0, &t2.1, &t2.2);
    let (nocca, noccb) = (eris.nocca, eris.noccb);
    let (nvira, nvirb) = (eris.nvira, eris.nvirb);
    let mo_ea_o = &eris.mo_energy.0[..nocca];
    let mo_eb_o = &eris.mo_energy.1[..noccb];
    let fova = eris.fova()?;
    let fovb = eris.fovb()?;

    let mut u1a = ZArr::zeros(&[nocca, nvira]);
    let mut u1b = ZArr::zeros(&[noccb, nvirb]);

    // `:62-66`
    let tau = make_tau(t2, t1, t1, 1.0)?;
    let (mut u2aa, mut u2ab, mut u2bb) = add_vvvv(&tau, eris)?;
    u2aa.scale(0.5);
    u2bb.scale(0.5);
    let (tauaa, tauab, taubb) = (&tau.0, &tau.1, &tau.2);

    // `:68-75`
    let mut fooa = einsum_scaled("me,ie->mi", &[&fova, t1a], 0.5)?;
    let mut foob = einsum_scaled("me,ie->mi", &[&fovb, t1b], 0.5)?;
    let mut fvva = einsum_scaled("me,ma->ae", &[&fova, t1a], -0.5)?;
    let mut fvvb = einsum_scaled("me,ma->ae", &[&fovb, t1b], -0.5)?;
    fooa.add_assign(&eris.fooa()?)?;
    foob.add_assign(&eris.foob()?)?;
    fvva.add_assign(&eris.fvva()?)?;
    fvvb.add_assign(&eris.fvvb()?)?;
    for (i, e) in mo_ea_o.iter().enumerate() {
        fooa.data_mut().re[i * nocca + i] -= e;
    }
    for (i, e) in mo_eb_o.iter().enumerate() {
        foob.data_mut().re[i * noccb + i] -= e;
    }
    for a in 0..nvira {
        fvva.data_mut().re[a * nvira + a] -= eris.mo_energy.0[nocca + a] + level_shift;
    }
    for a in 0..nvirb {
        fvvb.data_mut().re[a * nvirb + a] -= eris.mo_energy.1[noccb + a] + level_shift;
    }

    let mut wovvo = ZArr::zeros(&[nocca, nvira, nvira, nocca]);
    let mut wovvo_bb = ZArr::zeros(&[noccb, nvirb, nvirb, noccb]);
    let mut wovvo_ab = ZArr::zeros(&[nocca, nvirb, nvira, noccb]);
    let mut wovvo_abba = ZArr::zeros(&[nocca, nvirb, nvirb, nocca]);
    let mut wovvo_ba = ZArr::zeros(&[noccb, nvira, nvirb, nocca]);
    let mut wovvo_baab = ZArr::zeros(&[noccb, nvira, nvira, noccb]);

    // `:85-97` — the `ovvv` block, one range.
    if nvira > 0 && nocca > 0 {
        let mut ovvv = eris.ovvv.clone();
        ovvv.sub_assign(&eris.ovvv.transpose(&[0, 3, 2, 1])?)?;
        fvva.add_assign(&einsum("mf,mfae->ae", &[t1a, &ovvv])?)?;
        wovvo.add_assign(&einsum("jf,mebf->mbej", &[t1a, &ovvv])?)?;
        u1a.add_assign(&einsum_scaled("mief,meaf->ia", &[t2aa, &ovvv], 0.5)?)?;
        u2aa.add_assign(&einsum("ie,mbea->imab", &[t1a, &ovvv.conj()])?)?;
        let tmp1aa = einsum("ijef,mebf->ijmb", &[tauaa, &ovvv])?;
        u2aa.sub_assign(&einsum_scaled("ijmb,ma->ijab", &[&tmp1aa, t1a], 0.5)?)?;
    }

    // `:99-111` — `OVVV`.
    if nvirb > 0 && noccb > 0 {
        let mut ovvv = eris.ovvv_bb.clone();
        ovvv.sub_assign(&eris.ovvv_bb.transpose(&[0, 3, 2, 1])?)?;
        fvvb.add_assign(&einsum("mf,mfae->ae", &[t1b, &ovvv])?)?;
        // `:105` is `=`, not `+=`, and `wOVVO` is zero here either way.
        wovvo_bb = einsum("jf,mebf->mbej", &[t1b, &ovvv])?;
        u1b.add_assign(&einsum_scaled("mief,meaf->ia", &[t2bb, &ovvv], 0.5)?)?;
        u2bb.add_assign(&einsum("ie,mbea->imab", &[t1b, &ovvv.conj()])?)?;
        let tmp1bb = einsum("ijef,mebf->ijmb", &[taubb, &ovvv])?;
        u2bb.sub_assign(&einsum_scaled("ijmb,ma->ijab", &[&tmp1bb, t1b], 0.5)?)?;
    }

    // `:113-123` — `ovVV`. NOT antisymmetrised: the two spins are distinct.
    if nvirb > 0 && nocca > 0 {
        let ovvv = &eris.ov_vv;
        fvvb.add_assign(&einsum("mf,mfae->ae", &[t1a, ovvv])?)?;
        wovvo_ab = einsum("jf,mebf->mbej", &[t1b, ovvv])?;
        wovvo_abba = einsum_scaled("jf,mfbe->mbej", &[t1a, ovvv], -1.0)?;
        u1b.add_assign(&einsum("mief,meaf->ia", &[t2ab, ovvv])?)?;
        u2ab.add_assign(&einsum("ie,maeb->miab", &[t1b, &ovvv.conj()])?)?;
        let tmp1ab = einsum("ijef,mebf->ijmb", &[tauab, ovvv])?;
        u2ab.sub_assign(&einsum("ijmb,ma->ijab", &[&tmp1ab, t1a])?)?;
    }

    // `:125-135` — `OVvv`.
    if nvira > 0 && noccb > 0 {
        let ovvv = &eris.ovvv_ba;
        fvva.add_assign(&einsum("mf,mfae->ae", &[t1b, ovvv])?)?;
        wovvo_ba = einsum("jf,mebf->mbej", &[t1a, ovvv])?;
        wovvo_baab = einsum_scaled("jf,mfbe->mbej", &[t1b, ovvv], -1.0)?;
        u1a.add_assign(&einsum("imfe,meaf->ia", &[t2ab, ovvv])?)?;
        u2ab.add_assign(&einsum("ie,mbea->imab", &[t1a, &ovvv.conj()])?)?;
        let tmp1abba = einsum("ijef,mfbe->ijbm", &[tauab, ovvv])?;
        u2ab.sub_assign(&einsum("ijbm,ma->ijba", &[&tmp1abba, t1b])?)?;
    }

    // `:137-160` — the alpha-alpha `ovov` / `ovoo` ring.
    {
        let mut woooo = einsum("je,nemi->mnij", &[t1a, &eris.ovoo])?;
        let sw = woooo.transpose(&[0, 1, 3, 2])?;
        woooo.sub_assign(&sw)?;
        woooo.add_assign(&eris.oooo.transpose(&[0, 2, 1, 3])?)?;
        woooo.add_assign(&einsum_scaled(
            "ijef,menf->mnij",
            &[tauaa, &eris.ovov],
            0.5,
        )?)?;
        let mut w = woooo;
        w.scale(0.5);
        u2aa.add_assign(&einsum("mnab,mnij->ijab", &[tauaa, &w])?)?;

        let mut ovoo = eris.ovoo.clone();
        ovoo.sub_assign(&eris.ovoo.transpose(&[2, 1, 0, 3])?)?;
        fooa.add_assign(&einsum("ne,nemi->mi", &[t1a, &ovoo])?)?;
        u1a.add_assign(&einsum_scaled("mnae,meni->ia", &[t2aa, &ovoo], 0.5)?)?;
        wovvo.add_assign(&einsum("nb,nemj->mbej", &[t1a, &ovoo])?)?;
    }
    let mut fova_i;
    {
        let tilaa = make_tau_aa(t2aa, t1a, t1a, 0.5)?;
        let mut ovov = eris.ovov.clone();
        ovov.sub_assign(&eris.ovov.transpose(&[0, 3, 2, 1])?)?;
        fvva.sub_assign(&einsum_scaled("mnaf,menf->ae", &[&tilaa, &ovov], 0.5)?)?;
        fooa.add_assign(&einsum_scaled("inef,menf->mi", &[&tilaa, &ovov], 0.5)?)?;
        fova_i = einsum("nf,menf->me", &[t1a, &ovov])?;
        let mut c = ovov.conj().transpose(&[0, 2, 1, 3])?;
        c.scale(0.5);
        u2aa.add_assign(&c)?;
        wovvo.sub_assign(&einsum_scaled("jnfb,menf->mbej", &[t2aa, &ovov], 0.5)?)?;
        wovvo_ab.add_assign(&einsum_scaled("njfb,menf->mbej", &[t2ab, &ovov], 0.5)?)?;
        let tmpaa = einsum("jf,menf->mnej", &[t1a, &ovov])?;
        wovvo.sub_assign(&einsum("nb,mnej->mbej", &[t1a, &tmpaa])?)?;
    }

    // `:162-185` — the beta-beta ring.
    {
        let mut woooo = einsum("je,nemi->mnij", &[t1b, &eris.ovoo_bb])?;
        let sw = woooo.transpose(&[0, 1, 3, 2])?;
        woooo.sub_assign(&sw)?;
        woooo.add_assign(&eris.oooo_bb.transpose(&[0, 2, 1, 3])?)?;
        woooo.add_assign(&einsum_scaled(
            "ijef,menf->mnij",
            &[taubb, &eris.ovov_bb],
            0.5,
        )?)?;
        let mut w = woooo;
        w.scale(0.5);
        u2bb.add_assign(&einsum("mnab,mnij->ijab", &[taubb, &w])?)?;

        let mut ovoo = eris.ovoo_bb.clone();
        ovoo.sub_assign(&eris.ovoo_bb.transpose(&[2, 1, 0, 3])?)?;
        foob.add_assign(&einsum("ne,nemi->mi", &[t1b, &ovoo])?)?;
        u1b.add_assign(&einsum_scaled("mnae,meni->ia", &[t2bb, &ovoo], 0.5)?)?;
        wovvo_bb.add_assign(&einsum("nb,nemj->mbej", &[t1b, &ovoo])?)?;
    }
    let mut fovb_i;
    {
        let tilbb = make_tau_aa(t2bb, t1b, t1b, 0.5)?;
        let mut ovov = eris.ovov_bb.clone();
        ovov.sub_assign(&eris.ovov_bb.transpose(&[0, 3, 2, 1])?)?;
        fvvb.sub_assign(&einsum_scaled("mnaf,menf->ae", &[&tilbb, &ovov], 0.5)?)?;
        foob.add_assign(&einsum_scaled("inef,menf->mi", &[&tilbb, &ovov], 0.5)?)?;
        fovb_i = einsum("nf,menf->me", &[t1b, &ovov])?;
        let mut c = ovov.conj().transpose(&[0, 2, 1, 3])?;
        c.scale(0.5);
        u2bb.add_assign(&c)?;
        wovvo_bb.sub_assign(&einsum_scaled("jnfb,menf->mbej", &[t2bb, &ovov], 0.5)?)?;
        wovvo_ba.add_assign(&einsum_scaled("jnbf,menf->mbej", &[t2ab, &ovov], 0.5)?)?;
        let tmpbb = einsum("jf,menf->mnej", &[t1b, &ovov])?;
        wovvo_bb.sub_assign(&einsum("nb,mnej->mbej", &[t1b, &tmpbb])?)?;
    }

    // `:187-203` — the two mixed `ov*oo` blocks and `WoOoO`.
    let mut woooo_ab;
    {
        let ovoo_ba = &eris.ovoo_ba;
        let ovoo_ab = &eris.ov_oo;
        fooa.add_assign(&einsum("ne,nemi->mi", &[t1b, ovoo_ba])?)?;
        u1a.sub_assign(&einsum("nmae,meni->ia", &[t2ab, ovoo_ba])?)?;
        wovvo_ba.sub_assign(&einsum("nb,menj->mbej", &[t1a, ovoo_ba])?)?;
        wovvo_abba.add_assign(&einsum("nb,nemj->mbej", &[t1b, ovoo_ba])?)?;
        foob.add_assign(&einsum("ne,nemi->mi", &[t1a, ovoo_ab])?)?;
        u1b.sub_assign(&einsum("mnea,meni->ia", &[t2ab, ovoo_ab])?)?;
        wovvo_ab.sub_assign(&einsum("nb,menj->mbej", &[t1b, ovoo_ab])?)?;
        wovvo_baab.add_assign(&einsum("nb,nemj->mbej", &[t1a, ovoo_ab])?)?;
        // `:196-198` — `WoOoO`, kept in UPSTREAM's axis order
        // `(m:α-occ, N:β-occ, i:α-occ, J:β-occ)` so every spec below reads
        // against its own line. `:198` adds `ooOO.transpose(0,2,1,3)`, which
        // is what carries the chemists' `(mi|NJ)` into that order.
        woooo_ab = einsum("je,nemi->mnij", &[t1b, ovoo_ba])?;
        woooo_ab.add_assign(&einsum("je,nemi->nmji", &[t1a, ovoo_ab])?)?;
        woooo_ab.add_assign(&eris.oo_oo.transpose(&[0, 2, 1, 3])?)?;
    }

    // `:205-235` — the `ovOV` block, which every mixed intermediate touches.
    {
        let ovov = &eris.ov_ov;
        woooo_ab.add_assign(&einsum("ijef,menf->mnij", &[tauab, ovov])?)?;
        u2ab.add_assign(&einsum("mnab,mnij->ijab", &[tauab, &woooo_ab])?)?;

        let tilab = make_tau_ab(t2ab, t1, t1, 0.5)?;
        fvva.sub_assign(&einsum("mnaf,menf->ae", &[&tilab, ovov])?)?;
        fvvb.sub_assign(&einsum("nmfa,nfme->ae", &[&tilab, ovov])?)?;
        fooa.add_assign(&einsum("inef,menf->mi", &[&tilab, ovov])?)?;
        foob.add_assign(&einsum("nife,nfme->mi", &[&tilab, ovov])?)?;
        fova_i.add_assign(&einsum("nf,menf->me", &[t1b, ovov])?)?;
        fovb_i.add_assign(&einsum("nf,nfme->me", &[t1a, ovov])?)?;
        u2ab.add_assign(&ovov.conj().transpose(&[0, 2, 1, 3])?)?;
        wovvo.add_assign(&einsum_scaled("jnbf,menf->mbej", &[t2ab, ovov], 0.5)?)?;
        wovvo_bb.add_assign(&einsum_scaled("njfb,nfme->mbej", &[t2ab, ovov], 0.5)?)?;
        wovvo_ba.sub_assign(&einsum_scaled("jnfb,nfme->mbej", &[t2aa, ovov], 0.5)?)?;
        wovvo_ab.sub_assign(&einsum_scaled("jnfb,menf->mbej", &[t2bb, ovov], 0.5)?)?;
        wovvo_abba.add_assign(&einsum_scaled("jnfb,mfne->mbej", &[t2ab, ovov], 0.5)?)?;
        wovvo_baab.add_assign(&einsum_scaled("njbf,nemf->mbej", &[t2ab, ovov], 0.5)?)?;
        let tmpabab = einsum("jf,menf->mnej", &[t1b, ovov])?;
        let tmpbaba = einsum("jf,nfme->mnej", &[t1a, ovov])?;
        wovvo_ab.sub_assign(&einsum("nb,mnej->mbej", &[t1b, &tmpabab])?)?;
        wovvo_ba.sub_assign(&einsum("nb,mnej->mbej", &[t1a, &tmpbaba])?)?;
        wovvo_abba.add_assign(&einsum("nb,nmej->mbej", &[t1b, &tmpbaba])?)?;
        wovvo_baab.add_assign(&einsum("nb,nmej->mbej", &[t1a, &tmpabab])?)?;
    }

    // `:237-249`
    fova_i.add_assign(&fova)?;
    fovb_i.add_assign(&fovb)?;
    u1a.add_assign(&fova.conj())?;
    u1a.add_assign(&einsum("ie,ae->ia", &[t1a, &fvva])?)?;
    u1a.sub_assign(&einsum("ma,mi->ia", &[t1a, &fooa])?)?;
    u1a.sub_assign(&einsum("imea,me->ia", &[t2aa, &fova_i])?)?;
    u1a.add_assign(&einsum("imae,me->ia", &[t2ab, &fovb_i])?)?;
    u1b.add_assign(&fovb.conj())?;
    u1b.add_assign(&einsum("ie,ae->ia", &[t1b, &fvvb])?)?;
    u1b.sub_assign(&einsum("ma,mi->ia", &[t1b, &foob])?)?;
    u1b.sub_assign(&einsum("imea,me->ia", &[t2bb, &fovb_i])?)?;
    u1b.add_assign(&einsum("miea,me->ia", &[t2ab, &fova_i])?)?;

    // `:251-259` — the alpha-alpha `oovv` / `ovvo` pair.
    {
        let mut oovv = eris.oovv.clone();
        oovv.sub_assign(&eris.ovvo.transpose(&[0, 3, 2, 1])?)?;
        wovvo.sub_assign(&eris.oovv.transpose(&[0, 2, 3, 1])?)?;
        wovvo.add_assign(&eris.ovvo.transpose(&[0, 2, 1, 3])?)?;
        u1a.sub_assign(&einsum("nf,niaf->ia", &[t1a, &oovv])?)?;
        let tmp1aa = einsum("ie,mjbe->mbij", &[t1a, &oovv])?;
        u2aa.add_assign(&einsum_scaled("ma,mbij->ijab", &[t1a, &tmp1aa], 2.0)?)?;
    }
    // `:261-269`
    {
        let mut oovv = eris.oovv_bb.clone();
        oovv.sub_assign(&eris.ovvo_bb.transpose(&[0, 3, 2, 1])?)?;
        wovvo_bb.sub_assign(&eris.oovv_bb.transpose(&[0, 2, 3, 1])?)?;
        wovvo_bb.add_assign(&eris.ovvo_bb.transpose(&[0, 2, 1, 3])?)?;
        u1b.sub_assign(&einsum("nf,niaf->ia", &[t1b, &oovv])?)?;
        let tmp1bb = einsum("ie,mjbe->mbij", &[t1b, &oovv])?;
        u2bb.add_assign(&einsum_scaled("ma,mbij->ijab", &[t1b, &tmp1bb], 2.0)?)?;
    }
    // `:271-279` — `ooVV` / `ovVO`.
    {
        wovvo_abba.sub_assign(&eris.oo_vv.transpose(&[0, 2, 3, 1])?)?;
        wovvo_ab.add_assign(&eris.ov_vo.transpose(&[0, 2, 1, 3])?)?;
        u1b.add_assign(&einsum("nf,nfai->ia", &[t1a, &eris.ov_vo])?)?;
        let mut tmp1ab = einsum("ie,mebj->mbij", &[t1a, &eris.ov_vo])?;
        tmp1ab.add_assign(&einsum("ie,mjbe->mbji", &[t1b, &eris.oo_vv])?)?;
        u2ab.sub_assign(&einsum("ma,mbij->ijab", &[t1a, &tmp1ab])?)?;
    }
    // `:281-289` — `OOvv` / `OVvo`.
    {
        wovvo_baab.sub_assign(&eris.oovv_ba.transpose(&[0, 2, 3, 1])?)?;
        wovvo_ba.add_assign(&eris.ovvo_ba.transpose(&[0, 2, 1, 3])?)?;
        u1a.add_assign(&einsum("nf,nfai->ia", &[t1b, &eris.ovvo_ba])?)?;
        let mut tmp1ba = einsum("ie,mebj->mbij", &[t1b, &eris.ovvo_ba])?;
        tmp1ba.add_assign(&einsum("ie,mjbe->mbji", &[t1a, &eris.oovv_ba])?)?;
        u2ab.sub_assign(&einsum("ma,mbij->jiba", &[t1b, &tmp1ba])?)?;
    }

    // `:291-301` — the ring contractions.
    u2aa.add_assign(&einsum_scaled("imae,mbej->ijab", &[t2aa, &wovvo], 2.0)?)?;
    u2aa.add_assign(&einsum_scaled("imae,mbej->ijab", &[t2ab, &wovvo_ba], 2.0)?)?;
    u2bb.add_assign(&einsum_scaled("imae,mbej->ijab", &[t2bb, &wovvo_bb], 2.0)?)?;
    u2bb.add_assign(&einsum_scaled("miea,mbej->ijab", &[t2ab, &wovvo_ab], 2.0)?)?;
    u2ab.add_assign(&einsum("imae,mbej->ijab", &[t2aa, &wovvo_ab])?)?;
    u2ab.add_assign(&einsum("imae,mbej->ijab", &[t2ab, &wovvo_bb])?)?;
    u2ab.add_assign(&einsum("imea,mbej->ijba", &[t2ab, &wovvo_baab])?)?;
    u2ab.add_assign(&einsum("imae,mbej->jiba", &[t2bb, &wovvo_ba])?)?;
    u2ab.add_assign(&einsum("miea,mbej->jiba", &[t2ab, &wovvo])?)?;
    u2ab.add_assign(&einsum("miae,mbej->jiab", &[t2ab, &wovvo_abba])?)?;

    // `:303-315`
    let mut ftmpa = fvva.clone();
    ftmpa.sub_assign(&einsum_scaled("mb,me->be", &[t1a, &fova_i], 0.5)?)?;
    let mut ftmpb = fvvb.clone();
    ftmpb.sub_assign(&einsum_scaled("mb,me->be", &[t1b, &fovb_i], 0.5)?)?;
    u2aa.add_assign(&einsum("ijae,be->ijab", &[t2aa, &ftmpa])?)?;
    u2bb.add_assign(&einsum("ijae,be->ijab", &[t2bb, &ftmpb])?)?;
    u2ab.add_assign(&einsum("ijae,be->ijab", &[t2ab, &ftmpb])?)?;
    u2ab.add_assign(&einsum("ijea,be->ijba", &[t2ab, &ftmpa])?)?;
    let mut ftmpa = fooa.clone();
    ftmpa.add_assign(&einsum_scaled("je,me->mj", &[t1a, &fova_i], 0.5)?)?;
    let mut ftmpb = foob.clone();
    ftmpb.add_assign(&einsum_scaled("je,me->mj", &[t1b, &fovb_i], 0.5)?)?;
    u2aa.sub_assign(&einsum("imab,mj->ijab", &[t2aa, &ftmpa])?)?;
    u2bb.sub_assign(&einsum("imab,mj->ijab", &[t2bb, &ftmpb])?)?;
    u2ab.sub_assign(&einsum("imab,mj->ijab", &[t2ab, &ftmpb])?)?;
    u2ab.sub_assign(&einsum("miab,mj->jiab", &[t2ab, &ftmpa])?)?;

    // `:317-327` — the `ovoo.conj()` terms.
    {
        let ovoo_a = eris.ovoo.conj();
        let ovoo_b = eris.ovoo_bb.conj();
        let ovoo_ba = eris.ovoo_ba.conj();
        let ovoo_ab = eris.ov_oo.conj();
        let mut oa = ovoo_a.clone();
        oa.sub_assign(&ovoo_a.transpose(&[2, 1, 0, 3])?)?;
        let mut ob = ovoo_b.clone();
        ob.sub_assign(&ovoo_b.transpose(&[2, 1, 0, 3])?)?;
        u2aa.sub_assign(&einsum("ma,jbim->ijab", &[t1a, &oa])?)?;
        u2bb.sub_assign(&einsum("ma,jbim->ijab", &[t1b, &ob])?)?;
        u2ab.sub_assign(&einsum("ma,jbim->ijab", &[t1a, &ovoo_ba])?)?;
        u2ab.sub_assign(&einsum("ma,jbim->jiba", &[t1b, &ovoo_ab])?)?;
    }

    // `:329-334` — the `P(ij)P(ab)` antisymmetrisation of the same-spin blocks.
    u2aa.scale(0.5);
    u2bb.scale(0.5);
    let sw = u2aa.transpose(&[0, 1, 3, 2])?;
    u2aa.sub_assign(&sw)?;
    let sw = u2aa.transpose(&[1, 0, 2, 3])?;
    u2aa.sub_assign(&sw)?;
    let sw = u2bb.transpose(&[0, 1, 3, 2])?;
    u2bb.sub_assign(&sw)?;
    let sw = u2bb.transpose(&[1, 0, 2, 3])?;
    u2bb.sub_assign(&sw)?;

    // `:336-343` — the denominators.
    let ea = eia(&eris.mo_energy.0, nocca, nvira, level_shift);
    let eb = eia(&eris.mo_energy.1, noccb, nvirb, level_shift);
    for (f, d) in ea.iter().enumerate() {
        u1a.data_mut().re[f] /= d;
        u1a.data_mut().im[f] /= d;
    }
    for (f, d) in eb.iter().enumerate() {
        u1b.data_mut().re[f] /= d;
        u1b.data_mut().im[f] /= d;
    }
    divide_ijab(&mut u2aa, &ea, &ea, [nocca, nocca, nvira, nvira]);
    divide_ijab(&mut u2ab, &ea, &eb, [nocca, noccb, nvira, nvirb]);
    divide_ijab(&mut u2bb, &eb, &eb, [noccb, noccb, nvirb, nvirb]);

    Ok(((u1a, u1b), (u2aa, u2ab, u2bb)))
}

/// What [`kernel`] returns.
#[derive(Debug, Clone)]
pub struct UccsdResult {
    pub converged: bool,
    pub e_corr: f64,
    pub emp2: f64,
    pub t1: UT1,
    pub t2: UT2,
    pub niter: usize,
    /// The largest `|Im(E_corr)|` seen. `energy` (`:800`) warns above `1e-4`.
    pub max_imag: f64,
}

/// The DIIS iterate for the three-block unrestricted amplitudes.
fn flatten(t1: &UT1, t2: &UT2) -> Vec<f64> {
    let mut v = Vec::new();
    for z in [&t1.0, &t1.1, &t2.0, &t2.1, &t2.2] {
        v.extend_from_slice(&z.data().re);
        v.extend_from_slice(&z.data().im);
    }
    v
}

fn unflatten(flat: &[f64], t1: &UT1, t2: &UT2) -> (UT1, UT2) {
    let mut off = 0;
    let mut take = |z: &ZArr| -> ZArr {
        let n = z.len();
        let mut out = z.clone();
        out.data_mut().re.copy_from_slice(&flat[off..off + n]);
        out.data_mut()
            .im
            .copy_from_slice(&flat[off + n..off + 2 * n]);
        off += 2 * n;
        out
    };
    let a = take(&t1.0);
    let b = take(&t1.1);
    let c = take(&t2.0);
    let d = take(&t2.1);
    let e = take(&t2.2);
    ((a, b), (c, d, e))
}

/// `ccsd.CCSDBase.ccsd` driven with this module's `update_amps` and `energy`.
///
/// # Errors
/// Propagates every amplitude update and the DIIS solve.
pub fn kernel(
    eris: &ChemistsErisU,
    opts: &crate::rccsd::RccsdOpts,
) -> Result<UccsdResult, PbcCcError> {
    use pyscf_diis::{Diis, DiisStorable};

    /// The flat `[re…, im…]` packing, as [`crate::kccsd_rhf::KAmplitudeSubspace`]
    /// does for the restricted case and for the same reason: CDIIS forms a REAL
    /// combination, which acts on each plane independently.
    #[derive(Debug, Clone)]
    struct UAmp(Vec<f64>);
    impl DiisStorable for UAmp {
        fn as_flat(&self) -> &[f64] {
            &self.0
        }
        fn from_flat(&mut self, s: &[f64]) {
            self.0.copy_from_slice(s);
        }
        fn dot(&self, other: &Self) -> f64 {
            pyscf_algebra::oracle_dot(&self.0, &other.0)
        }
        fn len(&self) -> usize {
            self.0.len()
        }
    }

    let (emp2, mut t1, mut t2) = init_amps(eris)?;
    let (mut e_corr, mut max_imag) = energy(&t1, &t2, eris)?;
    max_imag = max_imag.abs();
    let mut converged = false;
    let mut niter = 0;
    let mut diis = if opts.diis {
        Some(Diis::<UAmp>::new(opts.diis_space))
    } else {
        None
    };

    for istep in 0..opts.max_cycle {
        niter = istep + 1;
        let (t1new, t2new) = update_amps(&t1, &t2, eris, opts.level_shift)?;
        let cur = flatten(&t1new, &t2new);
        let prev = flatten(&t1, &t2);
        let res: Vec<f64> = cur.iter().zip(&prev).map(|(a, b)| a - b).collect();
        let normt = pyscf_algebra::oracle_dot(&res, &res).sqrt();
        t1 = t1new;
        t2 = t2new;

        if let Some(stack) = diis.as_mut()
            && istep >= opts.diis_start_cycle
        {
            let cur = flatten(&t1, &t2);
            let err: Vec<f64> = cur.iter().zip(&prev).map(|(a, b)| a - b).collect();
            let extrap = stack
                .extrapolate(UAmp(cur), err)
                .map_err(|e| PbcCcError::Algebra(format!("amplitude DIIS: {e}")))?;
            let (a, b) = unflatten(&extrap.0, &t1, &t2);
            t1 = a;
            t2 = b;
        }

        let eold = e_corr;
        let (e, im) = energy(&t1, &t2, eris)?;
        e_corr = e;
        max_imag = max_imag.max(im.abs());
        if (e_corr - eold).abs() < opts.conv_tol && normt < opts.conv_tol_normt {
            converged = true;
            break;
        }
    }
    Ok(UccsdResult {
        converged,
        e_corr,
        emp2,
        t1,
        t2,
        niter,
        max_imag,
    })
}
