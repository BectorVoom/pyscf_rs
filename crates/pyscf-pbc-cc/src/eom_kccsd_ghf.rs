//! `eom_kccsd_ghf` — equation-of-motion CCSD over spin ORBITALS at k-points
//! (plan 16-09; `pyscf/pbc/cc/eom_kccsd_ghf.py`, 2011 l).
//!
//! # This module ships FIRST of the three EOM ports, and that is upstream's order
//!
//! `eom_kccsd_rhf.py:25` and `eom_kccsd_uhf.py:29` both `import eom_kccsd_ghf as
//! eom_kgccsd` and inherit its `EOMIP`/`EOMEA`/`EOMEE`. `PBC-MASTER-PLAN §8.8`
//! ordered the base class LAST; `16-CONTEXT §1` corrected that before any code,
//! and this is where the correction lands.
//!
//! # The `r2` packing is antisymmetric and stored as a strict lower triangle
//!
//! `vector_to_amplitudes_ip` (`:318-328`) reads `nkpts·nocc·(nkpts·nocc−1)/2`
//! blocks of `nvir` into the strictly-lower triangle of a
//! `(nkpts·nocc) × (nkpts·nocc) × nvir` array, mirrors them with a MINUS sign,
//! and only then splits the composite index back into `(k, orbital)`. The
//! diagonal is identically zero. A port that stored the full square would
//! agree on every matvec and disagree on `vector_size`, which is what the
//! Davidson allocates — so [`ip_vector_size`] is asserted directly.
//!
//! # `Woovv` is `eris.oovv`, not a built intermediate
//!
//! `_IMDS._make_shared` (`:1872`) assigns `self.Woovv = eris.oovv`. It is
//! spelled `Woovv` at every use site, so this port keeps the name and the
//! aliasing rather than substituting the ERI block silently.

use pyscf_algebra::CTensor;
use pyscf_pbc_lib::{KIdx, Kconserv, get_kconserv3};

use crate::error::PbcCcError;
use crate::kccsd::{GBlk, KgEris};
use crate::kintermediates as gimd;
use crate::zarr::{ZArr, einsum, einsum_scaled};

/// The intermediates `_IMDS` (`:1841-1966`) caches, built once per `kshift`
/// sweep and shared by the matvec, its left sibling and the diagonal.
///
/// `make_ip` and `make_ea` build overlapping sets — both want `Woooo` — so the
/// two flags record what is present rather than rebuilding.
pub struct EomImds<'a> {
    pub eris: &'a KgEris,
    pub t1: ZArr,
    pub t2: ZArr,
    /// `[nkpts, nocc, nocc]`.
    pub foo: ZArr,
    /// `[nkpts, nvir, nvir]`.
    pub fvv: ZArr,
    /// `[nkpts, nocc, nvir]`.
    pub fov: ZArr,
    /// `[nkpts³, nocc, nvir, nvir, nocc]`.
    pub wovvo: ZArr,
    /// `[nkpts³, nocc, nocc, nocc, nocc]` — IP and EA both want it.
    pub woooo: Option<ZArr>,
    /// `[nkpts³, nocc, nocc, nocc, nvir]` — IP only.
    pub wooov: Option<ZArr>,
    /// `[nkpts³, nocc, nvir, nocc, nocc]` — IP only.
    pub wovoo: Option<ZArr>,
    /// `[nkpts³, nvir, nocc, nvir, nvir]` — EA only.
    pub wvovv: Option<ZArr>,
    /// `[nkpts³, nvir, nvir, nvir, nvir]` — EA only.
    pub wvvvv: Option<ZArr>,
    /// `[nkpts³, nvir, nvir, nvir, nocc]` — EA only.
    pub wvvvo: Option<ZArr>,
}

impl<'a> EomImds<'a> {
    /// `_IMDS._make_shared` (`:1863-1876`) — `Foo`, `Fvv`, `Fov`, `Wovvo`.
    ///
    /// # Errors
    /// Propagates every intermediate build.
    pub fn make_shared(
        t1: &ZArr,
        t2: &ZArr,
        eris: &'a KgEris,
        kconserv: &Kconserv,
    ) -> Result<Self, PbcCcError> {
        Ok(Self {
            eris,
            t1: t1.clone(),
            t2: t2.clone(),
            foo: gimd::foo(t1, t2, eris, kconserv)?,
            fvv: gimd::fvv(t1, t2, eris, kconserv)?,
            fov: gimd::fov(t1, eris)?,
            wovvo: gimd::wovvo(t1, t2, eris, kconserv)?,
            woooo: None,
            wooov: None,
            wovoo: None,
            wvovv: None,
            wvvvv: None,
            wvvvo: None,
        })
    }

    /// `_IMDS.make_ip` (`:1878-1893`) — adds `Woooo`, `Wooov`, `Wovoo`.
    ///
    /// # Errors
    /// Propagates every intermediate build.
    pub fn make_ip(mut self, kconserv: &Kconserv) -> Result<Self, PbcCcError> {
        let (t1, t2) = (self.t1.clone(), self.t2.clone());
        if self.woooo.is_none() {
            self.woooo = Some(gimd::woooo(&t1, &t2, self.eris, kconserv)?);
        }
        self.wooov = Some(gimd::wooov(&t1, self.eris)?);
        self.wovoo = Some(gimd::wovoo(&t1, &t2, self.eris, kconserv)?);
        Ok(self)
    }

    /// `_IMDS.make_ea` (`:1913-1932`) — adds `Woooo`, `Wvovv`, `Wvvvv`, `Wvvvo`.
    ///
    /// `Wvvvo` is given the `Wvvvv` this call just built, which is what
    /// `make_ee` does explicitly at `:1966` and what `make_ea` gets for free by
    /// ordering; building it twice would be the phase's largest tensor twice.
    ///
    /// # Errors
    /// Propagates every intermediate build.
    pub fn make_ea(mut self, kconserv: &Kconserv) -> Result<Self, PbcCcError> {
        let (t1, t2) = (self.t1.clone(), self.t2.clone());
        if self.woooo.is_none() {
            self.woooo = Some(gimd::woooo(&t1, &t2, self.eris, kconserv)?);
        }
        self.wvovv = Some(gimd::wvovv(&t1, self.eris)?);
        let w4 = gimd::wvvvv(&t1, &t2, self.eris, kconserv)?;
        self.wvvvo = Some(gimd::wvvvo(&t1, &t2, self.eris, kconserv, Some(&w4))?);
        self.wvvvv = Some(w4);
        Ok(self)
    }

    fn need<'w>(&self, w: &'w Option<ZArr>, what: &'static str) -> Result<&'w ZArr, PbcCcError> {
        w.as_ref()
            .ok_or_else(|| PbcCcError::Shape(format!("{what} was not built; call make_ip/make_ea")))
    }
}

/// `EOMIP.vector_size` (`:751-755`).
pub fn ip_vector_size(nkpts: usize, nocc: usize, nvir: usize) -> usize {
    let n = nkpts * nocc;
    nocc + n * (n - 1) * nvir / 2
}

/// `vector_to_amplitudes_ip` (`:318-328`). Returns `(r1, r2)` with `r1` shaped
/// `[nocc]` and `r2` shaped `[nkpts, nkpts, nocc, nocc, nvir]`.
///
/// # Errors
/// [`PbcCcError::Shape`] if the vector is not [`ip_vector_size`] long.
pub fn vector_to_amplitudes_ip(
    vector: &ZArr,
    nkpts: usize,
    nocc: usize,
    nvir: usize,
) -> Result<(ZArr, ZArr), PbcCcError> {
    let want = ip_vector_size(nkpts, nocc, nvir);
    if vector.len() != want {
        return Err(PbcCcError::Shape(format!(
            "IP vector of {} elements, expected {want}",
            vector.len()
        )));
    }
    let mut r1 = ZArr::zeros(&[nocc]);
    r1.data_mut().re[..nocc].copy_from_slice(&vector.data().re[..nocc]);
    r1.data_mut().im[..nocc].copy_from_slice(&vector.data().im[..nocc]);

    let mut r2 = ZArr::zeros(&[nkpts, nkpts, nocc, nocc, nvir]);
    let n = nkpts * nocc;
    let mut cur = nocc;
    // `np.tril_indices(n, -1)` in row-major order: for each row `p`, every
    // column `q < p`. The mirrored entry carries a MINUS sign (`:326`).
    for p in 0..n {
        for q in 0..p {
            let (kp, i) = (p / nocc, p % nocc);
            let (kq, j) = (q / nocc, q % nocc);
            for a in 0..nvir {
                let (re, im) = (vector.data().re[cur + a], vector.data().im[cur + a]);
                let f = (((kp * nkpts + kq) * nocc + i) * nocc + j) * nvir + a;
                r2.data_mut().re[f] = re;
                r2.data_mut().im[f] = im;
                let g = (((kq * nkpts + kp) * nocc + j) * nocc + i) * nvir + a;
                r2.data_mut().re[g] = -re;
                r2.data_mut().im[g] = -im;
            }
            cur += nvir;
        }
    }
    Ok((r1, r2))
}

/// `amplitudes_to_vector_ip` (`:330-336`) — the inverse of
/// [`vector_to_amplitudes_ip`], reading the strict lower triangle only.
///
/// # Errors
/// [`PbcCcError::Shape`] on a shape mismatch.
pub fn amplitudes_to_vector_ip(r1: &ZArr, r2: &ZArr) -> Result<ZArr, PbcCcError> {
    let nkpts = r2.shape()[0];
    let nocc = r2.shape()[2];
    let nvir = r2.shape()[4];
    if r1.shape() != [nocc] {
        return Err(PbcCcError::Shape(format!(
            "IP r1 shape {:?}, expected [{nocc}]",
            r1.shape()
        )));
    }
    let mut v = ZArr::zeros(&[ip_vector_size(nkpts, nocc, nvir)]);
    v.data_mut().re[..nocc].copy_from_slice(&r1.data().re);
    v.data_mut().im[..nocc].copy_from_slice(&r1.data().im);
    let n = nkpts * nocc;
    let mut cur = nocc;
    for p in 0..n {
        for q in 0..p {
            let (kp, i) = (p / nocc, p % nocc);
            let (kq, j) = (q / nocc, q % nocc);
            for a in 0..nvir {
                let f = (((kp * nkpts + kq) * nocc + i) * nocc + j) * nvir + a;
                v.data_mut().re[cur + a] = r2.data().re[f];
                v.data_mut().im[cur + a] = r2.data().im[f];
            }
            cur += nvir;
        }
    }
    Ok(v)
}

/// `ipccsd_matvec` (`:338-384`).
///
/// # Errors
/// Propagates every intermediate access and shape check.
pub fn ipccsd_matvec(
    vector: &ZArr,
    kshift: usize,
    imds: &EomImds<'_>,
    kconserv: &Kconserv,
) -> Result<ZArr, PbcCcError> {
    let (nkpts, nocc, nvir) = (imds.eris.nkpts, imds.eris.nocc, imds.eris.nvir);
    let (r1, r2) = vector_to_amplitudes_ip(vector, nkpts, nocc, nvir)?;
    let woooo = imds.need(&imds.woooo, "Woooo")?;
    let wooov = imds.need(&imds.wooov, "Wooov")?;
    let wovoo = imds.need(&imds.wovoo, "Wovoo")?;

    // `:349-355`
    let mut hr1 = einsum("mi,m->i", &[&imds.foo.slice_leading(&[kshift])?, &r1])?;
    hr1.scale(-1.0);
    for km in 0..nkpts {
        hr1.add_assign(&einsum(
            "me,mie->i",
            &[
                &imds.fov.slice_leading(&[km])?,
                &r2.slice_leading(&[km, kshift])?,
            ],
        )?)?;
        for kn in 0..nkpts {
            hr1.sub_assign(&einsum_scaled(
                "nmie,mne->i",
                &[
                    &wooov.slice_leading(&[kn, km, kshift])?,
                    &r2.slice_leading(&[km, kn])?,
                ],
                0.5,
            )?)?;
        }
    }

    let mut hr2 = ZArr::zeros(&[nkpts, nkpts, nocc, nocc, nvir]);
    // `:357-370`
    for ki in 0..nkpts {
        for kj in 0..nkpts {
            let ka = kconserv.get(ki, kshift, kj) as usize;
            let mut blk = einsum(
                "ae,ije->ija",
                &[
                    &imds.fvv.slice_leading(&[ka])?,
                    &r2.slice_leading(&[ki, kj])?,
                ],
            )?;
            blk.sub_assign(&einsum(
                "mi,mja->ija",
                &[
                    &imds.foo.slice_leading(&[ki])?,
                    &r2.slice_leading(&[ki, kj])?,
                ],
            )?)?;
            blk.add_assign(&einsum(
                "mj,mia->ija",
                &[
                    &imds.foo.slice_leading(&[kj])?,
                    &r2.slice_leading(&[kj, ki])?,
                ],
            )?)?;
            blk.sub_assign(&einsum(
                "maji,m->ija",
                &[&wovoo.slice_leading(&[kshift, ka, kj])?, &r1],
            )?)?;
            for km in 0..nkpts {
                let kn = kconserv.get(ki, km, kj) as usize;
                blk.add_assign(&einsum_scaled(
                    "mnij,mna->ija",
                    &[
                        &woooo.slice_leading(&[km, kn, ki])?,
                        &r2.slice_leading(&[km, kn])?,
                    ],
                    0.5,
                )?)?;
            }
            hr2.set_leading(&[ki, kj], &blk)?;
        }
    }
    // `:372-380` — a SECOND loop upstream, kept separate.
    for ki in 0..nkpts {
        for kj in 0..nkpts {
            let ka = kconserv.get(ki, kshift, kj) as usize;
            let mut blk = hr2.slice_leading(&[ki, kj])?;
            for km in 0..nkpts {
                let ke = kconserv.get(km, kshift, kj) as usize;
                blk.add_assign(&einsum(
                    "maei,mje->ija",
                    &[
                        &imds.wovvo.slice_leading(&[km, ka, ke])?,
                        &r2.slice_leading(&[km, kj])?,
                    ],
                )?)?;
                let ke = kconserv.get(km, kshift, ki) as usize;
                blk.sub_assign(&einsum(
                    "maej,mie->ija",
                    &[
                        &imds.wovvo.slice_leading(&[km, ka, ke])?,
                        &r2.slice_leading(&[km, ki])?,
                    ],
                )?)?;
            }
            hr2.set_leading(&[ki, kj], &blk)?;
        }
    }

    // `:382-383` — the one term that contracts ALL of `r2` into a single
    // `nvir` vector and broadcasts it back.
    let mut tmp = ZArr::zeros(&[nvir]);
    for kx in 0..nkpts {
        for ky in 0..nkpts {
            tmp.add_assign(&einsum(
                "mnef,mnf->e",
                &[
                    &imds.eris.blk(GBlk::Oovv, kx, ky, kshift)?,
                    &r2.slice_leading(&[kx, ky])?,
                ],
            )?)?;
        }
    }
    for kx in 0..nkpts {
        for ky in 0..nkpts {
            let mut blk = hr2.slice_leading(&[kx, ky])?;
            blk.add_assign(&einsum_scaled(
                "e,jiea->ija",
                &[&tmp, &imds.t2.slice_leading(&[ky, kx, kshift])?],
                0.5,
            )?)?;
            hr2.set_leading(&[kx, ky], &blk)?;
        }
    }

    amplitudes_to_vector_ip(&hr1, &hr2)
}

/// `lipccsd_matvec` (`:386-439`) — the LEFT eigenvector matvec.
///
/// Not the conjugate transpose of [`ipccsd_matvec`] computed numerically: it is
/// a separately derived contraction list, and the `2ph` operator it acts on is
/// `s_{ij}^{ b}` rather than `s_{ij}^{a }` (upstream's own docstrings at `:339`
/// and `:387`).
///
/// # Errors
/// As [`ipccsd_matvec`].
pub fn lipccsd_matvec(
    vector: &ZArr,
    kshift: usize,
    imds: &EomImds<'_>,
    kconserv: &Kconserv,
) -> Result<ZArr, PbcCcError> {
    let (nkpts, nocc, nvir) = (imds.eris.nkpts, imds.eris.nocc, imds.eris.nvir);
    let (r1, r2) = vector_to_amplitudes_ip(vector, nkpts, nocc, nvir)?;
    let woooo = imds.need(&imds.woooo, "Woooo")?;
    let wooov = imds.need(&imds.wooov, "Wooov")?;
    let wovoo = imds.need(&imds.wovoo, "Wovoo")?;

    // `:397-401`
    let mut hr1 = einsum("mi,i->m", &[&imds.foo.slice_leading(&[kshift])?, &r1])?;
    hr1.scale(-1.0);
    for ki in 0..nkpts {
        for kj in 0..nkpts {
            let ka = kconserv.get(ki, kshift, kj) as usize;
            hr1.sub_assign(&einsum_scaled(
                "maji,ija->m",
                &[
                    &wovoo.slice_leading(&[kshift, ka, kj])?,
                    &r2.slice_leading(&[ki, kj])?,
                ],
                0.5,
            )?)?;
        }
    }

    let mut hr2 = ZArr::zeros(&[nkpts, nkpts, nocc, nocc, nvir]);
    // `:404-409` — note the two `(km==ke)` / `(kn==ke)` guards: they are
    // momentum conditions written as multiplications by a boolean upstream.
    for km in 0..nkpts {
        for kn in 0..nkpts {
            let ke = kconserv.get(km, kshift, kn) as usize;
            let mut blk = hr2.slice_leading(&[km, kn])?;
            blk.sub_assign(&einsum(
                "nmie,i->mne",
                &[&wooov.slice_leading(&[kn, km, kshift])?, &r1],
            )?)?;
            hr2.set_leading(&[km, kn], &blk)?;

            if km == ke {
                let v = einsum("me,n->mne", &[&imds.fov.slice_leading(&[km])?, &r1])?;
                let mut b = hr2.slice_leading(&[km, kshift])?;
                b.add_assign(&v)?;
                hr2.set_leading(&[km, kshift], &b)?;
            }
            if kn == ke {
                let v = einsum("ne,m->mne", &[&imds.fov.slice_leading(&[kn])?, &r1])?;
                let mut b = hr2.slice_leading(&[kshift, kn])?;
                b.sub_assign(&v)?;
                hr2.set_leading(&[kshift, kn], &b)?;
            }
        }
    }

    // `:411-426`
    for km in 0..nkpts {
        for kn in 0..nkpts {
            let ke = kconserv.get(km, kshift, kn) as usize;
            let mut blk = hr2.slice_leading(&[km, kn])?;
            blk.add_assign(&einsum(
                "ae,mna->mne",
                &[
                    &imds.fvv.slice_leading(&[ke])?,
                    &r2.slice_leading(&[km, kn])?,
                ],
            )?)?;
            blk.sub_assign(&einsum(
                "mi,ine->mne",
                &[
                    &imds.foo.slice_leading(&[km])?,
                    &r2.slice_leading(&[km, kn])?,
                ],
            )?)?;
            blk.add_assign(&einsum(
                "ni,ime->mne",
                &[
                    &imds.foo.slice_leading(&[kn])?,
                    &r2.slice_leading(&[kn, km])?,
                ],
            )?)?;
            for ki in 0..nkpts {
                let kj = kconserv.get(km, ki, kn) as usize;
                blk.add_assign(&einsum_scaled(
                    "mnij,ije->mne",
                    &[
                        &woooo.slice_leading(&[km, kn, ki])?,
                        &r2.slice_leading(&[ki, kj])?,
                    ],
                    0.5,
                )?)?;
                let ka = kconserv.get(ke, km, ki) as usize;
                blk.add_assign(&einsum(
                    "maei,ina->mne",
                    &[
                        &imds.wovvo.slice_leading(&[km, ka, ke])?,
                        &r2.slice_leading(&[ki, kn])?,
                    ],
                )?)?;
                let ka = kconserv.get(ke, kn, ki) as usize;
                blk.sub_assign(&einsum(
                    "naei,ima->mne",
                    &[
                        &imds.wovvo.slice_leading(&[kn, ka, ke])?,
                        &r2.slice_leading(&[ki, km])?,
                    ],
                )?)?;
            }
            hr2.set_leading(&[km, kn], &blk)?;
        }
    }

    // `:428-436` — `kf` is `kshift`, fixed outside the loop upstream (`:431`).
    let kf = kshift;
    let mut tmp = ZArr::zeros(&[nvir]);
    for ki in 0..nkpts {
        for kj in 0..nkpts {
            let ka = kconserv.get(ki, kshift, kj) as usize;
            tmp.add_assign(&einsum(
                "ija,ijaf->f",
                &[
                    &r2.slice_leading(&[ki, kj])?,
                    &imds.t2.slice_leading(&[ki, kj, ka])?,
                ],
            )?)?;
        }
    }
    for km in 0..nkpts {
        for kn in 0..nkpts {
            let mut blk = hr2.slice_leading(&[km, kn])?;
            blk.add_assign(&einsum_scaled(
                "mnfe,f->mne",
                &[&imds.eris.blk(GBlk::Oovv, km, kn, kf)?, &tmp],
                0.5,
            )?)?;
            hr2.set_leading(&[km, kn], &blk)?;
        }
    }

    amplitudes_to_vector_ip(&hr1, &hr2)
}

/// `ipccsd_diag` (`:441-476`), the `partition = None` branch.
///
/// The `'mp'` partition (`:449-457`) reads the bare Fock diagonal instead of
/// `Foo`/`Fvv` and drops the four `W` terms; it is selected by
/// `eom.partition == 'mp'` and is NOT ported here — nothing in this phase sets
/// it, and porting a branch with no caller means shipping untested arithmetic.
/// [`PbcCcError::NotImplementedUpstream`] is not the right answer either, since
/// upstream implements it; it is simply out of scope and recorded as such.
///
/// # Errors
/// As [`ipccsd_matvec`].
pub fn ipccsd_diag(
    kshift: usize,
    imds: &EomImds<'_>,
    kconserv: &Kconserv,
) -> Result<ZArr, PbcCcError> {
    let (nkpts, nocc, nvir) = (imds.eris.nkpts, imds.eris.nocc, imds.eris.nvir);
    let woooo = imds.need(&imds.woooo, "Woooo")?;

    let mut hr1 = ZArr::zeros(&[nocc]);
    let foo_s = imds.foo.slice_leading(&[kshift])?;
    for i in 0..nocc {
        let (re, im) = foo_s.at(&[i, i])?;
        hr1.data_mut().re[i] = -re;
        hr1.data_mut().im[i] = -im;
    }

    let mut hr2 = ZArr::zeros(&[nkpts, nkpts, nocc, nocc, nvir]);
    for ki in 0..nkpts {
        for kj in 0..nkpts {
            let ka = kconserv.get(ki, kshift, kj) as usize;
            let mut blk = ZArr::zeros(&[nocc, nocc, nvir]);
            let fi = imds.foo.slice_leading(&[ki])?;
            let fj = imds.foo.slice_leading(&[kj])?;
            let fa = imds.fvv.slice_leading(&[ka])?;
            for i in 0..nocc {
                for j in 0..nocc {
                    for a in 0..nvir {
                        let f = (i * nocc + j) * nvir + a;
                        let (r, m) = fi.at(&[i, i])?;
                        blk.data_mut().re[f] -= r;
                        blk.data_mut().im[f] -= m;
                        let (r, m) = fj.at(&[j, j])?;
                        blk.data_mut().re[f] -= r;
                        blk.data_mut().im[f] -= m;
                        let (r, m) = fa.at(&[a, a])?;
                        blk.data_mut().re[f] += r;
                        blk.data_mut().im[f] += m;
                    }
                }
            }
            // `:465-466` — `einsum('ijij->ij', Woooo[ki,kj,ki])`, a DOUBLE
            // diagonal, written as an index loop because it is not an einsum
            // this port's parser accepts (repeated letters within one operand).
            if ki == kconserv.get(ki, kj, kj) as usize {
                let w = woooo.slice_leading(&[ki, kj, ki])?;
                for i in 0..nocc {
                    for j in 0..nocc {
                        let (r, m) = w.at(&[i, j, i, j])?;
                        for a in 0..nvir {
                            let f = (i * nocc + j) * nvir + a;
                            blk.data_mut().re[f] += r;
                            blk.data_mut().im[f] += m;
                        }
                    }
                }
            }
            // `:468-469` — `einsum('iaai->ia', Wovvo[ki,ka,ka])` and its `j`
            // sibling, likewise diagonals.
            let wi = imds.wovvo.slice_leading(&[ki, ka, ka])?;
            let wj = imds.wovvo.slice_leading(&[kj, ka, ka])?;
            for a in 0..nvir {
                for i in 0..nocc {
                    let (r, m) = wi.at(&[i, a, a, i])?;
                    for j in 0..nocc {
                        let f = (i * nocc + j) * nvir + a;
                        blk.data_mut().re[f] += r;
                        blk.data_mut().im[f] += m;
                    }
                }
                for j in 0..nocc {
                    let (r, m) = wj.at(&[j, a, a, j])?;
                    for i in 0..nocc {
                        let f = (i * nocc + j) * nvir + a;
                        blk.data_mut().re[f] += r;
                        blk.data_mut().im[f] += m;
                    }
                }
            }
            // `:471`
            blk.add_assign(&einsum(
                "ijea,jiea->ija",
                &[
                    &imds.eris.blk(GBlk::Oovv, ki, kj, kshift)?,
                    &imds.t2.slice_leading(&[kj, ki, kshift])?,
                ],
            )?)?;
            hr2.set_leading(&[ki, kj], &blk)?;
        }
    }
    amplitudes_to_vector_ip(&hr1, &hr2)
}

/// `mask_frozen_ip` (`:663-682`) — replace every PADDED index with `const`.
///
/// The default `const` is [`crate::kccsd_rhf::LARGE_DENOM`], which is
/// arithmetic and not a skip: the Davidson still sees those entries, they are
/// just pushed far from every root.
///
/// # Errors
/// As [`ipccsd_matvec`].
pub fn mask_frozen_ip(
    vector: &ZArr,
    kshift: usize,
    nkpts: usize,
    nocc: usize,
    nvir: usize,
    nonzero_opadding: &[Vec<usize>],
    nonzero_vpadding: &[Vec<usize>],
    kconserv: &Kconserv,
    konst: f64,
) -> Result<ZArr, PbcCcError> {
    let (r1, r2) = vector_to_amplitudes_ip(vector, nkpts, nocc, nvir)?;
    let mut new_r1 = ZArr::zeros(&[nocc]);
    for v in new_r1.data_mut().re.iter_mut() {
        *v = konst;
    }
    for &i in &nonzero_opadding[kshift] {
        if i < nocc {
            new_r1.data_mut().re[i] = r1.data().re[i];
            new_r1.data_mut().im[i] = r1.data().im[i];
        }
    }

    let mut new_r2 = ZArr::zeros(&[nkpts, nkpts, nocc, nocc, nvir]);
    for v in new_r2.data_mut().re.iter_mut() {
        *v = konst;
    }
    for ki in 0..nkpts {
        for kj in 0..nkpts {
            let kb = kconserv.get(ki, kshift, kj) as usize;
            for &i in &nonzero_opadding[ki] {
                for &j in &nonzero_opadding[kj] {
                    for &a in &nonzero_vpadding[kb] {
                        if i >= nocc || j >= nocc || a >= nvir {
                            continue;
                        }
                        let f = (((ki * nkpts + kj) * nocc + i) * nocc + j) * nvir + a;
                        new_r2.data_mut().re[f] = r2.data().re[f];
                        new_r2.data_mut().im[f] = r2.data().im[f];
                    }
                }
            }
        }
    }
    amplitudes_to_vector_ip(&new_r1, &new_r2)
}

// ---------------------------------------------------------------------------
// EA (`:771-1199`)
// ---------------------------------------------------------------------------

/// The `(a, b)` virtual pairs one `(kj, ka)` block contributes to the EA
/// vector — `np.tril_indices(nvir, 0)` when `ka < kb`, `np.tril_indices(nvir,
/// -1)` otherwise (`:869-873`, `:889-893`).
///
/// The `ka < kb` test is on the k-point INDICES, not on anything physical: it
/// is upstream's way of choosing one representative of each `{ka, kb}` pair,
/// and the diagonal `a == b` belongs to exactly one of the two.
fn ea_pairs(nvir: usize, ka: usize, kb: usize) -> Vec<(usize, usize)> {
    let lo = if ka < kb { 0_i64 } else { -1 };
    let mut v = Vec::new();
    for a in 0..nvir {
        for b in 0..nvir {
            if (b as i64) <= a as i64 + lo {
                v.push((a, b));
            }
        }
    }
    v
}

/// `EOMEA.vector_size` — `nvir` plus the EA `r2` triangle.
///
/// Unlike [`ip_vector_size`] this depends on `kshift`, because the pair list
/// does. Upstream's closed form (`:889`) assumes the total is
/// `nocc·nkpts·nvir·(nkpts·nvir−1)/2` regardless; this counts what the packing
/// actually writes, and [`ea_vector_size_matches_upstreams_formula`] in the
/// test suite asserts the two agree.
pub fn ea_vector_size(
    nkpts: usize,
    nocc: usize,
    nvir: usize,
    kshift: usize,
    kconserv: &Kconserv,
) -> usize {
    let mut n = nvir;
    for kj in 0..nkpts {
        for ka in 0..nkpts {
            let kb = kconserv.get(kshift, ka, kj) as usize;
            n += nocc * ea_pairs(nvir, ka, kb).len();
        }
    }
    n
}

/// `vector_to_amplitudes_ea` (`:880-899`). Returns `(r1, r2)` with `r1` shaped
/// `[nvir]` and `r2` shaped `[nkpts, nkpts, nocc, nvir, nvir]`, indexed
/// `[kj, ka]`.
///
/// # Errors
/// [`PbcCcError::Shape`] if the vector length disagrees with the packing.
pub fn vector_to_amplitudes_ea(
    vector: &ZArr,
    kshift: usize,
    nkpts: usize,
    nocc: usize,
    nvir: usize,
    kconserv: &Kconserv,
) -> Result<(ZArr, ZArr), PbcCcError> {
    let want = ea_vector_size(nkpts, nocc, nvir, kshift, kconserv);
    if vector.len() != want {
        return Err(PbcCcError::Shape(format!(
            "EA vector of {} elements, expected {want}",
            vector.len()
        )));
    }
    let mut r1 = ZArr::zeros(&[nvir]);
    r1.data_mut().re[..nvir].copy_from_slice(&vector.data().re[..nvir]);
    r1.data_mut().im[..nvir].copy_from_slice(&vector.data().im[..nvir]);

    let mut r2 = ZArr::zeros(&[nkpts, nkpts, nocc, nvir, nvir]);
    let mut cur = nvir;
    for kj in 0..nkpts {
        for ka in 0..nkpts {
            let kb = kconserv.get(kshift, ka, kj) as usize;
            // NumPy puts the ADVANCED-index dimension first here, because the
            // two integer indices and the two arrays are separated by a slice
            // (`:897`'s `reshape(-1, nocc)` is the proof) — so the block is
            // pair-major, occupied-minor.
            for (a, b) in ea_pairs(nvir, ka, kb) {
                for o in 0..nocc {
                    let (re, im) = (vector.data().re[cur], vector.data().im[cur]);
                    cur += 1;
                    let f = (((kj * nkpts + ka) * nocc + o) * nvir + a) * nvir + b;
                    r2.data_mut().re[f] = re;
                    r2.data_mut().im[f] = im;
                    let g = (((kj * nkpts + kb) * nocc + o) * nvir + b) * nvir + a;
                    r2.data_mut().re[g] = -re;
                    r2.data_mut().im[g] = -im;
                }
            }
        }
    }
    Ok((r1, r2))
}

/// `amplitudes_to_vector_ea` (`:865-878`).
///
/// # Errors
/// [`PbcCcError::Shape`] on a shape mismatch.
pub fn amplitudes_to_vector_ea(
    r1: &ZArr,
    r2: &ZArr,
    kshift: usize,
    kconserv: &Kconserv,
) -> Result<ZArr, PbcCcError> {
    let nkpts = r2.shape()[0];
    let nocc = r2.shape()[2];
    let nvir = r2.shape()[3];
    if r1.shape() != [nvir] {
        return Err(PbcCcError::Shape(format!(
            "EA r1 shape {:?}, expected [{nvir}]",
            r1.shape()
        )));
    }
    let n = ea_vector_size(nkpts, nocc, nvir, kshift, kconserv);
    let mut v = ZArr::zeros(&[n]);
    v.data_mut().re[..nvir].copy_from_slice(&r1.data().re);
    v.data_mut().im[..nvir].copy_from_slice(&r1.data().im);
    let mut cur = nvir;
    for kj in 0..nkpts {
        for ka in 0..nkpts {
            let kb = kconserv.get(kshift, ka, kj) as usize;
            for (a, b) in ea_pairs(nvir, ka, kb) {
                for o in 0..nocc {
                    let f = (((kj * nkpts + ka) * nocc + o) * nvir + a) * nvir + b;
                    v.data_mut().re[cur] = r2.data().re[f];
                    v.data_mut().im[cur] = r2.data().im[f];
                    cur += 1;
                }
            }
        }
    }
    Ok(v)
}

/// `eaccsd_matvec` (`:908-946`).
///
/// # Errors
/// Propagates every intermediate access and shape check.
pub fn eaccsd_matvec(
    vector: &ZArr,
    kshift: usize,
    imds: &EomImds<'_>,
    kconserv: &Kconserv,
) -> Result<ZArr, PbcCcError> {
    let (nkpts, nocc, nvir) = (imds.eris.nkpts, imds.eris.nocc, imds.eris.nvir);
    let (r1, r2) = vector_to_amplitudes_ea(vector, kshift, nkpts, nocc, nvir, kconserv)?;
    let wvovv = imds.need(&imds.wvovv, "Wvovv")?;
    let wvvvv = imds.need(&imds.wvvvv, "Wvvvv")?;
    let wvvvo = imds.need(&imds.wvvvo, "Wvvvo")?;

    // `:917-922`
    let mut hr1 = einsum("ac,c->a", &[&imds.fvv.slice_leading(&[kshift])?, &r1])?;
    for kl in 0..nkpts {
        hr1.add_assign(&einsum(
            "ld,lad->a",
            &[
                &imds.fov.slice_leading(&[kl])?,
                &r2.slice_leading(&[kl, kshift])?,
            ],
        )?)?;
        for kc in 0..nkpts {
            hr1.add_assign(&einsum_scaled(
                "alcd,lcd->a",
                &[
                    &wvovv.slice_leading(&[kshift, kl, kc])?,
                    &r2.slice_leading(&[kl, kc])?,
                ],
                0.5,
            )?)?;
        }
    }

    let mut hr2 = ZArr::zeros(&[nkpts, nkpts, nocc, nvir, nvir]);
    // `:925-940`
    for kj in 0..nkpts {
        for ka in 0..nkpts {
            let kb = kconserv.get(kshift, ka, kj) as usize;
            let mut blk = einsum(
                "abcj,c->jab",
                &[&wvvvo.slice_leading(&[ka, kb, kshift])?, &r1],
            )?;
            blk.add_assign(&einsum(
                "ac,jcb->jab",
                &[
                    &imds.fvv.slice_leading(&[ka])?,
                    &r2.slice_leading(&[kj, ka])?,
                ],
            )?)?;
            blk.sub_assign(&einsum(
                "bc,jca->jab",
                &[
                    &imds.fvv.slice_leading(&[kb])?,
                    &r2.slice_leading(&[kj, kb])?,
                ],
            )?)?;
            blk.sub_assign(&einsum(
                "lj,lab->jab",
                &[
                    &imds.foo.slice_leading(&[kj])?,
                    &r2.slice_leading(&[kj, ka])?,
                ],
            )?)?;
            for kd in 0..nkpts {
                let kl = kconserv.get(kj, kb, kd) as usize;
                blk.add_assign(&einsum(
                    "lbdj,lad->jab",
                    &[
                        &imds.wovvo.slice_leading(&[kl, kb, kd])?,
                        &r2.slice_leading(&[kl, ka])?,
                    ],
                )?)?;
                // P(ab)
                let kl = kconserv.get(kj, ka, kd) as usize;
                blk.sub_assign(&einsum(
                    "ladj,lbd->jab",
                    &[
                        &imds.wovvo.slice_leading(&[kl, ka, kd])?,
                        &r2.slice_leading(&[kl, kb])?,
                    ],
                )?)?;
                let kc = kconserv.get(ka, kd, kb) as usize;
                blk.add_assign(&einsum_scaled(
                    "abcd,jcd->jab",
                    &[
                        &wvvvv.slice_leading(&[ka, kb, kc])?,
                        &r2.slice_leading(&[kj, kc])?,
                    ],
                    0.5,
                )?)?;
            }
            hr2.set_leading(&[kj, ka], &blk)?;
        }
    }

    // `:942-943`
    let mut tmp = ZArr::zeros(&[nocc]);
    for kx in 0..nkpts {
        for ky in 0..nkpts {
            tmp.add_assign(&einsum(
                "klcd,lcd->k",
                &[
                    &imds.eris.blk(GBlk::Oovv, kshift, kx, ky)?,
                    &r2.slice_leading(&[kx, ky])?,
                ],
            )?)?;
        }
    }
    for kx in 0..nkpts {
        for ky in 0..nkpts {
            let mut blk = hr2.slice_leading(&[kx, ky])?;
            blk.sub_assign(&einsum_scaled(
                "k,kjab->jab",
                &[&tmp, &imds.t2.slice_leading(&[kshift, kx, ky])?],
                0.5,
            )?)?;
            hr2.set_leading(&[kx, ky], &blk)?;
        }
    }

    amplitudes_to_vector_ea(&hr1, &hr2, kshift, kconserv)
}

/// `leaccsd_matvec` (`:948-1000`) — the LEFT eigenvector matvec.
///
/// # Errors
/// As [`eaccsd_matvec`].
pub fn leaccsd_matvec(
    vector: &ZArr,
    kshift: usize,
    imds: &EomImds<'_>,
    kconserv: &Kconserv,
) -> Result<ZArr, PbcCcError> {
    let (nkpts, nocc, nvir) = (imds.eris.nkpts, imds.eris.nocc, imds.eris.nvir);
    let (r1, r2) = vector_to_amplitudes_ea(vector, kshift, nkpts, nocc, nvir, kconserv)?;
    let wvovv = imds.need(&imds.wvovv, "Wvovv")?;
    let wvvvv = imds.need(&imds.wvvvv, "Wvvvv")?;
    let wvvvo = imds.need(&imds.wvvvo, "Wvvvo")?;

    // `:958-961` — note `'ca,c->a'`, the TRANSPOSED `Fvv` contraction.
    let mut hr1 = einsum("ca,c->a", &[&imds.fvv.slice_leading(&[kshift])?, &r1])?;
    for kj in 0..nkpts {
        for kb in 0..nkpts {
            let kc = kconserv.get(kshift, kb, kj) as usize;
            hr1.add_assign(&einsum_scaled(
                "cbaj,jcb->a",
                &[
                    &wvvvo.slice_leading(&[kc, kb, kshift])?,
                    &r2.slice_leading(&[kj, kc])?,
                ],
                0.5,
            )?)?;
        }
    }

    let mut hr2 = ZArr::zeros(&[nkpts, nkpts, nocc, nvir, nvir]);
    // `:964-968` — the two `(kj==kb)` / `(kj==ka)` guards are momentum
    // conditions written as multiplications by a boolean upstream.
    for kj in 0..nkpts {
        for ka in 0..nkpts {
            let kb = kconserv.get(kshift, ka, kj) as usize;
            let mut blk = hr2.slice_leading(&[kj, ka])?;
            blk.add_assign(&einsum(
                "cjab,c->jab",
                &[&wvovv.slice_leading(&[kshift, kj, ka])?, &r1],
            )?)?;
            hr2.set_leading(&[kj, ka], &blk)?;

            if kj == kb {
                let v = einsum("jb,a->jab", &[&imds.fov.slice_leading(&[kj])?, &r1])?;
                let mut b = hr2.slice_leading(&[kj, kshift])?;
                b.add_assign(&v)?;
                hr2.set_leading(&[kj, kshift], &b)?;
            }
            if kj == ka {
                let v = einsum("ja,b->jab", &[&imds.fov.slice_leading(&[kj])?, &r1])?;
                let mut b = hr2.slice_leading(&[kj, ka])?;
                b.sub_assign(&v)?;
                hr2.set_leading(&[kj, ka], &b)?;
            }
        }
    }

    // `:970-985`
    for kj in 0..nkpts {
        for ka in 0..nkpts {
            let kb = kconserv.get(kshift, ka, kj) as usize;
            let mut blk = hr2.slice_leading(&[kj, ka])?;
            blk.add_assign(&einsum(
                "ca,jcb->jab",
                &[
                    &imds.fvv.slice_leading(&[ka])?,
                    &r2.slice_leading(&[kj, ka])?,
                ],
            )?)?;
            blk.sub_assign(&einsum(
                "cb,jca->jab",
                &[
                    &imds.fvv.slice_leading(&[kb])?,
                    &r2.slice_leading(&[kj, kb])?,
                ],
            )?)?;
            blk.sub_assign(&einsum(
                "jl,lab->jab",
                &[
                    &imds.foo.slice_leading(&[kj])?,
                    &r2.slice_leading(&[kj, ka])?,
                ],
            )?)?;
            for kd in 0..nkpts {
                let km = kconserv.get(kj, kb, kd) as usize;
                blk.add_assign(&einsum(
                    "jdbm,mad->jab",
                    &[
                        &imds.wovvo.slice_leading(&[kj, kd, kb])?,
                        &r2.slice_leading(&[km, ka])?,
                    ],
                )?)?;
                let km = kconserv.get(kj, ka, kd) as usize;
                blk.sub_assign(&einsum(
                    "jdam,mbd->jab",
                    &[
                        &imds.wovvo.slice_leading(&[kj, kd, ka])?,
                        &r2.slice_leading(&[km, kb])?,
                    ],
                )?)?;
                let kc = kconserv.get(ka, kd, kb) as usize;
                blk.add_assign(&einsum_scaled(
                    "cdab,jcd->jab",
                    &[
                        &wvvvv.slice_leading(&[kc, kd, ka])?,
                        &r2.slice_leading(&[kj, kc])?,
                    ],
                    0.5,
                )?)?;
            }
            hr2.set_leading(&[kj, ka], &blk)?;
        }
    }

    // `:987-996`
    let mut tmp = ZArr::zeros(&[nocc]);
    for kj in 0..nkpts {
        for ka in 0..nkpts {
            tmp.add_assign(&einsum(
                "jab,kjab->k",
                &[
                    &r2.slice_leading(&[kj, ka])?,
                    &imds.t2.slice_leading(&[kshift, kj, ka])?,
                ],
            )?)?;
        }
    }
    for kj in 0..nkpts {
        for ka in 0..nkpts {
            let mut blk = hr2.slice_leading(&[kj, ka])?;
            blk.sub_assign(&einsum_scaled(
                "kjab,k->jab",
                &[&imds.eris.blk(GBlk::Oovv, kshift, kj, ka)?, &tmp],
                0.5,
            )?)?;
            hr2.set_leading(&[kj, ka], &blk)?;
        }
    }

    amplitudes_to_vector_ea(&hr1, &hr2, kshift, kconserv)
}

/// `eaccsd_diag` (`:1002-1036`), the `partition = None` branch — see
/// [`ipccsd_diag`] on the `'mp'` branch.
///
/// # Errors
/// As [`eaccsd_matvec`].
pub fn eaccsd_diag(
    kshift: usize,
    imds: &EomImds<'_>,
    kconserv: &Kconserv,
) -> Result<ZArr, PbcCcError> {
    let (nkpts, nocc, nvir) = (imds.eris.nkpts, imds.eris.nocc, imds.eris.nvir);
    let wvvvv = imds.need(&imds.wvvvv, "Wvvvv")?;

    let mut hr1 = ZArr::zeros(&[nvir]);
    let fvv_s = imds.fvv.slice_leading(&[kshift])?;
    for a in 0..nvir {
        let (re, im) = fvv_s.at(&[a, a])?;
        hr1.data_mut().re[a] = re;
        hr1.data_mut().im[a] = im;
    }

    let mut hr2 = ZArr::zeros(&[nkpts, nkpts, nocc, nvir, nvir]);
    for kj in 0..nkpts {
        for ka in 0..nkpts {
            let kb = kconserv.get(kshift, ka, kj) as usize;
            let mut blk = ZArr::zeros(&[nocc, nvir, nvir]);
            let fj = imds.foo.slice_leading(&[kj])?;
            let fa = imds.fvv.slice_leading(&[ka])?;
            let fb = imds.fvv.slice_leading(&[kb])?;
            for j in 0..nocc {
                for a in 0..nvir {
                    for b in 0..nvir {
                        let f = (j * nvir + a) * nvir + b;
                        let (r, m) = fj.at(&[j, j])?;
                        blk.data_mut().re[f] -= r;
                        blk.data_mut().im[f] -= m;
                        let (r, m) = fa.at(&[a, a])?;
                        blk.data_mut().re[f] += r;
                        blk.data_mut().im[f] += m;
                        let (r, m) = fb.at(&[b, b])?;
                        blk.data_mut().re[f] += r;
                        blk.data_mut().im[f] += m;
                    }
                }
            }
            // `:1027-1028` — `einsum('jbbj->jb', ...)` / `('jaaj->ja', ...)`,
            // diagonals, written as index loops.
            let wb = imds.wovvo.slice_leading(&[kj, kb, kb])?;
            let wa = imds.wovvo.slice_leading(&[kj, ka, ka])?;
            for j in 0..nocc {
                for b in 0..nvir {
                    let (r, m) = wb.at(&[j, b, b, j])?;
                    for a in 0..nvir {
                        let f = (j * nvir + a) * nvir + b;
                        blk.data_mut().re[f] += r;
                        blk.data_mut().im[f] += m;
                    }
                }
                for a in 0..nvir {
                    let (r, m) = wa.at(&[j, a, a, j])?;
                    for b in 0..nvir {
                        let f = (j * nvir + a) * nvir + b;
                        blk.data_mut().re[f] += r;
                        blk.data_mut().im[f] += m;
                    }
                }
            }
            // `:1030-1031` — `einsum('abab->ab', Wvvvv[ka,kb,ka])`.
            if ka == kconserv.get(ka, kb, kb) as usize {
                let w = wvvvv.slice_leading(&[ka, kb, ka])?;
                for a in 0..nvir {
                    for b in 0..nvir {
                        let (r, m) = w.at(&[a, b, a, b])?;
                        for j in 0..nocc {
                            let f = (j * nvir + a) * nvir + b;
                            blk.data_mut().re[f] += r;
                            blk.data_mut().im[f] += m;
                        }
                    }
                }
            }
            // `:1033`
            blk.sub_assign(&einsum(
                "kjab,kjab->jab",
                &[
                    &imds.eris.blk(GBlk::Oovv, kshift, kj, ka)?,
                    &imds.t2.slice_leading(&[kshift, kj, ka])?,
                ],
            )?)?;
            hr2.set_leading(&[kj, ka], &blk)?;
        }
    }
    amplitudes_to_vector_ea(&hr1, &hr2, kshift, kconserv)
}

/// `mask_frozen_ea` (`:1180-1199`).
///
/// # Errors
/// As [`eaccsd_matvec`].
#[allow(clippy::too_many_arguments)]
pub fn mask_frozen_ea(
    vector: &ZArr,
    kshift: usize,
    nkpts: usize,
    nocc: usize,
    nvir: usize,
    nonzero_opadding: &[Vec<usize>],
    nonzero_vpadding: &[Vec<usize>],
    kconserv: &Kconserv,
    konst: f64,
) -> Result<ZArr, PbcCcError> {
    let (r1, r2) = vector_to_amplitudes_ea(vector, kshift, nkpts, nocc, nvir, kconserv)?;
    let mut new_r1 = ZArr::zeros(&[nvir]);
    for v in new_r1.data_mut().re.iter_mut() {
        *v = konst;
    }
    for &a in &nonzero_vpadding[kshift] {
        if a < nvir {
            new_r1.data_mut().re[a] = r1.data().re[a];
            new_r1.data_mut().im[a] = r1.data().im[a];
        }
    }

    let mut new_r2 = ZArr::zeros(&[nkpts, nkpts, nocc, nvir, nvir]);
    for v in new_r2.data_mut().re.iter_mut() {
        *v = konst;
    }
    for kj in 0..nkpts {
        for ka in 0..nkpts {
            let kb = kconserv.get(kshift, ka, kj) as usize;
            for &j in &nonzero_opadding[kj] {
                for &a in &nonzero_vpadding[ka] {
                    for &b in &nonzero_vpadding[kb] {
                        if j >= nocc || a >= nvir || b >= nvir {
                            continue;
                        }
                        let f = (((kj * nkpts + ka) * nocc + j) * nvir + a) * nvir + b;
                        new_r2.data_mut().re[f] = r2.data().re[f];
                        new_r2.data_mut().im[f] = r2.data().im[f];
                    }
                }
            }
        }
    }
    amplitudes_to_vector_ea(&new_r1, &new_r2, kshift, kconserv)
}

// ---------------------------------------------------------------------------
// The Davidson driver (`:40-159`) and the two excitation surfaces
// ---------------------------------------------------------------------------

/// Which excitation the driver is solving for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Excitation {
    /// Ionisation potential — `EOMIP` (`:684`).
    Ip,
    /// Electron affinity — `EOMEA` (`:1201`).
    Ea,
    /// Electronic excitation — `EOMEE` (`:1691`), driven by `kernel_ee`
    /// (`:1288`) rather than `kernel`.
    Ee,
}

/// `EOMIP` / `EOMEA`'s solver knobs (`eom_rccsd.EOM`'s defaults, which
/// `eom_kccsd_ghf` does not override).
#[derive(Debug, Clone, Copy)]
pub struct EomOpts {
    /// `eom.conv_tol`.
    pub conv_tol: f64,
    /// `eom.max_cycle`.
    pub max_cycle: usize,
    /// `eom.max_space`. **A real memory knob, not a constant to inline**:
    /// the Davidson subspace is `2·max_space·nroots` vectors of the full
    /// amplitude length, which `16-REVIEW.md §7.2` sized at 16 MiB on
    /// `gth-szv` 2×2×2 but **5.4 GiB** for EA on `gth-dzvp` 3×3×3.
    pub max_space: usize,
    /// Roots per k-point.
    pub nroots: usize,
    /// `koopmans=True` targets quasiparticle states by overlap with the guess
    /// rather than by lowest eigenvalue (`:129-136`).
    pub koopmans: bool,
    /// Solve for LEFT eigenvectors.
    pub left: bool,
    /// `eom.partition`. Anything but [`Partition::None`] is REFUSED, which is
    /// upstream's own behaviour at `ipccsd`/`eaccsd` — see [`Partition`].
    pub partition: Partition,
}

impl Default for EomOpts {
    fn default() -> Self {
        Self {
            conv_tol: 1e-7,
            max_cycle: 50,
            max_space: 20,
            nroots: 1,
            koopmans: false,
            left: false,
            partition: Partition::None,
        }
    }
}

/// One k-shift's roots.
#[derive(Debug, Clone)]
pub struct EomRoots {
    /// The k-point index this block was solved at.
    pub kshift: usize,
    /// Per-root convergence.
    pub conv: Vec<bool>,
    /// The excitation energies.
    pub e: Vec<f64>,
    /// The eigenvectors, in the packed vector layout.
    pub v: Vec<ZArr>,
    /// `‖r1‖²`, upstream's `qp_weight` (`:154`) — the quasiparticle weight,
    /// printed per root and worth returning because it is how a caller tells a
    /// physical ionisation from a satellite.
    pub qp_weight: Vec<f64>,
}

/// The padded-orbital index sets a masking call needs, as
/// `get_padding_k_idx` (`:217-222`) returns them.
pub struct Padding {
    pub occupied: Vec<Vec<usize>>,
    pub virtuals: Vec<Vec<usize>>,
}

/// `get_padding_k_idx(eom, cc)` (`:217-222`) — `padding_k_idx(cc, "split")`.
///
/// # Errors
/// Propagates the padding surface.
pub fn padding_from(padded: &pyscf_pbc_mp::PaddedMos) -> Result<Padding, PbcCcError> {
    let (occupied, virtuals) = crate::kccsd_rhf::split_padding(padded)?;
    Ok(Padding { occupied, virtuals })
}

/// `kernel(eom, ...)` (`:40-159`) for one `kshift`.
///
/// # The guess, the mask and `LARGE_DENOM` are one mechanism
///
/// `:106` masks the DIAGONAL with [`crate::kccsd_rhf::LARGE_DENOM`] and
/// `:713`/`:717` masks each guess vector with `0.0`. Padded orbitals are
/// therefore pushed far from every root but still present in the vector — the
/// same "arithmetic, never a skip" rule the amplitude denominators follow
/// (`16-CONTEXT §3.3`). Dropping them instead would change the vector length
/// and hence the subspace.
///
/// # Errors
/// Propagates the matvec and the Davidson solve.
#[allow(clippy::too_many_arguments)]
pub fn eom_kernel(
    kind: Excitation,
    kshift: usize,
    imds: &EomImds<'_>,
    padding: &Padding,
    kconserv: &Kconserv,
    opts: &EomOpts,
) -> Result<EomRoots, PbcCcError> {
    // `ipccsd` (`:615-618`) and `eaccsd` (`:905`) refuse BOTH partitions at the
    // entry point, verified by running them. The branches live on inside the
    // matvecs and diagonals and are exposed here as `*_mp` functions.
    if opts.partition != Partition::None {
        return Err(partition_refusal());
    }
    let (nkpts, nocc, nvir) = (imds.eris.nkpts, imds.eris.nocc, imds.eris.nvir);
    let size = match kind {
        Excitation::Ip => ip_vector_size(nkpts, nocc, nvir),
        Excitation::Ea => ea_vector_size(nkpts, nocc, nvir, kshift, kconserv),
        Excitation::Ee => ee_vector_size(nkpts, nocc, nvir, kshift, kconserv),
    };
    if kind == Excitation::Ee && opts.koopmans {
        // `:1745-1749` — `EOMEE.get_init_guess` raises for `koopmans`, with a
        // `# TODO do Koopmans later`. Reproduced rather than invented.
        return Err(PbcCcError::NotImplementedUpstream {
            upstream: "pbc/cc/eom_kccsd_ghf.py:1749",
            what: "EOMEE.get_init_guess raises NotImplementedError for koopmans=True",
        });
    }

    let mask = |v: &ZArr, konst: f64| -> Result<ZArr, PbcCcError> {
        match kind {
            Excitation::Ip => mask_frozen_ip(
                v,
                kshift,
                nkpts,
                nocc,
                nvir,
                &padding.occupied,
                &padding.virtuals,
                kconserv,
                konst,
            ),
            Excitation::Ea => mask_frozen_ea(
                v,
                kshift,
                nkpts,
                nocc,
                nvir,
                &padding.occupied,
                &padding.virtuals,
                kconserv,
                konst,
            ),
            // `kernel_ee` (`:1288-1296`) has NO masking — its docstring says
            // it is "a simplified version of kernel() with a few parts
            // removed, such as those involving eom.mask_frozen()", with a
            // `# TODO mask frozen-orbital indices` at `:1313`. Masking here
            // anyway would be a silent divergence in the guess AND in the
            // diagonal, so the identity is what upstream does.
            Excitation::Ee => Ok(v.clone()),
        }
    };

    // `:85-91` — the root count is capped by the number of UNFROZEN entries,
    // found by masking a zero vector with `const = 1` and counting.
    let ones = mask(&ZArr::zeros(&[size]), 1.0)?;
    let nfrozen = ones.data().re.iter().filter(|v| **v == 1.0).count();
    let nroots = opts.nroots.min(size).min(size - nfrozen).max(1);

    // `:105-106` — the diagonal, masked with LARGE_DENOM.
    let diag = match kind {
        Excitation::Ip => ipccsd_diag(kshift, imds, kconserv)?,
        Excitation::Ea => eaccsd_diag(kshift, imds, kconserv)?,
        Excitation::Ee => eeccsd_diag(kshift, imds, kconserv)?,
    };
    let diag = mask(&diag, crate::kccsd_rhf::LARGE_DENOM)?;

    // `:108-121` / `:705-724` — the initial guess.
    let guess = init_guess(kind, kshift, &diag, nroots, opts.koopmans, padding, &mask)?;

    let aop = |xs: &[CTensor]| -> Vec<CTensor> {
        xs.iter()
            .map(|x| {
                let v = ZArr::from_ctensor(&[size], x.clone()).expect("guess shape");
                let out = match (kind, opts.left) {
                    (Excitation::Ip, false) => ipccsd_matvec(&v, kshift, imds, kconserv),
                    (Excitation::Ip, true) => lipccsd_matvec(&v, kshift, imds, kconserv),
                    (Excitation::Ea, false) => eaccsd_matvec(&v, kshift, imds, kconserv),
                    (Excitation::Ea, true) => leaccsd_matvec(&v, kshift, imds, kconserv),
                    // `EOMEE` declares no `l_matvec` (`:1701-1704`), so there
                    // is no left EE to port.
                    (Excitation::Ee, _) => eeccsd_matvec(&v, kshift, imds, kconserv),
                };
                out.expect("EOM matvec").into_ctensor()
            })
            .collect()
    };

    // `:126-127` — `precond(r, e0, x0) = r / (e0 - diag + 1e-12)`.
    let dre: Vec<f64> = diag.data().re.clone();
    let precond = |r: &CTensor, e0: f64, _x0: &CTensor| -> CTensor {
        let mut out = r.clone();
        for i in 0..out.re.len() {
            let d = e0 - dre[i] + 1e-12;
            out.re[i] /= d;
            out.im[i] /= d;
        }
        out
    };

    let dopts = pyscf_algebra::DavidsonOptions {
        tol: opts.conv_tol,
        tol_residual: None,
        max_cycle: opts.max_cycle,
        max_space: opts.max_space,
        nroots,
        left: false,
        real_dtype: false,
        ..Default::default()
    };
    let res = pyscf_algebra::davidson_nosym1(
        aop,
        guess.iter().map(|g| g.data().clone()).collect(),
        precond,
        &dopts,
        pyscf_algebra::pick_real_eigs,
    )
    .map_err(|e| PbcCcError::Algebra(format!("EOM Davidson: {e}")))?;

    // `:151-157` — the quasiparticle weight is `‖r1‖²`.
    let mut v = Vec::with_capacity(res.x.len());
    let mut qp_weight = Vec::with_capacity(res.x.len());
    for x in &res.x {
        let vec = ZArr::from_ctensor(&[size], x.clone())?;
        let n1 = match kind {
            Excitation::Ip => nocc,
            Excitation::Ea => nvir,
            Excitation::Ee => nkpts * nocc * nvir,
        };
        let w: f64 = (0..n1)
            .map(|i| vec.data().re[i] * vec.data().re[i] + vec.data().im[i] * vec.data().im[i])
            .sum();
        qp_weight.push(w);
        v.push(vec);
    }

    Ok(EomRoots {
        kshift,
        conv: res.conv,
        e: res.e,
        v,
        qp_weight,
    })
}

/// `EOMIP.get_init_guess` (`:705-724`) and its EA sibling (`:905-...`).
///
/// The `koopmans` branch seeds unit vectors on the NON-PADDED orbitals — the
/// occupied ones in REVERSE order for IP (highest occupied first), the virtual
/// ones in forward order for EA — and masks each with `const = 0.0`. The other
/// branch takes the `nroots` smallest diagonal entries.
fn init_guess(
    kind: Excitation,
    kshift: usize,
    diag: &ZArr,
    nroots: usize,
    koopmans: bool,
    padding: &Padding,
    mask: &dyn Fn(&ZArr, f64) -> Result<ZArr, PbcCcError>,
) -> Result<Vec<ZArr>, PbcCcError> {
    let size = diag.len();
    let mut guess = Vec::with_capacity(nroots);
    if koopmans {
        let seeds: Vec<usize> = match kind {
            Excitation::Ip => padding.occupied[kshift].iter().rev().copied().collect(),
            Excitation::Ea => padding.virtuals[kshift].to_vec(),
            // Unreachable: `eom_kernel` refuses `koopmans` for EE above.
            Excitation::Ee => Vec::new(),
        };
        for &n in seeds.iter().take(nroots) {
            let mut g = ZArr::zeros(&[size]);
            if n < size {
                g.data_mut().re[n] = 1.0;
            }
            guess.push(mask(&g, 0.0)?);
        }
    } else {
        // `diag.argsort()[:nroots]`, a STABLE sort on the real part — the
        // diagonal is real by construction (it is a sum of Fock diagonals and
        // Hermitian-block diagonals) and upstream sorts the complex array,
        // which numpy orders by real part then imaginary.
        let mut idx: Vec<usize> = (0..size).collect();
        idx.sort_by(|a, b| {
            diag.data().re[*a]
                .partial_cmp(&diag.data().re[*b])
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| {
                    diag.data().im[*a]
                        .partial_cmp(&diag.data().im[*b])
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
        });
        for &i in idx.iter().take(nroots) {
            let mut g = ZArr::zeros(&[size]);
            g.data_mut().re[i] = 1.0;
            guess.push(mask(&g, 0.0)?);
        }
    }
    if guess.is_empty() {
        return Err(PbcCcError::Shape(
            "EOM produced no initial guess vectors".into(),
        ));
    }
    Ok(guess)
}

// ---------------------------------------------------------------------------
// EE (`:1288-1839`)
// ---------------------------------------------------------------------------

/// `get_kconserv_ee_r1(kshift)` (`:1788-1801`) — `kconserv[:, kshift, 0]`.
///
/// `kconserv_r1[m] = n` with `k_m − k_n − k_shift = G`.
pub fn kconserv_ee_r1(nkpts: usize, kshift: usize, kconserv: &Kconserv) -> Vec<usize> {
    (0..nkpts)
        .map(|m| kconserv.get(m, kshift, 0) as usize)
        .collect()
}

/// `get_kconserv_ee_r2(kshift)` (`:1805-1834`) — `kconserv_r2[k,l,m] = n` with
/// `k_k − k_l + k_m − k_n − k_shift = G`.
///
/// Upstream rebuilds this geometrically from the k-point coordinates. Here it
/// is composed from the ordinary `kconserv`: `t = kconserv[k,l,m]` gives
/// `k_t ≡ k_k − k_l + k_m`, and `kconserv[t, kshift, 0]` then shifts by
/// `−k_shift + k_0`. **That is only the same array when `k_0 = 0`** — which is
/// upstream's own assumption, since `get_kconserv_ee_r1` is literally
/// `kconserv[:, kshift, 0]`. The composition is not left as an assumption:
/// `kgccsd_eom_ee_matches_upstream` compares the whole array against
/// upstream's geometric construction before using it.
pub fn kconserv_ee_r2(nkpts: usize, kshift: usize, kconserv: &Kconserv) -> Vec<usize> {
    let mut out = vec![0_usize; nkpts * nkpts * nkpts];
    for k in 0..nkpts {
        for l in 0..nkpts {
            for m in 0..nkpts {
                let t = kconserv.get(k, l, m) as usize;
                out[(k * nkpts + l) * nkpts + m] = kconserv.get(t, kshift, 0) as usize;
            }
        }
    }
    out
}

fn r2k(kr2: &[usize], nkpts: usize, k: usize, l: usize, m: usize) -> usize {
    kr2[(k * nkpts + l) * nkpts + m]
}

/// `EOMEE.vector_size(kshift)` (`:1707-1745`), computed by WALKING the packing
/// rather than by upstream's closed form.
///
/// Upstream has two branches: an odd-`nkpts` closed form and, for even `nkpts`,
/// the same triple loop this function runs — and its own docstring says "the
/// vector size is kshift-dependent if nkpts is an even number". Walking the
/// packing is right in both cases by construction, and the test asserts it
/// equals upstream's number for every shift.
pub fn ee_vector_size(
    nkpts: usize,
    nocc: usize,
    nvir: usize,
    kshift: usize,
    kconserv: &Kconserv,
) -> usize {
    let kr2 = kconserv_ee_r2(nkpts, kshift, kconserv);
    let n = nkpts * nocc;
    let mut size = nkpts * nocc * nvir;
    for p in 0..n {
        for q in 0..p {
            let (ki, kj) = (p / nocc, q / nocc);
            for ka in 0..nkpts {
                let kb = r2k(&kr2, nkpts, ki, ka, kj);
                if ka == kb {
                    size += nvir * (nvir - 1) / 2;
                } else if ka > kb {
                    size += nvir * nvir;
                }
            }
        }
    }
    size
}

/// `vector_to_amplitudes_ee` (`:1602-1652`). Returns `(r1, r2)` with `r1`
/// shaped `[nkpts, nocc, nvir]` and `r2` shaped
/// `[nkpts, nkpts, nkpts, nocc, nocc, nvir, nvir]`, indexed `[ki, kj, ka]`.
///
/// # Errors
/// [`PbcCcError::Shape`] if the vector length disagrees with the packing.
pub fn vector_to_amplitudes_ee(
    vector: &ZArr,
    kshift: usize,
    nkpts: usize,
    nocc: usize,
    nvir: usize,
    kconserv: &Kconserv,
) -> Result<(ZArr, ZArr), PbcCcError> {
    let want = ee_vector_size(nkpts, nocc, nvir, kshift, kconserv);
    if vector.len() != want {
        return Err(PbcCcError::Shape(format!(
            "EE vector of {} elements, expected {want}",
            vector.len()
        )));
    }
    let kr2 = kconserv_ee_r2(nkpts, kshift, kconserv);
    let n1 = nkpts * nocc * nvir;
    let mut r1 = ZArr::zeros(&[nkpts, nocc, nvir]);
    r1.data_mut().re.copy_from_slice(&vector.data().re[..n1]);
    r1.data_mut().im.copy_from_slice(&vector.data().im[..n1]);

    let mut r2 = ZArr::zeros(&[nkpts, nkpts, nkpts, nocc, nocc, nvir, nvir]);
    let n = nkpts * nocc;
    let mut cur = n1;
    // `r2` is assembled in the COMPOSITE layout `[(ki,i), (kj,j), ka, a, b]`
    // and only reshaped at the end (`:1651`); the loop below writes straight to
    // the final `[ki, kj, ka, i, j, a, b]` addresses instead.
    let put = |r2: &mut ZArr, ki, kj, ka, i, j, a, b, re: f64, im: f64| {
        let f =
            ((((((ki * nkpts + kj) * nkpts + ka) * nocc + i) * nocc + j) * nvir + a) * nvir) + b;
        r2.data_mut().re[f] = re;
        r2.data_mut().im[f] = im;
    };
    for p in 0..n {
        for q in 0..p {
            let (ki, i) = (p / nocc, p % nocc);
            let (kj, j) = (q / nocc, q % nocc);
            for ka in 0..nkpts {
                let kb = r2k(&kr2, nkpts, ki, ka, kj);
                if ka == kb {
                    for a in 0..nvir {
                        for b in 0..a {
                            let (re, im) = (vector.data().re[cur], vector.data().im[cur]);
                            cur += 1;
                            // `:1465-1466` — antisymmetric in (a,b) …
                            put(&mut r2, ki, kj, ka, i, j, a, b, re, im);
                            put(&mut r2, ki, kj, ka, i, j, b, a, -re, -im);
                            // … and in the OCCUPIED pair, with a/b unswapped
                            // (`:1476`, `r2[Q,P] = -r2_ka_ab`).
                            put(&mut r2, kj, ki, ka, j, i, a, b, -re, -im);
                            put(&mut r2, kj, ki, ka, j, i, b, a, re, im);
                        }
                    }
                } else if ka > kb {
                    for a in 0..nvir {
                        for b in 0..nvir {
                            let (re, im) = (vector.data().re[cur], vector.data().im[cur]);
                            cur += 1;
                            put(&mut r2, ki, kj, ka, i, j, a, b, re, im);
                            // `:1471` — `r2_ka_ab[kb] = -tmp.transpose()`.
                            put(&mut r2, ki, kj, kb, i, j, b, a, -re, -im);
                            put(&mut r2, kj, ki, ka, j, i, a, b, -re, -im);
                            put(&mut r2, kj, ki, kb, j, i, b, a, re, im);
                        }
                    }
                }
            }
        }
    }
    Ok((r1, r2))
}

/// `amplitudes_to_vector_ee` (`:1655-1689`).
///
/// # Errors
/// [`PbcCcError::Shape`] on a shape mismatch.
pub fn amplitudes_to_vector_ee(
    r1: &ZArr,
    r2: &ZArr,
    kshift: usize,
    kconserv: &Kconserv,
) -> Result<ZArr, PbcCcError> {
    let nkpts = r1.shape()[0];
    let nocc = r1.shape()[1];
    let nvir = r1.shape()[2];
    let kr2 = kconserv_ee_r2(nkpts, kshift, kconserv);
    let size = ee_vector_size(nkpts, nocc, nvir, kshift, kconserv);
    let mut v = ZArr::zeros(&[size]);
    let n1 = nkpts * nocc * nvir;
    v.data_mut().re[..n1].copy_from_slice(&r1.data().re);
    v.data_mut().im[..n1].copy_from_slice(&r1.data().im);

    let at = |ki: usize, kj: usize, ka: usize, i, j, a, b| {
        ((((((ki * nkpts + kj) * nkpts + ka) * nocc + i) * nocc + j) * nvir + a) * nvir) + b
    };
    let n = nkpts * nocc;
    let mut cur = n1;
    for p in 0..n {
        for q in 0..p {
            let (ki, i) = (p / nocc, p % nocc);
            let (kj, j) = (q / nocc, q % nocc);
            for ka in 0..nkpts {
                let kb = r2k(&kr2, nkpts, ki, ka, kj);
                if ka == kb {
                    for a in 0..nvir {
                        for b in 0..a {
                            let f = at(ki, kj, ka, i, j, a, b);
                            v.data_mut().re[cur] = r2.data().re[f];
                            v.data_mut().im[cur] = r2.data().im[f];
                            cur += 1;
                        }
                    }
                } else if ka > kb {
                    for a in 0..nvir {
                        for b in 0..nvir {
                            let f = at(ki, kj, ka, i, j, a, b);
                            v.data_mut().re[cur] = r2.data().re[f];
                            v.data_mut().im[cur] = r2.data().im[f];
                            cur += 1;
                        }
                    }
                }
            }
        }
    }
    Ok(v)
}

/// `eeccsd_matvec` (`:1397-1546`).
///
/// # Errors
/// Propagates every intermediate access and shape check.
#[allow(clippy::too_many_lines)]
pub fn eeccsd_matvec(
    vector: &ZArr,
    kshift: usize,
    imds: &EomImds<'_>,
    kconserv: &Kconserv,
) -> Result<ZArr, PbcCcError> {
    let (nkpts, nocc, nvir) = (imds.eris.nkpts, imds.eris.nocc, imds.eris.nvir);
    let kr1 = kconserv_ee_r1(nkpts, kshift, kconserv);
    let kr2v = kconserv_ee_r2(nkpts, kshift, kconserv);
    let kr2 = |k: usize, l: usize, m: usize| r2k(&kr2v, nkpts, k, l, m);
    let (r1, r2) = vector_to_amplitudes_ee(vector, kshift, nkpts, nocc, nvir, kconserv)?;
    let woooo = imds.need(&imds.woooo, "Woooo")?;
    let wooov = imds.need(&imds.wooov, "Wooov")?;
    let wovoo = imds.need(&imds.wovoo, "Wovoo")?;
    let wvovv = imds.need(&imds.wvovv, "Wvovv")?;
    let wvvvv = imds.need(&imds.wvvvv, "Wvvvv")?;
    let wvvvo = imds.need(&imds.wvvvo, "Wvvvo")?;

    // `:1412-1424`
    let mut hr1 = ZArr::zeros(&[nkpts, nocc, nvir]);
    for ki in 0..nkpts {
        let ka = kr1[ki];
        let mut blk = einsum(
            "ae,ie->ia",
            &[&imds.fvv.slice_leading(&[ka])?, &r1.slice_leading(&[ki])?],
        )?;
        blk.sub_assign(&einsum(
            "mi,ma->ia",
            &[&imds.foo.slice_leading(&[ki])?, &r1.slice_leading(&[ki])?],
        )?)?;
        for km in 0..nkpts {
            blk.add_assign(&einsum(
                "me,imae->ia",
                &[
                    &imds.fov.slice_leading(&[km])?,
                    &r2.slice_leading(&[ki, km, ka])?,
                ],
            )?)?;
            let ke = kr1[km];
            blk.add_assign(&einsum(
                "maei,me->ia",
                &[
                    &imds.wovvo.slice_leading(&[km, ka, ke])?,
                    &r1.slice_leading(&[km])?,
                ],
            )?)?;
            for kn in 0..nkpts {
                blk.sub_assign(&einsum_scaled(
                    "mnie,mnae->ia",
                    &[
                        &wooov.slice_leading(&[km, kn, ki])?,
                        &r2.slice_leading(&[km, kn, ka])?,
                    ],
                    0.5,
                )?)?;
                // `:1424` — upstream renames the dummy `kn -> ke` here; the
                // index is the same loop variable.
                blk.add_assign(&einsum_scaled(
                    "amef,imef->ia",
                    &[
                        &wvovv.slice_leading(&[ka, km, kn])?,
                        &r2.slice_leading(&[ki, km, kn])?,
                    ],
                    0.5,
                )?)?;
            }
        }
        hr1.set_leading(&[ki], &blk)?;
    }

    let mut hr2 = ZArr::zeros(&[nkpts, nkpts, nkpts, nocc, nocc, nvir, nvir]);
    let add = |h: &mut ZArr, k: [usize; 3], v: &ZArr, s: f64| -> Result<(), PbcCcError> {
        let mut cur = h.slice_leading(&k)?;
        cur.zip_assign(v, s)?;
        h.set_leading(&k, &cur)
    };

    // `:1427-1470`
    for ki in 0..nkpts {
        for kj in 0..nkpts {
            for ka in 0..nkpts {
                let kb = kr2(ki, ka, kj);
                // P(ij)
                let mut tmp_ij = einsum(
                    "mj,imab->ijab",
                    &[
                        &imds.foo.slice_leading(&[kj])?,
                        &r2.slice_leading(&[ki, kj, ka])?,
                    ],
                )?;
                tmp_ij.scale(-1.0);
                let ke = kr1[ki];
                tmp_ij.add_assign(&einsum(
                    "abej,ie->ijab",
                    &[
                        &wvvvo.slice_leading(&[ka, kb, ke])?,
                        &r1.slice_leading(&[ki])?,
                    ],
                )?)?;
                add(&mut hr2, [ki, kj, ka], &tmp_ij, 1.0)?;
                add(
                    &mut hr2,
                    [kj, ki, ka],
                    &tmp_ij.transpose(&[1, 0, 2, 3])?,
                    -1.0,
                )?;

                // P(ab)
                let mut tmp_ab = einsum(
                    "be,ijae->ijab",
                    &[
                        &imds.fvv.slice_leading(&[kb])?,
                        &r2.slice_leading(&[ki, kj, ka])?,
                    ],
                )?;
                let km = kconserv.get(ki, kb, kj) as usize;
                let mut w = einsum(
                    "mbij,ma->ijab",
                    &[
                        &wovoo.slice_leading(&[km, kb, ki])?,
                        &r1.slice_leading(&[km])?,
                    ],
                )?;
                w.scale(-1.0);
                tmp_ab.add_assign(&w)?;
                add(&mut hr2, [ki, kj, ka], &tmp_ab, 1.0)?;
                add(
                    &mut hr2,
                    [ki, kj, kb],
                    &tmp_ab.transpose(&[0, 1, 3, 2])?,
                    -1.0,
                )?;

                // `:1451-1461` — the `oooo` and `vvvv` ladders.
                let mut ladder = ZArr::zeros(&[nocc, nocc, nvir, nvir]);
                for km in 0..nkpts {
                    let kn = kconserv.get(ki, km, kj) as usize;
                    ladder.add_assign(&einsum_scaled(
                        "mnij,mnab->ijab",
                        &[
                            &woooo.slice_leading(&[km, kn, ki])?,
                            &r2.slice_leading(&[km, kn, ka])?,
                        ],
                        0.5,
                    )?)?;
                    ladder.add_assign(&einsum_scaled(
                        "abef,ijef->ijab",
                        &[
                            &wvvvv.slice_leading(&[ka, kb, km])?,
                            &r2.slice_leading(&[ki, kj, km])?,
                        ],
                        0.5,
                    )?)?;
                }
                add(&mut hr2, [ki, kj, ka], &ladder, 1.0)?;

                // `:1463-1470` — P(ij) P(ab) on the ring term.
                for km in 0..nkpts {
                    let ke = kconserv.get(km, kj, kb) as usize;
                    let tmp = einsum(
                        "mbej,imae->ijab",
                        &[
                            &imds.wovvo.slice_leading(&[km, kb, ke])?,
                            &r2.slice_leading(&[ki, km, ka])?,
                        ],
                    )?;
                    add(&mut hr2, [ki, kj, ka], &tmp, 1.0)?;
                    add(&mut hr2, [kj, ki, ka], &tmp.transpose(&[1, 0, 2, 3])?, -1.0)?;
                    add(&mut hr2, [ki, kj, kb], &tmp.transpose(&[0, 1, 3, 2])?, -1.0)?;
                    add(&mut hr2, [kj, ki, kb], &tmp.transpose(&[1, 0, 3, 2])?, 1.0)?;
                }
            }
        }
    }

    // `:1478-1517` — the four `M = W·r` intermediates.
    let mut tmp_eb = ZArr::zeros(&[nkpts, nvir, nvir]);
    let mut tmp_fa = ZArr::zeros(&[nkpts, nvir, nvir]);
    let mut tmp_jm = ZArr::zeros(&[nkpts, nocc, nocc]);
    let mut tmp_in = ZArr::zeros(&[nkpts, nocc, nocc]);
    for ke in 0..nkpts {
        let kb = kr1[ke];
        let mut eb = ZArr::zeros(&[nvir, nvir]);
        for km in 0..nkpts {
            for kn in 0..nkpts {
                eb.add_assign(&einsum(
                    "mnef,mnbf->eb",
                    &[
                        &imds.eris.blk(GBlk::Oovv, km, kn, ke)?,
                        &r2.slice_leading(&[km, kn, kb])?,
                    ],
                )?)?;
            }
        }
        tmp_eb.set_leading(&[ke], &eb)?;

        let kf = ke;
        let ka = kr1[kf];
        let mut fa = ZArr::zeros(&[nvir, nvir]);
        for km in 0..nkpts {
            fa.add_assign(&einsum(
                "amfe,me->fa",
                &[
                    &wvovv.slice_leading(&[ka, km, kf])?,
                    &r1.slice_leading(&[km])?,
                ],
            )?)?;
        }
        tmp_fa.set_leading(&[kf], &fa)?;

        let kj = ke;
        let km = kr1[kj];
        let mut jm = ZArr::zeros(&[nocc, nocc]);
        for kn in 0..nkpts {
            for ke2 in 0..nkpts {
                jm.add_assign(&einsum(
                    "mnef,jnef->jm",
                    &[
                        &imds.eris.blk(GBlk::Oovv, km, kn, ke2)?,
                        &r2.slice_leading(&[kj, kn, ke2])?,
                    ],
                )?)?;
            }
        }
        tmp_jm.set_leading(&[kj], &jm)?;

        let ki = ke;
        let kn = kr1[ki];
        let mut in_ = ZArr::zeros(&[nocc, nocc]);
        for km2 in 0..nkpts {
            in_.add_assign(&einsum(
                "mnie,me->in",
                &[
                    &wooov.slice_leading(&[km2, kn, ki])?,
                    &r1.slice_leading(&[km2])?,
                ],
            )?)?;
        }
        tmp_in.set_leading(&[ki], &in_)?;
    }

    // `:1519-1544`
    for ki in 0..nkpts {
        for kj in 0..nkpts {
            for ka in 0..nkpts {
                let kb = kr2(ki, ka, kj);
                let ke = kconserv.get(ki, ka, kj) as usize;
                let mut tmp_ab = einsum_scaled(
                    "eb,ijae->ijab",
                    &[
                        &tmp_eb.slice_leading(&[ke])?,
                        &imds.t2.slice_leading(&[ki, kj, ka])?,
                    ],
                    -0.5,
                )?;
                let kf = kconserv.get(ki, kb, kj) as usize;
                tmp_ab.add_assign(&einsum(
                    "fa,ijfb->ijab",
                    &[
                        &tmp_fa.slice_leading(&[kf])?,
                        &imds.t2.slice_leading(&[ki, kj, kf])?,
                    ],
                )?)?;
                add(&mut hr2, [ki, kj, ka], &tmp_ab, 1.0)?;
                add(
                    &mut hr2,
                    [ki, kj, kb],
                    &tmp_ab.transpose(&[0, 1, 3, 2])?,
                    -1.0,
                )?;

                let km = kr1[kj];
                let mut tmp_ij = einsum_scaled(
                    "jm,imab->ijab",
                    &[
                        &tmp_jm.slice_leading(&[kj])?,
                        &imds.t2.slice_leading(&[ki, km, ka])?,
                    ],
                    -0.5,
                )?;
                let kn = kr1[ki];
                tmp_ij.add_assign(&einsum(
                    "in,njab->ijab",
                    &[
                        &tmp_in.slice_leading(&[ki])?,
                        &imds.t2.slice_leading(&[kn, kj, ka])?,
                    ],
                )?)?;
                add(&mut hr2, [ki, kj, ka], &tmp_ij, 1.0)?;
                add(
                    &mut hr2,
                    [kj, ki, ka],
                    &tmp_ij.transpose(&[1, 0, 2, 3])?,
                    -1.0,
                )?;
            }
        }
    }

    amplitudes_to_vector_ee(&hr1, &hr2, kshift, kconserv)
}

/// `eeccsd_diag` (`:1548-1600`), the `partition = None` branch.
///
/// # Errors
/// As [`eeccsd_matvec`].
#[allow(clippy::too_many_lines)]
pub fn eeccsd_diag(
    kshift: usize,
    imds: &EomImds<'_>,
    kconserv: &Kconserv,
) -> Result<ZArr, PbcCcError> {
    let (nkpts, nocc, nvir) = (imds.eris.nkpts, imds.eris.nocc, imds.eris.nvir);
    let kr1 = kconserv_ee_r1(nkpts, kshift, kconserv);
    let kr2v = kconserv_ee_r2(nkpts, kshift, kconserv);
    let woooo = imds.need(&imds.woooo, "Woooo")?;
    let wvvvv = imds.need(&imds.wvvvv, "Wvvvv")?;

    let mut hr1 = ZArr::zeros(&[nkpts, nocc, nvir]);
    for ki in 0..nkpts {
        let ka = kr1[ki];
        let fi = imds.foo.slice_leading(&[ki])?;
        let fa = imds.fvv.slice_leading(&[ka])?;
        let w = imds.wovvo.slice_leading(&[ki, ka, ka])?;
        let mut blk = ZArr::zeros(&[nocc, nvir]);
        for i in 0..nocc {
            for a in 0..nvir {
                let f = i * nvir + a;
                let (r, m) = fi.at(&[i, i])?;
                blk.data_mut().re[f] -= r;
                blk.data_mut().im[f] -= m;
                let (r, m) = fa.at(&[a, a])?;
                blk.data_mut().re[f] += r;
                blk.data_mut().im[f] += m;
                // `einsum('iaai->ia', Wovvo[ki,ka,ka])`
                let (r, m) = w.at(&[i, a, a, i])?;
                blk.data_mut().re[f] += r;
                blk.data_mut().im[f] += m;
            }
        }
        hr1.set_leading(&[ki], &blk)?;
    }

    let mut hr2 = ZArr::zeros(&[nkpts, nkpts, nkpts, nocc, nocc, nvir, nvir]);
    for ki in 0..nkpts {
        for kj in 0..nkpts {
            for ka in 0..nkpts {
                let kb = r2k(&kr2v, nkpts, ki, ka, kj);
                let mut blk = ZArr::zeros(&[nocc, nocc, nvir, nvir]);
                let fi = imds.foo.slice_leading(&[ki])?;
                let fj = imds.foo.slice_leading(&[kj])?;
                let fa = imds.fvv.slice_leading(&[ka])?;
                let fb = imds.fvv.slice_leading(&[kb])?;
                let wjb = imds.wovvo.slice_leading(&[kj, kb, kb])?;
                let wib = imds.wovvo.slice_leading(&[ki, kb, kb])?;
                let wja = imds.wovvo.slice_leading(&[kj, ka, ka])?;
                let wia = imds.wovvo.slice_leading(&[ki, ka, ka])?;
                let woo = woooo.slice_leading(&[ki, kj, ki])?;
                let wvv = wvvvv.slice_leading(&[ka, kb, ka])?;
                for i in 0..nocc {
                    for j in 0..nocc {
                        for a in 0..nvir {
                            for b in 0..nvir {
                                let f = ((i * nocc + j) * nvir + a) * nvir + b;
                                let mut acc_re = 0.0;
                                let mut acc_im = 0.0;
                                let mut sub = |v: (f64, f64), s: f64| {
                                    acc_re += s * v.0;
                                    acc_im += s * v.1;
                                };
                                sub(fi.at(&[i, i])?, -1.0);
                                sub(fj.at(&[j, j])?, -1.0);
                                sub(fa.at(&[a, a])?, 1.0);
                                sub(fb.at(&[b, b])?, 1.0);
                                sub(wjb.at(&[j, b, b, j])?, 1.0);
                                sub(wib.at(&[i, b, b, i])?, 1.0);
                                sub(wja.at(&[j, a, a, j])?, 1.0);
                                sub(wia.at(&[i, a, a, i])?, 1.0);
                                sub(woo.at(&[i, j, i, j])?, 1.0);
                                sub(wvv.at(&[a, b, a, b])?, 1.0);
                                blk.data_mut().re[f] += acc_re;
                                blk.data_mut().im[f] += acc_im;
                            }
                        }
                    }
                }
                // `:1587-1596` — four `W·t2` diagonal terms. `kconserv`, NOT
                // `kconserv_r2`, is used here; upstream's comment at `:1586`
                // says so explicitly ("This is to make t2 are non-zero").
                let kk = kconserv.get(ka, kj, kb) as usize;
                let v = einsum(
                    "kjab,kjab->jab",
                    &[
                        &imds.eris.blk(GBlk::Oovv, kk, kj, ka)?,
                        &imds.t2.slice_leading(&[kk, kj, ka])?,
                    ],
                )?;
                for i in 0..nocc {
                    for j in 0..nocc {
                        for a in 0..nvir {
                            for b in 0..nvir {
                                let f = ((i * nocc + j) * nvir + a) * nvir + b;
                                let (r, m) = v.at(&[j, a, b])?;
                                blk.data_mut().re[f] -= r;
                                blk.data_mut().im[f] -= m;
                            }
                        }
                    }
                }
                let kk = kconserv.get(ka, ki, kb) as usize;
                // NOTE: upstream indexes `t2[kk, ka, ka]` on this line
                // (`:1591`), not `t2[kk, ki, ka]` as the pattern of the other
                // three would suggest. It is transcribed AS WRITTEN — a port
                // that "fixed" it would disagree with the reference it is
                // gated against, and this is upstream's number to change.
                let v = einsum(
                    "kiab,kiab->iab",
                    &[
                        &imds.eris.blk(GBlk::Oovv, kk, ki, ka)?,
                        &imds.t2.slice_leading(&[kk, ka, ka])?,
                    ],
                )?;
                for i in 0..nocc {
                    for j in 0..nocc {
                        for a in 0..nvir {
                            for b in 0..nvir {
                                let f = ((i * nocc + j) * nvir + a) * nvir + b;
                                let (r, m) = v.at(&[i, a, b])?;
                                blk.data_mut().re[f] -= r;
                                blk.data_mut().im[f] -= m;
                            }
                        }
                    }
                }
                let kc = kconserv.get(ki, kb, kj) as usize;
                let v = einsum(
                    "ijcb,ijcb->ijb",
                    &[
                        &imds.eris.blk(GBlk::Oovv, ki, kj, kc)?,
                        &imds.t2.slice_leading(&[ki, kj, kc])?,
                    ],
                )?;
                for i in 0..nocc {
                    for j in 0..nocc {
                        for a in 0..nvir {
                            for b in 0..nvir {
                                let f = ((i * nocc + j) * nvir + a) * nvir + b;
                                let (r, m) = v.at(&[i, j, b])?;
                                blk.data_mut().re[f] -= r;
                                blk.data_mut().im[f] -= m;
                            }
                        }
                    }
                }
                let kc = kconserv.get(ki, ka, kj) as usize;
                let v = einsum(
                    "ijca,ijca->ija",
                    &[
                        &imds.eris.blk(GBlk::Oovv, ki, kj, kc)?,
                        &imds.t2.slice_leading(&[ki, kj, kc])?,
                    ],
                )?;
                for i in 0..nocc {
                    for j in 0..nocc {
                        for a in 0..nvir {
                            for b in 0..nvir {
                                let f = ((i * nocc + j) * nvir + a) * nvir + b;
                                let (r, m) = v.at(&[i, j, a])?;
                                blk.data_mut().re[f] -= r;
                                blk.data_mut().im[f] -= m;
                            }
                        }
                    }
                }
                hr2.set_leading(&[ki, kj, ka], &blk)?;
            }
        }
    }
    amplitudes_to_vector_ee(&hr1, &hr2, kshift, kconserv)
}

// ---------------------------------------------------------------------------
// The `partition` branches (`:449-457`, `:1010-1018`)
// ---------------------------------------------------------------------------

/// Upstream's `eom.partition`, and what this port does with each value.
///
/// # Both non-`None` values are REFUSED at upstream's own driver
///
/// `ipccsd` (`:612-618`) and `eaccsd` (`:902-906`) both do
///
/// ```text
/// if partition:
///     eom.partition = partition.lower()
///     assert eom.partition in ['mp','full']
///     if eom.partition in ['mp', 'full']:
///         raise NotImplementedError
/// ```
///
/// and `eom_kccsd_rhf`'s and `eom_kccsd_uhf`'s `EOMIP`/`EOMEA` inherit those
/// drivers, so NO caller reaches a partition branch through the public API in
/// ANY of the three modules. Measured, not assumed: all four of
/// `{ip,ea}ccsd(partition=('mp'|'full'))` raise `NotImplementedError`. Every
/// `eom_kernel` in this crate reproduces that refusal.
///
/// # What is behind the refusal, module by module
///
/// The branches stay reachable the way upstream leaves them reachable — set
/// `eom.partition` and call the matvec or the diagonal directly — and this
/// crate exposes them as explicit `*_mp` / `*_partition` entry points:
///
/// | module | `'mp'` matvec | `'mp'` diagonal |
/// |---|---|---|
/// | `eom_kccsd_ghf` | no branch exists | [`ipccsd_diag_mp`], [`eaccsd_diag_mp`] |
/// | `eom_kccsd_rhf` | `ipccsd_matvec_partition`, `eaccsd_matvec_partition` | `ipccsd_diag_partition`, `eaccsd_diag_partition` |
/// | `eom_kccsd_uhf` | no branch exists | REFUSED — `raise Exception("MP diag is not tested")` |
///
/// **Four of those branches need an attribute upstream never sets.**
/// `ghf ipccsd_diag:449`, `ghf eaccsd_diag:1010`, `rhf ipccsd_diag:177` and
/// `rhf eaccsd_matvec:463` open with `eom.eris.fock`, and `eom.eris` does not
/// exist on an EOM object — `pyscf/cc/eom_rccsd.py`'s `EOM.__init__` never
/// assigns it, only `_IMDS` does. Measured: they raise `AttributeError:
/// 'EOMIP' object has no attribute 'eris'`. Their siblings
/// `rhf ipccsd_matvec:70` and `rhf eaccsd_diag:584` read `imds.eris.fock`
/// instead and run. `imds.eris` IS the same `_ERIS`, so this port reads the
/// Fock from the intermediates uniformly, and the oracle that gates these four
/// supplies `eom.eris` so that upstream's own arithmetic produces the
/// reference rather than this port inventing one.
///
/// # `'full'` computes nothing anywhere
///
/// Only the two RHF matvecs have a `'full'` branch at all
/// (`eom_kccsd_rhf.py:79-83`, `:472-476`). It reads
///
/// ```text
/// if diag is not None:
///     diag = eom.get_diag(imds=imds)
/// diag_matrix2 = eom.vector_to_amplitudes(diag, nmo, nocc)[1]
/// ```
///
/// — note the inverted guard — and `EOMIP.vector_to_amplitudes` there takes
/// `(self, vector, kshift=None)`, so the three-argument call raises
/// `TypeError: EOMIP.vector_to_amplitudes() takes from 2 to 3 positional
/// arguments but 4 were given` before any arithmetic happens. Measured for
/// both IP and EA. [`Partition::refuse_full`] is that `TypeError`.
///
/// The DIAGONALS have no `'full'` branch — their `if 'mp' … else …` falls
/// through — so `partition='full'` on a diagonal IS the `None` diagonal, and
/// this crate's `*_diag_partition` reproduces that rather than refusing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Partition {
    /// The only branch upstream's drivers permit.
    #[default]
    None,
    /// Møller-Plesset: bare Fock diagonals in place of `Foo`/`Fvv`, and the
    /// two-body `W` terms dropped.
    Mp,
    /// Unreachable upstream — see the type doc.
    Full,
}

/// The refusal `ipccsd`/`eaccsd` give a non-`None` `partition` (`:618`,
/// `:905`), shared by this module's and [`crate::eom_kccsd_rhf`]'s drivers —
/// `eom_kccsd_rhf`'s `EOMIP`/`EOMEA` inherit these very functions.
///
/// Exposed as a function so the refusal is testable without building a
/// fixture: it is the first statement of `eom_kernel`, before `imds` is read.
#[must_use]
pub fn partition_refusal() -> PbcCcError {
    PbcCcError::NotImplementedUpstream {
        upstream: "pbc/cc/eom_kccsd_ghf.py:618",
        what: "ipccsd/eaccsd raise NotImplementedError for partition in ('mp', 'full')",
    }
}

impl Partition {
    /// The refusal every `'full'` branch in this crate shares.
    ///
    /// # Errors
    /// [`PbcCcError::NotImplementedUpstream`] for [`Partition::Full`].
    pub fn refuse_full(p: Self) -> Result<(), PbcCcError> {
        if p == Self::Full {
            return Err(PbcCcError::NotImplementedUpstream {
                upstream: "pbc/cc/eom_kccsd_rhf.py:474",
                what: "partition='full' calls eom.vector_to_amplitudes(diag, nmo, nocc), which \
                       takes at most 3 positional arguments; upstream raises TypeError before \
                       any arithmetic",
            });
        }
        Ok(())
    }
}

/// `ipccsd_diag`'s `partition == 'mp'` branch (`:449-457`).
///
/// The singles diagonal is unchanged; only the doubles block differs, and there
/// it is the BARE Fock diagonal rather than `Foo`/`Fvv`, with every `W` term
/// dropped.
///
/// # Errors
/// As [`ipccsd_diag`].
pub fn ipccsd_diag_mp(
    kshift: usize,
    imds: &EomImds<'_>,
    kconserv: &Kconserv,
) -> Result<ZArr, PbcCcError> {
    let (nkpts, nocc, nvir) = (imds.eris.nkpts, imds.eris.nocc, imds.eris.nvir);
    let mut hr1 = ZArr::zeros(&[nocc]);
    let foo_s = imds.foo.slice_leading(&[kshift])?;
    for i in 0..nocc {
        let (re, im) = foo_s.at(&[i, i])?;
        hr1.data_mut().re[i] = -re;
        hr1.data_mut().im[i] = -im;
    }
    let mut hr2 = ZArr::zeros(&[nkpts, nkpts, nocc, nocc, nvir]);
    for ki in 0..nkpts {
        for kj in 0..nkpts {
            let ka = kconserv.get(ki, kshift, kj) as usize;
            let fi = imds.eris.foo(ki)?;
            let fj = imds.eris.foo(kj)?;
            let fa = imds.eris.fvv(ka)?;
            let mut blk = ZArr::zeros(&[nocc, nocc, nvir]);
            for i in 0..nocc {
                for j in 0..nocc {
                    for a in 0..nvir {
                        let f = (i * nocc + j) * nvir + a;
                        let (r, m) = fi.at(&[i, i])?;
                        blk.data_mut().re[f] -= r;
                        blk.data_mut().im[f] -= m;
                        let (r, m) = fj.at(&[j, j])?;
                        blk.data_mut().re[f] -= r;
                        blk.data_mut().im[f] -= m;
                        let (r, m) = fa.at(&[a, a])?;
                        blk.data_mut().re[f] += r;
                        blk.data_mut().im[f] += m;
                    }
                }
            }
            hr2.set_leading(&[ki, kj], &blk)?;
        }
    }
    amplitudes_to_vector_ip(&hr1, &hr2)
}

/// `eaccsd_diag`'s `partition == 'mp'` branch (`:1010-1018`), which upstream
/// itself labels `# This case is untested`.
///
/// **The sign on `fvv[ka]` is a MINUS here** where the `None` branch has a plus
/// (`:1024` vs `:1015`). Transcribed as written; see [`Partition`] on why this
/// port does not "fix" upstream.
///
/// # Errors
/// As [`eaccsd_diag`].
pub fn eaccsd_diag_mp(
    kshift: usize,
    imds: &EomImds<'_>,
    kconserv: &Kconserv,
) -> Result<ZArr, PbcCcError> {
    let (nkpts, nocc, nvir) = (imds.eris.nkpts, imds.eris.nocc, imds.eris.nvir);
    let mut hr1 = ZArr::zeros(&[nvir]);
    let fvv_s = imds.fvv.slice_leading(&[kshift])?;
    for a in 0..nvir {
        let (re, im) = fvv_s.at(&[a, a])?;
        hr1.data_mut().re[a] = re;
        hr1.data_mut().im[a] = im;
    }
    let mut hr2 = ZArr::zeros(&[nkpts, nkpts, nocc, nvir, nvir]);
    for kj in 0..nkpts {
        for ka in 0..nkpts {
            let kb = kconserv.get(kshift, ka, kj) as usize;
            let fj = imds.eris.foo(kj)?;
            let fa = imds.eris.fvv(ka)?;
            let fb = imds.eris.fvv(kb)?;
            let mut blk = ZArr::zeros(&[nocc, nvir, nvir]);
            for j in 0..nocc {
                for a in 0..nvir {
                    for b in 0..nvir {
                        let f = (j * nvir + a) * nvir + b;
                        let (r, m) = fj.at(&[j, j])?;
                        blk.data_mut().re[f] -= r;
                        blk.data_mut().im[f] -= m;
                        // `:1016` — MINUS, unlike `:1024`'s plus.
                        let (r, m) = fa.at(&[a, a])?;
                        blk.data_mut().re[f] -= r;
                        blk.data_mut().im[f] -= m;
                        let (r, m) = fb.at(&[b, b])?;
                        blk.data_mut().re[f] += r;
                        blk.data_mut().im[f] += m;
                    }
                }
            }
            hr2.set_leading(&[kj, ka], &blk)?;
        }
    }
    amplitudes_to_vector_ea(&hr1, &hr2, kshift, kconserv)
}

// ---------------------------------------------------------------------------
// The CCSD* corrections (`:478-608`, `:1038-1166`)
// ---------------------------------------------------------------------------

/// One `(right, left)` eigenpair with its EOM eigenvalue, the input a CCSD\*
/// correction consumes.
///
/// Upstream takes three parallel sequences — `evals`, `evecs`, `levecs` — and
/// `zip`s them (`:514`, `:1071`). This bundles one element of that zip so a
/// caller cannot silently pair a right vector with the wrong left one, which
/// is exactly the class of defect `16-VERIFICATION` records for free k-axis
/// gathers.
pub struct StarPair<'a> {
    /// The converged EOM-CCSD eigenvalue this root is corrected from.
    pub eval: f64,
    /// The RIGHT eigenvector, packed as the matvec packs it.
    pub r: &'a ZArr,
    /// The LEFT eigenvector, from `left = true`.
    pub l: &'a ZArr,
}

/// Everything `get_kconserv3` needs that a [`Kconserv`] does not carry.
///
/// `ipccsd_star_contract:539` calls `kpts_helper.get_kconserv3(eom._cc._scf.cell,
/// eom._cc.kpts, …)`, so the lattice and the k-mesh have to reach the
/// correction. [`crate::kccsd_t`] passes the same pair for the same reason.
pub struct StarLattice<'a> {
    /// `cell.lattice_vectors()`.
    pub a: &'a [[f64; 3]; 3],
    /// `cc.kpts`.
    pub kpts: &'a [[f64; 3]],
}

/// The `1/12` prefactor both corrections end with (`:603`, `:1161`).
const STAR_PREFACTOR: f64 = 1.0 / 12.0;

/// Upstream warns below this left-right overlap (`:521-524`, `:1081-1084`).
const STAR_SMALL_OVERLAP: f64 = 1e-7;

/// One corrected root, plus the diagnostics upstream logs on the way.
#[derive(Debug, Clone, Copy)]
pub struct StarRoot {
    /// `ip_eval + deltaE` — the CCSD\* energy (`:607`, `:1165`).
    pub e_star: f64,
    /// `deltaE` alone, the perturbative correction.
    pub delta_e: f64,
    /// The IMAGINARY part `:604`'s `deltaE.real` throws away, kept as a
    /// diagnostic: it is zero only when the left and right vectors really are
    /// a biorthogonal pair, so a caller that wants upstream's silent discard
    /// can ignore it and one that wants a check has it.
    pub delta_e_imag: f64,
    /// `<L|R>` BEFORE the normalisation, real and imaginary
    /// (`:520`, `:1080`). Upstream logs it and warns when `|<L|R>| < 1e-7`;
    /// this returns it so a caller can apply its own judgement.
    pub ldotr: (f64, f64),
    /// Whether `|<L|R>|` fell below [`STAR_SMALL_OVERLAP`] — upstream's
    /// "Results may be inaccurate" warning, as data rather than a log line.
    pub small_overlap: bool,
}

/// `<L|R> = l1·r1 + ½ l2·r2` (`:519`), UNCONJUGATED — upstream uses `np.dot`
/// on the raw amplitudes, not a Hermitian inner product.
fn ldotr(l1: &ZArr, l2: &ZArr, r1: &ZArr, r2: &ZArr) -> (f64, f64) {
    let mut re = 0.0;
    let mut im = 0.0;
    for i in 0..l1.len() {
        re += l1.data().re[i] * r1.data().re[i] - l1.data().im[i] * r1.data().im[i];
        im += l1.data().re[i] * r1.data().im[i] + l1.data().im[i] * r1.data().re[i];
    }
    for i in 0..l2.len() {
        re += 0.5 * (l2.data().re[i] * r2.data().re[i] - l2.data().im[i] * r2.data().im[i]);
        im += 0.5 * (l2.data().re[i] * r2.data().im[i] + l2.data().im[i] * r2.data().re[i]);
    }
    (re, im)
}

/// `l /= ldotr` for a complex scalar.
fn scale_by_inverse(x: &mut ZArr, (re, im): (f64, f64)) {
    let d = re * re + im * im;
    x.scale_complex(re / d, -im / d);
}

/// `get_kconserv3(cell, kpts, [p, q, kshift, range(nkpts), range(nkpts)])`,
/// returned as a `[nkpts, nkpts]` row-major table.
fn kklist(lat: &StarLattice<'_>, p: usize, q: usize, kshift: usize, nkpts: usize) -> Vec<usize> {
    let all: Vec<usize> = (0..nkpts).collect();
    get_kconserv3(
        lat.a,
        lat.kpts,
        &[
            KIdx::One(p),
            KIdx::One(q),
            KIdx::One(kshift),
            KIdx::Many(all.clone()),
            KIdx::Many(all),
        ],
    )
    .data
    .iter()
    .map(|v| *v as usize)
    .collect()
}

/// The occupied and virtual orbital energies `eris.mo_energy` splits into
/// (`:504-508`, `:1062-1066`).
fn split_mo_energy(eris: &KgEris, nocc: usize) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
    let o = eris.mo_energy.iter().map(|e| e[..nocc].to_vec()).collect();
    let v = eris.mo_energy.iter().map(|e| e[nocc..].to_vec()).collect();
    (o, v)
}

/// `ipccsd_star_contract(eom, evals, evecs, levecs, kshift, imds)` (`:478-608`)
/// — the IP-CCSD\* correction of Saeh and Stanton, JCP **111**, 8275 (1999).
///
/// # The 2h1p right amplitudes are `s^{a}_{ij}`
///
/// Upstream's docstring says so at `:489`: the `(ia)` indices are coupled.
/// That is the packing [`vector_to_amplitudes_ip`] already produces, so the
/// vectors handed in here are exactly the Davidson's.
///
/// # `partition` must be `None`
///
/// `:494` asserts it, and there is no `'mp'` form of this correction — see
/// [`Partition`].
///
/// # The three-body arrays are the cost
///
/// `lijkab` and `rijkab` are `[nkpts, nkpts, nocc³, nvir²]` EACH, and `P(ijk)`
/// makes a second copy of both, so the peak is four of them at once —
/// upstream's own shape (`:531-532`, `:582-583`), reproduced rather than
/// blocked, because a blocked form would change the summation order and this
/// is the first port of the equation.
///
/// # Errors
/// Propagates every block access; [`PbcCcError::Shape`] if a vector is not
/// [`ip_vector_size`] long.
#[allow(clippy::too_many_lines)]
pub fn ipccsd_star_contract(
    pairs: &[StarPair<'_>],
    kshift: usize,
    imds: &EomImds<'_>,
    padding: &Padding,
    kconserv: &Kconserv,
    lat: &StarLattice<'_>,
) -> Result<Vec<StarRoot>, PbcCcError> {
    let (nkpts, nocc, nvir) = (imds.eris.nkpts, imds.eris.nocc, imds.eris.nvir);
    let (mo_e_o, mo_e_v) = split_mo_energy(imds.eris, nocc);
    let t2 = &imds.t2;
    let mut out = Vec::with_capacity(pairs.len());

    for pair in pairs {
        // `:517-518` — both vectors unpacked with the SAME kshift.
        let (l1, mut l2) = vector_to_amplitudes_ip(pair.l, nkpts, nocc, nvir)?;
        let (r1, r2) = vector_to_amplitudes_ip(pair.r, nkpts, nocc, nvir)?;
        let mut l1 = l1;

        // `:519-528` — enforce `<L|R> = 1` by scaling the LEFT vector only.
        let dot = ldotr(&l1, &l2, &r1, &r2);
        let small = dot.0.hypot(dot.1) < STAR_SMALL_OVERLAP;
        scale_by_inverse(&mut l1, dot);
        scale_by_inverse(&mut l2, dot);

        let mut delta_re = 0.0_f64;
        let mut delta_im = 0.0_f64;

        for ka in 0..nkpts {
            for kb in 0..nkpts {
                let kk_of = kklist(lat, ka, kb, kshift, nkpts);
                let shape = [nkpts, nkpts, nocc, nocc, nocc, nvir, nvir];
                let mut lijkab = ZArr::zeros(&shape);
                let mut rijkab = ZArr::zeros(&shape);

                for ki in 0..nkpts {
                    for kj in 0..nkpts {
                        let kk = kk_of[ki * nkpts + kj];
                        let mut lblk = ZArr::zeros(&[nocc, nocc, nocc, nvir, nvir]);
                        let mut rblk = ZArr::zeros(&[nocc, nocc, nocc, nvir, nvir]);

                        // --- `lijkab` (`:545-561`)
                        // `:547-548`
                        if kk == kshift && kb == kconserv.get(ki, ka, kj) as usize {
                            lblk.add_assign(&einsum(
                                "ijab,k->ijkab",
                                &[&imds.eris.blk(GBlk::Oovv, ki, kj, ka)?, &l1],
                            )?)?;
                        }
                        // `:550-554` — `-tmp + tmpT`, the `a`/`b` exchange.
                        let km = kconserv.get(kj, ka, ki) as usize;
                        lblk.sub_assign(&einsum(
                            "jima,mkb->ijkab",
                            &[
                                &imds.eris.blk(GBlk::Ooov, kj, ki, km)?,
                                &l2.slice_leading(&[km, kk])?,
                            ],
                        )?)?;
                        let km = kconserv.get(kj, kb, ki) as usize;
                        lblk.add_assign(&einsum(
                            "jimb,mka->ijkab",
                            &[
                                &imds.eris.blk(GBlk::Ooov, kj, ki, km)?,
                                &l2.slice_leading(&[km, kk])?,
                            ],
                        )?)?;
                        // `:556-557`
                        let ke = kconserv.get(ka, ki, kb) as usize;
                        lblk.add_assign(&einsum(
                            "ieab,jke->ijkab",
                            &[
                                &imds.eris.blk(GBlk::Ovvv, ki, ke, ka)?,
                                &l2.slice_leading(&[kj, kk])?,
                            ],
                        )?)?;

                        // --- `rijkab` (`:559-577`)
                        // `:560-564` — `-(tmp - tmpT)`.
                        let tmp = einsum(
                            "mbke,m->bke",
                            &[&imds.eris.blk(GBlk::Ovov, kshift, kb, kk)?, &r1],
                        )?;
                        rblk.sub_assign(&einsum(
                            "bke,ijae->ijkab",
                            &[&tmp, &t2.slice_leading(&[ki, kj, ka])?],
                        )?)?;
                        let tmpt = einsum(
                            "make,m->ake",
                            &[&imds.eris.blk(GBlk::Ovov, kshift, ka, kk)?, &r1],
                        )?;
                        rblk.add_assign(&einsum(
                            "ake,ijbe->ijkab",
                            &[&tmpt, &t2.slice_leading(&[ki, kj, kb])?],
                        )?)?;
                        // `:566-569`
                        let km = kconserv.get(kj, kshift, kk) as usize;
                        let tmp = einsum(
                            "mnjk,n->mjk",
                            &[&imds.eris.blk(GBlk::Oooo, km, kshift, kj)?, &r1],
                        )?;
                        rblk.add_assign(&einsum(
                            "mjk,imab->ijkab",
                            &[&tmp, &t2.slice_leading(&[ki, km, ka])?],
                        )?)?;
                        // `:571-575` — note the `.conj()` upstream applies to
                        // the eris here and NOT in the `lijkab` block above.
                        let km = kconserv.get(kj, ka, ki) as usize;
                        rblk.sub_assign(&einsum(
                            "jima,mkb->ijkab",
                            &[
                                &imds.eris.blk(GBlk::Ooov, kj, ki, km)?.conj(),
                                &r2.slice_leading(&[km, kk])?,
                            ],
                        )?)?;
                        let km = kconserv.get(kj, kb, ki) as usize;
                        rblk.add_assign(&einsum(
                            "jimb,mka->ijkab",
                            &[
                                &imds.eris.blk(GBlk::Ooov, kj, ki, km)?.conj(),
                                &r2.slice_leading(&[km, kk])?,
                            ],
                        )?)?;
                        // `:577`
                        let ke = kconserv.get(ka, ki, kb) as usize;
                        rblk.add_assign(&einsum(
                            "ieab,jke->ijkab",
                            &[
                                &imds.eris.blk(GBlk::Ovvv, ki, ke, ka)?.conj(),
                                &r2.slice_leading(&[kj, kk])?,
                            ],
                        )?)?;

                        lijkab.set_leading(&[ki, kj], &lblk)?;
                        rijkab.set_leading(&[ki, kj], &rblk)?;
                    }
                }

                // `:583-597` — `P(ijk)`, the denominator, and the contraction.
                let eab = crate::kccsd_t::epq2(&mo_e_v, &padding.virtuals, ka, kb, nvir, -1.0);
                for ki in 0..nkpts {
                    for kj in 0..nkpts {
                        let kk = kk_of[ki * nkpts + kj];
                        let mut pl = lijkab.slice_leading(&[ki, kj])?;
                        pl.add_assign(
                            &lijkab
                                .slice_leading(&[kj, kk])?
                                .transpose(&[2, 0, 1, 3, 4])?,
                        )?;
                        pl.add_assign(
                            &lijkab
                                .slice_leading(&[kk, ki])?
                                .transpose(&[1, 2, 0, 3, 4])?,
                        )?;
                        let mut pr = rijkab.slice_leading(&[ki, kj])?;
                        pr.add_assign(
                            &rijkab
                                .slice_leading(&[kj, kk])?
                                .transpose(&[2, 0, 1, 3, 4])?,
                        )?;
                        pr.add_assign(
                            &rijkab
                                .slice_leading(&[kk, ki])?
                                .transpose(&[1, 2, 0, 3, 4])?,
                        )?;

                        let eijk = crate::kccsd_t::epqr3(
                            &mo_e_o,
                            &padding.occupied,
                            ki,
                            kj,
                            kk,
                            nocc,
                            1.0,
                        );
                        // `:601` — `Σ Pl · Pr / (eijkab + ip_eval)`, summed
                        // over every free index at once.
                        let mut f = 0;
                        for &e_ijk in &eijk {
                            for &e_ab in &eab {
                                let denom = e_ijk + e_ab + pair.eval;
                                let (lr, li) = (pl.data().re[f], pl.data().im[f]);
                                let (rr, ri) = (pr.data().re[f], pr.data().im[f]);
                                delta_re += (lr * rr - li * ri) / denom;
                                delta_im += (lr * ri + li * rr) / denom;
                                f += 1;
                            }
                        }
                    }
                }
            }
        }

        // `:603-604` — the `1/12`, then the REAL part.
        let delta_e = delta_re * STAR_PREFACTOR;
        out.push(StarRoot {
            e_star: pair.eval + delta_e,
            delta_e,
            delta_e_imag: delta_im * STAR_PREFACTOR,
            ldotr: dot,
            small_overlap: small,
        });
    }
    Ok(out)
}

/// `eaccsd_star_contract(eom, evals, evecs, levecs, kshift, imds)`
/// (`:1038-1166`) — the EA-CCSD\* correction, same reference.
///
/// # This is NOT [`ipccsd_star_contract`] with the spaces swapped
///
/// The outer loop is over `(ki, kj)` rather than `(ka, kb)` (`:1088`), the
/// three-body arrays are `[nocc², nvir³]`, `P(abc)` permutes the LAST three
/// axes (`:1143-1148`), and the `l1` term enters with a MINUS (`:1103`) where
/// the IP one enters with a plus (`:548`). Each is transcribed from the line
/// above it.
///
/// # Errors
/// As [`ipccsd_star_contract`].
#[allow(clippy::too_many_lines)]
pub fn eaccsd_star_contract(
    pairs: &[StarPair<'_>],
    kshift: usize,
    imds: &EomImds<'_>,
    padding: &Padding,
    kconserv: &Kconserv,
    lat: &StarLattice<'_>,
) -> Result<Vec<StarRoot>, PbcCcError> {
    let (nkpts, nocc, nvir) = (imds.eris.nkpts, imds.eris.nocc, imds.eris.nvir);
    let (mo_e_o, mo_e_v) = split_mo_energy(imds.eris, nocc);
    let t2 = &imds.t2;
    let mut out = Vec::with_capacity(pairs.len());

    for pair in pairs {
        let (l1, mut l2) = vector_to_amplitudes_ea(pair.l, kshift, nkpts, nocc, nvir, kconserv)?;
        let (r1, r2) = vector_to_amplitudes_ea(pair.r, kshift, nkpts, nocc, nvir, kconserv)?;
        let mut l1 = l1;

        let dot = ldotr(&l1, &l2, &r1, &r2);
        let small = dot.0.hypot(dot.1) < STAR_SMALL_OVERLAP;
        scale_by_inverse(&mut l1, dot);
        scale_by_inverse(&mut l2, dot);

        let mut delta_re = 0.0_f64;
        let mut delta_im = 0.0_f64;

        for ki in 0..nkpts {
            for kj in 0..nkpts {
                let kc_of = kklist(lat, ki, kj, kshift, nkpts);
                let shape = [nkpts, nkpts, nocc, nocc, nvir, nvir, nvir];
                let mut lijabc = ZArr::zeros(&shape);
                let mut rijabc = ZArr::zeros(&shape);

                for ka in 0..nkpts {
                    for kb in 0..nkpts {
                        let kc = kc_of[ka * nkpts + kb];
                        let mut lblk = ZArr::zeros(&[nocc, nocc, nvir, nvir, nvir]);
                        let mut rblk = ZArr::zeros(&[nocc, nocc, nvir, nvir, nvir]);

                        // --- `lijabc` (`:1101-1116`)
                        // `:1102-1103` — a MINUS, unlike the IP `l1` term.
                        if kc == kshift && kb == kconserv.get(ki, ka, kj) as usize {
                            lblk.sub_assign(&einsum(
                                "ijab,c->ijabc",
                                &[&imds.eris.blk(GBlk::Oovv, ki, kj, ka)?, &l1],
                            )?)?;
                        }
                        // `:1105-1106`
                        let km = kconserv.get(kj, ka, ki) as usize;
                        lblk.sub_assign(&einsum(
                            "jima,mbc->ijabc",
                            &[
                                &imds.eris.blk(GBlk::Ooov, kj, ki, km)?,
                                &l2.slice_leading(&[km, kb])?,
                            ],
                        )?)?;
                        // `:1108-1112` — `-(tmp - tmpT)`.
                        let ke = kconserv.get(ka, ki, kb) as usize;
                        lblk.sub_assign(&einsum(
                            "ieab,jce->ijabc",
                            &[
                                &imds.eris.blk(GBlk::Ovvv, ki, ke, ka)?,
                                &l2.slice_leading(&[kj, kc])?,
                            ],
                        )?)?;
                        let ke = kconserv.get(ka, kj, kb) as usize;
                        lblk.add_assign(&einsum(
                            "jeab,ice->ijabc",
                            &[
                                &imds.eris.blk(GBlk::Ovvv, kj, ke, ka)?,
                                &l2.slice_leading(&[ki, kc])?,
                            ],
                        )?)?;

                        // --- `rijabc` (`:1114-1132`)
                        // `:1115-1118`
                        let ke = kconserv.get(kb, kshift, kc) as usize;
                        let tmp = einsum(
                            "bcef,f->bce",
                            &[&imds.eris.blk(GBlk::Vvvv, kb, kc, ke)?, &r1],
                        )?;
                        rblk.sub_assign(&einsum(
                            "bce,ijae->ijabc",
                            &[&tmp, &t2.slice_leading(&[ki, kj, ka])?],
                        )?)?;
                        // `:1120-1126` — `+(tmp - tmpT)`, the `i`/`j` exchange.
                        let km = kconserv.get(kj, kc, kshift) as usize;
                        let tmp = einsum(
                            "mcje,e->mcj",
                            &[&imds.eris.blk(GBlk::Ovov, km, kc, kj)?, &r1],
                        )?;
                        rblk.add_assign(&einsum(
                            "mcj,imab->ijabc",
                            &[&tmp, &t2.slice_leading(&[ki, km, ka])?],
                        )?)?;
                        let km = kconserv.get(ki, kc, kshift) as usize;
                        let tmpt = einsum(
                            "mcie,e->mci",
                            &[&imds.eris.blk(GBlk::Ovov, km, kc, ki)?, &r1],
                        )?;
                        rblk.sub_assign(&einsum(
                            "mci,jmab->ijabc",
                            &[&tmpt, &t2.slice_leading(&[kj, km, ka])?],
                        )?)?;
                        // `:1128-1129`
                        let km = kconserv.get(kj, ka, ki) as usize;
                        rblk.add_assign(&einsum(
                            "jima,mcb->ijabc",
                            &[
                                &imds.eris.blk(GBlk::Ooov, kj, ki, km)?.conj(),
                                &r2.slice_leading(&[km, kc])?,
                            ],
                        )?)?;
                        // `:1131-1135`
                        let ke = kconserv.get(ka, ki, kb) as usize;
                        rblk.sub_assign(&einsum(
                            "ieab,jce->ijabc",
                            &[
                                &imds.eris.blk(GBlk::Ovvv, ki, ke, ka)?.conj(),
                                &r2.slice_leading(&[kj, kc])?,
                            ],
                        )?)?;
                        let ke = kconserv.get(ka, kj, kb) as usize;
                        rblk.add_assign(&einsum(
                            "jeab,ice->ijabc",
                            &[
                                &imds.eris.blk(GBlk::Ovvv, kj, ke, ka)?.conj(),
                                &r2.slice_leading(&[ki, kc])?,
                            ],
                        )?)?;

                        lijabc.set_leading(&[ka, kb], &lblk)?;
                        rijabc.set_leading(&[ka, kb], &rblk)?;
                    }
                }

                // `:1143-1158` — `P(abc)` on the LAST three axes.
                let eij = crate::kccsd_t::epq2(&mo_e_o, &padding.occupied, ki, kj, nocc, 1.0);
                for ka in 0..nkpts {
                    for kb in 0..nkpts {
                        let kc = kc_of[ka * nkpts + kb];
                        let mut pl = lijabc.slice_leading(&[ka, kb])?;
                        pl.add_assign(
                            &lijabc
                                .slice_leading(&[kb, kc])?
                                .transpose(&[0, 1, 4, 2, 3])?,
                        )?;
                        pl.add_assign(
                            &lijabc
                                .slice_leading(&[kc, ka])?
                                .transpose(&[0, 1, 3, 4, 2])?,
                        )?;
                        let mut pr = rijabc.slice_leading(&[ka, kb])?;
                        pr.add_assign(
                            &rijabc
                                .slice_leading(&[kb, kc])?
                                .transpose(&[0, 1, 4, 2, 3])?,
                        )?;
                        pr.add_assign(
                            &rijabc
                                .slice_leading(&[kc, ka])?
                                .transpose(&[0, 1, 3, 4, 2])?,
                        )?;

                        let eabc = crate::kccsd_t::epqr3(
                            &mo_e_v,
                            &padding.virtuals,
                            ka,
                            kb,
                            kc,
                            nvir,
                            -1.0,
                        );
                        let mut f = 0;
                        for &e_ij in &eij {
                            for &e_abc in &eabc {
                                let denom = e_ij + e_abc + pair.eval;
                                let (lr, li) = (pl.data().re[f], pl.data().im[f]);
                                let (rr, ri) = (pr.data().re[f], pr.data().im[f]);
                                delta_re += (lr * rr - li * ri) / denom;
                                delta_im += (lr * ri + li * rr) / denom;
                                f += 1;
                            }
                        }
                    }
                }
            }
        }

        let delta_e = delta_re * STAR_PREFACTOR;
        out.push(StarRoot {
            e_star: pair.eval + delta_e,
            delta_e,
            delta_e_imag: delta_im * STAR_PREFACTOR,
            ldotr: dot,
            small_overlap: small,
        });
    }
    Ok(out)
}

/// `_sort_left_right_eigensystem(eom, …)` — `pyscf/cc/eom_rccsd.py:144-206`.
///
/// Pairs each CONVERGED right root with the first as-yet-unclaimed converged
/// left root whose eigenvalue is within `tol`. A right root with no partner is
/// DROPPED — upstream logs "Will not perform perturbation on this state"
/// (`:198`) and simply omits it, so the returned list can be shorter than
/// either input.
///
/// The `tol` is upstream's default `1e-6` (`:145`).
#[must_use]
pub fn sort_left_right_eigensystem<'a>(
    right: &'a EomRoots,
    left: &'a EomRoots,
    tol: f64,
) -> Vec<StarPair<'a>> {
    let mut left_idx: Vec<usize> = (0..left.e.len()).filter(|i| left.conv[*i]).collect();
    let right_idx: Vec<usize> = (0..right.e.len()).filter(|i| right.conv[*i]).collect();
    let mut out = Vec::new();
    for ir in right_idx {
        if let Some(pos) = left_idx
            .iter()
            .position(|il| (right.e[ir] - left.e[*il]).abs() < tol)
        {
            let il = left_idx.remove(pos);
            out.push(StarPair {
                eval: right.e[ir],
                r: &right.v[ir],
                l: &left.v[il],
            });
        }
    }
    out
}

/// `perturbed_ccsd_kernel` (`:625-648`) for ONE `kshift` — solve the right
/// eigenproblem, solve the left one, pair them, contract.
///
/// `ipccsd_star` and `eaccsd_star` (`:652-661`, `:1169-1178`) are this with a
/// `partition` refusal in front, and [`eom_kernel`] already carries that
/// refusal, so there is one function here where upstream has three.
///
/// **Upstream passes `right_guess` to BOTH solves** (`:637` and `:642` both
/// read `guess=right_guess`; `left_guess` is accepted and never used). This
/// port has no guess parameter at all — [`eom_kernel`] builds its guess from
/// the diagonal — so the divergence cannot arise.
///
/// # Errors
/// Propagates both Davidson solves and the contraction.
pub fn perturbed_ccsd_kernel(
    kind: Excitation,
    kshift: usize,
    imds: &EomImds<'_>,
    padding: &Padding,
    kconserv: &Kconserv,
    lat: &StarLattice<'_>,
    opts: &EomOpts,
) -> Result<Vec<StarRoot>, PbcCcError> {
    if kind == Excitation::Ee {
        // `EOMEE` declares no `ccsd_star_contract` and no `*_star` driver
        // (`:1701-1704`): there is no EE-CCSD* upstream.
        return Err(PbcCcError::NotImplementedUpstream {
            upstream: "pbc/cc/eom_kccsd_ghf.py:1701",
            what: "EOMEE declares no ccsd_star_contract; there is no EE-CCSD* correction",
        });
    }
    let right = eom_kernel(
        kind,
        kshift,
        imds,
        padding,
        kconserv,
        &EomOpts {
            left: false,
            ..*opts
        },
    )?;
    let left = eom_kernel(
        kind,
        kshift,
        imds,
        padding,
        kconserv,
        &EomOpts {
            left: true,
            ..*opts
        },
    )?;
    let pairs = sort_left_right_eigensystem(&right, &left, 1e-6);
    match kind {
        Excitation::Ip => ipccsd_star_contract(&pairs, kshift, imds, padding, kconserv, lat),
        Excitation::Ea => eaccsd_star_contract(&pairs, kshift, imds, padding, kconserv, lat),
        Excitation::Ee => unreachable!("refused above"),
    }
}
