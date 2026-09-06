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
use pyscf_pbc_lib::Kconserv;

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
    let (nkpts, nocc, nvir) = (imds.eris.nkpts, imds.eris.nocc, imds.eris.nvir);
    let size = match kind {
        Excitation::Ip => ip_vector_size(nkpts, nocc, nvir),
        Excitation::Ea => ea_vector_size(nkpts, nocc, nvir, kshift, kconserv),
    };

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
