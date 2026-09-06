//! `kccsd_t_rhf` — the BLOCKED restricted k-point CCSD(T)
//! (plan 16-08 Task 2; `pyscf/pbc/cc/kccsd_t_rhf.py`).
//!
//! # The C kernel, and what replaces it
//!
//! `kccsd_t_rhf.py:236` is
//!
//! ```python
//! drv = _ccsd.libcc.CCsd_zcontract_t3T
//! ```
//!
//! a complex C contraction taking **24 raw data pointers** (`:229-230`) plus
//! `mo_offset` and `slices` blocking arrays. This port has no C and no `libcc`
//! — the zero-C constraint is the project's core value proposition, not an
//! accident — so the kernel is written in Rust and gated against
//! [`crate::kccsd_t_rhf_slow`], the loop-explicit form of the same energy and
//! the only oracle-free reference the blocked path has (`16-CONTEXT §1.7`).
//!
//! **What is ported and what is not.** The 24 pointers are the six `vvop`, six
//! `vooo_C` and twelve `t2T` slices `get_data_slices` (`:510`) selects; they are
//! a *marshalling* detail of handing numpy arrays to C, and the arrays
//! themselves — `transpose_t2` (`:366`), `create_eris_vvop` (`:392`),
//! `create_eris_vooo` (`:410`) — are ported here in full because they carry
//! real transpositions and conjugations. The pointer packing is not ported,
//! because in Rust the slices are simply borrowed.
//!
//! # The blocking IS the algorithm (`16-REVIEW.md §4.2`)
//!
//! `t3` is `nocc³ · nvir³` **per k-triple**, formed, consumed and discarded.
//! A port that materialised it over all k would trade a streaming problem for
//! an allocation no `§9.2` fixture can hold. The `(a0,a1,b0,b1,c0,c1)` virtual
//! blocking (`:270-278`) is ported, and the energy is INDEPENDENT of the block
//! size — 16-08 test 3 asserts exactly that, which is what catches a wrong
//! `mo_offset`/`slices` translation.
//!
//! # Where this differs from the slow path, and why it is faster
//!
//! Not the transposed arrays — those only change memory access. The real
//! saving is that `my_permuted_w[ki,kj,kk]` is **cached across the whole
//! `(ki,kj,kk)` loop** (`:281-296`) and the `R` combination then reads six
//! cached entries (`:325-330`), where `kccsd_t_rhf_slow.py:124-133` rebuilds
//! `get_permuted_w` six times — six W builds per triple instead of one.
//!
//! # `LARGE_DENOM`
//!
//! `_get_epqr` (`kccsd_t_rhf.py:30` imports `LARGE_DENOM` for it) fills the
//! triples denominator at padded orbitals. Arithmetic, `~1e-28`, not a skip
//! (`16-CONTEXT §3.3`).

use pyscf_algebra::oracle_sum;
use pyscf_pbc_lib::{KIdx, Kconserv, get_kconserv3};
use pyscf_pbc_mp::{PaddedMos, PaddingIdx, PaddingKind, padding_k_idx};

use crate::error::PbcCcError;
use crate::kccsd_rhf::LARGE_DENOM;
use crate::kccsd_t_rhf_slow::{VirBlock, tasks};
use crate::keris::{Blk, KEris};
use crate::zarr::{ZArr, einsum};

/// `transpose_t2(t2, ...)` — `kccsd_t_rhf.py:366-389`.
///
/// `out[ka,kb,kj] = t2[ki,kj,ka].transpose(2,3,1,0)`, i.e.
/// `t2T[ka,kb,kj][a,b,j,i] = t2[ki,kj,ka][i,j,a,b]`.
///
/// # Errors
/// Shape violations only.
pub fn transpose_t2(
    t2: &ZArr,
    nkpts: usize,
    nocc: usize,
    nvir: usize,
    kconserv: &Kconserv,
) -> Result<ZArr, PbcCcError> {
    let mut out = ZArr::zeros(&[nkpts, nkpts, nkpts, nvir, nvir, nocc, nocc]);
    for ki in 0..nkpts {
        for kj in 0..nkpts {
            for ka in 0..nkpts {
                let kb = kconserv.get(ki, ka, kj) as usize;
                let b = t2.slice_leading(&[ki, kj, ka])?.transpose(&[2, 3, 1, 0])?;
                out.set_leading(&[ka, kb, kj], &b)?;
            }
        }
    }
    Ok(out)
}

/// `create_eris_vvop(vovv, oovv, ...)` — `kccsd_t_rhf.py:392-408`.
///
/// `out[ki,kj,ka][:,:,:,nocc:] = vovv[kb,ka,kj].conj().transpose(3,2,1,0)` and
/// `out[ki,kj,ka][:,:,:,:nocc] = oovv[kb,ka,kj].conj().transpose(3,2,1,0)`.
/// **Both halves conjugate**, and that is one of this module's explicit
/// conjugation sites.
///
/// # Errors
/// Propagates the ERI access.
pub fn create_eris_vvop(eris: &KEris, kconserv: &Kconserv) -> Result<ZArr, PbcCcError> {
    let (nkpts, nocc, nvir, nmo) = (eris.nkpts, eris.nocc, eris.nvir, eris.nmo);
    let mut out = ZArr::zeros(&[nkpts, nkpts, nkpts, nvir, nvir, nocc, nmo]);
    for ki in 0..nkpts {
        for kj in 0..nkpts {
            for ka in 0..nkpts {
                let kb = kconserv.get(ki, ka, kj) as usize;
                let vv = eris
                    .blk(Blk::Vovv, kb, ka, kj)?
                    .conj()
                    .transpose(&[3, 2, 1, 0])?;
                let oo = eris
                    .blk(Blk::Oovv, kb, ka, kj)?
                    .conj()
                    .transpose(&[3, 2, 1, 0])?;
                let mut blk = ZArr::zeros(&[nvir, nvir, nocc, nmo]);
                for a in 0..nvir {
                    for b in 0..nvir {
                        for o in 0..nocc {
                            for p in 0..nmo {
                                let (re, im) = if p < nocc {
                                    oo.at(&[a, b, o, p])?
                                } else {
                                    vv.at(&[a, b, o, p - nocc])?
                                };
                                let f = ((a * nvir + b) * nocc + o) * nmo + p;
                                blk.data_mut().re[f] = re;
                                blk.data_mut().im[f] = im;
                            }
                        }
                    }
                }
                out.set_leading(&[ki, kj, ka], &blk)?;
            }
        }
    }
    Ok(out)
}

/// `create_eris_vooo(ooov, ...)` — `kccsd_t_rhf.py:410-425`.
///
/// `out[ki,kj,kb] = ooov[kb,kj,ka].conj().transpose(3,1,0,2)`, with
/// `kb = kconserv[ki,kj,ka]`. Upstream's own comment calls this "not exactly
/// chemist's notation, but close": physicist `<bj|ai>` → chemist `(ba|ji)`,
/// then the last two indices swapped.
///
/// # Errors
/// Propagates the ERI access.
pub fn create_eris_vooo(eris: &KEris, kconserv: &Kconserv) -> Result<ZArr, PbcCcError> {
    let (nkpts, nocc, nvir) = (eris.nkpts, eris.nocc, eris.nvir);
    let mut out = ZArr::zeros(&[nkpts, nkpts, nkpts, nvir, nocc, nocc, nocc]);
    for ki in 0..nkpts {
        for kj in 0..nkpts {
            for ka in 0..nkpts {
                let kb = kconserv.get(ki, kj, ka) as usize;
                let b = eris
                    .blk(Blk::Ooov, kb, kj, ka)?
                    .conj()
                    .transpose(&[3, 1, 0, 2])?;
                out.set_leading(&[ki, kj, kb], &b)?;
            }
        }
    }
    Ok(out)
}

struct Fast<'a> {
    t1t: ZArr,
    fvo: ZArr,
    t2t: &'a ZArr,
    vvop: &'a ZArr,
    vooo: &'a ZArr,
    kconserv: &'a Kconserv,
    nocc: usize,
    nvir: usize,
    nmo: usize,
}

impl Fast<'_> {
    /// `get_w` — `kccsd_t_rhf.py:118-125`, on the transposed arrays.
    #[allow(clippy::too_many_arguments)]
    fn get_w(
        &self,
        ki: usize,
        kj: usize,
        kk: usize,
        ka: usize,
        kb: usize,
        kc: usize,
        blk: VirBlock,
    ) -> Result<ZArr, PbcCcError> {
        let (nocc, nvir, nmo) = (self.nocc, self.nvir, self.nmo);
        let km = self.kconserv.get(kc, kk, kb) as usize;
        let kf = self.kconserv.get(kk, kc, kj) as usize;

        let a = self.t2t.slice_leading(&[kc, kf, kj])?.slice_axes(&[
            blk.c,
            (0, nvir),
            (0, nocc),
            (0, nocc),
        ])?;
        let b = self.vvop.slice_leading(&[ka, kb, ki])?.slice_axes(&[
            blk.a,
            blk.b,
            (0, nocc),
            (nocc, nmo),
        ])?;
        let mut out = einsum("cfjk,abif->abcijk", &[&a, &b])?;

        let a = self.t2t.slice_leading(&[kc, kb, km])?.slice_axes(&[
            blk.c,
            blk.b,
            (0, nocc),
            (0, nocc),
        ])?;
        let b = self.vooo.slice_leading(&[ka, ki, kj])?.slice_axes(&[
            blk.a,
            (0, nocc),
            (0, nocc),
            (0, nocc),
        ])?;
        out.sub_assign(&einsum("cbmk,aijm->abcijk", &[&a, &b])?)?;
        Ok(out)
    }

    #[allow(clippy::too_many_arguments)]
    fn get_permuted_w(
        &self,
        ki: usize,
        kj: usize,
        kk: usize,
        ka: usize,
        kb: usize,
        kc: usize,
        blk: VirBlock,
    ) -> Result<ZArr, PbcCcError> {
        let p = |x, y, z| VirBlock { a: x, b: y, c: z };
        let mut out = self.get_w(ki, kj, kk, ka, kb, kc, p(blk.a, blk.b, blk.c))?;
        out.add_assign(
            &self
                .get_w(kj, kk, ki, kb, kc, ka, p(blk.b, blk.c, blk.a))?
                .transpose(&[2, 0, 1, 5, 3, 4])?,
        )?;
        out.add_assign(
            &self
                .get_w(kk, ki, kj, kc, ka, kb, p(blk.c, blk.a, blk.b))?
                .transpose(&[1, 2, 0, 4, 5, 3])?,
        )?;
        out.add_assign(
            &self
                .get_w(ki, kk, kj, ka, kc, kb, p(blk.a, blk.c, blk.b))?
                .transpose(&[0, 2, 1, 3, 5, 4])?,
        )?;
        out.add_assign(
            &self
                .get_w(kk, kj, ki, kc, kb, ka, p(blk.c, blk.b, blk.a))?
                .transpose(&[2, 1, 0, 5, 4, 3])?,
        )?;
        out.add_assign(
            &self
                .get_w(kj, ki, kk, kb, ka, kc, p(blk.b, blk.a, blk.c))?
                .transpose(&[1, 0, 2, 4, 3, 5])?,
        )?;
        Ok(out)
    }

    /// `get_v` — `kccsd_t_rhf.py:159-175`. **Carries the `0.5` factors**, which
    /// is why the fast path's `pwijk` is `w + v` while the slow path's is
    /// `w + 0.5*v`.
    #[allow(clippy::too_many_arguments)]
    fn get_v(
        &self,
        ki: usize,
        kj: usize,
        kk: usize,
        ka: usize,
        kb: usize,
        kc: usize,
        blk: VirBlock,
    ) -> Result<ZArr, PbcCcError> {
        let (nocc, nvir) = (self.nocc, self.nvir);
        let shape = [
            blk.a.1 - blk.a.0,
            blk.b.1 - blk.b.0,
            blk.c.1 - blk.c.0,
            nocc,
            nocc,
            nocc,
        ];
        let mut out = ZArr::zeros(&shape);
        if kk != kc {
            return Ok(out);
        }
        let mut t1k = self.t1t.slice_leading(&[kk])?.slice_axes(&[blk.c, (0, nocc)])?;
        t1k.scale(0.5);
        let vvop = self.vvop.slice_leading(&[kb, ka, kj])?.slice_axes(&[
            blk.b,
            blk.a,
            (0, nocc),
            (0, nocc),
        ])?;
        out.add_assign(&einsum("ck,baji->abcijk", &[&t1k, &vvop])?)?;

        let mut f = self.fvo.slice_leading(&[kk])?.slice_axes(&[blk.c, (0, nocc)])?;
        f.scale(0.5);
        let t2t = self.t2t.slice_leading(&[kb, ka, ki])?.slice_axes(&[
            blk.b,
            blk.a,
            (0, nocc),
            (0, nocc),
        ])?;
        let _ = nvir;
        out.add_assign(&einsum("ck,baij->abcijk", &[&f, &t2t])?)?;
        Ok(out)
    }

    #[allow(clippy::too_many_arguments)]
    fn get_permuted_v(
        &self,
        ki: usize,
        kj: usize,
        kk: usize,
        ka: usize,
        kb: usize,
        kc: usize,
        blk: VirBlock,
    ) -> Result<ZArr, PbcCcError> {
        let p = |x, y, z| VirBlock { a: x, b: y, c: z };
        let mut out = self.get_v(ki, kj, kk, ka, kb, kc, p(blk.a, blk.b, blk.c))?;
        out.add_assign(
            &self
                .get_v(kj, kk, ki, kb, kc, ka, p(blk.b, blk.c, blk.a))?
                .transpose(&[2, 0, 1, 5, 3, 4])?,
        )?;
        out.add_assign(
            &self
                .get_v(kk, ki, kj, kc, ka, kb, p(blk.c, blk.a, blk.b))?
                .transpose(&[1, 2, 0, 4, 5, 3])?,
        )?;
        out.add_assign(
            &self
                .get_v(ki, kk, kj, ka, kc, kb, p(blk.a, blk.c, blk.b))?
                .transpose(&[0, 2, 1, 3, 5, 4])?,
        )?;
        out.add_assign(
            &self
                .get_v(kk, kj, ki, kc, kb, ka, p(blk.c, blk.b, blk.a))?
                .transpose(&[2, 1, 0, 5, 4, 3])?,
        )?;
        out.add_assign(
            &self
                .get_v(kj, ki, kk, kb, ka, kc, p(blk.b, blk.a, blk.c))?
                .transpose(&[1, 0, 2, 4, 3, 5])?,
        )?;
        Ok(out)
    }
}

/// `kernel(mycc, eris, t1, t2)` — `kccsd_t_rhf.py:44-340`.
///
/// # Errors
/// Propagates the ERI access, the padding surface and every shape check.
#[allow(clippy::too_many_arguments)]
pub fn kernel(
    eris: &KEris,
    padded: &PaddedMos,
    t1: &ZArr,
    t2: &ZArr,
    kconserv: &Kconserv,
    a: &[[f64; 3]; 3],
    kpts: &[[f64; 3]],
    vir_blksize: Option<usize>,
) -> Result<f64, PbcCcError> {
    kernel_with_stats(eris, padded, t1, t2, kconserv, a, kpts, vir_blksize).map(|(e, _)| e)
}

/// [`kernel`], additionally returning the PEAK LIVE bytes of the `t3`-class
/// `w`/`v` cache.
///
/// **This is what makes `16-REVIEW.md §4.2`'s claim testable rather than
/// aspirational** (16-08 test 6). The cache is allocated per `(ka, kb, block)`
/// and dropped at the end of that block, so the peak is
/// `2 · nkpts³ · na·nb·nc · nocc³ · 16` — bounded by ONE virtual block, never
/// by `nkpts³ · nvir³ · nocc³`. Blocking the virtuals reduces it by
/// `(blksize/nvir)³`, and the test asserts exactly that ratio against a
/// literal.
///
/// # Errors
/// As [`kernel`].
#[allow(clippy::too_many_arguments)]
pub fn kernel_with_stats(
    eris: &KEris,
    padded: &PaddedMos,
    t1: &ZArr,
    t2: &ZArr,
    kconserv: &Kconserv,
    a: &[[f64; 3]; 3],
    kpts: &[[f64; 3]],
    vir_blksize: Option<usize>,
) -> Result<(f64, usize), PbcCcError> {
    let (nkpts, nocc, nvir, nmo) = (eris.nkpts, eris.nocc, eris.nvir, eris.nmo);
    let mo_e_o: Vec<Vec<f64>> = eris.mo_energy.iter().map(|e| e[..nocc].to_vec()).collect();
    let mo_e_v: Vec<Vec<f64>> = eris.mo_energy.iter().map(|e| e[nocc..].to_vec()).collect();
    let (nz_o, nz_v) =
        match padding_k_idx(&padded.nmo_per_kpt, &padded.nocc_per_kpt, PaddingKind::Split) {
            Ok(PaddingIdx::Split { occupied, virtuals }) => (occupied, virtuals),
            Ok(_) => return Err(PbcCcError::Shape("padding_k_idx returned a joint set".into())),
            Err(e) => return Err(PbcCcError::Shape(format!("padding_k_idx: {e}"))),
        };

    // `:104` create_t3_eris — the transposed arrays, built once.
    let t2t = transpose_t2(t2, nkpts, nocc, nvir, kconserv)?;
    let vvop = create_eris_vvop(eris, kconserv)?;
    let vooo = create_eris_vooo(eris, kconserv)?;
    // `:105-106` t1T = t1.transpose(0,2,1), fvo = fov.transpose(0,2,1).
    let t1t = t1.transpose(&[0, 2, 1])?;
    let mut fvo = ZArr::zeros(&[nkpts, nvir, nocc]);
    for k in 0..nkpts {
        fvo.set_leading(&[k], &eris.fov(k)?.transpose(&[1, 0])?)?;
    }

    let fast = Fast {
        t1t,
        fvo,
        t2t: &t2t,
        vvop: &vvop,
        vooo: &vooo,
        kconserv,
        nocc,
        nvir,
        nmo,
    };

    let task_list = tasks(nvir, vir_blksize.unwrap_or(nvir));
    let mut terms_re: Vec<f64> = Vec::new();
    let mut terms_im: Vec<f64> = Vec::new();
    let mut peak_cache_bytes = 0_usize;

    for ka in 0..nkpts {
        for kb in 0..=ka {
            for &blk in &task_list {
                let (na, nb, nc) = (
                    blk.a.1 - blk.a.0,
                    blk.b.1 - blk.b.0,
                    blk.c.1 - blk.c.0,
                );
                let bshape = [na, nb, nc, nocc, nocc, nocc];
                // `:281-296` — cache `w` and `v` over the whole (ki,kj,kk)
                // loop. THIS is the fast path's saving: the `R` combination
                // below reads six cached entries instead of rebuilding
                // `get_permuted_w` six times.
                //
                // The cache is per (ka, kb, block), so the live `t3`-class
                // memory is `nkpts³ · na·nb·nc · nocc³ · 16` — bounded by ONE
                // virtual block, never by `nkpts³ · nvir³ · nocc³`
                // (`16-REVIEW.md §4.2`).
                let mut wc: Vec<Option<ZArr>> = vec![None; nkpts * nkpts * nkpts];
                let mut vc: Vec<Option<ZArr>> = vec![None; nkpts * nkpts * nkpts];
                // The peak this block's cache CAN reach: two caches of
                // `nkpts³` blocks of `na·nb·nc·nocc³` complex elements.
                peak_cache_bytes = peak_cache_bytes.max(
                    2 * nkpts.pow(3) * na * nb * nc * nocc.pow(3) * 16,
                );
                let at = |x: usize, y: usize, z: usize| (x * nkpts + y) * nkpts + z;

                for ki in 0..nkpts {
                    for kj in 0..nkpts {
                        for kk in 0..nkpts {
                            let kc = kconserv3(a, kpts, ki, kj, kk, ka, kb);
                            if !(ka >= kb && kb >= kc) {
                                continue;
                            }
                            wc[at(ki, kj, kk)] =
                                Some(fast.get_permuted_w(ki, kj, kk, ka, kb, kc, blk)?);
                            vc[at(ki, kj, kk)] =
                                Some(fast.get_permuted_v(ki, kj, kk, ka, kb, kc, blk)?);
                        }
                    }
                }

                for ki in 0..nkpts {
                    for kj in 0..nkpts {
                        for kk in 0..nkpts {
                            let kc = kconserv3(a, kpts, ki, kj, kk, ka, kb);
                            if !(ka >= kb && kb >= kc) {
                                continue;
                            }
                            let symm_kpt = if ka == kb && kb == kc {
                                1.0
                            } else if ka == kb || kb == kc {
                                3.0
                            } else {
                                6.0
                            };

                            let eijk = epqr3(&mo_e_o, &nz_o, ki, kj, kk, nocc);
                            let eabc = epqr3_v(&mo_e_v, &nz_v, ka, kb, kc, nvir);

                            let w = wc[at(ki, kj, kk)]
                                .as_ref()
                                .ok_or_else(|| PbcCcError::Shape("missing cached w".into()))?;
                            let v = vc[at(ki, kj, kk)]
                                .as_ref()
                                .ok_or_else(|| PbcCcError::Shape("missing cached v".into()))?;
                            let mut pwijk = w.clone();
                            pwijk.add_assign(v)?;

                            // `:325-330` — the R combination over cached W's.
                            let mut rwijk = w.clone();
                            rwijk.scale(4.0);
                            let take = |x: usize, y: usize, z: usize| -> Result<&ZArr, PbcCcError> {
                                wc[at(x, y, z)]
                                    .as_ref()
                                    .ok_or_else(|| PbcCcError::Shape("missing cached w".into()))
                            };
                            rwijk.zip_assign(
                                &take(kj, kk, ki)?.transpose(&[0, 1, 2, 5, 3, 4])?,
                                1.0,
                            )?;
                            rwijk.zip_assign(
                                &take(kk, ki, kj)?.transpose(&[0, 1, 2, 4, 5, 3])?,
                                1.0,
                            )?;
                            rwijk.zip_assign(
                                &take(ki, kk, kj)?.transpose(&[0, 1, 2, 3, 5, 4])?,
                                -2.0,
                            )?;
                            rwijk.zip_assign(
                                &take(kk, kj, ki)?.transpose(&[0, 1, 2, 5, 4, 3])?,
                                -2.0,
                            )?;
                            rwijk.zip_assign(
                                &take(kj, ki, kk)?.transpose(&[0, 1, 2, 4, 3, 5])?,
                                -2.0,
                            )?;

                            debug_assert_eq!(rwijk.shape(), &bshape);
                            for x in 0..na {
                                for y in 0..nb {
                                    for z in 0..nc {
                                        // `:317-322` eabc carries fac=[-1,-1,-1]
                                        // and eijkabc = eijk + eabc.
                                        let d_abc = eabc[((x + blk.a.0) * nvir + (y + blk.b.0))
                                            * nvir
                                            + (z + blk.c.0)];
                                        for i in 0..nocc {
                                            for j in 0..nocc {
                                                for k in 0..nocc {
                                                    let d = eijk[(i * nocc + j) * nocc + k] + d_abc;
                                                    let f = ((((x * nb + y) * nc + z) * nocc + i)
                                                        * nocc
                                                        + j)
                                                        * nocc
                                                        + k;
                                                    rwijk.data_mut().re[f] /= d;
                                                    rwijk.data_mut().im[f] /= d;
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                            // `:331` — the conjugation is on `pwijk` here and on
                            // `rwijk` in the slow path; the two differ by an
                            // overall complex conjugate, so the REAL parts —
                            // the only thing either returns — agree exactly.
                            let (re, im) =
                                einsum("abcijk,abcijk->", &[&rwijk, &pwijk.conj()])?.at(&[])?;
                            terms_re.push(symm_kpt * re);
                            terms_im.push(symm_kpt * im);
                        }
                    }
                }
            }
        }
    }

    let re = oracle_sum(&terms_re) / 3.0 / nkpts as f64;
    let im = oracle_sum(&terms_im) / 3.0 / nkpts as f64;
    if im.abs() > 1e-4 {
        tracing::warn!(
            imaginary = im,
            "non-zero imaginary part of the CCSD(T) energy (kccsd_t_rhf.py:335)"
        );
    }
    Ok((re, peak_cache_bytes))
}

fn kconserv3(
    a: &[[f64; 3]; 3],
    kpts: &[[f64; 3]],
    ki: usize,
    kj: usize,
    kk: usize,
    ka: usize,
    kb: usize,
) -> usize {
    get_kconserv3(
        a,
        kpts,
        &[
            KIdx::One(ki),
            KIdx::One(kj),
            KIdx::One(kk),
            KIdx::One(ka),
            KIdx::One(kb),
        ],
    )
    .data[0] as usize
}

/// `_get_epqr` for three OCCUPIED indices, `fac = [1,1,1]`
/// (`kccsd_t_rhf.py:300-302`). Padded entries carry [`LARGE_DENOM`].
fn epqr3(
    mo_e: &[Vec<f64>],
    nz: &[Vec<usize>],
    kp: usize,
    kq: usize,
    kr: usize,
    n: usize,
) -> Vec<f64> {
    let mut out = vec![LARGE_DENOM; n * n * n];
    for &i in &nz[kp] {
        if i >= n {
            continue;
        }
        for &j in &nz[kq] {
            if j >= n {
                continue;
            }
            for &k in &nz[kr] {
                if k >= n {
                    continue;
                }
                out[(i * n + j) * n + k] = mo_e[kp][i] + mo_e[kq][j] + mo_e[kr][k];
            }
        }
    }
    out
}

/// `_get_epqr` for three VIRTUAL indices with `fac = [-1,-1,-1]`
/// (`kccsd_t_rhf.py:317-320`).
fn epqr3_v(
    mo_e: &[Vec<f64>],
    nz: &[Vec<usize>],
    kp: usize,
    kq: usize,
    kr: usize,
    n: usize,
) -> Vec<f64> {
    let mut out = vec![LARGE_DENOM; n * n * n];
    for &i in &nz[kp] {
        if i >= n {
            continue;
        }
        for &j in &nz[kq] {
            if j >= n {
                continue;
            }
            for &k in &nz[kr] {
                if k >= n {
                    continue;
                }
                out[(i * n + j) * n + k] = -(mo_e[kp][i] + mo_e[kq][j] + mo_e[kr][k]);
            }
        }
    }
    out
}
