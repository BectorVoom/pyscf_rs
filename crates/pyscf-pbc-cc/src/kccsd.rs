//! `KGCCSD` — spin-orbital k-point coupled cluster (plan 16-07;
//! `pyscf/pbc/cc/kccsd.py`).
//!
//! # The ERI convention, named once
//!
//! Every block here is the ANTISYMMETRISED PHYSICIST integral `<pq||rs>`, not
//! the chemist `(pq|rs)` that [`crate::keris`] holds. `_make_eris_incore`
//! (`kccsd.py:471-640`) builds it in three steps and all three carry real
//! index work:
//!
//! ```text
//! :577-596  eri[kp,kq,kr] = Σ over the four spin-block combinations of
//!                           ao2mo(mo_a/mo_b …)          — chemist (pq|rs)
//! :621      eri -= eri.transpose(2,1,0,5,4,3,6)         — ANTISYMMETRISE,
//!                                                          k-axes 0↔2 AND
//!                                                          orbital axes 3↔5
//! :623      eri  = eri.transpose(0,2,1,3,5,4,6)         — chemist → physicist
//! ```
//!
//! The antisymmetrisation transposes the K-AXES as well as the orbital ones,
//! because `(ps|rq)` and `(rq|ps)` are the same integral and upstream is "not
//! tracking the k-point of orbital `s`" (its own comment at `:619-620`).
//! Getting either transpose wrong is the 14-05 `decompose_j2c` class of defect
//! (`16-CONTEXT §3.4`, `+6 306 866.73 Ha`), so both are written out here and
//! the convention lives in the type name.
//!
//! # `kccsd.py:414` is a live refusal; `:486` is NOT
//!
//! `:414` `raise NotImplementedError` is reachable and is ported as
//! [`PbcCcError::NotImplementedUpstream`] with its upstream line. `:486`
//! (`#    raise NotImplementedError('Different occupancies…')`) is COMMENTED
//! OUT upstream and is deliberately not ported as a refusal — porting it would
//! invent a restriction upstream does not impose.
//!
//! # Memory
//!
//! The spin-orbital basis doubles both `nocc` and `nvir`, so every tensor of
//! the `nvir⁴` class is **16×** its RHF counterpart at the same cell
//! (`16-REVIEW.md §2.3`): diamond `gth-szv` 2×2×2 `vvvv` is 32 MiB against the
//! RHF 2 MiB, and `gth-dzvp` 2×2×2 is **28.6 GiB**. The seven blocks go through
//! [`KBlocks`], whose tier comes from an exact byte count.

use std::sync::Arc;

use pyscf_algebra::{CTensor, oracle_sum};
use pyscf_pbc_df::{MoCoeff, PeriodicDf};
use pyscf_pbc_lib::{Kconserv, KptsHelper};
use pyscf_pbc_mp::{PaddedMos, PaddingIdx, PaddingKind, padding_k_idx};
use pyscf_runtime::ZWorkspacePool;

use crate::error::PbcCcError;
use crate::kccsd_rhf::LARGE_DENOM;
use crate::kintermediates as imdk;
use crate::ktensor::{KBlocks, Tier};
use crate::zarr::{ZArr, einsum, einsum_scaled};

/// Which of the seven spin-orbital blocks. **Note the set differs from the RHF
/// one** (`kccsd.py:627-633`): `ovoo` and `ovvv` here, `voov` and `vovv`
/// there.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GBlk {
    Oooo,
    Ooov,
    Ovoo,
    Oovv,
    Ovov,
    Ovvv,
    Vvvv,
}

impl GBlk {
    fn spaces(self) -> [bool; 4] {
        match self {
            GBlk::Oooo => [false, false, false, false],
            GBlk::Ooov => [false, false, false, true],
            GBlk::Ovoo => [false, true, false, false],
            GBlk::Oovv => [false, false, true, true],
            GBlk::Ovov => [false, true, false, true],
            GBlk::Ovvv => [false, true, true, true],
            GBlk::Vvvv => [true, true, true, true],
        }
    }

    /// Per-block dimensions at `(nocc, nvir)`.
    pub fn dims(self, nocc: usize, nvir: usize) -> [usize; 4] {
        self.spaces().map(|v| if v { nvir } else { nocc })
    }

    /// A short name for diagnostics.
    pub fn name(self) -> &'static str {
        match self {
            GBlk::Oooo => "oooo",
            GBlk::Ooov => "ooov",
            GBlk::Ovoo => "ovoo",
            GBlk::Oovv => "oovv",
            GBlk::Ovov => "ovov",
            GBlk::Ovvv => "ovvv",
            GBlk::Vvvv => "vvvv",
        }
    }

    /// The seven, in upstream's declaration order (`kccsd.py:627-633`).
    pub const ALL: [GBlk; 7] = [
        GBlk::Oooo,
        GBlk::Ooov,
        GBlk::Ovoo,
        GBlk::Oovv,
        GBlk::Ovov,
        GBlk::Ovvv,
        GBlk::Vvvv,
    ];
}

/// The spin-orbital `_PhysicistsERIs` — `gccsd._PhysicistsERIs()` at
/// `kccsd.py:477`, filled by `_make_eris_incore`.
#[derive(Debug)]
pub struct KgEris {
    pub nkpts: usize,
    pub nocc: usize,
    pub nmo: usize,
    pub nvir: usize,
    /// `[nkpts, nmo, nmo]`.
    pub fock: ZArr,
    /// Madelung-adjusted, unless the caller suppressed it.
    pub mo_energy: Vec<Vec<f64>>,
    blocks: Vec<(GBlk, KBlocks)>,
}

impl KgEris {
    /// Build the seven blocks from an ALREADY-BUILT spin-orbital mean field.
    ///
    /// `mo_coeff[k]` is `(2 · nao_scalar) × nmo`: the bottom `nao_scalar` rows
    /// are one spin block and the top `nao_scalar` the other
    /// (`kccsd.py:573-575`).
    ///
    /// # Errors
    /// Propagates the density-fitting builder, the complex arena and every
    /// shape check.
    pub fn from_parts(
        with_df: &dyn PeriodicDf,
        khelper: &KptsHelper,
        mo_coeff: &[MoCoeff],
        fock: ZArr,
        mo_energy: Vec<Vec<f64>>,
        nocc: usize,
        max_memory_bytes: usize,
    ) -> Result<Self, PbcCcError> {
        let nkpts = with_df.kpts().len();
        let nmo = mo_coeff[0].nmo;
        let nvir = nmo - nocc;
        let nao2 = mo_coeff[0].nao;
        if nao2 % 2 != 0 {
            return Err(PbcCcError::Shape(format!(
                "KGCCSD needs a spin-orbital MO block with an even row count, got {nao2}"
            )));
        }
        let nao = nao2 / 2;

        // `:573-575` — the two spin blocks.
        let split = |m: &MoCoeff, top: bool| -> MoCoeff {
            let off = if top { nao } else { 0 };
            let mut c = CTensor::zeros(nao * nmo);
            for a in 0..nao {
                for p in 0..nmo {
                    c.re[a * nmo + p] = m.c.re[(a + off) * nmo + p];
                    c.im[a * nmo + p] = m.c.im[(a + off) * nmo + p];
                }
            }
            MoCoeff::new(nao, nmo, c)
        };
        let mo_a: Vec<MoCoeff> = mo_coeff.iter().map(|m| split(m, false)).collect();
        let mo_b: Vec<MoCoeff> = mo_coeff.iter().map(|m| split(m, true)).collect();

        // `:577-596` — the chemist-notation four-spin-block sum.
        let mut eri = ZArr::zeros(&[nkpts, nkpts, nkpts, nmo, nmo, nmo, nmo]);
        for kp in 0..nkpts {
            for kq in 0..nkpts {
                for kr in 0..nkpts {
                    let ks = khelper.kconserv.get(kp, kq, kr) as usize;
                    let mut acc = ZArr::zeros(&[nmo, nmo, nmo, nmo]);
                    for (a, b) in [
                        (&mo_a, &mo_a),
                        (&mo_b, &mo_b),
                        (&mo_a, &mo_b),
                        (&mo_b, &mo_a),
                    ] {
                        let e = with_df
                            .ao2mo(
                                [&a[kp], &a[kq], &b[kr], &b[ks]],
                                [kp, kq, kr, ks],
                                false,
                            )
                            .map_err(|e| PbcCcError::Shape(format!("ao2mo: {e}")))?
                            .restore_s1();
                        acc.add_assign(&ZArr::from_ctensor(&[nmo; 4], e.data)?)?;
                    }
                    eri.set_leading(&[kp, kq, kr], &acc)?;
                }
            }
        }

        // `:621` eri -= eri.transpose(2,1,0,5,4,3,6) — k-axes 0↔2 AND orbital
        // axes 3↔5. Built into a fresh array because the operation reads
        // elements it would otherwise have already overwritten.
        let swapped = eri.transpose(&[2, 1, 0, 5, 4, 3, 6])?;
        eri.sub_assign(&swapped)?;
        // `:623` chemist -> physicist.
        let eri = eri.transpose(&[0, 2, 1, 3, 5, 4, 6])?;

        let inv = 1.0 / nkpts as f64;
        let pool = Arc::new(ZWorkspacePool::new(max_memory_bytes.max(1)));
        let mut blocks: Vec<(GBlk, KBlocks)> = Vec::with_capacity(7);
        for b in GBlk::ALL {
            let d = b.dims(nocc, nvir);
            let t = KBlocks::with_budget(&pool, nkpts, &d, max_memory_bytes)?;
            let sp = b.spaces();
            for k0 in 0..nkpts {
                for k1 in 0..nkpts {
                    for k2 in 0..nkpts {
                        let full = eri.slice_leading(&[k0, k1, k2])?;
                        let mut out = ZArr::zeros(&d);
                        for i0 in 0..d[0] {
                            for i1 in 0..d[1] {
                                for i2 in 0..d[2] {
                                    for i3 in 0..d[3] {
                                        let s = [
                                            if sp[0] { nocc + i0 } else { i0 },
                                            if sp[1] { nocc + i1 } else { i1 },
                                            if sp[2] { nocc + i2 } else { i2 },
                                            if sp[3] { nocc + i3 } else { i3 },
                                        ];
                                        let (re, im) = full.at(&s)?;
                                        let f =
                                            ((i0 * d[1] + i1) * d[2] + i2) * d[3] + i3;
                                        out.data_mut().re[f] = re * inv;
                                        out.data_mut().im[f] = im * inv;
                                    }
                                }
                            }
                        }
                        t.set([k0, k1, k2], &out)?;
                    }
                }
            }
            blocks.push((b, t));
        }

        Ok(Self {
            nkpts,
            nocc,
            nmo,
            nvir,
            fock,
            mo_energy,
            blocks,
        })
    }

    /// One block at a k-triple.
    ///
    /// # Errors
    /// [`PbcCcError`] for an unknown block or a bad k-address.
    pub fn blk(&self, b: GBlk, k0: usize, k1: usize, k2: usize) -> Result<ZArr, PbcCcError> {
        self.tensor(b)?.get([k0, k1, k2])
    }

    /// The blocks over a FREE MIDDLE k-index — `eris.oovv[km,:,ke]`.
    ///
    /// **Distinct from [`KgEris::blk_free1`], and the distinction is silent**:
    /// both produce the same SHAPE, so swapping them is a wrong number no
    /// shape check catches. `cc_Wovvo` (`kintermediates.py:183`) wants this
    /// one; `Soovv_tmp` in the RHF `cc_Wvoov` wants the other.
    ///
    /// # Errors
    /// As [`KgEris::blk`].
    pub fn blk_free_mid(&self, b: GBlk, k0: usize, k2: usize) -> Result<ZArr, PbcCcError> {
        let d = b.dims(self.nocc, self.nvir);
        let mut shape = vec![self.nkpts];
        shape.extend_from_slice(&d);
        let mut out = ZArr::zeros(&shape);
        for k in 0..self.nkpts {
            out.set_leading(&[k], &self.blk(b, k0, k, k2)?)?;
        }
        Ok(out)
    }

    /// The blocks over a FREE FIRST k-index — `eris.oovv[:,kk,kc]`.
    ///
    /// # Errors
    /// As [`KgEris::blk`].
    pub fn blk_free1(&self, b: GBlk, k1: usize, k2: usize) -> Result<ZArr, PbcCcError> {
        let d = b.dims(self.nocc, self.nvir);
        let mut shape = vec![self.nkpts];
        shape.extend_from_slice(&d);
        let mut out = ZArr::zeros(&shape);
        for k in 0..self.nkpts {
            out.set_leading(&[k], &self.blk(b, k, k1, k2)?)?;
        }
        Ok(out)
    }

    /// The blocks over a FREE THIRD k-index — `eris.oovv[km,kn]`.
    ///
    /// # Errors
    /// As [`KgEris::blk`].
    pub fn blk_free2(&self, b: GBlk, k0: usize, k1: usize) -> Result<ZArr, PbcCcError> {
        let d = b.dims(self.nocc, self.nvir);
        let mut shape = vec![self.nkpts];
        shape.extend_from_slice(&d);
        let mut out = ZArr::zeros(&shape);
        for k in 0..self.nkpts {
            out.set_leading(&[k], &self.blk(b, k0, k1, k)?)?;
        }
        Ok(out)
    }

    /// The storage tier a block landed in.
    ///
    /// # Errors
    /// [`PbcCcError`] for an unknown block.
    pub fn tier(&self, b: GBlk) -> Result<Tier, PbcCcError> {
        Ok(self.tensor(b)?.tier())
    }

    /// `fock[k][:nocc, nocc:]`.
    ///
    /// # Errors
    /// [`PbcCcError::Shape`] on a bad index.
    pub fn fov(&self, k: usize) -> Result<ZArr, PbcCcError> {
        self.fock_sub(k, 0, self.nocc, self.nocc, self.nmo)
    }

    /// `fock[k][:nocc, :nocc]`.
    ///
    /// # Errors
    /// As [`KgEris::fov`].
    pub fn foo(&self, k: usize) -> Result<ZArr, PbcCcError> {
        self.fock_sub(k, 0, self.nocc, 0, self.nocc)
    }

    /// `fock[k][nocc:, nocc:]`.
    ///
    /// # Errors
    /// As [`KgEris::fov`].
    pub fn fvv(&self, k: usize) -> Result<ZArr, PbcCcError> {
        self.fock_sub(k, self.nocc, self.nmo, self.nocc, self.nmo)
    }

    fn fock_sub(
        &self,
        k: usize,
        r0: usize,
        r1: usize,
        c0: usize,
        c1: usize,
    ) -> Result<ZArr, PbcCcError> {
        self.fock
            .slice_leading(&[k])?
            .slice_axes(&[(r0, r1), (c0, c1)])
    }

    fn tensor(&self, b: GBlk) -> Result<&KBlocks, PbcCcError> {
        self.blocks
            .iter()
            .find(|(x, _)| *x == b)
            .map(|(_, t)| t)
            .ok_or_else(|| PbcCcError::Shape(format!("no {} block", b.name())))
    }
}

/// `energy(cc, t1, t2, eris)` — `kccsd.py:47-65`.
///
/// # Errors
/// Propagates the ERI access and every shape check.
pub fn energy(t1: &ZArr, t2: &ZArr, eris: &KgEris) -> Result<f64, PbcCcError> {
    let (nk, nocc, nvir) = (eris.nkpts, eris.nocc, eris.nvir);
    let mut re: Vec<f64> = Vec::new();
    let mut im: Vec<f64> = Vec::new();
    for ki in 0..nk {
        let (a, b) = einsum(
            "ia,ia->",
            &[&eris.fov(ki)?, &t1.slice_leading(&[ki])?],
        )?
        .at(&[])?;
        re.push(a);
        im.push(b);
    }
    // `:54-60` t1t1[ki,kj,ki] = einsum('ia,jb->ijab', t1[ki], t1[kj]);
    //          tau = t2 + 2*t1t1
    let mut tau = t2.clone();
    for ki in 0..nk {
        let ka = ki;
        for kj in 0..nk {
            let mut blk = tau.slice_leading(&[ki, kj, ka])?;
            blk.zip_assign(
                &einsum(
                    "ia,jb->ijab",
                    &[&t1.slice_leading(&[ki])?, &t1.slice_leading(&[kj])?],
                )?,
                2.0,
            )?;
            tau.set_leading(&[ki, kj, ka], &blk)?;
        }
    }
    // `:61` e += 0.25 * dot(tau.flatten(), oovv.flatten()) — an UNCONJUGATED
    // flat dot, so `oracle_zdotu`'s pattern, expressed here as the einsum it
    // is (see `crate::zarr`).
    for ki in 0..nk {
        for kj in 0..nk {
            for ka in 0..nk {
                let (a, b) = einsum_scaled(
                    "ijab,ijab->",
                    &[
                        &tau.slice_leading(&[ki, kj, ka])?,
                        &eris.blk(GBlk::Oovv, ki, kj, ka)?,
                    ],
                    0.25,
                )?
                .at(&[])?;
                re.push(a);
                im.push(b);
            }
        }
    }
    let _ = (nocc, nvir);
    let e_re = oracle_sum(&re) / nk as f64;
    let e_im = oracle_sum(&im) / nk as f64;
    if e_im.abs() > 1e-4 {
        tracing::warn!(
            imaginary = e_im,
            "non-zero imaginary part in the KGCCSD energy (kccsd.py:64)"
        );
    }
    Ok(e_re)
}

/// `update_amps(cc, t1, t2, eris)` — `kccsd.py:68-220`.
///
/// # Errors
/// Propagates every intermediate build and shape check.
pub fn update_amps(
    t1: &ZArr,
    t2: &ZArr,
    eris: &KgEris,
    padded: &PaddedMos,
    kconserv: &Kconserv,
    level_shift: f64,
) -> Result<(ZArr, ZArr), PbcCcError> {
    let (nk, nocc, nvir) = (eris.nkpts, eris.nocc, eris.nvir);
    let mo_e_o: Vec<Vec<f64>> = eris.mo_energy.iter().map(|e| e[..nocc].to_vec()).collect();
    let mo_e_v: Vec<Vec<f64>> = eris
        .mo_energy
        .iter()
        .map(|e| e[nocc..].iter().map(|x| x + level_shift).collect())
        .collect();
    let (nz_o, nz_v) =
        match padding_k_idx(&padded.nmo_per_kpt, &padded.nocc_per_kpt, PaddingKind::Split) {
            Ok(PaddingIdx::Split { occupied, virtuals }) => (occupied, virtuals),
            Ok(_) => return Err(PbcCcError::Shape("padding_k_idx returned a joint set".into())),
            Err(e) => return Err(PbcCcError::Shape(format!("padding_k_idx: {e}"))),
        };

    let tau = imdk::make_tau(t2, t1, t1, kconserv, nk, nocc, nvir, 1.0)?;
    let mut fvv = imdk::cc_fvv(t1, t2, eris, kconserv)?;
    let mut foo = imdk::cc_foo(t1, t2, eris, kconserv)?;
    let fov = imdk::cc_fov(t1, eris)?;
    let woooo = imdk::cc_woooo(t1, t2, eris, kconserv)?;
    let wvvvv = imdk::cc_wvvvv(t1, t2, eris, kconserv)?;
    let wovvo = imdk::cc_wovvo(t1, t2, eris, kconserv)?;

    // `:175-177` — move the energy terms to the other side.
    for k in 0..nk {
        shift_diag(&mut foo, k, &mo_e_o[k])?;
        shift_diag(&mut fvv, k, &mo_e_v[k])?;
    }

    let (ovvo, oovo) = imdk::eris_ovvo_oovo(eris, kconserv)?;
    // `:186` eris_vvvo[ke,kj,kb] = -eris.ovvv[km,kb,ke].transpose(2,3,1,0).conj()
    let mut vvvo = ZArr::zeros(&[nk, nk, nk, nvir, nvir, nvir, nocc]);
    for km in 0..nk {
        for kb in 0..nk {
            for ke in 0..nk {
                let kj = kconserv.get(km, ke, kb) as usize;
                let mut b = eris
                    .blk(GBlk::Ovvv, km, kb, ke)?
                    .transpose(&[2, 3, 1, 0])?
                    .conj();
                b.scale(-1.0);
                vvvo.set_leading(&[ke, kj, kb], &b)?;
            }
        }
    }

    // ------------------------------------------------------------------ T1
    let mut t1new = ZArr::zeros(&[nk, nocc, nvir]);
    for ka in 0..nk {
        let ki = ka;
        // `:119` t1new[ka] += fov[ka].conj() — the FIRST explicit conjugation.
        let mut acc = eris.fov(ka)?.conj();
        acc.add_assign(&einsum(
            "ie,ae->ia",
            &[&t1.slice_leading(&[ka])?, &fvv.slice_leading(&[ka])?],
        )?)?;
        acc.sub_assign(&einsum(
            "ma,mi->ia",
            &[&t1.slice_leading(&[ka])?, &foo.slice_leading(&[ka])?],
        )?)?;
        for km in 0..nk {
            acc.add_assign(&einsum(
                "imae,me->ia",
                &[
                    &t2.slice_leading(&[ka, km, ka])?,
                    &fov.slice_leading(&[km])?,
                ],
            )?)?;
            acc.sub_assign(&einsum(
                "nf,naif->ia",
                &[
                    &t1.slice_leading(&[km])?,
                    &eris.blk(GBlk::Ovov, km, ka, ki)?,
                ],
            )?)?;
            for kn in 0..nk {
                let ke = kconserv.get(km, ki, kn) as usize;
                acc.add_assign(&einsum_scaled(
                    "imef,maef->ia",
                    &[
                        &t2.slice_leading(&[ki, km, ke])?,
                        &eris.blk(GBlk::Ovvv, km, ka, ke)?,
                    ],
                    -0.5,
                )?)?;
                acc.add_assign(&einsum_scaled(
                    "mnae,nmei->ia",
                    &[
                        &t2.slice_leading(&[km, kn, ka])?,
                        &oovo.slice_leading(&[kn, km, ke])?,
                    ],
                    -0.5,
                )?)?;
            }
        }
        t1new.set_leading(&[ka], &acc)?;
    }

    // ------------------------------------------------------------------ T2
    // `:130` t2new = eris.oovv.conj() — the SECOND explicit conjugation.
    let mut t2new = ZArr::zeros(&[nk, nk, nk, nocc, nocc, nvir, nvir]);
    for ki in 0..nk {
        for kj in 0..nk {
            for ka in 0..nk {
                t2new.set_leading(
                    &[ki, kj, ka],
                    &eris.blk(GBlk::Oovv, ki, kj, ka)?.conj(),
                )?;
            }
        }
    }

    for ki in 0..nk {
        for kj in 0..nk {
            for ka in 0..nk {
                let kb = kconserv.get(ki, ka, kj) as usize;
                let mut blk = t2new.slice_leading(&[ki, kj, ka])?;

                let mut ftmp = fvv.slice_leading(&[kb])?;
                ftmp.zip_assign(
                    &einsum(
                        "mb,me->be",
                        &[&t1.slice_leading(&[kb])?, &fov.slice_leading(&[kb])?],
                    )?,
                    -0.5,
                )?;
                blk.add_assign(&einsum(
                    "ijae,be->ijab",
                    &[&t2.slice_leading(&[ki, kj, ka])?, &ftmp],
                )?)?;

                let mut ftmp = fvv.slice_leading(&[ka])?;
                ftmp.zip_assign(
                    &einsum(
                        "ma,me->ae",
                        &[&t1.slice_leading(&[ka])?, &fov.slice_leading(&[ka])?],
                    )?,
                    -0.5,
                )?;
                blk.sub_assign(&einsum(
                    "ijbe,ae->ijab",
                    &[&t2.slice_leading(&[ki, kj, kb])?, &ftmp],
                )?)?;

                let mut ftmp = foo.slice_leading(&[kj])?;
                ftmp.zip_assign(
                    &einsum(
                        "je,me->mj",
                        &[&t1.slice_leading(&[kj])?, &fov.slice_leading(&[kj])?],
                    )?,
                    0.5,
                )?;
                blk.sub_assign(&einsum(
                    "imab,mj->ijab",
                    &[&t2.slice_leading(&[ki, kj, ka])?, &ftmp],
                )?)?;

                let mut ftmp = foo.slice_leading(&[ki])?;
                ftmp.zip_assign(
                    &einsum(
                        "ie,me->mi",
                        &[&t1.slice_leading(&[ki])?, &fov.slice_leading(&[ki])?],
                    )?,
                    0.5,
                )?;
                blk.add_assign(&einsum(
                    "jmab,mi->ijab",
                    &[&t2.slice_leading(&[kj, ki, ka])?, &ftmp],
                )?)?;

                t2new.set_leading(&[ki, kj, ka], &blk)?;
            }
        }
    }

    for ki in 0..nk {
        for kj in 0..nk {
            for ka in 0..nk {
                let kb = kconserv.get(ki, ka, kj) as usize;
                for km in 0..nk {
                    let kn = kconserv.get(ka, km, kb) as usize;
                    let mut blk = t2new.slice_leading(&[ki, kj, ka])?;
                    blk.add_assign(&einsum_scaled(
                        "mnab,mnij->ijab",
                        &[
                            &tau.slice_leading(&[km, kn, ka])?,
                            &woooo.slice_leading(&[km, kn, ki])?,
                        ],
                        0.5,
                    )?)?;
                    let ke = km;
                    blk.add_assign(&einsum_scaled(
                        "ijef,abef->ijab",
                        &[
                            &tau.slice_leading(&[ki, kj, ke])?,
                            &wvvvv.slice_leading(&[ka, kb, ke])?,
                        ],
                        0.5,
                    )?)?;
                    t2new.set_leading(&[ki, kj, ka], &blk)?;

                    // `:169-181` — the Wmbej ring term and its three P(ij)P(ab)
                    // partners, which write into FOUR different k-blocks.
                    let ke = kconserv.get(km, kj, kb) as usize;
                    let mut tmp = einsum(
                        "imae,mbej->ijab",
                        &[
                            &t2.slice_leading(&[ki, km, ka])?,
                            &wovvo.slice_leading(&[km, kb, ke])?,
                        ],
                    )?;
                    if km == ka && ke == ki {
                        tmp.sub_assign(&einsum(
                            "ie,ma,mbej->ijab",
                            &[
                                &t1.slice_leading(&[ki])?,
                                &t1.slice_leading(&[km])?,
                                &ovvo.slice_leading(&[km, kb, ke])?,
                            ],
                        )?)?;
                    }
                    add_into(&mut t2new, [ki, kj, ka], &tmp, 1.0, None)?;
                    add_into(&mut t2new, [ki, kj, kb], &tmp, -1.0, Some([0, 1, 3, 2]))?;
                    add_into(&mut t2new, [kj, ki, ka], &tmp, -1.0, Some([1, 0, 2, 3]))?;
                    add_into(&mut t2new, [kj, ki, kb], &tmp, 1.0, Some([1, 0, 3, 2]))?;
                }

                let mut blk = t2new.slice_leading(&[ki, kj, ka])?;
                let ke = ki;
                blk.add_assign(&einsum(
                    "ie,abej->ijab",
                    &[
                        &t1.slice_leading(&[ki])?,
                        &vvvo.slice_leading(&[ka, kb, ke])?,
                    ],
                )?)?;
                let ke = kj;
                blk.sub_assign(&einsum(
                    "je,abei->ijab",
                    &[
                        &t1.slice_leading(&[kj])?,
                        &vvvo.slice_leading(&[ka, kb, ke])?,
                    ],
                )?)?;
                let km = ka;
                blk.sub_assign(&einsum(
                    "ma,mbij->ijab",
                    &[
                        &t1.slice_leading(&[ka])?,
                        &eris.blk(GBlk::Ovoo, km, kb, ki)?,
                    ],
                )?)?;
                let km = kb;
                blk.add_assign(&einsum(
                    "mb,maij->ijab",
                    &[
                        &t1.slice_leading(&[kb])?,
                        &eris.blk(GBlk::Ovoo, km, ka, ki)?,
                    ],
                )?)?;
                t2new.set_leading(&[ki, kj, ka], &blk)?;
            }
        }
    }

    // ---------------------------------------------- the LARGE_DENOM divide
    for ki in 0..nk {
        let ka = ki;
        let eia = eia_large_denom(&mo_e_o, &mo_e_v, ki, ka, &nz_o, &nz_v);
        let mut blk = t1new.slice_leading(&[ki])?;
        for i in 0..nocc {
            for a in 0..nvir {
                let f = i * nvir + a;
                blk.data_mut().re[f] /= eia[f];
                blk.data_mut().im[f] /= eia[f];
            }
        }
        t1new.set_leading(&[ki], &blk)?;
    }
    for ki in 0..nk {
        for kj in 0..nk {
            for ka in 0..nk {
                let kb = kconserv.get(ki, ka, kj) as usize;
                let eia = eia_large_denom(&mo_e_o, &mo_e_v, ki, ka, &nz_o, &nz_v);
                let ejb = eia_large_denom(&mo_e_o, &mo_e_v, kj, kb, &nz_o, &nz_v);
                let mut blk = t2new.slice_leading(&[ki, kj, ka])?;
                for i in 0..nocc {
                    for j in 0..nocc {
                        for a in 0..nvir {
                            for b in 0..nvir {
                                let d = eia[i * nvir + a] + ejb[j * nvir + b];
                                let f = ((i * nocc + j) * nvir + a) * nvir + b;
                                blk.data_mut().re[f] /= d;
                                blk.data_mut().im[f] /= d;
                            }
                        }
                    }
                }
                t2new.set_leading(&[ki, kj, ka], &blk)?;
            }
        }
    }

    Ok((t1new, t2new))
}

fn add_into(
    t2new: &mut ZArr,
    k: [usize; 3],
    src: &ZArr,
    factor: f64,
    axes: Option<[usize; 4]>,
) -> Result<(), PbcCcError> {
    let v = match axes {
        Some(a) => src.transpose(&a)?,
        None => src.clone(),
    };
    let mut blk = t2new.slice_leading(&k)?;
    blk.zip_assign(&v, factor)?;
    t2new.set_leading(&k, &blk)
}

fn shift_diag(x: &mut ZArr, k: usize, e: &[f64]) -> Result<(), PbcCcError> {
    let mut blk = x.slice_leading(&[k])?;
    let n = e.len();
    for i in 0..n {
        blk.data_mut().re[i * n + i] -= e[i];
    }
    x.set_leading(&[k], &blk)
}

/// `e[kp, :nocc] - e[kq, nocc:]` with padded entries at [`LARGE_DENOM`]
/// (`kccsd.py:203-206`).
fn eia_large_denom(
    mo_e_o: &[Vec<f64>],
    mo_e_v: &[Vec<f64>],
    kp: usize,
    kq: usize,
    nz_o: &[Vec<usize>],
    nz_v: &[Vec<usize>],
) -> Vec<f64> {
    let (nocc, nvir) = (mo_e_o[kp].len(), mo_e_v[kq].len());
    let mut out = vec![LARGE_DENOM; nocc * nvir];
    for &i in &nz_o[kp] {
        if i >= nocc {
            continue;
        }
        for &a in &nz_v[kq] {
            if a >= nvir {
                continue;
            }
            out[i * nvir + a] = mo_e_o[kp][i] - mo_e_v[kq][a];
        }
    }
    out
}

/// `spatial2spin(t1_spatial, orbspin, kconserv)` for `t1` —
/// `kccsd.py:223-232` and `:253-260`.
///
/// `orbspin[k][p]` is 0 for an alpha spin-orbital and 1 for a beta one. The
/// RESTRICTED `t1` is used for both spin channels (`:229` calls
/// `spatial2spin((tx, tx), …)`), which is what makes a KRCCSD result liftable
/// into the spin-orbital basis and hence what makes 16-07 test 2 —
/// `KGCCSD.e_corr == KRCCSD.e_corr` on a closed shell — checkable at all.
///
/// # Errors
/// [`PbcCcError::Shape`] on a size or `orbspin` inconsistency.
pub fn spatial2spin_t1(
    t1a: &ZArr,
    t1b: &ZArr,
    orbspin: &[Vec<u8>],
    nocc: usize,
    nvir: usize,
) -> Result<ZArr, PbcCcError> {
    let nk = t1a.shape()[0];
    let mut out = ZArr::zeros(&[nk, nocc, nvir]);
    for k in 0..nk {
        for (src, spin) in [(t1a, 0_u8), (t1b, 1_u8)] {
            let o = spin_idx(&orbspin[k][..nocc], spin);
            let v = spin_idx(&orbspin[k][nocc..], spin);
            if o.len() != src.shape()[1] || v.len() != src.shape()[2] {
                return Err(PbcCcError::Shape(format!(
                    "spatial2spin: k {k} spin {spin} has {} occupied and {} virtual, \
                     but the spatial t1 is {} x {}",
                    o.len(),
                    v.len(),
                    src.shape()[1],
                    src.shape()[2]
                )));
            }
            for (i, &oi) in o.iter().enumerate() {
                for (a, &ai) in v.iter().enumerate() {
                    let (re, im) = src.at(&[k, i, a])?;
                    let f = (k * nocc + oi) * nvir + ai;
                    out.data_mut().re[f] = re;
                    out.data_mut().im[f] = im;
                }
            }
        }
    }
    Ok(out)
}

fn spin_idx(orbspin: &[u8], want: u8) -> Vec<usize> {
    orbspin
        .iter()
        .enumerate()
        .filter_map(|(i, &s)| (s == want).then_some(i))
        .collect()
}

/// `spin2spatial(t1_spin, orbspin, kconserv)` for `t1` — `kccsd.py:289-312`.
///
/// # Errors
/// [`PbcCcError::Shape`] on a size inconsistency.
pub fn spin2spatial_t1(
    t1: &ZArr,
    orbspin: &[Vec<u8>],
    nocc: usize,
) -> Result<(ZArr, ZArr), PbcCcError> {
    let nk = t1.shape()[0];
    let oa0 = spin_idx(&orbspin[0][..nocc], 0).len();
    let ob0 = spin_idx(&orbspin[0][..nocc], 1).len();
    let va0 = spin_idx(&orbspin[0][nocc..], 0).len();
    let vb0 = spin_idx(&orbspin[0][nocc..], 1).len();
    let mut t1a = ZArr::zeros(&[nk, oa0, va0]);
    let mut t1b = ZArr::zeros(&[nk, ob0, vb0]);
    for k in 0..nk {
        for (dst, o, v) in [
            (&mut t1a, spin_idx(&orbspin[k][..nocc], 0), spin_idx(&orbspin[k][nocc..], 0)),
            (&mut t1b, spin_idx(&orbspin[k][..nocc], 1), spin_idx(&orbspin[k][nocc..], 1)),
        ] {
            for (i, &oi) in o.iter().enumerate() {
                for (a, &ai) in v.iter().enumerate() {
                    let (re, im) = t1.at(&[k, oi, ai])?;
                    let n = dst.shape()[2];
                    let f = (k * dst.shape()[1] + i) * n + a;
                    dst.data_mut().re[f] = re;
                    dst.data_mut().im[f] = im;
                }
            }
        }
    }
    Ok((t1a, t1b))
}

/// `init_amps` for the spin-orbital case — the MP2 guess
/// (`gccsd.GCCSD.init_amps` via `kccsd.py:395`'s `ccsd.CCSD.ccsd` driver).
///
/// # Errors
/// Propagates the ERI access.
pub fn init_amps(
    eris: &KgEris,
    padded: &PaddedMos,
    kconserv: &Kconserv,
) -> Result<(f64, ZArr, ZArr), PbcCcError> {
    let (nk, nocc, nvir) = (eris.nkpts, eris.nocc, eris.nvir);
    let t1 = ZArr::zeros(&[nk, nocc, nvir]);
    let mut t2 = ZArr::zeros(&[nk, nk, nk, nocc, nocc, nvir, nvir]);
    let mo_e_o: Vec<Vec<f64>> = eris.mo_energy.iter().map(|e| e[..nocc].to_vec()).collect();
    let mo_e_v: Vec<Vec<f64>> = eris.mo_energy.iter().map(|e| e[nocc..].to_vec()).collect();
    let (nz_o, nz_v) =
        match padding_k_idx(&padded.nmo_per_kpt, &padded.nocc_per_kpt, PaddingKind::Split) {
            Ok(PaddingIdx::Split { occupied, virtuals }) => (occupied, virtuals),
            Ok(_) => return Err(PbcCcError::Shape("padding_k_idx returned a joint set".into())),
            Err(e) => return Err(PbcCcError::Shape(format!("padding_k_idx: {e}"))),
        };
    let mut terms: Vec<f64> = Vec::new();
    for ki in 0..nk {
        for kj in 0..nk {
            for ka in 0..nk {
                let kb = kconserv.get(ki, ka, kj) as usize;
                let eia = eia_large_denom(&mo_e_o, &mo_e_v, ki, ka, &nz_o, &nz_v);
                let ejb = eia_large_denom(&mo_e_o, &mo_e_v, kj, kb, &nz_o, &nz_v);
                let oovv = eris.blk(GBlk::Oovv, ki, kj, ka)?;
                let mut blk = oovv.conj();
                for i in 0..nocc {
                    for j in 0..nocc {
                        for a in 0..nvir {
                            for b in 0..nvir {
                                let d = eia[i * nvir + a] + ejb[j * nvir + b];
                                let f = ((i * nocc + j) * nvir + a) * nvir + b;
                                blk.data_mut().re[f] /= d;
                                blk.data_mut().im[f] /= d;
                            }
                        }
                    }
                }
                let (re, _) = einsum_scaled("ijab,ijab->", &[&blk, &oovv], 0.25)?.at(&[])?;
                terms.push(re);
                t2.set_leading(&[ki, kj, ka], &blk)?;
            }
        }
    }
    Ok((oracle_sum(&terms) / nk as f64, t1, t2))
}

/// The spin-orbital amplitude iteration — `ccsd.CCSD.ccsd`'s driver reached
/// through `kccsd.py:395`.
///
/// # Errors
/// Propagates `update_amps` and `energy`.
pub fn kernel(
    eris: &KgEris,
    padded: &PaddedMos,
    kconserv: &Kconserv,
    opts: &crate::kccsd_rhf::KrccsdOpts,
) -> Result<crate::kccsd_rhf::KrccsdResult, PbcCcError> {
    use pyscf_diis::Diis;

    use crate::kccsd_rhf::KAmplitudeSubspace;

    let (emp2, mut t1, mut t2) = init_amps(eris, padded, kconserv)?;
    let mut e = energy(&t1, &t2, eris)?;
    let mut converged = false;
    let mut cycles = 0;

    // The SAME amplitude-DIIS stack `kccsd_rhf` uses — `kccsd.py:395` reaches
    // `ccsd.CCSD.ccsd`'s driver, which is DIIS-accelerated. Without it the
    // spin-orbital iteration does not converge in `max_cycle`: measured 50
    // cycles and still moving on diamond `gth-szv` [1,1,2], against 21 with it.
    let mut diis: Option<Diis<KAmplitudeSubspace>> = if opts.diis {
        Some(Diis::new(opts.diis_space))
    } else {
        None
    };

    for istep in 0..opts.max_cycle {
        cycles = istep + 1;
        let (t1n, t2n) = update_amps(&t1, &t2, eris, padded, kconserv, opts.level_shift)?;
        let cur = KAmplitudeSubspace::from_amplitudes(&t1n, &t2n);
        let prev = KAmplitudeSubspace::from_amplitudes(&t1, &t2);
        let res = cur.residual(&prev);
        let normt = pyscf_algebra::oracle_dot(&res, &res).sqrt();
        t1 = t1n;
        t2 = t2n;

        if let Some(stack) = diis.as_mut()
            && istep >= opts.diis_start_cycle
        {
            let cur = KAmplitudeSubspace::from_amplitudes(&t1, &t2);
            let err = cur.residual(&prev);
            let extrap = stack
                .extrapolate(cur, err)
                .map_err(|e| PbcCcError::Algebra(format!("amplitude DIIS: {e}")))?;
            let (a, b) = extrap.to_amplitudes(&t1, &t2);
            t1 = a;
            t2 = b;
        }

        let eold = e;
        e = energy(&t1, &t2, eris)?;
        if (e - eold).abs() < opts.conv_tol && normt < opts.conv_tol_normt {
            converged = true;
            break;
        }
    }
    Ok(crate::kccsd_rhf::KrccsdResult {
        e_corr: e,
        emp2,
        converged,
        cycles,
        t1,
        t2,
    })
}

/// `kccsd.py:414` — the one LIVE refusal in this module.
///
/// `:486` is commented out upstream (`#    raise NotImplementedError('Different
/// occupancies…')`) and is deliberately NOT ported as a refusal: porting it
/// would invent a restriction upstream does not impose.
///
/// # Errors
/// Always. That is the point.
pub fn refuse_414() -> Result<(), PbcCcError> {
    Err(PbcCcError::NotImplementedUpstream {
        upstream: "pyscf/pbc/cc/kccsd.py:414",
        what: "the surface upstream PySCF 2.12.1 raises NotImplementedError for",
    })
}
