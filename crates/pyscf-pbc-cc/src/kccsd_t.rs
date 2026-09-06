//! `kccsd_t` — the SPIN-ORBITAL k-point CCSD(T) (plan 16-08 Task 3;
//! `pyscf/pbc/cc/kccsd_t.py`, equations from Tu, Yang, Wang and Guo,
//! J. Phys. Chem. **135** (2011)).
//!
//! Runs on [`crate::kccsd`]'s `KGCCSD` amplitudes. Unlike the RHF `(T)` this
//! has no C kernel and no blocking: it is an explicit `(a, b, c)` loop with
//! k-point AND orbital permutational symmetry factors, and `t3` is never
//! materialised beyond one `(nocc, nocc, nocc)` block — the streaming property
//! `16-REVIEW.md §4.2` asks for, here by construction.
//!
//! # `tril_product` — the enumeration order is observable
//!
//! `:139-149` selects the `(a, b, c)` loop by which of `ka`, `kb`, `kc`
//! coincide: `a >= b >= c` when `ka == kc`, `a >= b` when `ka == kb`,
//! `b >= c` when `kb == kc`, and the full product otherwise. Each carries its
//! own multiplicity (`:152-169`). Porting the ORDER matters, not only the set:
//! §9.3 gates bit-identity and the accumulation order is what that is about.
//!
//! # `LARGE_DENOM`
//!
//! `eabc` is built with `fac = [-1,-1,-1]` and upstream's own comment at
//! `:123` says why: "so the LARGE_DENOM does not cancel with the one from
//! eijk". Two padded orbitals must give a LARGER denominator, not zero.

use pyscf_algebra::oracle_sum;
use pyscf_pbc_lib::{KIdx, Kconserv, get_kconserv3};
use pyscf_pbc_mp::{PaddedMos, PaddingIdx, PaddingKind, padding_k_idx};

use crate::error::PbcCcError;
use crate::kccsd::{GBlk, KgEris};
use crate::kccsd_rhf::LARGE_DENOM;
use crate::zarr::{ZArr, einsum};

/// `lib.misc.tril_product(range(n), repeat=3, tril_idx=…)` for the three cases
/// `kccsd_t.py:139-149` uses.
///
/// * `Full` — the whole `n³` product.
/// * `Abc` — `a >= b >= c` (`tril_idx = [0,1,2]`).
/// * `Ab` — `a >= b`, `c` free (`tril_idx = [0,1]`).
/// * `Bc` — `b >= c`, `a` free (`tril_idx = [1,2]`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrilKind {
    Full,
    Abc,
    Ab,
    Bc,
}

/// The `(a, b, c)` list, in upstream's enumeration order.
pub fn tril_product(n: usize, kind: TrilKind) -> Vec<(usize, usize, usize)> {
    let mut out = Vec::new();
    for a in 0..n {
        for b in 0..n {
            if matches!(kind, TrilKind::Abc | TrilKind::Ab) && b > a {
                continue;
            }
            for c in 0..n {
                if matches!(kind, TrilKind::Abc | TrilKind::Bc) && c > b {
                    continue;
                }
                out.push((a, b, c));
            }
        }
    }
    out
}

/// `kernel(mycc, eris, t1, t2)` — `kccsd_t.py:42-276`.
///
/// # Errors
/// Propagates the ERI access, the padding surface and every shape check.
pub fn kernel(
    eris: &KgEris,
    padded: &PaddedMos,
    t1: &ZArr,
    t2: &ZArr,
    kconserv: &Kconserv,
    a_lat: &[[f64; 3]; 3],
    kpts: &[[f64; 3]],
) -> Result<f64, PbcCcError> {
    let (nkpts, nocc, nvir) = (eris.nkpts, eris.nocc, eris.nvir);
    let mo_e_o: Vec<Vec<f64>> = eris.mo_energy.iter().map(|e| e[..nocc].to_vec()).collect();
    let mo_e_v: Vec<Vec<f64>> = eris.mo_energy.iter().map(|e| e[nocc..].to_vec()).collect();
    let fov: Vec<ZArr> = (0..nkpts).map(|k| eris.fov(k)).collect::<Result<_, _>>()?;
    let (nz_o, nz_v) = match padding_k_idx(
        &padded.nmo_per_kpt,
        &padded.nocc_per_kpt,
        PaddingKind::Split,
    ) {
        Ok(PaddingIdx::Split { occupied, virtuals }) => (occupied, virtuals),
        Ok(_) => {
            return Err(PbcCcError::Shape(
                "padding_k_idx returned a joint set".into(),
            ));
        }
        Err(e) => return Err(PbcCcError::Shape(format!("padding_k_idx: {e}"))),
    };

    let mut terms_re: Vec<f64> = Vec::new();
    let mut terms_im: Vec<f64> = Vec::new();

    // `t2[kx,ky,kz][:, :, p, q]` — the rank-2 occupied block at fixed virtuals.
    let t2oo = |kx: usize, ky: usize, kz: usize, p: usize, q: usize| -> Result<ZArr, PbcCcError> {
        t2.slice_leading(&[kx, ky, kz])?
            .slice_axes(&[(0, nocc), (0, nocc), (p, p + 1), (q, q + 1)])?
            .reshape(&[nocc, nocc])
    };
    // `t2[kx,ky,kz][:, :, p, :]` — 'jke'.
    let t2ove = |kx: usize, ky: usize, kz: usize, p: usize| -> Result<ZArr, PbcCcError> {
        t2.slice_leading(&[kx, ky, kz])?
            .slice_axes(&[(0, nocc), (0, nocc), (p, p + 1), (0, nvir)])?
            .reshape(&[nocc, nocc, nvir])
    };
    // `-eris.ovvv[kx,ky,kz][:, :, p, q].conj()` — 'ie'.
    let movvv = |kx: usize, ky: usize, kz: usize, p: usize, q: usize| -> Result<ZArr, PbcCcError> {
        let mut b = eris
            .blk(GBlk::Ovvv, kx, ky, kz)?
            .slice_axes(&[(0, nocc), (0, nvir), (p, p + 1), (q, q + 1)])?
            .reshape(&[nocc, nvir])?
            .conj();
        b.scale(-1.0);
        Ok(b)
    };
    // `eris.ooov[kx,ky,kz][:, :, :, p].conj()` — 'jkm'.
    let ooovc = |kx: usize, ky: usize, kz: usize, p: usize| -> Result<ZArr, PbcCcError> {
        Ok(eris
            .blk(GBlk::Ooov, kx, ky, kz)?
            .slice_axes(&[(0, nocc), (0, nocc), (0, nocc), (p, p + 1)])?
            .reshape(&[nocc, nocc, nocc])?
            .conj())
    };
    // `-eris.oovv[kx,ky,kz][:, :, p, q].conj()` — 'jk'.
    let moovv = |kx: usize, ky: usize, kz: usize, p: usize, q: usize| -> Result<ZArr, PbcCcError> {
        let mut b = eris
            .blk(GBlk::Oovv, kx, ky, kz)?
            .slice_axes(&[(0, nocc), (0, nocc), (p, p + 1), (q, q + 1)])?
            .reshape(&[nocc, nocc])?
            .conj();
        b.scale(-1.0);
        Ok(b)
    };
    // `t1[k][:, p]` and `-fov[k][:, p]`.
    let t1o = |k: usize, p: usize| -> Result<ZArr, PbcCcError> {
        t1.slice_leading(&[k])?
            .slice_axes(&[(0, nocc), (p, p + 1)])?
            .reshape(&[nocc])
    };
    let mfov = |k: usize, p: usize| -> Result<ZArr, PbcCcError> {
        let mut b = fov[k]
            .slice_axes(&[(0, nocc), (p, p + 1)])?
            .reshape(&[nocc])?;
        b.scale(-1.0);
        Ok(b)
    };

    for ki in 0..nkpts {
        for kj in 0..=ki {
            for kk in 0..=kj {
                let eijk = epqr3(&mo_e_o, &nz_o, ki, kj, kk, nocc, 1.0);
                let symm_ijk = if ki == kj && kj == kk {
                    1.0
                } else if ki == kj || kj == kk {
                    3.0
                } else {
                    6.0
                };

                for ka in 0..nkpts {
                    for kb in 0..=ka {
                        let kc = get_kconserv3(
                            a_lat,
                            kpts,
                            &[
                                KIdx::One(ki),
                                KIdx::One(kj),
                                KIdx::One(kk),
                                KIdx::One(ka),
                                KIdx::One(kb),
                            ],
                        )
                        .data[0] as usize;
                        if kc > kb {
                            continue;
                        }
                        // `:123` fac = [-1,-1,-1] so the two LARGE_DENOMs add
                        // rather than cancel.
                        let eabc = epqr3(&mo_e_v, &nz_v, ka, kb, kc, nvir, -1.0);
                        let symm_abc_kpt = if ka == kb && kb == kc {
                            1.0
                        } else if ka == kb || kb == kc {
                            3.0
                        } else {
                            6.0
                        };
                        let (kind, mode) = if ka == kc {
                            (TrilKind::Abc, 3)
                        } else if ka == kb {
                            (TrilKind::Ab, 1)
                        } else if kb == kc {
                            (TrilKind::Bc, 2)
                        } else {
                            (TrilKind::Full, 0)
                        };

                        for (a, b, c) in tril_product(nvir, kind) {
                            let symm_abc = match mode {
                                3 => {
                                    if a == b && b == c {
                                        1.0
                                    } else if a == b || b == c {
                                        3.0
                                    } else {
                                        6.0
                                    }
                                }
                                1 => {
                                    if a == b {
                                        1.0
                                    } else {
                                        2.0
                                    }
                                }
                                2 => {
                                    if b == c {
                                        1.0
                                    } else {
                                        2.0
                                    }
                                }
                                _ => 1.0,
                            };

                            // ---- t3c: the connected triples amplitude
                            let mut t3c = ZArr::zeros(&[nocc, nocc, nocc]);
                            // First term: 1 - p(ij) - p(ik)
                            let ke = kconserv.get(kj, ka, kk) as usize;
                            t3c.add_assign(&einsum(
                                "jke,ie->ijk",
                                &[&t2ove(kj, kk, ka, a)?, &movvv(ki, ke, kc, c, b)?],
                            )?)?;
                            let ke = kconserv.get(ki, ka, kk) as usize;
                            t3c.sub_assign(&einsum(
                                "ike,je->ijk",
                                &[&t2ove(ki, kk, ka, a)?, &movvv(kj, ke, kc, c, b)?],
                            )?)?;
                            let ke = kconserv.get(kj, ka, ki) as usize;
                            t3c.sub_assign(&einsum(
                                "jie,ke->ijk",
                                &[&t2ove(kj, ki, ka, a)?, &movvv(kk, ke, kc, c, b)?],
                            )?)?;

                            let km = kconserv.get(kb, ki, kc) as usize;
                            t3c.sub_assign(&einsum(
                                "mi,jkm->ijk",
                                &[&t2oo(km, ki, kb, b, c)?, &ooovc(kj, kk, km, a)?],
                            )?)?;
                            let km = kconserv.get(kb, kj, kc) as usize;
                            t3c.add_assign(&einsum(
                                "mj,ikm->ijk",
                                &[&t2oo(km, kj, kb, b, c)?, &ooovc(ki, kk, km, a)?],
                            )?)?;
                            let km = kconserv.get(kb, kk, kc) as usize;
                            t3c.add_assign(&einsum(
                                "mk,jim->ijk",
                                &[&t2oo(km, kk, kb, b, c)?, &ooovc(kj, ki, km, a)?],
                            )?)?;

                            // Second term: - p(ab) + p(ab)p(ij) + p(ab)p(ik)
                            let ke = kconserv.get(kj, kb, kk) as usize;
                            t3c.sub_assign(&einsum(
                                "jke,ie->ijk",
                                &[&t2ove(kj, kk, kb, b)?, &movvv(ki, ke, kc, c, a)?],
                            )?)?;
                            let ke = kconserv.get(ki, kb, kk) as usize;
                            t3c.add_assign(&einsum(
                                "ike,je->ijk",
                                &[&t2ove(ki, kk, kb, b)?, &movvv(kj, ke, kc, c, a)?],
                            )?)?;
                            let ke = kconserv.get(kj, kb, ki) as usize;
                            t3c.add_assign(&einsum(
                                "jie,ke->ijk",
                                &[&t2ove(kj, ki, kb, b)?, &movvv(kk, ke, kc, c, a)?],
                            )?)?;

                            let km = kconserv.get(ka, ki, kc) as usize;
                            t3c.add_assign(&einsum(
                                "mi,jkm->ijk",
                                &[&t2oo(km, ki, ka, a, c)?, &ooovc(kj, kk, km, b)?],
                            )?)?;
                            let km = kconserv.get(ka, kj, kc) as usize;
                            t3c.sub_assign(&einsum(
                                "mj,ikm->ijk",
                                &[&t2oo(km, kj, ka, a, c)?, &ooovc(ki, kk, km, b)?],
                            )?)?;
                            let km = kconserv.get(ka, kk, kc) as usize;
                            t3c.sub_assign(&einsum(
                                "mk,jim->ijk",
                                &[&t2oo(km, kk, ka, a, c)?, &ooovc(kj, ki, km, b)?],
                            )?)?;

                            // Third term: - p(ac) + p(ac)p(ij) + p(ac)p(ik)
                            let ke = kconserv.get(kj, kc, kk) as usize;
                            t3c.sub_assign(&einsum(
                                "jke,ie->ijk",
                                &[&t2ove(kj, kk, kc, c)?, &movvv(ki, ke, ka, a, b)?],
                            )?)?;
                            let ke = kconserv.get(ki, kc, kk) as usize;
                            t3c.add_assign(&einsum(
                                "ike,je->ijk",
                                &[&t2ove(ki, kk, kc, c)?, &movvv(kj, ke, ka, a, b)?],
                            )?)?;
                            let ke = kconserv.get(kj, kc, ki) as usize;
                            t3c.add_assign(&einsum(
                                "jie,ke->ijk",
                                &[&t2ove(kj, ki, kc, c)?, &movvv(kk, ke, ka, a, b)?],
                            )?)?;

                            let km = kconserv.get(kb, ki, ka) as usize;
                            t3c.add_assign(&einsum(
                                "mi,jkm->ijk",
                                &[&t2oo(km, ki, kb, b, a)?, &ooovc(kj, kk, km, c)?],
                            )?)?;
                            let km = kconserv.get(kb, kj, ka) as usize;
                            t3c.sub_assign(&einsum(
                                "mj,ikm->ijk",
                                &[&t2oo(km, kj, kb, b, a)?, &ooovc(ki, kk, km, c)?],
                            )?)?;
                            let km = kconserv.get(kb, kk, ka) as usize;
                            t3c.sub_assign(&einsum(
                                "mk,jim->ijk",
                                &[&t2oo(km, kk, kb, b, a)?, &ooovc(kj, ki, km, c)?],
                            )?)?;

                            // ---- t3d: the disconnected contribution
                            let mut t3d = ZArr::zeros(&[nocc, nocc, nocc]);
                            // `:224-262`. The virtual index the `t1`/`fov`
                            // factor carries is `a`, `b`, `c` for the three
                            // terms respectively and is carried EXPLICITLY:
                            // inferring it from the `oovv` slice breaks the
                            // moment two of `a`, `b`, `c` coincide, which the
                            // `tril_product` loops make common.
                            for (cond, sign, ko, spec, orb, kx, ky, kz, p, q) in [
                                (ki == ka, 1.0, ki, "i,jk->ijk", a, kj, kk, kb, b, c),
                                (kj == ka, -1.0, kj, "j,ik->ijk", a, ki, kk, kb, b, c),
                                (kk == ka, -1.0, kk, "k,ji->ijk", a, kj, ki, kb, b, c),
                                (ki == kb, -1.0, ki, "i,jk->ijk", b, kj, kk, ka, a, c),
                                (kj == kb, 1.0, kj, "j,ik->ijk", b, ki, kk, ka, a, c),
                                (kk == kb, 1.0, kk, "k,ji->ijk", b, kj, ki, ka, a, c),
                                (ki == kc, -1.0, ki, "i,jk->ijk", c, kj, kk, kb, b, a),
                                (kj == kc, 1.0, kj, "j,ik->ijk", c, ki, kk, kb, b, a),
                                (kk == kc, 1.0, kk, "k,ji->ijk", c, kj, ki, kb, b, a),
                            ] {
                                if !cond {
                                    continue;
                                }
                                t3d.zip_assign(
                                    &einsum(spec, &[&t1o(ko, orb)?, &moovv(kx, ky, kz, p, q)?])?,
                                    sign,
                                )?;
                                t3d.zip_assign(
                                    &einsum(spec, &[&mfov(ko, orb)?, &t2oo(kx, ky, kz, p, q)?])?,
                                    sign,
                                )?;
                            }

                            // `:264-266` (t3c + t3d) / eijkabc
                            let mut t3cd = t3c.clone();
                            t3cd.add_assign(&t3d)?;
                            let d_abc = eabc[(a * nvir + b) * nvir + c];
                            for f in 0..nocc * nocc * nocc {
                                let d = eijk[f] + d_abc;
                                t3cd.data_mut().re[f] /= d;
                                t3cd.data_mut().im[f] /= d;
                            }
                            let (re, im) = einsum("ijk,ijk->", &[&t3c, &t3cd.conj()])?.at(&[])?;
                            let w = symm_abc_kpt * symm_ijk * symm_abc;
                            terms_re.push(w * re);
                            terms_im.push(w * im);
                        }
                    }
                }
            }
        }
    }

    let re = oracle_sum(&terms_re) / 36.0 / nkpts as f64;
    let im = oracle_sum(&terms_im) / 36.0 / nkpts as f64;
    if im.abs() > 1e-4 {
        tracing::warn!(
            imaginary = im,
            "non-zero imaginary part of the spin-orbital CCSD(T) energy (kccsd_t.py:271)"
        );
    }
    Ok(re)
}

/// `_get_epqr` over three indices of one space, with a uniform factor.
/// Padded orbitals carry [`LARGE_DENOM`].
fn epqr3(
    mo_e: &[Vec<f64>],
    nz: &[Vec<usize>],
    kp: usize,
    kq: usize,
    kr: usize,
    n: usize,
    fac: f64,
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
                out[(i * n + j) * n + k] = fac * (mo_e[kp][i] + mo_e[kq][j] + mo_e[kr][k]);
            }
        }
    }
    out
}
