//! `kccsd_t_rhf_slow` — the loop-explicit restricted k-point CCSD(T)
//! (plan 16-08 Task 1; `pyscf/pbc/cc/kccsd_t_rhf_slow.py`).
//!
//! **This file is ported FIRST, before either fast path** (16-08 Task 1), and
//! `PBC-MASTER-PLAN §8.8`'s table omits it entirely (`16-CONTEXT §1.7`). It is
//! the only oracle-free reference the blocked path can be gated against:
//! `kccsd_t_rhf.py:236` drives the C kernel `_ccsd.libcc.CCsd_zcontract_t3T`
//! (24 raw data pointers, `:229-245`), and this port has no C and no `libcc` —
//! that is the project's core value proposition, not an accident.
//!
//! 16-01 measured upstream's own fast-vs-slow agreement at **`3.27e-16`
//! absolute / `2.95e-13` relative** (`measurements/README.md §5`), which is
//! gate **G4** and the one place a Phase-16 number can be tight: same input,
//! same formula, two implementations, no convergence noise in between.
//!
//! # `LARGE_DENOM` again (`16-CONTEXT §3.3`)
//!
//! `kccsd_t_rhf_slow.py:174` and `:196` fill `eijk` / `eabc` with
//! `LARGE_DENOM` at PADDED orbitals rather than skipping them, so a padded
//! orbital contributes `~1e-28` to `t3`. Arithmetic, not a guard.
//!
//! # The conjugations
//!
//! Four sites conjugate, and each is an explicit [`ZArr::conj`] here because it
//! is an explicit `.conj()` there: `eris.vovv[...].conj()` (`:106`),
//! `eris.ooov[...].conj()` (`:107`), `eris.oovv[...].conj()` (`:137`) and
//! `rwijk.conj()` in the energy contraction (`:205`). Everything else is an
//! `einsum` and therefore unconjugated by construction (see `crate::zarr`).

use pyscf_algebra::oracle_sum;
use pyscf_pbc_lib::{KIdx, Kconserv, get_kconserv3};
use pyscf_pbc_mp::{PaddedMos, PaddingIdx, PaddingKind, padding_k_idx};

use crate::error::PbcCcError;
use crate::kccsd_rhf::LARGE_DENOM;
use crate::keris::{Blk, KEris};
use crate::zarr::{ZArr, einsum};

/// A virtual-orbital block range `[a0, a1) x [b0, b1) x [c0, c1)` — upstream's
/// `task` tuple (`kccsd_t_rhf_slow.py:167`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VirBlock {
    pub a: (usize, usize),
    pub b: (usize, usize),
    pub c: (usize, usize),
}

/// The task list for a virtual block size, `:161-167`.
pub fn tasks(nvir: usize, blksize: usize) -> Vec<VirBlock> {
    let bs = blksize.clamp(1, nvir.max(1));
    let mut out = Vec::new();
    let mut a0 = 0;
    while a0 < nvir {
        let a1 = (a0 + bs).min(nvir);
        let mut b0 = 0;
        while b0 < nvir {
            let b1 = (b0 + bs).min(nvir);
            let mut c0 = 0;
            while c0 < nvir {
                let c1 = (c0 + bs).min(nvir);
                out.push(VirBlock {
                    a: (a0, a1),
                    b: (b0, b1),
                    c: (c0, c1),
                });
                c0 = c1;
            }
            b0 = b1;
        }
        a0 = a1;
    }
    out
}

struct Ctx<'a> {
    eris: &'a KEris,
    t1: &'a ZArr,
    t2: &'a ZArr,
    fov: Vec<ZArr>,
    kconserv: &'a Kconserv,
    nocc: usize,
    nvir: usize,
}

impl Ctx<'_> {
    /// `get_w` — `kccsd_t_rhf_slow.py:104-110`. Returns `abcijk`.
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
        let (nocc, nvir) = (self.nocc, self.nvir);
        let km = self.kconserv.get(ki, ka, kj) as usize;
        let kf = self.kconserv.get(kk, kc, kj) as usize;

        let t2a = self.t2.slice_leading(&[kk, kj, kc])?.slice_axes(&[
            (0, nocc),
            (0, nocc),
            blk.c,
            (0, nvir),
        ])?;
        let vovv = self
            .eris
            .blk(Blk::Vovv, kf, ki, kb)?
            .slice_axes(&[(0, nvir), (0, nocc), blk.b, blk.a])?
            .conj();
        let mut ret = einsum("kjcf,fiba->abcijk", &[&t2a, &vovv])?;

        let t2b = self.t2.slice_leading(&[km, kk, kb])?.slice_axes(&[
            (0, nocc),
            (0, nocc),
            blk.b,
            blk.c,
        ])?;
        let ooov = self
            .eris
            .blk(Blk::Ooov, kj, ki, km)?
            .slice_axes(&[(0, nocc), (0, nocc), (0, nocc), blk.a])?
            .conj();
        ret.sub_assign(&einsum("mkbc,jima->abcijk", &[&t2b, &ooov])?)?;
        Ok(ret)
    }

    /// `get_permuted_w` — `:112-122`, the `Pijkabc` operator.
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
        let p = |x: (usize, usize), y: (usize, usize), z: (usize, usize)| VirBlock {
            a: x,
            b: y,
            c: z,
        };
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

    /// `get_rw` — `:124-133`, the `R` operator.
    #[allow(clippy::too_many_arguments)]
    fn get_rw(
        &self,
        ki: usize,
        kj: usize,
        kk: usize,
        ka: usize,
        kb: usize,
        kc: usize,
        blk: VirBlock,
    ) -> Result<ZArr, PbcCcError> {
        let mut out = self.get_permuted_w(ki, kj, kk, ka, kb, kc, blk)?;
        out.scale(4.0);
        out.zip_assign(
            &self
                .get_permuted_w(kj, kk, ki, ka, kb, kc, blk)?
                .transpose(&[0, 1, 2, 5, 3, 4])?,
            1.0,
        )?;
        out.zip_assign(
            &self
                .get_permuted_w(kk, ki, kj, ka, kb, kc, blk)?
                .transpose(&[0, 1, 2, 4, 5, 3])?,
            1.0,
        )?;
        out.zip_assign(
            &self
                .get_permuted_w(ki, kk, kj, ka, kb, kc, blk)?
                .transpose(&[0, 1, 2, 3, 5, 4])?,
            -2.0,
        )?;
        out.zip_assign(
            &self
                .get_permuted_w(kk, kj, ki, ka, kb, kc, blk)?
                .transpose(&[0, 1, 2, 5, 4, 3])?,
            -2.0,
        )?;
        out.zip_assign(
            &self
                .get_permuted_w(kj, ki, kk, ka, kb, kc, blk)?
                .transpose(&[0, 1, 2, 4, 3, 5])?,
            -2.0,
        )?;
        Ok(out)
    }

    /// `get_v` — `:135-140`.
    #[allow(clippy::too_many_arguments)]
    fn get_v(
        &self,
        ki: usize,
        kj: usize,
        kk: usize,
        ka: usize,
        _kb: usize,
        _kc: usize,
        blk: VirBlock,
        kk_eq_kc: bool,
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
        if !kk_eq_kc {
            return Ok(out);
        }
        let t1k = self
            .t1
            .slice_leading(&[kk])?
            .slice_axes(&[(0, nocc), blk.c])?;
        let oovv = self
            .eris
            .blk(Blk::Oovv, ki, kj, ka)?
            .slice_axes(&[(0, nocc), (0, nocc), blk.a, blk.b])?
            .conj();
        out.add_assign(&einsum("kc,ijab->abcijk", &[&t1k, &oovv])?)?;
        let fovk = self.fov[kk].slice_axes(&[(0, nocc), blk.c])?;
        let t2a = self.t2.slice_leading(&[ki, kj, ka])?.slice_axes(&[
            (0, nocc),
            (0, nocc),
            blk.a,
            blk.b,
        ])?;
        let _ = nvir;
        out.add_assign(&einsum("kc,ijab->abcijk", &[&fovk, &t2a])?)?;
        Ok(out)
    }

    /// `get_permuted_v` — `:142-151`.
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
        let p = |x: (usize, usize), y: (usize, usize), z: (usize, usize)| VirBlock {
            a: x,
            b: y,
            c: z,
        };
        let mut out = self.get_v(ki, kj, kk, ka, kb, kc, p(blk.a, blk.b, blk.c), kk == kc)?;
        out.add_assign(
            &self
                .get_v(kj, kk, ki, kb, kc, ka, p(blk.b, blk.c, blk.a), ki == ka)?
                .transpose(&[2, 0, 1, 5, 3, 4])?,
        )?;
        out.add_assign(
            &self
                .get_v(kk, ki, kj, kc, ka, kb, p(blk.c, blk.a, blk.b), kj == kb)?
                .transpose(&[1, 2, 0, 4, 5, 3])?,
        )?;
        out.add_assign(
            &self
                .get_v(ki, kk, kj, ka, kc, kb, p(blk.a, blk.c, blk.b), kj == kb)?
                .transpose(&[0, 2, 1, 3, 5, 4])?,
        )?;
        out.add_assign(
            &self
                .get_v(kk, kj, ki, kc, kb, ka, p(blk.c, blk.b, blk.a), ki == ka)?
                .transpose(&[2, 1, 0, 5, 4, 3])?,
        )?;
        out.add_assign(
            &self
                .get_v(kj, ki, kk, kb, ka, kc, p(blk.b, blk.a, blk.c), kk == kc)?
                .transpose(&[1, 0, 2, 4, 3, 5])?,
        )?;
        Ok(out)
    }
}

/// `kernel(mycc, eris, t1, t2)` — `kccsd_t_rhf_slow.py:48-215`.
///
/// `vir_blksize` is upstream's `tasks` blocking; `None` uses `nvir`, i.e. no
/// blocking. **The energy is independent of it** and 16-08 test 3 asserts so.
///
/// # Errors
/// Propagates the ERI access, the padding surface and every shape check.
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
    let (nkpts, nocc, nvir) = (eris.nkpts, eris.nocc, eris.nvir);
    let mo_e_o: Vec<Vec<f64>> = eris.mo_energy.iter().map(|e| e[..nocc].to_vec()).collect();
    let mo_e_v: Vec<Vec<f64>> = eris.mo_energy.iter().map(|e| e[nocc..].to_vec()).collect();
    let fov: Vec<ZArr> = (0..nkpts)
        .map(|k| eris.fov(k))
        .collect::<Result<_, _>>()?;
    let (nz_o, nz_v) =
        match padding_k_idx(&padded.nmo_per_kpt, &padded.nocc_per_kpt, PaddingKind::Split) {
            Ok(PaddingIdx::Split { occupied, virtuals }) => (occupied, virtuals),
            Ok(_) => return Err(PbcCcError::Shape("padding_k_idx returned a joint set".into())),
            Err(e) => return Err(PbcCcError::Shape(format!("padding_k_idx: {e}"))),
        };

    let ctx = Ctx {
        eris,
        t1,
        t2,
        fov,
        kconserv,
        nocc,
        nvir,
    };
    let task_list = tasks(nvir, vir_blksize.unwrap_or(nvir));
    let mut terms_re: Vec<f64> = Vec::new();
    let mut terms_im: Vec<f64> = Vec::new();

    for ka in 0..nkpts {
        for kb in 0..=ka {
            for ki in 0..nkpts {
                for kj in 0..nkpts {
                    for kk in 0..nkpts {
                        // `:172-178` — eijk with LARGE_DENOM at padded orbitals.
                        let mut eijk = vec![LARGE_DENOM; nocc * nocc * nocc];
                        for &i in &nz_o[ki] {
                            for &j in &nz_o[kj] {
                                for &k in &nz_o[kk] {
                                    if i < nocc && j < nocc && k < nocc {
                                        eijk[(i * nocc + j) * nocc + k] =
                                            mo_e_o[ki][i] + mo_e_o[kj][j] + mo_e_o[kk][k];
                                    }
                                }
                            }
                        }

                        // `:182` — the fourth k of the triples amplitude.
                        let k3 = get_kconserv3(
                            a,
                            kpts,
                            &[
                                KIdx::One(ki),
                                KIdx::One(kj),
                                KIdx::One(kk),
                                KIdx::One(ka),
                                KIdx::One(kb),
                            ],
                        );
                        let kc = k3.data[0] as usize;

                        // `:184-185` — the (ka >= kb >= kc) restriction and its
                        // multiplicity, `:187-192`.
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

                        let mut eabc = vec![LARGE_DENOM; nvir * nvir * nvir];
                        for &x in &nz_v[ka] {
                            for &y in &nz_v[kb] {
                                for &z in &nz_v[kc] {
                                    if x < nvir && y < nvir && z < nvir {
                                        eabc[(x * nvir + y) * nvir + z] =
                                            mo_e_v[ka][x] + mo_e_v[kb][y] + mo_e_v[kc][z];
                                    }
                                }
                            }
                        }

                        for &blk in &task_list {
                            let (na, nb, nc) = (
                                blk.a.1 - blk.a.0,
                                blk.b.1 - blk.b.0,
                                blk.c.1 - blk.c.0,
                            );
                            let mut pwijk = ctx.get_permuted_w(ki, kj, kk, ka, kb, kc, blk)?;
                            let mut v = ctx.get_permuted_v(ki, kj, kk, ka, kb, kc, blk)?;
                            v.scale(0.5);
                            pwijk.add_assign(&v)?;

                            let mut rwijk = ctx.get_rw(ki, kj, kk, ka, kb, kc, blk)?;
                            // `:203` rwijk / eijkabc, elementwise.
                            for x in 0..na {
                                for y in 0..nb {
                                    for z in 0..nc {
                                        let d_abc = eabc[((x + blk.a.0) * nvir + (y + blk.b.0))
                                            * nvir
                                            + (z + blk.c.0)];
                                        for i in 0..nocc {
                                            for j in 0..nocc {
                                                for k in 0..nocc {
                                                    let d = eijk[(i * nocc + j) * nocc + k] - d_abc;
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
                            // `:205` — the FOURTH conjugation, on `rwijk`.
                            let (re, im) =
                                einsum("abcijk,abcijk->", &[&pwijk, &rwijk.conj()])?.at(&[])?;
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
            "non-zero imaginary part of the CCSD(T) energy (kccsd_t_rhf_slow.py:210)"
        );
    }
    Ok(re)
}
