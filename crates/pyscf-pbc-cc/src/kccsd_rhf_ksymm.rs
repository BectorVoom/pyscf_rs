//! `KsymAdaptedRCCSD` — restricted k-point coupled cluster over an irreducible
//! k-quartet set (`pyscf/pbc/cc/kccsd_rhf_ksymm.py`, 806 l), plan 17-09.
//!
//! # What the symmetry buys, measured
//!
//! On upstream's own fixture (He, two GTOs, `[2,2,2]`,
//! `measurements/gate_kccsd_ksymm.out`) there are **120 IBZ k-quartets against
//! 512 k-triples**. Every integral transform and every intermediate block is
//! built on those 120 and unfolded to the zone by rotation; the amplitude
//! equations then read the unfolded arrays, because they index
//! `t2[kk, kj, kc]` at arbitrary triples and re-deriving each read through
//! `transform_4d` would cost more than the storage saves. That is upstream's
//! `ktensor_direct = False` default (`:389-395`) and it is what this port
//! ships. See [`crate::ksymm_common`] for the container seam.
//!
//! # DEVIATION D-17-09-01 — upstream's second T1 quartet term tests a STALE
//! # loop variable, and this port does not
//!
//! `update_amps`' second quartet loop (`kccsd_rhf_ksymm.py:103-117`) unpacks
//! `kk, kl, ki, kd = kq` and then guards its `t1 t1` term with
//!
//! ```text
//! if kk == ka and kl == kc:
//!     tau_term_1 += einsum('ka,lc->klac', t1[ka], t1[kc])
//! ```
//!
//! **`kc` is not bound in that loop.** It is left over from the PRECEDING one
//! (`:89`, `ki, kk, kc, kd = kq`), so the guard tests an index belonging to an
//! unrelated quartet. Momentum conservation on `kq` gives `kd = kk + kl - ki`,
//! and this term has `ka = ki`, so when `kk == ka` the partner index is `kl`
//! itself: the intended guard is `kk == ka` alone, with `t1[kl]`. (When the
//! stale `kc` happens to equal `kl`, upstream's term is right; the rest of the
//! time it is simply missing.)
//!
//! **Measured** — `measurements/gate_kccsd_stale_kc.py` / `.out`, upstream
//! PySCF 2.12.1 monkey-patched against itself on the He `[2,2,2]` fixture:
//!
//! | `e_corr` | value | vs full-BZ `KRCCSD` |
//! |---|---|---|
//! | upstream ksymm, stale `kc` | `-0.007379123020832` | **6.917e-11** |
//! | ksymm with `kk == ka` | `-0.007379123132509` | **4.250e-11** |
//! | full-BZ `KRCCSD`, same mean field | `-0.007379123090006` | — |
//!
//! The corrected guard moves `e_corr` by `1.117e-10` and lands **closer** to
//! the full-BZ answer, which is the only reference either version is trying to
//! reproduce. This port ships the corrected guard, gates against its own
//! full-BZ `KRCCSD` rather than against upstream's k-symmetric number, and
//! records the discrepancy for upstream (17-13 §8).
//!
//! # `ktensor_direct` is NOT shipped
//!
//! `ktensor_direct = True` keeps every tensor block-sparse and recomputes
//! non-IBZ blocks on the fly. It changes no number — it is a memory/CPU
//! trade — and upstream's own tests never set it. Shipping it would double the
//! surface with no oracle, so it is refused by name in [`KsymRccsdOpts`].

use pyscf_algebra::{CTensor, oracle_sum};
use pyscf_diis::Diis;
use pyscf_pbc_df::{MoCoeff, PeriodicDf};
use pyscf_pbc_lib::{Kconserv, KptsHelper};
use pyscf_pbc_mp::PaddedMos;
use pyscf_pbc_symm::kpts::{KPoints, KQuartets, MORotationMatrix};
use pyscf_pbc_symm::ktensor::OrbSpace;

use crate::error::PbcCcError;
use crate::kccsd_rhf::{KAmplitudeSubspace, KrccsdOpts, get_eia, split_padding};
use crate::keris::Blk;
use crate::kintermediates_rhf_ksymm as imdk;
use crate::ksymm_common::{
    KsymCtx, add_stored2, add_stored4, get_stored2, get_stored4, ibz4_index, rot, set_stored2,
    set_stored4, to_dense2,
};
use crate::zarr::{ZArr, einsum, einsum_scaled};

fn shape(what: impl Into<String>) -> PbcCcError {
    PbcCcError::Shape(what.into())
}

fn df_err(e: pyscf_pbc_df::PbcDfError) -> PbcCcError {
    PbcCcError::Shape(format!("density fitting: {e}"))
}

// =====================================================================
// `_PhysicistsERIs` (`:686-780`) + `_make_eris_incore` (`:503-573`)
// =====================================================================

/// The seven MO integral blocks, built on the IBZ k-quartets and unfolded.
///
/// Dense after construction, exactly as upstream's `ktensor_direct = False`
/// default leaves them (`:568-575`), so [`KsymEris::blk`] is a plain slice and
/// the amplitude equations read it like any other array.
#[derive(Debug)]
pub struct KsymEris {
    pub nkpts: usize,
    pub nocc: usize,
    pub nmo: usize,
    pub nvir: usize,
    /// `[nkpts, nmo, nmo]`.
    pub fock: ZArr,
    pub mo_energy: Vec<Vec<f64>>,
    pub mo_coeff: Vec<MoCoeff>,
    pub madelung: f64,
    /// How many `ao2mo` transforms the IBZ loop ran. REPORTED, not asserted:
    /// it is the saving this whole plan exists for.
    pub n_ao2mo: usize,
    blocks: Vec<(Blk, ZArr)>,
}

impl KsymEris {
    /// `ao2mo` (`:397-419`) + `_make_eris_incore` (`:503-573`).
    ///
    /// `fock` / `mo_energy` / `madelung` come from
    /// [`crate::keris::KEris::build_fock`], which this SHARES rather than
    /// duplicating: `_common_init_` (`:718-780`) is character for character
    /// the non-symmetric one.
    ///
    /// # Errors
    /// Propagates the density-fitting builder and every shape check.
    pub fn build(
        ctx: &KsymCtx<'_>,
        with_df: &dyn PeriodicDf,
        padded: &PaddedMos,
        fock: ZArr,
        mo_energy: Vec<Vec<f64>>,
        madelung: f64,
    ) -> Result<Self, PbcCcError> {
        let nkpts = ctx.nkpts();
        // Its OWN helper: the symmetry map here is built over a RESTRICTED
        // k-point list, which is not the map any other consumer wants.
        let mut khelper = KptsHelper::without_symm_map(&with_df.cell().a, with_df.kpts());
        if with_df.kpts().len() != nkpts {
            return Err(shape(
                "KsymEris needs a density-fitting object over the FULL BZ",
            ));
        }
        let (nocc, nmo) = (padded.nocc, padded.nmo);
        let nvir = nmo - nocc;
        let nq = ctx.kqrts_ibz().len();

        // `:526` — `kptlist = kqrts_ibz[:,:3][:,[0,2,1]]`: the IBZ quartets in
        // CHEMISTS' order. The symmetry map is built over that restricted list,
        // so every orbit member is itself an IBZ triple and no write is
        // discarded.
        let kptlist: Vec<[usize; 3]> = ctx.kqrts_ibz().iter().map(|q| [q[0], q[2], q[1]]).collect();
        khelper.build_symm_map(Some(&kptlist));
        let entries = khelper
            .symm_map
            .as_ref()
            .ok_or_else(|| shape("build_symm_map produced no map"))?
            .entries()
            .to_vec();

        let mut stores: Vec<(Blk, Vec<CTensor>)> = Blk::ALL
            .iter()
            .map(|&b| {
                let len: usize = b.dims(nocc, nvir).iter().product();
                (b, vec![CTensor::zeros(len); nq])
            })
            .collect();

        let mo_refs: Vec<&MoCoeff> = padded.mo_coeff.iter().collect();
        let inv = 1.0 / nkpts as f64;
        let mut n_ao2mo = 0usize;
        for (rep, orbit) in &entries {
            let [ikp, ikq, ikr] = *rep;
            let iks = khelper.kconserv.get(ikp, ikq, ikr) as usize;
            let eri = with_df
                .ao2mo(
                    [mo_refs[ikp], mo_refs[ikq], mo_refs[ikr], mo_refs[iks]],
                    [ikp, ikq, ikr, iks],
                    false,
                )
                .map_err(df_err)?
                .restore_s1();
            n_ao2mo += 1;
            let eri_kpt = ZArr::from_ctensor(&[nmo, nmo, nmo, nmo], eri.data)?;
            for &[kp, kq, kr] in orbit {
                let raw = ZArr::from_ctensor(
                    &[nmo; 4],
                    khelper
                        .transform_symm(eri_kpt.data(), [nmo; 4], kp, kq, kr)
                        .map_err(shape)?,
                )?;
                // `:544` — `.transpose(0, 2, 1, 3)`: chemists' -> physicists'.
                let symm_block = raw.transpose(&[0, 2, 1, 3])?;
                let klc = [kp, kr, kq];
                let slot = ibz4_index(ctx.kpts, ctx.kqrts, klc).ok_or_else(|| {
                    shape(format!(
                        "kccsd_rhf_ksymm: the symmetry orbit produced the \
                         non-representative triple {klc:?}"
                    ))
                })?;
                for (b, store) in stores.iter_mut() {
                    let sp = b.spaces();
                    let d = b.dims(nocc, nvir);
                    let mut out = CTensor::zeros(d.iter().product());
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
                                    let (re, im) = symm_block.at(&s)?;
                                    let f = ((i0 * d[1] + i1) * d[2] + i2) * d[3] + i3;
                                    out.re[f] = re * inv;
                                    out.im[f] = im * inv;
                                }
                            }
                        }
                    }
                    store[slot] = out;
                }
            }
        }

        // `:568-575` — `todense()`: unfold each block to the whole zone.
        let mut blocks = Vec::with_capacity(Blk::ALL.len());
        for (b, store) in stores {
            let d = b.dims(nocc, nvir);
            let blen: usize = d.iter().product();
            let mut ibz = ZArr::zeros(&[nq, d[0], d[1], d[2], d[3]]);
            for (i, blk) in store.iter().enumerate() {
                ibz.data_mut().re[i * blen..(i + 1) * blen].copy_from_slice(&blk.re);
                ibz.data_mut().im[i * blen..(i + 1) * blen].copy_from_slice(&blk.im);
            }
            let label = ctx.label_for(b);
            blocks.push((b, ctx.dense4(&ibz, d, label, &ctx.labels.ccnn)?));
        }

        Ok(Self {
            nkpts,
            nocc,
            nmo,
            nvir,
            fock,
            mo_energy,
            mo_coeff: padded.mo_coeff.clone(),
            madelung,
            n_ao2mo,
            blocks,
        })
    }

    /// One block at the physicists' triple `[ki, kj, ka]`.
    ///
    /// # Errors
    /// [`PbcCcError::Shape`] for an unknown block or an out-of-range k-index.
    pub fn blk(&self, b: Blk, k0: usize, k1: usize, k2: usize) -> Result<ZArr, PbcCcError> {
        let t = self
            .blocks
            .iter()
            .find(|(x, _)| *x == b)
            .ok_or_else(|| shape(format!("KsymEris has no {} block", b.name())))?;
        t.1.slice_leading(&[k0, k1, k2])
    }

    /// `eris.fock[k]`.
    ///
    /// # Errors
    /// As [`ZArr::slice_leading`].
    pub fn fock_at(&self, k: usize) -> Result<ZArr, PbcCcError> {
        self.fock.slice_leading(&[k])
    }

    /// `eris.fock[k, :nocc, nocc:]`.
    ///
    /// # Errors
    /// As [`ZArr::slice_axes`].
    pub fn fov(&self, k: usize) -> Result<ZArr, PbcCcError> {
        self.fock_at(k)?
            .slice_axes(&[(0, self.nocc), (self.nocc, self.nmo)])
    }

    /// `eris.fock[k, :nocc, :nocc]`.
    ///
    /// # Errors
    /// As [`ZArr::slice_axes`].
    pub fn foo(&self, k: usize) -> Result<ZArr, PbcCcError> {
        self.fock_at(k)?
            .slice_axes(&[(0, self.nocc), (0, self.nocc)])
    }

    /// `eris.fock[k, nocc:, nocc:]`.
    ///
    /// # Errors
    /// As [`ZArr::slice_axes`].
    pub fn fvv(&self, k: usize) -> Result<ZArr, PbcCcError> {
        self.fock_at(k)?
            .slice_axes(&[(self.nocc, self.nmo), (self.nocc, self.nmo)])
    }
}

// =====================================================================
// `energy` (`:359-386`)
// =====================================================================

/// `energy(cc, t1, t2, eris)` — `kccsd_rhf_ksymm.py:359-386`.
///
/// `t1` / `t2` are the DENSE, already-unfolded amplitudes; the sums run over
/// the IBZ k-points and IBZ k-quartets with their weights.
///
/// # Determinism
/// Every term is pushed onto a fixed-order buffer and reduced by
/// [`oracle_sum`], as `kccsd_rhf::energy` does — the `nkpts^3` accumulation is
/// a D-PBC-17 shape.
///
/// # Errors
/// Propagates every ERI access and shape check.
pub fn energy(ctx: &KsymCtx<'_>, t1: &ZArr, t2: &ZArr, eris: &KsymEris) -> Result<f64, PbcCcError> {
    let nkpts = ctx.nkpts();
    let mut re: Vec<f64> = Vec::new();
    let mut im: Vec<f64> = Vec::new();

    for i in 0..ctx.nkpts_ibz() {
        let ki = ctx.kpts.ibz2bz[i];
        let w = ctx.kpts.weights_ibz[i];
        let (r, m) = einsum_scaled(
            "ia,ia->",
            &[&eris.fov(ki)?, &t1.slice_leading(&[ki])?],
            2.0 * w,
        )?
        .at(&[])?;
        re.push(r);
        im.push(m);
    }

    // `:373-378` — `tau` is a `KsymmArray` upstream, but every read of it below
    // is at the SAME IBZ quartet that wrote it (`:381-384`), so the unfold is
    // never exercised and a per-quartet buffer is the identical computation.
    let n3 = (nkpts as f64).powi(3);
    for (k, kq) in ctx.kqrts_ibz().iter().enumerate() {
        let [ki, kj, ka, kb] = *kq;
        let mut tau = t2.slice_leading(&[ki, kj, ka])?;
        if ki == ka && kj == kb {
            tau.add_assign(&einsum(
                "ia,jb->ijab",
                &[&t1.slice_leading(&[ki])?, &t1.slice_leading(&[kj])?],
            )?)?;
        }
        let w = ctx.kqrts.weights_ibz[k] * n3;
        let (r, m) = einsum_scaled(
            "ijab,ijab->",
            &[&tau, &eris.blk(Blk::Oovv, ki, kj, ka)?],
            2.0 * w,
        )?
        .at(&[])?;
        re.push(r);
        im.push(m);
        let (r, m) = einsum_scaled(
            "ijab,ijba->",
            &[&tau, &eris.blk(Blk::Oovv, ki, kj, kb)?],
            -w,
        )?
        .at(&[])?;
        re.push(r);
        im.push(m);
    }

    let e_re = oracle_sum(&re) / nkpts as f64;
    let e_im = oracle_sum(&im) / nkpts as f64;
    if e_im.abs() > 1e-4 {
        tracing::warn!(
            imaginary = e_im,
            "non-zero imaginary part in the k-symmetric KRCCSD energy \
             (kccsd_rhf_ksymm.py:385)"
        );
    }
    Ok(e_re)
}

// =====================================================================
// `init_amps` (`:421-471`)
// =====================================================================

/// `init_amps(eris)` — `kccsd_rhf_ksymm.py:421-471`. Returns
/// `(emp2, t1_ibz, t2_ibz)` with the amplitudes in their **IBZ block stores**:
/// `t1_ibz` is `[nkpts_ibz, nocc, nvir]`, `t2_ibz` is
/// `[len(kqrts_ibz), nocc, nocc, nvir, nvir]`.
///
/// # Errors
/// Propagates every ERI access and shape check.
pub fn init_amps(
    ctx: &KsymCtx<'_>,
    eris: &KsymEris,
    padded: &PaddedMos,
) -> Result<(f64, ZArr, ZArr), PbcCcError> {
    let (no, nv, nkpts) = (ctx.nocc, ctx.nvir, ctx.nkpts());
    let nq = ctx.kqrts_ibz().len();
    let t1 = ZArr::zeros(&[ctx.nkpts_ibz(), no, nv]);
    let mut t2 = ZArr::zeros(&[nq, no, no, nv, nv]);
    let mo_e_o: Vec<Vec<f64>> = eris.mo_energy.iter().map(|e| e[..no].to_vec()).collect();
    let mo_e_v: Vec<Vec<f64>> = eris.mo_energy.iter().map(|e| e[no..].to_vec()).collect();
    let (nz_o, nz_v) = split_padding(padded)?;

    let n3 = (nkpts as f64).powi(3);
    let mut terms: Vec<f64> = Vec::with_capacity(nq);
    for (i, kq) in ctx.kqrts_ibz().iter().enumerate() {
        let [ki, kj, ka, kb] = *kq;
        let w = ctx.kqrts.weights_ibz[i] * n3;
        let eia = get_eia(&mo_e_o, &mo_e_v, ki, ka, &nz_o, &nz_v);
        let ejb = get_eia(&mo_e_o, &mo_e_v, kj, kb, &nz_o, &nz_v);
        let ijab = eris.blk(Blk::Oovv, ki, kj, ka)?;
        let ijba = eris.blk(Blk::Oovv, ki, kj, kb)?;
        let t = divide_by_eijab(&ijab.conj(), &eia, &ejb, no, nv)?;
        let mut woovv = ijab.clone();
        woovv.scale(2.0);
        woovv.sub_assign(&ijba.transpose(&[0, 1, 3, 2])?)?;
        let (r, _) = einsum_scaled("ijab,ijab->", &[&t, &woovv], w)?.at(&[])?;
        terms.push(r);
        t2.set_leading(&[i], &t)?;
    }
    Ok((oracle_sum(&terms) / nkpts as f64, t1, t2))
}

/// `x / eijab` with `eijab[i,j,a,b] = eia[i,a] + ejb[j,b]`.
fn divide_by_eijab(
    x: &ZArr,
    eia: &[f64],
    ejb: &[f64],
    no: usize,
    nv: usize,
) -> Result<ZArr, PbcCcError> {
    let mut out = x.clone();
    for i in 0..no {
        for j in 0..no {
            for a in 0..nv {
                for b in 0..nv {
                    let d = eia[i * nv + a] + ejb[j * nv + b];
                    let f = ((i * no + j) * nv + a) * nv + b;
                    out.data_mut().re[f] /= d;
                    out.data_mut().im[f] /= d;
                }
            }
        }
    }
    Ok(out)
}

// =====================================================================
// `update_amps` (`:41-274`) and `add_vvvv_` (`:276-357`)
// =====================================================================

/// `update_amps(cc, t1, t2, eris)` — `kccsd_rhf_ksymm.py:41-274`.
///
/// Takes and returns the **IBZ block stores**; the dense amplitudes are
/// materialised inside, exactly where upstream calls `t1.todense()` /
/// `t2.todense()` (`:74`, `:77`).
///
/// # Errors
/// Propagates every ERI access, intermediate build and shape check.
pub fn update_amps(
    ctx: &KsymCtx<'_>,
    t1_ibz: &ZArr,
    t2_ibz: &ZArr,
    eris: &KsymEris,
    padded: &PaddedMos,
    opts: &KrccsdOpts,
) -> Result<(ZArr, ZArr), PbcCcError> {
    let (no, nv, nkpts) = (ctx.nocc, ctx.nvir, ctx.nkpts());
    let nq = ctx.kqrts_ibz().len();
    let kconserv = ctx.kconserv;
    let mo_e_o: Vec<Vec<f64>> = eris.mo_energy.iter().map(|e| e[..no].to_vec()).collect();
    let mo_e_v: Vec<Vec<f64>> = eris
        .mo_energy
        .iter()
        .map(|e| e[no..].iter().map(|x| x + opts.level_shift).collect())
        .collect();
    let (nz_o, nz_v) = split_padding(padded)?;

    let t1 = ctx.dense2(t1_ibz, [no, nv], &ctx.labels.ov, &ctx.labels.nc)?;
    let t2 = ctx.dense4(t2_ibz, [no, no, nv, nv], &ctx.labels.oovv, &ctx.labels.nncc)?;

    let mut f_oo = imdk::cc_foo_arr(ctx, &t1, &t2, eris)?;
    let mut f_vv = imdk::cc_fvv_arr(ctx, &t1, &t2, eris)?;
    let fov_imd = imdk::cc_fov(ctx, &t1, &t2, eris)?;
    let mut l_oo = imdk::loo_arr(ctx, &t1, &t2, eris)?;
    let mut l_vv = imdk::lvv_arr(ctx, &t1, &t2, eris)?;

    // `:68-73` — move the energy terms to the other side. Upstream mutates the
    // store through the numpy VIEW `transform_2d` returns at a representative
    // (`ktensor.py:269-270`); this port does the read-modify-write explicitly.
    for i in 0..ctx.nkpts_ibz() {
        let ki = ctx.kpts.ibz2bz[i];
        shift_diag(&mut f_oo, ctx, ki, &mo_e_o[ki], no)?;
        shift_diag(&mut l_oo, ctx, ki, &mo_e_o[ki], no)?;
        shift_diag(&mut f_vv, ctx, ki, &mo_e_v[ki], nv)?;
        shift_diag(&mut l_vv, ctx, ki, &mo_e_v[ki], nv)?;
    }

    let mut t1new = ctx.empty2([no, nv], &ctx.labels.ov, &ctx.labels.nc)?;
    let mut t2new = ctx.empty4([no, no, nv, nv], &ctx.labels.oovv, &ctx.labels.nncc)?;

    // ---------------------------------------------------------------- T1
    // `:80-86`
    for i in 0..ctx.nkpts_ibz() {
        let ka = ctx.kpts.ibz2bz[i];
        let ki = ka;
        let fov = eris.fov(ka)?;
        let t1a = t1.slice_leading(&[ka])?;
        let t1i = t1.slice_leading(&[ki])?;
        let mut acc = fov.conj();
        acc.add_assign(&einsum_scaled(
            "kc,ka,ic->ia",
            &[&eris.fov(ki)?, &t1a, &t1i],
            -2.0,
        )?)?;
        acc.add_assign(&einsum(
            "ac,ic->ia",
            &[&get_stored2(&f_vv, ctx.kpts, ka, [nv, nv])?, &t1i],
        )?)?;
        acc.sub_assign(&einsum(
            "ki,ka->ia",
            &[&get_stored2(&f_oo, ctx.kpts, ki, [no, no])?, &t1a],
        )?)?;
        set_stored2(&mut t1new, ctx.kpts, ka, &acc)?;
    }

    // `:88-101` — the `Svovv` quartet term, symmetrised over the STABILISER.
    for (i, kq) in ctx.kqrts_ibz().iter().enumerate() {
        let [ki, kk, kc, kd] = *kq;
        let ka = ki;
        let mut svovv = eris.blk(Blk::Vovv, ka, kk, kc)?;
        svovv.scale(2.0);
        svovv.sub_assign(&eris.blk(Blk::Vovv, ka, kk, kd)?.transpose(&[0, 1, 3, 2])?)?;
        let mut tau = t2.slice_leading(&[ki, kk, kc])?;
        if ki == kc && kk == kd {
            tau.add_assign(&einsum(
                "ic,kd->ikcd",
                &[&t1.slice_leading(&[ki])?, &t1.slice_leading(&[kk])?],
            )?)?;
        }
        let fock = einsum("akcd,ikcd->ia", &[&svovv, &tau])?;
        for (_, iop) in ctx.kqrts.loop_stabilizer(i) {
            let roo = rot(ctx.rmat, OrbSpace::Occ, ka, iop)?;
            let rvv = rot(ctx.rmat, OrbSpace::Vir, ka, iop)?;
            let d = einsum("ia,im,ae->me", &[&fock, &roo, &rvv.conj()])?;
            add_stored2(&mut t1new, ctx.kpts, ka, &d)?;
        }
    }

    // `:103-117` — the `Sooov` quartet term. **This one SCATTERS** to every
    // IBZ image of `ka`, and its `t1 t1` guard is D-17-09-01 (see the module
    // doc): `kk == ka` alone, with `t1[kl]`, not upstream's stale `kc`.
    for (i, kq) in ctx.kqrts_ibz().iter().enumerate() {
        let [kk, kl, ki, _kd] = *kq;
        let ka = ki;
        let mut sooov = eris.blk(Blk::Ooov, kk, kl, ki)?;
        sooov.scale(2.0);
        sooov.sub_assign(&eris.blk(Blk::Ooov, kl, kk, ki)?.transpose(&[1, 0, 2, 3])?)?;
        let mut tau = t2.slice_leading(&[kk, kl, ka])?;
        if kk == ka {
            tau.add_assign(&einsum(
                "ka,lc->klac",
                &[&t1.slice_leading(&[ka])?, &t1.slice_leading(&[kl])?],
            )?)?;
        }
        let fock = einsum_scaled("klic,klac->ia", &[&sooov, &tau], -1.0)?;
        for &iop in &ctx.kqrts.stars_ops[i] {
            let ka_p = ctx.kpts.k2opk[ka][iop];
            if ka_p < 0 {
                continue;
            }
            let ka_p = ka_p as usize;
            if crate::ksymm_common::ibz2_index(ctx.kpts, ka_p).is_none() {
                continue;
            }
            let roo = rot(ctx.rmat, OrbSpace::Occ, ka, iop)?;
            let rvv = rot(ctx.rmat, OrbSpace::Vir, ka, iop)?;
            let d = einsum("ia,im,ae->me", &[&fock, &roo, &rvv.conj()])?;
            add_stored2(&mut t1new, ctx.kpts, ka_p, &d)?;
        }
    }

    // `:119-133` — the `Fov` / `voov` / `ovov` term.
    for (i, kq) in ctx.kqrts_ibz().iter().enumerate() {
        let [ki, kk, ka, kc] = *kq;
        if !(ka == ki && kk == kc) {
            continue;
        }
        let mut tau = t2.slice_leading(&[kk, ki, kk])?;
        tau.scale(2.0);
        tau.sub_assign(&t2.slice_leading(&[ki, kk, kk])?.transpose(&[1, 0, 2, 3])?)?;
        if ki == kk {
            tau.add_assign(&einsum(
                "ic,ka->kica",
                &[&t1.slice_leading(&[ki])?, &t1.slice_leading(&[ka])?],
            )?)?;
        }
        let mut fock = einsum("kc,kica->ia", &[&fov_imd.slice_leading(&[kc])?, &tau])?;
        fock.add_assign(&einsum_scaled(
            "akic,kc->ia",
            &[&eris.blk(Blk::Voov, ka, kk, ki)?, &t1.slice_leading(&[kc])?],
            2.0,
        )?)?;
        fock.sub_assign(&einsum(
            "kaic,kc->ia",
            &[&eris.blk(Blk::Ovov, kk, ka, ki)?, &t1.slice_leading(&[kc])?],
        )?)?;
        for (_, iop) in ctx.kqrts.loop_stabilizer(i) {
            let roo = rot(ctx.rmat, OrbSpace::Occ, ka, iop)?;
            let rvv = rot(ctx.rmat, OrbSpace::Vir, ka, iop)?;
            let d = einsum("ia,im,ae->me", &[&fock, &roo, &rvv.conj()])?;
            add_stored2(&mut t1new, ctx.kpts, ka, &d)?;
        }
    }

    // `:135-143` — the T1 denominator.
    for i in 0..ctx.nkpts_ibz() {
        let ki = ctx.kpts.ibz2bz[i];
        let ka = ki;
        let eia = get_eia(&mo_e_o, &mo_e_v, ki, ka, &nz_o, &nz_v);
        let mut blk = get_stored2(&t1new, ctx.kpts, ki, [no, nv])?;
        for (f, d) in eia.iter().enumerate().take(no * nv) {
            blk.data_mut().re[f] /= d;
            blk.data_mut().im[f] /= d;
        }
        set_stored2(&mut t1new, ctx.kpts, ki, &blk)?;
    }

    // ---------------------------------------------------------------- T2
    let loo_d = to_dense2(&l_oo, nkpts, [no, no])?;
    let lvv_d = to_dense2(&l_vv, nkpts, [nv, nv])?;

    // `:149-151`
    for kq in ctx.kqrts_ibz().iter() {
        let [ki, kj, ka, _kb] = *kq;
        set_stored4(
            &mut t2new,
            ctx.kpts,
            ctx.kqrts,
            [ki, kj, ka],
            &eris.blk(Blk::Oovv, ki, kj, ka)?.conj(),
        )?;
    }

    // `:153-192` — the oooo ladder.
    {
        let woooo = imdk::cc_woooo(ctx, &t1, &t2, eris)?;
        let t2_oooo = |ki: usize, kj: usize, ka: usize, kb: usize| -> Result<ZArr, PbcCcError> {
            let mut acc = ZArr::zeros(&[no, no, nv, nv]);
            for kl in 0..nkpts {
                let kk = kconserv.get(kj, kl, ki) as usize;
                let mut tau = t2.slice_leading(&[kk, kl, ka])?;
                if kl == kb && kk == ka {
                    tau.add_assign(&einsum(
                        "ic,jd->ijcd",
                        &[&t1.slice_leading(&[ka])?, &t1.slice_leading(&[kb])?],
                    )?)?;
                }
                acc.add_assign(&einsum_scaled(
                    "klij,klab->ijab",
                    &[&woooo.slice_leading(&[kk, kl, ki])?, &tau],
                    0.5,
                )?)?;
            }
            Ok(acc)
        };
        accumulate_quartets(ctx, &mut t2new, t2_oooo)?;
    }

    // `:194` — `add_vvvv_`.
    add_vvvv_(ctx, &mut t2new, &t1, &t2, eris)?;

    // `:196-218` — the L / singles-dressed terms.
    {
        let voov1 = |ki: usize, kj: usize, ka: usize, kb: usize| -> Result<ZArr, PbcCcError> {
            let t2ija = t2.slice_leading(&[ki, kj, ka])?;
            let mut acc = einsum("ac,ijcb->ijab", &[&lvv_d.slice_leading(&[ka])?, &t2ija])?;
            let mut nloo = loo_d.slice_leading(&[ki])?;
            nloo.scale(-1.0);
            acc.add_assign(&einsum("ki,kjab->ijab", &[&nloo, &t2ija])?)?;

            let kc = kj;
            let mut tmp2 = eris
                .blk(Blk::Vovv, kc, ki, kb)?
                .transpose(&[3, 2, 1, 0])?
                .conj();
            tmp2.sub_assign(&einsum(
                "kbic,ka->abic",
                &[&eris.blk(Blk::Ovov, ka, kb, ki)?, &t1.slice_leading(&[ka])?],
            )?)?;
            acc.add_assign(&einsum(
                "abic,jc->ijab",
                &[&tmp2, &t1.slice_leading(&[kj])?],
            )?)?;

            let kk = kb;
            let mut tmp2 = eris
                .blk(Blk::Ooov, kj, ki, kk)?
                .transpose(&[3, 2, 1, 0])?
                .conj();
            tmp2.add_assign(&einsum(
                "akic,jc->akij",
                &[&eris.blk(Blk::Voov, ka, kk, ki)?, &t1.slice_leading(&[kj])?],
            )?)?;
            acc.sub_assign(&einsum(
                "akij,kb->ijab",
                &[&tmp2, &t1.slice_leading(&[kb])?],
            )?)?;
            Ok(acc)
        };
        accumulate_quartets(ctx, &mut t2new, voov1)?;
    }

    // `:220-263` — the voov / vovo ring terms.
    {
        let wvoov = imdk::cc_wvoov(ctx, &t1, &t2, eris)?;
        let wvovo = imdk::cc_wvovo(ctx, &t1, &t2, eris)?;
        let voov2 = |ki: usize, kj: usize, ka: usize, kb: usize| -> Result<ZArr, PbcCcError> {
            let mut acc = ZArr::zeros(&[no, no, nv, nv]);
            for kk in 0..nkpts {
                let kc = kconserv.get(ka, ki, kk) as usize;
                let mut tv = wvoov.slice_leading(&[ka, kk, ki])?;
                tv.scale(2.0);
                tv.sub_assign(
                    &wvovo
                        .slice_leading(&[ka, kk, kc])?
                        .transpose(&[0, 1, 3, 2])?,
                )?;
                acc.add_assign(&einsum(
                    "akic,kjcb->ijab",
                    &[&tv, &t2.slice_leading(&[kk, kj, kc])?],
                )?)?;
                acc.sub_assign(&einsum(
                    "akic,kjbc->ijab",
                    &[
                        &wvoov.slice_leading(&[ka, kk, ki])?,
                        &t2.slice_leading(&[kk, kj, kb])?,
                    ],
                )?)?;
                let kc2 = kconserv.get(kk, ka, kj) as usize;
                acc.add_assign(&einsum_scaled(
                    "bkci,kjac->ijab",
                    &[
                        &wvovo.slice_leading(&[kb, kk, kc2])?,
                        &t2.slice_leading(&[kk, kj, ka])?,
                    ],
                    -1.0,
                )?)?;
            }
            Ok(acc)
        };
        accumulate_quartets(ctx, &mut t2new, voov2)?;
    }

    // `:265-274` — the T2 denominator.
    for kq in ctx.kqrts_ibz().iter() {
        let [ki, kj, ka, kb] = *kq;
        let eia = get_eia(&mo_e_o, &mo_e_v, ki, ka, &nz_o, &nz_v);
        let ejb = get_eia(&mo_e_o, &mo_e_v, kj, kb, &nz_o, &nz_v);
        let blk = get_stored4(&t2new, ctx.kpts, ctx.kqrts, [ki, kj, ka], [no, no, nv, nv])?;
        let out = divide_by_eijab(&blk, &eia, &ejb, no, nv)?;
        set_stored4(&mut t2new, ctx.kpts, ctx.kqrts, [ki, kj, ka], &out)?;
    }

    // Back to the IBZ block stores.
    let mut t1_out = ZArr::zeros(&[ctx.nkpts_ibz(), no, nv]);
    for i in 0..ctx.nkpts_ibz() {
        let ki = ctx.kpts.ibz2bz[i];
        t1_out.set_leading(&[i], &get_stored2(&t1new, ctx.kpts, ki, [no, nv])?)?;
    }
    let mut t2_out = ZArr::zeros(&[nq, no, no, nv, nv]);
    for (i, kq) in ctx.kqrts_ibz().iter().enumerate() {
        let [ki, kj, ka, _kb] = *kq;
        t2_out.set_leading(
            &[i],
            &get_stored4(&t2new, ctx.kpts, ctx.kqrts, [ki, kj, ka], [no, no, nv, nv])?,
        )?;
    }
    Ok((t1_out, t2_out))
}

/// The pattern upstream repeats at `:180-190`, `:209-217` and `:254-262`:
///
/// ```text
/// t2new[ki,kj,ka] += f(ki,kj,ka,kb)
/// if (kj,ki,kb,ka) is itself an IBZ quartet:
///     t2new[kj,ki,kb] += f(ki,kj,ka,kb).transpose(1,0,3,2)
/// else:
///     t2new[ki,kj,ka] += f(kj,ki,kb,ka).transpose(1,0,3,2)
/// ```
///
/// The `else` branch is the whole point and it is easy to get backwards: when
/// the interchanged quartet is NOT a representative, its contribution has
/// nowhere of its own to go, so it is folded back onto `(ki,kj,ka)` — which is
/// what `accumulate_pair` does implicitly in the non-symmetric kernel, where
/// every triple has a slot.
fn accumulate_quartets<F>(
    ctx: &KsymCtx<'_>,
    t2new: &mut pyscf_pbc_symm::ktensor::KsymmArray<'_>,
    f: F,
) -> Result<(), PbcCcError>
where
    F: Fn(usize, usize, usize, usize) -> Result<ZArr, PbcCcError>,
{
    let (no, nv) = (ctx.nocc, ctx.nvir);
    let dims = [no, no, nv, nv];
    for kq in ctx.kqrts_ibz().iter() {
        let [ki, kj, ka, kb] = *kq;
        let tmp = f(ki, kj, ka, kb)?;
        add_stored4(t2new, ctx.kpts, ctx.kqrts, [ki, kj, ka], &tmp)?;
        if ctx.kqrts_ibz().contains(&[kj, ki, kb, ka]) {
            add_stored4(
                t2new,
                ctx.kpts,
                ctx.kqrts,
                [kj, ki, kb],
                &tmp.transpose(&[1, 0, 3, 2])?,
            )?;
        } else {
            let swapped = f(kj, ki, kb, ka)?;
            add_stored4(
                t2new,
                ctx.kpts,
                ctx.kqrts,
                [ki, kj, ka],
                &swapped.transpose(&[1, 0, 3, 2])?,
            )?;
        }
    }
    let _ = dims;
    Ok(())
}

/// `add_vvvv_` (`:276-357`), the incore branch.
///
/// The `direct` / `Lpv` branch (`:288-303`) needs `eris.Lpv`, which this port
/// does not build for the k-symmetric ERIs, and the outcore branch differs only
/// in where `_Wvvvv` lives. Both collapse to this one once `_Wvvvv` is dense.
///
/// # Errors
/// Propagates every ERI access and shape check.
pub fn add_vvvv_(
    ctx: &KsymCtx<'_>,
    t2new: &mut pyscf_pbc_symm::ktensor::KsymmArray<'_>,
    t1: &ZArr,
    t2: &ZArr,
    eris: &KsymEris,
) -> Result<(), PbcCcError> {
    let nkpts = ctx.nkpts();
    let kconserv = ctx.kconserv;
    let wvvvv = imdk::cc_wvvvv(ctx, t1, t2, eris)?;

    // `:340-357` — group the IBZ quartets by their `(ka, kb)` pair so each
    // `Wvvvv[ka,kb,kc]` block is built once and consumed by every quartet that
    // shares it.
    let mut pairs: Vec<[usize; 2]> = ctx.kqrts_ibz().iter().map(|q| [q[2], q[3]]).collect();
    pairs.sort_unstable();
    pairs.dedup();
    for [ka, kb] in pairs {
        let members: Vec<[usize; 4]> = ctx
            .kqrts_ibz()
            .iter()
            .copied()
            .filter(|q| q[2] == ka && q[3] == kb)
            .collect();
        for kc in 0..nkpts {
            let kd = kconserv.get(ka, kc, kb) as usize;
            let w = wvvvv.slice_leading(&[ka, kb, kc])?;
            for q in &members {
                let [ki, kj, _, _] = *q;
                let mut tau = t2.slice_leading(&[ki, kj, kc])?;
                if ki == kc && kj == kd {
                    tau.add_assign(&einsum(
                        "ic,jd->ijcd",
                        &[&t1.slice_leading(&[ki])?, &t1.slice_leading(&[kj])?],
                    )?)?;
                }
                add_stored4(
                    t2new,
                    ctx.kpts,
                    ctx.kqrts,
                    [ki, kj, ka],
                    &einsum("abcd,ijcd->ijab", &[&w, &tau])?,
                )?;
            }
        }
    }
    Ok(())
}

/// `X[ki][diag_indices] -= e` on the STORED block.
fn shift_diag(
    x: &mut pyscf_pbc_symm::ktensor::KsymmArray<'_>,
    ctx: &KsymCtx<'_>,
    ki: usize,
    e: &[f64],
    n: usize,
) -> Result<(), PbcCcError> {
    let mut blk = get_stored2(x, ctx.kpts, ki, [n, n])?;
    for (i, ei) in e.iter().enumerate().take(n) {
        blk.data_mut().re[i * n + i] -= ei;
    }
    set_stored2(x, ctx.kpts, ki, &blk)
}

// =====================================================================
// The driver
// =====================================================================

/// Knobs of the k-symmetric amplitude iteration. A thin wrapper around
/// [`KrccsdOpts`] plus the one k-symmetric option, so nothing is re-defaulted.
#[derive(Debug, Clone, Copy, Default)]
pub struct KsymRccsdOpts {
    pub base: KrccsdOpts,
    /// `cc.ktensor_direct` (`:389-395`). **Only `false` is implemented** — see
    /// the module doc. `true` returns an error rather than silently ignoring
    /// the request.
    pub ktensor_direct: bool,
}

/// What [`kernel`] returns. `t1` / `t2` are the **IBZ block stores**; use
/// [`KsymCtx::dense2`] / [`KsymCtx::dense4`] to unfold them.
#[derive(Debug, Clone)]
pub struct KsymRccsdResult {
    pub e_corr: f64,
    pub emp2: f64,
    pub converged: bool,
    pub cycles: usize,
    /// `[nkpts_ibz, nocc, nvir]`.
    pub t1: ZArr,
    /// `[len(kqrts_ibz), nocc, nocc, nvir, nvir]`.
    pub t2: ZArr,
}

/// The amplitude iteration — `pyscf/cc/ccsd.py:kernel` over the IBZ stores.
///
/// # Errors
/// [`PbcCcError::Shape`] for `ktensor_direct = true`, plus every error
/// [`update_amps`] and [`energy`] raise.
pub fn kernel(
    ctx: &KsymCtx<'_>,
    eris: &KsymEris,
    padded: &PaddedMos,
    opts: &KsymRccsdOpts,
) -> Result<KsymRccsdResult, PbcCcError> {
    if opts.ktensor_direct {
        return Err(shape(
            "ktensor_direct = true is not implemented: it changes no number \
             (it trades memory for recomputation) and upstream's own tests \
             never set it, so this port ships no untested second surface. \
             kccsd_rhf_ksymm.py:389-395.",
        ));
    }
    let (no, nv) = (ctx.nocc, ctx.nvir);
    let (emp2, mut t1, mut t2) = init_amps(ctx, eris, padded)?;

    let t1_d = ctx.dense2(&t1, [no, nv], &ctx.labels.ov, &ctx.labels.nc)?;
    let t2_d = ctx.dense4(&t2, [no, no, nv, nv], &ctx.labels.oovv, &ctx.labels.nncc)?;
    let mut e_old = energy(ctx, &t1_d, &t2_d, eris)?;

    let mut diis: Option<Diis<KAmplitudeSubspace>> =
        opts.base.diis.then(|| Diis::new(opts.base.diis_space));
    let mut converged = false;
    let mut cycles = 0usize;

    for istep in 0..opts.base.max_cycle {
        cycles = istep + 1;
        let (mut t1new, mut t2new) = update_amps(ctx, &t1, &t2, eris, padded, &opts.base)?;

        // `ccsd.py:74-76` — `normt`, through `oracle_dot` so it is
        // thread-count invariant.
        let cur = KAmplitudeSubspace::from_amplitudes(&t1new, &t2new);
        let prev = KAmplitudeSubspace::from_amplitudes(&t1, &t2);
        let res = cur.residual(&prev);
        let normt = pyscf_algebra::oracle_dot(&res, &res).sqrt();

        if opts.base.iterative_damping < 1.0 {
            let f = opts.base.iterative_damping;
            damp(&mut t1new, &t1, f);
            damp(&mut t2new, &t2, f);
        }

        t1 = t1new;
        t2 = t2new;

        if let Some(stack) = diis.as_mut()
            && istep >= opts.base.diis_start_cycle
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

        let t1_d = ctx.dense2(&t1, [no, nv], &ctx.labels.ov, &ctx.labels.nc)?;
        let t2_d = ctx.dense4(&t2, [no, no, nv, nv], &ctx.labels.oovv, &ctx.labels.nncc)?;
        let e_new = energy(ctx, &t1_d, &t2_d, eris)?;
        let de = (e_new - e_old).abs();
        e_old = e_new;
        tracing::debug!(istep, e_corr = e_new, de, normt, "ksymm KRCCSD");
        if de < opts.base.conv_tol && normt < opts.base.conv_tol_normt {
            converged = true;
            break;
        }
    }

    Ok(KsymRccsdResult {
        e_corr: e_old,
        emp2,
        converged,
        cycles,
        t1,
        t2,
    })
}

fn damp(new: &mut ZArr, old: &ZArr, f: f64) {
    for i in 0..new.len() {
        new.data_mut().re[i] = f * new.data().re[i] + (1.0 - f) * old.data().re[i];
        new.data_mut().im[i] = f * new.data().im[i] + (1.0 - f) * old.data().im[i];
    }
}

// =====================================================================
// The assembled inputs — `KsymAdaptedRCCSD.__init__` (`:378-396`) +
// `ao2mo`'s `MORotationMatrix` build (`:401-403`)
// =====================================================================

/// Everything a k-symmetric KRCCSD run needs, derived once from a converged
/// k-symmetric SCF.
///
/// # The mean field must be the UNFOLDED one
///
/// `RCCSD.__init__` gets a NON-symmetry mean field (`kccsd_rhf_ksymm.py:381`'s
/// own comment), i.e. the full-BZ MO set that `KPoints::transform_mo_coeff`
/// produces from the IBZ one. That is not a convenience: the k-symmetric
/// amplitude equations assume `C[Rk]` IS `R C[k]`, and a mean field converged
/// independently at every BZ k-point is free to differ from that by a unitary
/// rotation inside each degenerate subspace — which would make every
/// `MORotationMatrix` block wrong in a way no shape check can see.
/// `pyscf_pbc_mp::unfold_kscf_result` produces the right input.
#[derive(Debug)]
pub struct KsymRccsdInputs {
    pub kqrts: KQuartets,
    pub rmat: MORotationMatrix,
    pub kconserv: Kconserv,
    pub padded: PaddedMos,
    /// `[nkpts, nmo, nmo]`, from [`crate::keris::KEris::build_fock`].
    pub fock: ZArr,
    pub mo_energy: Vec<Vec<f64>>,
    pub madelung: f64,
    pub e_hf: f64,
    pub converged: bool,
    pub nocc: usize,
    pub nvir: usize,
}

impl KsymRccsdInputs {
    /// Build from the **unfolded, full-BZ** result of a k-symmetric SCF.
    ///
    /// # Errors
    /// [`PbcCcError::Shape`] when `scf` is not a single restricted channel over
    /// the full BZ, plus the padding surface, the rotation-matrix build and
    /// the Fock build.
    pub fn build(
        scf: &pyscf_pbc_scf::KScfResult,
        with_df: &dyn PeriodicDf,
        kpts: &KPoints,
        eris_opts: crate::keris::KErisOpts,
    ) -> Result<Self, PbcCcError> {
        let cell = with_df.cell();
        let nkpts = kpts.nkpts();
        if scf.nset != 1 || scf.nkpts != nkpts || with_df.kpts().len() != nkpts {
            return Err(shape(format!(
                "KsymAdaptedRCCSD needs one restricted SCF channel over the FULL BZ \
                 ({nkpts} k-points); got nset = {}, nkpts = {}",
                scf.nset, scf.nkpts
            )));
        }
        let nao = cell.mol.nao_nr;
        let mf = pyscf_pbc_mp::spin_block(scf, 0).map_err(|e| shape(format!("spin_block: {e}")))?;
        let raw: Result<Vec<MoCoeff>, _> = mf
            .mo_coeff
            .iter()
            .zip(mf.mo_occ)
            .map(|(c, occ)| pyscf_pbc_mp::mo_coeff_from_kscf(c, nao, occ.len()))
            .collect();
        let raw = raw.map_err(|e| shape(format!("mo_coeff_from_kscf: {e}")))?;
        let frozen = pyscf_pbc_mp::FrozenK::default();
        let padded = pyscf_pbc_mp::add_padding(&raw, mf.mo_energy, mf.mo_occ, &frozen)
            .map_err(|e| shape(format!("add_padding: {e}")))?;
        let (nocc, nmo) = (padded.nocc, padded.nmo);
        let nvir = nmo - nocc;

        // `:401-403` — the rotation matrices are built from the PADDED MO
        // coefficients, not the raw ones, so their block sizes match `t1`/`t2`.
        let mo_rows: Vec<Vec<num_complex::Complex64>> = padded
            .mo_coeff
            .iter()
            .map(|m| {
                (0..m.nao)
                    .flat_map(|i| {
                        (0..m.nmo).map(move |p| {
                            num_complex::Complex64::new(
                                m.c.re[i * m.nmo + p],
                                m.c.im[i * m.nmo + p],
                            )
                        })
                    })
                    .collect()
            })
            .collect();
        let ovlp = pyscf_pbc_gto::get_ovlp(cell, &kpts.kpts)
            .map_err(|e| shape(format!("get_ovlp: {e}")))?;
        let ovlp_rows: Vec<Vec<num_complex::Complex64>> = ovlp
            .iter()
            .map(|s| {
                // `get_ovlp` returns COLUMN-MAJOR `nao x nao`; the transforms
                // want ROW-MAJOR. For a Hermitian overlap this is a conjugate,
                // not a no-op, so it is written out.
                (0..nao)
                    .flat_map(|i| {
                        (0..nao).map(move |j| {
                            num_complex::Complex64::new(s.re[j * nao + i], s.im[j * nao + i])
                        })
                    })
                    .collect()
            })
            .collect();
        let mut rmat = MORotationMatrix::new(nocc, nmo);
        rmat.build(kpts, cell, &mo_rows, &ovlp_rows, nao)
            .map_err(crate::ksymm_common::symm_err)?;

        let mut kqrts = KQuartets::build(kpts, cell).map_err(crate::ksymm_common::symm_err)?;
        // `loop_stabilizer` is used by six of the intermediates; upstream calls
        // `cache_stabilizer` lazily and this port cannot (it would need
        // `&mut self` behind an iterator), so it is built eagerly here.
        kqrts.cache_stabilizer(kpts);

        let dm = pyscf_pbc_scf::krdm::make_rdm1(mf.mo_coeff, mf.mo_occ, nao);
        let (fock, mo_energy, madelung) =
            crate::keris::KEris::build_fock(cell, with_df, &padded, &dm, eris_opts)?;

        Ok(Self {
            kqrts,
            rmat,
            kconserv: pyscf_pbc_lib::kpts_helper::get_kconserv(&cell.a, &kpts.kpts),
            padded,
            fock,
            mo_energy,
            madelung,
            e_hf: scf.e_tot,
            converged: scf.converged,
            nocc,
            nvir,
        })
    }

    /// The k-symmetry context the intermediates and the kernel index by.
    pub fn ctx<'a>(&'a self, kpts: &'a KPoints) -> KsymCtx<'a> {
        KsymCtx::new(
            kpts,
            &self.kqrts,
            &self.rmat,
            &self.kconserv,
            self.nocc,
            self.nvir,
        )
    }
}

/// The whole run: inputs, ERIs, amplitude iteration.
///
/// # Errors
/// [`PbcCcError::NotConverged`] for an unconverged mean field, plus everything
/// [`KsymRccsdInputs::build`], [`KsymEris::build`] and [`kernel`] raise.
pub fn run(
    inputs: &KsymRccsdInputs,
    with_df: &dyn PeriodicDf,
    kpts: &KPoints,
    opts: &KsymRccsdOpts,
) -> Result<(KsymRccsdResult, usize), PbcCcError> {
    if !inputs.converged {
        return Err(PbcCcError::NotConverged {
            what: "the reference k-symmetric SCF",
            detail: "KsymAdaptedRCCSD refuses an unconverged mean field".into(),
        });
    }
    let ctx = inputs.ctx(kpts);
    let eris = KsymEris::build(
        &ctx,
        with_df,
        &inputs.padded,
        inputs.fock.clone(),
        inputs.mo_energy.clone(),
        inputs.madelung,
    )?;
    let n_ao2mo = eris.n_ao2mo;
    let r = kernel(&ctx, &eris, &inputs.padded, opts)?;
    Ok((r, n_ao2mo))
}
