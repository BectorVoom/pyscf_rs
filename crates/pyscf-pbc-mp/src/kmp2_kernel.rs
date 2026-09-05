//! Deterministic two-pass KMP2 energy kernel.

use pyscf_algebra::{CTensor, oracle_sum, oracle_zdotu, oracle_zdotu_re};
use pyscf_pbc_df::CoulGCache;
use rayon::prelude::*;
use std::collections::HashMap;
use std::sync::Arc;

use crate::{
    Kmp2, LovTable, PaddedMos, PaddingIdx, PaddingKind, PbcMpError, T2, build_lov, mo_slice,
    padding_k_idx,
};

pub const LARGE_DENOM: f64 = 1.0e14;

pub(crate) fn df_oovv(
    lov: &LovTable,
    ki: usize,
    ka: usize,
    kj: usize,
    kb: usize,
) -> Result<CTensor, PbcMpError> {
    let nocc = lov.nocc;
    let nvir = lov.nvir;
    let (naux, x) = lov.block(ki, ka);
    let (naux_y, y) = lov.block(kj, kb);
    if naux != naux_y {
        return Err(PbcMpError::Shape {
            what: "Lov auxiliary ranks differ in a contraction".into(),
        });
    }
    let mut out = CTensor::zeros(nocc * nocc * nvir * nvir);
    for i in 0..nocc {
        for j in 0..nocc {
            for a in 0..nvir {
                let ix = (i * nvir + a) * *naux;
                let xs = CTensor::from_planes(
                    x.re[ix..ix + *naux].to_vec(),
                    x.im[ix..ix + *naux].to_vec(),
                );
                for b in 0..nvir {
                    let iy = (j * nvir + b) * *naux;
                    let ys = CTensor::from_planes(
                        y.re[iy..iy + *naux].to_vec(),
                        y.im[iy..iy + *naux].to_vec(),
                    );
                    // No conjugation: upstream is einsum("Lia,Ljb->iajb").
                    let (re, im) = oracle_zdotu(&xs, &ys);
                    let p = ((i * nocc + j) * nvir + a) * nvir + b;
                    out.re[p] = re / lov.nkpts as f64;
                    out.im[p] = im / lov.nkpts as f64;
                }
            }
        }
    }
    Ok(out)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn ao2mo_oovv(
    df: &dyn pyscf_pbc_df::PeriodicDf,
    mos: &[pyscf_pbc_df::MoCoeff],
    ki: usize,
    ka: usize,
    kj: usize,
    kb: usize,
    nocc: usize,
    nmo: usize,
    nkpts: usize,
    cache: Option<&CoulGCache>,
) -> Result<CTensor, PbcMpError> {
    let nvir = nmo - nocc;
    let oi = mo_slice(&mos[ki], 0, nocc)?;
    let va = mo_slice(&mos[ka], nocc, nmo)?;
    let oj = mo_slice(&mos[kj], 0, nocc)?;
    let vb = mo_slice(&mos[kb], nocc, nmo)?;
    let eri = df
        .ao2mo_cached([&oi, &va, &oj, &vb], [ki, ka, kj, kb], false, cache)?
        .restore_s1();
    let mut out = CTensor::zeros(nocc * nocc * nvir * nvir);
    for i in 0..nocc {
        for j in 0..nocc {
            for a in 0..nvir {
                for b in 0..nvir {
                    let src = (i * nvir + a) * (nocc * nvir) + j * nvir + b;
                    let dst = ((i * nocc + j) * nvir + a) * nvir + b;
                    out.re[dst] = eri.data.re[src] / nkpts as f64;
                    out.im[dst] = eri.data.im[src] / nkpts as f64;
                }
            }
        }
    }
    Ok(out)
}

pub fn kmp2_kernel(
    mp: &Kmp2<'_>,
    padded: &PaddedMos,
) -> Result<(f64, f64, Option<T2>), PbcMpError> {
    let nk = mp.kpts.len();
    let (nocc, nmo) = (padded.nocc, padded.nmo);
    if nocc == 0 || nocc >= nmo {
        return Err(PbcMpError::Shape {
            what: "KMP2 needs non-empty occupied and virtual spaces".into(),
        });
    }
    let nvir = nmo - nocc;
    let nov2 = (nocc * nvir).saturating_pow(2);
    let naux = if mp.with_df_ints {
        mp.with_df.get_naoaux()?
    } else {
        0
    };
    let t2_mb = if mp.with_t2 { nk.pow(3) * nov2 * 16 } else { 0 } as f64 / 1.0e6;
    let lov_mb = if mp.with_df_ints {
        nk.pow(2) * naux * nocc * nvir * 16
    } else {
        0
    } as f64
        / 1.0e6;
    let per_outer_mb = (nk * nov2 * 16) as f64 / 1.0e6;
    let fixed_mb = t2_mb + lov_mb;
    if fixed_mb + per_outer_mb > mp.max_memory {
        return Err(PbcMpError::Memory {
            required_mb: fixed_mb + per_outer_mb,
            available_mb: mp.max_memory,
        });
    }
    let live_outer = (((mp.max_memory - fixed_mb) / per_outer_mb.max(f64::MIN_POSITIVE)) as usize)
        .max(1)
        .min(rayon::current_num_threads())
        .min(nk * nk);
    let split = match padding_k_idx(
        &padded.nmo_per_kpt,
        &padded.nocc_per_kpt,
        PaddingKind::Split,
    )? {
        PaddingIdx::Split { occupied, virtuals } => (occupied, virtuals),
        PaddingIdx::Joint(_) => unreachable!(),
    };
    let lov = if mp.with_df_ints {
        Some(build_lov(mp.with_df, &padded.mo_coeff, nocc)?)
    } else {
        None
    };
    let caches = if lov.is_none() && matches!(mp.with_df.name(), "FFTDF" | "AFTDF") {
        let mut unique = HashMap::<[u64; 3], Arc<CoulGCache>>::new();
        let mut table = Vec::with_capacity(nk * nk);
        for ki in 0..nk {
            for ka in 0..nk {
                let q = [
                    mp.kpts[ka][0] - mp.kpts[ki][0],
                    mp.kpts[ka][1] - mp.kpts[ki][1],
                    mp.kpts[ka][2] - mp.kpts[ki][2],
                ];
                let key = q.map(f64::to_bits);
                let cache = match unique.get(&key) {
                    Some(v) => Arc::clone(v),
                    None => {
                        let v = Arc::new(CoulGCache::build(mp.cell, mp.with_df.mesh(), q)?);
                        unique.insert(key, Arc::clone(&v));
                        v
                    }
                };
                table.push(cache);
            }
        }
        Some(table)
    } else {
        None
    };
    let pairs: Vec<_> = (0..nk * nk).collect();
    let mut pair_results = Vec::with_capacity(pairs.len());
    for batch in pairs.chunks(live_outer) {
        let chunk: Result<Vec<_>, PbcMpError> = batch
            .par_iter()
            .map(|&flat| {
                let (ki, kj) = (flat / nk, flat % nk);
                // Upstream deliberately fills every ka block before the exchange pass,
                // because the latter reads oovv[kb], which may be a later block.
                let oovv: Result<Vec<_>, PbcMpError> = (0..nk)
                    .into_par_iter()
                    .map(|ka| {
                        let kb = mp.khelper.kconserv.get(ki, ka, kj) as usize;
                        match &lov {
                            Some(l) => df_oovv(l, ki, ka, kj, kb),
                            None => ao2mo_oovv(
                                mp.with_df,
                                &padded.mo_coeff,
                                ki,
                                ka,
                                kj,
                                kb,
                                nocc,
                                nmo,
                                nk,
                                caches.as_ref().map(|c| c[ki * nk + ka].as_ref()),
                            ),
                        }
                    })
                    .collect();
                let oovv = oovv?;
                let terms: Vec<_> = (0..nk)
                    .into_par_iter()
                    .map(|ka| {
                        let kb = mp.khelper.kconserv.get(ki, ka, kj) as usize;
                        let mut amp = CTensor::zeros(nov2);
                        let mut exch = CTensor::zeros(nov2);
                        for i in 0..nocc {
                            for j in 0..nocc {
                                for a in 0..nvir {
                                    for b in 0..nvir {
                                        let p = ((i * nocc + j) * nvir + a) * nvir + b;
                                        let eia = if split.0[ki].contains(&i)
                                            && split.1[ka].contains(&a)
                                        {
                                            padded.mo_energy[ki][i] - padded.mo_energy[ka][nocc + a]
                                        } else {
                                            LARGE_DENOM
                                        };
                                        let ejb = if split.0[kj].contains(&j)
                                            && split.1[kb].contains(&b)
                                        {
                                            padded.mo_energy[kj][j] - padded.mo_energy[kb][nocc + b]
                                        } else {
                                            LARGE_DENOM
                                        };
                                        let denom = eia + ejb;
                                        amp.re[p] = oovv[ka].re[p] / denom;
                                        amp.im[p] = -oovv[ka].im[p] / denom;
                                        let q = ((i * nocc + j) * nvir + b) * nvir + a;
                                        exch.re[p] = oovv[kb].re[q];
                                        exch.im[p] = oovv[kb].im[q];
                                    }
                                }
                            }
                        }
                        let edi = 2.0 * oracle_zdotu_re(&amp, &oovv[ka]);
                        let exi = -oracle_zdotu_re(&amp, &exch);
                        (edi * 0.5 + exi, edi * 0.5, amp)
                    })
                    .collect();
                let ss = oracle_sum(&terms.iter().map(|x| x.0).collect::<Vec<_>>());
                let os = oracle_sum(&terms.iter().map(|x| x.1).collect::<Vec<_>>());
                Ok((ss, os, terms.into_iter().map(|x| x.2).collect::<Vec<_>>()))
            })
            .collect();
        pair_results.extend(chunk?);
    }
    let ss = oracle_sum(&pair_results.iter().map(|x| x.0).collect::<Vec<_>>()) / nk as f64;
    let os = oracle_sum(&pair_results.iter().map(|x| x.1).collect::<Vec<_>>()) / nk as f64;
    let t2 = mp.with_t2.then(|| T2 {
        nkpts: nk,
        nocc,
        nvir,
        blocks: pair_results.into_iter().flat_map(|x| x.2).collect(),
    });
    Ok((ss, os, t2))
}
