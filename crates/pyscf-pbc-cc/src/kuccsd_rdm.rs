//! `kuccsd_rdm` — the one-particle density matrix of `KUCCSD`
//! (plan 16-12; `pyscf/pbc/cc/kuccsd_rdm.py`, 157 l).
//!
//! # `l1`/`l2` default to `conj(t1)`/`conj(t2)`, and that is not the Lambda equations
//!
//! `:26-29` sets `l1 = [amp.conj() for amp in t1]` when none is supplied. That
//! is upstream's default and it is what this port reproduces; it is NOT a
//! solution of the Lambda equations, so the density matrix is the one PySCF
//! gives you for `mycc.make_rdm1()` without a preceding `solve_lambda`, and
//! nothing more. `pbc/cc` ships no k-point Lambda solver at all — there is no
//! `kccsd_lambda.py` — so there is nothing further to port here, and a caller
//! wanting response-quality densities has to bring its own `l1`/`l2`.
//!
//! # The occupied-virtual response block is absent, by construction
//!
//! `make_rdm1`'s own docstring (`:92-94`): "the occupied-virtual blocks due to
//! the orbital response contribution are not included". The `ov` block that IS
//! written is `dov + dvo^H`, i.e. the amplitude part only.
//!
//! # The frozen-core branch is DEAD CODE upstream
//!
//! `:136-152` reads `if with_frozen and mycc.frozen is not None: raise
//! NotImplementedError` and then, AFTER the raise, twelve lines that would have
//! done the reindexing. They cannot run. This port reproduces the refusal and
//! does not port the unreachable body — writing it would mean shipping
//! untestable code whose only specification is that upstream never executes it.

use pyscf_pbc_lib::Kconserv;

use crate::error::PbcCcError;
use crate::kintermediates_uhf::{UT1, UT2};
use crate::zarr::{ZArr, einsum, einsum_scaled};

/// `(doo, dov, dvo, dvv)`, each a `(alpha, beta)` pair.
pub struct Gamma1 {
    /// `[nkpts, nocc, nocc]` per spin.
    pub doo: (ZArr, ZArr),
    /// `[nkpts, nocc, nvir]` per spin — this is `l1` itself (`:86-87`).
    pub dov: (ZArr, ZArr),
    /// `[nkpts, nvir, nocc]` per spin.
    pub dvo: (ZArr, ZArr),
    /// `[nkpts, nvir, nvir]` per spin.
    pub dvv: (ZArr, ZArr),
}

/// `_gamma1_intermediates(cc, t1, t2, l1, l2)` — `:25-89`.
///
/// `l1`/`l2` default to the elementwise conjugates of `t1`/`t2` (`:26-29`).
///
/// # Errors
/// Propagates every shape check.
pub fn gamma1_intermediates(
    t1: &UT1,
    t2: &UT2,
    l1: Option<&UT1>,
    l2: Option<&UT2>,
    kconserv: &Kconserv,
) -> Result<Gamma1, PbcCcError> {
    let owned_l1;
    let l1 = match l1 {
        Some(l) => l,
        None => {
            owned_l1 = (t1.0.conj(), t1.1.conj());
            &owned_l1
        }
    };
    let owned_l2;
    let l2 = match l2 {
        Some(l) => l,
        None => {
            owned_l2 = (t2.0.conj(), t2.1.conj(), t2.2.conj());
            &owned_l2
        }
    };

    let nkpts = t1.0.shape()[0];
    let (oa, va) = (t1.0.shape()[1], t1.0.shape()[2]);
    let (ob, vb) = (t1.1.shape()[1], t1.1.shape()[2]);

    let mut dooa = ZArr::zeros(&[nkpts, oa, oa]);
    let mut doob = ZArr::zeros(&[nkpts, ob, ob]);
    let mut dvva = ZArr::zeros(&[nkpts, va, va]);
    let mut dvvb = ZArr::zeros(&[nkpts, vb, vb]);

    // `:39-44` — `doo`. The output k-index is `x`, which for `doob`'s `ab`
    // term is the SECOND axis of `l2ab`/`t2ab` (`:43`, `yxz`), not the first.
    for kx in 0..nkpts {
        let mut a = einsum(
            "ie,je->ij",
            &[&l1.0.slice_leading(&[kx])?, &t1.0.slice_leading(&[kx])?],
        )?;
        a.scale(-1.0);
        let mut b = einsum(
            "ie,je->ij",
            &[&l1.1.slice_leading(&[kx])?, &t1.1.slice_leading(&[kx])?],
        )?;
        b.scale(-1.0);
        for ky in 0..nkpts {
            for kz in 0..nkpts {
                a.sub_assign(&einsum(
                    "imef,jmef->ij",
                    &[
                        &l2.1.slice_leading(&[kx, ky, kz])?,
                        &t2.1.slice_leading(&[kx, ky, kz])?,
                    ],
                )?)?;
                a.sub_assign(&einsum_scaled(
                    "imef,jmef->ij",
                    &[
                        &l2.0.slice_leading(&[kx, ky, kz])?,
                        &t2.0.slice_leading(&[kx, ky, kz])?,
                    ],
                    0.5,
                )?)?;
                b.sub_assign(&einsum(
                    "mief,mjef->ij",
                    &[
                        &l2.1.slice_leading(&[ky, kx, kz])?,
                        &t2.1.slice_leading(&[ky, kx, kz])?,
                    ],
                )?)?;
                b.sub_assign(&einsum_scaled(
                    "imef,jmef->ij",
                    &[
                        &l2.2.slice_leading(&[kx, ky, kz])?,
                        &t2.2.slice_leading(&[kx, ky, kz])?,
                    ],
                    0.5,
                )?)?;
            }
        }
        dooa.set_leading(&[kx], &a)?;
        doob.set_leading(&[kx], &b)?;
    }

    // `:46-49` / `:55` — `dvv`. Here the output k-index is `z`, the THIRD.
    for kz in 0..nkpts {
        let mut a = einsum(
            "ma,mb->ab",
            &[&t1.0.slice_leading(&[kz])?, &l1.0.slice_leading(&[kz])?],
        )?;
        let mut b = einsum(
            "ma,mb->ab",
            &[&t1.1.slice_leading(&[kz])?, &l1.1.slice_leading(&[kz])?],
        )?;
        for kx in 0..nkpts {
            for ky in 0..nkpts {
                a.add_assign(&einsum(
                    "mnae,mnbe->ab",
                    &[
                        &t2.1.slice_leading(&[kx, ky, kz])?,
                        &l2.1.slice_leading(&[kx, ky, kz])?,
                    ],
                )?)?;
                a.add_assign(&einsum_scaled(
                    "mnae,mnbe->ab",
                    &[
                        &t2.0.slice_leading(&[kx, ky, kz])?,
                        &l2.0.slice_leading(&[kx, ky, kz])?,
                    ],
                    0.5,
                )?)?;
                b.add_assign(&einsum_scaled(
                    "mnae,mnbe->ab",
                    &[
                        &t2.2.slice_leading(&[kx, ky, kz])?,
                        &l2.2.slice_leading(&[kx, ky, kz])?,
                    ],
                    0.5,
                )?)?;
            }
        }
        dvva.set_leading(&[kz], &a)?;
        dvvb.set_leading(&[kz], &b)?;
    }
    // `:51-53` — the ONE term upstream already writes as an explicit k-loop,
    // because its output index is `kconserv[km,ke,kn]` and no einsum subscript
    // can say that.
    for km in 0..nkpts {
        for kn in 0..nkpts {
            for ke in 0..nkpts {
                let ka = kconserv.get(km, ke, kn) as usize;
                let v = einsum(
                    "mnea,mneb->ab",
                    &[
                        &t2.1.slice_leading(&[km, kn, ke])?,
                        &l2.1.slice_leading(&[km, kn, ke])?,
                    ],
                )?;
                add_at1(&mut dvvb, ka, &v, 1.0)?;
            }
        }
    }

    // `:57-61` / `:70-76` — the `xt1`/`xt2` half-intermediates.
    let mut xt1a = ZArr::zeros(&[nkpts, oa, oa]);
    let mut xt1b = ZArr::zeros(&[nkpts, ob, ob]);
    let mut xt2a = ZArr::zeros(&[nkpts, va, va]);
    let mut xt2b = ZArr::zeros(&[nkpts, vb, vb]);
    for kx in 0..nkpts {
        for ky in 0..nkpts {
            for kz in 0..nkpts {
                // `:57-58` output k = x
                add_at1(
                    &mut xt1a,
                    kx,
                    &einsum_scaled(
                        "mnef,inef->mi",
                        &[
                            &l2.0.slice_leading(&[kx, ky, kz])?,
                            &t2.0.slice_leading(&[kx, ky, kz])?,
                        ],
                        0.5,
                    )?,
                    1.0,
                )?;
                add_at1(
                    &mut xt1a,
                    kx,
                    &einsum(
                        "mnef,inef->mi",
                        &[
                            &l2.1.slice_leading(&[kx, ky, kz])?,
                            &t2.1.slice_leading(&[kx, ky, kz])?,
                        ],
                    )?,
                    1.0,
                )?;
                // `:59-60` output k = z
                add_at1(
                    &mut xt2a,
                    kz,
                    &einsum_scaled(
                        "mnaf,mnef->ae",
                        &[
                            &t2.0.slice_leading(&[kx, ky, kz])?,
                            &l2.0.slice_leading(&[kx, ky, kz])?,
                        ],
                        0.5,
                    )?,
                    1.0,
                )?;
                add_at1(
                    &mut xt2a,
                    kz,
                    &einsum(
                        "mnaf,mnef->ae",
                        &[
                            &t2.1.slice_leading(&[kx, ky, kz])?,
                            &l2.1.slice_leading(&[kx, ky, kz])?,
                        ],
                    )?,
                    1.0,
                )?;
                // `:70` output k = x
                add_at1(
                    &mut xt1b,
                    kx,
                    &einsum_scaled(
                        "mnef,inef->mi",
                        &[
                            &l2.2.slice_leading(&[kx, ky, kz])?,
                            &t2.2.slice_leading(&[kx, ky, kz])?,
                        ],
                        0.5,
                    )?,
                    1.0,
                )?;
                // `:71` output k = y, the SECOND axis
                add_at1(
                    &mut xt1b,
                    ky,
                    &einsum(
                        "nmef,nief->mi",
                        &[
                            &l2.1.slice_leading(&[kx, ky, kz])?,
                            &t2.1.slice_leading(&[kx, ky, kz])?,
                        ],
                    )?,
                    1.0,
                )?;
                // `:72` output k = z
                add_at1(
                    &mut xt2b,
                    kz,
                    &einsum_scaled(
                        "mnaf,mnef->ae",
                        &[
                            &t2.2.slice_leading(&[kx, ky, kz])?,
                            &l2.2.slice_leading(&[kx, ky, kz])?,
                        ],
                        0.5,
                    )?,
                    1.0,
                )?;
            }
        }
    }
    // `:73-75` — the second explicit k-loop, `ka = kconserv[km,kf,kn]`.
    for km in 0..nkpts {
        for kn in 0..nkpts {
            for kf in 0..nkpts {
                let ka = kconserv.get(km, kf, kn) as usize;
                let v = einsum(
                    "mnfa,mnfe->ae",
                    &[
                        &t2.1.slice_leading(&[km, kn, kf])?,
                        &l2.1.slice_leading(&[km, kn, kf])?,
                    ],
                )?;
                add_at1(&mut xt2b, ka, &v, 1.0)?;
            }
        }
    }
    for k in 0..nkpts {
        add_at1(
            &mut xt2a,
            k,
            &einsum(
                "ma,me->ae",
                &[&t1.0.slice_leading(&[k])?, &l1.0.slice_leading(&[k])?],
            )?,
            1.0,
        )?;
        add_at1(
            &mut xt2b,
            k,
            &einsum(
                "ma,me->ae",
                &[&t1.1.slice_leading(&[k])?, &l1.1.slice_leading(&[k])?],
            )?,
            1.0,
        )?;
    }

    // `:63-68` / `:78-83` — `dvo`.
    let mut dvoa = ZArr::zeros(&[nkpts, va, oa]);
    let mut dvob = ZArr::zeros(&[nkpts, vb, ob]);
    for ka in 0..nkpts {
        dvoa.set_leading(&[ka], &t1.0.slice_leading(&[ka])?.transpose(&[1, 0])?)?;
        dvob.set_leading(&[ka], &t1.1.slice_leading(&[ka])?.transpose(&[1, 0])?)?;
    }
    for ka in 0..nkpts {
        for km in 0..nkpts {
            add_at1(
                &mut dvoa,
                ka,
                &einsum(
                    "imae,me->ai",
                    &[
                        &t2.0.slice_leading(&[ka, km, ka])?,
                        &l1.0.slice_leading(&[km])?,
                    ],
                )?,
                1.0,
            )?;
            add_at1(
                &mut dvoa,
                ka,
                &einsum(
                    "imae,me->ai",
                    &[
                        &t2.1.slice_leading(&[ka, km, ka])?,
                        &l1.1.slice_leading(&[km])?,
                    ],
                )?,
                1.0,
            )?;
            add_at1(
                &mut dvob,
                ka,
                &einsum(
                    "imae,me->ai",
                    &[
                        &t2.2.slice_leading(&[ka, km, ka])?,
                        &l1.1.slice_leading(&[km])?,
                    ],
                )?,
                1.0,
            )?;
            add_at1(
                &mut dvob,
                ka,
                &einsum(
                    "miea,me->ai",
                    &[
                        &t2.1.slice_leading(&[km, ka, km])?,
                        &l1.0.slice_leading(&[km])?,
                    ],
                )?,
                1.0,
            )?;
        }
    }
    for k in 0..nkpts {
        add_at1(
            &mut dvoa,
            k,
            &einsum(
                "mi,ma->ai",
                &[&xt1a.slice_leading(&[k])?, &t1.0.slice_leading(&[k])?],
            )?,
            -1.0,
        )?;
        add_at1(
            &mut dvoa,
            k,
            &einsum(
                "ie,ae->ai",
                &[&t1.0.slice_leading(&[k])?, &xt2a.slice_leading(&[k])?],
            )?,
            -1.0,
        )?;
        add_at1(
            &mut dvob,
            k,
            &einsum(
                "mi,ma->ai",
                &[&xt1b.slice_leading(&[k])?, &t1.1.slice_leading(&[k])?],
            )?,
            -1.0,
        )?;
        add_at1(
            &mut dvob,
            k,
            &einsum(
                "ie,ae->ai",
                &[&t1.1.slice_leading(&[k])?, &xt2b.slice_leading(&[k])?],
            )?,
            -1.0,
        )?;
    }

    Ok(Gamma1 {
        doo: (dooa, doob),
        // `:85-86` — `dov` IS `l1`, not a copy of anything derived.
        dov: (l1.0.clone(), l1.1.clone()),
        dvo: (dvoa, dvob),
        dvv: (dvva, dvvb),
    })
}

/// `make_rdm1(mycc, t1, t2, l1, l2)` — `:91-103`, then `_make_rdm1` (`:105-157`).
///
/// Returns `(dm1a, dm1b)`, each `[nkpts, nmo, nmo]` in the MO basis. Pass
/// `frozen = false` to reproduce upstream's `with_frozen=False`; `frozen = true`
/// with any frozen orbital REFUSES, as upstream does.
///
/// # Errors
/// [`PbcCcError::NotImplementedUpstream`] for the frozen-core branch;
/// otherwise propagates every shape check.
pub fn make_rdm1(
    t1: &UT1,
    t2: &UT2,
    l1: Option<&UT1>,
    l2: Option<&UT2>,
    kconserv: &Kconserv,
    has_frozen: bool,
) -> Result<(ZArr, ZArr), PbcCcError> {
    let d1 = gamma1_intermediates(t1, t2, l1, l2, kconserv)?;
    make_rdm1_from_gamma1(&d1, has_frozen)
}

/// `_make_rdm1(mycc, d1, with_frozen, ao_repr=False)` — `:105-157`.
///
/// # Errors
/// As [`make_rdm1`].
pub fn make_rdm1_from_gamma1(d1: &Gamma1, has_frozen: bool) -> Result<(ZArr, ZArr), PbcCcError> {
    if has_frozen {
        // `:136-137`. The twelve lines after the `raise` are unreachable and
        // are not ported — see the module doc.
        return Err(PbcCcError::NotImplementedUpstream {
            upstream: "pbc/cc/kuccsd_rdm.py:137",
            what: "_make_rdm1 raises NotImplementedError when with_frozen and cc.frozen",
        });
    }
    let a = one_spin(&d1.doo.0, &d1.dov.0, &d1.dvo.0, &d1.dvv.0)?;
    let b = one_spin(&d1.doo.1, &d1.dov.1, &d1.dvo.1, &d1.dvv.1)?;
    Ok((a, b))
}

/// `:119-133` for one spin — assemble, symmetrise, halve, and put the
/// reference occupation back on the diagonal.
fn one_spin(doo: &ZArr, dov: &ZArr, dvo: &ZArr, dvv: &ZArr) -> Result<ZArr, PbcCcError> {
    let nkpts = doo.shape()[0];
    let nocc = doo.shape()[1];
    let nvir = dvv.shape()[1];
    let nmo = nocc + nvir;
    let mut dm = ZArr::zeros(&[nkpts, nmo, nmo]);
    for k in 0..nkpts {
        let oo = herm_sum(&doo.slice_leading(&[k])?)?;
        let vv = herm_sum(&dvv.slice_leading(&[k])?)?;
        // `:122` — `dov + dvo^H`, NOT symmetrised against itself.
        let mut ov = dov.slice_leading(&[k])?;
        ov.add_assign(&dvo.slice_leading(&[k])?.conj().transpose(&[1, 0])?)?;
        // `:123` — the `vo` block is the conjugate transpose of the `ov` one,
        // so the whole matrix is Hermitian by construction.
        let vo = ov.conj().transpose(&[1, 0])?;

        let mut blk = ZArr::zeros(&[nmo, nmo]);
        put(&mut blk, 0, 0, &oo)?;
        put(&mut blk, 0, nocc, &ov)?;
        put(&mut blk, nocc, 0, &vo)?;
        put(&mut blk, nocc, nocc, &vv)?;
        blk.scale(0.5);
        // `:126-127` — the reference determinant, added AFTER the halving.
        for i in 0..nocc {
            blk.data_mut().re[i * nmo + i] += 1.0;
        }
        dm.set_leading(&[k], &blk)?;
    }
    Ok(dm)
}

/// `x + x.conj().transpose(0,2,1)` for one k-block.
fn herm_sum(x: &ZArr) -> Result<ZArr, PbcCcError> {
    let mut out = x.clone();
    out.add_assign(&x.conj().transpose(&[1, 0])?)?;
    Ok(out)
}

/// Write `src` into `dst` at `(r0, c0)`.
fn put(dst: &mut ZArr, r0: usize, c0: usize, src: &ZArr) -> Result<(), PbcCcError> {
    let n = dst.shape()[1];
    let (nr, nc) = (src.shape()[0], src.shape()[1]);
    for r in 0..nr {
        for c in 0..nc {
            let (re, im) = src.at(&[r, c])?;
            dst.data_mut().re[(r0 + r) * n + c0 + c] = re;
            dst.data_mut().im[(r0 + r) * n + c0 + c] = im;
        }
    }
    Ok(())
}

fn add_at1(t: &mut ZArr, k: usize, v: &ZArr, s: f64) -> Result<(), PbcCcError> {
    let mut cur = t.slice_leading(&[k])?;
    cur.zip_assign(v, s)?;
    t.set_leading(&[k], &cur)
}
