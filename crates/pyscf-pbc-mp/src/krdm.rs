//! KMP2 one- and two-particle density matrices (`kmp2.py:567-690`).

use pyscf_algebra::{CTensor, oracle_sum};
use pyscf_pbc_lib::Kconserv;

use crate::{PaddingIdx, PaddingKind, PbcMpError, T2, padding_k_idx};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RdmKind {
    Compact,
    Padded,
}

#[derive(Debug, Clone)]
pub enum Rdm2 {
    Padded { nmo: usize, data: CTensor },
    Compact(Vec<CTensor>),
}

fn t_idx(no: usize, nv: usize, i: usize, j: usize, a: usize, b: usize) -> usize {
    ((i * no + j) * nv + a) * nv + b
}

fn prod_conj(x: &CTensor, xi: usize, y: &CTensor, yi: usize) -> (f64, f64) {
    (
        x.re[xi] * y.re[yi] + x.im[xi] * y.im[yi],
        x.re[xi] * y.im[yi] - x.im[xi] * y.re[yi],
    )
}

/// `_gamma1_intermediates`; outputs are row-major `(nocc,nocc)` and
/// `(nvir,nvir)` blocks for every k-point.
pub fn gamma1_intermediates(t2: &T2, kconserv: &Kconserv) -> (Vec<CTensor>, Vec<CTensor>) {
    let (nk, no, nv) = (t2.nkpts, t2.nocc, t2.nvir);
    let mut doo = vec![CTensor::zeros(no * no); nk];
    let mut dvv = vec![CTensor::zeros(nv * nv); nk];

    for k in 0..nk {
        for x in 0..nv {
            for y in 0..nv {
                let mut re = Vec::new();
                let mut im = Vec::new();
                for ki in 0..nk {
                    for kj in 0..nk {
                        for ka in 0..nk {
                            let kb = kconserv.get(ki, ka, kj) as usize;
                            if kb != k {
                                continue;
                            }
                            let a = t2.block(ki, kj, ka);
                            let b = t2.block(ki, kj, kb);
                            for i in 0..no {
                                for j in 0..no {
                                    for c in 0..nv {
                                        let lhs = t_idx(no, nv, i, j, c, x);
                                        let (dr, di) =
                                            prod_conj(a, lhs, a, t_idx(no, nv, i, j, c, y));
                                        let (er, ei) =
                                            prod_conj(a, lhs, b, t_idx(no, nv, i, j, y, c));
                                        re.push(2.0 * dr - er);
                                        im.push(2.0 * di - ei);
                                    }
                                }
                            }
                        }
                    }
                }
                dvv[k].re[y * nv + x] = oracle_sum(&re);
                dvv[k].im[y * nv + x] = oracle_sum(&im);
            }
        }
        for x in 0..no {
            for y in 0..no {
                let mut re = Vec::new();
                let mut im = Vec::new();
                for ki in 0..nk {
                    for kj in 0..nk {
                        for ka in 0..nk {
                            if kj != k {
                                continue;
                            }
                            let kb = kconserv.get(ki, ka, kj) as usize;
                            let a = t2.block(ki, kj, ka);
                            let b = t2.block(ki, kj, kb);
                            for i in 0..no {
                                for av in 0..nv {
                                    for bv in 0..nv {
                                        let lhs = t_idx(no, nv, i, x, av, bv);
                                        let (dr, di) =
                                            prod_conj(a, lhs, a, t_idx(no, nv, i, y, av, bv));
                                        let (er, ei) =
                                            prod_conj(a, lhs, b, t_idx(no, nv, i, y, bv, av));
                                        re.push(2.0 * dr - er);
                                        im.push(2.0 * di - ei);
                                    }
                                }
                            }
                        }
                    }
                }
                doo[k].re[x * no + y] = -oracle_sum(&re);
                doo[k].im[x * no + y] = -oracle_sum(&im);
            }
        }
    }
    (doo, dvv)
}

pub fn make_rdm1(
    t2: &T2,
    kconserv: &Kconserv,
    nmo_per_kpt: &[usize],
    nocc_per_kpt: &[usize],
    kind: RdmKind,
) -> Result<Vec<CTensor>, PbcMpError> {
    let (doo, dvv) = gamma1_intermediates(t2, kconserv);
    let nmo = t2.nocc + t2.nvir;
    let joint = match padding_k_idx(nmo_per_kpt, nocc_per_kpt, PaddingKind::Joint)? {
        PaddingIdx::Joint(v) => v,
        PaddingIdx::Split { .. } => unreachable!(),
    };
    let mut out = Vec::with_capacity(t2.nkpts);
    for k in 0..t2.nkpts {
        let mut d = CTensor::zeros(nmo * nmo);
        for p in 0..t2.nocc {
            for q in 0..t2.nocc {
                let z = p * t2.nocc + q;
                d.re[p * nmo + q] = doo[k].re[z] + usize::from(p == q) as f64;
                d.im[p * nmo + q] = doo[k].im[z];
            }
        }
        for p in 0..t2.nvir {
            for q in 0..t2.nvir {
                let z = p * t2.nvir + q;
                d.re[(t2.nocc + p) * nmo + t2.nocc + q] = dvv[k].re[z];
                d.im[(t2.nocc + p) * nmo + t2.nocc + q] = dvv[k].im[z];
            }
        }
        let before = d.clone();
        for p in 0..nmo {
            for q in 0..nmo {
                let z = p * nmo + q;
                let zt = q * nmo + p;
                d.re[z] += before.re[zt];
                d.im[z] -= before.im[zt];
            }
        }
        if kind == RdmKind::Padded {
            out.push(d);
        } else {
            let idx = &joint[k];
            let mut c = CTensor::zeros(idx.len() * idx.len());
            for (p, &ip) in idx.iter().enumerate() {
                for (q, &iq) in idx.iter().enumerate() {
                    c.re[p * idx.len() + q] = d.re[ip * nmo + iq];
                    c.im[p * idx.len() + q] = d.im[ip * nmo + iq];
                }
            }
            out.push(c);
        }
    }
    Ok(out)
}

fn add_scaled(dst: &mut CTensor, di: usize, src: &CTensor, si: usize, scale: f64, conj: bool) {
    dst.re[di] += scale * src.re[si];
    dst.im[di] += scale * if conj { -src.im[si] } else { src.im[si] };
}

pub fn make_rdm2(
    t2: &T2,
    kconserv: &Kconserv,
    nmo_per_kpt: &[usize],
    nocc_per_kpt: &[usize],
    kind: RdmKind,
) -> Result<Rdm2, PbcMpError> {
    let (nk, no, nv) = (t2.nkpts, t2.nocc, t2.nvir);
    let nm = no + nv;
    let mut dm2 = CTensor::zeros(nk.pow(3) * nm.pow(4));
    let idx7 = |kp: usize, kq: usize, kr: usize, p: usize, q: usize, r: usize, s: usize| {
        ((((((kp * nk + kq) * nk + kr) * nm + p) * nm + q) * nm + r) * nm) + s
    };
    for ki in 0..nk {
        for kj in 0..nk {
            for ka in 0..nk {
                let kb = kconserv.get(ki, ka, kj) as usize;
                let a = t2.block(ki, kj, ka);
                let b = t2.block(kj, ki, ka);
                for i in 0..no {
                    for j in 0..no {
                        for av in 0..nv {
                            for bv in 0..nv {
                                let ai = t_idx(no, nv, i, j, av, bv);
                                let bi = t_idx(no, nv, j, i, av, bv);
                                let vr = 4.0 * a.re[ai] - 2.0 * b.re[bi];
                                let vi = 4.0 * a.im[ai] - 2.0 * b.im[bi];
                                let d1 = idx7(ki, ka, kj, i, no + av, j, no + bv);
                                dm2.re[d1] = vr;
                                dm2.im[d1] = vi;
                                let d2 = idx7(ka, ki, kb, no + av, i, no + bv, j);
                                dm2.re[d2] = vr;
                                dm2.im[d2] = -vi;
                            }
                        }
                    }
                }
            }
        }
    }
    let mut dm1 = make_rdm1(t2, kconserv, nmo_per_kpt, nocc_per_kpt, RdmKind::Padded)?;
    let occupied = match padding_k_idx(nmo_per_kpt, nocc_per_kpt, PaddingKind::Split)? {
        PaddingIdx::Split { occupied, .. } => occupied,
        PaddingIdx::Joint(_) => unreachable!(),
    };
    for ki in 0..nk {
        for &i in &occupied[ki] {
            dm1[ki].re[i * nm + i] -= 2.0;
        }
    }
    for (ki, occupied_ki) in occupied.iter().enumerate().take(nk) {
        for (kp, dm1_kp) in dm1.iter().enumerate().take(nk) {
            for &i in occupied_ki {
                for p in 0..nm {
                    for q in 0..nm {
                        let si = q * nm + p;
                        add_scaled(
                            &mut dm2,
                            idx7(ki, ki, kp, i, i, p, q),
                            dm1_kp,
                            si,
                            2.0,
                            false,
                        );
                        add_scaled(
                            &mut dm2,
                            idx7(kp, kp, ki, p, q, i, i),
                            dm1_kp,
                            si,
                            2.0,
                            false,
                        );
                        add_scaled(
                            &mut dm2,
                            idx7(kp, ki, ki, p, i, i, q),
                            dm1_kp,
                            si,
                            -1.0,
                            false,
                        );
                        add_scaled(
                            &mut dm2,
                            idx7(ki, kp, kp, i, p, q, i),
                            dm1_kp,
                            p * nm + q,
                            -1.0,
                            false,
                        );
                    }
                }
            }
        }
    }
    for ki in 0..nk {
        for kj in 0..nk {
            for &i in &occupied[ki] {
                for &j in &occupied[kj] {
                    dm2.re[idx7(ki, ki, kj, i, i, j, j)] += 4.0;
                    dm2.re[idx7(ki, kj, kj, i, j, j, i)] -= 2.0;
                }
            }
        }
    }
    if kind == RdmKind::Padded {
        return Ok(Rdm2::Padded { nmo: nm, data: dm2 });
    }
    let joint = match padding_k_idx(nmo_per_kpt, nocc_per_kpt, PaddingKind::Joint)? {
        PaddingIdx::Joint(v) => v,
        PaddingIdx::Split { .. } => unreachable!(),
    };
    let mut blocks = Vec::with_capacity(nk.pow(3));
    for kp in 0..nk {
        for kq in 0..nk {
            for kr in 0..nk {
                let ks = kconserv.get(kp, kq, kr) as usize;
                let shape = [
                    joint[kp].len(),
                    joint[kq].len(),
                    joint[kr].len(),
                    joint[ks].len(),
                ];
                let mut block = CTensor::zeros(shape.iter().product());
                let mut z = 0;
                for &p in &joint[kp] {
                    for &q in &joint[kq] {
                        for &r in &joint[kr] {
                            for &s in &joint[ks] {
                                let src = idx7(kp, kq, kr, p, q, r, s);
                                block.re[z] = dm2.re[src];
                                block.im[z] = dm2.im[src];
                                z += 1;
                            }
                        }
                    }
                }
                blocks.push(block);
            }
        }
    }
    Ok(Rdm2::Compact(blocks))
}
