//! `smearing::grad_tril` and `kocc::get_grad` hoist `(F C)[:, j]` out of the
//! per-element loop. That is a pure re-association of WHEN each inner sum is
//! computed, not of its operations, so the gradients must be BIT-identical to
//! the original O(nmo^2 nao^2) formulas reproduced here verbatim.

use pyscf_algebra::CTensor;
use pyscf_pbc_scf::{kocc, smearing};

fn lcg(seed: &mut u64) -> f64 {
    *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    ((*seed >> 11) as f64 / (1u64 << 53) as f64) - 0.5
}

fn random(n: usize, seed: &mut u64) -> CTensor {
    CTensor { re: (0..n).map(|_| lcg(seed)).collect(), im: (0..n).map(|_| lcg(seed)).collect() }
}

/// The pre-hoist `grad_tril`, verbatim.
fn grad_tril_naive(mo_coeff: &CTensor, fock: &CTensor, nao: usize, nmo: usize) -> Vec<f64> {
    let mut out = Vec::new();
    for i in 1..nmo {
        for j in 0..i {
            let mut re = 0.0_f64;
            let mut im = 0.0_f64;
            for mu in 0..nao {
                let mut fr = 0.0_f64;
                let mut fi = 0.0_f64;
                for nu in 0..nao {
                    let (x, y) = (fock.re[mu * nao + nu], fock.im[mu * nao + nu]);
                    let (u, v) = (mo_coeff.re[nu + j * nao], mo_coeff.im[nu + j * nao]);
                    fr += x * u - y * v;
                    fi += x * v + y * u;
                }
                let (cr, ci) = (mo_coeff.re[mu + i * nao], -mo_coeff.im[mu + i * nao]);
                re += cr * fr - ci * fi;
                im += cr * fi + ci * fr;
            }
            out.push(re);
            out.push(im);
        }
    }
    out
}

/// The pre-hoist `kocc::get_grad`, verbatim.
fn get_grad_naive(mo_coeff: &CTensor, mo_occ: &[f64], fock: &CTensor, nao: usize) -> Vec<f64> {
    let nmo = mo_occ.len();
    let occ: Vec<usize> = (0..nmo).filter(|i| mo_occ[*i] > 0.0).collect();
    let vir: Vec<usize> = (0..nmo).filter(|i| mo_occ[*i] <= 0.0).collect();
    let mut out = Vec::new();
    for &a in &vir {
        for &i in &occ {
            let mut re = 0.0_f64;
            let mut im = 0.0_f64;
            for mu in 0..nao {
                let mut fr = 0.0_f64;
                let mut fi = 0.0_f64;
                for nu in 0..nao {
                    let (x, y) = (fock.re[mu * nao + nu], fock.im[mu * nao + nu]);
                    let (u, v) = (mo_coeff.re[nu + i * nao], mo_coeff.im[nu + i * nao]);
                    fr += x * u - y * v;
                    fi += x * v + y * u;
                }
                let (cr, ci) = (mo_coeff.re[mu + a * nao], -mo_coeff.im[mu + a * nao]);
                re += cr * fr - ci * fi;
                im += cr * fi + ci * fr;
            }
            out.push(re);
            out.push(im);
        }
    }
    out
}

fn bits(v: &[f64]) -> Vec<u64> {
    v.iter().map(|x| x.to_bits()).collect()
}

#[test]
fn hoisted_gradients_are_bit_identical() {
    let mut seed = 7_u64;
    for nao in [1, 5, 23] {
        let c = random(nao * nao, &mut seed);
        let f = random(nao * nao, &mut seed);
        assert_eq!(bits(&smearing::grad_tril(&c, &f, nao, nao)), bits(&grad_tril_naive(&c, &f, nao, nao)));
        // Interleaved occupations (a smeared-like pattern included).
        let occ: Vec<f64> = (0..nao).map(|i| if i % 3 == 1 { 0.0 } else { 2.0 - 0.1 * i as f64 }).collect();
        assert_eq!(bits(&kocc::get_grad(&c, &occ, &f, nao)), bits(&get_grad_naive(&c, &occ, &f, nao)));
    }
}
