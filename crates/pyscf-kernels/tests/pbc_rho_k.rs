//! SCF-01 — `rho_k`, the device density at one k-point, against a serial
//! host reference in the host route's (`eval_rho_one`) operation order:
//! bitwise on the CPU runtime.
#![cfg(feature = "cpu")]

use cubecl::Runtime;
use pyscf_algebra::AlgebraClient;
use pyscf_kernels::pbc::rho_k;

fn lcg(seed: u64, len: usize) -> Vec<f64> {
    let mut x = seed;
    (0..len)
        .map(|_| {
            x = x
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((x >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
        })
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn reference(
    ar: &[f64],
    ai: &[f64],
    dr: &[f64],
    di: &[f64],
    ncomp: usize,
    nao: usize,
    ngrids: usize,
) -> (Vec<f64>, Vec<f64>) {
    let mut c0r = vec![0.0; nao * ngrids];
    let mut c0i = vec![0.0; nao * ngrids];
    for j in 0..nao {
        for i in 0..nao {
            let (r, m) = (dr[i * nao + j], di[i * nao + j]);
            if r == 0.0 && m == 0.0 {
                continue;
            }
            for g in 0..ngrids {
                let (a, b) = (ar[i * ngrids + g], ai[i * ngrids + g]);
                c0r[j * ngrids + g] += a * r - b * m;
                c0i[j * ngrids + g] += a * m + b * r;
            }
        }
    }
    let mut or = vec![0.0; ncomp * ngrids];
    let mut oi = vec![0.0; ncomp * ngrids];
    for c in 0..ncomp {
        for g in 0..ngrids {
            let (mut sr, mut si) = (0.0_f64, 0.0_f64);
            for j in 0..nao {
                let e = c * nao * ngrids + j * ngrids + g;
                let (a, b) = (ar[e], -ai[e]);
                let (p, q) = (c0r[j * ngrids + g], c0i[j * ngrids + g]);
                sr += a * p - b * q;
                si += a * q + b * p;
            }
            or[c * ngrids + g] = sr;
            oi[c * ngrids + g] = si;
        }
    }
    (or, oi)
}

#[test]
fn rho_k_matches_the_host_order_bitwise() {
    let client = AlgebraClient::Cpu(cubecl_cpu::CpuRuntime::client(&cubecl_cpu::CpuDevice));
    for ncomp in [1usize, 4] {
        let (nao, ngrids) = (7, 211);
        let ar = lcg(1, ncomp * nao * ngrids);
        let ai = lcg(2, ncomp * nao * ngrids);
        let mut dr = lcg(3, nao * nao);
        let di = lcg(4, nao * nao);
        dr[5] = 0.0; // a zero entry the host skips only when both parts are zero
        let (gr, gi) = rho_k(&client, &ar, &ai, &dr, &di, ncomp, nao, ngrids).expect("rho_k");
        let (wr, wi) = reference(&ar, &ai, &dr, &di, ncomp, nao, ngrids);
        for i in 0..gr.len() {
            assert_eq!(
                gr[i].to_bits(),
                wr[i].to_bits(),
                "ncomp {ncomp} re {i}: {} vs {}",
                gr[i],
                wr[i]
            );
            assert_eq!(
                gi[i].to_bits(),
                wi[i].to_bits(),
                "ncomp {ncomp} im {i}: {} vs {}",
                gi[i],
                wi[i]
            );
        }
    }
}
