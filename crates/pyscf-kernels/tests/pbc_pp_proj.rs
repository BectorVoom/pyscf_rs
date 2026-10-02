//! K-PP2 (projector block `S[row, g]`): device block against the host formula,
//! and block-offset selection.

#![cfg(feature = "cpu")]

use cubecl::Runtime;
use pyscf_algebra::AlgebraClient;
use pyscf_kernels::{PpProjTables, pp_proj_block};

struct Lcg(u64);

impl Lcg {
    fn new(seed: u64) -> Self {
        Self(seed)
    }
    fn next_f64(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let u = (self.0 >> 11) as f64 / (1u64 << 53) as f64;
        u * 6.0 - 3.0
    }
}

fn cpu_client() -> AlgebraClient {
    AlgebraClient::Cpu(cubecl_cpu::CpuRuntime::client(&cubecl_cpu::CpuDevice))
}

/// Three hand-built rows (no chemistry needed).
fn small_tables() -> PpProjTables {
    PpProjTables {
        // atom positions, one (x, y, z) per row
        row_r: vec![0.1, 0.2, 0.3, 1.0, -0.5, 0.25, -0.4, 0.6, 0.9],
        // (alpha, coef, c0, c1, c2) per row
        row_par: vec![
            0.4, 1.3, 2.0, 0.0, 0.0, //
            0.7, 0.9, 5.0, -1.0, 0.0, //
            0.3, 1.1, 63.0, -18.0, 1.0,
        ],
        row_term0: vec![0, 1, 2],
        row_nterm: vec![1, 1, 2],
        // term coefficients
        term_c: vec![1.0, 0.5, 0.7, -0.3],
        // (ix, iy, iz) per term
        term_pow: vec![0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 2],
    }
}

fn pseudo_gvectors(n: usize, seed: &mut Lcg) -> Vec<f64> {
    (0..3 * n).map(|_| seed.next_f64()).collect()
}

/// Host reference: the K-PP2 formula in plain Rust.
fn proj_reference(
    t: &PpProjTables,
    gv: &[f64],
    kpt: [f64; 3],
    g0: usize,
    nb: usize,
) -> (Vec<f64>, Vec<f64>) {
    let nrow = t.row_r.len() / 3;
    let mut out_re = vec![0.0; nrow * nb];
    let mut out_im = vec![0.0; nrow * nb];
    for p in 0..nrow {
        let (rx, ry, rz) = (t.row_r[p * 3], t.row_r[p * 3 + 1], t.row_r[p * 3 + 2]);
        let (alpha, coef) = (t.row_par[p * 5], t.row_par[p * 5 + 1]);
        let (c0, c1, c2) = (
            t.row_par[p * 5 + 2],
            t.row_par[p * 5 + 3],
            t.row_par[p * 5 + 4],
        );
        let t0 = t.row_term0[p] as usize;
        let nt = t.row_nterm[p] as usize;
        for g in 0..nb {
            let o = (g0 + g) * 3;
            let (gx, gy, gz) = (gv[o], gv[o + 1], gv[o + 2]);
            let (qx, qy, qz) = (gx + kpt[0], gy + kpt[1], gz + kpt[2]);
            let q2 = qx * qx + qy * qy + qz * qz;
            let x2 = q2 * 2.0 * alpha;
            let ql = c0 + c1 * x2 + c2 * x2 * x2;
            let mut poly = 0.0;
            for tt in t0..t0 + nt {
                let (ix, iy, iz) = (
                    t.term_pow[tt * 3] as i32,
                    t.term_pow[tt * 3 + 1] as i32,
                    t.term_pow[tt * 3 + 2] as i32,
                );
                poly += t.term_c[tt] * qx.powi(ix) * qy.powi(iy) * qz.powi(iz);
            }
            let val = poly * coef * (-alpha * q2).exp() * ql;
            let theta = gx * rx + gy * ry + gz * rz;
            let (sn, cs) = theta.sin_cos();
            out_re[p * nb + g] = val * cs;
            out_im[p * nb + g] = val * sn;
        }
    }
    (out_re, out_im)
}

#[test]
fn projector_block_matches_the_host_formula() {
    let t = small_tables();
    let mut rng = Lcg::new(7);
    let gv = pseudo_gvectors(40, &mut rng);
    let kpt = [0.11, -0.07, 0.05];
    let client = cpu_client();
    let (got_re, got_im) = pp_proj_block(&client, &t, &gv, kpt, 0, 40).expect("pp_proj_block");
    let (want_re, want_im) = proj_reference(&t, &gv, kpt, 0, 40);
    for i in 0..3 * 40 {
        let scale = 1.0 + want_re[i].abs();
        assert!((got_re[i] - want_re[i]).abs() < 1e-12 * scale, "re[{i}]");
        let scale = 1.0 + want_im[i].abs();
        assert!((got_im[i] - want_im[i]).abs() < 1e-12 * scale, "im[{i}]");
    }
}

#[test]
fn a_block_offset_selects_the_right_points() {
    let t = small_tables();
    let mut rng = Lcg::new(7);
    let gv = pseudo_gvectors(40, &mut rng);
    let kpt = [0.11, -0.07, 0.05];
    let client = cpu_client();
    let (full_re, full_im) = pp_proj_block(&client, &t, &gv, kpt, 0, 40).expect("full");
    let (got_re, got_im) = pp_proj_block(&client, &t, &gv, kpt, 17, 9).expect("offset");
    for p in 0..3 {
        for g in 0..9 {
            assert_eq!(
                got_re[p * 9 + g].to_bits(),
                full_re[p * 40 + 17 + g].to_bits(),
                "re[{p},{g}]"
            );
            assert_eq!(
                got_im[p * 9 + g].to_bits(),
                full_im[p * 40 + 17 + g].to_bits(),
                "im[{p},{g}]"
            );
        }
    }
}
