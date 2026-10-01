//! BAND-09 — the split contraction (`nsplit` lanes per output element, a
//! fixed-order fold of the partials) forced on the CPU runtime through
//! `PYSCF_PBC_BAND_VMAT_SPLIT`, against the unsplit kernel.
//!
//! The split reorders the grid sum, so this is a tolerance gate; the fold is
//! in a fixed order, so repeated runs must be bitwise identical.
//!
//! One test per file: the switch is a process-wide environment variable.
#![cfg(feature = "cpu")]

use cubecl::Runtime;
use pyscf_algebra::AlgebraClient;
use pyscf_kernels::pbc::{AoPlanes, band_vmat};

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

#[test]
fn split_contraction_matches_unsplit_and_is_deterministic() {
    let client = AlgebraClient::Cpu(cubecl_cpu::CpuRuntime::client(&cubecl_cpu::CpuDevice));
    // Γ present (per-k launches) and absent (one launch); a grid that is not
    // a multiple of any split; splits that exceed and equal small sizes.
    for gamma in [vec![true, false, false], vec![false, false, false]] {
        let (nkpts, nao, ngrids, comp, nvar) = (3, 5, 301, 4, 4);
        let re = lcg(7, nkpts * comp * nao * ngrids);
        let im = lcg(8, nkpts * comp * nao * ngrids);
        let wv = lcg(99, nvar * ngrids);
        let ao = AoPlanes { re: &re, im: &im };
        let run = |split: &str| {
            // SAFETY: the only test in this binary.
            unsafe { std::env::set_var("PYSCF_PBC_BAND_VMAT_SPLIT", split) };
            band_vmat(&client, &ao, &wv, nvar, comp, nkpts, nao, ngrids, &gamma).expect("band_vmat")
        };
        let base = run("1");
        for split in ["2", "8", "32", "301", "4096"] {
            let a = run(split);
            let b = run(split);
            for (k, (((ar, ai), (br, bi)), (wr, wi))) in a.iter().zip(&b).zip(&base).enumerate() {
                for i in 0..ar.len() {
                    assert_eq!(
                        ar[i].to_bits(),
                        br[i].to_bits(),
                        "split {split} not deterministic"
                    );
                    assert_eq!(
                        ai[i].to_bits(),
                        bi[i].to_bits(),
                        "split {split} not deterministic"
                    );
                    let tol = 1e-12 * (1.0 + wr[i].abs() + wi[i].abs());
                    assert!(
                        (ar[i] - wr[i]).abs() <= tol,
                        "split {split} re k={k} i={i}: {} vs {}",
                        ar[i],
                        wr[i]
                    );
                    assert!(
                        (ai[i] - wi[i]).abs() <= tol,
                        "split {split} im k={k} i={i}: {} vs {}",
                        ai[i],
                        wi[i]
                    );
                }
            }
        }
    }
    unsafe { std::env::remove_var("PYSCF_PBC_BAND_VMAT_SPLIT") };
}
