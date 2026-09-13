//! k-point CPHF seam tests (19-03): the k-aware `fvind` over the ONE solver.
//!
//! * Synthetic two-k CPHF against a per-k dense `solve_linear` reference.
//! * **Gate B**: the periodic path enters `pyscf_grad::cphf::solve`
//!   (invocation-counted `fvind` + bit-identical delegation check).
//! * `gen_response` absent on the PBC KS base, present on concrete RKS.
//! * `_get_jk` k-shift routing incl. both upstream refusals.
//! * Determinism at 1 vs 8 rayon workers inside one process.
//!
//! Run scoped: `cargo test -p pyscf-pbc-scf --test cphf_k`

use pyscf_algebra::solve_linear;
use pyscf_pbc_scf::cphf::{KCPHF_DEFAULT_MAX_CYCLE, KCPHF_DEFAULT_TOL, KcphfInput, dense_kvind, run_kcphf};
use pyscf_pbc_scf::response::{
    PbcKohnShamBase, ResponseError, ResponseJkBackend, RksGenResponse, resolve_jk_route,
};
use std::sync::atomic::{AtomicUsize, Ordering};

/// Two-k synthetic problem: k=0 is (nocc=2, nvir=3), k=1 is (nocc=1, nvir=2).
/// Spectra are occ-first then vir (the solver's split convention); kernels are
/// diagonally dominant so the Krylov solve converges fast.
struct TwoK {
    input: KcphfInput,
    kmat_k: Vec<Vec<f64>>,
    e_ai_k: Vec<Vec<f64>>,
}

fn two_k_fixture() -> TwoK {
    // k=0: occ energies [-0.6, -0.4], vir [0.3, 0.5, 0.8].
    let mo_energy_0 = vec![-0.6, -0.4, 0.3, 0.5, 0.8];
    let mo_occ_0 = vec![2.0, 2.0, 0.0, 0.0, 0.0];
    // k=1: occ [-0.5], vir [0.4, 0.7].
    let mo_energy_1 = vec![-0.5, 0.4, 0.7];
    let mo_occ_1 = vec![2.0, 0.0, 0.0];
    // RHS blocks (vir-major), arbitrary nonzero.
    let h1_0: Vec<f64> = (0..6).map(|i| 0.1 * (i as f64 + 1.0)).collect();
    let h1_1: Vec<f64> = vec![0.25, -0.15];
    // Diagonally-dominant dense kernels per k (vir-major ndim²).
    let kmat_0: Vec<f64> = (0..36)
        .map(|i| if i % 7 == 0 { 0.05 } else { 0.001 * ((i % 5) as f64) })
        .collect();
    let kmat_1: Vec<f64> = vec![0.04, 0.002, 0.002, 0.04];

    let e_ai = |e: &[f64], o: &[f64]| -> Vec<f64> {
        let eo: Vec<f64> = e.iter().zip(o.iter()).filter_map(|(&en, &oc)| {
            if oc > 0.0 {
                Some(en)
            } else {
                None
            }
        }).collect();
        let ea: Vec<f64> = e.iter().zip(o.iter()).filter_map(|(&en, &oc)| {
            if oc == 0.0 {
                Some(en)
            } else {
                None
            }
        }).collect();
        let mut v = Vec::with_capacity(ea.len() * eo.len());
        for &a in &ea {
            for &i in &eo {
                v.push(1.0 / (a - i));
            }
        }
        v
    };
    let e_ai_k = vec![
        e_ai(&mo_energy_0, &mo_occ_0),
        e_ai(&mo_energy_1, &mo_occ_1),
    ];
    TwoK {
        input: KcphfInput {
            mo_energy_k: vec![mo_energy_0, mo_energy_1],
            mo_occ_k: vec![mo_occ_0, mo_occ_1],
            h1_k: vec![h1_0, h1_1],
        },
        kmat_k: vec![kmat_0, kmat_1],
        e_ai_k,
    }
}

/// Dense reference for one k-block: `M·z = mo1base` with
/// `M = I + diag(e_ai)·Kmat`, `mo1base = h1·(−e_ai)`.
fn dense_ref(kmat: &[f64], e_ai: &[f64], h1: &[f64]) -> Vec<f64> {
    let nd = h1.len();
    let mut m = vec![0.0f64; nd * nd];
    for i in 0..nd {
        for j in 0..nd {
            m[i * nd + j] = e_ai[i] * kmat[i * nd + j];
        }
        m[i * nd + i] += 1.0;
    }
    let b: Vec<f64> = h1.iter().zip(e_ai.iter()).map(|(&h, &e)| h * -e).collect();
    solve_linear(&m, &b, nd).expect("dense reference must solve")
}

#[test]
fn kcphf_matches_per_k_dense_reference() {
    let fix = two_k_fixture();
    let fvind = dense_kvind(&fix.kmat_k);
    let z = run_kcphf(&fix.input, &fvind, KCPHF_DEFAULT_MAX_CYCLE, KCPHF_DEFAULT_TOL)
        .expect("kcphf must converge");
    assert_eq!(z.len(), 2);
    for k in 0..2 {
        let ref_k = dense_ref(&fix.kmat_k[k], &fix.e_ai_k[k], &fix.input.h1_k[k]);
        assert_eq!(z[k].len(), ref_k.len());
        let max: f64 = z[k]
            .iter()
            .zip(ref_k.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);
        assert!(max < 1e-8, "k={k} deviates from dense reference by {max:e}");
    }
}

/// Gate B: the periodic path enters `pyscf_grad::cphf::solve`.
#[test]
fn kcphf_routes_through_the_one_cphf_solver() {
    let fix = two_k_fixture();
    // (a) The fvind is invoked through the solver: count entries.
    let calls = AtomicUsize::new(0);
    let counted = |k: usize, z: &[f64]| -> Result<Vec<f64>, pyscf_core::PyscfRsError> {
        calls.fetch_add(1, Ordering::SeqCst);
        dense_kvind(&fix.kmat_k)(k, z)
    };
    let z = run_kcphf(&fix.input, &counted, KCPHF_DEFAULT_MAX_CYCLE, KCPHF_DEFAULT_TOL)
        .expect("kcphf must converge");
    assert!(calls.load(Ordering::SeqCst) > 0, "fvind was never entered");
    // (b) Delegation, not reimplementation: each block is bit-identical to a
    // direct `pyscf_grad::cphf::solve` call on the same block.
    for k in 0..2 {
        let direct = pyscf_grad::cphf::solve(
            &|zz| dense_kvind(&fix.kmat_k)(k, zz),
            &fix.input.mo_energy_k[k],
            &fix.input.mo_occ_k[k],
            &fix.input.h1_k[k],
            None,
            KCPHF_DEFAULT_MAX_CYCLE,
            KCPHF_DEFAULT_TOL,
            false,
            pyscf_grad::cphf::DEFAULT_LEVEL_SHIFT,
        )
        .expect("direct solve must converge");
        assert_eq!(z[k].len(), direct.len());
        for (a, b) in z[k].iter().zip(direct.iter()) {
            assert_eq!(a.to_bits(), b.to_bits(), "block {k} is not the solver's own output");
        }
    }
}

#[test]
fn gen_response_absent_on_base_present_on_rks() {
    // pbc/dft/rks.py:268 — the base leaves gen_response NotImplemented.
    assert!(!PbcKohnShamBase::HAS_GEN_RESPONSE);
    // :411 — only the concrete RKS rebinds it.
    assert!(RksGenResponse::HAS_GEN_RESPONSE);
}

#[test]
fn jk_kshift_routing_matches_upstream() {
    // kshift == 0 always takes the direct route, either backend.
    assert_eq!(
        resolve_jk_route(0, None, ResponseJkBackend::Direct),
        Ok(ResponseJkBackend::Direct)
    );
    assert_eq!(
        resolve_jk_route(0, Some(0.3), ResponseJkBackend::Fitted),
        Ok(ResponseJkBackend::Direct)
    );
    // Nonzero shift on a fitted backend takes the kshift path.
    assert_eq!(
        resolve_jk_route(1, None, ResponseJkBackend::Fitted),
        Ok(ResponseJkBackend::Fitted)
    );
    // Nonzero shift + range separation is refused.
    assert_eq!(
        resolve_jk_route(1, Some(0.3), ResponseJkBackend::Fitted),
        Err(ResponseError::KshiftWithRangeSeparation)
    );
    // Nonzero shift without a fitted backend is refused.
    assert_eq!(
        resolve_jk_route(2, None, ResponseJkBackend::Direct),
        Err(ResponseError::KshiftNeedsFittedJk)
    );
}

#[test]
fn kcphf_deterministic_across_thread_counts() {
    let run = || {
        let fix = two_k_fixture();
        let fvind = dense_kvind(&fix.kmat_k);
        run_kcphf(&fix.input, &fvind, KCPHF_DEFAULT_MAX_CYCLE, KCPHF_DEFAULT_TOL)
            .expect("kcphf must converge")
    };
    let pool1 = rayon::ThreadPoolBuilder::new().num_threads(1).build().unwrap();
    let pool8 = rayon::ThreadPoolBuilder::new().num_threads(8).build().unwrap();
    let z1 = pool1.install(run);
    let z8 = pool8.install(run);
    assert_eq!(z1.len(), z8.len());
    for (a, b) in z1.iter().zip(z8.iter()) {
        assert_eq!(a.len(), b.len());
        for (x, y) in a.iter().zip(b.iter()) {
            assert_eq!(x.to_bits(), y.to_bits());
        }
    }
}
