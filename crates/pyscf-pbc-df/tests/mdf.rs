//! Plan 14-06 acceptance — mixed density fitting, and Gate 2.
//!
//! # Gate 2 is stated against the CC ladder, not `mdfladder.out`
//!
//! `measurements/mdfladder.out` was recorded with `df.MDF`'s DEFAULT builder,
//! and `MDF._prefer_ccdf` is `False` (`mdf.py:79`) — so every row of it is the
//! RANGE-SEPARATED `_RSMDFBuilder`, which plan 14-07 owns. This plan ports
//! `_CCMDFBuilder`, exactly as 14-02 ported `_CCGDFBuilder`, so the ladder it
//! must reproduce is `measurements/mdfladder_cc.out`:
//!
//! ```text
//! He-fcc/sto-3g 2x2x2, CC route, against E_KRHF(FFTDF, mesh 31)
//!   GDF (no plane waves)   6.002e-05
//!   MDF mesh  7            1.695e-06
//!   MDF mesh  9 (default)  5.476e-08
//!   MDF mesh 11            6.684e-09   <- the plateau
//!   MDF mesh 15            3.216e-08
//!   MDF mesh 21            3.318e-08
//! ```
//!
//! **The ladder is not monotone and a monotone gate would fail a correct
//! implementation.** Two floors are in play — MDF's own auxiliary fit, and the
//! mesh-31 truncation of the FFTDF *reference* — and past the crossover the
//! comparison measures the reference. Phase 13's Gate 2 had the same structure.
//! The gate is therefore stated as: MDF beats GDF by ≥3 orders at the plateau,
//! falls by ≥2 orders from mesh 7 to mesh 11, and then stays within one order
//! of the plateau.
//!
//! Also corrected here: `14-06-PLAN.md` states MDF's default mesh as `[7,7,7]`.
//! It is not — it is `[11,11,11]` on diamond 2x2x2 and `[9,9,9]` on He-fcc
//! 2x2x2 (measured, `mdfladder_cc.out`). Mesh 7 is simply `mdfladder.py`'s
//! lowest rung.
//!
//! # Where Gate 2 lives
//!
//! Gate 2 and the MDF oracle drive a converged `KRHF`, and `pyscf-pbc-scf`
//! depends on this crate — so they are in
//! `crates/pyscf-pbc-scf/tests/df_swap.rs`, beside 14-04's Gate 1, rather than
//! behind a dev-dependency cycle. What stays here is everything that needs no
//! SCF: the defaults, the refusal, and the two structural properties of
//! `get_jk`.

mod common;

use pyscf_pbc_df::traits::{JkOpts, PeriodicDf};
use pyscf_pbc_df::Mdf;
use pyscf_pbc_gto::Cell;

fn kpts_of(cell: &Cell, mesh: [usize; 3]) -> Vec<[f64; 3]> {
    cell.make_kpts(mesh).expect("make_kpts")
}

/// `dm[k] = S(k)^{-1}`-free stand-in: a Hermitian, positive, k-dependent
/// density matrix. J/K structural tests need a density that is NOT real and
/// NOT the identity, or a missing conjugate cancels.
fn probe_dms(nao: usize, nkpts: usize) -> Vec<Vec<pyscf_algebra::CTensor>> {
    let mut seed = 0x2f6e_2b1cu64;
    let mut rand = || {
        seed = seed
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((seed >> 11) as f64 / (1u64 << 53) as f64) - 0.5
    };
    let mut out = Vec::with_capacity(nkpts);
    for _ in 0..nkpts {
        let mut m = pyscf_algebra::CTensor::zeros(nao * nao);
        for p in 0..nao {
            for q in 0..nao {
                m.re[p * nao + q] = rand();
                m.im[p * nao + q] = rand();
            }
        }
        // Hermitise, then make it diagonally dominant so it looks like a
        // density rather than noise.
        let src = m.clone();
        for p in 0..nao {
            for q in 0..nao {
                m.re[p * nao + q] = 0.5 * (src.re[p * nao + q] + src.re[q * nao + p]);
                m.im[p * nao + q] = 0.5 * (src.im[p * nao + q] - src.im[q * nao + p]);
            }
            m.re[p * nao + p] += 2.0;
            m.im[p * nao + p] = 0.0;
        }
        out.push(m);
    }
    vec![out]
}

// ---------------------------------------------------------------------------
// Task 4 — the class and its defaults
// ---------------------------------------------------------------------------

/// MDF's default mesh, measured. **The plan's `[7,7,7]` is wrong**; see the
/// module docs. This test is what stops the wrong number coming back.
#[test]
fn mdf_default_mesh_is_the_builders_own_estimate() {
    let cell = common::he_all_electron();
    let m = Mdf::new(cell.clone(), &kpts_of(&cell, [2, 2, 2]));
    assert_eq!(
        m.resolved_mesh().expect("mesh"),
        [9, 9, 9],
        "He-fcc 2x2x2: measured upstream at [9,9,9] (mdfladder_cc.out)"
    );
    let g = Mdf::new(cell.clone(), &[[0.0; 3]]);
    assert_eq!(g.resolved_mesh().expect("mesh"), [11, 11, 11], "He-fcc gamma");

    let d = common::diamond();
    let dm = Mdf::new(d.clone(), &kpts_of(&d, [2, 2, 2]));
    assert_eq!(dm.resolved_mesh().expect("mesh"), [11, 11, 11], "diamond 2x2x2");
}

/// An explicit mesh overrides the estimate, and `name()` reports MDF.
#[test]
fn mdf_mesh_is_settable_and_named() {
    let cell = common::he_all_electron();
    let mut m = Mdf::new(cell, &[[0.0; 3]]);
    m.mesh = Some([7, 7, 7]);
    assert_eq!(m.resolved_mesh().expect("mesh"), [7, 7, 7]);
    assert_eq!(m.name(), "MDF");
}

/// `_RSMDFBuilder` is REFUSED, not silently substituted — D-PBC-20. It is
/// upstream's default route and plan 14-07 owns it.
#[test]
fn prefer_ccdf_false_is_refused() {
    let cell = common::he_all_electron();
    let mut m = Mdf::new(cell, &[[0.0; 3]]);
    m.prefer_ccdf = false;
    let e = m.resolved_mesh().expect_err("must refuse the RS route");
    let msg = format!("{e}");
    assert!(
        msg.contains("_RSMDFBuilder") || msg.contains("14-07"),
        "the refusal must name the plan that owns it: {msg}"
    );
}

// ---------------------------------------------------------------------------
// Task 5.3 — Hermiticity
// ---------------------------------------------------------------------------

/// **Task 5.3** — `vj` and `vk` are Hermitian at every k-point, no oracle.
///
/// The sharpest cheap check on the sum of the two halves: `df_jk`'s
/// contraction and `aft_jk`'s are Hermitian separately, so a mis-paired
/// `(ki, kj)` in either summand shows up here immediately.
#[test]
fn mdf_vj_and_vk_are_hermitian() {
    let cell = common::he_all_electron();
    let kpts = kpts_of(&cell, [2, 1, 1]);
    let nao = cell.mol.nao_nr;
    let df = Mdf::new(cell, &kpts);
    let dms = probe_dms(nao, kpts.len());
    let out = df
        .get_jk(&dms, &kpts, JkOpts::hermitian())
        .expect("MDF get_jk");

    for (name, mats) in [("vj", out.vj.as_ref()), ("vk", out.vk.as_ref())] {
        let mats = mats.unwrap_or_else(|| panic!("{name} was not built"));
        let mut worst = 0.0f64;
        for k in &mats[0] {
            for p in 0..nao {
                for q in 0..nao {
                    let (a, b) = (p * nao + q, q * nao + p);
                    worst = worst.max((k.re[a] - k.re[b]).abs());
                    worst = worst.max((k.im[a] + k.im[b]).abs());
                }
            }
        }
        assert!(worst < 1e-13, "MDF {name} is not Hermitian: {worst:e}");
    }
}

/// `get_jk` is exactly `df_jk` + `aft_jk`, and `exxdiv` is carried by the
/// PLANE-WAVE half alone (`mdf_jk.py:61-62`). Verified by rebuilding the sum
/// from its two published summands.
#[test]
fn mdf_jk_is_the_sum_of_its_two_halves() {
    let cell = common::he_all_electron();
    let kpts = kpts_of(&cell, [2, 1, 1]);
    let nao = cell.mol.nao_nr;
    let df = Mdf::new(cell, &kpts);
    let dms = probe_dms(nao, kpts.len());

    let mut opts = JkOpts::hermitian();
    opts.exxdiv = Some(pyscf_pbc_gto::ExxDiv::Ewald);
    let out = df.get_jk(&dms, &kpts, opts).expect("get_jk");

    let g = pyscf_pbc_df::gdf::jk::get_k_kpts(df.gdf().expect("gdf"), &dms, &kpts, None)
        .expect("cderi half");
    let a = pyscf_pbc_df::aft_jk::get_k_kpts(
        df.aftdf().expect("aftdf"),
        &dms,
        &kpts,
        Some(pyscf_pbc_gto::ExxDiv::Ewald),
        None,
    )
    .expect("pw half");

    let vk = out.vk.expect("vk");
    let mut worst = 0.0f64;
    for k in 0..kpts.len() {
        for i in 0..nao * nao {
            worst = worst.max((vk[0][k].re[i] - (g[0][k].re[i] + a[0][k].re[i])).abs());
            worst = worst.max((vk[0][k].im[i] - (g[0][k].im[i] + a[0][k].im[i])).abs());
        }
    }
    assert!(worst == 0.0, "MDF vk is not exactly the two halves: {worst:e}");
}

