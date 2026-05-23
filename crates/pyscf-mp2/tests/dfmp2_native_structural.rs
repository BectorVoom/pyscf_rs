//! Native RI-MP2 structural / cross-check tests (always-on layer for the
//! `mp2-structural` CI job).
//!
//! Every arm here is ALWAYS-ON (no cintx, no live `intor`):
//!
//! 1. CROSS-PATH AGREEMENT (the un-gated correctness anchor, T-05-06-XCHECK):
//!    feed [`pyscf_mp2::emp2_rhf`] a hand-built synthetic [`DfIntegrals`] + toy
//!    [`Mp2Reference`] and assert it equals the conventional
//!    [`pyscf_mp2::dfrmp2_kernel`] `e_corr` on the SAME synthetic B-tensor. The
//!    native (occupied-pair) contraction and the conventional (full-ovov-block)
//!    contraction must give the same total correlation energy for the same
//!    integrals — this catches a divergent native math port WITHOUT needing the
//!    cintx#11 live integrals.
//! 2. UHF: assert [`pyscf_mp2::emp2_uhf`] on a synthetic open-shell B-tensor
//!    pair returns a finite, hand-checked correlation energy.
//! 3. STRUCTURAL: assert the native kernel `?`-propagates the `int3c2e_sph`
//!    cintx#11 gate as an `Err` (no panic) against a real Mole.
//!
//! The live-PySCF numeric parity assertions live in the cintx#11-gated
//! `dfmp2_native_energy` oracle arm (a separate CI job), not in this file.

use pyscf_core::{MOCoefficients, Mole};
use pyscf_df::{DfIntegrals, cholesky_eri};
use pyscf_mp2::{
    Frozen, Mp2Reference, NativeDFRMP2, NativeDFUMP2, UmpReference, dfrmp2_kernel, emp2_rhf,
    emp2_uhf,
};

/// Build a deterministic synthetic DF B-tensor `b_uvq` (ROW-MAJOR
/// `[nao,nao,naux]`): element `(μ,ν,Q)` at `μ*nao*naux + ν*naux + Q`.
///
/// The `seed` perturbs the fill so the αα and ββ open-shell B-tensors differ.
fn synthetic_b(nao: usize, naux: usize, seed: f64) -> DfIntegrals {
    let mut b = vec![0.0_f64; nao * nao * naux];
    for mu in 0..nao {
        for nu in 0..nao {
            for q in 0..naux {
                let v = 0.1 + seed + (mu as f64) * 0.5 + (nu as f64) * 0.25 + (q as f64) * 0.125;
                b[mu * nao * naux + nu * naux + q] = v;
            }
        }
    }
    DfIntegrals {
        b_uvq: b,
        naux,
        nao,
    }
}

/// Build a toy [`Mp2Reference`] with `nocc` occupied + `nvir` virtual orbitals
/// (so nao = nmo = nocc+nvir) and a deterministic non-trivial MO coefficient
/// block (column-major `[nao, nmo]`). Occupied columns come first.
///
/// `eshift` shifts the orbital energies so the α and β references differ in the
/// open-shell test.
fn toy_reference(nocc: usize, nvir: usize, eshift: f64) -> Mp2Reference {
    let nmo = nocc + nvir;
    let nao = nmo;
    let mut data = vec![0.0_f64; nao * nmo];
    for mo in 0..nmo {
        for ao in 0..nao {
            data[ao + mo * nao] = 0.3 + (ao as f64) * 0.2 - (mo as f64) * 0.15;
        }
    }
    let mut mo_occ = vec![0.0_f64; nmo];
    let mut mo_energy = vec![0.0_f64; nmo];
    for i in 0..nocc {
        mo_occ[i] = 2.0;
        mo_energy[i] = -0.8 + eshift + (i as f64) * 0.1; // occupied < 0
    }
    for a in 0..nvir {
        mo_energy[nocc + a] = 0.3 + eshift + (a as f64) * 0.2; // virtual > 0
    }
    Mp2Reference {
        mo_coeff: MOCoefficients {
            nao,
            nmo,
            data,
            energies: mo_energy.clone(),
            occupations: mo_occ.clone(),
        },
        mo_energy,
        mo_occ,
        e_hf: -1.0,
        converged: true,
        mol: Mole::default(),
    }
}

/// ALWAYS-ON cross-path agreement (T-05-06-XCHECK): the native `emp2_rhf` equals
/// the conventional `dfrmp2_kernel` `e_corr` on the SAME synthetic B-tensor.
///
/// Both ultimately contract the same DF `(ia|jb)` integrals; the native path
/// walks occupied pairs `j ≤ i` while the conventional path runs the full
/// closed-form over the materialized ovov block. With `ps = pt = 1.0` (the
/// native default) they MUST agree on the total correlation energy.
#[test]
fn dfmp2_native_emp2_rhf_agrees_with_conventional() {
    for &(nocc, nvir, naux) in &[(2usize, 2usize, 3usize), (2, 3, 4), (3, 2, 3)] {
        let nao = nocc + nvir;
        let df = synthetic_b(nao, naux, 0.0);
        let refr = toy_reference(nocc, nvir, 0.0);

        let native = emp2_rhf(&refr, &Frozen::None, &df).expect("native emp2_rhf");
        let conv = dfrmp2_kernel(&refr, &Frozen::None, &df, false)
            .expect("conventional dfrmp2_kernel")
            .e_corr;

        assert!(native.is_finite(), "native e_corr must be finite");
        // RELATIVE tolerance: the native (occupied-pair) and conventional
        // (full-ovov closed-form) paths sum the SAME DF (ia|jb) integrals in
        // DIFFERENT reduction orders, so the float results are bit-close but not
        // bit-identical. This synthetic case has near-degenerate denominators
        // (energy magnitude ~1e6), so an absolute tolerance is meaningless —
        // assert agreement to ~1e-10 RELATIVE (12+ significant figures).
        let rel = (native - conv).abs() / native.abs().max(1.0);
        assert!(
            rel < 1e-10,
            "native vs conventional disagree for (nocc={nocc}, nvir={nvir}, naux={naux}): \
             native={native}, conventional={conv}, rel={rel}"
        );
    }
}

/// ALWAYS-ON: the [`NativeDFRMP2`] driver `kernel()` returns the same `e_corr`
/// as the free [`emp2_rhf`] (structural wiring of the native driver) and reports
/// `e_corr` finite.
#[test]
fn dfmp2_native_dfrmp2_driver_kernel_matches_free_fn() {
    let nocc = 2;
    let nvir = 2;
    let nao = nocc + nvir;
    let naux = 3;
    let df = synthetic_b(nao, naux, 0.0);
    let refr = toy_reference(nocc, nvir, 0.0);

    let free = emp2_rhf(&refr, &Frozen::None, &df).expect("emp2_rhf");
    let driver = NativeDFRMP2 {
        reference: refr,
        df,
        frozen: Frozen::None,
    };
    let res = driver.kernel().expect("NativeDFRMP2::kernel");
    assert!(res.e_corr.is_finite());
    assert!(
        (res.e_corr - free).abs() < 1e-12,
        "driver e_corr {} vs free fn {}",
        res.e_corr,
        free
    );
    // The native fast path reports e_corr only (no SS/OS split).
    assert_eq!(res.e_os, res.e_corr);
    assert!(res.t2.is_none());
}

/// Independent longhand reference for the native UHF correlation energy on a
/// synthetic per-spin B-tensor pair. Strict scalar folds (never `oracle_*`), so
/// the assertion is an independent check of the production reduction.
///
/// Ports `emp2_uhf` (`dfump2_native.py:272-355`): same-spin (pt, antisymmetrized,
/// j<i) for each spin + opposite-spin (ps, direct, all i(α)×j(β)). `ps = pt = 1`.
//
// The raw `i`/`j` occupied indices are load-bearing: they index the energy
// slices AND compute the flat `[i,Q,b]` offset in the same body (mirrors the
// production `ump2.rs` discipline). The enumerate rewrite would obscure the
// flat-index layout rather than clarify it.
#[allow(clippy::needless_range_loop)]
fn reference_emp2_uhf(refr: &UmpReference, df_a: &DfIntegrals, df_b: &DfIntegrals) -> f64 {
    let ps = 1.0_f64;
    let pt = 1.0_f64;

    // Per-spin native [i,Q,b] transform + active energies (no frozen).
    let (ints_a, nocc_a, nvir_a, eo_a, ev_a) = longhand_iqb(&refr.alpha, df_a);
    let (ints_b, nocc_b, nvir_b, eo_b, ev_b) = longhand_iqb(&refr.beta, df_b);
    let naux = df_a.naux;

    let kab = |ints_x: &[f64], i: usize, nvir_x: usize, ints_y: &[f64], j: usize, nvir_y: usize| {
        // K[a,b] = Σ_Q B^Q_ia·B^Q_jb (longhand fold).
        let mut k = vec![0.0_f64; nvir_x * nvir_y];
        for a in 0..nvir_x {
            for b in 0..nvir_y {
                let mut acc = 0.0_f64;
                for q in 0..naux {
                    acc +=
                        ints_x[(i * naux + q) * nvir_x + a] * ints_y[(j * naux + q) * nvir_y + b];
                }
                k[a * nvir_y + b] = acc;
            }
        }
        k
    };

    let mut energy = 0.0_f64;
    // Same-spin αα.
    for i in 0..nocc_a {
        for j in 0..i {
            let k = kab(&ints_a, i, nvir_a, &ints_a, j, nvir_a);
            let mut acc = 0.0_f64;
            for a in 0..nvir_a {
                for b in 0..nvir_a {
                    let de = eo_a[i] + eo_a[j] - ev_a[a] - ev_a[b];
                    let t = (k[a * nvir_a + b] - k[b * nvir_a + a]) / de;
                    acc += t * k[a * nvir_a + b];
                }
            }
            energy += pt * acc;
        }
    }
    // Same-spin ββ.
    for i in 0..nocc_b {
        for j in 0..i {
            let k = kab(&ints_b, i, nvir_b, &ints_b, j, nvir_b);
            let mut acc = 0.0_f64;
            for a in 0..nvir_b {
                for b in 0..nvir_b {
                    let de = eo_b[i] + eo_b[j] - ev_b[a] - ev_b[b];
                    let t = (k[a * nvir_b + b] - k[b * nvir_b + a]) / de;
                    acc += t * k[a * nvir_b + b];
                }
            }
            energy += pt * acc;
        }
    }
    // Opposite-spin αβ.
    for i in 0..nocc_a {
        for j in 0..nocc_b {
            let k = kab(&ints_a, i, nvir_a, &ints_b, j, nvir_b);
            let mut acc = 0.0_f64;
            for a in 0..nvir_a {
                for b in 0..nvir_b {
                    let de = eo_a[i] + eo_b[j] - ev_a[a] - ev_b[b];
                    let t = k[a * nvir_b + b] / de;
                    acc += t * k[a * nvir_b + b];
                }
            }
            energy += ps * acc;
        }
    }
    energy
}

/// Longhand `[i,Q,b]` transform + active occ/vir energies for one spin
/// reference (no frozen, occupied columns first). Returns
/// `(ints, nocc, nvir, e_occ, e_vir)`.
fn longhand_iqb(
    refr: &Mp2Reference,
    df: &DfIntegrals,
) -> (Vec<f64>, usize, usize, Vec<f64>, Vec<f64>) {
    let nao = df.nao;
    let naux = df.naux;
    let nocc = refr.mo_occ.iter().filter(|&&o| o > 0.0).count();
    let nvir = refr.mo_coeff.nmo - nocc;
    let mut ints = vec![0.0_f64; nocc * naux * nvir];
    for i in 0..nocc {
        for q in 0..naux {
            for b in 0..nvir {
                let mut acc = 0.0_f64;
                for mu in 0..nao {
                    for nu in 0..nao {
                        let ci = refr.mo_coeff.data[mu + i * nao];
                        let cb = refr.mo_coeff.data[nu + (nocc + b) * nao];
                        acc += ci * df.b_uvq[mu * nao * naux + nu * naux + q] * cb;
                    }
                }
                ints[(i * naux + q) * nvir + b] = acc;
            }
        }
    }
    let mut e_occ = Vec::new();
    let mut e_vir = Vec::new();
    for col in 0..refr.mo_coeff.nmo {
        if refr.mo_occ[col] > 0.0 {
            e_occ.push(refr.mo_energy[col]);
        } else {
            e_vir.push(refr.mo_energy[col]);
        }
    }
    (ints, nocc, nvir, e_occ, e_vir)
}

/// ALWAYS-ON: `emp2_uhf` on a synthetic open-shell B-tensor pair returns a
/// finite correlation energy matching the independent longhand reference.
#[test]
fn dfmp2_native_emp2_uhf_finite_hand_checked() {
    let nocc = 2;
    let nvir = 2;
    let nao = nocc + nvir;
    let naux = 3;
    // Distinct α / β B-tensors and references (open-shell).
    let df_a = synthetic_b(nao, naux, 0.0);
    let df_b = synthetic_b(nao, naux, 0.07);
    let refr = UmpReference {
        alpha: toy_reference(nocc, nvir, 0.0),
        beta: toy_reference(nocc, nvir, 0.05),
    };

    let got = emp2_uhf(&refr, &Frozen::None, &df_a, &df_b).expect("emp2_uhf");
    assert!(got.is_finite(), "uhf e_corr must be finite");

    let want = reference_emp2_uhf(&refr, &df_a, &df_b);
    assert!(
        (got - want).abs() < 1e-10,
        "native emp2_uhf {got} vs longhand reference {want}"
    );

    // Driver wiring: NativeDFUMP2::kernel matches the free fn.
    let driver = NativeDFUMP2 {
        reference: refr,
        df_alpha: df_a,
        df_beta: df_b,
        frozen: Frozen::None,
    };
    let res = driver.kernel().expect("NativeDFUMP2::kernel");
    assert!(
        (res.e_corr - got).abs() < 1e-12,
        "driver uhf e_corr {} vs free fn {}",
        res.e_corr,
        got
    );
    assert!(res.t2.is_none());
}

/// ALWAYS-ON structural: a single occupied / single virtual closed-shell case
/// also agrees between native and conventional — the smallest cross-check.
#[test]
fn dfmp2_native_emp2_rhf_single_orbital_agrees() {
    let nocc = 1;
    let nvir = 1;
    let nao = nocc + nvir;
    let naux = 2;
    let df = synthetic_b(nao, naux, 0.0);
    let refr = toy_reference(nocc, nvir, 0.0);

    let native = emp2_rhf(&refr, &Frozen::None, &df).expect("native emp2_rhf");
    let conv = dfrmp2_kernel(&refr, &Frozen::None, &df, false)
        .expect("dfrmp2_kernel")
        .e_corr;
    assert!(native.is_finite());
    assert!(
        (native - conv).abs() < 1e-12,
        "single-orbital native {native} vs conventional {conv}"
    );
}

/// ALWAYS-ON structural: the native kernel `?`-propagates the `int3c2e_sph`
/// cintx#11 gate as an `Err` (or a well-formed all-zero gated tensor) when built
/// against a REAL Mole — it must NEVER panic. The numeric native-DF-MP2 energy
/// is cintx-gated; the error-propagation SHAPE is always-on (T-05-06-FFI).
#[test]
fn dfmp2_native_kernel_propagates_int3c2e_gate_no_panic() {
    use pyscf_gto::{AtomInput, BasisInput, M, MoleBuildArgs};

    let mol = M(MoleBuildArgs {
        atom: AtomInput::String("H 0 0 0; H 0 0 0.74".into()),
        basis: BasisInput::Name("sto-3g".into()),
        ..Default::default()
    })
    .expect("build H2");

    // default_ri (mp2fit *-ri aux) — NOT the JK-fit aux.
    let aux = pyscf_df::default_ri("sto-3g");
    // cholesky_eri must NOT panic: with the int3c2e_sph cintx#11 gap the (P|Q)
    // is all-zero → SingularAux Err is the expected gated shape; if cintx lands
    // a real int2c2e the call may succeed with an all-zero b_uvq. Either way the
    // result is a well-formed Result and the process does not abort.
    let res: Result<DfIntegrals, _> = cholesky_eri(&mol, aux);
    match res {
        Ok(df) => {
            // Gated all-zero path: emp2_rhf on the all-zero B-tensor must still
            // be a well-formed Result (never panics). A toy reference with the
            // right nao is fed so the transform shape-checks pass.
            assert_eq!(df.nao, mol.nao_nr);
            assert_eq!(df.b_uvq.len(), df.nao * df.nao * df.naux);
            // The H2/sto-3g reference has nao=2 (1 occ, 1 vir).
            let refr = toy_reference(1, 1, 0.0);
            // Only run emp2_rhf if the synthetic reference nao matches the gated
            // B-tensor nao (sto-3g H2 → nao 2). If it does, the all-zero tensor
            // yields a finite (zero) energy; either way no panic.
            if refr.mo_coeff.nao == df.nao {
                let e = emp2_rhf(&refr, &Frozen::None, &df);
                assert!(e.is_ok(), "all-zero gated tensor must not error/panic");
            }
        }
        Err(_) => {
            // SingularAux (all-zero (P|Q)) — the expected cintx#11-gated shape.
        }
    }
}
