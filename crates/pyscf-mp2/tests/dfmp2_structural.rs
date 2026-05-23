//! DF-MP2 structural / wiring + synthetic-contraction tests (always-on layer
//! for the `mp2-structural` CI job).
//!
//! Every arm here is ALWAYS-ON (no cintx, no live `intor`):
//!
//! 1. Feed [`df_ao2mo`] a HAND-BUILT synthetic [`DfIntegrals`] and a toy
//!    [`Mp2Reference`]; assert the resulting `(ia|jb)` block equals an
//!    independent longhand `Σ_Q B^Q_ia·B^Q_jb` reference computed inline (proves
//!    the DF contraction math un-gated, mirroring the 05-02 synthetic-ERI
//!    approach).
//! 2. Assert [`dfrmp2_kernel`] on the synthetic DF block returns a finite
//!    `e_corr` matching a hand-computed closed-form value.
//! 3. Structural: assert that [`pyscf_df::cholesky_eri`] against a real Mole
//!    propagates the `int3c2e_sph` gate as an `Err` (does NOT panic) — the
//!    numeric DF-MP2 energy oracle is cintx#11-gated, but the error-propagation
//!    SHAPE is always-on.
//!
//! The numeric upstream-PySCF parity assertions live in the cintx#11-gated
//! `dfmp2_energy` oracle arm (a separate CI job), not in this file.

use pyscf_core::{MOCoefficients, Mole};
use pyscf_df::{DfIntegrals, cholesky_eri};
use pyscf_mp2::{Frozen, Mp2Reference, df_ao2mo, dfrmp2_kernel};

/// Build a deterministic synthetic DF B-tensor `b_uvq` (ROW-MAJOR
/// `[nao,nao,naux]`): element `(μ,ν,Q)` at `μ*nao*naux + ν*naux + Q`.
fn synthetic_b(nao: usize, naux: usize) -> DfIntegrals {
    let mut b = vec![0.0_f64; nao * nao * naux];
    for mu in 0..nao {
        for nu in 0..nao {
            for q in 0..naux {
                // Non-symmetric, non-trivial fill so the transform exercises
                // every index distinctly.
                let v = 0.1 + (mu as f64) * 0.5 + (nu as f64) * 0.25 + (q as f64) * 0.125;
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
fn toy_reference(nocc: usize, nvir: usize) -> Mp2Reference {
    let nmo = nocc + nvir;
    let nao = nmo;
    let mut data = vec![0.0_f64; nao * nmo];
    for mo in 0..nmo {
        for ao in 0..nao {
            // Distinct, non-singular coefficient pattern.
            data[ao + mo * nao] = 0.3 + (ao as f64) * 0.2 - (mo as f64) * 0.15;
        }
    }
    let mut mo_occ = vec![0.0_f64; nmo];
    let mut mo_energy = vec![0.0_f64; nmo];
    for i in 0..nocc {
        mo_occ[i] = 2.0;
        mo_energy[i] = -0.8 + (i as f64) * 0.1; // occupied energies < 0
    }
    for a in 0..nvir {
        mo_energy[nocc + a] = 0.3 + (a as f64) * 0.2; // virtual energies > 0
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

/// Independent longhand reference for the DF `(ia|jb)` block:
/// `B^Q_ia = Σ_{μ,ν} C_μ^i · b[μ,ν,Q] · C_ν^a` then
/// `(ia|jb) = Σ_Q B^Q_ia · B^Q_jb`. All strict scalar folds (never `oracle_*`),
/// so the assertion is an independent check of the production reduction order.
///
/// `co`/`cv` are column-major `[nao,nocc]`/`[nao,nvir]`. Output is C-order
/// `[nocc,nvir,nocc,nvir]`: `(i,a,j,b)` at `((i*nvir + a)*nocc + j)*nvir + b`.
fn reference_ovov(df: &DfIntegrals, co: &[f64], nocc: usize, cv: &[f64], nvir: usize) -> Vec<f64> {
    let nao = df.nao;
    let naux = df.naux;
    // ovl[(i,a,Q)] = B^Q_ia, ROW-MAJOR [nocc,nvir,naux].
    let mut ovl = vec![0.0_f64; nocc * nvir * naux];
    for i in 0..nocc {
        for a in 0..nvir {
            for q in 0..naux {
                let mut acc = 0.0_f64;
                for mu in 0..nao {
                    for nu in 0..nao {
                        acc += co[mu + i * nao]
                            * df.b_uvq[mu * nao * naux + nu * naux + q]
                            * cv[nu + a * nao];
                    }
                }
                ovl[(i * nvir + a) * naux + q] = acc;
            }
        }
    }
    // (ia|jb) = Σ_Q B^Q_ia · B^Q_jb.
    let mut ovov = vec![0.0_f64; nocc * nvir * nocc * nvir];
    for i in 0..nocc {
        for a in 0..nvir {
            for j in 0..nocc {
                for b in 0..nvir {
                    let mut acc = 0.0_f64;
                    for q in 0..naux {
                        acc += ovl[(i * nvir + a) * naux + q] * ovl[(j * nvir + b) * naux + q];
                    }
                    ovov[((i * nvir + a) * nocc + j) * nvir + b] = acc;
                }
            }
        }
    }
    ovov
}

/// Extract the column-major occupied / virtual MO blocks from a toy reference
/// (occupied columns first, all active — no frozen). Returns `(co, cv)` as
/// column-major `[nao,nocc]` / `[nao,nvir]`.
fn split_co_cv(refr: &Mp2Reference, nocc: usize, nvir: usize) -> (Vec<f64>, Vec<f64>) {
    let nao = refr.mo_coeff.nao;
    let mut co = vec![0.0_f64; nao * nocc];
    let mut cv = vec![0.0_f64; nao * nvir];
    for i in 0..nocc {
        co[i * nao..(i + 1) * nao].copy_from_slice(&refr.mo_coeff.data[i * nao..(i + 1) * nao]);
    }
    for a in 0..nvir {
        let col = nocc + a;
        cv[a * nao..(a + 1) * nao].copy_from_slice(&refr.mo_coeff.data[col * nao..(col + 1) * nao]);
    }
    (co, cv)
}

/// ALWAYS-ON (the un-gated DF correctness check): `df_ao2mo` on a synthetic
/// `DfIntegrals` equals the independent longhand `Σ_Q B^Q_ia·B^Q_jb` reference.
/// Mirrors the 05-02 synthetic-ERI roundtrip approach.
#[test]
fn df_ao2mo_matches_longhand_reference() {
    let nocc = 2;
    let nvir = 2;
    let nao = nocc + nvir;
    let naux = 3;

    let df = synthetic_b(nao, naux);
    let refr = toy_reference(nocc, nvir);

    let got = df_ao2mo(&refr, &Frozen::None, &df).expect("df_ao2mo");
    assert_eq!(got.nocc, nocc);
    assert_eq!(got.nvir, nvir);
    assert_eq!(got.ovov.len(), nocc * nvir * nocc * nvir);

    let (co, cv) = split_co_cv(&refr, nocc, nvir);
    let want = reference_ovov(&df, &co, nocc, &cv, nvir);
    assert_eq!(want.len(), got.ovov.len());
    for (idx, (a, b)) in got.ovov.iter().zip(want.iter()).enumerate() {
        // Small case (every fold length ≤ nao*nao = 16, ≤ naux = 3), so the
        // oracle reduction base case is a strict left fold matching the
        // longhand reference to floating-point equality.
        assert!(
            (a - b).abs() < 1e-12,
            "df_ao2mo mismatch at flat index {idx}: got {a}, want {b}"
        );
    }
}

/// ALWAYS-ON: a single occupied / single virtual synthetic DF block fed to the
/// DFRMP2 hook produces the SAME `(ia|jb)` as feeding the same B-tensor
/// directly — confirms the DFRMP2 swap-the-source seam routes through
/// `df_ao2mo` (structural wiring of the conventional path).
#[test]
fn dfrmp2_hook_routes_through_df_ao2mo() {
    let nocc = 1;
    let nvir = 1;
    let nao = nocc + nvir;
    let naux = 2;
    let df = synthetic_b(nao, naux);
    let refr = toy_reference(nocc, nvir);

    let direct = df_ao2mo(&refr, &Frozen::None, &df).expect("df_ao2mo");

    let hooks = pyscf_mp2::DFRMP2 {
        reference: refr.clone(),
        df: df.clone(),
    };
    use pyscf_mp2::Mp2OverrideHooks;
    let via_hook = hooks.ao2mo(&refr, &Frozen::None).expect("hook ao2mo");

    assert_eq!(direct.ovov, via_hook.ovov);
    assert_eq!(direct.nocc, via_hook.nocc);
    assert_eq!(direct.nvir, via_hook.nvir);
}

/// ALWAYS-ON: `dfrmp2_kernel` on the synthetic DF block returns the
/// hand-computed closed-form `e_corr`.
///
/// nocc=1, nvir=1, naux=2. We compute the DF `(ia|jb)` longhand, then the
/// closed-form RMP2 (single i=j=0, single a=b=0):
///   g = (00|00); eia = εo − εv; denom = 2·eia; t2 = g/denom;
///   edi = 2·g·t2; exi = −g·t2;
///   e_ss = edi·0.5 + exi; e_os = edi·0.5; e_corr = e_ss + e_os.
/// For nocc=nvir=1 the exchange (ib|ja) == (ia|jb), so exi = −g·t2 and
/// e_ss = edi·0.5 + exi = g·t2 − g·t2 = 0; e_corr = e_os = g·t2.
#[test]
fn dfrmp2_kernel_hand_computed_energy() {
    let nocc = 1;
    let nvir = 1;
    let nao = nocc + nvir;
    let naux = 2;
    let df = synthetic_b(nao, naux);
    let refr = toy_reference(nocc, nvir);

    // Longhand DF (00|00).
    let (co, cv) = split_co_cv(&refr, nocc, nvir);
    let ovov = reference_ovov(&df, &co, nocc, &cv, nvir);
    let g = ovov[0];

    // εo (col 0) and εv (col 1) from the toy reference.
    let eo = refr.mo_energy[0];
    let ev = refr.mo_energy[1];
    let eia = eo - ev;
    let denom = 2.0 * eia;
    let t2 = g / denom;
    let edi = 2.0 * g * t2;
    let exi = -g * t2;
    let e_ss_ref = edi * 0.5 + exi;
    let e_os_ref = edi * 0.5;
    let e_corr_ref = e_ss_ref + e_os_ref;

    let res = dfrmp2_kernel(&refr, &Frozen::None, &df, true).expect("dfrmp2_kernel");
    assert!(res.e_corr.is_finite(), "e_corr must be finite");
    assert!(
        (res.e_corr - e_corr_ref).abs() < 1e-12,
        "e_corr {} vs ref {}",
        res.e_corr,
        e_corr_ref
    );
    assert!(
        (res.e_ss - e_ss_ref).abs() < 1e-12,
        "e_ss {} vs ref {}",
        res.e_ss,
        e_ss_ref
    );
    assert!(
        (res.e_os - e_os_ref).abs() < 1e-12,
        "e_os {} vs ref {}",
        res.e_os,
        e_os_ref
    );
    // with_t2 stored.
    let amp = res.t2.expect("t2 stored");
    assert_eq!(amp.nocc, 1);
    assert_eq!(amp.nvir, 1);
}

/// ALWAYS-ON structural: `df_ao2mo` validates the B-tensor / MO-block shape and
/// returns `Err` (never panics / OOB) when the synthetic B-tensor's `nao` does
/// not match the reference's `nao` (T-05-05-FFI caller→df_ao2mo boundary).
#[test]
fn df_ao2mo_shape_mismatch_errors_not_panics() {
    let nocc = 1;
    let nvir = 1;
    let refr = toy_reference(nocc, nvir); // nao = 2
    // A B-tensor with a DIFFERENT nao (3) than the reference (2).
    let bad_df = synthetic_b(3, 2);
    let r = df_ao2mo(&refr, &Frozen::None, &bad_df);
    assert!(r.is_err(), "nao mismatch must error, not panic");
}

/// ALWAYS-ON structural: `cholesky_eri` against a REAL built Mole propagates
/// the `int3c2e_sph` cintx#11 gate — it must return `Err` (or an all-zero
/// B-tensor that the DF-MP2 numeric oracle is gated on), never panic. The
/// numeric DF-MP2 energy is cintx-gated; the error-propagation SHAPE is
/// always-on.
///
/// Per `cholesky_eri.rs:104-108`, `int3c2e_sph` currently returns a zero-filled
/// buffer of the correct shape (the cintx-ops gap), so `cholesky_eri` may
/// either `Err` (singular all-zero `(P|Q)` Cholesky) or return zeros. Either is
/// the gated SHAPE; the assertion is that the call NEVER panics and the result
/// is a well-formed `Result`.
#[test]
fn cholesky_eri_propagates_int3c2e_gate_no_panic() {
    use pyscf_gto::{AtomInput, BasisInput, M, MoleBuildArgs};

    let mol = M(MoleBuildArgs {
        atom: AtomInput::String("H 0 0 0; H 0 0 0.74".into()),
        basis: BasisInput::Name("sto-3g".into()),
        ..Default::default()
    })
    .expect("build H2");

    // default_ri (mp2fit aux) for sto-3g resolves to the weigend universal aux.
    let aux = pyscf_df::default_ri("sto-3g");
    // The call must NOT panic. With the int3c2e_sph cintx#11 gap the (P|Q) is
    // all-zero → SingularAux Err is the expected gated shape; if cintx lands a
    // real int2c2e the call may succeed with an all-zero b_uvq. Either way the
    // result is a well-formed Result and the process does not abort.
    let res: Result<DfIntegrals, _> = cholesky_eri(&mol, aux);
    match res {
        Ok(df) => {
            // Gated all-zero path: a well-formed but numerically-empty tensor.
            assert_eq!(df.nao, mol.nao_nr);
            assert_eq!(df.b_uvq.len(), df.nao * df.nao * df.naux);
        }
        Err(_) => {
            // SingularAux (all-zero (P|Q)) — the expected cintx#11-gated shape.
        }
    }
}
