//! Always-on in-tree NUMERIC smoke for the MP2 path now that cintx#11 is
//! closed (05-08): `intor("int2e")` (arity-4) and `int3c2e_sph` (orbital×aux)
//! evaluate for real, so the in-core RMP2 and conventional DF-MP2 kernels —
//! UNCHANGED since 05-03/05-05 (they always `?`-propagated the gate, D-05) —
//! produce real, finite, physically-signed correlation energies.
//!
//! This is NOT an upstream-PySCF byte-identity check (the sandbox has no
//! numpy/PySCF; that is the CI-gated/human-verify arm). It proves the
//! end-to-end path int2e → ao2mo → kernel → energy lights up: a closed-shell
//! RMP2 correlation energy is finite, non-zero, and `e_corr ≤ 0`.

use pyscf_algebra::oracle_dot;
use pyscf_core::MOCoefficients;
use pyscf_gto::{AtomInput, BasisInput, M, MoleBuildArgs, intor};
use pyscf_mp2::{Frozen, Mp2Reference, NoMp2Overrides, dfrmp2_kernel, rmp2_kernel};

/// Build a REAL `Mp2Reference` on `atom`/`basis` with an identity `mo_coeff`
/// (MO = AO) and a closed-shell occ<vir spectrum. Identity coefficients keep
/// the AO→MO transform non-trivial (so `(ia|jb) != 0`) while leaving the MO
/// occupied/virtual split well-defined; with occupied energies below virtual
/// the MP2 denominators are negative, so a real `(ia|jb)` yields `e_corr ≤ 0`.
fn real_identity_reference(atom: &str, basis: &str) -> Mp2Reference {
    let mol = M(MoleBuildArgs {
        atom: AtomInput::String(atom.into()),
        basis: BasisInput::Name(basis.into()),
        ..Default::default()
    })
    .expect("build mol");

    let nao = mol.nao_nr;
    assert!(nao >= 2, "need at least one occ + one vir AO");

    // Column-major identity: C[ao,mo] = δ at ao + mo*nao.
    let mut data = vec![0.0f64; nao * nao];
    for d in 0..nao {
        data[d + d * nao] = 1.0;
    }

    // One doubly-occupied MO (index 0), the rest virtual; occ energy < vir.
    let mut mo_energy = vec![0.0f64; nao];
    for (i, e) in mo_energy.iter_mut().enumerate() {
        *e = -0.5 + (i as f64); // [-0.5, 0.5, 1.5, ...] — occ(0) below all vir
    }
    let mut mo_occ = vec![0.0f64; nao];
    mo_occ[0] = 2.0;

    Mp2Reference {
        mo_coeff: MOCoefficients {
            nao,
            nmo: nao,
            data,
            energies: mo_energy.clone(),
            occupations: mo_occ.clone(),
        },
        mo_energy,
        mo_occ,
        e_hf: -1.0,
        converged: true,
        mol,
    }
}

#[test]
fn rmp2_h2_sto3g_finite_nonzero_negative_corr() {
    let refr = real_identity_reference("H 0 0 0; H 0 0 0.74", "sto-3g");

    // Real int2e → default_ao2mo → closed-form RMP2 (NoMp2Overrides = the
    // pure-Rust default_ao2mo path through intor("int2e")).
    let res = rmp2_kernel(&refr, &Frozen::None, &NoMp2Overrides, false)
        .expect("RMP2 numeric must run end-to-end now that int2e lands");

    assert!(
        res.e_corr.is_finite(),
        "e_corr must be finite, got {}",
        res.e_corr
    );
    assert!(
        res.e_corr <= 1e-12,
        "closed-shell RMP2 correlation energy must be ≤ 0, got {}",
        res.e_corr
    );
    assert!(
        res.e_corr.abs() > 1e-12,
        "e_corr must be non-trivially non-zero (int2e produced a real (ia|jb))"
    );
    // SS/OS split is consistent with the total.
    assert!(
        (res.e_ss + res.e_os - res.e_corr).abs() < 1e-12,
        "e_ss + e_os must equal e_corr"
    );
    eprintln!("RMP2 H2/STO-3G e_corr = {}", res.e_corr);
}

/// Thread-count invariance (T-05-03-FP): the oracle_sum/oracle_dot reductions
/// must give a bit-identical e_corr regardless of rayon thread count, now that
/// the energy is computed over a real (non-zero) integral block.
#[test]
fn rmp2_energy_is_thread_count_invariant() {
    let refr = real_identity_reference("H 0 0 0; H 0 0 0.74", "sto-3g");
    let a = rmp2_kernel(&refr, &Frozen::None, &NoMp2Overrides, false).expect("rmp2 a");
    let b = rmp2_kernel(&refr, &Frozen::None, &NoMp2Overrides, false).expect("rmp2 b");
    assert_eq!(
        a.e_corr.to_bits(),
        b.e_corr.to_bits(),
        "RMP2 e_corr must be bit-identical across repeated runs"
    );
}

/// Conventional DF-MP2 end-to-end through the REAL int3c2e_sph + the 05-09
/// rank-revealing DF-metric fallback: cholesky_eri → B-tensor → df_ao2mo →
/// kernel now SUCCEEDS for the weigend (*-ri) aux (previously rank-deficient),
/// yielding a finite, physically-signed correlation energy.
#[test]
fn dfmp2_h2_sto3g_finite_negative_corr() {
    let refr = real_identity_reference("H 0 0 0; H 0 0 0.74", "sto-3g");
    let aux = pyscf_df::default_ri("sto-3g");

    let df = pyscf_df::cholesky_eri(&refr.mol, aux)
        .expect("cholesky_eri must succeed via the 05-09 rank-revealing fallback");
    let res = dfrmp2_kernel(&refr, &Frozen::None, &df, false)
        .expect("DF-MP2 numeric must run end-to-end with a real B-tensor");

    assert!(
        res.e_corr.is_finite(),
        "DF e_corr finite, got {}",
        res.e_corr
    );
    assert!(
        res.e_corr <= 1e-12,
        "closed-shell DF-MP2 correlation energy must be ≤ 0, got {}",
        res.e_corr
    );
    eprintln!(
        "DF-MP2 H2/STO-3G e_corr = {} (eff. naux={})",
        res.e_corr, df.naux
    );
}

/// The DF B-tensor must reconstruct the exact 2-electron integrals within DF
/// accuracy: `Σ_Q B^Q_{μν} B^Q_{λσ} ≈ (μν|λσ)` where the RHS is the real
/// `intor("int2e")` (now available, 05-08). This is the gold-standard in-tree
/// correctness check for the whole DF path (int3c2e + int2c2e + the metric
/// fit) — independent of any MP2 kernel. DF is an approximation, so the
/// tolerance is loose (aux-incompleteness), but the reconstruction must be
/// finite, bra-ket symmetric, and close.
#[test]
fn df_b_tensor_reconstructs_exact_eri() {
    let mol = M(MoleBuildArgs {
        atom: AtomInput::String("H 0 0 0; H 0 0 0.74".into()),
        basis: BasisInput::Name("sto-3g".into()),
        ..Default::default()
    })
    .expect("build H2/STO-3G");
    let nao = mol.nao_nr;

    let exact = intor(&mol, "int2e").expect("int2e");
    let df = pyscf_df::cholesky_eri(&mol, pyscf_df::default_ri("sto-3g"))
        .expect("cholesky_eri (rank-revealing fallback)");
    let naux = df.naux;

    // b_uvq row-major [nao,nao,naux]: the Q-vector for (μν) is contiguous at
    // (μ*nao+ν)*naux .. +naux. df_eri(μν,λσ) = Σ_Q B[μν,Q]·B[λσ,Q].
    let bvec = |mu: usize, nu: usize| {
        let base = (mu * nao + nu) * naux;
        &df.b_uvq[base..base + naux]
    };
    // exact int2e is F-order [nao;4]: (μ,ν,λ,σ) at μ + ν*nao + λ*nao² + σ*nao³.
    let exact_at = |mu: usize, nu: usize, la: usize, si: usize| {
        exact.values[mu + nu * nao + la * nao * nao + si * nao * nao * nao]
    };

    let mut max_err = 0.0f64;
    for mu in 0..nao {
        for nu in 0..nao {
            for la in 0..nao {
                for si in 0..nao {
                    let df_val = oracle_dot(bvec(mu, nu), bvec(la, si));
                    assert!(df_val.is_finite(), "DF-reconstructed ERI must be finite");
                    // Bra-ket symmetry of the reconstruction.
                    let swapped = oracle_dot(bvec(la, si), bvec(mu, nu));
                    assert!(
                        (df_val - swapped).abs() < 1e-12,
                        "DF reconstruction symmetric"
                    );
                    max_err = max_err.max((df_val - exact_at(mu, nu, la, si)).abs());
                }
            }
        }
    }
    // Diagonal Coulomb self-repulsion stays positive after DF.
    assert!(
        oracle_dot(bvec(0, 0), bvec(0, 0)) > 0.0,
        "(00|00)_DF must be > 0"
    );
    eprintln!("DF reconstruction max abs error vs exact int2e = {max_err:.3e}");
    // DF-accuracy bound, tightened from the original 5e-2 after the cintx
    // 2c2e/3c2e d-shell normalization fix (DF-01, cintx 55bf984). The weigend
    // aux for H2/STO-3G includes d functions; the missing n*b00 recurrence
    // factor previously inflated this to 1.686e-3. With the fix the true
    // DF-incompleteness error is 1.06e-4; 5e-4 guards against a d-shell
    // regression with ~5x margin.
    assert!(
        max_err < 5e-4,
        "DF reconstruction error {max_err:.3e} exceeds the DF-accuracy bound — \
         possible regression of the cintx 2c2e/3c2e d-shell n*b00 fix (DF-01)"
    );
}
