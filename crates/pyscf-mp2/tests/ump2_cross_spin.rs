//! Cross-spin `(o_α v_α | o_β v_β)` ao2mo acceptance scaffold for F-06.
//!
//! Two always-on, no-PySCF tests pin the `cross_spin_ao2mo` contract:
//!
//! - **Test A (F→C repack, layout-pinning):** a tiny REAL `int2e` case with
//!   deliberately NON-palindromic shapes (`nocc_a=1, nvir_a=2` vs
//!   `nocc_b=2, nvir_b=1`) so the αα palindromic `(ia|jb)==(jb|ia)` symmetry
//!   CANNOT mask a transpose bug. It asserts the returned C-order `ovov`
//!   reproduces, element-for-element, the F-order `ao2mo::general` output under
//!   the documented index map. This pins the repack independent of physics.
//!
//! - **Test B (closed-shell guard, NECESSARY-not-sufficient):** feed a
//!   closed-shell reference as an unrestricted pair (α==β); the `ump2_kernel`
//!   total using `cross_spin_ao2mo` for the αβ block must reproduce the
//!   `rmp2_kernel` `e_corr` to <1e-10.
//!
//!   ⚠ This guard does NOT certify the cross-spin layout: a
//!   symmetric-equivalent transpose bug is invisible when α==β (the F↔C swap is
//!   absorbed by the palindromic shape). The REAL acceptance gate is the
//!   open-shell live-PySCF byte-identity oracle
//!   (`crates/pyscf-py/tests/ump2_open_shell_oracle.py`, plan Task 3).
//!
//! Per `docs/rust_crate_test_guideline.md`: these are integration tests against
//! the public crate surface (`cross_spin_ao2mo`, `ump2_kernel`, `rmp2_kernel`),
//! with an explicit specification source (the `<interfaces>` index contracts)
//! and an in-test statement separating verified scope (the F→C re-stride and
//! the closed-shell invariant) from unverified scope (the open-shell layout,
//! deferred to the live oracle).

use pyscf_core::MOCoefficients;
use pyscf_gto::{AtomInput, BasisInput, M, MoleBuildArgs};
use pyscf_mp2::{
    ChemistsEris, Frozen, Mp2Reference, NoMp2Overrides, UmpReference, cross_spin_ao2mo,
    default_ao2mo, rmp2_kernel, ump2_kernel,
};

/// Build a REAL `Mp2Reference` on `atom`/`basis` with an identity `mo_coeff`
/// (MO = AO) and the supplied occupied/virtual spectrum. Identity coefficients
/// keep the AO→MO transform non-trivial (so `(ia|JB) != 0`) while leaving the
/// occupied/virtual split well-defined. `mo_occ` selects which columns are
/// occupied (occ > 0); the energies place every occupied below every virtual so
/// the MP2 denominators stay negative.
fn real_reference(atom: &str, basis: &str, occ_cols: &[usize]) -> Mp2Reference {
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

    // Occupied columns get occ=2.0 and energies below every virtual.
    let mut mo_occ = vec![0.0f64; nao];
    for &c in occ_cols {
        mo_occ[c] = 2.0;
    }
    // Occupied energies in [-1.0, -0.5, ...]; virtual energies in [0.5, 1.5, ...].
    let mut mo_energy = vec![0.0f64; nao];
    let mut occ_e = -1.0f64;
    let mut vir_e = 0.5f64;
    for (col, e) in mo_energy.iter_mut().enumerate() {
        if mo_occ[col] > 0.0 {
            *e = occ_e;
            occ_e += 0.1;
        } else {
            *e = vir_e;
            vir_e += 1.0;
        }
    }

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

/// Test A — the F-order → C-order repack is exactly the documented re-stride,
/// pinned with NON-palindromic shapes so an accidental same-axis transpose bug
/// (or skipping the repack entirely) is detectable.
///
/// H₃/STO-3G gives nao=3. α has 1 occupied column (→ nocc_a=1, nvir_a=2); β has
/// 2 occupied columns (→ nocc_b=2, nvir_b=1). nocc_a≠nocc_b AND nvir_a≠nvir_b,
/// so the αα palindrome CANNOT hide a layout error.
#[test]
fn test_a_f_to_c_repack_pins_layout() {
    // 3-AO molecule (H₃ linear, STO-3G).
    let alpha = real_reference("H 0 0 0; H 0 0 1.5; H 0 0 3.0", "sto-3g", &[0]);
    let beta = real_reference("H 0 0 0; H 0 0 1.5; H 0 0 3.0", "sto-3g", &[0, 1]);

    let nao = alpha.mol.nao_nr;
    assert_eq!(nao, 3, "H₃/STO-3G must give nao=3 for the 1/2-occ split");

    // Active counts: α = 1 occ / 2 vir; β = 2 occ / 1 vir (non-palindromic).
    let nocc_a = 1usize;
    let nvir_a = 2usize;
    let nocc_b = 2usize;
    let nvir_b = 1usize;

    // The wrapper's C-order αβ block.
    let eris_ab = cross_spin_ao2mo(&alpha, &beta, &Frozen::None)
        .expect("cross_spin_ao2mo must run end-to-end");
    assert_eq!(eris_ab.nocc, nocc_a, "ChemistsEris.nocc is the α side");
    assert_eq!(eris_ab.nvir, nvir_a, "ChemistsEris.nvir is the α side");
    assert_eq!(
        eris_ab.ovov.len(),
        nocc_a * nvir_a * nocc_b * nvir_b,
        "αβ block length must be nocc_a*nvir_a*nocc_b*nvir_b"
    );

    // Independently recompute the F-order ao2mo::general output with the SAME
    // column subsets the wrapper builds (1 occ / 2 vir for α, 2 occ / 1 vir for
    // β), then assert the documented F→C re-stride holds element-for-element.
    let mk_subset = |refr: &Mp2Reference, want_occ: bool| -> MOCoefficients {
        let mut cols = Vec::new();
        for (col, &occ) in refr.mo_occ.iter().enumerate() {
            if (occ > 0.0) == want_occ {
                cols.push(col);
            }
        }
        let mut data = Vec::new();
        let mut energies = Vec::new();
        let mut occupations = Vec::new();
        for &c in &cols {
            let start = c * nao;
            data.extend_from_slice(&refr.mo_coeff.data[start..start + nao]);
            energies.push(refr.mo_energy[c]);
            occupations.push(refr.mo_occ[c]);
        }
        MOCoefficients {
            nao,
            nmo: cols.len(),
            data,
            energies,
            occupations,
        }
    };
    let co_a = mk_subset(&alpha, true);
    let cv_a = mk_subset(&alpha, false);
    let co_b = mk_subset(&beta, true);
    let cv_b = mk_subset(&beta, false);
    assert_eq!((co_a.nmo, cv_a.nmo), (nocc_a, nvir_a));
    assert_eq!((co_b.nmo, cv_b.nmo), (nocc_b, nvir_b));

    let eri = pyscf_gto::intor(&alpha.mol, "int2e").expect("int2e");
    let f_order = pyscf_ao2mo::general(&eri.values, nao, [&co_a, &cv_a, &co_b, &cv_b])
        .expect("ao2mo::general");

    // For EVERY (i,a,J,B): C-order ovov == F-order general under the re-stride.
    for i in 0..nocc_a {
        for a in 0..nvir_a {
            for jj in 0..nocc_b {
                for bb in 0..nvir_b {
                    let c_idx = ((i * nvir_a + a) * nocc_b + jj) * nvir_b + bb;
                    let f_idx =
                        i + a * nocc_a + jj * nocc_a * nvir_a + bb * nocc_a * nvir_a * nocc_b;
                    assert_eq!(
                        eris_ab.ovov[c_idx].to_bits(),
                        f_order[f_idx].to_bits(),
                        "F→C repack mismatch at (i={i},a={a},J={jj},B={bb})"
                    );
                }
            }
        }
    }

    // Sanity: the block is non-trivially non-zero (a real int2e drove it), so
    // the assertions above are not vacuously comparing all-zeros.
    assert!(
        eris_ab.ovov.iter().any(|&v| v.abs() > 1e-12),
        "the αβ block must be non-trivially non-zero"
    );
}

/// Test B — closed-shell guard (NECESSARY, NOT sufficient).
///
/// Feed a closed-shell reference as an unrestricted pair (α==β). The UMP2 total
/// using `cross_spin_ao2mo` for the αβ block must reproduce the RMP2 `e_corr`
/// to <1e-10.
///
/// ⚠ A symmetric-equivalent transpose bug is INVISIBLE here (when α==β the F↔C
/// swap is absorbed by the palindromic shape). The REAL acceptance gate for the
/// cross-spin layout is the open-shell live-PySCF byte-identity oracle
/// (`crates/pyscf-py/tests/ump2_open_shell_oracle.py`). This test only proves
/// the wrapper is wired into the kernel and degenerates correctly.
#[test]
fn test_b_closed_shell_ump2_equals_rmp2() {
    // Closed-shell H₂/STO-3G: 1 doubly-occupied MO, 1 virtual.
    let refr = real_reference("H 0 0 0; H 0 0 0.74", "sto-3g", &[0]);

    // RMP2 reference value (the same-spin + opposite-spin closed-form).
    let rmp2 = rmp2_kernel(&refr, &Frozen::None, &NoMp2Overrides, false).expect("rmp2");

    // UMP2 with α==β. The αα/ββ same-spin blocks come from default_ao2mo (via
    // the kernel's own consumption); here we assemble the three blocks exactly
    // as the PyO3 bridge does: aa/bb from the per-spin in-core ao2mo, ab from
    // cross_spin_ao2mo.
    let ump_ref = UmpReference {
        alpha: refr.clone(),
        beta: refr.clone(),
    };
    let eris_aa =
        pyscf_mp2::default_ao2mo(&ump_ref.alpha, &Frozen::None).expect("aa default_ao2mo");
    let eris_bb = pyscf_mp2::default_ao2mo(&ump_ref.beta, &Frozen::None).expect("bb default_ao2mo");
    let eris_ab: ChemistsEris =
        cross_spin_ao2mo(&ump_ref.alpha, &ump_ref.beta, &Frozen::None).expect("ab cross_spin");

    let ump2 =
        ump2_kernel(&ump_ref, &Frozen::None, &eris_aa, &eris_ab, &eris_bb, false).expect("ump2");

    // For a closed-shell reference treated as α==β UHF, the UMP2 correlation
    // energy equals the RMP2 correlation energy (the αα+ββ same-spin pieces sum
    // to the RMP2 same-spin term, and the αβ piece to the RMP2 opposite-spin
    // term). This is the standard closed-shell degeneracy.
    assert!(
        rmp2.e_corr.is_finite() && ump2.e_corr.is_finite(),
        "both energies must be finite"
    );
    assert!(
        (ump2.e_corr - rmp2.e_corr).abs() < 1e-10,
        "closed-shell UMP2 e_corr ({}) must match RMP2 e_corr ({}) to <1e-10 \
         (necessary, NOT sufficient — open-shell oracle is the real gate)",
        ump2.e_corr,
        rmp2.e_corr
    );
}

/// Test C — `default_ao2mo` (the restricted / αα / ββ same-spin block) must obey
/// the bra-ket particle-exchange symmetry `(ia|jb) == (jb|ia)` when indexed
/// through `rmp2_kernel`'s C-order `idx`, on a case with `nvir > 1`.
///
/// REGRESSION GUARD (260601-nfb): `default_ao2mo` previously returned
/// `ao2mo::general`'s raw **F-order** `[nocc,nvir,nocc,nvir]` buffer while
/// `rmp2_kernel` reads it **C-order** (`((i*nvir+a)*nocc+j)*nvir+b`). The two
/// layouts coincide ONLY when `nvir == 1` (the F↔C difference collapses to an
/// i↔j swap that `(ia|ja)==(ja|ia)` absorbs) — so the long-standing H₂/STO-3G
/// smoke test (`nvir==1`) passed while restricted RMP2 was silently wrong by
/// ~mHa for every polyatomic (`nvir>1`: H₂O ~9.7 mHa, NH₃ ~10 mHa). This test
/// uses `nvir==2` so the raw F-order layout VIOLATES the physical
/// `(ia|jb)==(jb|ia)` symmetry and is caught oracle-free. The fix adds the
/// explicit F→C repack to `default_ao2mo` (mirroring `cross_spin_ao2mo`).
#[test]
fn test_c_default_ao2mo_obeys_bra_ket_symmetry_nvir_gt_1() {
    // H₄ linear / STO-3G → nao=4; 2 occupied columns → nocc=2, nvir=2 (nvir>1).
    let refr = real_reference(
        "H 0 0 0; H 0 0 1.0; H 0 0 2.0; H 0 0 3.0",
        "sto-3g",
        &[0, 1],
    );
    let nao = refr.mol.nao_nr;
    assert_eq!(nao, 4, "H₄/STO-3G must give nao=4");

    let eris = default_ao2mo(&refr, &Frozen::None).expect("default_ao2mo must run");
    let (nocc, nvir) = (eris.nocc, eris.nvir);
    assert_eq!((nocc, nvir), (2, 2), "expected nocc=2, nvir=2 (nvir>1)");

    // rmp2_kernel's flat C-order index into ovov.
    let idx = |i: usize, a: usize, j: usize, b: usize| ((i * nvir + a) * nocc + j) * nvir + b;

    // Physical invariant for real ERIs: (ia|jb) == (jb|ia). Indexed through the
    // kernel's C-order map, ovov[idx(i,a,j,b)] must equal ovov[idx(j,b,i,a)].
    // The pre-fix raw-F-order buffer violates this for nvir>1.
    let mut max_asym = 0.0f64;
    for i in 0..nocc {
        for a in 0..nvir {
            for j in 0..nocc {
                for b in 0..nvir {
                    let lhs = eris.ovov[idx(i, a, j, b)];
                    let rhs = eris.ovov[idx(j, b, i, a)];
                    max_asym = max_asym.max((lhs - rhs).abs());
                }
            }
        }
    }
    assert!(
        eris.ovov.iter().any(|&v| v.abs() > 1e-12),
        "the (ia|jb) block must be non-trivially non-zero (else the symmetry \
         check is vacuous)"
    );
    assert!(
        max_asym < 1e-12,
        "default_ao2mo must obey (ia|jb)==(jb|ia) through the kernel's C-order \
         index; max bra-ket asymmetry {max_asym:.3e} indicates the raw F-order \
         layout bug (missing F→C repack) for nvir>1"
    );
}
