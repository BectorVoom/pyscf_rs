//! **Open-shell UCCSD Λ byte-fixture (F-07 regression — PySCF-FREE).**
//!
//! The in-tree α==β collapse + the Λ fixed-point self-consistency invariant are
//! both NECESSARY but NOT SUFFICIENT: a transposed-index l-equation term (the two
//! F-07 bugs — `wvvvo`'s `kbad,jkcd` contraction and the l2 `tmp_c` `ovvv` a<->b
//! swap) is masked when α==β AND survives a self-consistency check (a buggy
//! equation has its OWN consistent fixed point). The ONLY sufficient witness is a
//! comparison to an INDEPENDENT correct reference.
//!
//! This test bakes that reference WITHOUT a live-PySCF dependency at test time:
//! the converged UHF MO arrays for the genuine open-shell OH doublet (spin=1,
//! STO-3G, nocc_α=5 ≠ nocc_β=4) are stored as a fixture (dumped once from PySCF
//! 2.12.1), and the assertion is on the converged spin-orbital Λ Frobenius norms
//! `‖l1‖²` and `‖l2‖²`. The Frobenius norm is INVARIANT under the occ-internal
//! and vir-internal permutation that distinguishes the rs blocked spin-orbital
//! ordering ([α-occ,β-occ,α-vir,β-vir]) from PySCF GCCSD's energy-sorted
//! ordering — so the reference value is ordering-independent and the bug-bearing
//! λ (which has a different norm) is caught. Reference norms from upstream GCCSD
//! (`gcc.solve_lambda()`, conv_tol_normt=1e-9): ‖l1‖²=1.882825202e-4,
//! ‖l2‖²=6.023989549e-2.
//!
//! Per `docs/rust_crate_test_guideline.md`: spec source = upstream PySCF 2.12.1;
//! verified scope = the open-shell spin-orbital Λ equation on OH/STO-3G; the
//! tolerance (1e-6 relative on the norms) + molecule are explicit.

use pyscf_ccsd::{CcsdReference, UccsdReference, solve_ulambda, uccsd_kernel};
use pyscf_core::MOCoefficients;
use pyscf_gto::{AtomInput, BasisInput, M, MoleBuildArgs};
use pyscf_mp2::Frozen;
use pyscf_runtime::WorkspacePool;

const NAO: usize = 6;
const NMO: usize = 6;

// Upstream UHF mo_coeff for OH/STO-3G spin=1, ROW-MAJOR [2,nao,nmo] (α then β).
#[rustfmt::skip]
const MO_COEFF_ROWMAJOR: [f64; 72] = [
    9.94283124751723646e-01, -2.47225183494367784e-01, 8.83820317790486865e-02, -3.21329469152460844e-16, 1.38354199038279903e-16, 8.85958897879606522e-02,
    2.46658819231369945e-02, 9.24052400948521924e-01, -4.60233253649214247e-01, 1.63417662278733083e-15, -7.43949258498867240e-16, -6.01433576604227360e-01,
    -2.36885176364703725e-19, 4.49611522107974693e-17, 3.52763437323189932e-15, 9.96999830232707551e-01, 7.74037370929354235e-02, 2.06154199562834903e-18,
    2.43812280531721951e-19, 1.71358104292423692e-18, -1.67389169952414112e-15, -7.74037370929353263e-02, 9.96999830232707329e-01, -1.36388508891171003e-16,
    3.78965861013546039e-03, 1.11452054632765987e-01, 6.99676194720547318e-01, -2.64998255934756870e-15, 7.61588544116635907e-16, -8.59708827505687112e-01,
    -6.74546771083781894e-03, 1.65988703351360956e-01, 4.98199012239988470e-01, -1.63055915425673404e-15, 8.86281415931800843e-16, 1.14789324991799124e+00,
    9.94837977793708972e-01, -2.35179367392098437e-01, 1.06384910605826485e-01, 1.02690086415258811e-16, -2.29107742551744981e-17, 9.51322901876878041e-02,
    2.23661440007633695e-02, 8.65413683537515244e-01, -5.39206811092270422e-01, -5.57792462985422982e-16, 8.54302584131853644e-17, -6.22762528642772484e-01,
    -2.56873543918472098e-19, -6.77866109346809507e-17, -5.14426243541123124e-17, 7.74037370929332724e-02, 9.96999830232707662e-01, 2.24950682949579780e-16,
    1.95053768132526978e-19, -2.79363521174529369e-17, -7.98388485031950069e-16, 9.96999830232707551e-01, -7.74037370929331198e-02, -1.96261322494459871e-16,
    3.50858793865961886e-03, 1.15962601016014674e-01, 6.60663621225666575e-01, 3.74227272927722026e-16, 3.44201268254848916e-17, -8.89463834064849523e-01,
    -6.15697039433366674e-03, 2.48486816066951027e-01, 5.26308825233173794e-01, 6.74980040600600993e-16, 2.54149344838516178e-17, 1.12012246402669846e+00,
];

#[rustfmt::skip]
const MO_ENERGY: [f64; 12] = [
    -2.02857109315115594e+01, -1.29218677429439976e+00, -5.50984034040802872e-01, -5.24556886036488312e-01, -4.29667995799722202e-01, 6.20054964554638821e-01,
    -2.02572357748975591e+01, -1.12820018702577896e+00, -5.02869893958302949e-01, -3.77793676740949325e-01, 3.60032081278943483e-01, 6.55760698011273790e-01,
];

#[rustfmt::skip]
const MO_OCC: [f64; 12] = [
    1.0, 1.0, 1.0, 1.0, 1.0, 0.0,
    1.0, 1.0, 1.0, 1.0, 0.0, 0.0,
];

const E_TOT: f64 = -74.36266919476726;

// Upstream GCCSD converged Λ Frobenius norms (permutation-invariant references).
const L1_SUMSQ_REF: f64 = 1.882825202235e-4;
const L2_SUMSQ_REF: f64 = 6.023989548843e-2;

/// Build a per-spin `CcsdReference` from the fixture, channel `spin` (0=α, 1=β).
/// The stored array is ROW-MAJOR `[nao,nmo]`; `MOCoefficients.data` is COLUMN-
/// major (`C[ao,mo]` at `ao + mo*nao`), so we transpose on the way in.
fn channel(spin: usize, mol: &pyscf_core::Mole) -> CcsdReference {
    let base = spin * NAO * NMO;
    let mut data = vec![0.0_f64; NAO * NMO];
    for ao in 0..NAO {
        for mo in 0..NMO {
            // row-major source (ao, mo) → ao*NMO + mo; col-major dest ao + mo*NAO.
            data[ao + mo * NAO] = MO_COEFF_ROWMAJOR[base + ao * NMO + mo];
        }
    }
    let eoff = spin * NMO;
    CcsdReference {
        mo_coeff: MOCoefficients {
            nao: NAO,
            nmo: NMO,
            data,
            energies: MO_ENERGY[eoff..eoff + NMO].to_vec(),
            occupations: MO_OCC[eoff..eoff + NMO].to_vec(),
        },
        mo_energy: MO_ENERGY[eoff..eoff + NMO].to_vec(),
        mo_occ: MO_OCC[eoff..eoff + NMO].to_vec(),
        e_hf: E_TOT,
        converged: true,
        mol: mol.clone(),
    }
}

#[test]
fn uccsd_open_shell_lambda_matches_upstream_oh_fixture() {
    let mol = M(MoleBuildArgs {
        atom: AtomInput::String("O 0 0 0; H 0 0 0.97".into()),
        basis: BasisInput::Name("sto-3g".into()),
        spin: 1,
        ..Default::default()
    })
    .expect("build OH mol");

    let uref = UccsdReference {
        alpha: channel(0, &mol),
        beta: channel(1, &mol),
    };

    let pool = WorkspacePool::new(WorkspacePool::DEFAULT_BUDGET_BYTES);
    let res = uccsd_kernel(&uref, &Frozen::None, &pool).expect("OH UCCSD must run");
    assert!(res.converged, "OH UCCSD must converge");

    let lam = solve_ulambda(&res.so_t1, &res.so_t2, &res.so_eris, &pool).expect("solve_ulambda");
    assert!(lam.converged, "OH spin-orbital Λ must converge");

    // Frobenius norms are invariant under the occ/vir-internal ordering
    // difference between the rs blocked and PySCF energy-sorted spin-orbital
    // layouts — so they compare directly to the upstream references.
    let l1_sumsq: f64 = lam.l1.iter().map(|x| x * x).sum();
    let l2_sumsq: f64 = lam.l2.iter().map(|x| x * x).sum();

    eprintln!(
        "OH/STO-3G UCCSD Λ: ‖l1‖²={l1_sumsq:.12e} (ref {L1_SUMSQ_REF:.12e}), \
         ‖l2‖²={l2_sumsq:.12e} (ref {L2_SUMSQ_REF:.12e})"
    );

    let rel1 = (l1_sumsq - L1_SUMSQ_REF).abs() / L1_SUMSQ_REF;
    let rel2 = (l2_sumsq - L2_SUMSQ_REF).abs() / L2_SUMSQ_REF;
    // 1e-6 relative: tight enough that either F-07 transposed-index bug (which
    // moved ‖l2‖² by ~O(1%)) fails, loose enough to absorb the rs no-DIIS
    // amplitude convergence tail vs PySCF's DIIS-converged reference.
    assert!(
        rel1 < 1e-6,
        "‖l1‖² must match upstream OH fixture (rel {rel1:.3e}): {l1_sumsq:.12e} vs {L1_SUMSQ_REF:.12e}"
    );
    assert!(
        rel2 < 1e-6,
        "‖l2‖² must match upstream OH fixture (rel {rel2:.3e}): {l2_sumsq:.12e} vs {L2_SUMSQ_REF:.12e}"
    );
}
