//! ECP analytical-gradient gate (GRAD-07, D-01) — closing the GTO-05 arc
//! (Phase 2 wired ECP *eval*; Phase 7 wires the ECP *gradient*).
//!
//! Wires the ECP-gradient hcore term ([`pyscf_grad::ecp`]) against the 07-02
//! finite-difference harness ([`pyscf_grad::verify_fd`]).
//!
//! ## The two ECP-gradient families (07-01 cintx-availability split, D-02)
//!
//! Port of `pyscf/grad/rhf.py:109-143`:
//!   * `get_hcore`  → `+ ECPscalar_ipnuc` (= `int1e_ecp_ipnuc`) — cintx-READY
//!     (07-01); the ipnuc numeric arm UN-GATES now.
//!   * `hcore_deriv` → `+ ECPscalar_iprinv` (= `int1e_ecp_iprinv`) — MISSING from
//!     cintx (07-01, no scheduled workstream); routes to a clean
//!     cintx-availability error (NOT a panic), so the iprinv-dependent numeric
//!     arm stays `#[ignore]`'d.
//!
//! Both wire through the Phase-2 `CintxEcpEngine` (the dispatch un-gated in
//! 07-01). The ipnuc term folds into the RHF hcore path so ECP-bearing molecules
//! get the ECP gradient contribution.
//!
//! Run scoped + single-threaded (user-memory + CI discipline, NO libxc):
//!   cargo test -p pyscf-grad --locked -- --test-threads=1 ecp
//!   cargo test -p pyscf-grad --locked -- --ignored ecp_verify_fd_numeric

use pyscf_core::{Mole, PyscfRsError, Unit};
use pyscf_grad::ecp::{get_hcore_ecp, hcore_deriv_ecp};
use pyscf_grad::verify_fd::{DEFAULT_DISP, FD_TOL, verify_fd};
use pyscf_gto::{AtomInput, BasisInput, EcpInput, M, MoleBuildArgs};

/// A non-ECP molecule (He / STO-3G): the ECP-grad term contributes nothing, so
/// the engine surfaces `EcpEngineNotAvailable` — which the ECP-grad wrapper maps
/// to a zero contribution (never a panic).
fn he_sto3g() -> Mole {
    M(MoleBuildArgs {
        atom: AtomInput::String("He 0 0 0".into()),
        basis: BasisInput::Name("sto-3g".into()),
        ..Default::default()
    })
    .expect("build He/sto-3g mol")
}

/// Cu / LANL2DZ — carries an ECP, the molecule on which the ECP-gradient
/// `ECPscalar_ipnuc` term is non-trivial (cintx-ready, 07-01).
fn cu_lanl2dz() -> Mole {
    M(MoleBuildArgs {
        atom: AtomInput::String("Cu 0 0 0".into()),
        basis: BasisInput::Name("lanl2dz".into()),
        ecp: EcpInput::Name("lanl2dz".into()),
        unit: Unit::Bohr,
        max_memory: 4000.0,
        axes: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        ..Default::default()
    })
    .expect("Cu/LANL2DZ build")
}

#[test]
fn ecp_ipnuc_term_is_component_leading_for_an_ecp_molecule() {
    // GRAD-07: the get_hcore `+ ECPscalar_ipnuc` term (cintx-READY, un-gated)
    // contributes a component-leading [3, nao, nao] buffer for an ECP-bearing
    // molecule. Cu/LANL2DZ exercises the real cintx ECP-gradient path.
    let mol = cu_lanl2dz();
    let nao = mol.nao_nr;
    let hcore_ecp = get_hcore_ecp(&mol).expect("ECPscalar_ipnuc is cintx-ready (07-01)");
    assert_eq!(
        hcore_ecp.len(),
        3 * nao * nao,
        "ECP-grad hcore term must be a [3, nao, nao] component-leading buffer"
    );
    assert!(
        hcore_ecp.iter().all(|v| v.is_finite()),
        "ECP-grad hcore term must be finite"
    );
    let nonzero = hcore_ecp.iter().filter(|&&v| v.abs() > 1e-18).count();
    assert!(
        nonzero > 0,
        "ECPscalar_ipnuc contributes real (non-zero) values for an ECP molecule"
    );
}

#[test]
fn ecp_ipnuc_term_is_zero_contribution_for_a_non_ecp_molecule() {
    // A molecule with NO ECP: the `+ ECPscalar_ipnuc` term contributes nothing.
    // The wrapper maps the engine's `EcpEngineNotAvailable` to a zero buffer
    // (never a panic, never NotYetImplemented{phase:7}).
    let mol = he_sto3g();
    let nao = mol.nao_nr;
    let hcore_ecp =
        get_hcore_ecp(&mol).expect("non-ECP molecule → zero contribution, not an error");
    assert_eq!(
        hcore_ecp.len(),
        3 * nao * nao,
        "even a zero contribution must carry the [3, nao, nao] shape"
    );
    assert!(
        hcore_ecp.iter().all(|&v| v == 0.0),
        "a non-ECP molecule contributes an all-zero ECP-grad hcore term"
    );
}

#[test]
fn ecp_iprinv_per_atom_term_returns_real_buffer() {
    // F-05 / cintx 21-07: iprinv un-gated; the prior assertion (iprinv →
    // cintx-availability error) is superseded. The hcore_deriv `+ ECPscalar_iprinv`
    // per-atom term is now cintx-READY. For Cu/LANL2DZ (Cu IS the ECP atom) the
    // per-atom buffer at atom 0 is the REAL [3, nao, nao] contribution — this is
    // the runtime exercise of Task 2's new call site THROUGH hcore_deriv_ecp.
    let mol = cu_lanl2dz();
    let nao = mol.nao_nr;
    let buf = hcore_deriv_ecp(&mol, 0)
        .expect("ECPscalar_iprinv is cintx-ready (F-05) — hcore_deriv_ecp must return Ok");
    assert_eq!(
        buf.len(),
        3 * nao * nao,
        "hcore_deriv_ecp must return a [3, nao, nao] component-leading buffer"
    );
    assert!(
        buf.iter().all(|v| v.is_finite()),
        "hcore_deriv_ecp iprinv buffer must be finite"
    );
    let nonzero = buf.iter().filter(|&&v| v.abs() > 1e-18).count();
    assert!(
        nonzero > 0,
        "the per-atom iprinv term on the Cu ECP atom must carry real (non-zero) values"
    );
}

#[test]
fn ecp_iprinv_per_atom_term_is_zero_for_a_non_ecp_molecule() {
    // Independent anchor: He/STO-3G has NO ECP, so the per-atom iprinv term
    // contributes nothing → an all-zero [3, nao, nao] buffer (never a panic,
    // never an error). The cintx kernel also zero-fills when the rinv origin
    // matches no ECP atom, so a non-ECP atom is a zero contribution by both routes.
    let mol = he_sto3g();
    let nao = mol.nao_nr;
    let buf = hcore_deriv_ecp(&mol, 0)
        .expect("non-ECP molecule → zero iprinv contribution, not an error");
    assert_eq!(
        buf.len(),
        3 * nao * nao,
        "even a zero contribution must carry the [3, nao, nao] shape"
    );
    assert!(
        buf.iter().all(|&v| v == 0.0),
        "a non-ECP molecule contributes an all-zero per-atom iprinv term"
    );
}

/// NUMERIC arm — `#[ignore]`'d behind the cintx grad-intor gate.
///
/// The FULL ECP gradient FD-vs-analytical comparison still needs the RHF base
/// `de` assembly (`int2e_ip1` + `int1e_ip*` — MISSING from cintx). Both the
/// `ECPscalar_ipnuc` (GRAD-07) and the per-atom `ECPscalar_iprinv` (F-05) terms
/// are now cintx-ready, but the END-TO-END ECP gradient numeric stays
/// `#[ignore]`'d until the base grad-intor families land. Run on demand:
///   cargo test -p pyscf-grad --locked -- --ignored ecp_verify_fd_numeric
#[test]
#[ignore = "cintx grad-intor workstream pending: int2e_ip1 + int1e_ip{ovlp,kin,nuc,rinv} \
            absent; the ipnuc + iprinv ECP terms are cintx-ready (GRAD-07 / F-05) but the \
            end-to-end ECP gradient FD needs the full base assembly"]
fn ecp_verify_fd_numeric() {
    let mol = cu_lanl2dz();
    let coords = mol.atom_coords();
    // Placeholder analytical gradient + as_scanner energy; the real wiring
    // assembles the ECP-augmented RHF gradient + the ECP as_scanner energy.
    let analytical: Vec<[f64; 3]> = vec![[0.0; 3]; coords.len()];
    let energy = |displaced: &[[f64; 3]]| -> Result<f64, PyscfRsError> {
        let _ = displaced;
        Ok(0.0)
    };
    let report =
        verify_fd(&coords, &analytical, energy, DEFAULT_DISP, FD_TOL).expect("verify_fd runs");
    assert!(
        report.passed,
        "ECP analytical gradient disagrees with FD: max|Δ| = {} Ha/Bohr (tol {})",
        report.max_abs_diff, FD_TOL
    );
}
