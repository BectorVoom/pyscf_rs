//! ECP gradient hcore term (GRAD-07) — closing the GTO-05 arc (Phase 2 wired
//! ECP *eval*; Phase 7 wires the ECP *gradient*).
//!
//! Port target: `pyscf/grad/rhf.py:109-143` (the `get_hcore` + `hcore_deriv` ECP
//! branches). Structural analog: the ECP dispatch in
//! `crates/pyscf-gto/src/ecp_engine_cintx.rs` (the `ecp_int1e_ipnuc` method
//! un-gated in 07-01) + the RHF `get_hcore`/`hcore_deriv` (07-03) the ECP term
//! extends.
//!
//! ## The two ECP-gradient families (07-01 cintx-availability split, D-02)
//!
//!   * `get_hcore`  (`rhf.py:116-117`): `h1 += mol.intor('ECPscalar_ipnuc',
//!     comp=3)` — `ECPscalar_ipnuc` (= `int1e_ecp_ipnuc`) is cintx-READY (07-01).
//!     The ipnuc numeric arm UN-GATES now: [`get_hcore_ecp`] returns the real
//!     component-leading `[3, nao, nao]` contribution for an ECP-bearing
//!     molecule (and a zero buffer for a non-ECP molecule — never a panic).
//!   * `hcore_deriv` (`rhf.py:139-140`): `vrinv += mol.intor('ECPscalar_iprinv',
//!     comp=3)` — `ECPscalar_iprinv` (= `int1e_ecp_iprinv`) is MISSING from cintx
//!     (07-01, no scheduled workstream). [`hcore_deriv_ecp`] routes to a clean
//!     cintx-availability error (NEVER a panic, NEVER a silent zero — T-07-27).
//!
//! Both dispatch through the Phase-2 [`pyscf_gto::CintxEcpEngine`] (the
//! `ecp_int1e_ipnuc` method un-gated in 07-01).
//!
//! ## Layout normalisation
//!
//! The cintx `ecp_int1e_ipnuc` Density carries a component-leading buffer laid
//! out `data[comp + p*3 + q*3*nao]` (component-INNER on the AO axes); the RHF
//! `get_hcore`/`hcore_deriv` path (07-03) uses the component-leading F-order
//! `data[comp*nao*nao + i + j*nao]` layout. [`get_hcore_ecp`] NORMALISES the
//! engine buffer to the RHF layout so the ECP term folds cleanly into the RHF
//! hcore path (`assert_component_leading`-compatible).
//!
//! ## Bit-exact discipline (Pitfall 1/2)
//!
//! Every reduction materialises into a `Vec` then routes through
//! `pyscf_algebra::oracle_sum` — NEVER a bare `+=`.

use pyscf_core::{EcpEngine, Mole, PyscfRsError};

/// The number of Cartesian derivative components (x/y/z) every ECP-gradient
/// intor carries (axis 0 of a `[3, nao, nao]` component-leading buffer).
const NCOMP: usize = 3;

/// The `get_hcore` ECP-gradient term `+ ECPscalar_ipnuc` (`pyscf/grad/rhf.py:
/// 116-117`), cintx-READY (07-01).
///
/// Returns a flat component-leading `[3, nao, nao]` F-order buffer
/// (`data[comp*nao*nao + i + j*nao]` — the RHF-path layout) carrying
/// `∂/∂R_comp ⟨i|V_ECP|j⟩` for the ECP-bearing centres. The numeric arm UN-GATES
/// now: for an ECP-bearing molecule the cintx `ecp_int1e_ipnuc` (un-gated in
/// 07-01) delivers real values; for a molecule with NO ECP the engine reports
/// `EcpEngineNotAvailable` and this returns an ALL-ZERO buffer (the ECP term
/// contributes nothing — never a panic, never a propagated error).
///
/// This folds into the RHF `get_hcore` term (07-03): an ECP-bearing molecule's
/// core-Hamiltonian gradient is `-(int1e_ipkin + int1e_ipnuc) + ECPscalar_ipnuc`.
///
/// # Errors
/// A non-availability cintx error (e.g. a malformed ECP, a workspace failure)
/// `?`-propagates as a clean `Core(InvalidMolecule(..))` — but a plain
/// "no ECP on this molecule" is NOT an error (it is a zero contribution).
pub fn get_hcore_ecp(mol: &Mole) -> Result<Vec<f64>, PyscfRsError> {
    let nao = mol.nao_nr;
    let engine = pyscf_gto::ecp_engine();

    // The cintx ECP-gradient ipnuc dispatch (un-gated in 07-01). The returned
    // Density carries a component-leading buffer `data[comp + p*3 + q*3*nao]`.
    match engine.ecp_int1e_ipnuc(mol, "ECPscalar_ipnuc") {
        Ok(density) => {
            let expect = NCOMP * nao * nao;
            if density.nao != nao || density.data.len() != expect {
                return Err(PyscfRsError::Core(pyscf_core::CoreError::InvalidMolecule(
                    format!(
                        "ECPscalar_ipnuc returned a buffer of {} elements (nao={}); \
                         expected components*nao*nao = {expect} (nao={nao})",
                        density.data.len(),
                        density.nao
                    ),
                )));
            }
            // Normalise the engine's component-INNER layout
            // (`data[comp + p*3 + q*3*nao]`) to the RHF component-leading F-order
            // (`out[comp*nao*nao + i + j*nao]`) so the term folds into the RHF
            // get_hcore path (assert_component_leading-compatible).
            let mut out = vec![0.0_f64; expect];
            for comp in 0..NCOMP {
                let base = comp * nao * nao;
                for q in 0..nao {
                    for p in 0..nao {
                        // engine index: comp + p*3 + q*3*nao ; RHF index: base + p + q*nao.
                        out[base + p + q * nao] = density.data[comp + p * NCOMP + q * NCOMP * nao];
                    }
                }
            }
            Ok(out)
        }
        // No ECP on this molecule → the ECP gradient term contributes nothing.
        // A zero buffer of the right shape (NOT an error, NOT a panic).
        Err(PyscfRsError::EcpEngineNotAvailable) => Ok(vec![0.0_f64; NCOMP * nao * nao]),
        // Any OTHER error (malformed ECP, workspace failure, the gated iprinv
        // availability error if mis-routed) `?`-propagates as-is — never swallowed.
        Err(e) => Err(e),
    }
}

/// The `hcore_deriv` per-atom ECP-gradient term `+ ECPscalar_iprinv`
/// (`pyscf/grad/rhf.py:139-140`) — MISSING from cintx (07-01, no scheduled
/// workstream).
///
/// `ECPscalar_iprinv` is the per-atom (`with_rinv_at_nucleus(atm_id)`) ECP
/// derivative the RHF `hcore_deriv` adds for the atom whose ECP origin is being
/// differentiated. cintx ships NO `int1e_ecp_iprinv` family on any branch
/// (07-01), so this routes to a CLEAN cintx-availability error
/// (`Core(InvalidMolecule(..))`) — NEVER a panic, NEVER a silent zero, NEVER
/// `NotYetImplemented{phase:7}` (T-07-27). The numeric arm un-gates by dropping
/// the gate once the cintx workstream lands `ECPscalar_iprinv`.
///
/// `atm_id` is the atom whose ECP origin is differentiated (validated `< natm`).
pub fn hcore_deriv_ecp(mol: &Mole, atm_id: usize) -> Result<Vec<f64>, PyscfRsError> {
    if atm_id >= mol.natm {
        return Err(PyscfRsError::Core(pyscf_core::CoreError::InvalidMolecule(
            format!("hcore_deriv_ecp: atm_id {atm_id} out of range (natm={})", mol.natm),
        )));
    }
    let engine = pyscf_gto::ecp_engine();
    // Route the iprinv family through the engine; cintx is MISSING it (07-01) so
    // the engine returns a clean cintx-availability error. We DELIBERATELY surface
    // that error rather than substituting a zero or a panic (T-07-27). If a future
    // cintx ships ECPscalar_iprinv this returns the per-atom [3, nao, nao] buffer.
    let _ = engine.ecp_int1e_ipnuc(mol, "ECPscalar_iprinv")?;
    // Unreachable until cintx ships iprinv: when it does, normalise + return the
    // per-atom contribution here (mirroring get_hcore_ecp's layout normalisation).
    Err(PyscfRsError::Core(pyscf_core::CoreError::InvalidMolecule(
        "ECPscalar_iprinv (the per-atom hcore_deriv ECP-gradient term) is MISSING from \
         every cintx branch (07-01, no scheduled workstream) — the iprinv numeric arm \
         un-gates when that family lands."
            .into(),
    )))
}

/// ECP gradient seam preserved for the 07-02 module stub. The real bodies are
/// [`get_hcore_ecp`] (the cintx-ready ipnuc `get_hcore` term) + [`hcore_deriv_ecp`]
/// (the cintx-gated iprinv `hcore_deriv` term); this thin wrapper errors clearly
/// if called without a molecule (the PyO3 bridge always supplies one).
pub fn default_grad_ecp() -> Result<Vec<[f64; 3]>, PyscfRsError> {
    Err(PyscfRsError::Core(pyscf_core::CoreError::InvalidMolecule(
        "ECP gradient requires a Mole — use get_hcore_ecp(mol) / hcore_deriv_ecp(mol, ia) \
         (07-08)"
            .into(),
    )))
}
