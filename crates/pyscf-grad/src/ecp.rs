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
//!     comp=3)` under `with_rinv_at_nucleus(atm_id)` — `ECPscalar_iprinv`
//!     (= `int1e_ecp_iprinv`) is now cintx-READY (F-05 / cintx workstream 21-07).
//!     [`hcore_deriv_ecp`] returns the REAL per-atom component-leading
//!     `[3, nao, nao]` buffer for an ECP-bearing atom (and an all-zero buffer for
//!     a non-ECP atom / molecule — never a panic). Consuming this per-atom buffer
//!     into total nuclear forces is F-08 (still out of scope here — this returns
//!     the integral only).
//!
//! Both dispatch through the Phase-2 [`pyscf_gto::CintxEcpEngine`]: the
//! `ecp_int1e_ipnuc` (07-01) and the `ecp_int1e_iprinv` (F-05) methods.
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
/// (`pyscf/grad/rhf.py:139-140`) — now cintx-READY (F-05 / cintx workstream
/// 21-07).
///
/// `ECPscalar_iprinv` is the per-atom (`with_rinv_at_nucleus(atm_id)`) ECP
/// derivative the RHF `hcore_deriv` adds for the atom whose ECP origin is being
/// differentiated. The cintx native `ecp_iprinv` kernel selects ONLY the ECP
/// slot whose coordinate matches the rinv origin; this returns the real per-atom
/// component-leading `[3, nao, nao]` buffer (normalised to the RHF
/// component-leading F-order, mirroring [`get_hcore_ecp`]).
///
/// `atm_id` is the atom whose ECP origin is differentiated (validated `< natm`);
/// its coordinate (Bohr — `Mole` storage is always Bohr) is the rinv origin.
/// For a non-ECP atom (or a molecule with no ECP) the iprinv term contributes
/// nothing → an all-zero `[3, nao, nao]` buffer (PySCF adds `ECPscalar_iprinv`
/// ONLY when `atm_id in ecp_atoms`; the cintx kernel likewise zero-fills when
/// the origin matches no ECP atom). Other engine errors propagate.
///
/// Consuming this per-atom buffer into total nuclear forces is F-08 (still out
/// of scope — this returns the integral only).
pub fn hcore_deriv_ecp(mol: &Mole, atm_id: usize) -> Result<Vec<f64>, PyscfRsError> {
    if atm_id >= mol.natm {
        return Err(PyscfRsError::Core(pyscf_core::CoreError::InvalidMolecule(
            format!(
                "hcore_deriv_ecp: atm_id {atm_id} out of range (natm={})",
                mol.natm
            ),
        )));
    }
    let nao = mol.nao_nr;
    // The rinv origin is the differentiated nucleus coordinate (Bohr); cintx's
    // ecp_iprinv selector matches ECP slots against it.
    let rinv_origin = mol.atom_coord(atm_id);
    let engine = pyscf_gto::ecp_engine();

    // F-05: route the per-atom iprinv family through the dedicated engine method
    // (cintx 21-07). The component-leading buffer is `data[comp + p*3 + q*3*nao]`.
    match engine.ecp_int1e_iprinv(mol, "ECPscalar_iprinv", rinv_origin) {
        Ok(density) => {
            let expect = NCOMP * nao * nao;
            if density.nao != nao || density.data.len() != expect {
                return Err(PyscfRsError::Core(pyscf_core::CoreError::InvalidMolecule(
                    format!(
                        "ECPscalar_iprinv returned a buffer of {} elements (nao={}); \
                         expected components*nao*nao = {expect} (nao={nao})",
                        density.data.len(),
                        density.nao
                    ),
                )));
            }
            // Normalise the engine's component-INNER layout
            // (`data[comp + p*3 + q*3*nao]`) to the RHF component-leading F-order
            // (`out[comp*nao*nao + p + q*nao]`) — same pure copy as get_hcore_ecp
            // (no accumulation, so no oracle_sum required).
            let mut out = vec![0.0_f64; expect];
            for comp in 0..NCOMP {
                let base = comp * nao * nao;
                for q in 0..nao {
                    for p in 0..nao {
                        out[base + p + q * nao] = density.data[comp + p * NCOMP + q * NCOMP * nao];
                    }
                }
            }
            Ok(out)
        }
        // A non-ECP atom (or a molecule with no ECP) contributes nothing to the
        // per-atom iprinv term → an all-zero [3, nao, nao] buffer (NOT an error,
        // NOT a panic). Matches PySCF: vrinv += ECPscalar_iprinv ONLY when
        // atm_id in ecp_atoms.
        Err(PyscfRsError::EcpEngineNotAvailable) => Ok(vec![0.0_f64; NCOMP * nao * nao]),
        // Any OTHER error (malformed ECP, workspace failure) propagates as-is.
        Err(e) => Err(e),
    }
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
