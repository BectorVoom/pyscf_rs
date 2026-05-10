//! pyscf-gto — Molecular structure & integrals.
//!
//! Phase 2 fills the bodies for GTO-01..11. Wave 0 (plan 02-01) laid out
//! modules and proved cintx reachability. Plan 02-02 (this commit) ships
//! the GTO-01 atom-input front-door + the GTO-08 ≥30-attribute Mole
//! floor wired via `pyscf_gto::M(MoleBuildArgs { ... })`.

#![forbid(unsafe_code)]

pub mod basis; // Plan 02-03 — GTO-03 (basis loading).
pub mod ecp_engine_stub; // Plan 02-07 — GTO-05 (engine half: stub trait impl).
pub mod eval_gto; // Plan 02-06 — GTO-07 (eval_gto user wrapper, algebra-wall friendly).
pub mod format_atom;
pub mod format_basis; // Plan 02-03 — GTO-02 (11→5 input-form dispatch).
pub mod format_ecp; // Plan 02-07 — GTO-05 (loading half: format_ecp + make_ecp_env).
pub mod intor; // Plan 02-05 — GTO-06 (mol.intor(name) dispatcher).
pub mod layout_table; // Wave 0 (plan 02-01); consumed by intor.rs in 02-05.
pub mod make_env; // Plan 02-04 — GTO-04 (flat-array projection, D-03).
pub mod projection; // Plan 02-04 — GTO-11 (zero-copy cintx_core::BasisSet build).
pub mod types;

// Plan 02-08 adds: dumps_loads.

pub use basis::{load_basis, parse as parse_basis};
pub use ecp_engine_stub::EcpEngineNotAvailable;
pub use eval_gto::{eval_gto, EvalGtoOutput};
pub use format_basis::format_basis;
pub use format_ecp::{format_ecp, make_ecp_env};
pub use intor::{intor, IntorOutput};
pub use pyscf_core::{Mole, Unit};
pub use types::{AtomInput, BasisInput, EcpInput, MoleBuildArgs};

/// Returns the active ECP engine. Phase 2 returns the
/// `EcpEngineNotAvailable` stub per D-06 sequencing; gap-closure plan
/// 02-10 swaps to the cintx-backed impl when cintx ECP merges.
///
/// The intor dispatcher (`pyscf_gto::intor::intor`) calls this for any
/// `int1e_ecp*` / `ECPscalar*` name and routes through the
/// `EcpEngine` trait — meaning 02-10 needs to change only the impl
/// returned here, not any caller's code.
pub fn ecp_engine() -> ecp_engine_stub::EcpEngineNotAvailable {
    ecp_engine_stub::EcpEngineNotAvailable
}

/// Shortcut to build a Mole. Equivalent to `pyscf.M(...)` upstream.
///
/// Source: `pyscf/gto/mole.py:106-118` (Apache-2.0).
///
/// # Example
///
/// ```ignore
/// use pyscf_core::Unit;
/// use pyscf_gto::{AtomInput, BasisInput, MoleBuildArgs, M};
///
/// let mol = M(MoleBuildArgs {
///     atom: AtomInput::String("H 0 0 0; H 0 0 1.4".into()),
///     basis: BasisInput::Name("sto-3g".into()),
///     unit: Unit::Bohr,
///     ..Default::default()
/// }).unwrap();
/// assert_eq!(mol.natm, 2);
/// ```
#[allow(non_snake_case)]
pub fn M(args: MoleBuildArgs) -> Result<Mole, pyscf_core::PyscfRsError> {
    let mut mol = Mole::default();
    build_from(&mut mol, args)?;
    Ok(mol)
}

/// Populate a Mole from `MoleBuildArgs`.
///
/// Plan 02-02 wires:
///   - `format_atom` (GTO-01) — parsed `_atom`, `natm`.
///   - The scalar fields (`charge`, `spin`, `unit`, `verbose`, ...).
///   - `nelectron` computation (`sum(atom_charges) - charge`).
///
/// Plan 02-04 will extend this to also call `format_basis` + `make_env`
/// (the cintx flat-array projection). Plan 02-07 will extend with
/// `format_ecp` + `make_ecp_env`.
pub fn build_from(
    mol: &mut Mole,
    args: MoleBuildArgs,
) -> Result<(), pyscf_core::PyscfRsError> {
    // Echo user input for `dumps()` round-trip later.
    mol.atom = format!("{:?}", args.atom);
    mol.basis = format!("{:?}", args.basis);
    mol.ecp = format!("{:?}", args.ecp);

    // Scalar state.
    mol.charge = args.charge;
    mol.spin = args.spin;
    mol.cart = args.cart;
    mol.unit = args.unit;
    mol.verbose = args.verbose;
    mol.max_memory = args.max_memory;
    mol.output = args.output;
    mol.symmetry = false; // out of scope per ROADMAP.
    mol.groupname = "C1".to_string();
    mol.topgroup = "C1".to_string();

    // GTO-01: format_atom.
    let parsed_atoms =
        format_atom::format_atom(&args.atom, args.unit, args.origin, args.axes)?;
    mol.natm = parsed_atoms.len();
    mol._atom = parsed_atoms;

    // Compute nelectron from atom_charges - mol.charge (per upstream
    // tot_electrons at pyscf/gto/mole.py:1162-1186).
    let total_z: i32 = mol
        ._atom
        .iter()
        .filter_map(|(s, _)| format_atom::charge_for_symbol(s))
        .sum();
    let nelec = total_z - mol.charge;
    if nelec < 0 {
        return Err(pyscf_core::PyscfRsError::Core(
            pyscf_core::CoreError::InvalidMolecule(format!(
                "negative electron count: {nelec} (total Z = {total_z}, charge = {})",
                mol.charge
            )),
        ));
    }
    mol.nelectron = nelec as usize;

    // GTO-02: format_basis — populate `mol._basis` (per-element-symbol map of
    // ParsedBasis). Plan 02-03 wired the dispatch + loader.
    let parsed_basis = format_basis::format_basis(&args.basis, &mol._atom)?;
    mol._basis = parsed_basis;

    // GTO-04 + GTO-11 (plan 02-04, this commit): project to libcint flat
    // arrays + build the zero-copy cintx_core::BasisSet Arc.
    let env_out = make_env::make_env(&mol._atom, &mol._basis, mol.cart);
    mol._atm = env_out._atm;
    mol._bas = env_out._bas;
    mol._env = env_out._env;
    mol.ao_loc_nr = env_out.ao_loc_nr;
    mol.nao_nr = env_out.nao_nr;
    mol.nbas = mol._bas.len() / pyscf_core::raw_layout::BAS_SLOTS;
    mol.nao_2c = 0; // Phase 2 stub — spinor not in scope; Phase 3 may extend.

    // GTO-11: zero-copy Arc<BasisSet>. Stored in mol.basis_set; consumers
    // (02-05 intor, 02-06 eval_gto, SCF, DFT, ...) clone the Arc rather than
    // rebuilding the typed view.
    mol.basis_set = Some(projection::build_cintx_basis_set(
        &mol._atom,
        &mol._basis,
        mol.cart,
    )?);

    // Plan 02-07 (GTO-05 loading half): ECP processing.
    //
    // Order matters: `make_env` (above) populates `_atm`, then
    // `make_ecp_env` mutates `_atm[CHARGE_OF]` per Pitfall 2. Doing
    // ECP processing AFTER make_env keeps the Pitfall-2 contract
    // ("subtract once") inside a single function.
    let parsed_ecp = format_ecp::format_ecp(&args.ecp, &mol._atom)?;
    mol._ecp = parsed_ecp;
    if !mol._ecp.is_empty() {
        mol._ecpbas = format_ecp::make_ecp_env(
            &mol._atom,
            &mol._ecp,
            &mut mol._atm,
            &mut mol._env,
        );

        // Recompute `nelectron` because CHARGE_OF was decremented for
        // ECP atoms.
        use pyscf_core::raw_layout::{ATM_SLOTS, CHARGE_OF};
        let total_z: i32 = (0..mol.natm)
            .map(|i| mol._atm[i * ATM_SLOTS + CHARGE_OF])
            .sum();
        let nelec = total_z - mol.charge;
        if nelec < 0 {
            return Err(pyscf_core::PyscfRsError::Core(
                pyscf_core::CoreError::InvalidMolecule(format!(
                    "ECP processing produced negative electron count: \
                     total_valence_Z={total_z}, charge={}",
                    mol.charge,
                )),
            ));
        }
        mol.nelectron = nelec as usize;
    }

    // Mark built — `Mole::build()` succeeds, `mol.intor(...)` (when 02-05
    // lands it) is allowed to dispatch.
    mol._built = true;

    Ok(())
}
