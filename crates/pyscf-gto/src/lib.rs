//! pyscf-gto — Molecular structure & integrals.
//!
//! Phase 2 fills the bodies for GTO-01..11. Wave 0 (plan 02-01) laid out
//! modules and proved cintx reachability. Plan 02-02 (this commit) ships
//! the GTO-01 atom-input front-door + the GTO-08 ≥30-attribute Mole
//! floor wired via `pyscf_gto::M(MoleBuildArgs { ... })`.

#![forbid(unsafe_code)]

pub mod basis; // Plan 02-03 — GTO-03 (basis loading).
pub mod format_atom;
pub mod format_basis; // Plan 02-03 — GTO-02 (11→5 input-form dispatch).
pub mod layout_table; // Wave 0 (plan 02-01); consumed by intor.rs in 02-05.
pub mod make_env; // Plan 02-04 — GTO-04 (flat-array projection, D-03).
pub mod projection; // Plan 02-04 — GTO-11 (zero-copy cintx_core::BasisSet build).
pub mod types;

// Plans 02-05..02-08 add: intor, eval_gto, ecp_engine_stub, dumps_loads.

pub use basis::{load_basis, parse as parse_basis};
pub use format_basis::format_basis;
pub use pyscf_core::{Mole, Unit};
pub use types::{AtomInput, BasisInput, EcpInput, MoleBuildArgs};

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

    // Mark built — `Mole::build()` succeeds, `mol.intor(...)` (when 02-05
    // lands it) is allowed to dispatch.
    mol._built = true;

    Ok(())
}
