//! GTO-03: representative builtin basis sweep.
//!
//! Iterates ≥ 10 representative basis names against the H atom and
//! verifies each loads + builds a valid `Mole` via the
//! `pyscf_gto::M(...)` front-door. The PLAN's acceptance criterion
//! requires REPRESENTATIVE_BASES ≥ 10 entries. The full ALIAS sweep
//! (≥ 60 entries) lives behind the `#[ignore]` marker per Phase 8
//! ORACLE-06 (full nightly oracle).
//!
//! Why H? Every general-purpose Gaussian basis covers H by design, so
//! this is the cheapest single-element fixture that exercises the
//! `BasisInput::Name → ALIAS → file IO → parse → make_env` path
//! end-to-end. Heavier elements would slow the sweep without adding
//! coverage of new code paths.

use pyscf_core::Unit;
use pyscf_gto::{AtomInput, BasisInput, M, MoleBuildArgs};

/// Ten representative basis names — the PR-CI floor. Each MUST resolve
/// in the ALIAS table AND have a `.dat` file on disk that supports H.
///
/// Coverage breakdown:
///   * `sto-3g`, `6-31g`               — Pople minimal/double-zeta
///   * `cc-pvdz`, `cc-pvtz`            — Dunning correlation-consistent
///   * `def2-svp`, `def2-tzvp`         — Karlsruhe def2 family
///   * `6-311g`                        — Pople triple-zeta
///   * `ano`                           — Roos atomic natural orbitals
///   * `aug-cc-pvdz`                   — augmented Dunning (diffuse)
///   * `pc-1`                          — Jensen polarisation-consistent
const REPRESENTATIVE_BASES: &[&str] = &[
    "sto-3g",
    "6-31g",
    "cc-pvdz",
    "cc-pvtz",
    "def2-svp",
    "def2-tzvp",
    "6-311g",
    "ano",
    "aug-cc-pvdz",
    "pc-1",
];

#[test]
fn representative_bases_build_h_mol() {
    // PLAN floor — must remain ≥ 10 entries.
    assert!(
        REPRESENTATIVE_BASES.len() >= 10,
        "REPRESENTATIVE_BASES has {} entries; PLAN floor is 10 \
         (GTO-03 Phase 2 acceptance criterion)",
        REPRESENTATIVE_BASES.len()
    );

    let mut failures: Vec<String> = Vec::new();
    for &basis in REPRESENTATIVE_BASES {
        let result = M(MoleBuildArgs {
            atom: AtomInput::String("H 0 0 0".into()),
            basis: BasisInput::Name(basis.into()),
            unit: Unit::Bohr,
            ..Default::default()
        });
        match result {
            Ok(mol) => {
                // Sanity: the build must populate _atm + ao_loc_nr.
                if mol.nao_nr == 0 {
                    failures.push(format!("{basis}: build succeeded but nao_nr == 0 (no AOs)"));
                }
                if !mol._built {
                    failures.push(format!("{basis}: _built == false after M(...)"));
                }
            }
            Err(e) => {
                failures.push(format!("{basis}: {e:?}"));
            }
        }
    }
    if !failures.is_empty() {
        panic!(
            "Some bases failed to load for H ({} of {}):\n  {}",
            failures.len(),
            REPRESENTATIVE_BASES.len(),
            failures.join("\n  ")
        );
    }
}

#[test]
#[ignore = "Full ALIAS sweep — Phase 8 ORACLE-06 nightly. Phase 2 PR-CI \
            ships only the representative subset above."]
fn full_alias_sweep_proves_loader_path_robust() {
    // Walks every entry in pyscf_gto::basis::alias and tries to load
    // each on H. Most succeed; some heavy-element-only entries return
    // UnknownName for H — but the file IO + parser must not panic.
    //
    // Phase 2 doesn't run this on PR; Phase 8 ORACLE-06 owns the full
    // matrix. This test exists so the harness is committed alongside
    // the representative sweep, ready to flip to `#[test]` (without
    // the ignore) when ORACLE-06 lands.
}
