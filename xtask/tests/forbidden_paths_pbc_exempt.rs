//! Tests verifying that pyscf-pbc-* crates are exempt from the forbidden-paths lint
//! and that molecular crates are still protected against out-of-scope imports.

use std::path::Path;
use xtask::forbidden_paths::{check_file_content, is_pbc_exempt_path};

#[test]
fn test_pbc_crate_is_exempt() {
    let pbc_path = Path::new("crates/pyscf-pbc-foo/src/lib.rs");
    assert!(is_pbc_exempt_path(pbc_path));

    let content = "use pyscf::x2c;\nuse pyscf::mcscf;\n";
    let violations = check_file_content(pbc_path, content);
    assert!(
        violations.is_empty(),
        "pyscf-pbc-* crate paths should be exempt from forbidden import checks"
    );
}

#[test]
fn test_molecular_crate_is_flagged_for_forbidden_imports() {
    let scf_path = Path::new("crates/pyscf-scf/src/foo.rs");
    assert!(!is_pbc_exempt_path(scf_path));

    let content = "use pyscf::x2c;\n";
    let violations = check_file_content(scf_path, content);
    assert_eq!(
        violations.len(),
        1,
        "Molecular crate should be flagged for `use pyscf::x2c`"
    );
    assert!(violations[0].contains("forbidden import `use pyscf::x2c`"));
}

#[test]
fn test_pbc_needle_is_removed() {
    let scf_path = Path::new("crates/pyscf-scf/src/foo.rs");
    let content = "use pyscf::pbc;\n";
    let violations = check_file_content(scf_path, content);
    assert!(
        violations.is_empty(),
        "`use pyscf::pbc` should no longer be in the forbidden needles list"
    );
}
