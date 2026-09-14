//! `pyscf._native.pbc` — the NESTED periodic module tree (plan 20-08).
//!
//! The seven molecular submodules in `lib.rs` are FLAT (`_native.scf`, …) and
//! reachable only by attribute access. PBC needs `import pyscf._native.pbc.scf`
//! to work as a real import statement, which an attribute alone does not give:
//! `pyscf._native` is an extension module, not a package, so the import system
//! can only resolve a dotted child it finds in `sys.modules`. This file is the
//! ONE place that pattern lives; 20-09 … 20-15 fill the children, they do not
//! re-derive the registration.
//!
//! The pattern, per child:
//!   1. `PyModule::new(py, "pyscf._native.pbc.<child>")` — the FULL dotted name
//!      becomes `__name__`, so `repr()` and pickling report the real path;
//!   2. `parent.add_submodule(&child)` — PyO3 strips the dotted prefix, so the
//!      attribute is the short `<child>`;
//!   3. `sys.modules["pyscf._native.pbc.<child>"] = child`.
//!
//! Classes bound into a child in later plans MUST say
//! `#[pyclass(module = "pyscf._native.pbc.<child>")]` for the same reason.
//!
//! Children are registered empty here and filled by their plan's `register`:
//! `gto` (20-09, `pbc/gto.rs`) and `df` (20-10, `pbc/df.rs`) so far.
//! `pyscf-pbc-dft` is a dependency since 20-09 Step 0a; its bindings are 20-13's.

use pyo3::prelude::*;

pub mod convert;
pub mod df;
pub mod gto;

/// Dotted name of the periodic parent module.
pub const PBC_MODULE: &str = "pyscf._native.pbc";

/// The ten children, in upstream `pyscf.pbc` package order, with the plan that
/// fills each one.
pub const PBC_CHILDREN: [(&str, &str); 10] = [
    ("gto", "20-09"),
    ("scf", "20-12"),
    ("dft", "20-13"),
    ("df", "20-10"),
    ("symm", "20-14"),
    ("lib", "20-14"),
    ("tools", "20-14"),
    ("mp", "20-15"),
    ("cc", "20-15"),
    ("ci", "20-15"),
];

/// Create `pyscf._native.pbc` and its ten empty children, attach `pbc` to the
/// `_native` root, and register every module in `sys.modules` under its full
/// dotted name so the nested import statement resolves.
pub fn register(py: Python<'_>, root: &Bound<'_, PyModule>) -> PyResult<()> {
    let sys_modules = PyModule::import(py, "sys")?.getattr("modules")?;

    let pbc = PyModule::new(py, PBC_MODULE)?;
    pbc.setattr(
        "__doc__",
        "Periodic (PBC) bindings. Children are registered empty until their plan lands.",
    )?;
    for (child, plan) in PBC_CHILDREN {
        let full = format!("{PBC_MODULE}.{child}");
        let m = PyModule::new(py, &full)?;
        m.setattr(
            "__doc__",
            match child {
                "gto" | "df" => format!("{full} — periodic bindings (plan {plan})."),
                _ => {
                    format!("{full} — registered empty (plan 20-08); bindings land in plan {plan}.")
                }
            },
        )?;
        match child {
            "gto" => gto::register(&m)?,
            "df" => df::register(&m)?,
            _ => {}
        }
        pbc.add_submodule(&m)?;
        sys_modules.set_item(&full, &m)?;
    }

    root.add_submodule(&pbc)?;
    sys_modules.set_item(PBC_MODULE, &pbc)?;
    Ok(())
}
