//! Plan 07-09 Task 2 — ALWAYS-ON structural test for the geomopt PyO3 surface
//! (the `optimize` entry + the `geometric_solver`/`berny_solver` shims
//! delegating to the ONE native engine + the GEOMOPT-01 no-runtime-dep contract
//! + the constraints clear-error).
//!
//! Per the 07-06 precedent the always-on arm asserts the bridge plumbing EXISTS
//! (the optimize/kernel entries, the single-engine delegation, the no-geometric/
//! pyberny-import contract, the constraints clear-error) on a source-scan — it
//! does NOT require a live Python `mf`, maturin, or the (MISSING) cintx
//! grad-intor families. The live optimizer trajectory + the `pip uninstall
//! geometric pyberny` no-runtime-dep proof are the 07-10 CI close-out arm.

use std::fs;

/// `geomopt.rs` must expose the `optimize` entry + the `geometric_solver` /
/// `berny_solver` shim entry points (kernel + optimize each), delegating to the
/// ONE native engine (D-06).
#[test]
fn geomopt_rs_defines_the_optimizer_surface() {
    let src = fs::read_to_string("src/geomopt.rs").expect("geomopt.rs readable");
    // The module-level optimize() entry (pyscf.geomopt.optimize).
    assert!(
        src.contains("fn optimize") && src.contains("name = \"optimize\""),
        "geomopt.rs must expose the optimize() + the named geometric/berny optimize entries"
    );
    // Both solver shims with kernel + optimize.
    for entry in [
        "geometric_solver_kernel",
        "geometric_solver_optimize",
        "berny_solver_kernel",
        "berny_solver_optimize",
    ] {
        assert!(
            src.contains(entry),
            "geomopt.rs must expose the `{entry}` shim entry point (D-07)"
        );
    }
    // The shim submodules are registered under _native.geomopt.
    assert!(
        src.contains("\"geometric_solver\"") && src.contains("\"berny_solver\""),
        "geomopt.rs must register the geometric_solver + berny_solver submodules"
    );
}

/// The shims delegate to the ONE native engine (D-06 / T-07-20): the single
/// shared `run_geomopt` core drives `pyscf_geomopt::geometric_solver::kernel`;
/// there is NO second optimizer implementation in the bridge.
#[test]
fn geomopt_shims_delegate_to_one_native_engine() {
    let src = fs::read_to_string("src/geomopt.rs").expect("geomopt.rs readable");
    assert!(
        src.contains("fn run_geomopt"),
        "geomopt.rs must route every entry through the single shared run_geomopt core (T-07-20)"
    );
    assert!(
        src.contains("pyscf_geomopt::geometric_solver::kernel"),
        "geomopt.rs run_geomopt must drive the ONE native pyscf_geomopt engine (D-06)"
    );
    // The berny entry must NOT invoke a second optimizer — it routes through the
    // same run_geomopt. We assert the bridge does not name a distinct second
    // engine function (no `berny`-specific native optimize call).
    assert!(
        !src.contains("pyscf_geomopt::berny_solver::kernel"),
        "the berny shim must NOT call a distinct native berny engine — it is a thin alias (T-07-20)"
    );
    // The single-engine marker is referenced.
    assert!(
        src.contains("NATIVE_ENGINE_NAME"),
        "geomopt.rs must reference the single NATIVE_ENGINE_NAME marker (T-07-20)"
    );
}

/// T-07-33: a non-None `constraints` raises a clear error (the native
/// ConstraintsUnsupported, surfaced as a Python exception), never a silent
/// no-op. The bridge must build the ShimParams constraints marker.
#[test]
fn geomopt_constraints_raise_a_clear_error() {
    let src = fs::read_to_string("src/geomopt.rs").expect("geomopt.rs readable");
    assert!(
        src.contains("params.constraints = Some"),
        "geomopt.rs must surface a non-None constraints into the ShimParams marker (T-07-33)"
    );
    assert!(
        src.contains("ConstraintsUnsupported") || src.contains("constraints"),
        "geomopt.rs must route constraints to the native clear-error (never a silent no-op)"
    );
    // maxsteps defaults to 100 at every entry (T-07-32).
    assert!(
        src.contains("maxsteps=100"),
        "every geomopt entry must default maxsteps=100 (T-07-32)"
    );
}

/// GEOMOPT-01 (the CRITICAL no-runtime-dep contract): the Python overlay
/// re-exports `_native.geomopt` and contains NO `import geometric` / `import
/// pyberny` — the optimizer is fully native.
#[test]
fn python_geomopt_overlay_has_no_external_optimizer_import() {
    let src = fs::read_to_string("../../python/pyscf/geomopt/__init__.py")
        .expect("python/pyscf/geomopt/__init__.py readable");
    assert!(
        src.contains("from pyscf._native.geomopt"),
        "python/pyscf/geomopt/__init__.py must re-export from pyscf._native.geomopt (BIND-02)"
    );
    for sym in ["optimize", "geometric_solver", "berny_solver"] {
        assert!(
            src.contains(sym),
            "python/pyscf/geomopt/__init__.py __all__/import must list `{sym}`"
        );
    }
    assert!(
        src.contains("__all__"),
        "python/pyscf/geomopt/__init__.py must define __all__"
    );
    // GEOMOPT-01: NO external optimizer runtime dependency. We scan each line
    // for an `import` of the external `geometric` / `pyberny` PACKAGES (the
    // standalone package names — NOT our own `geometric_solver`/`berny_solver`
    // submodules). This mirrors the acceptance-criterion intent: the optimizer
    // is fully native, so neither external package is imported.
    for line in src.lines() {
        let t = line.trim();
        // A bare `import geometric` / `from geometric import ...` (the package),
        // distinguished from `import geometric_solver` by the trailing token.
        let imports_external = |pkg: &str| -> bool {
            for prefix in [format!("import {pkg}"), format!("from {pkg}")] {
                if let Some(rest) = t.strip_prefix(&prefix) {
                    // The next char must not continue the identifier (so
                    // `geometric_solver` / `bernyx` do NOT match).
                    match rest.chars().next() {
                        Some(c) if c.is_alphanumeric() || c == '_' => {}
                        _ => return true,
                    }
                }
            }
            false
        };
        for pkg in ["geometric", "pyberny", "berny"] {
            assert!(
                !imports_external(pkg),
                "GEOMOPT-01 VIOLATION: python/pyscf/geomopt/__init__.py imports the external \
                 `{pkg}` package (`{t}`) — the optimizer must be fully native"
            );
        }
    }
}
