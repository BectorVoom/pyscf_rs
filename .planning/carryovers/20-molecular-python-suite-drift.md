# Carryover — pre-existing molecular Python test failures (not caused by Phase 20)

**Source:** `.planning/phases/20-pbc-python-bindings/measurements/python-suite-triage.md`
(A/B with Phase-20 `sys.modules` registrations stripped: identical failing ids); re-confirmed on the
final tree by 20-18 (`20-VERIFICATION.md` §7).

## The baseline set (10 failed + 9 errors)

| test | error | cause |
|---|---|---|
| `test_scf_{analyze ×3, chkfile::test_chkfile_upstream_writes_pyscf_rs_reads, df, diis, rhf_benzene, uhf, xplat_uhartree}` (9 errors) | in-process upstream loader (`conftest._load_upstream`); since 20-19 A the guard raises instead of a vacuous native-vs-native compare | R1: the in-process loader cannot work under the overlay; CI uses the subprocess oracle `PYSCF_RS_UPSTREAM_PYTHON` |
| `test_scf_rhf_h2o.py`, `test_scf_rhf_ccpvdz.py` ×3 | same loader | environment: pass with `PYSCF_RS_UPSTREAM_PYTHON=$PWD/.venv/bin/python` (4 passed) |
| `test_scf_cross_dispatch.py` ×3 | `DID NOT RAISE PyscfRsError` | test drift: `to_uks`/`to_rks` wired in `fbe1e35` (2026-05-22), test still expects the Phase-3 stub refusal |
| `test_scf_ghf.py` | `GHF` has no `run` | Rust surface gap in `crates/pyscf-py/src/scf.rs` `PyGHF` |
| `test_panic_to_exception.py` ×2 | raises base `_native.PyscfRsRuntimeError(kind 'Core')`, not the Python subclass `PyscfRsError` | R2 design defect in `errors.rs` (`create_exception!` base class) + non-convergence surfacing as `CoreError::InvalidMolecule` |

## Unblock

1. Retire the in-process loader, or make it spawn the vendored tree out of process.
2. Update `test_scf_cross_dispatch.py` to the wired behaviour.
3. Bind `PyGHF.run`.
4. Decide the exception hierarchy (raise the subclass, or make the Python class an alias of the native one) and map non-convergence to `ConvergenceFailure`.
