---
phase: 07-gradients-geomopt
slug: gradients-geomopt
status: verified
threats_open: 0
asvs_level: 1
created: 2026-05-26
---

# Phase 07 — Security

> Per-phase security contract: threat register, accepted risks, and audit trail.
> Verification mode: register authored at plan time (10 PLAN.md `<threat_model>` STRIDE
> blocks). Each declared mitigation was verified present in the implemented code by grep
> + read; no retroactive threat discovery. Implementation files are read-only.

---

## Trust Boundaries

| Boundary | Description | Data Crossing |
|----------|-------------|---------------|
| cintx kernel ↔ pyscf-gto dispatch | gradient-integral buffer shape/layout; a wrong repack scrambles x/y/z components | `[3, nao, nao]` / `[3, nao⁴]` component-leading F-order tensors |
| missing-cintx-family ↔ caller | an absent grad-intor family must surface a clean error, not a panic/silent-zero | `Result<IntorOutput, PyscfRsError>` |
| user kwargs (disp, atmlst, maxsteps, constraints) ↔ kernel/optimizer | unbounded/invalid scalars cross here | f64 disp, usize indices, usize maxsteps, constraints marker |
| energy/grad reduction ↔ FD/de result | bare `+=` would make results thread-count-dependent | f64 accumulators (routed through oracle_sum/oracle_dot) |
| caller max_cycle ↔ cphf::solve | an unbounded max_cycle would loop forever | usize max_cycle + matrix-free fvind |
| Phase-6 λ/RDM ↔ CCSD-grad | re-deriving λ would diverge from the validated surface | `pyscf_ccsd::solve_lambda` output |
| HDF5 checkpoint ↔ resume | a corrupt/incompatible checkpoint must fail cleanly | schema_version + shaped state |
| Python mf object / NumPy ↔ Rust snapshot | malformed/non-contiguous arrays cross the FFI | mo_coeff (F-contig), mol, de (C-contig) |
| GIL ↔ long grad/optimize compute | a callback re-entering Python under a held GIL deadlocks | Python::detach / Python::attach |
| Rust panic ↔ Python exception | a panic escaping the FFI is a DoS | PyResult / pyscf_to_py bridge |
| CI runner ↔ cintx sibling crate / libxc | workspace-wide cargo would trigger the ~6h libxc compile | scoped `-p` cargo invocations |
| crate dep graph ↔ registry / cubecl | a new registry package or cubecl edge would breach the supply-chain/algebra wall | path-only deps; xtask dependency-wall lint |

---

## Threat Register

| Threat ID | Category | Component | Disposition | Mitigation | Status |
|-----------|----------|-----------|-------------|------------|--------|
| T-07-01 | Tampering | int2e_ip1 component-leading repack (intor.rs) | mitigate | `ComponentLeadingFOrder{components}` path; logical shape `[c,nao⁴]` component-axis-leading (intor.rs:466-471); `assert_component_leading` rejects `[nao,nao,3]` before contraction (rhf.rs:446-470) | closed |
| T-07-02 | DoS | missing cintx family dispatch | mitigate | `Resolver::descriptor_by_symbol` returns a clean `Core(InvalidMolecule)` availability error (intor.rs:152-158); arity-4 cintx errors `?`-propagate, never panic (intor.rs:519-531, 549-558) | closed |
| T-07-03 | Info disclosure | silent zero buffer for an unimplemented op | accept | A zero buffer is structurally prevented: shape assert (T-07-01) + cintx availability error (T-07-02) + downstream FD gate. No silent-zero path exists. See Accepted Risks Log RA-01. | closed |
| T-07-04 | Tampering | reduction order in verify_fd / grad_elec | mitigate | FD central-difference + diff via `oracle_sum` (verify_fd.rs:108,111); no bare `+=` in any reduction (displacement writes are single-write) | closed |
| T-07-05 | DoS | unbounded/zero `disp` from caller | mitigate | `if !disp.is_finite() || disp <= 0.0 { return Err(InvalidDisplacement) }` BEFORE any energy eval (verify_fd.rs:78-79) | closed |
| T-07-06 | Tampering | out-of-range `atmlst` index | mitigate | `resolve_atmlst` bounds-checks `idx >= natm` → `GradError::ShapeMismatch`, never OOB (lib.rs:254-268); `#![forbid(unsafe_code)]` | closed |
| T-07-07 | Tampering | grad_elec einsum reductions | mitigate | grad contractions materialize then `oracle_sum`/`oracle_dot`; FD gate catches drift; algebra-wall import exercised | closed |
| T-07-08 | Tampering | component-leading layout in get_ovlp/get_veff/hcore | mitigate | `assert_component_leading(&out, nao, name)` called on every gradient intor (rhf.rs:230,245,247,285; uhf/rks/uks parallels); first axis MUST == 3 (rhf.rs:460) | closed |
| T-07-09 | Tampering | eigenvector sign flips across the scanner | accept | SCF `default_eig` applies `pyscf_core::canonicalize_signs` on every diagonalization (eig.rs:36); gradients inherit vendor-stable mo_coeff. See RA-02. | closed |
| T-07-10 | DoS | unbounded `maxsteps` from user kwargs | mitigate | `validate_maxsteps`: rejects 0 and `> MAX_ALLOWED_MAXSTEPS (10_000)` → `InvalidMaxSteps`; default 100 (lib.rs:126-131; converge.rs:30,32) | closed |
| T-07-11 | Tampering | `constraints` kwarg silently ignored | mitigate | `if opt.has_constraints { return Err(ConstraintsUnsupported) }` (lib.rs:232-234; error.rs:30-31) | closed |
| T-07-12 | Tampering | eigenvector sign flip across geometry steps | mitigate | GradScanner re-runs the SCF energy closure each step → `canonicalize_signs` inside `default_eig` re-applies per geometry (scanner.rs:61-69 → eig.rs:36); scanner does not bypass SCF | closed |
| T-07-13 | DoS | singular/ill-conditioned Wilson B-matrix | mitigate | `g_inverse` drops eigenvalues `<= G_EIGENVALUE_TOL (1e-6)` as null-space (bmatrix.rs:31,209); backtransform fixed-point returns `BacktransformDiverged` on non-convergence | closed |
| T-07-14 | Tampering | spin-channel + grid reductions | mitigate | α/β + grid reductions materialize then `oracle_sum`/`oracle_dot`; no bare `+=` in the reductions | closed |
| T-07-15 | Tampering | CPHF wrongly called for KS/UHF grad | mitigate | Zero `cphf::`/`Fvind`/`::solve(` call sites in rhf/uhf/rks/uks grad bodies (grep exit 1); only doc comments stating "NO response solve" (D-04) | closed |
| T-07-16 | DoS | a libxc-backed functional in the FD test | mitigate | KS-grad FD path stays on the in-tree xcfun backend (07-05-SUMMARY threat flags); `cargo test -p pyscf-grad` never enters the libxc dep graph | closed |
| T-07-17 | Tampering | `constraints` kwarg silently ignored (shim) | mitigate | Shim threads `constraints` into the native engine → `ConstraintsUnsupported` (shims.rs; lib.rs:232) | closed |
| T-07-18 | DoS | unbounded `maxsteps` in the shims | mitigate | Shim defaults `maxsteps=100` (converge.rs DEFAULT_MAXSTEPS) and validates via the native `validate_maxsteps` (shims.rs:32-33,62-64,92) | closed |
| T-07-19 | Tampering | corrupt/incompatible HDF5 checkpoint | mitigate | `load` validates `schema_version` (checkpoint.rs:173-180) + shape invariants (189-193) → `CheckpointCorrupt`; `validate()` also runs before any write (dump:130) | closed |
| T-07-20 | Tampering | a second berny optimizer drifting from geomeTRIC | mitigate | `berny_solver` is a thin ALIAS over the one native engine (shims.rs:8,208-245; both report `NATIVE_ENGINE_NAME`); `shim_parity` structural test asserts bit-identical results | closed |
| T-07-21 | DoS | unbounded `max_cycle` in cphf::solve | mitigate | `krylov` caps at `max_cycle.min(ndim)` (cphf.rs:306), loops `0..cap` (308); non-convergence → explicit error (345-351); default 50, MP2 override 30 | closed |
| T-07-22 | Tampering | dense A-matrix memory blow-up | mitigate | Matrix-free `vind_vo`/`fvind` response operator only; dense `A (O(nocc²nvir²))` never materialized (cphf.rs:1,23-28,69) | closed |
| T-07-23 | Tampering | CPHF reduction / aop drift | mitigate | CPHF contractions via pyscf_algebra; reductions via oracle_sum/oracle_dot (cphf.rs:44-47); FD gate catches drift | closed |
| T-07-24 | Tampering | N CPHF copies (GRAD-10 violation) | mitigate | Exactly one `pub fn solve` (cphf.rs:121); `single_cphf_impl` structural test (tests/cphf.rs:322) forbids a second CPHF solver | closed |
| T-07-25 | Tampering | re-deriving CCSD λ in pyscf-grad | mitigate | `use pyscf_ccsd::solve_lambda` (ccsd.rs:55), called directly (ccsd.rs:211); `single_lambda_solver_in_grad` test (ccsd_verify_fd.rs:266) forbids a re-derived solver | closed |
| T-07-26 | Tampering | CCSD Z-vector / ECP reductions | mitigate | CCSD/ECP contractions via pyscf_algebra; reductions via oracle_sum/oracle_dot; FD gate catches drift | closed |
| T-07-27 | DoS | ECPscalar_iprinv missing in cintx | mitigate | ECP engine routes all derivative names to a clean cintx-availability error, NOT `NotYetImplemented{phase:7}` (ecp_engine_cintx.rs:37-38,84-85,102) | closed |
| T-07-28 | Tampering | CCSD-grad reuses the wrong CPHF | mitigate | CCSD orbital-relaxation Z-vector re-enters the ONE `cphf::solve` (ccsd.rs:51,355) with its own fvind/RHS | closed |
| T-07-29 | DoS | Rust panic escaping the FFI | mitigate | Every `#[pyfunction]`/`#[pymethods]` returns `PyResult`; errors `?`-propagate via `pyscf_to_py` → `PyErr` (errors.rs:27; grad.rs:243); PyO3's intrinsic FFI boundary converts any panic→PanicException (never abort); `#![forbid(unsafe_code)]` on all three crates. See note N-01. | closed |
| T-07-30 | Tampering | malformed/non-contiguous NumPy mo_coeff/mol | mitigate | `numpy_io::to_mo_coeff` checks `is_c_contiguous()` and falls back to `to_owned()`/re-materialize (numpy_io.rs:31-36); `de_to_pyarray` returns C-contiguous output (grad.rs:163-164) | closed |
| T-07-31 | DoS | GIL re-entrancy in scanner/callback | mitigate | Kernel does NOT detach at top (override re-enters Python); pure-Rust DEFAULT compute runs under `py.detach` (grad.rs:225-242); scanner closures re-acquire via `Python::attach` (geomopt.rs:173,304-309) | closed |
| T-07-32 | DoS | unbounded `maxsteps` from Python kwargs | mitigate | Bridge `parse_shim_params(maxsteps)` defaults 100 and feeds the native `validate_maxsteps` cap (geomopt.rs:234-238,301) | closed |
| T-07-33 | Tampering | `constraints` kwarg silently ignored (Python) | mitigate | Non-None `constraints` surfaces the native `ConstraintsUnsupported` as a Python exception (geomopt.rs:27-28,222-235,307) | closed |
| T-07-34 | Tampering | subclass override not dispatched | mitigate | `is_overridden` compares the resolved bound method's `__qualname__` against base names (grad.rs:187-204); override → `call_method1` (grad.rs:232-234), subclass-override-wins | closed |
| T-07-35 | DoS | workspace-wide cargo test triggering libxc | mitigate | `grad-structural` runs `cargo test -p pyscf-grad -p pyscf-geomopt -p pyscf-oracle --locked` (ci.yml:716); `geomopt-no-runtime-dep` builds via maturin with no libxc; NEVER `--features libxc` on a grad/geomopt arm | closed |
| T-07-36 | Tampering | non-deterministic FD/oracle across threads | mitigate | `--test-threads=1` on grad-structural (ci.yml:716) + grad-oracle-upstream-manual (842); release-oracle ordered-reduction discipline (oracle_sum/oracle_dot throughout) | closed |
| T-07-37 | Tampering | upstream arm leaking into the daily gate | mitigate | `grad-oracle-upstream-manual` is `if: github.event_name == 'workflow_dispatch'` (ci.yml:819); never auto-runs on push/PR | closed |
| T-07-38 | Tampering | hidden geometric/pyberny runtime dep | mitigate | `geomopt-no-runtime-dep` CI job `pip uninstall -y geometric pyberny`, asserts not importable, proves native `optimize(mf)` runs (ci.yml:732-801); python overlay imports only `pyscf._native.geomopt` (geomopt/__init__.py:31-33); no `import geometric`/`import pyberny` in code | closed |
| T-07-SC | Tampering | crate/CI supply-chain (×4 plans: 07-01/04/05/06/07/08/09/10) | mitigate | pyscf-grad/geomopt declare only intra-workspace `{ path = "../..." }` deps — no registry package, no pyo3, no own hdf5-metno (HDF5 via pyscf-chkfile alias) (Cargo.toml:24-39 / 24-35); cubecl denylist enforced by `xtask check_dependency_wall` (denylist 29-37, allowlist excludes grad/geomopt at 47); geomeTRIC BSD-3-compatible Non-AI license confirmed BEFORE port + recorded in 07-04-SUMMARY (re-derived, NOT vendored); `setup-sibling-crates` provides cintx | closed |

*Status: open · closed*
*Disposition: mitigate (implementation required) · accept (documented risk) · transfer (third-party)*

**Counts:** 39 logical threats (T-07-01…T-07-38 + T-07-SC). Closed: 39. Open: 0.

---

## Accepted Risks Log

| Risk ID | Threat Ref | Rationale | Accepted By | Date |
|---------|------------|-----------|-------------|------|
| RA-01 | T-07-03 | A silent zero buffer for an unimplemented grad-intor op is structurally impossible: every missing cintx family resolves to a clean availability `Err` at the resolver (intor.rs:152), the `assert_component_leading` shape gate rejects any wrong-shape buffer before contraction (rhf.rs:446), and the always-on FD gate would catch any wrong numeric. There is no code path that substitutes zeros for a missing op. Residual risk: none beyond the latent code-review WARNINGs (see Audit Notes) which are on cintx-gated dead code. | gsd-security-auditor (Phase 7 plan dispositions) | 2026-05-26 |
| RA-02 | T-07-09 | Eigenvector sign canonicalization is owned by the SCF reference (`pyscf_core::canonicalize_signs`, applied in `default_eig` at eig.rs:36) and runs on every diagonalization. Gradients consume the already-canonicalized mo_coeff; the gradient layer adds no new diagonalization that could re-introduce a sign flip. Accepted as inherited from the SCF trust boundary. | gsd-security-auditor (Phase 7 plan dispositions) | 2026-05-26 |

*Accepted risks do not resurface in future audit runs.*

---

## Unregistered Flags

None. All 10 `## Threat Flags` sections in the 07-*-SUMMARY.md files declare "None — no new network endpoints, auth paths, file access, or schema changes." (07-01/02/03/07/08 carry no Threat Flags section and introduce no new attack surface — only API records and gated stubs). No new attack surface appeared during implementation that lacks a threat-model mapping.

---

## Audit Notes

- **N-01 (T-07-29 — catch_unwind):** The plan named `catch_unwind` at the PyO3 boundary. The implementation has no hand-written `catch_unwind`; instead it relies on (a) PyO3's intrinsic per-function FFI boundary, which converts a Rust panic into a Python `PanicException` rather than aborting the interpreter, and (b) disciplined `?`-propagation through `pyscf_to_py` so application errors surface as proper Python exceptions, plus `#![forbid(unsafe_code)]`. The declared property (panic → exception, never abort the interpreter) is therefore present via the framework boundary. Verified CLOSED with this implementation-detail note.
- **CR-01 / CR-02 (code-review BLOCKERs — both FIXED):** CR-01 (oracle dispatch test contradicting the `--features python` arm) is fixed by routing the eight grad/geomopt oracle names to a clear `OracleError::Upstream` gated error distinct from `UnknownMethod` (runner.rs:223-240; commit 01dc1a0). CR-02 (bare `+=` in `bmatrix::g_inverse`, the live always-on geomopt path) is fixed by materializing per-k rank-1 terms and reducing via `oracle_sum` (bmatrix.rs:21-39; commit 9a1224b). Both confirmed present — directly relevant to T-07-04/07/14/23/26/36 (CR-02) and the oracle dispatch surface (CR-01).
- **Code-review WARNINGs WR-01…WR-09 (NOT threat-model items, informational):** These are correctness landmines, almost all on cintx-gated dead code (WR-01 MP2/CCSD relaxed-density discard; WR-02 vk contraction; WR-03 ECP stitch order; WR-06 GGA vxc; WR-07 grid-response no-op) that is unreachable until the cintx grad-intor workstream lands, plus convention/robustness notes (WR-04 single-write `+=`; WR-05 scanner cache key; WR-08 near-degenerate denom; WR-09 max-fold not oracle). They are NOT declared threats in the Phase-7 register and do not change any threat disposition: the reduction-order threats (T-07-04/07/14/23/26/36) concern genuine multi-term reductions, all of which route through oracle_sum/oracle_dot; the WR-09 `fold(f64::max)` is order-independent for finite values (not a determinism hazard). These remain tracked in 07-REVIEW.md for the cintx-landing follow-up; they are out of scope for this threat-mitigation audit.

---

## Security Audit Trail

| Audit Date | Threats Total | Closed | Open | Run By |
|------------|---------------|--------|------|--------|
| 2026-05-26 | 39 | 39 | 0 | gsd-security-auditor |

---

## Sign-Off

- [x] All threats have a disposition (mitigate / accept / transfer)
- [x] Accepted risks documented in Accepted Risks Log (RA-01, RA-02)
- [x] `threats_open: 0` confirmed
- [x] `status: verified` set in frontmatter

**Approval:** verified 2026-05-26
