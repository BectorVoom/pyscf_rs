---
phase: 7
slug: gradients-geomopt
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-05-26
---

# Phase 7 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.
> Derived from 07-RESEARCH.md §"Validation Architecture". The PRIMARY always-on
> numeric gate is **FD self-verification** (`verify_fd`, D-01) — it needs NO upstream
> PySCF. Upstream byte-identity (≤1e-7 Ha/Bohr) + trajectory parity + the
> `pip uninstall geometric pyberny` no-runtime-dep proof are `workflow_dispatch` /
> human-verify arms (D-01/D-05), per the established 02-10 / 05-08 / 06-11 precedent.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | Rust `cargo test` (per-crate `tests/*.rs`) + `pyscf-oracle` fixtures (`KNOWN_METHODS`) + pytest (`pytest.ini`, `--import-mode=importlib`) for the Python drop-in arm |
| **Config file** | `pytest.ini` (Python arm); per-crate `Cargo.toml` `[dev-dependencies]` (Rust arm); `.github/workflows/ci.yml` (gate wiring) |
| **Quick run command** | `cargo test -p pyscf-grad` / `cargo test -p pyscf-geomopt` (scoped — MUST NOT pull `libxc_rs` into the dep graph; ~6h compile) |
| **Full suite command** | `cargo test -p pyscf-grad -p pyscf-geomopt -p pyscf-oracle` (scoped); upstream byte-identity is `workflow_dispatch` only |
| **Estimated runtime** | ~60–120 s scoped (FD harness on small molecules + H2O optimizer convergence); upstream/trajectory arms excluded from the daily gate |

---

## Sampling Rate

- **After every task commit:** Run `cargo test -p pyscf-grad` / `cargo test -p pyscf-geomopt` (FD-structural + optimizer-convergence; no libxc, no upstream).
- **After every plan wave:** Run `cargo test -p pyscf-grad -p pyscf-geomopt -p pyscf-oracle` + the `algebra_wall` dependency-wall lint + the no-FMA lint.
- **Before `/gsd:verify-work`:** Full scoped suite must be green.
- **Phase gate / human-verify close-out:** `workflow_dispatch` arms — upstream ≤1e-7 Ha/Bohr byte-identity, optimizer trajectory parity, and the `pip uninstall geometric pyberny` no-runtime-dep proof.
- **Max feedback latency:** ~120 s.

---

## Per-Task Verification Map

> Task IDs are assigned at plan time (gsd-planner) and reconciled by the Nyquist
> auditor. The rows below seed the map from the RESEARCH.md Req→Test matrix; the
> `Task ID` column is filled once PLAN.md files exist.

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| TBD | TBD | 0 | (cintx grad-intor workstream — un-gate prerequisite) | — | N/A | smoke | cintx round-trip vs libcint | ❌ W0 | ⬜ pending |
| TBD | TBD | — | GRAD-09 | — | N/A | gate (FD harness, central diff, per-atom/component, Bohr) | `cargo test -p pyscf-grad verify_fd` | ❌ W0 | ⬜ pending |
| TBD | TBD | — | GRAD-10 | — | N/A | structural (single-CPHF-impl assertion) | `cargo test -p pyscf-grad single_cphf_impl` | ❌ W0 | ⬜ pending |
| TBD | TBD | — | GRAD-08 | — | N/A | structural (`atmlst` row subset) | `cargo test -p pyscf-grad atmlst_subset` | ❌ W0 | ⬜ pending |
| TBD | TBD | — | GRAD-01 | — | N/A | unit (FD-gated; upstream byte = workflow_dispatch) | `cargo test -p pyscf-grad rhf_verify_fd` | ❌ W0 | ⬜ pending |
| TBD | TBD | — | GRAD-02 | — | N/A | unit (FD) | `cargo test -p pyscf-grad uhf_verify_fd` | ❌ W0 | ⬜ pending |
| TBD | TBD | — | GRAD-03 | — | N/A | unit (FD; `grid_response`) | `cargo test -p pyscf-grad rks_verify_fd` | ❌ W0 | ⬜ pending |
| TBD | TBD | — | GRAD-04 | — | N/A | unit (FD) | `cargo test -p pyscf-grad uks_verify_fd` | ❌ W0 | ⬜ pending |
| TBD | TBD | — | GRAD-05 | — | N/A | unit (FD; Z-vector via CPHF) | `cargo test -p pyscf-grad mp2_verify_fd` | ❌ W0 | ⬜ pending |
| TBD | TBD | — | GRAD-06 | — | N/A | unit (FD; Λ + Z-vector, consumes Phase-6 λ) | `cargo test -p pyscf-grad ccsd_verify_fd` | ❌ W0 | ⬜ pending |
| TBD | TBD | — | GRAD-07 | — | N/A | unit (FD; `ecp_ipnuc` ready, `iprinv` gated) | `cargo test -p pyscf-grad ecp_verify_fd` | ❌ W0 | ⬜ pending |
| TBD | TBD | — | GEOMOPT-04 | — | unbounded `maxsteps`/`max_cycle` capped | unit (constant assertion = geomeTRIC GAU defaults) | `cargo test -p pyscf-geomopt conv_defaults` | ❌ W0 | ⬜ pending |
| TBD | TBD | — | GEOMOPT-06 | — | N/A | unit (Wilson-B vs hand-calc; RFO step; neg-eig) | `cargo test -p pyscf-geomopt bmatrix rfo_step` | ❌ W0 | ⬜ pending |
| TBD | TBD | — | GEOMOPT-07 | — | N/A | integration (self-contained always-on: H2O→equilibrium) | `cargo test -p pyscf-geomopt h2o_equilibrium` | ❌ W0 | ⬜ pending |
| TBD | TBD | — | GEOMOPT-05 | — | N/A | unit (HDF5 checkpoint round-trip / resume) | `cargo test -p pyscf-geomopt checkpoint_resume` | ❌ W0 | ⬜ pending |
| TBD | TBD | — | GEOMOPT-02, GEOMOPT-03 | — | `constraints` kwarg → clear error, never silent no-op | structural (import + call-sig parity) + python smoke | `cargo test -p pyscf-geomopt shim_parity` | ❌ W0 | ⬜ pending |
| TBD | TBD | — | GEOMOPT-01 | — | N/A | CI proof (no `geometric`/`pyberny` runtime dep) | `pip uninstall -y geometric pyberny && python -c "import pyscf.geomopt; pyscf.geomopt.optimize(mf)"` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] **cintx gradient-integral workstream** — lands the 5 absent core families (`int2e_ip1`, `int1e_ip{ovlp,kin,nuc,rinv}`) + `ECPscalar_iprinv` + the rinv-origin-shift parameter. **BLOCKS numeric un-gating** (not a pyscf_rs test file but a hard prerequisite; per RESEARCH.md the CONTEXT D-02 "expected case" does NOT hold — only `int3c2e_ip1` + `int1e_ecp_ipnuc` are cintx-ready).
- [ ] `crates/pyscf-grad/tests/verify_fd.rs` — the FD harness (GRAD-09, D-01); gates all of GRAD-01..07
- [ ] `crates/pyscf-grad/tests/cphf.rs` — single-CPHF structural assertion (GRAD-10)
- [ ] `crates/pyscf-grad/tests/atmlst.rs` — `atmlst` subsetting (GRAD-08)
- [ ] `crates/pyscf-geomopt/tests/h2o_equilibrium.rs` — self-contained convergence gate (D-05 / GEOMOPT-07)
- [ ] `crates/pyscf-geomopt/tests/bmatrix.rs` + `rfo.rs` + `conv_defaults.rs` (GEOMOPT-04/06)
- [ ] `crates/pyscf-oracle` grad fixtures — `nuc_grad_*` method names (register-but-defer-dispatch, mirrors MP2/CCSD precedent); byte-identity arms `#[ignore]`'d / `workflow_dispatch`
- [ ] `.github/workflows/ci.yml` — FD always-on grad gates + self-contained geomopt gate + `workflow_dispatch` upstream/trajectory/pip-uninstall arms

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Upstream byte-identity ≤1e-7 Ha/Bohr vs `pyscf/grad/*` | GRAD-01..07 success-criterion bar | Sandbox cannot run upstream PySCF (02-10 / 05-08 / 06-11 precedent) | `workflow_dispatch` arm: install upstream pyscf, diff analytical grad vs `pyscf/grad/*` to ≤1e-7 |
| Optimizer trajectory / stationary-point parity vs `geometric_solver` | GEOMOPT-07 | geomeTRIC not importable in sandbox; bar is "same stationary point within chemical accuracy", not bit-for-bit | `workflow_dispatch` arm: install geomeTRIC, compare converged geometry within chemical accuracy |
| No-runtime-dep proof | GEOMOPT-01 | Requires a clean env with `geometric`/`pyberny` uninstalled | CI: `pip uninstall -y geometric pyberny && python -c "import pyscf.geomopt; pyscf.geomopt.optimize(mf)"` succeeds |

*All daily-gate behaviors have automated FD/structural verification; only the upstream/trajectory cross-checks are deferred to `workflow_dispatch`.*

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references (incl. the cintx grad-intor workstream prerequisite)
- [ ] No watch-mode flags
- [ ] Feedback latency < 120s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
