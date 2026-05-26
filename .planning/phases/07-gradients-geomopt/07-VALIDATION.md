---
phase: 7
slug: gradients-geomopt
status: complete
nyquist_compliant: true
wave_0_complete: true
created: 2026-05-26
closed: 2026-05-26
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
> `Task ID` column is filled now that all PLAN.md files (07-01..07-10) +
> SUMMARYs exist. The `Task ID` column is the owning `<plan>.<task>`; the always-on
> arms are green (07-10 close-out), the upstream byte-identity arms remain
> `workflow_dispatch` (gated on the cintx grad-intor workstream, 07-01).

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 07-01.1 | 07-01 | 0 | (cintx grad-intor workstream — un-gate prerequisite) | — | N/A | smoke | cintx round-trip vs libcint (`grad_intor_smoke.rs`) | ✅ | ⚠️ 2/8 ready, 6/8 cintx-MISSING (gated) |
| 07-02.2 | 07-02 | 1 | GRAD-09 | — | N/A | gate (FD harness, central diff, per-atom/component, Bohr) | `cargo test -p pyscf-grad verify_fd` | ✅ | ✅ green |
| 07-02.3 | 07-02 | 1 | GRAD-10 | — | N/A | structural (single-CPHF-impl assertion) | `cargo test -p pyscf-grad --test cphf single_cphf_impl` | ✅ | ✅ green |
| 07-02.1 | 07-02 | 1 | GRAD-08 | — | N/A | structural (`atmlst` row subset) | `cargo test -p pyscf-grad --test atmlst` | ✅ | ✅ green |
| 07-03.x | 07-03 | 2 | GRAD-01 | — | N/A | unit (FD-gated; upstream byte = workflow_dispatch) | `cargo test -p pyscf-grad --test rhf_verify_fd` | ✅ | ✅ green (FD); upstream gated |
| 07-05.x | 07-05 | 3 | GRAD-02 | — | N/A | unit (FD) | `cargo test -p pyscf-grad --test uhf_verify_fd` | ✅ | ✅ green (FD); upstream gated |
| 07-05.x | 07-05 | 3 | GRAD-03 | — | N/A | unit (FD; `grid_response`) | `cargo test -p pyscf-grad --test rks_verify_fd` | ✅ | ✅ green (FD); upstream gated |
| 07-05.x | 07-05 | 3 | GRAD-04 | — | N/A | unit (FD) | `cargo test -p pyscf-grad --test uks_verify_fd` | ✅ | ✅ green (FD); upstream gated |
| 07-06.x | 07-06 | 4 | GRAD-05 | — | N/A | unit (FD; Z-vector via CPHF) | `cargo test -p pyscf-grad --test mp2_verify_fd` | ✅ | ✅ green (FD); upstream gated |
| 07-08.x | 07-08 | 5 | GRAD-06 | — | N/A | unit (FD; Λ + Z-vector, consumes Phase-6 λ) | `cargo test -p pyscf-grad --test ccsd_verify_fd` | ✅ | ✅ green (FD); upstream gated |
| 07-04.x | 07-04 | 2 | GRAD-07 | — | N/A | unit (FD; `ecp_ipnuc` ready, `iprinv` gated) | `cargo test -p pyscf-grad --test ecp_verify_fd` | ✅ | ✅ green (FD); `iprinv` cintx-gated |
| 07-04.x | 07-04 | 2 | GEOMOPT-04 | — | unbounded `maxsteps`/`max_cycle` capped | unit (constant assertion = geomeTRIC GAU defaults) | `cargo test -p pyscf-geomopt --test conv_defaults` | ✅ | ✅ green |
| 07-04.x | 07-04 | 2 | GEOMOPT-06 | — | N/A | unit (Wilson-B vs hand-calc; RFO step; neg-eig) | `cargo test -p pyscf-geomopt --test bmatrix --test rfo` | ✅ | ✅ green |
| 07-04.x | 07-04 | 2 | GEOMOPT-07 | — | N/A | integration (self-contained always-on: H2O→equilibrium) | `cargo test -p pyscf-geomopt --test h2o_equilibrium` | ✅ | ✅ green |
| 07-04.x | 07-04 | 2 | GEOMOPT-05 | — | N/A | unit (HDF5 checkpoint round-trip / resume) | `cargo test -p pyscf-geomopt --test checkpoint_resume` | ✅ | ✅ green |
| 07-09.2 | 07-09 | 8 | GEOMOPT-02, GEOMOPT-03 | — | `constraints` kwarg → clear error, never silent no-op | structural (import + call-sig parity) + python smoke | `cargo test -p pyscf-geomopt --test shim_parity` | ✅ | ✅ green |
| 07-10.2 | 07-10 | 8 | GEOMOPT-01 | T-07-38 | N/A | CI proof (no `geometric`/`pyberny` runtime dep) | `pip uninstall -y geometric pyberny && python -c "import pyscf.grad; import pyscf.geomopt; pyscf.geomopt.optimize(mf)"` | ✅ | ✅ green (`geomopt-no-runtime-dep` CI job) |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky/gated*

---

## Wave 0 Requirements

- [~] **cintx gradient-integral workstream** — lands the 5 absent core families (`int2e_ip1`, `int1e_ip{ovlp,kin,nuc,rinv}`) + `ECPscalar_iprinv` + the rinv-origin-shift parameter. **BLOCKS numeric un-gating** (not a pyscf_rs test file but a hard prerequisite; per RESEARCH.md the CONTEXT D-02 "expected case" does NOT hold — only `int3c2e_ip1` + `int1e_ecp_ipnuc` are cintx-ready). **STATUS (07-01): 2/8 ready, 6/8 MISSING with NO scheduled cintx workstream — the upstream byte-identity arms stay `workflow_dispatch`-gated; the always-on FD/structural gates proceed regardless (D-01).**
- [x] `crates/pyscf-grad/tests/verify_fd.rs` (+ `rhf/uhf/rks/uks/mp2/ccsd/ecp_verify_fd.rs`) — the FD harness (GRAD-09, D-01); gates all of GRAD-01..07
- [x] `crates/pyscf-grad/tests/cphf.rs` — single-CPHF structural assertion (GRAD-10)
- [x] `crates/pyscf-grad/tests/atmlst.rs` — `atmlst` subsetting (GRAD-08)
- [x] `crates/pyscf-geomopt/tests/h2o_equilibrium.rs` — self-contained convergence gate (D-05 / GEOMOPT-07)
- [x] `crates/pyscf-geomopt/tests/bmatrix.rs` + `rfo.rs` + `conv_defaults.rs` (GEOMOPT-04/06)
- [x] `crates/pyscf-oracle/src/grad_oracle.rs` grad fixtures — `nuc_grad_*` + `geomopt_h2o` method names (register-but-defer-dispatch, mirrors MP2/CCSD precedent; KNOWN_METHODS 24→32); byte-identity arms `#[cfg(feature="python")]`/`#[ignore]`'d / `workflow_dispatch` (07-10)
- [x] `.github/workflows/ci.yml` — FD always-on `grad-structural` gate + self-contained geomopt gate + the `geomopt-no-runtime-dep` GEOMOPT-01 CI proof + the `workflow_dispatch` `grad-oracle-upstream-manual` upstream byte-identity / trajectory arms (07-10)

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

- [x] All tasks have `<automated>` verify or Wave 0 dependencies
- [x] Sampling continuity: no 3 consecutive tasks without automated verify
- [x] Wave 0 covers all MISSING references (incl. the cintx grad-intor workstream prerequisite — recorded as a `workflow_dispatch`-gated prerequisite, not blocking the always-on gate)
- [x] No watch-mode flags
- [x] Feedback latency < 120s (scoped `cargo test -p pyscf-grad -p pyscf-geomopt -p pyscf-oracle` runs in ~10s on the always-on path)
- [x] `nyquist_compliant: true` set in frontmatter

**Approval:** approved (07-10 close-out, 2026-05-26)

> **Nyquist coverage summary.** Every daily-gate behaviour has an automated
> always-on verify: the FD self-verification numeric gate (`verify_fd`, D-01,
> ≤1e-6 Ha/Bohr — the first always-on NUMERIC gate in the project, needs NO
> upstream PySCF) covers GRAD-01..07; the structural arms cover GRAD-08
> (`atmlst`) + GRAD-10 (`single_cphf_impl`); the self-contained H2O→equilibrium
> convergence + bmatrix/rfo/conv_defaults cover GEOMOPT-04/06/07; the
> `geomopt-no-runtime-dep` CI job (`pip uninstall -y geometric pyberny`) proves
> GEOMOPT-01; GEOMOPT-02/03 (the Python optimize entry point) ride `shim_parity`.
> The pyscf-oracle `grad_oracle.rs` registration arms prove the eight
> `nuc_grad_*`/`geomopt_h2o` names dispatch (always-on, no python, no libxc).
>
> **What stays gated on the cintx grad-intor workstream (07-01, D-02).** The
> upstream byte-identity analytical-gradient numerics (≤1e-7 Ha/Bohr vs
> `pyscf/grad/*`) + the geomopt trajectory/stationary-point parity vs
> `geometric_solver` are `workflow_dispatch`-only (the `grad-oracle-upstream-manual`
> arm), because (1) the sandbox cannot run upstream PySCF/geomeTRIC (the 02-10 /
> 05-08 / 06-11 precedent) AND (2) six of the eight gradient-integral families
> (`int2e_ip1`, `int1e_ip{ovlp,kin,nuc,rinv}`, `ECPscalar_iprinv` + the
> `with_rinv_at_nucleus` origin shift) are MISSING from cintx today with no
> scheduled workstream. Only `int3c2e_ip1` (DF-grad) and `int1e_ecp_ipnuc`
> (= `ECPscalar_ipnuc`, the ECP `get_hcore` term) are cintx-ready. These arms
> un-gate when the cintx grad-intor workstream lands the missing families
> (analogous to the int2e / d-shell-Rys workstream in project memory).
