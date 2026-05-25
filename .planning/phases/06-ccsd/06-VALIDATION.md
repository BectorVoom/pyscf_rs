---
phase: 06
slug: ccsd
status: draft
nyquist_compliant: true
wave_0_complete: true
created: 2026-05-24
---

# Phase 06 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.
> Derived from `06-RESEARCH.md` §"Validation Architecture". CCSD is the project's
> heaviest phase; the per-task map below is the requirement-level contract — exact
> task IDs are bound during planning and Wave 0.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | Rust built-in `#[test]` + `cargo test`; `pyscf-oracle` `oracle_check!` macro for energy fixtures; `--profile release-oracle` for bit-exact arms |
| **Config file** | none (Cargo workspace) — test targets live in `crates/pyscf-ccsd/tests/` + `crates/pyscf-oracle/tests/` |
| **Quick run command** | `cargo test -p pyscf-ccsd --locked` |
| **Full suite command** | `cargo test --workspace --locked` |
| **Estimated runtime** | quick ~seconds (small-system arms only); upstream/caffeine/DF-spill arms are CI/human-verify (`workflow_dispatch`), NOT always-on |

> ⚠ **User-memory constraint:** never trigger any cargo command that pulls `libxc_rs` into the dep graph (~6h compile). `pyscf-ccsd` deps must stay clear of it. Heavy fixtures (caffeine, DF-spill, `python3.13t` corpus, upstream byte-identity) run on a `workflow_dispatch`/human-verify arm only — do not make them always-on.

---

## Sampling Rate

- **After every task commit:** `cargo test -p pyscf-ccsd --locked` (fast structural + small-system arms; no upstream, no caffeine)
- **After every plan wave:** `cargo test --workspace --locked` + the `release-oracle` determinism arms (`-p pyscf-algebra --test oracle_determinism`, both rayon-1 and rayon-8) to guard Pitfall 1/2 on the new contractions
- **Before `/gsd:verify-work` (phase gate):** full workspace green + the `workflow_dispatch` human-verify arm run once manually (upstream byte-identity CCSD energy on small + caffeine, DF-CCSD spill proof on constrained `PYSCF_MAX_MEMORY`, λ/RDM byte-identity) + the `python3.13t` CCSD smoke (no GIL deadlock)
- **Max feedback latency:** quick arm in seconds; heavy arms deferred to CI/human-verify by design

---

## Per-Requirement Verification Map

> Task IDs (`06-PP-TT`) are assigned by the planner; this is the requirement→test contract the per-task map must satisfy.

| Requirement | Behavior | Test Type | Automated Command | File Exists |
|-------------|----------|-----------|-------------------|-------------|
| CCSD-01 | in-core RCCSD energy ≤1 µHartree (small system, always-on) | oracle (in-tree, small) | `cargo test -p pyscf-ccsd --test rccsd_numeric_smoke -- --test-threads=1` | ❌ W0 |
| CCSD-01 | RCCSD byte-identity vs upstream (caffeine, gated) | oracle (human-verify) | `cargo test -p pyscf-oracle --features python -- ccsd_rccsd` (`workflow_dispatch`) | ❌ W0 (CI arm) |
| CCSD-02 | UCCSD energy (small open-shell, structural always-on) | structural + oracle | `cargo test -p pyscf-ccsd --test uccsd_smoke` | ❌ W0 |
| CCSD-03 | T1/T2 dual convergence (energy target) | unit | `cargo test -p pyscf-ccsd convergence` | ❌ W0 |
| CCSD-04 | amplitude-DIIS converges in upstream iter count; vector packing byte-matches | unit (packing) + integration (iter count) | `cargo test -p pyscf-ccsd diis_amps` | ❌ W0 |
| CCSD-05 | `solve_lambda` λ amplitudes match upstream | oracle (small + human-verify) | `cargo test -p pyscf-ccsd lambda` / oracle arm | ❌ W0 |
| CCSD-06 | `make_rdm1`/`make_rdm2` (incl. `ao_repr`) match upstream | oracle | `cargo test -p pyscf-ccsd rdm` / oracle arm | ❌ W0 |
| CCSD-07 | AO-direct `direct=True` matches in-core | integration | `cargo test -p pyscf-ccsd direct` | ❌ W0 |
| CCSD-08 | DF-CCSD bounded memory; spills `Wabef` to HDF5 over budget; no leftover scratch | integration (small) + human-verify (benzene-dimer spill) | `cargo test -p pyscf-ccsd dfccsd_spill` / `workflow_dispatch` | ❌ W0 |
| CCSD-09 | `t1diagnostic`/`d1diagnostic` values match upstream | unit | `cargo test -p pyscf-ccsd diagnostics` | ❌ W0 |
| CCSD-10 | frozen `int`/`list`/`'auto'` match MP2 | unit (reuse MP2 helpers) | `cargo test -p pyscf-ccsd frozen` | ❌ W0 |
| CCSD-11 | `Wabef` allocated once across N iterations; over-budget in-core HARD-refuses | integration (alloc count) + unit (refusal) | `cargo test -p pyscf-ccsd --test heap_alloc_count` / `try_reserve` unit | ❌ W0 (needs counting allocator) |

*Status legend: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `crates/pyscf-ccsd/tests/rccsd_numeric_smoke.rs` — covers CCSD-01 (small-system in-core, always-on)
- [ ] `crates/pyscf-ccsd/tests/uccsd_smoke.rs` — covers CCSD-02
- [ ] `crates/pyscf-ccsd/tests/diis_amps.rs` — covers CCSD-04 (vector packing + iter count)
- [ ] `crates/pyscf-ccsd/tests/diagnostics.rs` — covers CCSD-09
- [ ] `crates/pyscf-ccsd/tests/frozen.rs` — covers CCSD-10 (reuse MP2 helper fixtures)
- [ ] `crates/pyscf-ccsd/tests/dfccsd_spill.rs` — covers CCSD-08 (spill + no-leftover-scratch)
- [ ] `crates/pyscf-ccsd/tests/direct.rs` — covers CCSD-07
- [ ] `crates/pyscf-ccsd/tests/heap_alloc_count.rs` — covers CCSD-11 (DEDICATED target with its own counting `#[global_allocator]`, NOT linked to oracle/determinism binaries)
- [ ] `crates/pyscf-ccsd/tests/refusal.rs` — covers CCSD-11 pre-flight `MemoryLimitExceeded`
- [ ] `crates/pyscf-oracle/tests/*` — CCSD energy/λ/RDM byte-identity fixtures (small always-on; caffeine/DF-spill gated behind `--features python` + `workflow_dispatch`, mirroring `mp2-oracle-upstream-manual` ci.yml:445)
- [ ] `.github/workflows/ci.yml` — `ccsd-structural`/`ccsd-oracle` always-on small arm; `ccsd-oracle-upstream-manual` (`workflow_dispatch`, caffeine + DF-spill + λ/RDM byte-identity); `python3.13t` CCSD smoke; heap-alloc-count gate
- [ ] `xtask/src/bin/check_no_fma.rs` — add `("pyscf-ccsd","pyscf_ccsd")` to `SCAN_TARGETS`

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Upstream byte-identity CCSD energy (caffeine/cc-pVDZ) | CCSD-01 | Sandbox can't install/run upstream PySCF; caffeine job is multi-GB / slow | Run `ccsd-oracle-upstream-manual` `workflow_dispatch` arm; compare to upstream reference fixture |
| DF-CCSD `Wabef`→HDF5 spill proof on constrained `PYSCF_MAX_MEMORY` (benzene-dimer/cc-pVDZ) | CCSD-08 | Requires a deliberately constrained memory budget + large system; not viable always-on | Run gated `dfccsd_spill` arm with a low `PYSCF_MAX_MEMORY`; assert spill file created+deleted, energy correct |
| λ / RDM byte-identity vs upstream | CCSD-05, CCSD-06 | Needs upstream PySCF reference values | `workflow_dispatch` oracle arm with `--features python` |
| `python3.13t` free-threaded CCSD smoke (no GIL deadlock) | (Pitfall 6, cross-cutting) | Free-threaded interpreter not in default sandbox; heaviest GIL re-validation in the project | Clone the `python3.13t SCF smoke` CI job; run CCSD corpus under `python3.13t` |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency acceptable (quick arm in seconds)
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
