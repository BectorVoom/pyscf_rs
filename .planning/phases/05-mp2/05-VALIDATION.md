---
phase: 5
slug: mp2
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-05-23
---

# Phase 5 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | Rust built-in `#[test]` + `pyscf-oracle` `oracle_check!` macro (subprocess-per-fixture, `Python::attach`, pyo3 dev-deps only) |
| **Config file** | none (cargo test discovery); oracle gated by `--features python` |
| **Quick run command** | `cargo test -p pyscf-mp2 -p pyscf-ao2mo` |
| **Full suite command** | `cargo test --workspace` then `cargo test -p pyscf-oracle --features python` (CI, with libpython + upstream pyscf) |
| **Estimated runtime** | ~30s quick; ~2-3min full workspace |

---

## Sampling Rate

- **After every task commit:** Run `cargo test -p pyscf-mp2 -p pyscf-ao2mo` (+ `cargo clippy -D warnings`, `cargo fmt --check`, `xtask check-dependency-wall`, `xtask check-no-fma` on touched crates)
- **After every plan wave:** Run `cargo test --workspace` + `xtask check-forbidden-paths` + `xtask check-cubecl-pin`
- **Before `/gsd:verify-work`:** Full suite green + `cargo test -p pyscf-oracle --features python` on CI (cintx-gated MP2 numeric arms green once `cintx#11` lands; structural arms always-on)
- **Max feedback latency:** ~30 seconds (quick), ~3 minutes (full)

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| ao2mo-scaffold | W0 | 0 | (infra) | — | N/A | unit | `cargo build -p pyscf-ao2mo` | ❌ W0 | ⬜ pending |
| ao2mo-transform | W1 | 1 | ao2mo general/full | — | shape/dtype validated before contraction | unit | `cargo test -p pyscf-ao2mo transform_roundtrip` | ❌ W0 | ⬜ pending |
| mp2-08-contract | W0 | 0 | MP2-08 | — | CCSD import call-site verbatim | contract | `cargo test -p pyscf-mp2 ccsd_import_contract` | ❌ W0 | ⬜ pending |
| rmp2-kernel | W1 | 1 | MP2-01 | T-5-01 | non-standard NumPy arrays to_owned; shapes validated | unit + oracle | `cargo test -p pyscf-mp2 rmp2_kernel` ; oracle `mp2_rmp2_energy` (cintx-gated) | ❌ W0 | ⬜ pending |
| ump2-kernel | W1 | 1 | MP2-02 | T-5-01 | spin-block shape validation | unit + oracle | `cargo test -p pyscf-mp2 ump2_kernel` ; oracle `mp2_ump2_energy` (cintx-gated) | ❌ W0 | ⬜ pending |
| frozen-core | W2 | 2 | MP2-03 | — | N/A | unit | `cargo test -p pyscf-mp2 frozen_core` | ❌ W0 | ⬜ pending |
| scs-factors | W2 | 2 | MP2-06 | — | N/A | unit | `cargo test -p pyscf-mp2 scs_factors` | ❌ W0 | ⬜ pending |
| make-rdm | W2 | 2 | MP2-05 | — | N/A | unit + oracle | `cargo test -p pyscf-mp2 make_rdm` ; oracle `mp2_rdm` (cintx-gated) | ❌ W0 | ⬜ pending |
| as-scanner | W2 | 2 | MP2-07 | T-5-02 | GIL detach around pure-Rust compute | unit + smoke | `cargo test -p pyscf-py mp2_scanner` (structural) | ❌ W0 | ⬜ pending |
| dfmp2-conv | W3 | 3 | MP2-04 | T-5-01 | same NumPy/shape guards; DF panic stays in Rust | unit + oracle | `cargo test -p pyscf-mp2 dfmp2` ; oracle `dfmp2_energy` (cintx-gated) | ❌ W0 | ⬜ pending |
| dfmp2-native | W3 | 3 | MP2-04 | T-5-01 | same guards | unit + oracle | `cargo test -p pyscf-mp2 dfmp2_native` ; oracle `dfmp2_native_energy` (cintx-gated) | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `crates/pyscf-ao2mo/Cargo.toml` + `src/lib.rs` skeleton — new crate scaffold (member registration + algebra/gto deps, no pyo3/cubecl)
- [ ] `crates/pyscf-mp2/tests/` — directory (currently no tests exist)
- [ ] `crates/pyscf-mp2/tests/ccsd_import_contract.rs` — MP2-08 verbatim-import contract test (mirror `cc/ccsd.py:35`)
- [ ] `crates/pyscf-mp2/tests/rmp2_structural.rs` + `ump2_structural.rs` — always-on shape/wiring + error-propagation tests (numeric separate, cintx-gated)
- [ ] `crates/pyscf-ao2mo/tests/transform_roundtrip.rs` — toy-ERI transform correctness (always-on; synthetic `nao^4` ERI, no cintx)
- [ ] `pyscf-oracle` new arms: `mp2_rmp2_energy`, `mp2_ump2_energy`, `dfmp2_energy`, `mp2_rdm` — extend `KNOWN_METHODS` + `--features python` driver, all cintx-gated
- [ ] `.github/workflows/ci.yml` — MP2 structural always-on job + MP2 numeric oracle job gated behind `cintx#11` (mirror DF-HF / DFT-01 gating)

*Existing infrastructure: `cargo test` discovery; `pyscf-oracle` macro infrastructure (Phase 3); `xtask` lint suite.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| `mf.MP2().run()` cross-module dispatch returns same energy as `mp.RMP2(mf).kernel()` | MP2-01 | Requires live Python/pyscf env + fully-wired PyO3 surface | Run `python -c "from pyscf import gto, scf, mp; mol=...; mf=scf.RHF(mol).run(); print(mf.MP2().run().e_corr, mp.RMP2(mf).kernel()[0])"` and confirm values match |
| `mf.density_fit().MP2()` routes to `DFMP2` | MP2-04 | Live Python env + cintx#11 gate | `python -c "import pyscf.mp as mp; ..."` after cintx lands |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 30s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
