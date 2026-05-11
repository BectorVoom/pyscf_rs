---
phase: 3
slug: scf-pyo3-bindings
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-05-11
---

# Phase 3 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.
> Source: `.planning/phases/03-scf-pyo3-bindings/03-RESEARCH.md` §"Validation Architecture".

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | `pytest >= 7.0` (Python side) + `cargo test` (Rust side) — both required. `h5py >= 3.10`, `numpy >= 1.26`, `maturin >= 1.12,<2.0`. |
| **Config file** | New `pyproject.toml` at repo root (maturin config + pytest deps). Existing root `pytest.ini` is upstream PySCF's test config — do NOT touch (Phase 1 D-03). |
| **Quick run command** | `cargo test -p pyscf-scf -p pyscf-diis -p pyscf-df -p pyscf-chkfile -p pyscf-core && maturin develop && pytest python/pyscf/tests/test_scf_smoke.py -x` |
| **Full suite command** | `cargo build --profile release-oracle --workspace && maturin develop --release && pytest python/pyscf/tests/ -x` |
| **Estimated runtime** | ~30 s quick · ~2 min wave · ~10 min phase gate · python3.13t and Linux/macOS µHartree matrix jobs add ~15 min on CI |

---

## Sampling Rate

- **After every task commit:** Run `cargo test -p pyscf-scf -p pyscf-diis -p pyscf-df -p pyscf-chkfile -p pyscf-core` (<30 s — Rust unit tests only)
- **After every plan wave:** Run quick command (full Rust + maturin develop + pytest smoke; <2 min)
- **Before `/gsd-verify-work`:** Full suite must be green; python3.13t job (BIND-05) and Linux/macOS µHartree matrix job (Pitfall 12) must pass
- **Max feedback latency:** 30 s per task; 120 s per wave; 600 s per phase gate

---

## Per-Task Verification Map

> Planner fills this map after PLAN.md files are generated. Every task `<verify>` block in every PLAN.md must reference one of the commands below.

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| _planner to fill_ | _XX_ | _N_ | REQ-{ID} | T-3-{N} / — | _expected secure behavior_ | unit / oracle-integration / matrix-CI / lint | _command_ | ✅ / ❌ W0 | ⬜ pending |

### Phase Requirements → Coverage Map (from RESEARCH.md §"Validation Architecture")

| Req ID | Behavior | Test Type | Automated Command | Wave 0 file |
|--------|----------|-----------|-------------------|-------------|
| SCF-01 | RHF on H2O/cc-pVDZ matches upstream ≤ 1 µHartree | oracle-integration | `pytest python/pyscf/tests/test_scf_rhf_h2o.py -x` | ❌ |
| SCF-01 | RHF on benzene/6-31G* matches upstream | oracle-integration | `pytest python/pyscf/tests/test_scf_rhf_benzene.py -x` | ❌ |
| SCF-02 | UHF on radical matches upstream | oracle-integration | `pytest python/pyscf/tests/test_scf_uhf.py -x` | ❌ |
| SCF-03 | GHF on H2 runs to completion | unit | `pytest python/pyscf/tests/test_scf_ghf.py -x` | ❌ |
| SCF-04 | C-DIIS converges in upstream iteration count ±1 | oracle-integration | `pytest python/pyscf/tests/test_scf_diis.py -x` | ❌ |
| SCF-05 | Each of 5 init_guess modes matches upstream first-iter density | oracle-integration | `pytest python/pyscf/tests/test_scf_init_guess.py -x` | ❌ |
| SCF-06 | `level_shift`, `damp`, `max_cycle`, `conv_tol`, `conv_tol_grad` semantics match upstream | unit + oracle | `pytest python/pyscf/tests/test_scf_controls.py -x` | ❌ |
| SCF-07 | `mf.density_fit().kernel()` matches upstream | oracle-integration | `pytest python/pyscf/tests/test_scf_df.py -x` | ❌ |
| SCF-08 | Subclass override `get_veff` invoked once per cycle | unit (Python) | `pytest python/pyscf/tests/test_scf_override_dispatch.py -x` | ❌ |
| SCF-09 | `mf.analyze()`/`mulliken_pop`/`mulliken_meta`/`dip_moment` match upstream | oracle | `pytest python/pyscf/tests/test_scf_analyze.py -x` | ❌ |
| SCF-10 | h5py-schema chkfile round-trip works | oracle | `pytest python/pyscf/tests/test_scf_chkfile.py -x` | ❌ (covers ORACLE-08) |
| SCF-11 | `mf.to_uhf()` / `to_rhf()` / `to_ghf()` work; `to_uks()`/`to_rks()` raise NotYetImplemented{phase:4} | unit | `pytest python/pyscf/tests/test_scf_cross_dispatch.py -x` | ❌ |
| SCF-12 | `mf.as_scanner()(mol2)` returns energy | unit | `pytest python/pyscf/tests/test_scf_scanner.py -x` | ❌ |
| SCF-13 | `canonicalize_signs` is largest-\|c\|-lowest-index-flip | unit (Rust) | `cargo test -p pyscf-core canonicalize_signs` | ❌ |
| SCF-13 | Linux x86_64 + macOS aarch64 µHartree assertion on H2O/cc-pVDZ | matrix-CI | GitHub Actions matrix job `xplat-uhartree` | ❌ (covers Pitfall 12) |
| SCF-14 | ≥ 30-attribute floor introspectable from Python | unit | `pytest python/pyscf/tests/test_scf_attributes.py -x` | ❌ |
| BIND-01 | `maturin develop` produces importable `pyscf._native` | CI smoke | `maturin develop && python -c 'from pyscf._native import scf'` | ❌ |
| BIND-02 | `from pyscf import scf` resolves to overlay `_native.scf` | unit (Python) | `pytest python/pyscf/tests/test_overlay_resolution.py -x` | ❌ |
| BIND-04 | Stride-fuzz: `a` / `a.T` / `a[::2]` / `a[:, 1:5]` all return identical answers | unit (Python) | `pytest python/pyscf/tests/test_scf_stride_fuzz.py -x` | ❌ |
| BIND-05 | python3.13t SCF smoke runs without deadlock | matrix-CI | GitHub Actions job `python313t-smoke` (NON-abi3 build — see Pitfall (new) in research) | ❌ |
| BIND-06 | No `lazy_static!` in `pyscf-py`; `PyOnceLock` used for caches | lint | `cargo run -p xtask -- lint forbid-lazy-static` | ❌ |
| BIND-07 | Subclass `get_veff` override invoked ≥ 1× per SCF cycle (re-shape of SCF-08) | unit (Python) | (same as SCF-08) | (same) |
| BIND-09 | Rust panic / convergence failure raises `PyscfRsError(kind='ConvergenceFailure')` preserving `__cause__` chain | unit (Python) | `pytest python/pyscf/tests/test_panic_to_exception.py -x` | ❌ |
| ORACLE-02 | `oracle_check!(method, tolerance, fixture)` macro invokable from `pyscf-oracle` dev-deps | unit (Rust) | `cargo test -p pyscf-oracle oracle_check` | ❌ |
| ORACLE-08 | chkfile round-trip both directions on H2O/cc-pVDZ | oracle | (same as SCF-10) | (same) |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

All wave-0 files are MISSING before Phase 3 execution starts. The planner must schedule a Wave 0 plan that creates these stubs so subsequent waves can assert green.

### Python test stubs (19 files)
- [ ] `python/pyscf/tests/__init__.py` — empty
- [ ] `python/pyscf/tests/conftest.py` — shared fixtures (H2O, benzene, water-trimer Mole)
- [ ] `python/pyscf/tests/test_scf_smoke.py` — wave-level smoke entry
- [ ] `python/pyscf/tests/test_scf_rhf_h2o.py` — SCF-01
- [ ] `python/pyscf/tests/test_scf_rhf_benzene.py` — SCF-01
- [ ] `python/pyscf/tests/test_scf_uhf.py` — SCF-02
- [ ] `python/pyscf/tests/test_scf_ghf.py` — SCF-03
- [ ] `python/pyscf/tests/test_scf_diis.py` — SCF-04
- [ ] `python/pyscf/tests/test_scf_init_guess.py` — SCF-05 (5 modes)
- [ ] `python/pyscf/tests/test_scf_controls.py` — SCF-06
- [ ] `python/pyscf/tests/test_scf_df.py` — SCF-07
- [ ] `python/pyscf/tests/test_scf_override_dispatch.py` — SCF-08 / BIND-07
- [ ] `python/pyscf/tests/test_scf_analyze.py` — SCF-09
- [ ] `python/pyscf/tests/test_scf_chkfile.py` — SCF-10 / ORACLE-08
- [ ] `python/pyscf/tests/test_scf_cross_dispatch.py` — SCF-11
- [ ] `python/pyscf/tests/test_scf_scanner.py` — SCF-12
- [ ] `python/pyscf/tests/test_scf_attributes.py` — SCF-14
- [ ] `python/pyscf/tests/test_overlay_resolution.py` — BIND-02
- [ ] `python/pyscf/tests/test_scf_stride_fuzz.py` — BIND-04
- [ ] `python/pyscf/tests/test_panic_to_exception.py` — BIND-09

### Rust test stubs (2 files)
- [ ] `crates/pyscf-oracle/src/lib.rs` — `oracle_check!` macro module (ORACLE-02)
- [ ] `crates/pyscf-oracle/tests/chkfile_roundtrip.rs` — ORACLE-08 macro invocation tests

### Infrastructure
- [ ] `pyproject.toml` at repo root — maturin config + pytest deps (`pytest>=7.0`, `h5py>=3.10`, `numpy>=1.26`, `maturin>=1.12,<2.0`)
- [ ] `python/pyscf/__init__.py` — overlay re-export shim (BIND-02)
- [ ] `crates/pyscf-core/src/lib.rs` extended with `pub fn canonicalize_signs` (SCF-13)
- [ ] `.github/workflows/ci.yml` — extends with **four** new jobs:
      - (a) `maturin-smoke` (BIND-01)
      - (b) `stride-fuzz` (BIND-04)
      - (c) `xplat-uhartree` Linux x86_64 + macOS aarch64 matrix job (Pitfall 12 mitigation, SCF-13 cross-validation)
      - (d) `python313t-smoke` (BIND-05) — runs a **separate non-abi3 build** of `pyscf-py` (per RESEARCH.md §"CRITICAL abi3 vs free-threaded ABI conflict": abi3-py310 wheels are incompatible with 3.13t)
- [ ] `xtask/src/lints/algebra_wall.rs` — extend allowlist with `pyscf-chkfile` (no algebra), `pyscf-diis` (algebra only), `pyscf-df` (algebra + gto)
- [ ] `xtask/src/lints/forbid_lazy_static.rs` (new) — BIND-06 lint module

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| `python3.13t` wheel install on developer machine | BIND-05 | Free-threaded interpreter is opt-in; not yet ubiquitous in CI runners | Document in phase SUMMARY: `pyenv install 3.13t-dev && pyenv local 3.13t-dev && maturin develop --features "free-threading-non-abi3"` |

*All other phase behaviors have automated verification.*

---

## Validation Sign-Off

- [ ] All tasks have `<verify>` blocks pointing to a command in this map (planner must enforce)
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 plan creates ALL files listed above before Wave 1 starts
- [ ] No watch-mode flags (CI must be one-shot)
- [ ] Feedback latency < 30 s per task, < 120 s per wave
- [ ] python3.13t job is a NON-abi3 build (separate `[features]` configuration in `pyscf-py/Cargo.toml`)
- [ ] Linux x86_64 + macOS aarch64 µHartree matrix job exists for Pitfall 12 mitigation
- [ ] `nyquist_compliant: true` set in frontmatter after planner fills the per-task map

**Approval:** pending

---

*Phase: 03-scf-pyo3-bindings*
*Created: 2026-05-11 via /gsd-plan-phase step 5.5 (Nyquist validation gate)*
