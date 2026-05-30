---
phase: 3
slug: scf-pyo3-bindings
status: complete
nyquist_compliant: true
wave_0_complete: true
created: 2026-05-11
audited: 2026-05-26
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

> **Post-execution audit (2026-05-26).** This map was reconciled against the shipped tree during `/gsd:validate-phase 3`. Every requirement has an automated verification command and a test file that exists on disk. Statuses below reflect what runs green in-sandbox (Rust `cargo test`) versus what is sampled in CI but requires the Python/maturin toolchain to assert (the `human_needed` items in `03-VERIFICATION.md`).

### Phase Requirements → Coverage Map (reconciled 2026-05-26)

| Req ID | Behavior | Test Type | Automated Command | File | Status |
|--------|----------|-----------|-------------------|------|--------|
| SCF-01 | RHF on H2O/cc-pVDZ matches upstream ≤ 1 µHartree | oracle-integration (CI) | `pytest python/pyscf/tests/test_scf_rhf_h2o.py -x` · `test_scf_rhf_ccpvdz.py` | ✅ | ⚠️ CI-gated |
| SCF-01 | RHF on benzene/6-31G* matches upstream | oracle-integration (CI) | `pytest python/pyscf/tests/test_scf_rhf_benzene.py -x` | ✅ | ⚠️ CI-gated |
| SCF-02 | UHF on radical matches upstream | oracle-integration (CI) | `pytest python/pyscf/tests/test_scf_uhf.py -x` | ✅ | ⚠️ CI-gated |
| SCF-03 | GHF on H2 runs to completion | unit (CI) | `pytest python/pyscf/tests/test_scf_ghf.py -x` | ✅ | ⚠️ CI-gated |
| SCF-04 | C-DIIS converges in upstream iteration count ±1 | oracle-integration + Rust | `cargo test -p pyscf-scf --test diis_adapter_wiring` · `pytest .../test_scf_diis.py -x` | ✅ | ✅ green (Rust) |
| SCF-05 | All 5 init_guess modes (minao/1e/chkfile/dm0 + atom/huckel) | oracle-integration + Rust | `cargo test -p pyscf-scf --test init_guess_atom_huckel` · `--test init_guess_minao` · `--test init_guess_chkfile` · `pytest .../test_scf_init_guess.py -x` | ✅ | ✅ green (Rust) |
| SCF-06 | `level_shift`/`damp`/`max_cycle`/`conv_tol`/`conv_tol_grad` semantics | unit + oracle | `cargo test -p pyscf-scf --test kernel_internals_unit` · `pytest .../test_scf_controls.py -x` | ✅ | ✅ green (Rust) |
| SCF-07 | `mf.density_fit().kernel()` matches upstream | oracle-integration + Rust | `cargo test -p pyscf-scf --test dfhf_end_to_end` · `--test density_fit_wiring` · `pytest .../test_scf_df.py -x` | ✅ | ✅ green (Rust) / ⚠️ upstream byte-id CI |
| SCF-08 | Subclass override `get_veff` invoked once per cycle | unit (Python) + Rust | `cargo test -p pyscf-scf --test hooks_kernel_types` · `pytest .../test_scf_override_dispatch.py -x` | ✅ | ✅ green (Rust) / ⚠️ runtime CI |
| SCF-09 | `mf.analyze`/`mulliken_pop`/`mulliken_meta`/`dip_moment` | oracle + Rust | `cargo test -p pyscf-scf --test mulliken_meta` · `--test analyze_convert_scanner` · `pytest .../test_scf_analyze.py -x` | ✅ | ✅ green (Rust) / ⚠️ upstream byte-id CI |
| SCF-10 | chkfile round-trip (h5py schema + rs auto-write + from_chk) | oracle + Rust | `cargo test -p pyscf-scf --test chkfile_dump_load` · `pytest .../test_scf_chkfile.py -x` | ✅ | ✅ green (Rust) / ⚠️ h5py CI |
| SCF-11 | `mf.to_uhf/to_rhf/to_ghf`; `to_uks/to_rks` raise NotYetImplemented{phase:4} | unit + Rust | `cargo test -p pyscf-scf --test analyze_convert_scanner` · `pytest .../test_scf_cross_dispatch.py -x` | ✅ | ✅ green (Rust) / ⚠️ CI |
| SCF-12 | `mf.as_scanner()(mol2)` returns energy | unit + Rust | `cargo test -p pyscf-scf --test analyze_convert_scanner` · `pytest .../test_scf_scanner.py -x` | ✅ | ✅ green (Rust) / ⚠️ CI |
| SCF-13 | `canonicalize_signs` is largest-\|c\|-lowest-index-flip | unit (Rust) | `cargo test -p pyscf-scf --test canonicalize_post_eigh` | ✅ | ✅ green |
| SCF-13 | Linux x86_64 + macOS aarch64 µHartree assertion on H2O/cc-pVDZ | matrix-CI | GitHub Actions job `xplat-uhartree` · `pytest .../test_scf_xplat_uhartree.py` | ✅ | ⚠️ CI-gated (Pitfall 12) |
| SCF-14 | ≥ 30-attribute floor introspectable from Python | unit + Rust | `cargo test -p pyscf-scf --test attribute_floor` · `pytest .../test_scf_attributes.py -x` | ✅ | ✅ green (Rust) / ⚠️ CI |
| BIND-01 | `maturin develop` produces importable `pyscf._native` | CI smoke | GitHub Actions job `maturin-smoke` · `crates/pyscf-py/tests/maturin_smoke.py` | ✅ | ⚠️ CI-gated |
| BIND-02 | `from pyscf import scf` resolves to overlay `_native.scf` | unit (Python) | `pytest python/pyscf/tests/test_overlay_resolution.py -x` | ✅ | ⚠️ CI-gated |
| BIND-04 | Stride-fuzz: `a` / `a.T` / `a[::2]` / `a[:, 1:5]` return identical answers | unit (Python) | GitHub Actions job `stride-fuzz` · `pytest .../test_scf_stride_fuzz.py -x` | ✅ | ⚠️ CI-gated |
| BIND-05 | python3.13t SCF smoke runs without deadlock | matrix-CI | GitHub Actions job `python313t-smoke` (NON-abi3 build) | ✅ | ⚠️ CI-gated |
| BIND-06 | No `lazy_static!` in `pyscf-py`; `PyOnceLock` used for caches | lint | `cargo run -p xtask --bin check-forbid-lazy-static` · CI job `xtask-forbid-lazy-static` | ✅ | ✅ green |
| BIND-07 | Subclass `get_veff` override invoked ≥ 1× per SCF cycle (re-shape of SCF-08) | unit (Python) | (same as SCF-08) | ✅ | ⚠️ runtime CI |
| BIND-09 | Rust panic / convergence failure raises `PyscfRsRuntimeError` with `.kind`/`.source_chain` | unit (Python) | `pytest python/pyscf/tests/test_panic_to_exception.py -x` | ✅ | ⚠️ runtime CI |
| ORACLE-02 | `oracle_check!(method, tolerance, fixture)` macro invokable | unit (Rust) | `cargo test -p pyscf-oracle --test oracle_check_smoke` | ✅ | ✅ green |
| ORACLE-08 | chkfile round-trip both directions on H2O/cc-pVDZ | oracle + Rust | `cargo test -p pyscf-oracle --test chkfile_roundtrip` · `pytest .../test_scf_chkfile.py -x` | ✅ | ✅ green (Rust) / ⚠️ h5py CI |

*File: ✅ exists on disk. Status: ✅ green = runs in-sandbox via `cargo test` and passes · ⚠️ CI-gated = automation exists (CI job + test body) but requires the Python/maturin/h5py/3.13t toolchain absent from the sandbox; sampled by CI and listed as `human_needed` in `03-VERIFICATION.md`.*

---

## Wave 0 Requirements

**All wave-0 files exist** (verified on disk 2026-05-26 during `/gsd:validate-phase 3`). `wave_0_complete: true`.

### Python test stubs (19 files) — all present
- [x] `python/pyscf/tests/__init__.py`
- [x] `python/pyscf/tests/conftest.py` — shared fixtures
- [x] `python/pyscf/tests/test_scf_smoke.py` — wave-level smoke entry
- [x] `python/pyscf/tests/test_scf_rhf_h2o.py` — SCF-01
- [x] `python/pyscf/tests/test_scf_rhf_benzene.py` — SCF-01
- [x] `python/pyscf/tests/test_scf_uhf.py` — SCF-02
- [x] `python/pyscf/tests/test_scf_ghf.py` — SCF-03
- [x] `python/pyscf/tests/test_scf_diis.py` — SCF-04
- [x] `python/pyscf/tests/test_scf_init_guess.py` — SCF-05 (5 modes)
- [x] `python/pyscf/tests/test_scf_controls.py` — SCF-06
- [x] `python/pyscf/tests/test_scf_df.py` — SCF-07
- [x] `python/pyscf/tests/test_scf_override_dispatch.py` — SCF-08 / BIND-07
- [x] `python/pyscf/tests/test_scf_analyze.py` — SCF-09
- [x] `python/pyscf/tests/test_scf_chkfile.py` — SCF-10 / ORACLE-08
- [x] `python/pyscf/tests/test_scf_cross_dispatch.py` — SCF-11
- [x] `python/pyscf/tests/test_scf_scanner.py` — SCF-12
- [x] `python/pyscf/tests/test_scf_attributes.py` — SCF-14
- [x] `python/pyscf/tests/test_overlay_resolution.py` — BIND-02
- [x] `python/pyscf/tests/test_scf_stride_fuzz.py` — BIND-04
- [x] `python/pyscf/tests/test_panic_to_exception.py` — BIND-09

> Plus 3 extras shipped beyond the original stub list: `test_scf_rhf_ccpvdz.py` (SCF-01 cc-pVDZ d-shell), `test_scf_xplat_uhartree.py` (SCF-13 matrix), `__pycache__` (build artifact).

### Rust test stubs — all present
- [x] `crates/pyscf-oracle/src/lib.rs` — `oracle_check!` macro module (ORACLE-02)
- [x] `crates/pyscf-oracle/tests/chkfile_roundtrip.rs` — ORACLE-08 macro invocation tests
- [x] `crates/pyscf-oracle/tests/oracle_check_smoke.rs` — ORACLE-02 macro smoke (shipped name)
- [x] `crates/pyscf-scf/tests/` — 16 integration tests (canonicalize_post_eigh, init_guess_atom_huckel, mulliken_meta, dfhf_end_to_end, chkfile_dump_load, attribute_floor, …)

### Infrastructure — all present
- [x] `pyproject.toml` at repo root — maturin config + pytest deps
- [x] `python/pyscf/__init__.py` — overlay re-export shim (BIND-02)
- [x] `crates/pyscf-core/src/canonicalize.rs` — `pub fn canonicalize_signs` (SCF-13) *(shipped in `canonicalize.rs`, not `lib.rs` as originally planned)*
- [x] `.github/workflows/ci.yml` — the four planned jobs all present: `maturin-smoke` (BIND-01), `stride-fuzz` (BIND-04), `xplat-uhartree` (Pitfall 12 / SCF-13), `python313t-smoke` (BIND-05) **plus** `xtask-forbid-lazy-static` (BIND-06) added during this audit
- [x] `xtask/src/bin/check_forbid_lazy_static.rs` — BIND-06 lint *(shipped as an xtask `[[bin]]`, not a `src/lints/` module as originally planned; invoked via `cargo run -p xtask --bin check-forbid-lazy-static`)*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| `python3.13t` wheel install on developer machine | BIND-05 | Free-threaded interpreter is opt-in; not yet ubiquitous in CI runners | Document in phase SUMMARY: `pyenv install 3.13t-dev && pyenv local 3.13t-dev && maturin develop --features "free-threading-non-abi3"` |

*All other phase behaviors have automated verification.*

---

## Validation Sign-Off

- [x] All tasks have `<verify>` blocks pointing to a command in this map — every requirement maps to a `cargo test`, `pytest`, CI job, or lint command
- [x] Sampling continuity: no 3 consecutive tasks without automated verify — Rust unit/integration tests run per-commit
- [x] Wave 0 files all exist (verified on disk 2026-05-26)
- [x] No watch-mode flags (CI is one-shot)
- [x] Feedback latency < 30 s per task, < 120 s per wave
- [x] python3.13t job is a NON-abi3 build (`python313t-smoke`: `--no-default-features --features free-threading`)
- [x] Linux x86_64 + macOS aarch64 µHartree matrix job exists for Pitfall 12 mitigation (`xplat-uhartree`)
- [x] `nyquist_compliant: true` set in frontmatter

**Approval:** validated 2026-05-26 via `/gsd:validate-phase 3`

---

## Validation Audit 2026-05-26

| Metric | Count |
|--------|-------|
| Requirements audited | 23 |
| Tests MISSING (no file) | 0 |
| Gaps found | 2 |
| Resolved | 2 |
| Escalated to manual-only | 0 |

**Gaps resolved:**

1. **BIND-06 lint not sampled in CI.** The `check-forbid-lazy-static` xtask binary existed and passed locally (exit 0) but, unlike the other 5 xtask lints, had no dedicated CI job. → Added `xtask-forbid-lazy-static` job to `.github/workflows/ci.yml`, mirroring the existing lint jobs. Lint confirmed green in-sandbox.
2. **Stale xfail markers masking SCF-10 / ORACLE-08.** `test_scf_chkfile.py` unconditionally `pytest.xfail()`-ed the rs-write and rs-read arms, claiming `mf.kernel()` auto-write and `mf.from_chk()` were unimplemented — both are shipped (`scf.rs:363-388` auto-write, `scf.rs:420` `from_chk`). → Removed both `pytest.xfail()` calls and corrected the module/test docstrings so the two arms run their real assertions under CI (`maturin-smoke`).

**Coverage outcome:** All 23 requirements have automated verification and an on-disk test file. 10 run green in-sandbox via `cargo test`; the remaining (µHartree parity, NumPy/PyO3 runtime contracts, h5py round-trip, free-threading) are sampled by CI jobs requiring the Python/maturin toolchain and are tracked as `human_needed` in `03-VERIFICATION.md`.

---

*Phase: 03-scf-pyo3-bindings*
*Created: 2026-05-11 via /gsd-plan-phase step 5.5 (Nyquist validation gate)*
*Audited: 2026-05-26 via /gsd:validate-phase 3 (State A — reconciled stale planning-time map against shipped tree)*
