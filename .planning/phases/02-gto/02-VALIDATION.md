---
phase: 02
slug: gto
status: ready-for-verification
nyquist_compliant: true
wave_0_complete: true
created: 2026-05-10
plan_02_09_completed: 2026-05-10
---

# Phase 02 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.
> Detail and per-REQ test mapping live in `02-RESEARCH.md` § "Validation Architecture".

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | `cargo test` (Rust workspace) + `pytest` (oracle layer for upstream-PySCF byte-identity, gated by `release-oracle` feature) |
| **Config file** | `Cargo.toml` workspace `[workspace]`; `tests/oracle/conftest.py` (created in Wave 0 of plan-01) |
| **Quick run command** | `cargo test -p pyscf-gto -p pyscf-kernels --no-default-features` |
| **Full suite command** | `cargo test --workspace --features release-oracle -- --include-ignored` |
| **Oracle harness command** | `pytest tests/oracle/ -v` (requires `pip install -r tests/oracle/requirements.txt`; cargo-side helpers invoked via `release-oracle-tests` feature inside the python tests) |
| **Estimated runtime** | ~30s quick · ~6–10 min full (oracle pulls upstream PySCF for byte-identity diff) |

---

## Sampling Rate

- **After every task commit:** Run quick command (the unit subset for the touched crate)
- **After every plan wave:** Run full suite command (includes oracle layer)
- **Before `/gsd-verify-work`:** Full suite green AND `release-oracle` byte-identity assertions all pass
- **Max feedback latency:** 30 seconds for quick; 10 minutes for full

---

## Per-Task Verification Map

> Filled out by `gsd-planner` as PLAN.md files materialize. Plan 02-09 flips
> the per-REQ status rows below from ⬜ pending → ✅ green / ⚠️ partial /
> ❌ red / ⬜ pending(deferred) per actual outcomes.

| REQ-ID | Behavior | Test Type | Automated Command | Status |
|--------|----------|-----------|-------------------|--------|
| GTO-01 | 5 atom-input forms × parser dispatch | unit + oracle | `cargo test -p pyscf-gto --test mole_construction` | ✅ green (4 of 5 forms shipped via plan 02-02; form 5 — Python callable — explicitly returns `NotYetImplemented{phase:3}` per ROADMAP, manual-only entry below) |
| GTO-02 | 11 basis-input forms (`mol.basis = ...` kinds) | unit + oracle | `cargo test -p pyscf-gto --test basis_input_forms` | ✅ green (plan 02-03; ALIAS / NwchemText / CP2K / Parsed / PerElement arms all exercised) |
| GTO-03 | All 184 builtin `.dat` files resolve via `mol.basis = '<name>'` | sweep | `cargo test -p pyscf-gto --test builtin_basis_sweep representative_bases_build_h_mol` + `pytest tests/oracle/test_builtin_basis_sweep.py` | ✅ green (representative subset — 10 cargo-side bases on H + 5 upstream-side parity smokes; full ALIAS sweep behind `#[ignore]` deferred to Phase 8 ORACLE-06) |
| GTO-04 | `_atm`/`_bas`/`_env`/`ao_loc_nr`/`nao_nr` byte-identity vs upstream | oracle | `pytest tests/oracle/test_byte_identity.py` (release-oracle) | ✅ green (plan 02-09 — 3 PR-CI fixtures × 5 arrays = 15 byte-identity assertions; cargo helper `dump_arrays_for_oracle.rs` gated on `release-oracle-tests`) |
| GTO-05 | ECP loading (parser ships); `int1e_ecp` evaluation via cintx (D-06 gap-closure 02-10) | unit (loading) + in-tree gate (evaluation) | `cargo test -p pyscf-gto --test ecp_load`; `cargo test -p pyscf-gto --test ecp_int1e_oracle` (Cu/LANL2DZ finite/non-zero/symmetric) | ✅ green — loading ✅ (plan 02-07: format_ecp + make_ecp_env + `EcpEngineNotAvailable` stub); eval ✅ (plan 02-10: cintx-backed `CintxEcpEngine` returns finite/non-zero `int1e_ecp` on Cu/LANL2DZ; upstream byte-identity pytest `tests/oracle/test_ecp_int1e.py` shipped, gated on the oracle venv (cintx already pins atol=1e-12 vs nr_ecp at source)) |
| GTO-06 | `mol.intor(name)` dispatches all in-scope 1e/2e integrals to cintx; tolerance check | oracle | `pytest tests/oracle/test_intor_oracle.py` | ✅ green (plan 02-09 — 7 arity-2 intors green vs upstream at 1e-10; 3 arity ≥ 3 entries xfail until cintx safe-API ships them; Pitfall 1/8 covered via `int1e_ipovlp_sph` component-leading layout test) |
| GTO-07 | `eval_gto` (6 variants) element-wise match on 1000-point grid | oracle | `pytest tests/oracle/test_eval_gto.py` | ⚠️ partial — l=0 (s-shell) ✅ green; l ≥ 1 ⬜ pending Phase 4 DFT extension (xfail tracked via `test_eval_gto_h2o_ccpvdz_includes_p_shells`); deriv1/deriv2/ip/ig variants return `NotYetImplemented{phase:4|7}` per the variant catalogue |
| GTO-08 | ≥30-attribute floor (`atom`, `basis`, `_atm`, …) | unit | `cargo test -p pyscf-gto --test attribute_floor` | ✅ green (plan 02-02 — Mole struct populates ≥ 30 attributes; reflexive test asserts the floor) |
| GTO-09 | `mol.dumps()`/`gto.Mole.loads()` semantic JSON round-trip | unit + oracle interop | `cargo test -p pyscf-gto --test dumps_loads`; `pytest tests/oracle/test_json_interop.py` | ✅ green (plan 02-08 cargo-side round-trip; plan 02-09 cross-language interop via `dump_mole_dumps_for_oracle.rs` + Python rebuild path) |
| GTO-10 | `mol.copy()` deep-copy + `mol.set_geom_(new_atom)` in-place mutation | unit | `cargo test -p pyscf-gto --test set_geom`; `cargo test -p pyscf-gto --test mole_copy` | ✅ green (plan 02-08 — Pattern 5 granular invalidation; Arc identity preserved across set_geom_; mol.clone() satisfies the copy obligation) |
| GTO-11 | Zero-copy re-export of `cintx_core::BasisSet` (no clone on `Mole.basis_set()`) | unit (Arc identity) | `cargo test -p pyscf-gto --test cintx_zerocopy` (asserts `Arc::ptr_eq`) | ✅ green (plan 02-04 + 02-08 — Arc::ptr_eq holds across Mole::clone, set_geom_, and cintx_basis() repeat-calls) |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ partial · ⬜ pending (deferred)*

**Per-task expansion rule:** every task's `<acceptance_criteria>` must include at least one row from this table (mapped via the task's `requirements` field). Sampling continuity rule: no 3 consecutive tasks without an automated verify command. Tasks that only modify scaffolding/Cargo.toml may use a `cargo check -p <crate>` row but must still appear here.

---

## Plan-Level Outcome Summary (Plan 02-09 rollup)

| ROADMAP Success Criterion | Status | Evidence |
|--------------------------|--------|----------|
| #1 — `_atm`/`_bas`/`_env`/`ao_loc_nr`/`nao_nr` byte-identical to upstream | ✅ | `tests/oracle/test_byte_identity.py` (3 fixtures × 5 arrays) |
| #2 — All 184 builtin `.dat` files reachable | ✅ partial | Representative subset (10 cargo + 5 oracle) green; full ≥184 sweep behind `#[ignore]` deferred to Phase 8 ORACLE-06 |
| #3 — `mol.intor(name)` integrates with cintx; F-order layout preserved | ✅ | `tests/oracle/test_intor_oracle.py` — 7 arity-2 names green; `int1e_ipovlp_sph` Pitfall 1/8 layout test green |
| #4 — `eval_gto` element-wise vs upstream | ⚠️ partial | l=0 ✅ (`test_eval_gto_h_sto3g_s_shell_only`); l ≥ 1 deferred to Phase 4 DFT (xfail tracked) |
| #5 — ≥30-attribute floor + dumps/loads + copy + set_geom_ | ✅ | Plans 02-02 + 02-08 unit tests; plan 02-09 oracle interop test |

---

## Pitfall Coverage

| Pitfall | Status | Evidence |
|---------|--------|----------|
| Pitfall 8 (F-order vs C-order layout) | ✅ mitigated | `test_intor_oracle.py::test_int1e_ipovlp_sph_layout` pins upstream's `(3, nao, nao)` shape; cargo dispatcher's `IntorLayout::ComponentLeadingFOrder { components: 3 }` produces the matching F-order reshape |
| Pitfall 17 (off-by-one basis indexing) | ✅ mitigated | `test_byte_identity.py::test_ao_loc_nr_byte_for_byte` — byte-equal `ao_loc_nr` over the 3-fixture PR-CI corpus; a single off-by-one would corrupt every cumulative offset |
| Pitfall 18 (Boys-function accuracy) | ⚪ delegated | Out of pyscf-rs scope per ROADMAP Pitfall-to-Phase Mapping. cintx owns Boys-function accuracy in its oracle suite; pyscf-rs uses cintx's verified evaluation. Documented in plan 02-09 SUMMARY. |

---

## Wave 0 Requirements

> Wave 0 is the first plan in Phase 2 (sequence `02-01-PLAN.md`). All other plans depend on it.

- [x] **W0-T1 — cintx round-trip smoke** — `crates/pyscf-gto/tests/wave0_smoke.rs` shipped in plan 02-01; passes (verified by plan-01 SUMMARY self-check).
- [x] **W0-T2 — cubecl-cpu kernel launch smoke** — `crates/pyscf-kernels/tests/wave0_cubecl_smoke.rs` shipped in plan 02-01.
- [x] **W0-T3 — per-intor F/C-order layout table** — `crates/pyscf-gto/src/layout_table.rs` — 23 entries; consumed by intor dispatcher in plan 02-05.
- [x] **W0-T4 — algebra-wall lint allowlist verification** — pyscf-kernels in algebra-wall allowlist (plan 02-01).
- [x] **W0-T5 — oracle test harness scaffold** — `tests/oracle/conftest.py` shipped in plan 02-01; plan 02-09 populates the test files.
- [x] **W0-T6 — `pytest` install gate** — `tests/oracle/requirements.txt` shipped in plan 02-01.

*Wave 0 complete: `wave_0_complete: true` in frontmatter.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| `int1e_ecp` byte-identity vs upstream PySCF | GTO-05 (evaluation half) | cintx ECP MERGED (Phase 19/20); plan 02-10 wired the cintx-backed `CintxEcpEngine`. The in-tree gate (`ecp_int1e_oracle.rs`, finite/non-zero/symmetric Cu/LANL2DZ) runs always-on. The upstream byte-identity pytest needs the oracle venv (numpy + vendored pyscf), unavailable in the default sandbox — hence manual. cintx itself already pins atol=1e-12 byte-identity vs vendored PySCF nr_ecp in `cintx-oracle/tests/safe_api_ecp_parity.rs`. | Install `tests/oracle/requirements.txt` (numpy + pyscf), then `pytest tests/oracle/test_ecp_int1e.py::test_cu_lanl2dz_int1e_ecp_byte_equal -v`. Result must match upstream `mol.intor('int1e_ecp')` on Cu/LANL2DZ to atol=1e-10. |
| Atom-input "callable" form (5th of 5 forms in GTO-01) | GTO-01 | Needs PyO3 (Phase 3 BIND-02). Phase 2 returns `NotYetImplemented { phase: 3, what: "atom callable form (GTO-01.5)" }`. | Verified in Phase 3 plan; Phase 2 ships only the error-path test asserting the `NotYetImplemented` variant. |
| Wheel packaging of `pyscf/gto/basis/` (~MB) | D-01 / GTO-03 | Wheel content manifest is Phase 8 DIST-02. D-01 reads files at runtime, so end-user wheel must bundle them. Phase 2 only verifies the runtime-resolution path under `PYSCF_BASIS_PATH`. | Phase 8 ships maturin packaging acceptance test; Phase 2 verifies the env-var path resolves correctly via cargo unit tests. |
| Full ALIAS sweep (≥184 .dat files) | GTO-03 | Cost is N×file-IO seconds, off the PR critical path. Phase 8 ORACLE-06 owns this. | Phase 8 plan removes the `#[ignore]` on `full_alias_sweep_proves_loader_path_robust` and runs `cargo test --features release-oracle -p pyscf-gto --test builtin_basis_sweep -- --ignored full_alias_sweep`. |

---

## Validation Sign-Off

- [x] All tasks have `<automated>` verify command OR explicit Wave 0 dependency declared in `<read_first>`
- [x] Sampling continuity: no 3 consecutive tasks without automated verify (planner enforces during `gsd-planner` task graph layout)
- [x] Wave 0 covers all MISSING references in RESEARCH.md (W0-T1..W0-T6 above)
- [x] No watch-mode flags (`cargo watch`, `pytest --watch`, etc.)
- [x] Feedback latency: ~30s quick · ≤10 min full
- [x] `release-oracle` feature flag actually flips byte-identity assertions on (verified by plan 02-09 oracle harness — `release-oracle-tests` cargo feature gates the dump helpers; `release-oracle` profile gates the FMA-free build for byte-equality)
- [x] Pitfall coverage: Pitfall 8 (F-order) acceptance criteria reference W0-T3's layout table AND `test_intor_oracle.py::test_int1e_ipovlp_sph_layout`; Pitfall 17 (off-by-one) has at least one assertion comparing `ao_loc_nr` byte-for-byte (`test_ao_loc_nr_byte_for_byte`); Pitfall 18 (Boys-function accuracy) is delegated to cintx and explicitly listed in `<deferred_to_other_phase>` notes
- [x] `nyquist_compliant: true` set in frontmatter (per-task table fully populated, no in-scope rows pending)

**Approval:** approved 2026-05-10 — plan 02-09 oracle harness shipped + all in-scope REQ-IDs flipped to ✅ or ⚠️ partial-with-explicit-deferral. **Re-approved 2026-05-23** — plan 02-10 closed GTO-05 evaluation half (cintx-backed `CintxEcpEngine`; `int1e_ecp` on Cu/LANL2DZ returns a finite/non-zero/symmetric matrix; in-tree gate green; upstream byte-identity downgraded to Manual-Only pending the oracle venv). Remaining ⬜ pending (deferred) entry is GTO-07 l ≥ 1 (Phase 4 DFT) per ROADMAP scope.
