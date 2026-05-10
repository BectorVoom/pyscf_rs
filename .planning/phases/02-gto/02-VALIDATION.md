---
phase: 02
slug: gto
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-05-10
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
| **Estimated runtime** | ~30s quick · ~6–10 min full (oracle pulls upstream PySCF for byte-identity diff) |

---

## Sampling Rate

- **After every task commit:** Run quick command (the unit subset for the touched crate)
- **After every plan wave:** Run full suite command (includes oracle layer)
- **Before `/gsd-verify-work`:** Full suite green AND `release-oracle` byte-identity assertions all pass
- **Max feedback latency:** 30 seconds for quick; 10 minutes for full

---

## Per-Task Verification Map

> Filled out by `gsd-planner` as PLAN.md files materialize. Below is the canonical REQ → test scaffold the planner copies into per-task `<acceptance_criteria>`. See `02-RESEARCH.md` § "Validation Architecture" for the full 21-assertion table.

| REQ-ID | Behavior | Test Type | Automated Command | Status |
|--------|----------|-----------|-------------------|--------|
| GTO-01 | 5 atom-input forms × parser dispatch | unit + oracle | `cargo test -p pyscf-gto atom_input_forms` | ⬜ pending |
| GTO-02 | 11 basis-input forms (`mol.basis = ...` kinds) | unit + oracle | `cargo test -p pyscf-gto basis_input_forms` | ⬜ pending |
| GTO-03 | All 184 builtin `.dat` files resolve via `mol.basis = '<name>'` | sweep | `cargo test -p pyscf-gto --test builtin_basis_sweep --features release-oracle` | ⬜ pending |
| GTO-04 | `_atm`/`_bas`/`_env`/`ao_loc_nr`/`nao_nr` byte-identity vs upstream | oracle | `pytest tests/oracle/test_byte_identity.py` (release-oracle) | ⬜ pending |
| GTO-05 | ECP loading (parser ships); `int1e_ecp` evaluation closes via gap-closure (D-06) | unit (loading) + ignored (evaluation) | `cargo test -p pyscf-gto ecp_load`; `int1e_ecp` test marked `#[ignore = "Pending cintx ECP merge"]` | ⬜ pending (split) |
| GTO-06 | `mol.intor(name)` dispatches all in-scope 1e/2e integrals to cintx; tolerance check | oracle | `pytest tests/oracle/test_intor_oracle.py -k "int1e_ovlp or int2e or int1e_kin or int1e_nuc"` | ⬜ pending |
| GTO-07 | `eval_gto` (6 variants) element-wise match on 1000-point grid | oracle | `pytest tests/oracle/test_eval_gto.py` | ⬜ pending |
| GTO-08 | ≥30-attribute floor (`atom`, `basis`, `_atm`, …) | unit | `cargo test -p pyscf-core mole_attribute_floor` | ⬜ pending |
| GTO-09 | `mol.dumps()`/`gto.Mole.loads()` semantic JSON round-trip | unit + oracle interop | `cargo test -p pyscf-gto dumps_loads_roundtrip`; `pytest tests/oracle/test_json_interop.py` | ⬜ pending |
| GTO-10 | `mol.copy()` deep-copy + `mol.set_geom_(new_atom)` in-place mutation | unit | `cargo test -p pyscf-core mole_copy mole_set_geom` | ⬜ pending |
| GTO-11 | Zero-copy re-export of `cintx_core::BasisSet` (no clone on `Mole.basis_set()`) | unit (Arc identity) | `cargo test -p pyscf-core basis_set_zero_copy` (asserts `Arc::ptr_eq`) | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

**Per-task expansion rule:** every task's `<acceptance_criteria>` must include at least one row from this table (mapped via the task's `requirements` field). Sampling continuity rule: no 3 consecutive tasks without an automated verify command. Tasks that only modify scaffolding/Cargo.toml may use a `cargo check -p <crate>` row but must still appear here.

---

## Wave 0 Requirements

> Wave 0 is the first plan in Phase 2 (sequence `02-01-PLAN.md`). All other plans depend on it.

- [ ] **W0-T1 — cintx round-trip smoke** — `crates/pyscf-gto/tests/wave0_smoke.rs` exercises `cintx_rs::SessionRequest` with H2/STO-3G `int1e_ovlp_sph` and asserts non-empty `IntegralTensor` (resolves the cintx integration reach risk before bulk porting starts; per RESEARCH.md Wave 0 gap #1)
- [ ] **W0-T2 — cubecl-cpu kernel launch smoke** — `crates/pyscf-kernels/tests/wave0_cubecl_smoke.rs` runs a minimal `#[cube(launch_unchecked)]` `vector_add` via `cubecl-cpu` and asserts output (resolves the cubecl kernel launch shape risk before `eval_gto` lands; per RESEARCH.md Wave 0 gap #2)
- [ ] **W0-T3 — per-intor F/C-order layout table** — `crates/pyscf-gto/src/layout_table.rs` enumerates the in-scope `intor` names from `cintx-compat::raw::RawApiId` and tags each F-order vs C-order by consulting `pyscf/gto/moleintor.py`. Output committed; downstream tasks reference it for the F-order preservation acceptance criterion (Pitfall 8). (Resolves RESEARCH.md Open Question #1)
- [ ] **W0-T4 — algebra-wall lint allowlist verification** — read `crates/pyscf-algebra/build.rs` (or wherever Phase 1 D-08 lint lives), confirm `pyscf-kernels` is in the allowlist (it must touch `cubecl::*` for the eval_gto kernel). If not, add it with comment citing Phase 2 D-04. (Resolves RESEARCH.md Open Question #2)
- [ ] **W0-T5 — oracle test harness scaffold** — `tests/oracle/conftest.py` + `tests/oracle/__init__.py` + `tests/oracle/Cargo.toml` (oracle xtask). Establishes the `release-oracle`-feature-gated harness that side-by-side-runs upstream PySCF and pyscf-rs on the small PR-CI corpus (H2O/cc-pVDZ + benzene/6-31G* + water-trimer/STO-3G).
- [ ] **W0-T6 — `pytest` install gate** — `tests/oracle/requirements.txt` lists `pyscf>=2.6` + `pytest>=7`; CI step asserts `pytest --version` succeeds before running any oracle test. (Project may already have pytest from Phase 1; verify in W0-T6 and skip install if so.)

*If Wave 0 produces NO red lines for any of W0-T1..W0-T6, set `wave_0_complete: true` in frontmatter and unblock plans 02-02 onward.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| `int1e_ecp` numerical accuracy vs upstream | GTO-05 (evaluation half) | Blocked on cintx ECP merge (D-06 parallel sequencing). Test exists but `#[ignore = "Pending cintx ECP merge"]` until the gap-closure plan flips it on. | After cintx ECP lands, the gap-closure plan removes the `#[ignore]` and runs `cargo test -p pyscf-gto int1e_ecp_oracle --features release-oracle`. Result must match upstream `mol.intor('int1e_ecp')` to oracle tolerance. |
| Atom-input "callable" form (5th of 5 forms in GTO-01) | GTO-01 | Needs PyO3 (Phase 3 BIND-02). Phase 2 returns `NotYetImplemented { phase: 3, what: "atom callable form (GTO-01.5)" }`. | Verified in Phase 3 plan; Phase 2 ships only the error-path test asserting the `NotYetImplemented` variant. |
| Wheel packaging of `pyscf/gto/basis/` (~MB) | D-01 / GTO-03 | Wheel content manifest is Phase 8 DIST-02. D-01 reads files at runtime, so end-user wheel must bundle them. Phase 2 only verifies the runtime-resolution path under `PYSCF_BASIS_PATH`. | Phase 8 ships maturin packaging acceptance test; Phase 2 verifies the env-var path resolves correctly via cargo unit tests. |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify command OR explicit Wave 0 dependency declared in `<read_first>`
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify (planner enforces during `gsd-planner` task graph layout)
- [ ] Wave 0 covers all MISSING references in RESEARCH.md (W0-T1..W0-T6 above)
- [ ] No watch-mode flags (`cargo watch`, `pytest --watch`, etc.)
- [ ] Feedback latency: ~30s quick · ≤10 min full
- [ ] `release-oracle` feature flag actually flips byte-identity assertions on (verified by W0-T5 harness scaffold)
- [ ] Pitfall coverage: Pitfall 8 (F-order) acceptance criteria reference W0-T3's layout table; Pitfall 17 (off-by-one basis indexing) has at least one assertion comparing `ao_loc_nr` byte-for-byte; Pitfall 18 (Boys-function accuracy) is delegated to cintx and explicitly listed in `<deferred_to_other_phase>` notes
- [ ] `nyquist_compliant: true` set in frontmatter (after planner fills the per-task table fully)

**Approval:** pending — to be flipped to `approved 2026-05-XX` once Wave 0 is green and the planner's per-task table has zero `⬜ pending` rows for in-scope behaviors (deferred items per D-06 / Phase 3 / Phase 8 may stay `⬜` with explicit deferral note).
