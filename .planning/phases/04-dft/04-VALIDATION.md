---
phase: 4
slug: dft
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-05-22
---

# Phase 4 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.
> Derived from `04-RESEARCH.md` § Validation Architecture. Per-task IDs are
> filled in by the planner/executor; requirement-level rows below are the
> Nyquist sampling contract.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | Rust `cargo test` + `pyscf-oracle` (`pyo3::Python::with_gil` driving upstream PySCF in-process, dev-dep only) + `oracle_check!` macro (ORACLE-02, Phase 3). PyO3 override tests via `pytest` against a maturin build. |
| **Config file** | workspace `Cargo.toml` profiles; `[profile.release-oracle]` (FMA-free, ordered reductions — Phase 1 FOUND-05) |
| **Quick run command** | `cargo test -p pyscf-grids` / `cargo test -p pyscf-dft` (CPU/xcfun default — **NEVER `--features libxc`**: pulls 266 kernel crates, ~6h freeze) |
| **Full suite command** | `cargo test --profile release-oracle -p pyscf-dft -p pyscf-grids` (CPU) |
| **Estimated runtime** | ~30s per-crate quick run; full release-oracle suite minutes (CPU). The gated `--features libxc` bit-exact build runs only in the dedicated cached CI job. |

---

## Sampling Rate

- **After every task commit:** Run `cargo test -p <crate>` (CPU/xcfun default; sub-30s where possible)
- **After every plan wave:** Run `cargo test --profile release-oracle -p pyscf-dft -p pyscf-grids` (CPU)
- **Before `/gsd:verify-work`:** Full CPU suite green + the dedicated `--features libxc` bit-exact CI job green (heavily cached) + the `wgpu-no-f64-fallback` CI job green
- **Max feedback latency:** ~30 seconds (per-task quick run)

---

## Per-Task Verification Map

> Requirement-level contract (per-task IDs assigned by the planner). Every plan
> task addressing one of these requirements MUST carry the matching automated
> verify command, or declare a Wave 0 dependency on the test file below.

| Requirement | Behavior | Test Type | Automated Command | File Exists | Status |
|-------------|----------|-----------|-------------------|-------------|--------|
| DFT-04 / DFT-09 | Grid points + weights byte-for-byte vs `gen_grid.py` for level 0..9 on corpus | oracle byte-compare | `cargo test -p pyscf-grids grid_weights_level_sweep` | ❌ W0 | ⬜ pending |
| DFT-02 | XC-string parser parity vs `libxc.py:parse_xc` (single/comma/shorthand/weights/alias) | unit parity | `cargo test -p pyscf-dft parse_xc_parity` | ❌ W0 | ⬜ pending |
| DFT-03 | libxc routes to `libxc_rs` / xcfun to `xcfun_rs`; bit-identical to C | oracle (gated, CI only) | `cargo test --features libxc -p pyscf-dft xc_eval_bitexact` | ❌ W0 | ⬜ pending |
| DFT-03 | ~100-functional libxc smoke (corpus subset) | smoke (gated, CI only) | `cargo test --features libxc -p pyscf-dft libxc_functional_smoke` | ❌ W0 | ⬜ pending |
| DFT-01 | RKS/UKS total energy ≤1 µHartree (SVWN/PBE/B3LYP) under release-oracle | oracle energy | `cargo test --profile release-oracle -p pyscf-dft rks_uks_bitexact` | ❌ W0 | ⬜ pending |
| DFT-05 | CAM-B3LYP / H2O range-separated-hybrid parity fixture | oracle energy (gated) | `cargo test --features libxc -p pyscf-dft cam_b3lyp_h2o_rsh` | ❌ W0 | ⬜ pending |
| DFT-06 | VV10 non-local-correlation energy match (`mf.nlc='VV10'`) | oracle energy | `cargo test -p pyscf-dft vv10_energy_match` | ❌ W0 | ⬜ pending |
| DFT-07 | DF-DFT `dft.RKS(mol).density_fit()` matches upstream | oracle energy | `cargo test -p pyscf-dft df_dft_match` | ❌ W0 | ⬜ pending |
| DFT-08 | Subclass `get_veff` + `define_xc_` overrides invoked every cycle | PyO3 dispatch assertion | `pytest python/tests/test_dft_override.py` (maturin build) | ❌ W0 | ⬜ pending |
| DFT-10 | `NumInt` signatures (`eval_xc`/`eval_rho`/`nr_rks`/`nr_uks`) match upstream | API/signature + numeric | `cargo test -p pyscf-dft numint_signatures` | ❌ W0 | ⬜ pending |
| DFT-11 | wgpu→CPU fallback with warning on `shader-f64`-less device | CI job (special runner) | `wgpu-no-f64-fallback` CI job: `dft.RKS(mol).run()`, assert warning + CPU-correct | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `crates/pyscf-grids/tests/grid_weights_level_sweep.rs` — byte-for-byte vs upstream, level 0..9, corpus (DFT-04/09)
- [ ] `crates/pyscf-dft/tests/parse_xc_parity.rs` — vs `libxc.py:parse_xc` (DFT-02)
- [ ] `crates/pyscf-dft/tests/rks_uks_bitexact.rs` — SVWN/PBE/B3LYP ≤1µHa (DFT-01)
- [ ] `crates/pyscf-dft/tests/xc_eval_bitexact.rs` + `libxc_functional_smoke.rs` — gated, CI-only (DFT-03)
- [ ] `crates/pyscf-dft/tests/cam_b3lyp_h2o_rsh.rs` (DFT-05), `vv10_energy_match.rs` (DFT-06), `df_dft_match.rs` (DFT-07)
- [ ] `crates/pyscf-dft/tests/numint_signatures.rs` (DFT-10)
- [ ] `python/tests/test_dft_override.py` — subclass override assertion (DFT-08)
- [ ] Oracle fixtures: upstream PySCF energies/grids generated under matched threading (`RAYON_NUM_THREADS=1`, `lib.num_threads(1)`, release-oracle) — extend the Phase 3 fixture corpus with DFT cases
- [ ] CI jobs: `--features libxc` DFT bit-exact (cached); `wgpu-no-f64-fallback` (special/emulated device); re-enable `libxc_rs` in `nightly-cross-crate.yml`
- [ ] Framework: reuse Phase 1/3 `oracle_check!` + `pyscf-oracle`; no new framework install

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| WGPU CPU-fallback warning on `shader-f64`-less hardware | DFT-11 | Requires a device (or emulation) lacking the `shader-f64` Vulkan extension; not reproducible on standard CI runners | Run the dedicated `wgpu-no-f64-fallback` CI job on the special/emulated runner; assert the `tracing::warn!` fallback message is emitted AND the energy matches CPU-correct numbers |

*All other phase behaviors have automated verification.*

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 30s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
