---
status: passed
verified_by: orchestrator (direct ground-truth — test+doc-only slice)
date: 2026-05-30
score: 4/4
---

# Verification — quick-260530-p40 (lock numint device AO-eval path)

Test + doc-only slice (no production/kernel/eval_rho code change). Verified by direct
ground-truth against source + fresh cargo, not a separate verifier agent.

| # | Must-have | Result | Evidence |
|---|-----------|--------|----------|
| 1 | Lock test exists + exercises the device AO path | PASS | `crates/pyscf-dft/tests/numint_device_ao_path.rs` (+186 lines, new). H2O/cc-pVDZ (maxl=2 → general l≥1 device kernel) on the default backend (select_backend → CpuRuntime kernel). |
| 2 | Assertion A — eval_rho correct over device AO block (non-circular) | PASS | ρ from `NumInt::eval_rho` over the `pyscf_gto::eval_gto("GTOval_sph")` device block matches an independent hand-written triple-sum (naive nested f64 loop, no shared code with `oracle_sum`) within 1e-9, all 8 grid points. |
| 3 | Assertion B — cross-kernel device AO correctness | PASS | `GTOval_sph_deriv1` comp-0 == `GTOval_sph` value block elementwise within 1e-12 (192 elements) — two independently-implemented device kernels agree. |
| 4 | No libxc + default-feature test green; doc note present | PASS | `cargo tree -p pyscf-dft -i libxc_rs` → "did not match any packages"; fresh `cargo test -p pyscf-dft --test numint_device_ao_path` → 1 passed/0 failed, zero libxc lines in the build. `numint.rs` +35 comment-only lines ("host by design" present); zero `eval_rho`/`contract_rho` code changes (git show: numint.rs additions only). |

## Notes
- Commit `fe4e24f` touched exactly the 2 intended files (numint.rs comment-only, new test). Unrelated `.claude/` churn untouched.
- ROCm variant intentionally omitted: pyscf-dft exposes no `rocm` feature, so `#[cfg(feature="rocm")]` would emit an unexpected-cfg warning; the always-on default-backend CpuRuntime test is the gate (the device kernels' real-ROCm correctness is already locked by the ljv/mlg/oms oracle tests on gfx1152).
- Closes Phase-8 GPU-enable: the device eval_gto surface (value + l≥1 + deriv1) is complete and numint inherits it transparently. GPU eval_rho is an explicitly-deferred perf item (user decision: keep host for FOUND-06 bit-exact SCF).
