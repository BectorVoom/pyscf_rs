---
phase: quick-260530-mlg
plan: 01
subsystem: gpu-enable / eval_gto
tags: [cubecl, eval_gto, cart2sph, l-ge-1, general-kernel, differential-oracle, rocm]
requires: [quick-260530-ljv]
provides:
  - "eval_gto_sph_kernel_general (#[cube] l 0..=4 cart->sph device kernel)"
  - "ipow (#[cube] exp-by-squaring helper)"
  - "build_angular_tables + launch_eval_gto_general<R> (9 prefix-summed angular device tables)"
  - "maxl<=4 device routing in eval_gto_sph (subsumes l=0)"
  - "differential oracle gate: mixed-l (p/d/f/g) cpu + rocm gfx1152"
affects: [crates/pyscf-kernels/src/eval_gto.rs, crates/pyscf-kernels/tests/eval_gto_oracle.rs]
tech-stack:
  patterns: ["host-precomputed angular device tables", "one-thread-per-(g,shell)", "#[cube] helper (Solution 1)", "independent-longhand differential oracle"]
key-files:
  modified:
    - crates/pyscf-kernels/src/eval_gto.rs
    - crates/pyscf-kernels/tests/eval_gto_oracle.rs
decisions:
  - "ipow compiled as a #[cube] helper (Solution 1); zero compile fix-forwards needed"
  - "device tables use 3 parallel cpow_lx/ly/lz i32 arrays (as planned, no deviation)"
  - "oracle = independent longhand l 0..=4 reference (different code path from kernel), g(l=4) included"
metrics:
  duration: "~25 min"
  completed: 2026-05-30
requirements: [GTO-07, ORACLE-07, D-04]
---

# Phase quick-260530-mlg Plan 01: GPU-enable eval_gto l>=1 cart->sph Summary

GPU-enabled the `l >= 1` cartesian-monomial + libcint cart→sph transform as a real
`#[cube(launch_unchecked)]` device kernel (`eval_gto_sph_kernel_general`, l 0..=4,
one thread per (g, shell)), making the device path the DEFAULT for any basis whose
max angular momentum is 1..=4 (subsumes l=0, so mixed s+p+d bases run uniformly on
the device). Validated bit-exact-to-tolerance against an independent longhand
reference on both CpuRuntime and real ROCm gfx1152 hardware.

## Tasks completed

- **T1+T2** (`2b33100`, `feat`): `ipow` `#[cube]` helper + `eval_gto_sph_kernel_general`
  + host `build_angular_tables` + `launch_eval_gto_general<R>` + maxl<=4 routing in
  `eval_gto_sph`. Build + lib tests + full default-cpu suite green (eval_gto_lge1
  independent-reference gate, s-shell oracle, lib, wave0 smoke). Clippy clean.
- **T3** (`bd293ff`, `test`): extended `tests/eval_gto_oracle.rs` with the independent
  longhand l 0..=4 reference (`lge1_reference`), `build_mixed_l_fixture`, 8 `MIXED_CASES`
  (pure p/d/f/g + mixed s+p+d / s..g), always-on CpuRuntime arm + `#[cfg(rocm)]` gfx1152
  arm. Both RUN green.

## Required statements (per plan output spec)

1. **Device-table schema as built** — Exactly as planned: 9 angular arrays built on
   the host for `l in 0..=maxl` (maxl<=4), all offsets prefix-summed:
   - `c2s_flat: Vec<f64>` — concatenated c2s matrices, `T[l][m][ci]` at
     `c2s_off_by_l[l] + m*ncart(l) + ci` (row-major [m][ci] per l block).
   - `cpow_lx / cpow_ly / cpow_lz: Vec<i32>` — **3 PARALLEL i32 arrays** (the plan's
     choice; NO deviation to interleaving), cart power per col at `cpow_off_by_l[l] + ci`.
   - `ncart_by_l / nsph_by_l: Vec<i32>`, `fac1_by_l: Vec<f64>` (= `common_fac_sp(l)`),
     `c2s_off_by_l: Vec<i32>` (prefix sum of `ncart(l)*nsph(l)`),
     `cpow_off_by_l: Vec<i32>` (prefix sum of `ncart(l)`). All length `maxl+1`.
   - Kernel device-table index math: `c2s_flat[c2s_off + m*ncart_l + ci]`,
     `cpow_l{x,y,z}[cpow_off + ci]`. l=0 subsumption: ncart=nsph=1, c2s=[[1.0]],
     (lx,ly,lz)=(0,0,0), fac1=Y00=0.28209… → out = 1.0*radial*Y00 (pinned by the mixed
     s+p+d fixture).

2. **ipow approach + compile fix-forwards** — `ipow` compiled cleanly as a `#[cube]`
   helper (Solution 1 from the host-fn-in-#[cube] pitfall guide), written in the
   STATEMENT form (`let mut r = 1.0; if n>=1 {r=base;} if n>=2 {r=r*base;} …`) per the
   mismatched-types guide §1.3. **ZERO compile fix-forwards were needed** — no i32→u32
   cast surprises, no usize-loop-bound errors, no `.exp()` rewrite. The shipped s-kernel
   idiom held verbatim: usize scalar args, `ABSOLUTE_POS` usize, `CubeDim::new_1d`,
   2-arg `ArrayArg::from_raw_parts`, method-style `(-alpha*r2).exp()`, i32-array indexing
   cast to usize. The only clippy nit (`assign_op_pattern` wanting `r *= base`) was
   suppressed with a documented `#[allow]` on `ipow` to keep the cubecl-IR-safe explicit
   assignment form. cart_pow i32 values are cast to u32 for the `ipow` exponent
   (`cpow_lx[...] as u32`) inside the kernel, as designed — no genuine cubecl blocker hit.

3. **Observed worst max_abs_diff on CpuRuntime (p/d/f/g)** —
   `[eval_gto_oracle] CPU mixed-l (p/d/f/g) worst max_abs_diff = 6.938893903907228e-18`
   (sub-ULP, the only divergence being `ipow` vs host `f64::powi` for l>=3 monomials),
   far inside TOL=1e-9. Covers pure p / pure d / pure f / **pure g (l=4)** plus mixed
   s+p+d (cc-pVDZ-like), p+d+f over 2 atoms, full s..g on one centre, and d+f over 3 atoms.

4. **rocm gfx1152 RUN + worst max_abs_diff** — The rocm arm was **actually RUN on
   hardware** (`cargo test -p pyscf-kernels --features rocm --test eval_gto_oracle`), not
   just compiled:
   `[eval_gto_oracle] ROCm mixed-l (p/d/f/g) worst max_abs_diff = 1.1102230246251565e-16`
   (1 ULP; device `exp` vs std f64 + ipow vs powi), well inside TOL=1e-9. All 4 oracle
   tests (s-shell cpu/rocm + mixed-l cpu/rocm) pass. (s-shell rocm worst was also 1.11e-16.)

5. **eval_gto_lge1 + full suite green (l>4 error path intact)** — `cargo test -p
   pyscf-kernels` (default cpu): lib 2/2, **eval_gto_lge1 4/4** (the independent s/p/d
   c2s reference now validating the GPU general kernel against a DIFFERENT code path —
   the key behavior gate), eval_gto_oracle 2/2, wave0 smoke 1/1. Zero FAILED. The l>4
   never-panic gate (`c2s_coeff_l5_returns_err_not_panic`) stays green: maxl>4 routing
   condition (`maxl <= 4`) keeps h-shells off the device; they fall through to the
   UNCHANGED `eval_gto_sph_cpu` → `NotYetImplemented{phase:4}`. The all_s s-shell fast
   path (`launch_eval_gto_s`) and `eval_gto_sph_cpu` fallback are untouched.

6. **Contradictions with the orchestrator's design** — **None.** Every idiom, offset
   formula, reduction order, and routing condition matched the plan and the
   design_verification notes. No architectural deviation; no Rule-4 checkpoint.

## Differential oracle ground-truth choice

Used the **PREFERRED independent-longhand reference** (not an inline byte-copy of the
private `eval_gto_sph_cpu`): replicated the `tests/eval_gto_lge1.rs` `reference` module
pattern as `lge1_reference` and extended its c2s tables to **l=3 (f) and l=4 (g)**,
copying the L0..L4 coefficients VERBATIM from the FROZEN `c2s_coeff` tables in
`eval_gto.rs`. This is a genuinely different code path from the kernel (naive
triple-nested cartesian-monomial loop, longhand c2s), so a wrong-convention kernel bug
(AO ordering vs ao_loc, transposed c2s, cart_pow order, fac1 placement) surfaces as a
diff > TOL. **g (l=4) fixtures ARE included** (pure-g case + the full s..g mixed case).

## Verification

| Check | Result |
|-------|--------|
| `cargo build -p pyscf-kernels` | OK (no libxc) — log/mlg-t1-build.log |
| `cargo build -p pyscf-kernels --features rocm` | OK — log/mlg-verify-build.log |
| `cargo test -p pyscf-kernels` (default cpu) | 9/9 pass, 0 FAILED — log/mlg-verify-cpu.log |
| `cargo test -p pyscf-kernels --features rocm --test eval_gto_oracle` | 4/4 pass on gfx1152 — log/mlg-verify-rocm.log |
| `cargo clippy -p pyscf-kernels --all-targets` | clean (only pre-existing fma4 target-feature warning) — log/mlg-verify-clippy.log |

NO libxc pulled in any build (all verification scoped `-p pyscf-kernels`; `-p pyscf-gto`
deliberately NOT run). All cargo output saved under `log/mlg-*.log`.

## Deviations from Plan

None - plan executed exactly as written. (T1 and T2 committed together per the plan's
merged-task allowance, since the kernel/launcher/table-builder/ipow/routing are one
coupled file; T3 is a separate commit.)

## Deferred (carry-forward remainder)

- **deriv1 / deriv2 device stencils** — `eval_gto_sph_deriv1` still routes to the host
  `eval_gto_sph_deriv1_cpu` (the GGA ∇ρ path); `GTOval_sph_deriv2` / `ip*` / `ig*` stay
  `NotYetImplemented`. GPU stencils for the gradient/hessian components remain a future
  GPU-enable increment.
- **pyscf-dft numint backend** — the DFT grid-loop numint integration that consumes
  `eval_gto` on the device is not wired here; this slice only GPU-enables the AO-value
  (`GTOval_sph`) path. Hooking numint to the device buffers is the next increment.

## Self-Check: PASSED

- `crates/pyscf-kernels/src/eval_gto.rs` — FOUND (modified, committed 2b33100)
- `crates/pyscf-kernels/tests/eval_gto_oracle.rs` — FOUND (modified, committed bd293ff)
- commit `2b33100` — FOUND
- commit `bd293ff` — FOUND
