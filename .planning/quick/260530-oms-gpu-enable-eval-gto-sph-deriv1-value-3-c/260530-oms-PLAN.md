---
phase: quick-260530-oms
plan: 01
type: execute
wave: 1
depends_on: [quick-260530-mlg]
files_modified:
  - crates/pyscf-kernels/src/eval_gto.rs
  - crates/pyscf-kernels/tests/eval_gto_oracle.rs
autonomous: true
requirements: [GTO-08, ORACLE-07, D-04]
subsystem: gpu-enable / eval_gto

must_haves:
  truths:
    - "eval_gto_sph_deriv1 runs on device (CpuRuntime + ROCm) for any non-empty basis with maxl<=4, producing the [4,ngrids,nao] value+grad buffer"
    - "Device deriv1 output matches an INDEPENDENT longhand deriv1 reference to <1e-9 over randomized p/d/f/g (and mixed-l) fixtures, ALL 4 components"
    - "l>4 / empty basis / empty grid still route to the UNCHANGED eval_gto_sph_deriv1_cpu (NotYetImplemented{phase:4} for l>4 preserved)"
    - "Full `cargo test -p pyscf-kernels` (default cpu) stays green; ROCm arm RUN on gfx1152 green"
  artifacts:
    - path: "crates/pyscf-kernels/src/eval_gto.rs"
      provides: "dpow #[cube] helper + eval_gto_sph_deriv1_kernel #[cube] + launch_eval_gto_deriv1<R> + device routing in eval_gto_sph_deriv1"
      contains: "fn eval_gto_sph_deriv1_kernel"
    - path: "crates/pyscf-kernels/tests/eval_gto_oracle.rs"
      provides: "deriv1 4-component differential oracle (CpuRuntime always-on + #[cfg(rocm)] gfx1152 arm)"
      contains: "deriv1"
  key_links:
    - from: "eval_gto_sph_deriv1"
      to: "launch_eval_gto_deriv1"
      via: "dispatch_backend! when maxl<=4 && comp_stride>0"
      pattern: "dispatch_backend!.*launch_eval_gto_deriv1"
    - from: "eval_gto_sph_deriv1_kernel"
      to: "dpow / ipow"
      via: "#[cube] helper calls (Solution 1)"
      pattern: "dpow"
---

<objective>
Port `eval_gto_sph_deriv1_cpu` (AO value + 3 analytic cartesian gradient
components, output `[4, ngrids, nao]`) into a real `#[cube(launch_unchecked)]`
GPU kernel that REUSES the shipped mlg general-kernel angular infrastructure
(`build_angular_tables` / `AngularTables` / `ipow`), routed via
`dispatch_backend!`. This is the GGA gradient AO-derivative device path and
COMPLETES the `eval_gto` device surface (deriv2 is OUT — no CPU impl, returns
`NotYetImplemented`, nothing to port).

Purpose: the DFT GGA numint path can run AO derivatives on device instead of
falling back to the host with a `tracing::warn!`.

Output:
- `dpow` `#[cube]` helper, `eval_gto_sph_deriv1_kernel` `#[cube]`,
  `launch_eval_gto_deriv1<R>` launcher, device routing wired into the existing
  `eval_gto_sph_deriv1` (all in `eval_gto.rs`).
- deriv1 4-component differential-oracle gate in `eval_gto_oracle.rs`
  (CpuRuntime always-on + `#[cfg(rocm)]` gfx1152), RUN on hardware.
</objective>

<execution_context>
@/home/user/Documents/workspace/pyscf_rs/.claude/get-shit-done/workflows/execute-plan.md
@/home/user/Documents/workspace/pyscf_rs/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@./CLAUDE.md
@.planning/quick/260530-mlg-gpu-enable-eval-gto-l-1-cart-sph-path-ge/260530-mlg-SUMMARY.md

<interfaces>
<!-- Contracts the executor builds against. All in crates/pyscf-kernels/src/eval_gto.rs. -->
<!-- REUSE VERBATIM (no changes): -->

ipow: a `#[cube] #[inline(always)] fn ipow(base: f64, n: u32) -> f64`
(eval_gto.rs ~657) — exp-by-squaring, STATEMENT form, l 0..=4 (`n==4` is the top
case). dpow MAY call it (Solution 1 legal).

AngularTables: struct with fields c2s_flat, cpow_lx, cpow_ly, cpow_lz,
ncart_by_l, nsph_by_l, fac1_by_l, c2s_off_by_l, cpow_off_by_l (Vec<f64>/Vec<i32>)
— eval_gto.rs ~774.

build_angular_tables: `fn build_angular_tables(maxl: u32) -> Result<AngularTables,
PyscfRsError>` (eval_gto.rs ~793) — prefix-summed device tables for l in 0..=maxl.
Index math: c2s_flat at `c2s_off + m*ncart_l + ci`; cpow_l{x,y,z} at `cpow_off + ci`.

launch_eval_gto_general: `fn launch_eval_gto_general<R: Runtime>(client, coords,
ngrids, atm, bas, env, ao_loc, nao, maxl) -> Result<Vec<f64>, PyscfRsError>`
(eval_gto.rs ~851) — the EXACT host-upload + launch template to clone for the
deriv1 launcher.

eval_gto_sph_kernel_general: `#[cube(launch_unchecked)] fn ...(...)`
(eval_gto.rs ~678) — the value-only kernel; the deriv1 kernel has the SAME arg
shape PLUS a longer `out` (4*ngrids*nao) and a `comp_stride: usize` scalar arg.

Constants: EVAL_GTO_BLOCK (u32 = 256), BAS_SLOTS, ATM_SLOTS, and the *_OF / PTR_*
offsets from `pyscf_core::raw_layout`. `dispatch_backend!` is imported from
`pyscf_algebra` (eval_gto.rs:85). `eval_gto_sph_deriv1` is already exported from
lib.rs:23.

<!-- MIRROR (the exact deriv1 math — eval_gto_sph_deriv1_cpu, eval_gto.rs 1263-1418): -->
Per (g, shell), per c_idx: radial = (sum over p of coef·exp(-alpha·r2))·fac1;
radial_2a = (sum over p of -2·alpha·coef·exp(-alpha·r2))·fac1 — BOTH in ONE
ordered loop over p_idx (host forms g0 = coef·exp(-alpha·r2) once, then e += g0;
e2a += -2.0·alpha·g0).
Per ci (lx,ly,lz): mono = dx^lx·dy^ly·dz^lz;
  cval = mono·radial;
  cdx  = radial_2a·dx·mono + radial·dpow(dx,lx)·dy^ly·dz^lz;
  cdy  = radial_2a·dy·mono + radial·dx^lx·dpow(dy,ly)·dz^lz;
  cdz  = radial_2a·dz·mono + radial·dx^lx·dy^ly·dpow(dz,lz).
c2s per m (t = c2s_flat[c2s_off + m·ncart_l + ci]): v/vx/vy/vz += t·cval/cdx/cdy/cdz.
off = g + (ao_off + c_idx·nsph_l + m)·ngrids; write
  out[off]=v, out[comp_stride+off]=vx, out[2·comp_stride+off]=vy, out[3·comp_stride+off]=vz.
comp_stride = ngrids·nao; shape [4, ngrids, nao].
dpow(q,lq) = lq·q^(lq-1), and 0 if lq==0 (host inner fn, eval_gto.rs 1299-1305).
</interfaces>
</context>

<tasks>

<task type="auto">
  <name>Task 1: dpow #[cube] helper + eval_gto_sph_deriv1_kernel + launch_eval_gto_deriv1 + device routing</name>
  <files>crates/pyscf-kernels/src/eval_gto.rs</files>
  <action>
One coupled change in eval_gto.rs. Read CLAUDE.md (the cubecl manual at
docs/manual/Cubecl is MANDATORY) before writing any #[cube] code. Save ALL cargo
output to log/oms-t1-*.log.

(a) dpow helper: add `#[cube] #[inline(always)] fn dpow(q: f64, lq: u32) -> f64`
in the STATEMENT form (like ipow, NOT `let r = if ... {}`, per the mismatched-types
guide section 1.3): start `let mut r = 0.0_f64;`, then `if lq >= 1 { r = (lq as
f64) * ipow(q, lq - 1); }`, return `r`. The `if lq >= 1` gate prevents `lq - 1`
u32 underflow (lq=0 short-circuits). dpow CALLS ipow (ipow is #[cube], Solution 1
legal). Semantics mirror host dpow (eval_gto.rs 1299-1305): 0 for lq==0 else
`lq·q^(lq-1)`. Document that this parallels host `q.powi(lq as i32 - 1)`
(ipow-vs-powi diverges <1 ULP at l>=3, inside TOL=1e-9 / ORACLE-07).

(b) Kernel: add `#[allow(clippy::too_many_arguments)] #[cube(launch_unchecked)] fn
eval_gto_sph_deriv1_kernel(...)` — clone eval_gto_sph_kernel_general's arg list and
APPEND a trailing `comp_stride: usize` scalar arg; `out` is now length
4*ngrids*nao. One thread per (g, shell), bounds-guard `if tid < ngrids*nbas`.
Reuse the same coords/bas/atm/env reads, dx/dy/dz, r2, and the l-indexed table
reads (ncart_l, nsph_l, fac1, c2s_off, cpow_off). Per c_idx compute radial AND
radial_2a in ONE ordered loop over p_idx: form `let g0 = coef*(-alpha*r2).exp();`
once, then `acc += g0;` and `acc2a += (-2.0)*alpha*g0;` — MIRROR host operand
order (eval_gto.rs 1359-1361), THEN `let radial = acc*fac1; let radial_2a =
acc2a*fac1;`. Use plain SEQUENTIAL acc (NOT oracle_sum): per ORACLE-07 / mlg,
oracle_sum == strict-sequential for nprim<=128 and the single-thread sequential
acc is the documented posture; the T2 independent oracle also sums sequentially.
Then the c2s transform: `for m in 0..nsph_l { for ci in 0..ncart_l { ... } }`,
and INSIDE the ci body recompute the per-ci quantities (no Vec scratch in #[cube]):
read lx/ly/lz from cpow tables (cast i32->u32);
`let mono = ipow(dx,lx)*ipow(dy,ly)*ipow(dz,lz);`
`let cval = mono*radial;`
`let cdx = radial_2a*dx*mono + radial*dpow(dx,lx)*ipow(dy,ly)*ipow(dz,lz);`
`let cdy = radial_2a*dy*mono + radial*ipow(dx,lx)*dpow(dy,ly)*ipow(dz,lz);`
`let cdz = radial_2a*dz*mono + radial*ipow(dx,lx)*ipow(dy,ly)*dpow(dz,lz);`
(operand order EXACTLY matches host eval_gto.rs 1382-1387); then `let t =
c2s_flat[c2s_off + m*ncart_l + ci];` and accumulate `v += t*cval; vx += t*cdx; vy
+= t*cdy; vz += t*cdz;` into four `let mut` accumulators declared per-m (reset each
m). After the ci loop, `let off = g + (ao_off + c_idx*nsph_l + m)*ngrids;` and
write out[off]=v, out[comp_stride+off]=vx, out[2*comp_stride+off]=vy,
out[3*comp_stride+off]=vz. Follow cubecl 0.10 idioms VERBATIM from the shipped
general kernel (usize scalar args, ABSOLUTE_POS usize, i32->u32 casts, `.exp()`
method, CubeDim::new_1d, 2-arg ArrayArg::from_raw_parts). Do NOT place fenced code
blocks; this is directive prose. The math identifiers above name the exact
operands the executor must emit as real cubecl statements.

(c) Launcher: add `#[allow(clippy::too_many_arguments)] fn launch_eval_gto_deriv1<R:
Runtime>(client, coords, ngrids, atm, bas, env, ao_loc, nao, maxl) ->
Result<Vec<f64>, PyscfRsError>` — clone launch_eval_gto_general byte-for-byte
EXCEPT: `let out_len = 4 * ngrids * nao;`, call
`eval_gto_sph_deriv1_kernel::launch_unchecked::<R>` with the matching
`ArrayArg::from_raw_parts(out_handle.clone(), out_len)` and pass the extra
`ngrids * nao` (comp_stride) scalar AFTER the existing trailing scalar args.
groups = `(ngrids*nbas).div_ceil(EVAL_GTO_BLOCK as usize) as u32`.

(d) Routing: rewrite the body of the EXISTING `pub fn eval_gto_sph_deriv1`
(eval_gto.rs 1234-1252) to mirror eval_gto_sph's mlg wiring. Compute
`let maxl = bas.chunks_exact(BAS_SLOTS).map(|r| r[ANG_OF]).max().unwrap_or(0) as
u32;` and `let comp_stride = ngrids * nao;`. If `!bas.is_empty() && maxl <= 4 &&
comp_stride > 0` → `let values = dispatch_backend!(client, c, Rt,
launch_eval_gto_deriv1::<Rt>(c, coords, ngrids, atm, bas, env, ao_loc, nao,
maxl))?;` and `return Ok(EvalGtoBuffers { values, shape: vec![4, ngrids, nao] });`.
ELSE (maxl>4 / empty basis / empty grid) → the UNCHANGED `eval_gto_sph_deriv1_cpu(
coords, ngrids, atm, bas, env, ao_loc, nao)`. Drop the old `client.kind()`
warn-guard. Do NOT modify `eval_gto_sph_deriv1_cpu` (it stays the l>4 fallback,
preserving NotYetImplemented{phase:4} and the comp_stride==0 early-return).

Builds: `cargo build -p pyscf-kernels 2>&1 | tee log/oms-t1-build.log` and
`cargo build -p pyscf-kernels --features rocm 2>&1 | tee log/oms-t1-build-rocm.log`;
clippy `cargo clippy -p pyscf-kernels --all-targets 2>&1 | tee log/oms-t1-clippy.log`.
NEVER run anything that pulls pyscf-gto/libxc (~6h compile). If a GENUINE cubecl
0.10 blocker appears (dpow-calls-ipow rejected, no-Vec scratch infeasible, etc.),
STOP and document it in the summary — do NOT silently descope.
  </action>
  <verify>
    <automated>cargo build -p pyscf-kernels 2>&1 | tee log/oms-t1-build.log | tail -5; cargo build -p pyscf-kernels --features rocm 2>&1 | tee log/oms-t1-build-rocm.log | tail -5; grep -v '^#' crates/pyscf-kernels/src/eval_gto.rs | grep -c 'fn eval_gto_sph_deriv1_kernel\|fn launch_eval_gto_deriv1\|fn dpow'</automated>
  </verify>
  <done>
dpow `#[cube]` helper, `eval_gto_sph_deriv1_kernel` `#[cube]`, and
`launch_eval_gto_deriv1<R>` exist; `eval_gto_sph_deriv1` dispatches to
`launch_eval_gto_deriv1` via `dispatch_backend!` for maxl<=4 and comp_stride>0,
and falls back to the UNCHANGED `eval_gto_sph_deriv1_cpu` otherwise. Both cpu and
rocm builds compile (no libxc pulled); clippy clean (pre-existing fma4
target-feature warning excepted). The grep count returns 3.
  </done>
</task>

<task type="auto">
  <name>Task 2: deriv1 4-component differential oracle (CpuRuntime + #[cfg(rocm)] gfx1152) — RUN</name>
  <files>crates/pyscf-kernels/tests/eval_gto_oracle.rs</files>
  <action>
Read docs/rust_crate_test_guideline.md before writing tests. Extend
tests/eval_gto_oracle.rs with a deriv1 4-component differential case, reusing the
existing mlg scaffolding (Lcg, max_abs_diff, TOL=1e-9, MixedFixture,
build_mixed_l_fixture, and the lge1_reference module whose
c2s_coeff/cart_powers/common_fac_sp/ncart/nsph already cover l 0..=4 VERBATIM from
the frozen tables).

Ground truth = an INDEPENDENT longhand deriv1 reference (a DIFFERENT code path
from the kernel), NOT eval_gto_sph_deriv1(cpu_client) (which now routes to the
kernel). Add `fn oracle_eval_deriv1(f: &MixedFixture) -> Vec<f64>` that ports the
radial/radial_2a/cval/cdx/cdy/cdz/c2s formula from eval_gto_sph_deriv1_cpu
(eval_gto.rs 1307-1410) but using lge1_reference::*. CRITICAL fidelity points:
(1) sum radial AND radial_2a SEQUENTIALLY in one p-loop (form g0 = coef·exp(-α·r2)
once; radial += g0; radial_2a += -2.0·α·g0), THEN multiply both by fac1 — match
the kernel's sequential acc, do NOT use oracle_sum (ORACLE-07: the kernel is
sequential). (2) define a local `fn dpow(q: f64, lq: u32) -> f64 { if lq==0 {0.0}
else { lq as f64 * q.powi(lq as i32 - 1) } }` (host longhand). (3) out length
4*ngrids*nao, comp_stride = ngrids*nao, write out[off]/out[comp_stride+off]/
out[2*comp_stride+off]/out[3*comp_stride+off], return-shape conceptually
[4, ngrids, nao]. (4) cdx/cdy/cdz operand order IDENTICAL to host (radial_2a·dq·mono
+ radial·dpow(dq,lq)·other_mono·other_mono).

Add `fn check_deriv1_case(client, n_atoms, l_pattern, shells_per_atom, ngrids,
seed) -> f64`: build a MixedFixture via build_mixed_l_fixture, call
`eval_gto_sph_deriv1(client, &f.coords, f.ngrids, &f.atm, &f.bas, &f.env,
&f.ao_loc, f.nao)`, assert `device.shape == vec![4, f.ngrids, f.nao]` and
`device.values.len() == 4*f.ngrids*f.nao`, diff ALL 4 components against
oracle_eval_deriv1 via max_abs_diff over the full flat buffer, return the max.
Add `eval_gto_sph_deriv1` to the existing `use pyscf_kernels::...` import line.

Add `const DERIV1_CASES: &[(usize, &[u32], usize, usize)]` mirroring MIXED_CASES:
pure p &[1], pure d &[2], pure f &[3], pure g &[4], mixed s+p+d &[0,1,2], mixed
p+d+f over 2 atoms &[1,2,3], full s..g &[0,1,2,3,4], d+f over 3 atoms &[2,3] — with
varied ngrids; build_mixed_l_fixture already randomizes nprim 1..4 (exercises the
>2-prim path) and nctr 1..2.

Add `#[test] fn eval_gto_deriv1_matches_oracle_on_cpu()` (always-on, CpuRuntime via
cubecl_cpu, distinct seed base e.g. 0xDE71_0001) and `#[cfg(feature = "rocm")]
#[test] fn eval_gto_deriv1_matches_oracle_on_rocm()` (HipRuntime::client on
AmdDevice::default, assert matches!(client, AlgebraClient::Rocm(_)), distinct seed
base e.g. 0xDE71_F00D). Both loop DERIV1_CASES, `worst = worst.max(diff)`,
`assert!(diff < TOL, ...)` with a descriptive message, and `eprintln!(
"[eval_gto_oracle] {CPU|ROCm} deriv1 (4-comp p/d/f/g) worst max_abs_diff = {worst:e}")`.

RUN both arms (do not just compile the rocm one):
`cargo test -p pyscf-kernels --test eval_gto_oracle 2>&1 | tee log/oms-t2-cpu.log`
and `cargo test -p pyscf-kernels --features rocm --test eval_gto_oracle 2>&1 | tee
log/oms-t2-rocm.log`. Capture both worst max_abs_diff values from the logs for the
summary. NEVER run -p pyscf-gto (libxc/cintx ~6h).
  </action>
  <verify>
    <automated>cargo test -p pyscf-kernels --test eval_gto_oracle 2>&1 | tee log/oms-t2-cpu.log | tail -8; cargo test -p pyscf-kernels --features rocm --test eval_gto_oracle eval_gto_deriv1_matches_oracle_on_rocm 2>&1 | tee log/oms-t2-rocm.log | tail -8</automated>
  </verify>
  <done>
oracle_eval_deriv1 (independent longhand, sequential radial+radial_2a, host
dpow, 4-component F-order write), check_deriv1_case, DERIV1_CASES, and both test
fns exist. CPU arm passes all DERIV1_CASES (worst max_abs_diff < 1e-9, all 4
components diffed); the ROCm arm RAN on gfx1152 and passed. No libxc pulled (scoped
-p pyscf-kernels). Worst max_abs_diff for cpu and rocm recorded in the summary.
  </done>
</task>

<task type="auto">
  <name>Task 3: behavior-preservation full-suite run (default cpu + rocm eval_gto_oracle)</name>
  <files>crates/pyscf-kernels/src/eval_gto.rs, crates/pyscf-kernels/tests/eval_gto_oracle.rs</files>
  <action>
No code changes (verification-only gate). Confirm the deriv1 device routing is
behavior-preserving and nothing regressed. Run the FULL default-cpu suite and the
rocm eval_gto_oracle suite, scoped to pyscf-kernels (NEVER -p pyscf-gto — pulls
libxc/cintx, ~6h):
- `cargo test -p pyscf-kernels 2>&1 | tee log/oms-t3-cpu-full.log` — expect zero
  FAILED: lib, eval_gto_oracle (s-shell + mixed-l + the NEW deriv1 4-comp cpu),
  eval_gto_lge1, wave0 smoke, and the l>4 never-panic gate
  (c2s_coeff_l5_returns_err_not_panic) ALL green. The l>4 / empty paths must still
  hit the UNCHANGED eval_gto_sph_deriv1_cpu fallback.
- `cargo test -p pyscf-kernels --features rocm --test eval_gto_oracle 2>&1 | tee
  log/oms-t3-rocm.log` — all 6 oracle tests (s-shell cpu/rocm, mixed-l cpu/rocm,
  deriv1 cpu/rocm) green on gfx1152.
- `cargo clippy -p pyscf-kernels --all-targets 2>&1 | tee log/oms-t3-clippy.log` —
  clean (pre-existing fma4 target-feature warning excepted).
Record the pass/fail counts in the summary. If any pre-existing test regressed,
STOP and document (do NOT mask by deleting/skipping tests).
  </action>
  <verify>
    <automated>cargo test -p pyscf-kernels 2>&1 | tee log/oms-t3-cpu-full.log | grep -E 'test result|FAILED'; cargo test -p pyscf-kernels --features rocm --test eval_gto_oracle 2>&1 | tee log/oms-t3-rocm.log | grep -E 'test result|FAILED'</automated>
  </verify>
  <done>
`cargo test -p pyscf-kernels` (default cpu) reports zero FAILED across lib +
eval_gto_oracle (incl. new deriv1) + eval_gto_lge1 + wave0 + l>4 never-panic gate.
`cargo test -p pyscf-kernels --features rocm --test eval_gto_oracle` reports all 6
oracle tests pass on gfx1152. Clippy clean. No libxc pulled. Counts recorded in
the summary.
  </done>
</task>

</tasks>

<threat_model>
## Trust Boundaries

| Boundary | Description |
|----------|-------------|
| host slice → device Array upload | flat libcint atm/bas/env/ao_loc + angular tables copied to the GPU; lengths must match handle sizes |
| #[cube] kernel index math | per-thread reads into env/c2s_flat/cpow/out; out-of-range write corrupts neighbouring AO slots silently |
| package installs | none — no new crates added (reuses cubecl 0.10 already in the dep graph) |

## STRIDE Threat Register

| Threat ID | Category | Component | Disposition | Mitigation Plan |
|-----------|----------|-----------|-------------|-----------------|
| T-oms-01 | Tampering | eval_gto_sph_deriv1_kernel out-buffer writes (4 components × comp_stride) | mitigate | clone launch_eval_gto_general's verified upload/handle-length pattern verbatim; out_len=4*ngrids*nao and comp_stride=ngrids*nao asserted via the T2 shape/length checks; bounds-guard `if tid < ngrids*nbas` |
| T-oms-02 | Information disclosure | reading wrong env offset (radial_2a chain) yields garbage ∇ρ feeding DFT | mitigate | differential oracle (T2) diffs ALL 4 components vs an INDEPENDENT longhand reference to <1e-9 over p/d/f/g + mixed-l; wrong-offset bug surfaces as diff > TOL |
| T-oms-03 | Denial of service | l>4 / empty basis reaching the device tables (c2s_coeff panics for l>4) | accept | routing keeps maxl<=4 on device; l>4/empty fall through to UNCHANGED eval_gto_sph_deriv1_cpu (NotYetImplemented{phase:4}); guarded by the c2s_coeff_l5_returns_err_not_panic gate (T3) |
| T-oms-SC | Tampering | npm/pip/cargo installs | accept | no new packages — reuses cubecl 0.10 / cubecl_hip already vetted in the mlg + ljv slices; no Package Legitimacy Gate triggered |
</threat_model>

<verification>
- `cargo build -p pyscf-kernels` and `... --features rocm` compile (no libxc).
- `cargo test -p pyscf-kernels` (default cpu): zero FAILED, incl. the new deriv1
  4-component oracle, the mlg/ljv oracles, eval_gto_lge1, and the l>4 never-panic gate.
- `cargo test -p pyscf-kernels --features rocm --test eval_gto_oracle`: all 6
  oracle tests pass on real gfx1152 hardware.
- `cargo clippy -p pyscf-kernels --all-targets`: clean (pre-existing fma4 warning aside).
- All cargo output saved under log/oms-*.log.
- NO `-p pyscf-gto` invocation anywhere (libxc/cintx ~6h compile prohibited).
</verification>

<success_criteria>
- `eval_gto_sph_deriv1` returns a device-computed `[4, ngrids, nao]` buffer (value
  + ∂x/∂y/∂z) for any non-empty basis with maxl<=4, on both CpuRuntime and ROCm.
- Device output matches the independent longhand deriv1 reference to <1e-9 over
  randomized p/d/f/g and mixed-l fixtures, ALL 4 components.
- l>4 / empty basis / empty grid still route to the UNCHANGED
  eval_gto_sph_deriv1_cpu (NotYetImplemented{phase:4} preserved; never panics).
- Full `cargo test -p pyscf-kernels` green; rocm eval_gto_oracle green on gfx1152;
  clippy clean. This COMPLETES the eval_gto device surface (deriv2 stays
  NotYetImplemented — out of scope, no CPU impl to port).
</success_criteria>

<output>
Create `.planning/quick/260530-oms-gpu-enable-eval-gto-sph-deriv1-value-3-c/260530-oms-SUMMARY.md` when done.
</output>
