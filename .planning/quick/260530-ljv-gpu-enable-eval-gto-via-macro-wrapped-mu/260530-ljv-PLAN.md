---
phase: quick-260530-ljv
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - crates/pyscf-algebra/src/lib.rs
  - crates/pyscf-algebra/src/dispatch.rs
  - crates/pyscf-kernels/src/eval_gto.rs
  - crates/pyscf-kernels/tests/eval_gto_oracle.rs
autonomous: true
requirements: [ORACLE-07, PERF-07]
must_haves:
  truths:
    - "dispatch_backend! is callable from pyscf-kernels (a downstream crate) and expands to the cfg-gated AlgebraClient fanout"
    - "pyscf-algebra's own 16 in-crate dispatch_backend! call sites still build after the macro_export change"
    - "eval_gto_sph on a pure s-shell basis runs a real #[cube(launch_unchecked)] kernel on the resolved backend and matches the CPU oracle within 1e-9"
    - "eval_gto_sph on any basis containing an l>=1 shell still returns the byte-identical existing CPU-path result"
    - "the new differential-oracle test passes on the always-on CpuRuntime arm (default features)"
  artifacts:
    - path: "crates/pyscf-algebra/src/dispatch.rs"
      provides: "dispatch_backend! hoisted to crate root via #[macro_export], importable cross-crate"
      contains: "macro_export"
    - path: "crates/pyscf-kernels/src/eval_gto.rs"
      provides: "s-shell #[cube(launch_unchecked)] kernel + macro-wrapped multi-backend launch fanout + all-l0 routing in eval_gto_sph"
      contains: "#[cube(launch_unchecked)]"
    - path: "crates/pyscf-kernels/tests/eval_gto_oracle.rs"
      provides: "differential oracle test: cube s-shell kernel vs eval_gto_sph_cpu, CpuRuntime always-on + #[cfg(feature=rocm)] gfx1152 arm"
      min_lines: 80
  key_links:
    - from: "crates/pyscf-kernels/src/eval_gto.rs"
      to: "pyscf_algebra::dispatch_backend!"
      via: "use pyscf_algebra::dispatch_backend; macro-wrapped launch fanout"
      pattern: "dispatch_backend!"
    - from: "crates/pyscf-kernels/src/eval_gto.rs"
      to: "eval_gto_sph_kernel cube launch"
      via: "every-shell-l0 fast path routes to the device kernel, else eval_gto_sph_cpu"
      pattern: "launch_unchecked"
    - from: "crates/pyscf-kernels/tests/eval_gto_oracle.rs"
      to: "eval_gto_sph_cpu ground truth"
      via: "differential comparison over randomized s-shell fixtures"
      pattern: "eval_gto_sph"
---

<objective>
GPU-enable `eval_gto` via a macro-wrapped multi-backend cube kernel fanout — Phase 8
GPU-enable, **first tractable slice**.

Today `eval_gto_sph` runs a `client.kind()` warn-guard then `eval_gto_sph_cpu`
(a pure host Rust loop): there is NO `#[cube]` launch and NO backend fanout. This
plan builds the FOUNDATION (`dispatch_backend!` exported cross-crate) plus the
FIRST real GPU compute path — the l=0 (s-shell) radial slice
`out = Σ_p coeff[c,p]·exp(-α_p·r²) · Y00`, F-order write `out[g + ao_idx*ngrids]`
— routed through that macro, and validates it with a differential oracle test
against the existing CPU longhand.

Behavior-preserving by construction: only pure-s-shell bases take the new device
path; ANY basis with an l>=1 shell falls back to the unchanged
`eval_gto_sph_cpu`, so every existing test passes byte-for-byte.

Purpose: prove the export + cube-kernel + oracle pattern on the smallest fully
validatable eval_gto slice, so the staged remainder drops in against a proven
template. Addresses ORACLE-07 (GPU at documented per-backend tolerance) and
PERF-07 (adaptive CPU fallback) on the eval_gto surface.

Output:
  * `dispatch_backend!` exported from pyscf-algebra (`#[macro_export]`), usable by
    pyscf-kernels (and every future downstream wall-allowlisted crate).
  * An s-shell `#[cube(launch_unchecked)]` kernel + host-slice launcher in
    pyscf-kernels, fanned out over all backends via the imported macro.
  * `eval_gto_sph` routes pure-s-shell bases to the device kernel, everything else
    to the existing CPU path.
  * A differential-oracle test (CpuRuntime always-on + rocm/gfx1152 arm).

### Staged remainder (DEFERRED — named, NOT executed this plan)
The full eval_gto GPU arc continues after this slice is validated. Do NOT pad
this plan with any of these; the user steers whether to continue:
  1. **l>=1 cart→sph kernel** — port the cartesian-monomial + libcint c2s
     transform path (`cart_powers`, `c2s_coeff`, `common_fac_sp`) into a cube
     kernel. Blocked-by: the "calling a normal Rust fn from inside #[cube]" pitfall
     (c2s tables must become device arrays or `#[comptime]` data).
  2. **deriv1 / deriv2 stencils** — the `eval_gto_sph_deriv1` analytic-gradient
     path (and the future deriv2) as cube kernels.
  3. **pyscf-dft numint cubecl backend** — wire the DFT grid-loop (numint) to call
     the device eval_gto path instead of the host loop.
</objective>

<execution_context>
@/home/user/Documents/workspace/pyscf_rs/.claude/get-shit-done/workflows/execute-plan.md
@/home/user/Documents/workspace/pyscf_rs/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@.planning/STATE.md
@.planning/ROADMAP.md
@./CLAUDE.md

# MANDATORY cubecl references (manual is STALE for the pinned 0.10.0 — follow the
# working gemm.rs template, NOT the manual examples)
@docs/manual/Cubecl/cubecl_macro_fanout_manual.md
@docs/manual/Cubecl/cubecl_error_solution_guide/

# TEMPLATES — copy these patterns exactly
@crates/pyscf-algebra/src/gemm.rs
@crates/pyscf-algebra/tests/gemm_oracle.rs
@crates/pyscf-kernels/tests/wave0_cubecl_smoke.rs

# THE TARGET (read fully)
@crates/pyscf-kernels/src/eval_gto.rs

<interfaces>
<!-- Contracts the executor needs. Extracted from the codebase — no exploration required. -->

From crates/pyscf-algebra/src/dispatch.rs (CURRENT — `macro_rules!` with NO export):
  macro_rules! dispatch_backend {
      ($client:expr, $c:ident, $rt:ident, $body:expr) => { match $client { ... } }
  }
  - Cpu arm:  type $rt = cubecl_cpu::CpuRuntime;
  - cuda arm: type $rt = cubecl_cuda::CudaRuntime;   (#[cfg(feature="cuda")])
  - wgpu arm: type $rt = cubecl_wgpu::WgpuRuntime;   (#[cfg(feature="wgpu")])
  - rocm arm: type $rt = cubecl_hip::HipRuntime;     (#[cfg(feature="rocm")])
  - Matches `$crate::AlgebraClient::{Cpu,Cuda,Wgpu,Rocm}`.
  - The bare runtime paths (`cubecl_cpu::CpuRuntime`, etc.) resolve in the
    CALLER's namespace — pyscf-kernels already has cfg-aligned cubecl-cpu/cuda/
    wgpu/hip optional deps + matching cpu/cuda/wgpu/rocm features.

From crates/pyscf-algebra/src/lib.rs:15-18 (CURRENT wiring to change):
  #[macro_use]
  mod dispatch;          // crate-internal, unqualified in-crate calls

From crates/pyscf-algebra/src/client.rs:
  pub enum AlgebraClient {
      Cpu(ComputeClient<cubecl_cpu::CpuRuntime>),
      #[cfg(feature="cuda")]  Cuda(ComputeClient<cubecl_cuda::CudaRuntime>),
      #[cfg(feature="wgpu")]  Wgpu(ComputeClient<cubecl_wgpu::WgpuRuntime>),
      #[cfg(feature="rocm")]  Rocm(ComputeClient<cubecl_hip::HipRuntime>),
  }
  pub fn kind(&self) -> BackendKind;   // Cpu/Cuda/Wgpu/Rocm

From crates/pyscf-algebra/src/gemm.rs (TEMPLATE kernel + launcher shape):
  #[cube(launch)]
  fn gemm_kernel<F: Float>(lhs: &Array<F>, rhs: &Array<F>, out: &mut Array<F>,
                           m: usize, k: usize, n: usize) {
      let tid = ABSOLUTE_POS;                 // usize in 0.10
      if tid < m*n { let mut acc = F::from_int(0); for j in 0..k { acc += ...; } out[tid]=acc; }
  }
  const BLOCK: u32 = 256;
  gemm_kernel::launch::<F, R>(client, CubeCount::Static(groups,1,1),
      CubeDim::new_1d(BLOCK),
      unsafe { ArrayArg::from_raw_parts(handle.clone(), len) },   // 2-arg, consumes handle
      m, k, n);                                // bare scalar args
  let lhs_handle = client.create(Bytes::from_elems(lhs.to_vec()));
  let out_handle = client.empty(len * core::mem::size_of::<F>());
  let bytes = client.read(vec![out_handle]); bytemuck::cast_slice::<u8,F>(&bytes[0]).to_vec()

From crates/pyscf-kernels/src/eval_gto.rs (the surface + the l=0 math to port):
  pub struct EvalGtoBuffers { pub values: Vec<f64>, pub shape: Vec<usize> }
  pub fn eval_gto_sph(client:&AlgebraClient, coords:&[f64], ngrids:usize,
      atm:&[i32], bas:&[i32], env:&[f64], ao_loc:&[i32], nao:usize, spherical:bool)
      -> Result<EvalGtoBuffers, PyscfRsError>
  // l=0 host math (eval_gto_sph_cpu, lines 590-615 — port THIS into the kernel):
  //   y00 = 0.5 / PI.sqrt()
  //   per shell: dx=gx-ax; r2=dx²+dy²+dz²
  //   per c_idx in 0..nctr: acc=0; for p_idx in 0..nprim {
  //       alpha=env[ptr_exp+p_idx]; coef=env[ptr_coeff + c_idx*nprim + p_idx];
  //       acc += coef * (-alpha*r2).exp(); }                 // ORDERED, sequential acc
  //   out[g + (ao_off + c_idx)*ngrids] = acc * y00;
  // libcint flat-array slot constants (re-exported via pyscf_core::raw_layout):
  //   ATM_SLOTS, BAS_SLOTS, ATOM_OF, ANG_OF, NPRIM_OF, NCTR_OF, PTR_EXP,
  //   PTR_COEFF, PTR_COORD
</interfaces>
</context>

<tasks>

<task type="auto">
  <name>Task 1: Export dispatch_backend! cross-crate; confirm pyscf-algebra still builds</name>
  <files>crates/pyscf-algebra/src/dispatch.rs, crates/pyscf-algebra/src/lib.rs</files>
  <action>
Make `dispatch_backend!` callable from downstream crates so pyscf-kernels can fan
out over the AlgebraClient backends without re-deriving the cfg-gated match.

In dispatch.rs: add `#[macro_export]` immediately above `macro_rules! dispatch_backend`.
`#[macro_export]` hoists the macro to the CRATE ROOT (`pyscf_algebra::dispatch_backend`),
which changes resolution for the EXISTING in-crate call sites. The macro body already
uses `$crate::AlgebraClient` and bare runtime paths, so no body change is needed.

In lib.rs (currently lines 15-18 `#[macro_use] mod dispatch;`): a `#[macro_export]`
macro is available at the crate root by name from anywhere in the crate AFTER its
textual definition, but the existing engine modules (axpy/dot/gemm/gemv/scal/
transpose/reduce/device_buffer — 16 call sites across these files) call it
UNQUALIFIED relying on `#[macro_use]` textual hoisting. After adding `#[macro_export]`,
verify those unqualified call sites still resolve. The robust pattern: keep `mod dispatch;`
(drop `#[macro_use]` since `#[macro_export]` makes it crate-root-global), and if any
in-crate call fails to resolve, add `use crate::dispatch_backend;` at the top of the
affected module(s) OR retain `#[macro_use] mod dispatch;` alongside `#[macro_export]`
(macro_use + macro_export coexist — macro_export is the cross-crate export, macro_use
keeps the unqualified in-crate textual scope). Prefer retaining `#[macro_use]` if the
build is clean — minimal diff, all 16 sites untouched. Update the module doc comment on
lib.rs:14-16 to note the macro is now ALSO exported cross-crate for pyscf-kernels.

Do NOT add it to lib.rs `pub use` list — `#[macro_export]` already publishes it at the
crate root; a `pub use self::dispatch::dispatch_backend;` would double-export and warn.

Save full cargo output to log/ before investigating any failure.
  </action>
  <verify>
    <automated>cargo build -p pyscf-algebra 2>&1 | tee log/ljv-task1-algebra-build.log | tail -5; grep -q "error\[" log/ljv-task1-algebra-build.log && exit 1 || true</automated>
    <automated>cargo test -p pyscf-algebra --lib 2>&1 | tee log/ljv-task1-algebra-test.log | tail -15</automated>
  </verify>
  <done>
`#[macro_export]` present on dispatch_backend!. `cargo build -p pyscf-algebra` and
`cargo test -p pyscf-algebra --lib` both succeed — confirming all 16 in-crate call
sites (axpy/dot/gemm/gemv/scal/transpose/reduce/device_buffer) still resolve. No
new warnings about unused/double macro export.
  </done>
</task>

<task type="auto" tdd="true">
  <name>Task 2: s-shell #[cube] kernel + macro-wrapped fanout + all-l0 routing in eval_gto</name>
  <files>crates/pyscf-kernels/src/eval_gto.rs</files>
  <behavior>
    - Pure-s-shell basis (all bas ANG_OF == 0): eval_gto_sph routes to the device
      kernel; result == eval_gto_sph_cpu within 1e-9 over the same inputs.
    - Basis with any l>=1 shell: eval_gto_sph returns the byte-identical existing
      eval_gto_sph_cpu result (no kernel launch — fallback path unchanged).
    - Empty grid (ngrids==0) or out_len==0: returns empty EvalGtoBuffers with
      shape [ngrids, nao] (same early-return as the host path).
    - F-order layout preserved: out[g + ao_idx*ngrids], ao_idx = ao_off + c_idx.
  </behavior>
  <action>
Add a real GPU compute path for the l=0 (s-shell) slice and route pure-s-shell
bases to it; everything else keeps the existing CPU path byte-for-byte.

(a) KERNEL — add `#[cube(launch_unchecked)] fn eval_gto_sph_kernel<F: Float>(...)`.
One thread per (grid-point g, contraction output ao_idx) flattened output element,
mirroring gemm_kernel's one-thread-per-output shape. The kernel needs device arrays
the GPU can index — upload the libcint flat arrays as cube `Array` args. Recommended
encoding (decide and state in the summary): coords as `&Array<F>` (length ngrids*3,
F-order), env as `&Array<F>`, and bas/atm/ao_loc as `&Array<i32>` (cubecl 0.10
indexes i32 arrays fine). Pass ngrids, nbas, nao and the libcint slot constants
(ATM_SLOTS/BAS_SLOTS/ATOM_OF/ANG_OF/NPRIM_OF/NCTR_OF/PTR_EXP/PTR_COEFF/PTR_COORD) as
bare `usize`/`u32` scalar args (LaunchArg for T=T, like gemm's m/k/n) — do NOT use
host-only helper fns inside the kernel (the "calling a normal Rust fn from inside
#[cube]" pitfall). y00 = 0.5/π.sqrt() is a constant; compute it as `F::from_int(1)`
scaled or pass it as a bare `F` scalar arg (cleanest — avoids π inside #[cube]).
Inside the kernel, replicate the host l=0 loop EXACTLY: per output thread resolve its
(g, shell, c_idx), compute r2, then the ORDERED sequential accumulation
`acc += coef * (-alpha*r2).exp()` over p_idx in 0..nprim (NOT a tree/parallel reduce
— bit-exact discipline, matches oracle_sum's ordered shape for the s-shell single-
thread case), then `out[g + ao_idx*ngrids] = acc * y00`. Bounds-guard the tail
(`if tid < ngrids*nao`). Inlined `(-alpha*r2).exp()` is verified to work inside
#[cube] per the eval_gto.rs header — confirm it compiles; if `.exp()` on the generic
`F` fails, restrict the kernel to f64 (the chemistry precision path) rather than
generic Float.

(b) LAUNCHER — add a `launch_eval_gto_s<R: Runtime>(client: &ComputeClient<R>, ...)
-> Vec<f64>` host-slice launcher modeled on gemm.rs `launch_gemm`: client.create the
coords/env/bas/atm/ao_loc handles, client.empty the out handle (ngrids*nao*size_of
::<f64>()), call `eval_gto_sph_kernel::launch_unchecked::<R>(...)` with
CubeCount::Static(groups,1,1) / CubeDim::new_1d(BLOCK) (BLOCK=256, groups =
(ngrids*nao).div_ceil(BLOCK)), ArrayArg::from_raw_parts(handle.clone(), len) per
array (2-arg form, consumes handle), then client.read + bytemuck::cast_slice back.

(c) FANOUT + ROUTING — at the top of eval_gto.rs add
`use pyscf_algebra::dispatch_backend;`. In `eval_gto_sph`, REPLACE the current
`client.kind()` warn-guard + unconditional `eval_gto_sph_cpu` call with: compute
`all_s = bas.chunks(BAS_SLOTS).all(|row| row[ANG_OF] == 0)` (guard nbas==0 / empty
grid → keep returning via the existing empty early-return, so route those to the CPU
path which already handles out_len==0). If `all_s && !bas.is_empty() && ngrids*nao>0`:
build the result via `let values = dispatch_backend!(client, c, Rt,
launch_eval_gto_s::<Rt>(c, coords, ngrids, atm, bas, env, ao_loc, nao));` and return
`EvalGtoBuffers { values, shape: vec![ngrids, nao] }`. ELSE: call the unchanged
`eval_gto_sph_cpu(...)` (the l>=1 fallback — NEVER changes l>=1 numerics). The
`spherical` flag for l=0 is a no-op (Y00 identity) so the device path ignores it
exactly as the host path does for s-shells. Keep `eval_gto_sph_cpu` and
`eval_gto_sph_deriv1*` entirely unchanged.

cubecl 0.10 idioms (follow gemm.rs, manual is STALE): ABSOLUTE_POS is usize;
CubeDim::new_1d(BLOCK); ArrayArg::from_raw_parts(handle, len) is 2-arg; bare scalar
args; launch_unchecked::<R>. Save full cargo output to log/ before investigating.
  </action>
  <verify>
    <automated>cargo build -p pyscf-kernels 2>&1 | tee log/ljv-task2-kernels-build.log | tail -8; grep -q "error\[" log/ljv-task2-kernels-build.log && exit 1 || true</automated>
    <automated>grep -v '^[[:space:]]*//' crates/pyscf-kernels/src/eval_gto.rs | grep -c "launch_unchecked" | { read n; [ "$n" -ge 1 ] || { echo "no cube launch in eval_gto"; exit 1; }; }</automated>
    <automated>cargo test -p pyscf-kernels 2>&1 | tee log/ljv-task2-kernels-test.log | tail -20</automated>
  </verify>
  <done>
pyscf-kernels builds (no libxc pulled). `eval_gto_sph` contains a
`#[cube(launch_unchecked)]` kernel reached via `dispatch_backend!` for pure-s-shell
bases, and falls back to the unchanged `eval_gto_sph_cpu` otherwise. ALL existing
pyscf-kernels tests pass (l>=1 paths byte-identical; s-shell smoke fixtures still
green).
  </done>
</task>

<task type="auto" tdd="true">
  <name>Task 3: differential-oracle test (CpuRuntime always-on + rocm gfx1152 arm)</name>
  <files>crates/pyscf-kernels/tests/eval_gto_oracle.rs</files>
  <behavior>
    - Over randomized s-shell fixtures (varied ngrids, nbas, nprim, nctr,
      exponents, coefficients, atom centers), the cube s-shell path matches
      eval_gto_sph_cpu within TOL = 1e-9.
    - The CpuRuntime test runs under default features (always-on gate).
    - The rocm test (#[cfg(feature="rocm")]) runs the SAME check on gfx1152 via
      HipRuntime; clients constructed directly (no select_backend) to dodge the
      PYSCF_BACKEND env race.
  </behavior>
  <action>
Create the differential-oracle test, modeled on
crates/pyscf-algebra/tests/gemm_oracle.rs (read docs/rust_crate_test_guideline.md
first per CLAUDE.md). Ground truth = `eval_gto_sph_cpu` (the FMA-free host longhand).
Device under test = the new cube path reached via `eval_gto_sph` on a pure-s-shell
basis (which now routes to the kernel).

Reuse gemm_oracle.rs's `Lcg` LCG + `max_abs_diff`. Add a fixture builder that
constructs a valid pure-s-shell libcint flat-array layout (atm/bas/env/ao_loc) for
randomized parameters — keep it minimal: N atoms at random centers, each with one or
more l=0 shells, nprim in {1,2,3,4} (exercise the >2-prim ordered-sum path),
nctr in {1,2}, random positive exponents and random coefficients packed F-order into
env at PTR_EXP/PTR_COEFF, ao_loc built as the running sum of nctr per shell. Mirror
the slot layout `eval_gto_sph_cpu` reads (BAS_SLOTS row: ATOM_OF, ANG_OF=0,
NPRIM_OF, NCTR_OF, PTR_EXP, PTR_COEFF). Build random F-order coords (ngrids*3).

For each fixture: call eval_gto_sph with an `AlgebraClient::Cpu(...)` (device path,
all-s → kernel) and compare against `eval_gto_sph_cpu(...)` called directly (the
oracle). assert max_abs_diff < TOL.

const TOL: f64 = 1e-9 with a comment: f-order single-thread sequential accumulation
in the kernel matches the host ordered acc within 1e-9 (not necessarily bit-identical
because the device exp() implementation may differ from std f64::exp by <1 ULP per
term; gemm_oracle uses the same 1e-9 bound).

Two tests:
  - `eval_gto_s_matches_oracle_on_cpu` — ALWAYS-ON (default cpu feature). Construct
    `AlgebraClient::Cpu(cubecl_cpu::CpuRuntime::client(&cubecl_cpu::CpuDevice))`
    directly (NOT select_backend — avoid the PYSCF_BACKEND race, per gemm_oracle.rs).
  - `#[cfg(feature="rocm")] eval_gto_s_matches_oracle_on_rocm` — construct
    `AlgebraClient::Rocm(cubecl_hip::HipRuntime::client(&cubecl_hip::AmdDevice::default()))`,
    assert matches!(client, AlgebraClient::Rocm(_)), run the SAME fixtures. This is
    the real-GPU confirmation on gfx1152; correctness is already PROVEN by the
    always-on CpuRuntime arm even if rocm linking is unavailable.

NOTE the eval_gto_sph_cpu oracle is a private fn; to compare against it from a tests/
integration file, either (i) re-derive the same l=0 longhand inline in the test as the
oracle (cleanest — keeps the test self-contained and the production fn private), or
(ii) if a crate-internal test is preferable, add a `#[cfg(test)] mod` inside
eval_gto.rs instead. Prefer (i): inline the y00·Σ coef·exp(-α·r2) longhand as the test
oracle so the differential check does not depend on production internals. Save full
cargo output to log/ before investigating.
  </action>
  <verify>
    <automated>cargo test -p pyscf-kernels --test eval_gto_oracle 2>&1 | tee log/ljv-task3-oracle-cpu.log | tail -20; grep -qE "test result: ok|running 0 tests" log/ljv-task3-oracle-cpu.log || exit 1</automated>
    <automated>grep -v '^[[:space:]]*//' crates/pyscf-kernels/tests/eval_gto_oracle.rs | grep -c 'cfg(feature = "rocm")' | { read n; [ "$n" -ge 1 ] || { echo "missing rocm arm"; exit 1; }; }</automated>
  </verify>
  <done>
`cargo test -p pyscf-kernels --test eval_gto_oracle` passes the always-on
`eval_gto_s_matches_oracle_on_cpu` gate (device s-shell == oracle within 1e-9 over
randomized fixtures). The `#[cfg(feature="rocm")]` gfx1152 arm exists for real-GPU
confirmation. No libxc pulled into the build.

ORACLE PIN (plan-checker advisory): the inline test oracle MUST use
`y00 = 0.5_f64 / std::f64::consts::PI.sqrt()` and the F-order write
`out[g + (ao_off + c_idx)*ngrids]`, byte-matching eval_gto_sph_cpu lines 603-614 —
otherwise the differential test could pass against a subtly-wrong oracle. Do NOT
re-derive the Y00 normalization; copy it verbatim from the production l=0 path.
  </done>
</task>

</tasks>

<threat_model>
## Trust Boundaries

| Boundary | Description |
|----------|-------------|
| host slices → device kernel | libcint flat arrays (atm/bas/env/ao_loc) + coords uploaded as device buffers indexed by the cube kernel; malformed lengths/offsets index out of range |
| pyscf-algebra macro → pyscf-kernels caller namespace | `#[macro_export]` makes `dispatch_backend!` resolve runtime type tokens in the caller's namespace; a missing cfg-aligned dep would fail to compile per-arm |

## STRIDE Threat Register

| Threat ID | Category | Component | Disposition | Mitigation Plan |
|-----------|----------|-----------|-------------|-----------------|
| T-ljv-01 | Tampering | eval_gto_sph_kernel device index math | mitigate | bounds-guard `if tid < ngrids*nao` like gemm_kernel; F-order index `g + ao_idx*ngrids` matches host exactly; differential oracle (Task 3) catches any index drift |
| T-ljv-02 | Information disclosure | l>=1 fallback split | mitigate | route to device ONLY when every shell is l=0; ANY l>=1 → unchanged eval_gto_sph_cpu, asserted by existing tests staying green (Task 2 verify) |
| T-ljv-03 | Denial of service | empty grid / empty basis | mitigate | route empty/out_len==0 through the existing CPU early-return; do not launch a zero-thread kernel |
| T-ljv-04 | Repudiation | numeric drift host vs device | accept | f-order sequential acc + device exp() differ from std exp by <1 ULP/term; bounded by documented TOL=1e-9 (same as gemm_oracle), NOT claimed bit-identical (ORACLE-07: GPU at documented tolerance, not bit-exact) |
| T-ljv-SC | Tampering | npm/pip/cargo installs | accept | NO new dependencies — uses cubecl-* already in pyscf-kernels/Cargo.toml + the exported pyscf-algebra macro; no install step |
</threat_model>

<verification>
- `cargo build -p pyscf-algebra` + `cargo test -p pyscf-algebra --lib` green (export
  did not break the 16 in-crate call sites).
- `cargo build -p pyscf-kernels` green WITHOUT pulling libxc (low-level crate).
- `eval_gto_sph` contains a `#[cube(launch_unchecked)]` kernel reached via
  `dispatch_backend!`; l>=1 falls back unchanged.
- `cargo test -p pyscf-kernels` — all existing tests pass byte-for-byte.
- `cargo test -p pyscf-kernels --test eval_gto_oracle` — always-on CpuRuntime arm
  matches the oracle within 1e-9; rocm/gfx1152 arm present for real-GPU confirmation.
- All cargo output saved under log/ljv-*.log (CLAUDE.md).
</verification>

<success_criteria>
- `dispatch_backend!` is `#[macro_export]`ed and callable from pyscf-kernels; all 16
  in-crate pyscf-algebra call sites still build.
- The l=0 s-shell eval_gto path runs a real cube kernel on the resolved backend,
  fanned out via the macro, and matches the CPU oracle within 1e-9.
- Every basis with an l>=1 shell falls back to the unchanged CPU path — all existing
  tests pass byte-for-byte (NO silent l>=1 numeric change).
- The differential-oracle test passes on the always-on CpuRuntime arm (default
  features), with the rocm gfx1152 arm wired for real-GPU confirmation.
- No libxc in the build graph; all verify commands scoped `-p pyscf-kernels` /
  `-p pyscf-algebra`.
</success_criteria>

<output>
Create `.planning/quick/260530-ljv-gpu-enable-eval-gto-via-macro-wrapped-mu/260530-ljv-SUMMARY.md` when done.
State explicitly: (1) the chosen i32-array vs F encoding for the device arrays;
(2) whether `#[macro_use]` was retained alongside `#[macro_export]` or replaced with
per-module `use crate::dispatch_backend;`; (3) the observed max_abs_diff on the CPU
oracle arm; (4) whether the rocm arm was actually exercised on gfx1152 or only
compiled. Carry forward the named DEFERRED remainder (l>=1 kernel, deriv1/deriv2,
pyscf-dft numint) for the user's continue/stop decision.
</output>
