---
phase: quick-260529-mtx
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - crates/pyscf-algebra/src/axpy.rs
  - crates/pyscf-algebra/src/lib.rs
  - crates/pyscf-algebra/tests/axpy_oracle.rs
autonomous: true
requirements: [ALG-AXPY]
must_haves:
  truths:
    - "axpy_dense computes y[i] += alpha*x[i] on the active backend (CPU always; ROCm under feature)"
    - "Mismatched x/y lengths return a clean AlgebraError (no panic), not a wrong result"
    - "Empty input is a no-op that returns Ok without launching a grid"
    - "The opaque Tensor-based axpy() remains a documented Phase-2 NotYetImplemented stub"
    - "No cubecl::Runtime type appears in any public signature (ALG-06 wall intact)"
    - "The CPU oracle differential test passes; the ROCm test compiles under --features rocm"
  artifacts:
    - path: "crates/pyscf-algebra/src/axpy.rs"
      provides: "axpy_kernel (#[cube(launch)]), launch_axpy (runtime-generic), axpy_dense (public host-slice), axpy (Tensor stub)"
      contains: "pub fn axpy_dense"
    - path: "crates/pyscf-algebra/tests/axpy_oracle.rs"
      provides: "axpy_kernel_matches_oracle_on_cpu (always) + axpy_kernel_matches_oracle_on_rocm (#[cfg rocm])"
      contains: "fn axpy_kernel_matches_oracle_on_cpu"
  key_links:
    - from: "crates/pyscf-algebra/src/lib.rs"
      to: "crates/pyscf-algebra/src/axpy.rs"
      via: "pub use axpy::{axpy, axpy_dense}"
      pattern: "axpy::\\{axpy, axpy_dense\\}"
    - from: "crates/pyscf-algebra/tests/axpy_oracle.rs"
      to: "pyscf_algebra::axpy_dense"
      via: "use pyscf_algebra::{AlgebraClient, axpy_dense}"
      pattern: "axpy_dense"
---

<objective>
Refactor the Phase-1 `NotYetImplemented` stub in `crates/pyscf-algebra/src/axpy.rs` into a real
cubecl generic-float kernel plus a backend-dispatched host-slice launcher, and add a randomized
oracle differential test. This mirrors the just-completed `scal` workstream (quick-260529-skl)
almost exactly — `axpy` is the two-array element-wise BLAS-1 analog of `scal`.

Purpose: Continue the dot→reduce→scal→axpy element-wise kernel workstream. AXPY (`y[i] += alpha*x[i]`)
is the cleanest next stub: one thread per element, no reduction, two arrays (x read-only, y in place).

Output:
- `axpy_kernel` — `#[cube(launch)]` generic over `F: Float`.
- `launch_axpy::<R: Runtime, F: DeviceScalar>` — runtime-generic launcher (Runtime stays inside the wall).
- `axpy_dense::<F: DeviceScalar>(client, alpha, x, y)` — public host-slice entry point, backend-dispatched.
- `axpy()` — left as a documented Phase-2 `NotYetImplemented` stub (Tensor has no device allocator yet).
- `crates/pyscf-algebra/tests/axpy_oracle.rs` — CPU oracle (always) + ROCm oracle (`#[cfg(feature = "rocm")]`).
</objective>

<execution_context>
@/home/user/Documents/workspace/pyscf_rs/.claude/get-shit-done/workflows/execute-plan.md
@/home/user/Documents/workspace/pyscf_rs/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@CLAUDE.md
@.planning/STATE.md

<!-- CANONICAL reference patterns — mirror structure and doc-comment density EXACTLY. -->
@crates/pyscf-algebra/src/scal.rs
@crates/pyscf-algebra/tests/scal_oracle.rs
@crates/pyscf-algebra/src/scalar.rs
@crates/pyscf-algebra/src/error.rs
@crates/pyscf-algebra/src/lib.rs

<interfaces>
<!-- Key contracts the executor needs. Use directly — no codebase exploration required. -->

DeviceScalar (crates/pyscf-algebra/src/scalar.rs):
  pub trait DeviceScalar: Scalar + cubecl::prelude::Float + bytemuck::Pod {}
  // sealed to f32 / f64.

AlgebraError variants relevant here (crates/pyscf-algebra/src/error.rs):
  DimensionMismatch { op: &'static str, lhs: Vec<usize>, rhs: Vec<usize> }  // USE THIS for x/y length mismatch
  NotYetImplemented { phase: u8, what: &'static str }                        // keep for the Tensor stub

AlgebraClient variants (used in dispatch match):
  AlgebraClient::Cpu(c)                       // always
  AlgebraClient::Cuda(c)  #[cfg(feature="cuda")]
  AlgebraClient::Wgpu(c)  #[cfg(feature="wgpu")]
  AlgebraClient::Rocm(c)  #[cfg(feature="rocm")]

Concrete runtimes for dispatch (match scal.rs exactly):
  cubecl_cpu::CpuRuntime, cubecl_cuda::CudaRuntime, cubecl_wgpu::WgpuRuntime, cubecl_hip::HipRuntime

cubecl 0.10.0 confirmed API (already proven in scal.rs):
  ABSOLUTE_POS : usize
  scalar alpha rides as a single-element Array<F> (NOT a bare generic scalar arg)
  dimension n rides as a bare usize LaunchArg
  ArrayArg::from_raw_parts(handle, len)  // 2-arg, consumes the handle — clone any handle needed for read-back
  CubeDim::new_1d(BLOCK)
  ComputeClient<R>  // one generic
  client.create(Bytes::from_elems(vec))  /  client.read(vec![handle])  /  bytemuck::cast_slice::<u8, F>(..)
</interfaces>
</context>

<tasks>

<task type="auto" tdd="true">
  <name>Task 1: Implement axpy_kernel + launch_axpy + axpy_dense; keep axpy() as Phase-2 stub</name>
  <files>crates/pyscf-algebra/src/axpy.rs, crates/pyscf-algebra/src/lib.rs</files>
  <behavior>
    - axpy_dense::<f64> on a small slice with alpha=2.0 yields y[i] + 2.0*x[i] elementwise (proven by Task 2 oracle).
    - axpy_dense with x.len() != y.len() returns Err(AlgebraError::DimensionMismatch { op: "axpy", .. }) — no panic, y unmodified.
    - axpy_dense with empty x AND empty y returns Ok(()) without launching a grid (no-op).
    - axpy() over Tensor still returns Err(AlgebraError::NotYetImplemented { phase: 2, .. }).
  </behavior>
  <action>
    Rewrite `crates/pyscf-algebra/src/axpy.rs` to mirror `scal.rs` line-for-line in structure,
    doc-comment density, and the cubecl 0.10 idioms, adapted for two arrays:

    Module doc-comment: open with `//! AXPY — generic-float cubecl element-wise y += alpha*x kernel
    + backend-dispatched host launcher.` Reference quick-260529-mtx and name the dot/reduce/scal
    siblings. Describe AXPY as `y[i] += alpha * x[i]` — element-wise BLAS-1, no reduction, one thread
    per element, two arrays (x read-only, y read-write in place). Document the two surfaces exactly as
    scal.rs does: `axpy()` is STILL a stub (Tensor sentinel BufferId, no device allocator until Phase 2),
    `axpy_dense()` is the working device path exercised by tests/axpy_oracle.rs on ROCm.

    Imports: same set as scal.rs — `crate::scalar::DeviceScalar`; `crate::{AlgebraClient, AlgebraError, Tensor}`;
    `cubecl::Runtime`; `cubecl::bytes::Bytes`; `cubecl::client::ComputeClient`; `cubecl::prelude::*`.

    Kernel: `#[cube(launch)] fn axpy_kernel<F: Float>(x: &Array<F>, y: &mut Array<F>, alpha: &Array<F>, n: usize)`.
    Body: `let tid = ABSOLUTE_POS;` then `if tid < n { y[tid] = y[tid] + alpha[0] * x[tid]; }`. x is `&Array`
    (read-only), y is `&mut Array` (in place). Keep the same bounds-guard rationale comment scal.rs uses
    (launch rounds up to whole blocks; tail threads must not write OOB). Keep the alpha-as-single-element-Array
    rationale comment verbatim in spirit (DeviceScalar lacks the CubeElement/ScalarArgSettings bounds a bare
    generic scalar launch arg needs).

    `const BLOCK: u32 = 256;` with the same doc-comment.

    Launcher: `fn launch_axpy<R: Runtime, F: DeviceScalar>(client: &ComputeClient<R>, alpha: F, x: &[F], y: &[F]) -> Vec<F>`.
    Upload x_handle, y_handle, alpha_handle via `client.create(Bytes::from_elems(..to_vec()))` (alpha via
    `vec![alpha]`). `let groups = x.len().div_ceil(BLOCK as usize) as u32;`. Call
    `axpy_kernel::launch::<F, R>(client, CubeCount::Static(groups,1,1), CubeDim::new_1d(BLOCK),
    ArrayArg::from_raw_parts(x_handle, x.len()), ArrayArg::from_raw_parts(y_handle.clone(), y.len()),
    ArrayArg::from_raw_parts(alpha_handle, 1), x.len())` inside `unsafe { }` per arg (match scal.rs's
    per-arg unsafe + SAFETY comment). Clone the **y_handle** for read-back (y is the output buffer that
    must survive): `let bytes = client.read(vec![y_handle]);` then
    `bytemuck::cast_slice::<u8, F>(&bytes[0]).to_vec()`. x_handle is consumed by from_raw_parts and is NOT
    read back. Note in the SAFETY comment that y is the output and its handle is cloned to survive read-back.

    Public dense entry: `pub fn axpy_dense<F: DeviceScalar>(client: &AlgebraClient, alpha: F, x: &[F],
    y: &mut [F]) -> Result<(), AlgebraError>`. Order of checks:
      1. Length mismatch FIRST: `if x.len() != y.len() { return Err(AlgebraError::DimensionMismatch {
         op: "axpy", lhs: vec![x.len()], rhs: vec![y.len()] }); }`.
      2. Empty no-op: `if x.is_empty() { return Ok(()); }` (after the length check, both are empty here).
      3. Dispatch `R` off `client` via the SAME match arms scal.rs uses (Cpu always; Cuda/Wgpu/Rocm under
         their `#[cfg(feature)]`), calling `launch_axpy::<RuntimeType, F>(c, alpha, x, y)`.
      4. `y.copy_from_slice(&result); Ok(())`.
    Doc-comment mirrors scal_dense: generic over F, empty input no-op, Runtime selected here and never in the
    signature (ALG-06 wall). Add one extra sentence documenting the length-mismatch DimensionMismatch behavior.

    Tensor stub: keep `pub fn axpy(_client: &AlgebraClient, _alpha: f64, _x: &Tensor, _y: &mut Tensor)
    -> Result<(), AlgebraError>` returning `Err(AlgebraError::NotYetImplemented { phase: 2, what:
    "axpy over Tensor (device allocator) — use axpy_dense for the host-slice device path" })`. Mirror
    scal.rs's stub doc-comment (Tensor sentinel BufferId; working path is axpy_dense).

    In `crates/pyscf-algebra/src/lib.rs`, change `pub use axpy::axpy;` to `pub use axpy::{axpy, axpy_dense};`
    (mirror the `pub use scal::{scal, scal_dense};` line).
  </action>
  <verify>
    <automated>cargo build -p pyscf-algebra 2>&1 | tee log/quick-260529-mtx-build.log; tail -5 log/quick-260529-mtx-build.log; grep -q "pub fn axpy_dense" crates/pyscf-algebra/src/axpy.rs && grep -q "axpy::{axpy, axpy_dense}" crates/pyscf-algebra/src/lib.rs && echo WIRED</automated>
  </verify>
  <done>pyscf-algebra builds; axpy.rs exports axpy_dense + retains the Tensor axpy() Phase-2 stub; lib.rs re-exports both; no cubecl::Runtime in any public signature.</done>
</task>

<task type="auto" tdd="true">
  <name>Task 2: Add axpy_oracle.rs randomized differential test (CPU always + ROCm under cfg)</name>
  <files>crates/pyscf-algebra/tests/axpy_oracle.rs</files>
  <behavior>
    - axpy_kernel_matches_oracle_on_cpu: for a spread of lengths and alphas, device y == host (y0[i] + alpha*x0[i]) within 1e-12.
    - Empty-input no-op: axpy_dense on two empty slices returns Ok and leaves them empty.
    - axpy_kernel_matches_oracle_on_rocm (#[cfg(feature="rocm")]): same differential check on AmdDevice::default() (gfx1152).
  </behavior>
  <action>
    Create `crates/pyscf-algebra/tests/axpy_oracle.rs` by adapting `tests/scal_oracle.rs` for the
    two-array AXPY. Keep the module doc-comment style, the `Lcg` (Knuth/MMIX constants → [-1,1)),
    `random_vector`, the direct-client construction rationale (no select_backend → no PYSCF_BACKEND race),
    and the always-on-CPU / cfg-ROCm test split EXACTLY.

    Module doc: reference quick-260529-mtx; state the oracle is exact element-wise (AXPY has no reduction,
    no pairwise tree needed): host ground truth is `y0[i] + alpha*x0[i]`. List verified-in-scope: device
    `y += alpha*x` == host within 1e-12 over a spread of random lengths (degenerate, prime, BLOCK-boundary
    straddling) and alphas (incl. 0, negative, identity); plus the empty-input no-op. Not verified: f32 path.

    Imports: `use cubecl::Runtime;` and `use pyscf_algebra::{AlgebraClient, axpy_dense};`.

    `check_case(client, len, alpha, seed) -> f64`: seed one Lcg, draw `x = random_vector(rng, len)` then
    `y0 = random_vector(rng, len)` (two distinct draws from the SAME rng so x and y differ). reference =
    `(0..len).map(|i| y0[i] + xi*alpha)` — i.e. `y0[i] + alpha*x[i]`. Clone y0 into `device`, call
    `axpy_dense::<f64>(client, alpha, &x, &mut device).expect("axpy_dense should always succeed")`,
    assert `device.len() == reference.len()` ("axpy must preserve length"), return max abs elementwise diff.

    Reuse the same constants: `TOL = 1e-12`, `LENS = [1, 2, 13, 31, 64, 97, 128, 255, 256, 257, 512, 1000]`,
    `ALPHAS = [1.0, 0.0, -1.0, 2.5, -3.75, 0.001, 1234.5]`.

    `run_all(client, base_seed, label)`: iterate every (len, alpha) pair through check_case with
    `base_seed.wrapping_add(i)`, assert `diff < TOL` with a `{label} axpy len {len} alpha {alpha}: max abs
    diff {diff:e} >= tol {TOL:e}` message. Then the empty no-op: two empty Vec<f64> (x and y), call
    `axpy_dense::<f64>(client, 3.0, &x_empty, &mut y_empty).expect("axpy_dense on empty slices is a no-op")`,
    assert y_empty.is_empty().

    Optional but matching the workstream: add a `#[test]` asserting the length-mismatch path returns
    DimensionMismatch (call axpy_dense with x len 4, y len 5, assert it is Err). Keep it CPU-only.

    `#[test] fn axpy_kernel_matches_oracle_on_cpu()`: build `AlgebraClient::Cpu(cubecl_cpu::CpuRuntime::client(
    &cubecl_cpu::CpuDevice))`, `run_all(&client, 0x_<fresh>_u64, "CPU")` — use a fresh distinct seed literal
    (NOT scal's 0x51EDC0DE).

    `#[cfg(feature = "rocm")] #[test] fn axpy_kernel_matches_oracle_on_rocm()`: build `AlgebraClient::Rocm(
    cubecl_hip::HipRuntime::client(&cubecl_hip::AmdDevice::default()))`, assert `matches!(client,
    AlgebraClient::Rocm(_))`, `run_all(&client, 0x_<fresh>_u64, "ROCm")` — fresh seed distinct from scal's.
  </action>
  <verify>
    <automated>cargo test -p pyscf-algebra --test axpy_oracle 2>&1 | tee log/quick-260529-mtx-test.log; tail -15 log/quick-260529-mtx-test.log; grep -q "test result: ok" log/quick-260529-mtx-test.log && echo OK</automated>
  </verify>
  <done>axpy_oracle.rs CPU differential test passes (device y == host y0+alpha*x within 1e-12 across all len/alpha pairs); empty no-op asserted; length-mismatch returns DimensionMismatch; ROCm test compiles under --features rocm (run on gfx1152 if hardware present).</done>
</task>

</tasks>

<threat_model>
## Trust Boundaries

| Boundary | Description |
|----------|-------------|
| host slice → device buffer | x/y host data uploaded via Bytes::from_elems; bounded by explicit length args |
| package installs | none — no new dependencies; cubecl/cubecl_cpu/cubecl_hip already in the crate |

## STRIDE Threat Register

| Threat ID | Category | Component | Disposition | Mitigation Plan |
|-----------|----------|-----------|-------------|-----------------|
| T-mtx-01 | Tampering | axpy_kernel out-of-bounds write | mitigate | `if tid < n` bounds guard (launch rounds up to whole blocks); oracle covers BLOCK-boundary lengths |
| T-mtx-02 | Denial of Service | zero-length / mismatched-length launch | mitigate | length-mismatch → DimensionMismatch before launch; empty → early Ok no-op (never launch zero grid) |
| T-mtx-03 | Information Disclosure | unsafe from_raw_parts handle/len mismatch | accept | lengths are the exact buffers just allocated; SAFETY comment documents the invariant (matches scal.rs) |
| T-mtx-SC | Tampering | npm/pip/cargo installs | accept | no new packages added; nothing to audit |
</threat_model>

<verification>
- `cargo build -p pyscf-algebra` succeeds (scoped to the crate — does NOT pull libxc_rs).
- `cargo test -p pyscf-algebra --test axpy_oracle` passes the CPU oracle (always-on gate).
- ROCm oracle: `cargo test -p pyscf-algebra --features rocm --test axpy_oracle` (~24s build, no libxc) on gfx1152 if hardware present; otherwise it compiles under the feature.
- `grep` confirms axpy_dense exported and lib.rs re-export updated.
- No cubecl::Runtime in any public signature (axpy_dense / axpy take only AlgebraClient + slices/Tensor).
- All cargo output saved under log/ per project convention before any build-issue investigation.
</verification>

<success_criteria>
- axpy.rs implements the generic-float kernel + launch_axpy + axpy_dense; axpy() over Tensor remains a Phase-2 NotYetImplemented stub.
- axpy_dense returns DimensionMismatch on length mismatch, is a no-op on empty input, and otherwise computes y[i] += alpha*x[i] on the active backend.
- axpy_oracle.rs CPU differential test passes within 1e-12; ROCm test gated behind #[cfg(feature = "rocm")].
- lib.rs re-exports `axpy::{axpy, axpy_dense}`.
- Structure and doc-comment density mirror the scal sibling.
</success_criteria>

<output>
Create `.planning/quick/260529-mtx-refactor-crates-pyscf-algebra-to-cubecl-/260529-mtx-01-SUMMARY.md` when done.
</output>
