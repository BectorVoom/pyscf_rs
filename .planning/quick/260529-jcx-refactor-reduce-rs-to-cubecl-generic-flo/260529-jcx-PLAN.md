---
phase: quick-260529-jcx
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - crates/pyscf-algebra/src/reduce.rs
  - crates/pyscf-algebra/src/lib.rs
  - crates/pyscf-algebra/tests/reduce_oracle.rs
autonomous: true
requirements: []
must_haves:
  truths:
    - "reduce_sum_dense::<f64>(client, x) returns the full sum of x within 1e-9 of oracle_sum"
    - "The reduction kernel runs on the CPU backend (always) and on the ROCm gfx1152 backend (under --features rocm)"
    - "The Tensor-surface reduce_sum stays a Phase-2 NotYetImplemented stub"
    - "cargo build and cargo clippy for pyscf-algebra are clean"
  artifacts:
    - path: "crates/pyscf-algebra/src/reduce.rs"
      provides: "reduce_kernel #[cube(launch)] generic over F: Float, launch_reduce_sum::<R,F>, reduce_sum_dense::<F>, reduce_sum Phase-2 stub"
      contains: "#[cube(launch)]"
    - path: "crates/pyscf-algebra/tests/reduce_oracle.rs"
      provides: "Randomized oracle differential test (CPU always + ROCm under cfg)"
      contains: "reduce_kernel_matches_oracle_on_cpu"
    - path: "crates/pyscf-algebra/src/lib.rs"
      provides: "pub use reduce::{reduce_sum, reduce_sum_dense}"
      contains: "reduce_sum_dense"
  key_links:
    - from: "crates/pyscf-algebra/src/reduce.rs"
      to: "crate::oracle::oracle_sum"
      via: "differential test ground truth"
      pattern: "reduce_sum_dense"
    - from: "crates/pyscf-algebra/tests/reduce_oracle.rs"
      to: "pyscf_algebra::reduce_sum_dense"
      via: "device-vs-oracle comparison"
      pattern: "reduce_sum_dense::<f64>"
---

<objective>
Refactor `crates/pyscf-algebra/src/reduce.rs` from the Phase-1 `NotYetImplemented`
stub into a real cubecl `#[cube(launch)]` reduction kernel generic over the device
float `F: Float`, plus a host-slice launcher dispatched off `AlgebraClient`, mirroring
the just-completed `dot.rs` sibling (quick-260529-iji) exactly. Add a randomized oracle
differential test that validates against `oracle_sum` on CPU (always) and on the ROCm
gfx1152 backend (under `--features rocm`).

Purpose: Lights up the second cubecl generic-float reduction on real AMD hardware,
keeping the cubecl `Runtime` generic inside the ALG-06 wall.
Output: refactored `reduce.rs`, `lib.rs` re-export, `tests/reduce_oracle.rs`.
</objective>

<execution_context>
@/home/user/Documents/workspace/pyscf_rs/.claude/get-shit-done/workflows/execute-plan.md
@/home/user/Documents/workspace/pyscf_rs/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@/home/user/Documents/workspace/pyscf_rs/CLAUDE.md
@/home/user/Documents/workspace/pyscf_rs/crates/pyscf-algebra/src/dot.rs
@/home/user/Documents/workspace/pyscf_rs/crates/pyscf-algebra/src/reduce.rs
@/home/user/Documents/workspace/pyscf_rs/crates/pyscf-algebra/src/oracle.rs
@/home/user/Documents/workspace/pyscf_rs/crates/pyscf-algebra/src/scalar.rs
@/home/user/Documents/workspace/pyscf_rs/crates/pyscf-algebra/src/lib.rs
@/home/user/Documents/workspace/pyscf_rs/crates/pyscf-algebra/tests/dot_oracle.rs

<interfaces>
<!-- Contracts the executor must implement against. dot.rs is the exact analog. -->

From src/dot.rs (THE TEMPLATE — mirror its module shape precisely):
```rust
#[cube(launch)]
fn dot_kernel<F: Float>(x: &Array<F>, y: &Array<F>, out: &mut Array<F>, n: usize) { /* ... */ }
const BLOCK: u32 = 256;
fn launch_dot<R: Runtime, F: DeviceScalar>(client: &ComputeClient<R>, x: &[F], y: &[F]) -> F { /* ... */ }
pub fn dot_dense<F: DeviceScalar>(client: &AlgebraClient, x: &[F], y: &[F]) -> Result<F, AlgebraError>;
pub fn dot(_client: &AlgebraClient, _x: &Tensor, _y: &Tensor) -> Result<f64, AlgebraError>; // Phase-2 stub
```

From src/scalar.rs:
```rust
pub trait DeviceScalar: Scalar + cubecl::prelude::Float + bytemuck::Pod { /* sealed */ }
```

From src/oracle.rs (differential ground truth):
```rust
pub fn oracle_sum(xs: &[f64]) -> f64; // pairwise tree, chunk=128, bit-deterministic
```

From src/error.rs (variants used):
```rust
AlgebraError::NotYetImplemented { phase: u32, what: &'static str }
```

Current stub signature to preserve as the Tensor surface:
```rust
pub fn reduce_sum(client, x: &Tensor, axis: usize, out: &mut Tensor) -> Result<(), AlgebraError>
```

cubecl 0.10.0 API gotchas (verified-working facts dot_kernel already uses):
- `ABSOLUTE_POS` is `usize`; `Array<F>` indices `usize`; dim scalars `usize` — no casts.
- Scalar kernel args passed BARE (LaunchArg for T = T): `n` not `ScalarArg::new(n)`.
- `ArrayArg::from_raw_parts(handle, len)` is 2-arg, `unsafe`, consumes handle by value (clone the output handle to read it back).
- `CubeDim::new_1d(BLOCK)`, `CubeCount::Static(groups, 1, 1)`.
- Launch: `kernel::launch::<F, R>(client, count, dim, args...)`.
- Backend arms: `AlgebraClient::Cpu(c) => launch_*::<cubecl_cpu::CpuRuntime, F>(c, ...)`,
  with `#[cfg(feature = "cuda"|"wgpu"|"rocm")]` arms using
  `cubecl_cuda::CudaRuntime` / `cubecl_wgpu::WgpuRuntime` / `cubecl_hip::HipRuntime`.
</interfaces>
</context>

<tasks>

<task type="auto" tdd="true">
  <name>Task 1: Refactor reduce.rs into a generic-float partial-sum kernel + dense launcher, re-export from lib.rs</name>
  <files>crates/pyscf-algebra/src/reduce.rs, crates/pyscf-algebra/src/lib.rs</files>
  <behavior>
    - reduce_sum_dense::<f64>(client, &[]) returns 0.0 (empty input → identity sum).
    - reduce_sum_dense::<f64>(client, &x) over a small known vector returns its sum within fp tolerance.
    - reduce_sum(client, x, axis, out) returns Err(NotYetImplemented { phase: 2, .. }) unchanged.
  </behavior>
  <action>
    Refactor src/reduce.rs to mirror src/dot.rs's module shape exactly. Replace the
    stub body with FOUR items, keeping the existing `reduce_sum` Tensor stub as the
    last one:

    1. A `#[cube(launch)] fn reduce_kernel<F: Float>(...)` generic over the device
       float. Prefer the PARTIAL-SUM shape: one thread sums a contiguous CHUNK of the
       input into one partial, bounds-guarded against the tail, producing `groups`
       partials. Thread `tid = ABSOLUTE_POS` sums input indices `[tid*chunk, (tid+1)*chunk)`
       clamped to `n`, writing the partial to `out[tid]`. Use a `while`/`for` loop in
       the thread to accumulate sequentially in `F`. Pass `n` and `chunk` as bare `usize`
       scalar args. If a dynamic-bound loop inside `#[cube]` proves problematic on
       0.10.0, FALL BACK to dot's exact shape (device does an identity pass `out[i]=x[i]`,
       host sums all elements) — but try the partial-sum kernel first.
       GROUND-TRUTH the kernel against the CPU oracle (Task 2) before assuming correct.
    2. `const BLOCK: u32 = 256;` (one thread per partial; grid sized to cover the
       partials). Choose CHUNK so the number of partials = `n.div_ceil(chunk)` and the
       launch covers them — keep it naive and obviously-correct.
    3. `fn launch_reduce_sum<R: Runtime, F: DeviceScalar>(client: &ComputeClient<R>, x: &[F]) -> F`:
       upload `x`, allocate the partials output buffer, launch `reduce_kernel`, read the
       partials back, and sum them on the host in `F` (plain `acc += p` — NOT an FMA).
       Handle the empty-input case (return `F::from_int(0)` without launching) so a
       zero-length grid is never launched.
    4. `pub fn reduce_sum_dense<F: DeviceScalar>(client: &AlgebraClient, x: &[F]) -> Result<F, AlgebraError>`:
       full axis-free sum matching `oracle_sum`'s signature. Dispatch `R` off
       `AlgebraClient` with the exact Cpu/Cuda/Wgpu/Rocm cfg arms from dot_dense. No
       shape validation needed (single input). Return `Ok(out)`.

    KEEP `pub fn reduce_sum(client, x: &Tensor, axis, out) -> Result<(), AlgebraError>`
    as a Phase-2 `NotYetImplemented { phase: 2, what: "..." }` stub — document the same
    rationale dot.rs uses (Tensor carries a sentinel BufferId; no device allocator until
    Phase 2). Adjust the param underscores so unused params don't warn.

    Update the module doc-comment to describe the new generic-float reduction (mirror
    dot.rs's header style). Then edit src/lib.rs: change `pub use reduce::reduce_sum;`
    to `pub use reduce::{reduce_sum, reduce_sum_dense};` (mirror the dot re-export).

    Per CLAUDE.md: read docs/manual/Cubecl before/while writing the kernel, and save
    full cargo output to log/ before investigating any build issue.
  </action>
  <verify>
    <automated>cargo build -p pyscf-algebra 2>&1 | tee log/reduce-jcx-build.log; cargo clippy -p pyscf-algebra 2>&1 | tee log/reduce-jcx-clippy.log</automated>
  </verify>
  <done>cargo build and cargo clippy for pyscf-algebra are clean (no errors, no new warnings); reduce.rs exports reduce_kernel, launch_reduce_sum, reduce_sum_dense, and the reduce_sum Phase-2 stub; lib.rs re-exports reduce_sum_dense.</done>
</task>

<task type="auto">
  <name>Task 2: Add tests/reduce_oracle.rs and run the CPU oracle differential test</name>
  <files>crates/pyscf-algebra/tests/reduce_oracle.rs</files>
  <action>
    Create tests/reduce_oracle.rs mirroring tests/dot_oracle.rs exactly, adapted for the
    single-input full sum:
    - Same `Lcg` (Knuth/MMIX constants, maps high bits to [-1.0, 1.0)) — NO `rand` crate.
    - `random_vector(rng, len)` helper.
    - `check_case(client, len, seed)`: build ONE random vector, call
      `reduce_sum_dense::<f64>(client, &x)`, compare against `oracle_sum(&x)`, return abs diff.
      Assert the oracle reference `is_finite()`.
    - `const TOL: f64 = 1e-9;`
    - `const LENS: &[usize] = &[1, 2, 13, 31, 64, 97, 128, 255, 256, 257, 512, 1000];`
      (degenerate, primes, straddling the BLOCK=256 boundary to exercise the tail guard).
      Optionally add `0` if reduce_sum_dense handles empty → 0.0.
    - `#[test] fn reduce_kernel_matches_oracle_on_cpu()` — builds
      `AlgebraClient::Cpu(cubecl_cpu::CpuRuntime::client(&cubecl_cpu::CpuDevice))`,
      loops LENS, asserts diff < TOL.
    - `#[cfg(feature = "rocm")] #[test] fn reduce_kernel_matches_oracle_on_rocm()` —
      builds `AlgebraClient::Rocm(cubecl_hip::HipRuntime::client(&cubecl_hip::AmdDevice::default()))`,
      asserts `matches!(client, AlgebraClient::Rocm(_))`, loops LENS, asserts diff < TOL.
    Import `use cubecl::Runtime;` and `use pyscf_algebra::{AlgebraClient, reduce_sum_dense, oracle_sum};`.
    Follow docs/rust_crate_test_guideline.md (differential testing, randomized inputs,
    reproducible seed) — see the dot_oracle.rs header for the pattern.

    Save cargo output to log/ before investigating any failure.
  </action>
  <verify>
    <automated>cargo test -p pyscf-algebra --test reduce_oracle reduce_kernel_matches_oracle_on_cpu 2>&1 | tee log/reduce-jcx-cpu-test.log</automated>
  </verify>
  <done>reduce_kernel_matches_oracle_on_cpu passes (device sum == oracle_sum within 1e-9 across all LENS).</done>
</task>

<task type="auto">
  <name>Task 3: Run the ROCm oracle differential test on gfx1152</name>
  <files>crates/pyscf-algebra/tests/reduce_oracle.rs</files>
  <action>
    Run the ROCm-gated differential test on real AMD gfx1152 hardware. The rocm build is
    ~24s and does NOT pull libxc (safe per MEMORY). If the test fails, the kernel is wrong
    on-device — fix reduce.rs (Task 1) and re-run Tasks 2 and 3. Save cargo output to log/
    before investigating any failure.
  </action>
  <verify>
    <automated>cargo test -p pyscf-algebra --features rocm --test reduce_oracle 2>&1 | tee log/reduce-jcx-rocm-test.log</automated>
  </verify>
  <done>reduce_kernel_matches_oracle_on_rocm passes on gfx1152 (device sum == oracle_sum within 1e-9 across all LENS); both CPU and ROCm tests green.</done>
</task>

</tasks>

<verification>
- `cargo build -p pyscf-algebra` clean.
- `cargo clippy -p pyscf-algebra` clean.
- `cargo test -p pyscf-algebra --test reduce_oracle reduce_kernel_matches_oracle_on_cpu` passes.
- `cargo test -p pyscf-algebra --features rocm --test reduce_oracle` passes on gfx1152.
- `reduce_sum` Tensor surface still returns `NotYetImplemented { phase: 2 }`.
</verification>

<success_criteria>
- reduce.rs is a real cubecl `#[cube(launch)]` generic-float reduction kernel + dense launcher dispatched off AlgebraClient.
- reduce_sum_dense::<F> matches oracle_sum within 1e-9 on both CPU and ROCm gfx1152.
- The Tensor-surface reduce_sum remains a Phase-2 stub.
- lib.rs re-exports reduce_sum_dense.
- No new clippy warnings; all cargo output captured under log/.
</success_criteria>

<output>
Create `.planning/quick/260529-jcx-refactor-reduce-rs-to-cubecl-generic-flo/260529-jcx-SUMMARY.md` when done.
</output>
