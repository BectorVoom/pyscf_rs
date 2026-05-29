---
phase: quick-260529-iji
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - crates/pyscf-algebra/src/dot.rs
  - crates/pyscf-algebra/src/lib.rs
  - crates/pyscf-algebra/tests/dot_oracle.rs
autonomous: true
requirements: [ALG-DOT-KERNEL, ALG-06]
must_haves:
  truths:
    - "dot_dense(client, x, y) returns sum(x[i]*y[i]) matching oracle_dot within 1e-9 on the CPU backend"
    - "dot_dense runs the SAME generic cubecl kernel on ROCm (gfx1152) and matches oracle_dot within 1e-9"
    - "dot_dense rejects mismatched-length vectors with AlgebraError::ShapeMismatch"
    - "dot() over the opaque Tensor surface still returns NotYetImplemented (Phase 2 contract preserved)"
    - "The cubecl Runtime generic never appears in any public signature (ALG-06 wall)"
  artifacts:
    - path: "crates/pyscf-algebra/src/dot.rs"
      provides: "dot_kernel (#[cube(launch)]), launch_dot<R,F>, dot_dense<F>, dot() stub"
      contains: "#[cube(launch)]"
      min_lines: 80
    - path: "crates/pyscf-algebra/tests/dot_oracle.rs"
      provides: "randomized oracle differential test, CPU always + ROCm under cfg"
      contains: "dot_kernel_matches_oracle_on_cpu"
    - path: "crates/pyscf-algebra/src/lib.rs"
      provides: "re-export of dot_dense alongside dot"
      contains: "dot_dense"
  key_links:
    - from: "crates/pyscf-algebra/src/dot.rs"
      to: "AlgebraClient variants (Cpu/Cuda/Wgpu/Rocm)"
      via: "match in dot_dense dispatching launch_dot::<Runtime, F>"
      pattern: "AlgebraClient::Cpu"
    - from: "crates/pyscf-algebra/tests/dot_oracle.rs"
      to: "pyscf_algebra::oracle_dot"
      via: "differential ground-truth comparison"
      pattern: "oracle_dot"
---

<objective>
Refactor `pyscf-algebra/src/dot.rs` from a `NotYetImplemented` stub into a real
generic-float cubecl `#[cube(launch)]` reduction kernel plus a backend-dispatched
host-slice launcher `dot_dense`, mirroring the JUST-LANDED `gemm.rs` precedent
(commits b720570 + 86d8187) exactly. Add a randomized oracle differential test
that runs on both CPU and ROCm (gfx1152).

Purpose: `dot(x, y) = sum(x[i] * y[i])` is the reduction sibling of GEMM. The
trusted bit-deterministic oracle `oracle_dot` already exists and is exported, so
this is a pure-additive refactor with a ready-made differential ground truth.
Output:
- `dot_kernel` (#[cube(launch)] one-thread-per-element products kernel)
- `launch_dot<R: Runtime, F: DeviceScalar>` host launcher (Runtime generic stays inside the wall)
- `dot_dense<F: DeviceScalar>(client, x, y) -> Result<F, AlgebraError>` public device path
- `dot()` Tensor stub preserved unchanged (Phase 2 contract)
- `tests/dot_oracle.rs` differential test (CPU + ROCm)
</objective>

<execution_context>
@/home/user/Documents/workspace/pyscf_rs/.claude/get-shit-done/workflows/execute-plan.md
@/home/user/Documents/workspace/pyscf_rs/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@.planning/STATE.md
@./CLAUDE.md

<!-- MANDATORY per CLAUDE.md: consult the cubecl manual before writing any cubecl code. -->
@docs/manual/Cubecl/Cubecl_generics.md

<!-- THE PRECEDENT — copy this structure exactly. gemm.rs is the ground truth for
     correct cubecl 0.10 API usage (docs/manual examples are STALE for 0.10.0). -->
@crates/pyscf-algebra/src/gemm.rs

<!-- The stub being refactored (preserve the dot() Tensor surface as a stub). -->
@crates/pyscf-algebra/src/dot.rs

<!-- THE TEST PRECEDENT — copy structure; swap oracle_einsum -> oracle_dot,
     2D shapes -> 1D vector lengths. -->
@crates/pyscf-algebra/tests/gemm_oracle.rs

<!-- DeviceScalar trait + bytemuck::Pod bound (already present). -->
@crates/pyscf-algebra/src/scalar.rs

<!-- The oracle: oracle_dot(a, b) -> f64, bit-deterministic, returns NaN on
     length mismatch. Already exported from lib.rs. -->
@crates/pyscf-algebra/src/oracle.rs

<interfaces>
<!-- Key contracts the executor needs — extracted from codebase, no exploration required. -->

From src/scalar.rs:
    pub trait DeviceScalar: Scalar + cubecl::prelude::Float + bytemuck::Pod {}
    impl DeviceScalar for f32 {}
    impl DeviceScalar for f64 {}

From src/error.rs (use this variant for the length-mismatch guard):
    AlgebraError::ShapeMismatch { expected: String, actual: String }
    AlgebraError::NotYetImplemented { phase: u8, what: &'static str }

From src/oracle.rs (the differential ground truth — already exported):
    pub fn oracle_dot(a: &[f64], b: &[f64]) -> f64   // bit-deterministic; NaN on len mismatch

From src/client.rs (variant set to match in dispatch — same shape as gemm_dense):
    enum AlgebraClient { Cpu(ComputeClient<cubecl_cpu::CpuRuntime>),
                         #[cfg(feature="cuda")] Cuda(..),
                         #[cfg(feature="wgpu")] Wgpu(..),
                         #[cfg(feature="rocm")] Rocm(ComputeClient<cubecl_hip::HipRuntime>) }

cubecl 0.10 API gotchas (gemm.rs is the canonical reference — NOT the stale manual examples):
    - ABSOLUTE_POS and Array indices are usize; scalar kernel dim args are usize too.
    - Scalar kernel args passed as BARE values (e.g. `n: usize`), not wrapped.
    - ArrayArg::from_raw_parts(handle, len) is the 2-arg form; consumes handle by value
      (clone the output handle so it survives the read-back).
    - CubeDim::new_1d(BLOCK), CubeCount::Static(groups, 1, 1).
    - launch::<F, R>(client, count, dim, args...) inside an `unsafe`-wrapped ArrayArg.
</interfaces>
</context>

<tasks>

<task type="auto" tdd="false">
  <name>Task 1: cubecl dot reduction kernel + dot_dense launcher + lib export</name>
  <files>crates/pyscf-algebra/src/dot.rs, crates/pyscf-algebra/src/lib.rs</files>
  <behavior>
    Verified by the oracle test in Task 2 (RED→GREEN handled there). This task
    produces the production code the test exercises.
    - dot_dense::&lt;f64&gt;(cpu, &amp;x, &amp;y) for equal-length vectors returns a value within 1e-9 of oracle_dot(&amp;x, &amp;y).
    - dot_dense rejects x.len() != y.len() with AlgebraError::ShapeMismatch.
    - dot() over Tensor still returns NotYetImplemented { phase: 2, .. }.
  </behavior>
  <action>
    Rewrite src/dot.rs mirroring src/gemm.rs's three-surface structure, adapted for a
    REDUCTION returning a scalar:

    (1) Module doc comment mirroring gemm.rs's: note this is the quick-260529-iji
    refactor of the Phase-1 NotYetImplemented stub into a generic-float cubecl kernel,
    that `dot(x,y) = sum(x[i]*y[i])` is the reduction sibling of GEMM, and that the
    Runtime generic stays inside the ALG-06 wall via dispatch off AlgebraClient.

    (2) Imports identical to gemm.rs: `use crate::scalar::DeviceScalar;`,
    `use crate::{AlgebraClient, AlgebraError, Tensor};`, `use cubecl::Runtime;`,
    `use cubecl::bytes::Bytes;`, `use cubecl::client::ComputeClient;`,
    `use cubecl::prelude::*;`.

    (3) `#[cube(launch)] fn dot_kernel&lt;F: Float&gt;(x: &amp;Array&lt;F&gt;, y: &amp;Array&lt;F&gt;, out: &amp;mut Array&lt;F&gt;, n: usize)`.
    DESIGN: one-thread-per-element kernel that writes the product into out — `let tid = ABSOLUTE_POS;`
    then `if tid < n { out[tid] = x[tid] * y[tid]; }`. This matches gemm.rs's "naive,
    obviously-correct" philosophy and avoids device atomics / tree-reduction. The FINAL
    sum happens on the host in f64 (see launcher). Do NOT pull in cubecl-reduce. `n` is
    a bare `usize` scalar arg (cubecl 0.10: ABSOLUTE_POS and indices are usize).

    (4) `const BLOCK: u32 = 256;` (same as gemm).

    (5) `fn launch_dot&lt;R: Runtime, F: DeviceScalar&gt;(client: &amp;ComputeClient&lt;R&gt;, x: &amp;[F], y: &amp;[F]) -> F`:
    upload x and y via `client.create(Bytes::from_elems(_.to_vec()))`; allocate
    `out_handle = client.empty(x.len() * core::mem::size_of::&lt;F&gt;())`;
    `groups = x.len().div_ceil(BLOCK as usize) as u32`; call
    `dot_kernel::launch::&lt;F, R&gt;(client, CubeCount::Static(groups,1,1), CubeDim::new_1d(BLOCK),
    unsafe { ArrayArg::from_raw_parts(x_handle, x.len()) },
    unsafe { ArrayArg::from_raw_parts(y_handle, y.len()) },
    unsafe { ArrayArg::from_raw_parts(out_handle.clone(), x.len()) }, x.len())`.
    Read back: `let bytes = client.read(vec![out_handle]); let products: Vec&lt;F&gt; = bytemuck::cast_slice::&lt;u8, F&gt;(&amp;bytes[0]).to_vec();`.
    Then HOST-SUM in F via a plain accumulator: `let mut acc = F::from_int(0); for &amp;p in &amp;products { acc += p; }  acc`.
    NOTE on the FMA guard: a plain `acc += p` host sum is FINE — check-no-fma only flags
    FMA in pyscf_* symbols and this is a simple add, not a fused multiply-add. Do NOT use
    `.mul_add` here. Keep the products kernel as the ONLY multiply.
    The Runtime generic R is confined to this fn — it must NOT escape into the public surface (ALG-06).

    (6) `pub fn dot_dense&lt;F: DeviceScalar&gt;(client: &amp;AlgebraClient, x: &amp;[F], y: &amp;[F]) -> Result&lt;F, AlgebraError&gt;`:
    length guard first — `if x.len() != y.len() { return Err(AlgebraError::ShapeMismatch {
    expected: format!("y len {}", x.len()), actual: y.len().to_string() }); }`.
    Then `let out = match client { AlgebraClient::Cpu(c) => launch_dot::&lt;cubecl_cpu::CpuRuntime, F&gt;(c, x, y), ... }`
    with the SAME #[cfg(feature=...)] arms as gemm_dense for cuda/wgpu/rocm
    (cuda -> cubecl_cuda::CudaRuntime, wgpu -> cubecl_wgpu::WgpuRuntime, rocm -> cubecl_hip::HipRuntime).
    `Ok(out)`.

    (7) Preserve `pub fn dot(_client: &amp;AlgebraClient, _x: &amp;Tensor, _y: &amp;Tensor) -> Result&lt;f64, AlgebraError&gt;`
    returning `Err(AlgebraError::NotYetImplemented { phase: 2, what: "dot over Tensor (device allocator) — use dot_dense for the host-slice device path" })`.
    Keep the SAME signature the existing stub has (client, x, y) so nothing downstream breaks;
    backend_matrix.rs does NOT pin dot(), but preserve the stub contract anyway.

    In src/lib.rs change `pub use dot::dot;` to `pub use dot::{dot, dot_dense};` (mirror the
    `pub use gemm::{gemm, gemm_dense};` line directly above it).
  </action>
  <verify>
    <automated>cargo build -p pyscf-algebra 2>&amp;1 | tee log/dot-iji-t1-build.log; test ${PIPESTATUS[0]} -eq 0</automated>
    <automated>grep -q "pub fn dot_dense" crates/pyscf-algebra/src/dot.rs &amp;&amp; grep -q "#\[cube(launch)\]" crates/pyscf-algebra/src/dot.rs &amp;&amp; grep -q "dot_dense" crates/pyscf-algebra/src/lib.rs</automated>
    <automated>grep -q "NotYetImplemented" crates/pyscf-algebra/src/dot.rs</automated>
    <automated>cargo clippy -p pyscf-algebra 2>&amp;1 | tee log/dot-iji-t1-clippy.log | grep -v '^#' | grep -c "warning:" | grep -qx 0 || (echo "clippy clean expected" &amp;&amp; false)</automated>
  </verify>
  <done>
    pyscf-algebra builds (CPU default feature) with dot_kernel (#[cube(launch)]),
    launch_dot, dot_dense, and the preserved dot() stub. dot_dense and dot both
    re-exported from lib.rs. The Runtime generic appears only inside launch_dot.
    clippy clean (no unwrap_used per FOUND-07, no FMA in pyscf_* symbols).
    Full cargo output saved under log/ per CLAUDE.md. Build stays scoped to
    -p pyscf-algebra (never pulls libxc_rs).
  </done>
</task>

<task type="auto" tdd="false">
  <name>Task 2: dot_oracle.rs differential test — CPU always + ROCm under cfg</name>
  <files>crates/pyscf-algebra/tests/dot_oracle.rs</files>
  <behavior>
    - dot_kernel_matches_oracle_on_cpu: for each length in LENS, dot_dense::&lt;f64&gt;(cpu, x, y) is within 1e-9 of oracle_dot(x, y).
    - dot_kernel_matches_oracle_on_rocm (#[cfg(feature="rocm")]): SAME check on the HipRuntime client; asserts the client is the Rocm variant (no silent fallback).
    - Lengths cover degenerate 1, a prime (e.g. 13, 31, 97), and powers/round numbers crossing the BLOCK=256 boundary (e.g. 256, 257, 1000).
  </behavior>
  <action>
    Create tests/dot_oracle.rs by copying the STRUCTURE of tests/gemm_oracle.rs, adapted
    from 2D GEMM shapes to 1D vector lengths and from oracle_einsum to oracle_dot:

    (1) Module doc comment mirroring gemm_oracle.rs's: quick-260529-iji randomized oracle
    differential test for the cubecl generic-float dot kernel (dot_dense); CPU
    oracle_dot is the bit-deterministic ground truth; CPU test always runs, ROCm test
    (#[cfg(feature="rocm")]) runs the same check on gfx1152 via cubecl_hip::HipRuntime;
    clients constructed directly (not via select_backend) to avoid racing on the
    process-global PYSCF_BACKEND env var.

    (2) `use cubecl::Runtime;` and `use pyscf_algebra::{AlgebraClient, dot_dense, oracle_dot};`.

    (3) Copy the `Lcg` deterministic LCG struct and `next_f64` VERBATIM from gemm_oracle.rs
    (Knuth/MMIX constants, maps to [-1,1)). Copy `random_matrix` and RENAME to `random_vector`
    (same body, takes len). It builds a `Vec&lt;f64&gt;` of length `len`.

    (4) `fn check_case(client: &amp;AlgebraClient, len: usize, seed: u64) -> f64`:
    `let mut rng = Lcg::new(seed); let x = random_vector(&amp;mut rng, len); let y = random_vector(&amp;mut rng, len);`
    `let device = dot_dense::&lt;f64&gt;(client, &amp;x, &amp;y).expect("dot_dense should succeed for equal-length vectors");`
    `let reference = oracle_dot(&amp;x, &amp;y);`
    `assert!(reference.is_finite(), "oracle_dot must be finite for equal-length vectors");`
    return `(device - reference).abs()`.

    (5) `const TOL: f64 = 1e-9;` with a comment: naive device products + host f64 sum vs
    pairwise oracle — differences are far below this in practice; generous enough to
    absorb backend reassociation while still catching a wrong kernel.

    (6) `const LENS: &amp;[usize] = &amp;[1, 2, 13, 31, 64, 97, 128, 255, 256, 257, 512, 1000];`
    with a comment: degenerate 1, primes, and lengths straddling the BLOCK=256 boundary
    to exercise the bounds-guard tail in the kernel.

    (7) `#[test] fn dot_kernel_matches_oracle_on_cpu()`: construct
    `let client = AlgebraClient::Cpu(cubecl_cpu::CpuRuntime::client(&amp;cubecl_cpu::CpuDevice));`
    loop over LENS.iter().enumerate(), seed `0x51ED_C0DE_u64 + i as u64`, assert
    `diff < TOL` with a message naming the length.

    (8) `#[cfg(feature = "rocm")] #[test] fn dot_kernel_matches_oracle_on_rocm()`: construct
    `let client = AlgebraClient::Rocm(cubecl_hip::HipRuntime::client(&amp;cubecl_hip::AmdDevice::default()));`,
    `assert!(matches!(client, AlgebraClient::Rocm(_)), "test must run on the ROCm backend, not a fallback");`
    loop over LENS, seed `0x0CA0_1A7E_u64 + i as u64`, assert `diff < TOL`.

    Match gemm_oracle.rs's exact import style, naming, and assertion-message format so the
    two oracle tests read identically.
  </action>
  <verify>
    <automated>cargo test -p pyscf-algebra --test dot_oracle dot_kernel_matches_oracle_on_cpu 2>&amp;1 | tee log/dot-iji-t2-cpu.log; test ${PIPESTATUS[0]} -eq 0</automated>
    <automated>cargo test -p pyscf-algebra --features rocm --test dot_oracle 2>&amp;1 | tee log/dot-iji-t2-rocm.log; test ${PIPESTATUS[0]} -eq 0</automated>
    <automated>grep -q "dot_kernel_matches_oracle_on_rocm" crates/pyscf-algebra/tests/dot_oracle.rs &amp;&amp; grep -q 'cfg(feature = "rocm")' crates/pyscf-algebra/tests/dot_oracle.rs</automated>
  </verify>
  <done>
    tests/dot_oracle.rs exists. dot_kernel_matches_oracle_on_cpu passes on the default
    CPU backend for all LENS within 1e-9. dot_kernel_matches_oracle_on_rocm passes on
    gfx1152 under --features rocm (the ROCm build is ~24s and pulls NO libxc). Full
    cargo output for both runs saved under log/ per CLAUDE.md. Both commands stay scoped
    to -p pyscf-algebra.
  </done>
</task>

</tasks>

<threat_model>
## Trust Boundaries

| Boundary | Description |
|----------|-------------|
| host slice → device buffer | `&[F]` host data crosses into a cubecl device buffer via `Bytes::from_elems`; `ArrayArg::from_raw_parts` lengths must match the just-allocated buffers |
| device buffer → host read-back | `client.read` bytes are reinterpreted as `&[F]` via `bytemuck::cast_slice`; length integrity matters |

## STRIDE Threat Register

| Threat ID | Category | Component | Disposition | Mitigation Plan |
|-----------|----------|-----------|-------------|-----------------|
| T-iji-01 | Tampering | `ArrayArg::from_raw_parts` length args in launch_dot | mitigate | lengths derived directly from `x.len()` / allocated buffer size; out_handle cloned before consumption (mirrors gemm.rs's verified pattern) |
| T-iji-02 | Denial of Service | mismatched-length vectors reaching the kernel | mitigate | `dot_dense` ShapeMismatch guard rejects `x.len() != y.len()` before any device upload |
| T-iji-03 | Information Disclosure | `bytemuck::cast_slice` over read-back bytes | accept | f64/f64 same-size cast on Pod types; buffer length == x.len()*size_of::<F>() by construction; no out-of-bounds reinterpretation |
| T-iji-SC | Tampering | npm/pip/cargo installs | accept | NO new dependencies — cubecl, cubecl-cpu, cubecl-hip, bytemuck already declared in Cargo.toml; no package-manager install task in this plan |
</threat_model>

<verification>
- `cargo build -p pyscf-algebra` succeeds (CPU default).
- `cargo test -p pyscf-algebra --test dot_oracle dot_kernel_matches_oracle_on_cpu` passes (≤1e-9 over all LENS).
- `cargo test -p pyscf-algebra --features rocm --test dot_oracle` passes on gfx1152.
- `cargo clippy -p pyscf-algebra` clean (FOUND-07: no unwrap_used; no FMA in pyscf_* symbols).
- `dot()` over Tensor still returns `NotYetImplemented`.
- No command pulls libxc_rs into the dep graph (all scoped to `-p pyscf-algebra`).
- All full cargo outputs saved under `log/` before investigating any build issue (CLAUDE.md).
</verification>

<success_criteria>
- `src/dot.rs` has a `#[cube(launch)]` `dot_kernel`, a Runtime-generic `launch_dot`, a public `dot_dense<F: DeviceScalar>`, and the preserved `dot()` Tensor stub.
- `lib.rs` re-exports both `dot` and `dot_dense`.
- `tests/dot_oracle.rs` matches `dot_dense` against `oracle_dot` within 1e-9 on CPU (always) and ROCm (under `--features rocm`), across degenerate/prime/boundary lengths.
- The cubecl `Runtime` generic never appears in a public signature (ALG-06 wall intact).
- Structure mirrors the gemm.rs / gemm_oracle.rs precedent so the two read identically.
</success_criteria>

<output>
Create `.planning/quick/260529-iji-refactor-pyscf-algebra-dot-rs-to-cubecl-/260529-iji-01-SUMMARY.md` when done.
</output>
