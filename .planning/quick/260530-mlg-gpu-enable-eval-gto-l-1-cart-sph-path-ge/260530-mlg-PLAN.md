---
phase: quick-260530-mlg
plan: 01
type: execute
wave: 1
subsystem: gpu-enable / eval_gto
tags: [cubecl, eval_gto, cart2sph, l-ge-1, general-kernel, differential-oracle, rocm]
depends_on: [quick-260530-ljv]
files_modified:
  - crates/pyscf-kernels/src/eval_gto.rs
  - crates/pyscf-kernels/tests/eval_gto_oracle.rs
autonomous: true
requirements: [GTO-07, ORACLE-07, D-04]
must_haves:
  truths:
    - "eval_gto_sph routes any basis whose max l is in 1..=4 to a real #[cube] device kernel (p/d/f/g) and the device result matches eval_gto_sph_cpu within 1e-9"
    - "Mixed-l bases (e.g. s+p+d like cc-pVDZ) evaluate uniformly on the device — l=0 is subsumed by the general kernel"
    - "l>4 (h-shell and above) NEVER reaches the kernel — it stays on eval_gto_sph_cpu and surfaces the c2s_coeff NotYetImplemented{phase:4} error, never a panic"
    - "Empty grid or empty basis falls back to eval_gto_sph_cpu (out_len==0 early return)"
    - "The existing pure-s-shell fast path (launch_eval_gto_s) is unchanged"
    - "All existing pyscf-kernels tests (eval_gto_lge1, eval_gto_oracle s-shell, lib, wave0 smoke) pass byte/ULP-for-byte after the change"
    - "The differential-oracle test gains randomized p/d/f (and g if feasible) mixed-l fixtures and a #[cfg(rocm)] gfx1152 arm"
  artifacts:
    - path: "crates/pyscf-kernels/src/eval_gto.rs"
      provides: "general #[cube] eval_gto_sph_kernel_general (l 0..=4) + #[cube] ipow helper + host angular-table builder + launch_eval_gto_general<R> + maxl<=4 routing in eval_gto_sph"
      contains: "eval_gto_sph_kernel_general"
    - path: "crates/pyscf-kernels/tests/eval_gto_oracle.rs"
      provides: "randomized mixed-l (l in {1,2,3}, plus a mixed s+p+d) differential fixtures vs eval_gto_sph_cpu, CpuRuntime always-on + rocm arm"
      contains: "matches_oracle"
  key_links:
    - from: "eval_gto_sph"
      to: "launch_eval_gto_general"
      via: "dispatch_backend! when maxl<=4 && ngrids*nao>0"
      pattern: "dispatch_backend!.*launch_eval_gto_general"
    - from: "eval_gto_sph_kernel_general"
      to: "ipow"
      via: "#[cube] helper call (Solution 1 from the host-fn-in-#[cube] pitfall guide)"
      pattern: "ipow"
    - from: "eval_gto_sph (l>4 / empty)"
      to: "eval_gto_sph_cpu"
      via: "unchanged CPU fallback preserving NotYetImplemented{phase:4}"
      pattern: "eval_gto_sph_cpu"
---

<objective>
Port the `l >= 1` cartesian-monomial + libcint cart→sph transform into a real
`#[cube]` GPU kernel so `eval_gto` runs on the device for bases with l>=1 shells
(p/d/f/g, l<=4), routed through the exported `dispatch_backend!`. The s-shell
slice (quick-260530-ljv) shipped and is validated; THIS slice adds the general
path and makes the device kernel the DEFAULT for any basis whose max angular
momentum is <= 4 (which subsumes l=0, so mixed s+p+d bases like cc-pVDZ run
uniformly on the device).

Purpose: GPU-enable the production AO-on-grid value path for real chemistry
bases (the s-only path only covered minimal-basis hydrogen-like cases). This is
the next Phase-8 GPU-enable increment.

Output:
- `eval_gto_sph_kernel_general` — a NEW f64-restricted `#[cube(launch_unchecked)]`
  kernel, ONE THREAD PER (g, shell), over HOST-PRECOMPUTED angular device tables.
- `ipow` — a `#[cube]` exp-by-squaring helper (Solution 1 from the pitfall guide).
- `launch_eval_gto_general<R: Runtime>` — host launcher building the 7 angular
  device tables + the libcint flat arrays.
- routing in `eval_gto_sph`: keep the all_s fast path; add a maxl<=4 device arm;
  l>4 / empty → unchanged `eval_gto_sph_cpu`.
- extended `tests/eval_gto_oracle.rs` differential gate (mixed-l, CPU + rocm).
</objective>

<execution_context>
@/home/user/Documents/workspace/pyscf_rs/.claude/get-shit-done/workflows/execute-plan.md
@/home/user/Documents/workspace/pyscf_rs/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@./CLAUDE.md
@/home/user/Documents/workspace/pyscf_rs/.planning/quick/260530-ljv-gpu-enable-eval-gto-via-macro-wrapped-mu/260530-ljv-SUMMARY.md

# cubecl 0.10 pitfall references (MANDATORY before touching #[cube]):
@/home/user/Documents/workspace/pyscf_rs/docs/manual/Cubecl/cubecl_error_solution_guide/calling a “normal” Rust function from inside a cube macro function fails in CubeCL.md
@/home/user/Documents/workspace/pyscf_rs/docs/manual/Cubecl/cubecl_error_solution_guide/mismatched types.md
@/home/user/Documents/workspace/pyscf_rs/docs/manual/Cubecl/Cubecl_conditionals.md

# The file being modified — READ FULLY before editing:
@crates/pyscf-kernels/src/eval_gto.rs

# The differential-oracle test to extend:
@crates/pyscf-kernels/tests/eval_gto_oracle.rs

# Existing l>=1 independent-reference gate (MUST stay green — it now validates the GPU path):
@crates/pyscf-kernels/tests/eval_gto_lge1.rs

<interfaces>
<!-- Extracted from the codebase — executor uses these DIRECTLY, no exploration. -->

From crates/pyscf-kernels/src/eval_gto.rs (HOST-ONLY angular helpers — they
return Vec/Result, so they are ILLEGAL inside #[cube]; call them ONLY on the
host inside the table builder):

  fn common_fac_sp(l: u32) -> f64           // l=0 → 0.28209…, l=1 → 0.48860…, else 1.0
  fn ncart(l: u32) -> usize                 // (l+1)(l+2)/2
  fn nsph(l: u32) -> usize                  // 2l+1
  fn cart_powers(l: u32) -> Vec<(u32,u32,u32)>   // (lx,ly,lz) per cart col, upstream order lx=l..0
  fn c2s_coeff(l: u32, m_row: usize, cart_col: usize) -> Result<f64, PyscfRsError>
        // FROZEN libcint c2s; Ok for l<=4, Err(NotYetImplemented{phase:4}) for l>4

From the host CPU oracle target (eval_gto_sph_cpu l>=1 branch, lines ~809-869) —
the EXACT reduction order the kernel must mirror:
  radial(c_idx) = ( Σ_p coef[c_idx*nprim+p] * exp(-alpha_p * r2) ) * fac1
                  // host uses oracle_sum for nprim>2, plain acc for nprim<=2;
                  // oracle_sum == pairwise(128) == STRICT SEQUENTIAL for nprim<=128,
                  // so a sequential acc in the kernel matches host bit-for-bit.
  mono[ci]      = dx^lx * dy^ly * dz^lz        // host: dx.powi(lx) etc.
  cart_vals[ci] = mono[ci] * radial
  v(m)          = Σ_ci c2s[l][m][ci] * cart_vals[ci]
  out[g + (ao_off + c_idx*nsph_l + m)*ngrids] = v(m)

From crates/pyscf-algebra (exported macro + client enum):
  pyscf_algebra::dispatch_backend!(client, c, Rt, expr_using::<Rt>(c, ...))
  pyscf_algebra::AlgebraClient   // ::Cpu(_), ::Rocm(_), … cfg-gated

From the EXISTING working s-shell kernel/launcher (the idiom to FOLLOW for 0.10):
  #[cube(launch_unchecked)] fn eval_gto_sph_kernel(coords:&Array<f64>, env:&Array<f64>,
      bas:&Array<i32>, atm:&Array<i32>, ao_loc:&Array<i32>, out:&mut Array<f64>,
      ngrids:usize, nbas:usize, nao:usize, y00:f64, atm_slots:usize, … : usize)
    // PROVEN in 0.10: usize scalar args OK; ABSOLUTE_POS is usize; i32-array
    // indexing OK (cast to usize); `(-alpha*r2).exp()` method-style OK here.
  fn launch_eval_gto_s<R: Runtime>(client:&ComputeClient<R>, …) -> Vec<f64>
    // client.create(Bytes::from_elems(v.to_vec())) per array; client.empty(out);
    // CubeCount::Static(groups,1,1); CubeDim::new_1d(EVAL_GTO_BLOCK);
    // ArrayArg::from_raw_parts(handle.clone(), len) (2-arg, no turbofish);
    // bare scalar args; client.read + bytemuck::cast_slice::<u8,f64>.
  const EVAL_GTO_BLOCK: u32 = 256;

From crates/pyscf-core::raw_layout (already imported in the file):
  ANG_OF, ATM_SLOTS, ATOM_OF, BAS_SLOTS, NCTR_OF, NPRIM_OF, PTR_COEFF, PTR_COORD, PTR_EXP
</interfaces>
</context>

<design_verification>
The orchestrator's design was checked against the code and HOLDS with no
contradictions. Key confirmations baked in:

1. PITFALL SIDESTEP CONFIRMED — `cart_powers`/`c2s_coeff`/`common_fac_sp`/`ncart`/
   `nsph` return `Vec`/`Result`, so they are host-only (illegal in `#[cube]`).
   The kernel must do pure indexed arithmetic over host-precomputed device arrays.
   Per the pitfall guide, the ONLY helper called from `#[cube]` (`ipow`) must
   itself be `#[cube]` (Solution 1).

2. BIT-EXACTNESS CONFIRMED — `oracle_sum` = `pairwise(xs, 128)` collapses to a
   STRICT SEQUENTIAL left-to-right sum for `len <= 128` (oracle.rs:79-86). Real
   shells have nprim <= ~30, so the kernel's sequential `acc += coef*exp(...)`
   == host `oracle_sum` bit-for-bit. NOT a tolerance gap. The host applies `fac1`
   AFTER the sum (`radial = radial * fac1`, line 850) — mirror that order exactly.

3. ONLY sub-ULP divergence = `ipow` vs host `f64::powi` (l>=3 monomials), ~1e-16,
   far inside TOL=1e-9 and inside the downstream `eval_gto_deriv1_oracle` 1e-10
   budget. So routing l>=1 to GPU BY DEFAULT is behavior-preserving to within
   documented tolerance (ORACLE-07).

4. usize scalar args + `(-x).exp()` method-style + i32-array indexing ALL work in
   this crate's pinned cubecl 0.10.0 — PROVEN by the shipped `eval_gto_sph_kernel`.
   FOLLOW that kernel's idiom (do NOT "fix" it to the generic error-guide advice).
   The error-guide caveats DO apply to the NEW `ipow` helper's `if`-as-value: use
   the statement form (`let mut r = 1.0; if n>=1 { r = base; } …`), NOT
   `let r = if n==0 {1.0} else {...}`.

5. l>4 NEVER reaches the kernel: routing computes `maxl` and only dispatches the
   device path when `maxl <= 4`. The host tables are built only for `l in 0..=maxl`
   with `maxl <= 4`, so `c2s_coeff` is never called with l>4 on the device path.
   l>4 stays on `eval_gto_sph_cpu` which preserves the `NotYetImplemented{phase:4}`
   error (verified: c2s_coeff wildcard arm + the `c2s_coeff_l5_returns_err_not_panic`
   unit test). This behavior MUST remain green.

6. EXISTING GATE BONUS — `tests/eval_gto_lge1.rs` already diffs `eval_gto_sph`
   (which after T2 routes l>=1 to the GPU kernel) against an INDEPENDENT longhand
   c2s reference over s/p/d shells. After this change it validates the GPU path
   against a different code path. It MUST stay green (strong behavior gate).

Note: tasks T1 and T2 are merged per the orchestrator's allowance (the kernel,
launcher, table builder, ipow helper, and routing all live in one file and are
mutually coupled — splitting them would create a non-compiling intermediate).
</design_verification>

<device_table_schema>
The settled angular-table schema (built on the host for `l in 0..=maxl`,
`maxl <= 4`). All offsets are prefix sums so the kernel indexes by l with O(1)
arithmetic. `cart_pow` uses 3 PARALLEL i32 arrays (lx/ly/lz) — simpler kernel
indexing than interleaving, and i32-array reads are proven to work.

  c2s_flat:     Array<f64>   // concatenated c2s matrices; T[l][m][ci] at
                             //   c2s_off_by_l[l] + m*ncart(l) + ci
                             //   (built by c2s_coeff(l,m,ci) on the host; l<=4 only)
  cpow_lx:      Array<i32>   // lx per cart col, at cpow_off_by_l[l] + ci
  cpow_ly:      Array<i32>   // ly per cart col, at cpow_off_by_l[l] + ci
  cpow_lz:      Array<i32>   // lz per cart col, at cpow_off_by_l[l] + ci
  ncart_by_l:   Array<i32>   // len maxl+1
  nsph_by_l:    Array<i32>   // len maxl+1
  fac1_by_l:    Array<f64>   // len maxl+1  (= common_fac_sp(l))
  c2s_off_by_l: Array<i32>   // len maxl+1  (prefix sum of ncart(l)*nsph(l))
  cpow_off_by_l:Array<i32>   // len maxl+1  (prefix sum of ncart(l))

Subsumption check (l=0): ncart=nsph=1, c2s=[[1.0]], (lx,ly,lz)=(0,0,0),
fac1=common_fac_sp(0)=0.28209…=Y00, mono=ipow(dx,0)*…=1.0 → out = 1.0*radial*Y00.
This MUST byte-match the host s-shell value (Y00 folded into the radial there);
the CpuRuntime oracle arm with a mixed s+p+d fixture pins it.
</device_table_schema>

<tasks>

<task type="auto" tdd="false">
  <name>Task 1: General l-0..4 #[cube] kernel + ipow helper + host table builder + launcher + maxl<=4 routing</name>
  <files>crates/pyscf-kernels/src/eval_gto.rs</files>
  <behavior>
    - Kernel result for an all-s basis byte-matches the existing s-shell path (Y00 subsumption).
    - Kernel result for p/d/f bases matches eval_gto_sph_cpu within <1e-9 (ipow vs powi is the only divergence).
    - l>4 basis: NO device dispatch; eval_gto_sph_cpu still returns NotYetImplemented{phase:4} (no panic).
    - Empty grid / empty basis: eval_gto_sph_cpu early-return path; no kernel launch.
    - all_s fast path (launch_eval_gto_s) still reached for pure-s bases (do NOT disturb it).
  </behavior>
  <action>
In crates/pyscf-kernels/src/eval_gto.rs, ADD (do NOT remove the existing s-shell
kernel/launcher or the all_s routing):

(a) `#[cube] #[inline(always)] fn ipow(base: f64, n: u32) -> f64` — exp-by-squaring
    for n<=4 using the STATEMENT form per the mismatched-types guide (Section 1.3):
    initialise `let mut r = 1.0f64;` then `if n >= 1 { r = base; } if n >= 2 { r = r * base; } if n >= 3 { r = r * base; } if n == 4 { r = r * base; }`
    — i.e. r = base^n via repeated multiply (n: 0→1.0, 1→base, 2→base*base, 3→base^3, 4→base^4).
    This is Solution 1 from the host-fn-in-#[cube] pitfall guide (a #[cube] helper).
    Document the doc-comment: this may differ from host `f64::powi` by <1 ULP at
    l>=3 but is bounded by TOL=1e-9 (ORACLE-07). Do NOT use `let r = if ... {}`
    expression form (triggers the ExpandElementTyped vs {float} mismatch). The loop
    bounds `n` come from i32 cart_pow values cast to u32 in the kernel.

(b) `#[cube(launch_unchecked)] fn eval_gto_sph_kernel_general(...)` — f64-restricted,
    ONE THREAD PER (g, shell): `let tid = ABSOLUTE_POS;` `if tid < ngrids*nbas { let g = tid % nbas-stride... }`.
    Concretely: `g = tid / nbas; let shell = tid % nbas;` (guard `tid < ngrids*nbas`).
    Args (follow the s-kernel idiom — usize scalar args OK):
      coords:&Array<f64>, env:&Array<f64>, bas:&Array<i32>, atm:&Array<i32>, ao_loc:&Array<i32>,
      c2s_flat:&Array<f64>, cpow_lx:&Array<i32>, cpow_ly:&Array<i32>, cpow_lz:&Array<i32>,
      ncart_by_l:&Array<i32>, nsph_by_l:&Array<i32>, fac1_by_l:&Array<f64>,
      c2s_off_by_l:&Array<i32>, cpow_off_by_l:&Array<i32>, out:&mut Array<f64>,
      ngrids:usize, nbas:usize, atm_slots:usize, bas_slots:usize, atom_of:usize, ang_of:usize,
      nprim_of:usize, nctr_of:usize, ptr_exp:usize, ptr_coeff:usize, ptr_coord:usize.
    Body per thread:
      - read coords[g], coords[g+ngrids], coords[g+2*ngrids].
      - bas_row = shell*bas_slots; l = bas[bas_row+ang_of] as u32; atom_id, nprim, nctr,
        pe=ptr_exp slot, pc=ptr_coeff slot, ao_off = ao_loc[shell] as usize.
      - atm coords via atm[atom_id*atm_slots+ptr_coord]; dx,dy,dz,r2.
      - l-indexed reads: `let lu = l as usize;` ncart_l = ncart_by_l[lu] as usize/u32 as needed;
        nsph_l, fac1 = fac1_by_l[lu]; c2s_off = c2s_off_by_l[lu] as usize; cpow_off = cpow_off_by_l[lu] as usize.
      - for c_idx in 0..nctr {
          radial: sequential `let mut acc = 0.0; for p in 0..nprim { let alpha = env[pe+p];
            let coef = env[pc + c_idx*nprim + p]; acc += coef * (-alpha*r2).exp(); }`
            (method-style .exp() — same as the shipped s-kernel); then `let radial = acc * fac1;`.
          for m in 0..nsph_l {
            `let mut v = 0.0;`
            for ci in 0..ncart_l {
              `let lx = cpow_lx[cpow_off + ci] as u32;` (ly, lz likewise);
              `let mono = ipow(dx,lx) * ipow(dy,ly) * ipow(dz,lz);`
              `let cart_val = mono * radial;`
              `v += c2s_flat[c2s_off + m*ncart_l + ci] * cart_val;`
            }
            `out[g + (ao_off + c_idx*nsph_l + m)*ngrids] = v;`
          }
        }
    If any cube op fails to compile (e.g. usize loop bound, i32→u32 cast, the
    ncart/nsph cast), fix forward: cast i32 array reads to u32 for loop bounds,
    keep f64 math; document the fix in the SUMMARY. Save cargo output to log/.

(c) Host table builder `fn build_angular_tables(maxl: u32) -> Result<AngularTables, PyscfRsError>`
    (or inline in the launcher) returning the 9 Vecs of the device_table_schema.
    Build for `l in 0..=maxl` (caller guarantees maxl<=4): call the HOST helpers
    `ncart(l)`, `nsph(l)`, `common_fac_sp(l)`, `cart_powers(l)`, and `c2s_coeff(l,m,ci)?`
    (propagate the Err — but caller only invokes with maxl<=4 so it never errors here).
    Prefix-sum c2s_off_by_l (Σ ncart(l)*nsph(l)) and cpow_off_by_l (Σ ncart(l)).

(d) `fn launch_eval_gto_general<R: Runtime>(client, coords, ngrids, atm, bas, env,
    ao_loc, nao, maxl) -> Result<Vec<f64>, PyscfRsError>` modeled on launch_eval_gto_s:
    build tables (propagate Err), client.create all 5 libcint arrays + the 9 angular
    arrays, client.empty(ngrids*nao*8), groups = (ngrids*nbas).div_ceil(EVAL_GTO_BLOCK),
    CubeDim::new_1d(EVAL_GTO_BLOCK), launch_unchecked with ArrayArg::from_raw_parts(h.clone(), len)
    per array (2-arg) + bare scalar args, client.read + bytemuck back. Reuse EVAL_GTO_BLOCK.

(e) In `eval_gto_sph`: KEEP the all_s fast-path block unchanged. After it, BEFORE the
    `eval_gto_sph_cpu` fallback, add:
      `let maxl = bas.chunks_exact(BAS_SLOTS).map(|r| r[ANG_OF]).max().unwrap_or(0) as u32;`
      `if !bas.is_empty() && maxl <= 4 && ngrids * nao > 0 {`
         `let _ = spherical;`  // l>=1 always emits SPHERICAL AOs, matching eval_gto_sph_cpu
         `let values = dispatch_backend!(client, c, Rt, launch_eval_gto_general::<Rt>(c, coords, ngrids, atm, bas, env, ao_loc, nao, maxl))?;`
         `return Ok(EvalGtoBuffers { values, shape: vec![ngrids, nao] });`
      `}`
    The final `eval_gto_sph_cpu(...)` call stays as the fallback for maxl>4 / empty.
    Note: since the all_s path already handles pure-s, the general arm primarily serves
    mixed/l>=1 bases — but it is harmless if all_s ever falls through (subsumes l=0).
    If `dispatch_backend!` arm returns a Result, propagate `?`; if it returns Vec<f64>
    directly (launcher returns Result), bind then `?` — match the launcher signature.

DO NOT inline fenced code as production code beyond what is described; follow the
shipped s-kernel for every 0.10 idiom (usize args, ABSOLUTE_POS usize, 2-arg
from_raw_parts, CubeDim::new_1d, method-style .exp()).
  </action>
  <verify>
    <automated>cargo build -p pyscf-kernels 2>&1 | tee log/mlg-t1-build.log | tail -20; cargo test -p pyscf-kernels --lib 2>&1 | tee log/mlg-t1-lib.log | tail -20</automated>
  </verify>
  <done>
    pyscf-kernels builds (no libxc pulled); the existing lib unit tests
    (c2s_coeff_l5_returns_err_not_panic, c2s_coeff_l_le_4_unchanged) pass;
    eval_gto_sph_kernel_general + ipow + launch_eval_gto_general + the maxl<=4
    routing arm exist; the all_s fast path and eval_gto_sph_cpu fallback are
    unchanged. log/mlg-t1-*.log saved.
  </done>
</task>

<task type="auto" tdd="false">
  <name>Task 2: Behavior-preservation — existing l>=1 + s-shell gates stay green on the new device path</name>
  <files>crates/pyscf-kernels/src/eval_gto.rs</files>
  <behavior>
    - eval_gto_lge1.rs (independent c2s reference, s/p/d shells) passes against eval_gto_sph now routing l>=1 to the GPU kernel.
    - eval_gto_oracle.rs s-shell arm still passes (pure-s bases reach all_s fast path, unaffected).
    - wave0 smoke still passes.
  </behavior>
  <action>
No new code expected in the common case — this task RUNS the existing
behavior-preservation gates against the changed routing and only fixes forward if a
gate fails (e.g. an index-math or fac1-order mismatch surfaced by eval_gto_lge1.rs).
Run the full default-feature pyscf-kernels test suite. If eval_gto_lge1.rs fails,
the kernel's reduction order / c2s indexing / cart_pow order diverges from the host
— diff against the eval_gto_sph_cpu l>=1 branch (lines ~809-869) and the
device_table_schema offsets, fix the kernel (NOT the test), re-run. Save cargo
output to log/mlg-t2-*.log. Do NOT run `-p pyscf-gto` here (it pulls cintx; the
pyscf-kernels-level eval_gto_lge1 + eval_gto_oracle gates fully cover the change
without the heavier dep graph — pyscf-gto has no direct libxc dep but cintx may
pull it, so scope verification to pyscf-kernels per the no-libxc constraint).
  </action>
  <verify>
    <automated>cargo test -p pyscf-kernels 2>&1 | tee log/mlg-t2-all.log | tail -40; grep -v '^#' log/mlg-t2-all.log | grep -c 'test result: FAILED'</automated>
  </verify>
  <done>
    `cargo test -p pyscf-kernels` (default features): eval_gto_lge1 (s/p/d
    independent-reference), eval_gto_oracle (s-shell), lib, and wave0 smoke ALL
    pass; zero `test result: FAILED` lines. The grep gate prints 0. No libxc in
    the build. log/mlg-t2-all.log saved.
  </done>
</task>

<task type="auto" tdd="false">
  <name>Task 3: Extend the differential oracle with mixed-l (p/d/f, g if feasible) + rocm arm</name>
  <files>crates/pyscf-kernels/tests/eval_gto_oracle.rs</files>
  <behavior>
    - A new fixture builds valid libcint atm/bas/env/ao_loc with shells of l in {1,2,3} (plus a mixed s+p+d basis), randomized via Lcg.
    - eval_gto_sph (device, now routing l>=1 to the general kernel) vs eval_gto_sph_cpu (production host longhand) — max_abs_diff < 1e-9 over the fixtures.
    - CpuRuntime arm always runs; #[cfg(feature="rocm")] arm runs on gfx1152.
  </behavior>
  <action>
Extend crates/pyscf-kernels/tests/eval_gto_oracle.rs (KEEP the existing s-shell
test + fixtures). Add a mixed-l differential test:

(a) GROUND TRUTH: import the production host longhand as the oracle. Since
    `eval_gto_sph_cpu` is private, the oracle is `eval_gto_sph` itself called on a
    CPU client (`AlgebraClient::Cpu(...)`) — BUT that now routes l>=1 to the device
    kernel too, so that is NOT an independent reference. Instead: compare the DEVICE
    client result against a CPU client result is invalid (both device). Per the
    hard constraint, ground truth = `eval_gto_sph_cpu` (the host longhand). Because
    it is private, ADD an inline longhand oracle in the test that BYTE-COPIES the
    `eval_gto_sph_cpu` l>=1 branch (lines ~809-869): the cart_powers order, the
    common_fac_sp(l) fac1, the FROZEN c2s matrices (copy the L1..L3 values from the
    file, or reuse the eval_gto_lge1.rs `reference::*` longhand approach which
    already encodes c2s for l<=2 — extend to l=3/f), the sequential radial sum, and
    the F-order write `out[g+(ao_off+c_idx*nsph_l+m)*ngrids]`. Copy VERBATIM, do not
    re-derive (ORACLE PIN, same discipline as the s-shell oracle and eval_gto_lge1).
    Alternatively (PREFERRED, less duplication): make the device-vs-oracle check
    delegate to the SAME independent longhand reference already proven in
    eval_gto_lge1.rs — replicate its `reference` module (s/p/d) and extend with f
    (l=3) c2s rows from the frozen L3 table in eval_gto.rs. This gives a DIFFERENT
    code path from the kernel (true differential test). The executor picks whichever
    keeps the oracle independent of the kernel; document the choice.

(b) FIXTURE: extend the Lcg-based `build_fixture` (or add `build_mixed_l_fixture`)
    to emit shells with l in {1,2,3} and a mixed s+p+d basis. Set bas[row+ANG_OF]=l,
    NCTR_OF/NPRIM_OF as today, ao_loc running sum uses `nctr * nsph(l)` per shell
    (NOT nctr — l>=1 has 2l+1 AOs per contraction). Coefficients F-order at PTR_COEFF.
    nao = final ao_loc running sum. Include g (l=4) if the c2s L4 reference is added;
    g is OPTIONAL (note it if skipped — l<=3 covers cc-pVTZ; g is cc-pV5Z territory).

(c) TESTS: `eval_gto_lge1_matches_oracle_on_cpu` (always-on, CpuRuntime) asserting
    max_abs_diff < TOL=1e-9 over a CASES spread (varied n_atoms/shells/ngrids/l-mix);
    `#[cfg(feature="rocm")] eval_gto_lge1_matches_oracle_on_rocm` constructing
    `AlgebraClient::Rocm(cubecl_hip::HipRuntime::client(&cubecl_hip::AmdDevice::default()))`,
    asserting `matches!(client, AlgebraClient::Rocm(_))` then the same diff < TOL.
    eprintln! the worst max_abs_diff per arm for the SUMMARY. Use distinct seeds
    from the s-shell test to avoid fixture collisions.
  </action>
  <verify>
    <automated>cargo test -p pyscf-kernels --test eval_gto_oracle 2>&1 | tee log/mlg-t3-cpu.log | tail -30; grep -v '^#' log/mlg-t3-cpu.log | grep -c 'test result: ok'</automated>
  </verify>
  <done>
    The CpuRuntime mixed-l differential test passes (max_abs_diff < 1e-9 over
    p/d/f, plus g if added, fixtures); the existing s-shell test still passes;
    grep gate shows >=1 `test result: ok`. The rocm arm compiles under
    `--features rocm`. log/mlg-t3-cpu.log saved. (rocm RUN happens in the
    phase-level verification below.)
  </done>
</task>

</tasks>

<threat_model>
## Trust Boundaries

| Boundary | Description |
|----------|-------------|
| host → device kernel | host-precomputed angular tables + libcint flat arrays cross to GPU memory; mis-sized handles or wrong offsets corrupt output |
| user basis → routing | a user-supplied basis with l>4 must NOT reach the kernel (host tables only built to l=4) |
| cubecl 0.10 install | pinned cubecl-* crates already in the dep graph (no NEW package installs) |

## STRIDE Threat Register

| Threat ID | Category | Component | Disposition | Mitigation Plan |
|-----------|----------|-----------|-------------|-----------------|
| T-mlg-01 | Tampering | l>4 basis reaching the device kernel (no c2s table) | mitigate | routing gates on `maxl <= 4`; host table builder only iterates `0..=maxl`; l>4 stays on eval_gto_sph_cpu → NotYetImplemented{phase:4}; the existing `c2s_coeff_l5_returns_err_not_panic` unit test stays green |
| T-mlg-02 | Information disclosure | wrong device-array length in from_raw_parts → OOB read of adjacent GPU memory | mitigate | launcher uses exact slice `.len()` per handle (mirrors launch_eval_gto_s); kernel bounds-guards `tid < ngrids*nbas`; differential oracle would surface garbage as a diff > TOL |
| T-mlg-03 | Denial of service | div_ceil group count overflow / empty grid launch | accept | empty grid/basis routes to eval_gto_sph_cpu early-return BEFORE any launch; ngrids*nbas>0 guaranteed on the device arm by the `ngrids*nao>0` gate |
| T-mlg-SC | Tampering | npm/pip/cargo installs | accept | NO new package installs — cubecl-* are already pinned in pyscf-kernels/Cargo.toml from quick-260530-ljv; no Package Legitimacy Gate needed |
</threat_model>

<verification>
Phase-level checks (run after all three tasks):

1. `cargo build -p pyscf-kernels` and `cargo build -p pyscf-kernels --features rocm`
   both succeed (no libxc in either). Save to log/mlg-verify-build.log.
2. `cargo test -p pyscf-kernels` (default cpu): ALL pass — eval_gto_lge1 (s/p/d
   independent reference now validating the GPU path), eval_gto_oracle (s-shell +
   new mixed-l), lib, wave0 smoke. Save to log/mlg-verify-cpu.log.
3. ROCm gfx1152 RUN (per hard constraint — gfx1152 is available):
   `cargo test -p pyscf-kernels --features rocm --test eval_gto_oracle 2>&1 | tee log/mlg-verify-rocm.log`
   — BOTH the s-shell and the new mixed-l rocm arms pass with worst
   max_abs_diff < 1e-9. Record the observed worst diff.
4. `cargo clippy -p pyscf-kernels --all-targets` clean (no new warnings). Save to
   log/mlg-verify-clippy.log.
5. pyscf-algebra untouched; verify the dependency wall is intact:
   `cargo run -p xtask -- check-dependency-wall` (if cheap) OR confirm no new
   cubecl import crossed into a method crate (only eval_gto.rs changed, already on
   the allowlist).
6. Do NOT run `-p pyscf-gto` (cintx → possible libxc). The pyscf-kernels-level
   eval_gto_lge1 + eval_gto_oracle gates fully cover the routing change.
</verification>

<success_criteria>
- `eval_gto_sph` routes any basis with `1 <= maxl <= 4` to `eval_gto_sph_kernel_general`
  via `dispatch_backend!`; device result matches `eval_gto_sph_cpu` within 1e-9
  (CpuRuntime: expected ~0 to <1e-9; rocm gfx1152: <1e-9, ~1e-12..1e-16 observed).
- Mixed s+p+d bases evaluate uniformly on the device (l=0 subsumed).
- l>4 and empty grid/basis stay on `eval_gto_sph_cpu` (NotYetImplemented{phase:4}
  preserved; `c2s_coeff_l5_returns_err_not_panic` green).
- The all_s fast path (`launch_eval_gto_s`) is unchanged.
- ALL existing pyscf-kernels tests pass (eval_gto_lge1 now validating the GPU path
  against an independent reference is the key behavior gate).
- The differential oracle gains randomized p/d/f (and g if feasible) mixed-l
  fixtures, CpuRuntime always-on + rocm gfx1152 arm; rocm arm RUN on hardware.
- clippy clean; no libxc; cargo output saved under log/mlg-*.log.
</success_criteria>

<output>
Create `.planning/quick/260530-mlg-gpu-enable-eval-gto-l-1-cart-sph-path-ge/260530-mlg-SUMMARY.md` when done.

Required statements in the SUMMARY:
1. The final device-table schema actually used (parallel cpow_lx/ly/lz vs
   interleaved — note any deviation from the plan's 3-parallel-array choice).
2. Whether `ipow` compiled as a `#[cube]` helper (Solution 1) or had to be inlined
   (Solution 2) — and any 0.10 cube-op fix-forwards (i32→u32 casts, usize loop
   bounds, .exp() form).
3. The differential oracle's ground-truth choice (inline byte-copy of
   eval_gto_sph_cpu vs the independent eval_gto_lge1-style longhand reference) and
   whether g (l=4) fixtures were included.
4. Observed worst max_abs_diff on the CpuRuntime mixed-l arm AND whether the
   eval_gto_lge1.rs independent-reference gate stayed green against the GPU path.
5. Whether the rocm gfx1152 arm was RUN (not just compiled) and its worst
   max_abs_diff.
6. Any place the code contradicted the orchestrator's design (expected: none).
</output>
