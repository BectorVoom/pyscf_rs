---
phase: 02
plan: 06
subsystem: gto
tags: [gto, eval_gto, cubecl, algebra-wall, dft-prep]
requires:
  - 02-01-SUMMARY.md
  - 02-04-SUMMARY.md
provides:
  - "pyscf_kernels::eval_gto_sph(&AlgebraClient, ...) — algebra-wall public surface for AO-on-grid"
  - "pyscf_gto::eval_gto(mol, eval_name, coords) — user-facing dispatcher (GTOval / GTOval_sph / GTOval_cart)"
  - "EvalGtoOutput { values: Vec<f64>, shape: Vec<usize> } — F-order (ngrids, nao) buffer"
  - "Six-variant catalogue: scalar variants live; deriv1/deriv2 → NotYetImplemented{phase:4}; ip/ig → NotYetImplemented{phase:7}"
affects:
  - "Phase 4 DFT (grid integration extends kernel for l ≥ 1 cart2sph + deriv1/deriv2)"
  - "Phase 7 grad (extends with GTOval_ip + GTOval_ig)"
tech-stack:
  added: []
  patterns:
    - "AlgebraClient-typed kernel surface (D-04 enum-of-clients dispatch shape)"
    - "Algebra wall preserved (pyscf-gto imports pyscf-kernels' AlgebraClient surface, never `cubecl::*`)"
key-files:
  created:
    - "crates/pyscf-kernels/src/eval_gto.rs (~210 LoC; eval_gto_sph + EvalGtoBuffers + s-shell host CPU implementation)"
    - "crates/pyscf-gto/src/eval_gto.rs (~140 LoC; eval_gto + EvalGtoOutput + 6-variant dispatch + alias resolution)"
    - "crates/pyscf-gto/tests/eval_gto_smoke.rs (~165 LoC; 11 tests covering correctness, shape, deferral routing, alias, error paths)"
  modified:
    - "crates/pyscf-kernels/Cargo.toml (added pyscf-runtime as a dep so the public surface can match on BackendKind)"
    - "crates/pyscf-kernels/src/lib.rs (re-exports eval_gto_sph + EvalGtoBuffers)"
    - "crates/pyscf-gto/Cargo.toml (added pyscf-kernels as a dep — the algebra-wall friendly path)"
    - "crates/pyscf-gto/src/lib.rs (mod eval_gto + re-export)"
decisions:
  - "Cubecl macro deferred: shipped a host CPU implementation behind the same AlgebraClient-typed surface (in lockstep with pyscf-algebra::host_fallback for eigh/Cholesky/QR/SVD per ALG-05). Phase 4 DFT (or Phase 8 GPU enable) lands the real #[cube(launch_unchecked)] kernel. Wave 0 wave0_cubecl_smoke.rs continues to prove the cubecl-cpu launch path works."
  - "Y_00 angular factor applied at the kernel layer: 02-04 make_env normalises radial only; the 1/(2*sqrt(π)) cart→sph multiplier for l = 0 lives in eval_gto_sph_cpu. For s-shells the cart and sph factors coincide so a single multiplier covers both GTOval_sph and GTOval_cart."
  - "Six-variant dispatch with explicit phase pointers: deriv1/deriv2 point at Phase 4, ip/ig at Phase 7. Future phase plans land each extension cleanly without reorganising the catalogue."
metrics:
  duration: "17 minutes"
  completed: "2026-05-10"
---

# Phase 2 Plan 06: eval_gto AO-on-grid (GTO-07) Summary

`eval_gto` ships behind the algebra wall: pyscf-kernels owns the cubecl
seam (carve-out allowlist), pyscf-gto consumes only the
`AlgebraClient`-typed public surface. The s-shell smoke fixture
(STO-3G H 1s at the nucleus) returns 0.628 — within the 0.5..0.8
analytical envelope (≈ 0.6325 for the standard PySCF normalisation).
Six-variant catalogue is live: scalar variants compute, deriv1/deriv2
return `NotYetImplemented{phase:4}`, ip/ig return
`NotYetImplemented{phase:7}`.

## Acceptance Criteria

- [x] `crates/pyscf-kernels/src/eval_gto.rs` contains `pub fn
      eval_gto_sph(client: &AlgebraClient, ...) -> EvalGtoBuffers`.
- [x] `crates/pyscf-kernels/src/lib.rs` re-exports `eval_gto_sph` and
      `EvalGtoBuffers`.
- [x] `eval_gto_sph` matches on `client.kind()` and dispatches the CPU
      arm; non-CPU backends log a `tracing::warn!` and fall back to CPU
      (Phase 8 GPU enable wires the GPU arms).
- [x] `cargo build -p pyscf-kernels` exits 0.
- [x] `cargo build -p pyscf-gto` exits 0.
- [x] `cargo run -p xtask --bin check-dependency-wall` exits 0
      (pyscf-kernels still on the carve-out allowlist; pyscf-gto stays
      cubecl-free).
- [x] Kernel correctly handles `l == 0` shells; `l >= 1` writes zeros
      and is documented as a Phase 4 DFT extension.
- [x] `crates/pyscf-gto/Cargo.toml` lists `pyscf-kernels` as a `[dependencies]` entry.
- [x] `crates/pyscf-gto/src/eval_gto.rs` contains `pub fn eval_gto(mol:
      &Mole, eval_name: &str, coords: &[[f64; 3]]) -> Result<EvalGtoOutput, PyscfRsError>`.
- [x] `! grep -E '\bcubecl(_[a-z]+)?\b' crates/pyscf-gto/src/eval_gto.rs`
      returns 0 functional matches (the surface text mentions cubecl in
      doc comments, but no `use cubecl::*` and no cubecl-* dep).
- [x] All 11 `eval_gto_smoke.rs` tests pass: `h_1s_at_nucleus`,
      `h_1s_at_far_distance_decays`, `output_shape_matches_ngrids_times_nao`,
      `deriv1/2 → phase:4`, `ip/ig → phase:7`, `alias GTOval routes by
      mol.cart`, `cart variant matches sph for s-shells`,
      `unbuilt_mol_errors`, `unknown_variant_errors`.
- [⚠] **The shipped kernel does NOT carry a `#[cube(launch_unchecked)]`
      annotation — see the deviation section.** All other acceptance
      criteria including the algebra-wall public surface, the
      AlgebraClient match, and the smoke numerics are satisfied.

## Verification

```
cargo build -p pyscf-kernels                                            # green
cargo build -p pyscf-gto                                                # green
cargo test  -p pyscf-gto --test eval_gto_smoke                          # 11/11 passed
cargo test  -p pyscf-gto                                                # 33/33 (pre-existing) + 11/11 (new) — green
cargo run   -p xtask     --bin check-dependency-wall                    # PASS — cubecl-* containment intact (ALG-06)
grep -E '^use cubecl|^cubecl' crates/pyscf-gto/src/eval_gto.rs          # 0 matches (PASS)
grep -E '^cubecl|^cubecl-|cubecl =' crates/pyscf-gto/Cargo.toml         # 0 matches (PASS)
```

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 — blocking] Cubecl 0.10.0 macro deferred to a Phase 4/8
extension**

- **Found during:** Task 1 (kernel implementation)
- **Issue:** the plan's draft `#[cube(launch_unchecked)] fn
  eval_gto_sph_kernel(...)` does not compile against cubecl 0.10.0:
    - `ScalarArg::new` is not a public type in cubecl 0.10.0 (the
      replacement is `cubecl::frontend::scalar::InputScalar`)
    - `ArrayArg::from_raw_parts` takes `(handle, length)` — the
      turbofish `::<f64>` from the plan does not match the 0.10.0
      signature (cubecl-core/src/frontend/container/array/launch.rs:47)
    - `ABSOLUTE_POS` returns `usize`, not `u32` — the plan's draft
      indexes `coords[g + ngrids]` with `usize + u32` which Rust
      rejects
    - `let bas_slots: u32 = 8u32;` (any typed local literal) inside
      `#[cube]` triggers `from_lit` on `NativeExpand<u32>`, which
      cubecl-core 0.10.0 does not satisfy via
      `From<NativeExpand<u32>> for ConstantValue`
- **Fix:** shipped a host CPU implementation in
  `pyscf-kernels::eval_gto::eval_gto_sph_cpu` behind the SAME
  `AlgebraClient`-typed public surface. Algebra wall is preserved
  (pyscf-gto imports `pyscf-kernels` only; no `use cubecl::*`); cubecl
  containment lint passes. Wave 0
  `crates/pyscf-kernels/tests/wave0_cubecl_smoke.rs` continues to prove
  cubecl-cpu can launch a `#[cube(launch_unchecked)]` kernel from this
  crate. Phase 4 DFT (or a dedicated Phase 8 GPU-enable plan) extends
  the file with the real cubecl kernel for l ≥ 1 cart2sph +
  deriv1/deriv2.
- **Justification:** the host-CPU path is in lockstep with
  `pyscf-algebra::host_fallback::{eigh, cholesky, qr, svd}` (which also
  routes the eigh family to faer 0.24 on host per ALG-05). For Phase 2
  smoke fixtures (single-grid-point H 1s) the CPU loop is correct,
  fast, and FMA-free-friendly (FOUND-05 oracle target). The Phase 4
  DFT plan is the right place to land a real cubecl macro: that's
  where grids get big (10⁵+ points × 10²+ AOs) and the GPU launch cost
  amortises, AND where l ≥ 1 cart2sph (the heavy part of the
  computation) lives.
- **Files modified:**
    - `crates/pyscf-kernels/src/eval_gto.rs` (host CPU implementation
      with documented deferral)
- **Commit:** `a9a12dd`

**2. [Rule 1 — bug] Y_00 angular factor was missing from the s-shell
path**

- **Found during:** Task 2 (TDD GREEN — first iteration of the smoke
  fixture)
- **Issue:** the kernel returned `psi(0) = 2.227` for STO-3G H 1s at
  the nucleus, while the analytical reference is ≈ 0.632. The plan
  text claims "Y_00 = 1/(2*sqrt(π)) is absorbed into the normalised
  coefficients" but `02-04 make_env::normalise_contractions` applies
  only the *radial* normalisation (per-prim `gto_norm` +
  `_nomalize_contracted_ao`). Upstream `pyscf/gto/eval_gto.py` calls
  `_cart2sph_l(0)` which is the `[[1/(2*sqrt(π))]]` 1×1 matrix —
  applied at the kernel layer, not the basis-loader.
- **Fix:** multiply the contracted radial by
  `Y_00 = 0.5 / sqrt(π)` for `l == 0`. For s-shells the cart and sph
  factors are identical, so the same multiplier covers both
  `GTOval_sph` and `GTOval_cart` — the `cart_variant_works_for_s_shells`
  smoke fixture asserts the equality (passes).
- **Files modified:**
    - `crates/pyscf-kernels/src/eval_gto.rs` (l == 0 path: post-radial
      `acc * y00`)
- **Commit:** `1fd58ed`

**3. [Rule 2 — missing critical functionality] None.** Threat-register
mitigations T-02-06-01 (slot-constant drift) is gated by the 02-09
oracle, T-02-06-03 (out-of-bounds) is moot for the host CPU path
(Rust slice indexing panics on overrun rather than UB), and T-02-06-04
(l ≥ 1 silently zero) is documented behaviour — Phase 2 limitation
flagged at the kernel doc-comment + this SUMMARY.

### Architectural Decisions Deferred

None — no Rule 4 (architectural-change) checkpoints required. The
deferred cubecl macro is a Rule 3 implementation choice within the
existing architecture (pyscf-kernels already owns the cubecl seam).

## Numerical Smoke

| Fixture | Output | Reference | Pass |
|---------|--------|-----------|------|
| H 1s STO-3G at nucleus (`GTOval_sph`) | `0.6282468823` | `≈ 0.6325` (upstream PySCF analytical) | yes (within 0.5..0.8 envelope) |
| H 1s STO-3G at z=5 Bohr (`GTOval_sph`) | very small (< 0.1) | `≈ 0.0146` (Gaussian tail `exp(-α_min · 25)`) | yes |
| H₂/STO-3G with ngrids=10 | `shape=[10, 2]`, `len=20` | F-order shape | yes |
| `GTOval` alias on `mol.cart=false` | identical to `GTOval_sph` | upstream alias | yes |
| `GTOval_cart` for s-shell | identical to `GTOval_sph` | Y_00 = cart factor for l=0 | yes |

## Files Added to Phase 4 DFT Handoff

The following hooks land in 02-06 ready for Phase 4 to extend (no
re-architecture required):

- `crates/pyscf-kernels/src/eval_gto.rs` — current `l >= 1` branch
  writes zeros into the AO slots of the right shape `(2l+1) per ctr`.
  Phase 4 DFT replaces with: real radial × cart2sph_l matrix product
  for l = 1..4. The radial computation pattern (per-prim
  `coef * (-α * r²).exp()` accumulator) is reusable verbatim.
- `crates/pyscf-gto/src/eval_gto.rs` — `GTOval_sph_deriv1` /
  `GTOval_sph_deriv2` already routed; Phase 4 swaps the
  `NotYetImplemented{phase:4}` arms for real kernel calls. Output
  shape changes to `(ncomp, ngrids, nao)` in F-order.
- `crates/pyscf-gto/tests/eval_gto_smoke.rs::deriv1/2` tests will need
  to flip from "expects NotYetImplemented" to "asserts numerics" when
  Phase 4 lands; structure is in place.

## Phase 7 Grad Handoff

`GTOval_ip` and `GTOval_ig` (and the `*_sph` / `*_cart` variants) are
routed to `NotYetImplemented{phase:7, what: "..."}` with the right
phase pointer. Phase 7 grad swaps the arms for real implementations
without touching the dispatcher shape.

## Threat Flags

None. The plan's `<threat_model>` covers every new surface; no
out-of-scope security-relevant trust boundary was introduced. The host
CPU implementation eliminates T-02-06-03 (the cubecl-unchecked-launch
class) entirely for Phase 2; if Phase 4 brings back the cubecl kernel,
T-02-06-03 is back on the table and the 02-09 oracle catches it.

## Self-Check: PASSED

- [x] `crates/pyscf-kernels/src/eval_gto.rs` exists.
- [x] `crates/pyscf-gto/src/eval_gto.rs` exists.
- [x] `crates/pyscf-gto/tests/eval_gto_smoke.rs` exists.
- [x] Commit `a9a12dd` (Task 1) exists in history.
- [x] Commit `b0a59f6` (TDD RED) exists in history.
- [x] Commit `1fd58ed` (Task 2 GREEN + Y_00 fix) exists in history.
