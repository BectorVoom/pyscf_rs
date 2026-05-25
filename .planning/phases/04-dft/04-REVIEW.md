---
phase: 04-dft
reviewed: 2026-05-22T00:00:00Z
depth: standard
files_reviewed: 59
files_reviewed_list:
  - .github/workflows/ci.yml
  - .github/workflows/nightly-cross-crate.yml
  - crates/pyscf-dft/Cargo.toml
  - crates/pyscf-dft/src/chkfile.rs
  - crates/pyscf-dft/src/df_dft.rs
  - crates/pyscf-dft/src/error.rs
  - crates/pyscf-dft/src/hooks.rs
  - crates/pyscf-dft/src/lib.rs
  - crates/pyscf-dft/src/numint.rs
  - crates/pyscf-dft/src/parser/libxc.rs
  - crates/pyscf-dft/src/parser/mod.rs
  - crates/pyscf-dft/src/parser/xcfun.rs
  - crates/pyscf-dft/src/rks.rs
  - crates/pyscf-dft/src/uks.rs
  - crates/pyscf-dft/src/veff.rs
  - crates/pyscf-dft/src/vv10.rs
  - crates/pyscf-dft/src/xc_backend.rs
  - crates/pyscf-dft/tests/cam_b3lyp_h2o_rsh.rs
  - crates/pyscf-dft/tests/df_dft_match.rs
  - crates/pyscf-dft/tests/dtype_f32_smoke.rs
  - crates/pyscf-dft/tests/ks_chkfile_roundtrip.rs
  - crates/pyscf-dft/tests/numint_signatures.rs
  - crates/pyscf-dft/tests/parse_xc_parity.rs
  - crates/pyscf-dft/tests/rks_uks_bitexact.rs
  - crates/pyscf-dft/tests/vv10_energy_match.rs
  - crates/pyscf-dft/tests/wgpu_f64_fallback.rs
  - crates/pyscf-grids/Cargo.toml
  - crates/pyscf-grids/src/lebedev.rs
  - crates/pyscf-grids/src/levels.rs
  - crates/pyscf-grids/src/lib.rs
  - crates/pyscf-grids/src/partition.rs
  - crates/pyscf-grids/src/prune.rs
  - crates/pyscf-grids/src/radial.rs
  - crates/pyscf-grids/src/radii.rs
  - crates/pyscf-grids/tests/grid_weights_level_sweep.rs
  - crates/pyscf-gto/src/eval_gto.rs
  - crates/pyscf-gto/src/lib.rs
  - crates/pyscf-gto/src/range_coulomb.rs
  - crates/pyscf-gto/tests/eval_gto_deriv1_oracle.rs
  - crates/pyscf-gto/tests/eval_gto_smoke.rs
  - crates/pyscf-gto/tests/range_coulomb_env.rs
  - crates/pyscf-kernels/src/eval_gto.rs
  - crates/pyscf-kernels/src/lib.rs
  - crates/pyscf-kernels/tests/eval_gto_lge1.rs
  - crates/pyscf-oracle/Cargo.toml
  - crates/pyscf-oracle/src/fixtures.rs
  - crates/pyscf-oracle/src/runner.rs
  - crates/pyscf-py/src/bridge.rs
  - crates/pyscf-py/src/dft.rs
  - crates/pyscf-py/src/lib.rs
  - crates/pyscf-py/src/scf.rs
  - crates/pyscf-scf/src/convert.rs
  - crates/pyscf-scf/src/lib.rs
  - docs/env-vars.md
  - python/pyscf/__init__.py
  - python/pyscf/dft/__init__.py
  - python/tests/test_dft_override.py
  - tests/oracle/test_eval_gto.py
findings:
  critical: 4
  warning: 9
  info: 6
  total: 19
status: issues_found
---

# Phase 4: Code Review Report

**Reviewed:** 2026-05-22
**Depth:** standard
**Files Reviewed:** 59
**Status:** issues_found

## Summary

Phase 4 ports PySCF's DFT stack (Becke grids, XC-string parsers, the `NumInt`
grid loop, RKS/UKS drivers, RSH, VV10, DF-DFT, KS chkfile) plus the PyO3
boundary. The structural scaffolding is careful and the bit-exact reduction
discipline (`oracle_sum`/`oracle_dot`) is consistently applied in the grid
loop, VV10, and Becke partition. The libxc backend is correctly cfg-gated, the
PyO3 subclass dispatch correctly routes through `call_method1` (MRO), and the
RSH env[8] guard is RAII-correct.

However, the review surfaces **correctness defects that break the open-shell
(UKS) path entirely** and several robustness/consistency problems. The most
serious: `NumInt::nr_uks` does not compute a genuine spin-resolved exchange-
correlation potential — it returns the closed-shell Vxc cloned into both spin
channels and the closed-shell Exc over the total density, which is physically
wrong for any open-shell system (the entire reason UKS exists). Two additional
BLOCKERs concern the `unwrap_or(0.0)` / `unwrap_or_else(S::zero)` swallowing of
scalar-conversion failures in the f32 path (silently corrupting results to
zero) and a `panic!` reachable from the untrusted-basis path in the cart→sph
kernel that violates the stated never-panic policy. A `1e-12` density-matrix
"fingerprint" used as a cache key is also a correctness hazard (collisions →
stale-energy reuse).

Note the project's hard constraint: the libxc-feature path was reviewed as
source only (never compiled). Findings in `parser/libxc.rs` and the
`#[cfg(feature="libxc")]` block of `xc_backend.rs` are static-analysis only.

## Critical Issues

### CR-01: `nr_uks` returns a fake spin-resolved Vxc/Exc (open-shell DFT is wrong)

**File:** `crates/pyscf-dft/src/uks.rs` (driver), `crates/pyscf-dft/src/numint.rs:537-583`
**Issue:** `NumInt::nr_uks` builds the *total* density `D = Dα + Dβ`, runs the
**closed-shell** `nr_rks` over it, then returns:

```rust
Ok(NrUksResult {
    nelec: (nelec_a, nelec_b),
    excsum: r.excsum,                 // closed-shell Exc over TOTAL density
    vmat: (r.vmat.clone(), r.vmat),   // SAME matrix in both spin channels
})
```

For an open-shell system (the only case UKS is used for) the XC potential is
spin-dependent: `Vxc^α = ∂Exc/∂Dα ≠ Vxc^β = ∂Exc/∂Dβ` whenever `Dα ≠ Dβ`, and
`Exc` is a functional of `(ρα, ρβ)` separately, not of `ρα+ρβ` evaluated as a
closed shell. The `xcfun` spin-split in `xc_backend.rs` further hard-codes
`rho_a = rho_b = rho/2`, so even the energy is the *spin-restricted* value of
the total density — it cannot represent spin polarization. This silently
produces incorrect total energies, incorrect Vα/Vβ, and (since Vα == Vβ is fed
back into the SCF) collapses the open-shell solution toward the restricted one.
The `vmat: (r.vmat.clone(), r.vmat)` line also clones once and moves once, so
the two channels are not even independent objects.

The module doc admits this ("v1 builds the spin-resolved Vxc by running the
closed-shell-shaped contraction over the total density … the closed-shell rho/2
split symmetrizes α/β"), but that approximation is not UKS — it is RKS wearing a
UKS signature. Worse, nothing routes `nr_uks` into the actual UKS SCF: `UKS::kernel`
(uks.rs:135) uses the *same* `crate::hooks::KsHooks` as RKS, whose `get_veff`
calls `default_get_veff` → `nr_rks` (closed-shell) on a single `dm`. So the
open-shell branch (`nr_uks`) is dead code in the driver, and `UKS::kernel`
actually runs a restricted Vxc.

**Fix:** Implement a genuine open-shell grid loop: evaluate ρα and ρβ
separately, build the spin-polarized `RhoBlock` (the xcfun `A_B…` vars already
support distinct α/β inputs — feed `rho_a = ρα`, `rho_b = ρβ`, not `rho/2`),
take `eval_xc` at `Vxc` order, and back-contract the two channels with their
own `vrho_a` / `vrho_b` (and the αα/αβ/ββ sigma derivatives for GGA). Wire the
UKS SCF to a hooks impl that calls `nr_uks` with `(Dα, Dβ)` rather than
reusing the RKS `KsHooks`. Until this lands, the UKS surface should return
`NotYetImplemented` rather than a plausible-but-wrong number.

### CR-02: f32 scalar-conversion failures are silently swallowed to 0.0

**File:** `crates/pyscf-dft/src/numint.rs:445-465, 506-513`
**Issue:** The f32 matmul chain converts every f64 value through
`S::from(x).unwrap_or_else(S::zero)` and converts back with
`t.to_f64().unwrap_or(0.0)`:

```rust
let phi_mu = S::from(ao_at(&ao, 0, g, mu)).unwrap_or_else(S::zero);
...
let wv = S::from(w * 0.5 * xc_out.vrho[g]).unwrap_or_else(S::zero);
...
vrow_terms[g] = t.to_f64().unwrap_or(0.0);
```

`f32::from(f64)` via the `num_traits::NumCast`/`FromPrimitive` path returns
`None` when the f64 magnitude exceeds `f32::MAX` (~3.4e38) — which is exactly
what happens for an overflowing intermediate. Instead of producing `inf`
(loud, debuggable) or erroring, the code substitutes `0.0`, which silently
corrupts the contraction with a wrong-but-finite value. A single
out-of-range AO/weight product drops a grid-point contribution to zero with no
warning, no error, and no `inf`/`nan` to flag it downstream. This defeats the
whole point of the "honest f32 escape hatch" — the result is not just
below-bit-exact, it can be silently and arbitrarily wrong.

**Fix:** Do not paper over conversion failure. Either propagate an error
(`S::from(x).ok_or(...)?`) or, since f32 cannot represent `±inf` only by
clamping, let the conversion produce `inf`/`nan` so the corruption is visible.
At minimum: `S::from(x).unwrap_or_else(|| S::from(x.signum() * f64::MAX).unwrap())`
is still wrong — the correct behavior is to surface a `PyscfRsError` on
non-finite/out-of-range conversion. Replace every `unwrap_or(0.0)` /
`unwrap_or_else(S::zero)` in the numeric chain with an explicit error path.

### CR-03: reachable `panic!` in the cart→sph kernel violates the never-panic policy on untrusted basis input

**File:** `crates/pyscf-kernels/src/eval_gto.rs:192-196`
**Issue:** `c2s_coeff` panics for any angular momentum `l > 4`:

```rust
_ => panic!(
    "eval_gto: cart→sph transform only supports l<=4 (g shells); got l={l}. ..."
),
```

The basis set is user-controlled (`dft.RKS(mol, ...)` with an arbitrary
`basis=`). A basis containing an h-shell or higher (l ≥ 5) — entirely valid
input, e.g. cc-pV5Z, ANO bases, or a hand-written GTO file — reaches this path
through `eval_gto_sph` → `nr_rks` and crashes the process (and, through the
PyO3 boundary, the host Python interpreter) instead of returning a clean error.
`cart_powers` and the `mono`/`cart_vals` buffers are sized for the real `l`, so
the panic is the only guard. The crate-level policy (and the DftError
doc-comment, error.rs:14-17) is "never a panic" for user input. The `nr_uks`
self-coupled VV10 path and the deriv1 path share the same `c2s_coeff` call
(`eval_gto.rs:599`) so both are affected.

**Fix:** Return `Result<EvalGtoBuffers, _>` (or have the kernel surface a
sentinel the wrapper converts to `PyscfRsError::NotYetImplemented{phase:4}` /
an "unsupported angular momentum" error) instead of `panic!`. The
`pyscf-gto::eval_gto` wrapper already returns `Result`, so threading the error
up is mechanical.

### CR-04: density-matrix cache key uses an absolute `1e-12` tolerance on `Σ|D|` — collision risk reuses stale energies

**File:** `crates/pyscf-dft/src/hooks.rs:238-241, 307-323`; `crates/pyscf-dft/src/df_dft.rs:184-187, 247-262`
**Issue:** The per-cycle KS-energy cache is keyed on a "fingerprint" that is the
`oracle_sum` of `|D|`, compared with an absolute tolerance:

```rust
Some(c) if (c.dm_fingerprint - dm_fingerprint(dm)).abs() < 1e-12 => {
    (c.exc, c.half_tr_d_vxc)
}
```

Two problems compound:
1. **`Σ|D|` is not injective.** Different density matrices routinely have the
   same (or near-identical) `Σ|D|` — `Σ|D|` is conserved-ish across SCF cycles
   near convergence and is trivially equal for any pair of DMs that redistribute
   weight without changing the L1 norm. A collision returns the *previous*
   cycle's `(Exc, ½Tr(D·Vxc))` for a *different* `dm`, injecting an inconsistent
   XC energy into `energy_elec` — a silent correctness bug that is hardest to
   catch precisely near convergence where energies are compared at µHartree.
2. **The absolute `1e-12` tolerance does not scale with system size.** `Σ|D|`
   grows with the number of electrons/AOs; for a large molecule the SCF step-to-
   step change in `Σ|D|` near convergence can itself be < 1e-12 in *relative*
   terms while the DM (and thus Exc) has genuinely changed, again returning a
   stale energy. Conversely it can never match for tiny systems, defeating the
   cache (a performance, not correctness, issue).

The comment claims "a mismatch just triggers a fresh grid-loop recompute" — but
the danger is a *false match*, which does the opposite (reuses stale data).

**Fix:** Do not use a lossy scalar fingerprint as a correctness-bearing cache
key. Either (a) cache by identity of the actual `dm` passed in the same SCF
iteration (the kernel calls `get_veff(dm)` then `energy_elec(dm, …)` back-to-
back — pass the bundle directly rather than re-deriving via a fingerprint), or
(b) on any fingerprint mismatch OR ambiguity, always recompute. If a fingerprint
is kept, hash the full `dm.data` (e.g. a content hash), not its L1 norm, and
treat only an exact hash match as a hit.

## Warnings

### WR-01: `nr_rks` recomputes the entire grid loop in `energy_elec` cache-miss path — and the GGA Vxc back-contraction is O(nao²·ngrids) computed twice

**File:** `crates/pyscf-dft/src/hooks.rs:313-322`; `crates/pyscf-dft/src/numint.rs:440-472`
**Issue:** On a cache miss `energy_elec` calls `self.ks_veff(self.mol, dm)?`
which re-runs `default_get_veff` → full `nr_rks` (AO eval + ρ contraction + XC
eval + Vxc back-contraction) purely to recover two scalars (`Exc`,
`½Tr(D·Vxc)`). Given CR-04, cache misses can be frequent. Separately, the Vxc
back-contraction loop (numint.rs:440-471) iterates the full `nao × nao` square
and writes both `[mu,nu]` and `[nu,mu]`, so every off-diagonal element is
computed twice (the result is numerically correct — the 0.5 factor compensates
— but it is 2× the necessary work in the hottest loop). Performance is out of
v1 scope, but the *correctness coupling* (energy depends on a possibly-stale
cache) makes this worth fixing alongside CR-04.

**Fix:** Have `get_veff` stash the full `KsVeff` bundle (not just the
fingerprint) keyed to the iteration, and have `energy_elec` read it directly.
Restrict the back-contraction inner loop to `nu <= mu` and write the transpose
once.

### WR-02: `dft_err` mislabels every grid/XC failure as `InvalidMolecule`

**File:** `crates/pyscf-dft/src/numint.rs:618-620`; `crates/pyscf-dft/src/vv10.rs:326-328`
**Issue:** Both `dft_err` helpers wrap arbitrary grid-loop failures (grids not
built, missing ∇ρ for GGA) as `CoreError::InvalidMolecule(msg)`. A "grids not
built" or "GGA ∇ρ missing" condition is not a malformed molecule — it is an
internal precondition violation or a programming error. This produces
misleading diagnostics ("InvalidMolecule: nr_rks: grids not built") that will
send users debugging their geometry instead of their grid build call.

**Fix:** Add a dedicated `DftError` variant (e.g. `GridNotBuilt`,
`InternalInvariant`) and surface it, or at least use a more accurate
`CoreError` variant. Do not overload `InvalidMolecule`.

### WR-03: `format_xc_code` indexes `frags[0]` after `split("RSH")` without an emptiness guard

**File:** `crates/pyscf-dft/src/parser/xcfun.rs:114-115`; `crates/pyscf-dft/src/parser/libxc.rs:177-178`
**Issue:** `let frags: Vec<&str> = cleaned.split("RSH").collect();` then
`String::from(frags[0])`. `str::split` always yields at least one element so
`frags[0]` cannot panic *today*, but the slice `&frag[..close]` /
`&rsh_key[1.min(rsh_key.len())..]` / `&frag[close + 1..]` byte-indexing into
arbitrary user input is fragile: `cleaned` has been uppercased but `find(')')`
returns a byte offset, and a multibyte UTF-8 char before the `)` (e.g. a
user pasting `RSH(α;…)`) would make `&frag[..close]` panic on a non-char-boundary
slice. The XC string is explicitly untrusted (T-04-05a). The whitespace filter
and uppercase do not remove non-ASCII letters.

**Fix:** Operate on `char_indices` / validate ASCII before byte-slicing, or use
`get(..close)` and handle `None` as a `MalformedToken` error. Apply to both
parser copies.

### WR-04: `xc_type_of` classifies GGA purely by a hardcoded id allowlist — MGGA and unknown ids silently fall through to LDA

**File:** `crates/pyscf-dft/src/numint.rs:55, 596-606`
**Issue:** `XCFUN_GGA_IDS` is a hand-maintained `&[u32]` of 10 ids; any
functional whose component id is not in that list is classified `XcType::Lda`,
which selects `GTOval_sph` (value only, no gradient). If a GGA/MGGA functional
slips in with an id not on the list (e.g. a raw-integer XC string `"5"`-style
input that maps to a different id, or a future table addition), `nr_rks`
evaluates it as LDA — feeding a gradient-dependent functional an LDA RhoBlock.
That then trips the `xc_backend.rs:374` family-mismatch check and errors — but
only by luck; an id that *is* GGA-family in xcfun yet absent from
`XCFUN_GGA_IDS` produces a hard error rather than the correct GGA path. MGGA is
unrepresented entirely (`XcType` has no `Mgga`), so any MGGA component is
silently treated as LDA.

**Fix:** Derive the family from the backend's own `func.is_gga()` /
`func.is_metagga()` (already used in `xcfun_eval`) rather than a parallel
hardcoded id list that can drift out of sync with the backend. Single source of
truth.

### WR-05: `RKS::kernel` (and `UKS::kernel`) ignore the result of `grids.build`

**File:** `crates/pyscf-dft/src/rks.rs:167`; `crates/pyscf-dft/src/uks.rs:136`; `crates/pyscf-py/src/dft.rs:207, 265, 524, 566`
**Issue:** `let _ = self.grids.build(&self.mol);` discards the `(coords,
weights)` return and any implicit failure. `Grids::build` cannot currently
return `Err` (it returns the tuple), so a degenerate molecule (zero atoms,
NaN coords) produces empty/garbage grids that surface much later as a confusing
"coords missing" or a silently-zero integral. The `let _ =` also hides that
`build` mutates `self.grids` for its side effect — a reader cannot tell the
return is intentionally dropped vs. an unhandled `Result`.

**Fix:** Make `Grids::build` return `Result` and propagate, or at minimum assert
non-empty grids before entering the SCF cycle so a bad geometry fails loudly at
build time rather than mid-SCF.

### WR-06: DF-DFT `get_jk` always builds the standard (expensive) K even for pure functionals

**File:** `crates/pyscf-dft/src/df_dft.rs:208-212`
**Issue:** `DfKsHooks::get_jk` unconditionally calls
`pyscf_scf::default_get_jk(mol, dm)` to obtain `k_std`, even when the functional
is a pure (non-hybrid) GGA/LDA where `hyb == 0` and K is never used in
`get_veff_ks` (`J + Vxc − hyb·K`). The code comment even acknowledges "for a
pure functional the `hyb·K` term vanishes … the standard K is never numerically
used" — yet it is still built every cycle, defeating the entire purpose of
density-fitting (avoiding the O(N⁴) exchange build). The standard K build is
also the `int2e_sph` arity-4 path that is `NotYetImplemented{phase:2}`, so for
pure functionals DF-DFT will *error out* on a call it did not need to make.

**Fix:** Gate the standard-K build on `hyb != 0` (parse `rsh_coeff` once, or
thread the hybrid coefficient into `DfKsHooks`). Return a zero K matrix when the
functional is pure.

### WR-07: env-vars doc / Python getter name mismatch (`mf.precision` vs `mf.dtype`)

**File:** `docs/env-vars.md:42-47`; `crates/pyscf-py/src/dft.rs:131-134`; `python/tests/test_dft_override.py:108-120`
**Issue:** `docs/env-vars.md` documents "the read-only `mf.precision` getter on
the Python `RKS`/`UKS` objects … returning the string `"f32"` / `"f64"`". The
actual implementation exposes the getter as `dtype` (`PyRKS::dtype`,
`PyUKS::dtype`), and the test asserts `mf.dtype == "f64"`. A user following the
documented surface (`mf.precision`) gets `AttributeError`. The single-source-of-
truth doc contradicts the shipped API.

**Fix:** Rename the doc reference to `mf.dtype` (or add a `precision` alias
getter if `precision` is the intended public name).

### WR-08: `parse_paren_omega` finds `(` and `)` independently — a stray `)` before `(` mis-slices

**File:** `crates/pyscf-dft/src/parser/xcfun.rs:188-208`; `crates/pyscf-dft/src/parser/libxc.rs:446-466`
**Issue:** `parse_paren_omega` does `key.find('(')` then `key[open..].find(')')`
and slices `&key[open + 1..open + close]`. The `close` offset is relative to
`open` (because of `key[open..].find`), so `key[open + 1 .. open + close]` is
correct *only* when `close` is measured from `open`. It is — `find` on the
`key[open..]` subslice returns an offset relative to `open`, and the slice adds
`open` back. This is correct. However, the subslice `&key[open+1..open+close]`
will be an empty/`""` string for `LR_HF()` (no digits), which `parse::<f64>()`
rejects as `MalformedToken` — acceptable. The real fragility: `key.contains("SR_HF")`
(line 289) is checked before the integer-id branch, so a raw functional id
string that happens to contain the substring `SR_HF` would be misrouted. Low
likelihood but the substring `contains` checks (vs. exact prefix match) are
looser than upstream's token equality.

**Fix:** Match the SR_HF/LR_HF token by prefix (`key.starts_with("SR_HF")`)
rather than `contains`, matching upstream's token-keyed dispatch.

### WR-09: `chkfile` load trusts `grids_level` f64→usize cast without range/finiteness check

**File:** `crates/pyscf-dft/src/chkfile.rs:195`
**Issue:** `let level = primitives::read_scalar_f64(scf_group, "grids_level")? as usize;`
reads an untrusted checkpoint value (T-04-08 tampering boundary is explicitly in
scope per the module doc) and does a saturating/UB-adjacent `f64 as usize` cast.
A tampered chkfile with `grids_level = -1.0`, `1e30`, or `NaN` produces a
nonsense level (`f64::NAN as usize == 0`, `-1.0 as usize == 0` in Rust's
saturating cast, `1e30 as usize == usize::MAX`) that later drives grid
allocation. While Rust's `as` cast is defined (saturating, not UB), a
`usize::MAX` level passed to grid sizing is a denial-of-service / OOM vector.

**Fix:** Validate `grids_level` is finite and within `0..=9` (the documented
range) on load, returning `ChkfileError` otherwise, consistent with the stated
never-panic / validate-before-allocate discipline.

## Info

### IN-01: `_markers` / `_py_mol_use_marker` dead-code functions ship in the binary

**File:** `crates/pyscf-py/src/dft.rs:705-716`; `crates/pyscf-py/src/scf.rs:686-701`
**Issue:** `#[allow(dead_code)] fn _markers(...)` and `_py_mol_use_marker`
construct throwaway values purely to silence unused-warnings. These are code
smells — they obscure genuinely-unused fields and add noise. The comment admits
"Clippy/rustc occasionally miss the cfg-flagged usage," which suggests the real
fix is correct `#[allow]` placement on the field, not a marker function.
**Fix:** Remove the marker fns; apply targeted `#[allow(dead_code)]` to the
specific field if needed.

### IN-02: `eval_rho` is an inherent `pub fn` taking no `&self` but lives on `NumInt`

**File:** `crates/pyscf-dft/src/numint.rs:190-228`
**Issue:** `NumInt::eval_rho` is an associated function (no `self`), yet it is a
public method on `NumInt`. Callers write `NumInt::eval_rho(...)`. This is
harmless but inconsistent with the upstream `ni.eval_rho` instance-method
surface and with `eval_xc` (which does take `&self`). Minor API-shape drift.
**Fix:** Consider a free function or document why it is associated.

### IN-03: magic constants in VV10 lack named provenance

**File:** `crates/pyscf-dft/src/vv10.rs:131-135`
**Issue:** `pi43 = 4π/3`, `kvv = bvv·1.5·π·(9π)^(−1/6)`, `beta = (3/bvv²)^0.75/32`
are inlined with only line-number comments to numint.py. The `1.5`, `(9π)^(−1/6)`,
`/32` factors are unexplained magic. Correct per the cited source, but a future
maintainer cannot verify without the upstream file.
**Fix:** Add the symbolic derivation (Vydrov-Van Voorhis constants) in a comment.

### IN-04: `XcType` enum has only `Lda`/`Gga` but docs/structures reference MGGA throughout

**File:** `crates/pyscf-dft/src/numint.rs:60-77`; `crates/pyscf-dft/src/xc_backend.rs:67-74`
**Issue:** `Family` (xc_backend) has `Mgga` and `RhoBlock` has an `Mgga` variant,
but `XcType` (numint) stops at `Gga`. The two type hierarchies are not aligned;
an MGGA functional cannot be expressed by `XcType` so the grid loop cannot
request the `[10, ngrids, nao]` deriv2/tau AO block. This is consistent with the
"MGGA deferred" decision but the half-present `Mgga` machinery (RhoBlock::Mgga,
Family::Mgga) is dead until `XcType::Mgga` exists.
**Fix:** Either remove the unreachable MGGA variants for v1 or add `XcType::Mgga`
and gate it behind a clear `NotYetImplemented`.

### IN-05: `define_xc_` adopts the description string verbatim as `self.xc` even when only the hyb is returned

**File:** `crates/pyscf-py/src/dft.rs:287-296, 585-593`
**Issue:** `define_xc_` sets `self.inner.xc = desc;` after parsing, returning only
`spec.hyb().0`. If the description is a recombination form like `"0.5*B3LYP +
0.5*PBE"`, storing the raw string as `xc` means the *next* `nr_rks` re-parses the
same compound string — fine — but the Python `mf.xc` getter now returns the
recombination expression, not a canonical functional name, which may surprise
callers expecting `define_xc_` to register a resolved functional.
**Fix:** Document that `define_xc_` stores the raw description; or store a
canonicalized form.

### IN-06: `nr_uks` ignores `relativity`/`hermi`/`max_memory`/`verbose` but `nr_rks` logs `max_memory`

**File:** `crates/pyscf-dft/src/numint.rs:537-559`
**Issue:** `nr_uks` forwards `relativity`, `hermi`, `max_memory`, `verbose` to its
internal `nr_rks` call but performs no UKS-specific logging, while `nr_rks` emits
a structured `tracing::info!` entry line. Observability is asymmetric between the
restricted and unrestricted paths (compounding CR-01's silent-wrongness).
**Fix:** Add the equivalent entry-log to `nr_uks` once CR-01 is addressed.

---

_Reviewed: 2026-05-22_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
