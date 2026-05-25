---
phase: 05-mp2
reviewed: 2026-05-23T00:00:00Z
depth: standard
files_reviewed: 27
files_reviewed_list:
  - .github/workflows/ci.yml
  - crates/pyscf-ao2mo/src/error.rs
  - crates/pyscf-ao2mo/src/incore.rs
  - crates/pyscf-ao2mo/src/lib.rs
  - crates/pyscf-ao2mo/src/transform.rs
  - crates/pyscf-ao2mo/tests/transform_roundtrip.rs
  - crates/pyscf-mp2/src/dfmp2.rs
  - crates/pyscf-mp2/src/dfmp2_native.rs
  - crates/pyscf-mp2/src/error.rs
  - crates/pyscf-mp2/src/frozen.rs
  - crates/pyscf-mp2/src/helpers.rs
  - crates/pyscf-mp2/src/hooks.rs
  - crates/pyscf-mp2/src/lib.rs
  - crates/pyscf-mp2/src/mp2.rs
  - crates/pyscf-mp2/src/rdm.rs
  - crates/pyscf-mp2/src/ump2.rs
  - crates/pyscf-mp2/tests/ccsd_import_contract.rs
  - crates/pyscf-mp2/tests/dfmp2_native_structural.rs
  - crates/pyscf-mp2/tests/dfmp2_structural.rs
  - crates/pyscf-mp2/tests/rmp2_structural.rs
  - crates/pyscf-mp2/tests/ump2_structural.rs
  - crates/pyscf-oracle/src/runner.rs
  - crates/pyscf-py/src/lib.rs
  - crates/pyscf-py/src/mp.rs
  - crates/pyscf-py/tests/mp2_scanner.rs
  - python/pyscf/mp/__init__.py
findings:
  critical: 1
  warning: 7
  info: 6
  total: 14
status: issues_found
---

# Phase 5: Code Review Report

**Reviewed:** 2026-05-23
**Depth:** standard
**Files Reviewed:** 27
**Status:** issues_found

## Summary

Reviewed the Phase 5 MP2 surface: the new `pyscf-ao2mo` crate (AO→MO 4-index
transform), the `pyscf-mp2` crate (RMP2 / UMP2 / conventional + native DF-MP2,
frozen-core, RDMs, helpers), the `pyscf-py` MP2 PyO3 bridge, the oracle runner
arms, the CI surface, and the always-on structural test suite.

The flat-index discipline in the quarter-transform and the closed-form RMP2
kernel is carefully documented and the always-on synthetic tests are genuinely
independent (longhand left-fold references, not self-checks). The pyo3/cubecl
wall is honoured: `pyscf-ao2mo` and `pyscf-mp2` name neither pyo3 nor cubecl in
their `Cargo.toml`, and every numeric reduction routes through
`oracle_sum`/`oracle_dot` (no bare `+=` in production compute paths — the only
`+=` accumulators are in `#[cfg(test)]` longhand references, which is correct).
The `int2e` / `int3c2e_sph` cintx#11 gates `?`-propagate correctly and are not
flagged as bugs per the phase constraints.

The headline concern is **CR-01**: the PyO3 UMP2 path constructs the αβ
opposite-spin ERI block from the α reference's *same-spin* transform. This is
labelled a "structural placeholder," but it is NOT behind any gate — once
cintx#11 lands arity-4 `int2e`, `PyUMP2.kernel` will run to completion and
return a *numerically wrong* open-shell correlation energy with no error. That
is exactly the "silently returns a wrong value" class the phase constraints ask
to distinguish from intentional gated stubs. The remaining warnings cover a
latent `naux` truncation in the native UHF cross-spin contraction, the α=β
reference clone in the UHF snapshot, and several robustness gaps.

## Critical Issues

### CR-01: PyUMP2 αβ block built from α same-spin transform — silently wrong UMP2 energy once int2e lands

**File:** `crates/pyscf-py/src/mp.rs:545-547` (also `mp.rs:802-806`)
**Issue:**
`PyUMP2::kernel` builds the three spin-block ERIs as:
```rust
let eris_aa = default_ao2mo(&refr.alpha, &frozen)?;
let eris_bb = default_ao2mo(&refr.beta, &frozen)?;
let eris_ab = default_ao2mo(&refr.alpha, &frozen)?;  // <-- αα, NOT αβ
```
`default_ao2mo` always transforms with `[&co, &cv, &co, &cv]` from a *single*
reference, so `eris_ab` is the αα `(ia|jb)` block, not the genuine cross-spin
`(i_α a_α | J_β B_β)` block that `ump2_kernel`'s `opposite_spin_channel`
consumes. The opposite-spin contraction (`ump2.rs:299`, direct, no exchange) is
the *dominant* MP2 correlation term, so feeding it the αα block yields a wrong
total `e_corr`.

Unlike `default_ao2mo`'s `int2e` gate (which returns `NotYetImplemented` and
is the deliberate cintx#11 stub), this code path has **no gate**: when cintx#11
lands arity-4 `int2e`, `default_ao2mo` returns a real αα block, `ump2_kernel`
accepts it (shapes match because nocc_a/nvir_a equal nocc_b/nvir_b for a
restricted-shaped snapshot), and `PyUMP2.kernel` returns a wrong number with no
error. The same defect is in `PyMp2Scanner::__call__` (`mp.rs:802-806`). The
inline comments acknowledge this ("structural cross-spin placeholder") but the
acknowledgement does not prevent the silent-wrong-value once the gate lifts.

**Fix:** Gate the wrong path explicitly so it cannot silently produce a number,
OR build the genuine cross-spin block. Minimal safe fix — refuse to compute
until the real cross-spin transform exists:
```rust
// UMP2 αβ requires the genuine (o_α v_α | o_β v_β) cross-spin transform,
// which needs a 2-reference ao2mo entry point. Until that exists, refuse
// rather than silently contract the αα block as if it were αβ.
return Err(pyscf_to_py(
    pyscf_mp2::Mp2Error::NotYetImplemented { plan: 4 }.into(),
));
```
The genuine fix is a cross-spin `ao2mo` (e.g. `general(eri, nao, [&co_a, &cv_a,
&co_b, &cv_b])`) once `int2e` lands; track it as a follow-on. Either way the
αα-as-αβ substitution must not reach a returned energy.

## Warnings

### WR-01: Native UHF cross-spin Q-fold uses `naux_a.min(naux_b)` with per-spin strides — truncates/mis-strides if aux differs

**File:** `crates/pyscf-mp2/src/dfmp2_native.rs:457`
**Issue:**
The opposite-spin block calls
`kab_from_slices(ints_i, ints_j, naux_a.min(naux_b), nvir_a, nvir_b)`.
Inside `kab_from_slices` (dfmp2_native.rs:255-266) the gather is
`ints_i[q * nvir_i + a]` for `q in 0..naux`, i.e. it assumes the slice is
row-major `[naux, nvir_i]` with exactly `naux` rows. `ints_i` is the per-`i`
slice of `ints_a` whose true row count is `naux_a`, and `ints_j`'s is `naux_b`.
Passing `naux_a.min(naux_b)`:
1. truncates the Q-sum when `naux_a != naux_b` (drops auxiliary functions), and
2. is inconsistent with the same-spin path which passes the full per-spin
   `naux`.
Today `df_a`/`df_b` share one `*-ri` aux so `naux_a == naux_b` and the bug is
latent, but the code does not enforce that, and `NativeDFUMP2` stores two
independent `DfIntegrals` (`df_alpha`, `df_beta`) that could legitimately
differ.
**Fix:** Validate `naux_a == naux_b` up front (return `ShapeMismatch` otherwise),
then pass the shared `naux`:
```rust
if naux_a != naux_b {
    return Err(crate::error::Mp2Error::ShapeMismatch {
        expected: naux_a, got: naux_b }.into());
}
// ...
let kab = kab_from_slices(ints_i, ints_j, naux_a, nvir_a, nvir_b);
```

### WR-02: UHF reference snapshot clones α onto β — wrong open-shell energy once int2e lands

**File:** `crates/pyscf-py/src/mp.rs:474-483`
**Issue:**
`snapshot_ump_reference` builds `beta = alpha.clone()`, so both spin channels
carry identical MO coefficients/energies/occupations even for a genuine UHF
`mf` (which exposes distinct `mo_coeff[0]`/`mo_coeff[1]`, `mo_energy[0/1]`,
`mo_occ[0/1]`). Combined with CR-01 this means a UHF MP2 would degenerate to a
restricted calculation. Like CR-01 it is masked only by the `int2e` gate, not by
an explicit guard, so it becomes a silent-wrong-value once the gate lifts.
**Fix:** Read the α/β slices from the UHF `mf` (`mo_coeff[0]`/`[1]` etc.) into
two distinct `Mp2Reference`s. Until that is wired, pair it with the CR-01 gate so
no wrong UMP2 number is returned.

### WR-03: `e_occ`/`e_vir` use `0.0` fallback for missing MO energies — masks shape bugs as silent zeros

**File:** `crates/pyscf-mp2/src/mp2.rs:212-213` (also `ump2.rs:127-128`,
`dfmp2_native.rs:151-152`)
**Issue:**
`let e = refr.mo_energy.get(col).copied().unwrap_or(0.0);` substitutes `0.0`
when `mo_energy` is shorter than the mask. A zero occupied/virtual energy
silently corrupts the `εi+εj−εa−εb` denominator (and can even make it 0 →
division producing `inf`/`NaN` that propagates into the energy) instead of
surfacing the inconsistency. The kernel already validates
`e_occ.len() == nocc && e_vir.len() == nvir` *after* the loop, but the
`unwrap_or(0.0)` can make the lengths line up while the values are wrong.
**Fix:** Validate `refr.mo_energy.len() == refr.mo_occ.len()` (== mask length)
at entry and index directly, or return `ShapeMismatch` from inside the loop when
`get(col)` is `None`, rather than fabricating a `0.0` energy.

### WR-04: `mo_subset`/`mo_without_core` infer `nmo` by integer division — a non-multiple length is silently rounded

**File:** `crates/pyscf-mp2/src/mp2.rs:108`, `crates/pyscf-mp2/src/dfmp2.rs:106`,
`crates/pyscf-mp2/src/dfmp2_native.rs:127`, `crates/pyscf-mp2/src/helpers.rs:121`
**Issue:**
`let n_selected = data.len().checked_div(nao).unwrap_or(0);` (and the
`data.len() / nao.max(1)` variant in helpers) computes the column count by
truncating integer division. Because `data` is grown by `extend_from_slice` of
`nao`-length slices the remainder should always be 0, but if a prior shape bug
ever pushes a partial column the truncation silently discards it and reports a
plausible-but-wrong `nmo`, propagating a quietly-misshaped `MOCoefficients`
downstream. `nao` can also be `0` (the `checked_div`/`max(1)` guards avoid a
panic but then report `nmo = 0`, hiding the real cause).
**Fix:** Assert `data.len() % nao == 0` (and `nao > 0`) and return
`ShapeMismatch` otherwise, so a partial column is a hard error, not a silent
round-down.

### WR-05: `reference_elements` clamps negative atomic charges to `0` then casts — silently mis-resolves frozen core

**File:** `crates/pyscf-mp2/src/mp2.rs:59-65` (duplicated in `dfmp2.rs:63-69`,
`dfmp2_native.rs:85-91`, `ump2.rs:114-119`)
**Issue:**
`.map(|z| z.max(0) as u32)` turns any negative `atom_charges()` entry into `0`
(treated as a ghost atom, `chemcore_count` 0). A negative charge here would
indicate upstream data corruption, but the clamp hides it and `Frozen::Auto`
then silently freezes the wrong number of orbitals. The four copies also drift
risk (see IN-01).
**Fix:** If a negative charge is genuinely impossible, return an error on
`z < 0` rather than clamping; if it is possible (anion partial charges should
not appear in `atom_charges`, which is nuclear `Z`), document why `0` is
correct. At minimum, deduplicate into one helper so the policy is single-sourced.

### WR-06: `oracle_dot` returns `NaN` on length mismatch — DF/native paths rely on un-asserted equal lengths

**File:** `crates/pyscf-mp2/src/dfmp2.rs:246`, `crates/pyscf-mp2/src/dfmp2_native.rs:263,345,359,464`
**Issue:**
`pyscf_algebra::oracle_dot` returns `f64::NAN` (not an error) when its two
slices differ in length (oracle.rs:31-33). The DF assembly relies on
`b_ia`/`b_jb` both being length `naux` and `tab`/`kab` both being `nvir*nvir`;
these are true by construction here, but a NaN from a future stride bug would
flow into the summed energy and produce a `NaN` correlation energy rather than a
diagnosable error. This is a latent robustness gap, not a present miscompute.
**Fix:** Add `debug_assert_eq!` on the slice lengths at each `oracle_dot` call
site (cheap, documents the invariant, and turns a silent NaN into a test-time
panic), or check-and-`ShapeMismatch` on the hot-path entries.

### WR-07: `make_rdm2` dm1 contribution iterates full `nmo0^2` per `i0` with no early exit — and assumes `oidx`/`vidx` partition is exhaustive

**File:** `crates/pyscf-mp2/src/rdm.rs:416-426`
**Issue:**
Step 3's triple loop `for i0 in 0..nocc0 { for p in 0..nmo0 { for q in 0..nmo0 }}`
writes `dm2[idx4(...)]` using `d1(q,p)`/`d1(p,q)` over the full `nmo0` range. The
correctness depends on `dm1` being fully populated over `[nmo0,nmo0]`, which
`make_rdm1(..., with_frozen=true)` does provide. However, this block is only
unit-tested for the no-frozen `nmo0 == nact` case
(`make_rdm2_known_subblock_placement`, `make_rdm2_length_is_nmo4_no_frozen`);
the *frozen* embedding path (where `oidx`/`vidx` are non-contiguous and `nmo0 >
nact`) has no numeric test asserting the placement, despite being the more
error-prone fancy-indexing case the module doc itself flags (T-05-04-LAYOUT,
"a silent transpose corrupts the RDM").
**Fix:** Add a frozen-path `make_rdm2` test (analogous to
`make_rdm1_with_frozen_places_core_diagonal`) that asserts a known sub-block
placement and the separable-HF diagonal under e.g. `Frozen::Count(1)`, so the
non-contiguous `oidx`/`vidx` scatter is exercised.

## Info

### IN-01: `reference_elements` and `mo_subset` duplicated verbatim across four modules

**File:** `crates/pyscf-mp2/src/mp2.rs:59-116`, `dfmp2.rs:63-114`,
`dfmp2_native.rs:85-135`, `ump2.rs:114-119`
**Issue:** `reference_elements` is copy-pasted in four files and `mo_subset` in
three, each with the doc note "kept local so the path has no cross-module
private dependency." This is deliberate but invites drift (e.g. mp2.rs's
`mo_subset` populates `energies`/`occupations`, while the dfmp2/native copies
leave them empty — a real behavioural divergence between copies).
**Fix:** Promote a single `pub(crate) fn mo_subset`/`reference_elements` (e.g. in
a `mp2::common` module) and call it from all paths; document the
energies/occupations policy once.

### IN-02: Hard-coded SCS factors `ps = pt = 1.0` as local magic numbers in native path

**File:** `crates/pyscf-mp2/src/dfmp2_native.rs:299-300, 386-387`
**Issue:** `let ps = 1.0_f64; let pt = 1.0_f64;` appear as bare literals in two
functions with a paragraph of comment each explaining they are the plain-MP2
default. The conventional path exposes SCS via `scs_energy`; the native path
fixes them.
**Fix:** Either lift to a named `const PLAIN_MP2_SCS: (f64, f64) = (1.0, 1.0);`
or thread `ps`/`pt` parameters through `emp2_rhf`/`emp2_uhf` so SCS-DF-RMP2 is
reachable without editing the literals.

### IN-03: `check_block`'s `label` parameter only feeds a `debug_assert!(!label.is_empty())`

**File:** `crates/pyscf-mp2/src/ump2.rs:139-172`
**Issue:** The `label: &str` argument is taken solely to satisfy
`debug_assert!(!label.is_empty())` (line 171); it never appears in the returned
`ShapeMismatch`, so a mismatch reports `expected/got` without saying which
spin-block (aa/ab/bb) failed.
**Fix:** Drop the unused `label` (and its `debug_assert`), or actually surface it
in the error so the failing channel is identifiable.

### IN-04: `mo2.rs` per-`i` exchange reorder reads `eris.ovov[idx(i,b,j,a)]` — correct but undocumented symmetry assumption

**File:** `crates/pyscf-mp2/src/mp2.rs:255-257`
**Issue:** `g_jba[p] = eris.ovov[idx(i, b, j, a)];` builds the `(ib|ja)`
exchange block by swapping `a<->b`. This is correct for the Chemist's `(ia|jb)`
layout the kernel assumes, but the comment only says "swap the virtual roles
a<->b in the i/j block" without stating it relies on the supplied `ovov` being a
genuine `(occ,vir|occ,vir)` block (an arbitrary user-overridden `ao2mo` returning
a non-(ia|jb) layout would make this read the wrong element).
**Fix:** Note in the doc that the kernel assumes `ovov` is the Chemist's
`(ia|jb)` block in the documented C-order; the override contract should restate
this for `Mp2OverrideHooks::ao2mo` implementers.

### IN-05: PyO3 override `ao2mo` infers `nvir` via `sqrt(len/(nocc*nocc))` — fragile shape recovery

**File:** `crates/pyscf-py/src/mp.rs:197-198`
**Issue:**
```rust
let nvir_sq = ovov.len() / (nocc * nocc).max(1);
let nvir = (nvir_sq as f64).sqrt() as usize;
```
recovers `nvir` from a flat array length via a float `sqrt` cast. If a Python
override returns a block whose length is not exactly `nocc*nocc*nvir*nvir`, this
silently yields a wrong `nvir` (floor of a non-integer sqrt) rather than
erroring; the downstream `rmp2_kernel` shape check will then fire with a
confusing `expected/got`, masking the real cause.
**Fix:** Validate `ovov.len() == nocc*nocc*nvir*nvir` after recovering `nvir`
(or have the override contract return `nvir` explicitly) and raise a clear
`PyValueError` on mismatch.

### IN-06: `python313t-smoke` CI job uses `sudo add-apt-repository`/`apt-get` with no failure trap on the PPA step

**File:** `.github/workflows/ci.yml:282-287`
**Issue:** The 3.13t install chains `add-apt-repository` → `apt-get update` →
`apt-get install ... || (uv fallback)`. Only the final `install` has a fallback;
a failure of `add-apt-repository` or `apt-get update` (e.g. PPA outage) aborts
the job before the `uv` fallback can run, making this job flaky on infra
hiccups. This is CI robustness, not a source defect.
**Fix:** Move the `|| uv` fallback to wrap the whole PPA+install sequence, or
prefer the `uv python install 3.13t` path first with apt as the fallback.

---

_Reviewed: 2026-05-23_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
