---
phase: 03-scf-pyo3-bindings
reviewed: 2026-05-24T00:00:00Z
depth: standard
files_reviewed: 7
files_reviewed_list:
  - crates/pyscf-scf/src/orth.rs
  - crates/pyscf-scf/src/atom_hf.rs
  - crates/pyscf-scf/src/analyze.rs
  - crates/pyscf-scf/src/init_guess.rs
  - crates/pyscf-scf/src/lib.rs
  - crates/pyscf-scf/tests/mulliken_meta.rs
  - crates/pyscf-scf/tests/init_guess_atom_huckel.rs
findings:
  critical: 0
  warning: 6
  info: 5
  total: 11
status: issues_found
---

# Phase 03: Code Review Report

**Reviewed:** 2026-05-24
**Depth:** standard
**Files Reviewed:** 7
**Status:** issues_found

> Scope note: this review covers the SCF-05 (`atom`/`huckel` init guesses +
> spherically-averaged atomic-RHF engine) and SCF-09 (`mulliken_meta` +
> meta-Löwdin `orth_ao`) gap-closure files only, per the `/gsd:code-review`
> file list (diff_base `aa639ee`). It supersedes the earlier 2026-05-11
> phase-wide pass that reviewed 53 files.

## Summary

Traced the row-major↔F-order index conventions through every matrix product in
`orth.rs`, `atom_hf.rs`, `analyze.rs`, and `init_guess.rs`. The AO-row /
MO-column indexing is **internally consistent and correct**: the `S·c` and
`cᵀSc` block builds in `orth_ao`, the `c_inv = C_orthᵀ·S` and
`D' = c_inv·D·c_invᵀ` products in `mulliken_meta`, the per-l-block scatter in
`angular_averaged_eig` (AO rows use native atom AO indices, MO columns use the
per-l reassembled order consistently with `spherical_occ`), and the
single-atom-rebuild coordinate handoff (`_atom` stores AU coords, rebuilt with
`Unit::Bohr` → factor 1.0, position-irrelevant for an isolated atom) all check
out. AO-range / atom-index overruns return `Err`, and the production paths
honor the never-panic rule (`?`/explicit `Err`, no `unwrap`/`expect`/`panic!`).

No BLOCKER-class defects: no injection surface, no data-loss, no guaranteed
crash, and no demonstrable wrong-number bug on the tested H2/H2O fixtures.

The findings are robustness and convention violations. Two of the disciplines
that the module headers explicitly promise are violated: the oracle-reduction
discipline (bare `+=` accumulators in `analyze.rs`) and accurate diagnostics
(a fabricated `last_diff` in the atom-SCF convergence error). Three latent
correctness gaps — `eigh_gen` linear-dependency padding consumed unchecked,
`lowdin` rank-deficiency accepted, and `default_get_occ` silent under-fill in
the huckel path — are masked today only by the conditioning of the tested
minimal bases and would produce silently wrong results (NaN/∞ energies, broken
electron conservation, or under-counted density) outside that envelope.

## Structural Findings (fallow)

No `<structural_findings>` block was provided by the workflow; this section is
intentionally empty. All findings below are narrative.

## Warnings

### WR-01: `aggregate_pop_to_charges` accumulates per-shell populations with bare `+=`, violating the oracle-reduction discipline

**File:** `crates/pyscf-scf/src/analyze.rs:137`
**Issue:** `atom_pop[atom] += pop_shell;` accumulates per-shell `oracle_sum`
results onto an atom with a bare floating-point `+=`. For any atom carrying more
than one shell (O in STO-3G has 1s + 2s + 2p), this sums three `oracle_sum`
outputs through an un-oracled accumulator. Every reviewed file's header — and
CONTRIBUTING.md Pitfall 9 / threat T-03-15-NUM — states "all reductions go
through `oracle_sum`/`oracle_dot`". A per-atom reduction across shells is exactly
such a reduction; the helper's own doc comment claims "a SINGLE oracle-reduction
site to audit". The shell loop is sequential so this is bit-stable under thread
reordering today, but it silently re-introduces the reduction-order coupling the
oracle wrappers exist to eliminate, and contradicts the documented contract.
**Fix:** Bucket per-shell sums per atom and reduce once:
```rust
let mut per_atom_terms: Vec<Vec<f64>> = vec![Vec::new(); natm];
for shell in 0..nbas {
    // ... existing atom / AO-range bounds checks ...
    per_atom_terms[atom].push(pyscf_algebra::oracle_sum(&ao_pop[lo..hi]));
}
let atom_pop: Vec<f64> = per_atom_terms
    .iter()
    .map(|t| pyscf_algebra::oracle_sum(t))
    .collect();
```

### WR-02: `dip_moment` nuclear sum accumulates over atoms with bare `+=`

**File:** `crates/pyscf-scf/src/analyze.rs:309-311`
**Issue:** `*item += z * coords[k];` accumulates the nuclear dipole
`Σ_A Z_A·r_A[k]` across atoms with a bare `+=` rather than `oracle_sum`. Same
convention violation as WR-01 (this code is pre-existing from plan 03-11 but
`analyze.rs` is in the review scope). For multi-atom molecules this is an
un-oracled reduction over the atom axis, while the electronic term `dip[k] = -e`
on line 293 is correctly oracled — the asymmetry is the tell.
**Fix:** Build per-component term Vecs across atoms and reduce once:
```rust
let mut nuc: [Vec<f64>; 3] = [Vec::new(), Vec::new(), Vec::new()];
for a in 0..rhf.mol.natm {
    let z = /* ... */; let coords = /* ... */;
    for k in 0..3 { nuc[k].push(z * coords[k]); }
}
for k in 0..3 { dip[k] += pyscf_algebra::oracle_sum(&nuc[k]); }
```

### WR-03: `angular_averaged_eig` consumes `eigh_gen` eigenpairs without rejecting the +∞ / zero-column rank-deficiency padding

**File:** `crates/pyscf-scf/src/atom_hf.rs:376-394`
**Issue:** `eigh_gen` documents (eigh_gen.rs:113-127) that when the input is
rank-deficient it pads the trailing eigenvalues with `f64::INFINITY` and zeroes
the corresponding eigenvector columns. This loop runs `for i in 0..nsh` and
unconditionally pushes `eigvals[i]` into `mo_energy` and scatters
`eigvecs[p + i*nsh]`. If a per-`l` averaged overlap block is linearly dependent
(`n_lin < nsh`), `+∞` energies enter `mo_energy` and zero MO columns enter
`mo_coeff` with no error — the per-atom SCF then feeds `inf` into its
energy-convergence test (`(e_elec - last_e).abs()` becomes NaN/`inf`) and into
`init_guess_by_huckel`'s `orb_e`, corrupting the GWH matrix. The tested minimal
bases are well-conditioned so this never fires, but the function offers no guard.
**Fix:** Reject non-finite eigenvalues in the kept range right after the solve:
```rust
if !eigvals.iter().take(nsh).all(|e| e.is_finite()) {
    return Err(PyscfRsError::Core(CoreError::InvalidMolecule(format!(
        "atom_hf: angular-averaged overlap block (l={_l}) is rank-deficient"
    ))));
}
```

### WR-04: `lowdin` accepts a rank-deficient block (`kept.len() < n`) and silently produces a defective `S^{-1/2}`, breaking the `orth_ao` orthonormality invariant

**File:** `crates/pyscf-scf/src/orth.rs:86-110`
**Issue:** `lowdin` returns `Err(Singular)` only when `kept.is_empty()`. When
`0 < kept.len() < n` (eigenvalues at/below `LOWDIN_EIG_TOL` dropped), it builds a
reduced-rank `S^{-1/2}`. In `orth_ao` step (d)/(e) this `x` is applied as
`c_block = c·x`, producing near-zero columns for the dropped directions; those
near-zero columns are written into `C_orth` and yield near-zero diagonals, which
the phase-adjust (line 324-330) leaves untouched. The result violates the
`C_orthᵀ·S·C_orth ≈ I` invariant that `mulliken_meta`'s electron-conservation
argument (analyze.rs:174-177) depends on — a zero diagonal is not 1, so
`Σ pop ≠ nelec`. The orthonormality test only exercises well-conditioned H2, so
a linearly-dependent channel would silently break conservation rather than error.
**Fix:** Reject rank deficiency explicitly (matches `lowdin`'s "linear-dependency
removal returns Err" promise in the header):
```rust
if kept.len() < n {
    return Err(ScfError::Algebra(pyscf_algebra::AlgebraError::Singular).into());
}
```
If reduced-rank channels must be supported, the column count written into
`C_orth` must shrink and `mulliken_meta`'s conservation derivation be revisited.

### WR-05: `init_guess_by_huckel` can silently under-fill the density when `nocc < nelectron/2`

**File:** `crates/pyscf-scf/src/init_guess.rs:472-495` (with `crate::occ::default_get_occ`, occ.rs:19-31)
**Issue:** `nocc` counts occupied atomic orbitals; `mo_e` has length `nocc`.
`default_get_occ(&mo_e, mol.nelectron)` computes `n_occ = nelectron/2` and fills
`occ.iter_mut().take(n_occ)`. `default_get_occ` does **not** guard
`n_occ > mo_energy.len()` — `.take()` silently caps at the slice length. So if
the summed atomic occupied-orbital count is less than `nelectron/2`, the Hückel
density holds fewer than `nelectron` electrons with **no error**. The huckel
test only exercises H2 (nocc=2, n_occ=1) so this never trips, but
`init_guess_by_huckel` is a general init-guess entry point.
**Fix:** Add a guard in `default_get_occ` (fixes both call sites):
```rust
if n_occ > mo_energy.len() {
    return Err(ScfError::Core(pyscf_core::CoreError::InvalidMolecule(format!(
        "Aufbau needs {n_occ} occupied MOs but only {} available", mo_energy.len()
    ))).into());
}
```
or assert `nocc >= mol.nelectron / 2` in `init_guess_by_huckel` before the fill.

### WR-06: `ConvergenceFailure.last_diff` is fabricated from an `unwrap_or(0.0)` on an unrelated quantity

**File:** `crates/pyscf-scf/src/atom_hf.rs:193-198`
**Issue:** On non-convergence the error reports
`last_diff: (last_e - mo_energy.first().copied().unwrap_or(0.0)).abs()`. This
subtracts the *first MO energy* from the *last electronic energy* — two unrelated
quantities — so the reported `last_diff` is meaningless, not the energy delta
that failed to converge. The real convergence metric `(e_elec - last_e).abs()`
(line 186) is out of scope by line 196. When a per-atom SCF genuinely fails to
converge, this prints a misleading number and wastes debugging time. (The
`unwrap_or` itself is fine for the never-panic rule, but it papers over the wrong
operand.)
**Fix:** Hoist the last computed delta into a variable visible at the error site:
```rust
let mut last_diff = f64::INFINITY;
// inside the loop, replacing the bare break test:
last_diff = (e_elec - last_e).abs();
if last_diff < ATOM_CONV_TOL { converged = true; break; }
// at the error site:
return Err(ScfError::ConvergenceFailure { cycles: ATOM_MAX_CYCLE, last_diff }.into());
```

## Info

### IN-01: First-cycle convergence test always compares against `f64::INFINITY`

**File:** `crates/pyscf-scf/src/atom_hf.rs:143,186-190`
**Issue:** `last_e` seeds to `f64::INFINITY`, so cycle 0's test
`(e_elec - INFINITY).abs() < ATOM_CONV_TOL` is always false — the loop always
runs at least two cycles. Harmless (correct, just one guaranteed wasted
iteration) but easy to misread as a bug.
**Fix:** Add a one-line comment near line 186 (`// cycle 0 never converges:
last_e = +inf; first real delta is cycle 1`).

### IN-02: Module-wide `#![allow(dead_code)]` on `atom_hf` is now stale

**File:** `crates/pyscf-scf/src/atom_hf.rs:31`
**Issue:** The header says this allow "becomes inert" once Tasks 2 & 3 wire
`get_atm_nrhf` into the init guesses — which has happened (init_guess.rs:299,378
call it). A blanket file-level `allow(dead_code)` now suppresses genuine
dead-code warnings for the whole module going forward (e.g. the unused
`AtomScfResult.mo_energy`/`mo_occ` fields once `mo_coeff` carries the same data).
**Fix:** Remove the module-level `#![allow(dead_code)]`; let clippy report any
genuinely unused items and annotate the specific ones intentionally kept.

### IN-03: `solve_one_element` silently maps an unknown element symbol to Z=0 (ghost)

**File:** `crates/pyscf-scf/src/atom_hf.rs:92`
**Issue:** `charge_for_symbol(&elem).unwrap_or(0).max(0) as usize` treats any
symbol `charge_for_symbol` does not recognize as Z=0 → a ghost with no electrons
→ `empty_result`. A malformed/typo'd element symbol thus yields a zero density
block instead of a diagnostic. Acceptable for the ghost convention but the silent
fallthrough can mask user-input errors.
**Fix:** Distinguish a recognized ghost marker from an unrecognized symbol and
`Err` on the latter, or emit a `tracing::warn!` on the Z=0 fallback.

### IN-04: `c -= oracle_sum(&dterms)` resembles the WR-01/WR-02 bad pattern but is correct

**File:** `crates/pyscf-scf/src/orth.rs:279`
**Issue:** `c[row + bj*nao] -= pyscf_algebra::oracle_sum(&dterms);` is correct —
the inner reduction over `k` is oracled, and the outer `-=` is a single
subtraction, not an accumulation. Flagging only so a future maintainer auditing
for bare `+=`/`-=` (per WR-01/WR-02) does not mistakenly "fix" it.
**Fix:** None required. Optionally annotate `// single subtraction of an oracled
projection — not an accumulation`.

### IN-05: H2O `mulliken_meta` test lacks per-atom charge polarity / symmetry assertions

**File:** `crates/pyscf-scf/tests/mulliken_meta.rs:98-135`
**Issue:** The H2O test checks conservation (`Σ pop ≈ nelec`, `Σ chg ≈ 0`) and
finiteness but never asserts the physically expected polarity (O negative, H
positive) nor the two-H symmetry it does assert for H2. A density that conserves
electron count but mis-attributes electrons across atoms (e.g. an AO→atom
aggregation index swap) would still pass — conservation is necessary but not
sufficient to catch an aggregation bug.
**Fix:** Add `assert!(res.atom_charges[0] < 0.0)` (O) and
`assert!((res.atom_charges[1] - res.atom_charges[2]).abs() < 1e-7)` (the two H,
given the symmetric input geometry) to pin polarity and per-atom symmetry.

---

_Reviewed: 2026-05-24_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
