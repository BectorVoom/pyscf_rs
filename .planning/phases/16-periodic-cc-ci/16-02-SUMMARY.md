# 16-02 — the complex arena, `symm_map`, `KTensor`. COMPLETE 2026-09-06.

## Task 2 was already shipped — by Phase 15

`KptsHelper::build_symm_map` / `transform_symm` / `_operation` **exist**, in
`crates/pyscf-pbc-lib/src/khelper.rs`, landed by Phase 15 and gated by
`tests/khelper.rs`. `16-CONTEXT §3.1` assumed otherwise. What this plan added
is the property set KCCSD depends on and KMP2 did not
(`crates/pyscf-pbc-lib/tests/symm_map.rs`, 5 tests): orbit completeness, the
≤4 bound, the four operations' mutual consistency (`op1∘op1 = id`,
`op2∘op2 = id`, `op3 = op1∘op2`), a bitwise check of `transform_symm` against a
hand-written reference for each of the four operations, and the laziness
`kccsd_rhf.py:512` requires.

**One upstream behaviour the tests had to be rewritten around, and it is not a
defect.** A triple may appear MORE THAN ONCE inside its own orbit: at a fixed
point of one of the four operations, several of `(kp,kq,kr)`, `(kr,ks,kp)`,
`(kq,kp,ks)`, `(ks,kr,kq)` coincide, and `kpts_helper.py:597-612` appends each
unconditionally while `completed` guards only the OUTER loop — so at
`nkpts = 1` the single triple is appended four times and `_operation` ends on
the LAST matching operation. The Rust port reproduces this exactly. The
assertions therefore count DISTINCT representatives, not orbit multiplicity.

## Task 1 — `ZWorkspacePool` (`crates/pyscf-runtime/src/zworkspace_pool.rs`)

`shape_bytes = product * 16`, a separate `ZBufferId`, planar `re`/`im` storage
(RULE 8 — complex never crosses the ALG-06 wall as an element type), the same
free-list / `InMemory | Spilled` / HARD-`MemoryLimitExceeded` shape as the f64
pool, and an HDF5 spill with two datasets.

The two `16-REVIEW.md §2.2` defects are fixed **in the new type**:
`with_slices` BORROWS instead of copying the buffer, and the registry lock is
released before the caller's closure runs while each allocation carries its own
`Mutex` — `in_use` is an `AtomicBool` so a `release` during another thread's
`with_slices` cannot silently leak a buffer off the free-list.

**`WorkspacePool` is byte-for-byte unchanged** — `git diff --stat
crates/pyscf-runtime/src/workspace_pool.rs` is empty and `cargo test -p
pyscf-ccsd` is green (51 + 3 + 5 + 6 + 3 + 2 + 5 + 2 + 3 + 1 + 4 + 4 + 1
passing).

### Deviation 1 — the plan's literal `192` is wrong and using it would have encoded the defect

16-02 Task 4 test 1 asks that a `[2,3,4]` complex reservation "reports **192**
bytes, not 96". `2*3*4 = 24`, so `24 * 8 = 192` is the **f64** count and
`24 * 16 = 384` is the complex one; both of the plan's numbers are f64-sized.
The test asserts **384**, and asserts `192` for the f64 sibling on the same
shape — which is exactly the "two arenas are deliberately different" property
clause 1 exists to protect. Following the plan's literal would have shipped a
complex arena sized with `* 8`, reporting half its footprint to the HARD
refusal: the precise failure `16-CONTEXT §1.3` forbids.

## Task 3 — `KTensor` (`crates/pyscf-pbc-cc/src/ktensor.rs`)

A `(nkpts^rank)`-indexed container of equal-shaped complex blocks over the
arena, **one pool buffer per k-address** so blocks are independently
borrowable — which is what lets the `nkpts³` loops run in parallel. Three
tiers matching `kccsd_rhf.py:132-137`/`:423-455`/`:777-832`
(`Tier::Absent | InMemory | Spilled`), selected from
`exact_bytes = nkpts^rank * prod(block_shape) * 16` and **never** from
`_mem_usage` — which 16-01 measured over-estimating by **9.143×**
(`gth-szv`) / **6.058×** (`gth-dzvp`), confirming the review's derived
9.1×/6.2×.

## Verification

* `cargo test -p pyscf-runtime -p pyscf-pbc-lib -p pyscf-pbc-cc` green
  (6 + 5 + 4 + 8 tests added).
* `cargo test -p pyscf-ccsd` green; the f64 pool's diff is empty.
* `check-orphan-modules` PASS (352 files, all reachable);
  `check-dependency-wall` PASS (ALG-06 intact). `pyscf-runtime` gained no
  dependency; `pyscf-pbc-cc` gained `pyscf-runtime` (the arena's owner, exactly
  as `pyscf-ccsd` depends on it for the f64 one) and `rayon` (D-PBC-29 clause 2).
* No tolerance appears anywhere in this plan's tests, by construction.
