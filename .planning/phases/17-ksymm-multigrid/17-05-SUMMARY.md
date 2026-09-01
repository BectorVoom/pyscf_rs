# 17-05 SUMMARY — `pyscf/pbc/lib/kpts.py` -> `pyscf-pbc-symm::kpts`

**Status:** in progress. This file is written INCREMENTALLY, one section per
task, because the execution environment restarts sessions periodically.

Plan: `.planning/phases/17-ksymm-multigrid/17-05-PLAN.md`.

---

## Task 1 — the fold: `make_kpts_ibz` / `map_k_points_fast` — DONE

**Shipped:**
* `crates/pyscf-pbc-symm/src/kpts.rs` (new) — `map_k_points_fast`
  (`kpts.py:305-325`), `make_kpts_ibz` (`kpts.py:39-114`), the `KPoints`
  struct with `kpts_ibz` / `ibz2bz` / `bz2ibz` / `weights_ibz` / `stars` /
  `stars_ops` / `stars_ops_bz` / `time_reversal_symm_bz` /
  `little_cogroup_ops` / `k2opk` / `nkpts_ibz`, `KPoints::build`
  (`kpts.py:1017-1033`) and `make_kpts` (`kpts.py:804-845`).
* `crates/pyscf-pbc-symm/src/symmetry.rs` — added
  `Symmetry::from_lattice_symmetry`, the inverse of the already-shipped
  `to_lattice_symmetry`. This is upstream's
  `self.__dict__.update(_lattice_symm.__dict__)` (`kpts.py:1019-1021`):
  `KPoints::build` REUSES the symmetry the `Cell` already built (with its own
  `check_mesh_symmetry` decision, `cell.py:1771-1772`) rather than re-running
  the space-group search.
* `crates/pyscf-pbc-symm/tests/kpts_ibz.rs` (new) — Gate A + the four
  oracle-free invariants + the thread-determinism test.
* `crates/pyscf-pbc-symm/Cargo.toml` — `rayon` as a normal dep (star search)
  and as a dev-dep (explicit `ThreadPool`s in the determinism test).

**Gate A — EXACT, no tolerance.** All six configurations at `[16,16,16]`:

| configuration | `si` | `diamond` | `lif` / `he_fcc` (Fm-3m control) |
|---|---|---|---|
| A `space_group_symmetry=true` | **145** | **145** | **145** |
| B `symmorphic=true, time_reversal=true` | **145** | **145** | **145** |
| C `symmorphic=true, time_reversal=false` | **245** | **245** | **145** (`C == A`) |
| D `with_gamma_point=false, space_group_symmetry=true` | **408** | **408** | **408** |
| E `with_gamma_point=false, symmorphic=true` | **816** | **816** | **408** (`E == D`) |
| F `time_reversal=true` only | **2052** | **2052** | **2052** |

`si` and `diamond` reproduce 17-01's measured `145/145/245/408/816/2052`
bit-for-bit. The symmorphic (Fm-3m) controls `lif`/`he_fcc` collapse to
`{145,145,145,408,408,2052}` exactly as 17-01 measured — asserted so that a
`symmorphic` branch that silently did nothing could not pass.

**Oracle-free invariants asserted for all six configurations on all four
cells:** `sum(weights_ibz) == 1` to 1e-15; `weights_ibz[i] ==
|stars[i]|/nkpts`; `bz2ibz` total and `ibz2bz[bz2ibz[k]]` in `k`'s own star;
`stars_ops[i][j] == stars_ops_bz[stars[i][j]]` (upstream's own loop,
`test_kpts_ksymm.py:60-65`); and `stars_ops[i][j] . kpts_ibz[i] == kpts[k]`
mod a reciprocal vector to `KPT_DIFF_TOL` — the last one re-derives the
rotation through `op.a2b(cell)` and is what catches an off-by-one op index.

**Parallelism.** The star search (`kpts.py:83-99`) is `rayon::par_iter` over
the outer BZ loop; it is a pure map with disjoint writes, no accumulation, so
no `oracle_sum` ordering to protect (unlike Task 4). Determinism is pinned by
`star_search_is_bit_identical_at_1_and_8_threads`, which varies the worker
count inside ONE process with explicit `rayon::ThreadPool`s — a strictly
stronger check than an env-var sweep across processes — and compares every
index array plus `weights_ibz` bit-for-bit (`to_bits()`).

**Green:** `cargo test -p pyscf-pbc-symm --release --test kpts_ibz`
(4/4), and a full `cargo test -p pyscf-pbc-symm --release` at 62/62.

**Deviations from upstream, recorded:**
1. `KPoints` lives in `pyscf-pbc-symm`, not `pyscf-pbc-lib` (D-PBC-25 /
   17-CONTEXT §4), and holds a `Symmetry` by COMPOSITION.
2. `KPoints` never stores a `Cell` (17-CONTEXT §3.9). Every method that
   needs one borrows `&Cell` for the call, exactly as `Symmetry::build`
   already does.
3. `map_kpts_tuples` is implemented for `ntuple = 1` (`map_k_points_fast`)
   only; the `ntuple > 1` entry point that takes an explicit `kpts_scaled`
   is not reachable from anything in this workspace — `make_ktuples_ibz`
   takes the `k2opk` path (`kpts.py:150-163`). Recorded rather than
   speculatively ported.

---

## Task 2 — `is_trim`, and time reversal — DONE

**Shipped:**
* `crates/pyscf-pbc-lib/src/kpts_helper.rs` — `is_trim`
  (`kpts_helper.py:39-63`), the last `kpts_helper` function still missing
  (17-CONTEXT §5). `khf_ksymm.py:126` needs it for the `eig_trs` branch.
  Takes `a = cell.lattice_vectors()` rather than a `Cell` (this crate is the
  bottom of the periodic DAG), exactly the split `get_kconserv` already uses.
  The module's "still missing" list was updated.
* `crates/pyscf-pbc-gto/src/kpts_mesh.rs` — `is_trim(cell, kpts, tol)`, the
  `Cell`-taking wrapper; re-exported from `pyscf_pbc_gto`.

Upstream's rounding is ported literally, not replaced by a tolerance test:
`logtol = ceil(-log10(tol))`, then `round(2*k_scaled, logtol+1) % 1 < tol`.

**The `2052 = 4096/2 + 4` decomposition is asserted explicitly**
(`is_trim_counts_the_time_reversal_invariant_momenta`): `[16,16,16]` has
`ntrim = 8` (every scaled coordinate 0 or 1/2), so
`(4096 - 8)/2 + 8 = 2052` and `4096/2 + 8/2 = 2052` — both forms asserted.
This pins the TRIM count independently of the fold, which is the point: Gate
A's `F = 2052` and this test fail for different reasons.
Also asserted: all 8 points of a `[2,2,2]` mesh are TRIM; only Gamma is on
`[3,3,3]`.

**Green:** `cargo test -p pyscf-pbc-gto --release --test kpts_mesh` (18/18).

---

## Task 6 — close the stub, let df/dft see the type — DONE

1. `crates/pyscf-pbc-gto/src/kpts_mesh.rs` — `make_kpts_with_symmetry` is
   **DELETED**, not implemented. It had refused with
   `NotYetImplemented { phase = 17 }` since plan 09-07 — the oldest Phase-17
   promise in the tree. Its doc comment is replaced by a module-level note
   redirecting to `pyscf_pbc_symm::kpts::make_kpts`, with the layering
   reason: a `Cell` returning a `KPoints` would invert D-PBC-25
   (`pyscf-pbc-symm` depends on `pyscf-pbc-gto`, not the other way round),
   and upstream's `cell.py:882` delegates to `libkpts.make_kpts` anyway.
   The `pub use` in `lib.rs` dropped the symbol.
2. `crates/pyscf-pbc-gto/tests/kpts_mesh.rs:221-232` — the test that asserted
   on the refusal is replaced by `kpoint_symmetry_is_no_longer_refused_here`
   plus the split-out `make_kpts_rejects_a_zero_axis`.
3. `crates/pyscf-pbc-df/Cargo.toml`, `crates/pyscf-pbc-dft/Cargo.toml` —
   `pyscf-pbc-symm` added as a dependency, **type visibility only**. No DF or
   numint behaviour changed; the seven `isinstance(kpts, KPoints)` branches of
   `pbc/dft/numint.py` are 17-08's work. Verified acyclic and green:
   `cargo build -p pyscf-pbc-df -p pyscf-pbc-dft`.

**`grep -rn "phase: 17" crates/pyscf-pbc-gto/` now returns nothing** (exit 1).

---

## Task 5 — `KPoints` itself, and the k-tuple machinery — DONE

**Shipped** (all in `crates/pyscf-pbc-symm/src/kpts.rs`):
`ktuple_to_index` / `index_to_ktuple` / `loop_ktuples` (`kpts.py:1035-1047`),
`addition_table` (`:1049-1063`), `inverse_table` (`:1065-1074`),
`get_kconserv` (`:1076-1082`), `make_gdf_kptij_lst_jk` (`:1017-1033`),
`little_cogroups` (`:1084-1100`), `little_cogroup_rep` (`:1102-1108`),
`make_ktuples_ibz` (`:116-199`), `make_k4_ibz` (`:205-217`, `sym="s1"`),
`KtuplesIbz` / `K4Ibz` result structs, and `KQuartets` with
`build` / `cache_stabilizer` / `loop_stabilizer` (`:1174-1223`).
`crates/pyscf-pbc-symm/src/error.rs` gained `UnsupportedK4Symmetry`.

**Composition, not inheritance (D-PBC-25 point 2).**
`KPoints { symmetry: Symmetry, .. }`, with `nop()` / `ops()` / `dmats()` /
`has_inversion()` re-exposed so adapters never reach through the field.

**`get_kconserv` DELEGATES** to the shipped
`pyscf_pbc_lib::kpts_helper::get_kconserv` (`kpts_helper.rs:282`), as the
plan requires — it is NOT re-ported. Upstream's own
`add_tab[add_tab[:, inv_tab], :]` (`kpts.py:1079-1081`) is only a faster
route to the identical table, and
`get_kconserv_matches_the_addition_inverse_table_route` proves the two agree
element-for-element, which is exactly what not re-porting buys.

**Green:** `cargo test -p pyscf-pbc-symm --release --test kpts_ktuples`
(8/8). The oracle-free tests:
* `add(k, inv(k)) == Gamma` for every `k`, `inverse_table` is an involution,
  every `addition_table` row is a permutation, and the table is symmetric.
* `ktuple_to_index` / `index_to_ktuple` round-trip over the WHOLE range for
  `ntuple` = 1, 2 and 3.
* `make_ktuples_ibz(ntuple = 2)` (and 3) PARTITION the `nkpts^ntuple` tuple
  space — exactly `nkpts^ntuple` covered, no duplicates — plus weights
  summing to 1 and the `stars_ops[i][j] == stars_ops_bz[stars[i][j]]`
  identity Task 1 asserts one dimension down.
* `make_k4_ibz("s1")` quartets satisfy `kb == kconserv[ki, ka, kj]`.
* `KQuartets::loop_stabilizer` elements genuinely fix the quartet's first
  index.
* every full-BZ little co-group has the same ORDER as its IBZ
  representative's (conjugation is an isomorphism), `indices[ki]` is a
  permutation, and `little_cogroup_rep` returns one character per element.

**Deviations, recorded:**
1. `addition_table` is built ROW BY ROW (fold `[kpts_scaled ; k_i +
   kpts_scaled]` and match) rather than through upstream's
   `(nkpts, nkpts, nkpts, 3)` difference tensor (`:1052-1053`), which is
   `O(nkpts^3)` memory — 400 GB at Gate A's `nkpts = 4096`. Same construction
   `map_k_points_fast` uses; `O(nkpts)` memory.
2. `make_k4_ibz` implements `sym = "s1"` only. The `"s2"` / `"s4"` branches
   (`kpts.py:218-300`) are consumed only by `kccsd_rhf_ksymm` (17-09) and
   return `PbcSymmError::UnsupportedK4Symmetry` rather than a wrong answer.
   `KQuartets` uses `"s1"`, so nothing in this plan is blocked.
3. `little_cogroups` REFUSES (`KptsSymmInputMismatch`) when a
   `little_cogroup_ops` entry indexes past `nop`. That is reachable only
   with `time_reversal = true`, where `k2opk` has `2*nop` columns while `ops`
   has `nop`; **upstream raises `IndexError` on the same input**
   (`kpts.py:1091`). Recorded rather than silently folded with `i % nop`,
   which would invent a different group.
4. `KQuartets` takes `&KPoints` at each call instead of holding one, for the
   same reason `KPoints` does not hold a `Cell` (17-CONTEXT §3.9).

**Follow-up for a later plan (not done here, per the plan's instruction):**
`crates/pyscf-pbc-symm/tests/basis.rs`'s test-local `little_cogroup` /
`sorted_little_pg` helpers are now superseded by
`KPoints::little_cogroup_ops` / `KPoints::little_cogroups`. 17-04's tests
were deliberately NOT refactored as part of 17-05.

---

## Task 3 — the unfolds — DONE, and Gate B is TIGHTER than 17-01's floor

**Shipped** (`crates/pyscf-pbc-symm/src/kpts.rs`): `transform_mo_coeff`
(`kpts.py:449`), `transform_mo_coeff_k` (`:494`), `transform_mo_occ`
(`:528`), `transform_dm` (`:556`), `dm_at_ref_cell` (`:622`),
`transform_mo_energy` (`:644`), `transform_1e_operator` / `transform_fock`
(`:663`), `check_mo_occ_symmetry` (`:717`), `get_rotation_mat_for_mos`
(`:757`), and `MORotationMatrix` (`:1127-1173`) as a lazily-built cache.
Two new error variants: `SymmetryBrokenOccupation`,
`SymmetrizeWavefunctionUnverified`.

Every per-op step delegates to 17-03's `symmetry::transform_*` /
`get_rotation_mat`; this layer is only the loop over stars plus the
time-reversal conjugation. There is still exactly ONE AO-rotation assembly
in the crate (17-CONTEXT §3.2).

### Gate B — against ONE converged full-BZ KRHF, never two

`si`, `[2,2,2]`, FFTDF, `conv_tol = 1e-11`. Max over every k and every
`(p, q)`, printed under `--nocapture`, never a first-violation assert
(17-04-MEASUREMENT.md's lesson):

| `precision` / `conv_tol_grad` | `transform_dm` | `make_rdm1(transform_mo_coeff)` | `transform_1e_operator` | `transform_mo_energy` | `dm_at_ref_cell` | `transform_mo_occ` |
|---|---|---|---|---|---|---|
| 1e-10 / 1e-10 | 1.784e-11 | 1.784e-11 | 1.099e-12 | 6.054e-12 | 6.106e-12 | 0 (exact) |
| **1e-12 / 1e-12 (shipped)** | **2.306e-13** | **2.306e-13** | **1.212e-14** | **3.686e-14** | **5.390e-14** | **0 (exact)** |

**Gate: `GATE_B_TOL = 1e-12`**, ~4x above the measured worst. That is three
orders TIGHTER than 17-01's ≤1e-9-at-default-precision floor, because the
fixture is tightened on BOTH axes rather than the gate being loosened.

Two readings recorded in the test's module doc:
* the ordering is stable across both fixtures — `transform_1e_operator`
  (a pure linear map on the Fock) is one to two orders tighter than
  `transform_dm`, because the density matrix carries the SCF's own residual
  asymmetry on top of the rotation's error. The rotation itself is pinned
  independently at 1e-10 by 17-03's `R S R^H == S`.
* `transform_dm` and `make_rdm1(transform_mo_coeff)` agree **to the last
  digit** at both fixtures, which is what says the two routes to a BZ
  density matrix are the same map.

### 17-CONTEXT §3.1 — the elementwise `mo_coeff` trap is pinned OPEN

`mo_coeff_elementwise_comparison_is_not_a_valid_gate` measures the
elementwise residual at **2.658** (O(1), exactly as 17-01 measured ~2.3 at
`[3,3,3]`) and asserts it is `> 1e-4` — with a failure message saying that a
SMALL value would mean the fixture stopped exercising a degenerate subspace,
not that an elementwise assert became valid. Nobody can "tighten" the DM
comparison into an MO one without deleting that test first.

Also gated: `check_mo_occ_symmetry` REFUSES a hand-broken occupation
(`SymmetryBrokenOccupation`), and `transform_mo_coeff_k` agrees with the
batched `transform_mo_coeff` BIT for BIT.

**Determinism:** `unfolds_are_bit_identical_at_1_and_8_threads` runs
`transform_dm`, `transform_1e_operator`, `transform_mo_coeff` and
`dm_at_ref_cell` in explicit 1- and 8-worker `rayon::ThreadPool`s inside ONE
process and compares every `f64` by `to_bits()`.

---

## Task 4 — `symmetrize_density` — DONE, D-PBC-17 from the FIRST version

**Shipped** (`crates/pyscf-pbc-symm/src/kpts.rs`): `symmetrize_density`
(`kpts.py:377-414`), `symmetrize_density_complex` (upstream's
`symmetrize_complex` branch, in the planar `(re, im)` split per RULE 8),
`symmetrize_wavefunction` (`:416-448`), plus `ft_offsets` /
`rotated_grid_index` / `star_grid_ops`. New error variants:
`MeshNotSymmetric`, `SymmetrizeWavefunctionUnverified`.

**D-PBC-17 is in the first version, not a retrofit.** The per-grid-point sum
over the star's operations goes through `pyscf_algebra::oracle_sum`'s
fixed-shape pairwise tree; the rayon loop is over GRID POINTS (disjoint
writes), not over operations. The §9.3 bit-identity test at 1 and 8 workers
shipped in the same commit, for both the real and the complex path, and also
asserts the complex path's real half is bit-identical to the real path.

**The grid rotation is asserted, not rounded.** Upstream's C kernel
(`pyscf/lib/pbc/symmetry.c:25-48`) computes the translation offset as
`(int)(ft * n)` — a TRUNCATION, silently wrong whenever `ft * n` is not
exactly representable (`(int)(0.25 * 7) == 1`, asserted in the test). This
port checks integrality and returns `MeshNotSymmetric` instead;
`check_mesh_symmetry` (17-03) is what guarantees the check passes.

**Oracle: upstream's own C kernel, transcribed independently in the test.**
`symmetrize_density_matches_upstreams_c_kernel` and
`symmetrize_density_fractional_translation_branch_matches_upstream` both
measure **0e0** — bit-identical to the naive left-to-right C accumulation,
`oracle_sum` notwithstanding, on an 8x8x8 mesh over every IBZ star.

**Finding worth recording:** on all §9.2 fixtures the star search NEVER
names a non-symmorphic op, so the `symmetrize_ft` branch is unreachable
end-to-end. Reason: `SPGElement`'s ordering is
`hash_key = trans * 3^9 + rot`, so zero-translation ops sort FIRST, and
`make_kpts_ibz`'s search `break`s at the first op that maps the IBZ point
onto the BZ point. The ft branch is therefore tested two ways instead —
`ft_offsets` directly (made `pub` with that reasoning in its doc), and
`symmetrize_density` white-box with a non-symmorphic op substituted into
`stars_ops`. A test also asserts the premise (`stars_ops` is all-symmorphic
here), so if that ever changes the reader is told.

`symmetrize_wavefunction` REFUSES: upstream's very first statement is
`raise RuntimeError('need verification')` (`kpts.py:415`), so every line
below it is dead code that has never run. RULE 2 makes that authoritative;
this port does not resurrect an algorithm upstream will not vouch for.

**Green:** `cargo test -p pyscf-pbc-symm --release --test kpts_transform`
(14/14, 311 s).

---

## Verification — all of `17-05-PLAN.md`'s `<verification>` block

| requirement | result |
|---|---|
| `cargo test -p pyscf-pbc-symm -p pyscf-pbc-lib -p pyscf-pbc-gto` green | `pyscf-pbc-lib` + `pyscf-pbc-gto`: every target, 0 failures. `pyscf-pbc-symm --release`: `geom` 7/7, `group` 8/8, `space_group` 15/15, `symmetry` 21/21, `kpts_ibz` 5/5, `kpts_ktuples` 8/8, `kpts_transform` 14/14. **`tests/basis.rs` (17-04's, 5204 s) was NOT re-run in this session** — its inputs are unchanged by this plan (`symmetry.rs` and `error.rs` gained only additive items). |
| Gate A: all six `nkpts_ibz` integers exact | **145 / 145 / 245 / 408 / 816 / 2052** on `si` AND `diamond`; `{145,145,145,408,408,2052}` on the Fm-3m controls `lif`/`he_fcc`. |
| Gate B at 17-01's measured floor, against a single converged SCF | Measured at two fixtures, gated at **1e-12** — three orders TIGHTER than 17-01's ≤1e-9-at-default-precision floor, by tightening the fixture rather than the gate. Worst residual 2.306e-13. |
| `symmetrize_density` bit-identical at 1 and 8 threads | `symmetrize_density_is_bit_identical_at_1_and_8_threads`, real AND complex paths, `to_bits()` equality, explicit `rayon::ThreadPool`s inside one process. |
| Task 1 star search + Task 3 unfolds bit-identical at 1 and 8 threads, **and measurably parallelised** | Bit-identity: `star_search_is_bit_identical_at_1_and_8_threads`, `unfolds_are_bit_identical_at_1_and_8_threads`. **Measurement: `make_kpts si [16,16,16]` (nkpts = 4096, nop = 48) — 31.2 ms at 1 worker, 6.6 ms at 8, 4.76x.** See the note below: the first measurement came out at 0.99x. |
| `grep -rn "phase: 17" crates/pyscf-pbc-gto/` returns nothing | exit 1, no matches. |
| `cargo build -p pyscf-pbc-df -p pyscf-pbc-dft` green with the new dep | green. |
| No `mod tests` in any `src/*.rs` | `grep -rn "mod tests" crates/pyscf-pbc-{symm,lib,gto}/src/` — no matches. |
| clippy | `cargo clippy -p pyscf-pbc-symm --all-targets` reports NOTHING in `src/kpts.rs` or the three new test files. (Pre-existing warnings remain in `src/basis.rs` and `tests/basis.rs`/`tests/space_group.rs`, and a pre-existing `-D warnings` failure lives in `pyscf-algebra/src/complex.rs:49` — neither is this plan's.) |

### The speed measurement changed the implementation

The plan predicted the `O(nkpts x nop)` star search would be the bottleneck
at `nkpts = 4096`. **It is not.** Parallelising only the star search measured
a **0.99x** speedup — the wall clock is dominated by `map_k_points_fast`,
which pays one `round_to_fbz` (an argsort of `2 * nkpts` values per
coordinate) plus one lexsort PER OP, i.e. 96 of each at Gate A. The op loop
there writes one disjoint COLUMN of `bz2bz_ks` each, so it parallelises with
no accumulation ordering either; with both loops parallel the same fixture
measures **4.76x**. Columns are collected in op order, so the result is
independent of the worker count — pinned by the bit-identity test.

This is recorded because the plan's "at nkpts = 4096 this is the difference
between a fixture that runs in the test suite and one that does not" is true
of the FOLD, but the star search was not where the time was.
