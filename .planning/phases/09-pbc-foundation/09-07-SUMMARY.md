---
phase: 09-pbc-foundation
plan: 07
subsystem: pbc-gto / pbc-lib
tags: [pbc, kpts, make_kpts, kpts_helper, get_kconserv, monkhorst-pack, is_zero]

# Dependency graph
requires:
  - phase: 09-pbc-foundation
    plan: 01
    provides: "the pyscf-pbc-lib / pyscf-pbc-gto crate scaffolds + path-scoped lint exemptions"
  - phase: 09-pbc-foundation
    plan: 03
    provides: "Cell, lattice_vectors, reciprocal_vectors, get_abs_kpts / get_scaled_kpts, the five §9.2 reference systems"
  - phase: 09-pbc-foundation
    plan: 06
    provides: "the pyscf-pbc-lib::kpts_helper module (round_to_fbz + the lib.cleanse port), which this plan extends rather than duplicates"
provides:
  - "pyscf_pbc_gto::kpts_mesh — make_kpts (+_default), make_kpts_with_symmetry, Cell::make_kpts, and the Cell-taking get_kconserv / get_kconserv3"
  - "pyscf_pbc_lib::kpts_helper — is_zero / is_gamma_point / gamma_point, member, intersection, unique (UniqueKpts), get_kconserv (Kconserv), get_kconserv3 (Kconserv3, KIdx), KCONSERV_TOL"
  - "WRAP_AROUND / WITH_GAMMA — upstream's make_kpts defaults (cell.py:42-43)"
affects: [09-08, 09-09, 11, 12, 13, 14, 15, 16, 17, 19, 20]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "a table-valued port returns a small newtype (Kconserv { nkpts, data }) whose `data` IS the flat C-order Vec the plan specifies, so the index arithmetic lives in one place instead of at every KCCSD call site"
    - "an upstream identity is asserted BOTH as a hard-coded table and as its defining congruence, so a table transcription error and an algorithm error cannot both hide"

key-files:
  created:
    - crates/pyscf-pbc-gto/src/kpts_mesh.rs
    - crates/pyscf-pbc-gto/tests/kpts_mesh.rs
    - crates/pyscf-pbc-gto/tests/common/kpts_reference.rs
  modified:
    - crates/pyscf-pbc-lib/src/kpts_helper.rs
    - crates/pyscf-pbc-lib/src/lib.rs
    - crates/pyscf-pbc-gto/src/lib.rs
    - crates/pyscf-pbc-tools/src/lattice.rs      # check_lattice_sum_range now calls the shared `intersection`
    - crates/pyscf-pbc-gto/tests/lattice.rs      # its hand-rolled MP grid replaced by make_kpts

key-decisions:
  - "`is_zero`'s threshold is KPT_DIFF_TOL = 1e-6, NOT the 1e-9 the plan's STEP 2 quotes. `kpts_helper.py:32` reads `abs(np.asarray(kpt)).sum() < KPT_DIFF_TOL`, and upstream confirms it: is_zero([1e-7,0,0]) is True, is_zero([1e-6,0,0]) is False. RULE 2 makes the Python authoritative; the plan text is wrong."
  - "`make_kpts`'s wrap_around is a PER-AXIS fold (`ks[ks>=.5] -= 1`, cell.py:866-867) applied BEFORE the cartesian product and BEFORE scaled_center — not `round_to_fbz(scaled_kpts)` on the finished product as the plan's STEP 1 says. The two differ observably (round_to_fbz also rounds to 6 decimals and runs `cleanse`, and it would fold the center shift too)."
  - "`get_kconserv` implements `_get_kconserv_slow` (kpts_helper.py:313-325) ONLY. Upstream first tries a k2gamma shortcut (`kpts_to_kmesh` + `double_translation_indices`, :303-311); `pyscf/pbc/tools/k2gamma.py` is not in this plan's PORT block. The two paths were verified IDENTICAL on every probed input before the choice was made."
  - "`get_kconserv` / `get_kconserv3` live in pyscf-pbc-lib and take `a` (the lattice vectors) rather than a `Cell`: pyscf-pbc-lib is the BOTTOM of the §4 DAG. Same split plans 09-04 and 09-06 used."
  - "`get_kconserv` returns `Kconserv { nkpts, data }`, not a bare `Vec<i32>`. `data` is exactly the flat C-order vector the plan specifies; the newtype adds `get(k, l, m)` so Phase 16's KCCSD does not re-derive the stride."
  - "`get_kconserv3`'s `kijkab` is a `&[KIdx]` with `KIdx::{One, Many}`, reproducing upstream's int-or-array entries and the `:436-438` squeeze that DROPS a pinned axis from the output shape."
  - "`unique` returns first-occurrence order. Upstream's docstring says 'sorted', but the argsort/argsort pair at :122-125 undoes NumPy's lexicographic order — verified against live output."
  - "`unique`'s merge test is 'equal after rounding to 6 decimals', not 'within KPT_DIFF_TOL'. That is what `np.unique(kpts.round(digits), axis=0)` does, and the two disagree at a rounding boundary."
  - "`space_group_symmetry` / `time_reversal_symmetry` return NotYetImplemented { phase: 17 } (D-PBC-15) — upstream hands back a `pbc.lib.kpts.KPoints` object there."
  - "`intersection` moved into kpts_helper and plan 09-06's private copy inside `pyscf_pbc_tools::lattice::check_lattice_sum_range` now calls it. One implementation in the workspace."

patterns-established:
  - "kconserv[k,k,m] == m and kconserv[k,l,m] == kconserv[m,l,k] as tier-1 invariants — they pin the sign convention and the K/M stride order without any reference table"
  - "the same kconserv table asserted on BOTH the Bohr-built and the Angstrom-built diamond, and on a shifted (non-gamma) mesh, to show it is integer topology and not a lattice-constant coincidence"

requirements-completed: [PBC-GTO-05]

# Metrics
duration: ~1h
completed: 2026-08-25
---

# Phase 9 Plan 07: k-point meshes — `make_kpts`, `kpts_helper`, `get_kconserv`

**`pyscf-pbc-gto` gains `kpts_mesh.rs` (the port of `cell.py:827-884` plus the
`Cell`-taking wrappers) and `pyscf-pbc-lib::kpts_helper` grows from the two functions
plan 09-06 seeded to the full Phase-9 set. 16 new tests are green; every `make_kpts`
variant reproduces upstream PySCF 2.12.1 to 1e-12 on a Bohr-specified diamond, and the
`kconserv` 8x8x8 / 6x6x6 tables and the `kconserv3` 4x4x4x4 table match EXACTLY.**

## What Shipped

### `crates/pyscf-pbc-lib/src/kpts_helper.rs`

| Rust | Upstream | Note |
|---|---|---|
| `is_zero` / `is_gamma_point` / `gamma_point` | `kpts_helper.py:31-37` | threshold is `KPT_DIFF_TOL`, see below |
| `member(kpt, kpts)` | `:90-97` | Chebyshev distance `< KPT_DIFF_TOL` |
| `intersection(kpts1, kpts2)` | `:99-106` | now also backs `check_lattice_sum_range` |
| `unique(kpts) -> UniqueKpts` | `:108-142` | first-occurrence order |
| `get_kconserv(a, kpts) -> Kconserv` | `:291-325` | `_get_kconserv_slow` body |
| `get_kconserv3(a, kpts, kijkab) -> Kconserv3` | `:409-439` | incl. the pinned-axis squeeze |
| `KCONSERV_TOL = 1e-9` | `:323`, `:433` | NOT `KPT_DIFF_TOL` — a separate constant |

### `crates/pyscf-pbc-gto/src/kpts_mesh.rs`

`make_kpts(cell, nks, wrap_around, with_gamma_point, scaled_center)`,
`make_kpts_default`, `Cell::make_kpts(nks)`, `make_kpts_with_symmetry` (the Phase-17
deferral), `WRAP_AROUND` / `WITH_GAMMA`, and the `Cell`-taking `get_kconserv` /
`get_kconserv3`. The `kpts_helper` surface is re-exported here so a caller needs one
`use`.

## Deviations from the plan

1. **`is_zero`'s threshold is `1e-6`, not `1e-9`.** PBC-MASTER-PLAN §8.1 plan 09-07
   step 2 says `< 1e-9`; `kpts_helper.py:32` says `< KPT_DIFF_TOL` and `KPT_DIFF_TOL`
   is `1e-6` (`:28`). Live upstream confirms: `is_zero([1e-7,0,0]) is True`,
   `is_zero([1e-6,0,0]) is False`. Since this predicate "decides real-vs-complex code
   paths everywhere", getting it wrong by three orders of magnitude would have been a
   silent divergence in every later phase. Pinned by
   `is_zero_uses_kpt_diff_tol_not_1e_9`.
2. **`wrap_around` is a per-axis fold, not `round_to_fbz`.** Step 1 describes
   `if wrap_around { scaled_kpts = round_to_fbz(scaled_kpts) }`. Upstream folds each
   1-D axis grid with `ks[ks>=.5] -= 1` BEFORE the cartesian product and BEFORE
   `scaled_center` is added (`cell.py:866-872`), and does no rounding or `cleanse`.
   Ported per RULE 2.
3. **`get_kconserv` implements the slow path only.** Upstream's `:303-311` shortcut
   needs `pyscf/pbc/tools/k2gamma.py` (`kpts_to_kmesh`, which wants
   `Fraction.limit_denominator`, and `double_translation_indices`), which is not in this
   plan's PORT block. Before choosing, both paths were run against each other on
   `[1,1,1]`, `[2,2,1]`, `[2,2,2]`, `[3,1,2]`, `[3,3,3]`, `[4,2,1]` and the
   `with_gamma_point=False` / `wrap_around=True` / `scaled_center` variants — identical
   every time. This is a performance deviation, not a numerical one. See Carry-overs.
4. **Files split across two crates.** The plan's FILES list has both
   `crates/pyscf-pbc-gto/src/kpts_mesh.rs` and
   `crates/pyscf-pbc-lib/src/kpts_helper.rs`, which is the split actually used:
   `pyscf-pbc-lib` cannot name `Cell`, so `get_kconserv` / `get_kconserv3` take the
   lattice vectors and `kpts_mesh` supplies them.
5. **`get_kconserv` returns `Kconserv`, not a bare `Vec<i32>`.** `Kconserv::data` is
   exactly the flat C-order `[nk][nk][nk]` vector the plan asks for; the wrapper adds
   `nkpts` and `get(k, l, m)`.
6. **Only the Phase-9 subset of `kpts_helper.py` is ported.** `is_trim`,
   `unique_with_wrap_around`, `members_with_wrap_around`, `group_by_conj_pairs`,
   `conj_mapping`, `kk_adapted_iter`, `KptsHelper` and `get_kconserv_ria` have no
   Phase-9 consumer; the PORT block does not list them.

## Green test commands

```
cargo test -p pyscf-pbc-gto --test kpts_mesh   # 16 passed
cargo test -p pyscf-pbc-gto                    # 98 passed
cargo test -p pyscf-pbc-tools                  # 30 passed
cargo clippy -p pyscf-pbc-gto -p pyscf-pbc-lib -p pyscf-pbc-tools --all-targets -- -D warnings
cargo build --workspace
cargo run -p xtask --bin check-dependency-wall  # PASS (ALG-06)
cargo run -p xtask --bin check-forbidden-paths  # PASS (FOUND-08, 350 files)
```

## Numeric acceptance

Diamond specified **directly in Bohr** (`H = 3.3701375705493315`), so absolute k-points
compare at **1e-12** with no Angstrom-conversion slack:

| case | `nks` | kwargs | matched |
|---|---|---|---|
| `222` | `[2,2,2]` | defaults | 8 kpts, 1e-12 |
| `222_nogamma` | `[2,2,2]` | `with_gamma_point=False` | 8 kpts, 1e-12 |
| `222_wrap` | `[2,2,2]` | `wrap_around=True` | 8 kpts, 1e-12 |
| `222_nogamma_wrap` | `[2,2,2]` | both | 8 kpts, 1e-12 |
| `321` | `[3,2,1]` | defaults | 6 kpts, 1e-12 |
| `333_wrap` | `[3,3,3]` | `wrap_around=True` | 27 kpts, 1e-12 |
| `222_center` | `[2,2,2]` | `scaled_center=[0.1,0.2,0.3]` | 8 kpts, 1e-12 |

`get_kconserv` matches the upstream **8x8x8** table for `make_kpts([2,2,2])` and the
**6x6x6** table for `make_kpts([3,2,1])` EXACTLY — on the Bohr diamond, on the
Angstrom-built §9.2 diamond, and on the shifted `with_gamma_point=False` mesh (which
yields the same table, as momentum conservation depends only on the mesh topology).
`get_kconserv3` matches the upstream **4x4x4x4** table for `make_kpts([2,2,1])` with
`kijkab = [r, 1, r, r, r]` EXACTLY, including the dropped `kj` axis.
`unique` matches upstream's `(uniq, index, inverse)` triple on the duplicate probe.

Independent of any table, the defining congruences are asserted directly:
`(k_K - k_L + k_M - k_N) . a / 2pi` and `(k_i + k_j + k_k - k_a - k_b - k_c) . a / 2pi`
are integral to 1e-9 for EVERY entry of both tables.

## Carry-overs

- **The `k2gamma` fast path for `get_kconserv` is not ported.** Today's cost is
  `O(nk^4)` where upstream's shortcut is `O(nk^2)` for a full Monkhorst-Pack mesh. At
  `nk = 64` that is ~17M inner tests — fine for Phase 9 and 11, plausibly not for
  Phase 16 KCCSD on a large mesh. Porting `kpts_to_kmesh` +
  `double_translation_indices` belongs with the rest of `pbc/tools/k2gamma.py`; the
  equivalence recorded above means it can be swapped in behind the same signature with
  the existing tests as the gate.
- **`Cell::make_kpts` is the only k-point constructor.** `pbc.lib.kpts.KPoints` (the
  symmetry-adapted container) is Phase 17 (D-PBC-15); until then
  `make_kpts_with_symmetry` returns `NotYetImplemented { phase: 17 }` rather than a
  plain `Vec` that silently ignores the request.
- **The rest of `kpts_helper.py` is unported** — see deviation 6. `is_trim` and
  `conj_mapping` are the first ones a later phase will want (Phase 17 k-point symmetry);
  `unique_with_wrap_around` / `kk_adapted_iter` are Phase 13/14 DF concerns.
- **`pyscf-pbc-lib` still has no test target of its own.** Its Phase-9 surface is
  exercised entirely through `crates/pyscf-pbc-gto/tests/kpts_mesh.rs` and
  `crates/pyscf-pbc-tools/tests/lattice.rs`, which is where the `Cell` and the reference
  systems live. Plan 09-09's rollup should decide whether the crate wants a direct
  `tests/kpts_helper.rs` too.
- **`pyscf_core::Unit::Ang.length_in_au()` still disagrees with upstream by 4.95e-9
  relative** (09-03 carry-over, unchanged). It is why the tier-2 k-point comparisons use
  `diamond_bohr()`; the integer `kconserv` tables are immune and are asserted on both
  cells.
