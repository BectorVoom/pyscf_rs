---
phase: 09-pbc-foundation
plan: 06
subsystem: pbc-tools / pbc-gto / pbc-lib
tags: [pbc, lattice-sum, get_lattice_Ls, super_cell, cell_plus_imgs, monkhorst-pack, qr, round_to_fbz]

# Dependency graph
requires:
  - phase: 09-pbc-foundation
    plan: 01
    provides: "the pyscf-pbc-lib / pyscf-pbc-tools / pyscf-pbc-gto crate scaffolds + path-scoped lint exemptions"
  - phase: 09-pbc-foundation
    plan: 03
    provides: "Cell (OWNS a Mole), lattice_vectors / vol / get_scaled_atom_coords / get_scaled_kpts / get_abs_kpts, the five §9.2 reference systems"
  - phase: 09-pbc-foundation
    plan: 04
    provides: "mat3 (det3/inv3/norm3/cross3/dot3), mesh::qr_r22_abs (the 3x3 Householder QR this plan generalises), Cell::try_rcut / Cell::try_mesh"
provides:
  - "pyscf_pbc_tools::lattice — get_lattice_ls, check_lattice_sum_range, max_atom_pair_distance, monkhorst_pack_size_from_scaled, round_to_cell0, qr_row2"
  - "pyscf_pbc_tools::supercell — super_cell_translations, cell_plus_imgs_translations, scale_lattice, image_atom_coords"
  - "pyscf_pbc_gto::lattice — get_lattice_ls (+_default), check_lattice_sum_range, get_monkhorst_pack_size (+_default), lattice_sum_dimension"
  - "pyscf_pbc_gto::supercell — super_cell, cell_plus_imgs"
  - "pyscf_pbc_lib::kpts_helper — KPT_DIFF_TOL, round_to_fbz (+ the lib.cleanse port it needs)"
affects: [09-07, 09-08, 09-09, 10, 11, 12, 13, 14, 16, 17, 18, 20]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "the geometry core lives in the LOWEST crate that can hold it and the Cell-taking wrapper lives in pyscf-pbc-gto — the same split plan 09-04 used for mesh::cutoff_to_mesh / Cell::cutoff_to_mesh, forced by the pbc-tools -> pbc-gto DAG edge"
    - "a supercell is rebuilt through pyscf_gto::build_from from the cell's PARSED _atom/_basis, the same trick pyscf_gto::loads uses, so basis normalisation still runs exactly once"
    - "an upstream quirk that looks like a bug is ported verbatim and labelled as such in the doc comment, then PINNED by a test, so a later 'fix' has to be deliberate"

key-files:
  created:
    - crates/pyscf-pbc-tools/src/lattice.rs
    - crates/pyscf-pbc-tools/src/supercell.rs
    - crates/pyscf-pbc-tools/tests/lattice.rs
    - crates/pyscf-pbc-gto/src/lattice.rs
    - crates/pyscf-pbc-gto/src/supercell.rs
    - crates/pyscf-pbc-gto/tests/lattice.rs
    - crates/pyscf-pbc-lib/src/kpts_helper.rs
  modified:
    - crates/pyscf-pbc-tools/src/lib.rs
    - crates/pyscf-pbc-gto/src/lib.rs
    - crates/pyscf-pbc-lib/src/lib.rs

key-decisions:
  - "`pyscf_algebra::qr` CANNOT serve find_boundary: its locked signature accepts only a SQUARE matrix (host_fallback.rs:146 `square_n`), and find_boundary factorises a 3x6. The plan's 'Use pyscf_algebra::qr' instruction is therefore a deviation — `qr_row2` generalises plan 09-04's `qr_r22_abs` to a variable column count, same Householder reflections in the same LAPACK dgeqrf order."
  - "get_lattice_Ls / super_cell / cell_plus_imgs / get_monkhorst_pack_size are SPLIT: geometry cores in pyscf-pbc-tools (per the plan's FILES), Cell-taking wrappers in pyscf-pbc-gto. `super_cell` RETURNS a Cell, which pyscf-pbc-tools cannot name without inverting the §4 DAG."
  - "super_cell rebuilds the molecular half through pyscf_gto::build_from instead of splicing _atm/_bas/_env by hand. Upstream avoids build() to stop a SECOND normalisation of the contraction coefficients; rebuilding from the pre-normalisation `_basis` normalises exactly once, so the result is equivalent and rides the one Mole build path (D-PBC-01)."
  - "`cell_plus_imgs` scales `a` by `nimgs[i]`, NOT by `2*nimgs[i]+1` (pbc.py:741). That leaves vol unchanged for nimgs=[1,1,1] while natm grows 27x. Ported verbatim per RULE 2 and PINNED by `cell_plus_imgs_matches_upstream_including_its_lattice_quirk`."
  - "Upstream's unused `nimgs` keyword on get_lattice_Ls (pbc.py:601) is NOT reproduced — the body never reads it, and a silently-ignored argument is a worse API than an absent one."
  - "round_to_fbz + lib.cleanse landed in pyscf-pbc-lib::kpts_helper (plan 09-07's file) rather than being duplicated inside pyscf-pbc-tools, because 09-06's round_to_cell0 is nothing but a round_to_fbz call. 09-07 EXTENDS that module; it does not create it."
  - "The plan's guessed TEST assertions were REGENERATED as instructed. `Ls[0] == [0,0,0]` is WRONG: Ls is the raw cartesian_prod starting at -bounds, so the origin sits mid-array (index len/2 before discard). The tests assert that instead."
  - "space_group_symmetry on a supercell returns NotYetImplemented { phase: 12 } (D-PBC-20) rather than silently returning a supercell with no lattice-symmetry object."

patterns-established:
  - "The 3x6 QR row-2 tail is cross-checked against its closed form |col_j . n_hat| with n_hat the unit normal of span(c0,c1) — a tier-1 invariant that pins every column, not just R[2,2]"
  - "Antisymmetry Ls[i] == -Ls[n-1-i] as a tier-1 test: it pins the cartesian_prod bounds, the axis order and the last-index-fastest convention at once"

requirements-completed: [PBC-TOOLS-01]

# Metrics
duration: ~1h
completed: 2026-08-25
---

# Phase 9 Plan 06: Lattice sums — `get_lattice_Ls`, `super_cell`, `cell_plus_imgs`

**`pyscf-pbc-tools` gains `lattice.rs` and `supercell.rs` — the geometry cores of
`pyscf/pbc/tools/pbc.py:587-786` and `:836-840`, ported line by line — and
`pyscf-pbc-gto` gains the `Cell`-taking wrappers, including a `super_cell` /
`cell_plus_imgs` that build a real `Cell` through the workspace's one `Mole` build path.
29 new tests are green; every image count, `Ls[0]` row and max-`|L|` reproduces upstream
PySCF 2.12.1 EXACTLY on a Bohr-specified diamond, and every count reproduces exactly on
all five §9.2 reference systems.**

## What Shipped

### `crates/pyscf-pbc-tools/src/lattice.rs`

| Rust | Upstream | Note |
|---|---|---|
| `qr_row2(cols)` | `np.linalg.qr(aR.T)[1][2]` | 3 x n generalisation of plan 09-04's `qr_r22_abs` |
| `get_lattice_ls(a, scaled, coords, rcut, dim, discard)` | `get_lattice_Ls`, `pbc.py:601-661` | the `find_boundary` / `cartesian_prod` / `discard` body |
| `check_lattice_sum_range(ls_full, ls, coords)` | `pbc.py:663-676` | incl. the `intersection` port (`kpts_helper.py:99-106`) |
| `max_atom_pair_distance(coords)` | `pbc.py:657-659` | |
| `monkhorst_pack_size_from_scaled(skpts, tol)` | `get_monkhorst_pack_size`, `pbc.py:587-599` | incl. the `10**(-int(-log10(1/nk))-2)` line |
| `round_to_cell0(r, tol)` (+ `_default`) | `pbc.py:836-840` | delegates to `pyscf_pbc_lib::round_to_fbz` |

### `crates/pyscf-pbc-tools/src/supercell.rs`

`super_cell_translations` (`pbc.py:706-713`, incl. the `wrap_around` index shift),
`cell_plus_imgs_translations` (`pbc.py:729-733`), `scale_lattice`
(`np.einsum('i,ij->ij', n, a)`) and `image_atom_coords` (the image-major atom layout of
`_build_supcell_`, `pbc.py:757-759`).

### `crates/pyscf-pbc-gto/src/lattice.rs` + `supercell.rs`

`lattice_sum_dimension` (the `pbc.py:609-614` default: a 2D cell still sums in all three
dimensions unless `low_dim_ft_type == inf_vacuum`), `get_lattice_ls` /
`get_lattice_ls_default`, `check_lattice_sum_range`, `get_monkhorst_pack_size` /
`_default`, `super_cell`, `cell_plus_imgs`.

### `crates/pyscf-pbc-lib/src/kpts_helper.rs`

`KPT_DIFF_TOL`, `round_to_fbz` (`kpts_helper.py:65-88`) and the `lib.cleanse(axis=0)`
port it calls (`numpy_helper.py:1561-1602`), including NumPy's round-half-to-EVEN and the
`decimal = -int(log10((tol+1e-16)/10))` truncation.

## Deviations from the plan

1. **`pyscf_algebra::qr` is not used** (plan STEP 1c). Its locked signature accepts only a
   square matrix; `find_boundary` factorises a 3x6. `qr_row2` generalises plan 09-04's
   already-proven Householder QR instead, and is cross-checked against `qr_r22_abs`,
   `qr_r22_abs_closed_form` and the closed-form perpendicular component of every column.
2. **The files are split across two crates.** The plan's FILES list only
   `crates/pyscf-pbc-tools/src/{lattice,supercell}.rs`, but `super_cell` returns a `Cell`
   and `pyscf-pbc-tools` sits BELOW `pyscf-pbc-gto` in the §4 DAG. Geometry cores stayed
   where the plan put them; `crates/pyscf-pbc-gto/src/{lattice,supercell}.rs` hold the
   `Cell` plumbing. This is why the plan's own verification block lints `pyscf-pbc-gto`.
3. **`round_to_fbz` landed in `pyscf-pbc-lib`, plan 09-07's file.** Writing a private copy
   in `pyscf-pbc-tools` would have handed 09-07 a de-duplication chore. 09-07 extends
   `kpts_helper.rs` with `get_kconserv`, `get_kconserv3`, `member`, `intersection`,
   `unique` and `is_zero`; `round_to_fbz` is already there.
4. **The plan's TEST assertions were regenerated** (as plan 09-04 also had to).
   `Ls[0] == [0,0,0]` is false: `Ls` is the raw `cartesian_prod` from `-bounds`, so the
   origin is at index `len/2` before `discard` reorders nothing but removes rows around
   it. The tests assert the true upstream `Ls[0]` per case, plus "the origin occurs
   exactly once".
5. **`get_lattice_Ls`'s `nimgs` keyword is omitted** — upstream accepts it and never reads
   it.
6. **`supcell.magmom` (`pbc.py:717-720`) is not ported**: this workspace's `Mole` has no
   `magmom` field yet.

## Green test commands

```
cargo test -p pyscf-pbc-tools --test lattice     # 18 passed
cargo test -p pyscf-pbc-tools                    # 30 passed (18 lattice + 12 mesh)
cargo test -p pyscf-pbc-gto  --test lattice      # 11 passed
cargo test -p pyscf-pbc-gto                      # 82 passed
cargo clippy -p pyscf-pbc-gto -p pyscf-pbc-tools -p pyscf-pbc-lib --all-targets -- -D warnings
cargo build --workspace
cargo run -p xtask --bin check-dependency-wall   # PASS (ALG-06)
cargo run -p xtask --bin check-forbidden-paths   # PASS (FOUND-08, 350 files)
```

## Numeric acceptance

Diamond specified **directly in Bohr** (`H = 3.3701375705493315`), so no Angstrom -> Bohr
constant enters and the match is exact rather than 1e-7:

| `rcut` | `discard` | `len(Ls)` | `Ls[0]` | `max |L|` |
|---:|---|---:|---|---:|
| 10.0 | true | 135 | `[3.3701375705, -6.7402751411, -10.1104127116]` | 12.60990013529029 |
| 10.0 | false | 729 | `[-26.9611005644]*3` | 46.69799600550547 |
| 5.0 | true | 19 | `[0, 0, -6.7402751411]` | 6.740275141098663 |
| 5.0 | false | 343 | `[-20.2208254233]*3` | 35.023497004129105 |
| 21.31940052177759 | true | 767 | `[10.1104127116, -13.4805502822, -16.8506878527]` | 23.830471296669888 |
| 21.31940052177759 | false | 3375 | `[-47.1819259877]*3` | 81.72149300963457 |

`check_lattice_sum_range(cell, get_lattice_Ls(cell))` = **22.035133643781318** (matched to
1e-9). `round_to_cell0` matches upstream on all four probe rows to 1e-12.

Image counts on the five §9.2 systems (Angstrom input, matched EXACTLY — the 4.95e-9
lattice gap is far from any `ceil` boundary):

| system | `rcut = 10` T/F | `rcut = 5` T/F |
|---|---|---|
| diamond | 135 / 729 | 19 / 343 |
| si | 43 / 343 | 13 / 125 |
| lif | 177 / 729 | 55 / 343 |
| he_fcc | 87 / 729 | 13 / 125 |
| graphene | 31 / 189 | 7 / 75 |

`super_cell(diamond, [2,2,2])`: `natm == 16`, `nao_nr == 64`, `mesh == [94,94,94]`,
`vol / cell.vol == 8` to 1e-9 and `vol == 612.4390450600976` to 1e-7 relative; all 16
atom positions match upstream. `super_cell(diamond, [3,1,2])`: `natm == 12`,
`mesh == [141,47,94]`, ratio 6. `cell_plus_imgs(diamond, [1,1,1])`: `natm == 54`,
`mesh == [141,141,141]`, `a` and `vol` UNCHANGED (the upstream quirk).
`get_monkhorst_pack_size` recovers `[1,1,1]`, `[2,2,2]`, `[3,2,1]`, `[4,4,4]`, `[2,1,3]`.

## Carry-overs

- **Plan 09-05's open item is CLOSED, negatively.** Its carry-over asked whether
  `get_uniform_grids(wrap_around = false)` needs "an extra image layer in
  `get_lattice_Ls` for 2D calculations". Upstream's `get_lattice_Ls` has no such knob —
  the extra layer is bought by passing a larger `rcut`, at the call site. Nothing for this
  plan to add; Phase 12's 2D work owns the choice.
- **`super_cell` inherits `cell.charge` and `cell.spin` unchanged** (upstream's
  `copy(deep=False)`), so an ODD-spin cell replicated an EVEN number of times will be
  rejected by `pyscf_gto::build_from`'s electron-parity check where upstream, which never
  calls `build()`, would not notice. All five §9.2 systems are closed-shell, so nothing
  hits this today. Revisit when Phase 11 needs spin-polarised supercells.
- **`super_cell` copies `rcut` from the primitive cell** — upstream does the same
  (`copy(deep=False)` preserves `_rcut`), so the supercell's `rcut` is NOT re-estimated
  for its larger geometry. Matching upstream is the acceptance criterion; note it before
  using a supercell's `rcut` as a lattice-sum radius.
- **`pyscf_core::Unit::Ang.length_in_au()` still disagrees with upstream by 4.95e-9
  relative** (09-03 carry-over, unchanged). It is why `crates/pyscf-pbc-gto/tests/lattice.rs`
  asserts floats at 1e-7 relative while `crates/pyscf-pbc-tools/tests/lattice.rs`, whose
  cell is given in Bohr, asserts at 1e-12.
- **`_build_supcell_`'s `magmom` line is unported** — blocked on `Mole` gaining the field.
