---
phase: 09-pbc-foundation
plan: 03
subsystem: pbc-gto
tags: [pbc, cell, lattice, reciprocal-vectors, kpoints, deref, reference-systems]

# Dependency graph
requires:
  - phase: 02-gto
    provides: "pyscf_gto::build_from / M / MoleBuildArgs / format_atom / format_basis / dumps / loads; the gth-szv GTH_ALIAS basis loader"
  - phase: 09-pbc-foundation
    plan: 01
    provides: "the pyscf-pbc-gto crate scaffold + path-scoped lint exemptions"
provides:
  - "Cell — the periodic analogue of Mole (D-PBC-01: OWNS a Mole, Derefs to it)"
  - "the lattice API: lattice_vectors, vol, reciprocal_vectors, get_abs_kpts, get_scaled_kpts, get_scaled_atom_coords, tot_electrons"
  - "closed-form 3x3 det3 / inv3 / transpose3 (no faer for a 3x3)"
  - "CellBuildArgs / ALattice / LowDimFtType / PseudoData placeholder"
  - "Cell pack / unpack / dumps / loads"
  - "the five shared reference systems (PBC-MASTER-PLAN §9.2) behind the `test-systems` feature"
  - "committed tier-2 upstream reference values for all five systems (D-PBC-19)"
affects: [09-04, 09-05, 09-06, 09-07, 09-08, 09-09, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Cell OWNS a Mole and Deref/DerefMut to it — Rust's answer to upstream's `class Cell(mole.MoleBase)`; ONE Mole build path in the workspace"
    - "lattice stored in Bohr, converted once in build (upstream converts lazily on every lattice_vectors() call)"
    - "sentinel + try_*() accessor for a value a later plan owns: the field carries a sentinel, the accessor returns NotYetImplemented so nothing reads a silent zero"
    - "shared test fixtures live in the producing crate behind a `test-systems` feature + a self dev-dependency; consumers enable the feature instead of copying"

key-files:
  created:
    - crates/pyscf-pbc-gto/src/cell.rs
    - crates/pyscf-pbc-gto/src/types.rs
    - crates/pyscf-pbc-gto/src/dumps_loads.rs
    - crates/pyscf-pbc-gto/src/pseudo/mod.rs
    - crates/pyscf-pbc-gto/src/test_systems.rs
    - crates/pyscf-pbc-gto/tests/common/mod.rs
    - crates/pyscf-pbc-gto/tests/common/systems.rs
    - crates/pyscf-pbc-gto/tests/cell_build.rs
  modified:
    - crates/pyscf-pbc-gto/src/lib.rs
    - crates/pyscf-pbc-gto/Cargo.toml
    - crates/pyscf-pbc-gto/tests/cintx_cross_basis_smoke.rs   # clippy identity_op only
    - crates/pyscf-algebra/src/axpy.rs                        # pre-existing clippy blocker
    - crates/pyscf-algebra/src/scal.rs                        # pre-existing clippy blocker
    - crates/pyscf-algebra/src/gemm.rs                        # pre-existing clippy blocker

key-decisions:
  - "D-PBC-01 realised as ownership + Deref/DerefMut, so cell.nao_nr / cell.natm / cell._env / cell.atom_coords() all resolve to the Mole with zero forwarding boilerplate."
  - "`a` is stored in BOHR; the input-unit scale is applied once in Cell::build. lattice_vectors() is then a pure accessor no call site can forget to scale."
  - "reciprocal_vectors returns Result (the lattice can be singular) and keeps upstream's dimension<3 ORTHOGONALITY ASSERTIONS — it does NOT zero rows (see DEVIATIONS)."
  - "rcut/mesh keep RCUT_UNSET/MESH_UNSET sentinels until plan 09-04; Cell::try_rcut()/try_mesh() turn the gap into a loud NotYetImplemented instead of a silent zero cutoff."
  - "exp_to_discard is implemented by filtering `_basis` BEFORE make_env, not by post-hoc `_bas`/`_env` surgery (see DEVIATIONS)."
  - "serde_json needs `float_roundtrip`: its default f64 parser is off by 1 ULP and silently perturbed the LiF lattice through dumps/loads."
  - "The five reference systems live in `src/test_systems.rs` behind a feature, with a self dev-dependency enabling it for this crate's own tests — §9.2's 'do not redefine them per crate' enforced by construction."

patterns-established:
  - "Tier-2 reference table as a `const [Reference; 5]` next to the constructors, generated once from live PySCF and committed"
  - "A test that pins a KNOWN upstream discrepancy to its single cause, so loose tolerances cannot hide a real bug"

requirements-completed: [PBC-GTO-01, PBC-GTO-02]

# Metrics
duration: ~1h
completed: 2026-08-25
---

# Phase 9 Plan 03: The `Cell` Type

**`pyscf-pbc-gto` now has `Cell` — the periodic analogue of `Mole`. Per D-PBC-01 it OWNS a `Mole` and `Deref`s to it, so `cell.nao_nr`, `cell.natm`, `cell._env` and every `Mole` method work unchanged and there is exactly one `Mole` build path in the workspace. Also lands the full lattice API (ported line-by-line from `cell.py:1811-1975`), `Cell::build` (`cell.py:1593-1810`), `pack`/`unpack`/`dumps`/`loads` (`cell.py:65-155`), and the five shared reference systems of PBC-MASTER-PLAN §9.2 with committed upstream reference values.**

## What Shipped

### Task 1 — `types.rs`

`CellBuildArgs` layers the periodic kwargs on top of `pyscf_gto::MoleBuildArgs`
(carried as a field, not duplicated — D-PBC-01): `a`, `mesh`, `ke_cutoff`, `rcut`,
`precision`, `dimension`, `low_dim_ft_type`, `fractional`, `exp_to_discard`,
`use_particle_mesh_ewald`, `space_group_symmetry`, `pseudo`.
`LowDimFtType { None, InfVacuum }` per §8.1.
`ALattice { Matrix([[f64;3];3]), Flat([f64;9]), Str(String) }` — the string form ports
`cell.py:1878-1882` (`;`, `,` and newlines are separators; exactly nine numbers).

### Task 2 — the `Cell` struct

Exactly the struct of PBC-MASTER-PLAN §8.1 plan 09-03 step 1, `#[derive(Debug, Clone)]`,
plus `Deref`/`DerefMut` to `Mole`. `pseudo: Option<PseudoData>` is declared and stays
`None` until plan 10-01; `PseudoData` is an empty placeholder in `pseudo/mod.rs` so the
field type is already final. The pseudopotential NAME the user asked for is preserved in
`Cell::pseudo_name` so it survives `dumps`/`loads` and plan 10-01 has it to parse.

### Task 3 — the lattice API (`cell.py:1811-1975`)

`lattice_vectors`, `vol` (`|det(a)|`), `reciprocal_vectors(norm_to)` +
`reciprocal_vectors_2pi`, `get_abs_kpts`, `get_scaled_kpts`, `get_scaled_atom_coords`,
`tot_electrons(nkpts)` (`cell.py:957-967`). `det3`/`inv3`/`transpose3` are closed-form
3x3 — the plan is explicit that faer must not be called for a 3x3, and `inv3` returns an
error on a singular lattice rather than infinities that would poison every k-point.

### Task 4 — `Cell::build` (`cell.py:1593-1810`)

Lattice resolved and converted to Bohr once; `dimension == 1` without `inf_vacuum`
rejected (`cell.py:1665-1666`); `dimension > 3` and singular lattices rejected; the
`fractional` coordinate transform (`cell.py:1582-1590`) and the `exp_to_discard` diffuse
filter (`cell.py:1671-1735`) applied to the molecular inputs; the molecular half built
through `pyscf_gto::build_from`; left-handed-lattice warning (`cell.py:1741-1746`);
`_built` set. `rcut`/`mesh` estimation is wired as an assignment from
`estimate_rcut`/`estimate_mesh` so plan 09-04 only has to fill those two bodies.

`dumps_loads.rs` ships `pack`/`unpack`/`dumps`/`loads`, layering the periodic fields on
top of `pyscf_gto::dumps` output exactly as upstream layers on `mole.pack`.

### Task 5 — the five reference systems (§9.2)

`diamond`, `si`, `lif`, `he_fcc`, `graphene` in `src/test_systems.rs`, behind the
`test-systems` feature, plus `all()` and the committed `REFERENCES` table.
`tests/common/systems.rs` re-exports them so this crate's tests read
`systems::diamond()`. Downstream PBC crates get the identical cells with
`pyscf-pbc-gto = { path = "...", features = ["test-systems"] }` in `[dev-dependencies]`.

## Verification Results

```
cargo test -p pyscf-pbc-gto --test cell_build        ✅ 22 passed / 0 failed
cargo test -p pyscf-pbc-gto --all-features           ✅ 23 passed / 0 failed
cargo clippy -p pyscf-pbc-gto --all-features -- -D warnings              ✅ clean
cargo clippy -p pyscf-pbc-gto --all-targets --all-features -- -D warnings ✅ clean
cargo build --workspace                              ✅ clean
cargo run -p xtask --bin check-dependency-wall       ✅ PASS
cargo run -p xtask --bin check-forbidden-paths       ✅ PASS (347 files)
rustfmt --edition 2024 --check <all touched files>   ✅ clean
```

Tier 2 (hard-coded upstream references, D-PBC-19) — generated once from live PySCF
2.12.1 in `.venv` and committed in `test_systems::REFERENCES`:

| system | vol (Bohr^3) | natm | nao_nr |
|---|---|---|---|
| diamond | 76.55488063251218 | 2 | 8 |
| si | 270.1967093603764 | 2 | 8 |
| lif | 110.42101837541341 | 2 | 6 |
| he_fcc | 45.551257834162435 | 1 | 1 |
| graphene | 707.3387370358154 | 2 | 8 |

Tier 1 invariants: `b . a^T == 2*pi*I` to 1e-12 for all five systems;
`get_abs_kpts(get_scaled_kpts(k)) == k` to 1e-12 for 10 fixed-seed pseudo-random k on all
five, plus a Gamma-centred 2x2x2 mesh both directions; scaled atom coords recover the
construction fractions (diamond `0.25`, LiF `0.5`) to 1e-12; `Deref` reaches `nao_nr`,
`natm`, `nbas`, `_env`, `basis_set`, `atom_coords()`, `atom_charges()`, and `DerefMut`
writes land on `cell.mol`; `dumps`/`loads` round-trips `a`, `mesh`, `precision`,
`dimension` (plus the rest of the periodic state AND `_atm`/`_bas`/`_env`) on all five.

## TWO REAL BUGS FOUND BY THE TESTS

**1. `serde_json`'s default f64 parser is not round-trip exact.** The `dumps`/`loads`
test failed on the LiF lattice: `3.8077981598514197` came back as `3.80779815985142`, a
different `f64`. `serde_json`'s default deserializer uses a fast path that can be off by
1 ULP. Fixed by enabling the `float_roundtrip` feature on this crate's `serde_json`
dependency. **This affects `pyscf-gto::loads` too** (same crate, same parser) — feature
unification means the fix applies there as well whenever `pyscf-pbc-gto` is in the build
graph, but `pyscf-gto` should declare the feature itself. Recorded as a carry-over.

**2. `pyscf_core::Unit::Ang.length_in_au()` does NOT match upstream PySCF.**
The constant is `1.8897261339213`; upstream converts Angstrom to Bohr by DIVIDING by
`pyscf/data/nist.py:BOHR = 0.52917721092`, an effective factor of `1.8897261245650618`.
They differ by **4.951e-9 relative**, so every lattice this port builds is 4.95e-9 long
versus upstream and every volume is 1.485e-8 large — 1.14e-6 Bohr^3 on diamond, 1.1e-5 on
graphene. The `pyscf-core` doc comment claiming the value "Matches upstream
`pyscf/data/nist.py BOHR` (verbatim)" is **false**.

This makes the plan's "vol matches to 1e-6 (absolute)" unreachable for any cell larger
than He/fcc regardless of port correctness. Handled by:
* using a RELATIVE tolerance (1e-7 on `vol`, 1e-8 on lattice components), and
* adding `bohr_constant_gap_vs_upstream_is_the_whole_lattice_error`, which asserts every
  lattice component equals `upstream * (1.8897261339213 * 0.52917721092)` to 1e-14 and
  the volume ratio is that ratio CUBED. The entire deviation is pinned to that one
  constant, so a genuine geometry bug cannot hide inside the loosened bound.

Correcting the constant is a workspace-wide change (it moves every molecular geometry and
every regression baseline in v1.0), so it is explicitly NOT done here. When it is, that
test's ratio becomes 1 and the tolerances drop to 1e-12 — the test says so in its docs.

## DEVIATIONS from the plan

**1. `reciprocal_vectors` does NOT zero the non-periodic rows.**
Plan Task 3 says to "Port the `dimension < 3` branch verbatim — it zeroes the
non-periodic rows." Upstream does no such thing. `cell.py:1908-1914` only ASSERTS
orthogonality (`dimension == 1`: all three lattice vectors mutually orthogonal;
`dimension == 2`: `a3` orthogonal to `a1` and `a2`) and then computes the full
`inv(a.T)` regardless. Zeroing rows would break `b . a^T == 2*pi*I`, which is the
acceptance test the same plan mandates. The port reproduces upstream: the same
orthogonality checks, downgraded from a hard `assert` to `debug_assert` + `tracing::warn!`
so a slightly-off lattice warns in release instead of aborting.

**2. `reciprocal_vectors` / `get_abs_kpts` / `get_scaled_atom_coords` return `Result`.**
The plan's signatures return bare arrays. They all invert the lattice, which can be
singular; returning `Result` keeps `inv3`'s singularity check meaningful rather than
propagating infinities into every k-point. `vol`, `lattice_vectors`, `get_scaled_kpts`
and `tot_electrons` need no inverse and keep the plan's infallible signatures.

**3. `exp_to_discard` filters `_basis`, not the projected `_bas`/`_env`.**
Upstream (`cell.py:1671-1735`) filters `_basis` AND then repeats the surgery on the
already-projected libcint arrays, re-invoking `_nomalize_contracted_ao` by hand. This
port filters `_basis` only and lets `pyscf_gto::make_env` do the projection and
normalisation once — the same functions upstream re-invokes, applied once instead of
twice, so the filtered and unfiltered paths cannot drift. Verified: a cutoff below every
`gth-szv` exponent leaves `_env`/`_bas` byte-identical to the unfiltered build, and a
cutoff of 0.5 genuinely shrinks `_env`.

**4. `rcut` / `mesh` are not estimated (owned by plan 09-04).**
Plan Task 4 steps 3-4 say to use "a `todo!()`-free placeholder that returns
`NotYetImplemented { phase: 9 }` and wire it in 09-04". `estimate_rcut` / `estimate_mesh`
are exactly that. `build` does NOT propagate their error — every lattice query (`vol`,
`reciprocal_vectors`, k-points) is fully usable without them, and failing the build would
make the whole plan untestable. Instead the fields keep the `RCUT_UNSET` / `MESH_UNSET`
sentinels and `Cell::try_rcut()` / `Cell::try_mesh()` surface the gap as a loud
`NotYetImplemented`, so no caller can consume a silent zero cutoff.
`unset_rcut_and_mesh_report_the_plan_09_04_gap` asserts this and is the reminder to flip
when 09-04 lands. A user-pinned `rcut`/`mesh` bypasses the estimator entirely and already
works (`user_supplied_rcut_and_mesh_are_honoured`).

**5. `unpack` returns a BUILT `Cell`.** Upstream's `unpack` only updates `__dict__` and
leaves the object un-built. Here the molecular half is rebuilt through
`pyscf_gto::loads`, which re-runs the deterministic `format_basis`/`make_env` pipeline
and therefore always yields a built `Mole`; a half-built `Cell` would be a footgun with
no upside.

Two additions beyond the plan's letter, both small: `CellBuildArgs::fractional` (the plan's
Task-1 field list omits it, but Task 4's `pack` port requires `cldic['fractional']`, and
§9.2's diamond is specified in scaled coordinates), and `Cell::pseudo_name` (so the
pseudopotential input survives `dumps`/`loads` before plan 10-01 can parse it).

## Out-of-scope fixes made to unblock the verification gate

`cargo clippy -p pyscf-pbc-gto -- -D warnings` lints the whole dependency graph, and
three PRE-EXISTING findings in `pyscf-algebra` (from the uncommitted perf work already in
the working tree) failed it. All three are mechanical no-ops, applied and regression-checked
with the full `pyscf-algebra` suite:

* `axpy.rs:43`, `scal.rs:51` — `ABSOLUTE_POS as usize` → `ABSOLUTE_POS` (`unnecessary_cast`)
* `gemm.rs:58` — `(k + TILE_DIM - 1) / TILE_DIM` → `k.div_ceil(TILE_DIM)` (`manual_div_ceil`)

Also `tests/cintx_cross_basis_smoke.rs:71` (from plan 09-01) — `s_ab.values[0 + 1 * nao]`
→ `s_ab.values[nao]` (`identity_op`).

## Carry-overs

- **`pyscf_core::Unit::Ang.length_in_au()` disagrees with upstream by 4.95e-9 relative**
  (see BUGS above). Workspace-wide blast radius; needs its own plan. Until then every PBC
  tier-2 comparison against upstream must use a relative tolerance of 1e-7 or looser, and
  `bohr_constant_gap_vs_upstream_is_the_whole_lattice_error` is the guard that keeps that
  honest. The false doc comment in `crates/pyscf-core/src/mole.rs:70-71` should be
  corrected regardless of whether the value changes.
- **`pyscf-gto` should declare `serde_json`'s `float_roundtrip` feature itself.** Today it
  only gets it through feature unification when `pyscf-pbc-gto` is in the graph; a build
  of `pyscf-gto` alone still has the 1-ULP `loads` bug.
- **`Cell::tot_electrons` returns the ALL-ELECTRON count.** Plan 10-01's GTH
  pseudopotentials replace each `Z` with its valence count (D-PBC-11). The upstream
  pseudopotential targets are already committed in `REFERENCES[i].nelectron_pp`
  (diamond 8, si 8, lif 10, he_fcc 2, graphene 8) so 10-01 has the numbers to hit.
- **`space_group_symmetry` is stored but not acted on** (upstream `build_lattice_symmetry`,
  `cell.py:1552-1580`) — Phase 12.
- **`ew_eta` / `ew_cut` are declared and always `None`** — plan 09-08's `get_ewald_params`.
- **The `dimension <= 2` vacuum-size warning (`cell.py:1748-1753`) is not ported**: it
  needs `rcut`, which plan 09-04 owns. Add it there.
- **`Cell::fromstring` / `fromfile` (POSCAR / CIF) are not ported** — not in this plan's
  scope; `cell.py:1108-1230`.
