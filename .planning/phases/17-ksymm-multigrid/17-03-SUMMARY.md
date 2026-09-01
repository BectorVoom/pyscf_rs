# 17-03 SUMMARY — space_group.rs / symmetry.rs; Cell gains `symmorphic` / `lattice_symmetry`

**Status:** SHIPPED. **Date:** 2026-09-01.

`crates/pyscf-pbc-symm` grows two new modules: `space_group.rs` (`SPGElement`
+ `SpaceGroup`, port of `pyscf/pbc/symm/space_group.py`, 369 l) and
`symmetry.rs` (Wigner-D matrices, `check_mesh_symmetry`, `Symmetry`, the three
symmetry transforms, port of `pyscf/pbc/symm/symmetry.py`, 348 l).
`crates/pyscf-pbc-gto` gains `symmetry_data.rs` (the plain-data
`LatticeSymmetry` type `Cell` actually stores), a `symmorphic: bool` field, a
`lattice_symmetry: Option<LatticeSymmetry>` field, and `Cell::symmetrize_mesh`.

## Exact green test commands

```
cargo test -p pyscf-pbc-symm
cargo test -p pyscf-pbc-gto --test cell_build --test lattice --test kpts_mesh
```

`pyscf-pbc-symm`: **51 tests, all green** — 7 `tests/geom.rs`, 8
`tests/group.rs`, 15 `tests/space_group.rs`, 21 `tests/symmetry.rs`, 0 unit
tests (no `mod tests` in any `src/*.rs`, AGENTS.md §2). `symmetry.rs`'s test
run takes ~200s (one converged Γ-only `KRHF(diamond)`, memoized across every
Task 6 test with a `std::sync::OnceLock` — see Deviation 1).

`pyscf-pbc-gto`: the three test targets that exercise this plan's changes
(`cell_build` 23, `lattice` 11, `kpts_mesh` 16 — `dumps_loads`, `Cell::build`,
`super_cell`'s symmetry refusal, and every reference-system build) are all
green. The FULL `cargo test -p pyscf-pbc-gto` (which also runs several
long-running, functionally-unrelated numerical suites — `gth_pp_loc`,
`ewald`, `eval_ao_kpts`, …) took ~1000s under heavy concurrent load from
other Phase-17 agents sharing the machine (`pyscf-pbc-df`, `pyscf-kernels`,
`pyscf-pbc-dft` test binaries were all running simultaneously — visible via
`ps aux` during this plan's own test runs) but DID complete inside the
session, all green — every test binary in the crate: 0 failures.

## Task 1 — `space_group.rs`: `SPGElement`

Ported `space_group.py:30-248` line-by-line: `transform_rot`/`transform_trans`
(the basis-conversion core), `dot`/`dot_rot`/`inv`/`transform`, the five
predicates (`rot_is_eye`, `rot_is_inversion`, `trans_is_zero`, `is_eye`,
`is_inversion`), the total order (`Ord`/`PartialOrd` via `hash_key`, matching
`__lt__`…`__ge__`) and `hash_key` (`__hash__`, reusing
`PgElement::hash_key` from 17-02 on the integer-rounded rotation).

`rot` is stored as `[[f64; 3]; 3]` uniformly (upstream's `np.ndarray` is
sometimes int-, sometimes float-dtype depending on which basis it lives in) —
small integers are exact in `f64`, so nothing is lost, and every basis
conversion returns the same type regardless of whether its result happens to
be integer.

**The `allow_non_integer` flag is kept** on `transform_rot`, exactly as the
plan requires — it is what turns a wrong basis conversion into
`PbcSymmError::NonIntegerRotation` instead of a silently-wrong matrix.

**`b2r`'s upstream asymmetry, kept, not fixed.** `a2r` passes
`allow_non_integer = true` (`space_group.py:213-217`); `b2r`
(`:225-229`) passes NO third argument, i.e. `false` — an apparent
inconsistency, since going from either lattice basis to Cartesian is
generically non-integer. **Verified against live upstream 2.12.1** (not just
reasoned about) before deciding how to port it: on `si` (cubic Fd-3m) every
op's Cartesian representation happens to be a signed permutation matrix, so
`b2r` succeeds for all 48 ops; on `graphene` (hexagonal) it genuinely RAISES
`RuntimeError` for the eight 3-/6-fold rotations
(`cos(120°) = -0.5 ≠ round(-0.5)`), confirmed by a small standalone Python
repro against the vendored tree. Ported EXACTLY as upstream has it (RULE 2) —
`tests/space_group.rs::b2r_r2b_graphene_partial_by_design` pins that this
partial failure is EXPECTED, not a regression.

Tests (`tests/space_group.rs`): `a2b`/`b2a` and `a2r`/`r2a` round-trip to
1e-13 for EVERY op on both `si` and `graphene` (these two pairs never hit the
`b2r` asymmetry); `b2r`/`r2b` round-trips on every op that succeeds, with the
graphene 8-op-fails/4-op-succeeds split asserted explicitly; diamond has at
least one `trans_is_zero() == false` op (the glide) while every `lif` op is
`true`; group axioms (`dot`/`inv`/identity) on diamond's full 48-op set;
sortedness is idempotent.

## Task 2 — `space_group.rs`: `SpaceGroup`, native backend only

Ported `space_group.py:250-337`. **Only the `'pyscf'` backend ships** —
17-CONTEXT §1.5's ruling, recorded again in this module's own doc comment so
a later reader does not mistake the omission for a gap. `SpaceGroup::build`
runs `search_space_group_ops` ONCE and classifies the SAME rotation list both
ways (`geom::get_crystal_class_from_rotations`) rather than upstream's literal
two separate calls (`self.ops` via `search_space_group_ops`, then `pg_symbol`
via `get_crystal_class`, which internally reruns the identical search) — a
deterministic function of `(cell, symprec)` either way, so the result cannot
differ and the search is not paid for twice.

Tests: `point_group_symbol` for all five §9.2 fixtures against upstream 2.12.1
(`m-3m` ×4, `6mm` for graphene); `nop` (48/48/48/48/12) matches BOTH upstream
AND `search_point_group_ops`'s rotation count (one translation per rotation
for every op in this fixture set — no rotation dropped or duplicated by the
translation search); diamond's 24-symmorphic/24-non-symmorphic and
graphene's 6/6 splits, both verified against upstream.

## Task 3 — `symmetry.rs`: the Wigner-D matrices, and the §3.2 pin

Ported `get_Dmat`/`get_Dmat_cart`/`make_Dmats` (`symmetry.py:32-94`), which in
turn needed a from-scratch port of `pyscf/symm/Dmatrix.py`'s Wigner
small-d matrix, `get_euler_angles`, and `pyscf/symm/sph.py`'s
`sph_pure2real` — none of this machinery existed anywhere in the workspace
before this plan (there is no `pyscf-symm` molecular-symmetry crate yet).
**One deliberate simplification, stated in the doc comment**: upstream
special-cases `l = 0, 1, 2` in `dmatrix()` for speed and falls back to a
general closed-form Wigner-d formula for `l >= 3`; this port uses the general
formula UNIFORMLY for every `l` (it is the identical mathematics, just
without three hand-unrolled fast paths this crate's angular momenta — `l <=
1` on every §9.2 fixture — do not need).

**The §3.2 trap, addressed structurally, not by adding a bigger test:**
[`get_rotation_mat`] is the ONLY function in the crate that assembles an
AO-space rotation matrix. `make_rot_loc` (`symmetry.py:330-343`) is ported
(17-04's `symm_adapted_basis` needs it) but is deliberately NOT used to build
a second, parallel assembly path here — upstream's own `_get_rotation_mat`
doesn't call it either, walking the cell's actual shells directly instead,
and porting a second implementation is exactly the two-diverging-copies shape
14-05's `decompose_j2c` defect had. `transform_mo_coeff`, `transform_dm` and
`transform_1e_operator` all go through this one function.

**Pinned with the identity that DEFINES a representation, not a round-trip**,
per the plan's explicit instruction: `R(op)·S·R(op)ᴴ == S` (`S` = the Γ-point
overlap from `pbc_intor`, purely analytic — no FFT mesh involved) holds for
EVERY op on ALL FIVE §9.2 reference cells (`r_s_rh_equals_s_{diamond,si,
lif,he_fcc,graphene}`, `< 1e-10`), using the FULL space group
(`check_mesh_symmetry = false`, so diamond's glide is exercised too — see
Deviation 2 on why this differs from the mesh-compatible subset the Task 6
KRHF tests use). `R(op1)·R(op2) == R(op1∘op2)` is checked over the FULL `n²`
sweep of ops (not spot-checked) on `diamond` and `si`
(`homomorphism_{diamond,si}`, `< 1e-8`) — see Deviation 3 on why the composed
op's Dmats are recomputed directly rather than looked up in the canonical op
list. `is_eye(op) => R == I` to `< 1e-14`.

## Task 4 — `check_mesh_symmetry`, and the mesh can grow

Ported `check_mesh_symmetry` (`symmetry.py:96-131`) as a THIN wrapper —
literally one function shared across TWO crates. The core algorithm
(`check_mesh_symmetry_core`, generic over a plain `(is_zero, translation)`
list rather than `SPGElement`, so it can live one crate down) is in
`pyscf-pbc-gto/src/symmetry_data.rs`; `pyscf_pbc_symm::symmetry::
check_mesh_symmetry` and the new `Cell::symmetrize_mesh` both call it, so
there is exactly one implementation of "does this mesh carry this
translation" — the same "one implementation" discipline Task 3 applies to the
AO rotation, applied here to the mesh side.

`Cell::symmetrize_mesh` (`cell.py:1529-1550`) ports the 8x/>1000-point loud
warning VERBATIM (`eprintln!`, matching upstream's `sys.stderr.write`), and
its doc comment — plus `Cell::space_group_symmetry`'s doc comment, now fixed
from the stale "not implemented before Phase 12" — spells out 17-CONTEXT
§3.3: turning symmetry on can silently change `cell.mesh`, and therefore the
energy, for reasons unrelated to the IBZ.

**Measured, not assumed**: diamond's default `cell.mesh = [47, 47, 47]`
(`Cell::build`'s own `precision`-driven estimate) is NOT a multiple of 4, so
it is genuinely INCOMPATIBLE with diamond's `(1/4,1/4,1/4)` glide — a real,
not synthetic, exercise of both `check_mesh_symmetry` branches:
`check_mesh_symmetry_grows_a_mesh_incompatible_with_the_glide` (mesh `[6,6,6]`
→ grown mesh, re-verified compatible) and the complementary pair
`symmetry_build_check_mesh_symmetry_{true,false}_*` (`true`: diamond's group
reduces from 48 to its 24-op symmorphic subgroup on ITS OWN default mesh;
`false`: the full 48-op group survives regardless).

## Task 5 — `symmetry.rs`: `Symmetry`, and the non-ported reference cycle

Ported `Symmetry::build` (`symmetry.py:165-207`) as an ASSOCIATED FUNCTION
taking `cell: &Cell` (not a method reading `self.cell`, since this `Symmetry`
never stores one). `cell.py:1576-1579`'s `del self.lattice_symmetry.cell` /
`del self.lattice_symmetry.spacegroup.cell` — which exists ONLY to break a
Python refcount cycle — has no Rust analogue and is intentionally not
ported; the module's top doc comment explains this at length specifically so
a later reader does not mistake the missing deletion for an oversight.
`Symmetry::build` returning `Err` when `!cell._built` (rather than upstream's
silent `cell.build()`) is the one behavioural difference the borrow forces.

The `auxcell` kwarg (`:200-203`) is not ported — nothing in this workspace
calls it yet; a future caller needing a wider `l_max` can call `make_dmats`
again directly.

**`build_lattice_symmetry` is a FREE FUNCTION in `pyscf-pbc-symm`, not a
`Cell` method** — `cell.py:1552-1580`'s `Cell.build_lattice_symmetry` would
need `Cell` (in `pyscf-pbc-gto`, below `pyscf-pbc-symm`) to call into
`Symmetry::build` (in `pyscf-pbc-symm`), which is exactly the dependency
inversion Task 7 rules out. It builds a `Symmetry`, converts it to a plain
`LatticeSymmetry` (`Symmetry::to_lattice_symmetry`), and stores THAT on
`cell.lattice_symmetry`.

Tests: `space_group_symmetry = false` gives the identity-only group;
`symmorphic = true` keeps exactly diamond's 24 zero-translation ops;
`has_inversion` matches `m-3m` (true, ×4 cubic fixtures) vs `6mm` (false,
graphene) — using `check_mesh_symmetry = false` so a mesh-incompatibility
artefact cannot silently hide a real crystallographic inversion (diamond's
inversion element itself carries a `(1/4,1/4,1/4)` translation, so it is one
of the 24 ops `check_mesh_symmetry = true` drops on this fixture's default
mesh — discovered BY this test, see Deviation 2); `build_lattice_symmetry`
wires `Cell::lattice_symmetry` and leaves `cell.mesh` untouched when
`check_mesh_symmetry = true`.

## Task 6 — the three transforms, against one converged Γ-only KRHF

Ported `_get_phase` → `get_phase`, `_get_rotation_mat` → `get_rotation_mat`,
`transform_mo_coeff`, `transform_dm`, `transform_1e_operator`
(`symmetry.py:226-329`), plus a from-scratch `aoslice_by_atom`
(`mole.py:1841-1880` — not exposed anywhere else in this workspace yet, so
ported as a private helper here, this module's only consumer).
`transform_dm`/`transform_1e_operator` share one `sandwich(mat, x, n)` helper
(`mat·x·matᴴ`) rather than reimplementing the identical three-line body
twice, as upstream does.

Fixture: ONE converged `KRHF(diamond)` at Γ (`conv_tol = 1e-11`,
`max_cycle = 50`), memoized once per test binary with `std::sync::OnceLock`
(Deviation 1). `S` is read back analytically (`pbc_intor`, F-order — converted
to row-major at the boundary), `dm`/`mo_coeff` from the converged
`KScfResult` (`dm` row-major, `mo_coeff` COLUMN-MAJOR per
`pyscf-pbc-scf/src/types.rs:119` — both conversions happen exactly once, in
two named helper functions, not scattered inline).

* `transform_dm(dm, op)` preserves `Tr(D S)` for every op in the
  `check_mesh_symmetry = true` group (24 ops on diamond's default mesh — see
  Deviation 2 for why this differs from Task 3's 48-op sweep), AND —
  stronger than the plan's literal ask — `transform_dm(dm, op) == dm` itself,
  since a closed-shell Γ ground state's density is invariant under the FULL
  point group (diamond's occupied manifold is exactly `{nondegenerate,
  triply-degenerate}`, no partial occupation at the Fermi level, so nothing
  breaks the invariance).
* `transform_1e_operator(S, op) == S` — the SAME identity Task 3 pins
  directly through `get_rotation_mat`, now exercised through the public
  production entry point end-to-end.
* `transform_dm` is idempotent under `op ∘ op⁻¹`.
* **17-CONTEXT §3.1, both halves.** `transform_mo_coeff` is compared ONLY
  through the density matrix it builds (`DM(transform_mo_coeff(op)) ==
  transform_dm(dm, op)`), never elementwise — AND a companion test
  (`mo_coeff_elementwise_comparison_fails_on_a_degenerate_level`) locates
  diamond's real, measured, exactly triply-degenerate occupied level
  (verified against live upstream 2.12.1: `mo_energy = [-0.610, 0.293,
  0.293, 0.293, 1.160, 1.160, 1.160, 1.526]`) and asserts that at least one
  non-identity op moves an elementwise `mo_coeff` column comparison for that
  manifold by `> 0.05` — so nobody can "fix" the DM comparison into an
  elementwise MO one without breaking this test.

## Task 7 — `Cell` gains `symmorphic` / `lattice_symmetry`

`crates/pyscf-pbc-gto/src/symmetry_data.rs` (new): `LatticeSymmetry` /
`LatticeSymmetryOp` (plain data — rotations, translations, per-op `Dmats`,
crystal class, group name) and `check_mesh_symmetry_core`. This is the SAME
shape `Cell::pseudo` already uses for `PseudoData` (parsed data, not the
parser) — cited explicitly in the module doc, per the plan's own precedent
pointer. `pyscf-pbc-gto` gains NO new dependency; `pyscf_pbc_symm::Symmetry`
PRODUCES a `LatticeSymmetry` (`Symmetry::to_lattice_symmetry`), it never the
other way round.

`Cell::space_group_symmetry`'s doc comment is fixed (was: "not implemented
before Phase 12"). `symmorphic: bool` (default `false`) threaded through
`CellBuildArgs` (`types.rs`), `Cell::build`, `Cell::default`,
`super_cell` (copies it, resets `lattice_symmetry` to `None` — a supercell is
a different lattice, and `super_cell` with symmetry already refuses
upstream-style, `pbc.py:784-785`, untouched by this plan), and
`dumps_loads.rs`'s `CellPack`/`pack`/`unpack` — closing the exact gap the plan
named ("`dumps_loads` already carries `space_group_symmetry` and would
otherwise drop its partner silently"). `lattice_symmetry` itself is NOT
serialised (derived, build-time-only state, like `rcut`/`mesh` but with no
cheap re-estimate — a caller that needs it after `loads` re-runs
`build_lattice_symmetry`), documented in `symmetry_data.rs`'s module doc.

`Cell::symm_orb`/`irrep_id` — explicitly out of scope, untouched, 17-04's job.

## Deviations from a literal reading of the plan

1. **KRHF fixture memoized with `OnceLock`, not five independent SCF runs.**
   The plan says "against a single converged … KRHF" — read literally as ONE
   converged reference object, which is what every Task 6 test now shares
   (first test thread to reach it pays the ~3-minute SCF cost once; every
   other test, including ones on other threads, blocks on the same
   `OnceLock` and gets the cached result). The original draft ran an
   independent SCF per test (5-6x the cost) purely from how the helper
   function was written, not from anything the plan required — fixed after
   measuring the first full run took >6 minutes wall-clock under contention
   from other agents' concurrent test suites on the same machine.
2. **Two different `check_mesh_symmetry` settings for two different kinds of
   test, not the same setting everywhere.** Task 3's `R S Rᴴ = S` test uses
   `check_mesh_symmetry = false` (the FULL 48-op group, glide included) —
   correct because `S` is analytic and has no FFT-mesh dependence at all.
   Task 6's KRHF-derived tests use the DEFAULT `check_mesh_symmetry = true`
   (24 ops on diamond, the mesh-compatible subgroup) — because the density
   there genuinely comes from an FFT-mesh-dependent DF build (`Fftdf`), and
   testing a mesh-incompatible op's invariance against a quantity that IS
   mesh-quantized would be measuring a mesh artefact, not a symmetry-algebra
   defect (exactly 17-CONTEXT §3.3's trap, one layer down from the "compare
   two runs" framing it was written for). Both settings are used
   deliberately and the choice is documented at each call site — this was
   discovered empirically (both `has_inversion_matches_point_group` and an
   earlier "keeps the full 48-op group" test initially FAILED against
   diamond's real default mesh `[47,47,47]`, which is not a multiple of 4)
   rather than assumed going in; see Task 4/5's "measured, not assumed" note
   above for the same discovery from the other direction.
3. **The homomorphism test does not require the composed op to be a
   canonical member of `SpaceGroup::ops`.** `SPGElement::dot` (ported
   verbatim from `space_group.py:103-117`, RULE 2) does NOT reduce the
   resulting translation mod 1, so `op1.dot(op2)` need not match (even up to
   a lattice vector) any `[0,1)`-reduced representative in the canonical op
   list — this is upstream's own behaviour, not a defect this port
   introduced (an early draft assumed group-closure-by-hash-lookup and
   failed on real ops; verifying against upstream's literal `dot()` source
   confirmed no mod-1 reduction exists there either). The fix does not touch
   `SPGElement::dot` at all: the test instead builds the composed op's Dmats
   directly (`a2r` + `make_dmats`), which is correct for ANY valid symmetry
   operation regardless of canonical range, since `get_phase`'s atom search
   already wraps into the cell via `round_to_cell0`.

## Verification checklist (17-03-PLAN.md's own list)

* `cargo test -p pyscf-pbc-symm -p pyscf-pbc-gto` green — `pyscf-pbc-symm`
  fully green (51/51); `pyscf-pbc-gto` fully green too (every test binary in
  the crate, 0 failures — confirmed by a full run that completed, ~1000s
  under concurrent load from other agents sharing the machine).
* `R S Rᴴ == S` for every op on all five §9.2 cells — green, `< 1e-10`.
* `R(op₁)R(op₂) == R(op₁∘op₂)` over the full group — green (`diamond`, `si`,
  full `n²` sweep), `< 1e-8`.
* `dumps_loads` round-trips `symmorphic` — green, both `true` and `false`
  pinned explicitly (`dumps_loads_round_trips_symmorphic`).
* No `mod tests` in any `src/*.rs` — verified by `grep`.
