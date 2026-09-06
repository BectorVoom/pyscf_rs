---
gsd_state_version: 1.0
milestone: v2.0
milestone_name: Periodic Boundary Conditions
status: in_progress
last_updated: "2026-09-05T12:00:00.000Z"
last_activity: 2026-09-05
progress:
  total_phases: 5
  completed_phases: 5
  total_plans: 48
  completed_plans: 48
  percent: 100
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-05-09)

**Core value:** Run mainstream molecular ground-state quantum chemistry (HF, DFT, MP2, CCSD, gradients) 2–5× faster than current PySCF + C extensions, with bit-exact agreement on regression tests, and zero C/CMake/libcint dependency hell at install time.
**Current focus:** Phase 16 (periodic CC/CI) — **IN PROGRESS, six of
fourteen plans complete and measured as of 2026-09-06.** `KRCCSD` ships and
matches upstream PySCF 2.12.1 to `6.560e-9`; `KCCSD(T)`'s RHF half ships with a
`8.363e-13` fast-vs-slow agreement. Phase 15, its hard blocker, is **CLOSED as
of 2026-09-05**.

## Current Position

**Phase 16 IN PROGRESS, 2026-09-06.** `.planning/phases/16-periodic-cc-ci/16-VERIFICATION.md`
is the authority. **Nine plans complete and measured** (16-01, 16-02, 16-03,
16-04, 16-05, 16-06, 16-07, 16-08, 16-13 and this verification), four not
started (16-09/10/11, 16-12), each recorded with its reason and its unblocking
work rather than silently dropped.

* **`KRCCSD e_corr` matches upstream to `6.560e-9`** — diamond `gth-szv`
  `[1,1,2]`, FFTDF, `cell.mesh = [15,15,15]`, `conv_tol = 1e-9`, an order and a
  half inside 16-01's measured `1e-7` gate. `init_amps emp2` `3.494e-9`; the
  seven `_ERIS` blocks `1.21e-8 … 2.34e-7`; all nine `cc_*` intermediates
  `3.5e-9 … 2.28e-7`; `update_amps`' `t1new`/`t2new` `1.84e-8`/`7.01e-8`. The
  block-level residuals ARE the FFT integral-transform floor at that mesh —
  upstream's own symmetry-loop and all-triples paths differ by `1.32e-7` on the
  same fixture.
* **`KCCSD(T)` fast vs slow: `8.363e-13` relative** (upstream's own two
  implementations: `2.946e-13`), blocking-invariant to `2.17e-19`, and
  `3.286e-10` from upstream. `kccsd_t_rhf_slow.py` — the file
  `PBC-MASTER-PLAN §8.8`'s table omits entirely — was ported FIRST, as the only
  oracle-free reference the blocked path has.
* **`KGCCSD e_corr` matches upstream to `2.066e-9`** in 19 cycles; the seven
  antisymmetrised `<pq||rs>` blocks to `2.42e-8 … 4.68e-7`. Two defects the
  gates caught: `cc_Wovvo` gathered the WRONG K-AXIS (`oovv[:,km,ke]` where
  `oovv[km,:,ke]` was meant — SAME SHAPE, so only a numerical comparison finds
  it: `t1new` still matched to `1.5e-8` while `t2new` was `7.7e-4` out), and the
  kernel had no DIIS (`e_corr` already `3.7e-9` from upstream but not converged
  in 50 cycles — which is why the test asserts `converged`, not just the
  number; with DIIS, 19).
* **`KCCSD(T)` complete, RHF and spin-orbital.** Spin-orbital vs upstream
  `3.459e-11`; spin-orbital vs RHF **`6.9e-12`**, which is **40× tighter than
  upstream's own two routes** (`2.86e-10`).
* **`KCIS` Davidson roots match upstream to `5.48e-10`**, dense to `1.84e-9`.
  More tellingly: at `kshift = 0` upstream's OWN Davidson and dense paths
  differ by `2.51e-3` — its Davidson converges to a different state — and this
  port **reproduces that spread to `1.29e-9`**. The two implementations agree
  on WHICH state the Davidson finds, which is a stronger statement than either
  root comparison alone.
* **The oracle-free gates**: incore vs spilled `_ERIS` BIT-IDENTICAL with the
  tier asserted on each side (D-PBC-29 clause 4, and the test fails if a
  fixture silently stays incore); `symm_map` vs all-triples `7.93e-7` with
  `vvvv` exactly `0e0`; `t1`/`t2`/`e_corr` bit-reproducible; `init_amps`' MP2
  vs Phase 15's `KMP2` `2.166e-10`; the arena charges exactly its derived byte
  count.
* **THE GATE WAS MEASURED FIRST, and five gates were found tighter than the
  thing they gate** — `ROADMAP`'s `1e-14`, `§7`'s `1e-8`, 16-05 test 5's
  bit-identity, 16-05 test 3's `1e-12`, and G4's own `1e-13` (below upstream's
  `2.95e-13`). None was loosened to pass a test; each was corrected by the
  measurement that proves it, and the old numbers are struck through, not
  deleted. `16-VERIFICATION §5`.
* **D-PBC-29 clause 3 is AMENDED**: `symm_map` is a measured **2.10×**, not the
  derived `~4×` — 176 orbit representatives for 512 triples at 2×2×2, and
  `vvvv` is built by `ao2mo_7d` in BOTH paths so it saves nothing. The clause
  stands at `~2×`. Clause 4's `_mem_usage` over-estimate is CONFIRMED at a
  measured `9.143×`/`6.058×`.
* **Three upstream anchor sets are excluded from every gate, for cause**:
  `test_krccsd.py::test_frozen_n3` FAILS on the vendored 2.12.1 tree, and every
  `cu_metallic` anchor sits in a test upstream itself disabled with
  `@unittest.skip('Results not match')`.
* **A finding that belongs to another phase**: this port's `KRHF` and
  upstream's differ by `1.348e-5 Ha` on diamond `[1,1,2]` with `cell.mesh`
  PINNED at `[15,15,15]`, while Phase 15 measured `4.772e-11` on the same cell
  at the DEFAULT mesh. Not the lattice sums (`rcut` and `nimgs` agree) — the
  FFT-grid-evaluated part of the mean field at a coarse grid. Every Phase-16 CC
  gate therefore runs on upstream's own mean field and prints the residual
  beside the result.
* **17-09's Phase-16 dependency is satisfied** (`KRCCSD` ships, oracle-green)
  but 17-09 is not thereby unblocked: its target is the k-SYMMETRY adapters,
  which also need Phase 17's `KPoints` IBZ machinery.

### The plan set as written, for reference

**Phase 16 PLANNED + REVIEWED, 2026-09-02 (no Rust written).**
`.planning/phases/16-periodic-cc-ci/` — `16-CONTEXT.md`, fourteen plan files
(`16-01`..`16-14`) and `16-REVIEW.md`, the speed + memory pass that produced
**D-PBC-29**. `PBC-MASTER-PLAN §8.8` sized the phase at ten plans and was
wrong about the starting state in **seven** ways, all found before any code:

* **HARD-BLOCKED on Phase 15.** All nine k-point CC/CI modules import
  `padding_k_idx`/`padded_mo_coeff`/`padded_mo_energy`/`get_nocc`/`get_nmo`/
  `get_frozen_mask` from `pbc.mp.kmp2`/`kump2` (nine file:line citations in
  `16-CONTEXT §1.1`) and `crates/pyscf-pbc-mp` is a **13-line stub**. Same
  block that stopped 17-09. Waves 0 (16-01/02/03) have no such dependency and
  start immediately; wave 1 onward defers explicitly rather than writing a
  second padding implementation — the convention is virtual-**TOP**-aligned
  (`kmp2.py:262-263`) and two of them is how a plausible wrong number ships.
* **Four molecular prerequisites costed at zero and absent**: `cc/gccsd.py`,
  `cc/rccsd.py`, `cc/eom_rccsd.py`'s `EOM`/`EOMIP`/`EOMEA` bases, and
  **`lib.davidson_nosym1`** — an iterative NON-symmetric Davidson required at
  `eom_kccsd_ghf.py:128`/`:1352` and `kcis_rhf.py:97`, of which this workspace
  has **none** (`eigh_gen` is symmetric; 17-02's `faer` path is dense). Four
  plans are dead without it, so it is its own wave-0 plan (16-03). Each
  molecular base is ported NARROWLY, to the entry points actually consumed.
* **The `§8.8` Reuse note cannot be followed literally.** `WorkspacePool` is
  in `pyscf-runtime`, not `pyscf-ccsd`; `pyscf-ccsd` has **zero** complex
  arithmetic; and the pool is f64 all the way down (`shape_bytes = product*8`
  `:278-280`, `InMemory(Box<[f64]>)`, `as_slice -> Vec<f64>` `:397`). Every
  k-point CC tensor is `complex128`.
* **`§8.8` builds the EOM base class LAST** — `eom_kccsd_rhf.py:25` and
  `eom_kccsd_uhf.py:29` both inherit from `eom_kccsd_ghf`. GHF ships first.
* **EOM-EE does not exist for UHF and is SINGLET-ONLY for RHF** — no `EOMEE`
  class in `eom_kccsd_uhf.py` at all, `_IMDS.make_ee` (`:1120`) raises;
  `EOMEETriplet` (`eom_kccsd_rhf.py:1483`) / `EOMEESpinFlip` (`:1489`) are
  shells whose only body is `vector_size -> None`. **`ROADMAP.md`'s own
  "IP/EA/EE (RHF/UHF/GHF)" claim was the error**; the port ships upstream's
  surface and upstream's refusals with oracle-gated tests (RULE 2, the
  `15-CONTEXT §1.3` discipline).
* **`pbc/ci/cisd.py` is a Γ-only shim** over molecular RCISD/UCISD/GCISD
  (`:24`, `:47`), and this port has **no molecular CI crate**. Deferred
  explicitly (16-13 Task 1). `kcis_rhf.py` — k-point CI **singles** — ships.
* **`kccsd_t_rhf.py:236` runs on a C kernel** (`CCsd_zcontract_t3T`, 24 raw
  pointers). Ported to Rust and gated against **`kccsd_t_rhf_slow.py`, the one
  `pbc/cc` file `§8.8`'s table omits entirely**.

**The gate was the fourth instance of the project's recurring defect.**
`ROADMAP` said **1e-14**, `PBC-MASTER-PLAN §7` said **1e-8** for the same
number, neither measured — and upstream's own suite asserts `KRCCSD` `e_corr`
at **6 decimals** (`test_krccsd.py:180`/`:226`/`:232`/`:338`/`:356`) and EOM
roots at **3** (`:359-366`). 1e-14 is eight orders tighter than upstream's own
tests. **16-01 measures the floor before the gate is written**, per DF route
(`kccsd_rhf.py:37` branches on `isinstance(with_df, GDF)`). Both old numbers
are struck through, not deleted, in all four documents.

**D-PBC-29 (`16-REVIEW.md §6`), four clauses, all derived with the line that
proves them** — `D-PBC-28` was already Phase 15's, so this is 29:
(1) the complex arena is a **new type**, never a reinterpretation of
`Box<[f64]>` — that would halve the number reaching the HARD refusal; the f64
pool also **copies on every `as_slice`** (`:397`) and **holds its global mutex
across the caller's closure** (`:461-483`), which would cap the phase at one
core; (2) contractions are host rayon loops with `oracle_*`, not
`zgemm_dense` (6-12× slower, 1.35e-10 off), every site naming its primitive;
(3) **`symm_map` is a genuine ~4×** that `§8.8` never mentions — and
`15-REVIEW D-15-R-04`'s ≤2× ruling explicitly does **not** carry over, because
KCCSD's `_ERIS` wants the full general block where KMP2 wanted only `(ov|ov)`;
(4) storage tiers come from an exact per-tensor byte count, never upstream's
`_mem_usage`, which over-estimates **6.2-9.1×** and would refuse jobs that fit.
Derived `vvvv`: `gth-szv` 2×2×2 **2.0 MiB** → `gth-dzvp` 3×3×3 **68.7 GiB**,
×16 for KGCCSD. **Every §9.2 fixture is `gth-szv`**, so a gate on those alone
would ship the HDF5 spill path never once executed — 17-12's exit-137 shape —
hence 16-01 Task 5 finds a tier-crossing fixture and 16-05 test 4 asserts which
tier each side used. On EOM the wall is the **Davidson subspace** (`2·max_space
·nroots` vectors), 16 MiB on `gth-szv` 2×2×2 but **5.4 GiB** for EA on
`gth-dzvp` 3×3×3; on (T) it is streaming — `t3` is per-k-triple and the
blocking IS the algorithm.

**Nothing is implemented.** Phase 15 remains the next phase in sequence and is
Phase 16's blocker.

**Phase 17 plan 01 — MEASURED, 2026-09-01 (no Rust written).**
`.planning/phases/17-ksymm-multigrid/17-01-PLAN.md` ran upstream PySCF 2.12.1
against itself to replace Phase 17's two unmeasured, mutually-contradictory
gate statements (`ROADMAP` said 1e-14, `PBC-MASTER-PLAN §7` said 1e-9, upstream's
own test suite asserts 5e-8…5e-9) with five measured gates. Full detail:
`.planning/phases/17-ksymm-multigrid/measurements/README.md`,
`17-01-SUMMARY.md`. Headline numbers, now also in `ROADMAP.md` and
`PBC-MASTER-PLAN.md §7`/`§8.9`:

* **Gate A** (IBZ integers, exact, no oracle): `145/145/245/408/816/2052`,
  reproduced bit-for-bit on upstream's own Si cell AND on this repo's `si`/
  `diamond` fixtures at different lattice constants — confirms the integers
  depend on the space-group TYPE, not the lattice constant.
* **Gate B** (transforms vs one converged SCF): floor is set by
  `cell.precision` (integral screening), not `conv_tol` — ≤1e-9 at PySCF's
  default, ≤1e-13 when `cell.precision=1e-13` on both sides. This CORRECTS
  17-CONTEXT's original "≥1e-12 unconditionally" guess.
* **Gate C/D** (energy, symmetry vs no-symmetry, mesh pinned): FFTDF
  ≤5.985e-11, GDF ≤3.433e-09 on `si`/`diamond`, both far inside upstream's
  own 5e-8/5e-7.
* **Gate E** (multigrid vs reference `numint`): v1 exact to 1e-12…1e-14; v2
  carries a MESH-INDEPENDENT ~2e-8(diamond)/1.5e-7(si) floor — a definitional
  gap, the Phase-17 analogue of Phase 14's GDF-vs-RSDF 4.5e-6.
* **New, unpredicted findings**: upstream's OWN multigrid (v1 and v2) measured
  SLOWER than reference `numint`/FFTDF on these reference systems (0.18x-0.49x)
  — 17-11/17-12 must not assume a speed win without re-measuring at their own
  target scale. A full-BZ-vs-IBZ-subset `get_jk` wall-clock bound for 17-07's
  D-PBC-26 fast path: 223x (FFTDF) / 40x (GDF) on `si [4,4,4]`, both well above
  the naive `nkpts/nkpts_ibz=8x` estimate.

**Carry-overs for 17-13** (resource-scoped by this measurement's time budget,
not by any found limitation): `lif`/`graphene` Gate C/D at PRODUCTION mesh
(this plan only reached a mesh-capped, degraded number for both — `lif`
1.461e-04, `graphene` non-convergent, both mesh-cap artefacts, not symmetry
defects); `lif`/`he_fcc`/`graphene`'s mesh-unpinning demonstration at true
default mesh; `diamond`'s remaining 4 cells of the full 2×2×2×2 grid (already
strongly suggested to land at the same scale as the 4 cells measured).

**Phase 17 plan 02 — SHIPPED, 2026-09-01.**
`crates/pyscf-pbc-symm` grows from a 13-line stub to `geom.rs` (`search_point_group_ops`
/ `search_space_group_ops` / `get_crystal_class`, port of `pyscf/pbc/symm/geom.py`),
`tables.rs` (`CrystalClass`/`LaueClass`/`SchoenfliesNotation` as `const` data,
port of `tables.py`) and `group.rs` (`PGElement`/`FiniteGroup`/`PointGroup`/
`Representation`, port of `group.py`). Full detail:
`.planning/phases/17-ksymm-multigrid/17-02-SUMMARY.md`.

* **D-PBC-25 recorded** (`PBC-MASTER-PLAN.md` §6, before any code): `KPoints`
  (17-05) will live in `pyscf-pbc-symm`, not `pyscf-pbc-lib`, by composition
  over a `Symmetry`; `pyscf-pbc-df`/`pyscf-pbc-dft` gain a `pyscf-pbc-symm`
  dependency (verified acyclic).
* **Measured, not guessed**: point-group op counts are 48 (`m-3m`/`Oh`) for
  `diamond`/`si`/`lif`/`he_fcc` and a simple-cubic control, but **12** (`6mm`/
  `C6v`) for `graphene` — the plan's own pre-measurement estimate was 24; the
  `dimension = 2` low-dim filter (no inverting the non-periodic axis)
  provably restricts the count to 12, and 12 is what is pinned.
  `diamond`/`si` are measured non-symmorphic (no zero-translation
  representative at the natural origin for every rotation) while
  `lif`/`he_fcc` are symmorphic — matching known crystallography.
* **`character_table`'s Burnside eigendecomposition needed a general
  (non-symmetric) complex eigensolver** `pyscf-algebra` doesn't expose
  (ALG-05 covers only symmetric `eigh_gen`). Added a direct, host-only `faer`
  dependency (`Eigen::new_from_real`) plus `num-complex` — not an ALG-06 wall
  violation since this crate touches no cubecl/device path. Upstream's
  `np.random.rand` draw (a generic tie-breaker, not a determinism
  requirement) is replaced with a fixed-seed PRNG so this port's own tests
  are reproducible.
* **15/15 tests green** (`cargo test -p pyscf-pbc-symm`), all oracle-free
  (group axioms, Latin-square multiplication table, Burnside orthogonality,
  `chi_to_rep(rep_to_chi(r)) == r`) plus `PointGroup::group_name` cross-checked
  against known crystallography on all five §9.2 fixtures.
  `check-orphan-modules` and `check-dependency-wall` both PASS.
* **Deviation, reported not silently worked around**: the literal
  `cargo clippy -p pyscf-pbc-symm -- -D warnings` (no `--no-deps`) fails
  transitively through a PRE-EXISTING `pyscf-algebra` lint
  (`clippy::chunks_exact_to_as_chunks` at `complex.rs:49`, reproduced on
  unmodified `main` and on the unrelated already-shipped `pyscf-pbc-lib`) —
  a repo-wide toolchain-drift issue, not introduced by this plan and out of
  its scope to fix. Verified instead with `--no-deps --all-targets`, which is
  clean. See `17-02-SUMMARY.md` Deviation 1 for the full reproduction.

**Phase 17 plan 03 — SHIPPED, 2026-09-01.**
`.planning/phases/17-ksymm-multigrid/17-03-PLAN.md` — `space_group.rs`
(`SPGElement`/`SpaceGroup`, port of `space_group.py`) and `symmetry.rs`
(Wigner-D matrices, `check_mesh_symmetry`, `Symmetry`, the three symmetry
transforms, port of `symmetry.py`) in `crates/pyscf-pbc-symm`; `Cell` gains
`symmorphic`/`lattice_symmetry` in `crates/pyscf-pbc-gto`. Full detail:
`17-03-SUMMARY.md`.

* **51/51 tests green** (`cargo test -p pyscf-pbc-symm`: 7 geom + 8 group + 15
  space_group + 21 symmetry). `cargo test -p pyscf-pbc-gto` is fully green
  too — every test binary in the crate, 0 failures (confirmed by a full run
  that completed inside the session, ~1000s under heavy concurrent load from
  other agents' test suites sharing the machine).
* **The §3.2 AO-rotation trap addressed structurally**: `get_rotation_mat` is
  the ONLY AO-rotation assembly in the crate; `transform_mo_coeff`/`_dm`/
  `_1e_operator` all go through it, and it is pinned by `R(op)·S·R(op)ᴴ = S`
  (S = the analytic Γ overlap) on EVERY op of ALL FIVE §9.2 fixtures
  (`< 1e-10`), plus `R(op₁)R(op₂) = R(op₁∘op₂)` over the FULL `n²` sweep on
  `diamond`/`si` (`< 1e-8`) — no round-trip test used anywhere.
* **`Symmetry` never owns a `Cell`** (17-CONTEXT §3.9): `Symmetry::build`
  takes a borrowed `&Cell`; upstream's `del self.lattice_symmetry.cell`
  (breaking a Python refcount cycle that cannot exist in Rust) is
  intentionally not ported, documented at length so it doesn't read as an
  omission. `build_lattice_symmetry` lives in `pyscf-pbc-symm` as a FREE
  FUNCTION, not a `Cell` method — `Cell` (below `pyscf-pbc-symm`) cannot call
  into `Symmetry::build` without inverting D-PBC-25's dependency direction.
* **`Cell::lattice_symmetry` is plain data** (`pyscf-pbc-gto/src/
  symmetry_data.rs`'s new `LatticeSymmetry`), same shape as `Cell::pseudo`
  holding `PseudoData` rather than the parser; `pyscf-pbc-gto` gains no new
  dependency. NOT serialised (derived, build-time state); `symmorphic` IS,
  closing the exact gap the plan named (`dumps_loads` carried
  `space_group_symmetry` but silently dropped its `symmorphic` partner,
  which did not exist as a field at all before this plan).
* **Measured, not assumed, mid-plan**: diamond's own default `cell.mesh =
  [47,47,47]` is NOT a multiple of 4, so it is genuinely incompatible with
  diamond's real `(1/4,1/4,1/4)` glide/inversion translation — this is a
  live exercise of 17-CONTEXT §3.3, not a synthetic one, and it flipped two
  tests from an initially-wrong assumption ("the default mesh carries the
  full group") to the correct, measured behaviour
  (`check_mesh_symmetry=true` reduces diamond to its 24-op symmorphic
  subgroup on its own default mesh; `=false` keeps all 48). See
  `17-03-SUMMARY.md` Deviations 1-3 for this and two related mid-plan
  discoveries (an `OnceLock`-memoized KRHF fixture, and why the
  homomorphism test does not require canonical-op-list membership under
  `SPGElement::dot`, which upstream itself never reduces mod 1).
* **Concurrency**: no file overlap with the concurrently-running 17-10/
  cubecl-kernel/multigrid agents in `crates/` (this plan touched only
  `pyscf-pbc-symm` and `pyscf-pbc-gto`); `.planning/STATE.md` and
  `PBC-MASTER-PLAN.md` edited additively, re-read immediately before each
  edit.

**Phase 17 plan 04 — SHIPPED, 2026-09-01.**
`.planning/phases/17-ksymm-multigrid/17-04-PLAN.md` — `basis.rs` in
`crates/pyscf-pbc-symm` (port of `pyscf/pbc/symm/basis.py`, the module
`PBC-MASTER-PLAN §8.9`'s table omitted entirely, 17-CONTEXT §1.2); `Cell`
gains `symm_orb`/`irrep_id` and the crate gains `build_symmetry`. Full
detail: `17-04-SUMMARY.md`.

* **62/62 tests green** (`cargo test -p pyscf-pbc-symm --release`: 7 basis +
  7 geom + 8 group + 4 kpts_ibz + 15 space_group + 21 symmetry, 0 failures,
  5384 s; the `#[ignore]`d probe correctly reports 1 ignored / 0 run).
  `tests/basis.rs` alone: 7/7 in 5204 s with its 4 fixtures in parallel on 16
  cores. `--release` is required, not a convenience — each fixture runs a
  converged full-BZ KRHF at tightened integral precision.
* **The four Task-4 properties, with MEASURED maxima** (not just pass/fail —
  every check now prints its worst residual over all k and all `(p,q)`):
  orthonormality `2.22e-16` (1 ulp, tol 1e-12) on all four fixtures;
  `S` block-diagonality `2.2e-14 .. 1.4e-13` (tol 1e-11); Fock
  block-diagonality `2.73e-13 .. 9.12e-13` (tol 1e-11, worst case 11x inside);
  invariance `8.88e-16` (tol 1e-8); completeness exact.
* **The 1e-11 Fock gate was MEASURED, not relaxed.** Its first `--release`
  run failed at a true max of `3.99e-10`. A 2-D `cell.precision` x
  `conv_tol_grad` sweep (`17-04-MEASUREMENT.md`, 7 rows) proved this a
  FIXTURE-CONFIGURATION floor, not an algebraic defect: the overlap control
  `S` is integral-precision-limited and *bit-identical* across every
  `conv_tol_grad` at fixed precision, while `F` needs BOTH axes — integrals
  alone leave 4.18e-10, convergence alone plateaus at ~1.92e-11. So
  `BLOCK_DIAG_TOL` **stayed at 1e-11** and the FIXTURE was tightened to
  `precision = 1e-10` / `conv_tol_grad = 1e-10`; `si_2x2x2` now measures
  `5.476113225217893e-13`, reproducing the probe's joint-tight point to the
  last digit. Same shape as 17-01 Task 2's Gate B; NOT the shape of 14-05's
  `decompose_j2c` defect that 17-04-PLAN.md named as the risk to watch for.
  The `#[ignore]`d `tests/basis_precision_probe.rs` is kept, trimmed to the
  two decisive sweep points, so a future failure is re-measured rather than
  tolerance-relaxed.
* **`crates/pyscf-pbc-gto/src/test_systems.rs` gains `si_precision(f64)` /
  `diamond_precision(f64)`** (`si()`/`diamond()` are now thin wrappers passing
  `DEFAULT_PRECISION`; every existing caller and committed reference number
  untouched). They go through `Cell::build(CellBuildArgs { .., precision, .. })`
  because mutating `cell.precision` on an already-built cell and calling the
  `build()` METHOD silently DROPS the pseudopotential (`Nocc 112 > Nmo 64`) —
  the trap is recorded in the constructor's doc comment.
* **Two documented corrections to the plan, both verified against live
  upstream PySCF 2.12.1.** (1) The plan's `must_haves` truth *"symm_orb columns
  are orthonormal in the S metric"* is WRONG: `_gram_schmidt`
  (`basis.py:93-108`) uses the PLAIN Hermitian inner product, no `S` anywhere;
  on live upstream diamond at Γ, `soᴴ S so = [[2.366, 2.209], [2.209, 2.366]]`
  (not `I`) while `soᴴ so` IS exactly `I`. The plan's one check became two —
  `ᴴ` orthonormality at 1e-12, plus `S` BLOCK-diagonality, which is the
  property that actually makes `khf_ksymm.eig`'s per-irrep generalized
  eigenproblems well-posed. (2) `symmorphic = true` is the tested scope
  because upstream's OWN `symm_adapted_basis` trips `assert nso == cell.nao`
  (`basis.py:90`) for both diamond and si at both meshes with
  `symmorphic = False` — an upstream non-symmorphic-glide + special-k-point
  limitation, not a port defect. Zero-translation ops still carry a
  non-trivial `_get_phase` phase, so Task 1's phase threading IS exercised.
* **Carry-over to 17-05**: `tests/basis.rs`'s test-local `little_cogroup` /
  `sorted_little_pg` helpers (duplicated in the probe) must be REPLACED by
  17-05's production `little_cogroup_ops` (`kpts.py:1084-1126`) once it lands,
  and the tests re-pointed at a real IBZ set. Flagged, not done. `basis.rs`
  itself is not expected to change: it already takes the four fields
  `basis.py:109-130` reads off `kpts` as a plain `SymmAdaptedBasisInput`.
  Tier-2 oracle (`irrep_id` multiset, `test_krhf_symorb` energy) still owed;
  the energy needs 17-07's `eig`.
* **Concurrency**: 17-05 was being worked concurrently in the same crate; this
  plan touched only `src/basis.rs` (pre-existing), `tests/basis.rs`,
  `tests/basis_precision_probe.rs` and `pyscf-pbc-gto`'s `test_systems.rs` —
  no overlap with `src/kpts.rs` / `tests/kpts_ibz.rs`. `.planning/STATE.md`
  edited additively, re-read immediately before the edit.

**Phase 17 plan 05 — SHIPPED, 2026-09-01.**
`.planning/phases/17-ksymm-multigrid/17-05-PLAN.md` — `pyscf/pbc/lib/kpts.py`
(1223 l, the largest single file in the phase) as
`crates/pyscf-pbc-symm/src/kpts.rs`, plus `is_trim` into `pyscf-pbc-lib` and
the closure of the oldest Phase-17 promise in the tree. Full detail:
`17-05-SUMMARY.md`.

* **Gate A — EXACT, no tolerance, reproduced on two cells.** `si` and
  `diamond` at `[16,16,16]` both give **145 / 145 / 245 / 408 / 816 / 2052**
  across the six configurations, bit-for-bit with 17-01's measurement. The
  Fm-3m controls `lif`/`he_fcc` collapse to `{145,145,145,408,408,2052}`
  (`C == A`, `E == D`) exactly as 17-01 measured — asserted so a `symmorphic`
  branch that silently did nothing could not pass the first two tests.
* **Gate B — measured, then the FIXTURE was tightened, not the gate.** Against
  ONE converged full-BZ KRHF (`si [2,2,2]`, FFTDF, `conv_tol = 1e-11`), max
  over every k and every `(p,q)`: at `precision`/`conv_tol_grad` = 1e-10/1e-10
  `transform_dm` = `make_rdm1(transform_mo_coeff)` = **1.784e-11**,
  `transform_1e_operator` 1.099e-12, `transform_mo_energy` 6.054e-12,
  `dm_at_ref_cell` 6.106e-12; at **1e-12/1e-12** they drop to **2.306e-13 /
  2.306e-13 / 1.212e-14 / 3.686e-14 / 5.390e-14**, `transform_mo_occ` exactly
  0. Shipped gate `GATE_B_TOL = 1e-12` — three orders TIGHTER than 17-01's
  ≤1e-9-at-default-precision floor. Confirms 17-04-MEASUREMENT's joint-floor
  finding on a third quantity class; `transform_dm` and
  `make_rdm1(transform_mo_coeff)` agree to the LAST DIGIT at both fixtures.
* **17-CONTEXT §3.1 is pinned OPEN.** The elementwise `mo_coeff` residual
  measures **2.658** (O(1), matching 17-01's ~2.3 at `[3,3,3]`) and a test
  asserts it is `> 1e-4`, with a failure message saying a SMALL value would
  mean the fixture stopped exercising a degenerate subspace — not that an
  elementwise assert became valid. Nobody can "tighten" the DM comparison
  into an MO one without deleting that test.
* **`symmetrize_density` went through `oracle_sum` in its FIRST version**
  (D-PBC-17 / 17-CONTEXT §3.8), with the §9.3 1-vs-8-worker bit-identity test
  in the same commit, for the real AND the complex (planar `re`/`im`, RULE 8)
  paths. Oracle: upstream's own C kernel (`pyscf/lib/pbc/symmetry.c:3-48`)
  transcribed independently in the test — agreement is **0e0**, bit-identical,
  on both the zero-translation and the fractional-translation branch.
* **A silent round is a wrong density — so it fails loudly.** Upstream's C
  computes the translation offset as `(int)(ft * n)`, a TRUNCATION that is
  wrong whenever `ft * n` is not exactly representable (`(int)(0.25 * 7) == 1`,
  asserted). This port checks integrality and returns `MeshNotSymmetric`.
  **Finding:** on every §9.2 fixture the star search NEVER names a
  non-symmorphic op — `SPGElement`'s ordering (`hash_key = trans*3^9 + rot`)
  sorts zero-translation ops first and `make_kpts_ibz` `break`s at the first
  match — so the `symmetrize_ft` branch is unreachable end-to-end and is
  tested two other ways (`ft_offsets` directly, and `symmetrize_density`
  white-box with a non-symmorphic op substituted into `stars_ops`). A test
  asserts that premise so a future change is announced.
* **Speed, measured not asserted:** `map_k_points_fast`'s op loop was
  parallelised alongside the star search after the first measurement showed
  the star search alone was NOT the bottleneck (0.99x). With both:
  `make_kpts si [16,16,16]` (`nkpts = 4096`, `nop = 48`) goes **31.2 ms at 1
  worker -> 6.6 ms at 8, a 4.76x speedup**, and every produced index array
  plus `weights_ibz` stays bit-identical between the two.
* **`get_kconserv` DELEGATES** to the shipped
  `pyscf_pbc_lib::kpts_helper::get_kconserv`, as the plan required. A test
  proves the delegation lands element-for-element on the table upstream's own
  fast path (`add_tab[add_tab[:, inv_tab], :]`) computes — which is exactly
  what not re-porting buys. `addition_table` is built row by row rather than
  through upstream's `(nkpts,nkpts,nkpts,3)` tensor, which would be 400 GB at
  Gate A's `nkpts = 4096`.
* **The oldest Phase-17 promise is closed.**
  `pyscf-pbc-gto/src/kpts_mesh.rs`'s `make_kpts_with_symmetry` (refusing since
  plan 09-07) is **DELETED**, its doc redirected to
  `pyscf_pbc_symm::kpts::make_kpts` — `Cell` cannot return a `KPoints` without
  inverting D-PBC-25. `grep -rn "phase: 17" crates/pyscf-pbc-gto/` now returns
  nothing. `pyscf-pbc-df` and `pyscf-pbc-dft` gained a `pyscf-pbc-symm`
  dependency for TYPE VISIBILITY ONLY; no DF or numint behaviour changed
  (the seven `isinstance(kpts, KPoints)` branches are 17-08's).
* **Deviations, all recorded in the SUMMARY:** `make_k4_ibz` ships `sym="s1"`
  only (`"s2"`/`"s4"` are 17-09's `kccsd_rhf_ksymm` and return
  `UnsupportedK4Symmetry` rather than a wrong answer);
  `symmetrize_wavefunction` REFUSES exactly as upstream does (its own first
  statement is `raise RuntimeError('need verification')`, `kpts.py:415`);
  `little_cogroups` refuses when a `little_cogroup_ops` entry indexes past
  `nop`, which is where upstream raises `IndexError` (reachable only with
  `time_reversal = true`); `map_kpts_tuples` at `ntuple > 1` with an explicit
  `kpts_scaled` is not ported (unreachable — `make_ktuples_ibz` takes the
  `k2opk` path).
* **Green:** `cargo test -p pyscf-pbc-lib -p pyscf-pbc-gto --release` (all
  targets, 0 failures) and, in `pyscf-pbc-symm --release`, `geom` 7/7,
  `group` 8/8, `space_group` 15/15, `symmetry` 21/21, **`kpts_ibz` 5/5**,
  **`kpts_ktuples` 8/8**, **`kpts_transform` 14/14** (311 s).
  `cargo build -p pyscf-pbc-df -p pyscf-pbc-dft` green with the new
  dependency. `cargo clippy -p pyscf-pbc-symm --all-targets` reports nothing
  in `src/kpts.rs` or the three new test files. No `mod tests` in any
  `src/*.rs`. **NOT re-run in this session:** `tests/basis.rs` (17-04's, 5204 s
  — its inputs are unchanged by this plan; `symmetry.rs`/`error.rs` gained
  only additive items).
* **Carry-over to a later plan:** 17-04's test-local `little_cogroup` /
  `sorted_little_pg` helpers are now superseded by `KPoints::little_cogroups`
  and could be replaced; 17-05-PLAN.md explicitly instructed NOT to refactor
  them here.
* **Concurrency**: touched `pyscf-pbc-symm` (`src/kpts.rs` new,
  `src/lib.rs`/`src/symmetry.rs`/`src/error.rs`/`Cargo.toml` additive, three
  new test files), `pyscf-pbc-lib` (`kpts_helper.rs`, `is_trim` appended),
  `pyscf-pbc-gto` (`kpts_mesh.rs`/`lib.rs`/`tests/kpts_mesh.rs`) and the two
  `Cargo.toml`s of `pyscf-pbc-df`/`pyscf-pbc-dft`. No overlap with 17-04's
  `src/basis.rs`/`tests/basis.rs` or 17-10's `pyscf-pbc-df/src`.
  `.planning/STATE.md` re-read immediately before this additive edit.

**Phase 17 plan 06 — SHIPPED, 2026-09-01.**
`.planning/phases/17-ksymm-multigrid/17-06-PLAN.md` — `pyscf/pbc/lib/ktensor.py`
(386 l), the `KsymmArray` container, into `crates/pyscf-pbc-symm/src/ktensor.rs`
(+ `tests/ktensor.rs`, 19 tests). Full detail: `17-06-SUMMARY.md`;
speed: `17-06-MEASUREMENT.md`.

* **What shipped**: `KsymmArray` (IBZ-only block storage with incore and
  out-of-core backing), `empty`/`empty_like`/`zeros`/`from_dense`/`to_dense`/
  `from_raw`/`to_raw`, the shape/ndim/order accessors, `set_2d`/`set_4d` and
  the `BlockSink`/`FlatBlocks` write abstraction, `transform_2d`/
  `transform_4d`, and the `index_to_coords`/`slice_to_coords` index algebra
  with a typed `Key`/`SliceSpec`/`Coords` vocabulary. `src/error.rs` gained 7
  additive `Ksymm*` variants; `Cargo.toml` gained `pyscf-chkfile` + `ndarray`.
* **D-07 honoured**: the out-of-core scratch goes through
  `pyscf_chkfile::hdf5` + `pyscf_chkfile::H5Complex` (the `pyscf-ao2mo`
  `OutcoreScratch` pattern, RAII-deleted temp file).
  `grep -c hdf5-metno crates/pyscf-pbc-symm/Cargo.toml` = **0**.
* **Metadata is borrowed, never cloned** — `KsymmMeta<'a>` holds
  `&'a KPoints` / `Option<&'a KQuartets>` / `Option<&'a MORotationMatrix>`,
  the same rule 17-CONTEXT §3.9 sets for `Symmetry` and a `Cell`.
  `subarray_order` is an enum on the struct (not a runtime string) and
  round-trips through `from_raw`.
* **D-17-06-01 — upstream's `fromdense` writes to the wrong keys, both
  branches** (`ktensor.py:194-198` passes an IBZ index where `set_2d`
  expects a BZ index; `:199-203` passes an integer `m` that
  `index_to_coords` expands to `nkpts**2` coordinates for one value block).
  It has no caller and no test upstream. The port writes at the keys
  `set_2d`/`set_4d` actually expect. Made falsifiable by
  `upstreams_fromdense_key_choice_would_drop_two_of_three_blocks_here`:
  `si [2,2,2]` has `ibz2bz = [0, 6, 7]`, so upstream's key choice keeps 1 of
  3 blocks.
* **The index map is gated against an INDEPENDENT dense tensor**, not a
  round-trip (the plan's Task-2 warning): all 512 full-BZ triples written,
  expectation built from `kqrts.kqrts_ibz` alone. Index arithmetic is
  exhaustive — 19 008 `slice_to_coords` cases, 585 `index_to_coords` cases.
* **Every `(label, trans)` combination is tested individually** — 16 for
  `transform_2d`, 256 for `transform_4d`, each against a directly
  written-out einsum, with the counts themselves asserted. Worst residuals
  **4.965e-16** and **1.343e-15** against a 1e-13 tolerance.
  `transform_*(block, identity)` is bit-exact for every `trans`; the
  unfold-then-refold identity is bit-exact; Hermiticity is preserved by
  `'nc'`/`'cn'` (7.85e-17 / 1.11e-16) and NOT by `'nn'`/`'cc'` (1.19 / 1.74,
  asserted large). **The plan's "Hermitian for every op and both `trans`
  values" is wrong as written and is corrected in the test.**
* **Acceptance on REAL data** (`si [2,2,2]` KRHF/FFTDF, `precision = 1e-10`,
  `conv_tol_grad = 1e-10`): four MO-basis one-electron blocks — exactly what
  `kintermediates_rhf_ksymm.py:26-80` stores with `('oo'|'ov'|'vv', 'cn')` —
  stored over the IBZ and read back at every BZ k-point. **Worst residual
  3.607e-12**, gated at 17-01's Gate-B floor 1e-9.
  **Finding: three of the four blocks cannot discriminate the `trans` flag**
  — Schur's lemma makes `C^H F C` / `C^H h C` / `C_v^H h C_v` real and
  DIAGONAL at every k of this mesh, so `'cn'` and `'nc'` coincide
  identically. The rectangular `C_o^H h C_v` block discriminates by 5e5
  (3.003e-12 vs 1.518e-6). A symmetric cell's canonical one-electron MO
  blocks are therefore a weak probe of the antiunitary convention; the
  synthetic 256-combination enumeration is the strong one.
* **17-07's Fock store did not exist**, so Task 4's "real `khf_ksymm` Fock
  store" was substituted, NOT fabricated. `17-06-SUMMARY.md`'s closing
  section lists exactly what 17-07 and 17-09 must add: the `khf_ksymm`-owned
  Fock-store acceptance test, a rank-4 test on real MO ERIs, re-running every
  index-map assertion if `make_k4_ibz("s2"/"s4")` lands, an out-of-core
  measurement at a size that genuinely does not fit, and
  `NDArrayOperatorsMixin` if a caller ever needs it.
* **Speed, MEASURED (`17-06-MEASUREMENT.md`)**: `ktensor.py:54-83` has NO
  size heuristic to span — `incore` is read off the metadata dict and every
  upstream caller decides with a pure `max_memory` test. So the measurement
  answers the useful question instead: the out-of-core UNFOLD costs only
  **1.1x-1.6x** and falling with size (`view()` reads the dataset once, not
  once per block), while the out-of-core BUILD carries a fixed **~1 ms**
  HDF5 file-creation cost that dominates below ~0.5 MiB. **No minimum-size
  floor was added**; the choice stays the caller's memory test. Explicitly
  NOT measured: a tensor that genuinely does not fit in RAM (17-09's).
  The unfold is a rayon `par_iter` (disjoint writes, no reduction, so no
  `oracle_sum` ordering) and is bit-identical at 1 and 8 workers inside one
  process.
* **Green**: `cargo test -p pyscf-pbc-symm --release` — `ktensor` 19/19
  (120 s), `geom` 7/7, `group` 8/8, `space_group` 15/15, `symmetry` 21/21,
  `kpts_ibz` 5/5, `kpts_ktuples` 8/8, `kpts_transform` 14/14 (496 s,
  re-run). **NOT re-run:** `tests/basis.rs` (17-04's, ~5200 s) — inputs
  unchanged, `error.rs` additive only, the same call 17-05 recorded.
  `cargo clippy -p pyscf-pbc-symm --all-targets` reports nothing in the new
  files. No `mod tests` in any `src/*.rs`.
* **Environment**: the sibling `libxc_rs` workspace was being regenerated
  concurrently, so `cargo` intermittently failed with `failed to get
  libxc-rkernel-mgga_*` / `no targets specified in the manifest`. Every such
  failure was transient; every test above was re-run to green.
* **Concurrency**: touched only `pyscf-pbc-symm`
  (`src/ktensor.rs` + `tests/ktensor.rs` new; `src/lib.rs`, `src/error.rs`,
  `Cargo.toml` additive) plus `Cargo.lock`. No overlap with 17-04's
  `basis.rs`, 17-05's `kpts.rs` or 17-10's `pyscf-pbc-df/src`.
  `.planning/STATE.md` re-read immediately before this additive edit.

**Phase 17 plan 10 — Tasks 1/2/3/5 SHIPPED 2026-09-01; Task 4 LANDED in a follow-up session (band k-points + MO-factorised `get_k`), its test EVIDENCE lost to restarts — and its band route was independently proven BIT-IDENTICAL to the direct route on 2026-09-02.**

*Update 2026-09-02 (supersedes "Task 4 CARRIED OVER" below):* a follow-up
session closed both band-k-point refusals (`gdf/jk.rs` and `mdf/mdf_jk.rs` now
build a band GDF/MDF over `kpts ∪ kpts_band` via `build_band_gdf` and proceed;
the only `NotYetImplemented` left in those functions is the pre-existing
Phase-14 `omega` one) and shipped the MO-factorised `get_k_kpts` behind
upstream's flag, with `tests/band_kpoints.rs` and `tests/gdf_mo_k.rs`. That
session was killed by the environment's restart cadence before it could record
the run results, so the band-k-point gate vs upstream `get_bands` and the
MO-route 1e-13 agreement + speedup are **written but unrecorded**. What IS
recorded, from 17-08's diagnostic `gdf_band_route_matches_the_direct_route`:
GDF's band route against its direct route, same `_cderi`, same density, at a
STRICT-SUBSET band set — **`max |dvj| = 0e0, max |dvk| = 0e0`**, bit-identical.
The `grep -rn "phase: 17" crates/pyscf-pbc-df/` check the plan asked for now
returns only historical doc-comment references, no live refusal. The
`exclude_dd_block` default remains `false` (deviation 5 stands) and the
diamond Ha-level oracle run remains `#[ignore]`d and unconfirmed on this
port's own code (deviation 6 stands). Original entry follows.


`.planning/phases/17-ksymm-multigrid/17-10-PLAN.md` — the independent
DF-accuracy track 17-CONTEXT §1.1 flagged as omitted from §8.9's original
eight-plan table. Full detail: `17-10-SUMMARY.md`.

* **Task 1** — `crates/pyscf-pbc-df/src/ft_ao/rs_cell.rs` (new):
  `ft_ao._RangeSeparatedCell`, built by copying ALREADY-NORMALISED `cintx::Shell`s
  out of the built `basis_set` rather than through `Cell::build` (which would
  silently re-normalise a decontracted shell's coefficients a second time).
  **D-PBC-21's "numerically transparent" premise is now VERIFIED, not
  asserted**: a bit-exact primitive-permutation test, plus `ft_aopair`
  evaluated over the RS cell and recontracted matching the direct lattice sum
  to `0.0` (He-fcc, no split at all) / `~1e-13` (diamond, every shell splits)
  with all screens off.
* **Task 2** — `crates/pyscf-pbc-df/src/ft_ao/supmol.rs` (new):
  `ft_ao.ExtendedMole`, represented as `(rs_cell, Ls, bvkmesh_Ls, bas_mask)`
  rather than a literal replicated `Mole` (every quantity gated is a function
  of shell parameters + geometry, not of an actual cint-drivable molecule —
  stated as a deliberate choice, not an omission). `strip_basis`/`estimate_rcut_per_shell`
  gated BOTH ways on diamond `gth-szv` 2×2×2: upstream's own per-shell radii
  to `<1e-9`, AND no regression against this port's pre-existing (plan 14-02)
  flattened-maximum `estimate_rcut` (their max matches to `1e-12`). Surviving
  triple count after `strip_basis` matches upstream exactly: **1450** (of
  12864 raw).
* **Task 3** — `exclude_dd_block` CLOSED. New
  `crates/pyscf-pbc-df/src/gdf_builder/dd_block.rs` ports `_outcore_dd_block`
  (the FFT re-route of the smooth-smooth `(ij|L)` block); `j3c.rs`'s new
  `make_j3c_scheme_dd` applies it as a post-hoc scatter-add correction
  (`dd_fft − dd_rs` at the smooth AO positions) so the EXISTING real-space
  pipeline is untouched for `dd_correction = None`. **Default LEFT at
  `false`** on both `CcGdfBuilder` and `RsGdfBuilder` — a deliberate
  deviation from the plan's "flip the default" instruction, because doing so
  touches every pre-existing test that builds a `CcGdfBuilder`/
  `RsGdfBuilder`/`Gdf` without naming the flag, several of which are gated
  tighter than D-PBC-23's own deltas, and a full-suite regression run to
  confirm safety did not finish inside this plan's time budget. The refusal
  itself IS fully closed — `exclude_dd_block = true` builds and runs — as a
  tested opt-in. **He-fcc's "exactly 0" is bit-identical `cderi`, BY
  CONSTRUCTION and CONFIRMED green** (a `rs_cell` with no SMOOTH shell makes
  the correction a no-op), non-`#[ignore]`d test. Diamond's two Ha-level
  numbers (1.835e-08 2×2×2, 2.900e-08 gamma) have a written, `#[ignore]`d
  oracle test (`crates/pyscf-pbc-scf/tests/exclude_dd_block_energy.rs`,
  needs `--release`) whose live run did NOT finish inside this session — the
  upstream/target side was independently re-confirmed
  (2.9002556800605817e-08 on gamma) but this PORT's own number is NOT YET
  CONFIRMED; see `17-10-SUMMARY.md` deviation 6. Both routes are pinned
  against their OWN upstream numbers separately in the test file, per 14-07
  Task 7d's lesson — pinning is written even though the diamond leg's live
  result is outstanding.
* **Task 5** — `crates/pyscf-pbc-scf/src/rsjk.rs`'s module doc updated:
  blocker 1 (the supermole) is closed by this plan; blocker 2 (a screened
  periodic 4-centre `int2e` driver) still has no implementation and no
  correct-but-slow fallback. Refusal LOGIC untouched — verified by diff.
* **Task 4 — NOT SHIPPED, carried over.** Neither the band-k-point `_cderi`
  rebuild (`gdf/jk.rs:243-253`, `mdf/mdf_jk.rs:80-90`) nor the MO-factorised
  `get_k_kpts` (`gdf/jk.rs:36-38`, which has no `force_dm_kbuild`-equivalent
  parameter to begin with) is implemented. `grep -rn "phase: 17"
  crates/pyscf-pbc-df/` therefore still returns two lines (both Task 4's).
  Reasoning and exact scope of what remains: `17-10-SUMMARY.md`'s Task 4
  section.
* **New decision**: `D-PBC-27` (`PBC-MASTER-PLAN.md`) — records the
  `ExtendedMole` non-literal-`Mole` representation and the `make_j3c_scheme_dd`
  post-hoc-correction integration shape.
* **Concurrency**: plan 17-02 (symmetry half) ran in the same window; no file
  overlap in `crates/` (17-02 touched only `pyscf-pbc-symm`), and the shared
  `.planning` files were edited additively.

**Phase 17 plan 11 — Tasks 1-4 SHIPPED, 2026-09-01.**
`.planning/phases/17-ksymm-multigrid/17-11-PLAN.md` — `multigrid.py` v1
(the ordered-last, droppable-if-overrun half of the phase per
`17-CONTEXT.md §1.4`). Full detail: `17-11-SUMMARY.md`.

* New kernel `crates/pyscf-kernels/src/multigrid_collocate.rs` — real-space
  Cartesian primitive collocation with periodic image summation, concrete
  `f64` (documented `exp`-libm exception, same class as `ft_aopair.rs`).
  Kernel-level tests (`crates/pyscf-kernels/tests/multigrid_collocate.rs`):
  analytic-norm, l=0..4-vs-`eval_gto_sph`, and periodic-wrap-exactness, all
  green at 1e-9..1e-12.
* New `crates/pyscf-pbc-dft/src/multigrid/{tasks,colloc,numint}.rs` — the
  grid-level task list (`_primitive_gto_cutoff`/`multi_grids_tasks_for_ke_cut`,
  decontracted per-PRIMITIVE `Pshell` representation rather than upstream's
  `h_coeff`/`t_coeff`/`t_cell` machinery — same math, simpler mechanism),
  the density/potential collocation driver (`level_rho`/`level_pass2`, both
  `oracle_sum`-ordered — D-PBC-17 bit-identity CONFIRMED at 1/2/3/8 rayon
  workers), and `MultiGridNumInt` (`get_nuc`/`get_pp` delegate to the
  already-shipped `Fftdf` — 17-01 measured that pass2 difference at
  1e-12..1e-13, i.e. noise; `get_j`/`nr_rks` go through the new engine).
  **Gamma point only** — stated scope, matches what Gate E's own upstream
  measurements use.
* **A real bug found and fixed**: `get_lattice_ls(..., discard=true)` —
  upstream's atom-PAIR image-discard heuristic silently dropped periodic
  images a single off-origin pshell's OWN self-collocation needed, breaking
  `∫ rho dr = Tr(dm.S)` by ~1.7e-3 on diamond's second (non-origin) atom
  while its origin atom stayed correct to ~2e-9 — the origin-only fixture
  trap this phase's own `LARGE_DENOM`/column-major precedents warn about.
  Fixed by `discard=false` at this one collocation call site, documented
  in-line. After the fix: `∫ rho dr == Tr(dm.S)` to **1.6e-11** (diamond) /
  **2.1e-11** (si), inside the plan's 1e-10 gate on both reference cells.
* **Gate E, measured**: `get_j`/`nr_rks(lda,vwn)` vs the reference
  `numint`/FFTDF at machine precision (1e-14..1e-15) on diamond and si, at
  a matched-but-coarsened 25³ mesh on both sides — NOT directly comparable
  to 17-01's natural-mesh 1e-12..1e-14 floor, and stated as such. Speed:
  this port's multigrid measured 4.0x-5.2x FASTER than its own `Fftdf`
  reconstruction at this one coarse test point — the OPPOSITE direction
  from 17-01's upstream-vs-upstream 0.18x-0.49x finding, and explicitly
  flagged as NOT comparable (different implementations, different mesh
  regime, no shell-cutoff submesh restriction in this port) — a later plan
  must NOT read this as "multigrid ships faster in general".
* **Task-1 gate deviation, recorded**: si's first-level mesh (33 vs
  upstream's 31) traced to a ~2.8e-5 relative discrepancy in the
  ALREADY-SHIPPED (Phase 9/11, not this plan) `pyscf_pbc_tools::mesh::
  mesh_to_cutoff` sitting at an integer `Gmax` boundary for si's geometry;
  diamond matches exactly. Out of this plan's scope to fix; recorded for
  whichever future plan owns `pyscf-pbc-tools::mesh`.
* **Not shipped, stated**: k-point multigrid, shell-cutoff submesh
  restriction (performance only — correctness unaffected, all gates
  green), a live GGA Gate-E number (code path exists, reuses the
  already-tested `xc.rs`, just not separately measured this session), and
  driver-level `MultiGridNumInt` selectability wiring into `Krks`/`veff.rs`
  (the standalone engine is complete and tested; SCF-loop plumbing is
  follow-up).
* **Concurrency**: no file overlap with the concurrently-running
  `pyscf-pbc-symm`/`pyscf-pbc-df` plans; only `crates/pyscf-kernels/` and
  `crates/pyscf-pbc-dft/` were touched.

Phase: 14-gdf-mdf-rsdf-rsjk — **CLOSED 2026-08-29; Gate 3 MET 2026-08-30.**
Full evidence in `.planning/phases/14-gdf-mdf-rsdf-rsjk/14-VERIFICATION.md`.

**Update 2026-08-30 — Gate 3 is no longer unreachable.** D-PBC-24 landed
`ExecutionOptions::range_omega` in cintx, and plan 14-07 sub-tasks 7b/7c ported
`rsdf_builder::_RSGDFBuilder` on top of it (`14-07-SUMMARY.md`, appended).
Measured against vendored PySCF 2.12.1, gated by
`crates/pyscf-pbc-scf/tests/gate3_rsdf.rs`:

* He-fcc `sto-3g` 2×2×2 — **RSDF 2.325e-10**, **GDF 2.750e-10** (Gate 1 level).
* diamond `gth-szv` gamma — **RSDF 1.615e-8**, **GDF 2.074e-8** (the GTH floor).

**The ORIGINAL Gate 3 criterion — the port's `|CC − RS|` landing on upstream's
own gap within 2× — is MET on diamond: 4.465597e-6 against 4.502481e-6, ratio
0.9918.** On He-fcc it does not discriminate (ratio 0.028) because upstream's
two routes differ there almost entirely through `exclude_d_aux` /
`exclude_dd_block`, which this port has in neither route; the gate therefore
asserts per-route agreement on both systems and reports the ratio.

`_RSNucBuilder`'s absence does not show at 1e-8 even on the pseudopotential
cell — RSDF's diamond error is smaller than GDF's.

Also shipped on the same foundation: **`_RSMDFBuilder`** (3.209e-10 / 1.897e-11
/ 7.808e-12 at matched meshes 11/15/21 vs upstream's `df.MDF()` default) and
**Task 7d — `Gdf::prefer_ccdf` flipped to `false`**, with `df_swap.rs` now
pinning BOTH routes against their own upstream numbers (they differ by 5.222e-10,
inside that test's 1e-9 bar, so pinning one would have hidden the flip).

Still open from D-PBC-24 stage 5:

* `_RSNucBuilder` — **performance carry-over, not a fidelity gap** (the same
  one 14-03 opened for `_CCNucBuilder`). This port uses neither split builder:
  `get_nuc`/`get_pp` go straight to AFTDF at the cell's converged mesh, gated at
  2.755e-12 and more accurate than either split. The split buys speed only.
* `rsdf_helper`'s prescreen — its absence keeps more primitives than upstream,
  the conservative direction.
* **`pyscf_pbc_scf::rsjk` — BLOCKED ON PHASE 17, not on cintx.** `range_omega`
  was necessary but not sufficient: `rsjk`'s short-range half needs
  `ft_ao._RangeSeparatedCell` + `ExtendedMole.strip_basis` (Phase 17,
  D-PBC-21/23) AND a periodic 4-centre screened `int2e` driver, of which this
  port has none. Unlike RSDF there is no all-compact fallback — the screening IS
  the algorithm. Sequence after Phase 17; size as its own plan.

**At close: four of five gates MET, one UNREACHABLE, and the fifth's blocker was
a missing capability in `cintx`, not in this port.**

| gate | result |
|---|---|
| **1** — the algebra vs upstream, all-electron control | **MET** — `KRHF` on GDF **2.750e-10**; `fuse(j3c)` 1.412e-12; `j2c` 7.105e-14; `df_ao2mo.get_eri` 1.667e-12; `ao2mo_7d` 1.984e-12; `KRHF` on MDF 2.827e-10 |
| **1b** — the same on diamond | **PARTIAL** — everything that needs no 3-centre build is gated at ≤1e-11; the flagship `make_j3c` is an unmeasured multi-hour run and its oracle is an `#[ignore]`d acceptance test |
| **2** — MDF converges to FFTDF | **MET** — 6.002e-05 (GDF) → 1.695e-06 → **3.433e-09** → 3.245e-08, on upstream's CC ladder to within 1 %, INCLUDING the non-monotone bounce |
| **3** — GDF vs RSDF | **MET 2026-08-30** (was UNREACHABLE) — per-route vs upstream: He-fcc RSDF 2.325e-10 / GDF 2.750e-10; diamond gamma 1.615e-8 / 2.074e-8. Original gap criterion MET on diamond, ratio **0.9918** |
| **4** — `_cderi` memory, k-mesh PINNED at 2×2×2 | **MET** — 6.08 % of the FFTDF AO table on diamond (upstream 6.17 %); 20.50 % at 3×3×3, which is why the mesh is pinned |

### The ROADMAP's gate was wrong in both halves and is rewritten

"Every DF builder gives the same KRHF energy to 1e-15 with GDF under 20% of
FFTDF memory" cannot stand: GDF is an APPROXIMATION whose fitting error is
1.222e-03 Ha on diamond 2×2×2, upstream's own two GDF builders disagree by up to
4.502e-06, and one f64 ulp at |E| ≈ 10.9 is 1.78e-15. The memory half is
k-mesh dependent and does not say so. `14-CONTEXT.md`'s five gates replace it;
the ROADMAP line and `PBC-MASTER-PLAN.md` §8.6's row are both corrected, not
quietly shipped against.

### 14-05 — `df_ao2mo` + `outcore`. **Phase 13's `ao2mo_7d` carry-over is CLOSED.**

The contract Phase 15 was blocked on:
`eri[ki, kj, kk][i, j, k, l]`, shape `(nk, nk, nk, nmoi, nmoj, nmok, nmol)`,
`kl = kconserv[ki, kj, kk]`, chemists' notation (the first index of each pair
conjugated). KMP2 reads it as `eri[ki, ka, kj][i, a, j, b]` = `(ia|jb)` with
`kb = kconserv[ki, ka, kj]` — the SAME table under two index namings, no
re-ordering. Asserted with four different `nmo`s (2/3/1/4) over every
`(ki,kj,kk)` of a 2×2×2 mesh at <1e-13, and against upstream at 1.984e-12.

**The attribution device worth keeping:** upstream's own `df_ao2mo.get_eri`, run
over THIS port's `cderi` through a stub `mydf`, agrees to **1.110e-16** — one
ulp at `|eri| ~ 0.5`, i.e. the two contractions differ only in summation order
(sequential here, BLAS `ddot` there). So "is the contraction upstream's?" and
"is the `cderi` upstream's?" are permanently separated.

**`_ao2mo.r_e2` conjugates the BRA only** — measured (2.512e-15 bra-only against
12.227 for both), because `r_ao2mo.c`'s two comments both say `^*` and its
arithmetic does not.

### 14-06 — MDF. Gate 2 met, and BOTH of the plan's premises were wrong

* MDF's default mesh is `[11,11,11]` (diamond 2×2×2) and `[9,9,9]` (He-fcc), not
  the plan's `[7,7,7]` — mesh 7 is `mdfladder.py`'s lowest rung.
* **`measurements/mdfladder.out` measures the WRONG BUILDER.** `MDF._prefer_ccdf`
  is `False`, so every row of it is `_RSMDFBuilder` — plan 14-07's route.
  `mdfladder_cc.py` / `.out` were added and are what Gate 2 asserts against.

MDF beats GDF by **170×** at gamma and **17 000×** at the 2×2×2 plateau.
`make_j3c` is ONE driver with a `Scheme` tag, mirroring upstream's
`_CCMDFBuilder(_CCGDFBuilder)` subclass; `Mdf` composes an inner `Gdf` (so
14-04's `df_jk` and 14-05's `df_ao2mo` are reused unchanged) and an inner
`Aftdf` at MDF's mesh (so Phase 13's `aft_jk` is).

### 14-07 / 14-08 — BLOCKED on cintx, and the block is documented (D-PBC-24)

Range separation is libcint's `PTR_RANGE_OMEGA` (`env[8]`) toggle around the
STANDARD `int3c2e`/`int2c2e`/`int2e` — upstream never calls an `int2e_sr_*`.
cintx's safe API cannot set it: `ExecutionOptions` carries `f12_zeta`
(`env[9]`), `rinv_orig` and `common_orig` and no `range_omega`; no kernel reads
`env[8]`; and the periodic 3-centre driver builds its `BasisSet` from the parsed
per-element basis rather than from an `_env` array, so even `pyscf-gto`'s own
direct-`_env` workaround is unreachable. **This is Phase 4's Open Question A5 /
cintx#11**, already documented in `crates/pyscf-gto/src/range_coulomb.rs`.

`14-07-PLAN.md` Task 7b named this failure mode in advance and required it be
REPORTED, not worked around — so `_RSGDFBuilder`, `_RSMDFBuilder`, `RSDF` and
`rsjk` return `NotYetImplemented` naming the gap, and the refusals are asserted.

**What DID ship:** 7a in full — all twelve ω estimators, `weighted_coulG_LR/_SR`
and `_gaussian_int`, every number gated at 1e-12 against
`measurements/omega.out` (added as Task 0, before any code). `rsjk`, RSH
functionals and Phase 17 all need them regardless. Plus `get_aux_chg` (equal to
14-01's monopole at 1e-14) and ONE shared `density_fit` for all four upstream
shims.

## SIX defects the phase's own tests caught

1. **`decompose_j2c` read `zeigh_gen`'s COLUMN-MAJOR eigenvectors row-major —
   worth +6 306 866.73 Ha.** The transpose of an orthogonal matrix is still
   orthogonal, so the factor had the right shape, rank and eigenvalues and
   nothing crashed. **No gate had ever reached the eigen branch**: `j2ctag` is
   `CD` on every measured system, including diamond, whose `eig_min` is
   3.17e-11 and which upstream still decomposes by Cholesky. MDF
   (`j2c_eig_always = True`) was its first consumer. New regression test:
   `V j2c Vᴴ = I` on the retained subspace — the identity that DEFINES the
   factor, 2.709e-14 (He-fcc) / 3.094e-08 (diamond, a conditioning floor).
2. **Two missing devices from `gen_uniq_kpts_groups`**, both invisible on
   Cholesky: `if self_conj: j2c = j2c.real` (a complex eigensolver may return an
   arbitrary phase, and `cderi` is contracted with NO conjugate, so it survives
   as `e^{2iθ}`), and the conjugate pass at `−kpt` with `_conj_j2c` rather than
   an independent decomposition of `j2c[−k]`.
3. **`get_naoaux` was STRICTER than upstream.** 14-03 made it raise on per-k
   rank disagreement; upstream takes one arbitrary block. The ranks legitimately
   differ per k-difference on the eigen route (MDF keeps 10 vectors for one
   group and 11 for another at mesh 15). Now returns the diagonal `(0,0)`
   block's rank, which is what `df_jk`'s `rho` accumulator needs.
4. **`ExtendedMole.strip_basis` is worth 1.054e-09 in `j3c` / 2.750e-09 in the
   ERI, and 14-02's gate could not see it** — it compared against a standalone
   `incore.Int3cBuilder`, which strips nothing. Localised by six measurements in
   which every INPUT matched and the assembly did not. Flattening upstream's
   per-shell-pair radius array to its own maximum collapses it to 7.333e-13:
   **the port is the MORE converged of the two.**
5. Both of 14-06's stated premises (above).
6. `_ao2mo.r_e2`'s conjugation convention (above).

**Phase 17 plan 07 — Tasks 0/1/2 + the `use_ao_symmetry` eig branch SHIPPED, 2026-09-01; `eig_trs`/KUHF/KGHF/fast-`get_jk` validation/Gates C-D CARRIED OVER.**
`.planning/phases/17-ksymm-multigrid/17-07-PLAN.md` — `khf_ksymm.py`.
Full detail: `17-07-SUMMARY.md`; reconnaissance in `17-07-BLUEPRINT.md`.

* **Written directly by the orchestrator session, not by an agent** — four
  consecutive agent sessions were killed by the ~20-40 min environment
  restart cadence during their READING phase, before writing a line. The
  reconnaissance was extracted into `17-07-BLUEPRINT.md` (the verbatim
  `KOverrideHooks` trait shape, the `krhf.rs` template, prerequisites, the
  phase's traps) so any successor starts at "write code".
* `crates/pyscf-pbc-scf/src/khf_ksymm.rs` (new) — `KsymAdaptedKrhf`, a
  `KOverrideHooks` implementation over an IBZ k-set. **D-PBC-15's central
  claim holds literally: `git diff crates/pyscf-pbc-scf/src/kscf.rs` is
  EMPTY** — the driver was not forked, copied or edited. The k-set
  indirection is one method: `kpts()` returns an owned `kpts_ibz`. The DF
  object is built over the FULL BZ (every DF entry point takes its k-points
  explicitly, `fftdf.rs:447`), so one object serves both the IBZ-length
  one-electron hooks and the full-BZ reference `get_veff`; the DF layer
  never learns about symmetry.
* `get_occ` computes ONE Fermi level over the UNFOLDED BZ (17-CONTEXT §3.4)
  with `nelectron = cell.tot_electrons(kpts.nkpts())` — the BZ count, not
  the IBZ one — and folds back through `check_mo_occ_symmetry`, whose failure
  is a typed error naming both k-points (a symmetry-broken state is
  physical, not internal). Every weighted sum is tabulated in the module doc
  (`weights_ibz` vs `1/nkpts` vs bare, per 17-CONTEXT §3.5).
* `eig_symm_adapted` — block-diagonalises the Fock one irrep at a time in
  17-04's `symm_orb`, with the layout contract (`symm_orb` column-major, Fock
  row-major, output column-major) written at the function because it is the
  exact shape of 14-05's +6 306 866 Ha defect. Made `pub` so 17-08's DFT
  adapters SHARE it rather than copy it.
* **Tests 4/4 green** (`--release`, 222 s): IBZ set is 3 of 8 BZ points;
  `weights_ibz` sums to 1 and matches every star size; the two `eig` routes
  agree on `e_tot` to **1.703e-11** (two converged SCFs) and, on IDENTICAL
  inputs, on every eigenvalue to **9.186e-11**. The first version of that
  eigenvalue test compared two independently converged SCFs and reported
  4.4e-9 — convergence noise, exactly what 17-05's plan warns against
  ("never two SCFs"); rerun on one SCF's Fock it tightened ~48x. The 9.2e-11
  residual is the off-block Fock leakage 17-04 measured at default
  `cell.precision`, not slack in the implementation.
* **D-17-07-01 — a latent UPSTREAM bug found**: `little_cogroup_ops` is
  filled from `np.where(k2opk[ki] == ki)[0]` (`kpts.py:112`), indices into
  `k2opk`'s `2*nop` columns when time-reversal is on, but its consumer
  indexes `kpts.ops[iop]` directly (`basis.py:113`). At Γ and every TRIM
  the second half is reachable, so **upstream would `IndexError`**. This
  port refuses with a typed `KptsSymmInputMismatch`, which is how it was
  found. NOT patched around — the test builds `KPoints` with
  `time_reversal_symmetry = false` and says why. It gates
  `use_ao_symmetry = true` + time reversal, which is upstream's DEFAULT
  combination, so it must be closed before the adapter is recommended at
  its defaults.
* Dependency: `pyscf-pbc-scf` now depends on `pyscf-pbc-symm`, which
  dev-depends on `pyscf-pbc-scf` — a cycle cargo permits (dev edges are
  excluded from build ordering), with the library direction still as
  D-PBC-25 ruled. The now-false comment in `pyscf-pbc-symm/Cargo.toml`
  ("no build-graph cycle since `pyscf-pbc-scf` does not depend on this
  crate") was corrected in place.
* **Carried over**: `eig_trs` (real `mo_coeff` at TRIMs — its TRIM test is
  the only proof the branch was taken); Task 4 (`get_rho`, chkfile
  round-trip incl. the k-count refusal, `to_khf`); Task 5 (KUHF/KGHF; KROHF
  has no upstream `*_ksymm` and is not invented); Task 6's fast-`get_jk`
  1e-13 validation against the reference route (the route is written
  behind `JkRoute`, reference is the default); Task 7's Gates C/D and the
  speed gate; plan 11-09's metal-occupancy test extended to ksymm.
  17-06's `KsymmArray` acceptance handoff was CLOSED on 2026-09-02 (see
  17-06's entry update below).

**Phase 17 plan 08 — Tasks 1/2/3/4 + `kukspu_ksymm` SHIPPED, 2026-09-02; Task 5 PARTIAL. TWO of the plan's premises were wrong and were corrected before building.**
`.planning/phases/17-ksymm-multigrid/17-08-PLAN.md` — `krks_ksymm.py`,
`kuks_ksymm.py`, `krkspu_ksymm.py`, `kukspu_ksymm.py`, and the seven
`isinstance(kpts, KPoints)` sites in `numint.py`. Full detail:
`17-08-SUMMARY.md`; the premise corrections in `17-08-FINDING-numint.md`.

* **D-17-08-01 — the plan's Task 1 premise is factually wrong.** It said all
  seven `numint` sites "evaluate the density at the IBZ points, then
  symmetrize the real-space density through `kpts.symmetrize_density`".
  Verified against vendored 2.12.1: **five** (`:328, :431, :859, :908,
  :956`) unfold to the FULL BZ via `transform_dm` and run the ordinary path;
  **two** (`:647`, `:779`) take `kpts_ibz` directly; and `symmetrize_density`
  has **no caller in `pyscf/pbc/` outside its own unit test**. Caught not by
  reading but by hitting the wall the wrong premise implies — the density is
  built per grid BLOCK and `symmetrize_density` rotates indices across the
  whole mesh, a fight upstream never has because upstream never does this.
  Consequence stated plainly: under symmetry `numint` does full-BZ work PLUS
  an unfold — a convenience interface, not an optimisation; the IBZ saving
  comes from the SCF side (D-PBC-26) only. That is the phase's THIRD speed
  assumption to fail in the same direction (after 17-01's upstream-multigrid
  0.18-0.49x and 17-05's 0.99x star-search parallelism).
* **Task 1, faithful**: `pub enum KSet { Full, Ibz(Box<KPoints>) }` on
  `KNumInt` (a field, not the plan's threaded parameter — so the `Full` path
  is not merely unedited, BOTH arms reach the same code, and bit-identity
  holds by construction). `unfold_dms`/`unfold_kdms`/`unfold_mos` (Group A)
  and `kpts_ibz()` (Group B) are wired at all seven sites, each carrying its
  upstream line. `cache_xc_kernel` unfolds the ORBITALS not the density, as
  `:859-863` does (RULE 2). The `Full` path's pre-existing suites are
  unchanged: `numint_blocking` 3/3, `numint_threads` 1/1, `modules` 8/8.
  Gate: unfolded-IBZ density vs full-BZ density **1.054e-13** (tol 1e-11) on
  a tight fixture — first measured 1.807e-10 at default precision; the
  FIXTURE was tightened (not the tolerance), the residual fell 1714x, and
  the tolerance was then set TIGHTER than the value that first failed. Third
  appearance of the joint precision/convergence floor in the phase, first
  predicted before being measured.
* **Task 2** `KsymAdaptedKrks` — the line that makes the shapes work is
  upstream's `kpts_band = kpts.kpts_ibz` (`krks_ksymm.py:41-42`): `nr_rks`
  evaluates rho over the full zone but builds the potential AT the band
  k-points, so both halves return `nkpts_ibz` matrices with nothing folded
  by hand. `eig`/`get_occ` SHARED with 17-07 (`pub` helpers, not copies;
  `KsymAdaptedKrhf` now routes through the same `ksymm_get_occ_restricted`).
  **Gate C for DFT, FFTDF: 3.109e-14 / 2.842e-14** (both `use_ao_symmetry`
  branches). A separate test asserts si `[2,2,2]`'s stars are UNEQUAL
  (`[1,3,4]`, `weights_ibz = [0.125,0.375,0.5]`) so a mistaken `1/nkpts`
  cannot coincide with `weights_ibz` and pass silently — the guard
  15-CONTEXT §3's KMP2 trap earned.
* **D-17-08-02 — Task 4's premise is wrong too.** The plan said the local
  projectors `C_ao_lo` "must be rotated with the space group". They must
  not: upstream's whole ksymm DFT+U is `krks_ksymm.get_veff` + the SHARED
  `krkspu._add_Vhubbard`, whose only symmetry-aware lines are
  `kpts = kpts.kpts_ibz` (`:77`) and `weight = weights_ibz` (`:93`); the
  projectors are built DIRECTLY at the IBZ points and nothing is unfolded.
  Same for `kukspu` (`:59`, `:78`), which also applies ONE `C_ao_lo` to both
  spins (`:68-70`). Shipped as `add_vhubbard_weighted` (existing
  `add_vhubbard` delegates with uniform `1/nkpts`, bit-exact pre-17-08
  behaviour) + `KsymAdaptedKrkspu` + `KsymAdaptedKukspu`.
  **Gate: `E_U` IBZ vs full BZ 6.939e-18**, with NO SCF — a Hermitian IBZ
  density pushed through `transform_dm` is symmetric BY CONSTRUCTION, so the
  residual is the weighting alone and the test runs in 0.29 s. Two fixture
  constraints were each found by a test refusing to pass vacuously: a `gth`
  cell gives a singular MINAO metric (must be all-electron), and `E_U`
  VANISHES on a filled shell — a converged He 1s gave `E_U = -2.04e-17`,
  agreeing to 2e-17 only because both sides were zero; an
  `assert!(e_u.abs() > 1e-6)` guard caught it and the fixture now uses
  fractional occupancy. Pre-existing asymmetry recorded, not fixed: Phase
  12's `Krkspu` has NO `KOverrideHooks` impl (U-08), so the plain DFT+U is
  not SCF-drivable while the k-symmetric one now is.
* **Task 3** `KsymAdaptedKuks` — shipped with `nset() == 2`, two Fermi
  levels each over the unfolded BZ (new shared `ksymm_get_occ_unrestricted`),
  `nr_uks` at `kpts_band = kpts_ibz`, a `weighted_trace_uks`; its
  `get_veff_tagged` deliberately mirrors `Kuks::veff_from_parts` rather than
  calling it, because that body derives `nkpts` from `with_df.kpts().len()`
  and forms `1/nkpts`, which come apart here. **D-17-08-03 — a Gate C
  precondition nobody had stated**: an IBZ-vs-full-BZ energy comparison is
  only valid if the FULL-BZ solution is itself symmetric. The IBZ run is
  constrained to symmetric occupations; an unconstrained full-BZ run is not.
  Measured on the open-shell fixture (RULE U satisfied, `|dm_a-dm_b| =
  1.19`): full-BZ occupations star-symmetric `alpha = true, beta = FALSE`,
  `|dE| = 4.533e-02` with the IBZ energy LOWER — a different, better state,
  physical, not a defect, and NOT absorbed by relaxing the tolerance. The
  energy gate now ASSERTS the precondition and is `#[ignore]`d pending a
  fixture whose full-BZ solution is symmetric in both channels; the
  machinery test (`kuks_ibz_runs_and_stays_symmetric`) passes.
* **Task 5, per DF route**: FFTDF Gate C PASS (above). **GDF Gate C RUN and
  FAILS at 1.432e-06** (tol 1e-8; e_full -7.774590218592, e_ibz
  -7.774588786147; 1381 s) — ~3 orders above GDF's measured floor and ~8
  orders worse than FFTDF on the identical comparison; recorded, not
  absorbed. The first hypothesis (GDF's `kpts_band` route rebuilds `_cderi`)
  was **tested and was WRONG**: `gdf_band_route_matches_the_direct_route`
  measured the band route against the direct one on the same `_cderi` and
  density at a STRICT-SUBSET band set — `max |dvj| = 0e0, max |dvk| = 0e0`,
  bit-identical. **The GDF band route is exonerated; 17-10's Task 4 work is
  correct.** (The diagnostic's first version passed the FULL k-set as
  `kpts_band`, and `band_is_kpts` short-circuits to the direct path when the
  two coincide — it would have compared direct-vs-direct and reported a
  false all-clear; fixed to use `kpts_ibz`.) `build_band_gdf` was also
  checked mechanically: every numeric `Gdf` field is copied, only the
  filesystem `cderi_to_save` is not. Current leading hypothesis, UNTESTED:
  GDF's `_cderi` is fit on a k-set with no symmetry adaptation, so the
  full-BZ GDF solution may be slightly symmetry-broken — the D-17-08-03
  class, GDF-specific because FFTDF is analytic; the check is
  `check_mo_occ_symmetry` on the full-BZ GDF solution. Erratum against the
  plan's Task 5: "upstream gates GDF TIGHTER than FFTDF" describes
  upstream's chosen test tolerances and points opposite to the measured
  floors (GDF is the looser route by ~3 orders); tolerances here follow the
  measurement. Gate D (port vs upstream) not attempted — needs the oracle
  harness; 17-01 already measured upstream's side per route; owned by
  17-13. RSH remains blocked on the Phase-14 `omega` carry-over at
  `gdf/jk.rs:674`, per the plan's own instruction to record rather than
  work around.
* Default suite: `cargo test -p pyscf-pbc-dft --release --test krks_ksymm
  -- --test-threads=1` — **7 passed, 0 failed**, 2 `#[ignore]`d with their
  reasons in the doc comments (GDF Gate C on cost+failure; KUKS Gate C on
  the D-17-08-03 precondition).

**Phase 17 plan 06 — handoff item 1 CLOSED, 2026-09-02** (the plan itself
SHIPPED 2026-09-01, above). `crates/pyscf-pbc-symm/tests/ktensor_ksymm_scf.rs`
fills a `KsymmArray` from `KsymAdaptedKrhf`'s OWN IBZ output via `set_2d_at`
at the irreducible representatives and reads back every BZ k-point. The
obvious comparison — against MO blocks from a SEPARATE full-BZ KRHF — is not
sound (orbitals are gauge-free within degenerate subspaces, 17-CONTEXT §3.1;
17-06 met the same wall via Schur's lemma), so both sides start from ONE
SCF's orbitals and two independent unfolds are compared: `KsymmArray::get_2d`
(17-06's `transform_2d` + `MORotationMatrix`) vs projecting with
`KPoints::transform_mo_coeff` (17-05). They share no code below `KPoints`.
`hcore` blocks, not the Fock — the MO-basis Fock is diagonal by construction
and could not see a wrong rotation. Measured `oo` 8.255e-14, `ov` 3.842e-13,
`vv` **3.318e-12** against the 1e-9 Gate-B floor. 17-06's own stand-in test is
kept, as its handoff asked. Items 2-4 of that handoff are 17-09's and remain
externally blocked.

**Phase 17 plan 12 — COMPLETE: kernel gates 8/8 and host-side Gate E 10/10 GREEN on this machine, 2026-09-02 (later session; supersedes the "UNVERIFIED" entry below).**

*Update 2026-09-02:* the exit-137 was not the machine, it was the port —
`collocate_pair_level` materialised one f64 per `(image × monomial × ci ×
cj) × grid point`, **192 GiB (si) / 231 GiB (diamond)** on the 25³ cells.
Replaced by density-contracted fused terms, kernels that reduce in-lane
(`collocate_pairs_rho` / `collocate_pairs_integrate`, gated vs the per-slot
values at 1e-13 and by the adjoint identity), and 5³-point spatial blocks
that see only the images reaching them: peak RSS 0.46 GiB, 7–9 s per
density. Running the suite then found three defects the OOM had hidden —
`p.coef·q.coef` applied twice on top of `E` (∫rho = 0.53 of 8.73 e), no
periodic wrap of the fused Gaussian (and, once added, a `[0,1)³` box on a
grid that is origin-centred in `[-0.5,0.5)`), and a polynomial-blind image
pre-screen that dropped negative far `p-p` terms — each located by the new
per-pair brute-force gate and fixed. **Gate E:** v2 `get_j` vs FFTDF
1.24e-8 diamond / 6.80e-8 si; v1-vs-v2 1.46e-8 / 7.41e-8 — at 17-01's
upstream floors (2e-8 / 1.5e-7). `nr_rks(lda,vwn)` Δnelec ≤ 1.5e-6, Δexc ≤
7.9e-7. Bit-identical at 1/2/3/8 threads. **Speed:** v2 `get_j` 21.8 s /
16.5 s vs reference 0.51 s / 0.46 s and v1 0.34 s / 0.37 s — 0.023× /
0.028×, ~10× worse than upstream's own 0.18–0.39× v2 floor; Phase 18 needs
v2 for `isinstance` only. `oracle_sum` deviation (in-kernel sequential
sums, still thread-invariant) recorded in `17-12-SUMMARY.md`. §8.10 of
`PBC-MASTER-PLAN` now carries the `MultiGridNumInt2` ↔ Phase 18 note.
Original entry follows.

**Phase 17 plan 12 — code SHIPPED and compiling, kernel gates GREEN, host-side Gate E UNVERIFIED, 2026-09-02 (earlier session).**
`.planning/phases/17-ksymm-multigrid/17-12-PLAN.md` — `multigrid_pair.py` +
`pp.py` + `utils.py`, `MultiGridNumInt2`, the twelve-entry-point half of
multigrid and the one Phase 18's `grad/rhf.py:44`/`grad/uhf.py:40` assert on.
Full detail: `17-12-SUMMARY.md` (reconstructed by the orchestrator; the
implementing agent was killed by a restart before writing it).

* Shipped: `crates/pyscf-kernels/src/{multigrid_pair,multigrid_gspace}.rs`,
  `crates/pyscf-pbc-dft/src/multigrid/{pair,pp,utils}.rs`, and both test
  files. ALG-06 held; the four C destructors are `Drop` by design.
* **Kernel side 5/5 green** (`pyscf-kernels --release --test multigrid_pair`,
  0.23 s): the collocate/integrate ADJOINT IDENTITY (the plan's "strongest
  oracle-free test", written first) at two sizes, `single_slot` vs a direct
  formula, `gradient_gs` vs its own documented einsum (the "free, exact
  oracle"), `get_gga_vrho_gs` vs its documented formula.
* **Host side PARTIAL**: `pair_task_list_is_sane` PASSES (diamond: 16
  pshells, 256/256 pairs, per-level `[0,4,12,240]`) — the load-bearing task
  list the plan says to gate before any number computed from it. Every
  SCF-bearing host test (`int_rho_matches_tr_dm_s_v2`,
  `gate_e_get_j_vs_reference_v2`, and the four not reached) is killed with
  **exit 137 (SIGKILL)** — as a suite, serialized, and individually in
  release. Not a timeout (124), not a panic (101): the process is killed,
  consistent with the OOM this session hit repeatedly on the shared machine.
  **No assertion has been observed to fail, and none has been observed to
  pass; this entry claims neither.**
* Gate E and the v1-vs-v2 ratio are therefore NOT confirmed by this plan's
  own run. The reference numbers exist (17-01: upstream v2 carries a
  mesh-independent ~2e-8 diamond / 1.5e-7 si floor vs FFTDF; upstream's v1
  and v2 both run 0.18-0.49x SLOWER than reference `numint`); the run is the
  carry-over, on a machine with enough memory or with reduced fixtures.
* `MultiGridNumInt2` ↔ Phase 18 is already recorded in `PBC-MASTER-PLAN`'s
  Phase-17 table (the 17-12 row). Inherits 17-11's stated reductions
  (gamma only; not yet a selectable SCF `numint`); `pp.rs`'s IBZ path is not
  yet connected to 17-05's `KPoints`, which now exists.

## Carry-overs

**ONE piece of work, not four.** `ft_ao._RangeSeparatedCell` + `ExtendedMole`
(with `strip_basis`, `_int_dd_block`, `merge_diffused_block`) closes:

| what | priced at |
|---|---|
| D-PBC-23 `exclude_dd_block` | 1.835e-08 Ha (diamond 2×2×2), 2.900e-08 (gamma), **0** (He-fcc) |
| `strip_basis` (new, 14-05) | 1.054e-09 in `j3c` → 2.750e-09 in the ERI |
| Phase 13's `ft_aopair` residual (D-PBC-21) | 5.121e-10 |

~600 + ~60 lines, and it feeds Phase 17.

**D-PBC-24, the cintx `range_omega` gap** — a cintx change, not a port change.
Blocks `_RSGDFBuilder`, `_RSMDFBuilder`, `RSDF`, `rsjk`, Gate 3, plan 14-07 Task
7d, and (already) Phase 4's numerical RSH assertion.
**Planned in `.planning/carryovers/D-PBC-24-cintx-range-omega-PLAN.md`** (five
stages, cintx-side). The finding that sizes it: `rys_order ≤ 3` on every system
this phase gates, and in that regime libcint gets the short-range integral from
`full − LR` with doubled Rys roots and the STANDARD root finder — so stage 2
unblocks Gate 3 without porting `CINTsr_rys_roots` at all.

**Smaller, each already refused rather than ignored:** `rsjk`'s MPI variants
(Phase 19, a named non-goal); `GDF`/`MDF` band k-points (Phase 17);
`GDF.get_jk(omega)` (same cintx gap); `exp_to_discard`; the MO-factorised
`get_k_kpts` (Phase 17); `outcore`'s k-pair blocking axis; diamond's `make_j3c`
wall time.

## Phase 15 is CLOSED

The two prerequisites described here were consumed successfully. Restricted
KMP2 and its AO2MO/Lov dependencies landed on 2026-09-05 and verification
closed the same day: the nine-part oracle matrix is green, the staggered energy
oracle ran (and found three defects — see Current Position), and both open
D-PBC-28 rows are measured. The GDF-vs-upstream numerical gap stays assigned to
Phase 14, because both Phase-15 GDF integral routes agree to `2e-15 Ha`.

### Phase 15 deferrals

| refusal | owner |
|---|---|
| `ktensor` / `KsymmArray`, `KPoints` inside `KMP2::new` | 17 (shipped there) |
| `KUMP2::kernel` energy, `kump2::_add_padding` | upstream — `kump2.py:38/:384/:402` all raise in 2.12.1; oracle-gated so the refusal cannot outlive its reason |
| `dimension == 2` in the DF `Lov`/`ao2mo` path | upstream |
| `kmp2_stagger` non-submesh at `dimension < 3`; odd Monkhorst-Pack submesh | upstream |
| fractional occupations in `get_nocc` | upstream — load-bearing here, because this port has live smearing that upstream's KMP2 never sees |
| `Frozen::{Auto, Window}` at k-points | 15, deliberately: upstream's `frozen='auto'` at k-points is molecular-only |
| `fft_ao2mo.general`'s gamma/all-real shortcut | 15, deliberately: a speed shortcut only, and KMP2 never reaches it |

---

Phase: 13-ft-ao-aftdf — **IMPLEMENTED, not closed** (retained below for continuity)
Plans: 13-01 … 13-05 and 13-07 shipped; **13-06 PARTIAL** (`get_eri` +
`get_ao_pairs_G` for both builders — oracle 4.172e-12 — but not `general` /
`get_mo_pairs_G` / `ao2mo_7d`); 13-08 written as
`.planning/phases/13-ft-ao-aftdf/13-VERIFICATION.md`.

**What runs.** `KRHF` runs on either builder with no driver change — that was the
point of the phase and it took a cross-crate refactor (D-PBC-22,
`Box<dyn PeriodicDf>` across all 8 drivers plus `veff::get_jk` and `get_hcore`)
which is proven BIT-IDENTICAL on the FFTDF path.

**Gate 1 MET in three parts. Gate 2 MET as a `(rcut, mesh)` ladder. Gate 3 MET
for `get_nuc`/`vj`/`vk`, near-met for `get_pp`.** Full numbers in `13-VERIFICATION.md`. The short
version: all three roadmap gate numbers were unmeasured, and measuring them first
is what made the phase tractable.
  * `ft_ao.estimate_rcut` (20.420 Bohr) is LOOSER than `cell.rcut` (21.319), so
    upstream's own `ft_aopair[G=0]` misses `int1e_ovlp` by 1.554e-9 and Gate 1's
    "1e-10" cannot pass. Past `rcut` x1.5 the FT sum is converged (x2.0 identical
    to four digits) and the residual sticks at 1.472e-10 — which is `pbc_intor`'s
    OWN truncation, not the kernel's. Hence 1a/1b/**1c**, and 1c (both sides over
    one identical image list, via `intor_cross_with_images`) passes at **1e-13**
    on diamond, He-fcc and all 8 k-points. That is the real gate on the algebra.
  * Gate 2 is a `(rcut, mesh)` ladder with TWO floors. Upstream plateaus at
    **2.607e-11 Ha**, BIT-IDENTICAL at mesh 31 and 41.
  * Gate 2 in the port: 9.309e-5 -> 5.066e-8 -> **2.378e-10** over meshes
    15/21/27 (diamond, gamma), the same ~1000x-per-6-mesh rate as upstream.
  * **Gate 3 is MET for 4 of 5 quantities** — `get_nuc` 2.755e-12, `vj`
    3.733e-12, `vk` 2.116e-12, `get_eri` 4.172e-12, all under the 1e-11 bar. That DISPROVES the
    obvious hypothesis that `ft_aopair`'s 5.121e-10 screening residual propagates
    broadly: all three run through the same `ft_loop`.
  * **`get_pp` is the exception at 1.806e-9, and the cause is upstream.**
    `aft.get_pp` builds part 2 with `_IntPPBuilder`; `pp_int.get_pp_loc_part2` is
    the reference route Phase 10 ported and `fft.get_pp` agrees with. **Those two
    upstream routes disagree with EACH OTHER by 1.7933e-9** — 99.3% of the gap.
    Substituting `pp_int` into upstream's own `AFTDF.get_pp` collapses the
    deviation to **3.982e-11** (45x). Asserted in the test suite so the
    attribution cannot rot. Worth reporting upstream on its own account.
  * `ft_aopair`'s own 5.121e-10 is separate and is screening: three upstream
    screens were ported (`strip_basis`, `get_ovlp_mask` over the
    `_RangeSeparatedCell` per-primitive grouping, libcint `PTR_EXPCUTOFF`),
    1.553e-9 -> 5.733e-10 -> 5.121e-10. The remainder is `ExtendedMole`, which
    D-PBC-21 declines to port and Phase 14 needs anyway for `gdf_builder`.

**The evidence that the Gate-3 residual is truncation and not algebra** is
oracle-free and sharp: Gate 1c at 1e-13, and the `get_pp` anti-Hermitian residue
falling from 5.133e-11 at upstream `rcut` to **2.665e-15** at a converged one.
Upstream tightens `precision` by 1e-2 for exactly that asymmetry
(`ft_ao.py:749-753`), and 5.13e-11 is on its 1e-10 target.

**Four defects the phase's own tests caught**, all in new code: (1) the
per-record screen reapplied `cell.precision*1e-2` — the IMAGE-LIST threshold — as
an absolute per-primitive-pair cutoff, accumulating to 1.66e-7 while every
angular-off-diagonal element stayed exact to 1e-16; (2) `estimate_rcut`'s `cs` is
the libcint contraction coefficient, NOT `gto_norm` (that is
`aft.estimate_ke_cutoff`), worth 21.186 vs 20.420 Bohr; (3) `Gamma(1.5)` returned
1 because the half-integer reduction stopped one step early, making `_fake_nuc`
short by exactly sqrt(pi)/2; (4) Phase-10's `get_pp_loc_part2`/`get_pp_nl` are
F-ORDER and were added raw, transposing the non-local block.

**Two performance corrections, both measurement-driven.** `FtKernel` was going to
be consolidated away as unnecessary; it is not — the record table is
`O(nimgs*nprim^2*npairs)` MD recursions and does not depend on `G`, so rebuilding
it per G-block made one `get_pp` take minutes. And `get_k_kpts` built one kernel
per `(ki,kj)` pair when the table depends on the ket k-point alone — `nkpts`
instead of `nkpts^2`.

Next: `ao2mo_7d` (the remaining 13-06 piece, which Phase 15's KMP2 is blocked on
and whose index order should be defined against KMP2 rather than guessed here),
then the Gate-3 closure inside Phase 14 alongside `gdf_builder`.
Last activity: 2026-08-29 — Phase 13 implemented and verified.

---

Phase: 11-fft-fftdf-periodic-hf — **COMPLETE** (retained below for continuity)
Plan: 11-12 COMPLETE (the phase verification rollup)
Status: **Phase 11 is CLOSED.** `KRHF(diamond, 2x2x2, gth-szv/gth-pade)` matches
live upstream PySCF **2.12.1** to **4.0e-12 Ha** at mesh 31 AND at the default
mesh 47; the ALL-ELECTRON control (`KRHF`/`KUHF` on He-fcc) meets the 1e-12 gate
outright at **2.2e-13**. Full results in
`.planning/phases/11-fft-fftdf-periodic-hf/11-VERIFICATION.md`.

### Two defects found by Phase 11's own tests

1. **`super_cell` silently dropped the pseudopotential** (a Phase-10 bug).
   `build_supcell` built its `Mole` through `pyscf_gto::build_from` — the
   MOLECULAR build — so `Cell::build`'s GTH valence-charge rewrite of
   `_atm[CHARGE_OF]` (plan 10-01, D-PBC-11) was never re-applied. Every
   supercell of a pseudopotential cell was therefore an ALL-ELECTRON system:
   diamond's `atom_charges()` came back `[6,6,6,6]` instead of `[4,4,4,4]`,
   taking `tot_electrons`, `ewald()` and the local pseudopotential with it. The
   supercell-equivalence identity read -10.53 versus -27.47. Fixed; the whole
   `pyscf-pbc-gto` suite still passes.
2. **The oracle was importing the WRONG PySCF.** Two installs are reachable:
   the vendored 2.12.1 at `<root>/pyscf` (the port target, whose line numbers
   every `PORT` comment cites) and 2.14.0 in `.venv/.../site-packages`. A script
   run picks up 2.14 because the script's own directory — not the CWD — lands on
   `sys.path[0]`. 2.14 **rewrote** `fft_jk.get_k_kpts` to fold the
   `exxdiv='ewald'` correction into `get_coulG` instead of applying
   `_ewald_exxdiv_for_G0` analytically, a **1.7e-5** difference in `vk`. Every
   Phase-11 oracle test now pins `PYTHONPATH` to the workspace root AND asserts
   `pyscf.__version__ == '2.12.1'` before comparing.

Previous: Phase 10-periodic-integrals — COMPLETE (10-08). `pbc_intor` within
1.29e-14 of upstream on diamond 2x2x2; GTH pseudopotentials complete bar the
long-range local term, which Phase 11 has now supplied.

### cintx status — R-13 fully resolved

**Wave 0.5 has LANDED.** `int1e_r{2,4}_origi` and `int3c1e_r{2,4,6}_origk` have
real kernels (`cintx-cubecl/src/kernels/unstable/{origi,origk}.rs`) behind the
`unstable-source-api` feature, which `pyscf-pbc-gto`'s default-on `gth-pp`
feature enables. The §2.4 Task-0 fail-open check passes.

**A second fail-open surface was found on 2026-08-26 and fixed the same day.**
Both families mishandled shells with `nctr > 1`: `origi` returned silently-wrong
values (only element (0,0) written, and wrong), `origk` PANICKED in
`cintx-cubecl/src/transform/c2s.rs:684` — the Cartesian->spherical step sized its
output from the angular momentum and forgot the contraction axis. cintx fixed
both (`kernels/unstable/{origi,origk,shared}.rs` + its own `*_genctr_parity`
oracle tests).

Verified and collected on this side: all seven symbols now match libcint to
<= 2.72e-15 relative on the `Li`/`gth-szv` general-contraction fixture; the
interim `pseudo::require_segmented_basis` guard and its two call sites are
DELETED; the two tests that pinned the broken behaviour are replaced by positive
libcint regression tests; and `part2_matches_upstream_on_lif` — the only §9.2
system with a general contraction, and the one system the bug had blocked — is
un-ignored and passes at **1.532e-10** (gate 1e-9). **No blockers remain in
Phase 10.**

Previous: Phase 09-pbc-foundation — COMPLETE (09-09, the verification rollup).
All nine plans shipped; `cell.ewald()` within 1e-9 Ha of upstream on
diamond/Si/LiF/He-fcc; 39 `pyscf-*` crates; complex algebra bit-reproducible.

## Performance Metrics

**Velocity:**

- Total plans completed: 103
- Average duration: — (no plans run yet)
- Total execution time: 0 hours

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 02 | 11 | - | - |
| 03 | 15 | - | - |
| 04 | 14 | - | - |
| 05 | 7 | - | - |
| 01 | 9 | - | - |
| 06 | 11 | - | - |
| 07 | 10 | - | - |

**Recent Trend:**

- Last 5 plans: —
- Trend: — (no data yet)

*Updated after each plan completion*
| Phase 02 P01 | 12min | 3 tasks | 13 files |
| Phase 02 P02 | 8min | 2 tasks | 9 files |
| Phase 04 P04-04 | 16min | 2 tasks | 12 files |
| Phase 04 P04-05 | 16min | 2 tasks | 9 files |
| Phase 04 P04-06 | 23min | 2 tasks | 14 files |
| Phase 04 P04-07 | 12min | 2 tasks | 9 files |
| Phase 04 P04-08 | 14min | 2 tasks | 7 files |
| Phase 04 P04-11 | 5min | 2 tasks | 3 files |
| Phase 04 P04-12 | 6min | 1 task (TDD) | 2 files |
| Phase 04 P04-13 | 11min | 1 task (TDD) | 2 files |
| Phase 04 P04-14 | 13min | 3 tasks | 6 files |
| Phase 05 P01 | 9min | 3 tasks | 24 files |
| Phase 05 P02 | 10min | 2 tasks | 3 files |
| Phase 05 P03 | 14min | 2 tasks | 7 files |
| Phase 05 P04 | 18min | 2 tasks | 5 files |
| Phase 05 P05 | 5min | 2 tasks | 3 files |
| Phase 05 P06 | 18min | 2 tasks | 3 files |
| Phase 05 P07 | 8min | 2 tasks | 6 files |
| Phase 02 P02-11 | 35min | 2 tasks | 5 files |
| Phase 03 P03-15 | 13min | 2 tasks | 5 files |
| Phase 03 P03-14 | 12min | 3 tasks | 5 files |
| Phase 06 P06-01 | 5min | 2 tasks | 20 files |
| Phase 06 P06-02 | 30min | 2 tasks | 9 files |
| Phase 06 P06-03 | 25min | 2 tasks (TDD) | 6 files |
| Phase 06 P06-04 | 30min | 2 tasks (TDD) | 5 files |
| Phase 06 P06-05 | 18min | 2 tasks (1 TDD) | 4 files |
| Phase 06 P06-06 | 22min | 2 tasks (TDD) | 7 files |
| Phase 06 P06-07 | 5min | 2 tasks (1 TDD) | 4 files |
| Phase 06 P06-08 | 8min | 2 tasks (1 TDD) | 5 files |
| Phase 06 P06-09 | 35min | 2 tasks | 7 files |
| Phase 06 P06-10 | 30min | 2 tasks | 5 files |
| Phase 06 P11 | 4min | 2 tasks | 4 files |
| Phase 07 P07-05 | 14min | 2 tasks (2 TDD) | 7 files |
| Phase 07 P07-04 | 18min | 3 tasks | 12 files |
| Phase 07 P07-03 | 24min | 2 tasks (1 TDD) | 2 files |
| Phase 07 P07-02 | 8min | 2 tasks (1 TDD) | 16 files |
| Phase 07 P07-01 | 39min | 2 tasks | 6 files |
| Phase 07 P07-07 | 9min | 2 tasks (2 TDD) | 5 files |
| Phase 07 P07-08 | 8min | 2 tasks (2 TDD) | 5 files |
| Phase 07 P07-06 | 7min | 2 tasks (2 TDD) | 6 files |
| Phase 07 P07-09 | 15min | 2 tasks | 9 files |
| Phase 07 P07-10 | 12min | 2 tasks | 5 files |
| Phase 09 P09-01 | 30min | 5 tasks | 77 files |
| Phase 09 P09-02 | ~1h | 9 tasks | 12 created, 2 modified |
| Phase 09 P09-03 | ~1h | 6 tasks | 8 created, 6 modified |

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.
Recent decisions affecting current work:

- [Phase 09 CLOSE-OUT]: **Phase 9 is COMPLETE and VERIFIED** — `.planning/phases/09-pbc-foundation/09-VERIFICATION.md` demonstrates all seven `09-CONTEXT.md` success criteria green with the command, observed value and verdict for each. Workspace is 39 `pyscf-*` crates + xtask; `check-dependency-wall` and `check-forbidden-paths` both PASS with the path-scoped `crates/pyscf-pbc-*` exemption (RULE 7, un-gated once in 09-01). **D-PBC-04 RESOLVED (09-02): `FAER_C64 = true`** — faer 0.24 has working native `c64` `SelfAdjointEigen`/`Llt`/`PartialPivLu`, probed against `H = [[2, 1-i], [1+i, 3]]`; `zeigh_gen`/`zcholesky`/`zsolve_linear` dispatch to the c64 route and the independent real-arithmetic route is always built as the CI cross-check. **R-02 RETIRED (09-01):** `cintx_cross_basis_smoke` proves `cintx` can evaluate a shell pair whose two shells come from different `Mole` instances (via `build_combined_basis`) to < 1e-12 against the single-`Mole` reference — the mechanism D-PBC-07 (periodic 1e integrals by image expansion, no new cintx operator) depends on. **Live-upstream oracle:** `crates/pyscf-pbc-gto/tests/oracle_phase9.rs`, 7 tests, `#[ignore]` + `PYSCF_ORACLE_VENV`-gated so it is NEVER a hard CI dep; `PYSCF_ORACLE_VENV=1 cargo test -p pyscf-pbc-gto --test oracle_phase9 -- --ignored` = 7 passed against PySCF 2.12.1. **PBC bindings:** `python/pyscf/pbc/__init__.py` is an import-path shim ONLY — every periodic PyO3 binding is plan 20-05 (D-PBC-14). **Two standing caveats every later PBC phase must know:** (1) the 4.951e-9 CODATA gap between `pyscf_core::Unit::Ang` (2014) and upstream `pyscf/data/nist.py` (2010) makes every Angstrom-built lattice relatively longer, so any 1e-12-class comparison MUST specify geometry in Bohr (09-05/09-07/09-08 and the whole oracle already do; two dedicated tests pin the deviation to exactly that constant so it fails loudly if either side changes); (2) `Cell::atom_charges()` is the ALL-ELECTRON `Z` until plan 10-01 parses GTH (D-PBC-11), so every Ewald reference is generated without `pseudo=` and the pseudised targets are committed in `ewald_reference::PSEUDISED_EWALD` for 10-01 to re-pin.
- [Phase 09 P09-08]: Ewald (K-05/K-06). **The 09-08 plan text's `ew_eta`-invariance gate is unsatisfiable as literally written** — holding `ew_cut` fixed while scaling `ew_eta` is not a physical invariance (weaker screening needs a longer real-space tail), and UPSTREAM ITSELF drifts 8.1e-7 Ha at `0.5*eta0` against the plan's 1e-8 bound; the gate re-derives `ew_cut = _estimate_rcut(eta^2, 0, 1., precision)` per `eta`, exactly as `get_ewald_params` does, and both sides then agree to < 1e-13. **New workspace dependency `libm = "0.2"`** (declared in `pyscf-pbc-gto` only): `std` has no `erfc`, cubecl's `Float` has none either, and the Abramowitz-Stegun 7.1.26 form the plan offered as a fallback is ~1.5e-7 accurate — two orders too coarse for a 1e-9 Ha gate; `libm::erfc` is FDLIBM, < 1 ulp, no-std, no transitive deps, already in `Cargo.lock`. `erfc` therefore runs on the HOST from device-computed distances (the plan's "preferred" option, now recorded as required). **Kernel scalars ride in an `Array<F>`, never as literals:** cubecl 0.10.0's `F::new` takes an `f32`, so `1e200` would become `inf` and `4*pi` would lose 29 mantissa bits — enough on its own to blow the gate (`ScalarArg::new` is still not public, same finding as `eval_gto.rs`). Both reductions are host-side `oracle_sum` (D-PBC-17), so `ewald()` is bit-reproducible. `_bspline` uses `powf` not `powi` to match numpy's correctly-rounded `**`. `cell.a is None` has no analogue here — `ewald` falls back to `Mole::enuc()` on a DEGENERATE lattice instead. Deferrals, each with a test pinning the phase number: `dimension == 2` truncated Coulomb -> `{phase: 12}` (plan 12-08, target -44.57202102404764 for graphene committed); `particle_mesh_ewald`'s G-space sum -> `{phase: 11}` (needs the plan 11-01 FFT; `Q`, `B`, `C`, `ewovrl`, `ewself` are all already computed inside it).
- [Phase 07 P07-10]: oracle/CI close-out — the phase Nyquist contract is CLOSED. grad_oracle lives in pyscf-oracle/SRC (a #[cfg(test)] dispatch_layer mod + a #[cfg(all(test,feature="python"))] live_arms mod), NOT in tests/ like the CCSD precedent — the plan named src/grad_oracle.rs as the deliverable (files_modified + artifacts.path). 8 names registered exactly as planned (nuc_grad_rhf/uhf/rks/uks/mp2/ccsd/ecp + geomopt_h2o), catalogue-len 24→32 (no drift). Three CI jobs: always-on grad-structural (cargo test -p pyscf-grad -p pyscf-geomopt -p pyscf-oracle, no python/libxc), geomopt-no-runtime-dep (GEOMOPT-01 pip-uninstall proof, runs in CI per D-05), workflow_dispatch grad-oracle-upstream-manual (upstream byte-identity ≤1e-7 + geomopt trajectory parity). The GEOMOPT-01 CI proof treats a cintx-availability error from optimize(mf) as the documented gated outcome and forbids ONLY a missing-geometric/pyberny ImportError (green today, stays green when the cintx numeric lands). The grad-oracle-upstream-manual arm installs geometric>=1.0 (the trajectory-parity reference) beyond the mp2/ccsd precedent's pyscf-only install. The upstream byte-identity numerics across ALL grad methods stay workflow_dispatch-gated on the 6 MISSING cintx grad-intor families (int2e_ip1 + int1e_ip{ovlp,kin,nuc,rinv} + ECPscalar_iprinv + with_rinv_at_nucleus, 07-01, no scheduled workstream); GRAD-01..07 + GEOMOPT-07 are recorded as Structural complete (always-on FD/structural green, upstream-match numeric cintx-gated); GEOMOPT-01 is fully Complete (the no-runtime-dep CI proof is green).

- [Phase 07 P07-08]: completed the gradient method surface (D-09 order, CCSD then ECP last). CCSD-grad (GRAD-06, the SECOND non-variational grad) CONSUMES the Phase-6 surface DIRECTLY — CcsdGradReference { mo_coeff/mo_energy/mo_occ + the converged Amplitudes (t1,t2) + ChemistsEris + mol }; relaxed_rdm1 runs pyscf_ccsd::solve_lambda then make_rdm1(ao_repr=false) (NO Λ re-derivation in pyscf-grad, D-04/T-07-25 — the single_lambda_solver_in_grad source-scan asserts NO `fn ...lambda...` declaration, only a path-prefixed CALL). The orbital-relaxation Z-vector re-enters the ONE 07-07 cphf::solve with its own ccsd_fvind (the ENERGY int2e get_veff, cintx-ready — identical in shape to mp2_fvind) + Xvo RHS, leaving max_cycle at the upstream default 50 (CCSD_CPHF_MAX_CYCLE = cphf::DEFAULT_MAX_CYCLE — NOT MP2's 30 override; pyscf/grad/ccsd.py does not override it). grad_elec assembles on the RHF base decomposition; the de assembly reaches the RHF get_ovlp (int1e_ipovlp — MISSING) first so kernel() ?-propagates a clean cintx-availability error (numeric #[ignore]'d). ECP-grad (GRAD-07): get_hcore_ecp wires the get_hcore '+ ECPscalar_ipnuc' term (cintx-READY/07-01 — numeric UN-GATED, real [3,nao,nao] on Cu/LANL2DZ via CintxEcpEngine::ecp_int1e_ipnuc; all-zero for a non-ECP molecule by mapping EcpEngineNotAvailable → zero buffer, never a panic) and NORMALISES the engine's component-INNER buffer (data[comp + p*3 + q*3*nao]) to the RHF component-leading F-order (out[comp*nao*nao + i + j*nao]) so it folds into get_hcore (07-03). hcore_deriv_ecp routes the hcore_deriv '+ ECPscalar_iprinv' per-atom term (MISSING from cintx) to a CLEAN cintx-availability error (T-07-27 — never a panic/silent-zero/NotYetImplemented{phase:7}). Closes the GTO-05 arc (Phase 2 wired ECP eval; Phase 7 wires ECP gradient). Both FD-gated (D-01); the CCSD numeric + the ECP end-to-end numeric stay #[ignore]'d on the 6 missing cintx grad-intor families (+ ECPscalar_iprinv for ECP end-to-end); the ECP ipnuc term un-gates now. Full pyscf-grad: 51 passed/6 ignored/0 failed; clippy clean; dependency-wall PASS. lib.rs flat re-exports CcsdGradients/CcsdGradReference/CCSD_CPHF_MAX_CYCLE + get_hcore_ecp/hcore_deriv_ecp (07-09 builds the CcsdGradReference from a CcsdReference + CcsdResult + the eris, and folds get_hcore_ecp into the RHF get_hcore for ECP molecules).
- [Phase 07 P07-07]: the ONE matrix-free Krylov CPHF/CPKS solver (cphf::solve, D-03/GRAD-10) — pyscf/scf/cphf.py:solve_nos1 + lib.krylov (Pople 1979); exact upstream defaults (max_cycle=50/tol=1e-9/level_shift=0); a caller-supplied &Fvind<'_> trait-object response operator (NO dense A-matrix); reductions via oracle_dot/oracle_sum, the projected nd×nd system via solve_linear; the single_cphf_impl gate asserts exactly ONE `pub fn solve(` in cphf.rs. MP2-grad (GRAD-05, the first non-variational Z-vector method) consumes it at max_cycle=30 (Pitfall 5): relaxed-density Lagrangian from gamma1_intermediates → Xvo RHS → response_dm1; the MP2 fvind uses the ENERGY int2e get_veff (cintx-ready) so the Z-vector solve is un-gated, only the de assembly is #[ignore]'d.
- [Phase 07 P07-06]: geomopt API surface around the 07-04 engine. (1) HDF5 optimizer-state checkpoint (checkpoint.rs): OptimizerState { coords, trust, hessian, nint, n_updates, step, prev_q, prev_g_int, prev_e, e_tot } with dump/load to/from an hdf5::Group via the pyscf_chkfile::hdf5 re-exported alias + the pyscf_chkfile::primitives scalar/1D helpers — NO own hdf5-metno dep (D-05/D-07 sole-owner discipline; Cargo.toml has no hdf5 line). Schema group /opt_state, schema_version=1.0, counts written as f64 read back as usize. optimize() + optimize_resume() refactored over a shared run_loop(opt, scanner, mol, init: Option<OptimizerState>); OptimizeResult gained .state (the live state on every run). Fail-clean guard (T-07-19): validate() rejects hessian≠nint·nint / prev-trio-not-all-present / prev-len≠nint on dump+load, load rejects unknown schema_version + coords≠3·natm, optimize_resume rejects resumed nint≠this molecule's internal count → GeomError::CheckpointCorrupt, never resume from garbage (GEOMOPT-05). (2) shims.rs: geometric_solver::{kernel,optimize} + berny_solver::{kernel,optimize} BOTH delegate to the ONE native optimize via a single shared run_shim core (D-06 — berny is a thin alias, NO second optimizer, T-07-20; NATIVE_ENGINE_NAME single-engine marker + engine_name() on both). ShimParams { conv_params: Option<ConvParams>, callback: Option<ShimCallback>, maxsteps: usize, constraints: Option<String> } mirrors the upstream kwargs (D-07); kernel->(conv,Mole), optimize->Mole (=kernel().1) — matches pyscf/geomopt/geometric_solver.py:96-192. constraints (non-None) -> clear GeomError::ConstraintsUnsupported, never a silent no-op (T-07-17); maxsteps default 100, validated at the shim boundary (T-07-18). ShimParams::default hand-impl (NOT #[derive] — derived gives maxsteps=0). Shims are pyo3-free (scanner, mol, params); the Python method-dispatch + GEOMOPT-01 no-runtime-dep proof are 07-09. All checkpoint+shim tests always-on (internal-only model scanner, no SCF/cintx); real-SCF arm stays #[ignore]'d per 07-04. Full pyscf-geomopt: 45 passed/1 ignored/0 failed; dependency-wall PASS.
- [Phase 07 P07-05]: UHF/RKS/UKS gradients (D-09 order) reuse the 07-03 RHF grad_elec decomposition. UhfReference = alpha/beta RhfReference pair (mirrors UmpReference); UHF grad = total-density Hellmann-Feynman + per-spin 2e Pulay + spin-summed overlap Pulay. RksGradients/UksGradients add the XC-potential derivative on the byte-exact Phase-4 Becke grid (get_vxc/get_vxc_uks via the NATIVE xcfun backend — numint eval_rho/eval_xc/eval_uks + GTOval_sph_deriv1, NEVER libxc per T-07-16; cargo tree confirms no libxc_rs) + the optional grid_response Becke-weight-derivative term (with_grid_response/extra_force, default OFF, fully supported on request). D-04: none call CPHF (grep -i cphf empty in all three) — grid_response is a grid-weight-derivative, NOT a response solve (Pitfall 5). All four variational NUMERIC arms #[ignore]'d on the SAME 6 missing cintx families (07-01/07-03 gate); the KS XC-grid term + grid_response run cintx-INDEPENDENTLY always-on. lib.rs EDITED (vs 07-03 unchanged): flat re-exports for RhfGradients..UksGradients + References (the surface 07-07 wires cphf/mp2 on). Known stub: the extra_force per-grid weight-gradient ∂w_g/∂R is a structural zero today (the grids weight-derivative surface not yet exposed); the on/off toggle + reduction shape are exercised always-on. eval_rho LDA takes the value-block (comp 0) slice of GTOval_sph_deriv1.
- [Phase 07 P07-04]: pyscf-geomopt — the native BFGS+RFO redundant-internal geometry optimizer (the phase's biggest novelty, NO in-tree analog) ported from geomeTRIC. geomeTRIC license CONFIRMED "BSD 3-clause (aka BSD 2.0) Non-AI License" (clauses 1-3 standard BSD-3, clause 4 anti-AI-training) — compatible for the algorithm port; deviates from RESEARCH A3's plain-BSD-3 expectation; recorded in lib.rs. API: GeometryOptimizer { conv_params, maxsteps, has_constraints } + optimize(opt, scanner, mol) -> OptimizeResult { coords, converged, nsteps, e_tot } drives the 07-02 GradScanner. The RFO step (aug-Hessian eig via pyscf_algebra::eigh_gen) + G- pseudo-inverse route the algebra wall; every reduction oracle_sum/oracle_dot. Displacement unit CONFIRMED Bohr (Pitfall 6). Added pyscf-gto (set_geom_) to the dep set (Rule 3, wall-clean). 07-06 (shims/checkpoint) + 07-09 (PyO3) wire against this engine.
- [Phase 07 P07-04]: h2o_equilibrium gate SPLIT (D-02) — equilibrium_via_model_scanner ALWAYS-ON (drives the FULL loop internals->B->RFO->back-transform->set_geom->converge via an internal-only translation/rotation-invariant analytic harmonic PES, converges grms 7e-6 in 6 steps, no SCF/cintx) + equilibrium_via_rhf_gradient #[ignore]'d on the cintx grad-integral workstream (int2e_ip1 + int1e_ip{ovlp,kin,nuc,rinv} MISSING, same gate as 07-03). The 5 GAU convergence defaults (1e-6/3e-4/4.5e-4/1.2e-3/1.8e-3) + maxsteps=100/trust=0.1/tmax=0.3 LOCKED (GEOMOPT-04). Blondel-Karplus dihedral s-vectors validated vs FD: s_b=-(p+1)s_a+q s_d, s_c=p s_a-(q+1)s_d.
- [Phase 07 P07-03]: RhfGradients { reference: RhfReference (mo_coeff/mo_energy/mo_occ/mol), atmlst, de } implements the base Gradients trait — grad_elec (Hellmann-Feynman + 2e Pulay + overlap Pulay), make_rdm1e, get_ovlp per-method; grad_nuc/kernel inherited. Free-fn forms (make_rdm1e/grad_elec/get_ovlp/aoslice_by_atom) are PyO3-bridge reusable (07-09). RhfReference is the converged-SCF snapshot shape every variational grad (07-05) + the geomopt scanner (07-04) reuse.
- [Phase 07 P07-03]: rhf_verify_fd NUMERIC arm is #[ignore]'d (cintx workstream pending) — int2e_ip1 + int1e_ip{ovlp,kin,nuc,rinv} + with_rinv_at_nucleus MISSING from cintx (07-01-SUMMARY, no scheduled workstream) so RhfGradients::kernel() ?-routes a clean Core(InvalidMolecule) cintx-availability error (NEVER NotYetImplemented{phase:7}); the STRUCTURAL arm (5 tests) + the verify_fd wiring are complete and un-gate by dropping the #[ignore]. lib.rs LEFT UNCHANGED (the 07-02 trait already carried grad_elec/make_rdm1e/get_ovlp). Reductions through oracle_sum/oracle_dot NOT pyscf_algebra::gemm (still a Phase-2 stub). D-04: no CPHF.
- [Phase 07 P07-02]: verify_fd operates over per-atom coords (Fn(&[[f64;3]])->Result<f64>) NOT a Mole — method-agnostic; per-method wave (07-03) adapts its Mole as_scanner into this coord closure. disp default 1e-4 Bohr, tol 1e-6 Ha/Bohr, all reductions through oracle_sum (no bare +=).
- [Phase 07 P07-02]: GradScanner is two boxed Send+Sync closures (EnergyClosure + GradClosure) returning (Energy, de) — fixes the geomopt seam (07-04 consumer) before any method body; mirrors pyscf-scf::as_scanner capture-by-value discipline.
- [Phase 07 P07-02]: atmlst (GRAD-08) + verify_fd (GRAD-09) are base-API-from-day-one (D-09) — built into the Gradients trait / resolve_atmlst helper so every GRAD-01..07 inherits them. grad_nuc is a real shared Coulomb-force port; get_ovlp/hcore_generator are NotYetImplemented seams (need cintx int1e_ipovlp/iprinv + with_rinv_at_nucleus, MISSING per 07-RESEARCH D-02 — RHF wave un-gates once cintx workstream lands).
- [Phase 07 P07-02]: new trait named Gradients (plural) to avoid colliding with the pre-existing pyscf_core::Gradient (singular). GradError carries Mp2Error + CcsdError #[from] bridges (grad consumes both post-SCF crates) + InvalidDisplacement (T-07-05).

- Roadmapping (2026-05-10): Compressed research's 12-phase suggestion to 8 phases (standard granularity). Merged `bindings` into `scf` (Phase 3) to lock PyO3 contract on RHF before DFT; merged `geomopt` into `grad` (Phase 7); merged `GPU enable + oracle hardening + distribution` into closing Phase 8.
- Roadmapping (2026-05-10): Phase 1 (Foundation) is the SHOWSTOPPER convergence point — 7 of 21 catalogued pitfalls have their primary mitigation here (FMA, reduction order, cubecl pin, panic policy, sibling-crate ABI, cross-platform libm, scope creep).
- Roadmapping (2026-05-10): Phase 3 (SCF + PyO3 bindings) is the second convergence point — 5 PyO3-related pitfalls (subclass override, NumPy stride, GIL deadlock, panic→exception, chkfile schema) plus eigenvector sign canonicalization land here on the small RHF surface.
- Algebra integration (2026-05-10): added a dedicated `pyscf-algebra` crate as the single owner of all linear algebra; only `pyscf-algebra` (and `pyscf-runtime` for client construction) may depend on `cubecl-*` runtime crates — enforced by a `cargo metadata` dependency-wall lint. Workspace grows 14 → 15 members.
- Algebra integration (2026-05-10): workspace `gpu` umbrella feature is OFF by default; CPU is the default backend. Per-backend features `cuda`/`wgpu`/`rocm`/`metal` opt in to each cubecl runtime at compile time. `PYSCF_BACKEND` env var selects among compiled-in backends at runtime; unrecognised/uncompiled values fall back to CPU with a `tracing::warn!`.
- Algebra integration (2026-05-10): host eigh/Cholesky/QR/SVD remain on `faer 0.24` behind the `pyscf-algebra` surface — even on a GPU build, these routines copy to host. Documented as the single intentional host-fallback path until `cubecl-linalg` ships an eigh.
- [Phase 02]: Wave 0 complete: cintx + cubecl-cpu reach proven; pyscf-kernels added to algebra-wall allowlist; 23-entry intor layout table; oracle harness scaffold + env-var docs in place
- [Phase 02]: pyscf-gto uses direct per-member cintx path-deps (cintx-core, cintx-rs, cintx-compat, cintx-ops, cintx-runtime) — workspace [patch.crates-io] cintx redirect alone is insufficient for subcrate consumers
- [Phase 02]: cubecl 0.10.0 ArrayArg::from_raw_parts signature is (Handle, usize) by value — no vectorization arg, no turbofish (older 0.9-era README sketch is stale)
- [Phase 02]: [Phase 02]: Mole >=30 attribute floor + format_atom 4-of-5 atom-input forms shipped via pyscf_gto::M(MoleBuildArgs); 5th Callable form returns NotYetImplemented{phase:3}; Local raw_atm_layout slot constants in pyscf-core::basis_set are TEMPORARY (02-04 deletes once cintx-compat dep lands)
- [Phase 04]: pyscf-grids byte-exact Becke grids (DFT-04/09) — generator-port Lebedev (SphGenOh + inline LEBEDEV_SEEDS, D-06, no codegen/build.rs), Treutler-Ahlrichs class-default radial, get_partition pure-Python fallback with pbecke.sum(axis=0) through oracle_sum (Pitfall 10). DFT-09 count sweep matches upstream level 0..9; DFT-04 byte-for-byte coords+weights is a CI-only grid_weights oracle arm (--features python).
- [Phase 04]: XC parsers + XcBackend seam (DFT-02/03) — libxc-default parse_xc (D-01, inline const XC_CODES/XC_ALIAS, part-aware possible_*_for fuzzy lookup, depth-bounded compound expansion T-04-05b) + xcfun-alternate parse_xc (0..77 ids, X/C/XC suffix fallback, LR_HF-zeroing tail). XcBackend cfg-gated enum mirrors AlgebraClient: Xcfun default-compiled, #[cfg(libxc)] Libxc in a gated submodule (default build never names a libxc_rs symbol). xcfun eval uses spin-resolved Vars (A_B/A_B_GAA_GAB_GBB/+TAU) with closed-shell rho/2 split (CPU launch supports spin-resolved only; Vars::N/A => NotConfigured). DFT-02 oracle = hand-transcribed parity table (PyO3-wall: no pyo3 dep in pyscf-dft); SLATERX bit-exact 1e-10 vs analytic. libxc NEVER compiled (cargo tree default = 0 libxc_rs).
- [Phase 04]: RKS/UKS core (DFT-01/08/10/11, D-07/D-08) — NumInt grid loop (nr_rks/nr_uks/eval_rho/eval_xc, upstream numint.py signatures) is algebra-orchestrated (AO via pyscf_gto::eval_gto behind the wall; dense ρ/Vxc contractions as host loops; Exc/nelec via oracle_sum) with NO #[cube] kernel (D-07; Tensor-API gemm/axpy stay NotYetImplemented{phase:2}, so the grid loop follows the Phase-3 SCF/DF inline-loop precedent). PARSE XC IN THE XCFUN NAMESPACE (default backend) — xcfun exposes the standard-hybrid mixing in hyb[0] (b3lyp→0.2); the libxc parser folds it inside compound id 402 (hyb=0), so using libxc::parse_xc would silently break hybrid_coeff AND feed libxc ids into the xcfun id→name map. D-08: NumInt reads DType::from_env() at construction + read-only dtype() accessor; f32/f64 enum-match dispatch of the matmul chain (F64 arm = unchanged bit-exact default; F32 casts ρ→f64 at the XcBackend::eval boundary since eval_gto/xcfun are f64-host) + one below-bit-exact tracing::warn!; no set_precision, no f32 tolerance gate. KS get_veff = J+Vxc−hyb·K (RSH omega!=0 seam → 04-07); KsHooks overrides energy_elec = Tr(D·h1e)+Ecoul+Exc via a per-cycle Exc cache (the SCF energy_elec signature has no mol). RKS/UKS reuse the Phase 3 kernel<H> verbatim. DFT-01 bit-exact energy gate is the CI-only --features python rks_energy/uks_energy oracle arms (live convergence needs working arity-3/4 ERIs = the Phase-2 int2e_sph/int3c2e_sph rollup gap, currently NotYetImplemented; minao init guess also not yet implemented) + an always-on structural layer; the RKS/UKS drivers are complete and converge once working ERIs land. From<DftError> for PyscfRsError bridge in pyscf-dft (no pyscf-core dep cycle). pyscf-dft stays pyo3-free + cubecl-free; libxc NEVER compiled.
- [Phase 06] (06-07, diagnostics + frozen contract / CCSD-09/10): get_t1_diagnostic/get_d1_diagnostic/get_d2_diagnostic port ccsd.py:748-776 into pyscf-ccsd/src/diagnostics.rs (was a Wave-3 stub). T1 (Lee-Taylor) = ||t1||_F/sqrt(nelec) via oracle_sum of squares; nelec is passed EXPLICITLY (not derived from t1.shape[0]) so the caller controls normalization — value equals upstream 2*nocc form when nelec=2*nocc. D1 (Janssen) / D2 (Nielsen) port the FULL upstream definition: max over BOTH the ij Gram block (einsum('ia,ja->ij')=t1·t1ᵀ / einsum('ikab,jkab->ij')) and the ab Gram block (einsum('ia,ib->ab')=t1ᵀ·t1 / einsum('ijac,ijbc->ab')) of sqrt(max|eigenvalue|) — the two blocks share nonzero eigenvalues so the values coincide, both computed for exact parity. Gram blocks built host-loop materialize-then-oracle_sum (no gemm/+=), diagonalized via pyscf_algebra::eigh_gen(matrix, identity, n) (S=I reduces the SCF generalized-eigh to a plain symmetric eigh — NO new eigh entry point); eigh_gen's +inf linear-dep padding skipped so a rank-deficient Gram stays finite. ShapeMismatch-validated, never OOB-panics. CCSD-10 frozen contract (tests/frozen.rs, NON-tdd validation over already-shipped helpers): CCSD's frozen-aware active space (default_ao2mo's eris.nocc/nvir) == the MP2-08 helpers (get_nocc/get_nmo/get_frozen_mask/mo_without_core) for Frozen::Count/List/Auto on a REAL in-tree LiH/STO-3G RHF→CCSD run (nocc=2,nmo=6); Count(1)/List([0]) drop exactly the frozen orbital; eris blocks (ovov/vvvv/oooo + mo_energy) sized to the active space; frozen-core e_corr (-0.0202318) rises toward zero vs all-electron (-0.0204491). KEY: Frozen::Auto through the CCSD path is ELEMENT-BLIND (count 0 == None) because default_ao2mo routes through pyscf_mp2::get_frozen_mask (the data-only helper surface carrying no charges) — this IS the verbatim cc/ccsd.py:35 reuse contract (no new frozen logic added); the MP2 numeric kernel resolves chemcore with real charges via frozen_mask directly (a separate path, out of CCSD-helper scope). ccsd.rs UNMODIFIED (kernel already routes frozen through helpers since 06-03). LiH chosen over H2/STO-3G (1 occ orbital can't be frozen meaningfully); H2O/HF all-electron error in the larger-nvir vvvv int2e transform (pre-existing, out of scope) but their frozen paths converge. fmt clean; no new crate dep; 0 libxc.
- [Phase 06] (06-05, amplitude-DIIS / CCSD-04 / D-06): AmplitudeSubspace impls DiisStorable by packing t1 + lower-triangular t2 (pack_tril of t2.transpose(0,2,1,3)) into one flat vector byte-matching ccsd.py:670 amplitudes_to_vector; vector_to_amplitudes round-trips symmetric t2 (unpack_tril filltriu=SYMMETRIC). dot routes through pyscf_algebra::oracle_dot (Pitfall 9 — DIIS path drift), NOT iter().sum(). ONE DIIS / second storable: reuses the entire Phase-3 pyscf_diis::Diis<S> machinery (B-matrix oracle_dot + solve_linear + oracle_sum extrap) with NO new DIIS body. The 06-03 ccsd_kernel run_diis NO-OP is replaced by Diis::<AmplitudeSubspace>::new(6) (diis_space=6 NOT SCF's 8, ccsd.py:926; diis_start_cycle=0, ccsd.py:928); error vector = packed residual (t1new-t1, t2new-t2). ccsd_kernel is now a thin wrapper over ccsd_kernel_diis(.., diis: bool) with diis=true (mycc.diis=True) — preserves all existing callers + exposes the no-DIIS path for the iter-count test. DECISION: the DIIS-vs-non-DIIS same-energy assert uses 2*CONV_TOL (2e-7), NOT the plan's literal 1e-9 — the dual criterion accepts a run at |dE|<CONV_TOL(1e-7) so two convergence paths legitimately land ~CONV_TOL apart (DIIS converges in 8 iters to e_corr=-0.020524527 matching the published H2/STO-3G ref; non-DIIS trips the loose CONV_TOL_NORMT=1e-5 criterion at 12 iters slightly earlier). 1e-9 is tighter than the kernel's own convergence guarantee. RAYON 1==8 bit-identical.
- [Phase 04]: RSH range-coulomb + VV10 NLC (DFT-05/06) — RSH via the env[8] (PTR_RANGE_OMEGA) mechanism: pyscf-gto::range_coulomb OmegaGuard (RAII set/restore of Mole._env[8], restore-on-drop incl. error/unwind path, T-04-07a) + intor_with_omega + get_k_with_omega drive the STANDARD int2e (NOT phantom int2e_lr_/int2e_sr_ symbols, Pitfall 1). veff::default_get_veff RSH branch (rks.py:108-129): omega!=0 → vk = hyb·K + (alpha−hyb)·K_lr via get_k_with_omega(+omega) on an Arc-backed Mole clone (shared &Mole needs no &mut, omega local + auto-restored). KsVeff gained half_tr_d_vxc so the energy cache is RSH-correct (the old `veff−J+hyb·K` Vxc reconstruction is wrong once vk carries the LR term — Rule-1 bug fix). OPEN QUESTION A5 RESOLVED: cintx safe API (ExecutionOptions/OperatorEnvParams) has f12_zeta (env[9]) + grids_params but NO range_omega (env[8]) setter, AND arity-4 int2e is NotYetImplemented{phase:2} — so the env[8] set/restore contract is owned at the pyscf-gto layer (complete+tested) and the numerical RSH ERI flips on only via a cintx#11-style gap-closure (safe-API env[8] reader + arity-4 int2e). VV10 (DFT-06) ports the pure-Python _vv10nlc double-loop (numint.py:526-538, Pitfall 4: NOT C VXC_vv10nlc) over a coarser nlcgrids (a separate Grids instance): per outer point double-loop over inner vv grid → F/U/W via oracle_sum (T-04-07b), exc/vrho/vsigma per numint.py:552-554; nr_nlc_vxc orchestrates (outer==inner==nlcgrids, excsum=oracle_dot(den,exc), symmetrized GGA Vxc). NlcCoeffs hardcodes only the bare 'VV10' default (5.9/0.0093, A1); per-functional → libxc nlc_coeff. CAM-B3LYP is libxc-only on the corpus (xcfun XC_CODES has no entry; libxc id 433) — the always-on RSH test uses an xcfun-namespace RSH(0.19*HF+0.46*LR_HF(0.33)+0.81*LYP); CAM-B3LYP/VV10 energy gates CI-gated. libxc NEVER compiled.
- [Phase 04]: Gap closure CR-04 (04-12) — The KS per-cycle energy cache in `hooks.rs` (KsHooks) and `df_dft.rs` (DfKsHooks) keyed `(Exc, 0.5·Tr(D·Vxc))` on `dm_fingerprint = Σ|D|` (L1 norm) with a `(c.fp - fp(dm)).abs() < 1e-12` hit guard. This is NON-INJECTIVE: two distinct density matrices can share an L1 norm, and at µHartree convergence (where the 1µH bit-exact gate operates) a genuine step-to-step Exc change can hide behind an unchanged Σ|D|, returning a STALE XC energy. Replaced with an INJECTIVE `dm_fingerprint(&Density) -> u64` that hashes each element's raw f64 bit pattern (`v.to_bits().hash(&mut h)`) via `std::collections::hash_map::DefaultHasher` (SipHash, stdlib — NO new crate dep, satisfying threat T-04-12-SC's no-install disposition). Both files use the IDENTICAL scheme; the cache-hit guard is now exact `c.dm_fingerprint == dm_fingerprint(dm)` (no float approximation). Hashing the bits (not the value) is deliberate: -0.0 != 0.0 and distinct NaN payloads differ, so the cache reuses Exc only for a byte-identical density. The cache mechanism + its grid-loop-recompute miss-fallback are retained (only the key changed). New `dm_fingerprint_is_injective` test: two dm with Σ|D|=4 but different entries (`[1,-1,1,-1]` vs `[2,-2,0,0]`) produce different u64 keys, and an identical dm is deterministic. `cargo test -p pyscf-dft` exits 0 (43 lib unit + all integration suites green).
- [Phase 04]: Gap closure CR-02 (04-13) — The f32 precision matmul chain in `numint.rs` (`eval_rho_scalar<f32>` `contract` closure + `nr_rks_inner<f32>` Vxc back-contraction) used `S::from(x).unwrap_or_else(S::zero)` and `t.to_f64().unwrap_or(0.0)` throughout, which on f64→f32 overflow silently substituted a wrong value — violating the D-08 "honest f32 path" contract (user should get a loud signal, not silent corruption). CRITICAL FINDING: the plan assumed `S::from(overflow)` returns `None`, but `num-traits 0.2.19` `f32::from(1e40_f64)` returns `Some(f32::INFINITY)` — so the prescribed `.ok_or_else(...)?` alone would have left `Ok([inf])` (still silent). Added `PyscfRsError::NumericOverflow { context: &'static str }` to error.rs and two helpers in numint.rs: `cast_finite<S>` (`ok_or` on the defensive `None` arm PLUS an `x.is_finite() && !s.is_finite()` check that catches a finite f64 narrowing to a non-finite f32 — the REAL overflow mode) and `back_to_f64<S>` (flags a non-finite f32 *accumulation* via `S::KIND != F64 && !t.is_finite()`). Every `unwrap_or_else(S::zero)`/`unwrap_or(0.0)` in the f32 numeric chain replaced with these `?`-propagating helpers; the `contract` closure became `Result<Vec<f64>, PyscfRsError>` with `?` at its 3 call sites. The f64 DEFAULT path is bit-identical: for `S=f64` both helpers are the identity (the `S::KIND==F64` guard skips the finiteness rejection so a legitimately non-finite f64 passes through exactly as the old `unwrap_or(0.0)`-on-f64 — which was simply `t` — did). `to_f64` is reached as a method via the `Scalar: num_traits::Float: ToPrimitive` supertrait (no num-traits dep added to pyscf-dft, honouring T-04-13-SC no-install + libxc-compile avoidance). New `f32_overflow_returns_err_not_zero` test (nao=1, ao=[1e40], dm=[1e40]) asserts `Err(NumericOverflow)` not `Ok(([0.0],None))`; `f64_path_unchanged_no_overflow_on_large_values` proves f64 computes 1e80 cleanly. `cargo test -p pyscf-dft` (45 lib + all integration incl. dtype_f32_smoke, rks_uks_bitexact) + `-p pyscf-core` (11 lib) green; clippy `-D warnings` + fmt clean. No new crate dep.
- [Phase 04]: Gap closure CR-01 (04-14) — `NumInt::nr_uks` was DEAD CODE: it ran `nr_rks` on the TOTAL density (`Dα + Dβ`, the closed-shell path) and returned `vmat: (r.vmat.clone(), r.vmat)` — the SAME Vxc matrix cloned into both spin channels, so it could never produce an open-shell potential. Fixed across 3 files. (1) `xc_backend.rs`: added `UksXcOutput` (per-spin `vrho_a`/`vrho_b` + GGA `vsigma_aa`/`ab`/`bb`) + `XcBackend::eval_uks` + private `xcfun_eval_uks` that builds the per-point xcfun input from the GENUINE `rho_a[ip]`/`rho_b[ip]` (NOT the closed-shell `rho/2` symmetric split), using spin-resolved `Vars::A_B` (LDA) / `A_B_GAA_GAB_GBB` (GGA) — so `vrho_a != vrho_b` for asymmetric densities; MGGA + libxc UKS arms return clean `BackendEval`. (2) `numint.rs`: rewrote `nr_uks` as a genuine open-shell loop — eval the AO block once, contract `rho_a`/`rho_b` (+`∇rho`) INDEPENDENTLY, build `sigma_aa`/`sigma_bb`/`sigma_ab` for GGA, call `eval_uks`, then a new `uks_vmat(grad_this, grad_other)` helper back-contracts TWO DISTINCT Vxc matrices (LDA `0.5·w·vrho·φμφν`; GGA adds same-spin `2·w·vsigma_same·(∇rho_this·∇φμ)·φν` + cross-spin `w·vsigma_ab·(∇rho_other·∇φμ)·φν`, `V+Vᵀ` symmetrized); per-spin nelec + combined excsum via `oracle_sum`. `nr_uks` runs f64-only (xcfun is f64-host; the D-08 f32 matmul chain stays the closed-shell `nr_rks` concern — scope boundary). (3) `hooks.rs` + `uks.rs` + `pyscf-py/dft.rs`: added `UksKsHooks` (open-shell KS hooks; `get_veff` routes through `nr_uks`, combined `Vxc = Vxc_a + Vxc_b`, RSH-aware vk; `UksEnergyCache` keyed on TWO injective fingerprints reusing the CR-04 `dm_fingerprint`); `UKS::kernel` uses `UksKsHooks::new` (NOT `KsHooks::new`); `PyUKS::get_veff` routes through `UksKsHooks::get_veff_ks`→`nr_uks` (NOT `ks_default_get_veff`/`nr_rks`). Symmetric `dm_a=dm_b=dm/2` split is the STRUCTURAL-WIRING contract: the generic `pyscf_scf::kernel<H>` (kernel_impl.rs:59,127) carries a SINGLE total `Density` and calls `get_veff(mol,&dm)` once per cycle, so genuine asymmetric alpha/beta SCF state is out of scope (requires generalizing `kernel<H>`) — the open-shell machinery is complete and yields distinct per-spin Vxc the moment an asymmetric `(dm_a,dm_b)` is fed to `nr_uks` directly. DEVIATION (Rule 3): the GGA input-build loop's `saa/sab/sbb.unwrap()` tripped the crate's `#![warn(clippy::unwrap_used)]` under the `-D warnings` CI gate → rewrote as `if let (Some,Some,Some)`. New `nr_uks_asymmetric_spin_gives_different_vmat` (H2/sto-3g, α in AO0, β in AO1) proves `vmat_alpha != vmat_beta` (RED failed on the clone; GREEN passes); `uks_kernel_uses_nr_uks_not_rks_path` structural test confirms `UksKsHooks` satisfies both `OverrideHooks` + `KsOverrideHooks`. `cargo test -p pyscf-dft -p pyscf-py` exits 0 (47 dft lib + all integration + py tests green); clippy `-D warnings` + fmt clean. No new crate dep; libxc NEVER compiled. This was the FINAL Phase-04 gap-closure plan — all 4 BLOCKERs (CR-01..04) now closed.
- [Phase 04]: Gap closure CR-03 (04-11) — `c2s_coeff` in `pyscf-kernels::eval_gto` was `fn(u32,usize,usize)->f64` with an unconditional `panic!` on l>4 (h-shells: cc-pV5Z, ANO). Through the PyO3 panic→exception bridge this still aborts the Python process (FOUND-07 never-panic violation). Converted to `-> Result<f64, PyscfRsError>`: l<=4 arms wrapped in `Ok(...)` (FROZEN libcint coeffs byte-unchanged), l>4 wildcard returns `Err(NotYetImplemented{phase:4})`. `?`-propagated through `eval_gto_sph_cpu`/`eval_gto_sph_deriv1_cpu` and the public `eval_gto_sph`/`eval_gto_sph_deriv1` (now `Result`-returning). `pyscf_gto::eval_gto` was ALREADY `Result`-returning, so its public signature is unchanged — `numint.rs` `eval_gto_block` and every other downstream consumer compile untouched; only the two internal `?` additions and 3 integration-test `.expect(...)` were needed. No new dependency; libxc never compiled.
- [Phase ?]: [Phase 04]: DF-DFT + KsResult chkfile (DFT-07, D-10/D-06 reuse) — RKS::density_fit precomputes pyscf_df B integrals; DfKsHooks routes the Coulomb-J build through get_jk_df ((J_df, K_standard) split, T-04-08b) while Vxc/K stay standard, so get_veff_ks is identical to the non-DF KS path. KsResult wraps ScfResult: on-disk /scf group byte-identical to the SCF schema (upstream from_chk compat) PLUS xc/grids_level/grids_scheme metadata; impl Checkpointable via pyscf_chkfile primitives + the re-exported hdf5 alias (NO own hdf5-metno dep, D-05); load bounded/validated, never panics (T-04-08). ndarray added (F-order view, not hdf5). DFT-07 energy + ORACLE-08 h5py gates CI-only behind the Phase-2 int3c2e_sph gap + libpython/h5py; structural + Rust-Rust round-trip layers always-on. libxc NEVER compiled.
- [Phase 05]: 05-01 scaffold — `pyscf-ao2mo` registered as the 20th `pyscf-*` member (D-01) with `general`/`full` stub surface + `Ao2moError` bridging to `PyscfRsError`. `pyscf-mp2` deps wired (ao2mo/scf/df/gto/algebra/runtime) strictly pyo3-free + cubecl-free (`xtask check-dependency-wall` PASS), 9-module skeleton + `Mp2Error` bridge. The five MP2-08 helper signatures (`get_nocc`/`get_nmo`/`get_frozen_mask`/`get_e_hf`/`mo_without_core` — the verbatim `cc/ccsd.py:35` CCSD import contract; Python `_mo_without_core`→Rust `mo_without_core` via `#[doc(alias)]`) exported; the always-on `ccsd_import_contract` symbol-existence arm passes. Five MP2 numeric oracle arms registered in `KNOWN_METHODS` (len 13→18: `mp2_rmp2_energy`/`mp2_ump2_energy`/`dfmp2_energy`/`dfmp2_native_energy`/`mp2_rdm`), len-assert updated. CI: always-on `mp2-structural` job + `if: false` cintx#11-gated `mp2-oracle-cintx-gated` numeric job (needs arity-4 `int2e` for in-core + arity-3 `int3c2e_sph` for DF; mirrors DF-HF/DFT-01 gating). MP2 python `dispatch` match arms + all numeric/kernel bodies deferred to 05-02..05-06 (catch-all `UnknownMethod` arm covers the names until then; gated job never runs). Pure scaffolding — ships NO compute.
- [Phase ?]: [Phase 05]: 05-02 AO→MO transform — transform::quarter_transform implements the (pq|rs)→(iq|rs)→(ij|rs)→(ij|ks)→(ij|kl) quarter-transform as host loops (gemm is NotYetImplemented{phase:2}); every per-index sum materializes products into a reused Vec then oracle_sum (4 call sites, 0 bare += in the contraction) → bit-exact + thread-count invariant (T-05-02-FP). general(eri_ao,nao,[&MOCoefficients;4]) ports the eri_ao.size==nao**4 einsum branch of ao2mo/incore.py:125-128 (real-only: .conj() no-op); full = general(..,[mo_coeff;4]). F-order flat-index doc-commented at every boundary (Pitfall 3). T-05-02-SHAPE: validated at entry → ShapeMismatch, never OOB/panic. The 05-01 stub signatures (&[&[f64]]/&[f64]) were replaced (no external callers). Always-on synthetic-ERI roundtrip (the ONE un-gated numeric assertion this phase) asserts general/full/identity bit-exact vs an independent staged longhand reference. check-no-fma + check-dependency-wall PASS.
- [Phase ?]: [Phase 05]: 05-03 in-core RMP2 — rmp2_kernel ports mp2.py:47-76 closed-form (t2=(ia|jb)/(εi+εj−εa−εb), edi=2·oracle_dot(gi,t2i), exi=−oracle_dot((ib|ja),t2i)); EVERY reduction via oracle_dot/oracle_sum (no += accumulator) → bit-exact + thread-count invariant (T-05-03-FP); verified vs synthetic ChemistsEris (1×1 e_corr=−0.125, 1×2 longhand). default_ao2mo builds frozen-aware co/cv subsets + ao2mo::general([co,cv,co,cv]) with eri_ao=intor(int2e); int2e NotYetImplemented{phase:2} propagates with ? (never panics/zeros, T-05-03-FFI), numeric flips on at cintx#11 (D-05). Five MP2-08 helpers real bodies over (mo_occ,&Frozen) = always-on CCSD import contract. Frozen enum None/Count/List/Auto/Window; chemcore element→ORBITAL OnceLock table VERBATIM from elements.py:1079 (119 entries bit-identical) summed DIRECTLY no ÷2 (PLAN said /2 but upstream chemcore() returns sum; O→1/Si→5). scs_energy 1.0/1.0=plain, 1/3,1.2=SCS. Mp2OverrideHooks+ChemistsEris+NoMp2Overrides pyo3-free (D-08); energy default→default_energy, rdm1/rdm2→NotYetImplemented{plan:4} (05-04). check-no-fma + dependency-wall PASS.
- [Phase ?]: [Phase 05]: 05-04 UMP2 + RDMs — ump2_kernel ports ump2.py:35-109 open-shell e_corr = e_aa + e_bb + e_ab: same-spin (aa/bb) antisymmetrized 0.5·(direct − exchange), opposite-spin (ab) direct-only (no exchange), each channel using its own α/β orbital energies; e_corr via oracle_sum([e_aa,e_ab,e_bb]). New UmpAmplitudes { t2aa, t2ab, t2bb } spin-resolved triple (NO in-repo analog; single-channel pyscf-core::Amplitudes does not cover it); UmpReference reuses Mp2Reference twice (alpha/beta, mirrors ump2.py mo_*[0/1]). ump2_kernel takes the three ChemistsEris blocks DIRECTLY (synthetic-block always-on test path; pyscf-py/default_ao2mo build them per channel once arity-4 int2e lands). RDMs: gamma1_intermediates (doo=−dm1occ / dvv) ports mp2.py:175-203; make_rdm1 assembles nmo×nmo MO 1-RDM (doo+dooᵀ, dvv+dvvᵀ, +2 occ diag) with ao_repr (C·γ·Cᵀ) + with_frozen core-diag embedding; make_rdm2 builds nmo0^4 Chemist 2-RDM (dovov occ/vir placement + Chemist transpose, dm1 contribution, separable HF +4/−2, frozen oidx/vidx fancy-index maps). make_rdm2 ao_repr=true returns NotYetImplemented (nmo^4 AO back-transform deferred to Phase-7 gradients) — never silently-wrong. EVERY reduction via oracle_sum/oracle_dot (no += ; T-05-04-FP, bit-exact + RAYON 1==8). Tr(γ)==2·nocc proven analytically (Tr(doo)==−Tr(dvv)). RDM hooks seams stay cintx#11-gated (need default_ao2mo int2e to produce t2); free fns ship now. check-no-fma + check-dependency-wall PASS. requirements MP2-02 + MP2-05 complete.
- [Phase ?]: [Phase 05]: 05-05 conventional DF-MP2 — DFRMP2/DFUMP2 reuse the RMP2/UMP2 base and swap the ERI source to the pyscf-df B-tensor (D-06). df_ao2mo MO-transforms b_uvq (AO row-major [nao,nao,naux]) into B^Q_ia = sum_{mu,nu} C_mu^i*b[mu,nu,Q]*C_nu^a (half-transform then second contraction, materialize-then-oracle_sum), then assembles (ia|jb) = sum_Q B^Q_ia*B^Q_jb via oracle_dot over the Q axis (the libmp.MP2_contract_d MATH, no C dep; T-05-05-FP no bare +=). DFRMP2 implements Mp2OverrideHooks::ao2mo->df_ao2mo; dfrmp2_kernel wires it into rmp2_kernel verbatim (a borrowing DfRmp2Hooks avoids cloning the B-tensor). DFUMP2/dfump2_kernel: same-spin aa/bb via df_ao2mo per spin; ab cross-spin (ia|JB)=sum_Q B^Q_ia(a)*B^Q_JB(b) from per-spin MO transforms of the SHARED *-ri aux. A2/T-05-05-AUX: aux default is pyscf_df::default_ri (mp2fit *-ri), NOT the JK-fit aux (pinned by acceptance grep; doc comments reworded to drop the literal jkfit token). df_ao2mo takes a pre-built DfIntegrals; the int3c2e_sph cintx#11 gate is surfaced at the caller's cholesky_eri and ?-propagates, never panics/zero-substitutes (T-05-05-FFI, the Phase-4 CR-02 lesson). Always-on tests: synthetic DfIntegrals+toy reference assert df_ao2mo == longhand sum_Q B^Q*B^Q, dfrmp2_kernel returns hand-computed e_corr, shape-mismatch errors (no panic/OOB), cholesky_eri on a real Mole propagates the gate without panicking; numeric oracle stays cintx#11-gated. check-no-fma + check-dependency-wall PASS. requirement MP2-04 complete.
- [Phase ?]: [Phase 05]: 05-06 native RI-MP2 fast path (D-06 additional) — dfmp2_native is a SEPARATE module from the conventional dfmp2 path (own path pyscf.mp.dfmp2_native; upstream DFRMP2 subclasses lib.StreamObject, NOT mp2.RMP2 — so NOT the default mp.DFMP2 factory). emp2_rhf ports dfmp2_native.py:374-427: transform_b_to_iqb MO-transforms pyscf_df::cholesky_eri's b_uvq (REUSE the shipped 3c Cholesky — NO second Cholesky, Don't-Hand-Roll) into the native [i,Q,b] layout (ints_cholesky order [mo1,aux,mo2]), then walks occupied PAIRS (j<=i) forming per-pair Kab[a,b]=Σ_Q B^Q_ia·B^Q_jb=(ia|jb) via kab_from_slices (oracle_dot per (a,b)), accumulating 2(ps+pt)·ΣTab·Kab − 2pt·ΣTab·Kabᵀ (j<i) + ps·ΣTab·Kab (j==i diag); ps=pt=1.0 (generic SCS lives in scs_energy). emp2_uhf ports dfump2_native.py:272-355: same-spin (aa,bb) pt-scaled antisymmetrized (Kab−Kabᵀ)/DE over j<i + opposite-spin (ab) ps-scaled direct over all i(α)×j(β). NativeDFRMP2/NativeDFUMP2 driver structs (+kernel()) compute e_corr ONLY (no SS/OS split, no t2 — the conventional dfmp2 path owns the decomposition+amplitudes); reported as (e_ss=0,e_os=e_corr)/(e_aa=0,e_ab=e_corr,e_bb=0) placeholders. default_ri (mp2fit *-ri) NOT default_jkfit. solve_cphf_rhf STATUS-MARKER STUB → NotYetImplemented{plan:5} (D-06: relaxed RDM is the optional native extra needing orbital-gradient machinery + arity-4 int2e to produce the response RHS un-gated; energy is the core MP2-04 deliverable; intended pyscf_algebra::solve_linear landing documented in the stub). EVERY reduction via oracle_dot/oracle_sum (22 sites, 0 bare += energy accumulation; T-05-06-FP). int3c2e_sph cintx#11 gate ?-propagates, never panics/zero-substitutes (T-05-06-FFI). Always-on cross-check (T-05-06-XCHECK): native emp2_rhf == conventional dfrmp2_kernel e_corr on the SAME synthetic B-tensor across (2,2,3)/(2,3,4)/(3,2,3)+1×1 (RELATIVE tolerance — different reduction orders are bit-close not bit-identical; an absolute 1e-10 falsely failed at the synthetic ~3e6 energy magnitude, Rule-1 test-tolerance fix). emp2_uhf matches an independent longhand reference on a synthetic open-shell pair. cargo test -p pyscf-mp2 green (19 lib + all integration); clippy -D warnings + fmt clean; check-dependency-wall PASS (cubecl-free). NO new crate dep; libxc NEVER compiled. MP2-04 native half complete (requirement was already complete from 05-05's conventional half).
- [Phase ?]: [Phase 05]: 05-07 MP2 PyO3 bridge — PyRMP2/PyUMP2/PyDFMP2 eager-snapshot mf into a plain-array Mp2Reference (D-07) + hold Py<PyAny> mf; Mp2PyBridge impls Mp2OverrideHooks with an is_overridden __qualname__ base-class check: subclass ao2mo/make_rdm1/make_rdm2 override -> slf.call_method1 (Pitfall 7), else pure-Rust default_ao2mo/df_ao2mo/NoMp2Overrides under py.detach (BIND-05); kernel itself does NOT detach (hooks re-enter Python). PyDFMP2::kernel routes the reused rmp2_kernel through the bridge with DefaultAo2mo::Df(B-tensor) so a DFMP2 subclass ao2mo override is honored. MP2() #[pyfunction] factory: istype('UHF')->UMP2 / with_df->DFMP2 / else RMP2 (GHF out of v1 scope); frozen=None/int/list/'auto'. PyMp2Scanner holds mf.as_scanner(), __call__(mol) re-runs reference -> re-snapshot -> MP2 kernel -> e_hf+scs_energy (MP2-07). DF aux=default_ri (mp2fit *-ri). python/pyscf/mp overlay re-exports _native.mp. Always-on 4-arm structural test incl. synthetic scanner-closure -> -1.125; numeric MP2 + live dispatch-parity stay cintx#11-gated/Manual-Only CI. No new crate dep; libxc NEVER compiled. MP2-01/02/04/05/06/07 complete.
- [Phase 03]: 03-13 minao init guess (SCF-05) — implemented the DEFAULT `minao` init guess (faithful port of `pyscf/scf/hf.py:init_guess_by_minao`), so `RHF(mol).density_fit().kernel()` + plain RHF work OUT-OF-THE-BOX (the default config no longer errors on minao). Three pieces: (1) NEW `pyscf_gto::intor_cross(mol_a, mol_b, name)` — cross-basis arity-2 overlap `<A|int1e_ovlp|B>` via the combined two-basis BasisSet off-diagonal block (generalized `build_int3c2e_combined_basis` → `build_combined_basis`; int3c2e is now a thin caller); (2) ported `NRSRHF_CONFIGURATION` (119 rows Z=0..118, [s,p,d,f]) + `frac_occ` from `pyscf/data/elements.py`/`atom_hf.py` into `pyscf-scf/src/atom_config.rs`; (3) `init_guess_by_minao` = build an ANO reference Mole (`'ano'` basis loads via ano.dat; minao.py is .py-only, NOT loadable), occ vector from frac_occ over the per-element l-shells, `dm = mo·diag(occ)·moᵀ` with `mo = S_working⁻¹·S_cross` (`S_cross = <working|ANO>` via intor_cross; the project_mo_nr2nr identity), oracle_sum reductions. GOLD-STANDARD: minao H2/STO-3G dm = [0.94758917, 0.09227308, 0.09227308, 0.94758917] BYTE-MATCHES the upstream `init_guess_by_minao` docstring to 1e-8. Default-config (minao) RHF converges + matches the 1e guess; H2O/STO-3G RHF converges to -74.963 out-of-the-box. Flipped `kernel_propagates_jk_not_yet_implemented` → `default_minao_config_converges`. DOCUMENTED CAVEAT (not chased): the vendored ano.dat loads 1 contraction per l for at least H/O, so atoms needing >1 contraction per l (O 1s+2s) under-normalize in minao (H2O Tr(dm·S)≈7.9 vs 10) — a DATA-coverage limit, not an algorithm bug (H byte-matches exactly); RHF still converges to the right energy; full ANO coverage is a follow-up. `atom`/`huckel` guesses remain NotYetImplemented. No new crate dep; 0 libxc; clippy -D warnings + fmt + check-no-fma + check-dependency-wall PASS. SCF-05 → `[~]` (minao/1e/chkfile/dm0 done); SCF-07 default path now fully out-of-the-box (only upstream byte-identity remains, CI-gated).
- [Phase 03]: 03-12 DF-HF end-to-end lock-in (SCF-07) — with int2e (05-08) + the rank-revealing DF-metric fit (05-09), DF-HF runs END-TO-END: `RHF::density_fit(aux)` + `DfHooks` (routes get_jk/get_veff through `pyscf_df::get_jk_df`) + the generic SCF `kernel<H>` loop converge to a DF-HF energy matching non-DF RHF within DF accuracy. Verified always-on in-tree (H2/STO-3G, `1e` guess): RHF -1.1168 (un-ignored `h2_no_overrides_converges` — the int2e_sph gap that blocked it is closed); DF-HF[weigend] |Δ vs RHF|=4.6e-5 (eff. naux=21), DF-HF[cc-pvdz-jkfit] |Δ|=2.0e-4 (eff. naux=43); minimal sto-3g aux converges (poor fit, SCF-loop robustness). New `crates/pyscf-scf/tests/dfhf_end_to_end.rs` cross-checks DF-HF vs non-DF RHF (the DF-MP2-vs-in-core pattern from 05-09) — no upstream PySCF needed. SCOPE: uses the `1e` init guess; the DEFAULT `minao` guess is still NotYetImplemented (→ 03-13) and the default `density_fit().kernel()` errors on it until then (`kernel_propagates_jk_not_yet_implemented` still passes via the minao error, flips in 03-13). `get_jk_df` + the SCF loop proven numerically correct. No new crate dep; 0 libxc; clippy -D warnings + fmt + check-no-fma + check-dependency-wall PASS. SCF-07 marked `[~]` partial (end-to-end numeric done; minao default + upstream byte-identity remain). Upstream-PySCF byte-identity is CI-gated/human-verify.
- [Phase 05]: 05-09 DF-metric robustness gap-closure — closed the LAST DF-MP2 (MP2-04) numeric blocker surfaced by 05-08: the DF 2-center metric `(P|Q)` is frequently ill-conditioned/rank-deficient for real aux bases (cc-pvdz-jkfit AND weigend), and `cholesky_eri`'s plain Cholesky-Banachiewicz (`s<=0` pivot) rejected it. NEW `pyscf_algebra::df_metric_fit(j2c,n,lindep) -> (W column-major n×rank, rank)` (df_metric.rs): faer `SelfAdjointEigen` rank-revealing inverse-sqrt fit — drops eigenvalues ≤ `DF_METRIC_LINEAR_DEP=1e-9` (upstream PySCF `LINEAR_DEP_THRESHOLD` route), `W·Wᵀ = (P|Q)⁻¹` on the kept subspace. `cholesky_eri` keeps the Cholesky+forward-sub PD fast path BIT-FOR-BIT and, on `SingularAux`, falls back to `df_metric_fit` building `b_uvq[μν,k]=Σ_P (μν|P)·W[P,k]` via oracle_dot (no bare +=); `DfIntegrals.naux` becomes the EFFECTIVE rank ≤ auxmol.nao_nr. Mirrors upstream PySCF's try-Cholesky-then-eigh exactly (maximizes eventual byte-identity). RESULTS: cc-pvdz-jkfit + weigend `(P|Q)` now build real finite B-tensors (un-ignored `df_integrals_shape::h2o_cc_pvdz_df_integrals_shape` → always-on); DF-MP2 e_corr **-0.04424** (eff. naux=21) ≈ in-core RMP2 **-0.04428** to ~4e-5 (the DF fitting error); the gold-standard `df_b_tensor_reconstructs_exact_eri` check shows `Σ_Q B B` reconstructs the real `intor(int2e)` (μν|λσ) to **1.7e-3** (validates int3c2e+int2c2e+metric fit independent of any kernel). df_metric unit tests: PD inverse / rank-deficient pseudo-inverse / tiny-negative dropped (no NaN) / Singular / shape. No new crate dep (faer already in pyscf-algebra); 0 libxc; clippy -D warnings + fmt + check-no-fma + check-dependency-wall PASS. DF-HF (Phase 3) shares `cholesky_eri` so its metric is now robust too (its own SCF closure remains). MP2-04 numeric fully lit up in-tree.
- [Phase 05]: 05-08 cintx#11 numeric gap-closure — cintx now ships arity-4 `int2e_{sph,cart}` (api_manifest.rs:166/183, SHELL_TUPLE_CAPACITY=4) + arity-3 `int3c2e_sph` (api_manifest.rs:404), libcint-byte-identical at the cintx source (`safe_api_arity4_parity`/`center_3c2e_parity`; cintx-rs now asserts against all-zero output, api.rs:750). Wired the missing pyscf-gto dispatch layer: `intor.rs` `evaluate_arity4` (shell-quad loop → SessionRequest → F-order `[nao;4]` stitch, the `eri_ao` convention pyscf_ao2mo::transform documents) replaces the `3|4 => NotYetImplemented{phase:2}` branch (component-leading arity-4 = int2e_ip gradients → `NotYetImplemented{phase:7}`; plain arity-3 → clear error, int3c2e uses intor_with_auxmol). `evaluate_int3c2e_with_auxmol` replaces the all-zeros stub with real evaluation over a COMBINED orbital+aux BasisSet (`projection::build_int3c2e_combined_basis`, PySCF fakemol/conc_mol — aux shells re-based onto appended aux atoms; i,j∈orbital, k∈aux; aux AO offset = shell_offset(k)−nao). Shape surprises ?-propagate (no panic/no zero-substitute, T-05-08-FFI). The MP2 kernels are UNCHANGED (D-05): in-core RMP2 numeric now runs end-to-end (mp2_numeric_smoke.rs: H2/STO-3G `e_corr=-0.04428`, finite ≤0, thread-invariant); flipped `default_ao2mo_propagates_int2e_not_yet_implemented` → `..._succeeds_after_cintx11_closure`. Always-on in-tree gates: int2e_arity4.rs (finite/non-zero/8-fold) + int3c2e_auxmol.rs (finite/non-zero/bra-symmetric). CI: `mp2-oracle-cintx-gated` (if: false) → `mp2-oracle-upstream-manual` (workflow_dispatch + pyscf install) — upstream byte-identity is the human-verify arm (sandbox lacks numpy/PySCF, 02-10 precedent). FINDING (NOT chased, T-05-08-SCOPE): unignoring the H2O/cc-pVDZ DF shape test surfaced a SEPARATE Phase-3 issue — the cc-pvdz-jkfit AND weigend `(P|Q)` DF metrics are ill-conditioned and the plain Cholesky-Banachiewicz (`s<=0` pivot) rejects them; int3c2e itself ships (int3c2e_auxmol.rs proves it). So DF-MP2 numeric (MP2-04) is unblocked at the integral layer but gated on a Phase-3 rank-revealing DF-metric Cholesky in pyscf-algebra. No new crate dep; Cargo.lock UNCHANGED; libxc NEVER compiled. clippy -D warnings + fmt + check-no-fma + check-dependency-wall PASS. Also unblocks Phase-3 DF-HF + Phase-4 bit-exact RKS/UKS (int2e/int3c2e now real) — their own oracle closures.
- [Phase 02]: 02-10 GTO-05 eval-half gap-closure — the cintx Phase-19/20 workstream SHIPPED `int1e_ecp_{cart,sph}` (Type-1 local + Type-2 projector) byte-identical to vendored PySCF nr_ecp (cintx `safe_api_ecp_parity.rs` pins atol=1e-12). cintx is a PATH dep already pointing at the merged tree (Cargo.lock UNCHANGED — no git-rev pin bump needed; the plan's git-rev sketch was superseded by the Phase-1 D-15 path-dep topology). New `ecp_engine_cintx::CintxEcpEngine` replaces `EcpEngineNotAvailable` as the default `pyscf_gto::ecp_engine()`; the stub stays in-tree (documentation + testable error path, exercised DIRECTLY in updated `ecp_engine_stub.rs` tests rather than via `ecp_engine()`). KEY DEVIATION from the plan's speculative sketch: the cintx safe-API ECP preflight (`SessionRequest::query_workspace`) returns `FacadeError::MissingEcpBasis` unless `basis.ecp_shells()` is non-empty, but `mol.basis_set` is built ECP-free (`build_cintx_basis_set` → `BasisSet::try_new`). So the engine builds an ECP-augmented `BasisSet` on demand via a NEW `projection::build_cintx_basis_set_with_ecp` (projects per-element `mol._ecp` ParsedEcp → cintx `EcpChannel::Local`/`Projected(l)` + `EcpShell`, one shell per (atom,channel,distinct n_power) — mirrors make_ecp_env's `_ecpbas` row grouping), then iterates AO shell pairs through `SessionRequest` exactly like the non-ECP `intor::evaluate_arity2` and stitches an F-order nao×nao matrix into `Density::from_flat` (new pyscf-core helper). int1e_ecp on an ECP-LESS mol returns the canonical `EcpEngineNotAvailable` via a `mol._ecp.is_empty()` guard (preserves the 02-07 user-facing error contract). Always-on in-tree gate `crates/pyscf-gto/tests/ecp_int1e_oracle.rs` (Cu/LANL2DZ → finite, non-zero, symmetric matrix) PASSES under `cargo test -p pyscf-gto --test ecp_int1e_oracle`. The upstream byte-identity pytest `tests/oracle/test_ecp_int1e.py` is shipped + the `dump_intor_for_oracle` harness extended with `PYSCF_RS_ORACLE_ECP`, but it CANNOT run in this sandbox (no numpy/upstream-pyscf; the entire oracle suite is gated on `tests/oracle/requirements.txt`) — downgraded to a human-verify item (cintx already pins 1e-12 byte-identity to nr_ecp at the source). xtask check-dependency-wall + check-cubecl-pin PASS. No `#[ignore = "Pending cintx ECP"]` annotations existed to remove. Phase 7 GRAD-07 (ECP gradients via `int1e_ecp_ipnuc_*`, manifest ids 28/29) now unblocked. GTO-05 fully closed: loading ✅ + eval ✅.
- [Phase 02]: 02-11 general-contraction parser fix: nwchem.rs emits N contractions per N coeff columns (was truncating to col 1); projection.rs feeds cintx ROW-MAJOR coeffs [prim*nctr+ctr]. Closes 03-13 minao heavy-atom caveat (H2O Tr(dm.S) 7.9->9.86; minao unnormalized so <nelec, H2 dm traces 1.976/2.0). cintx l>=3 nctr>1 asymmetry surfaced as DI-02-11-CINTX-NCTR-HIGHL. Consumed cintx 6b14d48 via path-dep. GTO-02/03 done.
- [Phase 03]: 03-15 mulliken_meta (SCF-09) — shipped the real `mulliken_meta` (meta-Löwdin population analysis, `pyscf/scf/hf.py:1301-1340`) via a NEW `crate::orth::orth_ao` (`crates/pyscf-scf/src/orth.rs`). orth_ao = the GLOBALLY-ORTHONORMAL sequential block-Löwdin scheme of `pyscf/lo/nao.py::_nao_sub`: partition AOs into per-`l` angular-momentum CHANNELS (cross-atom — `_bas[ANG_OF]` walk; spans atoms like upstream's core/valence/Rydberg classes so HOMONUCLEAR SYMMETRY is preserved), then for each channel project out the previously-orthogonalized span in the S-metric (`c ← c − C_done·C_doneᵀ·S·c`) and Löwdin within block (`lowdin` = `S^{-1/2}` via `eigh_gen(S,I)` + λ^{-1/2}, drop λ≤1e-15), final phase-adjust (flip column if diagonal<0). KEY DEVIATION (Rule 1): the plan said partition by (atom,l) + naive block-diagonal Löwdin, but that is NOT globally orthonormal — leaves cross-block overlap (H2 `C_orthᵀ·S·C_orth` off-diag 0.659; H2 charges ±0.659 asymmetric), failing both the orthonormality gate AND the Σ ao_pop≈nelec conservation invariant. Fixed to per-`l` channels + the `_nao_sub` project-then-Löwdin order. mulliken_meta then: `c_inv=C_orthᵀ·S`, `D'=c_inv·D·c_invᵀ`, `pop[μ]=D'[μ,μ]` (S=I), shared `aggregate_pop_to_charges` (refactored out of mulliken_pop — single oracle-reduction site, mulliken_pop tests stay green). Conservation MEASURED: H2 Σ ao_pop=2.0/Σ chg≈−2.7e-15/H charges equal to 1e-15; H2O Σ ao_pop=10.0/Σ chg≈−6.2e-15 (O +0.622, H −0.314/−0.309). All reductions via oracle_sum (10 sites in orth.rs); no FMA; no new unwrap; no new crate dep; `cargo tree -p pyscf-scf`=0 libxc; clippy -D warnings + fmt + check-no-fma + check-dependency-wall PASS. SCF-09 → `[~]` (partial; full NAO `_nao_sub` core/valence/Rydberg byte-identity is a documented future enhancement + human-verify, mirrors SCF-07). NotYetImplemented gone from analyze.rs.
- [Phase 03]: 03-14 atom + huckel init guesses (SCF-05 → [x]) — shipped the LAST 2 of 5 init_guess modes. NEW `crate::atom_hf::get_atm_nrhf` (`crates/pyscf-scf/src/atom_hf.rs`) is the shared per-unique-element spherically-averaged atomic-RHF engine (port of `pyscf/scf/atom_hf.py:27-205`): builds a single-atom neutral Mole (working basis restricted to that element, spin=Z%2, cart=false), runs a small SCF whose eig step is the ANGULAR-AVERAGED solve (atom_hf.py:109-140 — group AOs by l, average the per-l Fock/overlap over the m-diagonal `einsum('piqi->pq')/degen`, `eigh_gen` the nsh×nsh block, scatter eigvecs back over the 2l+1 m-components), occupations from `frac_occ(Z,l)` (atom_hf.py:142-171), 1-electron H takes the AtomHF1e no-2e branch. `init_guess_by_atom` (hf.py:495-535) superposes per-atom dm blocks `atm_dm[i,j]=Σ_p occ·c·c` block-diagonally at each atom's molecular AO range (new shared `aoslice_by_atom` ATOM_OF+ao_loc_nr walk). `init_guess_by_huckel` (hf.py:537-555 + _init_guess_huckel_orbitals:577-670, GWH Kgwh=1.75 NON-updated rule) collects occupied atomic orbitals into the molecular AO basis, builds `orb_S=orb_Cᵀ·S·orb_C`, GWH `orb_H[io,jo]=0.5·1.75·orb_S·(Ei+Ej)`, `eigh_gen(orb_H,orb_S)`, back-transform to AO, Aufbau-fill + make_rdm1. CLOSING GATE PASSES (T-03-14-CORRECT): atom & huckel-seeded RHF on H2/STO-3G each converge to the SAME e_tot as the 1e guess — all three = −1.1167143250625533 (bit-identical). Tr(D·S): atom=2.0 (exact, normalized atomic orbitals — minao non-normalization caveat does NOT apply), huckel=2.000000000000001. All reductions via oracle_sum/oracle_dot (10 sites atom_hf.rs + 6 sites init_guess.rs); no FMA; no new unwrap in production; no new crate dep; `cargo tree -p pyscf-scf`=0 libxc; clippy -D warnings + fmt + check-no-fma + check-dependency-wall PASS. Cartesian cart2sph branches in both return a clear NotYetImplemented Err (spherical-only; STO-3G is spherical). DEVIATION (Rule 3): module-scoped `#![allow(dead_code)]` on atom_hf.rs so the Task-1 commit is independently -D-warnings-clean while Tasks 2/3 wire in the consumers (inert once reached). SCF-05 → `[x]`: all 5 init_guess modes (minao/atom/1e/huckel/chkfile) + user-dm0 ship.

- [Phase 06]: 06-03 in-core RCCSD numeric headline (CCSD-01/03/11, D-02 — the UN-GATED headline) — filled the ccsd/rintermediates/update_amps Wave-1 stubs with a CONVERGING in-core RCCSD whose H2/STO-3G correlation energy `e_corr = -0.020524500477 Ha` matches the published FCI/CCSD reference `-0.020525 Ha` to ~0.5 µHartree, BIT-IDENTICAL under RAYON 1==8. **rintermediates.rs:** ported `rintermediates.py:30-188` as host loops — `make_tau`, `cc_Foo`/`cc_Fvv`/`cc_Fov`, `Loo`/`Lvv`, `cc_Woooo`/`cc_Wvvvv`(+`_into`)/`cc_Wvoov`/`cc_Wvovo`; EVERY einsum collects contracted-axis products into a Vec then `oracle_sum`s (no gemm, no bare += contracted accumulation); block lengths validated → ShapeMismatch before indexing (no OOB; `#![forbid(unsafe_code)]`); flat C-order offsets doc-commented at every boundary; `cc_Wvvvv_into` writes the nv⁴ tenant into a caller-supplied buffer. **update_amps.rs:** ported `rccsd.py:43-143` (Hirata Eqs.35-36) — `default_update_amps` + `default_update_amps_with_wvvvv`; the `t1new`/`t2new` assembly from the `cc_*` intermediates incl. the `'abcd,ijcd->ijab'` Wvvvv contraction (heaviest einsum). **ccsd.rs:** `ccsd_kernel<H>` (port `ccsd.py:44-101`) — HARD `pool.try_reserve(nv⁴·8)?` pre-flight BEFORE building eris (D-01, no downgrade), `pool.reserve` the Wvvvv buffer ONCE before the loop and reuse via `with_mut_slice` every cycle (Pitfall 20/CCSD-11), MP2-seeded `init_amps` (`ccsd.py:1050-1077`: t1=0, t2=(ia|jb)/Dijab, emp2 reuse ovov), dual-criterion convergence `|dE|<1e-7 AND normt<1e-5` within max_cycle=50 (converged in 12 iters), `release` after; `default_energy` (`rccsd.py:146-162`: 2·tau:ovov − tau:ovov via oracle_sum); `default_ao2mo` builds the FULL block set (oooo/ovoo/oovv/ovov/ovvo/ovvv/vvvv) from ONE intor('int2e') via 7 ao2mo::general calls (F-order→C-order reorder per block) + diagonal canonical Fock; verified-default constants MAX_CYCLE=50/CONV_TOL=1e-7/CONV_TOL_NORMT=1e-5/DIIS_SPACE=6/DIIS_START_CYCLE=0. **KEY DECISION (Rule-3 blocking deviation):** ported `rccsd.py:43-143` (the clean rintermediates-driven RCCSD) NOT the production `ccsd.py:104-285` update_amps (a blocking/prefetch/H5TmpFile-swap optimized impl that is NOT a 1:1 host-loop port); both converge to the IDENTICAL energy (verified ≤1µH vs FCI). **conv_tol_normt=1e-5** (the verified CCSDBase class attr, ccsd.py:923 — NOT the CONTEXT 1e-6; RESEARCH A1 resolved). Tests: `rccsd_numeric_smoke.rs` (always-on CCSD-01, real RHF→RCCSD→≤1µHartree), `convergence.rs` (CCSD-03/CCSD-11: dual-criterion flag, RAYON 1==8 bit-identity, over-budget MemoryLimitExceeded refusal). DIIS slot is a NO-OP this plan (06-04 wires AmplitudeSubspace; kernel converges without it). 10 lib + 4 numeric + 6 (06-02 arena) tests green; clippy -D warnings clean; NO new crate dep; Cargo.lock NOT staged (no new dep; dirty-tree lock already satisfies --locked); libxc NEVER compiled.
- [Phase 06]: 06-02 WorkspacePool arena body + opaque Amplitudes handles (CCSD-11/D-01/D-08) — filled the Phase-1 budget-check-only `WorkspacePool` skeleton into a REAL reuse-pool: `reserve(shape, allow_spill) -> BufferId` scans the free-list for a released allocation of fitting size and REUSES it before allocating fresh (allocate-once-reuse, Pitfall 20); `release(id)` returns the buffer to the free-list WITHOUT dropping it. `TensorBackend { InMemory(Box<[f64]>) | Spilled(SpillHandle) }` — `SpillHandle` wraps a `pyscf_chkfile::hdf5` temp file (D-07, NO new hdf5 dep) and RAII-drop-DELETES it (T-06-02-LEAK). HARD `PYSCF_MAX_MEMORY` refusal (D-01): over-budget in-core `reserve(allow_spill=false)` returns `MemoryLimitExceeded` BEFORE allocating (no buffer-id consumed, NO silent downgrade; T-06-02-OOM). Accessors `as_slice`/`write_slice`/`with_mut_slice` for the 06-03+ kernel working store. **A2 single-handle resolution:** pool-owned `pyscf-runtime` `BufferId` (NOT `pyscf_algebra::Tensor`'s — its inner field is `pub(crate)` to algebra AND pyscf-runtime must not dep pyscf-algebra, wrong direction). **D-01 handle home (deviation from the plan's literal text — Rule-4-avoided architectural constraint):** `AmplitudeStore { Owned(Vec<f64>) | Pooled(PooledRef{buffer_id:u64,shape}) }` lives IN `pyscf-core`, NOT pyscf-runtime — pyscf-runtime already deps pyscf-core (a reverse dep CYCLES) and its default `cpu` feature pulls cubecl (would violate pyscf-core FOUND-02). `PooledRef` is dependency-free; the CCSD call site reads the buffer through the pool. `Amplitudes::from_vec` (MP2/owned) + `from_pooled` (CCSD/arena) + `t1_slice`/`t2_slice` accessors keep the MP2 construction site (`mp2.rs`) + RDM readers (`rdm.rs` gamma1/make_rdm2 + toy_amps) + `rmp2_structural` test compiling — MP2 numeric behavior UNCHANGED (19 lib + integration green). Two CCSD-11 test targets: `tests/heap_alloc_count.rs` (A3 DEDICATED counting `#[global_allocator]` scoped to this test binary, NOT linked to oracle/determinism arms — proves bounded allocation across N=5 iters + pool never grows past 1 alloc under 50-cycle reuse), `tests/refusal.rs` (4 arms: over-budget try_reserve/reserve refuse with named bytes, one-byte-over still refuses, spill is opt-in not auto-downgrade). Rule-1 bug fixed: first cut called `try_reserve(need_bytes)?` before the spill decision so an over-budget buffer was refused even with `allow_spill=true` — moved the ceiling into the `!fits_inmem && !allow_spill` branch (spill bypasses the in-memory budget). DEFERRED: Cargo.lock NOT staged — the dirty working-tree lock has ~100 unrelated `libxc-kernel-*` drift entries (HEAD already stale at 856; the libxc_rs path members re-resolve on any cargo lock op); a surgical 2-line edit fails `--locked` (cargo wants the extra entries). The dirty lock-on-disk already satisfies `--locked` builds; lock-unification belongs to the integration gate (mirrors 06-01). `cargo check -p pyscf-runtime -p pyscf-core -p pyscf-mp2 -p pyscf-ccsd --locked` exits 0; clippy -D warnings clean; `cargo tree -p pyscf-runtime` libxc=0 + cubecl unchanged (only NEW transitive crate is hdf5-metno via pyscf-chkfile); libxc NEVER compiled.
- [Phase 06]: 06-01 pyscf-ccsd crate scaffold (CCSD-11) — filled the 5-line stub into a real compiling crate: full workspace-internal dep set (`pyscf-core`/`algebra`/`ao2mo`/`mp2`/`scf`/`df`/`diis`/`chkfile`/`gto`/`runtime` + thiserror/tracing) wired via `{ path = "../..." }` (NOT `{ workspace = true }` — the pyscf-* members are not registered as `[workspace.dependencies]`, only as `members`; the RESEARCH/PLAN sketch's `workspace = true` form would not resolve, so used the proven pyscf-mp2 sibling idiom). 17-module skeleton mirroring upstream `pyscf/cc/*.py` (error/eris/hooks/reference/ccsd/rintermediates/update_amps/uccsd/uintermediates/diis_amps/lambda/ulambda/rdm/urdm/diagnostics/dfccsd/direct). Four contract types: `CcsdError` (`ShapeMismatch{expected,got}` + `NotYetImplemented{wave:u8}` — field is `wave` NOT MP2's `plan` — + `#[from]` arms for `AlgebraError`/`CoreError`/`Ao2moError` PLUS the two CCSD additions `pyscf_runtime::BackendError` (D-01 try_reserve) and `pyscf_diis::DiisError`; bridge to PyscfRsError via `Core(InvalidMolecule)`); `ChemistsEris` (port of `_ChemistsERIs` ccsd.py:1389 — carries `oooo`/`ovoo`/`oovv`/`ovov`/`ovvo`/`ovvv`/`vvvv` flat C-order blocks + `fock`/`mo_energy` + nocc/nvir vs MP2's single ovov; re-exported from `eris` module not `hooks`); `CcsdOverrideHooks` (D-09 set: ao2mo/update_amps/energy/make_rdm1/make_rdm2; energy default-delegates to `ccsd::default_energy`, rdm1/rdm2→`NotYetImplemented{wave:3}`) + `NoCcsdOverrides`; `CcsdReference` (field-for-field Mp2Reference mirror). `update_amps` hook uses `pyscf_core::Amplitudes` for now (Wave-2 opaque-Tensor D-01 upgrade flows through same signature). Stub modules: `#![allow(dead_code)]` + bodies `NotYetImplemented{wave:N}` (1: ccsd/rintermediates/update_amps; 2: uccsd/uintermediates/diis_amps; 3: lambda/ulambda/rdm/urdm/diagnostics; 4: dfccsd/direct) — math lands 06-02..06-11. Added `("pyscf-ccsd","pyscf_ccsd")` to check_no_fma SCAN_TARGETS (FMA-scan scans 3 asm files now, PASS); check_dependency_wall LEFT UNMODIFIED (denylist already covers the cubecl-free crate — the CONTEXT.md "extend allowlist" phrasing was the inaccurate one per RESEARCH Pitfall 3; verified PASS). `cargo build/check -p pyscf-ccsd` exit 0; ZERO libxc + ZERO pyo3 anywhere in tree; transitive cubecl (via pyscf-algebra) + hdf5-metno (via pyscf-chkfile) are the legal carve-out owners (the wall invariant — NOT a violation; the PLAN's `cargo tree | grep` empty-expectation was a verify-command inaccuracy since pyscf-ccsd legitimately deps the cubecl/hdf5 owner crates). No external package installed; libxc NEVER compiled. Pure scaffolding — ships NO compute.

- [Phase 06]: 06-06 CCSD λ-equations + RDMs incl. ao_repr (CCSD-05/06, D-03) — filled the lambda/rdm Wave-3 stubs with the CCSD response surface. **lambda.rs (CCSD-05):** ported the closed-shell `ccsd_lambda.py` — `solve_lambda` (the CONCRETE `CCSD.solve_lambda`, `ccsd.py:1273`→`ccsd_lambda.kernel`, RESEARCH A6; the base `CCSDBase.solve_lambda` raises NotImplementedError — port the concrete-class behavior), `update_lambda` (the λ iterate; L1/L2 equations symmetrized `tmp + tmp.T(1,0,3,2)` like t2), `LambdaImds` (the `cc_Fov`/`Loo`/`Lvv`/`cc_Woooo`/`cc_Wvoov`/`cc_Wvovo` blocks reused from rintermediates). Seeds l1=t1/l2=t2 (the ground-state λ fixed point), iterates to `||Δl||<CONV_TOL_NORMT` within MAX_CYCLE (the verified 06-03 constants), HARD `try_reserve`+`reserve`-once on the wvvvv≈nv⁴ tenant; λ converges on H2/STO-3G. **rdm.rs (CCSD-06):** ported `ccsd_rdm.py` — `gamma1_intermediates` (doo/dov/dvo/dvv from t1/t2/l1/l2 via theta=2t2−t2_swap), `make_rdm1` (nmo×nmo MO 1-RDM: doo+dooᵀ / dvv+dvvᵀ / dov+dvoᵀ + `+2` occ-diag mean-field; Tr(γ)==nelec; ao_repr=true → C·γ·Cᵀ), `make_rdm2` (nmo⁴ MO 2-RDM: dovov + dm1 cross-term + HF separable +4/−2, mirroring pyscf-mp2/src/rdm.rs). **D-03 SHIPPED:** `make_rdm2(ao_repr=true)` returns a REAL nao⁴ AO 2-RDM NUMERICALLY this phase (unlike Phase-5 MP2's NotYetImplemented) — the MO 2-RDM routes through `pyscf_ao2mo::general` by treating it as the 'eri' (dim nmo) and passing the MO-coeff TRANSPOSE Cᵀ[mo,ao]=C[ao,mo] (shape [nmo,nao]) as each coefficient block → Γ_ao[μνλσ]=Σ C[μp]C[νq]C[λr]C[σs]Γ_mo[pqrs]; the nao⁴ AO RDM is the HEAVIEST arena tenant (try_reserve HARD pre-flight + reserve once + with_mut_slice write + as_slice read + release). EVERY contraction host-loop oracle_sum (the RDM `+=` are independent output-element scatter-adds, the MP2-rdm pattern, NOT contracted-axis accumulation); RAYON 1==8 bit-invariant on update_lambda AND the make_rdm2(ao_repr) path. **DEVIATION (Rule-1 test fix):** the first AO 'partial-trace invariant' assertion was mathematically wrong (Σ_μ C[μp]C[μq]=CCᵀ≠δ unless C is orthonormal in the IDENTITY metric, but RHF C is orthonormal in the AO-OVERLAP metric CᵀSC=I) — replaced with the rigorous C=I identity-transform check (AO==MO bit-for-bit) + the C≠I differs-from-MO check. **DEVIATION (documented deferral):** ulambda.rs/urdm.rs (open-shell UCCSD λ/RDM) shipped as documented module mirrors (NOT silent wrong numeric code) — the spin-resolved path reuses the closed-shell discipline and is wired with the Phase-7 open-shell response consumer (Known Stub). Tests: tests/lambda.rs (3 — λ converges/structural l~t/RAYON), tests/rdm.rs (4 — Tr==nelec/nmo⁴/ao_repr ships+differs/over-budget refusal); 6 rdm lib + 3 lambda lib green; prior-wave regression (rccsd_numeric_smoke/convergence/uccsd_smoke/diis_amps) green; AO_REPR_ARENA_OK confirmed. ccsd.rs NOT modified (constraint honored). NO new crate dep; Cargo.lock NOT staged (no new dep; dirty lock already satisfies --locked); libxc NEVER compiled. Out-of-scope: a pre-existing clippy absurd_extreme_comparisons in ccsd.rs:203 (06-05, file NOT modified by this plan) logged to deferred-items.md.

- [Phase 06]: 06-08 AO-direct CCSD (CCSD-07, mycc.direct=True) — ported the `_contract_vvvv_t2` AO-direct branch (`ccsd.py:473-570` / `_contract_s4vvvv_t2`) into `direct.rs`, trading the in-memory `nv^4` `vvvv` MO tensor for an on-the-fly AO-integral contraction. **RESEARCH Open Q4 RESOLVED-AT-EXECUTION (path b):** a grep of `pyscf-gto`'s intor surface (`intor.rs`) confirms there is NO shell-sliced streaming `int2e` primitive in-tree — only `intor("int2e")` returning the full arity-4 AO tensor. So `contract_vvvv_t2_aodirect` sources the full AO `int2e` ONCE and tiles the AO→MO `vvvv` transform over the LEADING virtual index `a` (one `[1,nv,nv,nv]` slice via `ao2mo::general([&cv_a,&cv,&cv,&cv])` at a time), contracting each slice against `tau` and discarding it → peak `vvvv`-MO buffer = `nv^3`, the full `nv^4` MO `vvvv` is NEVER materialized (satisfies CCSD-07's `direct=True` contract; a shell-sliced primitive that also streams the AO source is the documented v2 upgrade). **The vvvv-step split:** `cc_Wvvvv[a,b,c,d] = (two ovvv·t1 t1-corrections, nv^3) + vvvv[a,c,b,d] (the nv^4 integral part)`; only the heavy integral part moves to AO-direct (`= einsum('ijcd,acbd->ijab', tau, vvvv)`, `ccsd.py:474`), the t1-corrections stay in-core (`default_update_amps_direct` contracts them against `tau` touching only the `nv^3` ovvv block), both reassembled into the `[no,no,nv,nv]` block a NEW shared `update_amps_core` consumes — so in-core (`default_update_amps_with_wvvvv` via `vvvv_step_from_wvvvv`, byte-unchanged) and AO-direct differ ONLY in how the block is produced. **Kernel wiring:** `ccsd_kernel_direct`/`ccsd_kernel_direct_diis` route the vvvv step through the AO-direct branch + use `estimate_direct_vvvv_bytes`=`nv^3·8` (the LOWER pre-flight, skipping the full-`nv^4` arena reservation); wired as a separate entrypoint (not a bool threaded through `ccsd_kernel_diis`) because the AO-direct path needs the raw AO `int2e` + MO coeffs which only the in-tree default ao2mo exposes (not the generic `hooks.ao2mo` seam); the in-core `ccsd_kernel`/`ccsd_kernel_diis` are UNCHANGED. **PROOF (two levels):** (1) lib test — `contract_vvvv_t2_from_eris` (per-`a` tiled) == `contract_vvvv_t2_incore_full` (full `nv^4`) bit-close across (2,2)/(2,3)/(3,4); (2) integration test `tests/direct.rs` — `ccsd_kernel_direct` `e_corr` == `ccsd_kernel` `e_corr` on LiH/STO-3G, BIT-IDENTICAL `-0.020449057574` (both 8 iters, ≤1e-9 gate), PLUS the memory-frugality proof: a pool budget between `nv^3·8` and `nv^4·8` (LiH: 512 < 1280 < 2048) makes in-core HARD-REFUSE (`MemoryLimitExceeded` on the `nv^4` `try_reserve`, D-01 no downgrade) but AO-direct ACCEPT and converge — the on-disk witness of the lower peak reservation. RAYON 1==8 bit-invariant; ShapeMismatch-validated (T-06-08-SHAPE). **DEVIATION (Rule-1, plan-mandated incidental fix):** the clippy `absurd_extreme_comparisons` on `ccsd.rs` `Some(stack) if istep >= DIIS_START_CYCLE` (const `usize` 0 → always true, introduced by 06-05) is FIXED via a documented `#[allow(clippy::absurd_extreme_comparisons)]` preserving the configurable `>=` start-cycle semantics; confirmed gone (`cargo clippy -p pyscf-ccsd --tests | grep -i absurd` no match); `deferred-items.md` entry marked resolved. **DEVIATION (Rule-3):** the `update_amps` vvvv-step refactor (extract into a swappable block) was necessary to route the step through AO-direct without duplicating the amplitude equation; in-core path byte-unchanged (39 lib + all integration green; rccsd_numeric_smoke/uccsd_smoke/diis_amps no regression). System note: LiH/STO-3G (not H2/STO-3G) for the integration test because H2 has nvir=1 (`nv^4==nv^3==1`, can't distinguish reservations). NO new crate dep; Cargo.lock NOT staged; libxc NEVER compiled. CCSD-07 complete.
- [Phase 06]: 06-09 DF-CCSD + HDF5 spill (CCSD-08, D-05/D-07/D-08) — filled the `dfccsd.rs` Wave-4 stub + added the Phase-5 D-04 outcore AO→MO deferral to `pyscf-ao2mo`. **DF-CCSD = swap-the-ERI-source (the Phase-5 DFRMP2(RMP2) pattern, ZERO kernel rewrite):** `DfCcsdHooks::ao2mo` builds the FULL `ChemistsEris` block set (`oooo`/`ovoo`/`oovv`/`ovov`/`ovvo`/`ovvv`/`vvvv`) from the DF B-tensor — each `(pq|rs)=Σ_Q B^Q_pq·B^Q_rs` via `oracle_dot` over the auxiliary axis Q (the `dfmp2.rs::df_ao2mo` MATH generalized to all blocks), with the B-tensor MO-transform `transform_b_block` (materialize-then-`oracle_sum`, no gemm/+=); the in-core `ccsd_kernel<H>` is reused VERBATIM (only `ao2mo` swapped). `dfrccsd_kernel`/`DFRCCSD::kernel` wire it; aux default = `default_ri` (mp2fit `*-ri`, NOT jkfit), un-gated since 05-09. `block_sizing` ports the verified `dfccsd.py:93-96` dmax/vvblk formulas (D-08; `(nvira+3)//4` cap via `div_ceil`, BLKMIN floor). **vvL HDF5 spill (D-07/D-08):** the `vvL` half-tensor (the dominant tenant, `dfccsd.py:139`) is reserved with `allow_spill=true` so an over-budget run SPILLS to the 06-02 `WorkspacePool` `Spilled` backend (HDF5 via the `pyscf_chkfile::hdf5` alias — NO new hdf5-metno dep) instead of HARD-refusing; RAII drop-deleted, no leftover scratch. **DECISION (Rule-1, the load-bearing call):** `vvL` lives in a DEDICATED budget-matched `WorkspacePool::new(pool.budget_bytes)` inside `df_ao2mo`, NOT the kernel's shared pool — because the kernel reserves its in-core `Wvvvv` `[nv⁴]` from that pool with `allow_spill=false` + accesses it via `with_mut_slice` (in-memory-only); a released SPILLED vvL on the shared free-list would be wrongly reused for the next `Wvvvv` reserve (the free-list scans by SIZE, not backend → `with_mut_slice` fails on the spilled buffer), AND a larger reused buffer breaks `Wvvvv`'s exact-`nv⁴`-length check. The isolation pool keeps both tenants independent while spilling under the SAME PYSCF_MAX_MEMORY budget; the shared 06-02 pool's tested reuse contract is NOT modified (the size-only free-list match is a pre-existing behavior only this DF mixed-shape/backend usage exposes; the isolation sidesteps it cleanly). **ao2mo outcore (Task 1, the D-04 deferral):** `general_outcore`/`full_outcore`/`OutcoreScratch` port `pyscf/ao2mo/outcore.py`+`semi_incore.py` — the half-transform `[np,nq,nao,nao]` spills to an HDF5 scratch (the `pyscf_chkfile::hdf5` alias, D-07) between the first/second halves, peak resident = one s-slab; bit-exact == in-core `general`/`full` (same `oracle_sum` fold order); RAII drop-delete. **DECISION (Rule-3):** outcore slicing uses plain `Range<usize>`→hdf5 `Selection` (`impl From<Range<usize>> for Selection`) NOT the `ndarray::s!` macro — the macro emits `#[allow(unsafe_code)]`, colliding with the crate-wide `#![forbid(unsafe_code)]`. `pyscf-ao2mo` gains `pyscf-chkfile`+`ndarray` deps (no new cubecl/libxc source; libxc stays 0; the cubecl already in ao2mo's tree is the pre-existing pyscf-gto/pyscf-algebra chain — the `grep -ci cubecl==0` gate is the documented 06-02 verify-command inaccuracy). DFUCCSD ships its driver struct (open-shell wiring); the DF open-shell numeric parity is the 06-08-closeout human-verify arm (D-04). Tests: tests/dfccsd_spill.rs (5 always-on — ovov==longhand Σ_Q B^Q·B^Q reference 1e-12 rel, DF-CCSD converges + driver==free-fn, vvL spills under tiny budget + no leftover, spill-file observed created-then-deleted, in-core no-spill), 6 outcore lib tests; in-core RCCSD smoke regression green; `cargo check -p pyscf-mp2 --tests` (ao2mo consumer) green; ccsd.rs UNMODIFIED (constraint honored); RAYON-invariant oracle reductions; Cargo.lock NOT staged; libxc NEVER compiled. Out-of-scope: a pre-existing clippy `type_complexity` in `tests/rdm.rs:39` (verified on HEAD, NOT touched by this plan) logged to deferred-items.md. CCSD-08 complete.
- [Phase 06]: 06-10 PyO3 CCSD bridge (D-09, CCSD-01/02/05/06/08) — `pyscf-py::cc` is the ONLY pyo3 layer for CCSD (the PyO3 wall; `pyscf-ccsd` stays pyo3-free — `# NO pyo3` in its Cargo.toml). Copies `pyscf-py::mp` section-for-section (Mp2→Ccsd): **eager snapshot** — `snapshot_reference`/`snapshot_uccsd_reference` pull `mf.mol`/`mo_coeff`(F-order)/`mo_energy`/`mo_occ`/`e_tot`→`e_hf`/`converged` into a plain-array `CcsdReference`/`UccsdReference`; **override dispatch** — `is_overridden` (verbatim mp.rs:130-150 `__qualname__` base-class MRO comparison, Pitfall-7-immune) drives `CcsdPyBridge: CcsdOverrideHooks`: for each of the 5 hooks (`ao2mo`/`update_amps`/`make_rdm1`/`make_rdm2`/`energy`), override? → `slf.call_method1` (re-enters Python under the GIL) : → `Python::attach(|py| py.detach(|| <pure-Rust default>))` (BIND-05). **The kernel does NOT `py.detach` at the top** (hooks re-enter Python, the load-bearing mp.rs:359 comment) — the `update_amps` default is the biggest `py.detach` region in the project (the heaviest python3.13t GIL re-validation, the 06-11 smoke arm). `PyRCCSD`/`PyUCCSD`/`PyDFCCSD` `kernel` build `WorkspacePool::from_env()` + the bridge + call `ccsd_kernel`/`uccsd_kernel`; PyDFCCSD swaps the `ao2mo` default to `df_ao2mo` over a `default_ri`+`cholesky_eri` B-tensor. `solve_lambda`/`make_rdm1`/`make_rdm2(ao_repr=)`/`as_scanner` exposed; `PyCcsdScanner` is the Mole→energy callable (CCSD-07 geomopt seam, SCF-12/MP2-07 analog) with a self-less `ScannerDfBridge` for the DF re-run. `ccsd_factory` (#[pyfunction(name="CCSD")]): UHF→PyUCCSD / with_df→PyDFCCSD / else PyRCCSD (cc/__init__.py:83-139). **python/pyscf/cc/__init__.py** re-exports `_native.cc.{CCSD,RCCSD,UCCSD,DFCCSD,Scanner}` (BIND-02) + grafts `mf.CCSD()` onto the Rust `_native.scf.{RHF,UHF,GHF}` base classes (the upstream `scf.hf.SCF.CCSD = CCSD` cross-module dispatch); `mf.density_fit()` already carries `with_df` so `mf.density_fit().CCSD()` routes to DFCCSD via the factory. **DEVIATION (Rule-3 ×2):** (a) added the workspace-internal `pyscf-ccsd` dep to `pyscf-py/Cargo.toml` (NOT in the plan's files_modified — load-bearing for the bridge; default features only, libxc never pulled, T-06-10-SC accept); (b) `df_ao2mo`'s 06-09 signature is `(refr,frozen,df,pool)` not `(…,mo_coeff)` — dropped the bridge's `mo_coeff`/`refr`/`frozen` fields, the DF default path builds a fresh `WorkspacePool::from_env()` (the pool is Mutex-backed, not Clone). **KNOWN STUB:** the override paths (`ao2mo`/`update_amps`/`energy`) FIRE `call_method1` (the GIL re-entry is real) then run the pure-Rust default rather than marshalling the override's multi-block/amplitude NumPy return — the full round-trip is the 06-11 live `workflow_dispatch` arm (the 05-07 MP2 precedent; the sandbox has no maturin/PySCF). Tests: `crates/pyscf-py/tests/cc_bridge.rs` — 6 always-on arms (factory-dispatch + override-detect qualname logic + scanner-closure shape + the pyo3-free `default_energy` on a synthetic 1×1 ChemistsEris == -0.125 + the surface/GIL-discipline source assertions) GREEN. `cargo check -p pyscf-py` (DEFAULT features) + `cargo check -p pyscf-ccsd` pass; libxc NEVER compiled. Live numeric CCSD dispatch-parity + λ/RDM byte-identity + python3.13t GIL smoke deferred to 06-11. CCSD-01/02/05/06/08 surface complete.

### Pending Todos

[From .planning/todos/pending/ — ideas captured during sessions]

None yet.

### Blockers/Concerns

[Issues that affect future work]

- **cintx Wave 0.5 gates Phase 10 plan 10-05 (GTH pseudopotentials).** The 10 moment-weighted families (`int3c1e_r{2,4,6}_origk`, `int1e_r{2,4}_origi`) are a hard prerequisite: without them GTH pseudopotentials do not evaluate and no periodic SCF runs. Tracked in `cintx/.planning/notes/gradient-family-gap-closure-PLAN.md` §1.3b. Phase 10's other plans (10-01..10-04, 10-07) do not depend on it.
- **`pyscf_core::Unit::Ang` is CODATA-2014, upstream is CODATA-2010 — 4.951e-9 relative on every Angstrom-built geometry.** `crates/pyscf-core/src/mole.rs:483` still asserts `1.8897261339213` and its doc comment's claim to match `pyscf/data/nist.py BOHR` verbatim is FALSE. Deliberately NOT fixed: changing the constant moves every molecular v1.0 regression baseline. Every 1e-12-class PBC comparison specifies geometry in Bohr instead, and two tests (`oracle_phase9::angstrom_lattices_match_upstream_within_the_codata_gap`, `ewald::angstrom_reference_systems_match_upstream_within_the_unit_gap`) pin the deviation to exactly this constant so it fails loudly if either side changes.
- **cubecl 0.10.0 lockstep** with cintx/libxc_rs/xcfun_rs is a four-crate ABI contract. Any cubecl bump requires synchronized bumps in all four. Phase 1 documents the upgrade ritual; nightly cross-crate matrix CI is the early-warning system.
- **WGPU f64 holes** (cubecl issues #1316/#1317) may force `wgpu` feature to be gated on `shader-f64` Vulkan extension at runtime. Honest fallback to CPU with warning is the chosen path; verified in Phase 4 (DFT) and Phase 8 (GPU enable).
- **CCSD(T) deferral pressure** is real (~30–40% of CCSD users want it). v1.x P1 entry on the roadmap signals deferral is intentional; expect a feature request within weeks of v1 release.
- **`faer-ext 0.7.1` ↔ `faer 0.24.0` compatibility** needs build verification in Phase 1; if it fails, either bump faer-ext upstream or drop the dependency and round-trip via `Vec<f64>`.
- **h5py ↔ hdf5-metno chkfile round-trip** robustness needs empirical seal in Phase 3 (ORACLE-08 round-trip oracle).
- **libxc_rs per-functional feature gate — `PENDING_LIBXC_RS_FEATURE_GATE`** (Phase 04 plan 04-02, user checkpoint 2026-05-22 → *keep pending*). The sibling `~/Documents/workspace/libxc_rs` repo still unconditionally path-deps all 266 `libxc-kernel-*` crates (~6h compile). Deferred as a separate cross-repo workstream (its own PR/issue), mirroring the Phase 2 cintx-ECP coordination (cintx#11). The xcfun-default DFT path (04-04..04-08) is independent and proceeds; the `--features libxc` bit-exact assertions (04-05/04-06/04-09) and the dedicated libxc CI job (04-10) stay `#[cfg(feature="libxc")]`-gated and CI-only until this lands. Never trigger a default `cargo build` on libxc_rs.
- **cintx safe-API range-coulomb env[8] gap (Open Question A5 RESOLVED, plan 04-07)** — cintx *reads* `PTR_RANGE_OMEGA = env[8]` (verified in `cintx-compat::raw`) but its SAFE API (`cintx_runtime::ExecutionOptions` / `OperatorEnvParams`) exposes only `f12_zeta` (env[9]) + `grids_params` — there is NO `range_omega` (env[8]) setter, and arity-4 `int2e` is `NotYetImplemented{phase:2}` (the Phase-2 verification-rollup gap). pyscf-rs owns the env[8] set/restore contract at the pyscf-gto layer (`range_coulomb::OmegaGuard` over `Mole._env[8]`, complete + tested), so the RSH veff branch is correct; the NUMERICAL RSH ERI (and DF JK / bit-exact RKS energy) flips on only once cintx ships a safe-API env[8] reader on the int2e plan AND lands arity-4 int2e — a cintx#11-style cross-repo gap-closure. The CAM-B3LYP/H2O bit-exact energy assertion (DFT-05) is CI-gated behind this + the libxc backend; the VV10 energy match (DFT-06) is CI-gated behind the same Phase-2 ERI/init-guess gap as the 04-06 DFT-01 oracle. The RSH/VV10 code needs no change when these land.

### Quick Tasks Completed

| # | Description | Date | Commit | Directory |
|---|-------------|------|--------|-----------|
| 260512-8jv | Create issue in cintx repository about remaining tasks from pyscf_rs Phase 2 ([cintx#11](https://github.com/BectorVoom/cintx/issues/11)) | 2026-05-11 | 7dcdf08 | [260512-8jv-create-issue-in-cintx-repository-about-r](./quick/260512-8jv-create-issue-in-cintx-repository-about-r/) |
| 260512-8wb | Rewrite cintx#11 as cintx-only Phase 2 task list (drop pyscf_rs framing) | 2026-05-11 | f53cc0e | [260512-8wb-rewrite-cintx-11-as-cintx-only-phase-2-t](./quick/260512-8wb-rewrite-cintx-11-as-cintx-only-phase-2-t/) |
| 260522-b06 | implement f32/f64 precision switching using generics | 2026-05-22 | 4c6ab55 | [260522-b06-implement-f32-f64-precision-switching-us](./quick/260522-b06-implement-f32-f64-precision-switching-us/) |
| 260529-i2x | refactor gemm.rs to cubecl generic-float kernel + ROCm random-oracle test (passes on gfx1152) | 2026-05-29 | b720570 | [260529-i2x-refactor-gemm-rs-to-cubecl-generic-float](./quick/260529-i2x-refactor-gemm-rs-to-cubecl-generic-float/) |
| 260529-iji | refactor dot.rs to cubecl generic-float reduction kernel + ROCm random-oracle test (passes on gfx1152) | 2026-05-29 | 7ab843b | [260529-iji-refactor-pyscf-algebra-dot-rs-to-cubecl-](./quick/260529-iji-refactor-pyscf-algebra-dot-rs-to-cubecl-/) |
| 260529-jcx | refactor reduce.rs to cubecl generic-float partial-sum kernel + ROCm random-oracle test (passes on gfx1152) | 2026-05-29 | be22fe8 | [260529-jcx-refactor-reduce-rs-to-cubecl-generic-flo](./quick/260529-jcx-refactor-reduce-rs-to-cubecl-generic-flo/) |
| 260529-skl | refactor scal.rs to cubecl generic-float scale kernel + ROCm random-oracle test (passes on gfx1152) | 2026-05-29 | 687411a | (commits only — no quick dir) |
| 260529-mtx | refactor axpy.rs to cubecl generic-float kernel (y += alpha*x) + implement stub + ROCm random-oracle test (passes on gfx1152) | 2026-05-29 | 4ec6700 | [260529-mtx-refactor-crates-pyscf-algebra-to-cubecl-](./quick/260529-mtx-refactor-crates-pyscf-algebra-to-cubecl-/) |
| 260529-oj6 | refactor host_fallback.rs — implement eigh/cholesky/qr/svd via faer-on-host round-trip (ALG-05, NOT native cubecl kernels) + ROCm oracle differential tests (8/8 pass on gfx1152) | 2026-05-29 | 3100d3c | [260529-oj6-refactor-host-fallback-to-cubecl-faer-ho](./quick/260529-oj6-refactor-host-fallback-to-cubecl-faer-ho/) |
| 260530-l29 | consolidate cubecl backend-dispatch boilerplate — one `dispatch_backend!` macro replaces 16 hand-written cfg-gated 4-arm `match AlgebraClient` blocks across 8 pyscf-algebra engines (net −154 lines, byte-exact tests pass; ALG-06 wall intact; default+wgpu cfg-correct; clippy clean). pyscf-kernels/pyscf-dft scoped out (genuinely different dispatch axes) | 2026-05-30 | f70615c | [260530-l29-consolidate-cubecl-backend-dispatch-boil](./quick/260530-l29-consolidate-cubecl-backend-dispatch-boil/) |
| 260530-ljv | **[VALIDATED 5/5]** Phase-8 GPU-enable first slice: exported `dispatch_backend!` cross-crate (`#[macro_export]`); real `#[cube(launch_unchecked)]` s-shell eval_gto kernel + macro-wrapped multi-backend fanout in pyscf-kernels; device path gated on all-l0 (l≥1 → unchanged CPU, byte-identical). Differential oracle: CpuRuntime diff=0 (always-on), **real ROCm gfx1152 diff=1.11e-16** (~1 ULP, TOL 1e-9). DEFERRED remainder: l≥1 cart→sph kernel, deriv1/deriv2, pyscf-dft numint backend | 2026-05-30 | 12ea384 | [260530-ljv-gpu-enable-eval-gto-via-macro-wrapped-mu](./quick/260530-ljv-gpu-enable-eval-gto-via-macro-wrapped-mu/) |
| 260530-mlg | **[VALIDATED 7/7]** Phase-8 GPU-enable l≥1 slice: general `#[cube(launch_unchecked)]` eval_gto kernel (l 0..4, one thread per (g,shell)) over host-precomputed angular device tables (c2s_flat + cart-power + fac1 prefix-summed by l); `#[cube] ipow` helper (host helpers stay host-only — pitfall sidestepped); routed via `dispatch_backend!` on maxl≤4, l>4/empty → unchanged CPU (NotYetImplemented{phase:4} intact), all_s fast-path untouched. Differential oracle (p/d/f/g mixed): CpuRuntime diff=6.94e-18 (sub-ULP), **real ROCm gfx1152 diff=1.11e-16** (1 ULP). eval_gto_lge1 indep-reference gate green. DEFERRED: deriv1/deriv2 stencils, pyscf-dft numint backend | 2026-05-30 | bd293ff | [260530-mlg-gpu-enable-eval-gto-l-1-cart-sph-path-ge](./quick/260530-mlg-gpu-enable-eval-gto-l-1-cart-sph-path-ge/) |
| 260530-oms | **[VALIDATED 5/5]** Phase-8 GPU-enable deriv1 slice: `eval_gto_sph_deriv1` (value + 3 cartesian gradients, [4,ngrids,nao]) GPU kernel — REUSES the mlg angular tables + ipow, adds `#[cube] dpow` (lq·q^(lq-1)) + a second radial reduction (Σ−2α·c·exp), 4-component c2s write; routed via `dispatch_backend!` on maxl≤4, l>4/empty → unchanged CPU. Differential oracle (4-comp p/d/f/g): CpuRuntime diff=2.22e-16 (1 ULP), **real ROCm gfx1152 diff=4.44e-16** (2 ULP). deriv2 NOT portable (no CPU impl). DEFERRED: pyscf-dft numint device-path lock (AO eval already transitive via select_backend; eval_rho stays host-by-design for bit-exact SCF) | 2026-05-30 | e3f5221 | [260530-oms-gpu-enable-eval-gto-sph-deriv1-value-3-c](./quick/260530-oms-gpu-enable-eval-gto-sph-deriv1-value-3-c/) |
| 260530-p40 | **[VALIDATED 4/4]** Phase-8 GPU-enable FINALE — numint device AO-path lock (test + doc only, NO kernel/eval_rho change): explicit pyscf-dft test (H2O/cc-pVDZ, default CpuRuntime) proves numint's AO block routes through the GPU eval_gto kernel — (A) eval_rho over the device AO block == independent triple-sum within 1e-9; (B) GTOval_sph_deriv1 comp-0 == GTOval_sph within 1e-12 (cross-kernel lock). numint.rs documents: AO eval device-routed via select_backend (inherits ljv/mlg/oms kernels); **eval_rho stays HOST by design** (oracle_sum pairwise over nao² = FOUND-06 bit-exact; GPU eval_rho deferred perf item, would be tolerance-bounded). No libxc pulled (cargo tree guard). **Phase-8 eval_gto GPU surface COMPLETE** | 2026-05-30 | fe4e24f | [260530-p40-lock-the-numint-device-ao-eval-path-expl](./quick/260530-p40-lock-the-numint-device-ao-eval-path-expl/) |
| 260601-pbk | **DF-01 RESOLVED** (cintx `55bf984` + rs guards `a0b742a`). The mixed 2D Rys recurrence `g(n,m+1)+= n·b00·g(n-1,m)` was missing the `n` factor on the b00 cross term in BOTH 2c2e + 3c2e kernels (host + #[cube] device); only bites n≥2 (d/f/g), s/p + int2e/cc-pVDZ unaffected. (First pass fixed only 2c2e → regressed because int3c2e stayed buggy and broke the error-cancellation; fixing BOTH is the answer.) Validated: int2c2e_cart d/f byte-match upstream non-origin (new libcint-parity tests); (P\|Q) metric trace matches to 8 digits for d/f/g; DF reconstruction 1.7e-3→1.06e-4; DFUMP2 1.78e-3→2.6e-5. Residual ~e-5 is the metric-fit-inverse method (separate). No regression (313 cintx + all rs DF/MP2 suites green). | 2026-06-01 | `55bf984` | [260601-pbk-fix-df-01-bug-cintx-d-shell-cartesian-no](./quick/260601-pbk-fix-df-01-bug-cintx-d-shell-cartesian-no/) |
| 260601-nfb | **F-06 cross-spin MP2 + FOUND a silent RMP2 layout bug.** Shipped `cross_spin_ao2mo` (explicit F→C repack), un-gated both PyO3 UMP2 αβ sites, wired `dfump2_kernel` into the unrestricted scanner, genuine UHF α/β snapshot. Open-shell **conventional UMP2 now byte-matches live PySCF (Δ=2.3e-10)**. **Key find:** `default_ao2mo` returned `ao2mo::general`'s raw F-order buffer while `rmp2_kernel` reads C-order — coincide only for nvir==1, so restricted RMP2 + αα/ββ UMP2 blocks were **silently ~mHa-wrong for every polyatomic** (H₂O ~9.8 mHa, NH₃ ~10 mHa); the only in-tree test (H₂, nvir=1) hid it. Fixed + oracle-free regression (`test_c` bra-ket symmetry). **DEFERRED:** DFUMP2/DF-RMP2 byte-identity — separate pre-existing DF-subsystem accuracy gap (~1e-4..1e-3 even nvir==1, aux-independent → DF metric/B-tensor fit, not cross-spin). No libxc pulled. | 2026-06-01 | 2d2458a | [260601-nfb-cross-spin-o-a-v-a-o-b-v-b-ao2mo-wrapper](./quick/260601-nfb-cross-spin-o-a-v-a-o-b-v-b-ao2mo-wrapper/) |
| 260601-re6 | **F-06 RESOLVED — reclassified manual-only → FIXED in `AUDIT-FIX-2026-06-01.md`.** No source change: F-06's code had already landed (`d7e7fad`/`8540566`) and DF-01 was fixed (`pbk`). Ran the open-shell live-PySCF acceptance oracle that audit pass-5 claimed was *impossible in-sandbox* — it **PASSES**: OH doublet (spin=1, STO-3G, nocc_α≠nocc_β) conventional UMP2 = live PySCF 2.12.1 to **\|Δe_corr\|=2.307e-10** (tol 1e-9). `pyscf-mp2` full suite green. DFUMP2 `2.6e-5` residual = separate DF metric-fit-inverse method item (non-gating). No libxc pulled. | 2026-06-01 | `8540566` (code; doc-only reclass) | [260601-re6-fix-f-06-reclassify-mp2-cross-spin-as-re](./quick/260601-re6-fix-f-06-reclassify-mp2-cross-spin-as-re/) |
| 260601-rhc | **[VALIDATED 6/6] F-05 iprinv arm un-gated.** cintx **21-07 now ships native `ecp_iprinv`** (audit's "MISSING from every cintx branch, no scheduled workstream" was STALE — landed `dc9c0fc`/`84a5b77`, byte-identity parity vs vendor `nr_ecp_deriv`). Un-gated `int1e_ecp_iprinv`/`ECPscalar_iprinv` end-to-end: new `EcpEngine::ecp_int1e_iprinv(mol,name,rinv_origin)` (default `EcpEngineNotAvailable`) + `CintxEcpEngine` impl resolving the iprinv `OperatorId` via `Resolver::descriptor_by_symbol("int1e_ecp_iprinv_sph").id` (no const exists) + `ExecutionOptions{rinv_orig:Some(mol.atom_coord(ia))}` → per-atom `[3,nao,nao]`; `pyscf-grad::hcore_deriv_ecp` now returns the real per-atom buffer (was a hardcoded availability error); 3 stale gated tests flipped (scalar-path WR-01 `InvalidMolecule` assertion retained). Gates: `cargo +nightly test -p pyscf-gto -p pyscf-grad` 0 fail (incl. `ecp_iprinv_evaluates_real_per_atom_buffer`, Cu==ipnuc self-consistency smoke @1e-12, no-match-origin→zeros); F-05 files clean under the **real CI fmt gate** `rustfmt --edition 2024 --check` (verifier's `cargo fmt --check` fmt-gap was an edition-mismatch false positive — only PRE-EXISTING pyscf-mp2/F-06 drift fails it, out of scope, untouched). No libxc pulled. **OUT OF SCOPE (F-08 / waves 07-03..07-08):** end-to-end analytic ECP-gradient FD assembly + spinor iprinv (still fails closed). | 2026-06-01 | `e384188` | [260601-rhc-fix-f-05-un-gate-int1e-ecp-iprinv-ecp-gr](./quick/260601-rhc-fix-f-05-un-gate-int1e-ecp-iprinv-ecp-gr/) |
| 260601-sln | **F-08 iprinv rinv-origin plumbed through `intor` (NON-ecp nuclear arm).** The audit's only audit-fix-sized F-08 piece ("plumb `rinv_origin` through `intor` for the `hcore` `iprinv` arm"). Ground-truthed first: `int1e_iprinv` is a REAL cintx `AllCint1e` op (resolver.rs:321, libcint 6.1.3 grad1.c) — distinct from the cintx-MISSING `int1e_ecp_iprinv`; cintx's `validate_rinv_orig_env_params` REQUIRES `rinv_orig:Some` for any `"iprinv"` op (rejects `None` → `InvalidEnvParam{PTR_RINV_ORIG}`, accepts `Some`), so plain `intor()`'s `ExecutionOptions::default()` was the ONLY gap. Shipped `pyscf_gto::intor_with_rinv_origin(mol,name,origin)` + `intor_with_rinv_at_nucleus(mol,name,atm_id)` (threads `ExecutionOptions{rinv_orig:Some(origin)}` into a refactored single-source `evaluate_arity2`; resolves op via `Resolver::descriptor_by_symbol` — no `OperatorId` const); `pyscf-grad::hcore_deriv` now calls `intor_with_rinv_at_nucleus(mol,"int1e_iprinv",atm_id)` (= upstream `with_rinv_at_nucleus`, `pyscf/grad/rhf.py:121-143`) instead of the origin-less call; stale "iprinv MISSING from cintx" prose corrected. **In-tree physics oracle (rigorous, no live-PySCF/libxc):** translational-invariance identity `Σ_atoms (-Z)·int1e_iprinv\|rinv@atom == int1e_ipnuc` @atol 1e-10 — exactly the relation `hcore_deriv` exploits. Gates: `cargo +nightly test -p pyscf-gto -p pyscf-grad --locked` → 264 passed / 0 failed / 11 ignored (incl. `int1e_iprinv_sum_over_nuclei_equals_ipnuc`, `int1e_iprinv_with_origin_evaluates_component_leading`, `_at_nucleus_matches_explicit_origin`, non-iprinv-name reject; orchestrator re-ran grad_intor_smoke 11/11); `rustfmt --edition 2024 --check` clean on the 4 touched files; clippy clean; no libxc pulled. **OUT OF SCOPE (waves 07-03..07-08, still deferred):** `get_veff`/`int2e_ip1` 2e-response, `get_ovlp`, full `hcore_generator`/`grad_elec` force assembly; `rhf_verify_fd_numeric` stays `#[ignore]`d. | 2026-06-01 | `9e8b188` | [260601-sln-fix-f-08-plumb-rinv-origin-through-intor](./quick/260601-sln-fix-f-08-plumb-rinv-origin-through-intor/) |
| 260601-fmc | **F-12 FIXED — `format_atom` callable atom form (GTO-01.5), pure-Rust closure.** The audit's "Needs Rust-closure API design decision" resolved (user-confirmed): `AtomInput::Callable` → `Callable(Arc<dyn Fn() -> Result<AtomInput, PyscfRsError> + Send + Sync>)`; closure produces another atom spec resolved recursively through `format_atom` (one-level guard rejects nested callables); `Arc` keeps `AtomInput: Clone`, manual `Debug` impl; new `AtomInput::callable` ctor + `AtomCallable` re-export. Replaced the stale `NotYetImplemented{phase:3}` stub + test with 4 tests (callable→String congruence oracle vs direct, Å→Bohr through callable, nested-callable rejection, closure-error propagation) + doctest. Gates: `cargo +nightly test -p pyscf-gto --locked` 0 fail; `rustfmt --edition 2024 --check` clean; clippy clean on new code; no libxc pulled. **⚠ SHARED-TREE RACE:** ran concurrently with 2 other claude agents on the same working tree; the F-07 `si2` agent's `git commit` grabbed the shared index and swept the F-12 pyscf-gto files into its commit `444d868` (F-07 urdm.rs message). F-12 is committed + correct but co-mingled; history NOT rewritten (unsafe with live concurrent writers) — separable by path `crates/pyscf-gto/*`. OUT OF SCOPE: PyO3 bridge accepting a *Python* callable (Phase 3 BIND). | 2026-06-01 | `444d868` (co-mingled w/ F-07 urdm) | [260601-fmc-implement-format-atom-callable-form](./quick/260601-fmc-implement-format-atom-callable-form/) |
| 260601-si2 | **F-07 FIXED — open-shell UCCSD Λ + RDM + wave-3 hooks, live-PySCF certified.** Closed the documented Phase-7 open-shell deferral (06-CONTEXT D-03). Pivoted on the research finding that in-tree UCCSD is internally **spin-orbital** (`SpinOrbitalEris`), so ported PySCF's spin-orbital `gccsd_lambda.py`/`gccsd_rdm.py` (clean 1:1 analogs of the validated closed-shell `lambda.rs`/`rdm.rs`), NOT the spin-block soup. Shipped: surface so_t1/so_t2/so_eris on `UccsdResult`; full `ulambda.rs` (`solve_ulambda`/`update_ulambda`) + `urdm.rs` (`umake_rdm1`/`umake_rdm2` + γ1/γ2 + `pack_rdm1`/`pack_rdm2` + ao_repr); wired PyUCCSD `solve_lambda`/`make_rdm1`/`make_rdm2` (direct-in-bridge; hooks.rs RHF-shaped `wave:3` defaults legitimately stay). **The OH-doublet (nocc_α=5≠nocc_β=4) live oracle was the SUFFICIENT gate — it caught two real Λ l2 transposed-index bugs (a summed-vs-looped index in `wvvvo`; an a↔b swap in `tmp_c`) that were INVISIBLE at α==β** (the closed-shell H2 validation passed them — the F-06 lesson). Root-caused via two-venv injection (not guessing); also tightened open-shell (no-DIIS) convergence (`UCONV_TOL=1e-9`). **Final gate (live PySCF 2.12.1, OH/STO-3G): make_rdm1 dm1a \|Δ\|=4.3e-8, dm1b \|Δ\|=8.4e-9, e_corr \|Δ\|=6.2e-9, Tr=9, bonus dm2ab=4.3e-8 — all ≤1e-7 PASS.** Added PySCF-free OH Λ-norm fixture (fails-on-bug/passes-on-fix — the witness α==β checks can't be). In-tree 51 lib + integration green; verifier 7/7 must-haves; no libxc pulled. **DEFERRED (documented, not silent):** frozen-core active-only; αβ-cross-block ao_repr (CCSD-10 precedent). | 2026-06-01 | `bb82db7` | [260601-si2-implement-open-shell-uccsd-lambda-and-rd](./quick/260601-si2-implement-open-shell-uccsd-lambda-and-rd/) |
| 260601-tyg | **F-11 FIXED — `Mole::build()` IoC builder hook (reclassified won't-fix → FIXED).** User opted to fix the intentional FOUND-02 redirect via inversion of control. pyscf-core gains a process-global `MoleBuilderHook = fn(&mut Mole)` + `OnceLock` registry (`register_mole_builder`/`mole_builder_is_registered`); `Mole::build()` dispatches to the registered hook (always → the `mol.copy(); mol.basis=aux; mol.build()` rebuild pattern works), stays idempotent when `_built`, else returns an actionable `NotYetImplemented`. pyscf-gto's `build_in_place` reconstructs `MoleBuildArgs` losslessly from the Mole's fields (structured `_atom` in Bohr → `Tuples`/`unit=Bohr`; basis/ecp via `strip_name_echo` of the `{:?}` echo) and calls `build_from`; armed from `M`/`build_from`/PyO3 init. **FOUND-02 intact** — core holds only a `fn(&mut Mole)` pointer, no gto dep, zero compute. Verified: 9 pyscf-core unit (incl. genuinely-unregistered cold-start NYI arm) + 5 `mole_build_ioc.rs` (sto-3g→cc-pvdz copy-rebuild via `Mole::build()` == direct `M(cc-pvdz)`) + 21 mole/auxmol regression, all `cargo +nightly`, 0 libxc rows. Only the `Name` basis/ecp form round-trips the direct path (dominant case). **⚠ provenance:** the pyscf-core commit `fa38417` (prior interrupted run, 06-01 21:40) also swept 5 `.planning/research/*.md` deletions via a shared-index race (no `-- <pathspec>`); no content lost (survives in `research/v1.0-archive/` + `86a5bf8`), out of scope. | 2026-06-02 | `cc7de9e` (+ `fa38417`) | [260601-tyg-fix-f-11-mole-build-ioc-builder-hook-fou](./quick/260601-tyg-fix-f-11-mole-build-ioc-builder-hook-fou/) |
| 260602-b62 | **NYI sweep + `init_guess` cartesian `cart2sph` branch implemented (2 NYI removed).** Triaged all 58 `NotYetImplemented` markers (table in PLAN): the audit campaign + F-11 cleared the high-value ones; the rest are intentional (libxc-forbidden UKS, cintx-gated spinor/ECP, algebra Tensor-API stubs, MP2 bridge seams, user-error guards, l>6 c2s beyond the l≤4 ceiling, F-11 cold-start) or Phase-7 grad-assembly waves. Implemented the one genuinely-fixable user-facing gap: `init_guess_by_atom`/`by_huckel` on a `cart=true` molecule previously hard-errored (atomic RHF is spherical-only, `atom_hf.rs:114`). Added `pyscf_kernels::cart2sph_l_matrix` (per-l libcint g_trans_cart2sph) + `pyscf_gto::cart2sph_coeff(mol)` (block-diag `[nao_cart×nao_sph]`, `C[cart,sph]=c2s[sph,cart]`); both cart branches build a **spherical sibling Mole** via the **F-11 `Mole::build()` hook**, run the validated spherical guess, then project `D_cart=C·D_sph·Cᵀ` (oracle_sum). **Byte-exact vs live PySCF 2.12.1**: full `cart2sph_coeff` H2O/cc-pVDZ `[25,24]` max\|Δ\|=2.2e-16; `Tr(D_cart·S_cart)=nelec` for both guesses (d-shell molecule). Full kernels+gto+scf suites green, `cargo +nightly`, 0 libxc rows. Limitation: spherical-sibling rebuild uses the F-11 Name-basis echo (common case). | 2026-06-02 | `250670c` | [260602-b62-implement-remaining-fixable-notyetimplem](./quick/260602-b62-implement-remaining-fixable-notyetimplem/) |
| 260602-g8f | **F-08 RHF analytic gradient assembly FIXED + FD-certified (s-shells); remaining blocker re-diagnosed as an EXTERNAL cintx p-shell kernel bug.** The audit was doubly wrong: the grad integrals now EVALUATE (so the assembly runs end-to-end) but the RHF gradient was numerically GARBAGE — three real `pyscf-grad::rhf::grad_elec` bugs never FD-validated (written while integrals were "missing"). H2/STO-3G finite-difference gate **0.695 → 2.6e-9 Ha/Bohr**. Bugs: (1) **component layout** — cintx/pyscf-gto component integrals are F-order `(3,nao,nao)` with the component axis FASTEST (`out[comp + ncomp*(i+j*nao+…)]`), read component-slowest everywhere → scrambled x/y/z + broke molecular symmetry; (2) **`hcore_deriv`** sliced the ket/col `j` instead of numpy's AXIS-1 bra/row `i` (`vrinv[:,p0:p1]`) → under-counted Hellmann-Feynman; (3) **`get_veff` K-contraction** — PySCF `'jk->s1il'` outputs exchange on the integral's FOURTH axis `l` (`vk[x,i,l]=Σ_jk g(x,i,j,k,l)D[j,k]`), code output to `(i,j)` (coincided only for 1-fn-per-shell → H2 masked it). Verified term-by-term vs live PySCF 2.12.1; un-gated `rhf_verify_fd_numeric` (H2), added gated `rhf_verify_fd_numeric_pshell` (H2O). **Remaining = EXTERNAL cintx bug**: element-wise vs PySCF on bent H2O, cintx `int1e_ipnuc` (3.7e-1), `int1e_iprinv`@off-origin (2.3e-1), `int2e_ip1` (1.1) are WRONG for l≥1 while `int1e_ovlp`/`int2e`/`int1e_ipovlp`/`int1e_ipkin` are correct to ~7e-9 ("evaluate ≠ correct"). pyscf-grad+scf suites green, 0 libxc rows, fmt/clippy clean. **Out of scope:** MP2/CCSD WR-01 (bare RHF de) + p-shell gates; UHF/RKS/UKS. ⚠ done inline (numerically-delicate, F-13/F-01 precedent); committed scoped by pathspec (concurrent b62 had in-flight changes); mid-task ENOSPC truncated rhf.rs → restored from git + re-applied. | 2026-06-02 | `47298a0` | [260602-g8f-fix-f08-rhf-analytic-gradient-assembly](./quick/260602-g8f-fix-f08-rhf-analytic-gradient-assembly/) |

## Deferred Items

Items acknowledged and carried forward:

| Category | Item | Status | Deferred At |
|----------|------|--------|-------------|
| CCSD | CCSD(T) — perturbative triples | v1.x P1 | Roadmap creation |
| SCF | ROHF, SOSCF (`scf.newton`), ADIIS/EDIIS, symmetry-adapted SCF | v1.x | Roadmap creation |
| DFT | DFT-D3/D4 dispersion, custom-XC user functions | v1.x | Roadmap creation |
| Hessian | RHF/RKS Hessian, vibrational frequencies | v1.x | Roadmap creation |
| CCSD | FNO-CCSD, GHF/GMP2/GCCSD path | v1.x | Roadmap creation |
| Geomopt | Constrained geometry optimization | v1.x | Roadmap creation |
| Distribution | conda-forge channel | v1.x | Roadmap creation |

### Phase 11 deferrals (v2.0)

| item | why | lands in |
|---|---|---|
| `exxdiv = 'vcut_sph'` / `'vcut_ws'` (+ `precompute_exx`) | `NotYetImplemented { phase: 12 }` (D-PBC-20) | 12 |
| `get_coulG` for `dimension == 1` | upstream raises `NotImplementedError` too | 12 |
| `madelung` for a 2-D cell | propagates `ewald`'s Phase-12 deferral | 12 |
| `BeckeDFTGrids` (periodic Becke atomic grids) | no FFTDF consumer; it is a DFT quantity | 12 |
| `ft_ao`-based `get_pp` non-local half | this port uses the exact real-space route; matching upstream's planewave ACCUMULATION is what the last decade of the diamond gate costs | 13 |
| `project_mo_nr2nr` for periodic orbitals | `NotYetImplemented { phase: 20 }` | 20 |
| the `pyscf.pbc.scf` PyO3 surface | the whole `pyscf.pbc.*` binding layer is one phase | 20 |
| periodic `analyze` / `mulliken_meta` / `dip_moment` | they need `pbc/tools/k2gamma`, which the roadmap places with the rest of `pbc.tools` | 20 |
| `get_j_e1_kpts` / `get_k_e1_kpts` (J/K derivatives) | gradient quantities | 18 |
| a `#[cube]` Stockham FFT | the host engine is `O(n log n)` and the default mesh is PRIME, so radix-2/3/5 never applies; see 11-01/03 | — |

## Session Continuity

Last session: 2026-09-05
Stopped at: **Phase 15 CLOSED — 15-07 verification rollup completed.**

Completed the six missing parts of `15-07-PLAN.md` Task 1's nine-part oracle
matrix, ran the staggered-mesh energy oracle, measured both open D-PBC-28 rows,
and brought `15-VERIFICATION.md`, `ROADMAP.md`, `PBC-MASTER-PLAN.md §8.7` and
the `D-PBC-28` decision entry back into agreement.

**New files.** `.planning/phases/15-periodic-ao2mo-kmp2/measurements/stagger.py`
(+ `stagger.out`) and `measurements/oracle_rollup.py` (seven sections);
`crates/pyscf-pbc-mp/tests/perf_dpbc28.rs` and
`crates/pyscf-pbc-df/tests/perf_dpbc28_mofirst.rs` (both fully `#[ignore]`d).
`crates/pyscf-pbc-mp/tests/oracle_phase15.rs` grew from 3 tests to 10, every one
`#[ignore]`d **and** short-circuiting unless `PYSCF_ORACLE_VENV` is set.

**Three defects fixed, all found by the staggered oracle.** `Krhf::get_occ` and
`Kghf::get_occ` now take `nkpts` from `mo_energy.len()` (`khf.py:191-192`,
`kghf.py:109`) — a no-op during SCF, and a factor of 26.6 on `kmp2_stagger`'s
full-mesh path. `Kmp2Stagger` grew a real `flag_submesh` field and an
`integral_df()` that resolves the builder inside `kernel` the way upstream does
(`kmp2_stagger.py:73-75`, `:165-169`, `:279-282`).

Previous session: 2026-08-26
Stopped at: **Completed 09-08-PLAN.md and 09-09-PLAN.md — Phase 9 is CLOSED.**

**09-08 (Ewald, K-05 + K-06).** `pyscf-kernels/src/pbc/ewald.rs` adds TWO `#[cube(launch_unchecked)] fn ..<F: Float>` kernels launched via `dispatch_backend!` (AGENTS.md §3 / RULE 5): **K-05** `ewald_rlij` (the `(nL, natm, natm)` C-order table of `r = |R_i - R_j + L|`, `cell.py:729-732`) and **K-06** `ewald_gs_terms` (`term[g] = |ZSI[g]|^2 * exp(-absG2/4eta^2) * (4pi/absG2) * weights`, `cell.py:753-770`). Both REDUCE ON THE HOST with `oracle_sum` (D-PBC-17 / §9.3), so `ewald()` is bit-reproducible — pinned by a test. `pyscf-pbc-gto` gains `ewald.rs` (`get_ewald_params` with ALL FOUR upstream branches including the `dimension == 2` parameter algebra, `ewald` for the 3D path, the three public term functions `ewald_real_space`/`ewald_self`/`ewald_g_space` that Phase 18 will reuse, and `Cell::{get_ewald_params, ewald, energy_nuc}`) and `ewald_pme.rs` (`_bspline`/`_bspline_grad`/`bspline` in full including the Euler exponential-spline coefficients and the odd-order/even-grid Nyquist zeroing, the screened `get_ewald_direct` = the C loop `lib/pbc/cell.c:get_ewald_direct`, `pme_charge_mesh`, and `particle_mesh_ewald` which computes everything up to the FFT then defers). **Gates:** `cargo test -p pyscf-pbc-gto --test ewald` **22/0** (16 tier-1 + 6 tier-2); `-p pyscf-pbc-tools` 30/0; clippy `--all-targets -D warnings` clean on pyscf-pbc-gto + pyscf-kernels; `cargo build --workspace` clean; both xtask lints PASS. **Tier-2 vs live PySCF 2.12.1** (Bohr-specified §9.2 cells, no pseudo): `cell.ewald()` matches to **< 1e-9 Ha** on diamond (-28.771040577654524), si (-102.88216217333321), lif (-30.95510482656236) and he_fcc (-1.6174696832216189); `ew_eta`/`ew_cut` to 1e-12/1e-10 and `len(get_lattice_Ls(rcut=ew_cut))` + the internal `cutoff_to_mesh` EXACTLY, on all five. `ew_eta`-invariance spread **< 1e-13 Ha**. **SEVEN documented deviations in 09-08-SUMMARY.md**, the load-bearing ones: (1) the plan's invariance gate is unsatisfiable as written — upstream itself drifts 8.1e-7 Ha when `ew_cut` is pinned while `ew_eta` scales, so the gate re-derives `ew_cut` per `eta`; (2) the plan's reference snippet uses `pseudo='gth-pade'` but this port has no GTH parser before 10-01, so references are generated WITHOUT `pseudo=` and the pseudised targets are committed separately; (3) tier-2 cells are Bohr-specified because the 4.95e-9 CODATA gap costs 1.4e-7 Ha; (4) **new workspace dep `libm = "0.2"`** for FDLIBM `erfc` (A&S 7.1.26 is ~1.5e-7, two orders too coarse); (5) `4*pi` and the `1e200` sentinel ride in an `Array<F>` because cubecl's `F::new` takes an `f32`.

**09-09 (verification rollup).** `crates/pyscf-pbc-gto/tests/oracle_phase9.rs` — 7 tests, every one `#[ignore]`d AND short-circuiting unless `PYSCF_ORACLE_VENV` is set, so `cargo test --workspace` never touches Python and even `-- --ignored` on a machine without a venv prints a skip line and passes. Each test SPAWNS the gate interpreter on an embedded Python emitter and diffs: `vol`/`rcut`/`mesh`/`b` + `b.a^T == 2pi*I` (1e-12), `get_Gv` on [5,5,5] (1e-12 element-wise), `get_SI` + `|SI| == 1` (1e-12), `get_lattice_Ls` (count EXACT + values 1e-12), `make_kpts` in 5 variants (1e-12), `get_kconserv` (EXACT ints), `ewald()` (1e-9 Ha) — all on all five §9.2 systems — plus a dedicated test asserting the Angstrom-path deviation is EXACTLY the 4.951e-9 CODATA gap and nothing more. `PYSCF_ORACLE_VENV=1 ... -- --ignored` = **7 passed**. Also: `.planning/phases/09-pbc-foundation/09-VERIFICATION.md` (per-criterion table with command/observed/verdict, the full reference-value index with generating-snippet locators, the two standing caveats, the deferred-branch table and the Phase-10 carry-overs); `python/pyscf/pbc/__init__.py` (import-path shim only — D-PBC-14 keeps every periodic PyO3 binding in plan 20-05); ROADMAP Phase 9 checkbox ticked.

Previous session: 2026-08-25
Stopped at: Completed 09-05-PLAN.md (G-vectors, structure factors, uniform grids — PBC-GTO-04). **TWO new cubecl kernels**, both `#[cube(launch_unchecked)] fn ..<F: Float>` and launched via `dispatch_backend!` (AGENTS.md §3 / RULE 5): `pyscf-kernels/src/pbc/gv.rs` (**K-01** — `Gv[g] = rx[x]*b[0] + ry[y]*b[1] + rz[z]*b[2]`, one thread per grid point, body transcribed from `lib/pbc/cell.c:133-141` **including its accumulation order**, 1D launch inverting the C-order flat index `g = x*my*mz + y*mz + z`) and `struct_factor.rs` (**K-02** — `theta = -(Gv[g].R_a)`, planar `si_re = cos`/`si_im = sin`, one thread per `(a,g)`). `pyscf-pbc-gto` gains `gv.rs`: `fftfreq_scaled`/`fftfreq`, `get_gv`, `get_gv_weights` -> `GvWeights{gv,gvbase,weights,mesh}`, `get_si` (BOTH upstream branches — the separable Gvbase product used by default, and the direct K-02 device form), `get_uniform_grids`, plus the four `Cell` methods. `pyscf-pbc-gto` now depends on `pyscf-kernels` (RULE 6; it still never names `cubecl-*`, check-dependency-wall PASS). Gates: `cargo test -p pyscf-pbc-gto --test gv` **19/0**; `-p pyscf-kernels --test pbc_gv` **6/0**; `-p pyscf-pbc-gto --all-features` 71/0; `-p pyscf-kernels` 21/0; `-p pyscf-pbc-tools` 12/0; clippy `--all-targets --all-features -D warnings` clean on all three (`--all-features` on pyscf-kernels enables cpu+cuda+wgpu+rocm, so **all four dispatch_backend! arms of both kernels compile**); `cargo build --workspace` clean; check-dependency-wall + check-forbidden-paths PASS; rustfmt clean. **Tier-2 vs live PySCF 2.12.1:** `fftfreq_scaled`/`fftfreq` tables n=1..8 **EXACT**; the full **125x3 diamond `Gv`** at mesh [5,5,5] to **< 1e-15** (the plan's gate was 1e-12); `weights = 0.013062524449620905` (= 1/vol) to 1e-14; `SI` rows to 1e-14; both `get_uniform_grids` variants (rows + `abs().sum()` digest) to 1e-13. **Tier-1 highlights:** `Gv[g].a[i] == 2*pi*r_i` on all five §9.2 systems (pins row order + axis order + the 2*pi at once); `fftfreq_scaled` characterised twice for **n=1..=32**; `|SI|==1` and `SI[a,0]==1+0i` on both branches; `SI[a,-g]==conj(SI[a,g])`; and at the KERNEL level both kernels match a naive host reference **BIT-FOR-BIT** (`assert_eq!` on `Vec<f64>`) over 7 mesh shapes including the transposed pair [6,6,7]/[6,7,6] and the 256/729 tail cases. **SIX documented deviations in 09-05-SUMMARY.md**, notably: (1) `get_SI`'s SEPARABLE branch was ported too (the plan named only the direct K-02 form) — it is upstream's default and is `natm*(mx+my+mz)` transcendentals vs `natm*ngrids` (282 vs 207646 per atom on diamond's real [47,47,47] mesh); the two branches agree to 1.3e-15, matching upstream's own 1.8e-15 spread; (2) the plan's 1e-12 `Gv` tolerance is UNREACHABLE against the §9.2 Angstrom diamond (the 4.95e-9 `Unit::Ang` gap gives ~9e-9 absolute on |G|~1.86), so the test adds `diamond_bohr()` — the same cell with the lattice given directly in Bohr from upstream's own `lattice_vectors()` literals — which matches to **~1 ULP**, and checks the Angstrom cell separately at relative 1e-7 with the residual pinned < 2e-8. The `inf_vacuum` / non-uniform Gauss-Chebyshev `Gv` base returns `NotYetImplemented{phase:12}` (D-PBC-20) — Phase 12 must widen `GvWeights::weights` from `f64` to a per-grid `Vec<f64>` when it lands. Also fixed 1 PRE-EXISTING clippy blocker in `pyscf-runtime/src/probe/wgpu.rs` (`doc_lazy_continuation`, doc-formatting no-op) that `--all-features` newly surfaced.
Previous session: 2026-08-25 — Completed 09-04-PLAN.md (cutoffs, `rcut`, mesh — PBC-GTO-03). `pyscf-pbc-gto` gains `cutoff.rs`: every estimator of `cell.py:373-523` and `:968-1025` ported line-by-line — `_estimate_rcut`/`bas_rcut`/`estimate_rcut`, `_estimate_ke_cutoff`/`estimate_ke_cutoff`/`error_for_ke_cutoff`, `_extract_pgto_params` (min/max), `get_bounding_sphere`/`get_nimgs`, `pgf_rcut`, `rcut_by_shells`, `_mesh_inf_vaccum`, `estimate_mesh` — plus the Rust spelling of `bas_exp`/`_libcint_ctr_coeff(...).max(axis=1)`/`mol.omega` over `_bas`/`_env`. `pyscf-pbc-tools` gains `mesh.rs` (`tools/pbc.py:787-836`: `cutoff_to_mesh`/`mesh_to_cutoff`/`cutoff_to_gs`/`gs_to_cutoff` + a 3x3 Householder QR for `np.linalg.qr(...)[1][2,2]`) and `mat3.rs`. **`Cell::build` now fills `rcut` and `mesh` — plan 09-03's `RCUT_UNSET`/`MESH_UNSET` sentinels no longer survive a build**, and the 09-03 carry-over `dimension<=2` vacuum-size warning (`cell.py:1751-1758`) is closed. New: `Cell::cutoff_to_mesh`/`nimgs`/`rcut_by_shells`/`bas_rcut` methods and the `use_loose_rcut` field (Cell/CellBuildArgs/CellPack). Gates: `cargo test -p pyscf-pbc-gto --all-features` 52/0 (29 new in `tests/cutoff.rs`); `cargo test -p pyscf-pbc-tools` 12/0; clippy `--all-targets --all-features -D warnings` clean on both; `cargo build --workspace` clean; `cargo doc` no broken links; check-dependency-wall + check-forbidden-paths PASS; rustfmt clean. **Tier-2 vs live PySCF 2.12.1** (precision 1e-8, all five §9.2 systems): `estimate_rcut`, `estimate_ke_cutoff`, `rcut_by_shells`, `bas_rcut`, `error_for_ke_cutoff(100)` all match to a relative **1e-12** (basis-only quantities are unit-independent, so the `Unit::Ang` gap cannot reach them); `cell.mesh`, `cutoff_to_mesh(a,{50,100,200})`, `cutoff_to_gs` and `nimgs` match **EXACTLY** (diamond rcut = 21.319400521777592, mesh [47,47,47]; graphene [45,45,351] with nimgs[2]=0). **EIGHT documented deviations in 09-04-SUMMARY.md**, notably: (1) the plan's guessed acceptance numbers (`rcut ~ 15.6`, `mesh == [15,15,15]`) are WRONG and were regenerated as the plan itself instructs — the truth is 21.3194 / [23,23,23]; (2) `pgf_rcut` is ported TWICE because upstream's `cell.rcut_by_shells` dispatches to `libpbc`, whose C twin (`lib/pbc/cell.c:30-59`) adds a `gmax < precision` early return the Python version lacks — porting only the Python one could not have reproduced upstream's shell radii; (3) `det3`/`inv3`/`transpose3` MOVED from `pyscf-pbc-gto::cell` down to `pyscf_pbc_tools::mat3` (re-exported, so 09-03's API and tests are untouched) because `cutoff_to_mesh` needs `2*pi*inv(a.T)` and the dependency edge runs gto -> tools — one lattice inversion in the workspace, no drift; (4) `use_loose_rcut` had to be added to `Cell` or `estimate_rcut`'s `cell.py:430-431` branch (and therefore `rcut_by_shells`) would be dead code. No `NotYetImplemented` remains in this plan's surface — every branch, including `inf_vacuum` and `use_loose_rcut`, is implemented.
Previous session: 2026-08-25 — Completed 09-03-PLAN.md (the `Cell` type). `pyscf-pbc-gto` gains `cell.rs` (Cell OWNING a `Mole` + `Deref`/`DerefMut` — D-PBC-01, so `cell.nao_nr`/`cell.natm`/`cell._env` work unchanged and there is ONE Mole build path), the lattice API ported line-by-line from `cell.py:1811-1975` (`lattice_vectors`/`vol`/`reciprocal_vectors`/`get_abs_kpts`/`get_scaled_kpts`/`get_scaled_atom_coords`/`tot_electrons`) with closed-form 3x3 `det3`/`inv3`/`transpose3` (no faer for a 3x3), `Cell::build` (`cell.py:1593-1810`, incl. the `fractional` transform and the `exp_to_discard` diffuse filter), `types.rs` (`CellBuildArgs`/`ALattice`/`LowDimFtType`), `dumps_loads.rs` (`pack`/`unpack`/`dumps`/`loads`, `cell.py:65-155`), a `PseudoData` placeholder, and the five §9.2 reference systems (diamond/si/lif/he_fcc/graphene) in `src/test_systems.rs` behind a `test-systems` feature with committed live-PySCF-2.12.1 reference values (D-PBC-19). Gates: `cargo test -p pyscf-pbc-gto --all-features` 23/0; clippy `--all-targets --all-features -D warnings` clean; `cargo build --workspace` clean; check-dependency-wall + check-forbidden-paths PASS; rustfmt clean. **TWO REAL BUGS FOUND BY THE TESTS:** (1) serde_json's DEFAULT f64 parser is 1-ULP lossy — `loads(dumps(cell))` silently perturbed the LiF lattice; fixed by enabling the `float_roundtrip` feature (pyscf-gto should declare it too — carry-over); (2) `pyscf_core::Unit::Ang.length_in_au() = 1.8897261339213` disagrees with upstream PySCF's effective factor `1/0.52917721092 = 1.8897261245650618` by **4.951e-9 relative** (the pyscf-core doc comment claiming it matches upstream verbatim is FALSE), making every volume 1.485e-8 large; the plan's absolute 1e-6 vol check is unreachable for any cell larger than He/fcc, so tier-2 checks use a relative bound plus a dedicated test pinning the ENTIRE deviation to that one constant. Correcting the constant is workspace-wide (moves every molecular geometry + v1.0 baseline) — deferred, carry-over. FIVE documented deviations in 09-03-SUMMARY.md, notably: `reciprocal_vectors` does NOT zero non-periodic rows (the plan's claim about upstream is wrong — upstream only ASSERTS orthogonality), and `rcut`/`mesh` keep sentinels with `try_rcut()`/`try_mesh()` returning `NotYetImplemented{phase:9}` until plan 09-04 fills the two estimator bodies. Also fixed 3 PRE-EXISTING clippy blockers in pyscf-algebra (axpy/scal `unnecessary_cast`, gemm `manual_div_ceil`) that failed the `-D warnings` gate; full pyscf-algebra suite re-run green.
Previous session: 2026-08-25 — Completed 09-02-PLAN.md (the complex-algebra contract — PBC-MASTER-PLAN §5). `pyscf-algebra` gains `complex.rs` (planar `CTensor`, D-PBC-02/RULE 8), `zgemm.rs` (FOUR real `gemm_dense` calls in the mandated D-PBC-03 order — no Karatsuba, no fusion), `zblas.rs` (zaxpy/zscal/zdotc/zdotu/zreduce_sum/ztranspose/zhadamard), `zeigh.rs` (zeigh_gen/zcholesky/zsolve_linear, each with the faer-`c64` primary route AND the mandated real-arithmetic cross-check route) and `zoracle.rs` (oracle_zsum/oracle_zdot, D-PBC-17). `pyscf-kernels` gains `src/pbc/zhadamard.rs` (K-04 `#[cube(launch_unchecked)] fn zhadamard_kernel<F: Float>`). **D-PBC-04 RESOLVED: `FAER_C64 = true`** — faer 0.24 has working native `c64` SelfAdjointEigen/Llt/PartialPivLu (probe transcript recorded verbatim in 09-02-SUMMARY.md; every later PBC plan may assume the `c64` route exists). Gates: `cargo test -p pyscf-algebra -p pyscf-kernels` 0 failed; clippy clean on all new files; `check-dependency-wall` PASS (no method crate gained a cubecl dep, ALG-06 intact); `check-no-fma` PASS on the release-oracle build; `check-forbidden-paths` PASS; `rustfmt --edition 2024 --check` clean. THREE documented deviations (all in 09-02-SUMMARY.md §DEVIATIONS): (1) `zhadamard_dense` carries an in-crate byte-identical mirror of the K-04 kernel because `pyscf-kernels` depends on `pyscf-algebra` and the call cannot go upward — a bit-for-bit lockstep test guards the pair; (2) `zcholesky`'s second route is an explicit complex Crout factorization, not the 2n x 2n embedding, which is mathematically unusable for Cholesky; (3) `zeigh_gen_embedding` adds a degeneracy fallback beyond the fixed `0,2,4,…` stride (non-degenerate inputs still get exactly those columns).
Previous session: 2026-05-26T04:57:02Z
Stopped at: Completed 07-09-PLAN.md (PyO3 bridge — grad.rs: six PyGradients classes (eager SCF snapshot D-09) + Gradients() factory + PyGradScanner returning the (e_tot, de) TUPLE; geomopt.rs: optimize + geometric_solver/berny_solver over the ONE native engine D-06/T-07-20; python/pyscf/{grad,geomopt} overlays — mf.nuc_grad_method() graft over scf+dft pyclasses, NO geometric/pyberny import GEOMOPT-01; method crates stay pyo3-free; numeric stays cintx-gated, structural bridge always-on; completes the GEOMOPT-02/03 Python optimize(mf) entry point 07-06 left Partial)
Resume file: None
| Phase 09 P09-06 | ~1h | 3 tasks | 10 files |
| Phase 09 P09-07 | ~1h | 3 tasks | 8 files |
| Phase 09 P09-08 | ~1h | 3 tasks | 8 files |
| Phase 09 P09-09 | ~40min | 3 tasks | 5 files |
