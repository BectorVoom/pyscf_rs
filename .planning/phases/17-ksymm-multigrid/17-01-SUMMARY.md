# 17-01 SUMMARY — measure the floor before writing the gate

**Status:** SHIPPED (measurement only — **no Rust source was created or
edited**). **Date:** 2026-09-01.
**Interpreter:** vendored PySCF **2.12.1** at `<repo root>/pyscf`, run with
`PYTHONPATH=<repo root>` so `import pyscf` resolves there and not to
site-packages 2.14 or the port's own `python/pyscf` (Phase 11's oracle
mis-import defect — every script here asserts
`pyscf.__version__ == "2.12.1"` before doing anything else, following the
precedent in `.planning/phases/14-gdf-mdf-rsdf-rsjk/measurements/README.md`
and `15-01-PLAN.md`).

`git diff --name-only` / `git status` after this plan touch only
`.planning/`: `.planning/phases/17-ksymm-multigrid/{measurements/,17-CONTEXT.md,17-01-SUMMARY.md}`,
`.planning/ROADMAP.md`, `.planning/pbc/PBC-MASTER-PLAN.md`, `.planning/STATE.md`.

## What was measured

All seven tasks in `17-01-PLAN.md` were run; full detail, every generating
snippet and every raw `.out` file are in
`.planning/phases/17-ksymm-multigrid/measurements/README.md`. Headline
results:

### Task 1 — Gate A: the six IBZ integers

**Reproduced EXACTLY** on upstream's own Si cell (`lib/test/test_kpts_ksymm.py:30-89`,
`[16,16,16]`): `145 / 145 / 245 / 408 / 816 / 2052`. This was the
stop-the-phase check and it passed without qualification.

Repeated on five more cells at `[16,16,16]` to settle 17-CONTEXT §3.10's
question (do the integers travel with lattice TYPE or lattice constant?):
**they travel with the space-group type.** `si` (`a=5.4306 Å`) and `diamond`
(`a=3.5668 Å`) — both Fd-3m diamond structure, different constant from
upstream's Si — reproduce upstream's exact six-integer set bit-for-bit.
`lif`/`he_fcc` (Fm-3m rocksalt/fcc, symmorphic) collapse to a different,
mutually-identical set (`{145,145,145,408,408,2052}` — `symmorphic=True`
changes nothing because a symmorphic group has no non-symmorphic operations
to lose). `graphene` (2D) gives a third set. Only the `time_reversal_symmetry=True`-only
configuration (`F=2052`) is invariant across every system, matching
17-CONTEXT's `2052 = 4096/2 + 4` Γ/TRIM-count formula, which references only
the `[16,16,16]` mesh, never the lattice.

**Decision:** gate on the integers using the port's own `si`/`diamond`
fixtures (§9.2) — they already reproduce upstream's exact set, so no sixth
fixture is required for the integers. `finger(kpts_ibz)` does not travel (it
scales with `1/a`) and is not gated; upstream's exact Si cell is recorded as
an optional finger-only cross-check in `measurements/gate_a.out`.

**A real fixture bug was caught and fixed while writing this task**: an
earlier draft placed `si`/`diamond`'s second basis atom at `a/2·(1,1,1)`
(rocksalt's octahedral position) instead of `a/4·(1,1,1)` (diamond's
tetrahedral position), silently building a rocksalt-symmetry cell and
producing the WRONG integer set (`245`/`816` came out as `145`/`408`). Caught
because the wrong cell's integers did not match upstream's Si. Documented in
`measurements/README.md` so nobody re-derives it.

### Task 2 — Gate B: the transform floor against one converged SCF

`si [3,3,3]`, `KRKS(LDA,VWN)`, `conv_tol=1e-11`. **Correction to 17-CONTEXT's
"expects ≥1e-12" guess**: the floor is set by `cell.precision` (AO-integral
screening), not by SCF `conv_tol`. At PySCF's default `cell.precision=1e-8`:
`transform_dm`/`make_rdm1(transform_mo_coeff(...))` = **4.481e-10**,
`transform_mo_energy` = **5.331e-11**, `transform_1e_operator` = **2.931e-11**,
`symmetrize_density` = **6.607e-13** — none of them ≥1e-12. Tightening
`conv_tol` to 1e-13 changed nothing (isolated with a dedicated re-run,
`gate_b_tight.py`); tightening `cell.precision` to 1e-13 collapsed every
residual to machine epsilon (≤8e-15, `gate_b_prec.py`). The `mo_coeff`
elementwise-comparison trap (17-CONTEXT §3.1) measured exactly as large as
predicted (~2.3, O(1)) at both precisions, confirming the density-matrix
comparison is the only valid one.

### Task 3 — Gates C/D: the two-SCF energy floor, mesh pinned

**Resource-scoped** (see Deviations below) to: `si`'s full `KRHF`/`KRKS` ×
gamma/Monkhorst × FFTDF/GDF grid (8 pairs, default mesh `[36,36,36]`);
`diamond`'s reduced 4-pair set (default mesh `[48,48,48]`); one pair each for
`lif`/`he_fcc`/`graphene` at a capped `25³` mesh. Every pair pins `cell.mesh`
identically on both sides.

* `si`: FFTDF `2.807e-13`…`6.928e-14`e Ha; GDF `9.362e-12`…`5.544e-10` Ha.
* `diamond`: FFTDF `5.985e-11`/`1.279e-13`/`5.400e-11`; GDF `3.433e-09`.
* Every FFTDF row beats upstream's own 5e-8(gamma)/5e-7(Monkhorst) by three
  to six orders of magnitude; every GDF row is comfortably inside it.
* **Worst measured across everything run: FFTDF ≤5.985e-11, GDF ≤3.433e-09**
  — these are the numbers now in `ROADMAP.md`/`PBC-MASTER-PLAN.md`.
* `he_fcc` at the capped mesh reproduces the same ~1e-10 scale as `si`/
  `diamond`. `lif` (1.461e-04) and `graphene` (6.391e-01, symmetric run did
  not even converge) do NOT — both are **mesh-cap artefacts**, not symmetry
  defects: `lif`'s ionic electrostatics and `graphene`'s 20 Å vacuum both
  need a mesh far finer than the 25³ cap. Carried over to 17-13 (below).
* Mesh-unpinning demonstration (17-CONTEXT §3.3): on `si`/`diamond` at
  `[2,2,2]`, `symmetrize_mesh` does NOT enlarge the default mesh — pinned and
  natural-unpinned meshes are identical, so `|dE|` from mesh alone is
  `1.776e-15`/`0` respectively. This is real evidence the enlargement is
  system/k-mesh dependent, not evidence that pinning is unnecessary.
* Run-to-run/thread-count spread (`si`, `KRHF`, FFTDF, full BZ, 4 runs at
  threads ∈ {1,8}): spread is **2e-15** — 2 ulps, matching
  `.planning/pbc/SUMMARY.md`'s prior finding about upstream's own
  multi-threaded-BLAS noise. Every Gate C/D number above sits far above this
  floor.

### Task 4 — the post-SCF (KMP2) floor

He (`mp/test/test_ksym.py`'s own fixture): `|d e_corr|=3.096e-16`, `rdm1`
residual `1.332e-15` — both far tighter than upstream's own 5e-11 gate, and
post-SCF IS tighter than SCF here (`|d E_scf|=4.441e-16`).

`si [2,2,2]`: `|d e_corr|=1.067e-09`, `rdm1` residual `5.028e-09`, `|d E_scf|=5.544e-10`.
**Qualification to 17-CONTEXT §2.1's ordering claim**: post-SCF is here ~2x
LOOSER than the SCF, not tighter. The "post-SCF tighter than SCF" pattern is
a property of upstream's tiny, near-machine-precision He fixture, not a
general theorem — 17-09 must gate `e_corr` on its own measured number per
system.

`KRCCSD`: recorded as **unmeasured, Phase 16 not shipped**
(`crates/pyscf-pbc-cc/src` is a 13-line stub) — no extrapolation attempted,
per the plan's explicit instruction.

### Task 5 — Gate E: the multigrid floor

`diamond`/`si`, two meshes each. **v1 (`MultiGridNumInt`) is essentially
algebraically exact** against FFTDF/`numint` (1e-12…1e-14 on `get_pp`/
`get_nuc`/`get_j`/`vxc`/`exc`), beating upstream's own 7-8 decimal tolerance
by orders of magnitude — except when the mesh is genuinely too coarse for the
system (si `get_pp` at a 0.6×-coarser mesh: 4.384e-09, a mesh-CONVERGENCE
artefact, not definitional). **v2 (`MultiGridNumInt2`) carries a
MESH-INDEPENDENT floor** against FFTDF (~2.41e-08 diamond, ~1.47e-07 si,
unchanged at both meshes tested) — a DEFINITIONAL gap, the direct Phase-17
analogue of Phase 14's GDF-vs-RSDF 4.5e-6 finding. Converged `KRKS(LDA,VWN)`
`e_tot`: v1 beats upstream's 1e-7 gate by 4-5 orders (4.7e-12 diamond,
3.8e-12 si); v2 is markedly looser (3.986e-07 diamond, **1.938e-06 si** —
worse than 1e-7).

### Task 6 — the speed floor (D-PBC-26, 17-CONTEXT §8)

1. **`get_jk` full-BZ vs IBZ-subset** on `si [4,4,4]` (`nkpts=64`,
   `nkpts_ibz=8`): FFTDF **223x**, GDF **40x** wall-clock ratio — both far
   ABOVE the naive `nkpts/nkpts_ibz=8x` bound, because exact-exchange cost
   scales closer to quadratically than linearly in `nkpts`. Target the GDF
   number (~40x) as 17-07's realistic floor.
2. **Multigrid vs reference `numint`** (measured as part of Task 5's
   converged-`KRKS` timing): both v1 AND v2 measured **SLOWER** than the
   reference route on `diamond`/`si` — wall-clock ratio 0.18x-0.49x. This is
   the opposite of "multigrid is faster" and 17-11/17-12 must not assume a
   speed win exists at this repo's reference-system scale.

### Task 7 — restate the gates in three documents

Done together, in this plan: `17-CONTEXT.md §2.2`/§8 rewritten with the
measured numbers; `ROADMAP.md`'s Phase-17 line updated with a "17-01
MEASURED" addendum (correcting Gate B's guess, adding the two new speed
numbers); `PBC-MASTER-PLAN.md:446` (`§7`) replaced ("1e-9" → the five measured
gates) and `§8.9` rewritten from the stale 8-plan table to the correct
13-plan table (17-01 through 17-13), with D-PBC-26's measured `get_jk` bound
and the `spglib`/droppable-multigrid notes folded in.

## Deviations from the plan

1. **`speed_get_jk.py`'s mesh was coarsened from `si`'s default `[36,36,36]`
   to `[9,9,9]`.** FFTDF `get_jk(with_k=True)` at 64 k-points did not finish
   inside this measurement's time budget at the default mesh (an O(nkpts²)
   sweep over 4096 k-pairs). This is a pure wall-clock RATIO measurement
   (same DF object, same mesh on both sides of each comparison), so the ratio
   is not expected to depend materially on mesh resolution.
2. **Gate C/D's grid was resource-scoped**, not run at the plan's literal
   "5 systems × KRHF/KRKS × gamma/Monkhorst × FFTDF/GDF × sym-on/off, each at
   its own default mesh": `diamond`'s default mesh (`[48,48,48]`) made a
   single pair take ~120s and `lif`'s (`[81,81,81]`) did not finish a pair
   inside several minutes. Actually run: `si` full grid (8 pairs) at default
   mesh; `diamond` reduced 4-pair set at default mesh; `lif`/`he_fcc`/
   `graphene` one pair each at an explicit 25³ cap. Every pair still pins
   `cell.mesh` identically on both sides — the deviation is in WHICH mesh was
   chosen, never in whether both runs of a comparison share it.
3. **`lif`/`graphene`'s Gate C/D numbers at capped mesh are recorded as
   mesh-cap artefacts, not as gate results** — `graphene`'s symmetric run did
   not even converge at the cap. Both need a longer-running acceptance pass
   at production mesh; see carry-overs.
4. **The mesh-unpinning demonstration (17-CONTEXT §3.3) only completed for
   `si`/`diamond`**; `lif`/`he_fcc`/`graphene`'s demonstration at TRUE default
   mesh was cut short by the same `[81,81,81]`/`[45,45,351]` cost that drove
   deviation 2, and is carried over.
5. **`diamond`'s deep grid only ran 4 of the full 8 cells** (Monkhorst×KRKS
   and gamma×KRKS×GDF were not run) — the 4 cells measured already show the
   same FFTDF/GDF pattern as `si`'s full grid, so this is a confirmation gap,
   not an open question.

None of these deviations touch the plan's non-negotiable must_haves: every
number reported IS produced by the vendored 2.12.1 tree with its generating
snippet committed alongside it; every two-run energy comparison that was
actually run pins `cell.mesh` on both sides; the gates were written into the
three documents strictly AFTER the numbers were measured; and both required
speed numbers (Task 6.1 and 6.2) are recorded with their generating scripts.

## Carry-overs for later Phase-17 plans

* **17-13** (verification): re-run `lif`/`graphene` Gate C/D at production
  mesh (not the 25³ cap) and complete the mesh-unpinning demonstration for
  all five systems at true default mesh, as a longer-running acceptance pass
  — the same `#[ignore]`d-acceptance-run pattern Phase 14 used for diamond's
  `make_j3c` wall time. Also complete `diamond`'s remaining 4 deep-grid cells.
* **17-07** (`get_jk` fast path, D-PBC-26): target the measured GDF ratio
  (~40x) as the realistic speedup floor for the IBZ-restricted route; FFTDF
  has room for a larger win (~223x measured, though the FFTDF number came
  from a much coarser mesh than production and should be re-checked once the
  port's own FFTDF exists).
* **17-11/17-12** (multigrid): do NOT assume a wall-clock win over the
  reference `numint` route without re-measuring at whatever system scale is
  actually targeted — upstream's own multigrid measured SLOWER
  (0.18x-0.49x) at this repo's reference-system scale. Gate E's accuracy
  numbers (v1 exact to 1e-12…1e-14, v2's mesh-independent ~2e-8…2e-7 floor)
  stand independently of the speed finding.
* **17-09** (KMP2 ksymm): gate `e_corr` on its own measured number per
  system (`si`: ≤~2e-9) rather than assuming "post-SCF tighter than SCF"
  transfers from upstream's He fixture — measured, it does not on `si`.
* **17-03/17-05** (transforms): Gate B's floor is `cell.precision`-limited,
  not `conv_tol`-limited — any port-vs-port equivalence test for the
  transform algebra should tighten `cell.precision` (not `conv_tol`) to
  isolate the algebra from integral-screening noise, matching how
  `gate_b_prec.py` isolated it here.

## Files

* `.planning/phases/17-ksymm-multigrid/measurements/README.md` — the full
  writeup, every table, every generating snippet inline.
* `.planning/phases/17-ksymm-multigrid/measurements/*.py` / `*.out` — 13
  scripts (`gate_a`, `gate_b`/`gate_b_tight`/`gate_b_prec`, `gate_c_d` +
  `_part2`/`_part3`/`_repro`, `gate_mp2`, `gate_multigrid`, `speed_get_jk`)
  and their raw output.
* `.planning/phases/17-ksymm-multigrid/17-CONTEXT.md` — §2.2 and §8 rewritten
  with measured numbers.
* `.planning/ROADMAP.md` — Phase 17 line, "17-01 MEASURED" addendum.
* `.planning/pbc/PBC-MASTER-PLAN.md` — `§7` line 446, `§8.9` (8→13 plans).
* `.planning/STATE.md` — Current Position updated.

No file under `crates/` was created, edited, or touched.
