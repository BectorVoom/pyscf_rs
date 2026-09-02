# Phase 17 — k-point symmetry + multigrid — CONTEXT

**Written:** 2026-08-31, before any Phase-17 code.
**Read this before `17-01-PLAN.md`.** Every claim below was verified against the
vendored `pyscf/` tree (2.12.1) and the current Rust workspace on 2026-08-31 and
carries the file and line that proves it.

`PBC-MASTER-PLAN.md §8.9` sizes this phase at eight plans. **Three of its eight
rows are incomplete, one whole workstream is missing from the table entirely,
and both of the phase's recorded gate statements are tighter than upstream's own
test suite** — one of them by seven orders of magnitude. This document records
what is actually there, what is actually missing, the architectural decision that
must be made before the first crate is written, and the gates that cannot be
believed as written.

This is the third consecutive phase where the pre-implementation read found the
gate wrong (`14-CONTEXT.md §2`, `15-CONTEXT.md §2`). The discipline is working;
keep it.

---

## 1. Scope corrections, in order of consequence

### 1.1 An entire workstream is missing from §8.9: the RS/BvK supermole

`§8.9`'s eight rows are `symm` × 2, `kpts`, three `*_ksymm` rows, `multigrid`,
and verification. **None of them is `ft_ao._RangeSeparatedCell` / `ExtendedMole`
— and seven live Rust sites already promise that work to Phase 17 by number:**

| Rust site | What it refuses | Measured cost of the deferral |
|---|---|---|
| `crates/pyscf-pbc-df/src/gdf_builder/mod.rs:96-102` | `exclude_dd_block = true` | **1.835e-08 Ha** diamond 2×2×2, **2.900e-08** gamma (D-PBC-23) |
| `crates/pyscf-pbc-df/src/rsdf_builder/mod.rs:190-196` | `exclude_dd_block = true` | same machinery |
| `crates/pyscf-pbc-df/src/gdf_builder/eta.rs:196` | the `estimate_rcut` `true` half | — |
| `crates/pyscf-pbc-df/src/gdf/jk.rs:36-38` | the MO-factorised `get_k_kpts` | performance only |
| `crates/pyscf-pbc-df/src/gdf/jk.rs:243-253` | `GDF.get_jk` at band k-points | `_cderi` rebuild |
| `crates/pyscf-pbc-df/src/mdf/mdf_jk.rs:80-90` | `MDF.get_jk` at band k-points | `_cderi` rebuild |
| `crates/pyscf-pbc-scf/src/rsjk.rs:31-58` | all of `rsjk` | blocked, "sequence it after Phase 17's supermole" |

D-PBC-21 and D-PBC-23 both name Phase 17 in their text, and the `ROADMAP` Phase-14
line calls it "**ONE piece of work not four**" — `_RangeSeparatedCell` +
`ExtendedMole` closes `exclude_dd_block` (1.835e-08 Ha), `strip_basis`
(1.054e-09 in `j3c`, 2.750e-09 in the ERI) and Phase 13's `ft_aopair` residual
(5.121e-10) together.

**It belongs in this phase and it is plan 17-10.** Two honest caveats, recorded
so nobody has to re-derive them:

* It has **nothing to do with symmetry or multigrid**. It is a DF-accuracy
  carry-over that landed in "Phase 17" because Phase 13 needed a number to write
  in an error message. It shares no code, no test fixture and no gate with the
  rest of the phase, so it runs as an **independent track** and can be scheduled
  in parallel from day one.
* Moving it elsewhere is not free: seven doc comments, their `NotYetImplemented
  { phase: 17 }` payloads and the tests that assert on them would all have to
  change. Keeping the promise is cheaper than renegotiating it.

`rsjk` itself is **not** in this phase. `rsjk.rs:41-52` is explicit that its
second blocker is a screened periodic 4-centre `int2e` driver
(`PBCVHF_direct_drv1`) of which this port has no implementation at all, and that
"the screening **is** the algorithm" — there is no correct-but-slow fallback.
17-10 removes blocker 1 only, and says so.

### 1.2 §8.9 omits `symm/basis.py`, and the default SCF path needs it

`§8.9` plan 17-01 lists `geom.py`, `tables.py`, `group.py`; 17-02 lists
`space_group.py` and `symmetry.py`. **`symm/basis.py` (161 l) appears nowhere**,
and it is not optional:

* `khf_ksymm.eig` (`khf_ksymm.py:104-119`) reads `cell.symm_orb` and
  `cell.irrep_id` directly.
* Those are produced by `Cell._build_symmetry` → `symm_adapted_basis`
  (`cell.py:1515-1527`, `basis.py:109`).
* `ksymm_scf_common_init` (`khf_ksymm.py:142`) defaults `use_ao_symmetry = True`,
  so this is the **default** branch, not an opt-in one.
* Upstream gates it: `test_khf_ksym.py:94-100` (`test_krhf_symorb`) and
  `test_krks_ksym.py:131-137` (`test_krks_symorb`).

Without `basis.py` the phase can only ship `use_ao_symmetry = False`. That is a
legitimate first milestone, but it must be a *stated* one, not an accident.
It is plan 17-04.

### 1.3 `ktensor.py` was moved INTO this phase by Phase 15 and §8.9 never recorded it

`15-CONTEXT.md §1.1` ruled that `pyscf/pbc/lib/ktensor.py` (386 l, the
`KsymmArray` container) is Phase-17 work, because its only consumers are
`kmp2_ksymm.py`, `khf_ksymm.py` and `kccsd_rhf_ksymm.py` — building it in Phase 15
would have produced "a container with no caller and no way to test it". `§8.9`'s
table still does not list it. It is plan 17-06.

### 1.4 §8.9's 17-07 is two independent ports with two different gates

`multigrid/__init__.py` exports **two** classes:

```
from .multigrid      import MultiGridNumInt     # v1, multigrid.py, 1962 l
from .multigrid_pair import MultiGridNumInt as MultiGridNumInt2   # v2, 1257 l
```

They are separate implementations with separate C backends, and their kernel
surfaces are not comparable:

| | v1 `multigrid.py` | v2 `multigrid_pair.py` |
|---|---|---|
| C entry points | **2** — `libdft.NUMINT_fill` (`:146`), `libdft.NUMINT_rho_drv` (`:280`) | **12**, via `_backend_c.py`: `grid_collocate_drv`, `grid_integrate_drv`, `build_task_list`, `build_core_density`, `init_gridlevel_info`, `init_rs_grid`, `int_gauss_charge_v_rs`, `gradient_gs`, `get_gga_vrho_gs`, + 3 destructors |
| Extra modules | — | `pp.py` (256 l), `utils.py` (70 l) |
| Downstream consumer | none | **`pyscf/pbc/grad/rhf.py:28,44` and `grad/uhf.py:28,40` `assert isinstance(ni, MultiGridNumInt2)`** — Phase 18 |

Lumping them into one plan hides that v2 is roughly the size of the rest of the
phase, and that v2 — not v1 — is the one Phase 18 references. They are plans
17-11 and 17-12.

**Judgment call, stated out loud:** multigrid is a *speed* feature — an
alternative `numint` — and upstream's own tests gate it at 1e-7…1e-8 against the
reference `numint` (§2.3). Nothing in Phases 18–20 requires it for
*correctness*; `grad/rhf.py` only branches on it. **If this phase overruns,
multigrid is the droppable half, and 17-11/17-12 are ordered last so that
dropping them costs nothing already built.** Do not drop 17-10 instead: seven
error messages point at it.

Being a speed feature is not the same as being *measured* as one — §8's
corollary rules that 17-11/17-12 must report their wall-clock ratio against the
reference `numint`, not only their accuracy against it.

### 1.5 §8.9 understates the `spglib` question — the native path IS upstream's default

`§8.9`'s note says `pyscf_spglib.py` is "an *optional* bridge". Stronger and
more useful:

* `search_point_group_ops` (`geom.py:27-77`) is a self-contained brute force over
  `lib.cartesian_prod([[1,0,-1]]*9)` — **19 683 integer matrices**, filtered by a
  metric-preservation test. No spglib, no external tables.
* `SpaceGroup.backend` defaults to `'pyscf'` (`space_group.py:264`); spglib is
  reached only at `space_group.py:293` when the user sets `backend = 'spglib'`,
  and `space_group.py:288-290` warns that spglib does not handle
  `cell.dimension < 3` at all.

**Recommendation: do not ship the `spglib` feature in v2.0.** It is a second
implementation of a path this phase must implement natively anyway, it cannot
serve the `graphene` (`dimension = 2`) reference system, and no gate in the
phase needs it. Record it as a v2.1 nicety.

### 1.6 §8.9's 17-06 is blocked on two phases that have shipped no code

`§8.9` plan 17-06 is "`kmp2_ksymm`, `kccsd_rhf_ksymm`". As of 2026-08-31:

```
crates/pyscf-pbc-mp/src   — 13 lines (lib.rs + error.rs stubs)
crates/pyscf-pbc-cc/src   — 13 lines (lib.rs + error.rs stubs)
crates/pyscf-pbc-ao2mo/src — 13 lines
```

Phase 15 is **planned only** (`.planning/phases/15-periodic-ao2mo-kmp2/`, seven
plans, no implementation). Phase 16 has no plans at all. `kmp2_ksymm.kernel`
(`kmp2_ksymm.py:30`) is a rewrite of `kmp2.kernel` over IBZ k-tuples, and
`kccsd_rhf_ksymm.py` (806 l) + `kintermediates_rhf_ksymm.py` (265 l) sit on top
of a `KRCCSD` that does not exist.

**Hard prerequisite, stated in the plan rather than discovered in it:** 17-09
cannot start before Phase 15 ships `KMP2`, and its CC half cannot start before
Phase 16 ships `KRCCSD`. If Phase 16 slips, **17-09 ships the MP2 half and defers
the CC half explicitly** — it does not guess at intermediates it cannot test.

### 1.7 `KPoints` is not confined to the SCF layer, and that decides where it lives

D-PBC-15 says k-point symmetry is "a Phase 17 add-on layer, never a fork of the
Phase 11/12 drivers". That is correct **about the drivers** — and it is the whole
story only if `KPoints` never has to be seen below them. It does:

| upstream site | what it does |
|---|---|
| `pyscf/pbc/dft/numint.py:328, 431, 647, 779, 859, 908, 956` | **seven** `isinstance(kpts, KPoints)` branches |
| `pyscf/pbc/df/fft.py:230-246`, `aft.py:174, 613-641`, `df.py:189-217`, `mdf.py:59` | every DF builder's `kpts` setter accepts a `KPoints` |
| `pyscf/pbc/dft/multigrid/multigrid.py:43`, `multigrid_pair.py:26`, `pp.py` | multigrid takes it too |

So `pyscf-pbc-df` and `pyscf-pbc-dft` must both be able to *name* the type. See
§4 for the ruling that follows from this; it is the one decision that must be
made before the first line of 17-02.

---

## 2. The gate statements that cannot be believed as written

### 2.1 `1e-14` (ROADMAP) vs `1e-9` (master plan §7) vs **upstream's own 5e-8 / 5e-7**

`ROADMAP.md` Phase 17: "symmetry-restricted energies equal full-BZ energies to
**1e-14**". `PBC-MASTER-PLAN.md:446`: "`KRHF` with `space_group_symmetry=True`
equals the no-symmetry energy to **1e-9**". Five orders apart, in two documents
describing the same gate, **and neither was measured** — the identical failure
mode `14-CONTEXT.md §2` and `15-CONTEXT.md §2.1` already caught twice.

**Upstream's own test suite asserts the same comparison, and it is looser than
both:**

| upstream test | comparison | decimals | i.e. |
|---|---|---|---|
| `scf/test/test_khf_ksym.py:84` | `KRHF` ksymm vs full BZ, **gamma-centred** | 7 | 5e-8 |
| `scf/test/test_khf_ksym.py:92` | `KRHF` ksymm vs full BZ, **Monkhorst** | 6 | 5e-7 |
| `scf/test/test_khf_ksym.py:107, 117` | `KUHF`, gamma-centred / Monkhorst | 7 / 6 | 5e-8 / 5e-7 |
| `scf/test/test_khf_ksym.py:130` | `KUHF` + smearing, Monkhorst | 6 | 5e-7 |
| `scf/test/test_khf_ksym.py:138, 154` | `KRHF`/`KUHF` on **GDF** | 7 | 5e-8 |
| `dft/test/test_krks_ksym.py:86, 101` | `KRKS` gamma-centred / Monkhorst | 7 / 6 | 5e-8 / 5e-7 |
| `dft/test/test_krks_ksym.py:169, 181, 207` | `KRKS` LDA/GGA/RSH on **GDF** | 8 | 5e-9 |
| `mp/test/test_ksym.py:56` | `KMP2` `e_corr` ksymm vs full BZ | 10 | 5e-11 |
| `cc/test/test_kccsd_ksymm.py:60, 62` | `KRCCSD` `t1`/`t2` ksymm vs full BZ | 6 | 5e-7 |

**`1e-14` is six to seven orders tighter than what upstream asserts, and would
fail a byte-perfect port.** The reason is structural, not sloppy: the two runs
are **independently converged SCFs**. They take different DIIS paths, they hit
`conv_tol` from different sides, and at a Monkhorst grid the time-reversal fold
changes which member of a degenerate pair is diagonalised first. The residual is
a *convergence* difference; it is not a statement about the symmetry algebra.

Note the shape of the table, which is itself evidence: **post-SCF quantities are
gated TIGHTER than the SCF** (`KMP2` at 5e-11) because they read the *same*
converged orbitals through two index paths, while the SCF comparison runs the
whole cycle twice. Any restatement must respect that ordering or it is measuring
the wrong thing.

### 2.2 The restatement: split the algebra from the convergence

**MEASURED by 17-01 (`measurements/README.md`, 2026-09-01), before any of this
was written into a test**, and now the basis for `ROADMAP.md` and
`PBC-MASTER-PLAN.md §7`/`§8.9`, together with this file, as 15-01 requires.

* **Gate A — the IBZ counts. Exact integers, no tolerance, no oracle needed.**
  **REPRODUCED EXACTLY** on upstream's own Si cell
  (`lib/test/test_kpts_ksymm.py:56-89`, `[16,16,16]`):

  | configuration | `nkpts_ibz` |
  |---|---|
  | `space_group_symmetry=True` (non-symmorphic allowed) | **145** |
  | `symmorphic=True`, `+ time_reversal_symmetry=True` | **145** (same `kpts_ibz`, `:71`) |
  | `symmorphic=True`, `time_reversal_symmetry=False` | **245** |
  | `with_gamma_point=False`, `space_group_symmetry=True` | **408** |
  | `with_gamma_point=False`, `symmorphic=True` | **816** |
  | `time_reversal_symmetry=True` only (no space group) | **2052** |

  A single wrong symmorphic branch, a wrong time-reversal fold or a wrong
  `wrap_around` interaction moves one of these integers. `2052 = 4096/2 + 4`
  also independently pins the TRIM count. **§3.10's prediction is CONFIRMED
  BY MEASUREMENT**: this exact six-integer set reproduces bit-for-bit on
  `si` (`a=5.4306 Å`) and `diamond` (`a=3.5668 Å`) — both Fd-3m, same
  non-symmorphic space group as upstream's Si, different lattice constant —
  while `lif`/`he_fcc` (Fm-3m, symmorphic) collapse to `{145,145,145,408,408,2052}`
  (`C=A`, `E=D`, because a symmorphic group has no non-symmorphic ops to lose
  under `symmorphic=True`) and `graphene` (2D) gives a third, unrelated set.
  Only `F=2052` (`time_reversal_symmetry=True` only) is invariant across every
  system tested, including `graphene` — it depends on the `[16,16,16]` mesh's
  Γ/TRIM count alone, never the lattice. **Decision: gate on the integers
  using the port's own `si`/`diamond` fixtures (§9.2) — no sixth fixture is
  needed for the integers.** `finger(kpts_ibz)` does NOT travel (it scales
  with `1/a`, confirmed measured); upstream's exact Si cell is recorded in
  `measurements/gate_a.out` as an optional finger-only cross-check, not a gate.

* **Gate B — the transforms, against ONE converged SCF.** `transform_dm`,
  `transform_1e_operator`, `transform_mo_occ`, `transform_mo_energy` and
  `symmetrize_density` are exact linear maps; feed them the IBZ slice of a
  *single* full-BZ run and compare to that same run's full-BZ arrays. No second
  SCF, so no convergence noise. Upstream asserts 7–8 decimals here
  (`test_kpts_ksymm.py:95-143`) but does so against a `max_cycle=1` reference,
  which is not a floor. **MEASURED on `si [3,3,3]`, `KRKS(LDA,VWN)`,
  `conv_tol=1e-11`: the floor is set by `cell.precision` (the AO-integral
  screening tolerance), NOT by `conv_tol`.** At PySCF's default
  `cell.precision=1e-8`: `transform_dm`/`make_rdm1(transform_mo_coeff(...))`
  land at **4.481e-10**, `transform_mo_energy` at **5.331e-11**,
  `transform_1e_operator` at **2.931e-11**, `symmetrize_density` at
  **6.607e-13** — NOT uniformly ≥1e-12 as this section originally guessed.
  Tightening `conv_tol` to 1e-13 changed nothing; tightening `cell.precision`
  to 1e-13 collapsed every residual to machine epsilon (≤8e-15). **Gate B is
  therefore: ≤1e-9 at PySCF's default `cell.precision`, ≤1e-13 when
  `cell.precision=1e-13` on both sides.** The `mo_coeff`-elementwise trap
  (§3.1) measured exactly as large as predicted (~2.3, O(1)) at both
  precisions — confirming the density-matrix comparison is the only valid one.

* **Gate C — energy, port vs port.** `KRHF`/`KRKS` with symmetry vs without, on
  the port's own two runs, `conv_tol = 1e-11`, **mesh pinned on both sides**
  (§3.3). The target is upstream's own level or better: 5e-8 gamma-centred,
  5e-7 Monkhorst. **MEASURED on `si` (full `KRHF`/`KRKS` × gamma/Monkhorst ×
  FFTDF/GDF grid) and `diamond`/`lif`/`he_fcc` (reduced set, resource-scoped —
  see `measurements/README.md`): FFTDF lands at 6.9e-14…5.4e-11 (routinely
  BEATING upstream's 5e-8/5e-7 by three to six orders), GDF lands at
  1.8e-11…5.5e-10 — both comfortably inside upstream's own tolerance.**
  `lif` and `graphene` at an aggressively resource-capped mesh (25³, far below
  their natural mesh — `lif`'s default is 81³, `graphene`'s is
  `[45,45,351]`) produced degraded numbers (`lif` 1.461e-04, `graphene`
  non-convergent) that are **mesh-cap artefacts, not symmetry-algebra
  failures** — recorded as a resource deviation, not a gate result, in
  `measurements/README.md`.

* **Gate D — energy, port vs upstream, PER DF ROUTE.** `15-CONTEXT §2` already
  established that a "matches upstream" assertion that does not name its DF
  backend is untestable; the same applies here, and upstream's own table above
  gates FFTDF at 7 decimals and GDF at 8. Report both — **measured together
  with Gate C above, since this plan compares the port's own two DF routes
  against each other under symmetry (there is no Rust port yet to compare
  against upstream separately); 17-07/17-08/17-13 gate the port against
  upstream per route once it exists.**

* **Gate E — multigrid is NOT held to any of the above.** Upstream's own
  `dft/test/test_multigrid.py` gates `MultiGridNumInt` against the reference
  `numint` at **8 decimals** for `get_pp`/`get_nuc`/`get_j` (`:84-129`) and
  **7 decimals** for the XC potential and `exc`/`ecoul` (`:139-217`), and
  `test_krks_ksym.py:240` gates a multigrid `KRKS` energy at 7. Multigrid is a
  different quadrature, not a different implementation of the same one. A gate
  that demands 1e-9 of it fails a correct port. **MEASURED on `diamond`/`si`
  at two meshes each: v1 (`MultiGridNumInt`) is essentially algebraically
  EXACT against FFTDF/`numint` (1e-12…1e-14, beating upstream's own 7-8
  decimal tolerance by orders of magnitude) except when the mesh is
  genuinely too coarse for the system (si `get_pp` at a 0.6×-linear coarser
  mesh: 4.384e-09, three orders worse — a mesh-CONVERGENCE artefact, not
  definitional). v2 (`MultiGridNumInt2`) carries a MESH-INDEPENDENT floor
  against FFTDF (~2.4e-8 diamond, ~1.5e-7 si, unchanged across both meshes
  tested) — a DEFINITIONAL gap, the direct Phase-17 analogue of Phase 14's
  GDF-vs-RSDF 4.5e-6. Converged `KRKS` `e_tot`: v1 beats upstream's 1e-7 gate
  by 4-5 orders (4.7e-12 diamond, 3.8e-12 si); v2 is markedly looser
  (3.986e-07 diamond, 1.938e-06 si — WORSE than 1e-7 on `si`). See §8 below
  for the (unexpected) wall-clock finding.**

### 2.3 What is NOT in dispute — and one qualification MEASURED by 17-01

`KMP2` ksymm at **5e-11** (`mp/test/test_ksym.py:56`) and the `rdm1` at the same
level (`:66`) are tight because they reuse one SCF. Hold 17-09 to that. Do not
relax it to the SCF's 5e-8 by analogy.

**Qualification:** on upstream's own He fixture, post-SCF IS measurably
tighter than SCF (`|d e_corr|=3.096e-16` < `|d E_scf|=4.441e-16`) — but on
`si [2,2,2]` it is **not** (`|d e_corr|=1.067e-09` > `|d E_scf|=5.544e-10`,
post-SCF ~2× LOOSER). **The "post-SCF tighter than SCF" ordering is a property
of upstream's tiny, near-machine-precision He fixture, not a general
theorem** — 17-09 must gate `e_corr` on its own measured number per system,
not assume the ordering transfers. `KRCCSD` (`cc/test/test_kccsd_ksymm.py`)
is unmeasured: Phase 16 has not shipped as of this measurement.

---

## 3. Traps recorded in advance

Each carries the line that proves it. These are the Phase-17 equivalents of
`15-CONTEXT §3`'s `LARGE_DENOM` and column-major findings — the defects that are
invisible to the obvious test.

### 3.1 Never compare `mo_coeff` elementwise. Compare the density matrix.

`transform_mo_coeff` is only defined up to a unitary mixing **within each
degenerate subspace**, and every symmetric cell has degeneracies at high-symmetry
k-points by construction. Upstream knows this and its own test never compares
coefficients: `test_kpts_ksymm.py:96-99` transforms the MOs, builds a density
matrix from them with `khf.make_rdm1`, and compares **that** to the reference DM.
A port that asserts on `mo_coeff` directly will fail while being correct.

Related: `_get_phase` (`symmetry.py:226`) and `get_rotation_mat_for_mos`
(`kpts.py:757`) fix the phase convention. `MORotationMatrix` (`kpts.py:1127`)
caches it. Getting the convention wrong is the R-05 failure mode ("wrong by a
phase → silently wrong energies") one layer up.

### 3.2 `mo_coeff` is COLUMN-MAJOR and the rotation acts on the AO index

`crates/pyscf-pbc-scf/src/types.rs:119` — `mo_coeff[idx(set,k)]`, **COLUMN-MAJOR
`nao × nmo`**. The `Dmats` from `make_Dmats` (`symmetry.py:79`) are
block-diagonal over shells in the **AO** index, applied via `make_rot_loc`
(`symmetry.py:330`). This is exactly the shape of 14-05's `decompose_j2c` defect
— a column-major eigenvector read row-major, **worth +6 306 866.73 Ha and
invisible to every gate then existing.** Give the AO-rotation one implementation
and pin it with the identity that defines it (`R S Rᴴ = S` on the overlap for
every op), not with a round-trip.

### 3.3 Turning symmetry on can CHANGE `cell.mesh`, and then the gate is measuring the mesh

`check_mesh_symmetry` (`symmetry.py:96`) can *enlarge* the FFT mesh so it carries
the lattice symmetry; `Cell.symmetrize_mesh` (`cell.py:1529-1550`) applies it and
warns when the result grows more than 8×. So a naive
"`space_group_symmetry=True` vs `False`" energy comparison can differ for a
reason that has nothing to do with the IBZ.

**Every gate in §2.2 that compares two runs MUST pin the mesh on both sides.**
Phase 14 already paid for the analogous mistake — `14-VERIFICATION`'s MDF gate
had to force a matched mesh because "for MDF the mesh is DEFINITIONAL".

### 3.4 The Fermi level is a full-BZ quantity even though the SCF is IBZ

`khf_ksymm.get_occ` (`khf_ksymm.py:31-67`) calls
`kpts.transform_mo_energy(mo_energy_kpts)` and sorts over the **unfolded BZ**
before picking `fermi = mo_energy[nocc-1]`, then folds back through
`check_mo_occ_symmetry` (`:65`). Fusing that into the IBZ loop gives a different
occupation on any cell with a k-dependent gap — the R-06 failure mode.

### 3.5 `energy_elec` weights by `weights_ibz`; it is not `1/nkpts`

`khf_ksymm.py:74-76`:
`e_coul = einsum('k,kij,kji', kpts.weights_ibz, dm_kpts, vhf_kpts) * 0.5`.
And `get_init_guess` (`:396-398`) uses the same weights and then multiplies by
`nkpts`. `15-CONTEXT §3` recorded the same class of trap for KMP2's three
`1/nkpts` sites being two distinct divisions; this is the IBZ version. Enumerate
every weighted sum in the adapter and name which weight each one takes.

### 3.6 Op enumeration order is observable

`search_point_group_ops` (`geom.py:43-77`) iterates
`lib.cartesian_prod([[1,0,-1]]*9)` in a fixed order and `append`s survivors in
that order. That order propagates into `stars_ops`, `stars_ops_bz` and the
`finger(kpts_ibz)` values upstream pins (`test_kpts_ksymm.py:67, 75, 80, 85`).
**Preserve the iteration order exactly**; `cartesian_prod` varies the LAST axis
fastest.

### 3.7 `geom.py`'s `np.clip` before `arccos` is load-bearing (upstream issue 3113)

`geom.py:38-41` and `:57-59` clip the metric ratio into `[-1, 1]` before
`arccos`, with the pre-fix line left in as a comment. Diagonal entries exceed 1
by rounding on a perfectly cubic lattice, and unclipped `arccos` returns NaN,
which then silently fails the `>` comparison and **admits a wrong rotation**.
Port the clip and cite the issue.

### 3.8 `symmetrize_density` is a D-PBC-17 accumulation

`kpts.py:377-414` sums `nkpts × ngrids` real-space densities. That is the same
shape D-PBC-17 governs and `15-CONTEXT` ruled on for KMP2's
`nkpts³·nocc²·nvir²`: it goes through `oracle_sum`/`oracle_zsum` **from the first
version, not as a retrofit**, and is gated bit-identical at `RAYON_NUM_THREADS`
1 and 8 (§9.3).

### 3.9 The `Symmetry`→`Cell` reference cycle is a Python problem only

`cell.py:1576-1579` deletes `lattice_symmetry.cell` and
`lattice_symmetry.spacegroup.cell` after building, purely to break a refcount
cycle. In Rust the cycle does not exist and the deletion has no meaning — **but
do not let `Symmetry` own a `Cell` either.** Build it from a borrowed `&Cell`
and store only the lattice-derived data (rotations, translations, `Dmats`,
crystal class). Storing a cloned `Cell` would silently desynchronise on
`cell.build()`.

### 3.10 Upstream's ksymm fixture is not §9.2's `si`

`test_kpts_ksymm.py:30-40` uses Si₂ fcc with `a = 2.6935121974 Å` half-vectors,
i.e. a cubic constant of 5.3870243948 Å. `PBC-MASTER-PLAN §9.2`'s `si` is
5.4306 Å. The `nkpts_ibz` integers in Gate A depend only on the lattice *type*
and reproduce on either; **the `finger(kpts_ibz)` values do not** — they scale
with 1/a. If the phase wants to pin upstream's fingers it must add upstream's
exact cell as a sixth fixture. 17-01 decides and records which.

---

## 4. The crate-layering ruling — make this before writing 17-02

Upstream declares `class KPoints(symm.Symmetry, lib.StreamObject)`
(`kpts.py:847`): `KPoints` **is a** `Symmetry`. The naive file-name mirroring
puts `Symmetry` in `pyscf-pbc-symm` and `KPoints` in `pyscf-pbc-lib`, following
`lib/kpts.py`. **That does not compile.** Current declared dependencies:

```
pyscf-pbc-lib  → core, algebra                       (no pbc-gto!)
pyscf-pbc-symm → core, algebra, pbc-lib, pbc-tools, pbc-gto
pyscf-pbc-df   → …, pbc-lib, pbc-tools, pbc-gto      (no pbc-symm)
pyscf-pbc-dft  → …, pbc-lib, pbc-tools, pbc-gto, pbc-df, pbc-scf
```

`pyscf-pbc-symm` already depends on `pyscf-pbc-lib`, so putting `KPoints` in
`pyscf-pbc-lib` and having it hold a `Symmetry` is a **dependency cycle**. And
`pyscf-pbc-lib` cannot see `Cell` at all, which `KPoints::build` needs.

**RULING (adopt as D-PBC-25, recorded by plan 17-02):**

1. **`KPoints` lives in `pyscf-pbc-symm`, not `pyscf-pbc-lib`**, next to
   `Symmetry`. The file-name mirror is broken deliberately and the module doc
   says why.
2. **Composition, not inheritance** — `KPoints { symmetry: Symmetry, … }`.
   Rust forces this anyway; record it so the port of `KPoints`'s inherited
   methods is not mistaken for duplication.
3. **`pyscf-pbc-df` and `pyscf-pbc-dft` gain a `pyscf-pbc-symm` dependency**, so
   the `numint` and DF-builder branches of §1.7 can name the type. Verified
   acyclic against the four dependency lists above.
4. `xtask check_dependency_wall` polices **only** cubecl deps
   (`xtask/src/bin/check_dependency_wall.rs:28-60`), not inter-crate layering, so
   no exemption is needed for (3).
5. **Corollary that binds 17-11/17-12:** `pyscf-pbc-dft` may **not** declare a
   cubecl dependency (ALG-06; carve-out is `pyscf-algebra`, `pyscf-runtime`,
   `pyscf-kernels`, `pyscf-bench`). **Every multigrid collocation/integration
   kernel goes in `pyscf-kernels`**, with `pyscf-pbc-dft` holding only the host
   task-list logic. Same split `ft_aopair` already uses (D-PBC-21).

`pyscf-pbc-symm/src` is a 13-line stub today (`lib.rs` 6, `error.rs` 7), so
nothing has to be undone.

---

## 5. What already ships, and must not be re-ported

| upstream | Rust | status |
|---|---|---|
| `kpts_helper.get_kconserv` / `get_kconserv3` | `pyscf-pbc-lib/src/kpts_helper.rs:282, 367` | **shipped (09-07)** |
| `kpts_helper.member`/`unique`/`round_to_fbz`/`is_gamma_point` | same file `:77-224` | **shipped** |
| `kpts_helper.group_by_conj_pairs` / `kk_adapted_iter` | same file `:484, 571` | **shipped (14-xx)** |
| `cell.make_kpts` (no symmetry) | `pyscf-pbc-gto/src/kpts_mesh.rs:51` | **shipped (09-07)** |
| `cell.make_kpts(space_group_symmetry=…)` | `kpts_mesh.rs:112-121` | **refuses, `phase: 17`** — 17-05 closes it |
| `Cell.space_group_symmetry` field | `pyscf-pbc-gto/src/cell.rs:84` | **field exists, honoured nowhere**; its doc comment says "not implemented before Phase 12" and is stale |
| `Cell.symmorphic` / `lattice_symmetry` / `symm_orb` / `irrep_id` | — | **missing** — 17-03 / 17-04 |
| `super_cell` with symmetry | `pyscf-pbc-gto/src/supercell.rs:100-104` | refuses; upstream also refuses (`pbc.py:784-785`) — leave refusing |
| the SCF driver `KOverrideHooks` | `pyscf-pbc-scf/src/khooks.rs:21` | **shipped** — 17-07 implements the trait, it does not fork `kscf::kernel` |
| `pyscf-pbc-tools::fft` / `ifft` | `pyscf-pbc-tools/src/fft.rs:96, 110` | **shipped** — multigrid's G-space half needs no new FFT |
| `kpts_helper.is_trim` | — | **missing**; `khf_ksymm.py:126` needs it (`kpts_helper.py:39`). One function, lands in 17-05 |

---

## 6. Prerequisites and sequencing

```
        ┌──────────────────────────────────────────────┐
17-01   │ measure the floor, build fixtures, set gates │  ← blocks everything
        └──────────────────────────────────────────────┘
             │
   ┌─────────┴──────────┐                        ┌──────────────────┐
   ▼                    ▼                        ▼                  ▼
17-02 geom/tables/  17-10 supermole          17-11 multigrid v1  (independent
      group          (independent track,           │              tracks —
   ▼                  unblocks rsjk)          17-12 multigrid v2  start any time
17-03 space_group/                                                after 17-01)
      symmetry + Cell
   ▼
17-04 basis.py (symm_orb)
   ▼
17-05 KPoints  ──────────────►  17-06 ktensor
   ▼                                  │
17-07 khf/kuhf/kghf_ksymm  ◄──────────┘
   ▼
17-08 krks/kuks/krkspu/kukspu_ksymm
   ▼
17-09 kmp2_ksymm [needs Phase 15]  (+ kccsd_rhf_ksymm [needs Phase 16])
   ▼
17-13 verification
```

**Hard external prerequisites:**

* 17-09 MP2 half — Phase 15 shipped (currently *planned only*).
* 17-09 CC half — Phase 16 shipped (currently *unplanned*). Defer, do not guess.
* Nothing else in the phase depends on Phase 15 or 16.

---

## 7. Plan map

| Plan | Content | Upstream | Lines |
|---|---|---|---|
| **17-01** | Measure the floor; fixtures; restate every gate in three documents | — | — |
| **17-02** | `symm/geom.py` + `tables.py` + `group.py`; record D-PBC-25 | 245 + 100 + 476 | 821 |
| **17-03** | `symm/space_group.py` + `symmetry.py`; `Cell` symmetry fields, `symmetrize_mesh`, `build_lattice_symmetry` | 369 + 348 | 717 |
| **17-04** | `symm/basis.py` — `symm_adapted_basis`, `Cell::_build_symmetry` **(§8.9 omitted this)** | 161 | 161 |
| **17-05** | `lib/kpts.py` `KPoints`; close `make_kpts_with_symmetry`; `is_trim` | 1223 | 1223 |
| **17-06** | `lib/ktensor.py` `KsymmArray` **(moved here by 15-CONTEXT §1.1)** | 386 | 386 |
| **17-07** | `khf_ksymm` / `kuhf_ksymm` / `kghf_ksymm` as `KOverrideHooks` impls; record D-PBC-26 | 410 + 219 + 211 | 840 |
| **17-08** | `krks_ksymm` / `kuks_ksymm` / `krkspu_ksymm` / `kukspu_ksymm` | 144+147+72+59 | 422 |
| **17-09** | `kmp2_ksymm` (+ `kccsd_rhf_ksymm` if Phase 16 has landed) | 285 (+1071) | 285–1356 |
| **17-10** | `ft_ao._RangeSeparatedCell` + `ExtendedMole`; close `exclude_dd_block` **(§8.9 omitted this)** | ~600 of `ft_ao.py` | ~600 |
| **17-11** | multigrid **v1** — `multigrid.py`, 2 cubecl kernels | 1962 | 1962 |
| **17-12** | multigrid **v2** — `multigrid_pair.py` + `pp.py` + `utils.py`, 12 kernels | 1257+256+70 | 1583 |
| **17-13** | Verification against the restated gates | — | — |

**13 plans, ~10 700 lines of upstream Python** — larger than Phase 14 (9 plans).
§8.9's eight was an undercount, not a compression.

---

## 8. The speed ruling — symmetry only pays for itself if `get_jk` exploits it too

`khf_ksymm.get_jk` (`khf_ksymm.py:250-277`) transforms the IBZ density to the
full BZ, calls the **ordinary** `PeriodicDf::get_jk` over all `nkpts`, then
folds the result back to the IBZ. Upstream itself does **not** shrink the DF
cost under symmetry — the only savings it takes are in `eig`'s
block-diagonalisation (17-04) and, incidentally, in needing fewer converged
k-points' worth of downstream state. **The single most expensive step in every
SCF cycle — the DF `get_jk` build — is run at full `nkpts` cost regardless of
`nkpts_ibz`.** For Si at `[16,16,16]`, `nkpts_ibz = 145` against `nkpts = 4096`
(§2.2 Gate A) — a 28× gap upstream leaves entirely on the table. 17-07 Task 6
already says this out loud ("a version that is correct and no faster is a
finished correctness milestone and an unfinished feature") but, as recorded,
only *measures* the shortfall. This section rules on closing it.

**The physics that licenses closing it:** the Fock/Coulomb operator built over
a lattice-symmetric `Cell` is itself equivariant under the cell's space group.
Fed a density `{dm(k)}` that is already symmetric (i.e. unfolded from the IBZ
by `transform_dm`, so `dm(Rk) = R·dm(k)·Rᴴ` for every op `R` by construction),
the resulting one-particle operators inherit the same equivariance:
`vj(Rk) = R·vj(k)·Rᴴ` and `vk(Rk) = R·vk(k)·Rᴴ` (conjugated when `R` includes
time reversal). This is exactly the identity `transform_1e_operator`
(`symmetry.py:323`) already encodes for the overlap and Fock matrices
(17-03 Task 6, 17-05 Task 3) — it is not a new claim, only a new place to apply
an operation this phase builds anyway.

**RULING (adopt as D-PBC-26, recorded by plan 17-07):**

> ### ERRATUM on point 1 — MEASURED WRONG, 2026-09-02
>
> **Point 1 below does not compute the same `vj`/`vk`, and the 223x / 40x in
> point 3 compared two different quantities.** Full derivation and the
> measurement are in
> [`KUKS-KSYMM-MULTIGRID-OPTIMISATION-PLAN.md`](../../pbc/KUKS-KSYMM-MULTIGRID-OPTIMISATION-PLAN.md)
> §2.2.3 and the `PBC-MASTER-PLAN.md` D-PBC-26 entry; in one line: the Coulomb
> density built from an IBZ k-list is `Σ rho_k / N_ibz` and the true one is
> `Σ w_k <rho_k>_star`, and `rho_k` is not point-group invariant. Measured
> `max |d veff| = 9.486e-2 Ha` on `si [2,2,2]`
> (`khf_ksymm.rs::ibz_only_get_jk_is_not_an_identity`).
>
> The attainable bound is `nkpts / nkpts_ibz`, reached **bit-identically**
> (`max |d| = 0e0`) by restricting the OUTPUT set — `kpts_band = kpts_ibz` —
> rather than the sampling set. Points 2, 4, 5 and 6 stand; point 4 was
> already satisfied, since the DFT k-symmetric adapters have taken the band
> route since 17-08 Task 2.

1. `get_jk` gains a **second, faster route**: call `PeriodicDf::get_jk` only at
   `kpts.kpts_ibz`, then unfold `vj`/`vk` to the full BZ with
   `transform_1e_operator` instead of re-running the DF builder at every BZ
   k-point. The literal upstream route (unfold the density, call `get_jk` over
   all `nkpts`, fold the result back) **stays as the reference implementation**
   — it is what Gate C/D compare against and it is the fallback when
   `use_ao_symmetry` or the fast path is disabled.
2. **The fast path is validated the way 17-10 Task 4 validates the
   MO-factorised `get_k_kpts`: against the port's own reference route, at
   1e-13, not against upstream.** Two routes to the same number inside one
   process is a stronger test than either against a third implementation, and
   it means the fast path can be gated before any upstream oracle exists for
   it.
3. Once the equivalence test is green, 17-07 Task 6's "cost, reported not
   gated" becomes "cost, gated": the fast path's wall time on `si [4,4,4]`
   must be **measurably lower** than the reference route's, and the ratio is
   reported alongside the correctness gate, in the same table. **17-01
   measured the achievable bound: on `si [4,4,4]` (`nkpts=64`,
   `nkpts_ibz=8`), a full-BZ vs IBZ-subset `get_jk` wall-clock comparison
   (same DF object, same mesh, coarsened to `[9,9,9]` to fit the measurement's
   budget — a pure ratio measurement, not an accuracy one) gave
   FFTDF **223x**, GDF **40x** — both far ABOVE the naive `nkpts/nkpts_ibz=8x`
   bound this section originally assumed, because exact-exchange cost in both
   routes scales closer to quadratically than linearly in `nkpts` (the
   `(k,k')` pairwise structure). 17-07's fast path should target the GDF
   number (~40x) as its realistic floor, since GDF is the port's default
   route; FFTDF has room for a substantially larger win.**
4. **`krks_ksymm::get_veff` (17-08 Task 2) inherits the same route for its
   `vj`/`vk` half** — the XC half is already IBZ-native through `numint`'s
   `KSet::Ibz` arm (17-08 Task 1), so DFT's `get_veff` becomes fully
   IBZ-costed once both halves use it, closing the same 28×-class gap for
   `KRKS`/`KUKS`.
5. **This does not change what `pyscf-pbc-df` is allowed to know.** The DF
   crate still never sees `KPoints` or a star (D-PBC-15); it only ever
   receives a plain k-point list, which is now sometimes the IBZ list instead
   of the full BZ. The equivariance unfold happens entirely in
   `pyscf-pbc-scf`/`pyscf-pbc-dft`, using machinery 17-03/17-05 already ship.
6. If the fast path turns out **not** to beat the reference route on the CPU
   backend — the recorded precedent is `zgemm_dense`, 6–12× slower than a host
   rayon loop despite being the "obvious" faster route — say so with the
   measured numbers and ship the reference route as the only one, exactly as
   17-10 Task 4 requires for the MO-factorised `get_k_kpts`. Do not ship a
   "faster" path that measured slower.

**Corollary — multigrid's own speed claim, MEASURED by 17-01: upstream's OWN
multigrid is SLOWER than the reference `numint`/FFTDF on the systems this
plan can afford to run.** `multigrid` exists *only* as a speed feature (§1.4:
"nothing in Phases 18-20 requires it for correctness"), and a converged
`KRKS(LDA,VWN)` timing comparison on `diamond`/`si` (`measurements/README.md`)
gave a reference/multigrid wall-time ratio of **0.49x (diamond, v1)**,
**0.21x (si, v1)**, **0.39x (diamond, v2)**, **0.18x (si, v2)** — i.e. v1 and
v2 both ran SLOWER than plain FFTDF+`numint`, not faster, on 8-AO systems at
their natural `[35-65]³` meshes. **This does not mean multigrid is a bad
port target** — upstream's own collocation/task-list overhead plausibly
dominates only at small system sizes and the crossover to a real win (more
AOs, denser meshes, more k-points) is real in the literature and NOT ruled
out by this measurement — but it means **17-11/17-12 must not assume a speed
win exists at the reference-system scale this repo tests at, and must
re-measure the ratio at whatever scale they actually target** rather than
citing "multigrid is faster" as an unexamined premise. The `zgemm_dense`
precedent (6-12x slower than a host loop despite being the "obvious" faster
route) is directly on point here, and this is now a second, independent data
point for the same caution.

Both corrections keep the phase's standing discipline: a performance claim is
either measured and reported, or it is not made.
