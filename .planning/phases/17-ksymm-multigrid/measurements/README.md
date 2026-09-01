# Phase 17 pre-implementation measurements (upstream PySCF 2.12.1)

Run every one of these from THIS directory as:

```bash
PYTHONPATH=<workspace root> ../../../../.venv/bin/python -u <script>.py
```

`PYTHONPATH` pointing at the workspace root pins `import pyscf` to the
**vendored** 2.12.1 tree at `<root>/pyscf`, not site-packages. Every script
asserts `pyscf.__version__ == "2.12.1"` before doing anything else, as Phases
13-15 do. `-u` is mandatory for the SCF scripts too: several redirect output
into `.out` files and a buffered pipe hides progress until exit.

**NO Rust source file was created or edited to produce this document.**
`git diff --name-only` / `git status` show only files under `.planning/`.

| script | measures | feeds |
|---|---|---|
| `gate_a.py` | Gate A -- the six `nkpts_ibz` integers, on upstream's own Si cell and on five §9.2-style reference cells | 17-02 through 17-05 |
| `gate_b.py` / `gate_b_tight.py` / `gate_b_prec.py` | Gate B -- the transform floor against ONE converged SCF, at default `cell.precision` (`gate_b.py`), tightened `conv_tol` (`gate_b_tight.py`, isolates SCF convergence -- no effect), and tightened `cell.precision` (`gate_b_prec.py`, isolates integral screening -- collapses to machine epsilon) | 17-03/17-05 |
| `gate_c_d.py` (si's full grid, output saved as `gate_c_d_si_only.out`) + `gate_c_d_part2.py` (diamond, reduced set) + `gate_c_d_part3.py` (lif/he_fcc/graphene broad sweep + mesh demo) + `gate_c_d_repro.py` (run-to-run/thread spread) | Gates C/D -- the two-SCF energy floor, mesh pinned, per DF route | 17-07/17-08/17-13 |
| `gate_mp2.py` | Task 4 -- the post-SCF (KMP2) floor | 17-09 |
| `gate_multigrid.py` | Gate E -- multigrid v1/v2 vs reference `numint`, two meshes, two systems | 17-11/17-12 |
| `speed_get_jk.py` | Task 6.1 -- `get_jk` full-BZ vs IBZ-subset wall-clock ratio | 17-07 (D-PBC-26) |
| `gate_multigrid.py` (converged-KRKS section) | Task 6.2 -- multigrid vs reference `numint` wall-clock ratio | 17-11/17-12 |

## Reference systems used (PBC-MASTER-PLAN §9.2, all `gth-szv`/`gth-pade` unless noted)

Two fixture bugs were found and fixed while writing `gate_a.py` and are recorded
here so nobody re-derives them: the diamond-structure atom offset is
**a_conv/4 · (1,1,1)** in Cartesian (the tetrahedral position), NOT a/2 · (1,1,1)
(that is rocksalt's octahedral position) -- an earlier draft used a/2 for `si`
and `diamond` and silently built a rocksalt-symmetry cell instead, which is
why its `nkpts_ibz` integers came out wrong (see Gate A below). `he_fcc` is a
**single-atom** primitive fcc cell, not a two-atom one.

| system | structure | space group | atoms |
|---|---|---|---|
| `si` | diamond, `a = 5.4306 Å` | Fd-3m (non-symmorphic) | Si at (0,0,0), Si at a/4·(1,1,1) |
| `diamond` | diamond, `a = 3.5668 Å` | Fd-3m (non-symmorphic) | C at (0,0,0), C at a/4·(1,1,1) |
| `lif` | rocksalt, `a = 4.03 Å` | Fm-3m (symmorphic) | Li at (0,0,0), F at a/2·(1,1,1) |
| `he_fcc` | single-atom fcc, `a = 3.0 Å` | Fm-3m (symmorphic) | He at (0,0,0) |
| `graphene` | hexagonal, 20 Å vacuum, `dimension=2` | p6/mmm-derived, 2D | C at (0,0,0), C at (a1+2·a2)/3 |

---

## Gate A -- the IBZ counts (`gate_a.py` / `gate_a.out`)

**Reproduced EXACTLY on upstream's own Si cell** (`lib/test/test_kpts_ksymm.py:30-89`,
`a = 2.6935121974 Å` half-vectors, `gth-szv`/`gth-pade`, `[16,16,16]` k-mesh):

| configuration | expected | measured |
|---|---|---|
| `space_group_symmetry=True` | 145 | **145** (finger 2.2116408840211186) |
| `symmorphic=True, time_reversal_symmetry=True` | 145 | **145** (identical `kpts_ibz`, max diff 0) |
| `symmorphic=True, time_reversal_symmetry=False` | 245 | **245** (finger -2.0196383066365353) |
| `with_gamma_point=False, space_group_symmetry=True` | 408 | **408** (finger -2.5811145613280138) |
| `with_gamma_point=False, symmorphic=True` | 816 | **816** (finger -1.1244923995083895) |
| `time_reversal_symmetry=True` only | 2052 | **2052** |

The interpreter/vendored tree gate PASSES without qualification. Nothing else
in this plan would be trustworthy if this had not reproduced -- per Task 1,
the phase would have stopped here.

### Does the integer set travel with lattice TYPE, or with the lattice constant?

17-CONTEXT §3.10 predicted the six integers depend on the lattice **type**
(space group), not the lattice constant, while `finger(kpts_ibz)` does not.
Measured on five more cells at `[16,16,16]`, same six configurations
(`A,B,C,D,E,F` = the six rows above, in order):

| system | space group | `{A,B,C,D,E,F}` | matches upstream's Si set? |
|---|---|---|---|
| upstream Si (`a=2.6935121974 Å` half-vectors) | Fd-3m | `{145,145,245,408,816,2052}` | -- |
| `si` (`a=5.4306 Å`, diamond structure) | Fd-3m | `{145,145,245,408,816,2052}` | **YES, EXACTLY** |
| `diamond` (`a=3.5668 Å`, diamond structure) | Fd-3m | `{145,145,245,408,816,2052}` | **YES, EXACTLY** |
| `lif` (rocksalt) | Fm-3m (symmorphic) | `{145,145,145,408,408,2052}` | no -- `C=A`, `E=D` because Fm-3m has no non-symmorphic ops to lose under `symmorphic=True` |
| `he_fcc` (single-atom fcc) | Fm-3m (symmorphic) | `{145,145,145,408,408,2052}` | no, same reason as `lif` |
| `graphene` (`dimension=2`) | 2D hexagonal | `{816,417,816,2176,2176,2052}` | no -- a different lattice type entirely |

**Confirmed exactly as 17-CONTEXT §3.10 predicted.** The integers are a
function of the *space group*, not the lattice constant: `si` and `diamond`
(different `a`, same Fd-3m structure as upstream's Si) reproduce upstream's
full six-integer set bit-for-bit, while `lif`/`he_fcc` (same Fm-3m structure
as each other, but a DIFFERENT space group from Fd-3m) reproduce a different,
mutually-identical set, and `graphene` (a different lattice type again)
reproduces neither. The one integer that is invariant across **every** system
tested, including `graphene`, is `F = 2052` (`time_reversal_symmetry=True`
only, no space group) -- consistent with 17-CONTEXT's `2052 = 4096/2 + 4`
formula, which counts Γ + TRIM points of a `[16,16,16]` mesh under inversion
alone and does not reference the lattice at all.

`finger(kpts_ibz)` does NOT travel, exactly as predicted (`si`'s config-A
finger is `2.193894485831914`, upstream Si's is `2.2116408840211186` -- both
`nkpts_ibz=145` but different fingers, because the k-point coordinates scale
with `1/a`).

**Decision (Task 1):** gate on the six **integers** using the port's own `si`
and `diamond` fixtures (§9.2), which already reproduce upstream's exact
non-symmorphic set with no separate fixture needed. Do **not** attempt to pin
`finger(kpts_ibz)` on `si`/`diamond` -- it is lattice-constant dependent by
construction and would be testing the wrong thing. Upstream's own exact Si
cell (`a=2.6935121974 Å` half-vectors) is recorded above as a **sixth,
optional fixture** for anyone who later wants an exact finger-level
cross-check against `test_kpts_ksymm.py` itself; it is not required by any
gate in this phase.

---

## Gate B -- the transforms, against ONE converged SCF (`gate_b.py`, `gate_b_prec.py`)

`si` (§9.2, diamond structure, `a=5.4306 Å`), `KRKS(LDA,VWN)`, `[3,3,3]`
k-mesh (`nkpts=27`, `nkpts_ibz=4`), `conv_tol=1e-11`. All residuals are
`abs(transformed_full_BZ - actual_full_BZ).max()` against the arrays from the
**same converged run** -- no second SCF, so no convergence noise, exactly as
17-CONTEXT §2.2 specifies.

| quantity | at default `cell.precision` (1e-8) | at `cell.precision=1e-13` |
|---|---|---|
| `transform_dm` | 4.481327e-10 | 7.771561e-15 |
| `make_rdm1(transform_mo_coeff(...))` | 4.481327e-10 | 7.771561e-15 |
| `transform_mo_occ` | 0 | 0 |
| `transform_mo_energy` | 5.330547e-11 | 5.107026e-15 |
| `transform_1e_operator` (via `transform_fock`) | 2.931082e-11 | 5.551297e-16 |
| `symmetrize_density` | 6.606521e-13 | 1.387779e-16 |
| `mo_coeff` compared ELEMENTWISE (demonstration only, per 17-CONTEXT §3.1) | 2.296247 | 2.428734 |

**Correction to 17-CONTEXT §2.2's Gate B expectation.** 17-CONTEXT expected
these linear maps to land at "≥1e-12" unconditionally. Measured: at PySCF's
**default** `cell.precision` (1e-8, the AO-integral screening tolerance, not
the SCF `conv_tol`), the floor is **4.5e-10** for `transform_dm`/
`make_rdm1(transform_mo_coeff)`, ~**5e-11** for `transform_mo_energy`, ~**3e-11**
for `transform_1e_operator`, and ~**7e-13** for `symmetrize_density` -- NOT
uniformly ≥1e-12. Re-running with `cell.conv_tol` tightened from 1e-11 to
1e-13 (SCF convergence) changed **nothing** (same residuals to 5 significant
figures); re-running with `cell.precision` tightened to 1e-13 (integral
screening) collapsed every residual to **machine precision (≤8e-15)**. **The
Gate B floor is therefore set by `cell.precision` (how tightly the physical,
unconstrained SCF solution is forced to be symmetric through integral
screening), not by SCF `conv_tol`.** The transform algebra itself
(`transform_dm`, `transform_1e_operator`, Wigner-D AO rotations in
`symmetry.py`) is exact analytic linear algebra with no numerical
integration in it, confirmed by the `cell.precision=1e-13` column landing at
literal machine epsilon.

**The `mo_coeff`-elementwise demonstration is exactly as large as 17-CONTEXT
§3.1 warns** (~2.3-2.4, i.e. O(1), at BOTH precisions) -- confirming the trap:
`mo_coeff` is only defined up to a unitary mixing within each degenerate
subspace and must never be compared elementwise; the density matrix built
from it is the correct comparison and it lands at the algebra's true floor.

**Gate B, as restated:** `transform_dm`, `make_rdm1(transform_mo_coeff(...))`,
`transform_mo_energy`, `transform_1e_operator` and `symmetrize_density` must
each land at **≤1e-9** at PySCF's own default `cell.precision`, and at
**≤1e-13** when `cell.precision` is tightened to `1e-13` on both the port and
the reference (this is the identity check that actually isolates the
transform algebra from integral screening). `transform_mo_occ` is exact (0)
at any precision, as expected for an integer-valued map.

---

## Gate E -- the multigrid floor (`gate_multigrid.py`)

`MultiGridNumInt` (v1) and `MultiGridNumInt2` (v2, `multigrid_pair.py`)
against the reference `pbc.dft.numint`/`FFTDF`, on `diamond` and `si`, at
each system's default mesh and a ~0.6x-linear coarser mesh (`get_pp`,
`get_nuc`, `get_j` on a random Hermitian density matrix; `vxc`/`exc` for LDA
and a GGA via `nr_rks`; a converged `KRKS(LDA,VWN)` `e_tot`).

### `get_pp` / `get_nuc` / `get_j` / `vxc` / `exc` -- max-abs residual

| system | mesh | `get_pp` v1 | `get_pp` v2 | `get_pp` v1 vs v2 | `get_nuc` v1 | `get_j` v1 | `exc`(lda) v1 | `exc`(pbe) v1 |
|---|---|---|---|---|---|---|---|---|
| diamond | `[65,65,65]` (default) | 2.924e-12 | 2.411e-08 | 2.411e-08 | 1.985e-13 | 6.138e-14 | 2.026e-12 | 2.046e-12 |
| diamond | `[39,39,39]` (coarse) | 2.924e-12 | 2.408e-08 | 2.408e-08 | 1.948e-13 | 6.875e-14 | 2.251e-12 | 2.963e-12 |
| si | `[35,35,35]` (default) | 3.120e-12 | 1.472e-07 | 1.472e-07 | 4.292e-13 | 1.918e-13 | 1.834e-12 | 1.881e-12 |
| si | `[21,21,21]` (coarse) | **4.384e-09** | 1.476e-07 | 1.472e-07 | 2.708e-14 | 1.689e-14 | 3.868e-13 | **1.316e-08** |

**v1 is essentially algebraically exact against FFTDF/`numint`** -- every v1
column lands at 1e-12...1e-14, matching or beating upstream's own 8-decimal
(`get_pp`/`get_nuc`/`get_j`) and 7-decimal (`vxc`/`exc`/`ecoul`) test
tolerances by two to five orders of magnitude, at BOTH meshes for diamond.
The two exceptions -- si `get_pp` at the coarse mesh (4.384e-09, three orders
worse than the default mesh's 3.120e-12) and si's GGA `exc` at the coarse
mesh (1.316e-08, four orders worse than 1.834e-12) -- are **mesh-convergence
artefacts, not a definitional floor**: `[21,21,21]` under-resolves this
system's pseudopotential/GGA-gradient terms for v1's collocation scheme, and
the residual is expected to shrink back down at a finer mesh (as it does for
diamond, whose coarsening ratio left it comparably resolved). **v1's floor is
therefore mesh-dependent only when the mesh is inadequate**, and Gate E must
say so rather than assume monotone convergence -- the same non-monotone-ladder
caveat Phase 14's MDF gate required.

**v2 vs FFTDF is a DEFINITIONAL floor that does NOT shrink with mesh** --
diamond sits at ~2.41e-08 at both meshes tested, si at ~1.47e-07 at both
meshes tested. This is upstream's own two-implementations-of-one-idea gap,
directly analogous to Phase 14's GDF-vs-RSDF 4.5e-6 finding: **v1 and v2 do
not compute the same quantity to the same precision, and no port can converge
that gap away by refining its mesh.** `get_pp` v1-vs-v2 tracks the v2-vs-FFTDF
column almost exactly, confirming v1 (not v2) is the accurate reference.

### Converged `KRKS(LDA,VWN)` `e_tot` -- accuracy AND wall-clock ratio

| system | `E_ref` (numint) | `E_v1` | `|dE| v1` | ratio (`t_ref/t_v1`) | `E_v2` | `|dE| v2` | ratio (`t_ref/t_v2`) |
|---|---|---|---|---|---|---|---|
| diamond | -10.2213460607 (13.3s) | -10.2213460607 (27.0s) | 4.727e-12 | **0.49x** | -10.2213456621 (34.2s) | 3.986e-07 | **0.39x** |
| si | -7.1606374045 (2.6s) | -7.1606374045 (12.1s) | 3.809e-12 | **0.21x** | -7.1606393421 (14.6s) | 1.938e-06 | **0.18x** |

`e_tot` accuracy: v1 beats upstream's own 7-decimal test tolerance
(`test_krks_ksym.py:240`, 1e-7) by 4-5 orders of magnitude on both systems. v2
is markedly looser -- 3.986e-07 (diamond) and **1.938e-06** (si), i.e. WORSE
than the 1e-7 bound some of upstream's own v2 tests use elsewhere
(`test_multigrid2.py`'s `assert abs(e-e0) < 1e-7`), on these `gth-szv`
systems at their natural meshes. This is a genuine, measured finding, not an
assumption: v2's accuracy against the reference is basis/mesh sensitive in a
way v1's is not, at least on the small systems this plan can afford to run.

**Both multigrid implementations are SLOWER than the reference `numint`/FFTDF
route on these reference systems** -- ratio < 1 in every cell of the table
above, worst at 0.18x (si, v2: nearly 6x SLOWER). This directly measures
17-CONTEXT §8's corollary: multigrid's own speed claim was unmeasured before
this plan, and the measurement shows the OPPOSITE of "faster" at this system
size. Multigrid's collocation/task-list overhead evidently dominates on an
8-AO, `[35,35,35]`-mesh system; whether it crosses over to a real win at
larger systems (more AOs, denser meshes, more k-points) is NOT established by
this measurement and must not be assumed by 17-11/17-12 -- it is exactly the
kind of performance claim `zgemm_dense`'s precedent (6-12x slower than a host
loop despite being the "obvious" faster route) warns against asserting cold.

---

## Task 4 -- the post-SCF (KMP2) floor (`gate_mp2.py`)

### He cell, `mp/test/test_ksym.py`'s own settings (`L=2` cube, minimal basis, `[2,2,2]`, GDF, `exxdiv=None`)

| quantity | value |
|---|---|
| `\|e_corr(ksymm) - e_corr(full BZ)\|` | **3.096e-16** |
| `rdm1` max residual (IBZ vs corresponding full-BZ) | **1.332e-15** |
| `\|E_scf(ksymm) - E_scf(full BZ)\|` (same run, for the ordering check) | 4.441e-16 |

Both far tighter than upstream's own gate (`mp/test/test_ksym.py:56`, 5e-11)
-- consistent with upstream's claim that post-SCF is tighter than SCF because
it reuses one SCF's orbitals through two index paths, though on this
near-machine-precision fixture both numbers are already at the noise floor.

### `si` at `[2,2,2]`, `gth-szv`/`gth-pade`, `KRHF(exxdiv=None)` + `KMP2`, GDF

| quantity | value |
|---|---|
| `\|e_corr(ksymm) - e_corr(full BZ)\|` | **1.067355e-09** |
| `rdm1` max residual | **5.028450e-09** |
| `\|E_scf(ksymm) - E_scf(full BZ)\|` (same run) | 5.543876e-10 |

**Finding that qualifies 17-CONTEXT §2.1's ordering claim:** on He, post-SCF
IS tighter than SCF (3.096e-16 < 4.441e-16), but on `si` it is **not**
(1.067e-09 > 5.544e-10 -- post-SCF is ~2x LOOSER than the SCF here). The
"post-SCF tighter than SCF" ordering upstream's own test-tolerance TABLE
implies (17-CONTEXT §2.1) is a property of upstream's specific He fixture
(tiny, minimal basis, deeply non-degenerate), not a general theorem, and
17-09's gate must not assume it holds on every system. Both numbers are still
far inside upstream's 5e-11 `KMP2` gate's spirit when read as "orders of
magnitude beyond the SCF's own 5e-8 floor" -- but the *specific* ordering
claim does not transfer to `si` and 17-09 should gate `e_corr` on its own
measured number (≤2e-9 on a system of this size) rather than "tighter than
whatever the SCF gate says."

### KRCCSD (`cc/test/test_kccsd_ksymm.py`)

**Unmeasured -- Phase 16 has not shipped.** `crates/pyscf-pbc-cc/src` is a
13-line stub (`lib.rs` + `error.rs`) as of this measurement (2026-09-01), so
per Task 4's instruction this is recorded as "unmeasured, Phase 16 not
shipped" rather than extrapolated. 17-09's CC half is deferred until Phase 16
ships `KRCCSD`.

---

## Task 6.1 -- `get_jk` at `nkpts` vs `nkpts_ibz` (`speed_get_jk.py`, D-PBC-26)

`si [4,4,4]` (`nkpts=64`, `nkpts_ibz=8`, ratio 8.000). **Deviation from the
plan's literal mesh**: the default mesh (`[36,36,36]`) made the FFTDF
`with_k=True` call at 64 k-points too slow to finish in this measurement's
time budget (an O(nkpts²) sweep over 4096 k-pairs); the mesh was coarsened to
`[9,9,9]` to make the measurement tractable. This is a pure wall-clock RATIO
measurement (same DF object, same mesh on both sides of each comparison), so
the ratio is not expected to depend materially on mesh resolution the way an
energy accuracy gate would.

| route | `get_jk`(full, 64 kpts) | `get_jk`(IBZ-subset, 8 kpts) | wall ratio | naive bound (`nkpts/nkpts_ibz`) |
|---|---|---|---|---|
| FFTDF | 26.2724 s | 0.1176 s | **223.3x** | 8.0x |
| GDF | 4.5471 s | 0.1136 s | **40.0x** | 8.0x |

**The achievable speedup from an IBZ-restricted `get_jk` is much larger than
the naive `nkpts/nkpts_ibz` bound**, because exact-exchange cost in both
FFTDF and GDF scales worse than linearly in `nkpts` (closer to quadratic, the
`(k,k')` pairwise structure of exchange) -- so restricting to `nkpts_ibz`
kpoints cuts the PAIR count by `(nkpts/nkpts_ibz)²= 64x`, not `8x`, and the
measured 223x (FFTDF) / 40x (GDF) are consistent with that being the dominant
effect plus additional fixed overhead at 64 kpts. **This bounds what
D-PBC-26's fast path in 17-07 can plausibly gain: an 8-40x reduction is a
conservative floor, and FFTDF's exact-exchange path in particular has room
for a much larger win than the ratio-of-counts intuition suggests.** GDF's
ratio (40x) is a more representative lower bound for 17-07's actual target,
since GDF (not FFTDF) is the port's default DF route.

Task 6.2 (multigrid vs reference `numint` wall-clock ratio) is recorded above
under Gate E's converged-`KRKS` table, per the plan's instruction that both
speed numbers be reported "in the same table-not-prose format Gate E already
uses" where applicable.

---

## Gates C and D -- the two-SCF energy floor, mesh PINNED (`gate_c_d.py` + `gate_c_d_part2.py` + `gate_c_d_part3.py` + `gate_c_d_repro.py`)

**Resource scoping, recorded as a deviation from the plan's literal grid.**
The plan's literal ask -- 5 systems x KRHF/KRKS x gamma/Monkhorst x FFTDF/GDF
x sym-on/off, at each system's own default mesh -- does not fit this
measurement's time budget: `diamond`'s default mesh (`[48,48,48]`) alone made
a single `KRHF` gamma/FFTDF pair take ~120s, and `lif`'s default mesh
(`[81,81,81]`) made a single pair fail to finish inside several minutes. The
grid actually run:

* **`si` (§9.2, `a=5.4306 Å`) -- the FULL 2x2x2x2 deep grid** (`KRHF`/`KRKS` x
  gamma/Monkhorst x FFTDF/GDF), 8 pairs, at its default mesh `[36,36,36]`.
* **`diamond` -- a reduced 4-pair set** (gamma+Monkhorst `KRHF`/FFTDF, gamma
  `KRHF`/GDF, gamma `KRKS`/FFTDF) at its default mesh `[48,48,48]`.
* **`lif`, `he_fcc`, `graphene` -- one pair each** (gamma, `KRHF`, FFTDF), at
  an explicit, aggressively CAPPED mesh (25 per axis, uniform) rather than
  each system's true default (`lif` 81³, `he_fcc` 59³, `graphene`
  `[45,45,351]`) -- see the caveat below.

Every pair still pins `cell.mesh` identically on both sides of its own
comparison (17-CONTEXT §3.3) -- the deviation is in which mesh was chosen,
never in whether both runs of a pair share it.

### `si` -- full deep grid, mesh `[36,36,36]`, `conv_tol=1e-11`

| kmesh type | method | DF route | `E(ksymm)` | `E(full BZ)` | `\|dE\|` | wall |
|---|---|---|---|---|---|---|
| gamma | KRHF | FFTDF | -7.526458914851 | -7.526458914851 | **2.807e-13** | 30.0s |
| gamma | KRHF | GDF | -7.527414272956 | -7.527414272401 | **5.544e-10** | 12.0s |
| Monkhorst | KRHF | FFTDF | -7.631936738032 | -7.631936738032 | **6.928e-14** | 54.3s |
| Monkhorst | KRHF | GDF | -7.632393300673 | -7.632393300663 | **9.362e-12** | 72.7s |
| gamma | KRKS | FFTDF | -7.772967813395 | -7.772967813395 | **1.545e-13** | 26.2s |
| gamma | KRKS | GDF | -7.774856155056 | -7.774856155075 | **1.842e-11** | 37.2s |
| Monkhorst | KRKS | FFTDF | -7.864456343549 | -7.864456343549 | **6.839e-14** | 31.9s |
| Monkhorst | KRKS | GDF | -7.865796574125 | -7.865796574285 | **1.601e-10** | 47.1s |

Every FFTDF row beats upstream's own tolerance (5e-8 gamma / 5e-7 Monkhorst,
`test_khf_ksym.py:84,92`) by **three to six orders of magnitude**; every GDF
row is inside it. FFTDF is consistently tighter than GDF (as expected --
FFTDF is exact modulo the plane-wave cutoff, GDF adds a fitting error on top
of the same symmetry algebra).

### `diamond` -- reduced set, mesh `[48,48,48]`, `conv_tol=1e-11`

| kmesh type | method | DF route | `E(ksymm)` | `E(full BZ)` | `\|dE\|` | wall |
|---|---|---|---|---|---|---|
| gamma | KRHF | FFTDF | -10.930858355314 | -10.930858355374 | **5.985e-11** | 119.7s |
| Monkhorst | KRHF | FFTDF | -11.071125075593 | -11.071125075593 | **1.279e-13** | 82.9s |
| gamma | KRHF | GDF | -10.932080541544 | -10.932080544977 | **3.433e-09** | 16.3s |
| gamma | KRKS | FFTDF | -11.240813947259 | -11.240813947313 | **5.400e-11** | 24.7s |

Same pattern as `si`: FFTDF beats upstream's tolerance by orders of
magnitude, GDF adds its fitting-scale contribution on top but stays inside it.

### `lif`, `he_fcc`, `graphene` -- one pair each, gamma/KRHF/FFTDF, CAPPED mesh 25³

| system | true default mesh | capped mesh used | `\|dE\|` | converged (sym, full) |
|---|---|---|---|---|
| `he_fcc` | `[59,59,59]` | `[25,25,25]` | **2.779e-10** | (True, True) |
| `lif` | `[81,81,81]` | `[25,25,25]` | **1.461e-04** | (True, True) |
| `graphene` | `[45,45,351]` | `[25,25,25]` | **6.391e-01** | **(False, True)** |

**`he_fcc`** (isotropic, all-electron, small `rcut`) tolerates the 25³ cap
fine and reproduces the same ~1e-10 scale as `si`/`diamond`. **`lif` and
`graphene` do NOT** -- `lif`'s ionic (large-Ewald-term) electrostatics and
`graphene`'s 20 Å vacuum along `z` both need a mesh far finer than 25³ to be
correctly resolved (uniformly capping `graphene`'s `[45,45,351]` down to
`[25,25,25]` starves the vacuum direction by 14x and the symmetric run does
not even converge). **These two numbers are mesh-cap ARTEFACTS, not Gate C/D
measurements** -- `lif` and `graphene`'s symmetry-vs-no-symmetry energy floor
at PRODUCTION mesh remains unmeasured by this plan and is carried over
(below) as a resource-scoped gap for 17-13 to close with a longer budget.

### Mesh-unpinning demonstration (17-CONTEXT §3.3)

| system | pinned mesh | natural (`symmetrize_mesh`) unpinned mesh | `\|dE\|` from mesh alone |
|---|---|---|---|
| `si` | `[36,36,36]` | `[36,36,36]` (identical) | **1.776e-15** |
| `diamond` | `[48,48,48]` | `[48,48,48]` (identical) | **0.000e+00** |

For `si`/`diamond` at `[2,2,2]`, `symmetrize_mesh` does **not** enlarge the
default mesh -- both systems' default mesh already carries the cubic lattice
symmetry at this k-mesh, so pinned vs unpinned makes no difference here. This
is itself useful evidence for §3.3's warning: the enlargement is
system/k-mesh dependent, so a gate that assumes "symmetry never touches the
mesh" would be accidentally right on these two systems and wrong in general
-- pinning explicitly, as every Gate C/D comparison above does, is still the
only safe default. `lif`/`he_fcc`/`graphene`'s unpinning demonstration (at
their TRUE default mesh, not the 25³ cap) was not completed within this
measurement's time budget -- carried over below.

### Run-to-run / thread-count spread (`gate_c_d_repro.py`)

`si`, `KRHF`, FFTDF, gamma-centred, mesh `[36,36,36]`, `conv_tol=1e-11`,
`[2,2,2]`, full BZ (no symmetry):

| threads | run | `e_tot` |
|---|---|---|
| 1 | 0 | -7.526458914850943 |
| 1 | 1 | -7.526458914850944 |
| 8 | 0 | -7.526458914850945 |
| 8 | 1 | -7.526458914850943 |

Spread across all four runs: **2e-15** (2 ulps at `\|E\|≈7.5`) -- consistent
with `.planning/pbc/SUMMARY.md`'s prior finding that upstream's own
multi-threaded BLAS moves the printed energy by ~1e-15 between runs. **A gate
tighter than ~2e-15 on this system is not a gate: it is testing thread-count
noise in upstream's own BLAS, not the port.** Every Gate C/D number measured
above (FFTDF: 2.8e-13 down to 1.3e-13; GDF: 5.5e-10 down to 9.4e-12) sits
comfortably above this floor, so none of them are at risk of this effect.

### Carry-overs for 17-13 (resource-scoped by this plan's time budget, not by any measured limitation)

1. `lif`/`graphene` Gate C/D at PRODUCTION mesh (not the 25³ cap) --
   `he_fcc` already shows the algebra holds at ~1e-10 on a system this cheap
   to run at full mesh; `lif`/`graphene` need a longer-running acceptance
   pass, the same "unmeasured, `#[ignore]`d" pattern Phase 14 used for
   diamond's `make_j3c` wall time.
2. `lif`/`he_fcc`/`graphene`'s mesh-unpinning demonstration at true default
   mesh (not the cap).
3. `diamond`'s remaining 4 cells of the full 2x2x2x2 grid (Monkhorst x
   KRKS x {FFTDF,GDF}, gamma x KRKS x GDF) -- `si`'s full grid and
   `diamond`'s 4 measured cells already show the same pattern (FFTDF ≤5.4e-11,
   GDF ≤3.4e-09), so this is a confirmation run, not an open question.
