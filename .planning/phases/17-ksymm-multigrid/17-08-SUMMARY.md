# 17-08 — ksymm DFT — SUMMARY (IN PROGRESS)

**Status: PARTIAL.** Task 1 (`numint` takes an IBZ k-set) is landed, with 2 of
its 3 gate tests green and the third pending a release build. Tasks 2-5 are NOT
done. Written incrementally because this environment restarts every ~20-40
minutes.

## Landed — Task 1: `numint` learns the symmetric k-set

**Read `17-08-FINDING-numint.md` first — this plan's Task 1 premise was wrong,
and the correction changed what was built.**

### D-17-08-01, in one paragraph

17-08-PLAN.md Task 1 asserted that all seven `isinstance(kpts, KPoints)` sites
in `pbc/dft/numint.py` "evaluate the density at the IBZ points, then
**symmetrize the real-space density** through `kpts.symmetrize_density`".
Verified against vendored PySCF 2.12.1: **none of them do.** Five
(`:328, :431, :859, :908, :956`) unfold to the **full BZ** via
`kpts.transform_dm(dms)` and then run the ordinary full-BZ path; two
(`:647` `nr_rks_fxc`, `:779` `nr_uks_fxc`) take `kpts.kpts_ibz` directly; and
`symmetrize_density` has **no caller in `pyscf/pbc/` outside its own unit
test**. Full evidence, including how the error was caught, is in
`17-08-FINDING-numint.md`.

### What shipped, faithful to upstream

`crates/pyscf-pbc-dft/src/numint.rs`:

* **`pub enum KSet { Full, Ibz(Box<KPoints>) }`** on `KNumInt` — the real
  branch upstream takes seven times, hoisted into one place. `KNumInt::new`
  is unchanged and defaults to `Full`.
* **`KNumInt::with_symmetry(&KPoints)`** — `kpts` is the **full BZ**, matching
  Group A's `kpts = kpts.kpts`.
* **`KNumInt::unfold_dms(cell, dms, nao)`** — Group A's
  `dms = kpts.transform_dm(dms)`, delegating to 17-05's `transform_dm` (gated
  at 1e-12). A **bit-exact** no-op on already-full-BZ input, matching
  upstream's `if kpts.kpts.size > 3` guard; a needless round trip would
  perturb the density.
* **`KNumInt::kpts_ibz()`** — Group B's `kpts = kpts.kpts_ibz`, falling back
  to the full set under `Full` so those callers need no branch.

The `Full` path's bit-identity is now stronger than the plan asked: both arms
reach the **same** code, because `Ibz` unfolds and then joins it.

### Removed, deliberately

`KNumInt::symmetrize_rho` and its Becke-grid refusal were written to the
plan's wording before the premise was checked. They are **correct code for a
behaviour upstream does not have**, so they were removed rather than left as
an unreachable path — the capability already lives where it belongs, and is
already tested, as `KPoints::symmetrize_density` (17-05 Task 4). Keeping a
second unreachable copy in the DFT crate would be the "container with no
caller" 15-CONTEXT §1.1 ruled against.

### The cost consequence, stated

This removes the only place 17-08 was going to make DFT cheaper. Under
symmetry `numint` does the full-BZ work **plus** an unfold — a convenience
interface, not an optimisation. The IBZ saving in a ksymm DFT run comes
entirely from the SCF side (17-07's `get_jk` route, D-PBC-26).

That belongs next to this phase's other measured speed corrections: upstream's
own multigrid measured **slower** than its reference `numint` (0.18-0.49x,
17-01), and 17-05's predicted star-search parallelism measured **0.99x**.
Three speed assumptions, all wrong in the same direction.

## Verified

`cargo test -p pyscf-pbc-dft --release --test krks_ksymm -- --test-threads=1`
— **3/3 green**, 201 s.

| test | measured |
|---|---|
| `unfolded_ibz_density_equals_full_bz_density` | max &#124;rho_full − rho_unfolded&#124; = **1.054e-13** (tol 1e-11) |
| `unfold_is_a_bit_exact_no_op_on_full_bz_input` | bit-exact |
| `full_bz_path_is_untouched` | bit-exact |

### The floor showed up a third time, and the prediction held

The Group-A identity first measured **1.807e-10** on `si()` at its DEFAULT
`cell.precision = 1e-8` with `conv_tol_grad = 1e-8` — over the 1e-10 tolerance
the test shipped with. Rather than relax it, the fixture was tightened to
`si_precision(1e-10)` / `conv_tol_grad = 1e-10`, exactly as
`17-04-MEASUREMENT.md` prescribes:

| fixture | max &#124;rho_full − rho_unfolded&#124; |
|---|---|
| `precision 1e-8`, `grad 1e-8` (default) | 1.807e-10 |
| **`precision 1e-10`, `grad 1e-10`** | **1.054e-13** |

A factor of **1714** for a 100x tightening of the inputs. That is a floor, not
a defect: the residual rides on `transform_dm`, whose accuracy 17-05 measured
as a joint function of the same two knobs (1.784e-11 at 1e-10/1e-10, 2.306e-13
at 1e-12/1e-12). The tolerance was then set to **1e-11** — tighter than the
value that originally failed — and it passes with ~100x margin.

This is the **third** appearance of this exact joint floor in the phase
(17-04's Fock block-diagonality, 17-07's `eig` comparison, now this), and the
first where it was predicted before being measured rather than diagnosed after.

## The seven call sites are now WIRED

All seven of upstream's `isinstance(kpts, KPoints)` branches are implemented at
their Rust counterparts, each carrying its upstream line reference:

| upstream | Rust site | group | what it does |
|---|---|---|---|
| `numint.py:328-331` | `nr_rks` | A | `unfold_kdms` |
| `:431-435` | `nr_uks` | A | `unfold_kdms`, per spin channel |
| `:956-959` | `get_rho` | A | `unfold_dms` |
| `:908-911` | `cache_xc_kernel1` | A | `unfold_dms` per channel |
| `:859-863` | `cache_xc_kernel` | A | `unfold_mos` — the one site that unfolds the **orbitals** (`transform_mo_coeff`/`transform_mo_occ`), not the density |
| `:647-649` | `nr_rks_fxc` | B | `kpts_ibz()` directly, no unfold |
| `:779-781` | `nr_uks_fxc` | B | `kpts_ibz()` directly, no unfold |

Two details worth recording:

* **`cache_xc_kernel` unfolds the orbitals, not the density.** Building the DM
  and then unfolding it is mathematically equivalent, but upstream transforms
  `mo_coeff`/`mo_occ` at `:859-863`, so this port does too (RULE 2) — a reader
  diffing the two files finds the same call. The resulting `dms` are then
  full-BZ length, so `cache_xc_kernel1`'s own unfold correctly becomes a no-op.
* **The Group-B functions no longer reference `self.kpts` at all.** They bind
  `kset = self.kpts_ibz()` and `nk = kset.len()` once and use those for the AO
  evaluation, the block ranges and the `vmat` allocation — otherwise the
  potential would be allocated over the full BZ while the AOs were evaluated
  over the IBZ.

### Regression: the `Full` path did not move

`numint.rs` is the file the whole DFT stack sits on, so the plan requires the
full-BZ path stay bit-identical. Every pre-existing target that exercises it
passes unchanged:

| target | result |
|---|---|
| `numint_blocking` | **3 passed** |
| `numint_threads` | **1 passed** |
| `modules` | **8 passed** |
| `krks_ksymm` (fast subset) | **2 passed** |

(There is no `--test numint` target; the file is covered by `numint_blocking`,
`numint_threads` and the suites above.)

## Task 2 — `KsymAdaptedKrks` — DONE and gated

`crates/pyscf-pbc-dft/src/krks_ksymm.rs` (new), wired into `lib.rs`.

**The line that makes the shapes work** is upstream's `krks_ksymm.py:41-42`:
`kpts_band = kpts.kpts_ibz` when no band k-points are given. `nr_rks` then
evaluates `rho` over the whole zone (the Group-A unfold) but builds the
potential **at the band k-points** — the IBZ set — so both the XC and the J/K
halves return `nkpts_ibz` matrices and the `KOverrideHooks` contract is met
with nothing folded by hand.

Upstream gets `eig` and `get_occ` by inheriting `khf_ksymm.KRHF`
(`class KsymAdaptedKRKS(krks.KRKS, khf_ksymm.KRHF)`). This port has no
inheritance, so 17-07's `eig_symm_adapted` and `ksymm_get_occ_restricted` were
made `pub` and are **shared, not copied** — `KsymAdaptedKrhf` now routes its
own `get_occ` through the same helper.

The module doc carries the full `weights_ibz`-vs-`1/nkpts` table (17-CONTEXT
§3.5), and `weighted_trace` is the single ordered (D-PBC-17) contraction all
of `ecoul`, the hybrid `exc` correction and `energy_elec`'s `e1` go through.

### Gate C for DFT — port vs port

`si` `[2,2,2]`, `lda,vwn`, mesh pinned on both sides, tight fixture:

| branch | `e_full` | `e_ibz` | &#124;dE&#124; |
|---|---|---|---|
| `use_ao_symmetry = false` | -7.772967811717 | -7.772967811717 | **3.109e-14** |
| `use_ao_symmetry = true` | -7.772967811717 | -7.772967811717 | **2.842e-14** |

Essentially machine precision, and **both** branches ship — the plain one
exists so any 17-04 defect stays bisectable.

### The weighting is actually observable on this fixture

A separate test asserts `si [2,2,2]`'s stars have **unequal** sizes —
measured `[1, 3, 4]`, giving `weights_ibz = [0.125, 0.375, 0.5]` against a
uniform `1/nkpts = 0.125`. Without that check, Gate C could pass while
`weights_ibz` had been mistakenly written as `1/nkpts`, because the two
coincide whenever every star has the same size. This is the guard 15-CONTEXT
§3's KMP2 trap earned.

## Task 4 — DFT+U over the IBZ — DONE and gated

**D-17-08-02: the plan's premise was wrong here too** (second one in this
plan). It said the local projectors `C_ao_lo` "must be rotated with the space
group when the density is unfolded". They are not, and must not be: upstream's
entire ksymm DFT+U is `krks_ksymm.get_veff` followed by the SHARED
`krkspu._add_Vhubbard`, whose only symmetry-aware lines are
`kpts = kpts.kpts_ibz` (`krkspu.py:77`) and `weight = weights_ibz` (`:93`).
`_make_minao_lo` is then called with the IBZ k-points, so the projectors are
built **directly where they are used** and nothing is unfolded. Full evidence
in `17-08-FINDING-numint.md`.

### Shipped

* `kspu.rs`: `add_vhubbard_weighted(..., weights: &[f64])`, with the existing
  `add_vhubbard` delegating to it using a uniform `1.0 / nkpts` — **bit-exactly
  the pre-17-08 behaviour**, since that is the same scalar it used to inline.
* `krks_ksymm.rs`: `KsymAdaptedKrkspu` = `KsymAdaptedKrks` + that call with
  `kpts_ibz` and `weights_ibz`.

### Gate: `E_U` over the IBZ vs the full BZ

| quantity | value |
|---|---|
| `E_U` full BZ (uniform `1/nkpts`) | 0.026457839749303 |
| `E_U` IBZ (`weights_ibz`) | 0.026457839749303 |
| **&#124;dE_U&#124;** | **6.939e-18** |

**No SCF is involved, deliberately.** The density needs to be *symmetric*, not
*converged*: an arbitrary Hermitian IBZ density is pushed through
`transform_dm`, producing a full-BZ density symmetric **by construction**. Both
sides then see identical physics, the residual is the weighting alone, and the
test runs in 0.29 s instead of minutes. A random density would not do — it is
not related across stars, so the two sums would legitimately differ.

### Two fixture constraints, both found by tests refusing to pass vacuously

1. A `gth` pseudopotential cell gives *"DFT+U: the local-orbital metric is
   singular"* against the MINAO reference — the cell must be all-electron.
2. `E_U = (U/2)(Tr P − Tr P²)` **vanishes on a filled shell.** The first
   working version used a converged He 1s density and produced
   `E_U = −2.04e-17`: the two weightings agreed to 2e-17, but only because both
   were zero. An `assert!(e_u.abs() > 1e-6)` guard caught it, and the fixture
   now uses a deliberately FRACTIONAL occupancy where `P` lies strictly
   between 0 and 1.

That guard is the reusable lesson: a symmetry gate whose quantity is
identically zero passes for the wrong reason.

### A pre-existing asymmetry, recorded not fixed

`Krkspu` (Phase 12) has **no `KOverrideHooks` impl**, so the plain DFT+U cannot
be driven by `kscf::kernel` at all — independently recorded as U-08 in
`KUKS-OPTIMISATION-PLAN.md`. `KsymAdaptedKrkspu` *does* implement the hooks, so
the k-symmetric DFT+U is SCF-drivable while the plain one is not. This plan
does not close U-08; it is noted so the asymmetry is not mistaken for a
Phase-17 regression, and it is why the gate above compares `E_U` directly
rather than two SCF energies.

## Task 3 — `KsymAdaptedKuks` — SHIPPED, with its end-to-end gate blocked on a fixture

`krks_ksymm.rs` gains `KsymAdaptedKuks`: `nset() == 2`, two Fermi levels each
over the unfolded BZ, `nr_uks` at `kpts_band = kpts_ibz`, and a
`weighted_trace_uks` carrying `weights_ibz`.

Upstream declares `class KsymAdaptedKUKS(kuks.KUKS, kuhf_ksymm.KUHF)`. This
port has neither inheritance nor a `KsymAdaptedKuhf` (17-07 Task 5 never
shipped one), so the k-symmetry half comes from two **shared** helpers rather
than copies: `eig_symm_adapted` and a new
`ksymm_get_occ_unrestricted` (added to `khf_ksymm.rs` alongside its restricted
sibling). When `KsymAdaptedKuhf` lands it should take over `get_occ`/`eig`
here.

One deliberate duplication: `get_veff_tagged` mirrors
`Kuks::veff_from_parts` rather than calling it, because that function derives
`nkpts` from `with_df.kpts().len()` and forms `weight = 1.0 / nkpts`. Here the
DF is over the FULL BZ while the weights are `weights_ibz`, so the two come
apart and the shared body cannot serve both. The reason is recorded at the
function.

`KROKS`/`KGKS` have no upstream `*_ksymm` module and were not invented.

### D-17-08-03 — a Gate C precondition nobody had stated

The KUKS energy gate is written and **`#[ignore]`d**, because running it
surfaced a methodology trap that applies to Gate C generally:

> **An IBZ-vs-full-BZ energy comparison is only valid if the FULL-BZ solution
> is itself symmetric.**

An IBZ run is *constrained* to symmetric occupations — `get_occ` folds through
`check_mo_occ_symmetry`, which raises otherwise. An unconstrained full-BZ run
is under no such constraint. Measured on the open-shell fixture:

```text
max |dm_a - dm_b| = 1.194                    (RULE U satisfied: genuinely open-shell)
full-BZ occupations star-symmetric?  alpha = true, beta = FALSE
e_full = -2.679270749095, e_ibz = -2.724600963472, |dE| = 4.533e-02
```

The full-BZ beta channel is **symmetry-broken**, and the IBZ energy is *lower* —
the constrained run found a different, better state. The two SCFs are solving
different problems, so 4.5e-2 Ha is physical, **not a KUKS defect**. Relaxing
the tolerance to 5e-2 would have buried a real physical distinction under a
number that merely looks like a tolerance.

The closed-shell gates never hit this (a restricted solution on these cells is
symmetric — hence their 3e-14 agreement), which is why the precondition had
gone unnoticed through Tasks 1, 2 and 4.

The test now **asserts the precondition explicitly**, so it can never be
invalidated silently: given a fixture whose full-BZ solution is star-symmetric
in both channels, it either passes or fails for the right reason. Finding such
a fixture is the carry-over.

What IS verified, and passes, is `kuks_ibz_runs_and_stays_symmetric`: the
adapter drives the shared SCF loop over an IBZ k-set with two genuine spin
channels (`max |dm_a - dm_b| = 1.19` at convergence), IBZ-length outputs, and a
full-BZ electron count.

## `kukspu_ksymm` — DONE

`KsymAdaptedKukspu` (`krks_ksymm.rs`), the unrestricted twin of
`KsymAdaptedKrkspu`. Upstream's file has the same shape —
`kuks_ksymm.get_veff` + the SHARED `kukspu._add_Vhubbard`, whose symmetry
handling is again exactly `kpts = kpts.kpts_ibz` (`kukspu.py:59`) and
`weight = weights_ibz` (`:78`). **D-17-08-02 applies unchanged**: no projector
rotation in the two-channel case either.

Upstream detail worth naming: `kukspu.py:68-70` applies **the same `C_ao_lo`
to both spins**. The projectors are spin-independent, which is why
`add_vhubbard_weighted` needed no spin-aware change — it already loops over
whatever density channels it is handed, and its `vxc[spin]` write-back was
already general.

## Task 5 — the gates, per DF route

17-01 measured Gate C/D **separately per DF route**, and the floors differ by
orders of magnitude on the same systems:

| route | 17-01's measured Gate C/D residuals |
|---|---|
| FFTDF | 6.9e-14 … 2.8e-13 |
| GDF | 9.4e-12 … 3.4e-09 |

So the two routes get different tolerances, exactly as the plan requires.

| gate | route | state |
|---|---|---|
| Gate C, KRKS | FFTDF | **PASS — 3.109e-14 / 2.842e-14** (both `use_ao_symmetry` branches) |
| Gate C, KRKS | GDF | **RUN — FAILS at 1.432e-06** (tol 1e-8). See below; not a relaxed tolerance. |
| Gate C, KUKS | FFTDF | written, `#[ignore]`d on the D-17-08-03 precondition |
| DFT+U (`E_U` weighting) | — | **PASS — 6.939e-18** |
| `KSet::Full` bit-identity | — | **PASS** (plus `numint_blocking` 3/3, `numint_threads` 1/1, `modules` 8/8) |
| `use_ao_symmetry` both ways | FFTDF | **PASS** — covered by Gate C |

### An erratum against 17-08-PLAN.md Task 5

The plan notes: *"upstream gates DFT on GDF **tighter** than on FFTDF — 8
decimals vs 7."* That is a statement about upstream's chosen **test
tolerances**, and it points the opposite way from the **measured floors**:
17-01 measured GDF as the *looser* route by roughly three orders of magnitude.
The tolerances here follow the measurement, and the discrepancy is recorded at
`GDF_E_TOL` so a later reader does not "correct" it back.

### The GDF gate FAILS — recorded, not absorbed

It is `#[ignore]`d on cost (two full GDF SCFs, 1381 s), but it **has been run**
and the result is a genuine failure:

```text
e_full = -7.774590218592
e_ibz  = -7.774588786147
|dE|   = 1.432444577176e-06          (tolerance 1e-8)
```

That is ~3 orders above GDF's own measured floor (9.4e-12 ... 3.4e-09, 17-01)
and ~8 orders worse than FFTDF on the **identical** comparison (3.109e-14).
**The tolerance was not relaxed to absorb it**, and the `#[ignore]` should not
be read as "unverified" — the number is recorded at the test and here.

**The first hypothesis was WRONG. The GDF band route is EXONERATED.**

The hypothesis was that GDF's `kpts_band` route — which rebuilds `_cderi`
(`df_jk.py:86-92`, plan 17-10 Task 4) — loses accuracy relative to the direct
route. `gdf_band_route_matches_the_direct_route` tested it directly, with the
ksymm layer removed: one GDF object, one converged density, direct versus
`kpts_band = Some(kpts_ibz)`, compared at the IBZ points.

```text
max |dvj| = 0e0,  max |dvk| = 0e0        (EXACTLY zero, 433 s)
```

Bit-identical. **The band route must not be "fixed"** — 17-10's Task 4 work is
correct on this evidence, and the 1.432e-06 belongs to the ksymm layer, i.e.
to this plan.

### The diagnostic itself had a bug, and it would have concluded the opposite

The first version passed `kpts_band = Some(kpts_abs)` — the **full** sampling
set. `band_is_kpts` (`df_jk.rs:32-37`) returns `true` when the band set equals
the sampling set, so `get_jk` short-circuits to the **direct** path. That
version compared the direct route with itself: it would have passed trivially
and been read as "the band route is fine", which is the right conclusion
reached for an entirely wrong reason — and it would have been indistinguishable
from the real result until someone re-derived it.

`kpts_band` must be a **strict subset** for the band route to be exercised at
all. The corrected version uses `kpts_ibz` and maps the output back through
`ibz2bz`. The invariant is sound either way: the density sum runs over ALL
sampling k-points regardless, so restricting the OUTPUT set changes nothing
physical.

### Current leading hypothesis (untested)

GDF's fit is not exactly invariant under the space group — `_cderi` is built on
a k-set with no symmetry adaptation — so the unconstrained full-BZ GDF solution
can be slightly symmetry-broken while the IBZ run is constrained to the
symmetric one. **That is the same class as D-17-08-03**, and it is GDF-specific
precisely because FFTDF's evaluation is analytic. The check is the one
D-17-08-03 already used: run `check_mo_occ_symmetry` on the full-BZ GDF
solution.

A secondary contributor: this fixture runs at DEFAULT precision, unlike the
FFTDF gate, so part of the gap may be the ordinary joint precision floor.
Both are cheap to test and neither is a reason to touch `build_band_gdf`.

### Ruled out mechanically

`build_band_gdf` copies every numerically relevant `Gdf` field (`auxbasis`,
`exp_to_discard`, `aosym`, `prefer_ccdf`, `j2c_eig_always`, `exclude_dd_block`,
`rs_rcut`, `rs_mesh`, and `j_only` conditionally). The only uncopied field is
`cderi_to_save`, a filesystem path with no numerical effect. So this was never
a dropped-setting bug.

Its fixture deliberately uses **default** precision, unlike the FFTDF gate:
GDF's floor sits far above the precision-limited regime, so `si_precision(1e-10)`
would buy no accuracy and cost a great deal of time. The tight-fixture
reasoning from `17-04-MEASUREMENT.md` binds FFTDF, not GDF.

### Gate D (port vs upstream) — not attempted here

Gate D needs the `PYSCF_ORACLE_VENV` harness. 17-01 already measured
**upstream's own** side per route (its README table carries e.g. gamma/KRKS/FFTDF
`-7.772967813395` and gamma/KRKS/GDF `-7.774856155056`), so the remaining work
is running this port against those numbers with the k-mesh convention and mesh
pinned to match — not re-deriving the reference. Carried over to 17-13, which
owns the phase's oracle pass.

## NOT done — carried over

1. **Task 2** — `krks_ksymm`'s `get_veff` / `energy_elec` (`weights_ibz`, with
   the weight table in a module doc as 17-CONTEXT §3.5 requires).
2. **Task 3** — `kuks_ksymm`. **Blocked on 17-07 Task 5**, which has not
   shipped the `KUHF` ksymm adapter this would sit on. `KROKS`/`KGKS` have no
   upstream `*_ksymm` module — do not invent them.
3. **Task 4** — `krkspu_ksymm` / `kukspu_ksymm` over `kspu.rs`, including the
   oracle-free check that the occupation matrix `n_IJ` is invariant under the
   little co-group at each site. The point of these is that the local
   projectors `C_ao_lo` must rotate with the space group; forgetting to
   rotate them applies `U` to the wrong orbital at every symmetry-related
   site, which is a large and plausible-looking energy error.
4. **Task 5** — Gates C and D per DF route, mesh pinned, at 17-01's floors;
   `test_krks_symorb`; the DFT+U gates; and the speed report.

## RSH is blocked, and it is a Phase-14 carry-over, not a Phase-17 gap

17-08-PLAN.md Task 2 asks whether plan 14-07's `range_coulomb` work closed the
`omega` refusal. **It did not.**
`crates/pyscf-pbc-df/src/gdf/jk.rs:674` still returns
`NotYetImplemented { phase: 14, what: "GDF.get_jk(omega) — the range-separated
kernel needs GDF.range_coulomb (df.py:515-553), plan 14-07" }`.

So the hybrid/RSH rows of upstream's `test_krks_ksym.py:143-157` (`rsh`) and
`:197-214` (`rsh` on GDF) **cannot be gated by this plan**. Per the plan's own
instruction this ships the LDA/GGA gates and records RSH as blocked on the
Phase-14 carry-over with the file:line, rather than working around it.

## On the plan's "`git diff crates/pyscf-pbc-df/src/` is empty" check

Stated precisely, because the raw command is misleading at this point in the
phase: **this plan touched no file under `crates/pyscf-pbc-df/`.** Its only
production edit is `crates/pyscf-pbc-dft/src/numint.rs`, plus the new test
file. The DF layer still knows nothing about symmetry, which is what the check
is actually asserting.

The raw `git diff crates/pyscf-pbc-df/src/` is NOT empty — it currently shows
10 files and ~1000 insertions — but every one of those belongs to **plan
17-10** (`ft_ao/rs_cell.rs`, `ft_ao/supmol.rs`, `gdf_builder/dd_block.rs`, the
band-k-point closures), which is an independent track that happens to share
the working tree. Read against 17-08's own changes the check passes; read
literally against an unclean tree it cannot, and a future reader should not
mistake 17-10's diff for a D-PBC-15 violation here.

## Environment note

Four consecutive agent sessions on 17-07 were killed mid-reading by the
restart cadence, so 17-07's code and this plan's were written directly by the
orchestrator session, which survives restarts. `17-07-BLUEPRINT.md` holds the
reconnaissance for anyone continuing.
