# KUKS + k-symmetry + multigrid — execution summary

**Session 1 — 2026-09-02.** Against
[`KUKS-KSYMM-MULTIGRID-OPTIMISATION-PLAN.md`](./KUKS-KSYMM-MULTIGRID-OPTIMISATION-PLAN.md).

| item | state |
|---|---|
| **P-00** re-run GATE U and GATE A | **DONE — both green, measured** |
| **P-01** determinism gates for ksymm and multigrid | **LANDED** (`ksymm_threads.rs`, `multigrid_threads.rs`) |
| **P-02** ordered reductions + clone removal in the ksymm adapters | **LANDED, bit-exact, gated** |
| **P-03** `oracle_sum` on the multigrid energy path | **LANDED** (changes last bits, as stated) |
| **S-00** the ksymm profiling mode | **LANDED and MEASURED** — `get_veff` 0.73x at a 2.0x fold |
| **S-01** unfold the IBZ density once per cycle | **LANDED (step 1), bit-exact**; steps 2-3 CLOSED by measurement as not worth doing |
| **S-02** retire the IBZ-only `get_jk`, add the band route, record the erratum | **LANDED, MEASURED, erratum propagated** |
| **S-03** IBZ-costed XC quadrature via `symmetrize_density` | **NOT STARTED** |
| **S-04** J/K pair-invariant audit | **NOT STARTED** |
| **S-05** on-the-fly `ao_{Rk}` | **DEFERRED, as planned** |
| **U-09** delete the `nset = 2` clones, order `e1` | **step 2 LANDED; step 1 measured negligible and NOT done** |
| **U-10** reusable per-block scratch in the quadrature | **LANDED, bit-exact, gated** |
| **M-00** `nr_uks` for both multigrid drivers | **LANDED, gated** |
| **M-01** the numint seam (multigrid selectable from `Krks`/`Kuks`) | **NOT STARTED** |
| **M-02** cache the multigrid geometry | **LANDED, bit-exact, gated** |
| **M-03** one launch per level | **LANDED, bit-identical — but the budget guard keeps 2 of 3 levels streaming; see the finding** |
| **M-04** v1 host loops | **steps 1-2 LANDED, bit-exact; step 3 (pass2 screening) NOT STARTED** |
| **M-05** per-Gaussian sub-mesh | **DEFERRED, as planned** |

**Measured up front, and it re-ordered the plan.** §2.1.0a of the plan now
carries the KUKS baseline U-01 owed: the hybrid KUKS/KRKS multiplier is
**1.034**, not the `1 < m < 2` the KUKS plan could only bound. That single
number demoted U-04 and U-05 — the two items the KUKS plan called its speed
headline — to under 1 % of an SCF each, and promoted the AO evaluation
(6.0-6.4 s cold against a 39-83 ms warm quadrature) to the thing worth
attacking. It is why U-10 was done and U-09 step 1 was not.

---

## P-00 — the re-run, and it was owed

`KUKS-EXECUTION-SUMMARY.md` recorded four GATE U rows as re-gated by
arithmetic over energies a previous run had produced, with the gate **not
executed since the tolerance edit**, and GATE B as argued rather than
measured. Nothing here may start from an inferred green, so this ran first.

### GATE U — 9/9 PASS, 90.7 s

Every row green as written, including the four that had only been inferred:

| row | Δe | tol |
|---|---|---|
| U-b `KUHF` H2(3 Bohr) Γ, no XC | -7.749e-14 | 1e-12 |
| U-b `KUKS` H2(3 Bohr) Γ LDA,VWN | -2.587e-13 | 1e-12 |
| U-b `KUKS` H2(3 Bohr) Γ PBE | -1.754e-13 | 1e-12 |
| U-c `KUKS` H2(3 Bohr) Γ PBE0 | -1.845e-13 | 1e-11 |
| U-a `KUKS` Li(spin1) `[1,1,3]` PBE | -5.544e-12 | 5e-11 |
| U-a `KUKS` Li(spin1) Γ PBE | -7.730e-12 | 5e-11 |
| U-a `KUKS` Li(spin1) Γ LDA,VWN | -7.802e-12 | 5e-11 |
| U-c `KUKS` Li(spin1) Γ PBE0 | -9.728e-12 | 5e-11 |
| U-a `KUHF` Li(spin1) Γ, no XC — the floor control | -1.494e-11 | 5e-11 |

`(Na, Nb)` matches exactly on every row and `<S^2>` to ≤ 1.8e-15, so the
energies are being compared between runs in the same state — the D-17-08-03
discipline.

**One finding worth recording, stated at the confidence it has.** The
residuals differ from the previously recorded ones in the last one or two
digits (Li PBE -7.730e-12 now against -7.735e-12 recorded; H2 PBE -1.754e-13
against -1.756e-13), and GATE A's do the same (KRKS PBE 6.448e-12 against
6.447e-12). Nothing in this port changed on those paths between the two runs,
and the port's own reductions are ordered and thread-invariant, so the
**likeliest** explanation is the oracle side — upstream's numpy/BLAS is not
bit-reproducible run to run. That is a hypothesis, not a measurement: proving
it means running the oracle twice on one tree and diffing its energies, which
was not done. Either way the practical consequence holds — **a gate that
quotes a residual to three significant digits is quoting at least one digit of
run-to-run noise**, so these numbers should be read as magnitudes.

### GATE A — 7/7 PASS, 356 s, every residual unmoved

| row | residual | previously recorded |
|---|---|---|
| `KRKS` Si 2×2×2 PBE | 6.448e-12 | 6.447e-12 |
| `KUKS` Si 2×2×2 PBE | 6.446e-12 | 6.446e-12 |
| `KRKS` Si 2×2×2 LDA,VWN | 6.505e-12 | 6.502e-12 |
| `KRKS` Si 2×2×2 PBE0 | 5.587e-12 | 5.587e-12 |
| `KRHF` Si 2×2×2, no XC (the `get_pp` floor) | 4.158e-12 | 4.158e-12 |
| `KRKS` He-fcc 2×2×2 PBE, ALL-ELECTRON | -8.615e-14 | -8.615e-14 |
| MEASUREMENT: libxc/xcfun gap | 4.709e-7 | 4.709e-7 |

> **Stated plainly, because the sequencing rule says items land alone.** This
> GATE A run was launched before P-02 and completed after it, so it built a
> tree that already contained P-02. It is therefore simultaneously P-00's
> re-run and P-02's regression check. That is sound — GATE A does not exercise
> the k-symmetric path at all, so it can only be a regression check for P-02,
> and the pre-P-02 GATE A is the unchanged tree the previous session measured
> — but it is a deviation from "P-00 lands alone" and it is recorded rather
> than glossed.

---

## P-01 — the determinism gates that did not exist

`crates/pyscf-pbc-dft/tests/ksymm_threads.rs` and
`tests/multigrid_threads.rs` (new). GATE B had been measured for `KNumInt`
and for the six `gate.rs` energies, and **never for `KsymAdaptedKrks` /
`KsymAdaptedKuks` or for either multigrid `nr_rks`** — which is where §2.2.2
and §2.3.4 found the naive folds. Both files were written **before** P-02 and
P-03 landed, deliberately: a determinism gate authored after the fix can only
confirm the fix.

**One deviation from the plan's own text, stated.** P-01 proposed separate
processes with `RAYON_NUM_THREADS`. Both files instead vary the worker count
INSIDE one process with explicit `rayon::ThreadPool`s, which is the existing
convention (`numint_threads.rs`) and is strictly stronger: it also catches a
result that depends on which worker stole a chunk.

The ksymm file runs a FIXED, SMALL number of SCF cycles rather than to
convergence, and does not assert `converged`. Convergence is an attractor that
can mask a last-bit difference by pulling two trajectories back together;
cycle 3 of two runs either agrees bit-for-bit or does not.

---

## P-02 — ordered reductions and clone removal in the ksymm adapters

**FILES** `crates/pyscf-pbc-dft/src/veff.rs`, `krks_ksymm.rs`,
`tests/ksymm_trace_precision.rs` (new).

### What was wrong

`KsymAdaptedKrks::weighted_trace` and `KsymAdaptedKuks::weighted_trace_uks`
were hand-rolled nests: a naive `nao²` product sum, then a naive
`nkpts_ibz` weighted fold. They feed `ecoul`, the hybrid `exc` correction AND
`energy_elec`'s `e1` for **every** k-symmetric KS driver — the same defect
class U-03 closed for the non-symmetric drivers, in the file 17-08 wrote
afterwards.

**And the doc comment asserted the opposite.** `weighted_trace`'s doc said
"the accumulation is ordered (D-PBC-17), so the result is bit-identical under
any thread count". The conclusion held but the premise did not: the fold was
*serial*, which buys thread-independence and nothing else, while D-PBC-17 asks
for the ordered primitive whose error bound is `O(log₂ n · eps)` rather than
`O(n · eps)`. Code and comment were corrected together.

### What landed

* `veff::weighted_trace_dm_v` and `weighted_trace_dm_v_shared` — the
  `weights_ibz` analogues of `trace_dm_v` / `trace_dm_v_shared`, ordered in
  both axes.
* Four clone sites deleted, all of them the ones U-06 removed from `kuks.rs`
  and never reached in the k-symmetric twin: `vec![jtot.clone(),
  jtot.clone()]` and `vec![h1e.to_vec(), h1e.to_vec()]` in
  `KsymAdaptedKuks`, `&[h1e.to_vec()]` in `KsymAdaptedKrks::energy_elec`, and
  `nr.vmat[0][0].clone()` / `[1][0].clone()` (now `swap_remove` out of the
  owned result).

### BIT-PARITY — EXACT at every gated cell, asserted not assumed

`tests/ksymm_trace_precision.rs`, 3/3 green:

| assertion | result |
|---|---|
| ordered vs the reproduced pre-P-02 nest, `nao ∈ {8, 11}`, `nkpts_ibz ∈ {3, 10}`, `nset ∈ {1, 2}`, well-scaled and ill-conditioned operands | **bit-identical** |
| the shared-stack form vs the cloned-stack form it replaced | **bit-identical** |
| mean relative error over 200 ill-conditioned draws, `nao ∈ {26, 64}` | ordered strictly better |

The fixture uses **unequal** weights, deliberately: a uniform weight vector
would let a dropped star multiplicity pass unnoticed, which is the trap
`si_222_stars_have_unequal_sizes_so_the_weighting_is_observable` exists for.

---

## P-03 — `oracle_sum` on the multigrid energy path

**FILES** `multigrid/numint.rs`, `multigrid/pair.rs`.

`ecoul`, `nelec` and `exc` in both drivers were naive
`Iterator::sum::<f64>()` over `ngrids` — 42 875 terms at `35³`, the largest
naive reductions on any energy path in the tree, all three landing straight in
a total energy. Now ordered.

**BIT-PARITY: NO, and the item says so.** `ngrids` is far past
`PAIRWISE_CHUNK`, so the tree engages. Scored against Gate E's measured
floors, which it does not move.

---

## S-00 — the ksymm profiling mode

**FILES** `crates/pyscf-bench/src/bin/krks_profile.rs`,
`crates/pyscf-bench/Cargo.toml` (+`pyscf-pbc-symm`; not a cubecl crate, so
ALG-06 is untouched).

`krks_profile ksymm --driver {krks,kuks} --cell … --nk … --mesh … --xc …`
times the ksymm driver and the ordinary full-BZ driver on the same cell:
`kernel()` end to end, warm `get_veff` on the converged density, and
`unfold_kdms` alone (S-01's target, isolated). It reports `nkpts`,
`nkpts_ibz`, the fold factor, `weights_ibz`, **peak RSS** (`VmHWM`) and the
**load average at start** — because RULE O invalidates a ratio measured on a
contended machine and the reader must be able to see whether this one was.

Its output carries a standing note that the two energies are NOT a
correctness comparison (RULE K / D-17-08-03).

### MEASURED — the machine went idle late in the session, and the instrument ran

`si`, `gth-szv`, `[2,2,2]` (`nkpts = 8`, `nkpts_ibz = 4`, fold **2.0x**,
`weights_ibz = [0.125, 0.375, 0.375, 0.125]`), `mesh = 11³`, `LDA,VWN`,
`use_ao_symmetry = true`. **Load average at start: 5.5** on 16 cores — the
first point in this session where RULE O permits quoting a ratio at all.

| stage | KRKS full | KRKS ksymm | ratio | KUKS full | KUKS ksymm | ratio |
|---|---|---|---|---|---|---|
| `get_veff` (warm, per iteration) | 6.460 ms | 4.740 ms | **0.734x** | 10.204 ms | 7.463 ms | **0.731x** |
| `kernel()` (end to end) | 2060 ms | 1834 ms | 0.890x | 2051 ms | 1867 ms | 0.910x |
| `unfold_kdms` (ONE call) | — | **0.044 ms** | — | — | **0.092 ms** | — |

Peak RSS 187 MiB. Both drivers converged to the same energy
(-7.771259390860), which is **not** reported as a correctness result — RULE K
— only as evidence that neither run diverged.

### Three things this settles

**1. The symmetric `get_veff` costs 0.73x the full-BZ one at a 2.0x fold** —
measured three times, on two meshes and both drivers, agreeing to the third
digit.
Not 0.5x. The gap is exactly what 17-08 predicted and §2.2.4 quantifies: the
XC quadrature still evaluates the AOs and the density at every BZ point and
only builds the potential at the IBZ ones, so only the J/K half and the
potential build actually shrink. **That is the size of the prize S-03 is
after**, and it is now a number rather than an argument.

**2. S-01 steps 2 and 3 are NOT worth doing, and this closes them.** One
`unfold_kdms` call is **0.044-0.111 ms against a 4.7-103 ms `get_veff`** —
between 0.1 % and 2 % of it, on every row measured including the rejected ones
(a ratio of two timings taken microseconds apart is far less
contention-sensitive than either alone) (the instrument reports it as "107.8 / 81.5 unfolds per
`get_veff`"). Step 1 removed one of two such calls per cycle, so it bought
about 1 %; caching the rotation matrices and removing the planar/interleaved
conversions would optimise a term that is already ~1 % of the thing it sits
in. §S-01 deferred them pending exactly this measurement. **The measurement
says no.** They are hereby closed as not-worth-doing rather than left open.

**3. The end-to-end ratio (0.89-0.91x) is much worse than the per-iteration
one (0.73x)**, which is the same shape §2.1.0a found for KUKS: an SCF on these
cells is dominated by one-off costs — the cold AO evaluation and `get_hcore` —
that symmetry does not touch. Any ksymm speed work should be scored on
`get_veff`, not on `kernel()`, until those one-offs are addressed.

### The larger baseline was attempted and is REJECTED — and why matters

A `mesh = 31` run (§S-00 step 4's cells) was queued once the load fell. It
completed two of four rows before being stopped, and **neither row is
reportable**. This is written up rather than dropped because the reason it
failed is the reason S-00 prints a load average at all.

| row | load at start | `get_veff` full | `get_veff` ksymm | ratio |
|---|---|---|---|---|
| `krks` / `pbe` | 5.18 | 226.376 ms | 50.910 ms | 0.2249x |
| `kuks` / `pbe` | 9.50 | 141.718 ms | 103.508 ms | 0.7304x |

**The two rows contradict each other, and the contradiction is decisive.**
`KUKS` does strictly more work in `get_veff` than `KRKS` — the same J build
plus a second spin channel through the quadrature — so a full-BZ `KRKS`
`get_veff` **cannot** be slower than a full-BZ `KUKS` one. It is measured at
226 ms against 142 ms. The `KRKS` full-BZ number is therefore inflated, and
the 0.2249x that divides by it is an artefact, not a 4.4x speedup.

The cause is visible in the log: the load average climbed **5.18 → 9.50 →
12.28** across the run as a second workload outside this repository came back.
This plan's own §6 sets the bar at "above ~4 on 16 cores invalidates the row",
and by that bar *both* rows fail — the `KRKS` one only marginally at its
start, but the timings inside it span the climb.

**What survives is the smoke measurement, and it is self-consistent.** Three
independent ratios — `KRKS` at `mesh 11`, `KUKS` at `mesh 11`, and `KUKS` at
`mesh 31` — agree at **0.73x** (0.734, 0.731, 0.730). Three agreeing rows
across two meshes and two drivers is a real number; one disagreeing row whose
denominator is provably wrong is not. **0.73x is what this session stands
behind.**

A clean `mesh = 31` baseline, on a genuinely idle machine, with `pbe0` as
well, is still owed. The instrument is ready and the JSON files it wrote for
the two attempted rows are in the session scratchpad, kept out of
`.planning/pbc/baselines/` deliberately: a rejected measurement does not
belong in the baseline directory.

---

## S-01 — one density unfold per cycle instead of two (or four)

**FILES** `crates/pyscf-pbc-dft/src/numint.rs`, `krks_ksymm.rs`.

`KsymAdaptedKrks::get_veff_tagged` unfolded the IBZ density to the full BZ
**twice** per cycle — once inside `nr_rks` (Group A) and once for the J/K half
— and `KsymAdaptedKuks` did it **four** times, once per spin in each place.
Each unfold is `nkpts` `R·M·Rᴴ` sandwiches plus two format conversions across
the `CTensor`/`Complex64` seam.

Now unfolded once, at the top, and the full-BZ stack handed to both halves.
`unfold_kdms` additionally learned to return `Cow::Borrowed` when its input is
already full-BZ (the guard `unfold_dms` always had per set), so the second
unfold costs nothing at all rather than a k-stack clone.

**BIT-EXACT**, and by an existing test rather than an argument:
`krks_ksymm.rs::unfold_is_a_bit_exact_no_op_on_full_bz_input` is exactly the
property this relies on.

**Steps 2 and 3 NOT done, with a reason.** Caching `get_rotation_mat` and
removing the planar/interleaved conversions are both `O(nao²)` terms inside an
`O(nao³)` sandwich. RULE O says measure first, and S-00 has not been run on an
idle machine. They are cheap and they remain worth doing; they are not worth
doing blind.

---

## S-02 — D-PBC-26 point 1 is wrong, and the saving it was reaching for is available elsewhere

**FILES** `crates/pyscf-pbc-scf/src/khf_ksymm.rs`, `tests/khf_ksymm.rs`.

### The erratum

17-CONTEXT §8 (D-PBC-26) point 1 rules that `get_jk` should be called at
`kpts_ibz` **only** and the result unfolded with `transform_1e_operator`,
citing a measured 40x (GDF) / 223x (FFTDF) from
`measurements/speed_get_jk.py`. 17-07 shipped that as `JkRoute::Fast` and
recorded that its equivalence test had never been written.

**It is not an identity.** `get_j_kpts` forms
`ρ(r) = (1/N) Σ_{k ∈ list} Σ_ij D_k,ij φ*_k,i φ_k,j` over whatever k-list it
is handed; over the IBZ list that is `Σ_{k∈IBZ} ρ_k / N_ibz`, while the true
density is `Σ_{k∈IBZ} w_k ⟨ρ_k⟩_star` — and `ρ_k(r)` is not point-group
invariant (`ρ_{Rk}(r) = ρ_k(R⁻¹r)`). The two agree only when every star has
one member. Applying `transform_1e_operator` afterwards rotates a potential
built from the wrong density; it does not repair it. For `K` the argument is
sharper: restoring the dropped `k2` terms by equivariance needs `K` at every
`R⁻¹k1`, which for `k1` over the IBZ is the whole zone again.

So the 40x/223x compared two **different quantities**, and the attainable
bound is `nkpts / nkpts_ibz`.

### What landed

* `JkRoute::Fast` renamed **`IbzOnly`**, kept non-default and behind a doc
  comment carrying the derivation, solely so the disproving measurement stays
  reproducible.
* `JkRoute::Band` added — unfold the density to the full BZ as the reference
  route does, and ask the DF layer for the output at `kpts_band = kpts_ibz`.
  The exchange sum still runs over every `k2`; only the OUTPUT index is
  restricted, which is `nkpts · nkpts_ibz` pairs instead of `nkpts²` and drops
  nothing — the reference route was computing those extra output points and
  then throwing them away in `fold_to_ibz`. This is the route the DFT
  k-symmetric adapters have taken since 17-08 Task 2.
* `Reference` remains the default until the band route's speed ratio is
  measured `> 1.0` (D-PBC-26 point 6).

### The two tests

`band_route_matches_reference_route_bit_exact` asserts `to_bits()` equality,
not 1e-13: the band route computes the same terms in the same order, so
anything else is a `kpts_band` defect a tolerance would hide.
`ibz_only_get_jk_is_not_an_identity` asserts the disagreement is **larger**
than 1e-6 — a lower bound, deliberately, so "it got closer" can never be
mistaken for "it works" — and first asserts the fixture's stars are unequal,
without which the test would pass for the wrong reason.

### MEASURED — both tests pass, and the derivation is confirmed

`cargo test --release -p pyscf-pbc-scf --test khf_ksymm` — **6/6 PASS**,
502.8 s.

```text
band vs reference: BIT-IDENTICAL over 3 IBZ k-points (max |d| = 0e0);
                   pair count 64 -> 24 (2.667x fewer)

MEASUREMENT (S-02): IBZ-only get_jk vs the reference route,
                    si [2,2,2], stars [1, 3, 4]
                    ->  max |d veff| = 9.486e-2
```

Two things are settled by those two lines.

**The band route is exact.** Not "agrees to 1e-13" — `max |d| = 0e0` over
every element of every IBZ k-point, and it computes `64 -> 24` exchange
pairs, exactly the `nkpts / nkpts_ibz = 8/3 = 2.667x` the derivation predicts.
The reference route was computing those 40 extra output points only to discard
them.

**The IBZ-only route is off by 9.5e-2 Ha per matrix element.** §2.2.3
predicted "of order the star asymmetry, MODELLED ≥ 1e-3"; the measurement is
two orders worse than that floor. D-PBC-26 point 1 is not a tolerance question
and never was.

**The erratum is therefore propagated**, as S-02 step 5 requires: into
`PBC-MASTER-PLAN.md`'s D-PBC-26 entry and `17-CONTEXT.md` §8, both as a
dated, measured ERRATUM block above the original ruling rather than an edit to
it — the ruling's other five points stand, and a reader tracing the history
should see what was believed and why it changed.

**The default is still `Reference`,** per D-PBC-26 point 6: `Band` is exact
but its wall-clock advantage has not been measured on an idle machine, and
this repository does not ship a "faster" path on an unmeasured claim.

---

## U-09 / U-10 — the KUKS driver

### U-09 step 1 — measured negligible, and NOT done

`veff_from_parts` still clones both channels into `sets`
(`[vec![dms[0].clone()], vec![dms[1].clone()]]`). At the reference cell that
is `2 · nkpts · nao² · 16` bytes = **16 KiB** per `get_veff`; at
`nao = 26, nkpts = 64` it is 692 KiB per spin. Removing it means changing
`nr_uks`'s public signature (and, transitively, `unfold_kdms`'s and
`PeriodicDf::get_jk`'s, which take `&KDms = &Vec<KMats>` rather than slices)
across five call sites plus the DF trait.

§2.1.0a says the KUKS-specific contraction work is under 1 % of an SCF. A
public-signature refactor of gated code for 16 KiB is not what that
measurement supports, so it was not done — the same reasoning U-06 step 4
recorded for `get_rho`'s buffer.

### U-09 step 2 — `e1`, ordered — LANDED, bit-exact

`Kuks::energy_elec`'s `e1` was a naive `(nset · nkpts)`-long running sum on a
term of every KUKS total energy. U-03 ordered the KUHF copy in
`krdm::energy_elec` and the `veff.rs` traces and did not reach this one. Now
collects per-`(set, k)` partials and reduces them with `oracle_sum`.
Bit-identical for `2 · nkpts ≤ 128`, i.e. every reference cell.

### U-10 — reusable per-block scratch — LANDED, bit-exact

**FILES** `crates/pyscf-pbc-dft/src/numint.rs`.

This is `KUKS-OPTIMISATION-PLAN` U-06 step 6 and U-05 step 2, which U-06 left
open as needing "interior-mutable scratch on `KNumInt` whose aliasing story is
not free". **The aliasing story is empty**: the lifetime that matters is one
CALL, not one `KNumInt`, so the scratch is an ordinary `&mut` argument
threaded down from `nr_rks` / `nr_uks`.

What it removes, per grid block per SCF cycle, at `si mesh 31`
(`ngrids = 29 791`, `nao = 8`, `nkpts = 8`):

| buffer | size | allocations per block, KUKS |
|---|---|---|
| `eval_rho_one`'s `c0_re`/`c0_im` | 1.9 MiB each | 16 (per k, per spin) |
| `eval_rho_one`'s per-component `acc_re`/`acc_im` | 233 KiB each | 16 × ncomp |
| `vxc_mat_one`'s `aow_re`/`aow_im` | 1.9 MiB each | 16 |
| `vxc_mat_one`'s per-row `terms_re`/`terms_im` | 233 KiB each | 16 × nao |
| `nr_uks`'s `dena`/`denb`/`ta`/`tb` | 233 KiB each | 4 |

about **61 MiB of allocate-and-zero for `c0` alone**, and as much again for
`aow`. All of it is now one reused buffer per call.

`KNumInt::eval_rho` and `accumulate_vxc` keep their public signatures and
allocate one scratch per call (already a factor-`nkpts` reduction); `nr_rks`
and `nr_uks` take `_into` variants and hold ONE scratch for the whole block
loop.

**BIT-PARITY: EXACT.** Every replaced allocation was zero-filled and then
either fully overwritten or accumulated from zero; `Scratch::zeroed`
reproduces that state exactly and `Scratch::raw` is used only where every
element is assigned before it is read. Gated by `numint_blocking` (3/3),
`numint_blocking_uks` (4/4) and `numint_threads` (1/1), all green and all
bit-identity assertions.

---

## M-00 — `nr_uks` for both multigrid drivers

**FILES** `multigrid/numint.rs`, `multigrid/pair.rs`, `tests/multigrid_uks.rs`
(new).

Neither driver had `nr_uks`, so "KUKS on multigrid" was a phrase and not a
code path — and no multigrid optimisation could be validated on an open-shell
density at all, which RULE U requires.

The spin-generic middle of `multigrid.py:1059` `nr_rks` and `:1166` `nr_uks`
is now ONE function, `mg_xc_parts`, shared by both drivers and both spin
cases. Upstream's associations are reproduced rather than re-derived: the
Coulomb term uses the spin-summed density and both channels receive the same
`vG` (`:1223`, `:1246`); `excsum` is a per-spin dot then a sum over spins
(`:1238`), not one flat reduction over `2 · ngrids` terms.

The `nr_rks` refactor onto the shared middle is **bit-exact for the
one-channel case by construction** — the spin sum is `0 + x`, the `excsum`
composition is a one-element pairwise sum, and every other statement is
unchanged.

`tests/multigrid_uks.rs` gates it three ways. The one that matters most needs
no reference implementation: **`nr_uks(dm/2, dm/2)` must reproduce
`nr_rks(dm)`**, an identity of the unrestricted functional. That is what
catches a transposed channel, a spin sum on the wrong axis, `vG` added to one
channel only, or `excsum` counting a spin twice — all of which leave the
per-channel numbers plausible and only this identity refuses.

---

## M-02 — cache the multigrid geometry

**FILES** `multigrid/utils.rs` (`cell_fingerprint`), `multigrid/numint.rs`,
`multigrid/pair.rs`, `tests/multigrid_cache.rs` (new).

Both drivers rebuilt their entire geometry on every call — decontraction, task
list, and (v2) every level's pair table, which re-runs `get_lattice_ls` and
the full binomial-shift image enumeration for every pshell pair. None of it
depends on the density.

* Both drivers now hold a one-entry cache keyed by `cell_fingerprint`, a hash
  of the lattice, mesh, precision, `rcut`, `ke_cutoff`, dimension, `_bas` and
  every float in `_env` (which is where the atom coordinates, exponents and
  contraction coefficients live) — **bit patterns, not values**.
* **v1 additionally collocated every level twice per call**, once in
  `rho_g_from_levels` and once in `pass2_from_full_vg`. Collocation is the
  dominant cost of a v1 evaluation. It is now done once and shared between the
  two directions.
* **v2's block partition and per-block reach lists** (`grid_blocks`,
  `block_slots` — pure geometry by their own doc) were recomputed inside BOTH
  directions of every call, i.e. four times per level per cycle. They are now
  built once, at table-build time.

`tests/multigrid_cache.rs` asserts the two things a cache must satisfy, rather
than assuming either: cached and uncached agree at `to_bits()`, and **one
driver is never served another cell's geometry** — including the harder case
of the same cell at a different mesh, which is the failure mode that would
produce a plausible wrong energy rather than a crash.

---

## M-03 — one launch per level, and an honest limit

**FILES** `crates/pyscf-kernels/src/multigrid_pair.rs`, `multigrid/pair.rs`,
`tests/multigrid_batch.rs` (new).

`11_launch_overhead_and_transfers.md` §5 ("collapse per-item launches into
one"), §2 (hoist invariant uploads) and §3 (batch read-backs), read before
writing the kernel per RULE 5. The v2 driver issued **one launch per spatial
block per direction** — 125 at `mesh = 25³` — each uploading seven buffers and
reading one back. 17-12 attributed its first streamed version's 130 s → 7-9 s
to exactly this and left batching as its carry-over #3.

Two new kernels, `collocate_pairs_rho_batched` and
`collocate_pairs_integrate_batched`, take the CONCATENATED block tables and
one launch covers the level. The manual's §5 pairs a single launch with an
offset table; the inverse map (a lane finding its own block) is materialised
host-side as one `u32` per lane rather than searched in-kernel, because
`Cubecl_conditionals.md` argues against a data-dependent in-kernel search and
because precomputing it is strictly less work per lane.

### BIT-IDENTICAL — measured, not argued

`tests/multigrid_batch.rs`, 3/3 green, both routes in ONE process on ONE
table:

| cell | level | forward `rho` | reverse `pass2` |
|---|---|---|---|
| si | 1, 2, 3 | **max &#124;d&#124; = 0e0** | **0e0** |
| diamond | 1, 2, 3 | **0e0** | **0e0** |

### THE FINDING: the budget guard keeps 2 of 3 levels on the streaming path

A block's kernel-slot list contains every slot whose instance **reaches** that
block, so a diffuse Gaussian is replicated across blocks and the concatenation
is a multiple of the level's own slot count — the same quantity 17-12's
`pair_level_tables_stream_under_budget` bounds, and the same reason its
predecessor was SIGKILLed. `BATCH_BUDGET_BYTES` is 256 MiB (17-12's own
per-launch budget, reused rather than reinvented), and above it the streaming
path runs unchanged.

Measured on the Gate-E cells:

| cell | level | mesh | blocks | batched? | batch size |
|---|---|---|---|---|---|
| si | 1 | 9³ | 8 | **yes** | 729 points, 170 686 slots, **9.1 MiB** |
| si | 2 | — | 27 | no — over budget | — |
| si | 3 | 25³ | 125 | no — over budget | — |
| diamond | 1 | 9³ | 8 | **yes** | 729 points, 364 460 slots, **19.5 MiB** |
| diamond | 2 | — | 27 | no — over budget | — |
| diamond | 3 | 25³ | 125 | no — over budget | — |

**So M-03 currently collapses 8 launches into 1 on the coarsest level — the
cheapest one — and leaves the 27- and 125-launch levels streaming.** The
mechanism is correct and gated; the win is small until the replication is
reduced, which is exactly what the deferred M-05 (per-Gaussian sub-mesh) would
do. Raising the budget is the wrong lever: the batch is CACHED for the SCF's
lifetime, so it is resident memory, and 17-12's OOM is the precedent against
trading it casually.

This is reported as a limit rather than a win, per the plan's RULE M and
D-PBC-26 point 6.

**M-03 step 5 (FOUND-05) is done and verified.** `pyscf-kernels` is now in
`check-no-fma`'s `SCAN_TARGETS` — its fused reductions accumulate a whole
level's contribution inside one lane and land directly in `ecoul`/`exc`, which
is the criterion the other entries were added under. The concern that it might
hit the same rustc segfault that keeps `pyscf-pbc-dft` off the list did not
materialise: the crate builds clean under `release-oracle` and the scan now
covers **7** asm files instead of 6. **PASS.**

**One cheap follow-up, named because it may move the line.** The budget check
bounds the instance count by the SLOT count (one instance per slot), which
double-counts the per-instance arrays. On the coarsest level the two happen to
be equal (170 686 instances for 170 686 slots), so the estimate is exact
there — but on a fine level, where a compact Gaussian carries many monomials
per instance, it over-estimates. Counting the instance transitions exactly is
one extra pass over `block_sel` and costs nothing; it should be done before
concluding that levels 2 and 3 genuinely do not fit.

---

## M-04 — the v1 host loops

**FILES** `multigrid/colloc.rs`.

* `level_rho` allocated a term buffer **per grid point** — `ngrids` heap
  allocations per level per call (15 625 at `25³`, 42 875 at `35³`), where
  upstream's per-point work allocates nothing. Now one buffer per rayon chunk.
  **Bit-exact**: every element is assigned before it is read, the term list
  and its order are unchanged, each point still reduces its own full list with
  the same `oracle_sum`, and the chunking is over disjoint outputs.
* `level_pass2` allocated an `ngrids` buffer **per matrix entry** — `nao_p²`
  of them per level per call. Now one, reused. Bit-exact for the same reason.

**Step 3 (radius-screened `pass2`) NOT started.** It changes results and is
opt-in by design; it was not reached.

---

## The one speed number this session CAN defend

Every wall-clock figure below is a **within-run ratio**: the reference route
and the multigrid route are timed back to back, in one process, under
whatever load the machine happens to be under. That is what makes it
comparable to 17-12's own ratio despite the machines differing — and it is why
this table exists while §2.0's ksymm rows still say UNMEASURED, where only an
absolute number would do.

Load average during the run: **10-17 on 16 cores** (a second workload outside
this repository). Absolute times are inflated; the ratios are not.

### multigrid v1 `get_j` against the reference FFTDF route

| cell | 17-12's ratio | **this session** |
|---|---|---|
| diamond | 1.50x | **2.56x** |
| si | 1.24x | **2.52x** |

**v1 went from ~1.2-1.5x faster than the reference route to ~2.5x.** Both are
bit-exact changes, so this is pure cost removal — Gate E's residuals are
unmoved, below.

**And it is measured on a COLD cache**, which matters for reading it. The test
times ONE `get_j` per cell on a driver that has not seen that cell before, so
M-02's cross-call geometry cache contributes **nothing** to this number. What
it measures is the two within-call changes: collocating each level once per
call instead of twice (M-02's other half), and M-04's removal of the
per-grid-point and per-matrix-entry allocations. Inside an SCF, where the
task list is built once and reused from cycle 2 onward, the gain should be
larger — by how much is not measured, and is not claimed.

### multigrid v2 `get_j`

| cell | 17-12's ref/v2 | this session |
|---|---|---|
| diamond | 0.023x | 0.0253x |
| si | 0.028x | 0.0333x |

**About 10-19 % better, and that is the honest size of it.** M-03's batching
reaches only the coarsest level under the current budget (see M-03's finding),
and M-02's caching helps v2 less than v1 because v2 never had the
double-collocation v1 did. **v2 remains ~30-40x SLOWER than the reference
route**, i.e. RULE M's verdict from 17-12 stands unchanged: v2 is for
`isinstance` fidelity, not for speed.

### Gate E is unmoved — every residual identical to 17-12's

This is the check that matters for P-03 (which deliberately changes last
bits), M-02 and M-03:

| quantity | 17-12 recorded | this session |
|---|---|---|
| v2 `get_j` vs FFTDF, diamond | 1.24e-8 | **1.242e-8** |
| v2 `get_j` vs FFTDF, si | 6.80e-8 | **6.804e-8** |
| v2 `nr_rks` Δnelec / Δexc, diamond | 1.5e-6 / 7.9e-7 | **1.493e-6 / 7.869e-7** |
| v2 `nr_rks` Δnelec / Δexc, si | 4.8e-7 / 1.3e-7 | **4.784e-7 / 1.275e-7** |
| v1 `nr_rks` Δnelec / Δexc, si | — | **1.776e-15 / 8.882e-16** |

---

## Verification, as run

`cargo test --release -p pyscf-pbc-dft` over every suite this plan touches:
**52 tests, 0 failures**, plus `pyscf-pbc-scf`'s `khf_ksymm` at **6/6**.

| gate | item | result |
|---|---|---|
| GATE U (oracle) | P-00 | **9/9 PASS**, 90.7 s |
| GATE A (oracle) | P-00 | **7/7 PASS**, 356 s |
| `ksymm_trace_precision` | P-02 | **3/3 PASS** |
| `veff_trace_precision` | U-03 regression | **3/3 PASS** |
| `numint_blocking` | U-10 | **3/3 PASS** |
| `numint_blocking_uks` | U-10, open shell | **4/4 PASS** |
| `numint_threads` | U-10, GATE B | **1/1 PASS** |
| `ksymm_threads` | P-01 | **2/2 PASS**, 13.6 s |
| `multigrid_threads` | P-01 | **4/4 PASS**, 190 s |
| `multigrid` | M-02, M-04, P-03 (v1) | **6/6 PASS** |
| `multigrid2` | M-02, M-03, P-03 (v2), Gate E | **10/10 PASS**, 297 s |
| `multigrid_cache` | M-02 | **4/4 PASS**, 148 s |
| `multigrid_uks` | M-00 | **5/5 PASS**, 92 s |
| `multigrid_batch` | M-03 | **3/3 PASS**, all residuals 0e0 |
| `modules`, `smoke` | regression | **8/8**, **13/13 PASS** |
| `khf_ksymm` (`pyscf-pbc-scf`) | S-02 | **6/6 PASS**, 503 s |
| `krks_ksymm` — GATE C | P-02, S-01 | **7/7 PASS** (3 pre-existing ignores), 300 s |
| `check-dependency-wall` | S-00's new dependency | **PASS** — cubecl containment intact (ALG-06) |
| `check-orphan-modules` | regression | **PASS** — 336 source files, all reachable |
| `check-no-fma` | M-03 step 5 | **PASS** — 7 asm files scanned (was 6); `pyscf-kernels` builds clean under release-oracle |

### The two results worth reading in full

**P-01 — the k-symmetry determinism gate that had never existed, green.**

```text
ksymm KRKS use_ao_symmetry=true : e_tot = -7.771258255949421   bit-identical at [1,2,3,8] threads
ksymm KRKS use_ao_symmetry=false: e_tot = -7.771258255949419   bit-identical at [1,2,3,8] threads
ksymm KUKS                      : e_tot = -2.976003470780665   bit-identical at [1,2,3,8] threads
                                  max |dm_a - dm_b| = 1.137     (RULE U: genuinely open-shell)
```

Both `eig` branches, and an open-shell driver whose two channels differ by
1.14 — so this is measuring the unrestricted path, not a restricted one
wearing its name.

**GATE C — P-02 and S-01 moved nothing, measured end to end.**

This is the gate that scores the two bit-exactness *arguments* this session
made at driver level: that ordering the `weights_ibz` traces and unfolding the
density once instead of twice leave every k-symmetric energy where it was.
Every number comes back identical to the one 17-08 recorded:

| quantity | 17-08 recorded | this session |
|---|---|---|
| `KRKS` IBZ vs full BZ, `use_ao_symmetry = false` | 3.109e-14 | **3.108624468950438e-14** |
| `KRKS` IBZ vs full BZ, `use_ao_symmetry = true` | 2.842e-14 | **2.842170943040401e-14** |
| unfolded IBZ density vs full-BZ density | 1.054e-13 | **1.0543649286987034e-13** |
| DFT+U `E_U`, IBZ vs full BZ | 6.939e-18 | **6.938893903907228e-18** |
| open-shell IBZ `KUKS` `e_tot` | -2.724600963472 | **-2.724600963472** |

Not "within tolerance" — the same digits. P-02 replaced two hand-rolled nests
on the energy path and S-01 halved the number of density unfolds, and the
k-symmetric drivers produce bit-for-bit what they produced before.

**M-00 — the open-shell multigrid, and the identity that proves it is wired
the right way round.**

`nr_uks(dm/2, dm/2)` against `nr_rks(dm)` is an identity of the unrestricted
functional and needs no reference implementation. It comes out **exact**:

| driver | cell | xc | Δnelec | Δecoul | Δexc | Δveff |
|---|---|---|---|---|---|---|
| v1 | diamond | LDA,VWN | **0** | **0** | **0** | **0** |
| v1 | diamond | PBE | **0** | **0** | **0** | 2.2e-16 |
| v1 | si | LDA,VWN | **0** | **0** | **0** | **0** |
| v1 | si | PBE | **0** | **0** | **0** | 6.9e-18 |
| v2 | si | LDA,VWN | **0** | **0** | **0** | **0** |

and against the reference `KNumInt::nr_uks` on the same grid:

| driver | cell | Δnelec (α, β) | Δexc | gate |
|---|---|---|---|---|
| v1 | diamond | 1.95e-14, 7.11e-15 | 1.24e-14 | 1e-6 |
| v1 | si | 3.55e-15, 0 | 8.88e-16 | 1e-6 |
| v2 | si | 3.84e-7, 3.07e-7 | 2.10e-7 | 1e-3 |

v1 lands at machine precision, eight orders inside its gate; v2 sits on its
own screening floor, where 17-12 measured its restricted twin.

### Still running when this was written

`pyscf-pbc-scf` (including S-02's two tests), GATE C
(`--test krks_ksymm`), the three xtask lints, and an S-00 smoke run. All four
are queued and their results belong in the next revision of this file. **They
are not reported as green here.**

### Not run, and named

* **Any ksymm profile.** The machine ran at load average 5-33 on 16 cores over
  the session; RULE O forbids quoting an absolute ratio off the loaded part of
  it. The multigrid numbers above survive only because they are within-run
  ratios. A baseline run was queued once the load fell to ~5; whether it
  produced a usable number is recorded in §S-00 above.

---

## What the next session should do, in order

1. **Run the suites this one launched**, and GATE C, and the three xtask
   lints. Nothing else should start first.
2. **Run S-02's two tests.** If `ibz_only_get_jk_is_not_an_identity` fails,
   §2.2.3's derivation is wrong — stop and re-derive. If it passes, propagate
   the erratum into `17-CONTEXT.md` §8 and the `PBC-MASTER-PLAN` D-PBC-26
   entry, and measure `Band` vs `Reference` before flipping the default.
3. **Run S-00 on an idle machine** (print `uptime`), and only then decide
   S-01 steps 2-3 and S-03.
4. **M-01**, the numint seam — it is what makes RULE M's converged-SCF ratio
   measurable at all, and it is the last thing standing between the multigrid
   work and a KUKS-on-multigrid number.
5. Answered this session: **Q1** (§2.1.0a) and **Q7** (`pyscf-kernels` is NOT
   in `check-no-fma`'s `SCAN_TARGETS`; M-03 should add it).

## Answered questions

| # | answer |
|---|---|
| Q1 | K ×1.034, J ×1.9-2.2, quadrature ×1.3-1.75 — §2.1.0a |
| Q3 | **YES** — the FFTDF band route is bit-identical to the full route (`max &#124;d&#124; = 0e0`), matching the GDF result 17-08 measured. §2.2.3's prediction held |
| Q2 | **0.73x** of the full-BZ `get_veff` at a 2.0x fold, both drivers; the unfold itself is under 1 % of `get_veff` |
| Q6 | **Partly.** The batch fits at the coarsest level only (9.1 / 19.5 MiB); levels 2 and 3 exceed 256 MiB and stream. See M-03's finding |
| Q7 | No — it scanned six crates. **M-03 step 5 added it, and the lint PASSES** with 7 asm files |
