# KRKS + k-symmetry + multigrid — session 7 execution record

**Plan:** [`KUKS-KSYMM-MULTIGRID-OPTIMISATION-PLAN-2.md`](./KUKS-KSYMM-MULTIGRID-OPTIMISATION-PLAN-2.md);
session 6's record: [`KUKS-KSYMM-MULTIGRID-SESSION-6-EXECUTION-SUMMARY.md`](./KUKS-KSYMM-MULTIGRID-SESSION-6-EXECUTION-SUMMARY.md),
whose §6 left **"uniform-set fast path in the vector reverse"** as the one
named, modelled (`<= 1.3x` on the reverse's non-`exp` half) and **unmeasured**
multigrid row. This session measures it.
**Date:** 2026-09-12.
**Base commit:** `b554cfb`. **The work is NOT committed** — the tree carries
two modified files (`crates/pyscf-kernels/src/multigrid_pair.rs`,
`crates/pyscf-pbc-dft/tests/multigrid_batch.rs`, +277 / +96 lines).
**Ask (user, verbatim):** "Please optimise speed KRKS and k-point symmetry +
multigrid gpu kernel on-device by loop and comptime manual. Please use
codegraph mcp server."
**Machine / load:** 16 cores, 30 GiB, CubeCL CPU runtime (no f64 GPU here,
`rocm-igpu-no-f64`); every GPU statement is REASONED and marked. The
1-minute load is quoted beside every number (RULE O).
**Manuals read before touching a kernel (RULE 5):** `INDEX.md`,
`01_loop_unrolling.md`, `comptime_macro.md`,
`Cubecl_comptime_specialization.md`, `Cubecl_loop_control.md`,
`Cubecl_conditionals.md`, `plane_alignment.md`, `06_vectorization.md` +
`Cubecl_dynamic_vectorization.md`, `07_memory_coalescing.md`,
`10_grid_stride_occupancy.md` + `Cubecl_grid_stride_loop.md`,
`03_kernel_fusion.md`, `11_launch_overhead_and_transfers.md`,
`05_lazy_execution.md`, `13_memory_preallocation.md`,
`Cubecl_compilation_caching.md`, `Cubecl_generics.md`, `profiling_tools.md`,
`16_profiling_and_bottleneck_identification.md`.

---

## 0. What this session did

| item | what | result |
|---|---|---|
| **M-22** uniform-group reverse | `mg_integrate_vec_uniform_kernel<N>`: a group whose N lanes all share one term set has lane-uniform monomial powers, so M-19's predicated products (`poly *= dx·m + (1−m)`, six vector ops per slot per point) collapse to scalar-bounded `while` loops and the `pw` / `mask` locals disappear. `reverse_groups` partitions the group table uniform-first; two launches, one per arm | landed, **bit-exact**; **1.14x** level-3 reverse, **1.25x** level-2 (§3); arm `PYSCF_MG_PAIR_UNIFORM=0` |
| scope finding — k-symmetry | the multigrid collocation is per-IMAGE and flat in `nkpts`, so k-symmetry cannot speed it; the only `nkpts`-growing work is a **serial host contraction** (§5) | recorded, NOT taken |
| measurement correction | the group-uniform fraction is **58.2 %**, not the 85 % the occurrence histogram suggests (§6.1); cross-binary bench deltas on this box are **±10 %** and are not evidence (§6.2) | corrected in-session |

Everything above went through `tests/multigrid_batch.rs`'s `to_bits()`
comparison (§4): no tolerance anywhere.

---

## 1. Baseline (BEFORE), `mg_pair_bench`, Si gth-szv `25^3`, warm min over 3

Own binary, default target dir (`profile.release`, `lto = "thin"`), load
**1.36** at launch. This is NOT `target/gate`, so these absolute numbers are
not comparable with session 6's rows — §6.2 is the reason that matters.

| level | forward | reverse |
|---|---|---|
| 1 | 7 ms | 10 ms |
| 2 | 297 ms | 500 ms |
| 3 | 1 018 ms | **1 651 ms** |

The reverse is **1.63x the forward**, which is why the reverse is the half
worth attacking. Same-set run histogram, unchanged from session 6: level 3
93.1 % of occurrences in runs >= 4 and **85.1 % in runs >= 8**; level 2
99.6 % >= 8; level 1 98.4 % >= 8.

---

## 2. M-22 — the uniform-group reverse kernel

`crates/pyscf-kernels/src/multigrid_pair.rs`:
`reverse_groups` (partition), `mg_integrate_vec_uniform_kernel<N>` (the
kernel), `reverse_vec_uniform_local_bytes` (its stack budget),
`reverse_uniform_enabled` (the arm), `launch_integrate_resident` (two
launches). Single-spin only — see §7.

**Why a separate kernel and not a `#[comptime] uniform: bool` inside the
existing one.** Uniformity is a per-GROUP runtime property, so it cannot be a
comptime argument directly; the comptime-shaped form is to partition on the
host and specialise per launch, which also removes the branch entirely rather
than trading one for another (`plane_alignment.md` §2). A comptime flag with
the arrays declared inside the pruned branch would probably also drop them,
but "probably" is the wrong word next to
`cubecl-cpu-local-arrays-cost-stack-per-iteration`: on the CPU runtime a
kernel's `Array::new` locals are charged per cube ITERATION, and the budget
handed to `launch_1d_chunked` has to be right. A distinct kernel makes it
unambiguous.

**Why both group orders are resident.** The partition reorders the group
table, and reordering is itself a change — it alters block visit order and so
the reverse direction's locality. Uploading only the sorted table would make
`PYSCF_MG_PAIR_UNIFORM=0` mean "reorder without the specialised kernel"
instead of "the pre-M-22 launch", and the A/B would attribute the reorder's
effect to the kernel. **This mistake was actually made and measured before
being caught** (§6.2). Both orders are now resident and the choice is made per
LAUNCH — not per upload, because chunk geometry is cached for the life of the
level, so an upload-time switch would leave both arms of a comparison sharing
whichever order was built first: a gate that passes while testing nothing.
Cost: one extra `u32` per group, ~8.9 MB across level 3's four chunks against
267 MB of resident geometry.

**Bit-exactness (argued, then gated).** For a power `p ∈ {0,1,2}` the
predicated form multiplies by `dx` exactly `p` times and by exactly `1.0` for
the remaining `2 − p` steps, in that order; the uniform kernel performs the
same `p` multiplications in the same order and omits the `× 1.0`, which is
the identity on every finite value. The one residue is a `-0.0` the
predicated path turns into `+0.0` (`dx·1 + 0` with `dx = -0.0`) and this one
keeps — a sign of zero no accumulator can observe, exactly the contract
`mg_integrate_vec_kernel` already records for itself. Held at `to_bits()`,
not argued: §4.

**Reordering is bit-exact** because a group writes only its own occurrences'
slots (`out[occ_slot0[occ] + local]`), disjoint across groups, so no sum's
order changes; the device fold reads `out` in occurrence order afterwards
regardless of the order the groups ran in.

---

## 3. Measured (§RULE O: one binary, interleaved, idle start)

`mg_pair_bench`, Si gth-szv `25^3`, warm min over 3, **one binary**, arms
interleaved OFF / ON / OFF / ON. Load climbed across the run, so the
UNTOUCHED forward direction doubles as a drift monitor:

| pass | load | L2 fwd | L2 rev | L3 fwd | L3 rev |
|---|---|---|---|---|---|
| OFF-1 | 2.94 | 276 | 451 | 939 | 1 498 |
| ON-1 | 6.07 | 283 | **364** | 961 | **1 348** |
| OFF-2 | 8.80 | 291 | 473 | 1 021 | 1 635 |
| ON-2 | 10.60 | 297 | **392** | 1 032 | **1 447** |

The forward drifted +10 % (939 -> 1 032) purely with load, so raw min/min
overstates. Normalising each pass by its own forward:

| | L3 rev / L3 fwd | L2 rev / L2 fwd |
|---|---|---|
| OFF-1 | 1.595 | 1.634 |
| OFF-2 | 1.601 | 1.625 |
| ON-1 | **1.403** | **1.286** |
| ON-2 | **1.402** | **1.320** |

The two ON passes agree to **0.1 %** at level 3 and the two OFF to 0.4 %,
which is what makes the result trustworthy:

* **level-3 reverse 1.14x**, **level-2 reverse 1.25x**, bit-exact.
* whole pair-kernel pair (fwd + rev, adjacent passes OFF-1 vs ON-1)
  3 181 -> 2 972 ms = **1.07x**, and conservative, since ON-1 ran at twice
  OFF-1's load. The forward is untouched and `exp` is still ~half the
  reverse, which bounds it.
* consistent with session 6's model: `<= 1.3x` on the reverse's non-`exp`
  half is ~1.15x on the reverse overall.

Coverage: **1 286 922 / 2 212 606 groups (58.2 %) uniform at line 8**, printed
by the gate as its own precondition.

---

## 4. Bit-exactness and gates — all green

`rustfmt` on the two touched paths only (a blanket `cargo fmt` reformats
unrelated files — `cargo-fmt-reformats-unrelated-files`); `git diff --stat`
confirms exactly two files changed. RULE 6:
`check-dependency-wall: PASS — cubecl-* containment intact (ALG-06)`.

One `cargo test --release -j 4` call, named targets, `--test-threads=1` (the
`PYSCF_MG_PAIR_*` switches are process-wide):

| gate | result |
|---|---|
| `multigrid_batch` (incl. the new `uniform_group_reverse_matches_the_predicated_kernel`) | **7/7** |
| `multigrid2` | **10/10** |
| `multigrid_kpts` | **6/6** |
| `multigrid_uks` | **5/5** |
| `multigrid_device_scatter` | **4/4** |
| `multigrid_threads` (GATE B determinism) | **4/4** |
| `multigrid_scf` | **3/3** |
| `krks_ksymm_multigrid` (GATE: IBZ vs full BZ on multigrid) | **2/2** |
| **total** | **41 passed, 0 failed, 0 ignored** |

The new gate compares the two arms at `to_bits()` per level and carries a
**non-vacuity assertion**: it pins `PYSCF_MG_PAIR_LINE=8`, counts uniform
groups host-side, and FAILS if that count is zero — without it a fixture with
no uniform group would compare the predicated kernel with itself and pass
while testing nothing (`libxc-parity-gates-can-be-vacuous`,
`ksymm-ibz-kpts-were-not-a-bitwise-subset`).

---

## 5. k-symmetry: why this session added nothing there, and what is left

Confirmed by reading, not assumed: the multigrid collocation runs over
lattice **images**, so it is flat in `nkpts`
(`multigrid/kpts.rs` module doc; `tests/krks_ksymm_multigrid.rs` doc:
"there is no multigrid analogue of S-03's symmetrised quadrature, and none is
invented"). `KsymAdaptedKrks` already unfolds to the full BZ once per cycle
(S-01) and asks for `kpts_band = kpts_ibz`, which the k-resolved multigrid
absorbs as one more phase table. **k-symmetry therefore cannot make the
multigrid kernels faster, by construction** — consistent with
`multigrid-cost-is-per-image-not-per-kpoint`.

What *does* grow with `nkpts` is the HOST contraction, and it is **serial**:
`term_coef_kpts` (`multigrid/kpts.rs:165`) and `pairlevel_pass2_kpts`
(`:217`) are both `O(nslots · nkpts)` with a plain `for k in 0..nk` inner
loop, and `kpts.rs` contains **no rayon at all** (the crate uses
`rayon::join` elsewhere in `pair.rs`). At a dense k-mesh this is the only
`nkpts`-scaling term in the whole engine.

**MEASURED, and REFUTED as a speed item** (`tests/mg_kpts_bench.rs`, new this
session). The sweep needs no instrumentation to attribute: the collocation is
flat in `nkpts` by construction, so the slope is everything that scales with
k. Si gth-szv, `25^3`, warm min over 3, load 2.54 at start:

| nk | nkpts | `eval_rho_g_kpts` | `nr_rks_kpts` |
|---|---|---|---|
| 1x1x1 | 1 | 1 243 ms | 2 968 ms |
| 2x2x2 | 8 | 1 263 | 3 042 |
| 3x3x3 | 27 | 1 327 | 3 148 |
| 4x4x4 | 64 | **1 352** | **3 233** |

A **64x** increase in k-points buys **+8.8 %** wall (109 ms forward,
265 ms `nr_rks`); at the common 2x2x2 it is **+1.6 %**. Perfectly
parallelising the whole slope on 16 cores is therefore bounded by ~**1.08x**
at 64 k-points and ~1.015x at 8.

And the slope OVER-attributes: `PhaseTable::new` (`O(nkpts · nimg)` per level
per call), `KDmP::expand`, `contract_v_kpts` and the `v_re`/`v_im` allocation
are all `O(nkpts)` as well, so the two serial contractions are only a
fraction of the 8 %. The upper bound is already too small to justify the
change — which would also have to stand up a k-path determinism gate that
does not exist (below) — so the item is **not taken**. Refining the
attribution would not change that conclusion.

---

## 6. Two measurement errors made in-session, and corrected

### 6.1 Occurrence-fraction is not group-fraction

`mg_pair_bench` prints "occurrences in runs >= 8: 85.1 %", and the lever was
sized off it. The quantity that governs M-22 is the fraction of **groups**
that are uniform, which measured **58.2 %** — a group of 8 straddling a
same-set run boundary is mixed even when both runs are long. Sizing this
lever off the histogram overestimates it by ~1.5x. The gate now prints the
group fraction so the next reader gets the right number.

### 6.2 A cross-binary timing delta is not evidence

The first A/B measured OFF at 1 477 ms against a 1 651 ms baseline taken from
a **different binary**, and this was initially explained as the reordering
leaking into the control. That explanation was **wrong**. After the control
was fixed so OFF is the pre-M-22 launch exactly, OFF still did not return to
1 651 (it measured 1 498 at *lower* load than the pass that gave 1 635). The
real cause is inter-process / inter-binary variance of about **±10 %** —
code layout, allocator state, first-touch. Consequences, both now standing
rules for this engine:

* only an **interleaved within-binary** A/B counts, and it should be
  normalised by a direction the change does not touch;
* a kill-switch must restore the original **data order**, not just the
  original code path, or the control silently carries half the change.

The 1.29x figure derived from the cross-binary comparison is **retracted**;
1.14x (§3) is what is claimed.

---

## 7. Not done, and why

* **The two-spin twin** (`mg_integrate2_vec_kernel`). That is the KUKS path;
  the ask was KRKS, and adding it would have doubled the diff on the critical
  path for a benefit not requested. The uniform partition and both group
  tables are already resident, so the twin is a mechanical follow-up:
  duplicate `mg_integrate_vec_uniform_kernel` with two weights and two
  accumulators, and split `launch_integrate2_resident` the same way.
  Modelled at the same ~1.14x on the KUKS reverse; UNMEASURED.
* **The forward monomial loops** — an unclaimed lever session 6 never listed.
  `mg_rho_kernel` / `mg_rho2_kernel` still evaluate monomials with three
  RUNTIME-bounded `while i < ix { poly *= dx }` loops per slot, while
  `validate_batch` guarantees every power `<= 2` and the bench confirms it
  (`max-power 2`; level 1 is `max-power 0` with one slot per set, so those
  loops run zero iterations and are pure overhead). A loop with a statically
  known bound of 2 is exactly `01_loop_unrolling.md`'s case. Smaller than
  M-22 (the forward's non-`exp` half is ~455 ms of 1 018 ms) but real.
* **An SCF-level KRKS number.** Everything here is isolated-kernel.
  `krks_profile` lives in `pyscf-bench`, whose build shape re-unifies features
  and can rebuild the ~500-crate libxc tree (`libxc-release-builds-are-slow`),
  so it was not spent unprompted. Expect well under the 1.07x pair-kernel
  figure at SCF level, since `get_veff` also carries FFT and XC.
* **The serial k-resolved host contraction** — attempted, **measured and
  refuted** (§5): 8.8 % of the run at 64 k-points, 1.6 % at 8, upper-bounded
  at ~1.08x if perfectly parallelised. Recorded rather than built.
* **A k-path determinism gate is MISSING, independent of performance.**
  `multigrid_threads` covers only `v1_nr_rks`, `v1_get_j`, `v2_nr_rks`,
  `v2_get_j` — all gamma-point; there is not one `kpts` reference in the
  file. So `nr_rks_kpts`, which produces `ecoul`/`exc`/`nelec` for every
  k-symmetric multigrid SCF, has never been gated for the thread bit-identity
  D-PBC-17 requires. This is vacuously satisfied TODAY because the k-resolved
  contraction is serial, and stops being vacuous the moment anyone
  parallelises it — which is precisely what the refuted item above proposed.
  Worth adding on its own merits.
* **`krks_profile`'s multigrid mode is gamma-only** — `run_multigrid` passes
  `&[[0.0; 3]]` to both `Krks::new` and `Kuks::new`, so it cannot profile the
  k-resolved path at all, and an SCF-level number for any k-resolved item
  needs new harness code rather than just a `pyscf-bench` build.
* **`target/gate` / no-LTO binaries.** Session 6's rows were taken there;
  `target/gate` does not exist in this tree and building it is a cold libxc
  compile. All numbers here are same-binary ratios instead, which is what
  §6.2 says is the only trustworthy form anyway.
* **Not committed.** No commit or push was requested.
