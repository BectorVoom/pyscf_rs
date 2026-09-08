# KRKS + k-symmetry + multigrid — session 4 execution record

**Plan:** [`KUKS-KSYMM-MULTIGRID-OPTIMISATION-PLAN-2.md`](./KUKS-KSYMM-MULTIGRID-OPTIMISATION-PLAN-2.md)
(session 3's record: [`KUKS-KSYMM-MULTIGRID-SESSION-3-EXECUTION-SUMMARY.md`](./KUKS-KSYMM-MULTIGRID-SESSION-3-EXECUTION-SUMMARY.md)).
**Date:** 2026-09-08.
**Machine:** 16 cores, 30 GiB, CubeCL **CPU** runtime (`pyscf-algebra`
`default = ["cpu"]`; the ROCm iGPU has no f64). Every number is a CPU-runtime
number; every GPU claim stays UNVERIFIED (RULE G/T).
**Load:** the box was freshly booted and otherwise idle — the 1-minute load
average at launch is printed in every report and quoted beside every number;
the AO rows were taken at 1.7-6.2 (the previous run of the same series still
decaying), the multigrid row at 1.65. This is the first session of the plan
whose RULE O precondition held for the AO measurements.
**Manuals read before touching a kernel (RULE 5):** `INDEX.md`,
`profiling_tools.md`, `16_profiling_and_bottleneck_identification.md`
(§2.4 the portable timing harness, §2.6 the CPU runtime's execution model),
`06_vectorization.md`, `Cubecl_dynamic_vectorization.md`,
`07_memory_coalescing.md` §2-3, `Cubecl_conditionals.md`,
`plane_alignment.md`, `Backend-Agnostic_Buffer_Slicing…` (the in-kernel
offset idiom K-09 uses), `05_lazy_execution.md` (why host spans lie).

---

## 0. What this session did, in one table

| step | result |
|---|---|
| **A-04** (new): per-point, per-shell `rcut` test inside the three AO kernels | landed behind `PYSCF_PBC_AO_POINT_SCREEN=1`, **default OFF** — measured ≤ 5 % on every row, §1.1 (RULE S: drops terms, no speed ratio above 1.0) |
| the AO kernel's cost, decomposed by three measurement arms | **MEASURED**, §1.2 — the arithmetic is ~0; the exponential is ~0; the pass is K-08's accumulator traffic (72-84 %) |
| **K-09** (new): image-batched Bloch accumulation, one read-modify-write of the k-point planes per batch instead of per image | landed, **bit-exact** (`tests/eval_ao_image_batch.rs`), §1.3-1.4; kill switch `PYSCF_PBC_AO_IMAGE_BATCH=1` |
| **M-13** (new): v2 batch geometry — one `u32` per instance occurrence instead of 40 B; per-block reach lists released once batched; the reverse read-back folded without its `Vec` copy | landed, **bit-exact** (`multigrid_batch` 4/4), §2 |
| measurement arms kept in the tree, all garbage-output and env-gated | `PYSCF_PBC_AO_EXP={fast,none}`, `PYSCF_PBC_AO_RCUT2_OVERRIDE=<r²>`, `PYSCF_PBC_AO_SKIP_K08=1` — §1.2 explains what each isolates |

Every number below is `MEASURED (source)` unless marked otherwise.

---

## 1. GATE AO — the cold periodic AO pass

### 1.1 A-04 — the per-point screen does nothing here (REFUTED as a speed item)

`krks_profile ao --cell si --nk 2,2,2 --mesh 31,31,31`, same binary, block
screen on in both arms (`scratchpad/ab1`):

| row | point screen OFF | point screen ON | ratio |
|---|---|---|---|
| gth-szv deriv 0, cold `eval_ao_kpts` | 1 820 ms | 1 769 ms | 1.03 |
| gth-szv deriv 1 | 5 027 ms | 5 163 ms | 0.97 |
| gth-dzvp deriv 1 | 17 828 ms | 17 396 ms | 1.02 |

The model behind A-04 (plan-2 §2.1 and this session's own first reading —
"a kept 128-point block still holds many points outside every shell's cutoff
sphere") is wrong for these cells: `estimate_rcut_for_eval` returns ~26 bohr
at the default precision (a sphere of ~270 cells' volume), so a kept block is
almost entirely inside some shell's reach and the per-point test removes
nothing. Recorded as an erratum against the module doc's opposite claim too —
the doc argued per-element skips were forbidden by branch divergence; neither
claim was measured before this session. Its arithmetic footprint IS
measurable: the whole-table deltas are 2.4e-13 (szv) / 7.3e-13 (dzvp) against
block-only (`tests/eval_ao_point_screen.rs`), and a converged multigrid
`KRKS` energy moves by 6.3e-13 with it on (§2).

### 1.2 Where the AO pass's time actually goes — three arms, one binary

Same fixture, `deriv 0` / `deriv 1`, gth-szv (`scratchpad/ab3`, `ab4`, `ab5`):

| arm | what it removes | cold pass, deriv 0 | cold pass, deriv 1 |
|---|---|---|---|
| reference | — | 1 915-1 946 ms | 5 102-5 217 ms |
| `PYSCF_PBC_AO_EXP=fast` | the table-based exact `exp` → the degree-13 series | 1 889 ms | 5 120 ms |
| `PYSCF_PBC_AO_EXP=none` | the exponential entirely | 1 782 ms | 5 078 ms |
| `PYSCF_PBC_AO_RCUT2_OVERRIDE=0` | ALL of the kernel's arithmetic (every lane takes the zero-fill arm) | 1 773 ms | 5 194 ms |
| `PYSCF_PBC_AO_SKIP_K08=1` | the Bloch accumulate | **541 ms** | **826 ms** |
| `SKIP_K08` + `RCUT2_OVERRIDE=0` | accumulate AND arithmetic | 404 ms | 978 ms |
| `SKIP_K08` + block screen OFF | accumulate (1331 dense launches) | — | 850 ms |

Read off the table:

* **The AO kernel's arithmetic is not the cost.** Removing the exponential,
  or every operation in the lane, moves the pass by under 4 %. Session 3's
  "46 ns per lane, ~15 software `exp`s" (§5.1 there) was a per-lane
  attribution of a span that was measuring something else.
* **K-08 is 72 % (deriv 0) and 84 % (deriv 1) of the cold pass**, not the
  20-23 % its host span reported. `lazy-launches-blur-stage-spans` named the
  mechanism; this is its size. What K-08 pays for is the read-modify-write of
  BOTH `(nkpts, n)` accumulator planes on EVERY image — `4·nkpts·n` reals of
  traffic to fold `n` reals in: at deriv 1, 122 MB per image × 454 images =
  55 GB through memory for a 7.6 MB block each.
* A-03 (vectorising the AO kernel's grid axis) is therefore **closed without
  implementation**: there is nothing in the kernel worth vectorising on this
  runtime. Its DEFER-UNTIL clause asked exactly this question.

### 1.3 K-09 — image-batched Bloch accumulation (**bit-exact**)

`crates/pyscf-kernels/src/pbc/eval_ao_k.rs` (`eval_ao_k_accumulate_batch_kernel`,
`AoImageBatch`, `AoKAccumulator::accumulate_batch`),
`crates/pyscf-kernels/src/eval_gto.rs` (`eval_gto_sph_into_target`,
`eval_gto_sph_deriv1_into_target`, `eval_gto_device_capable`),
`crates/pyscf-pbc-gto/src/eval_gto.rs` (the image loop).

The AO kernels write each image's block into a fixed-stride slot of one
device buffer (`Handle::offset_start`, the slicing manual's idiom); after
`B` images one launch folds them all: per element `p`, gather the value from
each image (dense at `p`; screened through the image's inverse index
`pos[m·ngrids + g]`, absent when not kept), then per k `acc = out[k,p]; acc +=
pr[m,k]·v_m` over the images **in image order**; store. Each `(k, p)` receives
exactly the additions the per-image launches performed, in the same order —
an un-kept element received none from that image and receives none here — so
the planes are bit-identical. RULE T: per image the accumulator traffic drops
from `4·nkpts·n` to `4·nkpts·n / B`, plus `n` (the block, unchanged) and
`ngrids/2` (the inverse index). `B` is sized under a 256 MiB slot budget and
capped at 16 (`AO_IMAGE_BATCH_MAX`, the kernel's register width):
16 at gth-szv, 10 at gth-dzvp deriv 1. `PYSCF_PBC_AO_IMAGE_BATCH=1` is the
pre-K-09 path exactly (the dense vectorised K-08 and the K-08b scatter
kernel are untouched and still serve it).

**Bit-parity: EXACT**, asserted by `tests/eval_ao_image_batch.rs` — batch 1
vs 16 vs 3 (ragged tail), `GTOval_sph` and `_deriv1`, gth-szv and gth-dzvp,
screened (dense + gathered images) in-process and unscreened (1331 dense
images) in a child, whole tables at `to_bits()`.

### 1.4 K-09 measured — same binary, `PYSCF_PBC_AO_IMAGE_BATCH=1` vs batched

`krks_profile ao --cell si --mesh 31,31,31`, block screen on, load 4.2-11.9
(this series was run back to back, so the later rows carry their own
predecessors' decaying load; the A/B pairs are adjacent and same-binary —
`scratchpad/ab6`):

| row | `B = 1` (pre-K-09 kernels) | batched | `B` | **ratio** | peak RSS `B=1` → batched |
|---|---|---|---|---|---|
| gth-szv deriv 0, 2×2×2 | 1 873 ms | **1 062 ms** | 16 | **1.76×** | 260 → 317 MiB |
| gth-szv deriv 0, 2×2×2 | 1 873 ms | 1 193 ms | 8 | 1.57× | 260 → 307 MiB |
| gth-szv deriv 1, 2×2×2 | 5 098 ms | **2 364 ms** | 16 | **2.16×** | 521 → 800 MiB |
| gth-szv deriv 1, 2×2×2 | 5 098 ms | 2 941 ms | 8 | 1.73× | 521 → 738 MiB |
| gth-dzvp deriv 1, 2×2×2 | 18 043 ms | **9 522 ms** | 10 (budget) | **1.89×** | 1 306 → 2 074 MiB |
| gth-szv deriv 1, **4×4×4** | 32 930 ms | **8 909 ms** | 16 | **3.70×** | 2 955 → 3 243 MiB |

* The gain grows with the k-point count, as the RULE-T model says it must:
  the traffic K-09 divides by `B` is `4·nkpts·n`, the traffic it leaves is
  `n`. At 64 k-points the accumulate WAS the pass (session 3 §5.7 measured
  the cold tables at 92 % of a 4×4×4 SCF), and it is now under a quarter of
  what it was.
* `B = 16` beats `B = 8` by a further 1.2×; the cap is the kernel's
  per-lane register array (`AO_IMAGE_BATCH_MAX`), and the knee has not been
  found — a 32-wide arm is a one-constant experiment for the next session.
* The RSS increase is the staged blocks (`B · n · 8` B: 61 MB at szv deriv
  1, 250 MB at dzvp) plus what the pool retains; `AO_IMAGE_BATCH_BUDGET_BYTES`
  is the dial. The dzvp row's +770 MiB is larger than its 250 MB of slots —
  the pool's retention of the freed per-image blocks (session 3 §4.1 saw the
  same behaviour) is the likely remainder and is UNVERIFIED.
* On the GPU runtime the same change removes `(B−1)/B` of the accumulator's
  global read-modify-writes per image; the gain there is UNVERIFIED (RULE G).

### 1.5 K-09 on a whole SCF — GATE S rows, same binary

`krks_profile ksymm --driver krks --cell si --nk 2,2,2 --mesh 31,31,31 --xc pbe`
(one process runs the full-BZ KRKS and the k-symmetric KRKS), load 1.7 / 2.6
(`scratchpad/ksymm/krks-222-pbe-{b1,def}.json`):

| | `B = 1` | K-09 default (`B = 16`) | **ratio** |
|---|---|---|---|
| full-BZ `kernel()` | 7 848 ms | **4 534 ms** | **1.73×** |
| ksymm `kernel()` | 7 057 ms | **3 863 ms** | **1.83×** |
| cold AO tables inside the full-BZ SCF (2 calls) | 6 448 ms | 3 096 ms | 2.08× |
| warm `get_veff`, full / ksymm | 44.8 / 36.2 ms | 42.0 / 35.6 ms | (unchanged, as it must be — K-09 is cold-only) |
| `e_tot`, full / ksymm | −7.785668903669 / −7.785668903525 | identical to every printed digit | bit-exact by gate |
| peak RSS | 615 MiB | 916 MiB | the staged blocks (§1.4) |

The cold AO tables are now 68 % of a 4.5 s pure-PBE SCF (were 82 % of 7.8 s);
what is left of them is the AO kernel launches themselves (§1.2's
`SKIP_K08` arm: 0.54 + 0.83 s) plus `4·nkpts·n / 16` of accumulate traffic.
The next AO lever is the first one this plan has ever had that is NOT the
accumulate: the per-launch fixed cost of the CPU runtime (§1.2's floor arm,
0.4-1.0 s for 454 launches) — a launch-count item, i.e. batching the AO
evaluation itself across images the same way, which the slot buffer K-09
introduces already makes possible.

---

## 2. GATE MG — M-13, the v2 batch geometry

`crates/pyscf-kernels/src/multigrid_pair.rs`, `crates/pyscf-pbc-dft/src/multigrid/pair.rs`.

Session 3 §6 named the next v2 memory cut: per-instance data (13.8 M × 40 B,
host + device at `25³` level 3) and the host copies beside the device ones.
Three changes, all bit-exact (the kernels read the same values from a
different address; the fold visits the same slots in the same order):

1. **Instance indirection.** A kernel instance reaches many blocks of a chunk
   and was copied (`eta` + centre, 32 B) into every one; the chunk now stores
   each distinct instance once and every occurrence carries a `u32`
   (`inst_ref`). Level 3: 13 829 361 occurrences → 2 278 963 distinct.
2. **`block_sel` released once batched.** The concatenated batches ARE the
   per-block reach lists; holding both was 4 B per concatenated slot of host
   memory (311 MB at level 3). The streaming route (`use_batch = false`, or a
   level over budget) rebuilds a block's list through `block_sel_of`.
3. **Reverse read-back folded from the device bytes** (`integrate_fold` /
   `integrate2_fold`) instead of `read → Vec copy → fold`: the `nslots · 8` B
   copy (622 MB at level 3) is gone from the transient peak.

The instrument gained `batch_uinstances` and `batch_geometry_bytes_total` per
level (`krks_profile multigrid`, `pbc_mg_*_level` spans).

`krks_profile multigrid --driver krks --numint v2 --mesh 25,25,25`, Si
gth-szv LDA, load 1.65 (`scratchpad/mg/v2-25-after-m13.json`):

| | session 3 (§5.2, no-LTO, load 5.2) | this session, after M-13 |
|---|---|---|
| peak RSS | **3 962 MiB** | **3 070 MiB** (−23 %) |
| resident batch geometry, level 3 | — | 622 MB (13.8 M occurrences, 2.28 M distinct, 77.8 M slots) |
| warm `get_veff` | 9 842 ms | 10 035 ms (RULE M: v2 speed untouched, as planned) |
| `e_tot` | −7.160554062714283 (§2, pre-M-12 binary) | −7.160554062713652 |

The `e_tot` difference at 6e-13 was chased down and is NOT M-13: that first
run was on the binary that still had A-04 defaulted ON, and the A-04 screen
reaches this SCF through the FFT pseudopotential's AO table. Re-run on both
binaries with the switch pinned each way: A-04 off → −7.160554062714283 on
both (bit-identical to session 3's pre-M-12 value across two sessions and
three binaries, five runs); A-04 on → −7.160554062713652 on both. So M-13 is
bit-exact at the SCF level too, and A-04 moves a converged `KRKS` energy by
6.3e-13 — one more number behind its default-OFF. The BEFORE arm for RSS was
not re-taken on this box (a same-box BEFORE needs a stash-and-rebuild; §5
lists it as owed).

---

## 3. Gates re-run

No-LTO `target/gate` binaries, as in session 3 (bit-exactness does not depend
on LTO; both arms of every gate are one binary). `cargo test -p pyscf-pbc-dft`
still rebuilds the libxc tree under its own feature set (~25 min; memory
`libxc-release-builds-are-slow`), which is why the multigrid rows were run
first and once.

| gate | result |
|---|---|
| `pyscf-pbc-gto` `eval_ao_image_batch` (**K-09**: batch 1 vs 3 vs 16, `GTOval_sph` + `_deriv1`, gth-szv + gth-dzvp, screened in-process AND unscreened in a child) | **2/2** — 8 tables, every real bit-identical |
| `pyscf-pbc-gto` `eval_ao_point_screen` (**A-04**: point vs block-only vs unscreened at the W-09 bound; thread bit-identity; inert without the block screen; the switch demonstrably fires) | **2/2** — worst deltas 2.440e-13 (szv) / 7.325e-13 (dzvp) against a 1e-11 gate |
| `pyscf-pbc-gto` `eval_ao_stages` (GATE AO: thread bit-identity, screen ≤ 1e-11, K-08b byte-equal) on the K-09 default | **2/2** — screen delta 3.942e-14 |
| `pyscf-pbc-gto` `eval_ao_screen` (W-09) | **3/3** |
| `pyscf-pbc-dft` `multigrid_batch` (**M-13**: batched vs streamed — the streamed arm now rebuilds its reach lists through `block_sel_of` — resident vs plain, fused vs single) | **4/4**, Si + diamond, 0e0 |
| `pyscf-pbc-dft` `multigrid_level_cache` (M-11) | **1/1** |
| `pyscf-pbc-dft` `multigrid_uks` (M-10: v1/v2 vs reference, closed-shell identity) | **5/5** |
| `pyscf-kernels`: `multigrid_pair` 8, `pbc_eval_ao_k` 7, `eval_gto_oracle` 3, `eval_gto_lge1` 4 | **22/22** |

Not run this session: GATE A / GATE U (`tests/gate.rs`, `gate_openshell.rs`,
need `PYSCF_ORACLE_VENV=1`), `multigrid2`, `multigrid_threads`,
`krks_ksymm`, `ksymm_*`, `numint_threads`. K-09 and M-13 are bit-exact by
gate (above), A-04 is off by default, and the three measurement arms are
env-gated no-ops when unset, so the full-BZ oracle rows are unaffected by
construction; they should still be re-run before the next baseline capture,
as session 3 also said of itself.

---

## 3.1 P-10 — the owed idle-machine baselines, taken

Plan-2 §3 P-10 asked for mesh-31 ksymm rows for both drivers at `[2,2,2]`
and `[4,4,4]` on an idle box, with the load inside the JSON. Load at launch
2.6-3.0 on every row (the binary's own guard refuses above 4.0);
`.planning/pbc/baselines/2026-09-08-*.json`, K-09 default, A-04 off:

| row | full `kernel()` | ksymm `kernel()` | ksymm/full | warm `get_veff` full / ksymm | cold AO tables (full SCF) | peak RSS |
|---|---|---|---|---|---|---|
| KRKS si 2×2×2 PBE | 4 485 ms | 3 862 ms | 0.861 | 49.2 / 35.8 ms | 3 124 ms | 915 MiB |
| KUKS si 2×2×2 PBE | 4 734 ms | 4 155 ms | 0.878 | 82.0 / 69.1 ms | 3 093 ms | 914 MiB |
| KRKS si 2×2×2 PBE0 | 39 163 ms | 25 700 ms | **0.656** | 6 948 / 3 648 ms | 3 361 ms | 915 MiB |
| KUKS si 2×2×2 PBE0 | 47 134 ms | 26 912 ms | **0.571** | 7 206 / 3 940 ms | 3 211 ms | 920 MiB |
| KRKS si 4×4×4 PBE | **15 027 ms** | **13 329 ms** | 0.887 | 436 / 219 ms | 11 621 ms | 3 654 MiB |
| KUKS si 4×4×4 PBE | 18 395 ms | 15 663 ms | 0.851 | 896 / 450 ms | 12 005 ms | 3 655 MiB |

* Session 3 §5.7's 4×4×4 KRKS row (load 10.8-11.6, before K-09) was 45 750 /
  43 112 ms with 42 s of cold tables; the same row is now 15.0 / 13.3 s with
  11.6 s of cold tables — the whole SCF **3.0× faster**, the cold tables
  3.6×. (Different loads; the same-binary A/B is §1.4-1.5.)
* Under symmetry the hybrids take the `N·N_ibz` band route's 0.5× on
  `get_veff` and land at 0.57-0.66× on the whole SCF; the pure functionals
  are 0.85-0.89× because their SCF is now mostly the cold tables, which S-03
  (still opt-in) is the item to shrink.
* `jk --driver kuks … --compare 2026-09-02-kuks-si222-mesh31-pbe0.json`: `get_k_kpts`
  +6.7 %, `get_j_kpts` +11 % (noise at this scale; the 2026-09-02 run was at
  load 3.6), `nr_rks` −64 % / `nr_uks` −50 % (S-07's band-table reuse, landed
  since), `kernel()` −3.1 %, `e_tot` 8.9e-16. New file
  `2026-09-08-kuks-si222-mesh31-pbe0.json`.
* Multigrid rows for the ledger: v1 25³ `kernel()` 1 680 ms, warm 9.5 ms,
  236 MiB; v2 25³ 31 071 ms, warm 9 722 ms, **3 067 MiB**. AO row:
  `2026-09-08-ao-si222-mesh31-szv-deriv1.json`, cold 2 539 ms, K-08 span
  4.8 % (with K-09; the span is still a lower bound).

## 4. Refuted, closed, and re-attributed

* **A-04 refuted** as a speed item (§1.1); kept off, gated, for a GPU
  measurement.
* **A-03 closed** without implementation (§1.2): the AO kernel's arithmetic
  is not where the cold pass's time goes on this runtime.
* **Session 3 §1 / §5.1's stage attribution corrected**: K-08 was 72-84 % of
  the cold pass, `eval_gto` under 20 %; both host spans mis-attributed by the
  lazy runtime exactly as `lazy-launches-blur-stage-spans` warned. Every
  future stage claim for a launching stage needs a kill-switch arm, not a span.

## 5. Not done, and why

* **A per-launch AO evaluation batch** (one `eval_gto` launch over `B` images'
  shifted grids). §1.2's floor arm says the AO kernels' remaining cost is
  per-launch, not per-lane; K-09's slot buffer is the natural target. Not
  started: RULE O says one change per measurement, and K-09 was this
  session's.
* **`AO_IMAGE_BATCH_MAX = 32`** — a one-constant experiment; `B = 16` beat
  `B = 8` by 1.2× and the knee is not located. The kernel's per-lane register
  arrays are the only cost.
* **The batch's RSS footprint** (§1.4: +60 MB to +770 MB) — the budget dial
  exists (`AO_IMAGE_BATCH_BUDGET_BYTES`, 256 MiB); whether the pool retains
  more than the slots is UNVERIFIED.
* **M-13's same-box BEFORE arm** — the 3 962 MiB comparison row is session
  3's (no-LTO, load 5.2). A stash-and-rebuild BEFORE on this box is owed
  before the −23 % is quoted outside this record.
* **v2 memory, next cut** — `out_integrate` (`nslots · 8` B resident per
  chunk, 622 MB at level 3) and a device-side fold that would drop the host
  `slot_global` copy; and `batch_bytes` still models the pre-M-12/M-13 layout
  (conservative: more chunks than needed, bit-neutral).
* **S-03 default, GATE A / GATE U** — unchanged from session 3 §6.
* **GPU arm of everything above** — UNVERIFIED (RULE G): every kernel here is
  shaped by `launch_1d`, but no GPU with f64 was available.
