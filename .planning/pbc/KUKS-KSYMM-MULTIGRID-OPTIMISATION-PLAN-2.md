# KUKS + k-symmetry + multigrid — Memory & Speed Optimisation Plan, session 2

**Created:** 2026-09-03
**Successor to:** [`KUKS-KSYMM-MULTIGRID-OPTIMISATION-PLAN.md`](./KUKS-KSYMM-MULTIGRID-OPTIMISATION-PLAN.md)
(session 1, executed 2026-09-02 — record in
[`KUKS-KSYMM-MULTIGRID-EXECUTION-SUMMARY.md`](./KUKS-KSYMM-MULTIGRID-EXECUTION-SUMMARY.md)).
Session 1 is **not superseded**: its rules, gates and item numbering are
inherited verbatim, its open items (S-03, S-04, M-01, M-04 step 3, S-02 step 4)
are re-scheduled here with their original numbers, and every new item below
continues the numbering (**A-** is new: the AO-evaluation track session 1
recorded as out of scope and this plan pulls in, because it is the measured
headline).
**Target:** the same three surfaces — `Kuks`, the `KsymAdapted*` adapters, the
two multigrid drivers — plus the one surface session 1 measured to be the
actual cost of a pure-functional periodic SCF: `eval_ao_kpts` and the cubecl
kernels under it.
**Backend note:** `pyscf-algebra` defaults to CubeCL's **CPU runtime**
(`crates/pyscf-algebra/Cargo.toml:16`, `default = ["cpu"]`), and this machine's
ROCm iGPU has no f64, so `PYSCF_BACKEND=rocm` resolves to CPU. Every kernel
item here is therefore written to be measured on the CPU runtime and *reasoned*
for a GPU; GPU speed claims are marked UNVERIFIED throughout and the kernels
are shaped by `pyscf_algebra::launch` so they are correct on both.
**Audience:** an execution agent that follows instructions literally and does
NOT infer.

---

## 0. HOW TO EXECUTE THIS PLAN

Inherits every rule of session 1 §0 (RULE 4, 5, 6, O, U, V, K, M, S, D-PBC-17)
and of `AGENTS.md`. Restated because they gate every item below:

* **RULE 5** — read the named cubecl manual sections (§7) BEFORE touching a
  kernel; on ANY cubecl build error read
  `/home/user/Documents/workspace/cubecl_manual/manual/cubecl_error_guideline.md`
  before touching code. Kernels are `<F: Float + CubeElement>` except where the
  file header documents the `exp`-only `f64` exception
  (`multigrid_pair.rs`, `multigrid_collocate.rs` use `cube_math::double::exp`).
* **RULE 6** — no `cubecl-*` dependency outside `pyscf-algebra` /
  `pyscf-kernels`; cubecl `Handle`s cross the wall only as PRIVATE fields of an
  opaque type (`AoKAccumulator` is the precedent). Run
  `cargo run -p xtask --bin check-dependency-wall` after every item.
* **RULE O** — one change, one re-measure, machine idle (`uptime` load average
  under ~4 on 16 cores, printed in the report). Session 1's mesh-31 ksymm rows
  were REJECTED on this rule; nothing here is quoted off a loaded machine.
* **RULE V** — every number is `MEASURED (source)`, `MODELLED`, `VERIFIED`
  (read in the source, with the line) or `UNVERIFIED`.
* **D-PBC-17** — every reduction reaching an energy goes through
  `oracle_sum`/`oracle_dot`; thread-count bit-identity is asserted, not argued.

Two rules are new to this plan.

* **RULE T — a transfer removed is measured as bytes, not as time.** The CPU
  runtime's "upload" and "read-back" are host memcpys; a discrete GPU's are
  PCIe copies. An item that hoists a buffer states the bytes it stops moving
  per call (a number derivable from the shapes, VERIFIED), and reports the CPU
  wall-time delta as the *lower bound* of the GPU gain, not as the gain.
* **RULE G — a kernel shaped for a GPU is pinnable on the CPU.** Any kernel
  that branches on `has_planes`, plane width or line size gets an env-var
  override so the GPU arm is *executed and gated for correctness* on the CPU
  runtime (the `PYSCF_GEMM_KERNEL` precedent in
  `crates/pyscf-algebra/src/gemm.rs`). No item ships a GPU arm no test ran.

---

## 1. Scope and the gates

### 1.1 In scope

| surface | code (VERIFIED this session) | crate |
|---|---|---|
| periodic AO evaluation driver | `eval_gto.rs:184-381` `eval_ao_kpts_with_images` (per-image `eval_gto`, host scatter, K-08 accumulate) | `pyscf-pbc-gto` |
| molecular AO kernels it calls once per image | `eval_gto.rs:1255-1310` `launch_eval_gto_s`, `:1700-1780` `launch_eval_gto_general`, `:1785-1861` `launch_eval_gto_deriv1` | `pyscf-kernels` |
| Bloch-phase accumulator | `pbc/eval_ao_k.rs:70-183` kernel + `AoKAccumulator`, `:205-235` `accumulate` (uploads the AO block) | `pyscf-kernels` |
| numint AO cache and block loop | `numint.rs:429-458` `eval_ao` (cache keyed by full k-list), `:577-668` `nr_rks`, `:670-731` `nr_uks` | `pyscf-pbc-dft` |
| star-average of a density | `kpts.rs:1614-1638` `star_grid_ops`, `:1657-1681` `symmetrize_density`, `:1693-1721` `_complex` | `pyscf-pbc-symm` |
| ksymm adapters | `krks_ksymm.rs:155` / `:647` `get_veff_tagged` | `pyscf-pbc-dft` |
| ksymm HF route default | `khf_ksymm.rs:472-478` `JkRoute::{Reference,Band,IbzOnly}` | `pyscf-pbc-scf` |
| multigrid v1 | `multigrid/colloc.rs:53-160` `collocate_level`, `:217-256` `level_rho`, `:268-300` `level_pass2`; `multigrid/numint.rs:223-299` `nr_rks`/`nr_uks` | `pyscf-pbc-dft` |
| multigrid v2 driver | `multigrid/pair.rs:770-840` `build_batched_level`, `:1038-1097` `pairlevel_rho_with`, `:1127-1196` `pairlevel_pass2_with`, `:1361-1427` `nr_rks`/`nr_uks` | `pyscf-pbc-dft` |
| multigrid v2 kernels | `multigrid_pair.rs:724-769` `PairSlotBatch`, `:774-894` the two batched kernels, `:896-982` their launchers | `pyscf-kernels` |
| launch geometry helpers | `crates/pyscf-algebra/src/launch.rs` (`launch_1d`, `line_size_for`, `has_planes`, `upload`) | `pyscf-algebra` |
| the profiler | `crates/pyscf-bench/src/bin/krks_profile.rs` (`jk`, `contract`, `ksymm` modes; no `ao`, no `multigrid`) | `pyscf-bench` |

### 1.2 Out of scope (non-goals)

* **A new periodic AO evaluator that folds the Bloch sum into the radial
  kernel.** PBC-MASTER-PLAN plan 10-04 forbids it ("Do not write a new AO
  evaluator", `eval_ao_k.rs:9-13`); the A- items below make the existing
  two-kernel pipeline cheaper, they do not replace it.
* **k-point multigrid.** Both multigrid drivers stay gamma-only (17-11/17-12
  scope); M-01 refuses `nkpts > 1` with a typed error.
* **Everything the KRKS plan still owns** — W-03/W-04 (device-resident
  `fft_jk`), W-06 (GEMM in numint; measured to LOSE on this backend, see
  `zgemm-dense-loses-to-host-rayon`), W-09 follow-ups. Nothing here routes a
  host contraction through `zgemm_dense`.
* **Changing a gated energy without a re-baselined gate.** Items that change
  arithmetic say so in their heading and ship opt-in (RULE S).
* **U- speed items.** Session 1 §2.1.0a measured every KUKS-specific
  contraction item under 1 % of an SCF; U-04/U-05 keep their KUKS-plan
  sequencing and are not restated. The only KUKS work here is the multigrid
  seam (M-01) and the spin-shared multigrid kernels (M-10), both of which are
  KUKS-*enabling*, not KUKS-contraction, work.

### 1.3 The gates — all inherited, all green as of 2026-09-02

| gate | what | last MEASURED |
|---|---|---|
| **GATE A** | `tests/gate.rs`, 7 oracle rows, KUKS Si 2×2×2 PBE at 6.446e-12 / 1e-11 | 7/7, 2026-09-02 |
| **GATE U** | `tests/gate_openshell.rs`, 9 rows, worst 1.494e-11 / 5e-11 (the `get_nuc` floor) | 9/9, 2026-09-02 |
| **GATE B** | thread-count bit-identity: `numint_threads`, `ksymm_threads`, `multigrid_threads` | all green, 2026-09-02 |
| **GATE C** | `tests/krks_ksymm.rs` IBZ-vs-full, 3.109e-14 / 2.842e-14; GDF row known-failing 1.43e-6 (fidelity, not speed) | 7/7 + ignores |
| **GATE E** | `multigrid.rs` v1 ≤ 2.1e-11, `multigrid2.rs` v2 6.804e-8 (si) / 1.242e-8 (diamond), `multigrid_batch.rs` 0e0 | green |
| **GATE S** | the speed-and-memory ledger in the execution summary — one row per item, BEFORE/AFTER wall, peak RSS (`VmHWM`), residual, bit-exact y/n | ledger exists; rows added by this plan |

Two gates are **new** and are built by items in §3 before anything they score
lands:

* **GATE AO** (A-00) — `eval_ao_kpts` bit-identity and per-stage timing on the
  reference cells; every A- item is scored against it.
* **GATE MG-SCF** (M-01) — a converged `KRKS`/`KUKS` on the multigrid arm vs
  the `KNumInt` arm, energies AND wall time in one table (RULE M's blank row).

---

## 2. Where the time and the memory go — what session 1 settled and what it found

### 2.0 READ THIS FIRST — the ranking, by measurement

| quantity | status | consequence |
|---|---|---|
| **Cold `eval_ao_kpts` on the full mesh** = **6.0-6.4 s** of a 10.1 s pure-PBE KUKS SCF on `si 2×2×2 mesh 31`; **70.4 s of 161.8 s** on `si gth-dzvp` before W-09, 3.24x after | MEASURED (session 1 §2.1.0a; `baselines/README.md`) | **the headline of this plan** (§2.1, track A). Warm quadrature is 39-83 ms; nothing per-iteration competes |
| ksymm `get_veff` = **0.73x** the full-BZ one at a 2.0x fold, both drivers, three agreeing rows | MEASURED (session 1 §S-00) | the XC quadrature still evaluates AOs + density at all `N` points; S-03 is the item, its prize is now a number |
| `unfold_kdms` = 0.044-0.111 ms against a 4.7-103 ms `get_veff` | MEASURED | S-01 steps 2-3 CLOSED; not reopened |
| the FFTDF band route is bit-identical (`max |d| = 0e0`) and computes `N·N_ibz` pairs (64 → 24) | MEASURED (S-02) | default is still `Reference` pending an idle-machine ratio — **S-02 step 4 is owed** (§3) |
| multigrid v1 `get_j` = **2.5x FASTER** than the reference FFTDF route (cold cache) | MEASURED (session 1, within-run ratio) | v1 is the multigrid route worth making cheap; its pass2 is serial (§2.3.2) |
| multigrid v2 `get_j` = **0.025-0.033x** of the reference (30-40x slower) | MEASURED | RULE M stands: v2 items are for the memory shape and `isinstance`, not a promised win |
| M-03's one-launch-per-level fits **only the coarsest level** (9.1 / 19.5 MiB); levels 2-3 exceed 256 MiB and stream 27 / 125 launches | MEASURED (session 1 M-03 finding) | the budget check over-counts instances (§2.3.1); M-07 fixes the count and chunks the batch instead of falling all the way back |
| KUKS/KRKS multipliers: K ×1.034, J ×1.9-2.2, quadrature ×1.3-1.75 | MEASURED | no U- speed item (§1.2) |
| ksymm peak RSS on the gate mesh; any absolute ksymm number at mesh 31 | **STILL UNMEASURED** (load 5-33 all session) | P-10 |
| any GPU number | **UNMEASURABLE HERE** (`rocm-igpu-no-f64`) | RULE G / RULE T |

### 2.1 The AO evaluation — where 6 s goes (VERIFIED by reading, UNMEASURED per stage)

`eval_ao_kpts_with_images` (`crates/pyscf-pbc-gto/src/eval_gto.rs:184-381`),
per lattice image `L` (`nimgs` images, `ls` from `get_lattice_ls` at `rmax`):

1. **Host:** build `shifted: Vec<[f64;3]>` = `coords − L` (`:281-284`, or the
   W-09 `gather_kept` subset at `:290`). `ngrids·3` doubles allocated per image.
2. **Host → kernel crate:** `pyscf_gto::eval_gto` (`crates/pyscf-gto/src/eval_gto.rs:81`)
   re-packs the coordinates into F-order (`:125-128`, another `ngrids·3`
   allocation), resolves the backend (`select_backend()` **per call**, `:135`),
   and calls `pyscf_kernels::eval_gto_sph` / `_deriv1`.
3. **Kernel:** `launch_eval_gto_general` / `_deriv1` / `_s`
   (`crates/pyscf-kernels/src/eval_gto.rs:1734-1744`, `:1820-1829`, `:1276-1286`)
   uploads coords + basis tables, launches with
   **`CubeCount::Static(groups,1,1)` and `CubeDim::new_1d(EVAL_GTO_BLOCK = 256)`**
   (`:1168`), and **reads the whole AO block back** (`:1775`, `:1861`, `:1310`).
   This is the fixed 256-wide cube `launch.rs:3-10` documents as "a reasonable
   shape for a discrete GPU and a pathological one for CubeCL's CPU runtime",
   and it is the ONE kernel in the tree still launched that way on the hot
   path — every other engine went through `launch_1d` in the ALG-03 pass.
4. **Host:** on the screened path, `scatter_kept` writes the sub-block into a
   full-size zero-filled buffer (`:299-305`; `screened.fill(0.0)` is
   `comp·ngrids·nao` doubles per image).
5. **Host → device again:** `AoKAccumulator::accumulate` **uploads the AO
   block** (`eval_ao_k.rs:229`, `upload(c, ao)`) plus the `2·nkpts` phases, and
   launches K-08 — a memory-bound kernel (`:33-36` says so itself).

So per image the AO block `n = comp·ngrids·nao` crosses the device boundary
**twice** (down in step 3, up in step 5) and is zero-filled once on the host,
to feed one `4·nkpts·n`-flop kernel. RULE T bytes at the reference cell
(`comp = 1`, `ngrids = 29 791`, `nao = 8`, `n = 238 328` doubles = 1.9 MiB):
**3.8 MiB of pure transfer + 1.9 MiB of zero-fill per image**, times `nimgs`.
`nimgs` is UNMEASURED — it is printed by nothing; A-00 prints it.

Which of steps 3 and 5 dominates is **UNMEASURED**; the plan does not guess
(RULE O). A-00 attributes, A-01 and A-02 each remove one identified cost and
are scored separately.

### 2.2 k-symmetry — what S-03 needs that session 1 did not write down

* `symmetrize_density` (`kpts.rs:1667-1680`) allocates `terms: Vec<f64>` **per
  grid point** inside a `par_iter` — the exact defect class M-04 step 1 removed
  from `level_rho` (`ngrids` heap allocations per call; 29 791 at mesh 31).
  `symmetrize_density_complex` allocates three per point. It also recomputes
  `grid_xyz` + `rotated_grid_index` per `(op, g)` on every call although the
  permutation depends only on `(KPoints, mesh)`. **S-06** fixes both,
  bit-exact, before S-03 puts this function on the per-cycle path.
* The numint AO cache is keyed by the **full k-list** (`numint.rs:437-441`),
  so under `KSet::Ibz` it holds `N` tables and S-03's `N_ibz` tables would be a
  *second* entry, not a replacement. S-03 step 1 keys the route into the cache
  key (`AoKey` gains the route tag) so the two never coexist.
* GGA needs the rotated gradient (session 1 S-03 step 2); the Cartesian
  rotation is the `l = 1` `Dmat` — port it, do not derive it.

### 2.3 Multigrid — what session 1 left, and three things it did not see

#### 2.3.1 v2: the batch geometry is re-uploaded AND re-cloned on every call (VERIFIED)

`pairlevel_rho_with` (`pair.rs:1067-1075`) builds `PairSlotBatch { slot_coef,
..clone_batch_geometry(&bl.batch) }` — **a full host copy of the cached
geometry** (coords, `point_block`, `block_inst0`, `inst_block`,
`instance_alpha`, `instance_center`, `inst_slot0`, `slot_pow`) — and
`launch_rho_batched` (`multigrid_pair.rs:896-910`) then **uploads all eight
arrays** although only `slot_coef` changed. `pairlevel_pass2_with`
(`:1160-1165`, `:946-958`) does it again with `weight`. Per level per cycle
that is 2 × (one clone + 8 uploads) of geometry that M-02 cached precisely so
it would not be rebuilt — the cache saved the *construction*, not the
*movement*. `11_launch_overhead_and_transfers.md` §2 ("hoist invariant
uploads out of loops") applied verbatim: **M-06.**

The budget guard (`pair.rs:773-774`) bounds instances by `nslots` ("one per
slot"), which session 1 recorded as exact on level 1 and an over-estimate on
finer levels. And when the level does not fit, the driver falls **all the way
back** to one launch per block (27, 125) instead of to the smallest number of
under-budget chunks. **M-07.**

#### 2.3.2 v1: `level_pass2` is serial, and `values` has no memory rule (VERIFIED)

`level_pass2` (`colloc.rs:268-300`) is a plain nested loop —
`add_block` is a closure with no rayon — over `nao_p² · ngrids` products, each
`(ci, cj)` entry its own `oracle_sum`. `level_rho` is parallel over grid
chunks; `pass2` is not parallel at all. The entries are disjoint outputs, so a
rayon split over `(ci, cj)` rows with one `buf` per worker is bit-exact
(the M-04 step 2 argument, per worker instead of per call). **M-09 step 1.**

`collocate_level` materialises `values: (n_slots × ngrids)` per level and M-02
now keeps it for the whole `nr_rks`. Session 1's M-02 step 2 asked for a
`0.25 · max_memory` rule and a `tracing::debug!` of the decision; the landed
code has neither (`multigrid/numint.rs:233`, `:276` call
`collocate_all_levels` unconditionally). At 100 AOs and `65³` the table is
~660 MiB per level (session 1 §2.3.2, MODELLED). **M-09 step 2.**

#### 2.3.3 KUKS on multigrid does every transcendental twice (VERIFIED)

`MultiGridNumInt2::nr_uks` (`pair.rs:1397-1427`) calls
`rho_g_from_pair_levels` once per spin and `pass2_from_full_vg_pair` once per
spin, i.e. **two forward and two reverse launches per level**, and inside the
kernels the `exp(-η·|r-P|²)` at `(point, instance)` (`multigrid_pair.rs:802`,
`:867`) does not depend on the spin — only `slot_coef` (forward) and `weight`
(reverse) do. The v1 driver shares the collocation (`multigrid/numint.rs:276`)
but runs `rho_g_from_level_values` per spin over the same `values`. A kernel
taking two coefficient vectors and writing two outputs evaluates `exp` once
per `(point, instance)` and keeps each channel's arithmetic identical
(bit-exact per channel by construction). This is the spin fusion U-05 wanted,
placed where a transcendental is actually shared. **M-10.**

#### 2.3.4 Kernel access patterns (VERIFIED against `07_memory_coalescing.md`)

| access | kernel line | verdict per the manual |
|---|---|---|
| `coords[p*3 + {0,1,2}]`, lane = `p` | `multigrid_pair.rs:789-791` | AoS, stride 3 — §3 says SoA; bit-exact to change (a load, not an operation) |
| `slot_pow[slot*3 + {0,1,2}]` as three `u32` | `:806-808`, `:870-872` | 12 bytes per slot where one `u32` (`ix | iy<<8 | iz<<16`) or three `u8` carry the same — §5 "narrow the element type"; bit-exact |
| `out[slot] = out[slot] + …` per grid point, lane = instance | `:890` | read-modify-write to global inside the `g` loop; accumulate in a register per slot and store once per `(inst, slot)` — the sum ORDER over `g` is unchanged, so bit-exact; on a GPU it removes `npoints` global RMWs per slot |
| `ao[p]`, `out_re[i]` as `Vector<F,N>`, lane = `(k, p_line)` | `eval_ao_k.rs:81-92` | already coalesced and vectorised; nothing to do |

**M-08** bundles the three bit-exact changes; the arithmetic (`poly` via
`while` loops over the powers, `exp` per instance) is NOT touched — session 1
measured `exp` was not the sink, and reordering the monomial products would
not be bit-exact.

---

## 3. Work items

Prefix key: **P-** precision/instrumentation, **A-** AO evaluation (new),
**S-** k-symmetry, **M-** multigrid, **U-** KUKS. Each item states FILES,
WHY, STEPS, BIT-PARITY, TEST, and what it is measured against. Items carried
from session 1 keep their number and say what changed.

### P-10 — The owed idle-machine baselines (**do this first; measurement only**)

**FILES** `.planning/pbc/baselines/` (new JSON), `krks_profile.rs` (one guard).

**WHY** §2.0 rows 9-10. Session 1 rejected its own mesh-31 ksymm rows on RULE
O. Every S- item below is scored against a ksymm baseline that does not exist
at the gate mesh, and the S-02 default flip is blocked on it.

**STEPS**

1. Add to `krks_profile`: refuse `--json <path under baselines/>` when the
   1-minute load average (`/proc/loadavg`) exceeds 4.0, with the value in the
   error. A rejected row must not be *able* to land in the baseline directory.
2. On an idle machine (`uptime` printed): `krks_profile ksymm --driver
   {krks,kuks} --cell si --nk 2,2,2 --mesh 31,31,31 --xc {pbe,pbe0}` and
   `--nk 4,4,4 --xc pbe`. Record `get_veff` warm, `kernel()`, `VmHWM`, and the
   `ksymm_over_full_*` ratios in the GATE S ledger.
3. `krks_profile jk --driver kuks … --compare
   baselines/2026-09-02-kuks-si222-mesh31-{pbe,pbe0}.json` to confirm the
   session-1 KUKS baseline reproduces within noise before anything changes.

**DONE** when the ledger has MEASURED mesh-31 rows for both ksymm drivers at
`[2,2,2]` and `[4,4,4]`, and the JSON files are in `baselines/` with the load
average inside them.

---

### A-00 — GATE AO: the AO-evaluation instrument and bit-identity gate (**prerequisite for every A- item**)

**FILES** `crates/pyscf-bench/src/bin/krks_profile.rs` (new `ao` mode),
`crates/pyscf-pbc-gto/tests/eval_ao_stages.rs` (new).

**WHY** §2.1 — the largest measured cost in a pure-functional SCF has no
per-stage attribution: `nimgs`, per-image kernel time, per-image transfer
time, host scatter time and K-08 time are all UNMEASURED. RULE O forbids
A-01/A-02 without it.

**STEPS**

1. `krks_profile ao --cell … --nk … --mesh … --deriv {0,1}
   [--screen {on,off}]`: time `eval_ao_kpts` cold, and inside it (behind a
   `tracing` span per stage, read by the bench through a subscriber, so
   production code gains spans and no timers) the four stages of §2.1:
   `shift+pack`, `eval_gto` (kernel incl. its own upload/read), `scatter`,
   `k08_accumulate`. Report `nimgs`, `n`, RULE-T bytes per image (`2·n·8` for
   the round trip, `n·8` for the zero-fill), `VmHWM`, load average.
2. Baseline cells: `si gth-szv [2,2,2] mesh 31` (the KUKS baseline cell),
   `si gth-dzvp [2,2,2] mesh 31` (W-09's large-cell baseline), both `deriv 0`
   and `deriv 1`, screen on and off. Record in the GATE S ledger.
3. `eval_ao_stages.rs`: `eval_ao_kpts` output `to_bits()`-equal across
   `RAYON_NUM_THREADS ∈ {1, 8}` (GATE B for this driver; never asserted before
   — `pbc_eval_ao_k.rs` covers the kernel, not the image loop), and equal
   with `PYSCF_PBC_AO_SCREEN=0/1` where W-09's own gate says it must be.

**DONE** when a table exists with a MEASURED share for each of the four stages
on both cells.

---

### A-01 — Route the three AO kernel launchers through `launch_1d` (**bit-exact**)

**FILES** `crates/pyscf-kernels/src/eval_gto.rs` (`:1255-1310`, `:1700-1780`,
`:1785-1861`).

**WHY** §2.1 step 3. `EVAL_GTO_BLOCK = 256` (`:1168`) with
`CubeCount::Static(groups)` is the fixed GPU cube `launch.rs` was written to
retire; on the CPU runtime it dispatches 256 units per cube for
`ngrids·nbas` lanes of a few hundred flops each. The kernel is called `nimgs`
times per cold evaluation. Memory `pyscf-algebra-cpu-is-default-backend`
records 16.9 s → 0.058 ms for the same shape change on a GEMM; this kernel is
not that pathological (no barriers), so the gain is UNVERIFIED until A-00
measures it.

**READ FIRST** `10_grid_stride_occupancy.md` §3, `11_launch_overhead_and_transfers.md`
§4 (the kernels are already `launch_unchecked`; keep the host-side checks that
justify it).

**STEPS**

1. Replace the three `(CubeCount::Static(groups,1,1), CubeDim::new_1d(EVAL_GTO_BLOCK))`
   pairs with `launch_1d(client, lanes, work_per_lane)`, `lanes` = the same
   `out_len` / `ngrids·nbas` the `groups` were computed from, `work_per_lane`
   = a constant per kernel stating the per-lane flop count (primitive count ×
   `exp` ≈ 100 per primitive — write the estimate in a comment the way
   `ft_aopair.rs:208-217` does). Delete `EVAL_GTO_BLOCK`.
2. The kernels' own `if idx < len` guards stay; `launch_1d` rounds the lane
   count up exactly as `calculate_cube_count_elemwise` did.
3. RULE G: nothing here branches on the device, so no override is needed.

**BIT-PARITY** **EXACT** — every lane computes one output independently of
the cube shape (`10_grid_stride_occupancy.md` §2, "disjoint writes are
bit-exact on any geometry"). Assert, do not argue: `eval_gto_oracle.rs` and
`eval_gto_lge1.rs` unchanged, plus A-00's bit-identity test.

**TEST** `pyscf-kernels` `eval_gto_*` suites unchanged; `pyscf-gto` and
`pyscf-dft` molecular gates unchanged (this kernel is shared with the
molecular code — GATE A of the molecular plan applies); GATE A / GATE U
unchanged to printed residuals.

**MEASURED AGAINST** A-00's `eval_gto` stage, cold `nr_rks` in the KUKS
baseline JSON (`--compare`).

---

### A-02 — Keep the AO block on the device between `eval_gto` and K-08 (**bit-exact; RULE T item**)

**FILES** `crates/pyscf-kernels/src/eval_gto.rs`,
`crates/pyscf-kernels/src/pbc/eval_ao_k.rs`, `crates/pyscf-kernels/src/pbc/mod.rs`,
`crates/pyscf-pbc-gto/src/eval_gto.rs`, `crates/pyscf-gto/src/eval_gto.rs`.

**WHY** §2.1 steps 3-5: the AO block goes device → host → device per image and
is zero-filled on the host on the screened path. Both kernels live in
`pyscf-kernels`, so the handle never crosses the ALG-06 wall.

**READ FIRST** `11_launch_overhead_and_transfers.md` §2-3,
`13_memory_preallocation.md` §1-2,
`Backend-Agnostic_Buffer_Slicing_and_Multi-Logical_Array_Allocation.md`
(one resident buffer, sub-ranges addressed by offset).

**STEPS**

1. `pyscf-kernels`: `eval_gto_sph_into` / `_deriv1_into` variants that write
   into a caller-supplied resident output handle (an opaque `AoBlockDevice`
   struct — private `Handle`, public `shape()`; the `AoKAccumulator` pattern)
   and do NOT `client.read`. The public `eval_gto_sph` becomes
   `_into` + one read, so the molecular callers are unchanged and bit-exact by
   construction.
2. `AoKAccumulator::accumulate_device(&mut self, client, ao: &AoBlockDevice,
   pr, pi)` — the existing launch on the resident handle; `accumulate` (slice)
   stays for the tests and the single-shot API.
3. **Screened images (W-09):** the kept sub-block is `index.len()·nao·comp`
   long and lands in a full-size buffer by host scatter. Replace the scatter
   with a scatter-accumulate variant of K-08: one lane per
   `(k, kept_row_line)`, `out[k·n + (c·ngrids + index[j])·nao + a] += phase·sub[…]`.
   The accumulators for non-kept rows are simply not touched. Bit-parity
   argument: the host path added `phase · 0.0` to those rows, and
   `x + (+0.0) == x` with the same bit pattern for every finite `x` and for
   `x = +0.0` (accumulators start at `+0.0` and `pr·0.0` is `±0.0`; `+0.0 +
   (−0.0) = +0.0`), so the untouched value equals the old result bit for bit.
   **Assert it** (A-00's test with screen on), do not rely on the argument.
4. `select_backend()` is called per `eval_gto` call (`pyscf-gto/src/eval_gto.rs:135`)
   — resolve once in `eval_ao_kpts_with_images` and pass the client down (the
   `_into` API takes it). The two F-order re-packs per image become one
   resident `coords` handle plus a `shift_coords(coords, L)` kernel writing
   into a second resident buffer (`ngrids·3` doubles, allocated once); with
   W-09 on, `gather_kept` needs the kept-block index list — upload the block
   boxes once and let the shift kernel take `index`. If the kernel form costs
   more than it saves on the CPU runtime (A-00 measures), keep the host shift
   and only hoist the allocation (`13_memory_preallocation.md` §1).

**BIT-PARITY** **EXACT** — steps 1, 2, 4 move bytes, not operations; step 3
is argued above and asserted. If A-00's bit-identity test moves, stop.

**TEST** A-00's gate; `pbc_eval_ao_k.rs` gains
`device_path_matches_slice_path_bit_exact` (screened and unscreened);
`eval_ao_screen.rs` unchanged; GATE A/U unchanged.

**MEASURED AGAINST** A-00's `scatter` + `k08_accumulate` stages and RULE-T
bytes (must go from `3·n·8` per image to `≈ 0` + `2·nkpts·8`); cold `nr_rks`
via `--compare`. State the CPU delta as the GPU lower bound.

---

### A-03 — **DEFERRED:** vectorise the AO kernels' grid axis

**WHY** the general/deriv1 kernels evaluate one `(grid point, shell)` per lane
with scalar loads; `06_vectorization.md` / `Cubecl_dynamic_vectorization.md`
would put `N` adjacent grid points in one `Vector<F, N>` as K-08 already does.

**DEFER UNTIL** A-00 shows the `eval_gto` stage is still the largest AFTER
A-01 and A-02, on the CPU runtime. The transform is bit-exact per element but
it re-shapes a 2 500-line kernel with an `exp`-only `f64` exception, and
RULE O says do not touch it on a guess.

---

### S-06 — `symmetrize_density`: no per-point allocation, cached permutation (**bit-exact; prerequisite for S-03**)

**FILES** `crates/pyscf-pbc-symm/src/kpts.rs` (`:1614-1721`),
`crates/pyscf-pbc-symm/tests/kpts_transform.rs`.

**WHY** §2.2 bullet 1.

**STEPS**

1. `par_chunks_mut(CHUNK)` over grid points with ONE `terms` buffer per chunk
   (the M-04 step 1 idiom); `oracle_sum` per point over the same
   `star_grid_ops` order. Same for the complex variant (three buffers).
2. Cache the permutation: `OnceLock<Vec<u32>>` per `(ibz_k_idx, mesh)` on
   `KPoints` (the `addition_table` pattern, `kpts.rs:380-383`) holding
   `rotated_grid_index(rot, ft, mesh, xyz)` for every `(non-identity op in the
   star, g)` — `|star|·ngrids·4` B; 0.95 MiB at mesh 31 for an 8-op star.
   Keyed by `mesh` because the same `KPoints` serves several meshes
   (multigrid levels, `kpts_band` grids).
3. The vector variant S-03 needs (`symmetrize_density_vec`, the `l = 1`
   `Dmat` rotation of the three gradient rows after the permutation) is
   written HERE, gated against a host reference in `kpts_transform.rs`, so
   S-03 lands only the numint side.

**BIT-PARITY** **EXACT** for steps 1-2 (same terms, same order, same
reducer); step 3 is new code with its own test.

**TEST** `kpts_transform.rs` unchanged + `symmetrize_density_cached_matches_uncached_bit_exact`
on `si [2,2,2]`, `[3,3,3]`, meshes `11³` and `31³`; GATE C unchanged.

---

### S-03 — **Opt-in, changes results:** IBZ-costed XC quadrature (carried from session 1; the prize is now MEASURED)

**FILES** `crates/pyscf-pbc-dft/src/numint.rs`, `krks_ksymm.rs`;
`tests/ksymm_symmetrize_rho.rs` (new).

**WHY** session 1 §2.2.4, now with S-00's number: the symmetric `get_veff` is
0.73x of the full one at a 2.0x fold because the AO + density half still runs
at `N` points. This is the only item that shrinks the numint AO cache
(`comp·N·ngrids·nao·16` B → `N_ibz` in place of `N`).

**SEQUENCE** after P-10 (baseline), A-02 (so the AO cost it scales is the
post-A-02 one), S-06.

**STEPS** — session 1's S-03 steps 1-5 unchanged, with two additions:

1. (new) The AO cache key gains the route tag (`AoKey` at `numint.rs:434`),
   and under `Symmetrize` `eval_ao` is called with `self.kpts_ibz()` — the
   full-BZ table must never be built on this route, or the memory claim is
   false. A-00's `ao` mode gains `--driver kuks-ksymm --rho-route symmetrize`
   and reports the cache bytes (`nkpts_cached · comp · ngrids · nao · 16`).
2. (new) `nr_uks` under `Symmetrize`: both spins share one IBZ AO table (as
   they share the full one now); `symmetrize_density` runs once per spin per
   IBZ point.
3. Grid blocking under `Symmetrize`: accumulate the per-`k` real-space `ρ_k`
   over blocks first, then symmetrise once over the full mesh (session 1
   step 3's second option — chosen because it keeps `block_ranges`
   unchanged, so `nelec`/`excsum` partition identically to the `Unfold` route
   and the 1e-11 comparison isolates the star-average).

**BIT-PARITY** **NO** (star-average of rotated copies vs `|star|` independent
evaluations). Opt-in via `PYSCF_PBC_KSYMM_RHO=symmetrize`, default `unfold`.

**TEST** as session 1: `Symmetrize` vs `Unfold` `ρ(r)` at 1e-11 on
`si [2,2,2]`/`[3,3,3]` with `si_precision(1e-10)`, LDA and GGA, `nelec` 1e-12,
converged `e_tot` 1e-11 within ONE driver (RULE K); open-shell row on the
`kuks_ibz_runs_and_stays_symmetric` fixture (RULE U); GATE C re-baselined
separately with the flag on; GATE B (`ksymm_threads`) with the flag on.

**MEASURED AGAINST** P-10's ksymm rows: `get_veff` warm, cold `nr_rks`,
`VmHWM`. MODELLED expectation `× N_ibz/N` on the AO stages — 1/2 at the
`[2,2,2]` fold S-00 measured, 1/8 at `[4,4,4]`.

---

### S-02 step 4 — Measure the band route and flip the default (**measurement + one-line change**)

**FILES** `crates/pyscf-pbc-scf/src/khf_ksymm.rs` (`JkRoute` default).

**WHY** session 1 landed `Band` as bit-identical (`max |d| = 0e0`, 64 → 24
pairs) and left `Reference` the default pending an idle-machine ratio
(D-PBC-26 point 6).

**STEPS** P-10's instrument, `--driver krhf-ksymm --jk-route
{reference,band}` on `si [2,2,2]` and `[4,4,4]` FFTDF and GDF; flip the
default to `Band` only where the measured ratio is `> 1.0`; record the ratio
in the D-PBC-26 erratum block.

**BIT-PARITY** EXACT (measured). **TEST** `khf_ksymm.rs` unchanged.

---

### S-04 — J/K pair-invariant audit (carried, unchanged)

Audit only, as session 1 wrote it; close with the table. No sequencing
constraint.

---

### M-01 — The numint seam and GATE MG-SCF (carried; the multigrid prerequisite)

**FILES** `crates/pyscf-pbc-dft/src/krks.rs`, `kuks.rs`, `numint.rs`,
`tests/multigrid_scf.rs` (new), `krks_profile.rs` (`multigrid` mode).

Session 1's M-01 steps 1-2 unchanged (`enum KsNumInt { Grid, MultiGrid,
MultiGrid2 }`, gamma-only, `xc_with_j` fused, typed refusals for `nkpts > 1`,
hybrids, non-uniform grids; converged `KRKS`/`KUKS` LDA on `small_silicon()`
/ `li_atom_spin1()` multigrid arm vs `Grid` arm at the v1/v2 floors, wall time
in the same table). Two additions:

1. `krks_profile multigrid --driver {krks,kuks} --numint {grid,v1,v2}` times
   per cycle: `build_tasks` (cache hit/miss), per level forward / reverse
   launch count and wall, FFTs, `VmHWM`. This is 17-13 Task 2's blank row and
   Q5's instrument; every M- item below is scored on it.
2. RULE U: the KUKS row is asserted genuinely open-shell
   (`max |dm_a − dm_b| > 1`, the P-01 idiom).

**BIT-PARITY** the `Grid` arm is EXACT (existing code behind an enum).

---

### M-06 — Device-resident batch geometry: upload once per level per SCF, not per call (**bit-exact; RULE T**)

**FILES** `crates/pyscf-kernels/src/multigrid_pair.rs`,
`crates/pyscf-pbc-dft/src/multigrid/pair.rs`, `tests/multigrid_batch.rs`.

**WHY** §2.3.1 first paragraph.

**READ FIRST** `11_launch_overhead_and_transfers.md` §2-3,
`13_memory_preallocation.md` §1, `05_lazy_execution.md` (the reverse launch
may be queued behind the forward one; only the read-back synchronises).

**STEPS**

1. `pyscf-kernels`: `PairSlotBatchDevice` — opaque struct holding the eight
   geometry `Handle`s plus a resident `slot_coef` handle, a resident
   `weight` handle, and the two output handles (`npoints`, `nslots`), all
   allocated ONCE from a `&PairSlotBatch` (`PairSlotBatchDevice::new(client,
   &batch)`). Methods `rho(&self, client, slot_coef: &[f64]) -> Vec<f64>` and
   `integrate(&self, client, slot_coef, weight) -> Vec<f64>`: write the
   per-call arrays into the resident handles (`client.write`/re-`create` —
   whichever the runtime exposes as an in-place write; if only `create` is
   available, ONE upload per call of the varying array is the floor), launch,
   read ONE output. Zero the output on the device (a `fill` kernel or a
   resident zero buffer copied by kernel) rather than uploading `vec![0.0;
   n]` per call (`multigrid_pair.rs:898`, `:945`).
2. `BatchedLevel` gains `device: OnceLock<PairSlotBatchDevice>` built on first
   use; `clone_batch_geometry` and the per-call `PairSlotBatch { .. }`
   construction in `pairlevel_rho_with` / `pairlevel_pass2_with` are deleted.
   The `slot_coef` / `w` gathers (`pair.rs:1069-1073`, `:1166-1170`) stay —
   they ARE the per-call data.
3. RULE T ledger: bytes per level per cycle before (`2 × Σ geometry + 2
   outputs zero-uploads`) and after (`slot_coef + weight + 2 read-backs`),
   from the shapes.
4. Keep the un-resident `collocate_pairs_rho_batched` / `_integrate_batched`
   entry points (the `multigrid_batch.rs` test uses them); the resident path
   must be `to_bits()`-identical to them.

**BIT-PARITY** **EXACT** — same kernel, same lanes, same order; only where
the bytes live changes. Asserted by `multigrid_batch.rs` (resident vs
non-resident, all levels, 0e0).

**TEST** `multigrid_batch.rs` + the new assertion; GATE E unchanged;
`multigrid_threads.rs` unchanged; `multigrid_cache.rs` unchanged (a different
cell must drop the device batch too — add that row).

**MEASURED AGAINST** M-01's instrument, forward + reverse wall per level; v2
`get_j` vs reference (0.025x / 0.033x is the baseline).

---

### M-07 — Exact instance count in the budget, and chunked batches instead of per-block fallback (**bit-exact**)

**FILES** `crates/pyscf-pbc-dft/src/multigrid/pair.rs` (`build_batched_level`).

**WHY** §2.3.1 second paragraph; session 1's own "cheap follow-up".

**STEPS**

1. Count instances by the transitions of `kslot_instance` over `block_sel`
   (one pass, already the loop at `pair.rs:808-821`) and put the exact
   `ninst` in the byte estimate. Re-measure which levels fit on the Gate-E
   cells; record the table (session 1's M-03 finding is the BEFORE).
2. When a level still exceeds `BATCH_BUDGET_BYTES`, build
   `Vec<BatchedLevel>` by greedily grouping consecutive blocks under the
   budget (block order unchanged) instead of returning `None`. The driver
   loops over chunks; the per-block streaming path is kept only for a level
   whose single largest block exceeds the budget on its own (17-12's bound
   says none does on the reference cells; assert it).
3. `BATCH_BUDGET_BYTES` stays 256 MiB; it is resident memory for the SCF's
   life and 17-12's OOM is the precedent. Do NOT raise it.

**BIT-PARITY** **EXACT** — each lane still runs its block's slot list in
table order; a chunk boundary only decides which launch a block is in.
Asserted at `to_bits()` against the per-block route (`multigrid_batch.rs`).

**TEST** `multigrid_batch.rs` extended to levels 2-3 (currently streaming, so
never batch-tested); GATE E unchanged.

**MEASURED AGAINST** M-01's launch count per level (27 / 125 → a handful) and
wall.

---

### M-08 — Kernel access patterns in the batched pair kernels (**bit-exact**)

**FILES** `crates/pyscf-kernels/src/multigrid_pair.rs` (`:774-894`),
`crates/pyscf-pbc-dft/src/multigrid/pair.rs` (`build_batched_level` layout).

**WHY** §2.3.4.

**READ FIRST** `07_memory_coalescing.md` §2-3, §5; `06_vectorization.md`;
`Cubecl_conditionals.md` (no data-dependent branch is added).

**STEPS**

1. Coordinates SoA: `coords_x/y/z` (three `Array<F>`) in `PairSlotBatch`;
   the forward kernel's lane-`p` loads become unit-stride.
2. Powers packed: `slot_pow: Vec<u32>` of `ix | iy << 8 | iz << 16`; the
   kernel unpacks with shifts/masks. 4 bytes per slot instead of 12.
3. Reverse kernel: per `(inst, slot)` accumulate `acc[slot − s0]` in
   registers across the `g` loop and store once. `s1 − s0` is bounded by the
   monomial count of one instance (`≤ 10` for `l ≤ 2` pairs at `L = 4`… state
   the bound from `build_pair_level_table`); use a fixed-size register array
   sized to the maximum the table can produce, checked on the host before
   launch. The order of additions into each `acc[slot]` over `g` is unchanged.
4. RULE G: none of this branches on the device. Vectorising the forward
   kernel over `N` adjacent points (`Vector<F, N>` for `dx,dy,dz,e,acc` with
   scalar broadcast of the instance data) is the natural next step and is
   bit-exact per lane; do it only if M-01's instrument shows the forward
   kernel dominates after M-06/M-07, and pin the width with
   `PYSCF_MG_PAIR_LINE=1` for the CPU correctness run.

**BIT-PARITY** **EXACT** — loads and stores change, no floating-point
operation or its order does. Asserted (`multigrid_batch.rs`, 0e0).

**TEST** `pyscf-kernels/tests/multigrid_pair.rs` unchanged (the un-batched
kernels are not touched); `multigrid_batch.rs`; GATE E.

**MEASURED AGAINST** M-01's per-level kernel wall on the CPU runtime (RULE T
lower bound; the coalescing gain is a GPU claim, UNVERIFIED here).

---

### M-09 — v1: parallel `pass2`, and a memory rule for `values` (**step 1 bit-exact; step 2 bit-exact by default**)

**FILES** `crates/pyscf-pbc-dft/src/multigrid/colloc.rs`,
`multigrid/numint.rs`.

**WHY** §2.3.2. v1 is the multigrid route that beats the reference (2.5x on
`get_j`); its reverse half is serial.

**STEPS**

1. `level_pass2`: rayon over the `(ci, cj)` output entries of each
   `add_block` (disjoint `v_p` entries; collect `(idx, value)` per worker and
   add in `(i, j)` order afterwards so `v_p[idx] +=` keeps its one-term-per-
   level-per-entry shape), ONE `buf` per worker. Each entry's `oracle_sum`
   sees the identical `ngrids` terms in the identical order.
2. The `0.25 · max_memory` rule session 1's M-02 step 2 specified and the
   landed code lacks: if `Σ_levels n_slots·ngrids·8 > 0.25·max_memory`,
   `nr_rks`/`nr_uks` collocate one level at a time and drop it after both
   directions (the pre-M-02 memory shape, but still one collocation per
   level), else keep all levels as now. `tracing::debug!` the decision.
   Below the threshold nothing changes, so the default reference cells are
   bit-exact; above it the SAME `values` feed the same loops — also
   bit-exact, just later.
3. (memory, opt-in, changes results) Grid-chunked streaming of `values` for
   `level_rho` is bit-exact (per-point sums), for `level_pass2` it is NOT
   (the per-entry `oracle_sum` tree would span chunks). Not done here; named
   so nobody does it silently. Trigger: a cell where step 2's fallback still
   OOMs.

**BIT-PARITY** step 1 EXACT (asserted by `multigrid.rs::int_rho_matches_tr_dm_s`
and a new `pass2_parallel_matches_serial_bit_exact`); step 2 EXACT.

**MEASURED AGAINST** M-01's instrument (reverse-direction wall per level,
`VmHWM` with a forced-low `PYSCF_MAX_MEMORY`).

---

### M-10 — Spin-shared multigrid kernels for KUKS (**bit-exact per channel**)

**FILES** `crates/pyscf-kernels/src/multigrid_pair.rs`,
`crates/pyscf-pbc-dft/src/multigrid/pair.rs` (`nr_uks`,
`rho_g_from_pair_levels`, `pass2_from_full_vg_pair`), `tests/multigrid_uks.rs`.

**WHY** §2.3.3.

**READ FIRST** `03_kernel_fusion.md` §2 and §4 (register pressure: two
accumulators per lane, not two kernels).

**STEPS**

1. Forward kernel variant with `slot_coef_a`, `slot_coef_b`, `out_a`,
   `out_b`: `e` and `poly` computed once per `(p, inst, slot)`;
   `acc_a += coef_a·poly·e`, `acc_b += coef_b·poly·e`. Per channel the
   sequence of operations is exactly the single-channel kernel's.
2. Reverse variant with `weight_a`, `weight_b`, `out_a`, `out_b`:
   `e` once, `we_a = weight_a[g]·e`, `we_b = weight_b[g]·e`, two register
   accumulators per slot (M-08 step 3's shape).
3. `MultiGridNumInt2::nr_uks`: one forward sweep per level producing both
   channels' `rho_r`, then two FFTs (unavoidable); one reverse sweep per level
   consuming both `wv` fields. The `PairSlotBatchDevice` (M-06) gains the
   second coefficient/weight/output handles, allocated only when `nr_uks` is
   called.
4. v1 (`multigrid/numint.rs::nr_uks`): the host `level_rho` cannot share
   `(coeff·v_i)·v_j` across spins bit-exactly (the association differs);
   only the loads are shared. Leave v1 alone; say so in the code.

**BIT-PARITY** **EXACT per channel** — asserted three ways in
`multigrid_uks.rs`: fused vs two single-channel launches at `to_bits()`;
`nr_uks(dm/2, dm/2)` vs `nr_rks(dm)` unchanged (session 1's identity, 0 on
every row); the reference-numint comparison at the v2 floor unchanged.

**MEASURED AGAINST** M-01's instrument, `--driver kuks --numint v2`:
forward + reverse wall per level, nset=2 over nset=1 ratio (MODELLED ≈ 1.1x
where the `exp` dominates, vs 2.0x now).

---

### M-04 step 3 — Opt-in radius-screened v1 `pass2` (carried, unchanged)

As session 1 wrote it: `PYSCF_PBC_MULTIGRID_PASS2_SCREEN`, default off,
gate-scored against the v1 floor, lands ALONE. Sequence after M-09 step 1 (the
loop it screens is the one M-09 parallelises).

---

### M-05 — **DEFERRED:** per-Gaussian sub-mesh in v2 (carried)

Trigger unchanged: v2 `get_j` still below 0.3x of the reference AFTER M-06,
M-07 and M-08 are measured, AND Phase 18 needs v2 beyond `isinstance` (Q8).

---

### U- items — none scheduled (§1.2)

U-04 / U-05 stay in the KUKS plan's sequencing (after §8 Q5 / after W-06).
The KUKS-*enabling* work in this plan is M-01 (KUKS on multigrid exists as an
SCF) and M-10 (its kernels are not doubled).

---

## 4. Sequencing

```text
P-10 ──► S-02 step 4          (measurement; needs an idle machine)
  │
  ├──► A-00 ──► A-01 ──► A-02 ──► [A-03 deferred]
  │                        │
  ├──► S-06 ───────────────┴──► S-03 (opt-in, lands ALONE)      S-04 (audit, any time)
  │
  └──► M-01 ──► M-06 ──► M-07 ──► M-08 ──► M-10        M-09 (step 1, then 2)  ──► M-04 step 3 (opt-in, ALONE)
                                                        [M-05 deferred]
```

* **P-10 first** — it is measurement, it lands nothing, and every S- ratio
  and the S-02 flip depend on it. It needs the machine idle; if it is not, do
  A-00 (whose within-run stage shares are load-tolerant) and come back.
* **Instruments before items**: A-00 before any A-, M-01 before any M-. An
  item scored on an instrument that does not exist is session 1's "blank
  row" failure mode.
* **A-01 and A-02 land separately** with their own A-00 print — they remove
  different costs (dispatch vs transfer) and RULE O wants each attributed.
* **Bit-exact items land at most in pairs**, each with GATE A/U/E prints.
  **S-03 and M-04 step 3 land ALONE** behind their flags (the W-08 discipline).
* **The three tracks (A, S, M) are independent** and may run in different
  sessions. Cross-track edges: S-03 after A-02 (so its AO saving is measured
  against the post-A-02 cost); M-10 after M-06 (it extends
  `PairSlotBatchDevice`).

---

## 5. Verification protocol — after EVERY work item

```bash
# 1-2. GATE A / GATE U (oracle)
PYSCF_ORACLE_VENV=1 cargo test -p pyscf-pbc-dft --release --test gate -- --ignored --nocapture
PYSCF_ORACLE_VENV=1 cargo test -p pyscf-pbc-dft --release --test gate_openshell -- --ignored --nocapture
# 3. GATE B — thread-count bit-identity, now including the AO driver (A-00)
cargo test -p pyscf-pbc-dft --release --test numint_threads --test ksymm_threads --test multigrid_threads -- --nocapture
cargo test -p pyscf-pbc-gto --release --test eval_ao_stages -- --nocapture
# 4. GATE C
cargo test -p pyscf-pbc-dft --release --test krks_ksymm -- --test-threads=1
cargo test -p pyscf-pbc-scf --release --test khf_ksymm -- --test-threads=1
# 5. GATE E, incl. the batched/resident/fused kernel identities
cargo test -p pyscf-kernels --release --test multigrid_pair --test pbc_eval_ao_k --test eval_gto_oracle --test eval_gto_lge1
cargo test -p pyscf-pbc-dft --release --test multigrid --test multigrid2 --test multigrid_batch --test multigrid_uks --test multigrid_cache -- --test-threads=1
# 6. GATE MG-SCF (after M-01)
cargo test -p pyscf-pbc-dft --release --test multigrid_scf -- --nocapture
# 7. The lints (RULE 6; check-no-fma already scans pyscf-kernels)
cargo run -p xtask --bin check-dependency-wall
cargo run -p xtask --bin check-orphan-modules
cargo run -p xtask --bin check-no-fma
# 8. Downstream
cargo test --release -p pyscf-gto -p pyscf-dft -p pyscf-pbc-gto -p pyscf-pbc-symm -p pyscf-pbc-scf -p pyscf-pbc-dft -p pyscf-pbc-df -p pyscf-kernels -p pyscf-bench
# 9. Re-profile — ONE variable changed; machine idle (uptime printed); --compare the item's named baseline
cargo run -p pyscf-bench --release --bin krks_profile -- ao        --cell si --nk 2,2,2 --mesh 31,31,31 --deriv 1 --json after.json --compare before.json
cargo run -p pyscf-bench --release --bin krks_profile -- ksymm     --driver kuks --cell si --nk 2,2,2 --mesh 31,31,31 --xc pbe --json after.json --compare before.json
cargo run -p pyscf-bench --release --bin krks_profile -- multigrid --driver kuks --numint v2 --json after.json --compare before.json
```

A GATE S ledger row without a `VmHWM` number is incomplete for A-02, S-03,
M-06, M-07, M-09; a row without RULE-T bytes is incomplete for A-02, M-06.

---

## 6. Risks

| risk | mitigation |
|---|---|
| A-02 step 3's `x + 0.0` argument fails on some element (a NaN/Inf in an AO block, or a `-0.0` accumulator) | asserted at `to_bits()` with screening on; on failure, keep the host scatter and ship steps 1-2-4 only |
| the resident-buffer write API differs between the CPU runtime and the GPU runtimes (`client.write` vs re-`create`) | RULE 5: `13_memory_preallocation.md` and `cubecl_error_guideline.md` first; the floor is one `create` per call of the varying array, still a win over eight |
| M-07's chunked batch changes which lanes share a cube and a GPU-side reduction order | there is none — every output is one lane's private sum; the `to_bits()` test against the per-block route is the proof |
| M-08 step 3's register array is too large for the GPU register file (occupancy) | bound it from the table on the host; if `s1 − s0` can exceed ~16, fall back to the RMW form for those instances (a host-side split, not a device branch) |
| M-10 doubles the per-lane registers of the reverse kernel | measure on the CPU runtime (where it costs nothing) and state the GPU occupancy risk as UNVERIFIED |
| S-03's GGA rotation convention is wrong | the 1e-11 `∇ρ` comparison on `si` (non-symmorphic, stars `[1,3,4]`) catches a wrong sign immediately; not `he_fcc` alone |
| A-01 moves a MOLECULAR gate (the kernel is shared) | the molecular `eval_gto_oracle` / `pyscf-dft` gates are in §5 step 5/8; a bit-exact item cannot move them, so any movement is a bug |
| the machine is loaded when a ratio is taken | P-10 step 1 refuses to write a baseline above load 4.0; every report prints `uptime` |
| cubecl build error in any A-/M- kernel item | AGENTS.md §4 — read the guideline first, no blind fixes, document per its template |

---

## 7. CubeCL manual sections this plan depends on (RULE 5 — read BEFORE the item)

All under `/home/user/Documents/workspace/cubecl_manual/manual/Cubecl/`
(TreeFinder document ids in parentheses for `node_get`).

| section | item |
|---|---|
| `INDEX.md` | every A-/M- item touching `pyscf-kernels` |
| `11_launch_overhead_and_transfers.md` (`ec4c0c16…`) §2 hoist uploads, §3 batch read-backs, §4 `launch_unchecked` contract, §5 collapse launches, §6 re-attribute | A-01, A-02, M-06, M-07 |
| `13_memory_preallocation.md` (`1956a1a5…`) §1 host-side pre-allocation, §2 the pool | A-02, M-06 |
| `Backend-Agnostic_Buffer_Slicing_and_Multi-Logical_Array_Allocation.md` (`f36e622d…`) | A-02 step 4, M-06 (one resident allocation, offsets) |
| `05_lazy_execution.md` (`6dd86ea3…`) | M-06 (forward/reverse launches queue; only `read` synchronises) |
| `10_grid_stride_occupancy.md` (`77eb0b03…`) §2-3 | A-01 (why a fixed 256-cube is wrong on both runtimes), M-07 |
| `07_memory_coalescing.md` (`a89b919c…`) §2-3, §5, §6 | M-08 |
| `06_vectorization.md`, `Cubecl_dynamic_vectorization.md` | A-03 (deferred), M-08 step 4 |
| `03_kernel_fusion.md` (`3014238c…`) §2, §4 | M-10 |
| `Cubecl_conditionals.md`, `plane_alignment.md` | M-08, M-10 (no per-lane data-dependent branch added) |
| `profiling_tools.md`, `16_profiling_and_bottleneck_identification.md` | A-00, M-01's instruments |
| `Handling_Interleaved_Complex_Numbers_in_CubeCL_with_ROCm_Backend.md` | NOT applicable — every kernel here is planar (RULE 8 / D-PBC-02); cited so nobody "fixes" the layout |
| `../cubecl_error_guideline.md` | any build failure |

---

## 8. Open questions

| # | question | who answers |
|---|---|---|
| Q9 | What share of the cold `eval_ao_kpts` is the kernel dispatch (A-01) vs the AO block round trip (A-02) vs the host scatter, and what is `nimgs` on the reference cells? | A-00 |
| Q10 | Does A-02 step 3's scatter-accumulate hold `to_bits()` on the screened path? | A-02 test |
| Q11 | With the exact instance count, do v2 levels 2-3 fit 256 MiB, and how many chunks does M-07 need? | M-07 step 1 |
| Q12 | After M-06/M-07/M-08, where is v2's time — forward kernel, reverse kernel, FFT, or host gathers? (decides A-03-style vectorisation for M-08 step 4 and M-05) | M-01's instrument |
| Q13 | Is the CPU runtime's `create` of a varying array measurably cheaper than the eight geometry uploads it replaces, i.e. is M-06 visible on this backend at all, or only by RULE T? | M-06 ledger row |
| Q14 | Peak RSS of a ksymm KUKS at `si [4,4,4] mesh 31`, `Unfold` vs `Symmetrize` | P-10, then S-03 |
| Q15 | Does `Band` beat `Reference` in wall time on GDF as well as FFTDF? | S-02 step 4 |
| Q2 (carried) | fraction of a ksymm `get_veff` that is the unfold at `[4,4,4]` — expected < 2 % by S-00's mesh-11/31 rows | P-10 |
| Q8 (carried) | which multigrid driver Phase 18 needs beyond `isinstance` | Phase 18's context; gates M-05 |
