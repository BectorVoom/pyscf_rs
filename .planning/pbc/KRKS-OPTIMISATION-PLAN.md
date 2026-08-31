# KRKS Speed & Precision Optimisation Plan — `pyscf-pbc-dft`

**Created:** 2026-08-30
**Target:** `pyscf_pbc_dft::krks::Krks` (`pbc.dft.KRKS`) and everything on its SCF hot path
**Status:** draft — no code written yet. **§2.1 and §2.2 are MEASURED**, not modelled
(measured on an AMD Ryzen AI 7 350, 8 cores / 16 threads, CPU cubecl backend, warm
caches). The harness that produced them was a throwaway pair of examples and has been
**deleted** — reproducing it as a committed, re-runnable baseline is W-00.
**Audience:** an execution agent that follows instructions literally and does NOT infer.

---

## 0. HOW TO EXECUTE THIS PLAN

This plan inherits every standing rule of
[`.planning/pbc/PBC-MASTER-PLAN.md`](./PBC-MASTER-PLAN.md) §0 and of `AGENTS.md`.
The three that bind hardest here:

* **RULE 4 — tests live in separate files.** No `mod tests` at the bottom of a
  production source file. Integration tests go in `crates/<crate>/tests/<name>.rs`.
* **RULE 5 — cubecl.** Read `/home/user/Documents/workspace/cubecl_manual/manual/Cubecl/INDEX.md`
  before writing any kernel. Every kernel is generic over the device float:
  `#[cube(launch_unchecked)] fn k<F: Float + CubeElement>(...)`. On any cubecl build
  error, STOP and read `cubecl_error_guideline.md` first — blind fixes are a protocol
  violation.
* **RULE 6 — the algebra wall (ALG-06).** Only `pyscf-algebra`, `pyscf-runtime` and
  `pyscf-kernels` may name a `cubecl-*` type. Every new device kernel in this plan
  lands in `crates/pyscf-kernels/src/pbc/`, never in `pyscf-pbc-dft` or `pyscf-pbc-df`.
  `xtask/src/bin/check_dependency_wall.rs` enforces it (`cargo run -p xtask --bin
  check-dependency-wall`).

Plus one rule specific to an optimisation plan:

* **RULE O — measure, change ONE thing, re-measure.** Per
  [`11_launch_overhead_and_transfers.md` §6](../../../cubecl_manual/manual/Cubecl/11_launch_overhead_and_transfers.md):
  these levers *move* the bottleneck rather than removing it, so a stale profile picks
  the wrong next step. Every work item below ends with a re-profile against **W-00**.

---

## 1. Scope and the three acceptance gates

### 1.1 In scope

The KRKS SCF iteration, end to end:

| stage | code | crate |
|---|---|---|
| AO collocation on the FFT grid | `eval_ao_kpts` | `pyscf-pbc-gto` |
| `ρ(r)`, `ε_xc`, `V_xc` | `KNumInt::nr_rks` | `pyscf-pbc-dft` |
| `J` | `fft_jk::get_j_kpts` | `pyscf-pbc-df` |
| `K` (hybrids, and every KRHF) | `fft_jk::get_k_kpts` | `pyscf-pbc-df` |
| the 3-D transform under both | `fft` / `ifft` | `pyscf-pbc-tools` |
| the complex primitives under all of it | `zgemm_dense`, `oracle_z*` | `pyscf-algebra` |

### 1.2 Out of scope (non-goals)

* Changing the **result**. This plan is bit-parity-preserving except where a work item
  says otherwise **in its own heading**, and every such item is opt-in behind a flag.
* `KUKS` / `KGKS` / `KRKSpU`. They share `KNumInt` and `fft_jk`, so they inherit the
  wins for free, but their drivers are not touched.
* Multigrid (`pyscf/pbc/dft/multigrid/`). Not ported; a separate milestone.
* New density-fitting algebra (GDF/RSDF tuning). FFTDF only.

### 1.3 GATE A — accuracy must not regress

`crates/pyscf-pbc-dft/tests/gate.rs` is the contract. It must keep passing at its
**current** tolerances, unchanged:

| gate | tolerance |
|---|---|
| `KRKS Si 2×2×2 PBE` / `LDA,VWN` / `PBE0`, `KUKS`, `KRHF` (GTH-pade) | `1e-11` Ha |
| `KRKS He-fcc 2×2×2 PBE` (all-electron control) | `1e-12` Ha |
| `e_nuc` on every case | `1e-12` Ha |

```bash
PYSCF_ORACLE_VENV=1 cargo test -p pyscf-pbc-dft --release --test gate -- --ignored --nocapture
```

The GTH cases floor at ~4e-12 for structural reasons inherited from `get_pp`; the
all-electron He control is what proves 1e-12 is really reachable. **Do not relax either
number to land a work item.** If an item cannot hold the gate, it does not ship.

### 1.4 GATE B — determinism must not regress

**D-PBC-17** (`PBC-MASTER-PLAN.md:247`) says every complex reduction that reaches an
energy, a density matrix or a convergence test goes through the ordered primitives
`oracle_zsum` / `oracle_zdot`, and that the result is **bit-identical** under
`RAYON_NUM_THREADS=1` and `=8`. Every work item must preserve that:

```bash
RAYON_NUM_THREADS=1 cargo test -p pyscf-pbc-dft --release --test gate -- --ignored
RAYON_NUM_THREADS=8 cargo test -p pyscf-pbc-dft --release --test gate -- --ignored
# the reported total energies must agree to the last bit, not to 1e-11
```

Note the tension this creates with the GPU work below, and how it is resolved:
`10_grid_stride_occupancy.md` §6 — *"disjoint grid-stride writes are bit-exact on
every backend and every geometry"*. Every kernel this plan adds must therefore have a
**disjoint output partition** (one lane owns one output cell) and must never use a
float atomic or a cross-lane float reduction on a quantity that lands in the energy.
Where a reduction is genuinely needed, use the fixed-point integer route of
[`09_fixedpoint_atomics.md`](../../../cubecl_manual/manual/Cubecl/09_fixedpoint_atomics.md)
— integer addition is associative, so the result is order-independent by construction.

### 1.5 GATE C — no FMA contraction (FOUND-05)

`xtask/src/bin/check_no_fma.rs` compiles the scanned crates under the
`release-oracle` profile with `--emit=asm` and fails on any `vfmadd*` / `fmadd` /
`fma213` / `fma231` mnemonic. A fused multiply-add computes `a*b + c` with a single
rounding instead of two, so a code path that contracts is *not bit-reproducible against
one that does not* — which is the whole point of D-PBC-03's fixed four-GEMM order and of
`oracle_sum`'s fixed pairwise tree.

Two consequences for this plan:

* **Every new `#[cube]` kernel must be written on unfused arithmetic.** Do not reach for
  `F::mul_add` or any fused intrinsic in a kernel whose output reaches an energy or a
  density matrix.
* **The ROCm/HIP backend needs `-ffp-contract=off` explicitly** —
  [`15_kernel_cache_and_rocm_setup.md`](../../../cubecl_manual/manual/Cubecl/15_kernel_cache_and_rocm_setup.md)
  §3 documents this and includes a verified contraction probe. That manual also records
  a trap directly relevant here: **the on-disk kernel cache silently ignores a changed
  compile flag**, so turning `-ffp-contract=off` on without clearing the cache can leave
  contracted binaries in place. Clear `target`'s cubecl cache when changing the flag.

`SCAN_TARGETS` (`check_no_fma.rs:81-88`) is currently
`[pyscf-algebra, pyscf-core, pyscf-ccsd]` — **no `pyscf-kernels`, no `pyscf-pbc-*`.**
Every kernel this plan adds is therefore unguarded today. **W-04 must add
`pyscf-kernels` to that list**, and W-03/W-05/W-06 should add `pyscf-pbc-df` and
`pyscf-pbc-dft`. The file's own doc comment anticipates exactly this ("other crates
(pyscf-kernels etc.) join the scan list as they accrete oracle-relevant numeric paths");
this plan is when that becomes due.

---

---

## 2. Where the time goes

### 2.1 The cost model, and the measurement that overrides it

Let `nao` = AOs per cell, `Nk` = sampling k-points, `Ng` = `mesh[0]·mesh[1]·mesh[2]`.
Per SCF iteration, counting complex multiply-adds:

| routine | complex FMA | 3-D transforms |
|---|---|---|
| `nr_rks` → `eval_rho_one` (`numint.rs:820`) | `Nk · nao² · Ng` | — |
| `nr_rks` → `vxc_mat_one` (`numint.rs:770`) | `Nk · nao² · Ng` | — |
| `get_j_kpts` (`fft_jk.rs:31`) | `2 · Nk · nao² · Ng` | `2` |
| **`get_k_kpts` (`fft_jk.rs:196`)** | **`3 · Nk² · nao² · Ng`** | **`2 · Nk² · nao²`** |

The FMA column says `get_k_kpts` should be ~`Nk`× the rest. **The transform column is
what actually decides it**, and it is `Nk²·nao²` versus 2 — for the gate cell a factor of
**4096**.

#### Measured — Si, `gth-szv` (`nao = 8`), 2×2×2 (`Nk = 8`), `mesh = [21,21,21]`, PBE

| stage | wall time | share of a hybrid `get_veff` |
|---|---|---|
| `nr_rks` (warm AO cache) | **22.7 ms** | 0.3 % |
| `get_j_kpts` | **13.9 ms** | 0.2 % |
| **`get_k_kpts`** | **6.60 s** | **99.5 %** |
| `get_veff` total, *pure* functional (no K) | **30.4 ms** | — |
| `nr_rks`, cold AO cache (one-off) | 7.26 s | — |
| `get_hcore` (one-off) | 3.26 s | — |

Two conclusions that reorder this entire plan:

1. **A pure functional is already fast.** `KRKS(PBE)`'s whole `get_veff` is 30 ms. The
   performance problem is **hybrids** (`PBE0`, `HSE`) and **`KRHF`** — anything that
   builds `K`. Optimising `nr_rks` cannot matter until `get_k_kpts` is fixed.
2. **`get_k_kpts` is 490× `get_j_kpts` and 290× `nr_rks`.**

#### Measured — and 93 % of `get_k_kpts` is the 3-D transform, not the contractions

Isolating exactly the transform workload `get_k_kpts` issues — `2·Nk²·nao² = 8192`
three-dimensional transforms of a `(64, Ng)` batch:

| mesh | `Ng` | 8192 transforms | µs/transform | **ns per grid point** | plan |
|---|---|---|---|---|---|
| 16 | 4 096 | 0.74 s | 90 | **22.0** | radix-2 |
| **21** | 9 261 | **5.59 s** | 683 | **73.7** | direct `O(n²)` |
| 25 | 15 625 | 11.09 s | 1 354 | **86.7** | direct `O(n²)` |
| 27 | 19 683 | 14.36 s | 1 753 | **89.1** | direct `O(n²)` |
| **31** | 29 791 | **25.25 s** | 3 082 | **103.5** | direct `O(n²)` |
| **32** | 32 768 | **11.93 s** | 1 456 | **44.4** | radix-2 |

At `mesh = 21` the isolated transform cost is **5.59 s against a `get_k_kpts` of 6.60 s
— 93 %.** Every contraction in `fft_jk.rs`, every `get_coulG` rebuild and every
`expmikr` rebuild together account for the remaining 7 %.

**The single cleanest statement of the problem is the last two rows.** `mesh = 32` has
**10 % more grid points than `mesh = 31` and computes 2.1× faster**, purely because 32
factors and 31 does not. And `[31,31,31]` is `MESH_GATE` — the mesh the accuracy gate
itself runs on (`crates/pyscf-pbc-dft/tests/gate.rs:33`).

The power-of-two rows also show the memory-access half of the problem: 16 and 32 use the
*same* algorithm, and per-point arithmetic between them grows only as `log₂n` (1.25×),
yet measured cost per point grows 2.0×. The excess is `transform_axis`'s strided gather
(§2.3) falling out of cache as the batch grows.

### 2.2 Every contraction on the path is a scalar, single-threaded loop

This was the headline finding *before* §2.1 was measured; the measurement demoted it to
**the 7 %**. It is still worth fixing — it is an ALG-06/D-PBC-03 compliance gap as much
as a performance one — but it is not where the time is. Not one of the hot contractions
reaches `pyscf_algebra::zgemm_dense`, `oracle_zdot`, or any device kernel:

| routine | file:line | shape | what it actually is |
|---|---|---|---|
| `eval_rho_one` | `pyscf-pbc-dft/src/numint.rs:820` | `(Ng,nao)ᴴ·(nao,nao)` then row-wise dot | hand-rolled triple loop |
| `vxc_mat_one` | `pyscf-pbc-dft/src/numint.rs:770` | `(nao,Ng)ᴴ·(Ng,nao)` | hand-rolled triple loop |
| `accumulate_rho` | `pyscf-pbc-df/src/fft_jk.rs:122` | `(nao,nao)·(nao,Ng)` + row dot | hand-rolled triple loop |
| `contract_ao_v_ao` | `pyscf-pbc-df/src/fft_jk.rs:154` | `(nao,Ng)ᴴ·diag(v)·(nao,Ng)` | hand-rolled triple loop |
| `dm_times_conj_ao` | `pyscf-pbc-df/src/fft_jk.rs:372` | `(nao,nao)·conj(nao,Ng)` | hand-rolled triple loop |
| `build_rho1` | `pyscf-pbc-df/src/fft_jk.rs:395` | outer product over `Ng` | hand-rolled triple loop |
| `contract_vr_aodm` | `pyscf-pbc-df/src/fft_jk.rs:443` | batched `(nao,Ng)` reduction | hand-rolled triple loop |
| `accumulate_vk` | `pyscf-pbc-df/src/fft_jk.rs:474` | `(nao,Ng)·(nao,Ng)ᵀ` | hand-rolled triple loop |
| `zmm_small` | `pyscf-pbc-df/src/zlinalg.rs:40` | `(nao,nao)·(nao,nao)` | hand-rolled triple loop |

`eval_ao_kpts` is the **one** hot path that was already device-enabled
(`quick-260826-spd`): `AoKAccumulator` keeps both `(nkpts, n)` planes resident on the
device across the whole lattice-image loop and reads back once
(`pyscf-kernels/src/pbc/eval_ao_k.rs:144-270`). **That type is the template for
everything this plan adds** — it is the shape that already satisfies ALG-06, D-PBC-17
and the hoisting rule of `11_launch_overhead_and_transfers.md` §2 simultaneously.

### 2.3 The 3-D FFT runs an O(n²) DFT on every mesh a PBC calculation actually uses

`pyscf-pbc-tools/src/fft_kernel.rs:45` sets `DIRECT_MAX = 40`, and `Fft1d::new`
selects, per axis length `n`:

* power of two → radix-2 Cooley-Tukey, `~(n/2)·log₂n` butterflies;
* **`n ≤ 40` and not a power of two → the direct `O(n²)` DFT** (`fft_kernel.rs:209`);
* `n > 40` → Bluestein over a padded power of two.

Real PBC meshes are odd and small: 21, 25, 31, 35 — the whole band lands in the direct
branch. Per axis that is `n²` complex mults where a mixed-radix decomposition costs
`n·Σrᵢ`:

| mesh axis | factorisation | direct `n²` | mixed radix `n·Σrᵢ` | ratio |
|---|---|---|---|---|
| 21 | 3·7 | 441 | 210 | 2.1× |
| 25 | 5·5 | 625 | 250 | 2.5× |
| 35 | 5·7 | 1225 | 420 | 2.9× |
| 27 | 3·3·3 | 729 | 243 | 3.0× |
| 31 | prime → Rader over 30 = 2·3·5 | 961 | ~350 | 2.7× |
| 47 (default diamond mesh) | prime → currently Bluestein over 128 | ~1150 | ~450 (Rader over 46) | 2.5× |

**This is a rare optimisation that improves precision at the same time.** The forward
error of a direct DFT grows as `O(n)` in the length; a Cooley-Tukey decomposition has
`O(log n)` depth and its error grows as `O(log n)`. Fewer operations per output means a
strictly shorter dependency chain, so W-02 below is scored against *both* gates.

Additionally `transform_axis` (`fft_kernel.rs:335-382`) walks the strided axes by
gathering one length-`n` line at a time into a scratch buffer with stride `inner`. For
the `x` axis `inner = my·mz = 441`, so every single element access is a separate cache
line — the textbook anti-pattern of
[`07_memory_coalescing.md`](../../../cubecl_manual/manual/Cubecl/07_memory_coalescing.md)
§2, whose §7 notes *"a coalescing fix beats an arithmetic fix"*.

### 2.4 Redundant work in the `get_k_kpts` k-pair loop

Inside the `Nk × Nk` double loop (`fft_jk.rs:255-302`), **per pair**:

* `get_coulg(...)` is rebuilt from scratch (`fft_jk.rs:262`) — an `O(Ng)` allocation
  plus a `Vec<[f64;3]>` of `k+G`. It depends only on `dk = kpt2 − kpt1`, and a
  Monkhorst-Pack mesh has only `Nk` distinct `dk` modulo the reciprocal lattice, not
  `Nk²`. So this is `Nk²` builds where `Nk` would do.
* `expmikr` is rebuilt (`fft_jk.rs:276-288`) — `2·Ng` transcendental `cos`/`sin` calls,
  again a function of `dk` alone. `8·9261·2 = 148k` transcendentals per iteration for
  the gate cell, all recomputable once and cached.
* Neither depends on the density matrix, so both are invariant across **every SCF
  iteration**, not merely across the pair loop. This is exactly
  `11_launch_overhead_and_transfers.md` §2 "hoist invariant uploads".

### 2.5 Precision: `fft_jk` violates D-PBC-17

`vxc_mat_one` in `numint.rs:770` correctly reduces over the grid with
`pyscf_algebra::oracle_sum` (the FOUND-06 pairwise tree, chunk 128). Every reduction in
`fft_jk.rs` is a **naive sequential `+=` over `Ng` terms**:

* `accumulate_rho` (`:141-149`) — `Ng`-long running sum into `rho`, which becomes `vj`
  and hence `ecoul`;
* `contract_ao_v_ao` (`:173-180`) — `sr`/`si` accumulated over all `Ng` grid points,
  landing directly in `vj[p,q]`;
* `accumulate_vk` (`:474-496`) — same shape, landing directly in `vk[p,q]` and hence in
  `exc` via `krks.rs:get_veff_tagged`.

For `Ng ≈ 10⁴–10⁵` a sequential sum carries a worst-case relative error of `O(Ng·ε)`
≈ `2e-11`, against `O(log₂Ng · ε)` ≈ `4e-15` for the pairwise tree — i.e. *the naive
sum's error bound is at the same order as the 1e-11 gate itself*. `oracle_zdot`
(`pyscf-algebra/src/zoracle.rs:36`) is the primitive that already exists for this exact
contraction and is not being called. **W-05 is the precision headline.**

### 2.6 No AO screening

`eval_ao_kpts` (`pyscf-pbc-gto/src/eval_gto.rs:159-274`) evaluates and stores every
`(grid, AO)` amplitude. Upstream threads a `non0tab` / screen index through
`pbc_eval_gto` so that grid blocks where a shell is numerically zero are skipped
entirely. For a large or sparse cell that is where most of the AO table is. Deferred to
W-09 because it is the only item here whose payoff is strongly cell-dependent.

---

## 3. Work items

Ordered. Each is independently landable and independently revertible. Do not start
`W-(n+1)` before `W-n` is green under both gates.

---

### W-00 — The profiling harness (**do this first, it gates everything else**)

**FILES** `crates/pyscf-bench/src/bin/krks_profile.rs` (new), `crates/pyscf-bench/Cargo.toml`

**WHY** §2.1's numbers came from two throwaway examples that were deleted after the
run. They must become a committed, re-runnable baseline before anything is optimised —
`11_launch_overhead_and_transfers.md` §6 is explicit that optimisations must be
attributed against a warm profile, one variable at a time, and §2.1 is already proof
that the *model* pointed at the wrong item (it predicted `get_k_kpts` would be `Nk`× the
rest; it is 490×, and for a different reason than the FMA count suggested).

**STEPS**

1. A binary that builds `Krks` over a parameterised cell (Si/diamond/He), k-mesh and
   FFT mesh, and times, **warm** (i.e. after one throwaway pass so the AO cache, the
   FFT plan cache and the cubecl JIT are all paid for):
   `get_ovlp`, `get_hcore`, `nr_rks`, `get_j_kpts`, `get_k_kpts`, `fft`/`ifft` alone,
   and one full `kernel()` to convergence.
2. Report wall time, the count of 3-D transforms, and complex-FMA counts from §2.1 so
   measured-vs-model divergence is visible.
3. Emit machine-readable JSON so a later run can be diffed against the baseline.
4. Enable cubecl's own timings alongside — see
   [`profiling_tools.md`](../../../cubecl_manual/manual/Cubecl/profiling_tools.md)
   for the host/JIT-side feature flags.

5. Include an **isolated transform benchmark** (the `2·Nk²·nao²` batch of §2.1) as its
   own sub-command. It is what attributed 93 % of `get_k_kpts` in the first place, and
   it re-runs in seconds where a full `get_k_kpts` does not.
6. Sweep the mesh over `{16, 21, 25, 27, 31, 32}` so the factorisation cliff of §2.1
   stays visible as a regression signal, not just as a one-off finding.

**DONE** A committed baseline JSON for `Si 2×2×2 PBE` at `mesh=[21,21,21]` and
`mesh=[31,31,31]` (the gate mesh), for **both a pure functional and PBE0** (the pure
case exercises none of `get_k_kpts` and is the wrong baseline on its own), plus a
`--compare <baseline.json>` mode.

**RISK** none — additive.

---

### W-01 — Hoist `get_coulG` and `expmikr` out of the `Nk²` pair loop

**FILES** `crates/pyscf-pbc-df/src/fft_jk.rs`, `crates/pyscf-pbc-df/src/fftdf.rs`

**WHY** §2.4. `Nk²` rebuilds of two `dk`-only quantities, `Nk²` times per SCF
iteration, of something that is constant for the whole SCF.

**STEPS**

1. Key both tables on the **wrapped** `dk` (the `wrap_around: true` folding
   `get_coulg` already applies), not on the raw `(k1,k2)` index pair. Build a
   `HashMap<[u64;3], Arc<(Vec<f64> /*coulG*/, Option<CTensor> /*expmikr*/)>>` keyed on
   `dk.map(f64::to_bits)` after rounding to the reciprocal-lattice tolerance already
   used by `pyscf_pbc_gto::is_zero`.
2. Hang the cache off `Fftdf` next to the existing `ao_cache`
   (`fftdf.rs:169-199`) so it survives across SCF iterations, and clear it in
   `Fftdf::reset` alongside the AO cache.
3. Key it on `(dk, omega, exxdiv)` — an RSH functional calls `get_k_kpts` twice per
   iteration with different `omega` (`veff.rs:get_jk`), and the two must not collide.

**BIT-PARITY** exact. Same values, computed once instead of `Nk²` times; nothing is
reordered or re-associated.

**TEST** `crates/pyscf-pbc-df/tests/fft_jk_cache.rs` — assert the cached and uncached
`get_k_kpts` agree **bit-for-bit** (`==` on the raw `f64`, not `approx`), over
`{gamma, 2×2×2, 1×1×3}` × `{omega: None, Some(0.11), Some(-0.11)}`.

**DONE** Gate A + Gate B green; W-00 shows the transcendental and allocation cost gone.

---

### W-02 — Mixed-radix / Rader FFT: faster **and** more accurate — **THE ITEM**

**FILES** `crates/pyscf-pbc-tools/src/fft_kernel.rs`

**WHY** §2.1 and §2.3. This is **93 % of the dominant cost**, it is the one item that
moves both gates in the same direction, and §2.1's `mesh 31 vs 32` pair shows the loss
is pure algorithm choice rather than anything intrinsic to the problem.

**EXPECTED GAIN** From §2.3's operation counts against §2.1's measured `ns/point`:

| mesh | measured now | mixed-radix ratio | projected | sanity check |
|---|---|---|---|---|
| 21 | 73.7 ns/pt | 2.1× | ~35 ns/pt | radix-2 at this size measures 22–44 |
| 25 | 86.7 ns/pt | 2.5× | ~35 ns/pt | ″ |
| 27 | 89.1 ns/pt | 3.0× | ~30 ns/pt | ″ |
| 31 | 103.5 ns/pt | ~2.6× (Rader) | ~40 ns/pt | `mesh 32` measures 44.4 |

The `mesh 32` measurement is the honest ceiling: a correct mixed-radix implementation
should land the odd meshes in the same 22–44 ns/point band the power-of-two meshes
already occupy, i.e. `get_k_kpts` at `mesh=21` from 6.60 s to roughly **3.0–3.5 s**.
Anything better than that is coming from the memory-access half (§2.3's strided gather),
not from the radix change, and should be attributed separately.

**STEPS**

1. Add `Plan::MixedRadix` — a Cooley-Tukey decomposition over the factorisation of `n`
   with straight-line codelets for radix 2, 3, 4, 5, 7 and a generic radix-`r` stage
   for the rest. Keep the twiddle table cached per `n` exactly as the existing plans do.
2. Add `Plan::Rader` for prime `n` — a length-`n` prime DFT becomes a length-`(n−1)`
   cyclic convolution over the multiplicative group, which recurses into
   `MixedRadix`. This is what makes 31 and 47 cheap.
3. Plan selection becomes: power of two → `Radix2` (unchanged); `n` smooth (all prime
   factors ≤ 7) → `MixedRadix`; `n` prime → `Rader`; otherwise `MixedRadix` over the
   smooth part with `Rader`/`Bluestein` on the rough factor. **Delete `DIRECT_MAX`**
   and keep `Direct` only for `n ≤ 4` where it *is* the codelet.
4. Do NOT touch `transform_axis`'s staging or `fft_stockham`'s per-axis `1/mx`,
   `1/my`, `1/mz` scaling — the master plan (`PBC-MASTER-PLAN.md`, plan 11-01) fixes
   that staging because folding it into one final `1/ngrids` would change the rounding.

**BIT-PARITY** **NO — this changes the summation order and is expected to change the
last bits, in the direction of the true value.** That is why it is scored against the
gate and not against a bit-parity assertion.

**TEST** `crates/pyscf-pbc-tools/tests/fft_accuracy.rs`
- Round-trip `ifft(fft(x)) == x` to `1e-14` for meshes with dims in
  `{2,3,4,5,7,8,9,11,13,16,17,21,25,27,31,32,35,47,64}` — must be **tighter** than the
  existing `1e-12` round-trip, and record the achieved figure.
- Against a 128-bit reference (sum the DFT in `f64` pairs / a rational reference for a
  small mesh): the new plan's max relative error must be **strictly smaller** than
  `Plan::Direct`'s at `n ∈ {21, 25, 31, 35}`. If it is not, the item has a bug — stop.
- `fft` of a delta is all-ones; `fft` of a constant is `Ng·δ_{G,0}` — both exact.

**DONE** Gate A green *with headroom* (record the new residual for
`KRKS He-fcc PBE`, which should shrink), Gate B green, W-00 shows the transform time
down by the §2.3 ratio.

**RISK** medium. Rader for primes is fiddly (primitive-root search, index permutation).
Land `MixedRadix` alone first, keep `Bluestein` as the prime fallback, and add `Rader`
as a separate commit.

---

### W-02b — Parallelise the transform batch (bit-exact)

The `2·Nk²·nao² = 8192` transforms per `get_k_kpts` are **completely independent** — a
batch of `nao²` rows per `(k1,k2)` pair, and `Nk²` pairs. They currently run one after
another on a single core (`fft.rs:175`, `fft_kernel.rs:335`). This machine has 8 physical
cores; the §2.1 measurement used one of them.

**FILES** `crates/pyscf-pbc-tools/src/fft.rs`, `crates/pyscf-pbc-df/src/fft_jk.rs`

**STEPS** Parallelise at the **batch row** level — one whole 3-D transform per worker,
over disjoint buffers. Do **not** parallelise inside a transform.

**BIT-PARITY** **exact, for any thread count.** This is the whole reason for the batch/
inside distinction: each transform's internal summation order is untouched, so the result
is bit-identical to the serial run. Same argument `10_grid_stride_occupancy.md` §6 makes
for disjoint grid-stride writes, and the same one `oracle.rs:6-11` makes for the
fixed-shape pairwise tree.

**TEST** Gate B is the test, and it is a real one here: assert `get_k_kpts` output is
bit-identical (`==` on raw `f64`) across `RAYON_NUM_THREADS ∈ {1, 2, 8, 16}`.

**DONE** Combined with W-02's ~2× algorithmic gain this is the difference between
`get_k_kpts` at 6.6 s and at well under 1 s for the gate cell.

---

### W-03 — Route the `fft_jk` contractions through `zgemm_dense`

**FILES** `crates/pyscf-pbc-df/src/fft_jk.rs`, `crates/pyscf-pbc-df/src/zlinalg.rs`

**WHY** §2.2. Six triple loops that are literally GEMMs. `PBC-MASTER-PLAN.md:1051`
already mandates `zgemm_dense` for the contractions in the neighbouring plan 11-01;
`fft_jk` never got the same treatment.

**PRIORITY — LOWERED BY MEASUREMENT.** §2.1 attributes **7 %** of `get_k_kpts` to
everything that is not the transform, and W-01 already takes part of that 7 %. Do not
start this item until W-02 has landed and W-00 has re-attributed the profile: after a
2× on the transform these contractions are still only ~13 % of `get_k_kpts`, and the
`FftEngine::Blas` result in step 4 shows the naive version of this change is a
regression. This is a correctness-of-architecture item (ALG-06, D-PBC-03 compliance)
with a modest performance payoff, and it should be scheduled as one.

**STEPS**

1. Rewrite, in this order, each as a `zgemm_dense` / `zgemm_h_dense` call:
   `dm_times_conj_ao` (`:372`) → `(nao,nao)·(nao,Ng)`;
   `accumulate_rho`'s `c0` stage (`:124-140`) → the same shape;
   `accumulate_vk` (`:474`) → `(nao,Ng)·(nao,Ng)ᵀ`;
   `contract_ao_v_ao`'s outer stage (`:167-185`) → `(nao,Ng)ᴴ·(nao,Ng)`.
2. `zgemm_dense` is **four real `gemm_dense` calls in a fixed order** (D-PBC-03,
   `zgemm.rs:43-79`). Do not reorder, do not fuse, do not substitute Karatsuba. The
   doc comment on every new call site must cite D-PBC-03.
3. The Hadamard stages that are *not* GEMMs (`aow = ao * vR` at `:155-166`, the
   `expmikr` multiply at `:297-311`, the `coulG` multiply at `:286-292`) go to
   `pyscf_kernels::pbc::zhadamard`, which already exists
   (`pyscf-kernels/src/pbc/zhadamard.rs:125`).
4. **Watch the transfer cost — this is MEASURED, not hypothetical.**
   `gemm_dense` (`pyscf-algebra/src/gemm.rs:290`) uploads both operands and reads the
   result back on *every call*. There is already a natural experiment in the tree:
   `FftEngine::Blas` (`fft.rs:51`) routes the 3-D transform through `zgemm_dense`, and
   on this machine it is **1.43× SLOWER** than the scalar host engine
   (`get_k_kpts` 9.46 s vs 6.60 s at `mesh=21`) — despite `Blas` being what
   `PBC-MASTER-PLAN.md` plan 11-01 mandates. Per-call upload/read-back on the CPU
   backend eats the entire benefit and then some.
   **Therefore: W-03 MUST NOT land without W-04.** Routing these contractions through
   `zgemm_dense` while the data still round-trips per call is a measured regression, not
   a speculative one.

**BIT-PARITY** **NO** — a blocked GEMM sums in a different order from the naive loop.
Scored against the gate. Expect the residual to *shrink* (a blocked GEMM's error bound
is better than a naive `k`-loop's), but verify rather than assume.

**TEST** `crates/pyscf-pbc-df/tests/fft_jk_gemm.rs` — the new `get_j_kpts`/`get_k_kpts`
against the pre-W-03 implementations (keep them as `#[cfg(test)]` reference functions in
the test file, **not** in the source file — RULE 4) to `1e-13` relative, over the same
matrix of cells/k-meshes/omegas as W-01.

---

### W-04 — Keep the `get_k_kpts` operands device-resident

**FILES** `crates/pyscf-kernels/src/pbc/kbuild.rs` (new), `crates/pyscf-pbc-df/src/fft_jk.rs`

**WHY** §2.2 and `11_launch_overhead_and_transfers.md` §2 and §5. After W-03 the
arithmetic is on the device but the *data* still crosses the boundary once per call, and
the `Nk²` loop makes that `Nk²` round trips of `nao·Ng` complex numbers.

**STEPS**

1. Model it on `AoKAccumulator` (`pyscf-kernels/src/pbc/eval_ao_k.rs:144-270`) — the
   precedent that already solved exactly this for the lattice-image loop. Expose a
   `KBuildContext` whose **private** fields are cubecl `Handle`s and whose public
   surface names no cubecl type, so `pyscf-pbc-df` can drive it without breaching
   ALG-06.
2. Upload once, before the pair loop: the AO tables `ao1_kpts` / `ao2_kpts`, the
   `coulG` table per distinct `dk` (from W-01), and the `expmikr` table per distinct
   `dk`. Allocate `vk_kpts` on the device and accumulate into it in place.
3. Read back **once**, after the loop, with a single batched `client.read(vec![…])`
   (`11_launch_overhead_and_transfers.md` §3), never one handle at a time.
4. Pre-allocate the `rho1` / `vR` / `vR_dm` scratch **outside** the loop and reuse the
   handles — [`13_memory_preallocation.md`](../../../cubecl_manual/manual/Cubecl/13_memory_preallocation.md)
   §1; the runtime's pool still charges a lookup per allocation.
5. Collapse the per-`(k1,k2)` launches into one grid-addressed launch where the shapes
   allow (`11_launch_overhead_and_transfers.md` §5): `CUBE_POS_X` selects the pair,
   with an offset table into a concatenated buffer.
6. Every kernel: `#[cube(launch_unchecked)]`, generic `<F: Float + CubeElement>`, with
   the host-side validation and the `// SAFETY:` block enumerating why each access is in
   range (`11_launch_overhead_and_transfers.md` §4). `launch_unchecked` changes
   performance, never numerics — if the result moves, there was an out-of-bounds access.
7. Geometry comes from `pyscf_algebra::launch::launch_1d` and `line_size_for`
   (`pyscf-algebra/src/launch.rs`) — **never** a hard-coded `CubeDim::new_1d(256)`.
   The default backend here is the CPU runtime (`default = ["cpu"]`), where a "unit" is
   an OS thread and a fixed 256-wide cube is pathological; `launch_1d` already encodes
   that. See also
   [`Hardware-Adaptive_Launch_Geometry.md`](../../../cubecl_manual/manual/Cubecl/Hardware-Adaptive_Launch_Geometry.md).
8. Output partition must be **disjoint** — one lane owns one output cell — so Gate B
   holds by construction (`10_grid_stride_occupancy.md` §6). No float atomics.
9. Unfused arithmetic only, and add `pyscf-kernels` to `SCAN_TARGETS` in
   `xtask/src/bin/check_no_fma.rs` — Gate C (§1.5).

**BIT-PARITY** exact with respect to W-03 (the same GEMMs, the same order, just without
the round trips).

**TEST** `crates/pyscf-kernels/tests/pbc_kbuild.rs` (kernel-level, random inputs per
[`Cubecl_random_value_test.md`](../../../cubecl_manual/manual/Cubecl/Cubecl_random_value_test.md))
plus a `get_k_kpts` bit-parity assertion against W-03's output.

**RISK** high — the largest single change here. Land W-03 first and keep W-04 revertible
behind `PYSCF_PBC_JK_ENGINE={host,device}` until the gate has run green on both.

---

### W-05 — **Precision:** ordered reductions in `fft_jk` (D-PBC-17 compliance)

**FILES** `crates/pyscf-pbc-df/src/fft_jk.rs`, `crates/pyscf-pbc-df/src/zlinalg.rs`

**WHY** §2.5. Three naive `O(Ng)` sequential sums land directly in `vj`, `vk`, `ecoul`
and `exc`. Their error bound is the same order as the 1e-11 gate. This is a standing
violation of D-PBC-17, not a discretionary improvement.

**STEPS**

1. `accumulate_vk` (`:474-496`) and `contract_ao_v_ao` (`:167-185`): replace the
   `sr`/`si` running sums with `pyscf_algebra::oracle_zdot`
   (`zoracle.rs:36`), which is precisely `Σ conj(x)·y` as four `oracle_dot` calls.
2. `accumulate_rho` (`:141-149`): the per-`g` sum is over `nao` (small) and is fine;
   the `Ng`-long accumulation into `rho` is the one to fix.
3. `zmm_small` (`zlinalg.rs:40-60`): `k = nao` is small, so this is a *correctness of
   policy* fix rather than a numerical one — route it through `zgemm_dense` under W-03
   and the question disappears.
4. If W-04 has landed, the device kernels must reduce the same way: a pairwise tree
   with the **fixed** `PAIRWISE_CHUNK = 128` shape (`oracle.rs:15`), or the fixed-point
   integer accumulator of `09_fixedpoint_atomics.md` — never a float atomic, never a
   `plane_sum` over a quantity that reaches the energy. `PAIRWISE_CHUNK` is load-bearing
   for bit-compatibility with prior runs; do not change it.

**BIT-PARITY** **NO — this is the point.** Expect the gate residual to *shrink*.

**TEST** `crates/pyscf-pbc-df/tests/fft_jk_precision.rs`
- Construct an ill-conditioned case (a `vR` with alternating large/small magnitudes over
  `Ng = 10⁵`), compare naive vs `oracle_zdot` against a 128-bit reference, and assert the
  ordered route is at least 3 decimal digits better.
- `RAYON_NUM_THREADS=1` vs `=8` bit-identity on `get_j_kpts` and `get_k_kpts` output.

**DONE** Gate A green **with a smaller residual than the pre-W-05 baseline** — record
both numbers in the summary.

---

### W-06 — Route `numint`'s grid contractions through `zgemm_dense`

**FILES** `crates/pyscf-pbc-dft/src/numint.rs`

**WHY** §2.2. `eval_rho_one` (`:820`) and `vxc_mat_one` (`:770`) are `2·Nk·nao²·Ng`
complex FMA per iteration in scalar loops. Lower priority than `fft_jk` only because
`get_k_kpts` is `Nk`× bigger — but for a **pure** functional (`LDA`, `PBE`) there is no
`K` at all and this becomes the dominant cost.

**STEPS**

1. `eval_rho_one`'s `c0[g,j] = Σᵢ ao₀[g,i]·dm[i,j]` → `zgemm_dense`, and the
   `Σⱼ conj(ao_c)·c0` row reduction → a fused `zhadamard` + segment reduction. Keep the
   `hermi = 1` factor-2 on the gradient rows exactly where it is (`numint.rs:869-871`,
   upstream `numint.py:141`).
2. `vxc_mat_one`'s `aow` stage → `zhadamard`; the `Σ_g conj(ao)·aow` outer product →
   `zgemm_h_dense`. It **already** reduces with `oracle_sum`, so preserve that ordering
   or prove the replacement is better; do not silently drop to a naive GEMM reduction.
3. Delete the `if s == 0.0 { continue; }` and `if dr == 0.0 && di == 0.0 { continue; }`
   short-circuits (`numint.rs:795`, `:846`). They are a *serial* micro-optimisation that
   creates branch divergence on any SIMT backend — see
   [`plane_alignment.md`](../../../cubecl_manual/manual/Cubecl/plane_alignment.md) and
   `Cubecl_conditionals.md` ("avoid `if` expressions"). Real screening belongs in W-09,
   at block granularity, not per element.
4. Fuse the `den = rho·w`, `nelec`, `exc` and `wv = vxc·w` element-wise passes
   (`numint.rs:317-327`) into one kernel —
   [`03_kernel_fusion.md`](../../../cubecl_manual/manual/Cubecl/03_kernel_fusion.md) §4:
   four passes over `Ng` become one, and the three intermediates never reach global
   memory. Keep `oracle_sum` on the two scalars that become energies.

**BIT-PARITY** **NO** (GEMM reassociation). Gate-scored.

**TEST** `crates/pyscf-pbc-dft/tests/numint_gemm.rs` — new vs reference to `1e-13`
relative on `{LDA, GGA} × {gamma, 2×2×2} × {1, 2} density sets`.

---

### W-07 — Grid-block sizing and the AO cache

**FILES** `crates/pyscf-pbc-dft/src/numint.rs`

**WHY** `block_ranges` (`:195-211`) derives the block from `max_memory` with a
deliberately conservative denominator ("`nao` is folded in by the caller"), which for a
4000 MB default and a small cell collapses to a single block covering the whole grid —
so the `BLKSIZE = 128` blocking that upstream uses for cache locality does nothing. On
the CPU runtime that is a working set of `4·Ng·nao·16` bytes streamed with no reuse.

**STEPS**

1. Make `block_ranges` take `nao` and size the block to the **L2 cache**, not to
   `max_memory`: pick the largest multiple of `BLKSIZE` for which
   `comp·2·Nk·nao·blk·16` bytes fits a tunable working-set target.
2. Expose the target through `PYSCF_PBC_NUMINT_BLKSIZE` and default it from the device
   properties. Consider autotuning it —
   [`12_autotuning.md`](../../../cubecl_manual/manual/Cubecl/12_autotuning.md) §1;
   the on-disk autotune cache already defaults to `target`.
3. The AO cache (`numint.rs:151-190`) hashes the full coordinate array with FNV-1a on
   **every** `eval_ao` call (`coord_hash`, `:918-931`) — an `O(Ng)` pass per block per
   set per iteration purely to look up a cache entry. Key on a cheap grid identity (a
   `PeriodicGrids` generation counter plus `(p0, p1)`) instead.

**BIT-PARITY** **The block partition changes `oracle_sum`'s input lengths and therefore
the pairwise tree shape**, so `excsum`/`nelec` will move in the last bits. This is
inherent to changing the blocking and is why the item is called out separately rather
than folded into W-06. If the gate cannot absorb it, accumulate `excsum` into a
block-independent tree (collect per-block partials and `oracle_sum` those) — do that
**first** and the item becomes bit-stable across block sizes, which is worth having on
its own.

**TEST** `crates/pyscf-pbc-dft/tests/numint_blocking.rs` — `nr_rks` output must be
bit-identical across `PYSCF_PBC_NUMINT_BLKSIZE ∈ {128, 1024, 8192, whole-grid}` once the
block-independent accumulation of the previous paragraph is in.

---

### W-08 — **Opt-in, changes results:** k-point pair symmetry in `get_k_kpts`

**FILES** `crates/pyscf-pbc-df/src/fft_jk.rs`

**WHY** The `Nk²` pair loop computes `(k1,k2)` and `(k2,k1)` independently. For a
Hermitian density matrix the two contributions are related by conjugate transposition,
so half the loop is redundant — a clean 2× on the dominant term. Upstream does this in
`pyscf/pbc/df/aft_jk.py` via `kk_adapted_iter`.

**STEPS**

1. Port `kk_adapted_iter`'s k-pair classification from `aft_jk.py`, line by line
   (RULE 2). Do **not** invent a symmetry argument; use upstream's.
2. Gate it behind `JkOpts { kk_symmetry: bool }`, defaulting **off**.
3. Note the interaction with `_hermi` — `get_k_kpts` currently ignores its `hermi`
   argument (`fft_jk.rs:198`, `_hermi`). The symmetry is only valid for `hermi == 1`;
   make the flag an error when `hermi != 1` rather than silently wrong.

**BIT-PARITY** **NO — halving the loop changes which terms are summed and in what
order.** This is why it is opt-in and last.

**TEST** symmetric-vs-full agreement to `1e-13`, and a **separate** gate run with the
flag on whose tolerance is re-baselined and recorded, not inherited.

---

### W-09 — **Deferred:** AO screening (`non0tab`)

**FILES** `crates/pyscf-pbc-gto/src/eval_gto.rs`, `crates/pyscf-pbc-dft/src/numint.rs`

**WHY** §2.6. Payoff scales with cell size and sparsity; on the 8-AO gate cell it is
approximately zero, so it cannot be validated by the gate and must be justified on a
larger cell.

**STEPS** Port upstream's `make_screen_index` / `non0tab` at `BLKSIZE` block
granularity, thread it through `eval_ao_kpts` and `KNumInt::block_ranges`, and skip
whole `(block, shell)` pairs. **Block granularity, never per element** — a per-element
skip is the branch divergence W-06 §3 removes.

**BIT-PARITY** NO — screened terms are dropped. Needs its own cutoff-convergence test
showing the dropped mass is below the gate, on a cell large enough for it to matter.

**DEFER UNTIL** W-00 has a large-cell baseline (e.g. Si 2×2×2 supercell, `gth-dzvp`)
that shows AO evaluation is actually a material share of the time.

---

## 4. Sequencing

**Reordered by §2.1's measurements.** The pre-measurement draft treated W-02, W-03 and
W-05 as peers; the profile says W-02 is 93 % of the problem and W-03 addresses 7 % via a
route that is currently a measured regression.

```
W-00 (commit the harness + baseline)
  │
  ├─→ W-02 (mixed-radix / Rader FFT)      ← 93 % of get_k_kpts. Do this first.
  │     └─→ W-02b (parallelise the transform batch)   ← bit-exact; 7 idle cores
  │
  ├─→ W-05 (ordered reductions, D-PBC-17) ← precision headline, independent of all timing work
  │
  └─→ W-01 (hoist coulG/expmikr)          ← cheap, exact, part of the residual 7 %
        └─→ W-03 (zgemm in fft_jk)  ──┐
                                      ├─→ MUST land together (see W-03 step 4)
              W-04 (device residency) ─┘
                    └─→ W-06 (zgemm in numint)     ← only matters for PURE functionals
                          └─→ W-07 (blocking + AO-cache key)
                                ├─→ W-08 (kk symmetry)   opt-in, re-baselined
                                └─→ W-09 (AO screening)  deferred, needs a large cell
```

W-00, W-02 and W-05 are mutually independent and can be done in parallel. W-05 should
land early regardless of the timing work: it is the precision item, it touches none of
the same code as W-02, and landing it before W-03/W-04 keeps its effect on the gate
residual attributable rather than tangled with a GEMM reassociation.

## 5. Verification protocol — run this after EVERY work item

```bash
# 1. Gate A — accuracy against upstream PySCF 2.12.1 (vendored tree)
PYSCF_ORACLE_VENV=1 cargo test -p pyscf-pbc-dft --release --test gate -- --ignored --nocapture

# 2. Gate B — thread-count bit-identity (D-PBC-17)
RAYON_NUM_THREADS=1 cargo test -p pyscf-pbc-dft --release --test gate -- --ignored --nocapture > /tmp/t1.log
RAYON_NUM_THREADS=8 cargo test -p pyscf-pbc-dft --release --test gate -- --ignored --nocapture > /tmp/t8.log
diff /tmp/t1.log /tmp/t8.log     # must be empty

# 3. The ALG-06 dependency wall
cargo run -p xtask --bin check-dependency-wall

# 3b. Gate C — FOUND-05, no FMA contraction on the oracle numeric path
cargo run -p xtask --bin check-no-fma

# 4. Everything downstream of the touched crates
cargo test -p pyscf-pbc-tools -p pyscf-pbc-df -p pyscf-pbc-dft -p pyscf-pbc-scf --release

# 5. Re-profile — one variable changed since the last run
cargo run -p pyscf-bench --release --bin krks_profile -- --compare baseline.json
```

Record, per item, in a `SUMMARY.md` next to this file: the wall-time delta, the **gate
residual before and after** (not just pass/fail — the residual is the precision signal),
and whether bit-parity was preserved or deliberately broken.

---

## 6. Risks

| risk | mitigation |
|---|---|
| A GEMM reassociation pushes the GTH gate past 1e-11 | The all-electron He control (1e-12) isolates whether the move is the reassociation or the `get_pp` floor. Land W-05 first so precision has headroom before W-03/W-04 spend it. |
| `gemm_dense`'s upload/read-back per call makes W-03 *slower* on the CPU backend | W-00 measures it. W-03 and W-04 are designed to land together for exactly this reason; if W-03 alone regresses, hold it behind the engine flag until W-04 is ready. |
| The ROCm iGPU available locally has no f64, so `PYSCF_BACKEND=rocm` silently falls back to CPU | GPU paths cannot be measured locally. Validate device kernels for **correctness** on the CPU runtime, and mark throughput claims for GPU backends as unverified until CI has an f64-capable device. |
| Rader's primitive-root/index-permutation logic is subtle | Land `MixedRadix` alone first with `Bluestein` still handling primes; add `Rader` as a separate, separately-tested commit. |
| Device kernels reintroduce nondeterminism | Disjoint output partitions only (`10_grid_stride_occupancy.md` §6). No float atomics, no cross-lane float reductions on energy quantities; fixed-point integer atomics (`09_fixedpoint_atomics.md`) where a reduction is unavoidable. |
| A device kernel contracts to FMA and breaks reproducibility | Gate C (§1.5). Write kernels on unfused arithmetic, set `-ffp-contract=off` on ROCm, and clear the cubecl on-disk cache when changing that flag — it does not invalidate on a compile-flag change (`15_kernel_cache_and_rocm_setup.md` §2). |
| Kernel JIT cost swamps a small cell | `cubecl.toml` already sets `[compilation] cache = "target"`. Keep it, and profile **warm** (`15_kernel_cache_and_rocm_setup.md` reports 24.5 s → 2.5 s on a 143-kernel suite from this setting alone). |

---

## 7. CubeCL manual sections this plan depends on

Read before implementing the item that cites it.

| item | manual |
|---|---|
| W-01, W-04 | `11_launch_overhead_and_transfers.md` — hoisting, batched read-back, `launch_unchecked`, collapsing per-item launches |
| W-04 | `13_memory_preallocation.md` — pre-allocate scratch outside the hot loop |
| W-04 | `Hardware-Adaptive_Launch_Geometry.md` — why `launch_1d`/`line_size_for` exist and what a hard-coded `CubeDim` costs on the CPU runtime |
| W-04, W-06 | `10_grid_stride_occupancy.md` — grid-stride loops; **disjoint writes are bit-exact** |
| W-04, W-06 | `06_vectorization.md`, `Cubecl_dynamic_vectorization.md` — `Vector<F, N>` and dividing the grid by `N` |
| W-02, W-04 | `07_memory_coalescing.md` — the strided-gather problem in `transform_axis` |
| W-05 | `09_fixedpoint_atomics.md` — order-independent accumulation when a reduction is unavoidable |
| W-06 | `03_kernel_fusion.md` — fusing the four element-wise passes over the grid |
| W-06 | `plane_alignment.md`, `Cubecl_conditionals.md` — why the per-element zero-skip must go |
| W-07 | `12_autotuning.md`, `04_autotune_optimization.md` — tuning the block size with a persistent cache |
| all kernels | `Cubecl_generics.md` — `<F: Float>` generic kernels (RULE 5) |
| all kernels | `Cubecl_random_value_test.md` — random-value kernel validation |
| Gate C, W-04 | `15_kernel_cache_and_rocm_setup.md` — `-ffp-contract=off`, and the cache-ignores-changed-flags trap |
| profiling | `profiling_tools.md`, `15_kernel_cache_and_rocm_setup.md` |
| on any build error | `cubecl_error_solution_guide/` — **read before touching the code** |

---

## 8. Open questions — ANSWERED 2026-08-30

The four questions this plan opened with have been measured. Recorded here so W-00 does
not re-derive them, and so a future run can check they still hold.

1. **What is the split between `get_k_kpts`, `nr_rks` and `get_j_kpts`?**
   Answered, §2.1. For a hybrid: `get_k_kpts` **99.5 %**, `nr_rks` 0.3 %, `get_j_kpts`
   0.2 %. For a pure functional there is no `K` at all and the whole `get_veff` is
   30 ms. **The performance problem is hybrids and `KRHF`, not `KRKS(PBE)`.**

2. **What fraction of `get_k_kpts` is the transform versus the contractions?**
   Answered, §2.1. Transform **93 %** (5.59 s isolated, against a 6.60 s
   `get_k_kpts`). Contractions + `get_coulG` + `expmikr` = the remaining 7 %. This is
   what demoted W-03/W-04 and promoted W-02.

3. **Does `FftEngine::Blas` already beat `Stockham`?**
   **No — it is 1.43× slower** (`get_k_kpts` 9.46 s vs 6.60 s at `mesh=21`). Do not flip
   the default. The result is more useful as evidence than as an option: it is a direct
   measurement of what per-call `zgemm_dense` upload/read-back costs on the CPU backend,
   and it is why W-03 may not land without W-04. Note the shipped default (`Stockham`)
   therefore *disagrees with* `PBC-MASTER-PLAN.md` plan 11-01, which mandates the BLAS
   engine — the master plan's assumption was that `zgemm_dense` would be the fast route.
   Worth an explicit erratum against that plan when W-02 lands.

4. **Is the AO cache hit rate actually 100 % across SCF iterations?**
   **Partly answered and it matters less than expected.** `nr_rks` costs 7.26 s cold and
   22.7 ms warm, so the cache is working — a ~320× difference. But `coord_hash`
   (`numint.rs:918`) still walks all `Ng` coordinates on every lookup, and at 22.7 ms
   total for warm `nr_rks` that hash is now a visible share of it. Still worth W-07's
   fix; no longer urgent, since `nr_rks` is 0.3 % of a hybrid iteration.

### New question opened by the measurements

5. **How much of the per-point cost is arithmetic and how much is the strided gather?**
   §2.1's `mesh 16` vs `mesh 32` pair (same radix-2 algorithm, 22.0 → 44.4 ns/point
   where arithmetic predicts only 1.25×) says the gather is a large second term. W-02
   should attribute its gain between the two — if the radix change alone does not
   deliver the projected ratio, the remainder is `transform_axis` (`fft_kernel.rs:335`)
   and needs the layout fix of `07_memory_coalescing.md` §2 rather than more radix work.

---

## 9. ERRATA — recorded 2026-08-31 after executing W-00, W-02b, W-06, W-07, W-08

Every item here is a claim **in this document** that execution falsified, with
the measurement that falsified it. They are recorded against the plan rather
than quietly worked around, per RULE O. Full write-up in
[`SUMMARY.md`](./SUMMARY.md) §"Session 2"; the raw numbers are committed under
[`baselines/`](./baselines/).

### E-1 — §2.1's "transform = 93 % of `get_k_kpts`" no longer holds

It was 93 % *before* W-02. After the mixed-radix/Rader landing the isolated
transform batch is **71 %** of `get_k_kpts` at mesh 21 (1 016 ms of 1 428 ms),
because the transform got ~5× cheaper and the contractions did not move. The
non-transform share is therefore **29 %**, four times what W-03's PRIORITY note
was written against. §2.1's table should be read as a pre-W-02 snapshot.

### E-2 — §2.1's "a pure functional is already fast" is drawn from the wrong number

True per ITERATION, false per RUN. A converged pure-PBE SCF iteration is 27 ms
(mesh 21) / 94 ms (mesh 31) — under 1.5 s of a 22.7 s run. **74 % of that run
is the cold `eval_ao_kpts` pass** (16.8 s at mesh 31) and 21 % is `get_hcore`.
§2.1 measured the same 7.26 s cold figure and then reasoned from the warm one.

Consequence: **W-06 targets ~3 % of a pure-functional run, not "the dominant
cost"**, and the item that would move one is a faster `eval_ao_kpts`.

### E-3 — W-03 and W-04 do not land; the decision rule is W-03's own

W-03 step 4 made the item conditional on a measurement. `krks_profile contract`
is that measurement, on the two shapes involved, at the gate mesh:

* `(nao,nao)·(nao,Ng)` — host 22.9 GFLOP/s, `zgemm_dense` 3.4 → **6.7× slower**
* `(nao,Ng)·(Ng,nao)` — host 15.9 GFLOP/s, `zgemm_dense` 1.9 → **8.3× slower**,
  and the GEMM's unordered grid sum differs from `oracle_dot`'s pairwise tree by
  **1.35e-10 — 13× the 1e-11 gate tolerance.**

So W-03 step 1's `accumulate_vk → zgemm_dense` and W-06 step 2's
`vxc_mat_one → zgemm_h_dense` would **break Gate A**, not merely slow it. They
also directly contradict W-05, which exists to put those two reductions on the
ordered primitive.

W-04 cannot rescue them either, for a structural reason the plan does not
account for: **the 3-D transform is a HOST routine.** `rho1` must return to the
host and `vR` must go back out on every block, so the round trip W-04 exists to
remove is imposed by the FFT, not by the contractions — and transfer is only
about a third of the device time anyway. W-03/W-04 should be re-scoped as
"device J/K, **blocked on a device FFT**", with `krks_profile contract` as the
gate that re-opens them.

### E-4 — W-06 steps 3 and 4 are not appropriate on this backend

Step 3 (delete the per-element zero short-circuits) is justified by SIMT branch
divergence; the default backend is the CPU runtime, which has none. Removing
them is also a numerical change (`-0.0 + 0.0 == +0.0`). Step 4 is a *kernel*
fusion item; on the host those four passes are sub-millisecond against a 28.5 ms
`nr_rks`.

### E-5 — W-07's bit-identity criterion is unachievable, and its reasoning is wrong

"`nr_rks` output must be bit-identical across `PYSCF_PBC_NUMINT_BLKSIZE` once
the block-independent accumulation is in" cannot hold. `oracle_sum` is a
pairwise tree whose shape is a function of input LENGTH, so
`oracle_sum([oracle_sum(b₀), …])` is a different tree from
`oracle_sum(b₀ ++ b₁ ++ …)` for any partition with more than one block, and
floating-point addition is not associative. Only concatenating every block is
partition-independent, and that defeats blocking. The achievable contract —
bit-identical at the default whole-grid partition, 1e-13 relative across
partitions — is what `crates/pyscf-pbc-dft/tests/numint_blocking.rs` asserts.

The per-block-partial reduction is still worth having, and landed: it removes a
naive sequential `+=` from the two quantities that reach the total energy.

### E-6 — W-08 cites the wrong upstream mechanism

`kk_adapted_iter` (`pbc/df/aft_jk.py`) is **not** a conjugate-pair halving. It
groups `(ki, kj)` by unique wrapped `dk` so that one analytic `ft_ao_pair`
tensor serves the group. FFTDF has no `ft_ao_pair`: its per-pair cost is a
batched 3-D transform of `rho1`, which depends on the AO tables at `k1` **and**
`k2` individually. Ported verbatim it saves nothing here — the only thing it
groups is the `get_coulG` that W-01 already caches. So W-08's instruction "do
not invent a symmetry argument; use upstream's" cannot be followed as written:
upstream has no such argument to borrow for this builder.

The relation that does hold was derived and then **verified against the full
`Nk²` loop to 5.9e-15 relative** (`tests/fft_jk_kk_symmetry.rs`):

```text
rho1^{21}[(i,j),g] = conj( rho1^{12}[(j,i),g] );  FFT(conj x)[G] = conj(FFT(x)[-G]);
coulG_{-dk}[G] = coulG_{dk}[-G]   =>   vR^{21}[(i,j),g] = conj( vR^{12}[(j,i),g] )
```

One FFT/iFFT pair therefore serves both orientations. The two contributions
still differ (different density matrices, different AO tables), so **only the
transform is halved** — which is why the measured factor is **1.56× (mesh 21) /
1.75× (mesh 31)** and not the "clean 2×" W-08 predicts.

It needs `G_{-n} = -G_n` to be an exact permutation of the reciprocal grid,
which fails on an EVEN mesh axis (its Nyquist frequency `-m/2` has no `+m/2`
partner). That, `hermi == 1`, no band k-points, and the full `nao` block are
checked and are **errors**, never silent fallbacks.

Re-baselined gate (W-08's own TEST asks for this), `PYSCF_PBC_KK_SYMMETRY=1`:
7/7 pass and **only `KRKS Si 2×2×2 PBE0` moves at all — by one ulp**
(`-7.796816043923375` → `-7.796816043923376`, residual 5.589e-12 → 5.588e-12).
Every other case is bit-identical to the full loop. The existing tolerances
hold unchanged.

### E-7 — §8 Q3's erratum against `PBC-MASTER-PLAN.md` plan 11-01 is now due twice over

§8 Q3 already noted that the shipped `FftEngine::Stockham` default *disagrees
with* plan 11-01's mandate to route the transform through `zgemm_dense`, and
asked for an explicit erratum. E-3 strengthens it from a speed argument into an
**accuracy** one: on the grid-reduction shape the device GEMM is 1.35e-10 away
from the ordered route, i.e. outside the gate. Plan 11-01's assumption that
`zgemm_dense` is the fast *and* safe route is wrong on both counts for this
backend.


### E-8 — W-09 is no longer deferred, and it is the largest single win in the plan

W-09's DEFER UNTIL clause ("until W-00 has a large-cell baseline that shows AO
evaluation is actually a material share of the time") is met and then some:
AO collocation is **74 %** of a pure-functional run on the 8-AO gate cell and
**62 %** on Si `gth-dzvp` 2×2×2. It landed, on by default, with these results:

| | baseline | with W-09 |
|---|---|---|
| Si `gth-szv` 2×2×2 mesh 31 PBE, full `kernel()` | 22 669 ms | **9 460 ms** (2.40×) |
| Si `gth-dzvp` 2×2×2 mesh 31 PBE, full `kernel()` | 161 818 ms | **49 870 ms** (3.24×) |
| Si `gth-szv` 2×2×2 mesh 21 PBE0, full `kernel()` | 14 999 ms | **11 004 ms** (1.36×) |

Two corrections to the item's own text:

* It predicts the payoff is "approximately zero on the 8-AO gate cell". It is
  **2.4×** there. The plan reasoned from `nao`, but the cost is dominated by the
  LATTICE-IMAGE loop, which does not shrink with `nao` — the screen rejects 64 %
  of the images on that cell outright.
* It expects the accuracy to get worse ("screened terms are dropped"). **Gate A's
  residuals got SMALLER** — He AE −9.281e-14 → −8.615e-14, LDA 6.506e-12 →
  6.495e-12, PBE0 5.589e-12 → 5.587e-12 — because upstream screens and this port
  did not. Screening moves the port TOWARDS its oracle.

### E-9 — Gate C's scan list is extended, and one crate cannot join it

§1.5 says crates join `SCAN_TARGETS` as they accrete oracle-relevant numeric
paths. `pyscf-pbc-gto`, `pyscf-pbc-df` and `pyscf-pbc-tools` now do, and all
three are FMA-clean. `pyscf-pbc-dft` could not be added: it pulls in the
`libxc_rs` rayon kernels, and `libxc-rkernel-mgga_c_tpssloc` **segfaults rustc**
under `release-oracle`'s `codegen-units = 1`. Recorded in the scan list itself.

### E-10 — a D-PBC-17 gap this plan does not name: `ztrace_ab` / `trace_dm_v`

Chasing a 2-ulp wobble in `KRKS Si 2×2×2 PBE0` (see `SUMMARY.md`) ruled out
W-09, the AO screen, test scheduling and FMA contraction. What remains is that
`zlinalg::ztrace_ab` — `Tr(AB)` over `nao²` terms with a naive `sr += …`, feeding
`ecoul` through `krks::trace_dm_v` — is an unordered reduction on the energy
path. That is the same violation W-05 fixed in `fft_jk` and W-07 fixed in
`nr_rks`, in a routine neither item listed. It should be routed through
`oracle_sum`.
