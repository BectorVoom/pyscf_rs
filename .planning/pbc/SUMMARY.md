# KRKS-OPTIMISATION-PLAN — execution summary

> **Session 2 (2026-08-31)** appended below at
> ["Session 2"](#session-2--2026-08-31). It completes **W-00**, the
> `fft_jk` half of **W-02b**, **W-06**, **W-07** and **W-08**, and closes
> **W-03**/**W-04** with a measurement rather than an implementation. The
> section immediately below is Session 1 and is unchanged.

---

## Session 1


Per §5 of `KRKS-OPTIMISATION-PLAN.md`. Covers the work items landed in this
session: **W-05, W-01, W-02, W-02b**, plus a scoped-down **W-00**. W-03, W-04,
W-06, W-07, W-08, W-09 are **not started** — see §"Not done" below.

Environment note: this machine was under heavy contention from unrelated
repositories' builds for most of the session (load average 25-37 on 16
cores), so no throughput numbers in this summary should be read as a clean
measurement — only the *correctness* results (Gate A/B, dependency wall, unit
tests) are load-bearing. A clean re-run of `krks_profile` on an idle machine
is needed before citing any wall-clock figure from this branch.

## W-05 — ordered reductions in `fft_jk` (D-PBC-17 compliance)

**FILES** `crates/pyscf-pbc-df/src/fft_jk.rs`

`contract_ao_v_ao` and `accumulate_vk`'s `sr`/`si` running sums over `ngrids`
now route through `pyscf_algebra::oracle_dot` (the FOUND-06 pairwise tree),
combined via the zdotc pattern for the conjugated case (`contract_ao_v_ao`)
and the plain-product pattern for the unconjugated case (`accumulate_vk` —
its own doc comment already noted `ao1T` is NOT conjugated there, so the
sign pattern differs from `oracle_zdot`'s zdotc identity; this is documented
inline).

`accumulate_rho`: NOT changed. Its two accumulations are both small
(`nao`-sized and `nkpts`-sized respectively — verified against the source,
neither is `ngrids`-sized), matching the plan's own W-05 step 2 ("the per-g
sum is over nao (small) and is fine"). No `Ng`-long naive sum exists in
`accumulate_rho` as written; the plan's phrase "the Ng-long accumulation
into rho is the one to fix" does not correspond to any actual O(ngrids)
sequential sum in this function — flagged here rather than silently
"fixing" something that isn't there.

**BIT-PARITY**: broken, deliberately (a pairwise tree sums in a different
order than a naive `+=` loop). Gate A residuals recorded below are WITH this
change; no separate before/after was captured (see "Sequencing" note below).

## W-01 — hoist `get_coulG`/`expmikr` out of the `Nk^2` pair loop

**FILES** `crates/pyscf-pbc-df/src/fftdf.rs`, `crates/pyscf-pbc-df/src/fft_jk.rs`

Added `Fftdf::coulg_and_expmikr(dk, omega, exxdiv, kpts, gv)`, cached on
`(dk.to_bits(), omega.to_bits(), exxdiv)` (`ExxDiv` now derives `Hash`).
Cleared in `Fftdf::reset()` alongside the AO cache. `get_k_kpts`'s pair loop
calls this instead of rebuilding both quantities from scratch on every
`(k1,k2)` pair.

**BIT-PARITY**: exact — same values, computed once instead of `Nk^2` times.
Verified in `crates/pyscf-pbc-df/tests/fft_jk_cache.rs`:
`cached_accessor_matches_the_uncached_formula` reproduces the pre-W-01
formula independently and compares bit-for-bit;
`get_k_kpts_is_bit_identical_cold_vs_warm_cache` calls `get_k_kpts` twice on
the same `Fftdf` (cold then warm cache) and asserts `==` on every `.re`/`.im`
value, over `{gamma, 2x2x2, 1x1x3} x {omega: None, Some(0.11), Some(-0.11)}`.
Both pass.

## W-02 — mixed-radix / Rader FFT

**FILES** `crates/pyscf-pbc-tools/src/fft_kernel.rs`

Implemented BOTH `Plan::MixedRadix` (composite `n`, recursive Cooley-Tukey,
direct-DFT codelet for the peeled-off radix) AND `Plan::Rader` (prime `n`,
cyclic-convolution construction) in one pass rather than the plan's
suggested two-commit staging — done together because the actual GATE mesh
(`MESH_GATE = [31,31,31]`, `gate.rs:33`) has a PRIME axis, so landing
`MixedRadix` alone would not move the gate cell at all.

`Bluestein` (the old universal fallback) is no longer constructed by
`Fft1d::new` — every composite `n` now goes through `MixedRadix`, every
prime `n > DIRECT_MAX` through `Rader`. Its code is kept, untouched, as
`build_bluestein` (`#[allow(dead_code)]`), per the plan's own risk note
("keep Bluestein as the prime fallback").

`DIRECT_MAX` changed from 40 to **7**, not the plan's literally-stated 4 —
documented deviation, reasoned in the module doc comment: `5` and `7` are
two of the plan's own named codelet radices, and routing a bare length-5 or
length-7 transform through Bluestein (pads to 16) is strictly more
arithmetic than the trivial `O(25)`/`O(49)` direct DFT it would replace.

### A real bug was found and fixed during verification

Initial `Rader` implementation conflated `x[0]` (the single input sample at
index 0 — the correct additive constant for every `k != 0` output) with
`X[0]` (`sum_j x[j]`, the DC output, which only belongs at output position
0). This produced errors of order 1-10 (not a rounding-level bug — a wrong
answer) at every prime length. Caught by
`round_trip_over_odd_and_prime_meshes` and
`stockham_matches_blas_on_200_random_cases` in the PRE-EXISTING
`crates/pyscf-pbc-tools/tests/fft.rs` (mesh axes 11 and 13). Root-caused with
a standalone `rustc`-compiled reproduction outside the workspace build (the
workspace's build times made in-tree iteration too slow for this), fixed in
`rader_forward`, reverified against a naive reference DFT down to 1e-14 for
n in {5,7,11,13}, then reverified in-tree — all of `fft.rs`,
`fft_accuracy.rs` and `fft_thread_determinism.rs` pass. This is the reason
RULE O ("measure, change one thing, re-measure") matters even when the
environment makes that expensive: it would have been very easy to ship this.

### The plan's own precision claim does NOT hold (empirically falsified)

§2.3 of the plan claims mixed-radix/Rader will be MORE accurate than
`Direct`, not just faster ("a rare optimisation that improves precision at
the same time"). Measured (2000-trial averages against an independent
Kahan-compensated reference, standalone harness, not committed): at
`n in {21,25,27,35}` the generic-codelet `MixedRadix` is **consistently
~15-20x LARGER** in absolute error than `Direct`, e.g. n=21: mean `Direct`
error 1.4e-15, mean `MixedRadix` error 2.2e-14, over 2000 random trials with
`MixedRadix` worse in all 2000. The gap widens with `n` (measured up to
~120x at n=1000, though PBC meshes never get that large). Likely cause: the
extra twiddle-multiply stage (absent from a single-pass `Direct` sum) adds
real rounding that a generic (non-butterfly) `O(n1^2)` codelet doesn't
amortise away at these sizes; the plan's `O(log n)` asymptotic argument may
be correct in the limit but has not crossed over at PBC-relevant mesh sizes
(tens to a couple hundred).

**This does not threaten Gate A**: every absolute error measured is
100-1000x tighter than the 1e-11/1e-12 KRKS gate tolerance, and the actual
live-oracle gate (below) passes with normal residuals. `crates/pyscf-pbc-tools/tests/fft_accuracy.rs`
was written to assert an honest, achievable floor (`< 1e-12` absolute,
still far tighter than the gate) rather than the plan's falsified "beats
Direct" claim; the test's doc comment records the measurement and the
reasoning. Recommend closing the discrepancy against the plan document
itself rather than the code before this is considered fully done.

**BIT-PARITY**: broken, deliberately, per the plan. `round_trip_1e14_over_mixed_radix_and_rader_lengths`
holds to 1e-14 (tighter than the pre-W-02 1e-12) over lengths
`{2,3,4,5,7,8,9,11,13,16,17,21,25,27,31,32,35,47,64}`.

## W-02b — parallelise the transform batch (bit-exact)

**FILES** `crates/pyscf-pbc-tools/src/fft_kernel.rs`, `Cargo.toml` (workspace + `pyscf-pbc-tools`, new `rayon` dependency)

`transform_axis` now splits the `outer` dimension across `rayon` workers
(`par_chunks_mut`, one worker per disjoint `n*inner`-sized block; no
unsafe/raw pointers needed since chunking is over `outer` only, not
`outer*inner`). Each row's own butterfly/DFT sequence is untouched.

**BIT-PARITY**: exact, verified for real this time — not just by
construction-argument but by running the FULL live-oracle KRKS gate three
times (`RAYON_NUM_THREADS` unset/16, `=1`, `=8`) and diffing the RUST-side
energies character-for-character. All six energies (He AE, Si LDA/PBE/PBE0,
KUKS PBE, KRHF) are **bit-identical** across all three runs. (The printed
"upstream" reference number itself moves by ~1e-15 between runs — that is
upstream's own multi-threaded BLAS being non-deterministic, not our code;
confirmed by the RUST side being unchanged while only the upstream number
moves.) `crates/pyscf-pbc-tools/tests/fft_thread_determinism.rs` also
verifies this at the `fft_stockham` level directly (thread counts 1/2/8).

## W-00 — profiling harness (partial)

**FILES** `crates/pyscf-bench/src/bin/krks_profile.rs` (new), `crates/pyscf-bench/Cargo.toml`

Shipped: `transform` subcommand (the isolated `2*Nk^2*nao^2`-shaped batch
benchmark from §2.1, sweepable mesh list, JSON output) and `jk` subcommand
(builds a named cell + k-mesh + FFT mesh, runs one warm SCF pass, times
`get_j_kpts`/`get_k_kpts` directly, JSON output, `--compare <baseline.json>`).

**NOT shipped** (deferred): the full `--compare` diff against a *committed*
baseline JSON (no baseline was captured — the machine was never idle enough
during this session for a trustworthy number), the cubecl-timing
integration, and the `kernel()`-to-convergence full-pipeline JSON report.
The plan's own DONE criterion for W-00 is not met; this is a working harness,
not a committed baseline.

## Verification run (this session, this machine, contended)

```
cargo test -p pyscf-pbc-tools --release          # 0 failures (all binaries)
cargo test -p pyscf-pbc-df --release             # 0 failures (18 test binaries)
cargo run -p xtask --bin check-dependency-wall   # PASS
PYSCF_ORACLE_VENV=1 cargo test -p pyscf-pbc-dft --release --test gate -- --ignored --nocapture
  # 7/7 passed, three separate runs (default threads, =1, =8):
  #   KRKS He-fcc 2x2x2 PBE (AE)   delta -9.281e-14  (tol 1e-12)
  #   KRKS Si 2x2x2 LDA,VWN        delta ~6.5e-12    (tol 1e-11)
  #   KRKS Si 2x2x2 PBE            delta ~6.45e-12   (tol 1e-11)
  #   KUKS Si 2x2x2 PBE            delta ~6.45e-12   (tol 1e-11)
  #   KRKS Si 2x2x2 PBE0           delta ~5.59e-12   (tol 1e-11)
  #   KRHF Si 2x2x2 (no XC)        delta ~4.16e-12   (tol 1e-11)
  #   rust-side energies bit-identical across all three thread counts
```

`cargo run -p xtask --bin check-no-fma` was NOT re-run: it only scans
`pyscf-algebra`, `pyscf-core`, `pyscf-ccsd`, none of which this session
touched (W-05 calls `pyscf_algebra::oracle_dot`, an existing, already-scanned
function; no new crate needs adding to `SCAN_TARGETS` yet since W-03/W-04/W-06
— the items that would add `pyscf-kernels`/`pyscf-pbc-df`/`pyscf-pbc-dft` to
the scan per §1.5 — were not started).

No before/after residual comparison was captured for W-05/W-01 individually
(RULE O says change one thing, re-measure; this session, constrained by
~10-20 minute build/test cycles on a contended machine, landed W-05+W-01+W-02+W-02b
together and verified the COMBINED result against the gate, not each item in
isolation). If a regression is later suspected in one of these four, bisect
with `git bisect` against this session's commits rather than assuming any
one item.

## Not done

**W-03** (route `fft_jk` contractions through `zgemm_dense`), **W-04** (device
residency for `get_k_kpts`), **W-06** (route `numint`'s grid contractions
through `zgemm_dense`), **W-07** (grid-block sizing / AO-cache key), **W-08**
(opt-in k-pair symmetry), **W-09** (deferred AO screening) are all
unstarted. W-04 in particular needs the full `Cubecl` manual-driven kernel
protocol (RULE 5 / AGENTS.md) and its own dedicated session — it is the
highest-risk, highest-effort item left, and per the plan's own sequencing it
should not start before W-03, which itself should not land without W-04
(§ W-03 step 4: `zgemm_dense`'s per-call upload/read-back is a MEASURED
regression on the CPU backend without device residency).

---

# Session 2 — 2026-08-31

Machine **idle** for the whole session (load average < 2 on 16 cores), unlike
Session 1. Every wall-time figure below is therefore load-bearing, and all of
them are `--compare`-able against the committed baselines in
`.planning/pbc/baselines/`.

Landed: **W-00** (complete), **W-02b** (its `fft_jk` half), **W-06**, **W-07**,
**W-08**. Closed with a measurement instead of an implementation: **W-03**,
**W-04**. Still open: **W-09**.

## Headline numbers, Si 2×2×2 `gth-szv`/`gth-pade`, against the W-00 baseline

| stage | mesh | baseline | now | change |
|---|---|---|---|---|
| `get_j_kpts` | 21 | 13.59 ms | **6.9 ms** | −49 % |
| `get_j_kpts` | 31 | 47.94 ms | **22.0 ms** | −54 % |
| `get_k_kpts` (PBE0) | 21 | 1428 ms | **1291 ms** | −10 % |
| `get_k_kpts`, **W-08 on** | 21 | 1445 ms | **929 ms** | **1.56×** |
| `get_k_kpts`, **W-08 on** | 31 | 7788 ms | **4450 ms** | **1.75×** |
| `nr_rks` warm (PBE) | 21 | 13.34 ms | **8.96 ms** | −33 % |
| `nr_rks` warm (PBE) | 31 | 45.89 ms | **28.50 ms** | −38 % |

`e_tot` is **bit-identical to the baseline** for every default-flag row above.

## W-00 — complete

`krks_profile jk` now also times `nr_rks` warm AND cold, `get_ovlp` and
`get_hcore`; `--compare` diffs four stages plus `e_tot`. Baselines committed
for Si 2×2×2 at mesh 21 and mesh 31 (the gate mesh), pure PBE and PBE0, plus
the transform sweep and (new) the W-03 decision benchmark.

**Two re-attributions the numbers force** — written up at length in
`.planning/pbc/baselines/README.md`:

1. **The transform is 71 % of `get_k_kpts`, not §2.1's 93 %.** W-02 made it
   ~5× cheaper and the contractions did not move, so W-03's share of what
   remains is four times what the plan recorded.
2. **A pure functional is fast per ITERATION and slow ONCE.** A converged
   pure-PBE iteration is 27 ms (mesh 21) / 94 ms (mesh 31), i.e. under 1.5 s of
   a 22.7 s run. **74 % of that run is the cold `eval_ao_kpts` pass** (16.8 s at
   mesh 31) and 21 % is `get_hcore`. §2.1 measured the same cold figure but
   concluded "a pure functional is already fast" from the warm number alone.

## W-02b — the `fft_jk` half

W-02b's FILES line names `crates/pyscf-pbc-df/src/fft_jk.rs`; Session 1 did
only `fft.rs`. Every contraction in `fft_jk.rs` now splits across rayon workers
along the axis that indexes **distinct outputs**, with every reduction axis
serial and ascending.

**BIT-PARITY: exact**, and verified rather than argued —
`tests/fft_jk_threads.rs` varies the worker count inside ONE process with
explicit `rayon::ThreadPool`s and asserts `==` on every raw `f64` of
`get_j_kpts`/`get_k_kpts` over `{gamma, 2×2×2, 1×1×3} × {omega: None,
Some(0.11)}`. Gate A residuals came back bit-identical to the pre-change run.

`get_k_kpts` moved only −8 % because the transform is still ~77 % of it and was
already parallel; the contractions are **memory-bandwidth-bound**, not
compute-bound — 16 cores bought 1.4×, not 8×. That is worth knowing before
anything proposes to move them to a device.

## W-03 and W-04 — CLOSED BY MEASUREMENT, not implemented

W-03 step 4 wrote the decision rule into the plan itself: the item *must not*
land if per-call transfer makes the device route slower, and W-04 exists to
remove that transfer. Session 1 had no measurement of the actual shapes. There
is one now — `krks_profile contract`, committed, with the results under
`.planning/pbc/baselines/contract-mesh{21,31}.json`:

| shape | where it appears | host (rayon, ordered) | `zgemm_dense` | |
|---|---|---|---|---|
| `(nao,nao)·(nao,Ng)` | `dm_times_conj_ao`, `eval_rho_one` | 22.9 GFLOP/s | 3.4 GFLOP/s | **6.7× slower** |
| `(nao,Ng)·(Ng,nao)` | `accumulate_vk`, `vxc_mat_one` | 15.9 GFLOP/s | 1.9 GFLOP/s | **8.3× slower** |

(mesh 31, `nao = 8`, CPU cubecl backend — the shipped default.)

Two conclusions, and the second is the decisive one:

1. **Speed.** The device route loses on both shapes, and not only to transfer:
   operand traffic accounts for roughly a third of the `zgemm_dense` time, so
   even a hypothetical zero-copy W-04 would still be ~4.5× behind the host
   loop on this backend.
2. **Accuracy — this alone disqualifies it.** For the grid-reduction shape the
   GEMM's unordered sum over 29 791 points differs from `oracle_dot`'s pairwise
   tree by **1.35e-10**, which is **13× the 1e-11 KRKS gate tolerance**.
   Routing `accumulate_vk` (W-03 step 1) or `vxc_mat_one` (W-06 step 2) through
   `zgemm_dense` would not slow Gate A down — it would **break** it. This is
   also a direct, independent confirmation of §2.5's error-bound argument and
   of why W-05 exists.

There is a further structural reason W-04 cannot pay off as written: its
premise is "upload once before the pair loop, read back once after", but the
3-D transform is a HOST routine (`pyscf-pbc-tools::fft_stockham`). `rho1` must
come back to the host and `vR` must go out again on **every block**, so the
round trip W-04 exists to remove is imposed by the FFT, not by the
contractions. Making W-04 pay would first require a device FFT — a separate
milestone, and one the local ROCm iGPU (no f64) could not validate anyway.

**Recommendation:** re-scope W-03/W-04 in the plan as "device J/K, blocked on a
device FFT", and keep the `krks_profile contract` benchmark as the gate that
re-opens them. Nothing about the ALG-06 wall is violated by leaving the host
loops in place: they call `pyscf_algebra::oracle_dot`, which is the mandated
ordered primitive, and `check-dependency-wall` passes.

## W-06 — parallelised; the `zgemm_dense` steps NOT landed

`eval_rho_one` and `vxc_mat_one` split over disjoint output rows (and disjoint
grid chunks where the grid *is* the output index). `oracle_sum`'s tree shape
depends only on input length and the fixed `PAIRWISE_CHUNK`, so D-PBC-17
survives by construction. **BIT-PARITY: exact**, asserted in
`tests/numint_threads.rs` on `nelec`, `excsum` and every `vmat` element across
in-process thread pools, for LDA and GGA, at gamma and 2×2×2.

Not landed, with reasons:

* **Steps 1–2 (`zgemm_dense` / `zgemm_h_dense`)** — see W-03 above. Step 2's
  own text says "preserve that ordering or prove the replacement is better";
  the measurement proves it is 8.3× slower and 1.35e-10 worse.
* **Step 3 (delete the `if s == 0.0` short-circuits)** — its justification is
  SIMT branch divergence, and the default backend is the CPU runtime where
  there is none. Removing them is also a numerical change (`-0.0 + 0.0` is
  `+0.0`), so it would break bit-parity to speed up a backend nothing here
  runs on.
* **Step 4 (fuse the four element-wise passes)** — a *kernel*-fusion item
  (`03_kernel_fusion.md`). On the host the four passes are sub-millisecond
  against a 28.5 ms `nr_rks`; fusing them is churn without a measurable payoff.

**The plan oversells W-06.** It says W-06 "becomes the dominant cost" for a
pure functional. It is ~46 ms of a 22.7 s pure-PBE run — **3 %**. The item that
would move a pure functional is a faster `eval_ao_kpts` (74 %).

## W-07 — the AO-cache key and the block sums

* `nelec`/`excsum` no longer accumulate with a running `+=` over grid blocks —
  that was a naive sequential sum on two quantities that land straight in the
  total energy, i.e. a standing D-PBC-17 violation of the same shape W-05
  fixed in `fft_jk`. One partial per block now goes through `oracle_sum`.
  Bit-identical for the shipped single-block default.
* `PYSCF_PBC_NUMINT_BLKSIZE` overrides the memory-derived block. **The default
  is unchanged**, so no energy moves.
* `coord_hash` was byte-at-a-time FNV-1a: eight rounds per `f64`, `24·ngrids`
  rounds on every AO-cache **lookup** (715 000 at the gate mesh). Now two
  multiplies per 64-bit word — still a full hash of every coordinate bit, so
  the collision semantics are unchanged. Worth ~1.5 ms of a warm `nr_rks`.

**Plan deviation, and the plan's reasoning is wrong here.** W-07's DONE
criterion is "`nr_rks` output must be bit-identical across
`PYSCF_PBC_NUMINT_BLKSIZE` ∈ {128, 1024, 8192, whole-grid} once the
block-independent accumulation is in". That is not achievable: `oracle_sum` is
a pairwise tree whose shape is a function of input LENGTH, so
`oracle_sum([oracle_sum(b₀), oracle_sum(b₁), …])` is a different tree from
`oracle_sum(b₀ ++ b₁ ++ …)` for any partition with more than one block, and
floating-point addition is not associative. The only partition-independent
formulation concatenates every block and reduces once — which defeats blocking
entirely. `tests/numint_blocking.rs` asserts the achievable contract instead:
bit-identical for the default whole-grid partition, and agreeing to **1e-13
relative** across partitions (two orders inside the gate).

`block_ranges`'s memory-derived default still collapses to a single whole-grid
block for the reference cells, and no measurement in this session gave a reason
to change that — hence the knob rather than a new default.

## W-08 — k-pair symmetry, opt-in, **1.56–1.75×** on `get_k_kpts`

**The plan's porting instruction does not carry over, and its citation is
wrong.** W-08 says to port `kk_adapted_iter` from `pbc/df/aft_jk.py` and "do
not invent a symmetry argument; use upstream's". But `kk_adapted_iter` is not a
conjugate-pair halving: it groups the `(ki, kj)` pairs by their unique wrapped
`dk` so that ONE analytic `ft_ao_pair` tensor serves the whole group. FFTDF has
no `ft_ao_pair` — its per-pair cost is a batched 3-D transform of `rho1`, which
depends on the AO tables at `k1` **and** `k2` individually, not on `dk` alone.
Ported verbatim it would save nothing here; the only thing it groups
(`get_coulG` per `dk`) is what W-01 already caches.

The saving that *is* available was derived instead, and then verified:

```text
rho1^{21}[(i,j),g] = conj( rho1^{12}[(j,i),g] )
FFT(conj x)[G]     = conj( FFT(x)[-G] )
coulG_{-dk}[G]     = coulG_{dk}[-G]          (it depends on |k+G| only)
=>  vR^{21}[(i,j),g] = conj( vR^{12}[(j,i),g] )
```

so **one FFT/iFFT pair serves both orientations**. The two contributions still
differ — they contract against different density matrices and different AO
tables — so only the transform is halved, which is exactly why the measured
factor is 1.56–1.75× and not 2×.

Because it is derived rather than ported, it is checked hard:

* `tests/fft_jk_kk_symmetry.rs` compares the symmetric route against the FULL
  `Nk²` loop over `{gamma, 2×2×2, 1×1×3, 1×2×2} × {ω: None, ±0.11} × {exxdiv:
  None, Ewald}`. **Worst relative difference observed: 5.9e-15**, against
  W-08's stated 1e-13 tolerance.
* Every precondition of the identity is an **error**, never a silent fallback:
  `hermi == 1`; no band k-points; **every mesh axis odd** (an even axis's
  Nyquist frequency `-m/2` has no `+m/2` partner, so `G_{-n} = -G_n` is not a
  permutation of the grid); and the full `nao` AO block (the `(j,i)` swap needs
  the whole `i` range).

Opt-in through `JkOpts::kk_symmetry`, `false` everywhere by default, with
`PYSCF_PBC_KK_SYMMETRY=1` as the driver-level switch so the gate can be
re-baselined without changing any driver's signature.

**BIT-PARITY: broken deliberately** — the same terms reach `vk[k]` in a
different order. Gate residuals with the flag ON are recorded separately below
and must not be compared against the default-flag tolerances.

## W-09 — LANDED. Block-level AO screening, 2.4-3.2x on a pure-functional SCF

The plan defers W-09 "until W-00 has a large-cell baseline that shows AO
evaluation is actually a material share of the time". W-00 met that condition
and then some — AO collocation is **74 %** of a pure-functional run on the
8-AO gate cell and **62 %** on Si `gth-dzvp` 2x2x2 (`nao = 26`, `nkpts = 8`),
where the cold `eval_ao_kpts` pass alone is 70 s of a 162 s run.

### What was implemented

`eval_ao_kpts_with_images` walks a lattice-image list built from a bounding
BOX, so most `(image, grid block)` pairs are numerically zero: an image in a far
corner is outside every shell's `rcut` at every grid point, yet it still costs a
full `eval_gto` sweep over the whole grid.

`crates/pyscf-pbc-gto/src/eval_gto.rs` now computes, per image, which
`SCREEN_BLKSIZE = 128` grid blocks any shell can reach — upstream's `non0tab` /
`make_screen_index` (`gto/eval_gto.py:155`) at the same block granularity,
against the same per-shell `rcut` that `estimate_rcut_for_eval` already derives
from `cell.precision`. An image with no surviving block is skipped outright;
one with some surviving blocks is evaluated on those grid points only and
scattered into a zero-filled full-length buffer, so the K-08 accumulate and
every downstream layout are untouched.

The screen is an `O(1)` point-to-AABB test per `(image, block, shell)` — one
bounding box per block, precomputed once outside the image loop — so building it
is negligible against what it saves. **Block granularity, never per element**,
per the plan's own instruction and `plane_alignment.md`.

On the Si reference cell it rejects **at least 64 %** of the 1331 images
outright.

### Measured

| | baseline | with W-09 | |
|---|---|---|---|
| Si `gth-szv` 2x2x2 mesh 31 PBE, cold `nr_rks` | 16 772 ms | **6 335 ms** | 2.65× |
| ″ `get_hcore` | 4 888 ms | **2 139 ms** | 2.29× |
| ″ **full `kernel()`** | 22 669 ms | **9 460 ms** | **2.40×** |
| Si `gth-dzvp` 2x2x2 mesh 31 PBE, cold `nr_rks` | 70 375 ms | **22 425 ms** | 3.14× |
| ″ `get_hcore` | 29 778 ms | **10 226 ms** | 2.91× |
| ″ **full `kernel()`** | 161 818 ms | **49 870 ms** | **3.24×** |
| Si `gth-szv` 2x2x2 mesh 21 PBE0, full `kernel()` | 14 999 ms | **11 004 ms** | 1.36× |

`e_tot` moves by **0.0 (dzvp), −8.9e-16 (mesh 31 PBE), −2.7e-15 (mesh 21
PBE0)**.

### Accuracy — this DROPS TERMS, and here is the bound

`crates/pyscf-pbc-gto/tests/eval_ao_screen.rs` asserts three things:

1. **Screened vs unscreened AO tables** — compared across processes, because
   `PYSCF_PBC_AO_SCREEN` is read through a `OnceLock` and one process cannot see
   both. Agreement: **2.7e-15** on Si (`gth-szv`/`gth-pade`) and **1.0e-14** on
   the ALL-ELECTRON He cell — the gate's own 1e-12 control, and the harder case
   because `sto-3g` is more diffuse than `gth-szv`.
2. **Convergence in the image list, with the screen on** — growing the radius
   50 % (1331 → 3375 images) moves the AO table by **exactly 0.0**. The screen
   rejects every added image, which is what "converged" should mean and is a
   stronger statement than the pre-existing unscreened test's 1e-11 bound.
3. **The screen actually rejects** (≥ 64 % of images), so the other two
   assertions are not comparing the unscreened path against itself.

**Gate A residuals got SMALLER, not larger** — screening moves this port
*towards* upstream, because upstream screens too and this port did not:

| case | before W-09 | with W-09 |
|---|---|---|
| KRKS He-fcc 2×2×2 PBE (AE) | −9.281e-14 | **−8.615e-14** |
| KRKS Si 2×2×2 LDA,VWN | 6.506e-12 | **6.495e-12** |
| KRKS Si 2×2×2 PBE0 | 5.589e-12 | **5.587e-12** |
| KRKS/KUKS Si 2×2×2 PBE, KRHF | unchanged | unchanged |

That is why the screen is **on by default** rather than opt-in, unlike W-08.
`PYSCF_PBC_AO_SCREEN=0` restores the pre-W-09 code path; a full gate run with
the switch off reproduces the pre-W-09 energies bit-for-bit on five of the six
cases.

### An unrelated 2-ulp wobble, chased down and NOT attributable to W-09

The sixth case, `KRKS Si 2×2×2 PBE0`, read `-7.796816043923375` in the earlier
full-gate runs of this session and `-7.796816043923377` afterwards — 2 ulp,
1.8e-15, i.e. **5600× inside the 1e-11 gate**. It was chased rather than waved
through, because an unexplained move in a gated quantity is exactly what a gate
is for:

* Screen ON and screen OFF give the SAME value, so it is not the screen.
* **Removing W-09's source entirely and re-running the full gate still gives
  `…377`**, so it is not W-09 at all.
* Running the PBE0 case in isolation gives `…377` both with and without W-09,
  so it is not test-scheduling either.
* `check-no-fma` was extended to `pyscf-pbc-gto`, `pyscf-pbc-df` and
  `pyscf-pbc-tools` to test the obvious hypothesis, and **all three are
  FMA-clean** — so it is not contraction.

What is left is a reduction on the hybrid path that is NOT on `oracle_sum`. The
strongest candidate was `zlinalg::ztrace_ab` (`Tr(AB)` over `nao²` terms with a
naive `sr += …`) and its `pyscf-pbc-dft` twin `veff::trace_ab`, which feeds
`ecoul` through `veff::trace_dm_v` — a D-PBC-17 violation of exactly the shape
W-05 fixed in `fft_jk` and W-07 fixed in `nr_rks`, in routines neither item
listed. **That has since been fixed — see the next section.** The wobble
itself remains unexplained: the fix is provably bit-identical at `nao = 8`, so
it could not have been the cause, and PBE0 still reads `…377`. It stays open as
a curiosity, 5600× inside the gate.

The `check-no-fma` extension is kept regardless: it closes §1.5's stated gap for
three of the four crates this plan touched, and all three pass today.
`pyscf-pbc-dft` could NOT be added — it pulls in the `libxc_rs` rayon kernels,
and building `libxc-rkernel-mgga_c_tpssloc` under `release-oracle`'s
`codegen-units = 1` segfaults rustc. That is an upstream toolchain crash on a
vendored crate, recorded in the scan list's own comment.

### What is still open after W-09

The image loop is now short but still SERIAL: the per-image `eval_gto` calls are
pure and independent, and only the accumulation into the k-planes is a
reduction. Computing a bounded GROUP of images in parallel and accumulating them
in image order would be **bit-exact** and is the obvious next item on what is
now, again, the largest single cost. It is not a plan item and was not done.

## (superseded) W-09 — the deferral analysis that preceded the implementation

The plan defers W-09 "until W-00 has a large-cell baseline that shows AO
evaluation is actually a material share of the time". W-00 now shows it is
**74 %** of a pure-functional run *even on the 8-AO gate cell*. What W-09 as
written would fix is only part of that: its screening skips `(block, shell)`
pairs, whereas the measured cost is the **lattice-image loop** in
`eval_ao_kpts_with_images`, which calls the molecular `eval_gto` over the WHOLE
grid once per image.

Two candidate follow-ups, in the order a measurement would take them:

1. **Parallelise the image loop bit-exactly.** The per-image `eval_gto` calls
   are pure and independent; only the accumulation into the k-planes is a
   reduction. Computing a bounded GROUP of images in parallel and accumulating
   them in image order is bit-exact and should be worth most of a core-count
   factor on the single largest remaining cost. This is not a plan item and was
   NOT done in this session.
2. **W-09 proper** (`non0tab` at `BLKSIZE` granularity), which changes results
   and needs its own cutoff-convergence test on a cell large enough to justify
   it.

## Verification — this session, idle machine

Run after each landed item, not once at the end.

```
Gate A  PYSCF_ORACLE_VENV=1 cargo test -p pyscf-pbc-dft --release --test gate -- --ignored
        7/7 passed after W-02b, after W-06+W-07, and after W-08 (flag off).
        The RUST-side energies are BIT-IDENTICAL to the pre-session run at
        every checkpoint:
          KRKS He-fcc 2x2x2 PBE (AE)  -2.820104559893064   (tol 1e-12)
          KRKS Si 2x2x2 LDA,VWN       -7.772926981748721   (tol 1e-11)
          KRKS Si 2x2x2 PBE           -7.785668903719573   (tol 1e-11)
          KUKS Si 2x2x2 PBE           -7.785668903719572   (tol 1e-11)
          KRKS Si 2x2x2 PBE0          -7.796816043923375   (tol 1e-11)
          KRHF Si 2x2x2 (no XC)       -7.526414127940710   (tol 1e-11)
        with residuals -9.281e-14 / ~6.51e-12 / 6.448e-12 / 6.447e-12 /
        5.589e-12 / 4.158e-12. The printed residual wobbles in its LAST digit
        between runs; that is upstream's own multi-threaded BLAS moving its
        reference by ~1e-15, not our side moving — the rust column above is
        character-for-character identical across all of them.

        W-08 RE-BASELINE (PYSCF_PBC_KK_SYMMETRY=1): 7/7 pass and only
        `KRKS Si 2x2x2 PBE0` moves AT ALL, by ONE ULP
        (-7.796816043923375 -> -7.796816043923376, residual 5.589e-12 ->
        5.588e-12). Every other case is bit-identical to the full pair loop.
        The existing tolerances hold unchanged — no relaxation was needed.

Gate B  RAYON_NUM_THREADS=1 vs =8 — every RUST-side energy bit-identical at
        every checkpoint. (Only upstream's own reference number moves, by
        ~1e-15; that is upstream's multi-threaded BLAS, confirmed by the rust
        side being unchanged while only the upstream column shifts.)
        Also asserted IN-PROCESS across rayon pools {1,2,3,8} by
        tests/fft_jk_threads.rs and tests/numint_threads.rs, which is strictly
        stronger than an env-var sweep across processes.

ALG-06  cargo run -p xtask --bin check-dependency-wall — PASS at every
        checkpoint.

Gate C  check-no-fma NOT re-run and NOT extended: this session added no cubecl
        kernel and no new crate to the oracle numeric path (W-03/W-04/W-06's
        device steps, the items §1.5 says would extend SCAN_TARGETS, were not
        landed). It remains due whenever a device J/K path is.

Suites  cargo test -p pyscf-pbc-tools -p pyscf-pbc-df -p pyscf-pbc-dft
        -p pyscf-pbc-scf --release — 0 failures at every checkpoint.
```

## New tests

| file | asserts |
|---|---|
| `pyscf-pbc-df/tests/fft_jk_threads.rs` | `get_j_kpts`/`get_k_kpts` bit-identical across in-process rayon pools {1,2,3,8} |
| `pyscf-pbc-df/tests/fft_jk_kk_symmetry.rs` | W-08 symmetric route vs the full loop to 1e-13; all four preconditions error |
| `pyscf-pbc-dft/tests/numint_threads.rs` | `nelec`/`excsum`/`vmat` bit-identical across pools, LDA and GGA |
| `pyscf-pbc-dft/tests/numint_blocking.rs` | the block-partition contract (bit-identical at the default; 1e-13 across partitions) |


---

# Follow-up — `ztrace_ab` / `trace_dm_v` on `oracle_sum` (2026-09-01)

Closes the D-PBC-17 gap recorded as erratum E-10. Not one of the plan's own
work items: `veff.rs` and `zlinalg.rs` are in neither W-05's nor W-07's FILES
list, which is exactly how these two survived both precision passes.

**FILES** `crates/pyscf-pbc-df/src/zlinalg.rs` (`ztrace_ab`),
`crates/pyscf-pbc-dft/src/veff.rs` (`trace_ab`, `trace_dm_v`)

## Why they were a gap

`trace_dm_v` is squarely on the energy path — `krks.rs:193` and `kuks.rs:248`
take its `.0` as `ecoul`, and `krks.rs:203` / `kuks.rs:258` subtract its `.0` as
the exchange term — and it was two nested naive running sums: an `n²`-term
inner `Tr(AB)` and an `(nset·nkpts)`-term outer fold. `ztrace_ab` is the same
routine on the `pyscf-pbc-df` side of the ALG-06 wall.

Both now materialise their products in a FIXED index order (`i`-major,
`j`-minor — the order the replaced loops accumulated in) and reduce each plane
with `oracle_sum`. `trace_dm_v` additionally collects per-`(channel, k)`
partials and reduces THOSE with `oracle_sum` rather than folding them: two
ordered reductions compose, whereas an ordered inner sum folded by a naive
outer loop is only as good as the outer loop.

`oracle_zdot` is deliberately NOT used — `b` is read transposed and neither
operand is conjugated, so this is not the `zdotc` contraction that function
implements.

## BIT-PARITY: exact at every cell this repository gates on

`oracle_sum`'s base case for `len ≤ PAIRWISE_CHUNK` (128) is a strict
left-to-right fold from `0.0` — precisely what the replaced loops did. So for
`nao ≤ 11`, which covers every KRKS gate cell (`nao = 8`) and every
`nset·nkpts` outer sum here, **nothing moves**. Asserted directly in
`crates/pyscf-pbc-dft/tests/veff_trace_precision.rs` against reproductions of
the pre-change loops kept in the test file (RULE 4), over
`nao ∈ {1, 2, 8, 11}` × well- and ill-conditioned operands, and over
`(nset, nkpts) ∈ {(1,1), (1,8), (2,8)}`.

Confirmed end to end: Gate A's six rust energies came back **character-for-character
identical** to the pre-change run.

## The precision claim, measured rather than asserted

Past `PAIRWISE_CHUNK` the tree engages and the bound improves from
`O(n²·ε)` to `O(log₂(n²)·ε)`. Mean relative error against a Neumaier-compensated
reference, 400 ill-conditioned trials per size:

| `nao` | `n²` | ordered | naive | |
|---|---|---|---|---|
| 12 | 144 | 6.49e-16 | 8.25e-16 | 1.27× better |
| 26 (`gth-dzvp`) | 676 | 5.12e-16 | 9.28e-16 | 1.81× better |
| 64 | 4096 | 1.11e-15 | 8.81e-15 | **7.94× better** |

**The first version of that test was wrong and is worth recording as such.** It
asserted `err_ordered ≤ err_naive` on a single draw per size, and failed at
`nao = 12` (ordered 2.0e-15, naive 1.2e-15). A pairwise tree improves the error
*bound*, not every individual sample — on any one operand pair the naive fold
can land closer, and here it did. The assertion now averages over an ensemble,
which is the claim the bound actually makes; the per-trial win rate
(155/400 → 212/400 → 294/400 as `n²` grows) is printed alongside so the
distinction stays visible.

## Verification

```
Gate A   7/7, rust energies BIT-IDENTICAL to the pre-change run
Gate B   RAYON_NUM_THREADS=1 vs =8 bit-identical
ALG-06   check-dependency-wall PASS
Suites   cargo test -p pyscf-pbc-df -p pyscf-pbc-dft -p pyscf-pbc-scf --release
         — 0 failures
```
