# KRKS-OPTIMISATION-PLAN — execution summary

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
