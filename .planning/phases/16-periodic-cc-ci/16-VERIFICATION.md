# Phase 16 — VERIFICATION. Written 2026-09-06.

**Status: PARTIAL.** **Nine of fourteen plans ship complete and measured**;
five did not start (four of them the plan set's own designated droppable
half). Every number below is reproducible from a
committed command, and every tolerance traces to
`measurements/README.md`. **No tolerance was loosened to make a test pass**;
where a gate was too tight, the measurement that proves it is recorded and the
gate is restated with the evidence — five times, listed in §5.

| plan | state | evidence |
|---|---|---|
| **16-01** MEASURE | **COMPLETE** | `measurements/README.md` + 7 scripts + outputs |
| **16-02** substrate | **COMPLETE** | `zworkspace_pool.rs`, `ktensor.rs`, 15 tests |
| **16-03** Davidson | **COMPLETE** | `davidson.rs`, 8 tests, cross-checked against upstream |
| **16-04** `kintermediates_rhf` | **COMPLETE** | oracle-green, `1.1e-8 … 2.3e-7` |
| **16-05** `KRCCSD` | **COMPLETE** | **`e_corr` `6.56e-9`**, 3 oracle + 5 oracle-free tests |
| **16-07** `KGCCSD` | **COMPLETE** | **`e_corr` `2.07e-9`**, 19 cycles; + the narrow molecular `gccsd` surface |
| **16-08** `KCCSD(T)` | **COMPLETE** | fast-vs-slow `8.36e-13` rel; spin-orbital `3.46e-11` from upstream |
| **16-13** `KCIS` | **COMPLETE** | Davidson roots `5.48e-10`; both solver paths |
| 16-06 `KUCCSD` | **SHIPPED** | `16-06-SUMMARY.md` |
| 16-09/10/11 EOM | **NOT STARTED** | §6 — the plan set's own "droppable half" |
| 16-12 `kuccsd_rdm` + Γ shim | **NOT STARTED** | §6 |
| **16-14** verification | **this file** | |

---

## 1. The correctness gates that were run

All on `diamond` `gth-szv`/`gth-pade` (`§9.2`), **FFTDF**, `[1,1,2]`,
`cell.mesh = [15,15,15]`, `cell.precision = 1e-8`, `conv_tol = 1e-9`,
`conv_tol_normt = 1e-7`, against vendored PySCF **2.12.1**.

```bash
PYSCF_ORACLE_VENV=1 cargo test --release -p pyscf-pbc-cc \
  --test oracle_phase16 -- --ignored --nocapture
cargo test --release -p pyscf-pbc-cc --test kccsd_rhf -- --ignored --nocapture
```

| # | quantity | measured | gate | verdict |
|---|---|---|---|---|
| G1 | `KRCCSD e_corr` vs upstream | **`6.560e-9`** | `1e-7` | **MET** |
| G1 | `init_amps emp2` vs upstream | `3.494e-9` | `1e-7` | **MET** |
| — | `energy()` on synthetic amplitudes | `3.22e-9` | `1e-6` | **MET** |
| — | `_ERIS` blocks (7) vs upstream | `1.21e-8 … 2.34e-7` | `1e-6` | **MET** |
| — | `cc_*` intermediates (9) vs upstream | `3.51e-9 … 2.28e-7` | `1e-6` | **MET** |
| — | `update_amps` `t1new` / `t2new` | `1.84e-8` / `7.01e-8` | `1e-6` | **MET** |
| G4 | **(T) fast vs slow, relative** | **`8.363e-13`** | `1e-12` | **MET** |
| — | (T) fast/slow vs upstream | `3.286e-10` | `1e-6` | **MET** |
| G3 | **`KGCCSD e_corr`** vs upstream | **`2.066e-9`** | `1e-8` | **MET** |
| — | `KGCCSD` seven `<pq\|\|rs>` blocks | `2.42e-8 … 4.68e-7` | `1e-6` | **MET** |
| — | `KGCCSD` `energy()`, `update_amps` | `1.55e-9`, `1.10e-8`/`7.28e-8` | `1e-6` | **MET** |
| G5 | **spin-orbital (T)** vs upstream | **`3.459e-11`** | `1e-9` | **MET** |
| G5 | spin-orbital vs RHF (T), this port | **`6.9e-12`** | `1e-9` | **MET** (upstream's own gap: `2.86e-10`) |
| G7 | **`KCIS` Davidson roots** vs upstream | **`5.48e-10`** / `2.37e-9` | `1e-5` | **MET** |
| — | `KCIS` dense roots vs upstream's dense | `1.84e-9` / `2.37e-9` | `1e-6` | **MET** |
| — | `KCIS` diagonal vs upstream | `1.43e-9` / `3.06e-9` | `1e-6` | **MET** |
| G8 | `_ERIS` incore vs spilled | **bit-identical** | bit-identical | **MET** |
| G10 | `symm_map` vs all-triples | `3.72e-8 … 7.93e-7` | `1e-6` | **MET** |
| G11 | determinism of `t1`, `t2`, `e_corr` | **bit-identical** | bit-identical | **MET** |
| G2 | `KRCCSD e_corr`, GDF / RSDF route | — | `1e-7` | **NOT RUN** (§6) |
| G3 | `KUCCSD e_corr` | `6.05e-10` | `1e-7` | PASS (`oracle_kuccsd.rs::kuccsd_e_corr_matches_upstream`). The gate is `1e-7`, not the plan's `1e-8`: it is the same number `measurements/README.md §1` sets for G1 (`KRCCSD e_corr` vs upstream, FFTDF), and the measured residual sits two orders inside it. |
| G6 | EOM roots | — | `1e-5` | **NOT RUN** (§6) |
| G9 | supercell equivalence | `1.426e-11` | `1e-7` | PASS (`kccsd_rhf.rs::krccsd_matches_the_supercell_at_gamma`, `--ignored --release`) |

---

## 2. The oracle-free gates — the phase's real proof

These consume no Python and survive an oracle drift.

| gate | measured | note |
|---|---|---|
| `_ERIS` incore vs spilled, all 7 blocks | **bit-identical**, 229 376 bytes each way | and the test ASSERTS WHICH TIER each side used, so a fixture that silently stayed incore fails |
| `symm_map` loop vs all-triples loop | `7.93e-7` worst (`vovv`); **`vvvv` exactly `0e0`** | `vvvv` is built by `ao2mo_7d` in BOTH paths — the control that says the difference is the FFT, not the transposition |
| `t1`, `t2`, `e_corr` over two runs | **bit-identical**, 18 cycles both times | §9.3; `e_corr` alone would pass a non-deterministic `t2` |
| `init_amps emp2` vs Phase 15's `KMP2` | `2.166e-10` | cross-PHASE, and it needs `keep_exxdiv = true` — see §5.4 |
| arena byte accounting vs the derived count | **exact** (`229 376 == Σ nkpts³·dims·16`) | D-PBC-29 clause 1 |
| **`KCIS` Davidson-vs-dense SPREAD reproduced** | `1.289e-9` at `kshift = 0` | upstream's own two paths differ by `2.51e-3` there — its Davidson finds a different state, and this port finds the SAME different state |
| `KCIS` Davidson vs this port's own dense | `2.51e-3` / `9.99e-16` | matching upstream's `2.51e-3` / `1.05e-15` |
| spin-orbital (T) vs RHF (T), this port | `6.9e-12` | upstream's own two routes: `2.86e-10` — this port's agree **40× tighter** |
| `<pq\|\|rs>` antisymmetry, diagonal, and non-identity with the chemist block | exact | `crates/pyscf-ccsd/tests/gccsd.rs`, 3 tests |
| `davidson_nosym1` vs a dense `faer` solve | `4.96e-12 … 1.1e-13` | 16-03, `n = 40`/`80`, `nroots` 1/3/5 |
| `davidson_nosym1` vs `eigh_gen`, Hermitian | `<1e-11` | the sign/conjugation sanity check |
| `einsum` vs an explicit loop, and its NON-conjugation | exact | `Σ x·x = -14` for imaginary `x`, not `+14` |
| `symm_map` orbit completeness / ≤4 / partition | exact integers | `nkpts` 1, 2, 4, 8 |
| `ZWorkspacePool` byte count, HARD refusal, free list, spill round-trip | exact | no tolerance anywhere in 16-02 |

---

## 3. D-PBC-29's four claims, re-measured against the shipped code (16-14 Task 4)

1. **The complex arena's accounting is exact and the HARD refusal fires at the
   boundary.** `ZWorkspacePool::shape_bytes(&[2,3,4]) == 384` (the f64 sibling
   reports `192` on the same shape); a 383-byte budget refuses with
   `MemoryLimitExceeded { requested: 384, limit: 383 }` and allocates nothing;
   a 384-byte budget succeeds. On a real `_ERIS` the arena charges exactly the
   derived `229 376` bytes. **CONFIRMED.**
2. **Host rayon loops vs `zgemm_dense`** — **NOT RE-MEASURED.** Every
   contraction in this phase goes through `ZArr::einsum`, a host loop with
   `oracle_zsum` accumulators, per clause 2; no `zgemm_dense` call was written,
   so there is nothing to compare against without writing the alternative
   first. The clause stands on the standing measurement
   (`zgemm-dense-loses-to-host-rayon`) and the re-measurement is carried over.
3. **`symm_map`** — **AMENDED, and the amendment is 16-01's.** Measured at
   **2.10× wall clock** (`59.487 s` vs `125.029 s` on diamond `gth-szv` 2×2×2)
   against the review's derived `~4×`, with a count ratio of 2.91 (176
   representatives for 512 triples). Two reasons: the orbit collapses at fixed
   points, and `vvvv` is built by `ao2mo_7d` in BOTH paths so it saves nothing.
   The clause stands — `symm_map` is used from the first version — at `~2×`.
   `16-REVIEW.md §3`'s own "if the measured ratio is materially below 4, say
   so" is discharged.
4. **Storage tiers from an exact per-tensor byte count** — **CONFIRMED**, and
   at least one green test crosses a tier boundary
   (`eris_incore_and_spilled_are_bit_identical`, which asserts
   `Tier::InMemory` on one side and `Tier::Spilled` on the other). Upstream's
   `_mem_usage` over-estimate is measured at **9.143×** (`gth-szv`) /
   **6.058×** (`gth-dzvp`), confirming the review's derived 9.1×/6.2×.

---

## 4. Determinism (§9.3)

`t1`, `t2` and `e_corr` are bit-identical over two in-process runs, and the
property holds **by construction** rather than by luck: every `ZArr::einsum`
output element is one `oracle_zsum` over a fixed-length product buffer whose
pairwise recursion tree depends only on that length, output elements never mix,
and the DIIS residual norm goes through `oracle_dot`. 16-03's Davidson is
likewise bit-identical over repeated runs.

**The cross-thread half (`RAYON_NUM_THREADS` 1 vs 8) was not run as a separate
invocation** and is a carry-over; the in-process half is what a regression in
accumulation order breaks first, and the structural argument above covers the
rest. For scale, 16-01 measured UPSTREAM's own `OMP_NUM_THREADS` 1/4/8 spread
at `6.94e-17` — upstream does not have this property and this port does.

---

## 5. Five gates were tighter than the thing they gate. None was loosened to pass a test.

Each was corrected by a measurement, with the number recorded:

1. `ROADMAP`'s **`1e-14`** for `KRCCSD e_corr` — eight orders tighter than
   upstream's own 6-decimal assertion. Struck through, replaced by G1 `1e-7`.
2. `PBC-MASTER-PLAN §7`'s **`1e-8`** for the same number — two orders tighter,
   and it named a fixture (`He 1×1×2`) that cannot host the calculation at all
   (`nvir = 0` in `gth-szv`). Struck through.
3. `16-05-PLAN.md` test 5's **bit-identity** for the `symm_map` loop —
   upstream's own two paths differ by `1.32e-7`. Replaced by G10 `1e-6`; the
   `vvvv` control (`0e0`) proves the difference is the FFT.
4. `16-05-PLAN.md` test 3's **`1e-12`** for `init_amps emp2 == KMP2` — the two
   are different quantities unless `keep_exxdiv = true` (upstream's own log
   line says "with fock eigenvalue shift"), and even then they agree only to
   the SCF's `conv_tol = 1e-10`, measured `2.17e-10`. Gated at `1e-9`.
5. **G4's `1e-13`**, written into this phase's own measurements README, is
   below upstream's own fast-vs-slow agreement of `2.95e-13`. Corrected to
   `1e-12`; measured `8.36e-13`.

Three upstream anchor sets are excluded from every gate, for cause:
`test_krccsd.py::test_frozen_n3` FAILS on the vendored tree (`ehf_bench` off by
`1.56e-6` at a 6-decimal assertion), and every `cu_metallic` anchor sits in a
test upstream itself disabled with `@unittest.skip('Results not match')`
(`test_krccsd.py:403`); run anyway, all eight diverge by `2e-2 … 7e-2`.

---

## 6. What did NOT ship, with the reason and the unblocking work

**Nothing here was guessed at, stubbed or silently dropped.** The rule this
phase inherited from 17-09 is: defer explicitly, never guess.

### 6.1 Plans not started

| plan | what it is | why not | unblocked by |
|---|---|---|---|
| ~~**16-06** `KUCCSD`~~ | `kintermediates_uhf` (1225 l) + `kccsd_uhf` (1116 l) | **CLOSED.** `kccsd_uhf.py` and `kintermediates_uhf.py:26-588` ship; `:590-1225` is EOM-KUCCSD's and stays with 16-11. Found and fixed THREE latent defects in the already-shipped complex arena — see `16-06-SUMMARY.md`. | — |
| **16-09/10/11** EOM-KCCSD GHF/RHF/UHF | 2011 + 1874 + 1275 l | not reached. `16-CONTEXT §4` designates these the phase's **droppable half** — excited-state properties nothing in Phases 17-20 needs for correctness — and BOTH their hard prerequisites now ship and are tested: 16-03's Davidson and 16-07's `KGCCSD`, which the RHF and UHF EOM modules inherit from | — |
| **16-12** `kuccsd_rdm` + the Γ shim | 157 + 157 l | **`kuccsd_rdm` SHIPS** — `6.94e-18` against upstream on fixed synthetic amplitudes, exactly Hermitian, with the converged comparison gated from the measured `2.41e-9` amplitude spread rather than from the RDM. The Γ shim is still blocked and NOT by 16-06: it needs the molecular complex-capable `rccsd.RCCSD` this port does not have (`16-CONTEXT §1.2`), which is a phase, not a task. | molecular `RCCSD` |

**`scf.kghf.KGHF.CCSD` (`kccsd.py:805`), the surface Phase 19 reads, still does
NOT exist** — 16-07 ships the KGCCSD arithmetic and gates it on upstream's mean
field, but not the driver that would build the Fock from this port's own
`Kghf`. See `16-07-SUMMARY.md`.

**17-09 is NOT unblocked.** `PBC-MASTER-PLAN §8.9` defers its CC half "if Phase
16 has shipped `KRCCSD`", and `KRCCSD` HAS shipped and is oracle-green — but
17-09's target is `kccsd_rhf_ksymm.py` (806 l) + `kintermediates_rhf_ksymm.py`
(265 l), the k-SYMMETRY adapters, which need `KPoints` IBZ machinery from
Phase 17 as well. The Phase-16 half of its dependency is now satisfied; state
that plainly rather than claim the plan is unblocked.

### 6.2 Gates and tests not run within the plans that DID ship

| item | plan | why | the number it would be held to |
|---|---|---|---|
| **G2 — the GDF / RSDF / MDF routes** | 16-05 | only FFTDF was run end to end. The route split is `9.22e-4 Ha` (§4 of the README), so this is not a formality | `1e-7` per route |
| ~~**G9 — supercell equivalence**~~ | 16-05 test 1 | **CLOSED.** `pyscf_pbc_gto::super_cell` already existed (it is in `pyscf-pbc-gto`, not `pyscf-pbc-tools` — that is what the original survey missed). `e_corr/nkpts` agrees to `1.426e-11`; the two mean fields to `2.057e-10`. | `1e-7` (upstream's own two routes differ by `2.97e-8`) |
| **Γ reduction vs molecular RCCSD** | 16-05 test 2, 16-04 test 1 | needs 16-12's Γ-point `pbc/cc/ccsd.py` shim | `1e-12` |
| **cross-thread determinism** | 16-05 test 7 | in-process half only; see §4 | bit-identical |
| ~~**`KUCCSD`**~~ | 16-06 | **CLOSED** at `6.05e-10`. `kuccsd_rdm` (16-12) is now unblocked. | `1e-7` |
| **the (T) peak-memory bound** | 16-08 test 6 | needs the `t3`-class allocations routed through `ZWorkspacePool`; the blocking IS ported and its invariance measured at `2.17e-19` | one block's `nocc³·nvir³` |
| **`zgemm_dense` re-measurement** | 16-14 Task 4.2 | nothing to compare against without writing the alternative | — |
| **`cell.precision` ladder** | 16-01 Task 2 | one `[2,2,2]` SCF at the default `[47,47,47]` mesh exceeds the session budget; 17-01 Gate B's "the floor is integral-screening-limited" is carried into this phase UNVERIFIED | — |
| **`gth-dzvp` at any mesh** | 16-01 Task 5 | the byte counts in `README §6` are DERIVED, not run; `gth-dzvp` 3×3×3's 90 GiB of ERI blocks cannot be built on this machine | — |

### 6.3 Absent UPSTREAM, not deferred by this port

Recorded so a later reader does not look for them (`16-CONTEXT §1.5`, `§1.6`,
16-01 Task 4's measurement):

* **EOM-EE triplet and spin-flip (RHF)** — `EOMEETriplet`
  (`eom_kccsd_rhf.py:1483`) and `EOMEESpinFlip` (`:1489`) are shells; both
  raise through `eeccsd` at `:835`, and `EOMEE.vector_size` (`:1417`) raises.
* **EOM-EE (UHF)** — `eom_kccsd_uhf` has **no `EOMEE` class at all**
  (`AttributeError`), and `_IMDS.make_ee` (`:1120`) raises.
* **`kuccsd_rdm`'s `with_frozen` path** — `kuccsd_rdm.py:136` raises and
  everything after it in that block is unreachable upstream.
* **`pbc/ci/cisd.py`** — a Γ-only shim (`:24`, `:47` refuse `kpt != 0`) over
  molecular RCISD/UCISD/GCISD, and this port has **no molecular CI crate**.
  Porting it means porting molecular CISD first, which is a phase, not a task.

`ROADMAP.md`'s "EOM-KCCSD IP/EA/**EE** (RHF/UHF/GHF)" claim is wrong on both
UHF-EE and RHF-EE-triplet/spin-flip, independently of what this port shipped.

---

## 7. The one finding this phase produced that belongs to another phase

**This port's `KRHF` and upstream's differ by `1.348e-5 Ha` on diamond
`gth-szv` `[1,1,2]` when `cell.mesh` is PINNED at `[15,15,15]`** — while Phase
15 measured the same two agreeing to **`4.772e-11` on the same cell at the
DEFAULT mesh**. Upstream's `rcut` (21.319) and `nimgs` ([6,6,6]) are identical
at both meshes, so it is not the lattice sums; it is the FFT-grid-evaluated
part of the mean field at a coarse grid.

Phase 16 does not own it and did not cause it. It is recorded in
`measurements/README.md §10` and carried to whichever phase owns FFTDF's
coarse-mesh behaviour. **Its consequence for this phase is procedural and
load-bearing:** every CC oracle gate here drives `KEris::from_parts` with
upstream's own `fock` / `mo_energy` / `mo_coeff`, so the numbers in §1 measure
the CC code and not two different SCFs — the discipline `15-VERIFICATION` used
when it drove `Lov` from "upstream's own padded MOs" and got `2e-15`. The
mean-field residual is printed beside every result and is never absorbed into a
tolerance.

---

## 7.1 One test failure worth recording — and the diagnosis was WRONG

`pyscf-pbc-cc::ktensor::block_roundtrip_is_bit_identical_on_both_tiers` failed
once, in `KTensor::zeros(.., allow_spill = true)`, and passed on rerun. This
section originally attributed it to `/tmp` — a 16 GiB tmpfs — having filled to
80% with a scratch `CARGO_TARGET_DIR`, and called the failure environmental.

**That was wrong.** 16-06 reproduced it deterministically by adding a fifth
test to the same file. The spill file is named
`pyscf_kcc_zspill_<pid>_<buffer id>.h5` and the buffer id restarts at 0 for
every `ZWorkspacePool`, so two pools in one process — two `#[test]`s on two
threads — both ask for `..._0.h5` and HDF5 refuses the second with *"unable to
truncate a file which is already open"*. A full `/tmp` produces a different
message (`No space left on device`), which the original note did not check.

Fixed with a process-global counter in `ZSpillHandle::create`. The lesson is the
one `16-CONTEXT §3.4` states about the whole phase: an intermittent failure with
a plausible environmental story is still an unread error message.

The part that WAS right: the arena reports the failure as
`BackendError::ProbeFailed` naming the path, not a silent fallback — the
behaviour D-01 asks for. That is why the message was available to re-read.

---

## 8. Workspace state after this phase

```
cargo test --release -p pyscf-runtime -p pyscf-pbc-lib -p pyscf-algebra \
           -p pyscf-ccsd -p pyscf-pbc-cc -p pyscf-pbc-ci
    62 suites GREEN, 0 failed

cargo test --release -p pyscf-pbc-cc --test kccsd_rhf -- --ignored
    5 passed (the oracle-free gates)

PYSCF_ORACLE_VENV=1 cargo test --release -p pyscf-pbc-cc \
    --test oracle_phase16 -- --ignored        5 passed
PYSCF_ORACLE_VENV=1 cargo test --release -p pyscf-pbc-ci \
    --test oracle_kcis   -- --ignored         1 passed

cargo clippy --release -p pyscf-pbc-cc -p pyscf-pbc-ci -p pyscf-runtime \
             -p pyscf-algebra -p pyscf-ccsd --no-deps --all-targets
    no lints in this workspace's crates

check-orphan-modules   PASS — 364 source files, all reachable
check-dependency-wall  PASS — cubecl-* containment intact (ALG-06)
```

`pyscf-ccsd`'s suite is green and `crates/pyscf-runtime/src/workspace_pool.rs`
has an EMPTY diff — 16-02's "do not edit the f64 pool" clause verified by the
diff, not by assertion.

---

## 9. Reproducing every number in this file

```bash
# 16-01's measurements (Python, vendored PySCF 2.12.1)
PYTHONPATH=$PWD .venv/bin/python -u \
  .planning/phases/16-periodic-cc-ci/measurements/m1_anchors.py
#   … m2_spread.py m3_df_routes.py m4_eom_and_t.py m5_tiers.py
#   … m6_symm_map.py m7_davidson_ref.py

# the always-on Rust tests
cargo test -p pyscf-runtime -p pyscf-pbc-lib -p pyscf-algebra -p pyscf-pbc-cc

# the SCF-converging oracle-free gates
cargo test --release -p pyscf-pbc-cc --test kccsd_rhf -- --ignored --nocapture

# the opt-in PySCF oracle
PYSCF_ORACLE_VENV=1 cargo test --release -p pyscf-pbc-cc \
  --test oracle_phase16 -- --ignored --nocapture
```
