# Phase 16 — measured floor and the RESTATED gate (plan 16-01)

**Measured 2026-09-06** against the vendored PySCF **2.12.1**
(`pyscf/__init__.py:38`, asserted at the top of every script here).
**No Rust source file was created or edited by plan 16-01**; `git diff --stat
crates/` for that plan is empty.

Every number below is reproducible from a committed script in this directory:

| script | plan task | output |
|---|---|---|
| `m1_anchors.py` | Task 1 — upstream's own committed anchors | `m1_anchors.out`, `m1_anchors_cu.out` |
| `m2_spread.py` | Task 2 — run-to-run / thread / conv_tol / precision spread | `m2_spread.out` |
| `m3_df_routes.py` | Task 3 — the DF-route split | `m3_df_routes.out`, `m3_df_routes_he.out` |
| `m4_eom_and_t.py` | Task 4 — EOM roots, (T), the EE refusals | `m4_eom_and_t.out` |
| `m5_tiers.py` | Task 5 — the storage-tier crossover | `m5_tiers.out` |
| `m6_symm_map.py` | Task 6 — pricing the `symm_map` saving | `m6_symm_map.out` |
| `m7_davidson_ref.py` | 16-03 cross-check — upstream `davidson_nosym1` on the Rust fixtures | `m7_davidson_ref.out` |

**Fixture pin.** `diamond` `gth-szv`/`gth-pade` (`§9.2`) at
`cell.precision = 1e-8` (default) has a DEFAULT mesh of **`[47,47,47]`**, at
which one `KRHF` at `[1,1,2]` alone costs **79 s** and a 2×2×2 `conv_tol`
ladder is hours. Every spread/route/tier measurement therefore PINS
`cell.mesh = [15,15,15]` and says so beside the number. A spread measurement
does not need the converged-basis energy; it needs two runs of the same thing.
The `precision` leg is the exception, since varying the mesh is what it
measures.

---

## 1. THE RESTATED GATE

`ROADMAP.md` said `KRCCSD e_corr` to **1e-14**. `PBC-MASTER-PLAN §7` said
**1e-8 on He 1×1×2**. Six orders apart, about the same number, neither
measured — the fourth instance of this defect in the project (Phases 14, 15,
17). **Both are struck through, not deleted, in all four documents.**

| # | quantity | fixture / route | **GATE** | measured basis |
|---|---|---|---|---|
| G1 | `KRCCSD e_corr` vs upstream | diamond `gth-szv`, **FFTDF**, mesh pinned on both sides, `conv_tol ≤ 1e-9` | **1e-7** | §2, §3: upstream's own run-to-run spread is `1.9e-16`, its `conv_tol` plateau `6.3e-10`, but its FFTDF-vs-MDF *route* spread on the same cell is `2.6e-7` |
| G2 | `KRCCSD e_corr` vs upstream | diamond `gth-szv`, **GDF** or **RSDF** | **1e-7**, stated separately | §3 — GDF and RSDF agree with each other to `6.8e-9`; both are `9.2e-4` from FFTDF/MDF |
| G3 | `KUCCSD`/`KGCCSD` `e_corr` vs this port's `KRCCSD`, closed shell | diamond `gth-szv` `[1,1,2]` | **1e-8** | §4 — upstream's own `KGCCSD` and `KRCCSD` differ by `4.95e-9` on this fixture |
| G4 | `KCCSD(T)` fast vs slow, same amplitudes | any | **1e-13 relative** | §4 — upstream's two implementations agree to `3.27e-16` absolute / `2.95e-13` relative |
| G5 | spin-orbital (T) vs RHF (T), closed shell | diamond `gth-szv` `[1,1,2]` | **1e-9** | §4 — upstream's own two routes differ by `2.86e-10` |
| G6 | EOM-IP/EA roots vs upstream | diamond `gth-szv` | **1e-5** | §5 — upstream's own Davidson-`conv_tol` spread reaches `4.85e-7` and its `nroots` spread `5.11e-7`; its own suite asserts **3 decimals** |
| G7 | `KCIS` roots vs upstream | diamond `gth-szv` | **1e-5** | same solver, same Davidson floor as G6 |
| G8 | incore vs spilled ERI blocks, same process | any | **bit-identical** | §6 — same code, same inputs, no convergence noise; upstream gates its analogue at 12 decimals (`test_krccsd.py:250-256`) |
| G9 | supercell equivalence `e_corr(k-mesh)` vs `e_corr(supercell)/nk` | diamond `gth-szv` `[1,1,2]` | **1e-7** | §2 — upstream's own two routes differ by `2.97e-8` |
| G10 | `symm_map` loop vs all-triples loop | diamond `gth-szv` `[2,2,2]` | **1e-6**, **NOT bit-identical** | §7 — upstream's own two paths differ by up to `1.32e-7` |
| G11 | determinism at `RAYON_NUM_THREADS` 1 and 8 | every shipped path | **bit-identical** on `t1`, `t2` AND `e_corr` | §9.3; upstream's own thread spread is `6.9e-17`, i.e. it does NOT have this property and this port does |

**Every energy gate names its DF route, its fixture, its mesh and its
`cell.precision`.** A "matches upstream" number that does not is untestable —
§3 shows the FFTDF/GDF split alone is `9.2e-4 Ha` on diamond, four orders
above G1.

---

## 2. Upstream's own committed anchors — and the three that no longer reproduce

`m1_anchors.out`. Each row: the constant upstream's source pins, the value the
vendored 2.12.1 tree actually produces, and the decimal place
`assertAlmostEqual` asserts.

| anchor | source | pinned | produced | residual | asserted | verdict |
|---|---|---|---|---|---|---|
| `hf_311` | `test_krccsd.py:177/:179` | `-0.9268762991822949` | `-0.9268762991429551` | `3.93e-11` | 8 dp | REPRODUCES |
| `cc_311` | `:178/:180` | `-0.04270217758641424` | `-0.04270218735912376` | `9.77e-09` | 6 dp | REPRODUCES |
| `ehf_bench` | `:210/:225` | `-8.648503065380389` | `-8.648501503147841` | **`1.56e-06`** | 6 dp | **DOES NOT REPRODUCE** |
| `ecc_bench` | `:211/:226` | `-0.100045112503651` | `-0.10004455671597806` | **`5.56e-07`** | 6 dp | **DOES NOT REPRODUCE** |
| `ehf_bench` supercell/2 | `:229` | same | `-8.648501501162452` | `1.56e-06` | 5 dp | REPRODUCES |
| `ecc_bench` supercell/2 | `:232` | same | `-0.10004473676128034` | `3.76e-07` | 6 dp | REPRODUCES |
| `ercc/prod(nk)` | `:478` | `-0.15632445245405927` | `-0.1563244457272586` | `6.73e-09` | 4 dp | REPRODUCES |
| `ercc_t/prod(nk)` | `:481` | `-0.00114619248449` | `-0.0011461924913841962` | `6.89e-12` | 5 dp | REPRODUCES |

**`test_krccsd.py::KnownValues::test_frozen_n3` FAILS on the vendored tree**,
confirmed by running upstream's own suite:

```
$ PYTHONPATH=$PWD .venv/bin/python -m pytest pyscf/pbc/cc/test/test_krccsd.py -k frozen_n3
AssertionError: -8.64850150314784 != -8.648503065380389 within 6 places
                (1.5622325495456835e-06 difference)
1 failed
```

**This is the fourth instance of the standing caveat `15-VERIFICATION.md §7`
records** ("never gate against a constant embedded in upstream's source"),
after `kmp2.py:820`'s `2.1e-10` and `kmp2_stagger.py:385/390/395`'s
`2.8e-7…3.5e-7`. No Phase-16 gate cites `ecc_bench` or `ehf_bench`.

**The `cu_metallic` anchors (`:338` `ecc2_bench`, `:356` `ecc3_bench`,
`:359-366` the IP/EA roots) come from tests upstream has ITSELF DISABLED**:
`test_cu_metallic_high_cost` carries `@unittest.skip('Results not match')` at
`test_krccsd.py:403`. Run anyway, with upstream's exact setup
(`scaled_center=[0,0,0]`, `wrap_around=True`, `conv_tol_grad=1e-6`), every one
of the eight diverges (`m1_anchors_cu.out`): `ehf_bench_cu` by `3.25e-2`,
`ecc2_bench` by `4.05e-2`, `ecc3_bench` by `4.72e-2`, the IP roots by
`2.03e-2…`, the EA roots by `7.06e-2…`. **No Phase-16 gate cites any of
them**, and 16-14 records the reason rather than leaving a reader to wonder why
a table row is missing.

### Supercell equivalence, oracle-free (`test_krccsd.py:478-482`)

```
ekcc            -0.1563244160359924
ercc/prod(nk)   -0.1563244457272586    |diff| 2.969126619567497e-08
ekcc_t          -0.0011461929877441429
ercc_t/prod(nk) -0.0011461924913841962 |diff| 4.9635994663413396e-10
```

Upstream asserts these at 5 and 6 decimals. **This is G9's basis** and it is
the single most valuable oracle-free gate in the phase: it alone catches a
wrong `kconserv` argument order, a transposed `t2` index order and a misplaced
`1/nkpts`.

---

## 3. Run-to-run, thread, `conv_tol` and `precision` spread — the true floor

`m2_spread.out`. diamond `gth-szv`, `nk = [2,2,2]`, `mesh = [15,15,15]`,
`cell.precision = 1e-8`, `KRHF conv_tol = 1e-10`, `KRCCSD conv_tol = 1e-9`.

* **Run-to-run (5 runs, same threads): `max-min = 1.943e-16`**, `std =
  6.48e-17`, mean `-0.11866306760186804`. The DIIS path is deterministic
  in-process, so the "two independently converged runs take different DIIS
  paths" worry does NOT materialise here — the floor is not set by this.
* **`OMP_NUM_THREADS` 1 / 4 / 8: `max-min = 6.939e-17`.**
  `-0.11866306760186801` / `-0.11866306760186797` / `-0.11866306760186804`.
  Upstream is thread-invariant to the last two bits but NOT bit-identical;
  this port's `oracle_*` reductions are (G11).
* **`conv_tol` × `conv_tol_normt` ladder** — the plateau:

| `conv_tol` | `conv_tol_normt` | `e_corr` | Δ vs previous rung | wall |
|---|---|---|---|---|
| 1e-07 | 1e-05 | `-0.11866301486029165` | — | 13.9 s |
| 1e-08 | 1e-06 | `-0.11866306697155163` | `5.21e-08` | 15.7 s |
| **1e-09** | **1e-07** | `-0.11866306760186805` | `6.30e-10` | 16.1 s |
| 1e-10 | 1e-08 | `-0.11866306840351801` | `8.02e-10` | 188.1 s |
| 1e-11 | 1e-09 | `-0.11866306862822656` | `2.25e-10` | 745.9 s |

  `e_corr` plateaus at `conv_tol = 1e-9` to within `~8e-10`; tightening further
  costs **46×** the wall clock and moves the energy by less than `1e-9`. The
  `t1`/`t2` fingerprints plateau at the same rung. **`conv_tol = 1e-9`,
  `conv_tol_normt = 1e-7` is the setting every Phase-16 gate is run at.**

---

## 4. The DF-route split — why the gate is stated PER ROUTE

`m3_df_routes.out`. `kccsd_rhf.py:37` imports `GDF, RSGDF` and `:824-832`
branches the whole `_ERIS` build on the mean field's DF class, exactly as
`kmp2.py:69` does.

**diamond `gth-szv` `[1,1,2]`, mesh `[15,15,15]`:**

| route | `e_hf` | `e_corr` | wall |
|---|---|---|---|
| FFTDF | `-8.651997841505` | `-0.1552984784873315` | 13.8 s |
| GDF | `-8.655276573450518` | `-0.1543761305810659` | 22.8 s |
| MDF | `-8.651923985178758` | `-0.15529821981195518` | 67.9 s |
| RSDF | `-8.655276596707562` | `-0.15437612376295026` | 56.6 s |

| pair | `|Δe_corr|` | `|Δe_hf|` |
|---|---|---|
| FFTDF vs GDF | **`9.223479e-04`** | `3.278732e-03` |
| FFTDF vs MDF | `2.586754e-07` | `7.385633e-05` |
| FFTDF vs RSDF | `9.223547e-04` | `3.278755e-03` |
| GDF vs MDF | `9.220892e-04` | `3.352588e-03` |
| **GDF vs RSDF** | **`6.818116e-09`** | `2.325704e-08` |
| MDF vs RSDF | `9.220960e-04` | `3.352612e-03` |

**Two routes, not four.** The plane-wave pair (FFTDF, MDF) agrees to
`2.6e-7`; the Gaussian pair (GDF, RSDF) agrees to `6.8e-9`; the two PAIRS are
`9.2e-4` apart. That `9.2e-4` is upstream disagreeing with itself, and it is
**three orders larger** than the `4.5e-6` the standing memory
`rsdf-gdf-disagree-on-diamond` records at the SCF level — because a `3.3e-3`
mean-field difference propagates. **A single "matches upstream" gate would be
untestable**; G1 and G2 are stated separately, and 16-14 reports this port's
own inter-route ratio the way `14-VERIFICATION` Gate 3 does.

`he_fcc` `gth-szv` **cannot host this measurement**: one He atom in `gth-szv`
is a single AO, so `nocc = 1`, `nvir = 0`, and `KRCCSD` dies with
`IndexError: index 2 is out of bounds for axis 0 with size 2`. This is the same
finding Phase 15 recorded (`STATE.md`: "`he_fcc` `gth-szv` cannot host the
`Lov`/MO-first oracles … `nvir = 0`"); the all-electron control uses **`6-31g`**
(`m3_df_routes_he.out`), as Phase 15's did.

---

## 5. (T), the spin-orbital cross-check, and the EOM floor

`m4_eom_and_t.out`, diamond `gth-szv` `[1,1,2]`, mesh `[15,15,15]`.

### (T) — the tight gate (G4)

```
e_corr                                  -0.1552984784873314
fast (kccsd_t_rhf, C kernel at :236)    -0.001111305625626701   0.17 s
slow (kccsd_t_rhf_slow)                 -0.0011113056256270284  0.10 s
|fast - slow| = 3.2742905609062234e-16   relative 2.946e-13
```

Same input, same formula, two implementations — no convergence noise — so this
is where a gate CAN be tight, and it needs no oracle beyond upstream's own two
implementations. **G4 = 1e-13 relative.** `kccsd_t_rhf_slow.py` is the file
`PBC-MASTER-PLAN §8.8`'s table omits entirely and it is the only oracle-free
reference for the blocked path (`16-CONTEXT §1.7`).

### spin-orbital vs RHF (G3, G5)

```
KGCCSD e_corr            -0.1552984834364875
KRCCSD e_corr            -0.1552984784873314    |Δ| 4.949e-09
spin-orbital (T)         -0.0011113059117248167
RHF (T)                  -0.001111305625626701  |Δ| 2.861e-10
```

**`16-07`'s plan says "`KGCCSD.e_corr` equals 16-05's `KRCCSD.e_corr` to
1e-10".** Upstream's own two routes differ by `4.95e-9` on this fixture, so
1e-10 would fail a correct implementation. **G3 = 1e-8.** Likewise
`16-08` test 2's "1e-11" for spin-orbital-vs-RHF (T) is below upstream's own
`2.86e-10`; **G5 = 1e-9.**

### EOM-IP/EA (G6)

`EOMIP`/`EOMEA` at `kshift = 0`, `koopmans = True`, `nroots = 3`:

```
IP  -0.8268859724113091  -0.7532755389443693  -0.7532027768695687
EA   1.0737187523628222   1.094274695221061    1.094371048889709
```

| perturbation | IP max\|Δ\| | EA max\|Δ\| |
|---|---|---|
| run-to-run | `0.000e+00` | `0.000e+00` |
| Davidson `conv_tol` 1e-6 vs 1e-7 | `2.143e-07` | `0.000e+00` |
| Davidson `conv_tol` 1e-8 vs 1e-7 | `4.419e-09` | **`4.804e-07`** |
| Davidson `conv_tol` 1e-9 vs 1e-7 | `1.389e-09` | `4.849e-07` |
| `nroots` 2 vs 3 | `1.921e-08` | `6.217e-15` |
| `nroots` 4 vs 3 | `3.351e-09` | **`5.108e-07`** |

**Upstream's 3-decimal assertion is necessity, not pessimism** for the EA
branch: its own roots move by `5e-7` when `nroots` or the Davidson threshold
changes. **G6 = 1e-5**, two orders inside upstream's own 3-decimal assertion
and two orders outside its own instability.

### The EE surface and its refusals

`EOMEESinglet` (`eom_kccsd_rhf.py:1425`) **runs**: `vector_size = 1088`,
roots `[0.2677977, 0.26870819]` on this fixture. Everything else refuses, and
16-10/16-11 quote these lines in their payloads:

| surface | upstream line | behaviour |
|---|---|---|
| `EOMEE.vector_size` | `eom_kccsd_rhf.py:1417` | `raise NotImplementedError` |
| `EOMEETriplet.kernel` | `:1483` → `eeccsd` at `:835` | `raise NotImplementedError` |
| `EOMEESpinFlip.kernel` | `:1489` → `eeccsd` at `:835` | `raise NotImplementedError` |
| `eom_kccsd_uhf.EOMEE` | — | **the class does not exist** (`AttributeError`) |
| `eom_kccsd_uhf._IMDS.make_ee` | `eom_kccsd_uhf.py:1120` | `raise NotImplementedError` |

**`ROADMAP.md`'s "EOM-KCCSD IP/EA/EE (RHF/UHF/GHF)" is wrong** and is
corrected: RHF EE is singlet-only, UHF EE does not exist
(`16-CONTEXT §1.5`).

---

## 6. The storage tiers (G8), and upstream's over-estimate

`m5_tiers.out`. Exact per-block byte counts at 16 B/element, `nkpts³ · block`:

| cell / basis | mesh | `oooo` | `oovv` | `vovv` | **`vvvv`** | 7 blocks | `_mem_usage` | over-estimate |
|---|---|---|---|---|---|---|---|---|
| diamond `gth-szv` (nocc 4, nvir 4) | 1×1×2 | 0.031 MiB | 0.031 | 0.031 | **0.031** | **0.219 MiB** | 2.000 MiB | **9.143×** |
| diamond `gth-szv` | 2×2×2 | 2.000 | 2.000 | 2.000 | **2.000** | **14.000 MiB** | 128.000 MiB | **9.143×** |
| diamond `gth-szv` | 3×3×3 | 76.9 | 76.9 | 76.9 | **76.9** | **538.207 MiB** | 4920.750 MiB | **9.143×** |
| diamond `gth-dzvp` (nocc 4, nvir 22) | 2×2×2 | 2.000 | 60.5 | 332.8 | **1830.1** | **2357.375 MiB** | 14280.500 MiB | **6.058×** |
| diamond `gth-dzvp` | 3×3×3 | 76.9 | 2296 | 12628 | **69456** | **90625.414 MiB** | 548990.394 MiB | **6.058×** |

`16-REVIEW.md §2.4` derived 9.1× and 6.2×; **measured 9.143× and 6.058×.**
Porting `kccsd_rhf.py:1100-1107` literally would import that factor into this
port's HARD `MemoryLimitExceeded` refusal — it would refuse jobs that fit.
**D-PBC-29 clause 4 stands: the tier comes from the exact per-tensor byte
count** (`KTensor::exact_bytes`, `16-02`).

**The tier flip, measured** (diamond `gth-szv` 2×2×2): incore iff
`mem_incore (134.218 MB) + mem_now (130.662 MB) < cc.max_memory`, i.e.
`max_memory > 264.880 MB` is incore and below it spills. The outcore branch has
its OWN floor at `:912` (`mem_now + nvir⁴·16·2/1e6 = 130.671 MB`), so the
window a test can set is **`(130.7, 264.9)` MB** — that window is the fixture
16-05 test 4 must use. At `[1,1,2]` the window is EMPTY (`137.695` vs
`137.7`), which is why the tier-crossing gate is stated at **2×2×2, not
1×1×2**.

---

## 7. `symm_map` — measured at **2.10×**, not the derived ~4×

`m6_symm_map.out`. **This is a correction to `16-REVIEW.md §3`**, which
derived "a genuine ~4×", and to D-PBC-29 clause 3, which cited it. §3 itself
required this: "if the measured ratio is materially below 4, it is reported".
It is.

### Orbit counts (diamond, `build_symm_map`)

| `nk` | `nkpts` | `nkpts³` | representatives | count ratio | `build_symm_map` |
|---|---|---|---|---|---|
| 1×1×1 | 1 | 1 | 1 | 1.0000 | 0.0001 s |
| 1×1×2 | 2 | 8 | 5 | 1.6000 | 0.0001 s |
| 1×2×2 | 4 | 64 | 28 | 2.2857 | 0.0003 s |
| 2×2×2 | 8 | 512 | 176 | 2.9091 | 0.0019 s |
| 3×3×3 | 27 | 19683 | 5292 | 3.7194 | 0.1276 s |
| 4×4×4 | 64 | 262144 | 67712 | 3.8715 | 1.5114 s |

The orbit collapses at fixed points, so the ratio approaches 4 only
asymptotically and is **2.91 at the 2×2×2 fixture the phase actually gates
on**. `build_symm_map` is `O(nkpts³)` and costs 1.5 s at `nkpts = 64` —
which is why `kccsd_rhf.py:512` builds it LAZILY and why 16-02 ports the
laziness.

### Wall clock (diamond `gth-szv` 2×2×2, `_ERIS` incore)

```
symmetry loop  (176 representatives of 512 triples)   59.487 s
all-triples loop (512 transforms)                    125.029 s
SPEED RATIO (all / symm) = 2.102x
```

**2.10×, against a derived ~4×.** The gap is the count ratio (2.91) minus the
cost of the transposes and of the `vvvv` block, which is built by
`ao2mo_7d` in BOTH paths (`kccsd_rhf.py:798`, the `self.vvvv[...]` line being
commented out) and therefore saves nothing. The saving is still the phase's
single largest speed item and still ships from the first version — but it is
**~2×, not ~4×**, and 16-14 re-measures it against the shipped Rust.

### G10 — the two paths are NOT bit-identical

```
max|oooo_symm - oooo_all| = 9.5554025721207554e-09
max|ooov_symm - ooov_all| = 3.3881521120141785e-08
max|oovv_symm - oovv_all| = 9.6038076179013399e-08
max|ovov_symm - ovov_all| = 5.1177143918550227e-08
max|voov_symm - voov_all| = 5.0749936402648878e-08
max|vovv_symm - vovv_all| = 1.3196230946222721e-07
max|vvvv_symm - vvvv_all| = 2.0816681711721685e-17
```

**`16-05-PLAN.md` test 5 asks for BIT-IDENTITY between the symmetry loop and
the all-triples loop. Upstream's own two paths differ by up to `1.32e-7`** at
`mesh = [15,15,15]` — because a symmetry-related k-quadruple's FFT transform
and its transposed sibling are not the same floating-point computation. The
`vvvv` row agrees to `2.08e-17` precisely because it does NOT go through the
symmetry loop. **G10 is therefore `1e-6`, not bit-identity**, and 16-05's test
is written to the measured number.

---

## 8. 16-03 cross-check — the Davidson port reproduces upstream, stalls included

`m7_davidson_ref.out`. Plan 16-03 requires no oracle and the shipped Rust tests
consume none; this was run because the port was seen to STALL, and the question
"is that a defect?" has a measurable answer.

On a dense random complex matrix (`n = 40`, SplitMix64 seed 12345, amplitude
0.05, diag `1+2i`, unit-vector guess, diagonal preconditioner,
`tol_residual = 1e-12`):

| | upstream `lib.davidson_nosym1` | `pyscf-algebra::davidson_nosym1` |
|---|---|---|
| eigenvalue | `1.0003286056665808` | `1.0003286056665817` |
| residual | `0.0001705039877410091` | `1.7050398774100944e-4` |
| cycles / `aop` calls | 4 | 4 |
| trajectory | `0.131 → 0.00171 → 0.000171 → 0.000171`, then "Linear dependency in trial subspace" | identical |

The stall is the METHOD, not the port. At coupling `≥ 0.2` **upstream itself**
converges to a spurious `2.565e-15` eigenvalue against a true lowest root of
`0.98032995`, and the Rust port does the same. The shipped fixtures therefore
stay at coupling 0.05, where both reach `|e − dense| = 4.96e-12`, and the
divergence is recorded here rather than papered over.

---

## 9. What plan 16-01 did NOT reach

Stated rather than extrapolated, the `15-VERIFICATION §8` discipline:

* **`diamond` `gth-dzvp`** at any mesh — the byte counts in §6 are DERIVED
  (`nkpts³ · block · 16`), not run. `gth-dzvp` `3×3×3`'s 90 GiB of ERI blocks
  cannot be built on this machine at all.
* **`si`, `lif`, `graphene`** — `§9.2` fixtures not measured here. `graphene`
  matters for `16-12` test 4 (the Madelung shift on `dimension = 2`) and
  `16-13` test 5 (the `kcis_rhf.py:637` refusal); both plans measure it
  themselves.
* **The `cell.precision` leg of Task 2** (`m2_spread.py precision`) was written
  but not run to completion — at the default mesh `[47,47,47]` one `[2,2,2]`
  SCF alone exceeds the session budget. `17-01` Gate B's finding (the floor is
  integral-screening-limited) is therefore carried over UNVERIFIED for Phase 16
  and is listed as a 16-14 carry-over.
* **`nkpts = 27` / `64` `_ERIS` wall clock** — only the orbit COUNTS were taken
  at those meshes, not the integral transform.
