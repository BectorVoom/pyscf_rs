# Phase 15 verification — Periodic AO2MO + KMP2

**Implementation:** 2026-09-05. **Verification closed:** 2026-09-05 (follow-up).
**Oracle:** vendored PySCF 2.12.1, asserted by every measurement script and by
`crates/pyscf-pbc-mp/tests/oracle_phase15.rs` itself.

**Verdict: CLOSED, with one gate NOT MET and owned by another phase.**
**UPDATED 2026-09-05 — that gate is now MET.** Phase 14 found and fixed the two
defects behind it (`14-VERIFICATION.md §11`): an `s2` k-pair packing error in
`gdf::cderi_store::sr_loop` and a nuclear-attraction mesh taken from `cell.mesh`
where upstream's `_CCNucBuilder` is mesh-independent. Row 5 below carries the
re-measured numbers; the text under it is kept as written because the ANALYSIS
was right — the defect was the mean field, not KMP2 — and because how it was
found is the point.

The restricted KMP2 implementation, both integral routes, the padding/frozen
surface, the KUMP2 refusal and staggered KMP2 all ship and are gated against
upstream. The nine-part opt-in oracle matrix is complete and green. The
staggered-energy oracle — recorded as `NOT MET` in the 2026-09-05 implementation
pass — is now run, and **it caught three real defects**, one of them worth a
factor of 26.6 in the reported energy. Completing the oracle matrix caught a
**fourth**, in the AFTDF AO2MO dispatch, worth `4.596e-1` against the route it
is supposed to equal (§3). The one criterion that remains `NOT MET` is the GDF
route's agreement with upstream, and it is **not a KMP2 defect**: this port's
GDF *mean field* on diamond `[1,1,2]` is **`1.523e+1 Ha`** from upstream's,
against FFTDF's `4.772e-11` on the same cell (§1a). That is Phase 14's, and it
is a **new** finding — `14-VERIFICATION`'s GDF gates are a one-AO `sto-3g` He
cell and diamond **at gamma only**, already recorded `PARTIAL`.

---

## §1 Per-criterion table

Every row carries the number that was measured. No row says "passes".

| # | criterion | command | observed | tolerance | verdict |
|---|---|---|---|---|---|
| 1 | upstream diamond FFTDF anchor reproduces its own source constant | `measurements/anchor.py` | `-0.20472143304034049`; `2.113e-10` from `kmp2.py:820`'s `-0.204721432828996` | — (measurement) | **MET** |
| 2 | KMP2 `e_corr`, diamond `gth-szv` `[1,1,2]`, FFTDF, `exxdiv=None` | `oracle_phase15::kmp2_energies` | Rust `-0.20472143298615961`, upstream `-0.20472143304034035`, residual **`5.418e-11`** | `2e-6` | **MET** |
| 3 | the same, SS and OS separately | same | SS residual `1.677e-11`, OS residual `3.741e-11` | `2e-6` | **MET** |
| 4 | KMP2 `e_corr`, He/6-31g `[1,1,2]` mesh 9, FFTDF | same | Rust `-0.03324144673276666`, upstream `-0.03324144675995790`, residual **`2.719e-11`** (SS `1.886e-14`, OS `2.721e-11`) | `2e-6` | **MET** |
| 5 | KMP2 `e_corr` on the **GDF** route | same | ~~diamond residual `1.289e-1`, He residual `1.417e-3` — on mean fields that are themselves `1.523e+1` and `1.461e-1` Ha out~~ **RE-MEASURED 2026-09-05 after `14-VERIFICATION §11`: He residual `4.090e-10` on a mean field `3.017e-9` out; the diamond mean field went `1.523e+1` -> `2.173e-8`** | `2e-6` | **MET (2026-09-05)** — was **NOT MET — Phase 14**; see §1a |
| 6 | GDF `Lov` route vs GDF four-index route, same mean field | `tests/kmp2.rs` | `< 2e-15 Ha` | `2e-15` | **MET** |
| 7 | `e_corr_ss + e_corr_os == e_corr` | `oracle_phase15::kmp2_energies`, `tests/kmp2.rs` | exact `f64` equality on every route and system | bitwise | **MET** |
| 8 | primitive-kmesh / supercell equivalence, no oracle | `tests/kmp2_supercell.rs` | He/6-31g `[1,1,2]` equals the gamma `1×1×2` supercell per cell | `2e-8` | **MET** |
| 9 | cache vs no cache | `tests/pbc_ao2mo_mofirst.rs` | bit-identical FFT/AFT AO-ERI and MO-first | bitwise | **MET** |
| 10 | rayon determinism, KMP2 `e_corr` and `t2` at 1 and 8 threads | `tests/kmp2_determinism.rs`, `tests/perf_dpbc28.rs` | bit-identical | bitwise | **MET** |
| 11 | `KptsHelper.symm_map` insertion order **and** `_operation` | `oracle_phase15::symm_map_and_operation` | exact integer equality, diamond `[1,1,2]` (5 orbits) and `[2,2,2]` (176 orbits) | exact | **MET** |
| 12 | `padding_k_idx`/`get_nocc`/`get_nmo`/`get_frozen_mask`, all four `frozen` forms | `oracle_phase15::padding_surface` | exact integer equality on upstream's ragged `nmo=(6,6,5)`, `nocc=(2,3,2)` example | exact | **MET** |
| 13 | `ao2mo` + `ao2mo_7d` through the trait, FFTDF | `oracle_phase15::ao2mo_and_ao2mo_7d` | `5.793e-12` on upstream's own random MO draw | `6e-11` | **MET** |
| 13a | the same, AFTDF | same | `1.891e-4`, attributable to the AO integral (see §1b) | `6e-4`, measured | **MET at the measured integral floor** |
| 14 | the same for the Gaussian builders | same | reported, not gated — the residual is row 5's `j3c` baseline | — | **REPORTED** |
| 14a | `Lov` element-wise vs `_init_mp_df_eris`, diamond `[1,1,2]`, on upstream's own padded MOs | `oracle_phase15::lov_blocks` | ~~max deviation `7.518e-1` — the same `j3c` baseline as rows 5 and 14; reported, not gated~~ **RE-MEASURED 2026-09-05: `1.569866e-5`, and now GATED at `2e-4`** | `2e-4` | **MET (2026-09-05)** — was **REPORTED — Phase 14** |
| 15 | `t2` blocks, `make_rdm1` (padded and compact), `gamma1_intermediates` | `oracle_phase15::t2_rdm1_and_gamma1` | element-wise vs upstream, He/6-31g `[1,1,2]` FFTDF | `2e-8` | **MET** |
| 16 | `Tr(rdm1)` and Hermiticity | `tests/kmp2.rs` | trace `8.0` to `2e-10`; Hermitian to `2e-12` | as shown | **MET** |
| 17 | MO-first `ao2mo` block, FFTDF | `oracle_phase15::mo_first_ao2mo_block` | **`5.815e-14`** / **`5.818e-14`** (two runs) element-wise over **every** conserving quadruple, diamond `gth-szv` `[1,1,2]`, on upstream's own padded MOs and the same 47³ mesh (asserted) | `2e-12` | **MET** |
| 17a | the same, AFTDF | `oracle_phase15::ao2mo_and_ao2mo_7d` (row 13a) | **covered there, not here.** `Aftdf::ao2mo` dispatches to the same `aft_general_mo_first`, and row 13a gates it on He/6-31g mesh 9. On diamond's 47³ mesh a single AFT quadruple ran past **19 CPU-minutes** without finishing on either side, which buys nothing the cheaper fixture does not already prove. | see row 13a | **MET via row 13a** |
| 18 | **staggered KMP2 energy**, submesh (`flag_submesh=True`) | `tests/kmp2_stagger.rs`, `measurements/stagger.py` | Rust `-0.01608990037894895`, upstream `-0.016089900380356827`, residual **`1.408e-12`** | `2e-6` | **MET** |
| 19 | **staggered KMP2 energy**, full mesh (`flag_submesh=False`) | same | Rust `-0.01402871682328140`, upstream `-0.014028716824109303`, residual **`8.279e-13`** | `2e-6` | **MET** |
| 20 | standard KMP2 on the same staggered fixture | same | Rust `-0.01439020371308994`, upstream `-0.014390203713094872`, residual **`4.928e-15`** | `2e-6` | **MET** |
| 21 | the staggered submesh index maps | same | `kpts_idx_occ = [7]`, `kpts_idx_vir = [0]`, matching upstream exactly | exact | **MET** |
| 22 | KUMP2 | `tests/kump2.rs`, `oracle_phase15` part 8 | the unrestricted padding surface ships; the energy kernel refuses, and upstream 2.12.1 still raises | exact | **MET** |
| 23 | `check-no-fma` | — | **deliberately not run.** `AGENTS.md` says FMA checking is not required for this project. Phases 13 and 14 ran it; this phase does not, and this row exists so the difference reads as a ruling rather than an oversight. | — | **N/A by ruling** |

### §1b Every AO2MO residual is its AO integral, not the transform

The test emits the **AO-level `get_eri` block** alongside the MO one, on the
same k-quadruple, so a residual is attributable without a second oracle run.
He/6-31g `[1,1,2]`, mesh 9, upstream's own random complex MO draw:

| builder | AO `get_eri` | `ao2mo` | `ao2mo_7d` |
|---|---|---|---|
| FFTDF | **`2.325e-13`** | `5.793e-12` | `5.794e-12` |
| AFTDF | **`2.761e-5`** | `1.891e-4` | `1.891e-4` |
| MDF | ~~`8.194e-6`~~ **`1.243e-9`** | ~~`4.908e-5`~~ `2.611e-8` | ~~`4.908e-5`~~ `2.611e-8` |
| GDF | ~~`1.221e-1`~~ **`1.988e-10`** | ~~`1.930e+0`~~ `1.774e-9` | ~~`1.930e+0`~~ `1.774e-9` |

The two Gaussian rows are the RE-MEASURED values (2026-09-05, after
`14-VERIFICATION.md §11`); the struck-through ones are what this table
originally recorded. GDF fell by **nine orders** at the AO level, which is the
cleanest single statement of what the `s2` off-diagonal k-pair packing defect
was doing: it corrupted `cderi` itself, so every consumer inherited it. Both
rows are now GATED in `oracle_phase15::ao2mo_and_ao2mo_7d` (`2e-8` for GDF,
`3e-7` for MDF) where they used to be checked only for non-finiteness.

Every row scales by the same ~7-16× factor from AO to MO — that is the
magnitude of upstream's random coefficients, nothing else. `ao2mo` and
`ao2mo_7d` agree exactly on every builder. **No residual in this table is
produced by anything Phase 15 built**; each is the AO integral the builder
hands over. FFTDF, the route the phase's gates run on, is exact to `2.3e-13`.
GDF's `1.221e-1` at the AO level is the same defect §1a prices at the SCF
level.

### §1a Why row 5 is Phase 14's — and it is worse than Phase 14 knew

`oracle_phase15::kmp2_energies` now prints the **mean-field** residual beside
the KMP2 one. The mean field is where the whole thing goes wrong:

| system, route | `e_tot` (this port) | `e_tot` (upstream) | SCF residual | KMP2 residual |
|---|---|---|---|---|
| diamond, FFTDF | `-8.65192328388683407` | `-8.65192328393455057` | **`4.772e-11`** | `5.418e-11` |
| diamond, **GDF** | `-23.88282979704741749` | `-8.65527634481766839` | **`1.523e+1`** | `1.289e-1` |
| He/6-31g, FFTDF | `-2.95037354933798257` | `-2.95037354933213347` | **`5.849e-12`** | `2.719e-11` |
| He/6-31g, **GDF** | `-2.34276106854969646` | `-2.48883216137747532` | **`1.461e-1`** | `1.417e-3` |

A KMP2 correlation energy computed on a Hartree-Fock reference that is **15.2
Hartree** wrong cannot be evaluated as an MP2 result at all. Three further
facts place the defect outside this phase:

* On the same GDF mean field, the `Lov` route and the four-index AO2MO route
  agree to `< 2e-15 Ha` (row 6). Two independent integral paths through the
  same KMP2 kernel cannot both be wrong and agree with each other to `2e-15`.
* Upstream's own two routes on **its** GDF mean field agree to `6.9e-13`
  (`-0.20043610112149332` via `Lov`, `-0.20043610112218085` forced through
  AO2MO), so the route split is not the source upstream-side either.
* The `Lov` blocks themselves differ from `_init_mp_df_eris` by `7.518e-1`
  (row 14a) when driven from **upstream's own padded MO coefficients** — i.e.
  the `j3c` tensor the transform reads is already wrong before any MO enters.
  *(This bullet was the decisive one, and it pointed at the right place: the
  defect was in `cderi`. Post-fix the same measurement is `1.569866e-5`.)*

**This is a new finding, not a restatement of Phase 14's.** `14-VERIFICATION`
gates GDF against upstream on `he_all_electron` (`sto-3g` — **one** AO) at
`2×2×2`, and on **diamond at gamma only**, where it is already recorded
`PARTIAL` (`gate3_rsdf.rs:143-147` says so, and gives the reason: diamond's
three-centre build is slow). **Diamond GDF at a non-gamma k-mesh with a
pseudopotential had never been run against upstream.** It now has, by
`oracle_phase15::kmp2_energies`, and it is `1.523e+1 Ha` out. The reproduction
is three lines:

```rust
let cell = common::diamond_anchor();                 // gth-szv / gth-pade, Bohr
let kpts = cell.make_kpts([1, 1, 2]).unwrap();
let mut mf = Krhf::from_df(Box::new(Gdf::new(cell, &kpts)));
mf.exxdiv = None;                                    // upstream: KRHF(..., exxdiv=None)
// cfg.conv_tol = 1e-11 -> e_tot = -23.8828…, upstream -8.6552…
```

Phase 14 owns it. Phase 16's `16-01` must not gate anything on the GDF route
until it is fixed.

#### RESOLVED 2026-09-05 — `14-VERIFICATION.md §11`

Two defects, and this section's diagnosis was right on every count: the fault
was in the `j3c` tensor the transform reads, not in KMP2.

1. **`sr_loop` served the wrong half of every off-diagonal k-pair.** An `s2`
   store holds only `mu >= nu` of each pair, and `(L | mu^{ki} nu^{kj})` is
   Hermitian in `(mu, nu)` ONLY at `ki == kj`; upstream joins the two stored
   triangles in `PBCunpack_tril_triu` (`lib/pbc/fill_ints.c:1460-1483`). The
   `s2 -> s1` unpack used `lib.ANTIHERMI` on the same block instead, and
   `ff01948` added a second, opposite half-error on top of it.
2. **`gdf::nuc::get_nuc`/`get_pp` ran AFTDF on `cell.mesh`**, where
   `_CCNucBuilder` is mesh-INDEPENDENT — worth `0.2 Ha` per element on the
   He/`6-31g` fixture, whose mesh is pinned to `[9,9,9]`.

Re-measured, same commands:

| system, route | `e_tot` (this port) | `e_tot` (upstream) | SCF residual | KMP2 residual |
|---|---|---|---|---|
| diamond, **GDF** | `-8.65527636655032495` | `-8.65527634481766839` | **`2.173e-8`** | not re-run |
| He/6-31g, **GDF** | `-2.48883215836059124` | `-2.48883216137747532` | **`3.017e-9`** | **`4.090e-10`** |

Diamond's `2.173e-8` is its ORDINARY GDF fitting residual, not a remainder:
`14-VERIFICATION §5` records `2.074e-8` for the same builder on the same cell at
`2x2x2`, and the DF fitting error is reproduced — `|E_FFTDF - E_GDF|` is
`3.353083e-3` here against upstream's own `3.353061e-3`.

**Row 6 survived the fix**, which is the load-bearing check: `ff01948`'s
`reverse` branch existed to force the `Lov`/AO2MO agreement, and removing it
left that agreement intact at `2e-15` because both routes now read the same
CORRECT square rather than the same wrong one.

Two `tests/kmp2.rs` assertions were corrected alongside
(`measurements/kmp2_gdf_and_rdm1.py`): the GDF `e_corr` self-pin became an
oracle gate, and `diamond_anchor_and_without_t2`'s per-k-point
`Tr(gamma_k) == nelec` assertion was replaced — that identity holds only after
the k-AVERAGE, and upstream misses the per-k form by `2.8e-2`. The second
failure is INDEPENDENT of the two defects above: it reproduces bit-identically
with the fixes stashed, and its cell runs on FFTDF.

---

## §2 Reference-value index

Every committed constant, with what produced it.

| constant | value | generated by | lives in |
|---|---|---|---|
| diamond FFTDF anchor | `-0.204721432828996` | `pyscf/pbc/mp/kmp2.py:820` (upstream source) | `tests/kmp2.rs`, `oracle_phase15.rs` |
| the same, live | `-0.20472143304034024` | `measurements/anchor.py` → `anchor.out` | `measurements/README.md` |
| anchor SS / OS | `-0.034594521893337379` / `-0.17012691114700285` | `measurements/anchor.py` | `measurements/README.md` |
| He/6-31g FFTDF | `-0.033241446759957924` | `measurements/routes.py` → `routes.out` | `tests/kmp2.rs` |
| He/6-31g GDF (upstream) | `-0.016989369077568279` | `measurements/routes.py` | `measurements/README.md` |
| He/6-31g GDF (this port) | ~~`-0.015572369890603862`~~ **`-0.01698936866861078`** (2026-09-05) | `tests/kmp2.rs`, now an ORACLE gate rather than a self-pin | `tests/kmp2.rs`, `15-.../measurements/kmp2_gdf_and_rdm1.out` |
| ragged padding fixture | `nmo=(6,6,5)`, `nocc=(2,3,2)`, dense `7` | `measurements/padding.py` → `padding.out` | `tests/padding.rs` |
| H2 dimer stagger, submesh | `-0.016089900380356827` | `measurements/stagger.py` → `stagger.out` | `tests/kmp2_stagger.rs` |
| H2 dimer stagger, full mesh | `-0.014028716824109303` | same | `tests/kmp2_stagger.rs` |
| H2 dimer standard KMP2 | `-0.014390203713094872` | same | `tests/kmp2_stagger.rs` |
| H2 dimer KRHF `e_tot` | `-1.1004620466064836` | same | `tests/kmp2_stagger.rs` |
| H2 dimer mesh | `[29,29,29]` (from `ke_cutoff = 100`) | same | `tests/kmp2_stagger.rs` (asserted) |
| upstream's own stagger constants | `-0.0160902544091997` / `-0.0140289970302513` / `-0.0143904878990777` | `kmp2_stagger.py:385/390/395` | quoted in `stagger.py`; **not** used as gates — they sit `2.8e-7…3.5e-7` from what 2.12.1 actually produces, which is why the live values above are the gates |
| upstream diamond GDF | `-0.20043610112149332` (`Lov`) / `-0.20043610112218085` (AO2MO) | `measurements/oracle_rollup.py kmp2` | §1a |
| FFTDF `ao2mo` tolerance | `6e-11` | measured `5.793e-12`, one order of headroom | `oracle_phase15.rs` |
| AFTDF `ao2mo` tolerance | `6e-4` | measured `1.891e-4`, whose AO-level source is `2.761e-5` (§1b) — the analytic-integral floor at mesh 9, not a transform residual | `oracle_phase15.rs` |

The headline tolerance is **`2e-6 Ha` per DF route**, measured in 15-01 and
unchanged. It is deliberately far above the observed FFTDF residuals
(`2.7e-11`…`5.4e-11`) because it has to survive two independent SCF paths; the
observed numbers are recorded above so a future regression is visible long
before the gate trips.

---

## §3 What the phase's own tests caught

Three defects, all found by the staggered-energy oracle that the implementation
pass left unrun. Each is stated with what it was worth.

### 3.1 `get_occ` took its k-point count from `mf.kpts`, not from its argument

`khf.py:191-192` is `nkpts = len(mo_energy_kpts)` — the electron count comes
from the **argument**, not from the mean field. This port's `Krhf::get_occ`
used `self.kpts().len()`.

The two agree on every SCF iteration, so no existing gate could see it. They
diverge exactly when `get_occ` is called on bands evaluated on a *different*
mesh — which is what `kmp2_stagger.py:271` does. On the H2 fixture the port
filled 8 of 16 k-points, leaving eight k-points with **zero** occupied
orbitals, padding the dense dimension from 2 to 3, and returning

* **`-0.3736910783398668 Ha`** where upstream gives `-0.014028716824109303 Ha`
  — **a factor of 26.6**.

Fixed in `crates/pyscf-pbc-scf/src/krhf.rs`. `Kghf::get_occ` carried the
identical divergence against `kghf.py:109` and is fixed with it; both are
no-ops during SCF, so nothing already gated moves.

### 3.2 The staggered four-index path used the mean field's own builder

`kmp2_stagger.py:73-75` builds a **fresh `df.FFTDF(mp.cell, mp.kpts)`** for the
non-DF path — *even on a GDF mean field*, and unlike plain `KMP2`, which uses
`mp._scf.with_df.ao2mo` (`kmp2.py:92`). This port used `self.df`.

Measured on the H2 fixture with a GDF mean field: upstream's `Lov` route gives
`-0.015836452346190341` and its forced four-index route gives
`-0.015886558478292751` — **`5.01e-5 Ha` apart**, because they are different
integral approximations. The port would have reported the wrong one of the two
whenever a caller set `with_df_ints = false` on a GDF stagger.

Fixed by `Kmp2Stagger::integral_df`, which now resolves the builder inside
`kernel` the way upstream does.

### 3.3 `new_full_mesh` hardcoded `with_df_ints = false` and an FFTDF

`kmp2_stagger.py:279-282` sets `with_df_ints` from the **mean field's**
builder regardless of `flag_submesh`; only the *reuse* of an existing `_cderi`
is gated on `flag_submesh` (`:165`), and the non-submesh case rebuilds
`df.GDF(mp.cell, mp.kpts)` over the combined mesh (`:169`). The port always
built an FFTDF and always took the four-index path, so a full-mesh stagger on a
GDF mean field silently computed plane-wave integrals. Fixed alongside 3.2;
`flag_submesh` is now a real field rather than an implicit constructor choice.

### 3.4 `aft_general_mo_first` conjugated the ket pair on the wrong side

`aft_ao2mo.py:215-216` conjugates the ket AO pair **before** the MO transform
(`rskR - rskI*1j`, then `_ao2mo.r_e2`, which itself conjugates the bra
coefficient). This port conjugated the **transformed** pair instead, which
computes `conj(conj(c_k) c_l · A) = c_k · conj(c_l) · conj(A)` — the wrong one
of the two coefficients.

It is **exactly equivalent for real coefficients**, which is why nothing caught
it. `Aftdf::ao2mo` and `ao2mo_cached` — the Phase-15 `PeriodicDf` AO2MO
dispatch for AFTDF, and therefore KMP2's integral source on an AFTDF mean field
— both route to it.

Measured on He/6-31g `[1,1,2]` mesh 9 with complex MO coefficients:

* `aft_general` (AO-first) vs `aft_general_mo_first`: **`4.596e-1`** before,
  **`1.804e-16`** after. FFT's two routes were `8.327e-17` throughout.
* against upstream's own `AFTDF.ao2mo`: **`2.323e+1`** on a block whose largest
  element is `33.6` — i.e. the values were unrelated, not merely imprecise.

**Why no test could see it.** The only non-gamma MO-first test,
`pbc_ao2mo_mofirst.rs::mo_first_matches_ao_first_away_from_gamma`, runs on
`common::he_all_electron` — which is **`sto-3g`: one AO** — with
`MoCoeff::identity`. Its entire MO transform is the 1×1 identity. A new test,
`mo_first_matches_ao_first_with_complex_coefficients`, runs the same sweep on a
two-AO He/6-31g cell with complex, non-identity coefficients at every k-point,
gated at `2e-13`, and asserts `nao > 1` so it cannot silently degenerate the
same way.

### 3.5 Found earlier, during implementation

Recorded here so the ledger is complete: the non-gamma FFT AO-ERI reciprocal-bin
permutation bug (15-08) and `sr_loop`'s missing lower-k-pair/conjugate-transpose
reconstruction (15-04), both of which produced plausible wrong numbers and both
of which are now pinned by tests.

---

## §4 Premises that turned out wrong

`15-CONTEXT.md §1` recorded three corrections to `PBC-MASTER-PLAN §8.7` before
any code: `kconserv` was already shipped (09-07), most of 15-02 was already
shipped (14-05), and `KUMP2`'s energy kernel does not exist upstream. 15-01
then replaced two unmeasured gate numbers (`1e-14` in `ROADMAP`, `1e-8` in the
master plan) with a measured `2e-6`.

**Which of the three would have cost a rewrite if found later?** The `ktensor`
one. Building the `KsymmArray` container in Phase 15 would have produced a type
with no caller and no way to test it, and Phase 17's `KPoints` machinery — which
shipped in the meantime — would then have had to either adopt or delete it. The
other two would have cost duplicated work, not a rewrite.

**Corrections implementation forced, beyond those three:**

1. **The upstream constants in `kmp2_stagger.py` are not 2.12.1's own output.**
   `:385/:390/:395` assert at `1e-5`; 2.12.1 produces values `2.8e-7`…`3.5e-7`
   away from them. Gating the port against the source-tree constants would have
   pinned a number upstream itself no longer reproduces, so `stagger.py`
   measures them live and the tests gate on the measured values. This is the
   same failure mode as the diamond anchor, whose live value sits `2.1e-10`
   from `kmp2.py:820`.
2. **`he_fcc` `gth-szv` cannot host the `Lov` oracle** that `15-07-PLAN.md`
   Task 1 part 4 specified: it is a single He atom with **one** AO, so
   `nvir = 0` and the block is empty. Parts 4 and 9 use diamond `gth-szv`
   `[1,1,2]` — the phase's own anchor system — and part 3 uses He/6-31g.
3. **`kmp2_stagger`'s builder choice is not `kmp2`'s** (§3.2), and it is made
   inside the kernel, not at construction (§3.3).
4. **§7.0's `(nao²/(nocc·nvir))² = 16×` prediction for the MO-first route
   over-states the measured gain**: the diamond `[1,1,2]` sweep measures
   **9.784×** (§8). The prediction counts only the pair-transform flops and
   ignores the grid work both routes share.

---

## §5 Deferred branches

Every refusal this phase ships, with the phase that owns it.

| refusal | site | owner |
|---|---|---|
| `ktensor` / `KsymmArray` | not built | **17** (shipped there) |
| `KPoints` / IBZ construction inside `KMP2::new` | not built | **17** |
| `KUMP2::kernel` energy | `PbcMpError::Kump2NotImplemented`, `kump2.rs:104/108` | **upstream** — `kump2.py:38/:384/:402` all raise in 2.12.1; the port ships the same surface and the same refusal, oracle-gated so it cannot outlive its reason |
| `kump2::_add_padding` | same | **upstream** |
| `dimension == 2` in the DF `Lov`/`ao2mo` path | `lov.rs:73` (negative `cderi` block), `_init_mp_df_eris`'s own `NotImplementedError` | **upstream** |
| `Frozen::{Auto, Window}` at k-points | `PbcMpError::UnsupportedFrozen`, `frozen_k.rs:55/161/224` | **15**, deliberately: upstream's `frozen='auto'` at k-points is molecular-only |
| `kmp2_stagger` non-submesh for `dimension < 3` | `kmp2_stagger.rs`, `PbcMpError::Shape` | **upstream** — `kmp2_stagger.py:256-262` raises `NotImplementedError` for the same reason |
| odd Monkhorst-Pack mesh for a submesh stagger | `PbcMpError::OddStaggerMesh` | **upstream** — `kmp2_stagger.py:234-240` raises `RuntimeError` |
| fractional occupations in `get_nocc` | `PbcMpError::FractionalOccupation` | **upstream** — and load-bearing here, because this port has live smearing that upstream's KMP2 never sees |
| `fft_ao2mo.general`'s gamma-point/all-real shortcut | not ported | **15**, deliberately: `fft_ao2mo.py:126-143` is a speed shortcut; the general complex path is correct there, only slower, and KMP2 never reaches it |

---

## §6 Carry-overs into Phase 16

* **`KptsHelper::build_symm_map` exists and is oracle-gated** (§1 row 11),
  including its insertion order and `_operation`. KCCSD should consume it
  rather than recompute orbits. Its cost is measured in §8.
* **`Eri7d`'s index contract is stable and unchanged**: `eri[ki,kj,kk][i,j,k,l]`
  with `kl = kconserv[ki,kj,kk]`, chemists' notation (`df_ao2mo.rs:34-70`).
  Phase 16 inherits it as-is.
* **The `WorkspacePool` arena was NOT needed at Phase-15 sizes.** KMP2's largest
  live object is `nkpts³ · (nocc·nvir)²` complex for `t2`, plus one
  `nkpts · (nocc·nvir)²` scratch per live outer pair; the thread-aware preflight
  in `kmp2_kernel.rs` bounds it and no test tripped it. KCCSD's `Wvvvv` is
  `nkpts³ · nvir⁴` and will need the arena regardless — see **D-PBC-29 (1)**.
* **`symm_map` is deliberately unused in `KMP2::kernel`** (`Kmp2::new` calls
  `KptsHelper::without_symm_map`). The available saving there is 2-fold, not
  8-fold, because two of the four operations produce `(vo|vo)` blocks KMP2 never
  asks for. **This ruling does not carry to KCCSD** — see D-PBC-29 (3).
* **The GDF numerical baseline** (§1 row 5) — **UNBLOCKED 2026-09-05.** It read:
  *"blocks any Phase-16 gate that runs on the GDF route — the mean field itself
  is `15.2 Ha` out on diamond at `[1,1,2]`, which `14-VERIFICATION` never
  measured because its diamond GDF gate is gamma-only and already `PARTIAL`."*
  Phase 14 fixed both causes (`§1a`'s RESOLVED note, `14-VERIFICATION §11`);
  diamond GDF is now `2.173e-8` and He/6-31g `3.017e-9`. Phase 16's `16-01`
  should still measure its own floor per DF route — that instruction was good
  practice independently of this defect — but a GDF row is no longer blocked.

---

## §7 Standing caveats

* **Ångström/Bohr.** Any new reference cell must state its unit and, if it is
  specified in Ångström, be re-derived against the CODATA constant this port
  uses. The diamond anchor is Bohr-specified precisely to avoid this.
* **The two-route rule** (`15-CONTEXT §2.2`). `kmp2.py:69` branches the entire
  `(ia|jb)` computation on `isinstance(with_df, GDF)`. A gate that says
  "matches upstream" without naming its DF route is untestable. Every row in §1
  names its route.
* **Live-vs-source constants.** Two of upstream's own committed constants
  (`kmp2.py:820`, `kmp2_stagger.py:385/390/395`) no longer reproduce exactly in
  2.12.1. Gate against a measured value and record the drift; never against a
  constant embedded in the source.

---

## §8 The performance ledger (D-PBC-28, `15-CONTEXT §7`)

Separate from the correctness gates. **No row here is gated to a tolerance** —
wall clock is not reproducible — except the residuals, which are.

| § | claim | measured | source |
|---|---|---|---|
| 7.1 | `Lov` block build, 1 vs 8 threads, **and** bit-identity | diamond `gth-szv` `[1,1,2]`, GDF. Bit-identical block for block in every run. Wall clock is **not measurable at this size** — the whole build is under a millisecond and three runs gave `0.728/0.356 ms = 2.04×`, `0.698/3.206 ms = 0.22×`, i.e. rayon pool startup, not scaling. | `tests/perf_dpbc28.rs::lov_build_and_kmp2_kernel_thread_scaling` |
| 7.1 | KMP2 `(ki,kj,ka)` loop, `Lov` route, same | `e_corr` and every `t2` block bit-identical. Same size problem: `1.906/0.365 ms = 5.22×`, `1.053/0.339 ms = 3.10×`, `2.334/5.150 ms = 0.45×` across three runs of a **2 ms** kernel. | same test |
| 7.1 | KMP2 `(ki,kj,ka)` loop, **four-index FFTDF route** — the row that answers the question | `t1 = 56.825 s`, `t8 = 4.522 s`, **`12.566×`**; `e_corr` and every `t2` block **bit-identical** at both thread counts (`-0.20472143370251261`). Each `(ki,kj,ka)` task is a full plane-wave transform, so the parallel work dominates and the ratio is real. | `tests/perf_dpbc28.rs::kmp2_four_index_thread_scaling` |
| 7.2 | `oracle_zdotu` shipped and used at all three unconjugated sites | `kmp2_kernel.rs` `df_oovv` (`Lov·Lov`), `edi`, `exi` — three sites, all `oracle_zdotu`/`oracle_zdotu_re`, none `oracle_zdot` | `grep`, `crates/pyscf-algebra/tests/zoracle_zdotu.rs` |
| 7.3 | no `zgemm_dense`/`gemm_dense` on any Phase-15 path | zero occurrences in `pyscf-pbc-mp`, `pyscf-pbc-ao2mo`, and `pyscf-pbc-df`'s ao2mo modules | `grep` |
| 7.4 | `Lov` stored `L`-fastest; the ordered dot is contiguous | `lov.rs` stores `(ia, L)`; `df_oovv` slices `naux` contiguously and feeds `oracle_zdotu` | `tests/lov.rs` |
| 7.5 | MO-first vs AO-ERI: wall clock, memory, residual | diamond `gth-szv` `[1,1,2]`, 8 conserving quadruples, `ngrids = 103823`. **Four runs**: `80.995/8.049 s = 10.062×` (quiet machine), `78.289/8.002 = 9.784×`, `150.528/20.220 = 7.444×`, `210.419/35.201 = 5.978×` (progressively more loaded). Residual **`1.735e-17`** in every run — bit-level. Derived scratch: AO pair-grid **101.4 MiB** vs MO pair-grid **25.3 MiB**; process `VmHWM` 638.0 MiB on the quiet run. | `crates/pyscf-pbc-df/tests/perf_dpbc28_mofirst.rs` |
| 7.5 | the same against §7.0's prediction | predicted `(nao²/(nocc·nvir))² = 16×`; measured **10.1× on a quiet machine**, degrading to `6×` under load. The spread is machine load, not the algorithm — the residual is identical to the last bit in all four runs. **The prediction over-states the gain** because both routes pay the same grid-side cost (§4.4); the direction and order of magnitude hold. | same |
| 7.5 | the same ratio in upstream | not separable: upstream has no AO-first FFT AO2MO to compare against — `fft_ao2mo.general` is MO-first (`fft_ao2mo.py:145-152`). This is why 15-08 exists. | `15-01` Task 7b |
| 7.6 | per-`q` `CoulGCache` + hoisted MO slices: bit-identical, and the saving | bit-identical (§1 row 9). The cache collapses `nkpts²` `CoulGCache::build` calls to one per distinct momentum transfer — on diamond `[2,2,2]` that is 64 → 8. | `tests/pbc_ao2mo_mofirst.rs`, `kmp2_kernel.rs` |
| 7.7 | `symm_map` declined in `KMP2::kernel` | ≤2×, logged as a Phase-16 carry-over (§6) | `15-REVIEW.md D-15-R-04` |
| 7.8 | peak-memory formula incl. the thread factor; pool bounded | `kmp2_kernel.rs` preflights `t2 + Lov + live_outer · nkpts · (nocc·nvir)² · 16 B` and caps `live_outer` at `rayon::current_num_threads()` | `kmp2_kernel.rs` |
| 7.9 | upstream DF-vs-non-DF ratio | **5.6313×** in GDF/`Lov`'s favour, upstream He/6-31g fixture, median of three kernel-only runs | `measurements/routes.py` → `routes.out` |
| — | `build_symm_map`'s `O(nkpts³)` growth | Orbit counts are exact and run-independent: `176 / 5 292 / 67 712 / 496 125` at `nkpts = 8 / 27 / 64 / 125`. Quiet-machine run: **0.082 ms / 5.17 ms / 178 ms / 1.520 s**; loaded run: **0.143 ms / 15.6 ms / 503 ms / 4.747 s**. Consecutive ratios `62.9 / 34.4 / 8.5` against `n³` predictions `38.4 / 13.3 / 7.5`: the `27→64` step grows **2.6-3.3× faster than `n³`** in every one of five runs. | `tests/perf_dpbc28.rs::build_symm_map_growth_curve` |

**Reading the `build_symm_map` row.** It is not free at Phase-16 sizes: 1.4-4.7 s
of single-threaded, deliberately sequential work at `nkpts = 125`, growing
super-cubically at the `27→64` step because the `HashMap` claim pass leaves
cache. It is still
cheap next to one KCCSD iteration, and it is why `Kmp2::new` does not build it —
but Phase 16, which *does* want it, should build it once and keep it.

**Rows not reached, and why.** The `[2,2,2]` and `gth-dzvp` legs of the §7.5
sweep are 64× and ~10× the `[1,1,2]` row respectively on the same 47³ mesh; a
`[2,2,2]` AO-first half alone was measured at **~26 s per quadruple × 512**
under load, i.e. multiple hours, and was stopped rather than reported from a
partial run. The `[1,1,2]` row above establishes the ratio and the residual;
the larger legs would refine the ratio, not change the finding. The test that
runs them is committed as `mo_first_vs_ao_first_cost_222` so the measurement is
one command away.

**A ratio near 1.0 in a parallelism row is a finding, not a pass** — and the
first two §7.1 rows range from `0.22×` to `5.22×` *across repeats of the same
measurement*, which is a stronger finding still. On the `Lov` route at diamond
`[1,1,2]` the entire block build is **0.7 ms** and the entire KMP2 kernel
**2 ms**; at that size `rayon`'s pool startup *is* the measurement, and the
run-to-run spread swamps any scaling signal. That is not evidence the loops
fail to parallelise — the bit-identity holds and the results are identical —
it is evidence that **this fixture is too small to measure scaling on**, which
is exactly the trap the plan named. `[2,2,2]` would be the natural larger
fixture and is not reachable on the `Lov` route: this port's GDF `make_j3c`
over the fused cell did not finish an hour there
(`tests/gdf_builder.rs:477` already prices `[1,1,2]` at tens of minutes).

**The four-index FFTDF route settles it.** Each `(ki,kj,ka)` task there is a
full plane-wave transform, the kernel runs **56.8 s** single-threaded, and
eight threads bring it to **4.5 s** — `12.566×`, with `e_corr` and every `t2`
block bit-identical between the two. The parallelism is real, it scales, and
the ordered `oracle_*` reductions cost nothing in determinism. (The ratio
exceeding 8 on an 8-thread pool is a cache/turbo effect at this size, not a
claim about the scaling law; the load-bearing statement is that the work
distributes and the answer does not move.)

---

## §9 Verification commands

```bash
cargo build --workspace
cargo test --workspace
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo run -p xtask --bin check_dependency_wall
cargo run -p xtask --bin check_forbidden_paths
cargo run -p xtask --bin check_orphan_modules
cargo fmt --all --check

# the opt-in oracle matrix (never runs under a plain `cargo test`)
PYSCF_ORACLE_VENV=1 cargo test --release -p pyscf-pbc-mp \
  --test oracle_phase15 -- --ignored --nocapture

# the D-PBC-28 measurements
cargo test --release -p pyscf-pbc-mp --test perf_dpbc28 -- --ignored --nocapture
cargo test --release -p pyscf-pbc-df --test perf_dpbc28_mofirst -- --ignored --nocapture
```

`check-no-fma` is deliberately absent — see §1 row 23.

### §9a What those commands actually returned

| command | result |
|---|---|
| `cargo build --workspace` | **clean** |
| `cargo run -p xtask --bin check-dependency-wall` | **PASS** — cubecl-* containment intact (ALG-06) |
| `cargo run -p xtask --bin check-forbidden-paths` | **PASS** — 382 `.rs` files, no out-of-scope upstream imports |
| `cargo run -p xtask --bin check-orphan-modules` | **PASS** — 348 source files, all reachable |
| `cargo fmt --all --check` | **clean inside this repository** — every reported diff is in the sibling `cintx` / `libxc_rs` **path dependencies**, which have pre-existing drift outside this repo (the deviation Phase 17 also recorded) |
| `cargo clippy` on the four crates this phase touched, `--all-targets --no-deps` | **clean apart from four pre-existing `collapsible_if` warnings** at `fftdf.rs:182/197/242` and `aft_jk.rs:444` — all outside this phase's diff hunks, all a `let`-chain lint from the current toolchain |
| `cargo test --workspace` | **one failure, not this phase's**: `pyscf-dft::hooks::tests::define_xc_string_form_parses` (`hooks.rs:654`, `spec.hyb()` ≠ 0.2). Verified by **stashing this phase's three source edits and re-running** — it still fails. `crates/pyscf-dft/src/hooks.rs` is itself **unmodified** in the working tree, and `pyscf-dft` has **no `pyscf-pbc-*` dependency**. The working tree also carries unrelated in-progress edits to `pyscf-core` and `pyscf-gto` (a basis-set-exchange feature) that `pyscf-dft` *does* depend on. It is also a `mod tests` inside a production source file, which `AGENTS.md §2` forbids. Not Phase 15's to fix, and recorded rather than hidden. |
| `PYSCF_ORACLE_VENV=1 … --test oracle_phase15 -- --ignored` | **10 / 10 green**, 1576 s, on the final gates in this document |

The two Phase-15 source edits outside `pyscf-pbc-mp` — `Krhf::get_occ` and
`Kghf::get_occ` — are no-ops during SCF by construction (`mo_energy.len()`
equals `mf.kpts().len()` on every iteration), and the `pyscf-pbc-dft` and
`pyscf-pbc-scf` suites are unchanged by them, which is the diff-against-
committed-residuals check `15-07-PLAN.md` Task 4 asks for.
