# Phase 20 floor table — the v2.0 (Phases 9-19) measured floors, in one place

**Written:** 2026-09-13 by plan 20-01. **Nothing was re-run to produce this
document.** Every number below is COPIED from the file and line named beside it;
where a plan-quoted number could not be found at its cited location, the row
says so. Oracle throughout: vendored PySCF **2.12.1** (`<root>/pyscf`), never
`>=`.

**NO Rust source file was created or edited to produce this document.**
`git diff --stat -- crates/` shows other agents' concurrent edits, none from
20-01; this plan touched only `.planning/` Markdown.

This is the authoritative per-method floor table for the v2.0 milestone. A later
Phase-20 plan that asserts a number against upstream takes its tolerance from
the **floor** column here, never from a round number.

| section | what it holds |
|---|---|
| §1 | the per-method floors the port is gated at (achievable, and in most rows achieved) |
| §2 | the IRREDUCIBLE gaps — properties of the method or of upstream, not defects |
| §3 | bitwise equality with upstream: measured unattainable, and the rule that follows |
| §4 | the Phase-20 identity gate, and the evidence that it discriminates (D-20-C) |
| §5 | Phases 18-19: gates as restated by their own phases (added at execution) |

---

## §1 — Per-method floors

"Floor" is the measured agreement the gate is stated against. "Status" is the
verdict the owning phase recorded. Paths are relative to `.planning/phases/`
unless they start `crates/`.

| # | quantity | fixture | floor | status | source |
|---|---|---|---|---|---|
| 1 | `KRHF` vs upstream, all-electron | He-fcc `sto-3g`, 2×2×2, mesh 15³ | **2.18e-13** (prose rounds it to 2.2e-13, `:120`) | PASS at 1e-12 | `11-fft-fftdf-periodic-hf/11-VERIFICATION.md:90` |
| 2 | `KRHF` vs upstream, `gth-pade` | diamond `gth-szv`, 2×2×2, mesh 47³ (default) | **4.00e-12** (4.02e-12 at mesh 31³, `:92`) | PASS at 1e-11 | `11-fft-fftdf-periodic-hf/11-VERIFICATION.md:93` |
| 3 | `KRKS` vs upstream | Si `gth-szv`, 2×2×2, PBE | **6.45e-12** | PASS at 1e-11 | `12-periodic-dft/12-VERIFICATION.md:77` |
| 3a | `KRKS` vs upstream, all-electron control | He-fcc `sto-3g`, 2×2×2, PBE | **9.81e-14** (221 ulp) | PASS at 1e-12 | `12-periodic-dft/12-VERIFICATION.md:81`, `:144` |
| 4 | FFTDF − AFTDF `KRHF` energy (upstream vs upstream) | diamond 2×2×2, mesh 31 and 41 bit-identical | **2.607e-11** | the Phase-13 Gate-2 plateau | `13-ft-ao-aftdf/measurements/README.md:53` |
| 5 | GDF − RSDF `KRHF` energy (upstream vs upstream) | diamond `gth-szv` 2×2×2 | **1.353e-08** | Phase-14 Gate 3 floor | `14-gdf-mdf-rsdf-rsjk/measurements/README.md:119` |
| 6 | FFTDF − MDF `KRHF` energy (upstream vs upstream) | diamond `gth-szv` 2×2×2, MDF default mesh | **1.124e-06** | Phase-14 Gate 2 floor | `14-gdf-mdf-rsdf-rsjk/measurements/README.md:120` |
| 7 | `KMP2 e_corr` vs upstream | diamond `gth-szv` `[1,1,2]`, FFTDF, `exxdiv=None` | gate **2e-6 per DF route**; measured **5.418e-11** | **MET** | `15-periodic-ao2mo-kmp2/15-VERIFICATION.md:39` (row 2), headline tolerance `:207` |
| 7a | `KMP2 e_corr` vs upstream, GDF route | He/`6-31g` `[1,1,2]` | gate 2e-6; measured **4.090e-10** | **MET (2026-09-05)**, was NOT MET before `14-VERIFICATION §11` | `15-periodic-ao2mo-kmp2/15-VERIFICATION.md:42` (row 5) |
| 8 | `KRCCSD e_corr` vs upstream, **per DF route** | diamond `gth-szv` `[1,1,2]`, mesh `[15,15,15]` pinned | gate **1e-7** (G1 FFTDF, G2 GDF/RSDF) | G1 FFTDF **MET at 6.560e-9**; **G2 GDF/RSDF NOT RUN** | gate `16-periodic-cc-ci/measurements/README.md:40-41` (§1); G1 `16-periodic-cc-ci/16-VERIFICATION.md:44`; G2 `:63` |
| 9 | ksymm **Gate B** — transforms vs one converged SCF | `si`/`diamond` | **≤1e-9** at default `cell.precision`; **≤1e-13** at `precision = 1e-13` | MET as restated | `17-ksymm-multigrid/17-VERIFICATION.md:107-108` |
| 10 | ksymm **Gate C**, FFTDF — `KRHF` ksymm vs full BZ | `si [2,2,2]`, mesh pinned | **8.793e-14** | MET | `17-ksymm-multigrid/17-VERIFICATION.md:123` |
| 10a | ksymm **Gate C**, FFTDF — `KRKS` ksymm vs full BZ | `si [2,2,2]`, both `use_ao_symmetry` branches | **3.109e-14 / 2.842e-14** | MET | `17-ksymm-multigrid/17-VERIFICATION.md:125` |
| 10b | ksymm **Gate C**, GDF — `KRHF` ksymm vs full BZ | `si [2,2,2]` | **2.486e-10** | MET | `17-ksymm-multigrid/17-VERIFICATION.md:124` |
| 11 | ksymm **Gate C**, **GDF** — `KRKS` ksymm vs full BZ | `si [2,2,2]` | upstream vs itself 1.8e-11 … 1.6e-10; measured **1.432e-06** | **NOT MET** — recorded, not absorbed (`§12`, `:801`, states it "against a 1e-8 tolerance") | `17-ksymm-multigrid/17-VERIFICATION.md:126` |
| 12 | `get_bands` vs upstream (off mesh, 2 band k-points) | He-fcc `sto-3g`, 1 AO, 2×2×2, mesh `[15,15,15]` | **1.68e-11** (`mo_energy` on mesh: 6.10e-11) | measured 2026-09-12 | `20-pbc-python-bindings/20-CONTEXT.md:149-150`; produced by `crates/pyscf-pbc-scf/tests/krhf_bands_oracle.rs` (printed, not a literal in the file) |
| 13 | band J/K via `kpts_band`, GDF, model density | 8 sampling + 2 band k-points | **~1.394e-9** (asserted `< 2e-9`) | reproducible to the 4th digit | `crates/pyscf-pbc-df/tests/band_kpoints.rs:211` (comment), `:220-221` (assertion) |
| 14 | row 11 bisected (20-04, added 2026-09-14): GDF `KRKS` ksymm vs full BZ at ONE density, full-BZ arm's XC grid matched to `cell.mesh` | `si [2,2,2]`, default precision, `lda,vwn` | `E_elec[D]` **2.719e-10**; converged SCF Gate C **2.720e-10** (as the gate builds the arms: **1.432e-06**, reproduced to 12 digits) | row 11 is a **DEFECT (a), not a floor**: `Krks::from_df` (`crates/pyscf-pbc-dft/src/krks.rs:93`) grids XC on `Gdf::mesh()` = `[13,13,13]`, not `cell.mesh` = `[35,35,35]`; `_cderi`/J/K/band/hcore steps all 0e0. Gate (1e-8) NOT restated; fix owed | `20-pbc-python-bindings/measurements/gdf-ksymm-bisect.md` |

**Correction to the plan's row "ksymm Gate C, FFTDF — 1.703e-11".** That number
IS in `17-VERIFICATION.md` (`:138`), but the same paragraph says it is *"easy to
mistake for Gate C and is NOT"*: it is the two `eig` routes (`use_ao_symmetry`
on vs off) agreeing on `e_tot`, a Schur's-lemma identity check inside the
k-symmetric SCF. Gate C's FFTDF numbers are rows 10/10a. `PBC-MASTER-PLAN §7`
row 17 carried the same mislabel. It is fixed there by a note, not by deleting
the text.

---

## §2 — Irreducible gaps: NOT defects, NOT to be planned against

Each of these is present in upstream itself, or is a property of the method.
**No implementation reaches them.** A plan that lists one as a defect to fix is
chasing upstream's own numbers and will never close.

| gap | size | source | why no implementation reaches it |
|---|---|---|---|
| FFTDF − GDF `KRHF` energy | **1.222e-03 Ha** | `14-gdf-mdf-rsdf-rsjk/measurements/README.md:121` | GDF *fits* the Coulomb integrals in a finite auxiliary basis; FFTDF evaluates them exactly on the mesh. The gap is the DF **fitting error** — a property of the auxiliary basis, identical in upstream. |
| upstream's own two GDF builders (`_RSGDFBuilder` vs `_CCGDFBuilder`) | **4.502e-06** | `14-gdf-mdf-rsdf-rsjk/measurements/README.md:139` (diamond gamma) | Two upstream constructions of the same fitted quantity disagree with each other. A port can match one route (and gates each against its own upstream number); no single answer matches both. |
| CC route split, FFTDF vs GDF `e_corr` | **9.223479e-04 Ha** (quoted as 9.22e-4) | `16-periodic-cc-ci/measurements/README.md:165` (§4) | The plane-wave pair (FFTDF, MDF) and the Gaussian pair (GDF, RSDF) are two routes in upstream; they sit 9.2e-4 apart on diamond `[1,1,2]`. That is upstream disagreeing with itself, which is why every CC gate names its DF route. |
| `pp_int.get_pp_loc_part2` vs `_IntPPBuilder.get_pp_loc_part2` | **1.7933e-9** (quoted as 1.79e-9) | `13-ft-ao-aftdf/13-VERIFICATION.md:110`, `:114` | Two upstream routes for the same pseudopotential term disagree by this much on diamond. Substituting the `pp_int` route into upstream's own `AFTDF.get_pp` collapses the port's deviation to 3.98e-11 (`:123`). Worth reporting upstream; not reachable by matching both. |
| band energies vs upstream | **~1.7e-11** (1.68e-11 off mesh, 6.10e-11 on mesh) | `20-pbc-python-bindings/20-CONTEXT.md:146-150` | Cross-implementation summation order: the band chain (hcore, plane-wave lattice sum, J2C fit, `kpts_band` J/K, per-k eigendecomposition) runs in C/BLAS upstream and in Rust/CubeCL here. It is ~1e-11 even on the 1-AO all-electron cell with no pseudopotential residual. |

---

## §3 — Bitwise equality with upstream is unattainable

Measured 2026-09-12 on the cheapest periodic fixture that exists — He-fcc
`sto-3g`, **one AO**, all-electron, 2×2×2, mesh `[15,15,15]` — against vendored
2.12.1 (`20-CONTEXT.md:146-156`):

| quantity | agreement | bitwise identical |
|---|---|---|
| `e_tot` | 2.17e-13 | — |
| `mo_energy` (on mesh) | 6.10e-11 | **0 / 8** |
| `get_bands` (off mesh) | 1.68e-11 | **0 / 2** |

Run twice, **the port is not bitwise reproducible against itself** on this
comparison: `6.102252e-11` vs `6.102218e-11` (`20-CONTEXT.md:152`), consistent
with parallel reduction order. `12-VERIFICATION.md:137-153` independently
records that one f64 ulp at `|E| ≈ 7.79` is 8.88e-16, so a 1e-15 gate is ~1.1
ulp, and that the best result anywhere in the port is **221 ulp**.

**The rule.** `to_bits()` / bit-identical assertions are for
**same-implementation A/B only** — an env kill switch, a route swap, a thread
count, incore vs spilled (e.g. `ksymm_band_ao_reuse.rs`; 17-08's
`max |dvj| = 0e0`; Phase-16 G8/G11). Never against upstream. No gate written in
this phase may be tighter than one f64 ulp of the quantity it bounds.

---

## §4 — The Phase-20 identity gate, and the evidence it discriminates (D-20-C)

The original Phase-20 gate — *"an unmodified upstream `pyscf.pbc` script runs on
pyscf-rs"* — is TRUE today with zero Phase-20 work, because
`python/pyscf/pbc/__init__.py` is a `pkgutil.extend_path` passthrough and every
`pyscf.pbc.*` name resolves to the vendored upstream tree (`20-CONTEXT.md §1.1`).
A gate a no-op passes is not a gate.

**Restated (20-01; enforced by 20-18):** each public name the overlay exports is
the SAME object as its native binding,

```python
assert pyscf.pbc.scf.KRHF is pyscf._native.pbc.scf.KRHF
```

for the 18 names in `python/pyscf/tests/test_pbc_identity_gate.py` (`Cell`, `M`;
`FFTDF`, `AFTDF`, `GDF`, `MDF`, `RSDF`; `KRHF`, `KUHF`, `KROHF`, `KGHF`; `KRKS`,
`KUKS`, `KROKS`, `KGKS`; `KPoints`; `KMP2`; `KRCCSD`), **plus** the unmodified
upstream scripts `examples/pbc/20-k_points_scf.py` and
`examples/pbc/22-k_points_mp2.py` running with that identity holding, their
numbers landing on §1 and `which_impl` reporting `"native"` (20-18 Task 2).

**Discriminating evidence** — [`identity-gate-pre-20-09.out`](identity-gate-pre-20-09.out).
`test_pbc_identity_gate.py` was run on HEAD `0f8b58d` with the stale 2026-08-20
`_native.abi3.so`, before any binding existed (`20-EXECUTION-NOTES.md` D-20-C).
Its summary line is `18 failed in 0.05s`; every case fails with
`ModuleNotFoundError: No module named 'pyscf._native.pbc'; 'pyscf._native' is
not a package`. This is 20-18 Task 1's "fails on the pre-20-09 tree" half,
captured without a checkout. The passing half is owed by 20-18.

---

## §5 — Phases 18-19: the gates as their own phases restated them

Added at execution: 20-CONTEXT (2026-09-12) predates Phase 19's closure and
Phase 18's start (`20-EXECUTION-NOTES.md §1`).

**Phase 18 — IN PROGRESS, paused** (`18-periodic-gradients-stress-geomopt/18-IMPLEMENTATION-CHECKPOINT.md:11`, `:52`).
Gates from `18-CONTEXT.md §2.3` (`:349-396`); no Rust method gate has run yet
(`18-.../measurements/README.md:78`).

| gate | statement | upstream floor measured by 18-01 | source |
|---|---|---|---|
| A1 / A2 / A3 | component strain derivatives vs own FD: **1e-9** (13 tests) / **2e-9** (2) / **1e-8** (`test_get_pp`) | raw upstream `test_get_vxc_lda` residual **1.0564055741291156e-9** against A1's 1e-9 — diagnosed as FD energy-reduction roundoff; passes in the stable-summation reference mode, raw mode still fails and remains the default | `18-CONTEXT.md:362-366`; `18-IMPLEMENTATION-CHECKPOINT.md:3-7`, `:53-54` |
| B | analytic gradient vs `verify_fd`: **1e-6 Ha/Bohr** (`FD_TOL`); **5e-6** for `krkspu`/`kukspu` | central-difference minimum: KRHF **5.278303349953717e-10** at full h=1e-5; PBE **4.3241903113777624e-10** at full h=1e-4 | `18-CONTEXT.md:375-379`; `18-.../measurements/README.md:68-70` |
| C | analytic gradient vs upstream `lib.fp(g)` at upstream's decimal count (6; 5 for DFT+U) | HF residuals ~1.72e-8, DFT ~1.5e-9, DFT+U ~8.32e-9 vs committed constants | `18-CONTEXT.md:380-385`; `18-.../measurements/README.md:29-30` |
| D | stress vs FD of `E(ε)`: **1e-6 Ha/Bohr³** (pressure units, `/vol`) | pending 18-17/18-21 | `18-CONTEXT.md:386-389` |
| E | gamma-point multigrid-v2 bodies gated against their own measured floor, not B/C | pending | `18-CONTEXT.md:390-396` |

**Phase 19 — CLOSED 2026-09-13** (`19-periodic-response-relativistic/19-VERIFICATION.md:1`).
A1 HF-family roots **1e-5**; A2 KS-family roots **1e-8 Ha**; B single-CPHF seam
(structural); C G0W0 QP **1e-4 per route**; D ADC IP/EA **1e-4**; E X2C
**1e-8 Ha** — A1/A2/B/D/E MET; C MET for AC live and for CD/UAC/slow on
analytic arms, the three live-vs-upstream GW arms **NOT RUN**
(`19-VERIFICATION.md` gate matrix; `.planning/carryovers/19-gw-live-oracle.md`).
