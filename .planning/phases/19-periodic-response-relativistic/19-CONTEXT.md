# Phase 19 — Periodic response + relativistic — CONTEXT

**Written:** 2026-09-12, before any Phase-19 code.
**Read this before `19-01-PLAN.md`.** Every claim was verified against the
vendored PySCF **2.12.1** tree and the current Rust workspace on 2026-09-12,
and carries the file, line or count that proves it.

`PBC-MASTER-PLAN §8.11` sizes this phase at **eight plans**. **That table is
wrong about the starting state in four ways and its gate is arithmetically
impossible**, and this document replaces it with nineteen.

---

## 1. The scope corrections, in order of consequence

### 1.1 The gate is impossible, and ten orders tighter than upstream's own tests

`ROADMAP.md:465` states:

> **Gate:** KRHF-TDA lowest excitation matches upstream to **1e-15 eV**.

A typical excitation is ~5 eV. One f64 ulp at 5 is `8.9e-16`, so **1e-15 eV is
about one ulp** — it demands bit-identical arithmetic to NumPy/LAPACK through
an SCF, a response build and a Davidson solve. `12-VERIFICATION` already
recorded that the best result anywhere in this port is **221 ulp**.

It is also ten orders tighter than upstream's own suite. Counting every
`assertAlmostEqual(..., places)` in `pyscf/pbc/tdscf/test/`:

| decimals | 10 | 8 | 7 | 5 | 4 | 3 | 2 |
|---|---|---|---|---|---|---|---|
| assertions | 6 | 10 | 2 | **15** | 7 | 1 | 1 |

The **modal** assertion is 5 decimals (~1e-5) and the tightest anywhere is 10.
The other three families are looser still:

| family | modal assertion | counts |
|---|---|---|
| `pbc/gw` | 4–5 decimals | 16 at 5, 16 at 4, 8 at 6 |
| `pbc/adc` | **4 decimals** | **58** at 4, 11 at 6, 9 at 2, 2 at 3 |
| `pbc/x2c` | 8 decimals | 29 at 8, 3 at 7, 2 at 6 |

This is the **eighth** time this project has found a gate written before it was
measured (Phases 14, 15, 16, 17, 18, the ROADMAP rows, Phase 20, here). 19-01
measures the floors and restates the gate before any of them is written.

### 1.2 Nothing exists. Five crates are 13-line stubs

`pyscf-pbc-{tdscf, gw, adc, x2c, eph}` are each **13 lines** — `src/lib.rs` (6)
plus `src/error.rs` (7) — exporting one error enum and nothing else. There is
no `todo!()` to fill in; there is no module.

The upstream surface to port is **~9,511 lines**:

| upstream | lines | Rust crate |
|---|---:|---|
| `pbc/adc` | 3,437 | `pyscf-pbc-adc` (stub) |
| `pbc/gw` | 2,504 | `pyscf-pbc-gw` (stub) |
| `pbc/tdscf` | 1,880 | `pyscf-pbc-tdscf` (stub) |
| `pbc/scf/{cphf,newton_ah,stability,_response_functions}` | 855 | **no crate — lands in `pyscf-pbc-scf`** |
| `pbc/x2c` | 654 | `pyscf-pbc-x2c` (stub) |
| `pbc/eph` | 181 | `pyscf-pbc-eph` (stub) |

`pbc/tddft` has **no crate at all** — upstream is a thin alias over `tdscf`.

### 1.3 There is already ONE CPHF solver, and a CI gate that forbids a second

`§8.11`'s 19-01 row says *"Reuse `pyscf-grad`'s single matrix-free Krylov CPHF
solver (GRAD-10) — one implementation only, enforced by the existing
source-scan gate."* **That is correct and it is load-bearing.**

`crates/pyscf-grad/src/cphf.rs` is that solver; its own module doc states the
rule, and `crates/pyscf-grad/tests/cphf.rs`'s `single_cphf_impl` assertion
*"guards against a second CPHF `pub fn solve` implementation landing in the
crate."* MP2 and CCSD already pass their own matrix-free `fvind` into it.

The periodic response equation is the same equation with a k-index on the
orbital-rotation space. **19-03 supplies a k-aware `fvind` and reuses
`cphf::solve`; it must not add a second solver**, and 19-02 extends the
structural gate to cover the PBC crates so that rule is machine-checked rather
than remembered.

### 1.4 `§8.11`'s eight plans are unbalanced by an order of magnitude

Its **19-07** is *"`adc/kadc_rhf` + `kadc_ao2mo` + `kadc_rhf_amplitudes` + IP
(1061 l) + EA (1324 l) + `dfadc`"* — **3,437 lines in one plan**, while its
19-06 is 328. A plan set whose largest member is ten times its smallest cannot
be scheduled, and cannot be executed by a bounded-context agent at all.

### 1.5 Phase 19 does NOT depend on Phase 18

Worth stating because the phase numbers suggest otherwise. The CPHF solver
Phase 19 reuses is **Phase 7's** (`crates/pyscf-grad/src/cphf.rs`, molecular,
shipped). Nothing in `pbc/tdscf`, `pbc/gw`, `pbc/adc`, `pbc/x2c` or `pbc/eph`
consumes a nuclear gradient. Phase 19 can start while Phase 18 is unfinished —
and Phase 18 currently has **zero SUMMARY files**, so it is planned but never
executed.

What Phase 19 *does* depend on is already shipped: `KRHF`/`KRKS` (Phases 11/12),
the periodic AO2MO + `KMP2` machinery (Phase 15), and `KPoints` (Phase 17).

### 1.6 `gen_response` is `NotImplemented` on the PBC Kohn-Sham base

`pyscf/pbc/dft/rks.py:268` sets `gen_response = NotImplemented` on the PBC
`KohnShamDFT` base, and only the concrete `RKS` class rebinds it (`:411`, from
the module-level definition at `:146`). Every consumer in this phase —
`newton_ah`, `stability`, TDA/TDHF — goes through that seam, so the *absence*
on the base class is part of the contract to port, not a bug to fix. A binding
that supplies `gen_response` universally would diverge from upstream silently.

### 1.7 Four of the five stub crates drag in libxc

`pyscf-pbc-{adc, gw, tdscf, eph}` depend on `pyscf-pbc-dft`, which takes
`pyscf-dft` with default features, whose default is `["libxc"]` — the
266-kernel, ~6h compile that keeps `ci.yml:418` disabled at `if: false`. Only
`pyscf-pbc-x2c` is free of it. 19-02 decides the feature posture **once**, for
all four, rather than four plans discovering it separately.

---

## 2. The gate: measured, per family, before it is written

Following 17-01's and 18-01's ruling exactly — *measure the floor, then write
the gate*. 19-01 measures and 19-19 enforces:

* **Gate A — TDA/TDHF excitation energies vs upstream**, per method
  (`krhf`, `kuhf`, `rhf`, `uhf`, and the four KS subclasses), at the decimal
  count **upstream's own test asserts for that method**, never a phase-wide
  number. The modal upstream assertion is 5 decimals; the tightest is 10.
* **Gate B — the response seam is exercised, not bypassed.** A structural
  assertion that the periodic CPHF path enters `pyscf_grad::cphf::solve` and
  that no second `pub fn solve` exists anywhere in the workspace (§1.3).
* **Gate C — G0W0 quasiparticle energies vs upstream**, at upstream's own 4–5
  decimals, **per route** (AC vs CD are different approximations and must not
  share a number).
* **Gate D — ADC IP/EA roots vs upstream**, at upstream's own **4 decimals**
  (58 of its 80 assertions).
* **Gate E — X2C one-electron energies vs upstream**, at its own 8 decimals —
  the tightest family in the phase, and the one with no iterative solver.

`ROADMAP.md:465` and `PBC-MASTER-PLAN §7`'s Phase-19 row are both restated by
19-01, in one edit, with the measured numbers.

---

## 3. The plan set: nineteen plans, eight waves

| plan | content | blocked on |
|---|---|---|
| **19-01** | **MEASURE** — Gates A–E floors from upstream's own suites; restate `ROADMAP:465` and `§8.11`. **No Rust.** | — |
| **19-02** | Substrate: the five crate surfaces, the libxc posture (§1.7), and the GRAD-10 structural gate extended to the PBC crates | — |
| **19-03** | `scf/cphf` (176 l) + `_response_functions` (47 l) — the k-aware `fvind` over the **existing** solver (§1.3) | 19-02 |
| **19-04** | `scf/newton_ah` (303 l) — second-order SCF | 19-03 |
| **19-05** | `scf/stability` (329 l) | 19-03 |
| **19-06** | `tdscf/rhf` (238 l) — gamma TDA/TDHF, the smallest complete method | 19-03 |
| **19-07** | `tdscf/krhf` (537 l) — k-point TDA/TDHF | 19-06 |
| **19-08** | `tdscf/uhf` (268 l) + `tdscf/kuhf` (540 l) | 19-07 |
| **19-09** | `tdscf/{rks,uks,krks,kuks}` (205 l) — the KS subclasses over §1.6's seam | 19-08 |
| **19-10** | `gw/krgw_ac` (644 l) — analytic-continuation G0W0 | 19-07 |
| **19-11** | `gw/krgw_cd` (704 l) — contour deformation | 19-10 |
| **19-12** | `gw/kugw_ac` (784 l) — unrestricted AC | 19-10 |
| **19-13** | `gw/{kgw_slow, kgw_slow_supercell, gw_slow}` (328 l) | 19-10 |
| **19-14** | `adc/kadc_ao2mo` (294 l) + `kadc_rhf` (326 l) + `kadc_rhf_amplitudes` (346 l) — the ADC base | 19-02 |
| **19-15** | `adc/kadc_rhf_ip` (1061 l) | 19-14 |
| **19-16** | `adc/kadc_rhf_ea` (1324 l) | 19-14 |
| **19-17** | `adc/dfadc` (62 l) + the ADC gate D rollup | 19-15, 19-16 |
| **19-18** | `x2c/sfx2c1e` (355 l) + `x2c/x2c1e` (286 l) + `eph/eph_fd` (181 l) | 19-02 |
| **19-19** | Verification rollup: Gates A–E, `FEATURES`, `STATE.md`, `ROADMAP` | all |

**Waves — eight, every dependency strictly earlier.**
W0 `19-01 19-02` · W1 `19-03 19-14 19-18` · W2 `19-04 19-05 19-06 19-15 19-16` ·
W3 `19-07 19-17` · W4 `19-08 19-10` · W5 `19-09 19-11 19-12 19-13` · W6 `19-19`.

**The droppable half is ADC** (19-14 … 19-17, 3,437 lines — the largest family
and the loosest-gated at 4 decimals), **and the trigger is explicit**: drop it
only when *both* (a) 19-01 has closed so the gates are written, and (b) the
wave-4 boundary is reached with Gate A still unmet for `krhf`. Do **not** drop
`tdscf` instead — `pbc/tddft` aliases it and `gw` consumes its response seam,
so dropping `tdscf` breaks two other families.

**Deliberate non-ports:** `pbc/tddft` as a separate module (upstream is an
alias over `tdscf`, §1.2), and multi-rank MPI for any of these (Phase 20's
`mpi` feature, default OFF, D-PBC-18).

---

## 4. Executing this phase with a Sonnet-class agent

Nineteen plans, six rules. Each removes a way a smaller-context executor
predictably fails.

1. **Bounded reading.** Every plan's `<context>` block is the complete input
   list; no plan explores the tree to find its own inputs.
2. **One concern per plan.** `§8.11`'s 3,437-line ADC plan became four
   (19-14 … 19-17); its 1,880-line tdscf pair became four (19-06 … 19-09); its
   2,504-line GW row became four (19-10 … 19-13). No plan exceeds ~1,300 lines
   of upstream source, and most are under 700.
3. **Copy a named precedent.** `crates/pyscf-grad/src/cphf.rs` for the response
   solve, `crates/pyscf-pbc-mp/src/kmp2.rs` for the k-resolved post-SCF driver
   shape, `17-01`'s `measurements/README.md` for the measurement format.
4. **No open judgment.** The libxc posture is decided once in **19-02**; the
   ADC drop has the two-condition trigger in §3; every gate tolerance comes
   from **19-01**'s measurement, never from the executor's choice.
5. **A binary command per gate.** Every `<verification>` names the command and
   what counts as success, with the test target **named** — a bare `--tests` at
   high parallelism has been OOM-killed in this workspace, and process absence
   is not success.
6. **Traps, inlined where they bite.**
   - `CARGO_TARGET_DIR` on `/home`, never `/tmp` (a 16 GB RAM tmpfs).
   - Never `ulimit -v` — it makes every CubeCL launch panic. Use
     `systemd-run --scope -p MemoryMax=`.
   - The oracle is **`pyscf==2.12.1`** exactly; 47 Rust test files assert it.
   - `rustfmt` only the files you touched.
   - Davidson roots are **order-sensitive**: compare sorted eigenvalues with an
     explicit root count, never positionally — upstream's own `nroots` spread
     reached `5.11e-7` in Phase 16.

**If a gate fails, report the measured number and stop.** Do not loosen a
tolerance to get a green run — an unmeasured gate is the defect §1.1 exists to
correct.
