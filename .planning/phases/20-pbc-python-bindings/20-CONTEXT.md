# Phase 20 — PBC Python bindings + oracle enforcement — CONTEXT

**Written:** 2026-09-12, before any Phase-20 code.
**Read this before `20-01-PLAN.md`.** Every claim below was verified against the
vendored PySCF **2.12.1** tree, the current Rust workspace, `.github/workflows/ci.yml`
and the `.planning` tree on 2026-09-12, and carries the file and line that proves it.

`PBC-MASTER-PLAN §8.12` sizes this phase at **eight plans** (20-01 … 20-08) and
puts the whole PyO3 surface in a single row, **20-05**. That row is the one this
document mostly replaces. **The eight-plan table is wrong about the starting
state in nine ways, its gate is satisfied vacuously today, and one of its eight
plans cannot be written at all because the Rust it would bind does not exist.**

---

## 1. The scope corrections, in order of consequence

### 1.1 The Phase-20 gate is ALREADY TRUE, vacuously, and must be restated

`ROADMAP.md:466` and `§7` row 20 both state the gate as:

> **Gate:** an unmodified upstream `pyscf.pbc` script runs on pyscf-rs.

That is **true right now, with zero Phase-20 work done**, and it proves nothing.
`python/pyscf/pbc/__init__.py` is a `pkgutil.extend_path` passthrough with
`__all__ = []`, and `pyscf/__init__.py` extends the path too, so
`import pyscf.pbc.scf` resolves to the **vendored upstream Python tree**:

```
pyscf.pbc      = /home/user/Documents/workspace/pyscf_rs/pyscf/pbc/__init__.py
pyscf.pbc.scf  = /home/user/Documents/workspace/pyscf_rs/pyscf/pbc/scf/__init__.py
```

(measured 2026-09-12). Every upstream `pyscf.pbc` script already runs — on
upstream's own pure-Python implementation. A gate that a no-op passes is not a
gate. **20-12 restates it** as the Phase-3 identity contract already used for
the molecular surface (`python/pyscf/tests/test_overlay_resolution.py:20`):

```python
assert pbc_scf.KRHF is pyscf._native.pbc.scf.KRHF
```

This is the single most important correction in this document. It is the same
defect class this project has now found **seven** times (Phases 14, 15, 16, 17,
18, the ROADMAP rows in §1.2, and here) — a gate written before it was measured.

### 1.2 The ROADMAP is stale for exactly the five phases that shipped

`*-VERIFICATION.md` exists for phases 13, 14, 15, 16 and 17, yet
`ROADMAP.md:459-463` still shows all five as `[ ]` unchecked (9–12 are `[x]`).
The same five rows still carry gates their own phases measured and superseded:

| ROADMAP row | states | the phase actually measured |
|---|---|---|
| `:461` Phase 15 | `KMP2 e_corr` to **1e-14** | `2e-6`, MET 2026-09-05 (`15-VERIFICATION §1a`) |
| `:462` Phase 16 | `KRCCSD e_corr` to **1e-14** | **1e-7 per DF route** (`16 measurements/README.md §1`) |
| `:463` Phase 17 | symmetry == full-BZ to **1e-14** | five gates; C FFTDF `1.703e-11` (`17-VERIFICATION`) |
| `:464` Phase 18 | `verify_fd` to **1e-15 Ha/Bohr** | below the central-difference floor (`18-CONTEXT §2.1`) |
| `:465` Phase 19 | TDA excitation to **1e-15 eV** | unmeasured; Phase 19 does not exist |

`12-VERIFICATION` already recorded why every one of these is unreachable: one
f64 ulp at `|E| ≈ 7.79` is `8.88e-16`, so a 1e-15 gate is ~1.1 ulp, **and the
best result anywhere in this port is 221 ulp**. 20-01 fixes these rows.

### 1.3 `PBC-DRIVER-INVENTORY.md`'s status column is not maintained

`0` rows marked `[x]`, `0` marked `[~]`, **167 marked `[ ]`** — after nine PBC
phases shipped. It is an accurate *scope map* (186 modules, ~78,000 lines) and a
useless *status source*. Nothing in this phase may cite it as status.

### 1.4 `STATE.md`'s counters contradict its own prose

Front-matter says `total_phases: 5, completed_phases: 5, total_plans: 48,
completed_plans: 48, percent: 100` while the body says Phase 17 closed and
**Phase 16 remains IN PROGRESS**, and the tree holds nine PBC phase directories.
20-01 reconciles the counters.

### 1.5 There is ZERO CI coverage of PBC

`grep -c pbc .github/workflows/ci.yml` → **0**. Not one job mentions it. And
**161 PBC gates are `#[ignore]`d**:

| crate | ignored | crate | ignored |
|---|---:|---|---:|
| `pyscf-pbc-cc` | 59 | `pyscf-pbc-scf` | 16 |
| `pyscf-pbc-dft` | 24 | `pyscf-pbc-mp` | 13 |
| `pyscf-pbc-gto` | 24 | `pyscf-pbc-ci` | 1 |
| `pyscf-pbc-df` | 22 | `pyscf-pbc-symm` | 1 |
| | | `pyscf-pbc-tools` | 1 |

~88 of them are ignored **solely** because they need `PYSCF_ORACLE_VENV`. So
every measured floor in phases 9–17 is enforced by nothing automated. **This,
not bitwise arithmetic, is what "fix the bit-exact issues" actually means here**
— see §2.

### 1.6 Two oracle entry points float to the WRONG upstream

`tests/oracle/requirements.txt:14` pins `pyscf>=2.6` and the
`*-oracle-upstream-manual` CI jobs `pip install "pyscf>=2.5"`
(`ci.yml:542`). Both resolve to **2.14.0**. PySCF 2.14 rewrote
`fft_jk.get_k_kpts` to fold `exxdiv='ewald'` into `get_coulG`, moving K by
**~1e-5** — the trap Phase 11 pinned and that **47 Rust PBC test files** assert
against by requiring `version == "2.12.1"`. CI *does* pin `pyscf==2.12.1`
correctly in three places (`ci.yml:284`, `:326`, `:328`); the two floating pins
are the ones a new PBC oracle job would most plausibly be copied from.

### 1.7 Eight of the twenty upstream module families cannot be bound at all

`pyscf-pbc-{grad, geomopt, tdscf, gw, adc, x2c, eph, mpi}` are **13-line stubs**
exporting one error enum each (`src/lib.rs` 6 lines + `src/error.rs` 7). Phase 18
is *planned but never executed* — 15 PLAN files, **zero** SUMMARY files, no
VERIFICATION. Phase 19 has no directory at all.

`§8.12`'s **20-04** (`mpicc`, 4,976 lines) binds `pyscf-pbc-mpi`, which is a
stub. **That plan cannot be written and is dropped from this phase.**

And because of the §1.1 passthrough, an unported module does **not** raise — it
silently serves upstream Python. A user calling `pyscf.pbc.grad` today gets
upstream's implementation with none of this port's performance and none of its
gates. 20-11 makes that a deliberate, documented policy instead of an accident.

### 1.8 `Cell::get_hcore` always refuses — hcore belongs to the DF object

`pyscf-pbc-gto/src/hcore.rs:181` refuses unconditionally (`phase: 11`); the real
entry point is `pyscf_pbc_df::get_hcore(&dyn PeriodicDf, kpts)`
(`pyscf-pbc-df/src/fftdf.rs:458`). A binding that mirrors upstream's
`cell.get_hcore()` shape naively will bind a permanent refusal.

### 1.9 `pyscf-pbc-dft` is the only binding target that drags in libxc

`crates/pyscf-pbc-dft/Cargo.toml:14` takes `pyscf-dft` with **default features
on**, and `pyscf-dft`'s default is `["libxc"]`. Binding `pbc.dft` therefore pulls
the 266-kernel libxc compile into the wheel — the same cost that keeps
`dft-libxc-bitexact` hard-disabled at `ci.yml:418` (`if: false`, ~6h). 20-08
decides this explicitly rather than discovering it in a wheel build.

---

## 2. "Fix all bit-exact issues" — what is and is not achievable

**Bitwise equality with upstream is not achievable and must not be planned for.**
Measured this session on the cheapest periodic fixture that exists (He-fcc
`sto-3g`, ONE AO, all-electron, 2×2×2, mesh `[15,15,15]`), against vendored
2.12.1:

| quantity | agreement | bitwise identical |
|---|---|---|
| `e_tot` | 2.17e-13 | — |
| `mo_energy` (on mesh) | 6.10e-11 | **0 / 8** |
| `get_bands` (off mesh) | 1.68e-11 | **0 / 2** |

Run twice, the port is **not bitwise reproducible against itself** (`6.102252e-11`
vs `6.102218e-11`), consistent with parallel reduction order. `12-VERIFICATION`
independently reached the same conclusion (221 ulp best-in-port). Bitwise
equality survives only **same-implementation A/B** comparisons, which this repo
already reserves it for (`ksymm_band_ao_reuse.rs`, 17-08's `max|dvj| = 0e0`).

So this phase splits the request into three things that ARE achievable:

**(a) Enforce the floors that are already measured.** 161 ignored gates, zero CI.
→ 20-02.

**(b) Close the genuinely-unmet numeric gates.** Only three qualify:

| item | measured | status |
|---|---|---|
| `KRKS` ksymm vs full BZ, **GDF** | **1.432e-06** vs a 1.8e-11…1.6e-10 target | **NOT MET**, `17-VERIFICATION:126` |
| `rsjk` build/get_jk | always refuses | D-PBC-24 carryover, blocked on Phase 17 supermole |
| `GDF/MDF.get_jk(omega)` | refuses | `gdf/jk.rs:673`, `mdf/mdf_jk.rs:132` |

→ 20-03.

**(c) Do NOT chase the irreducible ones.** These are properties of the method,
present in upstream too, and no implementation reaches them:

| gap | size | why it is not a bug |
|---|---|---|
| FFTDF − GDF | **1.222e-03** | the DF **fitting error** (`14 measurements:121`) |
| upstream's own two GDF builders | **4.502e-06** | `_RSGDFBuilder` vs `_CCGDFBuilder` (`:139`) |
| FFTDF/MDF vs GDF/RSDF in CC | **9.22e-04 Ha** | upstream's own route split (`16 measurements §3`) |
| `pp_int` vs `_IntPPBuilder` `get_pp` | **1.79e-09** | two upstream routes disagreeing (`13-VERIFICATION`) |
| band energies vs upstream | **~1.7e-11** | cross-implementation summation order (§2 above) |

A plan that lists these as defects to fix would be chasing upstream's own
numbers and would never close.

---

## 3. Ground truth — what exists to bind (verified 2026-09-12)

**Bindable now (~83,000 lines of Rust):**

| crate | src lines | key entry points |
|---|---:|---|
| `pyscf-pbc-cc` | 29,147 | `Krccsd`/`Kuccsd`/`Kgccsd`, `(T)`, EOM IP/EA/EE, `KsymEris` |
| `pyscf-pbc-df` | 15,255 | `Fftdf`/`Aftdf`/`Gdf`/`Mdf`/`Rsdf` behind `trait PeriodicDf` |
| `pyscf-pbc-dft` | 11,987 | `Krks`/`Kuks`/`Kroks`/`Kgks` + 4 ksymm + DFT+U + 3 numint backends |
| `pyscf-pbc-gto` | 8,775 | `Cell`, `M`, `make_kpts`, `band_path`, `super_cell` |
| `pyscf-pbc-symm` | 7,195 | `KPoints` IBZ machinery |
| `pyscf-pbc-scf` | 4,350 | `Krhf`/`Kuhf`/`Krohf`/`Kghf`, `KsymAdaptedKrhf`, smearing, chkfile |
| `pyscf-pbc-mp` | 2,771 | `Kmp2`/`Kump2`/`Kmp2Stagger`/`KsymAdaptedKmp2` |
| `pyscf-pbc-tools` | 2,108 | `fft`/`ifft`, `get_coulg`, `ExxDiv` |
| `pyscf-pbc-lib` | 868 | `get_kconserv`, `kpts_helper` |
| `pyscf-pbc-ci` | 433 | `kernel_at_kshift` (KCIS) |
| `pyscf-pbc-ao2mo` | 160 | thin wrappers over `pyscf-pbc-df` |

**Two seams make this tractable, and both already have molecular precedent:**

1. **`trait PeriodicDf`** (`pyscf-pbc-df/src/traits.rs:102`) is object-safe and
   every SCF/KS driver stores `Box<dyn PeriodicDf>` via `from_df`. **One**
   `PyPeriodicDf` wrapper therefore covers all five builders.
2. **`trait KOverrideHooks`** (`pyscf-pbc-scf/src/khooks.rs:21`) is the exact
   k-point analogue of the molecular `OverrideHooks` that
   `crates/pyscf-py/src/bridge.rs:74` already implements via
   `slf.call_method1`. The subclass-override contract ports 1:1.

**The one genuinely new piece** is the complex, k-resolved NumPy boundary.
`numpy_io.rs` today handles only real `Density`/`MOCoefficients`
(`to_density:22`, `density_to_pyarray:91`). PBC needs `CTensor { re, im }`
(planar, D-PBC-02) ↔ `Complex64`, and per-k *lists* (`KMats = Vec<CTensor>`).
**The precedent exists**: `crates/pyscf-py/src/gto.rs:98-111` already interleaves
planar `re`/`im` into `numpy::Complex64` with correct **F-order**
(`IxDyn(&shape).f()`), and `num-complex 0.4.6` is already a dep in five PBC
crates. 20-04 generalises that one function; it is not greenfield.

**Starting state of the binding crate:** `crates/pyscf-py/src/` has
`gto.rs, scf.rs, dft.rs, cc.rs, mp.rs, grad.rs, geomopt.rs, bridge.rs,
caches.rs, numpy_io.rs, errors.rs, lib.rs` and **no `pbc.rs`**; `lib.rs:70-114`
registers seven **flat** submodules. PBC needs **nested** ones
(`_native.pbc.gto`, `.scf`, `.dft`, …). Zero `pbc` references exist in
`crates/pyscf-py/` today.

---

## 4. Walls this phase must not breach

| wall | enforced by | status for this phase |
|---|---|---|
| ALG-06: only algebra/runtime/kernels/bench name `cubecl-*` | `xtask check-dependency-wall` (direct deps only) | Safe — cubecl is transitive in all 19 PBC crates via `pyscf-algebra`, none name it directly |
| D-PBC-14: every `pyscf-pbc-*` crate is pyo3-free | **NOTHING — plan text only** | **Gap.** Zero PBC crate names pyo3 today; 20-10 makes it machine-checked |
| FMA / reduction determinism | `xtask check-no-fma`, `[profile.release-oracle]` | Applies unchanged |
| No `lazy_static` in `pyscf-py` | `xtask check-forbid-lazy-static` | Use `PyOnceLock` (`caches.rs:28`) |
| Panic must not cross FFI | `xtask check-catch-unwind` | Applies to every new `#[pymethods]` |

---

## 5. The plan set (18 plans, replacing `§8.12`'s eight)

Sized for a **Claude Sonnet** executor — see §6. One concern per plan, one
bounded `<context>` list, a binary command in every `<verification>`.

| Plan | Type | Wave | Content |
|---|---|---|---|
| **20-01** | measure | 0 | Reconcile the gates: ROADMAP 459-466, `STATE.md` counters, restate the Phase-20 gate non-vacuously. **No Rust.** |
| **20-02** | measure | 0 | Tier the 161 ignored gates by runtime; fix the two floating upstream pins (§1.6) |
| **20-03** | execute | 1 | Add the `pbc-oracle` CI job (the first job to touch PBC at all) |
| **20-04** | measure | 1 | Bisect the GDF ksymm defect **by route**; name the failing step. No fix. |
| **20-05** | implement | 2 | `GDF/MDF.get_jk(omega)` — delete the two refusals |
| **20-06** | execute | 2 | Re-assess `rsjk`: implement, or update the D-PBC-24 carryover |
| **20-07** | implement | 1 | Complex k-resolved NumPy boundary in `numpy_io.rs` |
| **20-08** | execute | 1 | Nested `_native.pbc.*` module registration (empty children) |
| **20-09** | implement | 2 | `pbc.gto` — `PyCell`, `M`, `make_kpts`, `band_path`, `super_cell` |
| **20-10** | implement | 3 | `pbc.df` — one `PyPeriodicDf` over five builders + `density_fit` |
| **20-11** | implement | 3 | `KPyOverrideBridge` — the k-point override contract, alone |
| **20-12** | implement | 4 | `pbc.scf` — the four drivers + ksymm + smearing + chkfile + `get_bands` |
| **20-13** | implement | 5 | `pbc.dft` — 4 KS + 4 ksymm + DFT+U (libxc posture has a **default**, §1.9) |
| **20-14** | implement | 5 | `pbc.symm` / `pbc.lib` / `pbc.tools` |
| **20-15** | implement | 5 | `pbc.mp` / `pbc.cc` / `pbc.ci` / `pbc.ao2mo` |
| **20-16** | execute | 5 | Extend the dependency wall to forbid `pyo3` outside `pyscf-py` (§4) |
| **20-17** | execute | 6 | Overlay shims + the fallthrough policy (**default: announced**, §1.7) |
| **20-18** | verify | 7 | Identity-contract gate (§1.1), upstream suite, determinism, rollup |

**Dropped from `§8.12`:** its **20-04** (`mpicc`) — binds a stub crate (§1.7).
`k2gamma`/`lattice`/`pyscf_ase` (its 20-01/20-02) are **not** bindings work and
belong to a Rust porting phase; 20-17 records them as unported rather than
silently dropping them.

---

## 6. Executing this phase with a Sonnet-class agent

Every plan in this set obeys six rules. They exist because a smaller-context
executor fails in predictable ways, and each rule removes one of them.

1. **Bounded reading.** The `<context>` block is the *complete* list of files to
   read. No plan requires exploring the tree to find its own inputs. Where a
   fact matters, the plan cites `file:line` inline so it need not be re-derived.
2. **One concern per plan.** No plan touches more than ~2 production files. The
   three biggest units were split: oracle tiering from the CI job, the override
   bridge from the SCF drivers, and the `mp`/`cc` bindings from the wall lint.
3. **Copy a named precedent, never invent.** Every binding plan names the exact
   existing function it mirrors — `bridge.rs:74-269` for hook dispatch,
   `gto.rs:98-111` for the complex interleave, `python/pyscf/scf/__init__.py:8`
   for an overlay shim.
4. **No open judgment.** Where `§8.12` would have said "decide", these plans
   state a **default** and a narrow **escalation trigger**: do the default; stop
   and report only if the named condition holds. The two such points are the
   libxc posture (20-13) and the fallthrough policy (20-17).
5. **A binary command per gate.** Every `<verification>` names the exact command
   and the exit code that means success — never "confirm it works".
6. **Traps inlined at the point they bite**, not collected in a preamble:
   - `CARGO_TARGET_DIR` must live on `/home`, never `/tmp` (a 16 GB RAM tmpfs).
   - **Name test targets explicitly.** A bare `--tests` at high parallelism has
     been OOM-killed in this workspace. Confirm `exit = 0`; process absence is
     not success.
   - Never cap memory with `ulimit -v` — it makes every CubeCL launch panic and
     produces false gate failures. Use `systemd-run --scope -p MemoryMax=`.
   - `rustfmt` only the files you touched; the tree predates the installed
     rustfmt and a repo-wide format churns unrelated files.
   - The oracle is **`pyscf==2.12.1`**, never `>=`. 47 Rust test files assert it.

**If a gate fails, report the measured number and stop.** Do not loosen a
tolerance to get a green run — that is the failure mode this whole phase exists
to correct.
