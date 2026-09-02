# Phase 16 — Periodic Coupled Cluster + CI — CONTEXT

**Written:** 2026-09-02, before any Phase-16 code.
**Read this before `16-01-PLAN.md`.** Everything here was verified against the
vendored PySCF **2.12.1** tree (`pyscf/__init__.py:38`) and the current Rust
workspace on 2026-09-02; every claim carries the file and line that proves it.

`PBC-MASTER-PLAN.md §8.8` sizes this phase at **ten plans** and calls it "the
largest phase by line count (13,675 + 852 lines upstream)". **The line count is
exactly right** — `wc -l pyscf/pbc/cc/*.py` = 13,675 and `pyscf/pbc/ci/*.py` =
852, and the per-plan table accounts for 12,276 of the 13,675, the remaining
1,399 being `kccsd_rhf_ksymm.py` (806) + `kintermediates_rhf_ksymm.py` (265),
which are Phase 17's 17-09 by design, plus `kccsd_t_rhf_slow.py` (271) and
`__init__.py` (57).

**But the ten-plan table is wrong about the starting state in six ways, and its
gate is the fourth unmeasured, self-contradictory gate this project has
found.** This document records what is actually there, what is actually
missing, and what cannot be believed as written.

---

## 1. The scope corrections, in order of consequence

### 1.1 Phase 16 is HARD-BLOCKED on Phase 15, which has shipped no code

**Every** k-point CC and CI module imports the padding surface from
`pyscf.pbc.mp.kmp2` / `kump2`:

```
pyscf/pbc/cc/kccsd_rhf.py:31       from pyscf.pbc.mp.kmp2 import (get_frozen_mask, get_nocc, get_nmo,
pyscf/pbc/cc/kccsd.py:29           from pyscf.pbc.mp.kmp2 import (get_frozen_mask, get_nmo, get_nocc,
pyscf/pbc/cc/kccsd_uhf.py:36       from pyscf.pbc.mp.kump2 import (get_frozen_mask, get_nocc, get_nmo,
pyscf/pbc/cc/kccsd_t.py:31         from pyscf.pbc.mp.kmp2 import (get_frozen_mask, get_nocc, get_nmo,
pyscf/pbc/cc/kccsd_t_rhf.py:32     from pyscf.pbc.mp.kmp2 import (get_frozen_mask, get_nocc, get_nmo,
pyscf/pbc/cc/eom_kccsd_rhf.py:30   from pyscf.pbc.mp.kmp2 import (get_frozen_mask, get_nocc, get_nmo,
pyscf/pbc/cc/eom_kccsd_ghf.py:35   from pyscf.pbc.mp.kmp2 import (get_frozen_mask, get_nocc, get_nmo,
pyscf/pbc/cc/eom_kccsd_uhf.py:34   from pyscf.pbc.mp.kump2 import (get_frozen_mask, get_nocc, get_nmo,
pyscf/pbc/ci/kcis_rhf.py:31        from pyscf.pbc.mp.kmp2 import (get_nocc, get_nmo, padding_k_idx,
```

The continuation lines are `padding_k_idx, padded_mo_coeff, padded_mo_energy`.
`crates/pyscf-pbc-mp/` is **13 lines** (`src/lib.rs` 6, `src/error.rs` 7) — a
stub. `crates/pyscf-pbc-ao2mo/` is **13 lines**. Phase 15 is planned-only
(eight plans, no code).

**This is the same block that stopped 17-09**, which per its own
`autonomous: false` must-haves deferred rather than guessed. Phase 16 must not
start plan 16-04 (the first `kintermediates_rhf` plan) until 15-03's
`padding_k_idx` / `padded_mo_coeff` / `padded_mo_energy` and 15-02's
`PeriodicDf` ao2mo dispatch exist. **16-01, 16-02 and 16-03 have no such
dependency and are the phase's wave 0.**

Do not "temporarily" reimplement the padding surface inside `pyscf-pbc-cc`.
`padding_k_idx`'s convention is occupied-bottom-aligned and **virtual-TOP**-
aligned (`kmp2.py:262-263`, recorded as `15-CONTEXT §3.4`); two independent
implementations of it is precisely how a plausible wrong number ships.

### 1.2 The molecular prerequisites are real, they do not exist, and §8.8 counts none of them

`§8.8`'s "Port from" column lists only `pyscf/pbc/cc/*`. Three of its ten plans
inherit from **molecular** modules this port has never written:

| upstream base | used by | Rust state |
|---|---|---|
| `pyscf/cc/gccsd.py` (550 l) | `kccsd.py:332` `class GCCSD(gccsd.GCCSD)`, `:339`, `:352`, `:477` `gccsd._PhysicistsERIs()` | **absent** — `crates/pyscf-ccsd/` has `ccsd`, `uccsd`, `lambda`, `ulambda`, `rdm`, `urdm`, `dfccsd`, `direct`, no `gccsd` |
| `pyscf/cc/rccsd.py` (432 l) | `pbc/cc/ccsd.py:24` `class RCCSD(rccsd.RCCSD)`; `pbc/ci/cisd.py:33` `rccsd._make_eris_incore` | **absent** (the port has `ccsd.CCSD`, not the complex-capable `rccsd.RCCSD`) |
| `pyscf/cc/eom_rccsd.py` (2,109 l) | `eom_kccsd_ghf.py:29`, `:684` `EOMIP(eom_rccsd.EOMIP)`, `:1201` `EOMEA(...)`, `:1691` `EOMEE(eom_rccsd.EOM)` | **absent** — `grep -rn "EOM" crates/ --include=*.rs` returns nothing but false positives on "geom" |
| `pyscf/lib/linalg_helper.py:741` `davidson_nosym1` + `pick_real_eigs` | `eom_kccsd_ghf.py:128`, `:1352`; `kcis_rhf.py:97` | **absent** — no Davidson of any kind in the workspace |

**Sized honestly, not by the whole file.** The k-point subclasses override
nearly everything; what they actually consume of the molecular bases is narrow
and must be ported narrowly:

* From `gccsd`: `__init__`, `dump_flags`, `_PhysicistsERIs` and
  `ccsd.CCSD.ccsd`'s kernel driver (`kccsd.py:395`). Not the molecular
  `update_amps`, which `kccsd.py:68` replaces wholesale.
* From `eom_rccsd`: the `EOM` / `EOMIP` / `EOMEA` **base-class shape** —
  `kernel`, `get_init_guess`, `gen_matvec`, the `nroots`/`koopmans` surface —
  not the molecular matvecs, which every k-point subclass replaces.
* `davidson_nosym1` is a **non-symmetric** (non-Hermitian) Davidson with a
  `pick` callback. `pyscf-algebra` exposes `eigh_gen` (symmetric) and 17-02
  added a direct `faer` `Eigen::new_from_real` for a *dense* general complex
  eigenproblem. **Neither is an iterative non-symmetric Davidson.** There is no
  shortcut here: four plans (16-09/10/11/13) are dead without it.

**This is why the phase needs 14 plans, not 10**: 16-01 (measure), 16-02
(substrate), 16-03 (Davidson) are all work `§8.8` costed at zero.

### 1.3 `pyscf-ccsd` is f64-only. Every k-point CC tensor is complex.

`§8.8`'s Reuse note says: *"`pyscf-ccsd` already owns the molecular
`WorkspacePool` tensor arena and `PYSCF_MAX_MEMORY` pre-flight refusal.
`pyscf-pbc-cc` MUST reuse both."* Both halves need correcting.

* `grep -c "Complex64\|Complex<f64>\|c64" crates/pyscf-ccsd/src/*.rs` returns
  **zero on every file**. The molecular crate is real-arithmetic throughout.
  KCCSD amplitudes are `complex128` at every k-point that is not Γ
  (`kccsd_rhf.py:553-554`, `dtype=eris.fock.dtype`).
* The arena is **not in `pyscf-ccsd`** — it is
  `crates/pyscf-runtime/src/workspace_pool.rs`, which `pyscf-ccsd` *uses*
  (`rdm.rs:32`, `lambda.rs:37`). Correcting the owner matters because Phase 16
  must change it, and changing it touches `pyscf-runtime`, a crate below the
  whole workspace.
* **The pool is f64-typed all the way down**: `shape_bytes` is
  `product * 8` (`:278-280`), `TensorBackend::InMemory(Box<[f64]>)`,
  `as_slice(&self, id) -> Result<Vec<f64>, _>` (`:397`),
  `write_slice(&self, id, data: &[f64])` (`:422`),
  `with_mut_slice(.., impl FnOnce(&mut [f64]) -> R)` (`:461`).

So "MUST reuse both" **cannot be followed literally**. What is reusable is the
*shape*: the budget ceiling, the free-list, the `InMemory | Spilled` backend
split, the HARD `MemoryLimitExceeded` refusal with no silent downgrade. What is
not reusable is the element type. See §5 for the ruling, and `16-REVIEW.md
§2` for two further defects in the pool that Phase 16 would inherit.

### 1.4 `§8.8` builds the EOM base class LAST. It is inherited by the other two.

`§8.8` orders EOM as **16-06 RHF → 16-07 UHF → 16-08 GHF**. The dependency runs
the other way:

```
pyscf/pbc/cc/eom_kccsd_rhf.py:25   from pyscf.pbc.cc import eom_kccsd_ghf as eom_kgccsd
pyscf/pbc/cc/eom_kccsd_rhf.py:378  class EOMIP(eom_kgccsd.EOMIP)
pyscf/pbc/cc/eom_kccsd_rhf.py:777  class EOMEA(eom_kgccsd.EOMEA)
pyscf/pbc/cc/eom_kccsd_rhf.py:1406 class EOMEE(eom_kgccsd.EOMEE)
pyscf/pbc/cc/eom_kccsd_uhf.py:29   from pyscf.pbc.cc import eom_kccsd_ghf as eom_kgccsd
pyscf/pbc/cc/eom_kccsd_uhf.py:458  class EOMIP(eom_kgccsd.EOMIP)
pyscf/pbc/cc/eom_kccsd_uhf.py:959  class EOMEA(eom_kgccsd.EOMEA)
```

`eom_kccsd_ghf.py` is the module that owns the k-shift machinery, the
`davidson_nosym1` driver (`:128`, `:1352`) and the `EOMIP_Ta`/`EOMEA_Ta`
variants. **It ships first here (16-09), and RHF/UHF follow (16-10, 16-11).**

### 1.5 `EOM-EE` does not exist for UHF, and exists only as a SINGLET for RHF

`ROADMAP.md`'s Phase-16 entry promises "`EOM-KCCSD` IP/EA/EE (RHF/UHF/GHF)".
Read the source:

* **UHF EE does not exist.** `eom_kccsd_uhf.py` declares `EOMIP` (`:458`) and
  `EOMEA` (`:959`) and **no `EOMEE` class at all**; its `_IMDS.make_ee`
  (`:1120`) is `raise NotImplementedError`.
* **RHF EE is singlet-only.** `EOMEESinglet` (`eom_kccsd_rhf.py:1425`) is
  complete — `kernel`, `matvec`, `vector_size`, `get_init_guess`.
  `EOMEETriplet` (`:1483`) and `EOMEESpinFlip` (`:1489`) are **shells whose
  only body is `def vector_size(...): return None`** — no matvec, no kernel.
  The parent `EOMEE.vector_size` is `raise NotImplementedError` (`:1417`), and
  `gen_matvec`'s `left=True` branch is `raise NotImplementedError` (`:1466`).
* **GHF EE exists** — `eom_kccsd_ghf.py:1691 class EOMEE(eom_rccsd.EOM)`.

**RULE 2 makes the Python authoritative.** This port ships the same surface and
the same refusals, with the upstream line numbers in the error payloads, and an
oracle-gated test asserting upstream still raises — the discipline
`15-CONTEXT §1.3` set for `KUMP2`'s absent kernel. Inventing a k-point
UHF-EE kernel here would be the one thing no oracle in the workspace could
check. **`ROADMAP.md`'s Phase-16 line is corrected by this plan set.**

### 1.6 `pbc/ci/cisd.py` is not k-point CI, and it has no base to shim

`§8.8`'s plan 16-09 is "`KCIS` + `pbc/ci/cisd`". These are unrelated:

* `kcis_rhf.py` (700 l) is a real **k-point CIS** — singles only, despite the
  phase's "CI" label — with its own Davidson (`:97`) and a dense fallback
  (`:113 np.linalg.eig`).
* `pbc/ci/cisd.py` (116 l) is a **Γ-point-only shim**: `RCISD.__init__` is
  `if abs(mf.kpt).max() > 1e-9: raise NotImplementedError` (`:24`), and the
  same at `:47` for `UCISD`. It subclasses molecular `cisd.RCISD` /
  `ucisd.UCISD` / `gcisd` (`:18`), and **this port has no molecular CI crate at
  all** — there is no `crates/pyscf-ci`, and `crates/pyscf-pbc-ci/` is a
  13-line stub.

Porting `pbc/ci/cisd.py` therefore means first porting molecular RCISD/UCISD/
GCISD, which is not in any phase of this project's roadmap. **It is deferred
explicitly with the reason recorded (16-13), not guessed and not silently
dropped.** `kcis_rhf.py` ships.

Note also `kcis_rhf.py:36 from pyscf.pbc.cc.ccsd import _adjust_occ` — KCIS
depends on the Γ-point shim's Madelung occupied-orbital shift, so 16-12 lands
before 16-13.

### 1.7 `kccsd_t_rhf.py` runs on a C kernel, and its fallback is the file §8.8 omitted

`§8.8`'s 16-05 is "`KCCSD(T)`: `kccsd_t`, `kccsd_t_rhf` (970 l)" — that is
`kccsd_t.py` (319) + `kccsd_t_rhf.py` (651). But `kccsd_t_rhf.py:236` is:

```python
drv = _ccsd.libcc.CCsd_zcontract_t3T
```

a **complex C contraction kernel** taking 24 raw data pointers
(`:229-230`) plus `mo_offset`/`slices` blocking arrays. This port has no C and
no `libcc`. The zero-C constraint is the project's core value proposition, so
the kernel is ported to Rust (or `pyscf-kernels` cubecl) — and the reference to
port it *against* is **`kccsd_t_rhf_slow.py` (271 l), the one `pbc/cc` file
`§8.8`'s table omits entirely.** It is the readable, loop-explicit form of the
same energy and it is the only way to gate the fast path without a C oracle.

---

## 2. The gate cannot be believed as written — for the fourth time

| document | Phase-16 gate |
|---|---|
| `ROADMAP.md` (Phase 16 line) | "`KRCCSD` `e_corr` matches upstream to **1e-14**" |
| `PBC-MASTER-PLAN.md §7` (gate table, row 16) | "`KRCCSD` `e_corr` matches upstream to **1e-8** on He 1×1×2" |

Six orders apart, in two documents describing the same number, **neither
measured**. This is the identical failure Phase 14 paid for once, Phase 15
caught pre-implementation (`15-CONTEXT §2`) and Phase 17 caught again
(`17-CONTEXT §2`, replaced by five measured gates).

**Upstream's own test suite is the evidence that both numbers are wrong**, and
it is unambiguous. Decimal places asserted in `pyscf/pbc/cc/test/`:

| file | `e_corr` / energy | eigenvalues | ERI fingerprints |
|---|---|---|---|
| `test_krccsd.py` | **6** (`:180`, `:226`, `:232`, `:338`, `:356`) | **3** (`:359-361`), **2** (`:364-366`) | 10 (`:239-245`), 8 (`:263-268`) |
| `test_kgccsd.py` | **6** (37 assertions) | — | 7 |
| `test_eom_krccsd.py` | **6** (44 assertions) | **3** (12 assertions) | — |
| `test_kccsd_ksymm.py` | **8** (1 assertion) | — | — |

Upstream gates its own `KRCCSD` correlation energy at **6 decimals (5e-7)** and
its own EOM roots at **3 decimals (5e-4)**. `ROADMAP`'s `1e-14` is **eight
orders tighter than upstream's own suite**; §7's `1e-8` is still two orders
tighter. A gate that demands either will fail a correct implementation.

Two more numbers from the same file worth carrying, because they are the shape
of gate that *does* work:

* `test_krccsd.py:250-256` — `eris2` vs `eris1` (incore vs outcore, **same
  code, same inputs**) at **12 decimals**. A port-vs-port storage-tier gate has
  no convergence noise in it and can be tight.
* `test_krccsd.py:478` — supercell equivalence `ercc/prod(nk)` at **4
  decimals**. Oracle-free, and it alone catches a wrong `kconserv` argument
  order or a misplaced `1/nkpts`.

Why `e_corr` cannot be tight: it is a *quadratic* functional of ERIs that this
port gates at 1.7e-12…4.2e-12, divided by orbital energies from an SCF
converged to `conv_tol`, and then **iterated to `conv_tol_normt` through a DIIS
path that is not reproducible between two independently converged runs**. The
amplitude iteration is exactly why upstream drops from `KMP2`-ksymm's 5e-11
(`mp/test/test_ksym.py:56`) to `KRCCSD`-ksymm's 6 decimals
(`test_kccsd_ksymm.py`, and `17-CONTEXT §2` already noted this shape).

**Plan 16-01 measures the floor before the gate is written** and restates it in
`ROADMAP.md`, `PBC-MASTER-PLAN §7`/`§8.8` and this file together. Do not write
the gate number first and measure second. **And the gate must name its DF
route** — `kccsd_rhf.py:37` imports `GDF, RSGDF` and branches the whole ERI
build on the mean-field's DF class, exactly as `kmp2.py:69` does, and Phase 14
measured that split at **4.5e-6 Ha** on diamond with upstream's own two routes
apart by the same amount (standing memory `rsdf-gdf-disagree-on-diamond`).

---

## 3. Traps recorded in advance, each with the line that proves it

### 3.1 `symm_map` is 4-fold, and unlike Phase 15 it is worth taking

`kpts_helper.py:572-612` — `build_symm_map` assigns each `(kp,kq,kr)` triple to
one of **four** operations (`_operation` values 0-3), and `transform_symm`
(`:614-630`) realises them as `identity`, `transpose(2,3,0,1)`,
`conj(transpose(1,0,3,2))`, `conj(transpose(3,2,1,0))`.

`15-REVIEW.md §2 D-15-R-04` correctly ruled the saving **≤2× for KMP2**,
because ops 2 and 3 map an `(ov|ov)` block to a `(vo|vo)` block that KMP2 never
asks for. **That ruling does not carry over.** KCCSD's `_ERIS` needs the full
general `(pq|rs)` block — `oooo`, `ooov`, `oovv`, `ovov`, `voov`, `vovv`,
`vvvv` (`kccsd_rhf.py:789-794`, `:819-832`) — so all four operations land
inside the set it wants, and upstream duly uses all four:
`kccsd_rhf.py:783 khelper.build_symm_map()`, `:798`/`:804` iterate
`symm_map.keys()` and its orbit lists, `:805` and `:909` call `transform_symm`.

**The saving is a genuine ~4× on the phase's single most expensive step** and
it is not mentioned anywhere in `§8.8`. It goes into 16-05 from the first
version, not as a retrofit. See `16-REVIEW.md §3` for the arithmetic.

Note `kccsd_rhf.py:512` passes `init_symm_map=False` — the map is built lazily
in `_ERIS` (`:783`), not in the constructor. Port the laziness; `build_symm_map`
is `O(nkpts³)` and a `KRCCSD` object that is constructed but never run must not
pay it.

### 3.2 `oracle_zdot` is `zdotc`. Name the primitive at every site.

`crates/pyscf-algebra/src/zoracle.rs:36 oracle_zdot` computes `Σ conj(x)·y`.
`15-REVIEW.md §2 D-15-R-02` found that all three KMP2 hot contractions are
**unconjugated** and that following the plan literally produces a plausible
wrong number no gate but the last would catch; 15-04 therefore ships
`oracle_zdotu` / `oracle_zdotu_re` / `oracle_zdot_re`.

Phase 16 has **many more** such sites and a mix of both kinds — `kccsd_rhf.py`
alone conjugates in `energy` (`:47-66`), in `transform_symm` ops 2/3, and in
the `t2`/`tau` assembly. **Every plan in this phase names the primitive per
contraction site in its task text**, and no plan may say only "route through
`oracle_dot`". Phase 16 depends on 15-04 having landed `oracle_zdotu`.

### 3.3 `LARGE_DENOM` is arithmetic, not a guard

`kccsd_rhf.py:34` and every sibling import
`from pyscf.lib.parameters import LOOSE_ZERO_TOL, LARGE_DENOM`, and
`_get_epq` (`kccsd_rhf.py:263`) fills padded entries with
`large_num * np.ones(...)`. Padded orbitals must contribute `~1e-28` to the
amplitude, not be skipped: skipping them is a different program.
`15-CONTEXT §3.3` recorded this for KMP2; it is identically load-bearing here,
and `kccsd_t_rhf.py:30` imports `LARGE_DENOM` for the same purpose in the
triples denominator.

### 3.4 `mo_coeff` is column-major on the SCF side, row-major on the ao2mo side

`crates/pyscf-pbc-scf/src/types.rs:119` (`KScfResult.mo_coeff`, COLUMN-major)
vs `crates/pyscf-pbc-df/src/df_ao2mo.rs:362` (`MoCoeff`, ROW-major). This is
the same shape of defect as 14-05's `decompose_j2c` misread, which was worth
**+6 306 866.73 Ha** and was invisible to every gate then existing.
`15-CONTEXT §3.1` requires ONE conversion implementation and a `CᴴSC = I`
test rather than a round-trip. **Phase 16 reuses Phase 15's conversion; it does
not write a second one.**

### 3.5 The `exxdiv` / Madelung shift is not optional bookkeeping

`pbc/cc/ccsd.py:41` wraps the ERI build in
`lib.temporary_env(self._scf, exxdiv=None)` and then `:57-58` re-adds the
correction through `_adjust_occ(eris.mo_energy, eris.nocc, -madelung)`, with a
comment (`:53-56`) stating that without it "MP2 energy may be largely off the
correct value" and that it matters especially for low-dimension systems where
occupied and virtual energies overlap. `madelung` exists in this port
(`crates/pyscf-pbc-gto/src/coulg.rs:244`, `pbc.py:548-586`). Both halves —
the `exxdiv=None` ERI build **and** the `_adjust_occ` re-add — must ship
together or the energy is quietly wrong; `graphene` (`dimension = 2`) is the
fixture that exercises the overlap case.

### 3.6 Upstream's own memory estimate is a documented TODO — do not port it

`kccsd_rhf.py:1100-1107`:

```python
def _mem_usage(nkpts, nocc, nvir):
    incore = nkpts ** 3 * (nocc + nvir) ** 4
    incore *= 4          # "factor of two for intermediates and two for safety"
    # TODO: Improve incore estimate and add outcore estimate
    outcore = basic = incore
    return incore * 16 / 1e6, outcore * 16 / 1e6, basic * 16 / 1e6
```

It is `nmo⁴`, not the sum of the seven blocks actually allocated, and upstream
says so. On diamond `gth-dzvp` 2×2×2 it returns **13.9 GiB** where the seven blocks
sum to well under 2 GiB. Porting it literally imports a ~7× over-estimate into
this port's HARD `MemoryLimitExceeded` refusal, i.e. it would refuse jobs that
fit. **Compute the per-tensor requirement instead** — see `16-REVIEW.md §2.3`.

### 3.7 Determinism (§9.3) applies to the amplitude iteration, not just the energy

The `nkpts³ · nocc² · nvir²` amplitude accumulation is exactly the D-PBC-17
shape. `oracle_sum`/`oracle_zdotu` go in **from the first version, not as a
retrofit** — `0bcff45`'s D-PBC-17 fix had to retrofit `ztrace_ab`/`trace_ab`/
`trace_dm_v` and it is the more expensive order. Bit-identity at
`RAYON_NUM_THREADS` 1 and 8 is gated on `t1`, `t2` **and** `e_corr`, not on
`e_corr` alone: a non-deterministic `t2` that converges to the same energy will
pass an energy-only gate and fail EOM.

---

## 4. This phase's plan structure — 14 plans, and why each of the 4 extra exists

| plan | content | wave | new? |
|---|---|---|---|
| **16-01** | MEASURE the floor (`e_corr`, EOM roots, (T), the DF-route split, the storage-tier crossover); restate the gate in three documents. **No Rust.** | 0 | **NEW** (§2) |
| **16-02** | Substrate: `KptsHelper::build_symm_map`/`transform_symm`/`_operation`; the **complex** k-indexed tensor container + its spill backend | 0 | **NEW** (§1.3, §3.1) |
| **16-03** | `davidson_nosym1` + `pick_real_eigs` — iterative non-symmetric Davidson | 0 | **NEW** (§1.2) |
| **16-04** | `kintermediates_rhf` (926 l) | 1 | §8.8 16-01 |
| **16-05** | `KRCCSD`: `_ERIS` (3 storage tiers + `symm_map`), `update_amps`, `energy`, `kernel` (1203 l) | 2 | §8.8 16-02 |
| **16-06** | `kintermediates_uhf` + `KUCCSD` (1225 + 1116 l) | 3 | §8.8 16-03 |
| **16-07** | `KGCCSD` + `kintermediates` + the molecular `gccsd` surface it needs (833 + 529 l) | 3 | §8.8 16-04 |
| **16-08** | `KCCSD(T)`: `kccsd_t`, `kccsd_t_rhf`, gated against `kccsd_t_rhf_slow` (319 + 651 + 271 l) | 4 | §8.8 16-05 (+§1.7) |
| **16-09** | `EOM-KCCSD-GHF` IP/EA/EE — **the base module** (2011 l) | 4 | §8.8 16-08, **reordered** (§1.4) |
| **16-10** | `EOM-KCCSD-RHF` IP/EA + EE-**singlet**; Triplet/SpinFlip refuse as upstream does (1716 + 158 l) | 5 | §8.8 16-06 (+§1.5) |
| **16-11** | `EOM-KCCSD-UHF` IP/EA; **EE refuses** as upstream does (1275 l) | 5 | §8.8 16-07 (+§1.5) |
| **16-12** | `kuccsd_rdm` + the Γ-point `pbc/cc/ccsd.py` shim (157 + 157 l) | 4 | §8.8 16-10 (split) |
| **16-13** | `KCIS` (`kcis_rhf.py`, 700 l); `pbc/ci/cisd.py` **deferred explicitly** (§1.6) | 5 | §8.8 16-09 (+§1.6) |
| **16-14** | Verification against the restated gates | 6 | §8.8 16-10 (split) |

**Droppable half if the phase overruns:** 16-09/10/11/13 (EOM + KCIS) are
excited-state properties; nothing in Phases 17-20 needs them for correctness,
and they are ordered last so dropping them costs nothing already built. Do
**not** drop 16-05 or 16-07 instead — Phase 17's 17-09 is blocked by name on
`KRCCSD` (`PBC-MASTER-PLAN §8.9`, "+1071 l if Phase 16 has landed"), and
`scf.kghf.KGHF.CCSD` (`kccsd.py:805`) is a surface Phase 19 reads.

---

## 5. RULING — adopt as **D-PBC-29**

**`D-PBC-28` is taken** (Phase 15's speed ruling, `15-CONTEXT §7`, renumbered
from 27 by `15-REVIEW.md D-15-R-01` after 17-10 claimed 27). Phase 16's ruling
is **D-PBC-29**. The full statement, its evidence and its arithmetic are in
`16-REVIEW.md §6`; in brief:

1. **The complex tensor arena is a new type in `pyscf-runtime`, not a fork and
   not a cast.** `WorkspacePool` keeps its f64 API unchanged (every molecular
   caller is untouched); Phase 16 adds a complex-element sibling sharing the
   budget accounting, the free-list and the `InMemory | Spilled` split.
   Re-interpreting a `Box<[f64]>` as complex pairs is not done: `shape_bytes`'s
   `* 8` and every `Vec<f64>` boundary would silently halve the reported size,
   which lands directly in the HARD refusal.
2. **Contractions are host rayon loops over k-point triples with `oracle_*`
   accumulators, not `zgemm_dense`.** The standing measurement
   `zgemm-dense-loses-to-host-rayon` records `zgemm_dense` at **6-12× slower**
   on the CPU backend (the default here, `pyscf-algebra-cpu-is-default-backend`)
   **and 1.35e-10 off** on grid reductions — outside this project's 1e-11 gate.
3. **`symm_map` is used from the first version** (§3.1) — a genuine ~4× on the
   ERI transform, which is the phase's dominant cost.
4. **The storage tier is chosen from a per-tensor byte count, never from
   upstream's `_mem_usage`** (§3.6), and the gate must cross a tier boundary on
   a real fixture or the spill path ships untested.
