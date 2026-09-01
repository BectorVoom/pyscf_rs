# Phase 15 — Periodic AO2MO + KMP2 — CONTEXT

**Written:** 2026-08-31, before any Phase-15 code.
**Read this before `15-01-PLAN.md`.** Everything here was verified against the
vendored `pyscf/` tree and the current Rust workspace on 2026-08-31; every claim
carries the file and line that proves it.

`PBC-MASTER-PLAN.md §8.7` sizes this phase at five plans. **Three of those five
are wrong about the starting state**, in both directions: two are largely
already shipped, and one names a kernel that does not exist upstream at all.
This document records what is actually there, what is actually missing, and the
two gate statements that cannot be believed as written.

---

## 1. The scope corrections, in order of consequence

### 1.1 `kconserv` shipped in Phase 9. `ktensor` is Phase 17, not Phase 15.

`§8.7` puts "`kconserv` tables (K-16), `KptsHelper`, `ktensor`" in plan 15-01.

* **`get_kconserv` and `get_kconserv3` already exist** —
  `crates/pyscf-pbc-lib/src/kpts_helper.rs`, shipped by plan 09-07, exported as
  `pyscf_pbc_lib::kpts_helper::{Kconserv, Kconserv3, get_kconserv, get_kconserv3}`
  with `KCONSERV_TOL = 1e-9` and a documented deviation (upstream's `k2gamma`
  shortcut is skipped for the brute-force branch). `df_ao2mo::ao2mo_7d` already
  consumes it. **Do not re-port it.**
* **`KptsHelper` is NOT ported.** `kpts_helper.py:544-632` is the class with
  `symm_map`, `_operation` and `transform_symm` — the 8-fold ERI symmetry map.
  `KMP2.__init__` constructs one (`kmp2.py:715/723`) and `kmp2.kernel` uses
  **only `khelper.kconserv`** (`kmp2.py:93`). So KMP2 needs the field, not the
  map. `build_symm_map` is `O(nkpts³)` work that only KCCSD (Phase 16) reads.
* **`ktensor.py` is the `KsymmArray` container for k-point SYMMETRY** — its
  `transform_2d`/`transform_4d` take `rmat` rotation matrices and its only
  consumers are `kmp2_ksymm.py`, `khf_ksymm.py` and friends. That is
  `PBC-MASTER-PLAN §8.9`'s Phase 17 (`pbc/symm`, `KPoints` IBZ machinery,
  `*_ksymm` adapters). **Porting it here would build a container with no caller
  and no way to test it.** It moves to Phase 17 and this phase says so out loud.

### 1.2 Plan 14-05 already shipped most of `§8.7`'s plan 15-02.

`§8.7` puts "`pbc/ao2mo/eris`: `general`, `get_eri` at k-quadruples; per-DF
`fft_ao2mo`, `df_ao2mo` wiring" in 15-02. Current state:

| upstream | Rust | status |
|---|---|---|
| `df_ao2mo.get_eri/general/ao2mo_7d` | `crates/pyscf-pbc-df/src/df_ao2mo.rs:498/619/744` | **shipped (14-05)** |
| `mdf_ao2mo.get_eri/general/ao2mo_7d` | `crates/pyscf-pbc-df/src/mdf/mdf_ao2mo.rs:44/56/70` | **shipped (14-05)** |
| `aft_ao2mo.*` / `fft_ao2mo.*` | `crates/pyscf-pbc-df/src/pbc_ao2mo.rs:195/231/383/434/449/474/484` | **shipped (13-06 + 14-05)** |
| `rsdf`'s ao2mo | — | **missing** (`rsdf.rs` has no ao2mo surface at all) |
| the `PeriodicDf` dispatch | `crates/pyscf-pbc-df/src/traits.rs:66-100` | **missing** — the trait has `get_nuc`/`get_pp`/`get_jk` and no ao2mo |
| `_init_mp_df_eris` (`Lov`) | — | **missing** |
| `pbc/ao2mo/eris.py` | `crates/pyscf-pbc-ao2mo/` is a two-file stub | **missing** |

**The index contract is already settled and asserted** —
`df_ao2mo.rs:34-70` fixes `eri[ki,kj,kk][i,j,k,l]` with `kl = kconserv[ki,kj,kk]`
in chemists' notation, states that KMP2 reads it as `eri[ki,ka,kj][i,a,j,b]`
with `kb = kconserv[ki,ka,kj]`, and `tests/df_ao2mo.rs` pins it. Phase 13's
carry-over is closed. **Nothing in Phase 15 may re-derive or re-order it.**

So the real 15-02-shaped work is **an interface problem, not an integral
problem**: `kmp2.kernel:92` calls `fao2mo = mp._scf.with_df.ao2mo`, i.e. the
transform must be reachable through the `Box<dyn PeriodicDf>` that
`Krhf`/`Kuhf` own (`krhf.rs:29`, `kuhf.rs:31`). That is plan 15-04 here.

### 1.3 **`KUMP2`'s kernel does not exist upstream.**

`§8.7` plan 15-04 says "`KUMP2` + `kmp2_stagger`". Read the source:

```
pyscf/pbc/mp/kump2.py:38-39    def kernel(mp, mo_energy, mo_coeff, verbose): raise NotImplementedError
pyscf/pbc/mp/kump2.py:384-385  def _add_padding(...): raise NotImplementedError("Implementation needs to be checked first")
pyscf/pbc/mp/kump2.py:402-403  class KUMP2.kernel(...): raise NotImplementedError
```

What `kump2.py` *does* ship is the unrestricted **padding surface**:
`padding_k_idx` (`:41`), `padded_mo_energy` (`:104`), `padded_mo_coeff` (`:128`),
`get_nocc` (`:157`), `get_nmo` (`:261`), `get_frozen_mask` (`:334`) — all
two-spin-channel versions of the `kmp2.py` functions, all reachable, all
testable. **RULE 2 makes the Python authoritative**: this port ships the same
surface and the same `NotYetImplemented` for the kernel, with the upstream line
numbers in the error message. Inventing a KUMP2 energy here would be the one
thing no oracle in the workspace could check.

`kmp2_stagger.py` is a full, working, separate method and gets its own plan.

---

## 2. The two gate statements that cannot be believed as written

### 2.1 `1e-14` (ROADMAP) vs `1e-8` (master plan §8.7 plan 15-05)

`ROADMAP.md` line for Phase 15 says "`KMP2` `e_corr` matches upstream to
**1e-14**". `PBC-MASTER-PLAN.md §8.7` plan 15-05 says "**1e-8**". Six orders
apart, in two documents describing the same gate, and **neither was measured**.

This is exactly the failure Phase 14 already paid for once: `§8.6`'s original
gate ("every DF builder gives the same KRHF energy to 1e-15") was wrong in both
halves, and `14-CONTEXT.md` had to replace it with five measured gates. The
correction there was found *before* implementation, and that is the only reason
it did not cost a re-write. **Repeat that discipline here.**

What is already known about the achievable floor, from measurements this
workspace has committed:

* `KRHF` on GDF, He-fcc `sto-3g` 2×2×2, all-electron: **2.750e-10** vs upstream
  (14-VERIFICATION Gate 1).
* `KRHF` on RSDF, same cell: **2.325e-10**.
* `df_ao2mo.get_eri`: **1.667e-12**; `ao2mo_7d`: **1.984e-12** (14-05).
* AFTDF `get_eri` vs upstream: **4.172e-12** (13-06).
* `KRKS`/`KRHF` FFTDF gate residuals: **~4e-12 … 6.5e-12** at `conv_tol=1e-11`
  (`.planning/pbc/SUMMARY.md`, seven-test gate run).

`e_corr` is a *quadratic* functional of those integrals divided by orbital-energy
denominators, and the orbital energies come out of an SCF that is itself
converged to `conv_tol`, not to machine precision. **An `e_corr` agreeing to
1e-14 with upstream is not physically reachable through this stack**, and a gate
that demands it will fail a correct implementation. Equally, `1e-8` may be far
looser than what is actually achieved and would not catch a real defect.

**Plan 15-01 measures the floor before anything is implemented** and the phase's
gate is restated from that measurement, with the reasoning recorded, exactly as
`14-CONTEXT.md` did. Do not write the gate number first and measure second.

### 2.2 There are TWO upstream KMP2 routes and they do not agree with each other

`kmp2.kernel:69` sets `with_df_ints = mp.with_df_ints and isinstance(mp._scf.with_df, df.GDF)`.
So the SAME `KMP2` object computes `(ia|jb)` two different ways depending on the
mean-field's DF class:

* **GDF/MDF/RSDF** → `_init_mp_df_eris` (`kmp2.py:156-227`) builds
  `Lov[ki,kj][L,i,a]` from `cderi` and contracts
  `einsum("Lia,Ljb->iajb")` (`kmp2.py:96`).
* **FFTDF/AFTDF** → `fao2mo(...)` = `with_df.ao2mo`, a four-index MO transform
  per `(ki,kj,ka)` (`kmp2.py:98-104`).

These are the same two routes whose *SCF* energies Phase 14 measured **4.5e-6 Ha
apart on diamond** (the DF fitting error; `14-VERIFICATION` Gate 3, and the
standing memory `rsdf-gdf-disagree-on-diamond`). The correlation energies will
disagree by the same *class* of amount. **Gate each route against its own
upstream number.** A single "KMP2 matches upstream" assertion that does not name
its DF backend is untestable.

---

## 3. Traps this phase will hit, with the line that proves each

### 3.1 `mo_coeff` changes storage order at the SCF→MP2 seam — **column-major vs row-major**

* `KScfResult.mo_coeff` is `Vec<CTensor>`, documented **COLUMN-MAJOR `nao × nmo`**
  (`crates/pyscf-pbc-scf/src/types.rs:119-120`).
* `df_ao2mo::MoCoeff` is **ROW-MAJOR**: `r_e2` indexes `a.c.re[p * ni + i]`
  with `p` the AO and `i` the MO (`df_ao2mo.rs:362`, `:382`).

Feeding one to the other transposes every MO coefficient silently. This is the
same shape of defect as 14-05's `decompose_j2c` reading column-major eigenvectors
row-major — **worth +6 306 866.73 Ha** and invisible to every gate that existed
at the time, because no test had ever reached that branch. There must be exactly
ONE conversion function, in one place, with a test that is not a round-trip.

### 3.2 The `1/nkpts` factor appears twice and in different places

`kmp2.py:96` divides the DF route by `nkpts` inside the einsum;
`kmp2.py:104` divides the non-DF route by `nkpts` after the transform; and
`kmp2.py:116-117` divides `emp2_ss`/`emp2_os` by `nkpts` again at the end. Both
divisions are real and they are not the same one. Port the placement, not the
algebra.

### 3.3 `LARGE_DENOM = 1e14`, and it is load-bearing, not a guard

`pyscf/lib/parameters.py:55`. `kmp2.py:129-136` fills `eia`/`ejb` with
`LARGE_DENOM` and then overwrites **only the non-padded entries**
(`np.ix_(nonzero_opadding[ki], nonzero_vpadding[ka])`). Padded orbitals
therefore contribute `~1e-28` rather than dividing by zero. A port that skips
padded entries instead of dividing by `1e14` is *arithmetically different* — it
drops terms that upstream keeps at `O(1e-28)`. Copy the mechanism.

### 3.4 The padding convention puts virtuals at the TOP, not the bottom

`_padding_k_idx` (`kmp2.py:262-263`): occupied indices are `arange(k_o)` —
bottom-aligned — while virtual indices are `arange(dense_v - k_v, dense_v)` —
**top-aligned**. The padding sits *at the Fermi level*, between them. The
docstring table at `kmp2.py:288-305` is the specification and it should be
transcribed into the Rust doc comment verbatim; it is the only readable
statement of the convention.

For every §9.2 reference system at every k-mesh the port has run so far, `nocc`
is k-independent and **the padding is a no-op**. That means the default test
cell cannot see a padding bug. A test that exercises padding must construct
k-dependent `nocc` deliberately (see 15-03 Task 5).

### 3.5 `get_nocc` refuses fractional occupations — and this port has smearing

`kmp2.py:416-421` raises when `any(moocc % 1 != 0)`, naming
`mf.smearing_method` as the cause. `crates/pyscf-pbc-scf/src/smearing.rs` is
shipped and `Krhf.smearing` is a live field (`krhf.rs:34`). Port the refusal
with upstream's message. It is a real user-facing behaviour, not a stub.

### 3.6 `_gamma1_intermediates` returns `(-dm1occ, dm1vir)` — note the sign

`kmp2.py:690`. And `make_rdm1` then does `oo += np.eye(...)` **before**
`d += d.conj().T` (`kmp2.py:592-594`), so the identity is added once and the
Hermitisation doubles it. Both are easy to "clean up" into a wrong answer.

### 3.7 `t2` is `nkpts³ × nocc² × nvir²` complex and is built by default

`WITH_T2 = True` (`kmp2.py:43`). `kmp2.kernel:70-85` does a memory pre-flight
and raises `MemoryError` before allocating. `pyscf-ccsd` already owns a
`WorkspacePool` arena and a `PYSCF_MAX_MEMORY` pre-flight refusal
(`PBC-MASTER-PLAN §8.8` "Reuse note"); Phase 15 is the first periodic consumer
and should reuse it rather than grow a second one.

### 3.8 `kmp2_stagger` needs `exxdiv='vcut_sph'`, which Phase 11 deferred

`kmp2_stagger.py:270` runs `get_bands` inside
`lib.temporary_env(mf, exxdiv='vcut_sph', with_df=df.FFTDF(...))`.
`ExxDiv::VcutSph` exists as an enum variant (`pyscf-pbc-tools/src/coulg.rs:19`)
but `STATE.md`'s "Phase 11 deferrals" table lists `exxdiv='vcut_sph'` as
`NotYetImplemented { phase: 12 }`. **Check whether Phase 12 actually closed it
before planning 15-06's non-submesh branch** — if it did not, that branch defers
and the `flag_submesh = true` branch (which needs no `get_bands`) still ships.

### 3.9 `_init_mp_df_eris` raises on `cell.dimension == 2`

`kmp2.py:145-153`: the 2-D negative-part `cderi` block is not handled and
upstream raises `NotImplementedError`. Port the refusal; do not silently drop
the negative part, which is what "just skip it" would do.

---

## 4. What the phase builds on — the exact call surface

Everything below exists today and was read on 2026-08-31.

| need | where it is | note |
|---|---|---|
| `kconserv[ki,kj,kk]` | `pyscf_pbc_lib::kpts_helper::get_kconserv(&cell.a, &kpts) -> Kconserv`, `.get(k,l,m) -> i32` | takes `cell.a`, not `cell` |
| `round_to_fbz(kpts, wrap_around, tol)` | same module | `kmp2_stagger` needs `wrap_around=true, tol=1e-8` |
| `is_zero` / `member` / `unique` | same module | `KPT_DIFF_TOL = 1e-6` |
| MO-basis 4-index at a k-quadruple | `pbc_ao2mo::{aft_general, fft_general}`, `df_ao2mo::general`, `mdf_ao2mo::general` | k-points addressed by **index**, not vector |
| `ao2mo_7d` | `df_ao2mo::ao2mo_7d`, `mdf_ao2mo::ao2mo_7d`, `pbc_ao2mo::{aft,fft}_ao2mo_7d` | returns `Eri7d` |
| the half-transform `Σ_pq conj(C_i[p]) L[p,q] C_j[q]` | `df_ao2mo::r_e2(blk, nao, a, b)` — already `pub` | **this IS `_init_mp_df_eris`'s inner call** |
| `cderi` blocks | `Gdf::sr_loop(ki, kj, compact) -> Vec<SrBlock>`, `Gdf::get_naoaux()` | `Mdf` has the same pair |
| SCF outputs | `KScfResult { mo_energy: Vec<Vec<f64>>, mo_coeff: Vec<CTensor>, mo_occ: Vec<Vec<f64>>, e_tot, .. }` | `mo_coeff` **column-major**; `nset`-blocked for KUHF |
| `get_bands` | `Krhf::get_bands(kpts_band, dm_kpts)` | `kmp2_stagger`'s non-submesh branch |
| `get_occ` | `pyscf_pbc_scf::kocc::{get_occ_restricted, get_occ_unrestricted}` | |
| MP2 vocabulary | `pyscf_mp2::{Frozen, frozen_mask, Mp2Result}` | `Frozen::{None, Count, List, Auto, Window}` |
| reference cells | `pyscf-pbc-gto` `test-systems` feature; `crates/pyscf-pbc-gto/tests/common/systems.rs` | §9.2 — do not redefine |

**Crates:** `pyscf-pbc-ao2mo` and `pyscf-pbc-mp` are stubs (`lib.rs` + `error.rs`
only) with their dependency lists already correct — `pyscf-pbc-mp` already
declares `pyscf-pbc-lib`, `pyscf-pbc-df`, `pyscf-pbc-scf`, `pyscf-pbc-ao2mo`
and `pyscf-mp2`. No `Cargo.toml` dependency edges need adding for 15-02/03/05.

---

## 5. One committed upstream number to start from

`pyscf/pbc/mp/kmp2.py:807-820` — diamond `gth-szv`/`gth-pade`, lattice given in
**Bohr** (`cell.unit = 'B'`, `a = 3.370137329`), `KRHF` with `kpts =
make_kpts([1,1,2])` and **`exxdiv=None`**, then `KMP2`:

```
e_corr = -0.204721432828996
```

That is FFTDF (the KRHF default), i.e. the **non-DF route**, at a 1×1×2 mesh —
small enough to run in a test. It is the single cheapest tier-2 anchor available
and 15-01 must reproduce it from the vendored 2.12.1 before trusting anything
else it measures. Note the Bohr units and `exxdiv=None`: both matter, and the
§9.2 `diamond` constructor is Ångström-specified, which carries the known
4.951e-9 CODATA gap (`STATE.md`, 09-03). Use a Bohr-specified cell for this
anchor, as `09-05` already had to for `Gv`.

---

## 6. Standing rules that bite specifically here

* **RULE 2** — port line by line, same names, same order. `kmp2.py`'s three
  divisions by `nkpts` and its `LARGE_DENOM` fill are the test of this.
* **RULE 4** — no `mod tests` in a source file (AGENTS.md §2).
* **RULE 6** — `pyscf-pbc-mp` and `pyscf-pbc-ao2mo` must never name `cubecl-*`.
  Every contraction goes through `pyscf_algebra`. `check_dependency_wall` runs
  in every plan's verification block.
* **RULE 8** — planar `CTensor { re, im }` everywhere; no `Complex<f64>`.
* **§9.3 determinism** — the `emp2` accumulation is a sum over
  `nkpts³ × nocc² × nvir²` terms and is exactly the shape D-PBC-17 governs.
  Route the reductions through `pyscf_algebra::oracle_sum` / `oracle_dot`
  (the FOUND-06 pairwise tree) and pin bit-identity at
  `RAYON_NUM_THREADS=1` and `=8`, as `fft_jk` had to (W-05, `.planning/pbc/SUMMARY.md`).
* **RULE 9** — every plan ends with a `15-PP-SUMMARY.md` and a `STATE.md` update.

---

## 7. The speed ruling — parallelise the independent k-work, and measure the route choice's cost

§2.2 already rules that `kmp2.py:69`'s GDF-vs-FFTDF branch is a **correctness**
decision (the two routes disagree by the same 4.5e-6-Ha class of amount Phase 14
measured at the SCF level, so each must be gated against its own upstream
number). What §2.2 does not address, and no plan currently measures, is that the
same branch is also a **cost** decision users make when they choose a DF
backend, and this phase ships no number that tells them what it costs.

**RULING (adopt as D-PBC-27, recorded by plan 15-01/15-05):**

1. **Measure the DF-route-vs-non-DF-route wall-clock gap in upstream, not only
   the energy gap.** 15-01 already runs both routes on the same cells for Gate
   §2.2's sake (Task 3); the marginal cost of also timing them is one more
   column in a table already being built. Record it, so plan 15-05 has a floor
   to compare the port's own two routes against.
2. **The per-k-pair work in `Lov` (plan 15-04) and in `KMP2`'s `(ki, kj)` outer
   loop (plan 15-05) is embarrassingly parallel across k-pairs — parallelise it
   with rayon.** Every `Lov[ki,kj]` block and every `(ki,kj)`'s `oovv_ij`
   assembly writes to a disjoint slot; the reduction that is *not* disjoint —
   `edi`/`exi` accumulating into `emp2_ss`/`emp2_os` across `nkpts³` triples —
   already goes through `oracle_sum`/`oracle_dot` per §9.3 above. Parallelising
   the disjoint outer loop and reducing the shared accumulator deterministically
   are two different obligations; ship both from the first version, not the
   accumulator alone with a single-threaded outer loop bolted on top.
3. Once both are in, plan 15-05's tests report the port's own DF-route vs
   non-DF-route wall-clock ratio next to upstream's (from point 1) — a
   correctness gate with no accompanying cost number leaves a user unable to
   tell which backend to pick for performance, which is exactly the choice
   `kmp2.py:69` forces on them.

**Corollary — do not retrofit `KptsHelper.symm_map`'s 8-fold ERI symmetry into
`KMP2.kernel`, and say why in the code.** The infrastructure sits right next to
the kernel (15-02 ships it in the same plan wave), and it is tempting to use it
to touch only `nkpts³/8` triples instead of `nkpts³`. **Upstream itself does
not** — `kmp2.kernel` reads only `khelper.kconserv` (15-CONTEXT §1.1) — and the
reason generalises: the dominant cost per triple is the `nocc²·nvir²`
`edi`/`exi` contraction, which every member of a symmetry orbit still has to
pay once its `oovv` block is recovered by `transform_symm`, so the 8-fold
reduction would only shrink the cheaper `oovv`-assembly step. `build_symm_map`
itself is `O(nkpts³)` (15-CONTEXT §1.1) — spending that cost to save a smaller
one is not a win, and it would silently diverge from RULE 2's "same order" for
no measured benefit. This is not an omission; it has been considered and
declined, and 15-05 records the decision next to the kernel it applies to.

Both keep the phase's standing discipline: a performance claim is either
measured and reported, or it is not made.
