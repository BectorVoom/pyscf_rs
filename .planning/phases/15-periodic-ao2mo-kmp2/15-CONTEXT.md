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

### 3.10 `oracle_zdot` is `zdotc`. Every KMP2 contraction is `zdotu`.

`crates/pyscf-algebra/src/zoracle.rs:35` — `oracle_zdot(x, y)` computes
**`xᴴ·y = Σ conj(x[i])·y[i]`** (`re = dot(xr,yr) + dot(xi,yi)`). It is the only
bit-deterministic complex inner product the workspace exports
(`lib.rs:116`); `zblas::zdotu_dense` is the unconjugated one and `zoracle`'s own
module doc (`:1-14`) rules it out for anything that lands in an energy.

Three of Phase 15's four hot contractions have **no conjugate**:

| contraction | upstream | conjugation |
|---|---|---|
| `Lov·Lov -> oovv` | `kmp2.py:96`, `einsum("Lia,Ljb->iajb")` | **none** — `df_ao2mo.rs:12-19` explains why the second factor is already conjugated inside `cderi` |
| `edi = 2·Re⟨t2, oovv[ka]⟩` | `kmp2.py:113` | `t2` is **already** `conj(oovv/eijab)` (`:110`), so the einsum is unconjugated |
| `exi = −Re⟨t2, oovv[kb]ᵀ⟩` | `kmp2.py:114` | same |

Calling `oracle_zdot(t2, oovv)` therefore computes `Σ (oovv/e)·oovv` — the
**unconjugated square** — instead of `Σ conj(oovv/e)·oovv`. It runs, it returns
a plausible number, and it is wrong. Two ways out, and the plan must pick one
out loud rather than leave it to the implementer:

* **(a)** add `oracle_zdotu` next to `oracle_zdot` in `zoracle.rs` — four
  `oracle_dot` calls in the `zdotu` pattern (`re = rr − ii`, `im = ri + ir`) —
  and use it everywhere §3.10's table says "none"; or
* **(b)** never materialise the conjugation: keep `x = oovv/eijab`
  **unconjugated**, call `oracle_zdot(x, oovv)`, and conjugate only when storing
  `t2`. This is exact for `edi`/`exi` but does **not** help `Lov·Lov`, which has
  no conjugate on either side.

`Lov·Lov` forces (a). **15-04 adds `oracle_zdotu`** (a four-line sibling of an
existing function, host-only, no device path) and both consumers use it.

There is a second, free win in the same place: `edi`/`exi` take only
`Re`, so two of the four `oracle_dot` calls are computed and thrown away.
A `oracle_zdotu_re` / `oracle_zdot_re` that returns just the real part halves
that contraction. Ship it with (a); it is the same file.

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
  **The complex sibling `oracle_zdot` is `zdotc` and every KMP2 contraction is
  `zdotu` — see §3.10 before writing a single one of them.**
* **RULE 9** — every plan ends with a `15-PP-SUMMARY.md` and a `STATE.md` update.

---

## 7. The speed ruling — D-PBC-28

**Numbering:** an earlier draft of this section proposed `D-PBC-27`. That number
was taken on 2026-09-01 by plan 17-10 (`RsCell`/`ExtendedMole`,
`PBC-MASTER-PLAN.md §6`). **The Phase-15 speed ruling is D-PBC-28** and every
plan file in this phase now says so. If an older document (or a summary written
before 2026-09-02) says `D-PBC-27` in a *speed* context, it means this section;
`D-PBC-27` itself is 17-10's `RsCell`/`ExtendedMole` ruling and is unrelated.

§2.2 rules that `kmp2.py:69`'s GDF-vs-FFTDF branch is a **correctness**
decision. It is also a **cost** decision users make when they pick a DF backend,
and nothing in this phase measured it until now. Eight sub-rulings follow. Each
is either a measurement obligation or a construction obligation; none is a
tolerance.

### 7.0 The cost model this section argues from

Per `(ki, kj, ka)` triple, with `no = nocc`, `nv = nvir`, `nx = naux`,
`Ng = ngrids`, `n = nao`. These are **derived operation counts, not
measurements** — 15-01 Task 7 and 15-05 test 7b supply the wall clock.

| step | route | complex mults per triple |
|---|---|---|
| `oovv` assembly, `Σ_L Lov·Lov` | DF | `nx·(no·nv)²` |
| `oovv` assembly, `with_df.ao2mo` | non-DF, **upstream** | `Ng·(no·nv)²` (`_contract_plain`, `fft_ao2mo.py:145-152`) |
| `oovv` assembly, `with_df.ao2mo` | non-DF, **this port today** | `Ng·n⁴ + n⁴·no` (§7.5) |
| `edi` + `exi` | both | `2·(no·nv)²` |

**The assembly dominates by a factor of `naux` (DF) or `ngrids` (non-DF).**
Everything below follows from that one line, and it is the line 15-CONTEXT's
first draft got backwards.

### 7.1 Parallelise the independent k-work — with a granularity rule

`Lov[ki,kj]` (15-04) and `KMP2`'s `(ki,kj)` outer loop (15-05) write to disjoint
slots, so both take `rayon::par_iter` over the flattened index. **But the flat
index is only `nkpts²` long, and the phase's own anchor cell is `[1,1,2]` —
`nkpts = 2`, i.e. FOUR tasks on a sixteen-thread machine.** A bare
`par_iter` over `(ki,kj)` leaves 75 % of the machine idle on the plan's most-run
fixture.

**RULING:** parallelise over the flattened `(ki, kj, ka)` triple where the
consumer allows it, and where it does not (the `oovv_ij[ka]` first pass must
complete for all `ka` before the second pass reads `kb`), nest a second
`par_iter` over `ka` inside the `(ki,kj)` task via `rayon::join`/`par_iter`
on the inner loop. For `Lov`, whose blocks are only `nkpts²`, also parallelise
`r_e2` over its auxiliary index `l` — `df_ao2mo.rs:349-392` is a serial loop
over `blk.naux` whose iterations write disjoint `l * ni * nj` output slices,
so it is a one-line `par_chunks_mut` and it is bit-parity-preserving.

Report the measured thread scaling, not just bit-identity (15-05 test 12
already asks for this; 15-04 test 9b must ask for it too).

### 7.2 The reduction obligation is separate, and it needs a primitive that does not exist

The `emp2_ss`/`emp2_os` accumulation over `nkpts³` triples goes through
`oracle_sum`; the `(no·nv)²`-long `edi`/`exi` dots go through the ordered
complex dot. **See §3.10: the ordered complex dot the workspace exports is
`zdotc` and all three of these contractions are `zdotu`.** 15-04 adds
`oracle_zdotu` (+ the real-part-only variants) before anything consumes it.

### 7.3 Do NOT route the assembly through `zgemm_dense`

Measured on this machine 2026-08-31 (`krks_profile contract`,
`.planning/pbc/baselines/contract-mesh{21,31}.json`): `zgemm_dense` is
**6.7-8.3× SLOWER** than a plain rayon host loop over disjoint output rows on
the default CPU cubecl backend, and on a long reduction its unordered sum is
**1.35e-10** away from `oracle_dot`'s pairwise tree — thirteen times the KRKS
gate. `pyscf-algebra`'s Cargo default is `default = ["cpu"]` (ALG-03), so this
is the default execution path, not a fallback.

**RULING:** the `oovv` assembly is a **rayon loop over disjoint output rows plus
an ordered dot per output element**. No `zgemm_dense`, no `gemm_dense`, in
`pyscf-pbc-mp` or on any Phase-15 path. If a future device GEMM becomes
competitive, `krks_profile contract` is the gate that re-opens the question.

### 7.4 `Lov`'s memory layout is a speed decision — store the auxiliary index FASTEST

§7.3 makes every `oovv[i,a,j,b]` element an ordered dot over `L`. That dot is
contiguous only if `Lov` is stored `[i, a][L]`. Upstream stores `[L, i, a]`
(`kmp2.py:180-190`) because NumPy's `einsum` wants it that way; this port's
inner loop wants the transpose.

**RULING:** `LovTable`'s blocks are stored **`(nocc·nvir, naux)` row-major —
`L` fastest**, and the doc comment records the deviation from `kmp2.py:190`'s
`(naux, nocc, nvir)` with this reason. It is a storage order, not an algebra
change; §3.1's warning applies (there must be exactly one place that knows it).

### 7.5 The non-DF route materialises the `nao⁴` AO ERI, and upstream does not

**This is the phase's largest single speed item and no plan currently names
it.** `crates/pyscf-pbc-df/src/pbc_ao2mo.rs:449-457`:

```rust
pub fn fft_general(df, mos, kptijkl) -> Result<Eri, _> {
    let ao = fft_get_eri(df, kptijkl)?;   // the FULL nao² x nao² AO block
    Ok(mo_eri(&ao, nao, mos))             // then a 4-index MO transform
}
```

Upstream's `fft_ao2mo.general` (`fft_ao2mo.py:102-153`) never builds it: it
transforms AO→MO **on the real-space grid first**
(`mos = [dot(c.T, aos[i].T) for ...]`, `:147-148`) and hands the MO-resolved
grid arrays to `_contract_plain`. `aft_ao2mo.general` (`aft_ao2mo.py:126-...`)
is the same shape. So:

* **flops:** `Ng·n⁴` here vs `Ng·(no·nv)²` upstream — a ratio of
  `(n²/(no·nv))²`. Diamond `gth-szv` (`n=8, no=nv=4`): **16×**. A `gth-dzvp`
  cell (`n=26, no=4, nv=22`): **59×**. And KMP2 calls it `nkpts³` times.
* **memory:** `n⁴` complex per call — 1.6 GB at `n = 100`, on a machine where
  17-12's host tests already exit-137 (`STATE.md`) and where the standing
  memory `materialised-grid-values-oom` records the same failure mode.
* **RULE 2:** this is an unrecorded structural deviation from upstream shipped
  by 14-05, not a Phase-15 defect — but Phase 15 is its first `nkpts³` caller.

**RULING:** **plan 15-08** adds the MO-first path to `fft_general` /
`aft_general`, keeping the existing AO-ERI route reachable and bit-compared
against it. It depends on nothing in this phase and starts at wave 0.

### 7.6 Hoist what does not depend on the triple

`fft_get_eri` (`pbc_ao2mo.rs:231-268`) recomputes `get_gv_weights`, `get_gv`
and `get_coulG` on **every call**. Inside KMP2 that is `nkpts³` rebuilds of
quantities that depend only on `q = k_j − k_i`, of which there are at most
`nkpts` distinct values. Same for the MO slices `orbo[ki]` / `orbv[ka]`, which
`kmp2.py:99-102` re-slices per triple.

**RULING:** 15-08 gives the plane-wave `general` path a caller-supplied,
per-`q` cache for `(Gv, weights, coulG)`; 15-05 hoists the `mo_slice` calls out
of the `(ki,kj,ka)` loop into a `nkpts`-long table built once. Both are
bit-parity-preserving and both are checked by an `assert_eq!` against the
un-hoisted form on the anchor cell.

### 7.7 The `symm_map` corollary — right conclusion, wrong reason, and the right reason is worth recording

The first draft of this section declined to use `KptsHelper.symm_map`'s ERI
symmetry inside `KMP2.kernel` on the grounds that "the dominant cost per triple
is the `nocc²·nvir²` `edi`/`exi` contraction, which every member of a symmetry
orbit still has to pay." **§7.0 shows that premise is backwards**: the
`edi`/`exi` contraction is `2·(no·nv)²` and the assembly it was called cheaper
than is `naux·(no·nv)²` — the assembly is `naux/2` times MORE expensive, and
`naux` is several times `nao`. The symmetry would shrink the dominant term, not
the cheap one.

The conclusion still holds, for two reasons that are actually true:

1. **The usable symmetry inside the `(o v | o v)` pattern is 2-fold, not
   8-fold.** Of `transform_symm`'s four operations (15-02), only op 0 and op 1
   (`transpose(2,3,0,1)`, `(ia|jb) -> (jb|ia)`) map an `oovv` block to another
   `oovv` block. Ops 2 and 3 conjugate-transpose *within* each pair and produce
   `(ai|bj)` / `(vo|vo)` blocks, which KMP2 never asks for. The available saving
   on the assembly is **≤ 2×**, against `build_symm_map`'s own `O(nkpts³)`
   construction (15-02) and a second `nkpts`-long `oovv` buffer to hold the
   partner block.
2. **Upstream does not** (`kmp2.py:93` reads only `khelper.kconserv`), and
   RULE 2's "same order" is not something to spend on an unmeasured ≤2×.

**RULING:** unchanged — `Kmp2` builds its helper with
`KptsHelper::without_symm_map` and the kernel does not use the map. **But the
reason recorded next to the kernel is the one above, not the arithmetic that
was wrong**, and the ≤2× is logged as a Phase-16 carry-over (KCCSD pays the
`build_symm_map` cost anyway, so the trade there is different).

### 7.8 The rayon fan-out has a peak-memory budget, and this machine OOMs

`oovv_ij` is `nkpts × (no·nv)²` complex **per `(ki,kj)` task**. Parallelising
the outer loop multiplies peak resident memory by the number of live tasks:

```
peak ≈ min(threads, nkpts²) · nkpts · (nocc·nvir)² · 16 bytes
```

At the reference systems this is kilobytes (diamond `gth-szv` `[2,2,2]`:
8 · 8 · 256 · 16 = 256 KiB) and the ruling is free. It is **not** free at the
scale a user reaches, and 17-12's whole host suite is currently SIGKILLed
(exit 137) on this machine (`STATE.md`, `ROADMAP.md` Phase-17 rollup).

**RULING:** 15-05's memory pre-flight (`kmp2.py:70-85`) evaluates the formula
above **including the thread factor**, not just upstream's two single-threaded
formulas, and the rayon pool for the outer loop is bounded so the product stays
under the `PYSCF_MAX_MEMORY` budget `pyscf-ccsd` already owns. A refusal is
correct; an exit-137 is not.

### 7.9 What gets reported

Every claim above is either measured or not made:

* **15-01 Task 7** — upstream's DF-vs-non-DF wall clock, per system and mesh.
* **15-02 test 7** — `build_symm_map`'s `O(nkpts³)` growth curve.
* **15-04 test 9b** — `Lov` bit-identity at 1 and 8 threads **and** the measured
  speed-up (bit-identity alone cannot tell a parallel loop from a serial one).
* **15-05 test 7b** — the port's own DF-vs-non-DF ratio next to 15-01's.
* **15-05 test 12** — `e_corr`/`t2` bit-identity at 1 and 8 threads, plus the
  wall-clock drop.
* **15-08** — the MO-first vs AO-ERI route: bit-comparison, flop ratio, wall
  clock, and peak RSS, on the anchor cell and one larger basis.
* **15-07 §8** — the ledger that collects all of it. A blank row there reads as
  "never measured", which is the failure mode the correctness gates exist to
  prevent, applied to speed.
