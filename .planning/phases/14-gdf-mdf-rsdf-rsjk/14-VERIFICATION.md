# Phase 14 verification — GDF / MDF / RSDF / RSJK

**Date:** 2026-08-29 · **Oracle:** vendored PySCF **2.12.1** at `<root>/pyscf`
(`PYTHONPATH` pinned, `pyscf.__version__` asserted in every oracle script)

**Verdict (REOPENED and updated 2026-08-30): four of five gates MET at close;
Gate 3 was UNREACHABLE and is now MET.** The `cintx` capability it was blocked
on landed as D-PBC-24, and plan 14-07 sub-tasks 7b/7c ported `_RSGDFBuilder` on
top of it. See §5, rewritten.

**REOPENED AGAIN 2026-09-05.** Phase 15 measured this port's GDF `KRHF` **15.23
Ha** off upstream on diamond `[1,1,2]` and **0.146** on He/`6-31g` — systems
with `nao > 1` at a k-mesh, which no gate below covers. Two defects, both fixed;
**§11** is the record, and Gate 1c is the gate that would have caught them.

| gate | what it measures | result |
|---|---|---|
| **1** | GDF vs upstream on the all-electron control | **MET** — 2.750e-10 on the converged `KRHF`; 1.667e-12 / 1.984e-12 on the ERIs |
| **1b** | the same on diamond | **PARTIAL** — see §3; the flagship `make_j3c` is an unmeasured multi-hour run |
| **1c** | the same with `nao > 1` AND `ki != kj` — the two blind spots gates 1/1b share | **MET (2026-09-05, REOPENED)** — **3.017e-9** on He/`6-31g` `[1,1,2]` after fixing two defects neither of the gates above could see; §11 |
| **2** | MDF converges to FFTDF | **MET** — 1.695e-06 → 3.433e-09 → 3.245e-08, on upstream's ladder to within 1 % |
| **3** | GDF vs RSDF | **MET (2026-08-30)** — RSDF **2.325e-10** and GDF **2.750e-10** against upstream's own two routes on He-fcc 2×2×2 (§5) |
| **4** | `_cderi` under 20 % of the FFTDF AO table at 2×2×2 | **MET** — see §4 |

---

## §1 — The ROADMAP's gate was wrong in both halves, and the correction is measured

The ROADMAP said: *"every DF builder gives the same KRHF energy to 1e-15 with
GDF under 20% of FFTDF memory."* Neither half survives contact with a
measurement, and both corrections were recorded **before** implementation
started (`measurements/README.md`, 2026-08-29).

**"1e-15 across builders" is category-wrong, not merely tight.** GDF *fits* the
Coulomb integrals in a finite auxiliary basis; FFTDF and AFTDF evaluate them
exactly. `|E_FFTDF − E_GDF|` on diamond `gth-szv` 2×2×2 is **1.222e-03 Ha** —
the DF fitting error, a property of the auxiliary basis, present in upstream and
reachable by no implementation. Three independent reasons the number cannot
stand:

1. the fitting error itself, 1.222e-03;
2. upstream's OWN two GDF builders disagree by up to **4.502e-06**
   (`_RSGDFBuilder`, the `_prefer_ccdf = False` default, against
   `_CCGDFBuilder`) — `measurements/ccdf.py`;
3. one f64 ulp at `|E| ≈ 10.9` is 1.78e-15, so 1e-15 is under one ulp of the
   quantity being compared. Phase 12 §1d made the same finding for its own
   gate.

**"20 % of FFTDF memory" is k-mesh dependent and the roadmap does not say so.**
`_cderi` is `O(nkpts² · naux · nao_pair)`; the FFTDF AO table is
`O(nkpts · ngrids · nao)`. The ratio grows **linearly in `nkpts`** and crosses
20 % between 2×2×2 and 3×3×3 on diamond — upstream itself is at 6.17 % and
20.95 % respectively. The gate is only meaningful with the k-mesh pinned.

`14-CONTEXT.md`'s five gates replace it, and the ROADMAP line is rewritten to
match. This is the same correction Phase 12 §1d and Phase 13 made, for the same
reason: **do not quietly ship against a different number than the roadmap
claims.**

### Which measurement scripts were re-run, and which are new (Task 0)

`14-09-PLAN.md` Task 0 asks that the pre-implementation measurements be
**re-run, not re-derived**, and that any script missing at the time be named
plainly. Two were missing and both were added during this phase:

| script | status |
|---|---|
| `_cells.py`, `smoke.py`, `params.py`, `rscell.py`, `ddblock.py`, `memory.py`, `builders.py`, `ccdf.py`, `mdfladder.py` | pre-existing, results in `README.md`, re-read and used unchanged |
| **`omega.py` / `omega.out`** | **NEW, plan 14-07 Task 0** — every ω estimator on all four configurations, recorded before a line of 7a was written |
| **`mdfladder_cc.py` / `mdfladder_cc.out`** | **NEW, plan 14-06** — because `mdfladder.out` measures `_RSMDFBuilder`, not the `_CCMDFBuilder` this phase ships (see §6.5) |

---

## §2 — Gate 1 (MET): the algebra, on the all-electron control

He-fcc/`sto-3g` 2×2×2 is the control because D-PBC-23's deferral is provably
inert there: `measurements/ddblock.py` measures `exclude_dd_block`'s effect at
**exactly 0** on this system (`bas_type = [1]`, no smooth shell), against
1.835e-8 on diamond. So the gate has no escape hatch.

| quantity | port | upstream | \|d\| | command |
|---|---|---|---|---|
| `KRHF` on GDF | −2.80842508692377 | −2.80842508664874 | **2.750e-10** | `cargo test -p pyscf-pbc-scf --test df_swap --release -- krhf_on_gdf` |
| `fuse(j3c)` (gamma) | — | — | **1.412e-12** | `--test gdf_builder -- he_fuse_j3c` |
| `j2c` (gamma) | — | — | **7.105e-14** | ditto |
| `df_ao2mo.get_eri`, 3 quadruples | — | — | **1.667e-12** | `--test df_ao2mo -- get_eri_matches_upstream` |
| `df_ao2mo.ao2mo_7d`, complex MOs | — | — | **1.984e-12** | `--test df_ao2mo -- ao2mo_7d_matches_upstream` |
| `KRHF` on MDF, mesh 9 | −2.80848516472422 | −2.80848516444156 | **2.827e-10** | `--test df_swap -- krhf_on_mdf` |
| `KRHF` on MDF, gamma, mesh 11 | −3.20863596869297 | −3.20863596509184 | **3.601e-09** | ditto |
| **port `get_eri` vs UPSTREAM `get_eri`, both over the PORT's own `cderi`** | — | — | **1.110e-16** | `--test df_ao2mo -- get_eri_is_bit_exact` |

That last row is the attribution device, and it is the one worth keeping. It
runs upstream's actual `df_ao2mo.get_eri` over a stub `mydf` whose `sr_loop`
yields this port's blocks: **1.110e-16 is one ulp at `|eri| ~ 0.5`**, i.e. the
two sides run the same contraction over the same inputs and differ only in
SUMMATION ORDER (sequential reduction here, BLAS `ddot` there). So "is the
contraction upstream's?" and "is the `cderi` upstream's?" are permanently
separated, and a future `cderi` regression cannot be mistaken for a `df_ao2mo`
regression.

The DF **fitting error** is reproduced and asserted, so nobody can "fix" it:
`|E_GDF − E_FFTDF|` = **6.0023e-05** against upstream's 6.006e-05.

### The three upstream substitutions every oracle in this phase carries

Each is a measured finding, and each lives in the oracle scripts so the
attribution cannot rot — the device Phase 13 used for its `get_pp` /
`_IntPPBuilder` attribution.

1. **`exclude_dd_block = False`** — D-PBC-23. Inert on He-fcc (measured 0).
2. **`direct_scf_tol = 1e-14`** — upstream's default derives
   `cell.precision / lattice_sum_factor² · 0.1` = 1.46e-11 here, four orders
   looser than this port's Gaussian-product prescreen. 14-02 measured the
   difference at 1.98e-9 in `fuse(j3c)`, a P-INDEPENDENT term (the `q_P·S`
   signature). Note `_CCGDFBuilder.build` OVERWRITES this unconditionally
   (`gdf_builder.py:107-111`), so it must be set AFTER `build`, not in
   `__init__` — setting it in `__init__` changes nothing, which is measurable:
   2.750202510171107e-9 with and without.
3. **`estimate_rcut` flattened to its own maximum** — **new in 14-05, and it is
   the largest of the three.** See §6.

---

## §3 — Gate 1b (PARTIAL): diamond

**Not met, and the reason is wall-clock, not accuracy.** One diamond
`make_j3c` at GAMMA is a single screening group of ~77 M cintx shell triples at
~28 µs each (14-02's SUMMARY). A release run showed `user 280m59s` against
`real 22m20s` — 12.6 of 16 cores busy — and was terminated before finishing, so
**diamond's `cderi` wall time remains unmeasured**. The 2×2×2 case is 8× that
work.

What IS in place: the acceptance test
`tests/df_ao2mo.rs::get_eri_matches_upstream_on_diamond_gamma` is written,
`#[ignore]`d on cost, and carries all three substitutions of §2. It is an
opt-in run and its number is owed.

What is measured on diamond without a 3-centre build:
* the auxiliary cell (`nao`/`nbas` 108/36), `eta`, `mesh`, `ke_cutoff`,
  `fused_cell` (126/42), `auxbar` (12 nnz / 0.23012787965177506),
  `estimate_rcut` — all 1e-11 or better (14-01, 14-02);
* every ω estimator (14-07 7a), 1e-11 or better;
* the eigen-factor identity `V j2c Vᴴ = I` at 3.094e-08 on a rank-105-of-108
  metric (§6);
* Gate 4's memory ratio (§4).

---

## §4 — Gate 4 (MET): memory, with the k-mesh pinned

`cargo test -p pyscf-pbc-df --test memory --release -- --nocapture`

Sizes at mesh `[40,40,40]`, `s2` packing, 16 B per complex:

| system | FFTDF AO table | GDF `_cderi` payload | ratio | upstream (file) |
|---|---|---|---|---|
| **diamond 2×2×2** | 62.50 MiB | 3.80 MiB | **6.08 %** | 3.86 MiB → 6.17 % |
| diamond 3×3×3 | 210.94 MiB | 43.25 MiB | **20.50 %** | 44.20 MiB → 20.95 % |
| He-fcc 2×2×2 | 7.81 MiB | 0.0225 MiB | **0.29 %** | 0.12 MiB → 1.48 % |

The port's numbers are the exact payload `nkpts² · naux · nao_pair · 16`;
upstream's are HDF5 FILE sizes. On diamond the container overhead is under 2 %
and the two agree; **on He-fcc it is not comparable at all** — the payload is
23 552 B and this port's own file is 57 054 B, so upstream's 0.12 MiB is
container, not integrals. The He-fcc row is a record with that caveat attached,
not a gate; diamond is where Gate 4 is stated.

The formula is validated against a `_cderi` file **this port actually wrote** on
He-fcc 2×2×2 (`the_cderi_size_formula_matches_a_real_file`): the payload is
exact and the HDF5 overhead is under 64 KiB.

The 3×3×3 row is not decoration — it is why the gate names its k-mesh. The
ratio scales as `nkpts²/nkpts = nkpts`, asserted exactly (27/8 = 3.375).

---

## §5 — Gate 3 (MET, 2026-08-30): RSDF

**Superseded.** This section recorded Gate 3 as UNREACHABLE because
`_RSGDFBuilder.get_2c2e` needed a short-range `int2c2e`, `outcore_auxe2` a
short-range `int3c2e`, and cintx's safe API could not be asked to set libcint's
`PTR_RANGE_OMEGA` (`env[8]`). The original analysis is preserved in
`.planning/carryovers/D-PBC-24-cintx-range-omega-PLAN.md` §1.

### What closed it

**D-PBC-24 landed in cintx.** `ExecutionOptions::range_omega` exists, is part of
the WORKSPACE query (short range doubles the Rys roots), and the `CINTg0_2e`
omega branch is ported once in `cintx-cubecl::math::range_separation`. The
second obstruction recorded above — that `incore::aux_e2` builds its `BasisSet`
without an `_env`, so the `range_coulomb.rs` workaround was unreachable — turned
out never to matter: **ω rides in the OPTIONS, not in the basis.**
`incore::aux_e2`, `incore::fill_2c2e` and `pbc_intor::PbcIntorOpts` all take an
`omega` now, gated by `SR(ω) + LR(ω) == full` in `tests/incore.rs`.

**Plan 14-07 sub-tasks 7b/7c then ported `_RSGDFBuilder`**, as
`crate::rsdf_builder::{j2c, RsGdfBuilder}` plus
`gdf_builder::j3c::Scheme::RangeSeparated` — one fitting pipeline, three
schemes, the same shape `_CCMDFBuilder` takes upstream.

### The measurement

`PYSCF_ORACLE_VENV=1 cargo test -p pyscf-pbc-scf --release --test gate3_rsdf --
--ignored`, `conv_tol = 1e-12`, against vendored PySCF 2.12.1:

**He-fcc `sto-3g` 2×2×2** (all-electron control):

| route | upstream | this port | error |
|---|---|---|---|
| **RSDF** (upstream's DEFAULT, `_prefer_ccdf = False`) | −2.80842508717097 | −2.80842508693849 | **2.325e-10** |
| **GDF**, compensated charge | −2.80842508664874 | −2.80842508692377 | **2.750e-10** |
| `\|CC − RS\|` | 5.222351e-10 | 1.471179e-11 | ratio **0.028** |

**Diamond `gth-szv`/`gth-pade` gamma** (pseudopotential):

| route | upstream | this port | error |
|---|---|---|---|
| **RSDF** | −10.14369692267123 | −10.14369690652303 | **1.615e-8** |
| **GDF**, compensated charge | −10.14369242019033 | −10.14369244092593 | **2.074e-8** |
| `\|CC − RS\|` | 4.502481e-6 | 4.465597e-6 | ratio **0.9918** |

Two things follow, and the second corrects an earlier reading in this file.

**1. Each route reproduces upstream's corresponding route** at the port's own
accuracy for that system — 2-3e-10 on the all-electron control (Gate 1's level),
1.6-2.1e-8 on the pseudopotential cell (the GTH floor §3 prices). Notably
RSDF's diamond error (1.615e-8) is *smaller* than GDF's (2.074e-8), so
`_RSNucBuilder`'s absence — both schemes take `get_nuc`/`get_pp` from the
compensated route — does not show at this level even on a pseudopotential cell.

**2. The ORIGINAL Gate 3 criterion is MET, on diamond.** `14-07-PLAN.md` Task 7e
item 5 and this file's first draft of §5 asked for `|E(CC) − E(RS)|` to land on
upstream's own gap within a factor of 2, reasoning that "two independent
implementations of the same fitted quantity reproducing upstream's *disagreement*
is stronger evidence than either matching alone". On diamond gamma the port
gives **4.465597e-6** against upstream's **4.502481e-6** — a ratio of
**0.9918**. That is the criterion passing on its own terms, and it is the
single strongest piece of evidence in this phase.

### Where the original criterion does NOT discriminate, and why

On **He-fcc** the same criterion would fail: the port's gap is 1.471e-11 against
upstream's 5.222e-10, a ratio of 0.028. **That is not a defect and must not be
read as one.** Upstream's two routes differ partly through `exclude_d_aux` and
`exclude_dd_block`, which this port has in NEITHER route (D-PBC-21 / D-PBC-23
defer `ft_ao._RangeSeparatedCell` to Phase 17). On a 1-AO all-electron cell those
splits are essentially the whole of the inter-route difference, so removing them
from both routes leaves the two converging to the same quantity — the port
agrees with itself better than upstream does. On diamond, where the fitting error
is four orders larger and the routes genuinely diverge on the auxiliary fit
itself, the splits are a small part of the gap and the criterion recovers.

The gate therefore asserts **per-route agreement with upstream** on both systems
(the statement that holds everywhere) and **reports** the gap ratio, which is
0.9918 on diamond and uninformative on He-fcc. Gating on the ratio alone would
have produced a false negative on the all-electron control.

### The one deliberate divergence, priced

`_RSGDFBuilder.build` calls `_guess_omega` with the **orbital cell** where
upstream passes the **auxcell** (`rsdf_builder.py:145`). Upstream can afford the
auxcell's coarser answer because `exclude_d_aux` and `exclude_dd_block` route
what a coarse grid cannot resolve around the grid; this port has neither split,
so it resolves it instead:

| `(omega, mesh)` from | value | error vs upstream RSDF |
|---|---|---|
| `_guess_omega(auxcell)` — upstream's | (0.421018, [7,7,7]) | **8.670e-7** |
| `_guess_omega(cell)` — this port's | (0.739359, [11,11,11]) | **1.97e-10** |

`[11,11,11]` is also exactly what `measurements/omega.out` records and what
`tests/rsdf_builder.rs::guess_omega_matches_upstream` already pinned.

### Two bugs this gate caught, both real

1. **The SR 2-centre lattice sum was truncated.** `auxcell.rcut` is an *orbital*
   radius; the metric sums a two-centre *Coulomb* interaction `erfc(ωR)/R`,
   which reaches much further. Upstream sets
   `auxcell_c.rcut = estimate_rs_2c2e_rcut(...)` at `rsdf_builder.py:274` and
   this port had not. Measured: the real-space SR metric differed from the
   reciprocal `Σ_G conj(auxG) coulG_SR auxG` by **1.25e-4** at every k-point,
   worth **8.57e-5 Ha** in the converged energy. With it, the metric agrees with
   a converged reciprocal reference to **2e-12**.
2. **The diagnostic's own reference was the unconverged one.** `FT_SR` in
   reciprocal space converges *slowly* — the SR kernel is not smooth in `G` —
   so the first reference mesh made a correct metric look 7.6e-5 wrong. Meshes
   21/41/61/81 give 1.02e-1 / 7.64e-5 / 2.02e-9 / 1.38e-12 against the analytic
   SR sum. Recorded because the same trap will catch the next reader.

### Also shipped on the same foundation

* **`_RSMDFBuilder`** (`mdf.py:238-353`) — `_RSGDFBuilder` with `mixed` set;
  upstream expresses it as a subclass overriding three methods, this port as a
  flag on one builder. Gated **at matched meshes** against upstream's
  `df.MDF()` default route: **3.209e-10** (mesh 11), **1.897e-11** (15),
  **7.808e-12** (21). `measurements/mdfladder.out`, recorded on this route and
  unreachable in 14-06, is reachable again.

  Two bugs it caught. `_RSMDFBuilder` ends with
  `weighted_coulG = MDF.weighted_coulG` (`mdf.py:353`), so **every** kernel it
  uses — short-range included — carries MDF's `±Gmax ± 0.5` edge screen, which
  fires at every k-difference on a 2×2×2 mesh; omitting it was worth
  **1.176e-4 Ha**. And `mdf.py:265` passes no `precision` to
  `estimate_rs_2c2e_rcut` where `rsdf_builder.py:274` passes `precision**1.5`;
  matching upstream's looser value measured WORSE (1.324e-6 against 1.160e-6),
  because a smaller precision gives a larger radius, so both schemes keep the
  tighter one.

  **For MDF the mesh is definitional, not a convergence knob**: the plane-wave
  set is part of the basis, so two meshes are two different valid MDF
  approximations and an MDF energy is only comparable at a matched mesh. That
  is why this gate forces the mesh on both sides, and why the port's default
  (`[11,11,11]`, from the cell) sitting 1.160e-6 from upstream's default
  (`[7,7,7]`, from the auxcell) is a difference of grid rather than of algebra —
  the ladder above is the proof.

* **Task 7d — `Gdf::prefer_ccdf` flipped to `false`**, matching upstream.
  `df_swap.rs` now pins **both** routes against their own upstream numbers.
  That mattered: the two disagree by 5.222e-10 on He-fcc, *inside* that test's
  1e-9 bar, so the pre-flip test would have kept passing while silently
  measuring the other route — precisely the drift Task 7d exists to prevent.
  The default route is also the faster one, 1.3 s against 6.6 s on He-fcc 2×2×2.

### What is still NOT ported

* **`_RSNucBuilder`** (`rsdf_builder.py:1098-1311`) — **a performance
  carry-over, not a fidelity gap, and the same one 14-03 opened for
  `_CCNucBuilder`.** This port uses NEITHER split nuclear builder:
  `gdf::nuc::get_nuc` goes straight to AFTDF at the cell's converged mesh, where
  it is oracle-gated at 2.755e-12 — strictly more accurate than either split.
  What a split buys is speed (`[9,9,9]` instead of `[43,43,43]`), and 14-04
  measured that evaluating the whole nuclear attraction on the small mesh
  without the split is worth **0.0743 Ha**. Consistent with that, the flipped
  default shows no nuclear penalty: on diamond gamma the RS route's error
  (1.615e-8) is *smaller* than the CC route's (2.074e-8).
* `rsdf_helper.py`'s prescreen (`get_q_cond`, the Schwarz bound). Its absence
  keeps MORE primitives than upstream — conservative, as 14-05 was toward
  `ExtendedMole.strip_basis`.
* **`pyscf_pbc_scf::rsjk` (14-08 Task 4) — and its blocker is NOT the one this
  phase reported.** D-PBC-24's `range_omega` was necessary but not sufficient.
  Two independent things are still missing:

  1. **The supermole.** `rsjk.py:150-200` builds the short-range half over
     `ft_ao._RangeSeparatedCell` + `ft_ao.ExtendedMole.strip_basis`, and
     `_get_jk_sr` (`:267-436`) indexes it by `supmol.bas_mask`
     `(bvk_ncells, rs_nbas, nimgs)`. Both types are Phase 17 (D-PBC-21/23) —
     the same carry-over `exclude_dd_block` and `strip_basis` wait on.
  2. **A periodic 4-centre `int2e` driver.** `_get_jk_sr` drives
     `PBCVHF_direct_drv1`, a SCREENED direct sweep. `grep int2e` across every
     `pyscf-pbc-*` crate finds one doc comment and no implementation;
     `incore::aux_e2` is 3-centre.

  Point 2 is why the manoeuvre that unblocked RSDF does not transfer.
  `_RSGDFBuilder` could be ported by treating every function as compact and
  paying for the missing split in grid points — a degenerate case that is merely
  slower. For `rsjk` the screening **is** the algorithm: an unscreened 4-centre
  sweep over the BvK images is not slower but infeasible, so there is no
  correct-but-slow fallback. **Sequence it after Phase 17 and size it as its own
  plan.**

Substituting the full-range kernel would still give builders that run, converge
and are silently different methods; for `rsjk` — which is EXACT — a wrong answer
would land inside a correct GDF's 1.2e-3 fitting error and look plausible.

## §6 — Defects the phase's own tests caught, and what each was worth

Phases 11, 12 and 13 each ended with this section; it is the part of a
verification document worth re-reading.

### 1. `decompose_j2c` read `zeigh_gen`'s eigenvectors TRANSPOSED — **6.3e6 Ha**

`pyscf_algebra::zeigh_gen` returns its eigenvector matrix COLUMN-MAJOR; the
eigen route read it row-major. The transpose of an orthogonal matrix is still
orthogonal, so the factor had the right shape, rank and eigenvalues and nothing
crashed — it simply built the fitted tensor in the wrong basis.

**No gate had ever exercised that branch.** `j2ctag` is `CD` on every system in
`measurements/params.py`, including diamond, whose metric has
`eig_min = 3.17e-11` (below `linear_dep_threshold`) and which upstream still
decomposes by Cholesky because Cholesky is tried first and succeeds. MDF
(`j2c_eig_always = True`, `mdf.py:365`) was the first consumer:
**+6 306 866.73 Ha** on He-fcc 2×2×2, against −2.808485 after the fix.

New regression test: `V j2c Vᴴ = I` on the retained subspace — three matrix
products, the identity that *defines* the factor, measured at 2.709e-14
(He-fcc) and 3.094e-08 (diamond, rank 105/108, a conditioning floor). It fails
in milliseconds on a transposed, permuted or mis-phased factor, all of which
passed the previous shape/rank/tag assertions.

### 2. Two missing devices from `gen_uniq_kpts_groups`, both eigen-route-only

* **`if self_conj: j2c = j2c.real`** (`rsdf_builder.py:866-868`). A complex
  Hermitian eigensolver applied to a numerically-real metric may return each
  eigenvector with an arbitrary phase `e^{iθ}`; `cderi` is contracted as
  `Σ_L c_L c_L` with **no conjugate**, so the phase survives as `e^{2iθ}`.
  Cholesky has no such freedom.
* **The conjugate pass.** Upstream yields a second entry per non-self-conjugate
  group at `−kpt` with the pairs swapped and the SAME decomposition conjugated,
  rather than decomposing `j2c[−k]` independently, because the two
  decompositions can land on different `cderi` dimensions. Latent on a 2×2×2
  mesh (every difference there is self-conjugate) and real from 3×3×3 up.

### 3. `get_naoaux` was STRICTER than upstream, and wrongly so

14-03 made it raise when the per-k-pair ranks disagreed, on the stated reasoning
that upstream "raises rather than silently truncating". Upstream takes
`next(iter(...))` — one arbitrary block — and returns its leading dimension
(`df.py:592-597`). The ranks legitimately differ per k-difference on the eigen
route: **MDF on He-fcc 2×2×2 at mesh 15 keeps 10 vectors for one group and 11
for another**, and that is correct, because the auxiliary index is only
comparable within a group. Now returns the diagonal `(0,0)` block's rank — what
`df_jk::get_j_kpts`'s `rho` accumulator actually needs — and the cross-group
consumers in `df_ao2mo` check the pair they use.

### 4. `ExtendedMole.strip_basis` is worth 1.054e-09 in `j3c`, and 14-02's gate could not see it

14-05's first oracle came in at 2.750e-09 against a 1e-11 target. Localising it
needed six measurements, and every input matched while the assembly did not:

| stage | port vs upstream |
|---|---|
| `weighted_ft_ao` | 6.939e-18 |
| `auxbar` | 2.776e-17 |
| `int1e_ovlp` | 2.887e-15 |
| `ft_aopair` at the GDF mesh | 2.001e-11 (predicted `j3c` effect 2.255e-13) |
| Cholesky factor `L` | 1.068e-13 |
| real-space `fuse(j3c)`, both `rcut`s | 1.448e-12 / 1.451e-12 |
| **fused `j3c` after `add_ft_j3c`** | **1.054e-09** |

The cause is upstream's per-shell-pair `estimate_rcut` fed to
`ft_ao.ExtendedMole.strip_basis` (`gdf_builder.py:116-121`). This port has no
`ExtendedMole` (D-PBC-21/23) and evaluates every pair to the maximum radius.
Flattening upstream's radius array to its own maximum collapses the gap to
**7.333e-13** — **the port is the more converged of the two.** 14-02's own
gate compared against a standalone `incore.Int3cBuilder`, which strips nothing,
so it could not have seen this.

### 5. Both of 14-06's stated premises were wrong

* MDF's default mesh is `[11,11,11]` on diamond 2×2×2 and `[9,9,9]` on He-fcc,
  not the plan's `[7,7,7]` — mesh 7 is `mdfladder.py`'s lowest rung.
* **`measurements/mdfladder.out` measures the wrong builder.** Every row was
  recorded with `df.MDF`'s default, and `MDF._prefer_ccdf` is `False`
  (`mdf.py:79`), so the whole table is `_RSMDFBuilder` — plan 14-07's route.
  `mdfladder_cc.py` / `.out` were added and are what Gate 2 asserts against.

### 6. The `r_e2` conjugation convention had to be measured, not read

`pyscf/lib/ao2mo/r_ao2mo.c:AO2MOmmm_r_iltj` carries two comments that both say
`^*`; its arithmetic conjugates the **bra alone** (measured: 2.512e-15 for
bra-only against 12.227 for both).

---

## §7 — Gate 2 (MET): MDF converges to FFTDF, GDF does not

He-fcc/`sto-3g` 2×2×2, against `E_KRHF(FFTDF, mesh 31)`. Upstream's ladder is
`measurements/mdfladder_cc.out`.

| builder | upstream (CC) | **port** |
|---|---|---|
| GDF | 6.002e-05 | **6.002e-05** |
| MDF mesh 7 | 1.695e-06 | **1.695e-06** |
| MDF mesh 11 | 6.684e-09 | **3.433e-09** |
| MDF mesh 15 | 3.216e-08 | **3.245e-08** |

At gamma: `|GDF − FFTDF|` 1.476e-05 (upstream 1.476e-05), `|MDF − FFTDF|`
8.694e-08 (upstream 8.788e-08).

**The ladder is not monotone and a monotone gate would fail a correct
implementation.** MDF's own auxiliary fit and the mesh-31 truncation of the
FFTDF *reference* are two independent floors; past the crossover the comparison
measures the reference. The port reproduces the bounce to within 1 %, which is
stronger evidence than reproducing the descent alone. Phase 13's Gate 2 had the
same two-floor structure.

Gate 2 is therefore stated as: beat GDF by an order at mesh 7, fall two more
orders by mesh 11, beat GDF by three orders at the plateau, and stay within an
order of the plateau afterwards.

---

## §8 — Carry-overs

**ONE piece of work, not four.** `ft_ao._RangeSeparatedCell` + `ExtendedMole`
(with `strip_basis`, `_int_dd_block`, `merge_diffused_block`) closes all of:

| what | priced at |
|---|---|
| D-PBC-23 `exclude_dd_block` | 1.835e-08 Ha (diamond 2×2×2), 2.900e-08 (gamma), **0** (He-fcc) |
| `strip_basis` (new, 14-05) | 1.054e-09 in `j3c` → 2.750e-09 in the ERI |
| Phase 13's `ft_aopair` screening residual (D-PBC-21) | 5.121e-10 |

It also feeds Phase 17. ~600 + ~60 lines.

**The cintx `range_omega` gap** — §5. Blocks `_RSGDFBuilder`, `_RSMDFBuilder`,
`RSDF`, `rsjk`, Gate 3, plan 14-07 Task 7d, and (already) Phase 4's numerical
RSH assertion. It is a cintx change, not a port change.

**Smaller, and each already refused rather than ignored:**
* `rsjk`'s MPI / multi-threaded partitioning variants — Phase 19, a named
  non-goal of this phase.
* `GDF`/`MDF` band k-points (`kpts_band`) — Phase 17; upstream rebuilds
  `_cderi` to cover them.
* `GDF.get_jk(omega)` — the range-separated kernel, same cintx gap.
* `exp_to_discard` — changes `naux` silently.
* The MO-factorised `get_k_kpts` (`force_dm_kbuild = False`) — a performance
  variant, Phase 17.
* `outcore`'s blocking axis is the k-point pair rather than the auxiliary shell
  (14-05, stated deviation).
* Diamond's `make_j3c` wall time is unmeasured (§3).

---

## §9 — Performance, and where this port inverts upstream's ordering

`cargo test -p pyscf-pbc-scf --test df_swap --release -- wall_clock --ignored --nocapture`

**He-fcc/`sto-3g` 2×2×2** (`nao = 1`), release build, 16 cores:

| builder | build | SCF | total | `E_KRHF` |
|---|---|---|---|---|
| FFTDF (mesh 31) | 0.74 s | 1.28 s | **2.02 s** | −2.80848510969440 |
| MDF (mesh 9) | 4.46 s | 1.59 s | 6.05 s | −2.80848516472422 |
| GDF | 5.28 s | 1.30 s | 6.58 s | −2.80842508692377 |
| AFTDF (mesh 31) | 0.00 s | 12.44 s | 12.44 s | −2.80848508650343 |

Upstream's reference ordering on **diamond** 2×2×2 is GDF **6.4 s** < RSDF
13.5 s < MDF 16.9 s < FFTDF 30.0 s < AFTDF 450.6 s, and *GDF being the fastest
builder is the phase's whole point.*

**This port inverts that ordering on He-fcc, and the reason is the system, not
the port.** He-fcc has `nao = 1` and `naux = 23`: FFTDF's whole cost is one
31³ grid against a single AO, which is trivial, while GDF still has to build a
3-centre double lattice sum over 23 auxiliary functions. The DF advantage is
`O(ngrids · nao)` against `O(naux · nao_pair)`, and it only pays once `nao` is
large enough for the AO table to dominate — which is exactly why upstream
measured its ordering on diamond (`nao = 8`, mesh 47³) and not here.

**Diamond's ordering is therefore NOT verified in this port**, because its
`make_j3c` wall time is unmeasured (§3). That is the honest statement, and the
number is owed. What IS verified is the relative ordering AFTDF > everything —
reproduced, and for the same reason upstream sees it (AFTDF re-evaluates the
analytic AO-pair FT every SCF iteration; FFTDF caches an AO table and GDF/MDF
cache `cderi`).

`_cderi` file sizes against `measurements/memory.py` are in §4.

### Static checks

| check | result |
|---|---|
| `cargo run -p xtask --bin check-orphan-modules` | **PASS** — 309 source files, all reachable |
| `cargo run -p xtask --bin check-dependency-wall` | **PASS** — cubecl-* containment intact (ALG-06), and D-07's `hdf5-metno` resolution from 14-03 Task 0 holds |
| `cargo run -p xtask --bin check-no-fma` | **PASS** — no FMA mnemonics in `release-oracle` asm (FOUND-05) |
| `cargo test -p pyscf-pbc-df -p pyscf-pbc-scf -p pyscf-pbc-lib --release` | **PASS** — 0 failed across 20 suites |
| `cargo clippy` on the two crates | the phase's new files are clean; the pre-existing warnings in `fft_jk`/`fftdf`/`gdf/jk`/`incore`/`pp_int`/`auxcell`/`j2c` and in `pyscf-pbc-scf`'s drivers predate this phase |

---

## §10 — Phase 15 is unblocked

Both prerequisites are met:

1. **`ao2mo_7d`'s index order is fixed and asserted** —
   `eri[ki, kj, kk][i, j, k, l]` with `kl = kconserv[ki, kj, kk]`, chemists'
   notation, first index of each pair conjugated. KMP2 reads it as
   `eri[ki, ka, kj][i, a, j, b]` = `(ia|jb)` with `kb = kconserv[ki, ka, kj]` —
   the same table under two index namings, no re-ordering. Gated against
   upstream at 1.984e-12 and against `general` at 1e-13 over every `(ki,kj,kk)`
   of a 2×2×2 mesh with four different `nmo`s.
2. **A GDF that KMP2 can read** — `sr_loop`, `get_naoaux`, and an HDF5 `_cderi`
   in upstream's layout (14-03), now also constructible from an existing store
   through `Gdf::with_cderi` / `Gdf::load_cderi`.

---

## §11 — REOPENED 2026-09-05: two GDF defects the phase's own gates could not see

Phase 15's `oracle_phase15::kmp2_energies` printed the MEAN-FIELD residual
beside the KMP2 one and found this port's GDF `KRHF` **1.523e+1 Ha** off
upstream on diamond `[1,1,2]` and **1.461e-1** on He/`6-31g` `[1,1,2]`, against
`4.772e-11` / `5.849e-12` for FFTDF on the identical cells. `15-VERIFICATION`
row 4 assigned it here. Two independent defects, both now fixed.

**Neither is a tolerance question, and neither could have failed a gate above.**
§3's control is `he_all_electron` — `sto-3g` on He, **one AO** — and §5's
diamond leg is at **gamma**. A defect that needs `nao > 1` AND `ki != kj` has no
gate to fail: `nao == 1` makes `nao_pair == nao * nao`, so the `s2` store and
the `s1` square are the same array, and at gamma every k-pair is diagonal.
`measurements/offgamma_multiao.py` is the fixture that closes both, and
`crates/pyscf-pbc-scf/tests/df_swap.rs::krhf_on_gdf_matches_upstream_he_631g_off_gamma`
is the gate (**GATE 1c**).

### Defect 1 — `sr_loop` served the wrong half of every off-diagonal k-pair

`(L | mu^{ki} nu^{kj})` is Hermitian in `(mu, nu)` **only** at `ki == kj`. An
`s2` store therefore keeps the lower triangle of `(ki, kj)` AND the lower
triangle of `(kj, ki)`, and upstream joins them in `PBCunpack_tril_triu`
(`pyscf/lib/pbc/fill_ints.c:1460-1483`), reached through
`_KPair3CLoader.__getitem__` (`df.py:990-1009`):

```text
out[mu, nu] = tril[mu, nu]          mu >= nu   (from the (ki, kj) block)
out[nu, mu] = conj(triu[mu, nu])    mu >  nu   (from the (kj, ki) block)
```

`crates/pyscf-pbc-df/src/gdf/cderi_store.rs` did two different wrong things,
and **each got exactly half the square right**:

* `reshape_block`'s `s2 -> s1` unpack filled the upper triangle by
  `lib.ANTIHERMI` on the SAME block — correct at gamma and nowhere else. This
  is Phase 14's own, present since 14-03.
* Commit `ff01948` (Phase 15) then added a `reverse` branch that, for `ki < kj`,
  DISCARDED the correctly built `(ki, kj)` block and substituted a conjugated
  `(kj, ki)` — in packed storage that is a different integral, not a transpose,
  because the packed store has no `mu < nu` entry to transpose into place. Its
  stated motive was to make the KMP2 `Lov` route agree with `df_ao2mo`; both
  now read the same correct square, so the agreement holds without it.

The store itself was never wrong: the port's four `cderi` blocks match
upstream's four `j3c` datasets elementwise to `~1e-9`
(`offgamma_multiao.out` §1, against the RAW blocks).

### Defect 2 — `get_nuc` / `get_pp` evaluated on `cell.mesh`

`_CCNucBuilder`'s answer does **not** depend on `cell.mesh`: it splits the
compact part into a real-space `_int_nuc_vloc` and leaves a smooth remainder its
own `[9,9,9]` grid resolves exactly. `gdf/nuc.rs` does not port that split (a
standing carry-over) and substitutes AFTDF — but ran it on `cell.mesh`, which is
converged only if the caller happened to pin a converged one.
`offgamma_multiao.out` §3, He/`6-31g` `[1,1,2]`, `v_nuc[0][0,0]`:

| `cell.mesh` | upstream GDF | upstream AFTDF | upstream FFTDF |
|---|---|---|---|
| `[9,9,9]` (pinned) | **−3.229030131116** | −3.027263280742 | −3.795643296556 |
| `[99,99,99]` (the cell's own estimate) | **−3.229030131116** | −3.229030132539 | −3.229030132539 |

So the mesh is now the cell's own `estimate_mesh` — the grid `cell.precision`
demands for exactly this integral — never coarsened, and never made coarser than
a finer mesh the caller pinned. A cell that does not pin a mesh is UNCHANGED
(`Cell::build` already set `mesh = estimate_mesh`), which is why §3's He-fcc
number and diamond's `[47,47,47]` are unmoved. The carry-over is now honestly a
SPEED carry-over, which is what §3 always claimed it was.

### Measured, after both fixes

He/`6-31g` `[1,1,2]`, `KRHF`, `exxdiv = None`, `conv_tol = 1e-11`:

| stage | port | upstream | \|d\| |
|---|---|---|---|
| before | −2.34276106854970 | −2.48883216137748 | 1.461e-1 |
| defect 1 fixed | −2.36073554561004 | " | 1.281e-1 |
| **both fixed** | **−2.48883215836059** | " | **3.017e-9** |
| FFTDF control (unmoved by either) | −2.95037354933798 | −2.95037354933213 | 5.849e-12 |

Attribution, on the same cell:

| quantity | \|d\| vs upstream |
|---|---|
| `vj` / `vk` on a fixed model density, all 16 elements | **6.142e-10** |
| GDF `v_nuc[0][0,0]` | **4.98e-9** |
| the four raw `cderi` blocks, elementwise | **~1e-9** |

Diamond `[1,1,2]`, `gth-szv`/`gth-pade`:

| quantity | port | upstream | \|d\| |
|---|---|---|---|
| GDF `get_pp[0][0,0]` | +1.54584046364934e-1 | +1.54584050839746e-1 | 4.47e-9 |
| FFTDF `get_pp[0][0,0]` | +1.54584046391967e-1 | +1.54584046392055e-1 | 8.8e-13 |
| FFTDF `KRHF` | −8.65192328388683 | −8.65192328393455 | 4.77e-11 |
| **GDF `KRHF`** | **−8.65527636655032** | −8.65527634481768 | **2.173e-8** |

**The headline defect is closed: diamond went `1.523e+1 Ha` -> `2.173e-8`.**
Diamond's `get_pp` is unmoved by defect 2 (its mesh already IS the estimate), so
this leg measures defect 1 alone — and `2.173e-8` is diamond's ORDINARY GDF
residual, not a remainder: §5 records `2.074e-8` for the same builder on the
same cell at 2x2x2, and §1 records that GDF *fits* where FFTDF evaluates
(`|E_FFTDF - E_GDF|` = `3.353e-3` here, upstream's own `3.353e-3`). He/`6-31g`
lands two orders tighter (`3.017e-9`) because it is all-electron with `naux = 8`
for `nao = 2`.

### What it unblocks downstream

`15-VERIFICATION` row 4's GDF leg was NOT MET because the mean field under it
was wrong. With that fixed, `pyscf-pbc-mp`'s two GDF assertions became real
oracle gates (`.planning/phases/15-periodic-ao2mo-kmp2/measurements/kmp2_gdf_and_rdm1.out`):

| quantity | port | upstream | \|d\| |
|---|---|---|---|
| KMP2 `e_corr` on GDF, He/`6-31g` `[1,1,2]` | −0.01698936866861078 | −0.01698936907756816 | **4.090e-10** |

`tests/kmp2.rs` had pinned that to **−0.015572369890603862**, `1.417e-3` from
upstream — by its own comment, "what THIS port currently produces", precisely
because there was no usable reference under it. It is now gated against
upstream, and the `Lov` vs four-index AO2MO route agreement it also asserts
(2e-15) survives the removal of `ff01948`'s `reverse` branch, which existed to
force exactly that agreement.

### Tests added

| test | what it pins |
|---|---|
| `pyscf-pbc-df --test gdf -- sr_loop_takes_the_upper_triangle_from_the_conjugate_pair` | the `PBCunpack_tril_triu` assembly on a synthetic, deliberately non-Hermitian `L`; that a compact request is the stored block verbatim; and that an `s1` request with no conjugate pair is REFUSED rather than filled from the wrong block |
| `pyscf-pbc-scf --test df_swap -- krhf_on_gdf_matches_upstream_he_631g_off_gamma` | GATE 1c — the energy, both routes, on the `nao > 1` + off-gamma + pinned-coarse-mesh fixture |

### Two `tests/kmp2.rs` assertions corrected in passing

Both were measured against upstream rather than guessed
(`15-.../measurements/kmp2_gdf_and_rdm1.py`):

1. the GDF `e_corr` pin above, now an oracle gate;
2. `diamond_anchor_and_without_t2` asserted `Tr(gamma_k) == nelec` at EVERY
   k-point, to 2e-10. **That identity does not hold per k-point and upstream
   misses it by 2.8e-2**: the occupied- and virtual-block MP2 corrections cancel
   only after the k-average. Upstream's traces on this anchor are
   `[8.028298787714228, 7.971701212285773]`, mean exactly `8.0`. The test now
   pins each trace against upstream (this port matches to **4.368e-9**) and
   gates the k-AVERAGE at 2e-10. This failure is INDEPENDENT of the two defects
   above — it reproduces bit-identically with the fixes stashed, and its cell
   runs on FFTDF.
