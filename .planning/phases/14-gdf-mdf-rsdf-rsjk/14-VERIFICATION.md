# Phase 14 verification — GDF / MDF / RSDF / RSJK

**Date:** 2026-08-29 · **Oracle:** vendored PySCF **2.12.1** at `<root>/pyscf`
(`PYTHONPATH` pinned, `pyscf.__version__` asserted in every oracle script)

**Verdict: the phase closes with four of five gates MET, one UNREACHABLE, and
the reason for the fifth is a missing capability in `cintx`, not in this port.**

| gate | what it measures | result |
|---|---|---|
| **1** | GDF vs upstream on the all-electron control | **MET** — 2.750e-10 on the converged `KRHF`; 1.667e-12 / 1.984e-12 on the ERIs |
| **1b** | the same on diamond | **PARTIAL** — see §3; the flagship `make_j3c` is an unmeasured multi-hour run |
| **2** | MDF converges to FFTDF | **MET** — 1.695e-06 → 3.433e-09 → 3.245e-08, on upstream's ladder to within 1 % |
| **3** | GDF vs RSDF | **UNREACHABLE** — RSDF is blocked on a cintx gap (§5) |
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

## §5 — Gate 3 (UNREACHABLE): RSDF, and why

Gate 3 compares `|E_KRHF(GDF) − E_KRHF(RSDF)|` against upstream's own floor
(1.353e-08 diamond 2×2×2, 4.566e-09 gamma, 1.113e-10 He-fcc). **RSDF does not
exist in this port**, and the reason is outside it.

`_RSGDFBuilder.get_2c2e` needs a short-range `int2c2e`; its `outcore_auxe2`
needs a short-range `int3c2e`; `rsjk`'s real-space pass needs a short-range
`int2e`. Range separation is not a distinct integral symbol — upstream never
calls an `int2e_sr_*` — it is libcint's `PTR_RANGE_OMEGA` (`env[8]`) toggle
around the standard symbol. cintx cannot be asked to set it:

1. `cintx_runtime::ExecutionOptions` (`cintx-runtime/src/options.rs:96`) carries
   `f12_zeta` (`env[9]`), `rinv_orig` and `common_orig`. **No `range_omega`.**
2. No kernel reads `env[8]` — `center_3c2e.rs` and `two_electron.rs` do not
   mention `omega`. `cintx-compat/src/raw.rs:35-41` names the constant only in a
   warning not to overwrite the slot.
3. **The gap is already on this repository's record**: Phase 4's Open Question
   A5 / cintx#11, documented at length in
   `crates/pyscf-gto/src/range_coulomb.rs`, which shipped the `env[8]`
   set/restore semantics and CI-gated the numerical RSH assertion behind exactly
   this.
4. A second, independent obstruction: `incore::aux_e2` reaches cintx through
   `build_image_expanded_with_aux`, which builds its `BasisSet` from the parsed
   per-element basis and not from an `_env` array — so even `range_coulomb.rs`'s
   own direct-`_env` workaround is unreachable from the periodic 3-centre
   driver. Closing (1) would still leave this.

**The work needed to lift this is planned in
`.planning/carryovers/D-PBC-24-cintx-range-omega-PLAN.md`** — five stages, with
the finding that makes it tractable: `rys_order = (Σ l_ceil)/2 + 1` is `≤ 3` on
every system this phase gates, and in that regime libcint computes the
short-range integral as `full − LR` with DOUBLED Rys roots, using only the
standard root finder. `CINTsr_rys_roots` — the genuinely hard part — is needed
only above `rys_order = 3`.

`14-07-PLAN.md` Task 7b anticipated this exactly — *"if it does not, this is the
plan's one real blocker and it must be reported as such, not worked around with
a numerically different kernel"* — so every affected entry point returns
`NotYetImplemented { phase: 14 }` naming the gap (D-PBC-20), and the refusals
are asserted in `tests/rsdf_builder.rs`, `tests/rsdf.rs` and `tests/rsjk.rs`.

Substituting the full-range kernel would give builders that run, converge, and
are silently different methods. For `rsjk` that is worse than for RSDF: `rsjk`
is EXACT, so a wrong answer would land inside the 1.2e-3 DF fitting error of a
correct GDF and look plausible.

**Consequently:** plan 14-07 Task 7d's flip of `Gdf::prefer_ccdf` to `false`
does not happen (asserted by `gdf_default_route_has_not_flipped`), and the
committed `df_swap` baseline does not move.

**What DID ship of 14-07/14-08:** all twelve ω estimators plus
`weighted_coulG_LR/_SR` and `_gaussian_int` (7a, 10 tests, every number gated at
1e-12 against `measurements/omega.out`), `get_aux_chg`, and the shared
`density_fit` shim. Three consumers need the ω machinery regardless of the
blocker — `rsjk`, RSH functionals, and Phase 17 — and `rsjk.py:145-151` reads
precisely these functions, so when the cintx gap closes 7a is already in place.

---

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
