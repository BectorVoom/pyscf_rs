# Phase 14 Context — GDF / MDF / RSDF / RSJK

**Milestone:** v2.0 PBC · **Depends on:** Phases 9–13 (shipped) · **Blocks:** Phase 15 (KMP2 on GDF), Phase 16 (KCCSD), Phase 17 (k-symmetry), Phase 18 (periodic gradients)
**Master plan:** `.planning/pbc/PBC-MASTER-PLAN.md` — read §0, §3 (D-PBC-09/20/21/22), §8.6 and §10 before starting.
**Measurements:** `.planning/phases/14-gdf-mdf-rsdf-rsjk/measurements/` — **read `README.md` first.** Every number below is from there; re-run the scripts, do not re-derive them.

## Goal

Production density fitting: the builders real solid-state calculations actually
use. `GDF` is a `PeriodicDf` implementor with an on-disk `_cderi`, `MDF` and
`RSDF` are two more, `df_jk` builds J/K from `cderi` instead of from a grid, and
`rsjk` builds J/K with no fitting at all.

Phases 11–13 shipped the two *exact* builders (FFTDF, AFTDF). Everything in this
phase is an *approximation* with a controllable error, and that difference is
what the roadmap's gate gets wrong (§ "The gates the roadmap gets wrong").

## Success criteria (all must be TRUE to close the phase)

1. **Gate 1 (oracle) — the algebra.** On **He-fcc/`sto-3g` 2×2×2**, the
   all-electron control where `exclude_dd_block` is provably inert (measured: it
   moves the energy by exactly **0**), the port's `j2c`, `j3c`, `cderi` and
   converged `KRHF` energy match live upstream **2.12.1** `GDF` to **1e-11**.
   This is the gate on the algebra and it has no escape hatch.
2. **Gate 1b (oracle) — the flagship, with the dd-block deferral priced in.** On
   diamond/`gth-szv` 2×2×2 and gamma, the same quantities match upstream to
   **3e-8**, and the port additionally matches upstream **run with
   `exclude_dd_block=False`** to **1e-11**. The second half is the real
   assertion; the first half only records what the deferral costs. See D-PBC-23.
3. **Gate 2 (no oracle) — MDF converges to FFTDF.** `|E_KRHF(MDF) −
   E_KRHF(FFTDF)|` falls monotonically as MDF's `mesh` rises, on a
   pre-recorded upstream ladder. GDF alone does **not** converge to FFTDF and
   must not be gated as though it did (§ "The gates the roadmap gets wrong").
4. **Gate 3 (oracle) — cross-builder.** `|E_KRHF(GDF) − E_KRHF(RSDF)|` sits on
   the upstream floor for that pair, at the same aux basis. Two independent
   builders of the *same* fitted quantity; upstream's own difference is the
   only defensible target.
5. **Gate 4 (no oracle) — memory.** GDF's `_cderi` is under **20 % of the FFTDF
   AO table at mesh `[40,40,40]` on diamond 2×2×2** — measured upstream at
   **6.17 %**. The k-mesh must be stated: at 3×3×3 upstream itself is at
   **20.95 %** and the gate would fail a correct implementation.
6. Every driver (`KRHF`/`KUHF`/`KROHF`/`KGHF`/`KRKS`/`KUKS`/`KROKS`/`KGKS`)
   accepts `GDF`, `MDF` and `RSDF` through `Box<dyn PeriodicDf>` (D-PBC-22) with
   no driver change, and the FFTDF/AFTDF paths stay **bit-identical**.
7. `cargo test --workspace` green; `xtask check-orphan-modules`,
   `check_dependency_wall` and `check_no_fma` green.

## The gates the roadmap gets wrong — read this first

The ROADMAP says **"every DF builder gives the same KRHF energy to 1e-15 with
GDF under 20% of FFTDF memory."** The master plan's §8.6 plan 14-09 softens the
first half to 1e-6. **Both are wrong, and in different ways.**

### 1e-15 (and 1e-6) is category-wrong, not merely tight

Measured on diamond/`gth-szv` 2×2×2 (`measurements/ddblock.py`, `builders.py`):

| builder | `E_KRHF` |
|---|---|
| FFTDF, mesh 31 | −10.93087316795858 |
| GDF, default aux | **−10.93209469510983** |

That is a **1.2e-3 Ha** gap, and it is not an error in either builder. FFTDF and
AFTDF evaluate the Coulomb integrals *exactly* (to a mesh / lattice-sum
truncation); GDF **fits** them in a finite auxiliary basis. The gap is the DF
fitting error and it is a property of the aux basis, not of the port. No
implementation of GDF — upstream's included — can make it 1e-15, or 1e-6.

So the phase's cross-builder gates are restated as:

* **GDF vs upstream GDF** (Gate 1/1b) — the only gate that tests the port.
* **MDF vs FFTDF** (Gate 2) — MDF *is* GDF plus the AFT residual, so it is the
  builder that legitimately converges to FFTDF as its mesh rises. Upstream's
  measured value is **1.124e-06**, against **1.222e-03** for GDF alone: three
  orders, which is what makes it a real gate.
* **GDF vs RSDF** (Gate 3) — same fitted quantity, two independent builders.
  Upstream's own difference is the floor: **1.353e-08**.

And a third, independent reason the 1e-15 cannot stand: **upstream's own two GDF
builders disagree by up to 4.5e-6.** `GDF._prefer_ccdf = False`, so `df.GDF()`
runs `rsdf_builder._RSGDFBuilder`; `_CCGDFBuilder` — the one plan 14-02 ports,
because it is the self-contained one — is the fallback. Measured
(`measurements/ccdf.py`): 5.960e-07 on diamond 2×2×2, 4.502e-06 at gamma,
5.222e-10 on He-fcc. **Plans 14-02/14-03 therefore gate against upstream run
with `_prefer_ccdf = True`**; the port's default flips to the RS route in 14-07,
matching upstream.

### 20 % of FFTDF memory is k-mesh dependent and the roadmap does not say so

`measurements/memory.py`, mesh `[40,40,40]`:

| system | FFTDF AO table | GDF `_cderi` | ratio |
|---|---|---|---|
| diamond 2×2×2 | 62.50 MiB | 3.86 MiB | **6.17 %** |
| diamond 3×3×3 | 210.94 MiB | 44.20 MiB | **20.95 %** |

`_cderi` is `O(nkpts² · naux · nao_pair)`; the FFTDF AO table is
`O(nkpts · ngrids · nao)`. The ratio grows **linearly in `nkpts`** and crosses
20 % between 2×2×2 and 3×3×3 for this system. Gate 4 therefore pins the k-mesh
at 2×2×2 and records the 3×3×3 number as the reason the pin exists.

## D-PBC-23 — `exclude_dd_block` is DEFERRED, and it is worth 1.8e-8 Ha

**Decision.** Phase 14 ports the *definition* of the 3-centre lattice sum —
i.e. upstream's `exclude_dd_block = False` route — and does **not** port
`ft_ao._RangeSeparatedCell` / `_int_dd_block` / `merge_diffused_block`. This
extends D-PBC-21 (which made the same call for `ft_aopair`) to the 3-centre
tensor.

**Why this needed a measurement, not a judgement.** For `ft_aopair`, D-PBC-21
could argue the RS machinery is numerically transparent — it decontracts and
recontracts and only *drops* terms under a threshold. **That argument does not
carry to `exclude_dd_block`, and `measurements/ddblock.py` proves it:**

| system | `=True` (upstream default) | `=False` (this port) | \|dE\| |
|---|---|---|---|
| diamond 2×2×2 | −10.93209469510983 | −10.93209471346292 | **1.835e-8** |
| diamond gamma | −10.14369242019067 | −10.14369244919373 | **2.900e-8** |
| He-fcc 2×2×2 | −2.80842508664874 | −2.80842508664874 | **0** |

`exclude_dd_block` is not screening. It **re-routes** the smooth–smooth block of
`(ij|L)` out of the real-space lattice sum and into an FFT, because that block
converges slowly in real space. Upstream's default is the *more* accurate route,
so the port at `False` is genuinely 1.8e-8 worse, not merely different.

**Consequences, all of which the plans must honour:**

1. On diamond/`gth-szv`, `rs_cell.nbas` is 8 where `cell.nbas` is 4 and
   `bas_type = [1 2 1 2 1 2 1 2]` — the split is LIVE on the flagship system.
   A port that skips it cannot match upstream's default better than ~3e-8.
2. **He-fcc/`sto-3g` has no smooth shell at all** (`bas_type = [1]`), so the two
   routes are bit-identical there. That is why Gate 1 is stated on He-fcc: it
   pins the algebra at 1e-11 with no deferral in the way.
3. Gate 1b asserts against upstream **run with `exclude_dd_block=False`** at
   1e-11. That is a real assertion about the port, and it makes the 3e-8 a
   *priced* deferral rather than an unexplained residual. The test suite must
   carry that substitution so the attribution cannot rot — the same device
   Phase 13 used for the `get_pp` / `_IntPPBuilder` attribution.
4. If a later phase needs the last 1.8e-8, the work is `_RangeSeparatedCell` +
   `_int_dd_block`, ~600 + ~60 lines, and it also closes Phase 13's remaining
   `ft_aopair` screening residual (5.121e-10) and feeds Phase 17. Record it as
   one carry-over, not three.

## The one genuinely new primitive: `int3c2e` over a DOUBLE lattice sum

Everything else in this phase is assembly. Plan 14-01 is the primitive, and it
is where the phase can go wrong quietly.

```text
T[ki,kj][μν, P] = Σ_{L1,L2} e^{-i ki·L1} e^{i kj·L2} ( μ(L1) ν(L2) | P(0) )
```

Translating the whole integrand by `+L1` and substituting `La = −L1`,
`ΔL = L2 − L1` turns the double sum over AO images into a sum over an **aux
image** and an **AO-pair image**:

```text
T[ki,kj][μν, P] = Σ_{La} e^{i(ki−kj)·La}  Σ_{ΔL} e^{i kj·ΔL}  ( μ(0) ν(ΔL) | P(La) )
```

which is exactly the loop shape `pyscf_pbc_gto::pbc_intor::lattice_sum` already
has for the 2-centre case, with one extra index. The inner `ΔL` sum reuses the
existing shell-pair neighbour list; the outer `La` sum is bounded by
`incore::estimate_rcut(cell, auxcell)` (**17.266** Bohr on diamond, **9.532** on
He-fcc — both well inside `cell.rcut`). `pyscf-gto` needs one new basis builder,
`build_image_expanded_triple_basis`, alongside the existing
`build_image_expanded_cross_basis`.

**The sign and index conventions are NOT to be guessed.** Plan 14-01's first
test is `aux_e2` at gamma against a supercell molecular `int3c2e` — oracle-free —
and its second is upstream `incore.aux_e2` itself.

## Non-goals (do NOT do these in Phase 14)

- `ft_ao._RangeSeparatedCell`, `ExtendedMole`, `strip_basis`, `_int_dd_block`,
  `merge_diffused_block`, `_outcore_dd_block` — D-PBC-23 above. Any code path
  that would need them takes the `exclude_dd_block = false` branch; a caller
  that explicitly asks for `true` gets `NotYetImplemented { phase: 17 }`.
- `cell.dimension < 3` and `low_dim_ft_type = 'inf_vacuum'` beyond what Phases
  11–12 already closed. The `j2c_negative` / `cderi_negative` branch exists only
  for 2-D truncated Coulomb; keep the field, refuse the path.
- The `_prefer_ccdf = False` fast path in `df.GDF.build` (which picks
  `_RSGDFBuilder` over `_CCGDFBuilder`) is a *performance* choice between two
  builders that this phase ships both of; wire the selection, do not add a third.
- `KMP2`/`KCCSD` consumption of `df_ao2mo` — Phases 15/16. Plan 14-05 ships and
  tests the tensors; it does not wire a correlated method.
- The full `pyscf.pbc.df` PyO3 surface — Phase 20. Plan 14-03 wires only
  `mf.with_df = GDF(...)` (and MDF/RSDF), matching what 13-07 did for AFTDF.
- `rsjk`'s MPI / multi-threaded partitioning variants — one correct serial path.

## Plans and waves

| Wave | Plans |
|---|---|
| 1 | 14-01 (`int3c2e` double lattice sum, `aux_e2`, `fill_2c2e`, `make_modrho_basis`) |
| 2 | 14-02 (`gdf_builder`: compensating charge, `j2c`, `j3c`, `make_j3c`) |
| 3 | 14-03 (`GDF` + HDF5 `_cderi` + `PeriodicDf`), 14-04 (`df_jk`) |
| 4 | 14-05 (`df_ao2mo`, `outcore`), 14-06 (`MDF`) |
| 5 | 14-07 (`rsdf_helper` + `rsdf_builder`) |
| 6 | 14-08 (`RSDF` + `rsdf_jk` + `rsjk`) |
| 7 | 14-09 (verification rollup) |

14-03 and 14-04 are the same wave because `df_jk` needs only `sr_loop`'s
*signature*, which 14-03 fixes on its first line; 14-05 and 14-06 are the same
wave because `MDF` needs `GDF` + the already-shipped `AFTDF`, not `df_ao2mo`.

## What already ships and must be REUSED, not rewritten

| Need | Where it already lives |
|---|---|
| `ft_aopair_kpts`, `FtKernel`, `ft_ao` single-centre FT | `pyscf_pbc_df::ft_ao` (Phase 13) |
| `AFTDF`, `ft_loop`, `weighted_coulG`, `_fake_nuc` | `pyscf_pbc_df::aftdf` (Phase 13) |
| `get_coulG` incl. the `exxdiv='ewald'` fold at `G+k=0` | `pyscf_pbc_gto::coulg::get_coulg` |
| `madelung`, `_ewald_exxdiv_for_G0` | `pyscf_pbc_gto::coulg::madelung`, `pyscf_pbc_df::df_jk::ewald_exxdiv_for_g0` |
| `_format_dms`/`_format_jks`/`_format_kpts_band` | `pyscf_pbc_df::df_jk` (plan 11-07 put them in ONE place) |
| `Gv`, `Gvbase`, `kws`, structure factors | `pyscf_pbc_gto::gv` |
| lattice images, `rcut`, `_estimate_rcut` | `pyscf_pbc_gto::{lattice, cutoff}` |
| shell-pair neighbour list + screening | `pyscf_pbc_gto::neighborlist` |
| the 2-centre lattice-sum driver to copy | `pyscf_pbc_gto::pbc_intor::{lattice_sum, cross_basis}` |
| molecular `int3c2e`/`int2c2e` through cintx | `pyscf_gto::intor_with_auxmol` |
| the aux-basis name table + `aug_etb` fallback | `pyscf_df::auxbasis` (extend, do not fork) |
| molecular Cholesky of `(P|Q)` | `pyscf_df::cholesky_eri` (the host Cholesky-Banachiewicz) |
| eigh / Cholesky / triangular solve | `pyscf_algebra` |
| complex GEMM, ordered complex reductions | `pyscf_algebra::{zgemm_dense, oracle_zsum, oracle_zdot}` |
| HDF5 datasets | `pyscf_chkfile`'s re-exported `hdf5` alias — **D-07: chkfile is the sole `hdf5-metno` owner** |
| reference cells | `pyscf_pbc_gto::test_systems` (feature `test-systems`) |
| the oracle subprocess protocol | `crates/pyscf-pbc-df/tests/common/mod.rs::run_python` |

## What does NOT exist yet and must be ported inside this phase

- `pyscf_gto::build_image_expanded_triple_basis` — plan 14-01. The existing
  `build_image_expanded_cross_basis` handles two cells; the 3-centre lattice sum
  needs three.
- `kpts_helper.kk_adapted_iter` and `group_by_conj_pairs`
  (`pyscf/pbc/lib/kpts_helper.py:170-268`). Phase 13 listed them as its own and
  did **not** ship them; `gen_uniq_kpts_groups` (14-02) is the first caller that
  genuinely needs them. Plan 14-02 owns them.
- `members_with_wrap_around` (`rsdf_builder.py`) — needed by `make_j3c`'s
  custom-`kptij_lst` branch.
- An HDF5 3-index store with the `j3c/{ki*nkpts+kj}/{istep}` layout, its
  `_load3c` reader and the `s2 → s1` unpack. Plan 14-03.
- `pyscf-pbc-df` **already declares `hdf5-metno` directly**
  (`crates/pyscf-pbc-df/Cargo.toml:22`). That contradicts D-07. Plan 14-03 Task 0
  resolves it: either route through `pyscf_chkfile`'s re-export or record a
  dependency-wall exemption — do not leave it undecided.

## Standing constraints inherited from v1.0 / earlier PBC phases

- Tests in separate files (AGENTS.md §2). No `mod tests` in a source file.
  `xtask check-orphan-modules` makes an undeclared or untested module a CI
  failure — Phase 12 shipped a `lib.rs` that declared only `mod error`.
- cubecl kernels generic over `F: Float`; read the manual at
  `/home/user/Documents/workspace/cubecl_manual/manual/manual/Cubecl/INDEX.md`
  BEFORE writing kernel code (AGENTS.md §3).
- On any cubecl build error, read `cubecl_error_guideline.md` first (AGENTS.md §4).
- Scalar math inside a `#[cube]` body is `cube-math` with `MathConfig::EXACT`;
  host-side scalar math is `std`/`rmath`, never `cube-math`.
- Only `pyscf-algebra`/`pyscf-runtime`/`pyscf-kernels` may name `cubecl-*` (ALG-06).
- `release-oracle` stays FMA-free (`xtask check_no_fma`).
- Ordered reductions only (`oracle_sum`/`oracle_zsum`) in numerical paths.
- Complex tensors are PLANAR `CTensor { re, im }` (D-PBC-02); complex GEMM is
  four real GEMMs, never Karatsuba (D-PBC-03).
- Any deferred branch returns `NotYetImplemented { phase, what }` naming the
  plan that owns it — never a silently wrong answer (D-PBC-20).
- The oracle is the **vendored** PySCF 2.12.1 at `<root>/pyscf`, not
  site-packages 2.14 — `PYTHONPATH` pinned, `pyscf.__version__` asserted.
