# Phase 13 Context — `ft_ao` + AFTDF

**Milestone:** v2.0 PBC · **Depends on:** Phases 9–12 (shipped) · **Blocks:** Phase 14 (GDF/MDF/RSDF), Phase 15 (KMP2), Phase 16 (KCCSD)
**Master plan:** `.planning/pbc/PBC-MASTER-PLAN.md` — read §0, §3 (D-PBC-09/10/21/22), §6 (K-15), §8.5 and §11 before starting.

## Goal

The analytic Fourier transform of an AO pair exists, is a cubecl kernel, and is
correct; `AFTDF` is a `PeriodicDf` implementor; and `KRHF(cell, kpts)` runs on
**either** builder with no driver change and gets the same answer.

Everything Phase 14 builds — GDF's compensating-charge scheme, MDF, RSDF — sits
directly on `ft_aopair`. A defect that survives this phase is a defect in every
production density-fitting path in the milestone.

## Success criteria (all must be TRUE to close the phase)

1. **Gate 1 (no oracle), in three parts** — see "The two gates the roadmap gets
   wrong" below for the measurements that force this.
   `ft_aopair[μν, G=0]` vs the periodic overlap, diamond/`gth-szv` `2×2×2` and
   He-fcc/`sto-3g` gamma:
   **1a** ≤ **2e-9** at `rcut = estimate_rcut(cell)` vs `pbc_intor` (upstream
   itself measures 1.554e-9 / 5.322e-10);
   **1b** ≤ **2e-10** at `rcut = 1.5 × cell.rcut` vs `pbc_intor` (floor 1.472e-10,
   set by `pbc_intor`'s own truncation — a consistency check, not a kernel gate);
   **1c** ≤ **1e-13** at `rcut = 1.5 × cell.rcut` vs `intor_cross_with_images` over
   the **same `Ls`**. **1c is the real gate on the McMurchie–Davidson algebra.**
2. **Gate 2 (no oracle).** `|E_KRHF(AFTDF) − E_KRHF(FFTDF)|` on diamond `2×2×2`
   measured on the **pairs** `(rcut, mesh)` = `(×1.0, 21)`, `(×1.0, 31)`,
   `(×1.5, 31)`, `(×1.5, 41)` — **not** a mesh sweep. It must fall as the pair
   tightens and then sit on the pre-recorded upstream floor for that pair. There
   are TWO floors — AFTDF's `rcut` screening and FFTDF's mesh aliasing — and
   lowering one alone stalls against the other.
3. **Gate 3 (oracle-gated).** `AFTDF.get_nuc`, `.get_pp`, `.get_jk`, `.get_eri`
   and the converged `KRHF` energy match live upstream PySCF **2.12.1** to
   **1e-11**, `PYTHONPATH` pinned to the vendored tree.
4. `ft_aopair` matches a dense-grid numerical FT to 1e-6 for `l ≤ 2`, and the
   `s`-`s` closed form to 1e-14, with no oracle involved.
5. Every driver (`KRHF`/`KUHF`/`KROHF`/`KGHF`/`KRKS`/`KUKS`/`KROKS`/`KGKS`)
   accepts either builder, and the FFTDF path is **bit-identical** to what it
   produced before D-PBC-22 landed.
6. `cargo test --workspace` green; `xtask check-orphan-modules`,
   `check_dependency_wall` and `check_no_fma` green.

## The two gates the roadmap gets wrong — read this first

The ROADMAP says "AFTDF KRHF == FFTDF KRHF to 1e-13"; the master plan's original
13-06 said 1e-6. **Neither is a measurement, and a "monotone convergence in the
mesh" restatement is also wrong** — it would fail a correct implementation.
Measured on upstream 2026-08-28/29, diamond `2×2×2`, fixed init-guess density,
upstream's default `rcut`:

| mesh | `dvj` | `dvk` (`exxdiv='ewald'`) | `dvk` (`exxdiv=None`) |
|---|---|---|---|
| 15 | 3.827e-7 | 1.112e-6 | — |
| 21 | 2.365e-10 | 1.804e-9 | 1.690e-9 |
| 31 | 1.996e-11 | 6.487e-10 | 2.653e-11 |
| 41 | 1.996e-11 | 6.485e-10 | — |

**Mesh 41 is identical to mesh 31 to three digits.** Above mesh ~31 the mesh is
not the controlling parameter at all — the difference plateaus on `ft_aopair`'s
screening residual, and the `rcut` table in the next section shows the other floor.
Two independent floors, so Gate 2 is a `(rcut, mesh)` ladder, not a sweep.

Two further traps in that table. With `exxdiv='ewald'` — the SCF default — the
R-15 G=0 asymmetry is **~96% of `dvk` at mesh 31** (6.2e-10 of 6.487e-10) and
switching exxdiv off drops `dvk` 25×. And **do not characterise any of this at
mesh 21**: there the general screening residual still dominates, the two error
sources partially cancel in the max-abs norm, and the exxdiv term misleadingly
looks like ~1.1e-10.

Energy series measured so far (mesh ≥ 31 was still running at planning time; plan
13-08 Task 0 completes it and pastes it into `13-VERIFICATION.md` §1):

| mesh | `E_FFTDF` | `E_AFTDF` | abs diff |
|---|---|---|---|
| 15 | −10.93090153682113 | −10.93087523588556 | 2.630e-5 |
| 21 | −10.93087319091901 | −10.93087316555834 | 2.536e-8 |
| 31 | −10.93087316795859 | −10.93087316798466 | **2.607e-11** |
| 41 | −10.93087316795859 | −10.93087316798466 | **2.607e-11** |

**Mesh 31 and mesh 41 are BIT-IDENTICAL — all 14 printed digits of both energies.
The floor at upstream's default `rcut` is 2.607e-11 Ha; the roadmap's 1e-13 is
unreachable, three orders out.** Meshes 55/71 add nothing — the ladder stops at 41. It lands with `dvj`'s plateau
(1.996e-11) and the `exxdiv=None` `dvk` plateau (2.653e-11), NOT with the
`exxdiv='ewald'` `dvk` plateau (6.487e-10): the G=0 asymmetry is a near-uniform
shift that largely cancels in `Tr(D·vk)`, so it dominates the MATRIX difference
while barely touching the ENERGY. **Gate matrices and energies at different
levels — do not carry one number across both.** Risk R-16.
Measurement scripts and the full recorded results are committed at
`.planning/phases/13-ft-ao-aftdf/measurements/` — re-run them, don't re-derive.

**Gate 1's 1e-10 is likewise unachievable as stated, and the reason turned out to
be the reference side, not the kernel.** Measured on upstream 2026-08-28/29,
diamond/`gth-szv` `2×2×2`, mesh 31:

| quantity | value |
|---|---|
| `cell.rcut` | 21.319 Bohr |
| `ft_ao.estimate_rcut(cell).max()` | **20.420** Bohr |
| upstream `ft_aopair[G=0]` vs `int1e_ovlp`, gamma | **1.554e-9** |
| upstream, `k ≠ 0` | **5.322e-10** |

Scaling `ft_ao.estimate_rcut` and re-measuring:

| `rcut` | Gate 1 residual | `dvj` | `dvk` |
|---|---|---|---|
| ×1.0 = 20.42 | 1.554e-9 | 1.996e-11 | 6.487e-10 |
| ×1.5 = 30.63 | **1.472e-10** | 7.727e-13 | 1.609e-10 |
| ×2.0 = 40.84 | **1.472e-10** | 7.726e-13 | 1.609e-10 |

×1.5 and ×2.0 are identical to four digits: the FT lattice sum is fully converged
at ~30.6 Bohr and `ft_aopair` stops changing — yet the residual sits at 1.472e-10
and will not move. **That residual belongs to `pbc_intor("int1e_ovlp")`**, which
runs its own lattice sum only to `cell.rcut` = 21.319 at `precision` = 1e-8. Once
the FT side is converged, Gate 1 measures the OVERLAP's truncation error.

So a Gate 1 stated as "≤1e-11 against `pbc_intor`" cannot pass however correct the
kernel is. The fix is already in the port:
`pyscf_pbc_gto::pbc_intor::intor_cross_with_images` takes an explicit image list
(plan 10-06 added it so several operators could share one `Ls`). Building the
reference overlap over the **same `Ls`** as `ft_aopair` puts both sides on
identical images, and then nothing but the McMurchie–Davidson algebra can move the
result — that is variant **1c**, and it is the only one of the three that gates the
kernel.

## Non-goals (do NOT do these in Phase 13)

- `_RangeSeparatedCell` / `ExtendedMole` / the BvK supermole — D-PBC-21 says the
  direct lattice sum is the port. The BvK bucket contraction is a Phase-14
  performance question.
- `int3c2e` over lattice images, `gdf_builder`, `GDF`, HDF5 `_cderi` — Phase 14.
- The four alternative `_update_vk*` variants in `aft_jk.py` — they are
  thread-count and MO-factorisation optimisations producing the same numbers.
  Any path that would select them returns `NotYetImplemented { phase: 14 }`.
- `KMP2` / `KCCSD` consumption of `ao2mo_7d` — Phases 15/16. Phase 13 only fixes
  and tests the tensor's shape contract.
- The full `pyscf.pbc.df` PyO3 surface — Phase 20. Plan 13-07 wires only
  `mf.with_df = AFTDF(...)`.
- `cell.dimension < 3` for the AFTDF paths beyond what Phase 12 already closed.

## Plans and waves

| Wave | Plans |
|---|---|
| 1 | 13-01 (`ft_aopair` MD kernel, K-15), 13-02 (`ft_ao` single-centre FT) |
| 2 | 13-03 (`ft_aopair_kpts`, `FtKernel`, aosym, G-blocking) |
| 3 | 13-04 (`AFTDF`: `build`/`ft_loop`/`pw_loop`/`weighted_coulG`/`get_nuc`/`get_pp`) |
| 4 | 13-05 (`aft_jk`), 13-06 (`fft_ao2mo` + `aft_ao2mo`) |
| 5 | 13-07 (`Box<dyn PeriodicDf>`, D-PBC-22) |
| 6 | 13-08 (verification rollup) |

## What already ships and must be REUSED, not rewritten

| Need | Where it already lives |
|---|---|
| `get_coulG` incl. the `exxdiv='ewald'` fold at `G+k=0` | `pyscf_pbc_gto::coulg::get_coulg` (`coulg.rs:198-203`) |
| `madelung`, `_ewald_exxdiv_for_G0` | `pyscf_pbc_gto::coulg::madelung`, `pyscf_pbc_df::df_jk::ewald_exxdiv_for_g0` |
| `Gv`, `Gvbase`, `kws` | `pyscf_pbc_gto::gv::get_gv_weights` |
| structure factors `SI` | `pyscf_pbc_gto::gv::get_si` (K-02) |
| GTH `V_loc` part 1 in G space | `pyscf_pbc_gto::pseudo::vloc::get_gth_vlocg_part1` |
| GTH `V_loc` part 2 (real space) | `pyscf_pbc_gto::pseudo::vloc_part2::get_pp_loc_part2` |
| GTH `V_nl` | `pyscf_pbc_gto::pseudo::vnl::get_pp_nl` |
| lattice images, `rcut` | `pyscf_pbc_gto::lattice::get_lattice_ls`, `cutoff::estimate_rcut_pgto` |
| periodic overlap for Gate 1 | `pyscf_pbc_gto::pbc_intor::pbc_intor("int1e_ovlp")` |
| cart→sph | `pyscf_gto::cart2sph_coeff` |
| `_format_dms`/`_format_jks`/`_format_kpts_band` | `pyscf_pbc_df::df_jk` (plan 11-07 put them in ONE place) |
| complex GEMM, ordered complex reductions | `pyscf_algebra::{zgemm_dense, oracle_zsum, oracle_zdot}` |
| reference cells | `pyscf_pbc_gto::test_systems` (feature `test-systems`) — do NOT redefine |

## What does NOT exist yet and must be ported inside this phase

- `kpts_helper.kk_adapted_iter` and `group_by_conj_pairs`
  (`pyscf/pbc/lib/kpts_helper.py:170-268`) — `pyscf-pbc-lib` has `unique`,
  `member`, `get_kconserv`, but neither of these. Plan 13-05 owns them.
- `pyscf-pbc-df` does not depend on `pyscf-kernels` yet. Plan 13-01 adds it;
  `pyscf-pbc-gto` already does, so no new dependency-wall exemption is needed —
  but run `xtask check_dependency_wall` and confirm.
- `fft_ao2mo` was skipped in Phase 11. Plan 13-06 ships it, so that AFTDF's
  `get_eri` has an independent same-phase cross-check.

## Standing constraints inherited from v1.0 / earlier PBC phases

- Tests in separate files (AGENTS.md §2). No `mod tests` in a source file. Phase
  12's `lib.rs` declared only `mod error` and shipped zero tests — `xtask
  check-orphan-modules` now makes that a CI failure. Every new module this phase
  adds must be declared and reached by a test.
- cubecl kernels generic over `F: Float`; read the manual at
  `/home/user/Documents/workspace/cubecl_manual/manual/manual/Cubecl/INDEX.md`
  BEFORE writing kernel code (AGENTS.md §3).
- On any cubecl build error, read `cubecl_error_guideline.md` first (AGENTS.md §4).
- **Scalar math inside a `#[cube]` body is `cube-math`**
  (`/home/user/Documents/workspace/cube-math`), `MathConfig::EXACT` — precedent:
  `pbc/struct_factor.rs:46`, `pbc/ewald.rs:217`, `eval_gto.rs:1235`. Host-side
  scalar math is `std` or `rmath`, never `cube-math` (it is a DEVICE libm and
  panics outside a kernel — see `pbc-gto/src/ewald.rs:387`).
- Only `pyscf-algebra`/`pyscf-runtime`/`pyscf-kernels` may name `cubecl-*` (ALG-06).
- `release-oracle` profile stays FMA-free (`xtask check_no_fma`).
- Ordered reductions only (`oracle_sum`/`oracle_zsum`) in numerical paths.
- Complex tensors are PLANAR `CTensor { re, im }` (D-PBC-02); complex GEMM is
  four real GEMMs, never Karatsuba (D-PBC-03).
- The oracle is the **vendored** PySCF 2.12.1 at `<root>/pyscf`, not
  site-packages 2.14 — `PYTHONPATH` pinned, `pyscf.__version__` asserted.
