# 17-12 — multigrid v2 (`MultiGridNumInt2`) — SUMMARY

**Status: COMPLETE — kernel-side gates GREEN (8/8), host-side Gate E GREEN
(10/10) on the plan's own fixtures, on this machine.** Written 2026-09-02.
The first version of this summary (same date, earlier session) recorded the
host suite as UNVERIFIED because every density-bearing test was SIGKILLed
(exit 137). That was not an environment limit; it was the collocation's
memory shape, and fixing it exposed three real defects that the OOM had
been hiding. All four are recorded below because "the code has never been
observed to fail" was true only because it had never been observed to run.

## What shipped

| file | role |
|---|---|
| `crates/pyscf-kernels/src/multigrid_pair.rs` | `collocate_pairs` (per-slot), `collocate_pairs_rho` / `collocate_pairs_integrate` (reduction fused into the kernel — the `grid_collocate_drv` / `grid_integrate_drv` pair), plus a grouped variant bit-identical to the per-slot one |
| `crates/pyscf-kernels/src/multigrid_gspace.rs` | `gradient_gs`, `get_gga_vrho_gs` |
| `crates/pyscf-kernels/tests/multigrid_pair.rs` | 8 kernel gates |
| `crates/pyscf-pbc-dft/src/multigrid/pair.rs` | `PairTaskList` / `GridLevelSpec`, `PairLevelTable` (host-side fused-pair geometry), block-streamed `pairlevel_rho` / `pairlevel_pass2`, `MultiGridNumInt2` |
| `crates/pyscf-pbc-dft/src/multigrid/pp.rs`, `utils.rs` | `pp.py`, `utils.py` |
| `crates/pyscf-pbc-dft/tests/multigrid2.rs` | 10 host gates |

ALG-06 held: kernels in `pyscf-kernels`, `pyscf-pbc-dft` names no cubecl
crate. The four C destructors are `Drop`/ownership by design (module doc).

## The OOM, and what it was hiding

**Memory.** `collocate_pair_level` materialised one f64 per `(kernel slot ×
grid point)`, where a kernel slot is every `(pair, image, ci, cj, monomial)`
term. On the 25³ Gate-E cells that is **192 GiB (si) / 231 GiB (diamond)**
across the three populated levels — the exit-137. Now measured by
`pair_level_tables_stream_under_budget`, which prints the dense size each
run and asserts no launch's working set exceeds 256 MiB (largest observed:
42 MiB).

Fix, in three layers, each gated:

1. **Density-contracted fused terms.** The density matrix is contracted
   into per-`(instance, monomial)` coefficients on the host BEFORE
   collocation — what upstream's `grid_collocate_drv` does with its dm
   argument — so the kernel never sees a `(ci, cj)` index.
2. **Reduction inside the kernel.** `collocate_pairs_rho` sums its block's
   terms per grid point in one lane; `collocate_pairs_integrate` integrates
   the weighted grid per instance in one lane. No `(slot × point)` buffer
   exists anywhere. Gated: fused kernels vs the materialised per-slot
   values at 1e-13, and the adjoint identity `<integrate(w),1> = <w,rho>`
   on the fused pair with no reference at all.
3. **Spatial blocks.** Each level's mesh is cut into ~5³-point blocks; a
   block sees only the images whose cutoff ball reaches it (a per-axis slab
   distance, a lower bound on the true distance, so nothing that reaches is
   ever dropped). This is upstream's rcut sub-mesh turned inside out.
   Evaluates 35–63 % of the dense product on these cells.

**Ordering note (deviation from Task 0's "`oracle_sum` on every grid
accumulation").** The in-kernel sums are strictly sequential in table
order, not `oracle_sum`'s pairwise-128 tree. The property `oracle_sum`
exists for — bit-identity under any thread count — holds regardless: the
block partition and each block's slot list are geometry only, and
`eval_rho_g_is_bit_identical_across_thread_counts_v2` passes at 1, 2, 3
and 8 rayon threads. The pairwise host reduction needed the dense buffer,
which is the thing this plan could not afford. Recorded, not hidden.

Peak RSS for the density test went **>30 GiB (killed) → 0.46 GiB**; wall
time for one density evaluation 130 s (first streamed version, dominated by
per-launch buffer copies) → **7–9 s**.

**Defect 1 — double-scaled coefficients.** `Decontracted::expand` already
carries `raw_coeff · common_fac_sp` in every row (the convention v1's
`colloc.rs` documents with `pshell_coef = 1.0`); the fused pair prefactor
multiplied `p.coef · q.coef` in a second time. `∫rho` came out at **0.53 of
8.73 electrons** on diamond. Fix: the slot coefficient is `K · fx·fy·fz`
alone; `p.coef · q.coef` is used for screening magnitudes only.

**Defect 2 — no periodic wrap of the fused Gaussian.** Each product
Gaussian was evaluated at one centre `P` inside the cell; the tail that
belongs on the other side of a cell boundary was lost (an atom at the
origin kept one octant). Upstream's C indexes its sub-mesh modulo the mesh.
Fix: every image `P + L1` whose own radius reaches the mesh is its own
kernel instance sharing the term's coefficients. First cut of this used the
primitives' radii (~1000 images per Gaussian, 6-minute densities); the
shipped version uses the fused function's own radius
(`fused_radius`: where `|C| r^k e^{-eta r²}` drops below `precision ·
EXTRA_PREC`).

**Defect 2b — the wrap assumed a `[0,1)³` grid.** This port's
`get_uniform_grids` is origin-centred (fractions in `[-0.5, 0.5)`); the
image test against `[0,1]³` dropped every image on the negative side.
Residual before/after: si `6.1e-3 → 7.4e-5`. Found by the per-pair
brute-force gate (below), which printed sample points with negative
coordinates. The image box is now measured from the mesh's coordinates.

**Defect 3 — polynomial-blind image pre-screen.** Relative images `L` were
kept on `K · scale ≥ thr` alone; for `l > 0` pairs the binomial shift
multiplies the fused coefficient by up to `|L|^{l_p+l_q}`, so far `p-p`
images with NEGATIVE shifted terms were dropped and `∫rho` was 7e-5 too
LARGE. The pre-screen now bounds the shifted peak
(`pair_prescreen_bound`); the exact per-instance radius does the real
screening. Residual: si `7.4e-5 → 3.1e-7`.

## Verified — kernel side (`cargo test -p pyscf-kernels --release --test multigrid_pair`)

**8/8 green.** The original five (adjoint identity ×2, single slot vs
direct formula, `gradient_gs` vs its einsum, `get_gga_vrho_gs` vs its
formula) plus: grouped kernel bit-identical to per-slot; fused `rho` /
`integrate` vs per-slot values at 1e-13; adjoint identity on the fused pair.

## Verified — host side (`cargo test -p pyscf-pbc-dft --release --test multigrid2`)

**10/10 green, 416 s serialised.** Cells: `common::diamond()` /
`common::silicon()` (both gth-szv / gth-pade, no core) at `mesh = 25³`.

| test | plan task | result |
|---|---|---|
| `pair_task_list_is_sane` | 1 | 16 pshells, 256/256 pairs, per level `[0, 4, 12, 240]` (both cells) |
| `pair_level_tables_stream_under_budget` | memory | dense would be 192 / 231 GiB; max launch 42 MiB; 35–63 % of dense evaluated |
| `fused_pairs_match_brute_force_periodic_products` | 2 (pair form of 17-11's per-`l` tests) | worst ordered pair `max|fused − brute|` 7.0e-9 (si, diffuse α=0.0576 s-s); sum over pairs 9.4e-9 |
| `v2_rho_g_matches_v1_with_and_without_the_ladder` | 1/2 | si: ladder 3.1e-7, single finest level 1.3e-6; diamond 1.4e-6 / 1.7e-6 (max over G vs v1) |
| `int_rho_matches_tr_dm_s_v2` | 2/3 | si 3.1e-7, diamond 1.4e-6 (v1: 2e-11) |
| `gate_e_get_j_vs_reference_v2` | 5, Gate E | max|Δ| vs FFTDF: **diamond 1.24e-8, si 6.80e-8** (gate 1e-3) |
| `gate_e_nr_rks_lda_vs_reference_v2` | 5, Gate E | Δnelec 1.5e-6 / 4.8e-7, Δexc 7.9e-7 / 1.3e-7 (gate 1e-3) |
| `v1_vs_v2_gap_reported` | 5 | max|v1.get_j − v2.get_j| **diamond 1.46e-8, si 7.41e-8** |
| `gate_e_speed_ratio_reported_v2` | 5 | see table below |
| `eval_rho_g_is_bit_identical_across_thread_counts_v2` | D-PBC-17 | bit-identical at 1/2/3/8 threads |

**Gate E, accuracy and speed in one table (the plan's requirement):**

| cell | v2 `get_j` vs FFTDF | v1 vs v2 | 17-01 upstream v2-vs-FFTDF floor | wall `get_j`: ref / v1 / v2 | ref/v2 | v1/v2 | 17-01 upstream ref/v2 |
|---|---|---|---|---|---|---|---|
| diamond | 1.24e-8 | 1.46e-8 | ~2e-8 | 0.51 s / 0.34 s / 21.8 s | 0.023× | 0.016× | 0.18–0.39× |
| si | 6.80e-8 | 7.41e-8 | ~1.5e-7 | 0.46 s / 0.37 s / 16.5 s | 0.028× | 0.022× | 0.18–0.39× |

Accuracy: the port's v2-vs-reference and v1-vs-v2 numbers sit AT 17-01's
measured upstream floors (2e-8 diamond, 1.5e-7 si) — the inter-route gap is
reproduced to within a factor ~2 on both cells, the same "strongest single
piece of evidence" role 14-VERIFICATION Gate 3 gave GDF-vs-RSDF. The
per-electron-count floor (~1e-6) is the screening threshold
`precision · EXTRA_PREC = 1e-10` per image, upstream's own rule; v1 sits at
1e-11 because it screens per PRIMITIVE at `precision / vol`.

Speed: **v2 is ~40× slower than the reference route and ~50× slower than
v1 on these cells, i.e. ~10× further off than upstream's own v2 floor.**
Phase 18 must read this as: require v2 for `isinstance` fidelity only.
Where the time goes, for whoever picks it up: 35–63 % of the dense
`(image × monomial × point)` product is still evaluated (blocks are 5³;
upstream's sub-mesh is per Gaussian), and every image re-derives its
monomials from one `exp` — a Hermite recursion would share more.

## Scope reductions, inherited from 17-11 and stated

Gamma-point only; `MultiGridNumInt2` is not wired into the SCF driver as a
selectable `numint`; the IBZ path through `pp.rs` is not connected to
17-05's `KPoints`. The task-list membership rule (coarsest level whose
cutoff resolves `max(ke_p, ke_q)`) remains a documented reformulation of
upstream's C `build_task_list`, priced by the ladder-vs-single-level gate
(they agree to ~1e-6 on the electron count).

## Verification list, as run

- `cargo test -p pyscf-kernels --release` — all green (8/8 in `multigrid_pair`).
- `cargo test -p pyscf-pbc-dft --release --test multigrid2` — 10/10 green.
- `cargo run -p xtask --bin check-dependency-wall` — PASS (cubecl containment intact, ALG-06); `check-no-fma` — PASS (FOUND-05). Clippy clean on every touched file; touched files rustfmt'd.
- Adjoint identity 1e-13 ✓; `gradient_gs` einsum 1e-14 ✓; Gate E ✓; v1-vs-v2 ratio ✓; bit-identity ✓; speed table ✓.
- `PBC-MASTER-PLAN §8.10` now records the `MultiGridNumInt2` ↔ Phase 18 dependency.

## Carry-overs

1. Wire `MultiGridNumInt2` into the KRKS driver as a selectable `numint` (shared with 17-11).
2. Connect `pp.rs`'s IBZ path to 17-05's `KPoints`.
3. Speed: per-Gaussian sub-mesh (or finer blocks with batched launches) and a Hermite recursion — only if Phase 18 needs v2 for more than `isinstance`.
