# 14-02 SUMMARY — `gdf_builder`: the compensating charge, `j2c`, `j3c`, `cderi`

**Status:** SHIPPED. **Date:** 2026-08-29.
**Green:** `cargo test -p pyscf-pbc-lib --test kk_adapted` (8),
`cargo test -p pyscf-pbc-df --test gdf_builder` (15 + 1 slow-ignored),
`PYSCF_ORACLE_VENV=1 … --include-ignored` (the 1e-11 gate).
`check-orphan-modules` PASS (293 files).

## What shipped

| Module | Content |
|---|---|
| `pyscf-pbc-lib/src/kpts_helper.rs` | `unique_with_wrap_around`, `group_by_conj_pairs`, `kk_adapted_iter`, `KkGroup` — `kpts_helper.py:144-268`. Phase 13 named these and did not ship them. |
| `gdf_builder/eta.rs` | `guess_eta`, `estimate_eta_min`, `estimate_eta_for_ke_cutoff`, `estimate_ke_cutoff_for_eta`, `estimate_rcut` — `gdf_builder.py:888-1062` |
| `gdf_builder/fuse.rs` | `make_modchg_basis`, `fuse_auxcell`, `FusedCell`, `auxbar`, `compensate_nuccell` — `:729-931` |
| `gdf_builder/j2c.rs` | `get_2c2e`, `decompose_j2c`, Cholesky/eigen routes, `weighted_coulg` — `:139-196`, `rsdf_builder.py:215-247` |
| `gdf_builder/j3c.rs` | `outcore_auxe2`, `weighted_ft_ao`, `gen_j3c_loader`, `add_ft_j3c`, `solve_cderi`, `gen_uniq_kpts_groups`, `make_j3c`, `Cderi` — `:198-495`, `rsdf_builder.py:830-1011` |
| `gdf_builder/mod.rs` | `CcGdfBuilder` and the `exclude_dd_block` refusal seam |

## THE GATE — plan 14-01's retired 1e-11 lands here, and it PASSES

He-fcc/`sto-3g`, gamma, against upstream `_CCGDFBuilder` with
`exclude_dd_block = False`:

| quantity | port vs upstream |
|---|---|
| **`fuse(j3c)`**, both screens at 1e-14 | **1.412e-12** |
| **`j2c`** | **7.105e-14** |
| `fuse(j3c)` vs upstream's DEFAULT screen | 1.917e-9 (recorded as an upper bound, attributed below) |

and, oracle-free, against `measurements/params.py`:

| quantity | port | upstream |
|---|---|---|
| `eta` / `mesh` / `ke_cutoff`, all three cells | exact to 1e-12 | — |
| `fused_cell.nao` / `.nbas` | 126/42, 32/12 | 126/42, 32/12 |
| `auxbar` nnz / ‖·‖ | 12 / 0.2301278797, 4 / 0.3187837521 | same |
| `gdf_builder::estimate_rcut` | 16.729034885581783 / 10.750308556151602 | same |
| `‖j2c(k=0)‖` | 9.774955865744985 / 10.064640251330108 | same |
| `j2ctag` | `CD` on both | `CD` |
| **`‖cderi[0,0]‖`, He-fcc 2×2×2** | **0.6068683433161949** | same |

`kk_adapted_iter` reproduces upstream's grouping exactly on 2×2×2 (8
self-conjugate groups, every `kj` permutation), gamma, 2×1×1 and 3×1×1 (where
the first NON-self-conjugate group appears).

## Three defects the plan's own tests caught

1. **`libcint deduplicates identical basis blocks across atoms of the same
   element.**** `make_modrho_basis` rewrites `_env` contraction coefficients in
   place, and two carbons share one `PTR_COEFF` slot — so the naive per-shell
   loop scaled the same entries twice, leaving `_env` SQUARED for the second
   atom. Diamond's auxiliary metric came out `‖j2c‖ = 4495` against upstream's
   `251.96191223`, **with atom 0 exactly right and atom 1 untouched**. Keying
   the normalisation on the coefficient pointer makes it idempotent. He-fcc has
   one atom and could never have caught this — the flagship system did.
2. **The shell-pair neighbour list must be aggregated per auxiliary ATOM.** A
   compact auxiliary function and its diffuse model charge share a centre and
   are subtracted from each other; screening them by their own radii gave them
   different image lists. `fuse(j3c)[0]` came out −3.408 against upstream's
   0.942475 — with the COMPACT auxiliaries wrong and the diffuse ones already
   exact, which is the diagnostic signature.
3. **Same for the Gaussian-product prescreen**, and it was worth more:
   `fuse(j3c)` = [−4.90, −4.88, −5.17, −0.72] against [0.942, 0.805, 0.511,
   0.103]. Both aggregations are strictly more conservative than the per-shell
   bound, and they are what upstream's `strip_basis` does implicitly (its
   `estimate_rcut` returns one radius per ORBITAL shell and takes a single
   global `aux_exps.argmin()`).

All three are recorded with their numbers in `measurements/README.md`
§ "Recorded during 14-02".

## The 1.98e-9 residual is upstream's screen, and that is MEASURED

The port and upstream's default route differ by a **P-independent** 1.98e-9 —
1.981e-9, 1.978e-9, 1.9e-9 across auxiliary functions whose exponents span
66.2 → 0.82. P-independence is the `q_P · S_μν` signature, the same one 14-01
found in the raw tensor.

`Int3cBuilder.direct_scf_tol = None` derives
`cell.precision / lattice_sum_factor² · 0.1` = **1.46e-11** here — four orders
LOOSER than the port's 1e-14 prescreen. Set it to 1e-14 and upstream moves to
0.94247478635665 against the port's 0.94247478635764: **9.9e-13**. Sweeping the
port's own prescreen 1e-14 → 1e-20 moves nothing (saturated), and the port's
`fuse(j3c)` is rcut-converged from ×1.3 to ×2.5. So it is upstream discarding a
term the port retains, not the port being wrong — and the test suite asserts
BOTH numbers so the attribution cannot rot.

## Deviations from the plan

* `FusedCell` is built as ONE cell with the model-charge shells appended to each
  element's basis, not two cells concatenated: `Cell::build` goes through a
  per-element basis map, so `gto.conc_env`'s `[all aux | all chg]` layout is not
  reachable. The fused layout is atom-major and `fuse` works through explicit
  index maps (`aux_ao`, `partner`) instead of a slice split. The auxiliary AOs
  keep the auxiliary cell's own ORDER, so `fuse`'s output is indexable exactly
  as upstream's is.
* One normalisation pass covers both halves: `make_modchg_basis` writes
  `half_sph_norm / gaussian_int(2l+2, eta)`, which is what `apply_modrho`
  computes for a single-primitive shell. No second code path.
* `decompose_j2c` uses `pyscf_algebra::{zcholesky, zeigh_gen}` rather than a
  hand-rolled factorisation (D-PBC-04 keeps the complex algebra in one crate).
* The diamond `cderi` fingerprint is `#[ignore]`d as a slow acceptance run —
  see the carry-over below. He-fcc carries the per-commit gate.

## Performance — FIXED, 8.4x on the k-point build (added 2026-08-29)

Two changes; full tables in `measurements/README.md` § "Recorded during the
14-02 performance pass".

1. **`get_2c2e` and `outcore_auxe2` were called PER GROUP where upstream calls
   each ONCE** (`rsdf_builder.py:930`, `:853`). Both `aux_e2` and `pbc_intor`
   fold every lattice image into EVERY requested k-point in one sweep, so
   `nkpts` calls cost `nkpts` times one call and return identical numbers. This
   was a faithfulness bug as much as a performance bug. **He-fcc 2×2×2, all 64
   k-pairs: 135 s → 16.0 s.**
2. **The bra-image loop is threaded** with `std::thread::scope`, each worker
   accumulating into its own output and the partials reduced in chunk order — so
   the result does not depend on the thread count (FOUND-06 by construction).
   Worth 1.6x on He (14.85 s → 9.20 s at 4 threads) and it must be SIZED: the
   first attempt split 16 ways and made He *slower* (2.56 s → 7.30 s) because
   `EvaluationContext::new()` costs ~0.3 s. The pool is sized by the estimated
   triple count.

Every gate was re-run after both changes and is unmoved: `fuse(j3c)` 1.448e-12,
`j2c` 7.105e-14, `cderi` 0.6068683433161949.

## Carry-overs

* **Diamond at GAMMA is still slow**, because it is ONE group so the hoisting
  buys nothing there: `429 images² × 10 s2 shell pairs × 42 fused auxiliary
  shells ≈ 77 million` cintx `SessionRequest`s at ~28 us. The threading engages
  (a release run showed `user 280m59s` against `real 22m20s` — 12.6 of 16 cores
  busy) but the run was terminated at 22 min before finishing, so **diamond's
  wall time is unmeasured** and that number is owed. The per-request cost is a
  floor —
  `cintx::ShellTuple` is fixed-arity, so a shell RANGE cannot be requested in
  one call, and ~15 us of the 28 is fixed overhead. Cutting further needs either
  a batched cintx entry point or upstream's `direct_scf_tol` Schwarz screen
  (2.05e-12 for diamond) applied to a per-atom-aggregated bound.
* `j2c_negative` / `cderi_negative` are carried as fields and never populated —
  the 2-D truncated-Coulomb path is refused per 14-CONTEXT's non-goals.
* `exclude_dd_block = true` returns `NotYetImplemented { phase: 17 }` (D-PBC-23).
* The pre-existing `pyscf-dft --lib hooks::tests::define_xc_string_form_parses`
  failure is unchanged and unrelated.
