# Phase 9 Context — PBC Foundation + Complex Algebra

**Milestone:** v2.0 PBC · **Depends on:** Phase 1–7 (shipped) · **Blocks:** every later PBC phase
**Master plan:** `.planning/pbc/PBC-MASTER-PLAN.md` (read §0, §2, §3, §5, §8.1 before starting)

## Goal

The workspace has 39 members, complex linear algebra works and is bit-reproducible,
and a `Cell` can be built and produce: its reciprocal lattice, its G-vector grid,
its lattice-image list, its k-point mesh, and its Ewald energy.

## Success criteria (all must be TRUE to close the phase)

1. `cargo build --workspace` is clean with 39 members; `cargo run -p xtask --bin check_dependency_wall`
   and `check_forbidden_paths` both pass with the new `pyscf-pbc-*` exemption.
2. `pyscf_algebra::zeigh_gen` on a random 8×8 Hermitian pair matches a faer reference
   to 1e-12, and `Cᴴ S C == I` to 1e-12.
3. `oracle_zsum` over 1e6 complex elements is **bit-identical** at `RAYON_NUM_THREADS=1` and `=8`.
4. `Cell::build` on diamond/Si/LiF/He-fcc/graphene produces `vol`, `b`, `rcut`, `mesh`
   matching hard-coded upstream values; `b·aᵀ == 2π·I` to 1e-12.
5. `get_Gv` on a `[5,5,5]` mesh matches upstream element-wise to 1e-12; `|SI| == 1`.
6. `get_lattice_Ls` and `make_kpts` match upstream counts and values; `get_kconserv`
   matches the upstream 8×8×8 table exactly.
7. `cell.ewald()` for diamond matches upstream to **1e-9 Ha** and is invariant to
   `ew_eta` over `[0.5η₀, 2η₀]` to 1e-8.

## Non-goals (do NOT do these in Phase 9)

- Any periodic integral (`pbc_intor`) — Phase 10.
- Any FFT — Phase 11.
- Any SCF — Phase 11.
- `dimension < 3` Coulomb/Ewald branches — Phase 12 plan 12-08 (D-PBC-20).
- Any PyO3 binding — Phase 20 plan 20-05 (D-PBC-14).

## Plans and waves

| Wave | Plans |
|---|---|
| 1 | 09-01 (workspace scaffolding), 09-02 (complex algebra) |
| 2 | 09-03 (`Cell`), 09-04 (cutoffs/mesh), 09-05 (Gv/SI/uniform grids) |
| 3 | 09-06 (lattice Ls / supercell), 09-07 (k-point meshes) |
| 4 | 09-08 (Ewald), 09-09 (verification rollup) |

## Reference systems (define once, in `crates/pyscf-pbc-gto/tests/common/systems.rs`)

`diamond`, `si`, `lif`, `he_fcc`, `graphene` — see `PBC-MASTER-PLAN.md §9.2` for exact
lattice constants, basis and pseudopotential.

## Standing constraints inherited from v1.0

- Tests in separate files (AGENTS.md §2). No `mod tests` in a source file.
- cubecl kernels generic over `F: Float`; read the manual first (AGENTS.md §3).
- On any cubecl build error, read `cubecl_error_guideline.md` before fixing (AGENTS.md §4).
- Only `pyscf-algebra`/`pyscf-runtime`/`pyscf-kernels` may name `cubecl-*` (ALG-06).
- `release-oracle` profile must stay FMA-free (`xtask check_no_fma`).
- Ordered reductions only (`oracle_sum`/`oracle_zsum`) in numerical paths.
