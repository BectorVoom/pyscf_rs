# 20-06 SUMMARY — rsjk re-assessment: still blocked (branch B), ω half corrected

**Executed:** 2026-09-13. Branch **B** taken. Refusals stay; carryover and
refusal message now name the real blocker. One silently wrong shipped number
found and fixed along the way (deviation, below).

## Task 1 — sufficiency check

Upstream: `pyscf/pbc/scf/rsjk.py` (vendored 2.12.1) `build` (`:136-238`) and
`_get_jk_sr` (`:267-435`), plus the `libpbc` C they call. "absent" includes
"a similar Rust function exists but computes something different".

| # | upstream construct (file:line) | Rust equivalent | verdict |
|---|---|---|---|
| 1 | `_guess_omega` (**rsjk.py:1263**, rsjk's own — not rsdf_builder's) | was `rsdf_builder::guess_omega` (`omega.rs:337`, ports `rsdf_builder.py:1330`) — DIFFERENT formula: He-fcc 2×2×2 ω 0.73936/mesh 11 vs upstream **1.312754030266949/mesh 15**; diamond 0.60196 vs **0.7645412133957928** | was wrong → **fixed** (`crates/pyscf-pbc-scf/src/rsjk.rs` `guess_omega`) |
| 2 | `estimate_ke_cutoff_for_omega` (rsjk.py:1293) / `estimate_omega_for_ke_cutoff` (:1306) | rsdf_builder versions (`omega.rs:109/:129`) are different formulas | was wrong → **fixed** (rsjk.rs) |
| 3 | `rsdf_builder._estimate_meshz` (2-D, rsjk.py:153) | `rsdf_builder/omega.rs:162` | present |
| 4 | `ft_ao._RangeSeparatedCell.from_cell(.., in_rsjk=True)` (ft_ao.py:267) | `pyscf-pbc-df/src/ft_ao/rs_cell.rs:117` (`in_rsjk` → `estimate_ke_cutoff_pgto_4c`, :194) | present |
| 5 | `k2gamma.kpts_to_kmesh` (k2gamma.py:39) | none (`grep` finds only a doc mention, `kpts_helper.rs:314`) | **absent** |
| 6 | `direct_scf_tol` lattice-sum cutoff (rsjk.py:177-180) | none (trivial) | absent |
| 7 | `rsjk.estimate_rcut` (rsjk.py:1182) — 4-centre SR radius | `rsdf_builder::estimate_rcut` is the 3-centre `(rs_cell, rs_auxcell)` one (rsdf_builder.py:1418) | **absent** |
| 8 | `ExtendedMole.from_cell` + `strip_basis` (ft_ao.py:596/631) | `ft_ao/supmol.rs:96/:179` (gated, plan 17-10) | present (mask/segment level) |
| 9 | supermole `_atm/_bas/_env` of translated shells, `PTR_EXPCUTOFF`, `omega=-ω` (ft_ao.py:614-628, rsjk.py:186) | `supmol.rs:57` is compact — no shell table any integral can be called on | **absent** |
| 10 | SR `int2e` with `env[8]<0` | cintx scalar route honours `ExecutionOptions::range_omega` (incl. `sr_rys_roots_host`, `cintx-cubecl/src/math/rys_wheeler.rs:4690`); pyscf-gto consumer `intor_with_options` (`pyscf-gto/src/intor.rs:94`) is whole-Mole dense only; quartet-batch route REFUSES omega (`cintx-rs/src/api.rs:3405`) | primitive present, no per-quartet periodic consumer |
| 11 | `_vhf.make_cintopt` / `with_integral_screen` (rsjk.py:208-216) | none | absent |
| 12 | `PBCVHFnr_int2e_q_cond` (pyscf/lib/pbc/nr_direct.c:1037) — Schwarz `qindex[0:2]` | none (`grep q_cond\|qindex` finds only doc text; `rsdf_helper.py`'s `get_q_cond` is NOT what rsjk uses) | **absent** |
| 13 | `PBCVHFnr_sindex` (nr_direct.c:1120) — `qindex[2]` | none | **absent** |
| 14 | dd-block `INDEX_MIN` mask (rsjk.py:230-233) | `ExtendedMole::bas_type_to_indices` `supmol.rs:299` present; mask itself absent | partial |
| 15 | `_sort_qcond_cell0` / `_qcond_cell0_abstract` (rsjk.py:255/:1336) | none | **absent** |
| 16 | nodddd AO remap (`lib.locs_to_indices`, `take_2d`, rsjk.py:299-316) | `RsCell::recontract2d` `rs_cell.rs:546` (different direction) | absent |
| 17 | `sc_dm` BvK transform, `expLk` (rsjk.py:330-351) | none | absent |
| 18 | `k2gamma.double_translation_indices` (k2gamma.py:104) | none | **absent** |
| 19 | `lib.condense('NP_absmax')` `dmindex` (rsjk.py:358-368) | none | absent |
| 20 | `PBCVHF_direct_drv` / `PBCVHF_direct_drv_nodddd` (nr_direct.c:721/:855) | none | **absent** |
| 21 | `PBCVHF_contract_{j,k,jk}_{s1,s2kl}` (nr_direct.c:38-530) | none | **absent** |
| 22 | `approx_bvk_rcond0` / `PBCapprox_bvk_rcond` / `qindex_abstract` (nr_direct.c:531/:591/:688) | none | **absent** |
| 23 | `PBCint2e_sph` (pyscf/lib/pbc/cint2e.c:330) | none | **absent** |
| 24 | `vs` k-phase back-transform, `kpts_band` (rsjk.py:421-434) | none | absent |
| 25 | LR: `coulG − coulG_SR` + `π/ω²` G0 (rsjk.py:596-612) | `aft_jk::get_j_kpts/get_k_kpts(omega)` (`aft_jk.rs:44/:126`) — full image list, single-kernel coulG | absent (different computation) |
| 26 | `_ExtendedMoleFT` (rsdf_builder.py:1312), `smooth_basis_cell` | `rs_cell.rs:388` present; `_ExtendedMoleFT` none | partial |
| 27 | `dm_factor` K path, `_mo_k2gamma`, `_update_vk_dmf`, `PBC_ft_fuse_dd_s1` (rsjk.py:817-1170) | none | absent |
| 28 | `_purify` (rsjk.py:1172) | none | absent |

**Rows 5, 7, 9, 12-13, 15, 18, 20-23 (and the LR rows) are absent → branch B.**
The premise "RsCell + ExtendedMole unblock rsjk" does not hold: those are the
two rows that exist. What is missing is `rsjk.py`'s own screened SR body —
effectively a port of `pyscf/lib/pbc/nr_direct.c` + `cint2e.c` on a
materialised supermole. Upstream `supmol_sr` sizes (2.12.1, measured):
He-fcc sto-3g 2×2×2 **98 shells**, `bas_mask (8,2,79)`; diamond gth-szv 2×2×2
**2094 shells**, `bas_mask (8,8,249)`.

Branch A was not attempted, so no He-fcc gate at 1.353e-08 was run.

## Task 3 — carryover

`.planning/carryovers/D-PBC-24-cintx-range-omega-PLAN.md` `status:` block: the
"BLOCKED ON PHASE 17" bullet replaced by the six missing items above with
upstream `file:line` and host files, the stale supermole claim deleted, and the
ω fix recorded.

## Task 4 — refusal message

`RS_BUILDER_GAP` (`crates/pyscf-pbc-df/src/rsdf_builder/mod.rs`, string value
only) now says: not blocked on integrals or supermole types; missing
`rsjk.estimate_rcut`, the libcint-shaped supermole, `PBCVHFnr_int2e_q_cond` /
`PBCVHFnr_sindex`, `_qcond_cell0_abstract`, `PBCVHF_direct_drv` +
`PBCVHF_contract_*_s2kl` + `PBCint2e_sph`, BvK `dm_translation`/`dmindex`, and
rsjk's LR composition. Still contains `range_omega` and `env[8]` (existing test
assertion). Only test asserting the text: `crates/pyscf-pbc-scf/tests/rsjk.rs`
(grep of `crates python xtask`); extended to require the new names.

## Deviation — ω half was silently wrong (fixed)

`rsjk.py:145-150` resolves `_guess_omega` / `estimate_ke_cutoff_for_omega` to
rsjk.py's OWN module-level definitions, not rsdf_builder's. The shipped
`RangeSeparatedJkBuilder::guess_omega` returned RSDF's numbers and its test
pinned them. Ported `guess_omega` (rsjk.py:1263), `estimate_ke_cutoff_for_omega`
(:1293), `estimate_omega_for_ke_cutoff` (:1306, with OMEGA_MIN clamp) into
`crates/pyscf-pbc-scf/src/rsjk.rs`; the method now mirrors `build :142-156`
(explicit ω discards a preset mesh, as upstream does; 2-D meshz). Oracle
(`PYTHONPATH=$PWD .venv/bin/python`, pyscf 2.12.1 from `pyscf/`):

| case | ω | mesh | ke_cutoff |
|---|---|---|---|
| He-fcc 2×2×2 | 1.312754030266949 | 15 | 60.18879248031004 |
| He-fcc Γ | 1.533462187962161 | 17 | 78.613933035507 |
| He-fcc 2×2×2, mesh 9 | 0.7176440705624877 | 9 | 19.65348325887675 |
| diamond 2×2×2 | 0.7645412133957928 | 11 | 21.721883440437864 |
| diamond 2×2×2, mesh 9 | 0.6043412565112721 | 9 | 13.902005401880233 |
| He, ω=0.5, preset mesh 5 | 0.5 | 7 | 9.545851180141373 |
| He, ω=0.3, preset mesh 5 | 0.3 | 5 | 3.531257370504433 |

Gated in `tests/rsjk.rs` (`rsjk_guesses_rsjk_py_omega_not_rsdf`, ω at 1e-12,
ke at 1e-10; `an_explicit_omega_overrides_the_guess`). No other caller of
`RangeSeparatedJkBuilder::guess_omega` exists (it is not bound in Python).

## Files touched

- `crates/pyscf-pbc-scf/src/rsjk.rs` — module docs (Task-1 result), three ported
  ω helpers, `guess_omega` rewired; refusals unchanged.
- `crates/pyscf-pbc-scf/tests/rsjk.rs` — ω targets corrected, refusal assertion
  extended, stale header marked historical.
- `crates/pyscf-pbc-df/src/rsdf_builder/mod.rs` — `RS_BUILDER_GAP` value only.
- `.planning/carryovers/D-PBC-24-cintx-range-omega-PLAN.md` — `status:` block.

`rustfmt --edition 2024` on the two rsjk files; `--check` on `mod.rs` clean.
No git operations (D-20-A).

## Verification

Build protocol §2 (`target/gate`, `LTO=false`). The first build rebuilt the
libxc tree under concurrent load (92 min).

1. **rsjk tests** — `CARGO_TARGET_DIR=$PWD/target/gate CARGO_PROFILE_RELEASE_LTO=false cargo test --release -p pyscf-pbc-scf --test rsjk -j 6`
   → **exit=0**; `test result: ok. 5 passed; 0 failed` (includes
   `rsjk_guesses_rsjk_py_omega_not_rsdf`, `an_explicit_omega_overrides_the_guess`,
   `rsjk_refuses_and_names_what_is_unported`, `the_partitioning_variants_are_a_named_non_goal`).
   Log `target/gate-20-06-rsjk.log`.
2. **Refusals still present** — `grep -n 'NotYetImplemented' crates/pyscf-pbc-scf/src/rsjk.rs`
   → `build` `phase: 14` at **:275**, `get_jk` `phase: 14` at **:294**
   (moved from :149/:163 because three ω helpers were inserted above them —
   same two refusals, same text source). `grep -n 'phase: 19'` → `get_jk_mpi`
   at **:309**, still refusing.
3. **`RS_BUILDER_GAP` names no closed blocker** — it now states integrals and
   supermole types are NOT the blocker, and lists items 1-6.
4. **Regression** — `CARGO_TARGET_DIR=$PWD/target/gate CARGO_PROFILE_RELEASE_LTO=false cargo test --release -p pyscf-pbc-scf --test kscf --test gate3_rsdf --test rsjk -j 6`
   → **exit=101**, NOT from this plan:
   - `gate3_rsdf`: `ok. 0 passed; 0 failed; 3 ignored` (all its gates are `#[ignore]`).
   - `kscf`: `FAILED. 8 passed; 1 failed; 4 ignored` (908 s) —
     `supercell_equivalence_holds` panicked at `kscf.rs:139`:
     `supercell equivalence broken: -10.347315387195717 vs -10.531064341612561`
     (diamond gth-pade, FFTDF, mesh 15 / 30×15×15).
   - `rsjk` not reached in this invocation (cargo stops at the first failed
     target); passed in (1).

   **Attribution:** this plan's diff cannot reach that test. `kscf.rs` never
   names `rsjk`/`RangeSeparatedJkBuilder`/`RS_BUILDER_GAP`; the only reference
   to `rsjk` in `pyscf-pbc-scf/src` outside `rsjk.rs` is the `lib.rs:30/:47`
   module/re-export; `RS_BUILDER_GAP` is read only by the two refusals. The
   0.18 Ha break is in the FFTDF/GTH-PP path, which the working tree changes
   uncommitted (Phase-18 paused work: `pyscf-pbc-df/src/fftdf.rs` +245/−57,
   `fft_jk.rs`, `pyscf-pbc-gto/src/pseudo/{vnl,vloc_part2,mod}.rs`,
   `eval_gto.rs`). Not investigated further here (out of scope, D-20-A forbids
   reverting others' files); **flag for the Phase-18 owner / 20-VERIFICATION**.
   Log `target/gate-20-06-regress.log`.
