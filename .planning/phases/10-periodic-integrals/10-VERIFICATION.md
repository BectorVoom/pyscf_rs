---
phase: 10-periodic-integrals
type: verification
milestone: v2.0
status: PASS (no blockers; the cintx general-contraction bug of §4 was fixed upstream and is now gated)
verified: 2026-08-26
plans: [10-01, 10-02, 10-03, 10-04, 10-05, 10-06, 10-07, 10-08]
---

# Phase 10 Verification — Periodic integrals + GTH pseudopotentials

Closes PBC-MASTER-PLAN §8.2.

**Upstream reference:** PySCF **2.12.1**, the tree vendored at `<root>/pyscf`,
run through `.venv/bin/python`. Every oracle test is `#[ignore]`d AND gated on
`PYSCF_ORACLE_VENV`, so `cargo test --workspace` never touches Python:

```bash
PYSCF_ORACLE_VENV=1 cargo test -p pyscf-pbc-gto --release -- --ignored
```

**Geometry is specified in BOHR** throughout. `pyscf_core::Unit::Ang` is
CODATA-2014 and upstream is CODATA-2010, so an Angstrom cell differs in the 8th
digit of every lattice vector before an integral is evaluated (same note as
`09-VERIFICATION.md`).

---

## 1. The phase gate

> **Gate:** `pbc_intor('int1e_ovlp', kpts)` matches upstream to 1e-12 on
> diamond 2×2×2.

```
PYSCF_ORACLE_VENV=1 cargo test -p pyscf-pbc-gto --test pbc_intor --release -- --ignored --nocapture
```

| test | quantity | max \|Δ\| vs upstream | tolerance | verdict |
|---|---|---|---|---|
| `ovlp_matches_upstream_on_diamond_222` | `pbc_intor('int1e_ovlp', make_kpts([2,2,2]))`, all 8 k, all 64 elements, re and im | **1.288e-14** | 1e-12 | **PASS** (77× margin) |
| `kin_matches_upstream_on_diamond_222` | `pbc_intor('int1e_kin', …)` | 2.064e-14 | 1e-12 | PASS |
| `ovlp_matches_upstream_on_diamond_321` | 3×2×1 mesh (unequal, non-power-of-two) | 1.288e-14 | 1e-12 | PASS |

Each of those tests also pins the PRECONDITIONS, so a pass cannot come from
comparing two different systems: `nao`, `cell.rcut` (1e-10), `|Ls|` (exact),
`atom_charges()` (exact, PP-adjusted) and every k-point component (1e-12) are
asserted against upstream before a single integral is compared.

---

## 2. Per-plan results

### 10-01 — GTH pseudopotential data model

`crates/pyscf-pbc-gto/src/pseudo/mod.rs`, `cell.rs`, `pyscf-gto/src/basis/cp2k_pp.rs`

| item | observed | verdict |
|---|---|---|
| `gth-pade` C: `nelec`, `rloc`, `cexp`, one `l=0` block `h = [9.52284179]` | exact vs `gth-pade.dat` | PASS |
| `gth-pade` Si: 2×2 `h^0` with the continuation line mirrored | exact | PASS |
| `cell.atom_charges()` (diamond) | `[4, 4]` (was `[6, 6]`) | PASS |
| `cell.tot_electrons(8)` | 64 | PASS |
| `loads(dumps(cell))` keeps the PP | `back.pseudo == cell.pseudo` | PASS |

`cargo test -p pyscf-pbc-gto --test pseudo_parse` — 6 passed.

**Knock-on effect, now gated.** `cell.ewald()` reads `atom_charges()`, so a
`gth-pade` cell's nuclear repulsion changed the moment 10-01 landed — diamond
goes from **-28.771040577654524 Ha** (all-electron) to **-12.78712914562424 Ha**
(valence). Phase 9 had recorded upstream's pseudised numbers in
`tests/common/ewald_reference.rs::PSEUDISED_EWALD` as "plan 10-01's target, not
asserted"; that debt is now collected —
`ewald.rs::pseudised_ewald_matches_the_recorded_upstream_targets` asserts all
four 3-D systems to 1e-9 Ha, and
`angstrom_reference_systems_match_upstream_within_the_unit_gap` (which builds the
`test_systems` cells, and those DO carry `pseudo`) was switched from the
all-electron table to the pseudised one. Both charge conventions are now gated
side by side; neither can drift.

**Upstream-parity bug fixed in the existing parser.** `cp2k_pp::find_element_block`
took the FIRST block matching an element. A `.dat` file holds one block per
valence configuration, and upstream (`parse_cp2k_pp.py:145-158`) selects the one
whose last alias is *not* `-q<n>`. For sodium the file lists
`Na GTH-PADE-q1 …` before `Na GTH-PADE-q9 … GTH-PADE GTH-LDA`, so the old code
returned `Zion = 1` instead of 9 — eight electrons missing from every sodium
calculation. `basis::tests::load_pseudo_gth_pade_resolves_from_pseudo_tree` now
pins the q9 values and the explicit-`-q1`-suffix path.

### 10-02 — Neighbor list + screening

`crates/pyscf-pbc-gto/src/neighborlist.rs` (port of `neighbor_list.c:80-128`)

| item | observed | verdict |
|---|---|---|
| every on-site pair kept at `L = 0` | yes | PASS |
| decisions match `\|R_j + L − R_i\| < r_i + r_j` exactly, on all 767 images × 16 pairs, with 3-Bohr radii | exact agreement | PASS |
| screening error scales with the precision the radii were built for | 1e-2 → 2.0e-6; 1e-4 → 3.2e-11; 1e-6 → 6.7e-16 | PASS |

**Measured, worth recording:** with DEFAULT radii, diamond/`gth-szv` screening
drops *nothing* — the per-shell radii sum to ~27.8 Bohr while `cell.rcut`
already truncates the image list at 21.3 Bohr. Screening is strictly weaker than
the truncation already applied, which is why `screened_and_unscreened_agree`
holds at **1e-12**, not at `cell.precision`.

### 10-03 — `pbc_intor` / `intor_cross` (the core)

`crates/pyscf-pbc-gto/src/pbc_intor.rs` + `crates/pyscf-kernels/src/pbc/bloch.rs` (K-07)

Oracle-free gates (D-PBC-19), all PASS:

| gate | observed |
|---|---|
| gamma `S` real, symmetric, positive definite (Cholesky on the `2n×2n` real embedding) | PASS |
| `S^k` Hermitian + positive definite at all 8 k of a 2×2×2 mesh | PASS |
| **time reversal** `S^{-k} == conj(S^k)` | < 1e-13 |
| **`rcut` convergence** — `Ls(rcut)` vs `Ls(1.5·rcut)` | < 1e-9 |
| `hermi=1` reproduces the full `hermi=0` matrix | < 1e-13 |
| `int1e_kin` Hermitian, positive real diagonal; `int1e_ipovlp` carries 3 components | PASS |
| an out-of-scope family is refused (`NotYetImplemented{phase:13}`) | PASS |

### 10-04 — Periodic `eval_gto` / `eval_ao_kpts`

`crates/pyscf-pbc-gto/src/eval_gto.rs` + `crates/pyscf-kernels/src/pbc/eval_ao_k.rs` (K-08)

| gate | observed | verdict |
|---|---|---|
| **`ao_k(r + L) == exp(i k·L) · ao_k(r)`** — the primary gate, 3 k-points × 4 lattice vectors × 5 probe points × 8 AOs | < 1e-10 | **PASS** |
| gamma AO exactly real | max\|Im\| = 0 | PASS |
| `ao_{-k} == conj(ao_k)` | < 1e-12 | PASS |
| image sum converged (`rcut` vs `1.5·rcut`) | < 1e-11 | PASS |
| `GTOval_sph_deriv1` has 4 components; component 0 == `GTOval_sph` | < 1e-12 | PASS |
| deriv1 gradients vs central finite difference (`h = 1e-5`) | < 1e-6 | PASS |

Plan 10-04's instruction "do **not** write a new AO evaluator" is honoured: the
existing `pyscf_kernels::eval_gto` is called on `coords − L` and K-08 is only the
phase-accumulate step.

### 10-05 — GTH local pseudopotential

`crates/pyscf-pbc-gto/src/pseudo/vloc.rs` (G space) + `vloc_part2.rs` (real space)

G-space factors on a `[5,5,5]` mesh, vs upstream:

| quantity | diamond | LiF | tolerance | verdict |
|---|---|---|---|---|
| `get_coulG` | 0.0 | 1.1e-16 | 1e-12 | PASS |
| `get_gth_vlocG_part1` | 3.55e-15 | 1.78e-15 | 1e-12 | PASS |
| `get_gth_vlocG` | 3.55e-15 | 2.00e-15 | 1e-12 | PASS |
| `get_alphas` | 0.0 | 4.44e-16 | 1e-12 | PASS |

Real-space short-range matrix (the double lattice sum over 3-centre `origk`
integrals):

| quantity | observed | tolerance | verdict |
|---|---|---|---|
| `get_pp_loc_part2` (diamond, gamma) | **1.777e-12** | 1e-9 | PASS |
| `get_pp_loc_part2` (LiF, gamma) | **1.532e-10** | 1e-9 | PASS (was blocked; see §4) |

`fake_cell_vloc` is pinned against `pp_int.py:554`'s
`C_cn / rloc^{2cn−2} / half_sph_norm` for `cn = 1, 2`, and `cn = 3, 4` correctly
produce nothing for carbon (`nexp = 2`).

### 10-06 — GTH non-local pseudopotential

`crates/pyscf-pbc-gto/src/pseudo/vnl.rs`

| quantity | observed | tolerance | verdict |
|---|---|---|---|
| `get_pp_nl` diamond, 2×2×2, all 8 k | **1.887e-15** (‖V_nl‖ = 0.78) | 1e-11 | PASS |
| `get_pp_nl` silicon, 2×2×2 — exercises `nproj = 2`, i.e. the `int1e_r2_origi` half-overlap and the `PLI_FAC` rescaling | **5.551e-16** (‖V_nl‖ = 0.43) | 1e-11 | PASS |
| Hermitian at every k; exactly real at gamma | PASS | | PASS |
| `V_nl(-k) == conj(V_nl(k))` | < 1e-13 | | PASS |
| `PLI_FAC` rescaling of Si's `h^0` | matches the closed form to 1e-10 | | PASS |

### 10-07 — `get_hcore` / `get_ovlp`

`crates/pyscf-pbc-gto/src/hcore.rs`

| item | observed | verdict |
|---|---|---|
| `get_ovlp` off-diagonal Hermiticity | **exact** (bit-identical, `hermi = 1` mirrors with a conjugate) | PASS |
| `get_ovlp` == `pbc_intor("int1e_ovlp")` | < 1e-13 | PASS |
| `get_t` Hermitian, positive real diagonal | PASS | PASS |
| `get_hcore` | `NotYetImplemented{phase:11}` pointing at `get_hcore_parts` | PASS (see §3) |
| `get_hcore_parts` at gamma: `T + V_nl + V_loc,2` Hermitian, real, all three terms non-trivial | PASS | PASS |
| away from gamma: `V_loc,2` withheld (not faked with the gamma matrix) | PASS | PASS |

### 10-08 — rollup

This document, plus the architecture gates:

```
cargo run -p xtask --bin check-dependency-wall   -> PASS — cubecl-* containment intact (ALG-06)
cargo run -p xtask --bin check-forbidden-paths   -> PASS — 353 .rs file(s); no out-of-scope upstream PySCF imports (FOUND-08)
cargo clippy -p pyscf-pbc-gto -p pyscf-kernels -p pyscf-gto --all-targets  -> no warnings from any file this phase touched
   (two warnings remain in PRE-EXISTING pyscf-gto tests this phase did not touch:
    tests/spinor_intor.rs:7 doc-list indentation, tests/grad_intor_smoke.rs:412
    loop-variable indexing. Note `cargo clippy` CACHES: a second run with the
    same arguments prints nothing even when warnings exist — touch a lib.rs to
    force a real re-lint.)
```

Full-workspace regression sweep:

```
cargo test --workspace --release      -> 246 test binaries, 0 failures
cargo test -p pyscf-pbc-gto --release -> 12 binaries, 0 failures (9 ignored, ALL oracle-gated, none blocked)
PYSCF_ORACLE_VENV=1 cargo test -p pyscf-pbc-gto --release -- --ignored
                                      -> 16 passed, 0 failed (Phase 9's 7 oracle tests re-run green)
```

---

## 3. Deviations from PBC-MASTER-PLAN §8.2 — deliberate, with reasons

### D1. The image-expansion basis is per-image, not one-shot (plan 10-03)

The plan mandates ONE `BasisSet` holding cell-0 plus all `nimgs` image blocks,
indexed `[ish, nbas + l_idx*nbas + jsh]`, with a per-`L` fallback above a
20 000-shell memory guard. **That is 20× slower.** A cintx `SessionRequest`
costs O(total shells in the basis), not O(1):

| basis | shells | per shell-pair evaluation |
|---|---|---|
| one image | 8 | ~20–24 µs |
| 10 images | 44 | 32 µs |
| 100 images | 404 | 65 µs |
| 767 images | 3 072 | 400 µs |

Measured end-to-end on diamond gamma `int1e_ovlp` (12 272 shell-pair
evaluations): one-shot basis **498 s**, per-image basis **0.57 s** — identical
numbers to the last printed digit. The named constant survives as
`PBC_INTOR_SHELL_WARN_LIMIT`, now an advisory that only warns.

### D2. The Bloch contraction is a host fold, not `gemm_dense` (plan 10-03)

The plan's `dgemm`-shaped contraction (`fill_ints.c:1382-1385`) was implemented
first and measured: `pyscf_algebra::gemm_dense` on the cubecl CPU runtime takes
**17 s for a 64³ product** (its tiled `SharedMemory` + `sync_cube` kernel is
pathological on that backend), and the 32 GEMM calls a single diamond
`pbc_intor` needs took **487 of the 498 s** above. The contraction is now an
in-order host fold over images — the same summation order as upstream's two
`dgemm_` calls — and K-07 keeps the trigonometric part (`exp(i k·L)`) on device.
This is a `pyscf-algebra` performance problem, not a PBC one; it is recorded
here because it also caps what Phase 11's FFT can assume.

### D3. `get_pp_loc_part2` adds a second, conservative screen (plan 10-05)

Upstream's neighbor-list screen alone leaves O(nimgs²) shell triples per
`(ish, jsh, P)` — ~40 000 for diamond — which its OpenMP C driver absorbs and a
per-request safe API does not. The port adds the standard Gaussian-product bound
`exp(−θ_ab|A−B|² − θ_(ab)c|P_ab−C|²)` evaluated on each shell's most diffuse
primitive, against `PRESCREEN_EPS = 1e-14` (six decades below `cell.precision`).
It is a genuine upper bound, so nothing it drops can matter; the 1.78e-12 match
against upstream — which applies no such screen — is the proof. (At the first
value tried, 1e-12, the accumulated deviation was 7.1e-10.)

### D4. `get_hcore` returns `NotYetImplemented{phase:11}` — as the plan specifies

Plan 10-07 says the all-electron branch defers to Phase 11 and "PP path must work
in Phase 10". The PP path is *two thirds* done and cannot be finished here:
`get_pp = V_loc,1 + V_loc,2 + V_nl`, and **upstream's own
`pp_int.get_pp_loc_part1` raises `NotImplementedError`** — the long-range local
term is an FFT (`ifft(vlocG·SI)`, FFTDF) or AFT (`ft_aopair`) quantity, and
neither exists before Phase 11 / Phase 13 by D-PBC-09. Everything Phase 10 can
own is delivered and reachable through `get_hcore_parts`
(`T`, `V_nl`, `V_loc,2`, plus `HcoreParts::partial_hcore()`), and the G-space
factor Phase 11 needs, `get_gth_vlocG_part1`, is finished and gated. `get_hcore`
itself refuses rather than returning a matrix that is silently missing a term.

### D5. `pbc_intor` output is F-order

Matching `pyscf_gto::IntorOutput` and the rest of the workspace; upstream's numpy
arrays are C-order. Documented at every entry point.

---

## 4. cintx general contractions — a bug found here, fixed in cintx, now gated

PBC-MASTER-PLAN §2.4 / risk R-13 flagged the moment-weighted families as
`oracle_covered: false` with no dispatch arm. **That is resolved**: cintx Wave
0.5 has landed (`cintx-cubecl/src/kernels/unstable/{origi,origk}.rs`), the
symbols are gated behind `unstable-source-api`, and this crate's default-on
`gth-pp` feature enables them.

`tests/cintx_moment_weighted_available.rs` runs the §2.4 Task-0 fail-open check
on every build and passes: the symbols evaluate, and they differ from their
unweighted parents.

### The second fail-open surface — found 2026-08-26, fixed the same day

§2.4 predicted the operator NAME falling through to the unweighted parent. The
failure that actually showed up was different: both families mishandled a shell
with `nctr > 1`.

| symbol | `nctr = 1` | `nctr = 2` (`Li` / `gth-szv`), BEFORE the fix |
|---|---|---|
| `int1e_r{2,4}_origi_sph` | correct | **silently wrong** — returned `[19.13, 0, 0, 0]`; libcint gives `[0.0786, 1.612, 0.284, 17.157]`. Note element 0 was wrong too, so "the first entry looks plausible" was never a safe check. |
| `int3c1e_r{2,4,6}_origk_sph` | correct | **panicked** — `cintx-cubecl/src/transform/c2s.rs:684`, `copy_from_slice: source slice length (4) does not match destination slice length (1)` |
| `int1e_ovlp`, `int3c1e` (unweighted parents) | correct | correct |

The Cartesian→spherical step sized its output from the shell's angular momentum
and forgot the contraction axis. **cintx fixed both**
(`kernels/unstable/{origi,origk,shared.rs}`, plus its own
`origi_genctr_parity` / `origk_genctr_parity` oracle tests).

### Verified and collected

Re-verified against PySCF 2.12.1 / libcint on the `Li`/`gth-szv` fixture — the
correctness check the panic-only test could not make:

| symbol | max rel \|Δ\| vs libcint |
|---|---|
| `int1e_ovlp_sph` | 6.66e-16 |
| `int1e_r2_origi_sph` | 6.21e-16 |
| `int1e_r4_origi_sph` | 2.72e-15 |
| `int3c1e_sph` | 2.78e-17 |
| `int3c1e_r2_origk_sph` | 5.00e-16 |
| `int3c1e_r4_origk_sph` | 9.27e-16 |
| `int3c1e_r6_origk_sph` | 1.24e-15 |

Consequently:

* `pseudo::require_segmented_basis` — the guard that refused those code paths
  while the bug stood — is **deleted**, along with its two call sites in
  `int_vnl` and `get_pp_loc_part2`.
* The two tests that PINNED the broken behaviour are replaced by positive
  regression tests, `origi_matches_libcint_for_general_contractions` and
  `origk_matches_libcint_for_general_contractions`, which hold the corrected
  values above to 1e-12 relative. A regression now surfaces as a numeric
  failure rather than as a wrong pseudopotential. The length assertion is doing
  real work for `origk`: 2x2x2 = 8 values, not 1.
* `gth_pp_loc.rs::part2_matches_upstream_on_lif` is **un-blocked and passing**
  at **1.532e-10** (gate 1e-9). It is the only §9.2 system with a general
  contraction, so it is the end-to-end proof of the fix.

**No blockers remain in Phase 10.**

## 5. New public surface

| crate | module | what |
|---|---|---|
| `pyscf-pbc-gto` | `neighborlist` | `NeighborList`, `NeighborPair`, `build_neighbor_list{,_for_shlpairs}` |
| | `pbc_intor` | `pbc_intor`, `intor_cross`, `intor_cross_with_images`, `PbcIntorOpts`, `PbcIntorOutput`, `Cell::pbc_intor` |
| | `eval_gto` | `eval_ao_kpts`, `eval_ao_kpts_with_images`, `estimate_rcut_for_eval`, `Cell::pbc_eval_gto` |
| | `hcore` | `get_ovlp`, `get_t`, `get_hcore`, `get_hcore_parts`, `HcoreParts`, `Cell::get_ovlp` |
| | `pseudo` | `PseudoData`, `resolve_pseudo`, `require_segmented_basis`, `Cell::atom_pseudo` |
| | `pseudo::vloc` | `get_coulg`, `get_gth_vlocg{,_part1}`, `get_vlocg`, `get_alphas{,_gth}`, `fake_cell_vloc`, `VlocAux` |
| | `pseudo::vloc_part2` | `get_pp_loc_part2{,_gamma}`, `prescreen_exponent` |
| | `pseudo::vnl` | `fake_cell_vnl`, `int_vnl`, `get_pp_nl`, `FakeCellVnl`, `HlBlock`, `PLI_FAC` |
| `pyscf-kernels` | `pbc::bloch` | **K-07** `bloch_phase` |
| | `pbc::eval_ao_k` | **K-08** `eval_ao_k_accumulate` |
| `pyscf-gto` | `projection` | `build_image_expanded_basis`, `build_image_expanded_cross_basis`, `build_image_expanded_with_aux` |
| | `intor` | `add_suffix` (was private) |
| | `layout_table` | 6 new entries: `int1e_r{2,4}_origi_sph`, `int3c1e_sph`, `int3c1e_r{2,4,6}_origk_sph` |
| `pyscf-core` | `mole` | `GthPseudo` / `GthProjector` gain `PartialEq` |
