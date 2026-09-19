# 20-13-FIX SUMMARY — KUKSpU Hubbard term (D3) and ksymm GGA on an s-only basis (D4)

**Date:** 2026-09-14. **Work location:** an isolated copy (`pyscf_rs-fix2013`, rsync with `target/gate` warm-started). Pristine snapshots were diffed against the main tree before copy-back, and none of the touched files had changed in the meantime. The copy is now deleted. Both crate defects are fixed, and both binding refusals are removed.

## Root causes

### D3 — `kspu::add_vhubbard_weighted` (`crates/pyscf-pbc-dft/src/kspu.rs`)

**What was wrong.** The routine looped over density channels and applied the RESTRICTED expressions to every channel:

- `E_U += w (U/2)(Tr P − Tr P²/2)`
- `V = (1 − P) U/2`

That is `krkspu.py:111-113`. Upstream KUKSpU (`kukspu.py:96-97`) uses different expressions per spin:

- `Tr P_s − Tr P_s²`
- `(1 − 2P_s) U/2`

The doc's "U/2 is per CHANNEL" argument was wrong. Two consumers shared the defect: `KsymAdaptedKukspu`, and the binding's `PuDriver`.

**Fix.** The channel count now selects the upstream routine:

- 1 channel → `krkspu.py:111-118`, kept bit-for-bit;
- 2 channels → `kukspu.py:96-102`, written as upstream writes it;
- any other count → an error.

The loop order follows upstream. SC (the overlap times the local orbitals at each k) is spin-independent (`kukspu.py:69-71`), so it is computed once per `(site, k)`. Doc comments in `krks_ksymm.rs` (`KsymAdaptedKukspu`) and `pyscf-pbc-grad/src/stress/kuks.rs` were updated to match.

### D4 — `KPoints::symmetrize_density_vec` (`crates/pyscf-pbc-symm/src/kpts.rs`)

**What was wrong.** The routine indexed `dmats()[iop][1]`. `make_Dmats` builds `Dmats` only up to the basis' highest `l` (`symmetry.py:83-86`), so on an s-only cell `Dmats[iop]` holds only `[D⁰]`. The GGA gradient is a Cartesian vector whatever the basis, so the missing block was a real gap.

**Fix.**
- The signature is now `symmetrize_density_vec(&self, cell, rho, ibz, mesh)`.
- When the stored set lacks `l = 1`, the block is built from `ops[iop].a2r(cell).rot` through `make_dmats(cell, &[rot], Some(1))`. That is upstream's `l_max` widening argument.
- A stored `l = 1` block is used unchanged.
- The one caller (`numint.rs`) passes `cell`.

## Bitwise safety of the restricted path

- I ran a probe before and after the change: `add_vhubbard` and `add_vhubbard_weighted`, on sto-3g and 6-31g, with and without `alpha`. `E_U` bits and a hash of the potential are **bit-identical** between the pristine and fixed `kspu.rs`.
- The s-only vector symmetrization is **bitwise** equal to that of a p-basis twin cell.

## Tests (TDD — RED was captured before each fix)

| test | RED (pre-fix) | GREEN |
|---|---|---|
| `pyscf-pbc-dft/tests/kukspu.rs::kukspu_e_u_and_potential_equal_krkspu_on_a_closed_shell_density` (oracle-free) | \|ΔE_U\| 1.149e-2, max\|ΔV\| 5.39e-2 | sto-3g: 0, 3.5e-18; 6-31g: 6.9e-18, 1.4e-17 |
| `kukspu.rs::kukspu_he_matches_upstream` (`T1`, 1.6 s; He sto-3g 2×2×2, mesh 15, U = 5 eV) | e_tot off by 9.187e-2 | e_tot **6.08e-14** (tol 1e-12); E_U converged 5.62e-14; E_U at 0.35/0.35 **3.45e-10**; E_U at 0.5/0.2 **3.10e-10** (tol 1e-9, the D7 bound) |
| `pyscf-pbc-dft/tests/ksymm_gga_s_only.rs` (not ignored, 2.7 s) | panic `kpts.rs:2121` | KRKS **8.44e-14**, KUKS **8.48e-14** vs full BZ (tol 1e-9; mesh 43³; 3 of 8 k) |
| `pyscf-pbc-symm/tests/kpts_transform.rs::vector_density_on_an_s_only_basis_matches_a_p_basis_twin` | (signature change) | bitwise |
| `pyscf-pbc-grad/tests/kuks_stress_hubbard.rs` (new; the gate `stress/kuks.rs` cited did not exist) | — | (1,0) 5.18e-10, (2,2) 2.02e-9 (tol 1e-8) |

### Phase-18 KUKS stress

- `hubbard_u_deriv1_uks` never called `add_vhubbard`. It was already the unrestricted derivative (`*2` / `*4`), so nothing had relied on the wrong formula.
- The new FD gate ties it to the corrected `E_U`.
- At a closed-shell density, the UKS analytic derivative equals `hubbard_u_deriv1` (KRKS) to **2.8e-17**.

**Why the new test uses a full step of 1e-4:** at upstream's 2e-4 the central difference's O(h²) truncation error on this compact fixture exceeds the gate. Full-step series (2,2), measured:

| full step | UKS error | KRKS error |
|---|---|---|
| 8e-4 | 3.28e-7 | — |
| 4e-4 | 8.19e-8 | — |
| 2e-4 | 2.04e-8 | 1.84e-8 |
| 1e-4 | 5.04e-9 | 4.55e-9 |
| 5e-5 | 1.22e-9 | — |

The error falls by exactly 4× per halving, which is pure truncation. The 1e-8 tolerance is unchanged.

## Affected targets (copy, `target/gate`, LTO=false, -j 4)

| target | result |
|---|---|
| pbc-symm `kpts_transform` | 17 passed (376 s) |
| pbc-dft `modules` | 8 passed |
| pbc-dft `krks_ksymm` | 7 passed, 3 ignored |
| pbc-dft `ksymm_symmetrize_rho` | 2 passed |
| pbc-dft `ksymm_threads` | 2 passed |
| pbc-dft `ksymm_band_ao_reuse` | 3 passed |
| pbc-dft `ksymm_trace_precision` | 3 passed |
| pbc-dft `krks_ksymm_multigrid` | 2 passed |
| pbc-grad `uks_stress` | 9 passed |
| pbc-grad `kuks_stress_hubbard` | 1 passed |
| **pbc-grad `krks_stress`** | **killed at 600 s (D-20-D)**; see the pre-existing failures below |

### Pre-existing `krks_stress` failures (not caused by this fix)

- **`hubbard_u_deriv1_matches_eu_fd_at_1e8`** fails at (2,2), 1.841e-8 against a 1e-8 gate.
  - The A/B was run with pristine `kspu.rs` and gave the identical 1.841e-8, so this failure predates the fix.
  - The cause is the same truncation series as in the table above.
  - Phase 18 owns the test, so it was not edited.
- **`gate_a2_krks_get_j_matches_fd`** fails at 2.362e-9 and **`gate_a2_krks_get_nuc_matches_fd`** at 3.061e-9, both against a 2e-9 gate.
  - They exercise the get_j / get_nuc strain derivatives, which the diff does not touch: no `kspu` code and no ksymm path.
  - No A/B was run for these two.
  - The likely cause is Phase-18 in-progress work or the D-20-E coulG re-key.

## Binding (main tree, done last)

**Changes.**
- `crates/pyscf-py/src/pbc/dft.rs`: the KUKSpU/KsymAdaptedKUKSpU refusal and the ksymm-GGA s-only refusal are removed. The now-unused `grid_numint` binding is removed too.
- `cargo check -p pyscf-py`: exit 0, no dft.rs warnings.
- Extension rebuilt per EXECUTION-NOTES §2: maturin exit 0, cargo `Finished` in 47.68 s (`target/py-fix2013-build.log`).

**`python/pyscf/tests/test_pbc_dft.py`: the two refusal tests were replaced.**
- The upstream script now also runs KUKSpU.
- `test_kukspu_closed_shell_e_u_equals_krkspu`: full BZ and IBZ.
- `test_kukspu_matches_upstream`: e_tot 5.995e-14; E_U converged 5.6e-14, 0.35/0.35 3.45e-10, 0.5/0.2 3.10e-10; bridged run bitwise equal to the unbridged run.
- `test_dft_u_e_u_over_the_ibz_matches_the_full_bz`: now parametrized to include KUKSpU (6.9e-18).
- `test_dft_u_scf`: now parametrized to include KUKSpU (E_U −2.0e-17, |E − E_KUKS| = 0).
- `test_ksymm_gga_on_an_s_only_basis_matches_full_bz[KRKS, KUKS]`: 8.44e-14 and 8.48e-14.

**Result:** `.venv/bin/pytest python/pyscf/tests/test_pbc_dft.py -q -p no:cacheprovider` gives **30 passed in 77.84 s**, exit 0 (`target/py-fix2013-test.log`).

## Files

- **Modified:**
  - `crates/pyscf-pbc-dft/src/{kspu.rs,krks_ksymm.rs,numint.rs}`
  - `crates/pyscf-pbc-symm/src/kpts.rs`
  - `crates/pyscf-pbc-symm/tests/kpts_transform.rs`
  - `crates/pyscf-pbc-grad/src/stress/kuks.rs` (doc only; not rustfmt'd, because that file predates the formatter)
  - `crates/pyscf-py/src/pbc/dft.rs`
  - `python/pyscf/tests/test_pbc_dft.py`
- **New:**
  - `crates/pyscf-pbc-dft/tests/{kukspu.rs,ksymm_gga_s_only.rs}`
  - `crates/pyscf-pbc-grad/tests/kuks_stress_hubbard.rs`
- **CubeCL manual:** not consulted. No kernel was written; this is host-side algebra only.
