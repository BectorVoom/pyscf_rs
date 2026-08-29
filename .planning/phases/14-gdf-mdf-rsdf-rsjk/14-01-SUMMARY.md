# 14-01 SUMMARY — `incore`: the auxiliary cell and the 3-centre double lattice sum

**Status:** SHIPPED. **Date:** 2026-08-29.
**Green:** `cargo test -p pyscf-df -p pyscf-pbc-df` (49 tests) and
`PYSCF_ORACLE_VENV=1 cargo test -p pyscf-pbc-df --test incore -- --include-ignored`
(10/10). `check-dependency-wall` PASS, `check-orphan-modules` PASS (288 files).

## What shipped

| Module | Content |
|---|---|
| `pyscf-df/src/etb.rs` | `aug_etb_element`, `expand_etbs`, `aug_etb` — `df/addons.py:76-168`, the `USE_VERSION_26_AUXBASIS = true` (geometric-average) branch, which is the live one in 2.12.1 |
| `pyscf-df/src/configuration.rs` | `CONFIGURATION[Z]` — `data/elements.py:457-576` |
| `pyscf-df/src/psi4_auxbasis.rs` | `DEFAULT_AUXBASIS`, 27 rows, CANONICAL keys |
| `pyscf-df/src/bse_auxbasis.rs` | the 95 `BSE_META` rows that carry auxiliaries |
| `pyscf-df/src/make_auxbasis.rs` | `predefined_auxbasis` + `make_auxbasis` — Psi4 table → BSE metadata → even-tempered fallback, in upstream's order |
| `pyscf-pbc-df/src/incore/auxcell.rs` | `make_auxcell`, `make_modrho_basis`, `AuxCell`, `gaussian_int` |
| `pyscf-pbc-df/src/incore/int3c.rs` | `estimate_rcut`, `aux_e2`, `fill_2c2e`, `conc_locs`, `Aosym` |
| `pyscf-gto/src/basis/alias.rs` | 21 def2 `jfit`/`jkfit`/`ri` alias rows |
| `pyscf-gto/src/lib.rs` | `strip_name_echo` made `pub` |
| `pyscf-pbc-gto/src/pbc_intor.rs` | `int2c2e` added to `SUPPORTED_INTORS` |

`build_image_expanded_triple_basis` (plan 14-01 Task 2) turned out to be
**unnecessary**: `pyscf_gto::build_image_expanded_with_aux` already ships the
exact layout (Phase 10 added it for `get_pp_loc_part2`), so the plan's new
builder was not written. Recorded as a deviation, not a gap.

## Numbers

| quantity | port | upstream | status |
|---|---|---|---|
| `auxcell.nbas` / `.nao`, diamond | 36 / 108 | 36 / 108 | exact |
| `auxcell.nbas` / `.nao`, He-fcc | 9 / 23 | 9 / 23 | exact |
| auxiliary monopole | `half_sph_norm` to **1e-14** | — | oracle-free |
| `incore.estimate_rcut`, diamond | 17.266040957536866 | 17.266040957536866 | exact |
| `incore.estimate_rcut`, He-fcc | 9.53235156147295 | 9.53235156147295 | exact |
| `auxcell.rcut` after modrho, He-fcc | 7.723468338327722 | 7.723468338327722 | exact |
| **`aux_e2`, isolated cell, all 23 components** | — | — | **max\|diff\| 8.88e-16** |

`aux_e2` also passes, with no oracle: `s2` packs `s1` bit-identically, the
`T[ki,kj][mu nu] == conj(T[kj,ki][nu mu])` conjugation identity to 1e-12, and
reality at gamma to 1e-14. `fill_2c2e` is Hermitian to 1e-9 and real at gamma.

**Wall clock:** one `aux_e2` on diamond `gth-szv` at `aosym = s2`, release
profile: **215.1 s**. Upstream's ENTIRE GDF-driven `KRHF` on the same system is
6.4 s (`measurements/builders.py`), so the port is ~30-40x off on this
primitive. Plan 14-02 must size its blocking around 215 s per pass and treat the
gap as a named carry-over, not a surprise.

## THE finding: `aux_e2` against a CHARGED auxiliary cell has no well-defined value

Full evidence and tables in `measurements/README.md` § "Recorded during 14-01".
The short version:

* The port's per-triple algebra is **exact** — the isolated-cell gate above is
  machine precision on every component.
* On a real periodic cell, the double lattice sum **diverges with `rcut`**
  (16.11 at 9.53 Bohr, 34.05 at 14.0), because each auxiliary function carries
  net charge and its Coulomb interaction with a distant AO pair decays as `1/R`.
  Brute-forced inside upstream itself, so this is not a port artifact.
* Upstream's `incore.aux_e2` is that double sum minus a **P-INDEPENDENT** offset
  (5.1814 at 9.53, 23.12 at 14.0 — four identical digits across every `P`),
  i.e. it removes the divergent `G = 0` background-charge piece. P-independence
  is the signature of a term proportional to `S_mu_nu * q_P`, and every
  modrho-normalised auxiliary function has the same monopole by construction.
* The **compensated** tensor `fuse(j3c)` — the one GDF actually consumes — is
  screening-independent: upstream's is bit-identical at `rcut` x1.0, x1.5, x2.0.

**Consequence, and it is a plan change:** the 1e-11 oracle gate on the raw
periodic `aux_e2` is **retired**, the way Phase 12 §1d and Phase 13 retired
theirs — it gates a quantity with no screening-independent value. It is replaced
by the isolated-cell identity (1e-13 asserted, 8.88e-16 measured) plus the four
structural identities, and **the 1e-11 oracle gate moves to plan 14-02, on
`fuse(j3c)`**. `14-01-PLAN.md` Task 6 assertion 6 and `14-02-PLAN.md` Task 8 are
updated accordingly.

## Two defects this plan's own tests caught, both in EXISTING code

1. **`pyscf_df::auxbasis::DEFAULT_AUXBASIS` had the wrong `sto-3g` row and the
   wrong key convention.** It said `("weigend", "weigend")`; upstream says
   `("def2-svp-jkfit", "def2-svp-ri")`, and its keys are `_format_basis_name`-
   canonical (`sto3g`), not dash-preserving. Found because `make_auxbasis` on
   the He-fcc reference cell went through it. Fixed, with the stale test
   expectation corrected and a pointer to the upstream-faithful chain.
2. **`estimate_rcut` has two upstream namesakes that are NOT interchangeable.**
   `incore.estimate_rcut` (`incore.py:440`) uses `cs = gto_norm(l, e)`;
   `gdf_builder.estimate_rcut` (`gdf_builder.py:932`) uses the libcint
   contraction coefficient from `_extract_pgto_params`. They also differ in the
   `fac` prefactor and in the `(sfac*r0)` exponent (`l3 - 2` vs `l3 - 1`).
   Porting the wrong one gave 15.815 against upstream's 17.266. This is the same
   confusion as Phase 13's defect #2, one function along.

## Deviations from the plan

* Task 2's `build_image_expanded_triple_basis` not written — already existed.
* Task 6 assertion 6 (periodic oracle at 1e-11) retired with evidence; see above.
* `make_modrho_basis(drop_eta = Some(_))` returns `NotYetImplemented { phase: 14 }`
  rather than silently changing `naux`. No Phase-14 caller sets it.
* `AuxCell` carries `modrho_scale` BESIDE the cell rather than inside it,
  because `make_env::normalise_contractions` is scale-invariant and no
  coefficient written into `_basis` survives a rebuild. `_env` IS rewritten, so
  every estimator that reads it (`estimate_rcut`, `estimate_ke_cutoff`,
  `_extract_pgto_params`) is upstream-exact. Same separation
  `pseudo::vloc_part2` already uses for `fake_cell_vloc`.

## Carry-overs

* **Performance**: 215 s per diamond `aux_e2`. The loop is `nimgs x pairs x
  nauxbas` = 106M shell triples before screening; the Gaussian-product prescreen
  is what makes it finite. 14-02 needs this fast enough to run inside `make_j3c`.
* `hdf5-metno` is still declared directly by `pyscf-pbc-df`, contradicting D-07.
  Plan 14-03 Task 0 owns it; `check-dependency-wall` passes today because it only
  guards `cubecl-*`.
* **Unrelated pre-existing failure**: `pyscf-dft --lib
  hooks::tests::define_xc_string_form_parses` fails
  (`(spec.hyb().0 - 0.2).abs() < 1e-9`). It is in the uncommitted Phase-12/13
  working tree, was failing before 14-01 started, and nothing in this plan
  touches `pyscf-dft`. Reported, not fixed.
