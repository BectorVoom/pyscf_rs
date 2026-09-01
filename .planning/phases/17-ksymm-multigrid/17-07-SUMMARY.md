# 17-07 — ksymm SCF adapters — SUMMARY (IN PROGRESS)

**Status: PARTIAL.** Task 0 (the k-set indirection) and the KRHF adapter's
structure + Tasks 1/2 hooks are landed and COMPILING. Tasks 3 (the two extra
`eig` branches), 5 (KUHF/KGHF), 6 (fast `get_jk` validation) and 7 (Gates C/D)
are NOT done. Written incrementally — see the note at the bottom on why.

## Landed

`crates/pyscf-pbc-scf/src/khf_ksymm.rs` (new) — `KsymAdaptedKrhf`, a
`KOverrideHooks` implementation over an IBZ k-set.

* **Task 0 — the k-set indirection, settled.** `KOverrideHooks::kpts()`
  returns a borrowed slice, so the adapter owns `kpts_ibz: Vec<[f64;3]>` and
  returns that. Everything downstream (`nkpts`, the `idx(set,k)` layout in
  `KScfResult`, DIIS's Fock subspace) then works on IBZ indices unchanged.
  `KScfResult` keeps IBZ-length arrays; nothing is unfolded silently
  (upstream's `to_khf()` is the explicit converter).
* **The DF object is built over the FULL BZ.** `get_hcore`, `get_ovlp` and
  `get_jk` all take their k-points as an explicit argument
  (`fftdf.rs:447`), so ONE full-BZ DF serves both routes: the one-electron
  hooks pass `kpts_ibz`, the reference `get_veff` passes the full BZ. **The DF
  layer never learns about symmetry** — the crux of D-PBC-15.
* **Task 1 — `get_occ` over the UNFOLDED BZ** (17-CONTEXT §3.4). Unfolds
  `mo_energy` via `transform_mo_energy`, assigns one Fermi level with
  `nelectron() = cell.tot_electrons(kpts.nkpts())` (the **BZ** count, not
  `nkpts_ibz`), then folds back through `check_mo_occ_symmetry`, whose failure
  is surfaced as a typed error naming both k-points (a symmetry-broken state
  is a physical condition, not an internal error). `OCC_SYMMETRY_TOL = 1e-4`.
* **Task 2 — every weighted sum named.** The module doc carries the full table
  (which of `weights_ibz` / `1/nkpts` / a bare sum each site takes, and why),
  as 17-CONTEXT §3.5 requires and as 15-CONTEXT §3's KMP2 trap motivates.
  `energy_elec` uses `weights_ibz` (which already sums to 1, so no further
  `1/nkpts`); `get_init_guess`'s rescale uses `* nkpts`.
* **Task 6 (structure only)** — both `get_jk` routes are written behind
  `JkRoute`, with `Reference` as the **default**. `veff_fast` is NOT yet
  validated against `veff_reference`; until its 1e-13 equivalence test exists
  it must not be made the default.

### Dependency change, and a stale comment corrected

`pyscf-pbc-scf` gains `pyscf-pbc-symm` (+ `num-complex 0.4.6`, matching how
`pyscf-pbc-symm` itself pins it). `pyscf-pbc-symm` **dev**-depends on
`pyscf-pbc-scf` (17-03/17-04/17-05 tests need a converged KRHF), so these now
form a cycle. **Cargo permits it** — a dev-dependency edge is excluded from the
build-graph ordering; `cargo metadata` and `cargo build -p pyscf-pbc-scf` both
resolve. The library direction still matches D-PBC-25: `pyscf-pbc-symm` never
depends on `pyscf-pbc-scf` to compile itself, only to run its tests. The
comment in `pyscf-pbc-symm/Cargo.toml` that asserted "no build-graph cycle
since `pyscf-pbc-scf` does not depend on this crate" was TRUE when written and
is now FALSE; it has been corrected in place rather than left to mislead.

## Verified so far

`cargo test -p pyscf-pbc-scf --release --test khf_ksymm -- --test-threads=1`
— **4/4 green**, 222 s.

| test | result |
|---|---|
| `kpts_hook_returns_the_ibz_set` | IBZ set is 3 of 8 BZ points; `nelectron` counts over the BZ |
| `ibz_weights_carry_the_star_multiplicities` | `sum = 1` to 1e-15; every `weights_ibz[i] == |stars[i]|/nkpts` |
| `ao_symmetry_eig_matches_the_plain_route` | `|dE_tot|` = **1.703e-11** (two converged SCFs) |
| `ao_symmetry_eig_matches_the_plain_route_on_identical_inputs` | max `|d eigenvalue|` = **9.186e-11** |

### Two findings from writing those tests

**1. The first version of the eig comparison was wrong, in the way 17-05 warned
about.** It compared two *independently converged* SCFs and reported a 4.4e-9
eigenvalue spread against a 1.7e-11 total-energy agreement — i.e. it was
measuring convergence noise, not algebra. 17-05's plan states the rule for
Gate B in as many words: *"never two SCFs, because then the residual is
convergence noise, not algebra."* The test now runs ONE SCF and calls both
`eig` routes on its converged Fock and overlap. That tightened the residual
by ~48x, to 9.186e-11, which is the honest number for the algebra.

**2. Why 9.186e-11 and not ~1e-14.** Schur's lemma makes the
block-diagonalisation exact *for an exactly block-diagonal Fock*, so the
residual should be solver precision. It is not, and the reason is already
measured: `17-04-MEASUREMENT.md` records that at `si()`'s default
`cell.precision = 1e-8` the off-block Fock elements are ~4e-10 (dropping to
5.5e-13 once `precision` AND `conv_tol_grad` are both tightened). The
symmetry-adapted route *discards* exactly that inter-irrep weight, so it must
differ from the plain route by about that much. **The 9.186e-11 is the known
off-block leakage of the fixture, not slack in the implementation** — and it
is the same joint precision/convergence floor that episode identified. A
tight-fixture rerun should push it toward 1e-13; that is worth doing when
Gate C/D land, and would let `EIG_TOL` drop from 1e-10.

* `cargo build -p pyscf-pbc-scf` — clean (no errors, no new warnings).
* `git diff crates/pyscf-pbc-scf/src/kscf.rs` — **EMPTY.** The driver was not
  forked, copied or edited. D-PBC-15's central claim holds: the ksymm layer is
  another implementation of the eleven-method trait.
* `git diff crates/pyscf-pbc-df/` — untouched by this plan.

## NOT done — carried over

1. **Task 3 — the `eig_trs` branch only.** The `use_ao_symmetry = true` branch
   **is landed and tested** (see above); `eig_symm_adapted` solves one irrep
   block at a time, with the layout contract (`symm_orb` column-major, Fock
   row-major, output column-major) written out at the function because this is
   the 14-05 defect's exact shape. Still missing: the `eig_trs` branch (real
   `mo_coeff` at time-reversal-invariant momenta, via 17-05's `is_trim`).
   **The TRIM test (`abs(mo_coeff[k].imag).max() == 0`) is the only thing that
   proves that branch was taken — it must ship with the branch.**
2. **Task 4** — `get_rho`, `dump_chk`/`init_guess_by_chkfile` (including
   upstream's refusal when a chkfile's k-count disagrees with
   `nkpts_ibz`), `get_orbsym`, `to_khf`.
3. **Task 5** — `KUHF`/`KGHF` adapters. KROHF has no upstream `*_ksymm`
   module: refuse, do not invent one.
4. **Task 6 (validation)** — `veff_fast` vs `veff_reference` at 1e-13 on every
   fixture, both k-mesh types, GDF and FFTDF. Do NOT loosen that tolerance to
   make it pass; a failure localises a broken equivariance assumption.
5. **Task 7** — Gates C and D at 17-01's measured floors, mesh pinned on both
   sides, Gate D reported per DF route separately; plus the speed gate.
6. ~~**The `KsymmArray` Fock-store acceptance test**~~ — **CLOSED 2026-09-02**,
   `crates/pyscf-pbc-symm/tests/ktensor_ksymm_scf.rs`. Fills the store from
   this adapter's own IBZ output and reads back every BZ k-point; worst
   residual **3.318e-12** against the 1e-9 Gate-B floor. See
   `17-06-SUMMARY.md`'s handoff item 1 for the numbers and for why the
   comparison is against `transform_mo_coeff` rather than a second full-BZ
   SCF (gauge freedom).
7. **17-04's tier-2 oracle** (`test_krhf_symorb` energy), which was blocked on
   this plan's `eig`.
8. Extend plan 11-09's existing metal-occupancy test (risk R-06) to the ksymm
   path rather than writing a parallel one.

## D-17-07-01 — `little_cogroup_ops` and `ops` are two different index spaces

Found by running the first `use_ao_symmetry = true` test, which failed with
`little_cogroup_ops[0] references op index 24, but ops has 24 entries`.

`little_cogroup_ops` is filled from `np.where(kpts.k2opk[ki] == ki)[0]`
(`kpts.py:112`) — indices into `k2opk`'s **columns**, of which there are
`nop * (time_reversal + 1)`. Its consumer indexes the op list directly:
`kpts.ops[iop]` (`basis.py:113`, and this port's `symm_adapted_basis`). With
time reversal on, the second half of that column space is reachable — at Γ and
at every other TRIM, `-op·k == k` matches too — so **upstream would raise
`IndexError` there.** This port refuses with a typed `KptsSymmInputMismatch`
instead, which is how it was found.

This is the same class of latent upstream defect 17-06 recorded as
D-17-06-01 (`fromdense` writing to the wrong keys, no caller and no test
upstream), and 17-05 had already recorded the adjacent symptom:
*"`little_cogroups` refuses where upstream raises `IndexError` (time-reversal +
`k2opk`'s `2*nop` columns)."*

**Not resolved here.** `tests/khf_ksymm.rs` builds its `KPoints` with
`time_reversal_symmetry = false` and documents why, so `use_ao_symmetry = true`
is exercised on the space-group fold — the part `symm_adapted_basis` actually
consumes. Reconciling the two index spaces (either by mapping the doubled
indices down and handling the time-reversal half explicitly, or by having
`little_cogroup_ops` expose the `ops`-space subset separately) is real work
that should be decided deliberately, not patched under a failing test. It
also gates `use_ao_symmetry = true` + time reversal, which is upstream's
DEFAULT combination — so it must be closed before this adapter is
recommended at its defaults.

## Why this file is partial rather than absent

This environment restarts every ~20-40 minutes and kills background agents.
**Four consecutive agent sessions on this plan were killed during their
reading phase, before writing a single line of code.** The reconnaissance was
therefore extracted into `17-07-BLUEPRINT.md` (the verbatim trait shape, the
template file, the prerequisites, the traps) and the code above was written
directly by the orchestrator session, which survives restarts. Anyone
continuing should read the blueprint first and keep landing work in
compiling increments.
