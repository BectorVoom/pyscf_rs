# T06 — Final diagonalisation: implement in the SCF driver

**Goal.** `crates/pyscf-pbc-scf/src/kscf.rs::kernel` does upstream's final
step. T05's test turns green.

**Upstream** (`pyscf/scf/hf.py:211-232`), executed only if the loop
converged and `conv_check` is on:

```
eps, C  = eig(fock, s1e)        # fock = plain Fock of the latest density
occ     = get_occ(eps, C)
dm      = make_rdm1(C, occ)
vhf     = get_veff(dm)
last_e, e_tot = e_tot, energy_tot(dm, h1e, vhf)
norm_gorb = |get_grad(C, occ, plain Fock of the NEW dm)|
converged = |e_tot − last_e| < conv_tol·10   OR   norm_gorb < conv_tol_grad·3
```

Note the **OR**, and that the cycle counter is NOT increased.

## Read first

- `crates/pyscf-pbc-scf/src/kscf.rs` lines 110–250 (whole `kernel`).
- `crates/pyscf-pbc-scf/src/types.rs` lines 45–112 (`KScfConfig`,
  `CycleState`).
- `crates/pyscf-pbc-scf/tests/kscf_resume.rs` (how the hook is used).

## Do

1. `types.rs`, struct `KScfConfig`: add the field

```rust
/// Upstream's `conv_check` (`scf/hf.py:211-232`): after convergence,
/// diagonalise the plain Fock matrix once more and re-test.
pub conv_check: bool,
```
   and in its `Default` impl set `conv_check: true`.
2. `types.rs`, struct `CycleState`: add the field

```rust
/// `true` only for the extra call made after the final diagonalisation.
pub final_state: bool,
```
3. `kscf.rs`, inside the loop, the existing hook call (line ~215): change
   `converged: converged_now` to
   `converged: converged_now && !cfg.conv_check, final_state: false`.
   (With `conv_check` on, the converging cycle is reported as NOT yet
   converged; the final call below reports the real result.)
4. `kscf.rs`, directly after the `for cycle …` loop closes and BEFORE the
   `if mo_coeff.is_empty()` block, insert:

```rust
// scf/hf.py:211-232 — the final diagonalisation ("conv_check").
if converged && cfg.conv_check {
    let fock = hooks.get_fock(&h1e, &vhf, &dm)?;
    let (e, c) = hooks.eig(&fock, &s1e)?;
    let (occ, f_levels) = hooks.get_occ(&e)?;
    dm = hooks.make_rdm1(&c, &occ)?;
    vhf = hooks.get_veff(&dm)?;
    let last_e = e_tot;
    let (ee, ec) = hooks.energy_elec(&dm, &h1e, &vhf)?;
    e_elec = ee;
    e_coul = ec;
    e_tot = e_elec + e_nuc;
    let norm_gorb = norm(&hooks.get_grad(&c, &occ, &h1e, &vhf));
    converged = (e_tot - last_e).abs() < cfg.conv_tol * 10.0 || norm_gorb < grad_tol * 3.0;
    if cfg.verbose {
        tracing::info!(e_tot, de = e_tot - last_e, norm_gorb, converged, "periodic SCF final diagonalisation");
    }
    mo_energy = e;
    mo_coeff = c;
    mo_occ = occ;
    fermi = f_levels;
    if let Some(hook) = &cfg.on_cycle {
        let bare = hooks.get_fock(&h1e, &vhf, &dm)?;
        (hook.0)(&crate::types::CycleState {
            cycle: cycles.saturating_sub(1),
            e_tot,
            dm: &dm,
            fock: &bare,
            converged,
            final_state: true,
        });
    }
}
```
   Use the same call forms the loop already uses for each hook (copy them
   from the loop body if a signature differs from the snippet).
5. Every place that builds a `KScfConfig` with a struct literal WITHOUT
   `..Default::default()` / `..KScfConfig::default()` / `..base.clone()`
   needs the new field. Find them with CHECK on `pyscf-pbc-scf`,
   `pyscf-pbc-dft`, `pyscf-py`; add `conv_check: true` where the compiler
   asks.
6. Every place that builds a `CycleState { … }` literal: add
   `final_state: false` (find with
   `grep -rn "CycleState {" crates/ --include=*.rs`).
7. `crates/pyscf-pbc-scf/tests/kscf_resume.rs`: the test resumes from a
   hook state. Add `conv_check: false` to every `KScfConfig` it builds, so
   it keeps testing the loop itself.

## Verify

1. ORACLE `-p pyscf-pbc-dft --test kscf_conv_check_oracle` → passes; note
   the printed `Γ mo_energy max|Δ|`.
2. TEST `-p pyscf-pbc-scf --test kscf --test kscf_resume --test grad_fc_hoist`
   → no new failure against `baseline.txt`.
3. ORACLE `-p pyscf-pbc-dft --test scf_ovlp_smearing_oracle` → passes.

## If it fails

- A test in `kscf` pins an energy with `assert_eq!`/`to_bits()` between two
  Rust runs: both runs now do the final step, so they still agree. If one
  compares against a literal number: STOP and report the test name and both
  numbers. Do not edit the literal.
- Borrow error on `dm` inside the hook call: build `bare` first (as in the
  snippet), then call the hook.
