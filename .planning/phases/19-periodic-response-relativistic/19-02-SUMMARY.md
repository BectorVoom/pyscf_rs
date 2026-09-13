# 19-02 SUMMARY — substrate: five crate surfaces, libxc posture, single-CPHF gate

**Shipped:** 2026-09-12. All four tasks complete; every verification command green.

## Task 1 — the five crate surfaces

| crate | modules created | shape |
|---|---|---|
| `pyscf-pbc-tdscf` | `error`, `types` (`TdaConfig`/`TdaResult`), `davidson` (**full** dense TDA solver), `rhf`/`krhf`/`uhf`/`kuhf`/`rks`/`uks`/`krks`/`kuks` (signatures + `NotYetImplemented`, filled by 19-06…19-09) | `pyscf-pbc-mp` |
| `pyscf-pbc-gw` | `error`, `types` (`GwRoute`/`GwConfig`/`QpResult`), `pade` (**full** Thiele Padé), `sigma` (**full** imag grid + RPA polarizability), `krgw_ac`/`krgw_cd`/`kugw_ac`/`kgw_slow`/`kgw_slow_supercell`/`gw_slow` (signatures, 19-10…19-13) | `pyscf-pbc-mp` |
| `pyscf-pbc-adc` | `error`, `types` (`AdcLevel`/`AdcConfig`/`AdcRoots`), `kadc_ao2mo`/`kadc_rhf`/`amplitudes` (19-14), `ip` (19-15), `ea` (19-16), `dfadc` (19-17) | `pyscf-pbc-mp` |
| `pyscf-pbc-x2c` | `error`, `sfx2c1e`/`x2c1e` (19-18) | `pyscf-pbc-mp` |
| `pyscf-pbc-eph` | `error`, `eph_fd` (19-18) | `pyscf-pbc-mp` |

Shared infrastructure was implemented **fully now** (not stubbed) because every
later plan consumes it unchanged: the dense TDA eigensolver (`eigh_gen` +
`oracle_sum` oscillator strengths, TDA-only — full TDHF refused loudly) and
the GW imaginary-grid/Padé machinery. Method kernels stay `NotYetImplemented`
with the owning plan named in the variant, so a premature call fails loudly.

## Task 2 — libxc posture (decided once, §1.7)

- `pyscf-pbc-dft`: new `[features] default = ["libxc"]`, `libxc =
  ["pyscf-dft/libxc"]`; its `pyscf-dft` dep is now `default-features = false`.
  Pre-19-02 behavior preserved for all existing dependents (only
  `pyscf-bench` takes it with defaults — verified by grep).
- `tdscf`/`gw`/`adc`/`eph`: `pyscf-pbc-dft` with `default-features = false` +
  forwarding `libxc` feature. `x2c` untouched (no `pyscf-pbc-dft` dep — the
  one crate already free of libxc, §1.7 confirmed).
- Measured: `cargo tree -p pyscf-pbc-tdscf | grep -c libxc` → **0** off,
  **11** with `--features pyscf-pbc-tdscf/libxc`; `pyscf-bench` still **11**
  (unchanged). Scoped build of all five crates + `pyscf-pbc-dft`: ~6 s
  incremental (libxc-free path; the libxc cold build remains the documented
  >40 min in `pyscf-dft/Cargo.toml`).
- No Gate-A number can move with the feature off: no method body exists yet
  that reads the XC backend (all kernels refuse). The escalation condition is
  therefore vacuous at this plan and becomes live in 19-09 (KS response).

## Task 3 — single-CPHF gate (GRAD-10 extended)

- New `xtask check-single-cphf`: exactly one bare `pub fn solve(` in
  `crates/*/src` — `crates/pyscf-grad/src/cphf.rs`. Suffixed names
  (`solve_linear`, `solve_lambda`, `solve_cphf_rhf`) are unaffected by
  construction (prefix match on `solve(` / `solve<`).
- **Failure demonstrated**: injected `pub fn solve` into
  `pyscf-pbc-tdscf/src/davidson.rs` → `FAIL` naming the file, exit **2**;
  reverted → `PASS`, exit 0. (A lint only ever seen green is not known to work.)
- Consequence for 19-03: `pyscf-pbc-scf/src/cphf.rs` must expose k-aware
  `fvind` builders and call `pyscf_grad::cphf::solve` — any local `pub fn
  solve` fails this gate.

## Task 4 — CI

- `xtask-single-cphf` job added to `ci.yml` mirroring `xtask-dependency-wall`.

## Verification (all green 2026-09-12)

- `cargo build -p pyscf-pbc-tdscf -p pyscf-pbc-gw -p pyscf-pbc-adc -p pyscf-pbc-x2c -p pyscf-pbc-eph -p pyscf-pbc-dft` → `Finished`, no errors.
- `cargo tree -p pyscf-pbc-tdscf | grep -c libxc` → 0; with `/libxc` → 11.
- `check-single-cphf` → 0 (and 2 on injected violation, reverted).
- `check-orphan-modules`, `check-dependency-wall` → 0.
- No `pyscf-pbc-*` crate names `pyo3` (grep-clean; no new deps added at all).
