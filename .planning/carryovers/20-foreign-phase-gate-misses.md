# Carryover — gate misses observed during Phase 20 but OWNED by Phases 18 and 19

**Source:** `.planning/phases/20-pbc-python-bindings/20-13-FIX-SUMMARY.md` (krks_stress),
`measurements/pbc-oracle-tiers.md` §5c (Phase-19 arms); recorded by 20-18 (`20-VERIFICATION.md`).
Phase 20 did not edit these tests; no tolerance changed.

## (h) `pyscf-pbc-grad krks_stress` — three gate misses (Phase 18, in progress / paused)

| test | measured | gate | A/B |
|---|---|---|---|
| `hubbard_u_deriv1_matches_eu_fd_at_1e8` (2,2) | **1.841e-8** | 1e-8 | identical 1.841e-8 with pristine `kspu.rs` → predates 20-13-FIX; central-difference truncation (error falls 4× per step halving: 2e-4 → 1.84e-8, 1e-4 → 4.55e-9) |
| `gate_a2_krks_get_j_matches_fd` | **2.362e-9** | 2e-9 | not run |
| `gate_a2_krks_get_nuc_matches_fd` | **3.061e-9** | 2e-9 | not run; likely Phase-18 in-progress work or the D-20-E `coulG` re-key |

The whole `krks_stress` target was killed at 600 s (D-20-D) in 20-13-FIX. Also: every Phase-18
exchange-gradient number measured with the old `coulG` class key must be re-measured (D-20-E).

## (j) Phase-19 test stubs failing (Phase 19 / in-progress session)

| test | measured | expected |
|---|---|---|
| `pyscf-pbc-adc ea::gate_d_ip_spec_factors` (T1) | IP spec factors deviate **9.358e-1** (re-measured 9.3579e-1 in 20-03, exit 101) | Gate D `< 5e-4`; needs `get_trans_moments` + ADC-norm renormalisation |
| `pyscf-pbc-tdscf uhf::live_bigbox_matches_molecular` (T1) | `panic!` (unwired arm) | 2 dp vs molecular UHF-TDA; needs matrix-free `vind` for non-uniform per-k fillings |
| `pyscf-pbc-scf newton_ah::live_newton_matches_upstream_energy` (T1) | `panic!("live arm not yet wired to the KRHF Fock build")` | upstream Newton energy (see `20-periodic-newton-ah.md`) |

All three are excluded from the push `pbc-oracle` job and run as KNOWN-RED in nightly/full (20-03).
`19-VERIFICATION.md` / `.planning/carryovers/19-gw-live-oracle.md` list them as NOT RUN; the
measured values above say two panic and one computes a wrong number.
