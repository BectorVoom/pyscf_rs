# REPORT-T10 — G-space route on the host, and the switch (verify deviation)

## Verifies as written: all green

- ORACLE `pp_gspace_oracle`: 3 passed (projector 1.04e-14, ft_ao
  1.07e-14/1.51e-14, KTaO3 get_pp 3.9e-13 — was 1.4e-2).
- ORACLE `fftdf`: all pass, coarse diamond 1.27e-13 (was 1.8e-3).
- TEST `fftdf`: 7 passed (baseline) + 1 new ignored (the coarse oracle gate).
- TEST `numint_ao_budget`: 2 passed.
- ORACLE `kscf_conv_check_oracle` (2.7e-8) and `scf_ovlp_smearing_oracle`: pass.

## Deviation: `krks_dzvp_small_cell.rs` expectations re-derived from upstream

The TEST `krks_dzvp_small_cell` failed after the switch (both tests,
`converged=false` where `true` was asserted). Isolation
(`PYSCF_PBC_FFTDF_PP_NL=realspace`) reproduces the failure, so the cause is
T06's `conv_check`, not the T10 route: the loop converges, then the final
diagonalisation re-smears the occupations (shift 0.3 + gauss 0.03) and the
re-test flips `converged` to false. This is upstream behaviour (DESIGN T2),
not a regression.

Live upstream numbers (vendored PySCF 2.12.1, same TaO cell/recipe):

| case | upstream | port (new code) |
|---|---|---|
| SZV 1e-7, shift 0.15 | false, 21c, e −74.12862803 | e −74.12862733 (Δ 7e-7), flag TRUE |
| DZVP 1e-7, shift 0.15 | false, 22c, e −74.20875209 | e −74.20875119 (Δ 9e-7), false |
| DZVP/MINAO 1e-9, shift 0.15 | false, 29c, e −74.20875536486477 | 29c, e −74.2087553990 (Δ 3.4e-9), false |

Changes made (test file only, T03-clause + T06-principle):

1. `level_shift: 0.3 → 0.15` (3 sites) + one-line comment — T03's if-fails
   clause verbatim: for a restricted run the new formula at half the shift
   gives the same orbitals as the old formula at full shift. Restored
   baseline convergence speed (29/25 cycles, was 29/24).
2. Test 1 asserts upstream-derived energies (2e-6) + DZVP flag false +
   DZVP-below-SZV, instead of `converged == true`.
3. Test 2 asserts upstream-derived MINAO energy (2e-6) + flag false; keeps
   projection identity, electron count, same-solution (4e-8), cycle-saving.

## Known boundary left unasserted (out of scope, D1–D3)

SZV `converged`: upstream false, port true, energies agreeing to 7e-7. The
re-test gradient is exactly 0.0 on the port (wide smearing leaves no
exactly-zero occupation, so the virtual set — and `kocc::get_grad` — is
empty) while upstream's is ~1e-3. Occupation-threshold dust in the smearing
path, untouched by this plan; asserting either value as a gate would enshrine
a dust boundary. Recorded here so it is not lost.
