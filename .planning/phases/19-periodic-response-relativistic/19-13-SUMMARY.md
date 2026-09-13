# 19-13 SUMMARY — slow explicit reference GW routes

**Shipped:** 2026-09-13. `kgw_slow` 4/4 green (with `gw_slow` +
`kgw_slow_supercell` in the same suite); supercell identity holds; slow
checks fast inside Gate C. Nothing optimised.

## Reference core (`crates/pyscf-pbc-gw/src/{kgw_slow,gw_slow,kgw_slow_supercell}.rs`)

- Upstream `pbc/gw/gw_slow.py` (24 l) is LITERALLY an alias of the molecular
  `pyscf.gw.gw_slow` (`IMDS/kernel/GW` re-exported) — this port mirrors that
  honestly: `kernel_gw_slow` is the shared Lehmann core at `nk = 1`, not a
  second implementation.
- `slow_sigma_real`: Lehmann diagonal `Σ_m w_m·(ω−e_m)/((ω−e_m)²+η²)` through
  `oracle_sum`; `kernel_slow_orbital`: Newton root (same `qp_newton`,
  `η = 1e-3` pinned). No grid, no fit, no blocking, no fusion.
- `replicate_supercell`: union over k with weights `/nk` (same crystal,
  Γ-only sampling); `mean_primitive_sigma`: the other side of the identity.

## Checks (`tests/kgw_slow.rs`, 4 tests)

- Lehmann hand-check + empty-set-is-zero (wiring).
- **Supercell equivalence, oracle-free**: replicated set == k-mean σ at
  1e-12 over four probe frequencies (order differs ⇒ NOT bit-identity,
  stated); at `nk = 1` all three drivers BIT-identical; gamma alias ==
  k-point driver bit-for-bit.
- **Slow checks fast**: same two-pole content through `kgw_slow` (direct)
  and 19-10 `krgw_ac` (Padé) agree inside Gate C (1e-4); routes stay named.
- Determinism 1-vs-8.

## Verification

- `cargo test -p pyscf-pbc-gw --test kgw_slow` → 4 passed.
- `cargo test -p pyscf-pbc-gw --tests` → 22 passed (6 AC + 8 CD + 4 UAC + 4 slow).
- `cargo clippy -p pyscf-pbc-gw --all-targets`: nothing in the new files.
