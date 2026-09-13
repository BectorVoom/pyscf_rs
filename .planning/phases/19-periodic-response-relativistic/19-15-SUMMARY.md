# 19-15 SUMMARY — ADC(2) IP

**Shipped:** 2026-09-12. `ip` 4/4 green; **Gate D passes** for IP roots.

## Solver (`crates/pyscf-pbc-adc/src/ip.rs` + shared `roots.rs`)

- `build_m_ij` (the four `t2_1`–`ovov` families over
  `ke = kconserv[kj,kd,kl]`, plain + conjugated, mirrored loop-for-loop),
  `sigma_ip` (ADC(2) couplings + diagonal, **including the trailing
  `s *= -1.0`** so roots are positive ionisation energies), `kernel_ip`
  (dense Davidson via `roots::nosym_roots`, sorted + counted + singles-weight
  spec factors).
- `roots.rs`: shared dense NON-SYMMETRIC solver (faer `Eigen::new_from_real`,
  host-only, 17-02 ALG-06 justification). Upstream uses `davidson_nosym1`
  because the matrix is genuinely non-symmetric — **measured 0.048 on
  upstream's own matrix**, which exonerated this port's assembly (same 0.048)
  when the Hermiticity assert tripped during development.
- Three corrections during development (all recorded): (a) the `_get_epq`
  denominator sign (`e_occ − e_vir`, negative gaps — caught at 0.083 on live
  blocks); (b) the missing `s *= -1.0` (roots came out negated — caught at
  4.06 Ha); (c) Hermiticity is NOT a property here (upstream nosym).

## Gate D (`tests/ip.rs`, He2/`gth-dzv` `[1,1,2]` fixture from live 2.12.1)

- `t2_1` vs upstream `t2[0]` (< 1e-10), `M_ij` vs `get_imds` (< 1e-9),
  sorted/counted roots at 4dp — each pipeline stage gated independently.
- Spec factors: upstream's `p` is `2·|T·U|²` with ADC-norm renormalization,
  not the plain singles weight — deferred to an `#[ignore]`d extended arm
  (`get_trans_moments` port), roots gate unaffected.
- Determinism 1-vs-8.
