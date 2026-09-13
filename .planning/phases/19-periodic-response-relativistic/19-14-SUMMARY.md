# 19-14 SUMMARY — ADC base (kadc_ao2mo + kadc_rhf + amplitudes)

**Shipped:** 2026-09-12. All four tasks complete; `kadc_base` 7/7 green.

## Task 1 — `kadc_ao2mo` over the shipped transform

- `KadcEris` (six incore blocks as `CTensor`, row-major `[ki][kj][ka]` +
  MO axes) and `build_incore` (per-triple `iks = kconserv`, full-`nmo⁴`
  fetch, six-way slicing with `/nkpts`).
- The 4-index MO block comes from **`pyscf_pbc_ao2mo::general`** via
  `ao2mo_block` (new `pyscf-pbc-ao2mo`/`pyscf-pbc-df`/`pyscf-pbc-gto` deps;
  `Eri.data` row-major `[(p,q),(r,s)]` IS `[p][q][r][s]` — no reshape, with a
  hard `nmo⁴` length assertion so a packed backend fails loudly). No new
  4-index transform written. Named accessors (`ovov_at`, `oovv_at`) replace
  free gathers.

## Task 2 — driver base and amplitudes

- `t2_first_order` (`t2[ki,kj,ka][i,j,a,b] = conj(ovov[ki,ka,kj][i,a,j,b]) /
  eijab`, `kb = kconserv[ki,ka,kj]`), `adc2_energy` (`2·direct − exchange`,
  real part, `/nkpts`), `KadcDriver::kernel_gs`. Only `adc(2)` executes;
  `(2)-x`/`(3)` refuse (their `t1_2`/`t2_2` land with the manifolds).

## Task 3 — element-wise index oracle

- Synthetic provider encodes every index (k AND mo) into distinguishable
  re/im values; all six blocks asserted element-wise (`nocc=2 != nvir=3`).
- Discrimination demonstrated in-test: the swapped-gather read provably
  differs (`oracle_catches_transposed_k_gather`).

## Task 4 — arena discipline

- Blocks allocated with `CTensor::zeros` + explicit re-zero before reuse;
  asserted zero (Task-4 test), on the `nocc != nvir` fixture throughout.

## Verification

- `cargo test -p pyscf-pbc-adc --test kadc_base` → 7 passed.
- One investigation closed: hand-energy vs `adc2_energy` differed in the last
  ulp (different summation orders on a 1e11-scale total) — test now uses
  relative tolerance with the reason recorded, not a loosened absolute one.
- Determinism 1-vs-8 in-process (bit-identical).
