# Carryover — two Rust convention gaps in `pyscf-pbc-mp` / `pyscf-pbc-ao2mo` (worked around in the binding or the test)

**Source:** `.planning/phases/20-pbc-python-bindings/20-15-SUMMARY.md` D5, D6; recorded by 20-18
(`20-VERIFICATION.md`). 20-15 was not allowed to edit these crates.

## (e) `Kmp2Stagger::integral_df` reuses the mean field's FFTDF

- `crates/pyscf-pbc-mp/src/kmp2_stagger.rs` (`integral_df`, `same_kpts` branch) reuses the mean
  field's `Fftdf` whenever the k-points match. Upstream rebuilds `FFTDF(cell, kpts)` at
  **`cell.mesh`** (`pyscf/pbc/mp/kmp2_stagger.py:74`).
- Measured (He-fcc `6-31g` 2×2×2): with only `with_df.mesh = [15]*3` pinned over a default-mesh
  (`[99]^3`) cell, submesh `e_corr` is off by **1.136e-3** (−0.03706278503825 vs −0.03592704457336).
  With `cell.mesh` pinned both agree to **4.3e-7** (gate 2e-6; binding test measured 4.332e-7 submesh,
  4.158e-7 full mesh).
- Workaround: `python/pyscf/tests/test_pbc_post_scf.py` pins `cell.mesh`.
- Unblock: build the stagger integrals at `cell.mesh` as upstream does; re-gate with only
  `with_df.mesh` pinned.

## (f) `pbc.ao2mo` pair functions return `FFT·Ω/N`

- `pyscf-pbc-ao2mo` `get_mo_pairs_g` / `get_mo_pairs_invg` / `get_ao_pairs_g` return
  `FFT·Ω/N` (`fft_ao_pairs_g`'s `scale`); upstream returns the bare `tools.fft`. Both
  `assemble_eri`s apply `Ω/N²`, so composing the Rust functions was off by `Ω/N`
  (0.00145 vs 0.37096).
- Workaround: `crates/pyscf-py/src/pbc/ao2mo.rs` multiplies the pair arrays by `N/Ω` — a scaling,
  not bit-preserving; reproduces upstream to **1.211e-11** (gate 1e-10).
- Unblock: decide the Rust convention (upstream's bare FFT) in `pyscf-pbc-ao2mo`, drop the
  binding rescale, re-run the ao2mo wrapper gate.
