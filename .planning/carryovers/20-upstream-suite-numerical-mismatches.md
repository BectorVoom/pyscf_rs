# Carryover — 13 numerical mismatches surfaced by the upstream `pyscf/pbc` suite

**Source:** `.planning/phases/20-pbc-python-bindings/measurements/upstream-pbc-suite-after-20-19.md`
§(2) (`.so` 18:49:53, i.e. BEFORE 20-19 D's overlap-precision fix); status after D recorded in
`20-VERIFICATION.md` §3. Not triaged in Phase 20.

| node id | Δ | test tolerance |
|---|---:|---|
| `pyscf/pbc/dft/test/test_rks.py::KnownValues::test_density_fit` | **3.134** (−7.852143828842326 vs −4.717699891018736) | places=6 |
| `pyscf/pbc/dft/test/test_kuks.py::KnownValues::test_kuks_as_kuhf` | 9.36e-3 (−4.204045409900913 vs −4.213403459087) | places=9 |
| `pyscf/pbc/dft/test/test_rks.py::KnownValues::test_density_fit_2d` | 7.48e-3 | places=7 |
| `pyscf/pbc/dft/test/test_krks.py::KnownValues::test_klda8_cubic_kpt_222` | 4.32e-3 | places=7 |
| `pyscf/pbc/dft/test/test_krks.py::KnownValues::test_klda8_cubic_gamma` | 3.39e-3 | places=7 |
| `pyscf/pbc/dft/test/test_rks.py::KnownValues::test_rsh_0d` | 4.45e-4 | places=7 |
| `pyscf/pbc/dft/test/test_uks.py::KnownValues::test_pp_UKS` | 2.10e-4 | places=7 |
| `pyscf/pbc/dft/test/test_krks.py::KnownValues::test_klda8_primitive_kpt_222` | 6.95e-7 | places=7 |
| `pyscf/pbc/dft/test/test_krks.py::KnownValues::test_klda8_primitive_gamma` | 3.28e-7 | places=7 |
| `pyscf/pbc/df/test/test_band.py::KnownValues::test_fft_bands` | 7.86e-8 | places=7 |
| `pyscf/pbc/mp/test/test_ksym.py::KnownValues::test_kmp2` | 3.77e-10 | places=10 |
| `pyscf/pbc/mp/test/test_ksym.py::KnownValues::test_rdm1` | 1.53e-10 (vs 0.0) | places=10 |
| `pyscf/pbc/dft/test/test_gen_grid.py::KnownValues::test_becke_grids_round_error` | `assert 0.5111870535736216 < 0.1` | — |

## Triage order

1. **Done by 20-18:** re-measured on the post-20-19-D `.so` (19:52:34, whole-suite re-run
   `run-overlay-final`). All 13 still fail, values unchanged at the reported precision
   (`test_kuks_as_kuhf` −4.204045409901784, `test_density_fit` −7.852143828842326,
   `test_klda8_cubic_kpt_222` Δ 4.321e-3). The overlap-precision fix is NOT their cause.
   `test_becke_grids_round_error` compares `cell.vol` 1276.2505491552904 with the Becke weight sum
   1276.761736208864 on a 12-atom cell whose native mesh is `[2685, 2685, 1109]` — check the mesh too.
2. `test_density_fit` (3.13 Ha) is far too large for the overlap and is a separate defect
   (candidate: native `density_fit()` on a gamma `RKS` shim — `_j_only` / Becke grid swap, 20-13 D8).
3. `test_klda8_cubic_*` (~4e-3) vs `_primitive_*` (~5e-7): same functional, the cubic cell is
   larger — candidate: the XC grid mesh / `cell.mesh` choice for a pinned `ke_cutoff`.
4. `test_kuks_as_kuhf` 9.4e-3: KUKS with `xc='hf'`-like settings.
5. `test_becke_grids_round_error`: native `BeckeGrids` point count/weights vs upstream.
