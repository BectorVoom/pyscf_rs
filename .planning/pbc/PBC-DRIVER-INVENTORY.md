# PBC Driver Inventory — every upstream module → Rust target → phase

**Companion to** `PBC-MASTER-PLAN.md`. **186 modules, ~78,000 lines.**

Columns:
- **Upstream** — the file under `pyscf/pbc/` to port (line count).
- **Rust target** — the file to create.
- **Ph** — phase from `PBC-MASTER-PLAN.md §7`.
- **GPU** — `Y` if the plan assigns a cubecl kernel (K-nn ids from §6); `—` otherwise.

Status legend for the tracking column: `[ ]` not started · `[~]` partial · `[x]` done.

---

## pbc/gto — Cell, lattice, pseudopotentials  (16 files, 4,593 lines)

| St | Upstream | L | Rust target | Ph | GPU |
|---|---|---:|---|---|---|
| [ ] | `gto/cell.py` (class + build + properties) | 2142 | `pyscf-pbc-gto/src/cell.rs`, `types.rs` | 9 | — |
| [ ] | `gto/cell.py` `get_Gv`/`get_Gv_weights`/`get_SI`/`get_uniform_grids` | — | `pyscf-pbc-gto/src/gv.rs` + `pyscf-kernels/src/pbc/{gv,struct_factor}.rs` | 9 | K-01,K-02 |
| [ ] | `gto/cell.py` `estimate_rcut`/`ke_cutoff`/`pgf_rcut`/`rcut_by_shells` | — | `pyscf-pbc-gto/src/cutoff.rs` | 9 | — |
| [ ] | `gto/cell.py` `make_kpts` | — | `pyscf-pbc-gto/src/kpts_mesh.rs` | 9 | — |
| [ ] | `gto/cell.py` `ewald`/`get_ewald_params` | — | `pyscf-pbc-gto/src/ewald.rs` + `pyscf-kernels/src/pbc/ewald.rs` | 9 | K-05,K-06 |
| [ ] | `gto/cell.py` `intor_cross`/`pbc_intor`/`_intor_cross_screened`/`conc_cell` | — | `pyscf-pbc-gto/src/pbc_intor.rs` + `pyscf-kernels/src/pbc/bloch.rs` | 10 | K-07 |
| [ ] | `gto/cell.py` `tostring`/`tofile`/`fromfile`/`_parse_poscar`/`_parse_cif` | — | `pyscf-pbc-gto/src/io.rs` | 20 | — |
| [ ] | `gto/cell.py` `pack`/`unpack`/`dumps`/`loads` | — | `pyscf-pbc-gto/src/dumps_loads.rs` | 9 | — |
| [ ] | `gto/ewald_methods.py` (b-splines, particle-mesh Ewald) | 293 | `pyscf-pbc-gto/src/ewald_pme.rs` | 9 | K-06 |
| [ ] | `gto/eval_gto.py` | 257 | `pyscf-pbc-gto/src/eval_gto.rs` + `pyscf-kernels/src/pbc/eval_ao_k.rs` | 10 | K-08 |
| [ ] | `gto/neighborlist.py` | 200 | `pyscf-pbc-gto/src/neighborlist.rs` | 10 | — |
| [ ] | `gto/pseudo/pp.py` | 289 | `pyscf-pbc-gto/src/pseudo/vloc.rs`, `vnl.rs` + `pyscf-kernels/src/pbc/{gth_vloc,gth_projg}.rs` | 10 | K-09,K-10 |
| [ ] | `gto/pseudo/pp_int.py` | 675 | `pyscf-pbc-gto/src/pseudo/{vloc,vnl,fakecell}.rs` | 10 (grad half → 18) | K-09,K-10 |
| [ ] | `gto/pseudo/ppnl_velgauge.py` | 305 | `pyscf-pbc-gto/src/pseudo/velgauge.rs` | 19 | — |
| [ ] | `gto/pseudo/__init__.py` | 23 | `pyscf-pbc-gto/src/pseudo/mod.rs` | 10 | — |
| [ ] | `gto/ecp.py` | 78 | `pyscf-pbc-gto/src/ecp.rs` (reuses `pyscf_gto::CintxEcpEngine` over images) | 10 | — |
| [ ] | `gto/_pbcintor.py` | 67 | folded into `pbc_intor.rs` | 10 | — |
| [ ] | `gto/basis/*` (3 splitter scripts) | 149 | already covered by `pyscf-gto/src/basis/cp2k.rs` | — | — |

## pbc/tools  (9 files, 3,102 lines)

| St | Upstream | L | Rust target | Ph | GPU |
|---|---|---:|---|---|---|
| [ ] | `tools/pbc.py` `fft`/`ifft`/`fftk`/`ifftk`/`_fftn_blas`/`_ifftn_blas` | — | `pyscf-pbc-tools/src/fft.rs` + `pyscf-kernels/src/pbc/fft.rs` | 11 | K-11,K-12 |
| [ ] | `tools/pbc.py` `get_coulG`/`precompute_exx`/`_Gv_wrap_around` | — | `pyscf-pbc-tools/src/coulg.rs` + `pyscf-kernels/src/pbc/coulg.rs` | 11 | K-03 |
| [ ] | `tools/pbc.py` `madelung`/`get_monkhorst_pack_size` | — | `pyscf-pbc-tools/src/madelung.rs` | 11 | — |
| [ ] | `tools/pbc.py` `get_lattice_Ls`/`check_lattice_sum_range`/`round_to_cell0` | — | `pyscf-pbc-tools/src/lattice.rs` | 9 | — |
| [ ] | `tools/pbc.py` `super_cell`/`cell_plus_imgs`/`_build_supcell_` | — | `pyscf-pbc-tools/src/supercell.rs` | 9 | — |
| [ ] | `tools/pbc.py` `cutoff_to_mesh`/`mesh_to_cutoff`/`cutoff_to_gs`/`gs_to_cutoff` | — | `pyscf-pbc-tools/src/mesh.rs` | 9 | — |
| [ ] | `tools/k2gamma.py` | 345 | `pyscf-pbc-tools/src/k2gamma.rs` | 20 | — |
| [ ] | `tools/lattice.py` | 171 | `pyscf-pbc-tools/src/lattice_db.rs` | 20 | — |
| [ ] | `tools/make_test_cell.py` | 160 | `pyscf-pbc-gto/tests/common/systems.rs` | 9 | — |
| [ ] | `tools/tril.py` | 52 | `pyscf-pbc-tools/src/tril.rs` | 20 | — |
| [ ] | `tools/print_funcs.py` | 48 | `pyscf-pbc-tools/src/print_funcs.rs` | 20 | — |
| [ ] | `tools/pyscf_ase.py` | 286 | `pyscf-py` binding + `python/pyscf/pbc/tools/pyscf_ase.py` | 20 | — |
| [ ] | `tools/pywannier90.py` | 1184 | optional shim, `wannier90` feature | 20 | — |

## pbc/lib  (7 files, 3,447 lines)

| St | Upstream | L | Rust target | Ph | GPU |
|---|---|---:|---|---|---|
| [ ] | `lib/kpts_helper.py` | 635 | `pyscf-pbc-lib/src/kpts_helper.rs` + `pyscf-kernels/src/pbc/kconserv.rs` | 9 (basic) / 15 (kconserv3) | K-16 |
| [ ] | `lib/kpts.py` (`KPoints`, IBZ machinery) | 1223 | `pyscf-pbc-lib/src/kpts.rs` | 17 | — |
| [ ] | `lib/ktensor.py` | 386 | `pyscf-pbc-lib/src/ktensor.rs` | 15 | — |
| [ ] | `lib/linalg_helper.py` | 858 | `pyscf-pbc-lib/src/linalg_helper.rs` (complex Davidson) | 16 | — |
| [ ] | `lib/arnoldi.py` | 277 | `pyscf-pbc-lib/src/arnoldi.rs` | 16 | — |
| [ ] | `lib/chkfile.py` | 54 | `pyscf-pbc-lib/src/chkfile.rs` | 11 | — |

## pbc/df  (21 files, 13,666 lines)

| St | Upstream | L | Rust target | Ph | GPU |
|---|---|---:|---|---|---|
| [ ] | `df/fft.py` | 406 | `pyscf-pbc-df/src/fftdf.rs`, `traits.rs` | 11 | — |
| [ ] | `df/fft_jk.py` | 520 | `pyscf-pbc-df/src/fft_jk.rs` + `pyscf-kernels/src/pbc/{rho_k,vmat}.rs` | 11 | K-13,K-14 |
| [ ] | `df/fft_ao2mo.py` | 484 | `pyscf-pbc-df/src/fft_ao2mo.rs` | 15 | — |
| [ ] | `df/ft_ao.py` | 790 | `pyscf-pbc-df/src/ft_ao.rs` + `pyscf-kernels/src/pbc/ft_aopair.rs` | 13 | **K-15** |
| [ ] | `df/aft.py` | 776 | `pyscf-pbc-df/src/aftdf.rs` | 13 | — |
| [ ] | `df/aft_jk.py` | 753 | `pyscf-pbc-df/src/aft_jk.rs` | 13 | — |
| [ ] | `df/aft_ao2mo.py` | 434 | `pyscf-pbc-df/src/aft_ao2mo.rs` | 13 | — |
| [ ] | `df/incore.py` | 731 | `pyscf-pbc-df/src/incore.rs` | 14 | — |
| [ ] | `df/gdf_builder.py` | 1062 | `pyscf-pbc-df/src/gdf_builder.rs` | 14 | — |
| [ ] | `df/df.py` (`GDF`) | 1029 | `pyscf-pbc-df/src/gdf.rs` | 14 | — |
| [ ] | `df/df_jk.py` | 1552 | `pyscf-pbc-df/src/df_jk.rs` (+ `exxdiv.rs`) | 11 (exxdiv half) / 14 | — |
| [ ] | `df/df_ao2mo.py` | 351 | `pyscf-pbc-df/src/df_ao2mo.rs` | 14 | — |
| [ ] | `df/outcore.py` | 250 | `pyscf-pbc-df/src/outcore.rs` (HDF5) | 14 | — |
| [ ] | `df/mdf.py` | 460 | `pyscf-pbc-df/src/mdf.rs` | 14 | — |
| [ ] | `df/mdf_jk.py` | 149 | `pyscf-pbc-df/src/mdf_jk.rs` | 14 | — |
| [ ] | `df/mdf_ao2mo.py` | 176 | `pyscf-pbc-df/src/mdf_ao2mo.rs` | 14 | — |
| [ ] | `df/rsdf_helper.py` | 1348 | `pyscf-pbc-df/src/rsdf_helper.rs` | 14 | — |
| [ ] | `df/rsdf_builder.py` | 1631 | `pyscf-pbc-df/src/rsdf_builder.rs` | 14 | — |
| [ ] | `df/rsdf.py` | 680 | `pyscf-pbc-df/src/rsdf.rs` | 14 | — |
| [ ] | `df/rsdf_jk.py` | 53 | `pyscf-pbc-df/src/rsdf_jk.rs` | 14 | — |

## pbc/scf  (21 files, 7,693 lines)

| St | Upstream | L | Rust target | Ph | GPU |
|---|---|---:|---|---|---|
| [ ] | `scf/khf.py` (`KSCF`, `KRHF`) | 864 | `pyscf-pbc-scf/src/{kscf,krhf,khooks,kocc,krdm,kenergy,kdiis}.rs` | 11 | — |
| [ ] | `scf/kuhf.py` | 635 | `pyscf-pbc-scf/src/kuhf.rs` | 11 | — |
| [ ] | `scf/krohf.py` | 386 | `pyscf-pbc-scf/src/krohf.rs` | 11 | — |
| [ ] | `scf/kghf.py` | 323 | `pyscf-pbc-scf/src/kghf.rs` | 11 | — |
| [ ] | `scf/hf.py` (gamma-point SCF) | 1003 | `pyscf-pbc-scf/src/hf.rs` | 11 | — |
| [ ] | `scf/uhf.py` | 292 | `pyscf-pbc-scf/src/uhf.rs` | 11 | — |
| [ ] | `scf/rohf.py` | 136 | `pyscf-pbc-scf/src/rohf.rs` | 11 | — |
| [ ] | `scf/ghf.py` | 196 | `pyscf-pbc-scf/src/ghf.rs` | 11 | — |
| [ ] | `scf/smearing.py` | 191 | `pyscf-pbc-scf/src/smearing.rs` | 11 | — |
| [ ] | `scf/addons.py` | 379 | `pyscf-pbc-scf/src/addons.rs` | 11 | — |
| [ ] | `scf/chkfile.py` | 25 | `pyscf-pbc-scf/src/chkfile.rs` | 11 | — |
| [ ] | `scf/scfint.py` | 70 | `pyscf-pbc-scf/src/scfint.rs` | 11 | — |
| [ ] | `scf/khf_ksymm.py` | 410 | `pyscf-pbc-scf/src/khf_ksymm.rs` | 17 | — |
| [ ] | `scf/kuhf_ksymm.py` | 219 | `pyscf-pbc-scf/src/kuhf_ksymm.rs` | 17 | — |
| [ ] | `scf/kghf_ksymm.py` | 211 | `pyscf-pbc-scf/src/kghf_ksymm.rs` | 17 | — |
| [ ] | `scf/rsjk.py` | 1355 | `pyscf-pbc-df/src/rsjk.rs` | 14 | — |
| [ ] | `scf/newton_ah.py` | 303 | `pyscf-pbc-scf/src/newton_ah.rs` | 19 | — |
| [ ] | `scf/stability.py` | 329 | `pyscf-pbc-scf/src/stability.rs` | 19 | — |
| [ ] | `scf/cphf.py` | 176 | `pyscf-pbc-scf/src/cphf.rs` (**reuse `pyscf-grad`'s single Krylov solver**) | 19 | — |
| [ ] | `scf/_response_functions.py` | 47 | `pyscf-pbc-scf/src/response.rs` | 19 | — |

## pbc/dft  (25 files, 9,313 lines)

| St | Upstream | L | Rust target | Ph | GPU |
|---|---|---:|---|---|---|
| [ ] | `dft/gen_grid.py` | 294 | `pyscf-pbc-gto/src/grids.rs` (re-exported by `pyscf-pbc-dft`) | 11 | — |
| [ ] | `dft/numint.py` | 1346 | `pyscf-pbc-dft/src/numint.rs` | 12 | K-13,K-14 |
| [ ] | `dft/numint2c.py` | 642 | `pyscf-pbc-dft/src/numint2c.rs` | 12 | — |
| [ ] | `dft/krks.py` | 292 | `pyscf-pbc-dft/src/krks.rs` | 12 | — |
| [ ] | `dft/kuks.py` | 189 | `pyscf-pbc-dft/src/kuks.rs` | 12 | — |
| [ ] | `dft/kroks.py` | 75 | `pyscf-pbc-dft/src/kroks.rs` | 12 | — |
| [ ] | `dft/kgks.py` | 173 | `pyscf-pbc-dft/src/kgks.rs` | 12 | — |
| [ ] | `dft/rks.py` | 447 | `pyscf-pbc-dft/src/rks.rs` | 12 | — |
| [ ] | `dft/uks.py` | 173 | `pyscf-pbc-dft/src/uks.rs` | 12 | — |
| [ ] | `dft/roks.py` | 70 | `pyscf-pbc-dft/src/roks.rs` | 12 | — |
| [ ] | `dft/gks.py` | 158 | `pyscf-pbc-dft/src/gks.rs` | 12 | — |
| [ ] | `dft/krkspu.py` | 325 | `pyscf-pbc-dft/src/krkspu.rs` | 12 | — |
| [ ] | `dft/kukspu.py` | 301 | `pyscf-pbc-dft/src/kukspu.rs` | 12 | — |
| [ ] | `dft/cdft.py` | 154 | `pyscf-pbc-dft/src/cdft.rs` | 12 | — |
| [ ] | `dft/krks_ksymm.py` | 144 | `pyscf-pbc-dft/src/krks_ksymm.rs` | 17 | — |
| [ ] | `dft/kuks_ksymm.py` | 147 | `pyscf-pbc-dft/src/kuks_ksymm.rs` | 17 | — |
| [ ] | `dft/krkspu_ksymm.py` | 72 | `pyscf-pbc-dft/src/krkspu_ksymm.rs` | 17 | — |
| [ ] | `dft/kukspu_ksymm.py` | 59 | `pyscf-pbc-dft/src/kukspu_ksymm.rs` | 17 | — |
| [ ] | `dft/multigrid/multigrid.py` | 1962 | `pyscf-pbc-dft/src/multigrid/mod.rs`, `collocate.rs` | 17 | Y (new) |
| [ ] | `dft/multigrid/multigrid_pair.py` | 1257 | `pyscf-pbc-dft/src/multigrid/pair.rs` | 17 | Y (new) |
| [ ] | `dft/multigrid/pp.py` | 256 | `pyscf-pbc-dft/src/multigrid/pp.rs` | 17 | — |
| [ ] | `dft/multigrid/utils.py` + `_backend_c.py` | 640 | `pyscf-pbc-dft/src/multigrid/utils.rs` | 17 | — |

## pbc/ao2mo + pbc/mp  (8 files, 2,358 lines)

| St | Upstream | L | Rust target | Ph |
|---|---|---:|---|---|
| [ ] | `ao2mo/eris.py` | 258 | `pyscf-pbc-ao2mo/src/eris.rs` | 15 |
| [ ] | `mp/kmp2.py` | 821 | `pyscf-pbc-mp/src/kmp2.rs` | 15 |
| [ ] | `mp/kump2.py` | 423 | `pyscf-pbc-mp/src/kump2.rs` | 15 |
| [ ] | `mp/kmp2_stagger.py` | 419 | `pyscf-pbc-mp/src/kmp2_stagger.rs` | 15 |
| [ ] | `mp/kmp2_ksymm.py` | 285 | `pyscf-pbc-mp/src/kmp2_ksymm.rs` | 17 |
| [ ] | `mp/mp2.py` | 95 | `pyscf-pbc-mp/src/mp2.rs` (gamma shim) | 15 |

## pbc/cc + pbc/ci  (22 files, 14,527 lines)

| St | Upstream | L | Rust target | Ph |
|---|---|---:|---|---|
| [ ] | `cc/kintermediates_rhf.py` | 926 | `pyscf-pbc-cc/src/kintermediates_rhf.rs` | 16 |
| [ ] | `cc/kccsd_rhf.py` | 1203 | `pyscf-pbc-cc/src/kccsd_rhf.rs` | 16 |
| [ ] | `cc/kintermediates_uhf.py` | 1225 | `pyscf-pbc-cc/src/kintermediates_uhf.rs` | 16 |
| [ ] | `cc/kccsd_uhf.py` | 1116 | `pyscf-pbc-cc/src/kccsd_uhf.rs` | 16 |
| [ ] | `cc/kintermediates.py` | 529 | `pyscf-pbc-cc/src/kintermediates.rs` | 16 |
| [ ] | `cc/kccsd.py` (GHF) | 833 | `pyscf-pbc-cc/src/kccsd.rs` | 16 |
| [ ] | `cc/kccsd_t.py` | 319 | `pyscf-pbc-cc/src/kccsd_t.rs` | 16 |
| [ ] | `cc/kccsd_t_rhf.py` + `_slow.py` | 922 | `pyscf-pbc-cc/src/kccsd_t_rhf.rs` | 16 |
| [ ] | `cc/eom_kccsd_rhf.py` + ip + ea | 1874 | `pyscf-pbc-cc/src/eom_kccsd_rhf.rs` | 16 |
| [ ] | `cc/eom_kccsd_uhf.py` | 1275 | `pyscf-pbc-cc/src/eom_kccsd_uhf.rs` | 16 |
| [ ] | `cc/eom_kccsd_ghf.py` | 2011 | `pyscf-pbc-cc/src/eom_kccsd_ghf.rs` | 16 |
| [ ] | `cc/kuccsd_rdm.py` | 157 | `pyscf-pbc-cc/src/kuccsd_rdm.rs` | 16 |
| [ ] | `cc/ccsd.py` | 157 | `pyscf-pbc-cc/src/ccsd.rs` (gamma shim) | 16 |
| [ ] | `cc/kccsd_rhf_ksymm.py` + `kintermediates_rhf_ksymm.py` | 1071 | `pyscf-pbc-cc/src/kccsd_rhf_ksymm.rs` | 17 |
| [ ] | `ci/kcis_rhf.py` | 700 | `pyscf-pbc-ci/src/kcis_rhf.rs` | 16 |
| [ ] | `ci/cisd.py` | 116 | `pyscf-pbc-ci/src/cisd.rs` | 16 |

## pbc/symm  (8 files, 1,767 lines)

| St | Upstream | L | Rust target | Ph |
|---|---|---:|---|---|
| [ ] | `symm/geom.py` | 245 | `pyscf-pbc-symm/src/geom.rs` | 17 |
| [ ] | `symm/group.py` | 476 | `pyscf-pbc-symm/src/group.rs` | 17 |
| [ ] | `symm/space_group.py` | 369 | `pyscf-pbc-symm/src/space_group.rs` | 17 |
| [ ] | `symm/symmetry.py` | 348 | `pyscf-pbc-symm/src/symmetry.rs` | 17 |
| [ ] | `symm/basis.py` | 161 | `pyscf-pbc-symm/src/basis.rs` | 17 |
| [ ] | `symm/tables.py` | 100 | `pyscf-pbc-symm/src/tables.rs` | 17 |
| [ ] | `symm/pyscf_spglib.py` | 49 | optional `spglib` feature shim | 17 |

## pbc/grad + pbc/geomopt  (17 files, 3,117 lines)

| St | Upstream | L | Rust target | Ph |
|---|---|---:|---|---|
| [ ] | `grad/krhf.py` | 418 | `pyscf-pbc-grad/src/krhf.rs` | 18 |
| [ ] | `grad/kuhf.py` | 124 | `pyscf-pbc-grad/src/kuhf.rs` | 18 |
| [ ] | `grad/krks.py` | 141 | `pyscf-pbc-grad/src/krks.rs` | 18 |
| [ ] | `grad/kuks.py` | 135 | `pyscf-pbc-grad/src/kuks.rs` | 18 |
| [ ] | `grad/krkspu.py` | 142 | `pyscf-pbc-grad/src/krkspu.rs` | 18 |
| [ ] | `grad/kukspu.py` | 83 | `pyscf-pbc-grad/src/kukspu.rs` | 18 |
| [ ] | `grad/rhf.py` | 188 | `pyscf-pbc-grad/src/rhf.rs` | 18 |
| [ ] | `grad/uhf.py` | 103 | `pyscf-pbc-grad/src/uhf.rs` | 18 |
| [ ] | `grad/rks.py` + `uks.py` | 58 | `pyscf-pbc-grad/src/{rks,uks}.rs` | 18 |
| [ ] | `grad/krks_stress.py` | 404 | `pyscf-pbc-grad/src/krks_stress.rs` | 18 |
| [ ] | `grad/kuks_stress.py` | 308 | `pyscf-pbc-grad/src/kuks_stress.rs` | 18 |
| [ ] | `grad/rks_stress.py` | 462 | `pyscf-pbc-grad/src/rks_stress.rs` | 18 |
| [ ] | `grad/uks_stress.py` | 246 | `pyscf-pbc-grad/src/uks_stress.rs` | 18 |
| [ ] | `geomopt/geometric_solver.py` | 246 | `pyscf-pbc-geomopt/src/solver.rs` | 18 |

## pbc/tdscf + gw + adc + x2c + eph  (28 files, 8,676 lines)

| St | Upstream | L | Rust target | Ph |
|---|---|---:|---|---|
| [ ] | `tdscf/krhf.py` | 537 | `pyscf-pbc-tdscf/src/krhf.rs` | 19 |
| [ ] | `tdscf/kuhf.py` | 540 | `pyscf-pbc-tdscf/src/kuhf.rs` | 19 |
| [ ] | `tdscf/rhf.py` | 238 | `pyscf-pbc-tdscf/src/rhf.rs` | 19 |
| [ ] | `tdscf/uhf.py` | 268 | `pyscf-pbc-tdscf/src/uhf.rs` | 19 |
| [ ] | `tdscf/{krks,kuks,rks,uks}.py` | 205 | `pyscf-pbc-tdscf/src/{krks,kuks,rks,uks}.rs` | 19 |
| [ ] | `gw/krgw_ac.py` | 644 | `pyscf-pbc-gw/src/krgw_ac.rs` | 19 |
| [ ] | `gw/krgw_cd.py` | 704 | `pyscf-pbc-gw/src/krgw_cd.rs` | 19 |
| [ ] | `gw/kugw_ac.py` | 784 | `pyscf-pbc-gw/src/kugw_ac.rs` | 19 |
| [ ] | `gw/kgw_slow.py` + `kgw_slow_supercell.py` + `gw_slow.py` | 328 | `pyscf-pbc-gw/src/slow.rs` | 19 |
| [ ] | `adc/kadc_rhf.py` | 326 | `pyscf-pbc-adc/src/kadc_rhf.rs` | 19 |
| [ ] | `adc/kadc_rhf_ip.py` | 1061 | `pyscf-pbc-adc/src/kadc_rhf_ip.rs` | 19 |
| [ ] | `adc/kadc_rhf_ea.py` | 1324 | `pyscf-pbc-adc/src/kadc_rhf_ea.rs` | 19 |
| [ ] | `adc/kadc_rhf_amplitudes.py` | 346 | `pyscf-pbc-adc/src/amplitudes.rs` | 19 |
| [ ] | `adc/kadc_ao2mo.py` | 294 | `pyscf-pbc-adc/src/ao2mo.rs` | 19 |
| [ ] | `adc/dfadc.py` | 62 | `pyscf-pbc-adc/src/dfadc.rs` | 19 |
| [ ] | `x2c/sfx2c1e.py` | 355 | `pyscf-pbc-x2c/src/sfx2c1e.rs` | 19 |
| [ ] | `x2c/x2c1e.py` | 286 | `pyscf-pbc-x2c/src/x2c1e.rs` | 19 |
| [ ] | `eph/eph_fd.py` | 181 | `pyscf-pbc-eph/src/eph_fd.rs` | 19 |

## pbc/mpicc + pbc/mpitools  (10 files, 5,797 lines) — feature `mpi`, default OFF

| St | Upstream | L | Rust target | Ph |
|---|---|---:|---|---|
| [ ] | `mpicc/kccsd_rhf.py` | 3279 | `pyscf-pbc-mpi/src/kccsd_rhf.rs` | 20 |
| [ ] | `mpicc/kintermediates_rhf.py` | 1553 | `pyscf-pbc-mpi/src/kintermediates_rhf.rs` | 20 |
| [ ] | `mpicc/mpi_kpoint_helper.py` | 119 | `pyscf-pbc-mpi/src/kpoint_helper.rs` | 20 |
| [ ] | `mpitools/mpi.py` | 308 | `pyscf-pbc-mpi/src/mpi.rs` | 20 |
| [ ] | `mpitools/mpi_pool.py` | 176 | `pyscf-pbc-mpi/src/pool.rs` | 20 |
| [ ] | `mpitools/mpi_load_balancer.py` | 173 | `pyscf-pbc-mpi/src/load_balancer.rs` | 20 |
| [ ] | `mpitools/mpi_blksize.py` + `mpi_helper.py` | 151 | `pyscf-pbc-mpi/src/{blksize,helper}.rs` | 20 |

---

## Python-shim surface (`pyscf-py` + `python/pyscf/pbc/`) — Phase 20 plan 20-05

Every `__init__.py` under `pyscf/pbc/` gets a matching re-export shim so that
`from pyscf.pbc import gto, scf, dft, df, mp, cc, ci, grad, geomopt, tools, symm, tdscf, gw, adc, x2c` works:

```
python/pyscf/pbc/__init__.py          gto/  scf/  dft/  df/  mp/  cc/  ci/
python/pyscf/pbc/ao2mo/  grad/  geomopt/  tools/  lib/  symm/
python/pyscf/pbc/tdscf/  tddft/  gw/  adc/  eph/  x2c/  mpicc/  mpitools/
```
Each shim is 3–10 lines: `from pyscf._native.pbc.<mod> import *`.
