# PBC oracle tiers — measured runtime and outcome of every `#[ignore]`d PBC gate

**Written:** 2026-09-14 by plan 20-02 (finalisation). **Tiering is by measured
runtime only, never by outcome.** A fast failing gate is T1 and failing. No
tolerance was loosened. The only source edits were the `#[ignore = "..."]`
strings (Task 5).

| tier | rule | CI trigger (owned by 20-03) |
|---|---|---|
| **T1** | < 30 s | every push / PR |
| **T2** | 30 s – 5 min | nightly `schedule` |
| **T3** | > 5 min, or `TIMEOUT@600 s` (D-20-D: killed, outcome not observed) | `workflow_dispatch` only |

Raw material: `target/p20-02-logs/` — `run.status` (one JSON line per run), per-test
`<crate>__<target>__<test>.log` + `.log.time` (`/usr/bin/time -f '%e %M'`),
`gates-v3.json` (the enumeration below), `tiers_table.py` (generates §3 from those),
`encode_tiers.py` (Task 5). Pre-rerun logs are kept in `pre-20-05/` and `pre-20-04fix/`.

## §1 — Enumeration, and why it is not 161

`grep -rn '#\[ignore' crates/pyscf-pbc-*/tests/*.rs`, run **2026-09-14 14:46:01**:
**177 hits = 157 `#[ignore]` attributes + 20 doc-comment mentions** of the word.

| date | grep hits | attributes | source |
|---|---:|---:|---|
| 2026-09-12 | 161 | — | 20-CONTEXT §1.5 / this plan's objective |
| 2026-09-13 | 170 | 150 | 20-EXECUTION-NOTES §1; `gates-v1.json` |
| 2026-09-14 09:22 | 177 | 157 | `gates.json`: +4 `pyscf-pbc-df/tests/gdf_omega.rs` (20-05), +3 `pyscf-pbc-dft/tests/gdf_ksymm_bisect.rs` (20-04) |
| **2026-09-14 14:46** | **177** | **157** | `gates-v3.json` — identical key set to `gates.json`; nothing new since |

The tree moved, so the total is not 161. Files checked for gates added since 09:22:
`pyscf-pbc-scf/tests/rsjk.rs` and every `pyscf-pbc-grad/tests/*.rs` (Phase 18) have 0
`#[ignore]`; `pyscf-pbc-df/tests/fft_jk_grad.rs` has 1 attribute
(`e1_matches_upstream_gamma_112_222`, already in `gates.json`) plus 1 doc mention.
The verification one-liner `grep -rc ... | awk` counts the 20 comment lines too.
They are listed in §6 so the reconciliation can be checked.

**Every one of the 157 gates was measured.** No gate went unmeasured.

## §2 — How it was measured

- **Build.** One cargo call per crate, naming every target that holds a gate:
  `CARGO_TARGET_DIR=$PWD/target/gate CARGO_PROFILE_RELEASE_LTO=false cargo test --release -p <crate> --test … --no-run -j 6`.
  Every build exited 0 (`build.status`).
- **Run.** One gate at a time from the built binary:
  `systemd-run --user --scope -p MemoryMax=16G /usr/bin/time -f '%e %M' timeout -k 15 600 <bin> --ignored --exact <name> --test-threads=1 --nocapture`,
  with `PYSCF_ORACLE_VENV=1` and `cwd` set to the crate dir. PASS means the log contains
  `test result: ok. 1 passed`; the process merely exiting was not counted. No run
  printed a `skip:` line, so none of these passes is vacuous.
- **Binary generations** (the `runs` column):

| gen | built | what the binaries include | runs |
|---|---|---|---|
| **P1** | 2026-09-13 21:19–21:59 | HEAD `0f8b58d` + Phase-18 working tree, **before** the D-20-E coulG fix | `tools symm ci adc tdscf gto`, most of `mp`, `dft` `gate`/`gate_openshell` |
| **P2** | 2026-09-14 06:39–06:41 | + D-20-E FFTDF coulG cache fix | re-runs of P1 FAIL/TIMEOUT, then `df scf mp dft cc` |
| **P3** | 2026-09-14 09:22 | + 20-05 GDF/MDF omega + RS builder fixes (sources 09:05) | `df_ao2mo` FAIL re-runs, `gdf_omega`, `gdf_ksymm_bisect` |
| **P4** | 2026-09-14 14:48 | + 20-04 fix: KS drivers grid XC on `cell.mesh`, `gate*.rs` pin `mf.grids` (sources 09:22–09:24, commit `5e6dd65`) | all 16 `dft` `gate.rs`/`gate_openshell.rs` gates, `kuks_ibz_energy_matches_full_bz` |

  Since P3/P4, `pyscf-pbc-df`/`-dft`/`-kernels` `src/` differ from `5e6dd65` only by
  formatting churn from another session (`git diff` read: re-wrapped expressions, reordered
  `pub mod` lines). That churn cannot move an outcome.
- **Gates whose route a later fix touched, but which were not re-run by 20-02.** Each
  PASSED on P2, and 20-05 re-ran it after its own fixes, still passing
  (`20-05-SUMMARY.md` "Blast radius"): `df_swap::krhf_on_mdf_matches_upstream_he_fcc`,
  `gate3_rsdf::{gate3_both_routes_match_upstream_he_fcc, rs_mdf_matches_upstream_he_fcc}`,
  `band_kpoints::{gdf,mdf}_get_jk_at_band_kpoints_matches_upstream`,
  `gdf_builder::helium_fused_j3c_and_j2c_match_upstream`, and `incore::*`. The
  P1 gates that PASSED before the D-20-E fix (`gto`, most of `mp`, `ci`, `tools`) were
  not re-run. D-20-E is confined to FFTDF multi-k exchange, and on the one P1 route known
  to exercise it (`dft gate.rs`) it moved energies by 7.4e-2 – 2.9e-1 Ha (§4). A P1 pass on
  that route would therefore be implausible, but it is not re-verified. The risk is recorded,
  not measured away.

### Concurrency noise — read the seconds as ±3×

Two lanes ran at once in every generation. Other agents' `cargo` builds shared the
16-core host at unrecorded times. The P4 re-run is the best evidence of how
large this noise is: the **same 17 gates, same sources ± the XC-grid pin, ran 1.1–7.7× faster
(median ≈2.9×)** on a quieter host, and **10 of them changed tier** (9 T2 → T1, 1 T3 → T2).
A sample:

| gate | P1/P2 s | P4 s | ratio |
|---|---:|---:|---:|
| `krhf_si_222_is_the_pseudopotential_floor` | 443.62 | 185.47 | 2.4× (T3 → **T2**) |
| `krks_si_222_pbe0_matches_upstream` | 289.96 | 94.20 | 3.1× |
| `kuks_li_atom_113_pbe_matches_upstream` | 116.74 | 15.11 | 7.7× (T2 → **T1**) |
| `kuks_li_atom_gamma_pbe0_matches_upstream` | 66.41 | 29.67 | 2.2× (T2 → **T1**) |
| `krks_si_222_pbe_matches_upstream` | 46.54 | 23.28 | 2.0× (T2 → **T1**) |
| `kuks_si_222_pbe_matches_upstream` | 31.51 | 9.58 | 3.3× (T2 → **T1**) |

The P4 rows (lighter load) are what the tier column uses, because a tier is the latest
measurement. **⚑ marks a row within 2× of a tier edge** (15–60 s or 150–600 s). There
are **66** such rows, and a single CI run can move any of them across the edge. 20-03
should size its job timeouts from the upper end of that band, not from the number
printed here. The seconds include the Python oracle subprocess and the per-process
CubeCL warm-up.

## §3 — Summary

| crate | ignored | T1 | T2 | T3 | PASS | FAIL | TIMEOUT@600 | ⚑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `pyscf-pbc-adc` | 1 | 1 | 0 | 0 | 0 | 1 | 0 | 0 |
| `pyscf-pbc-cc` | 57 | 21 | 36 | 0 | 57 | 0 | 0 | 34 |
| `pyscf-pbc-ci` | 1 | 1 | 0 | 0 | 1 | 0 | 0 | 1 |
| `pyscf-pbc-df` | 24 | 14 | 5 | 5 | 16 | 3 | 5 | 7 |
| `pyscf-pbc-dft` | 24 | 15 | 5 | 4 | 19 | 1 | 4 | 9 |
| `pyscf-pbc-gto` | 21 | 17 | 3 | 1 | 18 | 2 | 1 | 3 |
| `pyscf-pbc-mp` | 13 | 6 | 2 | 5 | 12 | 0 | 1 | 7 |
| `pyscf-pbc-scf` | 13 | 4 | 4 | 5 | 8 | 1 | 4 | 5 |
| `pyscf-pbc-symm` | 1 | 0 | 0 | 1 | 0 | 0 | 1 | 0 |
| `pyscf-pbc-tdscf` | 1 | 1 | 0 | 0 | 0 | 1 | 0 | 0 |
| `pyscf-pbc-tools` | 1 | 1 | 0 | 0 | 1 | 0 | 0 | 0 |
| **total** | **157** | **81** | **55** | **21** | **132** | **9** | **16** | **66** |

Of the 9 FAILs, **3 are not oracle defects**: two child entry points and one
fixture-precondition refusal (§5b). **3 are unwired stub arms** that panic
unconditionally (§5c). **3 are real oracle regressions** (§5a). The 16 TIMEOUTs are
T3 with the outcome not observed. For 2 of them another plan observed the outcome
outside this budget (noted in their rows).

**Reason categories** (`categories.py`): `oracle` = venv-gated upstream comparison;
`slow` = offline or oracle gate ignored on cost; `measurement` = an instrument that
asserts little or nothing; `child` = a subprocess entry point that a non-ignored parent
test spawns (running it standalone is not a gate); `pending` = an ignored arm whose
reason string names missing machinery or a missing fixture; `diagnostic` = the 20-04
bisect.

**CI note for 20-03.** A T1 job that runs `--ignored` over whole binaries would pull in
the 3 stub arms, the 2 children and the fixture refusal. Select by `--exact` name, or
skip those 6. Do not demote them to a lower tier.

## §4 — Re-runs on current binaries: before / after

| gate | before (gen, secs, outcome, number) | after (gen, secs, outcome, number) | route change that forced the re-run |
|---|---|---|---|
| `dft gate::krhf_si_222_is_the_pseudopotential_floor` | P1, 544.09 s, **FAIL**, 2.884e-1 | P2 443.62 s PASS 4.158e-12 → **P4 185.47 s PASS 4.159e-12** (tol 1e-11) | D-20-E coulG (FFTDF multi-k K), then 20-04 grid pin |
| `dft gate::krks_si_222_pbe0_matches_upstream` | P1, 301.41 s, **FAIL**, 7.369e-2 | P2 289.96 s PASS 5.588e-12 → **P4 94.20 s PASS 5.590e-12** (tol 1e-11) | D-20-E, then 20-04 |
| `dft gate::krks_si_222_pbe_matches_upstream` (floor row 3, 6.45e-12) | P1, 46.54 s, PASS, 6.450e-12 | **P4 23.28 s PASS 6.453e-12** | 20-04 grid pin |
| `dft gate::krks_si_222_lda_matches_upstream` | P1 42.47 s PASS 6.509e-12 | **P4 13.35 s PASS 6.490e-12** | 20-04 |
| `dft gate::krks_he_all_electron_222_pbe_matches_upstream` (row 3a) | P1 11.17 s PASS 8.482e-14 | **P4 7.28 s PASS 8.482e-14** (tol 1e-12) | 20-04 |
| `dft gate::kuks_si_222_pbe_matches_upstream` | P1 31.51 s PASS 6.449e-12 | **P4 9.58 s PASS 6.448e-12** | 20-04 |
| `dft gate::krks_si_222_pbe_against_xcfun` (measurement) | P1 20.21 s PASS 4.709e-7 | **P4 8.93 s PASS 4.709e-7** | 20-04 |
| `dft gate_openshell::kuks_li_atom_gamma_pbe_matches_upstream` | P1 60.38 s PASS 7.726e-12 | **P4 18.53 s PASS 7.730e-12** (tol 5e-11) | 20-04 |
| `… kuks_li_atom_gamma_lda_matches_upstream` | P1 67.18 s PASS 7.796e-12 | **P4 14.09 s PASS 7.799e-12** | 20-04 |
| `… kuks_li_atom_113_pbe_matches_upstream` | P1 116.74 s PASS 5.473e-12 | **P4 15.11 s PASS 5.451e-12** | 20-04 |
| `… kuhf_li_atom_gamma_is_the_open_shell_floor` | P1 34.12 s PASS 1.493e-11 | **P4 16.73 s PASS 1.493e-11** (tol 5e-11) | 20-04 |
| `… kuks_h2_stretched_gamma_pbe_matches_upstream` | P1 34.36 s PASS 1.750e-13 | **P4 12.02 s PASS 1.752e-13** (tol 1e-12) | 20-04 |
| `… kuks_h2_stretched_gamma_lda_matches_upstream` | P1 29.06 s PASS 2.582e-13 | **P4 11.38 s PASS 2.578e-13** | 20-04 |
| `… kuhf_h2_stretched_gamma_matches_upstream` | P1 24.38 s PASS 7.794e-14 | **P4 15.40 s PASS 7.772e-14** | 20-04 |
| `… kuks_li_atom_gamma_pbe0_matches_upstream` | P1 66.41 s PASS 9.723e-12 | **P4 29.67 s PASS 9.720e-12** | 20-04 |
| `… kuks_h2_stretched_gamma_pbe0_matches_upstream` | P1 28.81 s PASS 1.832e-13 | **P4 9.12 s PASS 1.843e-13** (tol 1e-11) | 20-04 |
| `dft krks_ksymm::kuks_ibz_energy_matches_full_bz` | P1 109.59 s FAIL, P2 63.65 s FAIL — precondition | **P4 60.51 s FAIL**, same precondition, max\|dm_a−dm_b\| 1.194 | 20-04 (`kuks.rs` grid) |
| `df df_ao2mo::get_eri_matches_upstream_on_he_fcc` | P2 1.75 s FAIL 2.329752e-10 | **P3 1.50 s FAIL 2.329753e-10** | 20-05 (`int3c.rs`, `j2c.rs`) |
| `df df_ao2mo::ao2mo_7d_matches_upstream_on_he_fcc` | P2 2.07 s FAIL 8.214629e-11 | **P3 1.57 s FAIL 8.214648e-11** | 20-05 |
| `df df_ao2mo::get_eri_matches_upstream_on_diamond_gamma` | P2 82.96 s FAIL 2.3735e-7 | **P3 146.79 s FAIL 2.3500e-7** | 20-05 |

The 20-04 XC-grid pin changed **no gate number beyond the last printed digit**. Every
`gate.rs` fixture already pinned the XC mesh through the FFTDF mesh, and FFTDF's DF mesh
is `cell.mesh`. Some last-digit shifts come from the **upstream** side (e.g. LDA upstream
−7.772926981755228 → …210 with the Rust value bit-identical). That is upstream run-to-run
noise, consistent with `measurements/README.md` §3. The KRKS Si PBE floor
(`README.md` §1 row 3, 6.45e-12) is **re-confirmed at 6.453e-12**.

## §5 — Failures for follow-up

### §5a — real oracle regressions (route owner: `pyscf-pbc-df` GDF `df_ao2mo`)

| test | measured (P3, current) | expected floor / gate | pre-20-05 (P2) | notes |
|---|---|---|---|---|
| `df_ao2mo::get_eri_matches_upstream_on_he_fcc` | **2.3298e-10** | gate `< 1e-11` (screens equalised at 1e-14); Phase 14 measured **1.667e-12** (`14-VERIFICATION.md:86`) | 2.3298e-10 | **≈140× worse than Phase 14. Not caused by 20-05**: pre and post agree to 6 digits. The regression predates 2026-09-14 06:41. `get_eri_is_bit_exact_with_upstream_over_the_same_cderi` (same He-fcc fixture) PASSES at 3.85e-37, so the contraction is exact. The gap is in the port's `cderi` build versus upstream's. Candidates: the §11 GDF reopen (2026-09-05, `14-VERIFICATION.md:578`) or later `gdf`/`incore` edits. Bisect owed. |
| `df_ao2mo::ao2mo_7d_matches_upstream_on_he_fcc` | **8.2146e-11** | gate `< 1e-11`; Phase 14 measured **1.984e-12** (`14-VERIFICATION.md:87`) | 8.2146e-11 | Same fixture and `Gdf::build`, so most likely the same cause as the row above. |
| `df_ao2mo::get_eri_matches_upstream_on_diamond_gamma` | **2.3500e-7** | gate `< 1e-11`; **no prior number**. Phase 14 left it "owed" (`14-VERIFICATION.md:134`), and this is its first completed run | 2.3735e-7 | 20-05 moved it by 2.4e-9. Upstream is run with `exclude_dd_block=False`, and 14 measured that switch at 1.835e-8 on diamond. So 2.35e-7 is more than 10× the dd-block effect. Needs its own bisect. The bit-exact attribution device exists only on He-fcc, so on diamond `cderi` vs contraction is not yet separated. |

None of these three gates was loosened. All three are T1/T2, so 20-03 cannot put
`df_ao2mo` into a green job until they are fixed or explicitly quarantined by name.

### §5b — not oracle defects (fail by construction when run standalone)

| test | measured | why |
|---|---|---|
| `pyscf-pbc-gto eval_ao_point_screen::emit_ao_bits` | panics `child output path` in 0.01 s | Child entry point (reason: "child process for the A-04 gate"). It reads `PYSCF_AO_STAGE_OUTPUT`, which only its non-ignored parent (`run_child`) sets. The parent was not part of this measurement. |
| `pyscf-pbc-gto eval_ao_stages::emit_ao_bits` | same, 0.02 s | Child of `image_loop_is_thread_bit_exact_and_screen_stays_inside_its_gate` (non-ignored parent), with the same env contract. |
| `pyscf-pbc-dft krks_ksymm::kuks_ibz_energy_matches_full_bz` | `PRECONDITION FAILED`; full-BZ beta solution symmetry-broken (max\|dm_a−dm_b\| = 1.194), 60.5 s | The reason string says so: the test "needs an open-shell fixture whose FULL-BZ solution is star-symmetric". It is a fixture gap, not a KUKS number, and it is unchanged by the 20-04 fix. |

### §5c — unwired arms (two panic unconditionally; ADC computes a non-renormalised number; the owner is named in the reason)

| test | measured | expected | owner |
|---|---|---|---|
| `pyscf-pbc-adc ea::gate_d_ip_spec_factors` | IP spec factors deviate **9.358e-1** | Gate D `< 5e-4` | 19-15/19-16 follow-up: needs `get_trans_moments` + ADC-norm renormalisation |
| `pyscf-pbc-tdscf uhf::live_bigbox_matches_molecular` | `panic!("live arm: KUHF (2,1,1) Davidson vs molecular UHF-TDA at 2dp")` | 2 dp vs molecular UHF-TDA | 19-08 follow-up: matrix-free `vind` for non-uniform per-k fillings |
| `pyscf-pbc-scf newton_ah::live_newton_matches_upstream_energy` | `panic!("live arm not yet wired to the KRHF Fock build")` | upstream Newton energy | 19-04 Task 3 human-verify |

### §5d — T3 gates whose outcome was observed elsewhere (not in this budget)

| test | where observed | outcome |
|---|---|---|
| `krks_ksymm::krks_ibz_energy_matches_full_bz_on_gdf` | `20-04-FIX-SUMMARY.md`, post-fix binary | PASS, \|dE\| **2.1997e-10** (bound 1e-8) |
| `gdf_ksymm_bisect::gdf_gate_c_with_and_without_matched_xc_grid` | `20-04-SUMMARY.md`, 2544 s | exit 0; 1.4324e-6 as written / 2.720e-10 matched |

The other 14 TIMEOUT rows have **no observed outcome** in Phase 20.

## §6 — The 20 comment lines that `grep '#\[ignore'` also counts

`pyscf-pbc-cc/tests/oracle_eom_partition.rs:311`, `oracle_phase16.rs:3`;
`pyscf-pbc-df/tests/df_ao2mo.rs:978`, `fft_jk_grad.rs:11`, `fftdf.rs:9`, `perf_dpbc28_mofirst.rs:5`;
`pyscf-pbc-dft/tests/krks_ksymm.rs:451`, `:618`, `:737`;
`pyscf-pbc-gto/tests/cintx_moment_weighted_available.rs:44`, `eval_ao_screen.rs:169`, `oracle_phase9.rs:3`, `pbc_intor.rs:7`, `:524`;
`pyscf-pbc-scf/tests/df_swap.rs:576`, `exclude_dd_block_energy.rs:15`, `:93`, `kscf.rs:7`, `newton_ah.rs:11`;
`pyscf-pbc-tdscf/tests/uhf.rs:16`. 157 + 20 = 177.

## §7 — Per-gate table (157 rows)

`secs` is `/usr/bin/time` elapsed for the single-test process. `>600` means killed at
600 s. The tier is the one written into the `#[ignore = "Tn: …"]` string (Task 5).

| # | crate | file:line | test | category | secs | peak RSS (MB) | outcome | measured deviation | tier | runs |
|---:|---|---|---|---|---:|---:|---|---|---|---|
| 1 | `pbc-adc` | `ea.rs:133` | `gate_d_ip_spec_factors` | pending | 0.04 | 12 | FAIL | IP spec factors deviate 9.358e-1 vs 5e-4 (stub arm) | **T1** | P1 |
| 2 | `pbc-cc` | `kccsd_rhf.rs:82` | `eris_incore_and_spilled_are_bit_identical` | slow | 26.87 | 237 | PASS | bit-identical, 7 blocks | **T1**⚑ | P2 |
| 3 | `pbc-cc` | `kccsd_rhf.rs:145` | `symm_map_loop_matches_the_all_triples_loop` | slow | 23.34 | 212 | PASS | max 7.93e-7 (vovv) | **T1**⚑ | P2 |
| 4 | `pbc-cc` | `kccsd_rhf.rs:189` | `amplitudes_and_energy_are_bit_reproducible` | slow | 12.86 | 206 | PASS | bit-identical over two runs | **T1** | P2 |
| 5 | `pbc-cc` | `kccsd_rhf.rs:218` | `init_amps_emp2_equals_kmp2` | slow | 14.22 | 259 | PASS | 2.166e-10 | **T1** | P2 |
| 6 | `pbc-cc` | `kccsd_rhf.rs:258` | `eris_charges_exactly_what_it_allocates` | slow | 15.46 | 208 | PASS | exact byte count | **T1**⚑ | P2 |
| 7 | `pbc-cc` | `kccsd_rhf.rs:299` | `krccsd_matches_the_supercell_at_gamma` | slow | 41.98 | 454 | PASS | e_corr 1.426e-11 (G9 1e-7) | **T2**⚑ | P2 |
| 8 | `pbc-cc` | `kgccsd.rs:29` | `kgccsd_equals_krccsd_on_a_closed_shell` | slow | 108.53 | 250 | PASS | 4.562e-10 | **T2** | P2 |
| 9 | `pbc-cc` | `kgccsd.rs:93` | `ccsd_t_peak_memory_is_bounded_by_one_block` | slow | 16.08 | 212 | PASS | peak t3-cache == derived bound | **T1**⚑ | P2 |
| 10 | `pbc-cc` | `kgccsd.rs:170` | `amplitudes_are_bit_identical_across_thread_counts` | slow | 14.19 | 210 | PASS | bit-identical 1 vs 8 threads | **T1** | P2 |
| 11 | `pbc-cc` | `krccsd_smoke.rs:17` | `krccsd_runs_on_diamond_112` | slow | 14.66 | 207 | PASS | converged, 18 cycles | **T1** | P2 |
| 12 | `pbc-cc` | `kuccsd.rs:42` | `update_amps_at_zero_amplitudes_reproduces_init_amps` | slow | 21.55 | 203 | PASS | 6.94e-18 | **T1**⚑ | P2 |
| 13 | `pbc-cc` | `oracle_eom_ee_singlet.rs:42` | `ee_singlet_equations_match_upstream` | oracle | 44.97 | 208 | PASS | worst 2.37e-7 | **T2**⚑ | P2 |
| 14 | `pbc-cc` | `oracle_eom_ee_singlet.rs:224` | `ee_singlet_roots_match_upstream` | oracle | 42.07 | 210 | PASS | worst 3.70e-7 | **T2**⚑ | P2 |
| 15 | `pbc-cc` | `oracle_eom_partition.rs:61` | `upstream_refuses_every_partition_the_drivers_are_given` | oracle | 59.45 | 211 | PASS | refusals confirmed | **T2**⚑ | P2 |
| 16 | `pbc-cc` | `oracle_eom_partition.rs:121` | `rhf_partition_mp_matches_upstream` | oracle | 39.94 | 212 | PASS | worst 3.89e-7 | **T2**⚑ | P2 |
| 17 | `pbc-cc` | `oracle_eom_partition.rs:230` | `ghf_partition_mp_diagonals_match_upstream` | oracle | 142.28 | 235 | PASS | worst 5.59e-9 | **T2** | P2 |
| 18 | `pbc-cc` | `oracle_eom_star.rs:113` | `rhf_ccsd_star_matches_upstream` | oracle | 24.39 | 209 | PASS | 1.522e-10 | **T1**⚑ | P2 |
| 19 | `pbc-cc` | `oracle_eom_star.rs:192` | `ghf_ccsd_star_matches_upstream` | oracle | 153.40 | 236 | PASS | 1.521e-10 | **T2**⚑ | P2 |
| 20 | `pbc-cc` | `oracle_eom_ta.rs:75` | `ghf_t3p2_and_ta_roots_match_upstream` | oracle | 102.00 | 488 | PASS | worst printed 6.37e-10 | **T2** | P2 |
| 21 | `pbc-cc` | `oracle_eom_ta.rs:218` | `rhf_t3p2_and_ta_roots_match_upstream` | oracle | 12.38 | 211 | PASS | worst printed 1.37e-9 | **T1** | P2 |
| 22 | `pbc-cc` | `oracle_gamma_rccsd.rs:99` | `rccsd_equations_match_upstream` | oracle | 3.77 | 179 | PASS | init_t2 8.90e-10 | **T1** | P2 |
| 23 | `pbc-cc` | `oracle_gamma_rccsd.rs:189` | `rccsd_e_corr_matches_upstream` | oracle | 2.47 | 178 | PASS | gamma e_corr 2.231e-8 | **T1** | P2 |
| 24 | `pbc-cc` | `oracle_gamma_rccsd.rs:269` | `gamma_shim_and_krccsd_at_one_kpoint_agree` | oracle | 8.02 | 176 | PASS | port vs upstream 2.127e-8 | **T1** | P2 |
| 25 | `pbc-cc` | `oracle_gamma_rccsd.rs:318` | `gamma_shim_runs_on_this_ports_own_mean_field` | oracle | 13.28 | 190 | PASS | e_corr 6.128e-7 (reported, own mean field) | **T1** | P2 |
| 26 | `pbc-cc` | `oracle_gamma_ug.rs:133` | `gccsd_equations_match_upstream` | oracle | 44.71 | 204 | PASS | init_t2 8.90e-10 | **T2**⚑ | P2 |
| 27 | `pbc-cc` | `oracle_gamma_ug.rs:250` | `gccsd_e_corr_matches_upstream` | oracle | 44.58 | 203 | PASS | worst 3.52e-9 | **T2**⚑ | P2 |
| 28 | `pbc-cc` | `oracle_gamma_ug.rs:394` | `uccsd_equations_match_upstream` | oracle | 28.35 | 203 | PASS | init_t2 8.90e-10 | **T1**⚑ | P2 |
| 29 | `pbc-cc` | `oracle_gamma_ug.rs:482` | `uccsd_e_corr_matches_upstream` | oracle | 16.43 | 203 | PASS | worst 3.52e-9 | **T1**⚑ | P2 |
| 30 | `pbc-cc` | `oracle_gamma_ug.rs:533` | `gamma_shims_and_the_kpoint_routes_agree` | oracle | 20.95 | 203 | PASS | port U vs G 9.42e-11 | **T1**⚑ | P2 |
| 31 | `pbc-cc` | `oracle_kuccsd.rs:380` | `kueris_blocks_match_upstream` | oracle | 26.65 | 391 | PASS | worst ooOO 8.073e-10 (gate 1e-9) | **T1**⚑ | P2 |
| 32 | `pbc-cc` | `oracle_kuccsd.rs:409` | `the_eri_residual_is_the_mesh_and_not_the_port` | oracle | 62.85 | 384 | PASS | [31]^3 8.073e-10 | **T2** | P2 |
| 33 | `pbc-cc` | `oracle_kuccsd.rs:481` | `kuccsd_init_amps_matches_upstream` | oracle | 30.32 | 374 | PASS | emp2 6.439e-12 | **T2**⚑ | P2 |
| 34 | `pbc-cc` | `oracle_kuccsd.rs:499` | `kuccsd_update_amps_matches_upstream` | oracle | 39.88 | 385 | PASS | worst 1.63e-10 | **T2**⚑ | P2 |
| 35 | `pbc-cc` | `oracle_kuccsd.rs:570` | `kuccsd_e_corr_matches_upstream` | oracle | 43.71 | 391 | PASS | 4.691e-9 (gate 1e-7) | **T2**⚑ | P2 |
| 36 | `pbc-cc` | `oracle_kuccsd.rs:652` | `kuccsd_intermediates_match_upstream` | oracle | 69.11 | 389 | PASS | worst printed 4.73e-11 | **T2** | P2 |
| 37 | `pbc-cc` | `oracle_kuccsd.rs:765` | `kuccsd_wovvo_block_matches_upstream` | oracle | 85.10 | 388 | PASS | 8.85e-11 | **T2** | P2 |
| 38 | `pbc-cc` | `oracle_kuccsd.rs:815` | `kuccsd_woooo_block_matches_upstream` | oracle | 75.61 | 401 | PASS | 2.24e-10 | **T2** | P2 |
| 39 | `pbc-cc` | `oracle_kuccsd.rs:865` | `kuccsd_fock_block_matches_upstream` | oracle | 90.52 | 390 | PASS | 1.10e-10 | **T2** | P2 |
| 40 | `pbc-cc` | `oracle_kuccsd.rs:924` | `kuccsd_rdm1_matches_upstream` | oracle | 118.71 | 399 | PASS | 1.895e-8 (gate 1e-7) | **T2** | P2 |
| 41 | `pbc-cc` | `oracle_kuccsd.rs:1031` | `kuccsd_eom_intermediates_match_upstream` | oracle | 85.06 | 402 | PASS | worst printed 5.87e-10 | **T2** | P2 |
| 42 | `pbc-cc` | `oracle_kuccsd.rs:1175` | `kuccsd_eom_ip_matches_upstream` | oracle | 76.64 | 406 | PASS | 1.246e-9 (gate 1e-8) | **T2** | P2 |
| 43 | `pbc-cc` | `oracle_kuccsd.rs:1242` | `kuccsd_eom_ea_matches_upstream` | oracle | 86.46 | 395 | PASS | 1.512e-9 (gate 1e-8) | **T2** | P2 |
| 44 | `pbc-cc` | `oracle_kuccsd.rs:1314` | `kuccsd_eom_roots_match_upstream` | oracle | 97.99 | 393 | PASS | worst printed 4.11e-10 | **T2** | P2 |
| 45 | `pbc-cc` | `oracle_phase16.rs:60` | `eris_blocks_match_upstream` | oracle | 26.11 | 216 | PASS | worst printed 2.34e-7 | **T1**⚑ | P2 |
| 46 | `pbc-cc` | `oracle_phase16.rs:116` | `intermediates_and_update_amps_match_upstream` | oracle | 28.56 | 213 | PASS | t2new 7.01e-8 | **T1**⚑ | P2 |
| 47 | `pbc-cc` | `oracle_phase16.rs:216` | `krccsd_e_corr_matches_upstream_fftdf` | oracle | 44.31 | 214 | PASS | G1 6.560e-9 (1e-7) | **T2**⚑ | P2 |
| 48 | `pbc-cc` | `oracle_phase16.rs:267` | `ccsd_t_fast_equals_slow_and_matches_upstream` | oracle | 64.67 | 216 | PASS | (T) 3.287e-10 | **T2** | P2 |
| 49 | `pbc-cc` | `oracle_phase16.rs:377` | `kgccsd_matches_upstream` | oracle | 202.46 | 268 | PASS | e_corr 2.066e-9 | **T2**⚑ | P2 |
| 50 | `pbc-cc` | `oracle_phase16.rs:546` | `krccsd_e_corr_matches_upstream_gdf` | oracle | 180.44 | 222 | PASS | G2 GDF 1.504e-8 (1e-7) | **T2**⚑ | P2 |
| 51 | `pbc-cc` | `oracle_phase16.rs:640` | `kgccsd_eom_intermediates_match_upstream` | oracle | 181.80 | 268 | PASS | 4.68e-7 | **T2**⚑ | P2 |
| 52 | `pbc-cc` | `oracle_phase16.rs:775` | `kgccsd_eom_ip_and_ea_match_upstream` | oracle | 262.48 | 247 | PASS | worst printed 3.60e-7 | **T2**⚑ | P2 |
| 53 | `pbc-cc` | `oracle_phase16.rs:944` | `kgccsd_eom_roots_match_upstream` | oracle | 291.95 | 248 | PASS | worst printed 4.78e-9 | **T2**⚑ | P2 |
| 54 | `pbc-cc` | `oracle_phase16.rs:1081` | `kgccsd_eom_ee_matches_upstream` | oracle | 239.07 | 259 | PASS | worst printed 3.88e-7 | **T2**⚑ | P2 |
| 55 | `pbc-cc` | `oracle_phase16.rs:1223` | `krccsd_eom_intermediates_match_upstream` | oracle | 30.20 | 232 | PASS | 2.39e-7 | **T2**⚑ | P2 |
| 56 | `pbc-cc` | `oracle_phase16.rs:1325` | `krccsd_eom_ip_and_ea_match_upstream` | oracle | 41.75 | 230 | PASS | worst printed 5.43e-7 | **T2**⚑ | P2 |
| 57 | `pbc-cc` | `oracle_phase16.rs:1420` | `krccsd_eom_roots_match_upstream` | oracle | 41.57 | 232 | PASS | worst printed 1.40e-9 | **T2**⚑ | P2 |
| 58 | `pbc-cc` | `zgemm_vs_host.rs:135` | `zgemm_dense_versus_the_host_loop_on_this_phases_shapes` | measurement | 43.70 | 290 | PASS | timing; agreement 2.09e-15 rel | **T2**⚑ | P2 |
| 59 | `pbc-ci` | `oracle_kcis.rs:137` | `kcis_roots_match_upstream` | oracle | 15.19 | 195 | PASS | davidson vs upstream roots ~2.4e-9 | **T1**⚑ | P1 |
| 60 | `pbc-df` | `band_kpoints.rs:173` | `gdf_get_jk_at_band_kpoints_matches_upstream` | oracle | 7.87 | 178 | PASS | \|dvj\| 1.394e-9, \|dvk\| 1.98e-11 | **T1** | P2 |
| 61 | `pbc-df` | `band_kpoints.rs:230` | `mdf_get_jk_at_band_kpoints_matches_upstream` | oracle | 8.69 | 163 | PASS | asserted, no number printed | **T1** | P2 |
| 62 | `pbc-df` | `df_ao2mo.rs:871` | `get_eri_matches_upstream_on_he_fcc` | oracle | 1.50 | 124 | FAIL | 2.3298e-10 vs gate 1e-11 (14-VERIFICATION: 1.667e-12) | **T1** | P3 (P2 FAIL → P3 FAIL) |
| 63 | `pbc-df` | `df_ao2mo.rs:942` | `ao2mo_7d_matches_upstream_on_he_fcc` | oracle | 1.57 | 129 | FAIL | 8.2146e-11 vs gate 1e-11 (14-VERIFICATION: 1.984e-12) | **T1** | P3 (P2 FAIL → P3 FAIL) |
| 64 | `pbc-df` | `df_ao2mo.rs:983` | `get_eri_matches_upstream_on_diamond_gamma` | oracle | 146.79 | 192 | FAIL | 2.3500e-7 vs gate 1e-11 (first completed run; number was owed by Phase 14) | **T2** | P3 (P2 FAIL → P3 FAIL) |
| 65 | `pbc-df` | `df_ao2mo.rs:1028` | `get_eri_is_bit_exact_with_upstream_over_the_same_cderi` | oracle | 1.06 | 128 | PASS | 3.85e-37 (port cderi both sides) | **T1** | P2 |
| 66 | `pbc-df` | `exclude_dd_block.rs:113` | `diamond_gdf_both_routes_produce_a_cderi` | slow | >600 | 242 | TIMEOUT@600 s — outcome not observed | — | **T3** | P2 |
| 67 | `pbc-df` | `fft_jk_grad.rs:415` | `e1_matches_upstream_gamma_112_222` | oracle | 15.76 | 201 | PASS | worst 2.78e-13 | **T1**⚑ | P2 |
| 68 | `pbc-df` | `fftdf.rs:375` | `get_pp_matches_upstream_on_diamond_222` | oracle | 6.74 | 471 | PASS | 1.897e-13 | **T1** | P2 |
| 69 | `pbc-df` | `fftdf.rs:392` | `get_hcore_matches_upstream_on_diamond_222` | oracle | 6.99 | 473 | PASS | 1.895e-13 | **T1** | P2 |
| 70 | `pbc-df` | `fftdf.rs:411` | `get_nuc_matches_upstream_on_he` | oracle | 1.70 | 126 | PASS | 2.038e-13 | **T1** | P2 |
| 71 | `pbc-df` | `fftdf.rs:429` | `jk_matches_upstream_on_diamond_222` | oracle | 15.61 | 149 | PASS | vk 5.44e-13 | **T1**⚑ | P2 |
| 72 | `pbc-df` | `gdf.rs:433` | `get_pp_works_at_k_points` | slow | >600 | 371 | TIMEOUT@600 s — outcome not observed | — | **T3** | P2 |
| 73 | `pbc-df` | `gdf_builder.rs:477` | `cderi_fingerprint_matches_upstream_diamond` | slow | >600 | 248 | TIMEOUT@600 s — outcome not observed | — | **T3** | P2 |
| 74 | `pbc-df` | `gdf_builder.rs:660` | `helium_fused_j3c_and_j2c_match_upstream` | oracle | 18.55 | 163 | PASS | j3c 1.451e-12, j2c 7.11e-14 | **T1**⚑ | P2 |
| 75 | `pbc-df` | `gdf_mo_k.rs:119` | `mo_route_wall_clock_vs_dm_route_diamond` | measurement | >600 | 346 | TIMEOUT@600 s — outcome not observed | — | **T3** | P2 |
| 76 | `pbc-df` | `gdf_omega.rs:201` | `gdf_lr_matches_upstream` | oracle | 39.28 | 134 | PASS | \|dvj\| 8.034e-17, \|dvk\| 2.223e-11 (ewald) vs 1.353e-8 | **T2**⚑ | P3 |
| 77 | `pbc-df` | `gdf_omega.rs:213` | `gdf_sr_matches_upstream` | oracle | 27.44 | 151 | PASS | \|dvj\| 1.939e-9, \|dvk\| 2.823e-10 vs 1.353e-8 | **T1**⚑ | P3 |
| 78 | `pbc-df` | `gdf_omega.rs:224` | `mdf_lr_matches_upstream` | oracle | 37.28 | 135 | PASS | \|dvj\| 8.034e-17, \|dvk\| 2.223e-11 (ewald) vs 1.353e-8 | **T2**⚑ | P3 |
| 79 | `pbc-df` | `gdf_omega.rs:239` | `mdf_sr_matches_upstream` | oracle | 69.36 | 147 | PASS | \|dvj\| 1.830e-9, \|dvk\| 5.130e-10 vs 1.353e-8 | **T2** | P3 |
| 80 | `pbc-df` | `incore.rs:414` | `isolated_cell_aux_e2_matches_upstream` | oracle | 2.08 | 156 | PASS | 8.88e-16 | **T1** | P2 |
| 81 | `pbc-df` | `incore.rs:498` | `isolated_cell_fill_2c2e_is_symmetric_and_positive_definite` | oracle | 1.70 | 150 | PASS | asserted, no number printed | **T1** | P2 |
| 82 | `pbc-df` | `perf_dpbc28_mofirst.rs:180` | `mo_first_vs_ao_first_cost` | measurement | 222.84 | 1014 | PASS | ratio 6.16, residual 1.7e-17 | **T2**⚑ | P2 |
| 83 | `pbc-df` | `perf_dpbc28_mofirst.rs:188` | `mo_first_vs_ao_first_cost_222` | measurement | >600 | 696 | TIMEOUT@600 s — outcome not observed | — | **T3** | P2 |
| 84 | `pbc-dft` | `gate.rs:188` | `krks_si_222_pbe_matches_upstream` | oracle | 23.28 | 812 | PASS | 6.453e-12 (tol 1e-11); pre-20-04-fix 6.450e-12 | **T1**⚑ | P4 (P1 PASS → P4 PASS) |
| 85 | `pbc-dft` | `gate.rs:216` | `krks_si_222_lda_matches_upstream` | oracle | 13.35 | 569 | PASS | 6.490e-12 (tol 1e-11); pre 6.509e-12 | **T1** | P4 (P1 PASS → P4 PASS) |
| 86 | `pbc-dft` | `gate.rs:247` | `krks_he_all_electron_222_pbe_matches_upstream` | oracle | 7.28 | 362 | PASS | 8.482e-14 (tol 1e-12); pre 8.482e-14 | **T1** | P4 (P1 PASS → P4 PASS) |
| 87 | `pbc-dft` | `gate.rs:276` | `krhf_si_222_is_the_pseudopotential_floor` | oracle | 185.47 | 633 | PASS | 4.159e-12 (tol 1e-11); pre-20-04-fix 4.158e-12; pre-D-20-E binary 2.884e-1 FAIL | **T2**⚑ | P4 (P1 FAIL → P2 PASS → P4 PASS) |
| 88 | `pbc-dft` | `gate.rs:305` | `kuks_si_222_pbe_matches_upstream` | oracle | 9.58 | 804 | PASS | 6.448e-12 (tol 1e-11); pre 6.449e-12 | **T1** | P4 (P1 PASS → P4 PASS) |
| 89 | `pbc-dft` | `gate.rs:336` | `krks_si_222_pbe0_matches_upstream` | oracle | 94.20 | 935 | PASS | 5.590e-12 (tol 1e-11); pre-20-04-fix 5.588e-12; pre-D-20-E binary 7.369e-2 FAIL | **T2** | P4 (P1 FAIL → P2 PASS → P4 PASS) |
| 90 | `pbc-dft` | `gate.rs:368` | `krks_si_222_pbe_against_xcfun` | oracle | 8.93 | 794 | PASS | measurement vs xcfun upstream 4.709e-7 (unchanged) | **T1** | P4 (P1 PASS → P4 PASS) |
| 91 | `pbc-dft` | `gate_openshell.rs:362` | `kuks_li_atom_gamma_pbe_matches_upstream` | oracle | 18.53 | 439 | PASS | 7.730e-12 (tol 5e-11); pre 7.726e-12 | **T1**⚑ | P4 (P1 PASS → P4 PASS) |
| 92 | `pbc-dft` | `gate_openshell.rs:382` | `kuks_li_atom_gamma_lda_matches_upstream` | oracle | 14.09 | 415 | PASS | 7.799e-12 (tol 5e-11); pre 7.796e-12 | **T1** | P4 (P1 PASS → P4 PASS) |
| 93 | `pbc-dft` | `gate_openshell.rs:403` | `kuks_li_atom_113_pbe_matches_upstream` | oracle | 15.11 | 515 | PASS | 5.451e-12 (tol 5e-11); pre 5.473e-12 | **T1**⚑ | P4 (P1 PASS → P4 PASS) |
| 94 | `pbc-dft` | `gate_openshell.rs:423` | `kuhf_li_atom_gamma_is_the_open_shell_floor` | oracle | 16.73 | 442 | PASS | 1.493e-11 (tol 5e-11); pre 1.493e-11 | **T1**⚑ | P4 (P1 PASS → P4 PASS) |
| 95 | `pbc-dft` | `gate_openshell.rs:452` | `kuks_h2_stretched_gamma_pbe_matches_upstream` | oracle | 12.02 | 481 | PASS | 1.752e-13 (tol 1e-12); pre 1.750e-13 | **T1** | P4 (P1 PASS → P4 PASS) |
| 96 | `pbc-dft` | `gate_openshell.rs:471` | `kuks_h2_stretched_gamma_lda_matches_upstream` | oracle | 11.38 | 307 | PASS | 2.578e-13 (tol 1e-12); pre 2.582e-13 | **T1** | P4 (P1 PASS → P4 PASS) |
| 97 | `pbc-dft` | `gate_openshell.rs:492` | `kuhf_h2_stretched_gamma_matches_upstream` | oracle | 15.40 | 262 | PASS | 7.772e-14 (tol 1e-12); pre 7.794e-14 | **T1**⚑ | P4 (P1 PASS → P4 PASS) |
| 98 | `pbc-dft` | `gate_openshell.rs:517` | `kuks_li_atom_gamma_pbe0_matches_upstream` | oracle | 29.67 | 468 | PASS | 9.720e-12 (tol 5e-11); pre 9.723e-12 | **T1**⚑ | P4 (P1 PASS → P4 PASS) |
| 99 | `pbc-dft` | `gate_openshell.rs:537` | `kuks_h2_stretched_gamma_pbe0_matches_upstream` | oracle | 9.12 | 501 | PASS | 1.843e-13 (tol 1e-11); pre 1.832e-13 | **T1** | P4 (P1 PASS → P4 PASS) |
| 100 | `pbc-dft` | `gdf_ksymm_bisect.rs:386` | `bisect_fixture_and_arm_grids` | diagnostic | 0.00 | 48 | PASS | nkpts 8 / ibz 3; every arm XC grid [35,35,35] | **T1** | P3 |
| 101 | `pbc-dft` | `gdf_ksymm_bisect.rs:416` | `bisect_gdf_and_fftdf_routes_at_one_density` | diagnostic | >600 | 812 | TIMEOUT@600 s — outcome not observed | density symmetry-breaking 5.43e-10 printed before kill (20-04: all steps, 2544 s class) | **T3** | P3 |
| 102 | `pbc-dft` | `gdf_ksymm_bisect.rs:499` | `gdf_gate_c_with_and_without_matched_xc_grid` | diagnostic | >600 | 785 | TIMEOUT@600 s — outcome not observed | — (20-04 ran it to completion: 2544 s, 1.432e-6 as written / 2.720e-10 matched) | **T3** | P3 |
| 103 | `pbc-dft` | `krks_ksymm.rs:498` | `kuks_ibz_energy_matches_full_bz` | pending | 60.51 | 4605 | FAIL | PRECONDITION FAILED (fixture): full-BZ beta symmetry-broken, max\|dm_a-dm_b\| 1.194; unchanged post-20-04-fix | **T2** | P4 (P1 FAIL → P2 FAIL → P4 FAIL) |
| 104 | `pbc-dft` | `krks_ksymm.rs:781` | `krks_ibz_energy_matches_full_bz_on_gdf` | slow | >600 | 397 | TIMEOUT@600 s — outcome not observed | — in budget; 20-04-FIX ran it to completion post-fix: \|dE\| 2.1997e-10 (bound 1e-8) | **T3** | P2 |
| 105 | `pbc-dft` | `krks_ksymm.rs:879` | `gdf_band_route_matches_the_direct_route` | measurement | >600 | 360 | TIMEOUT@600 s — outcome not observed | — | **T3** | P2 |
| 106 | `pbc-dft` | `mg_kpts_bench.rs:96` | `mg_kpts_bench` | measurement | 257.54 | 1675 | PASS | instrument | **T2**⚑ | P2 |
| 107 | `pbc-dft` | `mg_pair_bench.rs:30` | `mg_pair_bench` | measurement | 46.18 | 1665 | PASS | instrument (warm forward/reverse ms) | **T2**⚑ | P2 |
| 108 | `pbc-gto` | `eval_ao_image_batch.rs:224` | `compare_unscreened_child` | child | 38.51 | 217 | PASS | bit-identical across batch sizes (child) | **T2**⚑ | P1 |
| 109 | `pbc-gto` | `eval_ao_point_screen.rs:158` | `emit_ao_bits` | child | 0.01 | 37 | FAIL | panics "child output path": needs the parent's env var | **T1** | P1 |
| 110 | `pbc-gto` | `eval_ao_screen.rs:172` | `print_reference_unscreened` | child | 0.86 | 140 | PASS | prints reference sums (child) | **T1** | P1 |
| 111 | `pbc-gto` | `eval_ao_stages.rs:142` | `emit_ao_bits` | child | 0.02 | 37 | FAIL | panics "child output path": needs the parent's env var | **T1** | P1 |
| 112 | `pbc-gto` | `gth_pp_loc.rs:258` | `g_space_factors_match_upstream_on_diamond` | oracle | 2.30 | 109 | PASS | 3.55e-15 | **T1** | P1 |
| 113 | `pbc-gto` | `gth_pp_loc.rs:264` | `g_space_factors_match_upstream_on_lif` | oracle | 5.73 | 109 | PASS | 2.00e-15 | **T1** | P1 |
| 114 | `pbc-gto` | `gth_pp_loc.rs:270` | `part2_matches_upstream_on_diamond` | oracle | 216.64 | 120 | PASS | 1.777e-12 | **T2**⚑ | P1 |
| 115 | `pbc-gto` | `gth_pp_loc.rs:284` | `part2_matches_upstream_on_lif` | oracle | >600 | 172 | TIMEOUT@600 s — outcome not observed | — | **T3** | P2 (P1 TIME → P2 TIME) |
| 116 | `pbc-gto` | `gth_pp_nl.rs:197` | `vnl_matches_upstream_on_diamond_222` | oracle | 2.67 | 115 | PASS | 1.67e-15 | **T1** | P1 |
| 117 | `pbc-gto` | `gth_pp_nl.rs:204` | `vnl_matches_upstream_on_silicon_222` | oracle | 3.37 | 119 | PASS | 4.44e-16 | **T1** | P1 |
| 118 | `pbc-gto` | `oracle_phase9.rs:312` | `cell_scalars_match_upstream` | oracle | 4.18 | 99 | PASS | asserted, no number printed | **T1** | P1 |
| 119 | `pbc-gto` | `oracle_phase9.rs:381` | `gv_and_si_match_upstream` | oracle | 4.08 | 108 | PASS | asserted, no number printed | **T1** | P1 |
| 120 | `pbc-gto` | `oracle_phase9.rs:423` | `lattice_ls_match_upstream` | oracle | 3.91 | 99 | PASS | asserted, no number printed | **T1** | P1 |
| 121 | `pbc-gto` | `oracle_phase9.rs:445` | `kpts_and_kconserv_match_upstream` | oracle | 4.14 | 99 | PASS | asserted, no number printed | **T1** | P1 |
| 122 | `pbc-gto` | `oracle_phase9.rs:483` | `ewald_matches_upstream` | oracle | 5.55 | 112 | PASS | asserted, no number printed | **T1** | P1 |
| 123 | `pbc-gto` | `oracle_phase9.rs:512` | `make_kpts_variants_match_upstream` | oracle | 0.69 | 97 | PASS | asserted, no number printed | **T1** | P1 |
| 124 | `pbc-gto` | `oracle_phase9.rs:572` | `angstrom_lattices_match_upstream_within_the_codata_gap` | oracle | 3.23 | 97 | PASS | asserted, no number printed | **T1** | P1 |
| 125 | `pbc-gto` | `pbc_intor.rs:488` | `ovlp_matches_upstream_on_diamond_222` | oracle | 3.89 | 115 | PASS | 1.288e-14 | **T1** | P1 |
| 126 | `pbc-gto` | `pbc_intor.rs:496` | `kin_matches_upstream_on_diamond_222` | oracle | 4.26 | 117 | PASS | 4.00e-15 | **T1** | P1 |
| 127 | `pbc-gto` | `pbc_intor.rs:504` | `ovlp_matches_upstream_on_diamond_321` | oracle | 4.44 | 115 | PASS | 1.288e-14 | **T1** | P1 |
| 128 | `pbc-gto` | `pbc_intor.rs:761` | `derivative_families_match_upstream_all_reference_cells` | oracle | 45.98 | 121 | PASS | worst printed 1.98e-14 | **T2**⚑ | P1 |
| 129 | `pbc-mp` | `oracle_phase15.rs:179` | `symm_map_and_operation` | oracle | 0.58 | 99 | PASS | integer surface, exact | **T1** | P1 |
| 130 | `pbc-mp` | `oracle_phase15.rs:229` | `padding_surface` | oracle | 0.56 | 98 | PASS | integer surface, exact | **T1** | P1 |
| 131 | `pbc-mp` | `oracle_phase15.rs:289` | `ao2mo_and_ao2mo_7d` | oracle | 25.65 | 150 | PASS | GDF 1.77e-9, MDF 2.61e-8, AFTDF 1.89e-4 (per-route gates) | **T1**⚑ | P1 |
| 132 | `pbc-mp` | `oracle_phase15.rs:410` | `lov_blocks` | oracle | 427.96 | 374 | PASS | max_dev 1.570e-5 (diamond [1,1,2]) | **T3**⚑ | P1 |
| 133 | `pbc-mp` | `oracle_phase15.rs:478` | `kmp2_energies` | oracle | >600 | 1797 | TIMEOUT@600 s — outcome not observed | first arm printed e_corr residual 5.418e-11 before kill | **T3** | P2 |
| 134 | `pbc-mp` | `oracle_phase15.rs:562` | `t2_rdm1_and_gamma1` | oracle | 15.73 | 167 | PASS | rdm1 3.46e-11, gamma1 1.73e-11 | **T1**⚑ | P2 |
| 135 | `pbc-mp` | `oracle_phase15.rs:641` | `stagger_energies` | oracle | 464.08 | 157 | PASS | upstream vs committed constants, last-digit | **T3**⚑ | P2 |
| 136 | `pbc-mp` | `oracle_phase15.rs:668` | `upstream_kump2_kernel_remains_an_explicit_refusal` | oracle | 1.00 | 97 | PASS | refusal confirmed | **T1** | P2 |
| 137 | `pbc-mp` | `oracle_phase15.rs:688` | `mo_first_ao2mo_block` | oracle | 542.73 | 960 | PASS | 5.818e-14 | **T3**⚑ | P2 |
| 138 | `pbc-mp` | `oracle_phase15.rs:765` | `committed_diamond_anchor_is_still_reproducible` | oracle | 45.53 | 485 | PASS | residual 2.113e-10 | **T2**⚑ | P2 |
| 139 | `pbc-mp` | `perf_dpbc28.rs:36` | `lov_build_and_kmp2_kernel_thread_scaling` | measurement | 458.55 | 300 | PASS | instrument | **T3**⚑ | P2 |
| 140 | `pbc-mp` | `perf_dpbc28.rs:123` | `kmp2_four_index_thread_scaling` | measurement | 94.01 | 1266 | PASS | instrument | **T2** | P2 |
| 141 | `pbc-mp` | `perf_dpbc28.rs:161` | `build_symm_map_growth_curve` | measurement | 2.43 | 213 | PASS | instrument | **T1** | P2 |
| 142 | `pbc-scf` | `df_swap.rs:466` | `krhf_on_mdf_matches_upstream_he_fcc` | oracle | 171.12 | 189 | PASS | gamma 3.601e-9, 2x2x2 2.826e-10 | **T2**⚑ | P2 |
| 143 | `pbc-scf` | `df_swap.rs:588` | `wall_clock_per_builder` | measurement | 128.04 | 315 | PASS | instrument | **T2** | P2 |
| 144 | `pbc-scf` | `exclude_dd_block_energy.rs:64` | `diamond_gamma_matches_upstream` | slow | >600 | 284 | TIMEOUT@600 s — outcome not observed | — | **T3** | P2 |
| 145 | `pbc-scf` | `exclude_dd_block_energy.rs:78` | `diamond_2x2x2_matches_upstream` | slow | >600 | 388 | TIMEOUT@600 s — outcome not observed | — | **T3** | P2 |
| 146 | `pbc-scf` | `gate3_rsdf.rs:93` | `gate3_both_routes_match_upstream_he_fcc` | oracle | 56.55 | 177 | PASS | RSDF 2.325e-10, GDF 2.750e-10 | **T2**⚑ | P2 |
| 147 | `pbc-scf` | `gate3_rsdf.rs:147` | `gate3_both_routes_match_upstream_diamond_gamma` | oracle | >600 | 301 | TIMEOUT@600 s — outcome not observed | — | **T3** | P2 |
| 148 | `pbc-scf` | `gate3_rsdf.rs:286` | `rs_mdf_matches_upstream_he_fcc` | oracle | 469.98 | 180 | PASS | 3.208e-10 | **T3**⚑ | P2 |
| 149 | `pbc-scf` | `krhf_bands_oracle.rs:144` | `krhf_get_bands_matches_upstream` | oracle | 13.33 | 156 | PASS | e_tot 2.167e-13 (bands asserted < 1e-9) | **T1** | P2 |
| 150 | `pbc-scf` | `kscf.rs:453` | `krhf_diamond_222_matches_upstream` | oracle | 270.89 | 627 | PASS | 3.997e-12 | **T2**⚑ | P2 |
| 151 | `pbc-scf` | `kscf.rs:477` | `krhf_he_all_electron_matches_upstream` | oracle | 13.76 | 148 | PASS | 2.167e-13 | **T1** | P2 |
| 152 | `pbc-scf` | `kscf.rs:494` | `kuhf_he_all_electron_matches_upstream` | oracle | 22.41 | 149 | PASS | 2.172e-13 | **T1**⚑ | P2 |
| 153 | `pbc-scf` | `kscf.rs:513` | `krhf_diamond_222_matches_upstream_at_the_default_mesh` | oracle | >600 | 1040 | TIMEOUT@600 s — outcome not observed | — | **T3** | P2 |
| 154 | `pbc-scf` | `newton_ah.rs:229` | `live_newton_matches_upstream_energy` | pending | 0.02 | 10 | FAIL | unconditional panic! (arm not wired) | **T1** | P2 |
| 155 | `pbc-symm` | `basis_precision_probe.rs:204` | `fock_block_diagonality_floor` | measurement | >600 | 906 | TIMEOUT@600 s — outcome not observed | first sweep printed max\|off-block F\| 3.99e-10 before kill | **T3** | P2 (P1 TIME → P2 TIME) |
| 156 | `pbc-tdscf` | `uhf.rs:495` | `live_bigbox_matches_molecular` | pending | 0.02 | 11 | FAIL | unconditional panic! (arm not wired) | **T1** | P1 |
| 157 | `pbc-tools` | `fft_thread_determinism.rs:38` | `fft_thread_child_emits_bits` | child | 0.08 | 21 | PASS | emits FFT bit hashes (child) | **T1** | P1 |

