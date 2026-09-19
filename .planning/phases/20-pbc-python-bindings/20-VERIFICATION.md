# Phase 20 verification — PBC Python bindings + oracle enforcement

**Written:** 2026-09-14, closing plan 20-18 (verify + rollup).
**Format:** `17-VERIFICATION.md`'s — every claim carries the measurement that supports it; every
gate that was wrong is recorded as wrong along with what replaced it; every gate NOT met is recorded
as not met with its measured number and moved to `.planning/carryovers/`.
**Tree:** HEAD `5e6dd65` + the uncommitted Phase-18/19/20 working tree (D-20-A: no commits).
`python/pyscf/_native.abi3.so` mtime **2026-09-14 19:52:34** (20-19 D's rebuild) for every
measurement in §3–§7 unless a row says otherwise. Raw logs: `target/p20-18-final/`.

**The one-line summary.** The Phase-20 gate as restated (identity contract) is MET 18/18 and
proven to discriminate (18/18 failed before any binding); the example-script gate, restated
because upstream itself cannot run the named scripts, is MET on legs 1–3 (examples 22, 22-ksymm,
23-smearing) and NOT MET on leg 4 (no bounded native replacement for example 20 completes: the one
qualifying script, `40-custom_gdf.py`, was killed after 82 min against upstream's 53 s); one periodic driver
is bit-identical at 1 and 8 rayon threads; the **≥ 80 % upstream-suite target is NOT MET
(11 / 815 after 20-19; 11 / 815 again on the final tree)** for a structural reason recorded as a
carryover; native FFTDF `get_jk` is **5.95× slower** than upstream on a quiet box (performance
carryover). Phase 20 therefore stays `[ ]` in ROADMAP.

---

## §1 — Decisions taken for this rollup (with provenance)

| id | decision | taken by | provenance |
|---|---|---|---|
| D-20-18-1 | **Example-script gate RESTATED.** The 20-01 gate named `examples/pbc/20-k_points_scf.py` and `22-k_points_mp2.py` "run unmodified". Upstream 2.12.1 itself cannot: example 20 needs periodic meta-GGA (`xc='m06,m06'`) and a production k-point Newton-AH (both absent; estimated ≥ 4–6 days upstream, weeks native at 64 k / 65³), and example 22 exits `NotImplementedError` at **line 62** (`pbc/mp/mp2.py:23-24`, `kpt ≠ Γ`; measured exit 1 after 554 s). New gate: (1) `22-k_points_mp2.py` unmodified reaches upstream's terminal state — lines 1–58 native, `KMP2 e_tot` within 2e-6 (floor row 7) of −11.02605159764625 (2×2×2) and −11.01065485358922 (1 k), `NotImplementedError` at line 62; (2) `22-k_points_mp2_ksymm.py` fully native; (3) `23-smearing.py` fully native; (4) a bounded native replacement for example 20 (KRHF + KRKS(GGA) + BeckeGrids/`density_fit` on a small cell). | orchestrator, 2026-09-14 | `measurements/example-gate-gap-scope.md` §0, §5 |
| D-20-18-2 | **The ≥ 80 % upstream-suite target is recorded NOT MET**, with its full taxonomy, as a structural carryover: upstream PBC Python subclasses molecular classes (`gto.Mole`, `scf.hf.RHF`/`uhf.UHF`) that the Phase-3 identity overlay replaces with non-subclassable native classes (counterfactual with both lifted: 49 / 815). | orchestrator, 2026-09-14 | `measurements/upstream-pbc-suite-after-20-19.md`; carryover `20-upstream-pbc-suite-80pct-structural.md` |
| D-20-18-3 | **Pre-20-09 discrimination evidence is the D-20-C capture, not a checkout.** `20-18-PLAN` Task 1 says "check out the tree as of 20-08"; D-20-A forbids checkout, and D-20-C ran the identical test on HEAD `0f8b58d` + the stale 2026-08-20 `.so` before any binding existed. | 20-EXECUTION-NOTES D-20-A/D-20-C | `measurements/identity-gate-pre-20-09.out` |
| D-20-18-4 | **Periodic determinism lives in the `pbc-oracle` job, not in the `oracle-determinism` matrix.** That job's `setup-sibling-crates` does not clone `cube-math`/`rmath`, which `pyscf-pbc-*` need; `pbc-oracle-setup` does. | 20-18 | 20-03-SUMMARY §Deviations; `ci.yml` |
| D-20-18-5 | **The upstream suite is re-run whole on the final tree** (the `.so` changed after the 20-19 AC run: 18:49:53 → 19:52:34), not only the ported families — the whole overlay run is 2 min 41 s, i.e. cheap. | 20-18 | `.so` mtime vs `upstream-pbc-suite-after-20-19.md` header |

---

## §2 — Gates

| # | gate | statement | measured (final tree) | verdict | provenance |
|---|---|---|---|---|---|
| G1 | identity contract | `pyscf.pbc.<fam>.<Name> is pyscf._native.pbc.<fam>.<Name>` for the 18 names of `test_pbc_identity_gate.py` | **18 passed** (0.04 s); pre-binding tree **18 failed** (`ModuleNotFoundError: No module named 'pyscf._native.pbc'`) | **MET, discriminating** | §3.1 |
| G2.1 | example 22 terminal state | lines 1–58 native; KMP2 `e_tot` ≤ 2e-6; `NotImplementedError` at line 62 | 2×2×2 **4.558e-7**; 1 k **6.543e-9**; `NotImplementedError` at **line 62** (`mypt = mp.RMP2(mf).run()`); every touched name native, `mp.RMP2` refused | **MET** | §3.2 |
| G2.2 | example 22-ksymm | fully native, KMP2 `e_tot` ≤ 2e-6 | exit 0; **1.695e-6**; all names native, 0 upstream `pyscf.pbc` modules | **MET** (1.18× inside the floor — flagged) | §3.2 |
| G2.3 | example 23-smearing | fully native | exit 0; free energy (σ=0.1) **2.2e-9**, entropy 2.3e-8 | **MET** | §3.2 |
| G2.4 | bounded native replacement for example 20 | KRHF + KRKS(GGA) + Becke/`density_fit`, small cell | only `examples/pbc/40-custom_gdf.py` qualifies; natively it was **killed after 4927 s (82 min)** inside line 35 `mf.run()` (KRHF over the saved GDF `_cderi`); upstream runs the whole script in **53.5 s**; a single native GDF `get_jk` (exxdiv `ewald`) on the same cell did not return in 580 s | **NOT MET — none qualifies** (performance carryover) | §3.2 |
| G3 | upstream suite ≥ 80 % | `passed / collected` over the 815 `pytest.ini` selects | 20-19: **11 / 815** (1.35 %); final tree (`.so` 19:52:34) **11 / 815**, 0 per-test outcome changes; control 808 / 815 | **NOT MET** → carryover | §3.3 |
| G4 | periodic determinism | one periodic driver bit-identical at `RAYON_NUM_THREADS=1` and `8` | KRHF He-fcc 2×2×2: `e_tot_bits=0xc0067587e69e6ce6` in both processes; pool(1) == pool(8) == global, `e_tot`/`e_elec`/`mo_energy` `to_bits()` | **MET** | §3.4 |
| G5 | tolerance hygiene | no tolerance in this phase tighter than one ulp of its quantity; none loosened | no new tolerance below 1e-14 relative; bitwise used only for same-implementation A/B; no tolerance loosened in any plan (each SUMMARY states it); 20-18 re-referenced three stale post-20-19-D references with tolerances unchanged (§7) | **MET** | per-plan SUMMARYs; §7 |
| P1 | performance sanity (orchestrator) | native FFTDF `get_jk` within 2× of upstream | **114.26 s vs 19.19 s = 5.95×** (quiet box) | **NOT MET** → carryover | §3.5 |

Rust-side numeric gates re-established during the phase (details §4):

| gate | before Phase 20 | after | verdict |
|---|---|---|---|
| ksymm Gate C, GDF `KRKS` (17-08, 1e-8) | **1.432e-06 NOT MET** | **2.1997e-10** | MET (20-04 + fix) |
| `GDF/MDF.get_jk(omega)` (1.353e-08 floor) | refused | LR 2.223e-11, SR 1.939e-9 (GDF) / 1.830e-9 (MDF) | MET (20-05) |
| `rsjk` build/get_jk | refused (D-PBC-24) | still refused; ω half corrected | NOT MET, carryover `D-PBC-24-cintx-range-omega-PLAN.md` (20-06 branch B) |
| `df_ao2mo` He-fcc (1e-11) | 2.3298e-10 / 8.2146e-11 FAIL | 1.6667e-12 / 1.9839e-12 | MET (route pin, test-only) |
| `df_ao2mo` diamond gamma (1e-11) | never met | **3.8427e-8** (2793 s) | **NOT MET** → `20-gdf-df-ao2mo-diamond-gamma.md` |
| 157 ignored PBC gates tiered, CI | 0 CI coverage | 74 T1 in push `pbc-oracle` (local replay 74/74 green), nightly 130, full 152 | MET locally; first GitHub run unobserved (carryover) |

---

## §3 — Evidence

### §3.1 — Identity gate (Task 1)

| run | tree | command | result |
|---|---|---|---|
| before | HEAD `0f8b58d` + stale 2026-08-20 `.so`, before any binding (D-20-C) | `.venv/bin/pytest python/pyscf/tests/test_pbc_identity_gate.py` | **`18 failed in 0.05s`**, every case `ModuleNotFoundError: No module named 'pyscf._native.pbc'; 'pyscf._native' is not a package` — `measurements/identity-gate-pre-20-09.out` |
| after | final tree, `.so` 19:52:34 | `.venv/bin/pytest python/pyscf/tests/test_pbc_identity_gate.py -q -p no:cacheprovider` | **`18 passed in 0.04s`**, exit 0 — `target/p20-18-final/identity-gate-final{,-verbose}.out` |

The 18 names: `gto.{Cell,M}`, `df.{FFTDF,AFTDF,GDF,MDF,RSDF}`, `scf.{KRHF,KUHF,KROHF,KGHF}`,
`dft.{KRKS,KUKS,KROKS,KGKS}`, `symm.KPoints`, `mp.KMP2`, `cc.KRCCSD`. Intermediate states recorded by
the plans: 20-08 18 failed (moved one step later), 20-09 7/18, 20-14 8/18, 20-12 12/18, 20-13/20-15 18/18.

### §3.2 — Example-script gate (Task 2, restated per D-20-18-1)

**Wrapper:** `measurements/example_gate_wrapper.py` (recorded in the repo). It compiles the
unmodified example from its own file and `exec`s it (source, line numbers and tracebacks are
upstream's), with a line tracer confined to the script's module frame that snapshots the type of
every pyscf-typed global after each line; afterwards it records the terminal state (exception type
+ script line), every `(global, type)` pair seen, `pyscf.pbc._unported.which_impl(name)` for the
names the script touches, every `pyscf.pbc.*` module loaded from outside the overlay, and every
`PbcUpstreamFallthroughWarning`. Command:
`PYTHONPATH=$REPO/python .venv/bin/python example_gate_wrapper.py $REPO/examples/pbc/<x>.py out.json NAMES…`,
cwd `target/p20-18-final/` (never the repo root — 20-17 D1). The box was loaded by concurrent runs,
so wall times below are upper bounds.

| script | terminal state | quantity | native | upstream 2.12.1 | \|Δ\| | bound |
|---|---|---|---|---|---|---|
| `22-k_points_mp2.py` | **`NotImplementedError` at line 62** `mypt = mp.RMP2(mf).run()` (native refusal; upstream raises the same class at the same line, `pbc/mp/mp2.py:24`); wall 1637.7 s (27.3 min, loaded), RSS 1.87 GB | KMP2 `e_tot` 2×2×2 | −11.02605114183888 | −11.02605159764625 | **4.558e-7** | 2e-6 |
| | | KMP2 `e_tot` 1 k-point | −11.010654847046348 | −11.01065485358922 | **6.543e-9** | 2e-6 |
| `22-k_points_mp2_ksymm.py` | exit 0, 87.5 s (loaded; 16.4 s idle in 20-18-PRE), RSS 0.57 GB | KMP2 `e_tot` | −7.575651916077467 | −7.575653611089853 | **1.695e-6** | 2e-6 |
| `23-smearing.py` | exit 0, 119.0 s (loaded), RSS 2.56 GB | entropy (σ=0.1) | 3.3826092138536534 | 3.382609236505189 | 2.3e-8 | — |
| | | free energy (σ=0.1) | −2.2426373775484656 | −2.2426373797484516 | **2.2e-9** | — |
| | | ≈zero-T energy (σ=0.1) | −2.073506916855783 | −2.073506917923192 | 1.1e-9 | — |
| | | `e_tot` / `e_free` after σ=0.001 | −2.0565136230905225 / −2.0569087063365643 | −2.056513625508845 / −2.0569087061617015 | 2.4e-9 / 1.7e-10 | — |
| `40-custom_gdf.py` (item 4) | **killed at 4927 s** (82 min, the D-20-D rule applied by hand; box loaded for the first ~55 min, alone afterwards): Python stack via `gdb` + `PyRun_SimpleString` at 61 min showed line 35 `mf.run()`; `/proc/PID/io` flat over 20 s (compute-bound, ~8 cores, RSS 0.91 GB); `gdf-sample.h5` (7.4 MB) was written at ~3 min, so lines 27–34 completed | KRHF (GDF from `_cderi`) `e_tot` | not reached | −74.8147884422133 | — | — |
| | upstream: exit 0, **53.5 s**, RSS 0.33 GB | KRKS PBE (GDF, `_j_only=False`) `e_tot` | not reached | −75.1330890828251 | — | — |

`which_impl` (from the wrapper JSONs):

| script | types seen | `which_impl` | upstream `pyscf.pbc` modules loaded | fallthrough warnings |
|---|---|---|---|---|
| 22 | `cell` `pyscf._native.pbc.gto.Cell`; `kmf`, `mf` `pyscf._native.pbc.scf.KRHF` (gamma `scf.RHF(cell, kpt=)` returns the native KRHF at nk=1, 20-12 D3); `mypt` `pyscf._native.pbc.mp.KMP2` | `gto.Cell`, `scf.KRHF`, `mp.KMP2`, `scf.RHF`, `df.FFTDF` native; `mp.RMP2` **refused** | none | none |
| 22-ksymm | `cell` Cell; `kpts` `pyscf._native.pbc.symm.KPoints`; `kmf` KRHF; `kmp2` KMP2 | `gto.M`, `scf.KRHF`, `mp.KMP2`, `symm.KPoints` native | none | none |
| 23-smearing | `cell` Cell (via `pyscf.M`, 20-18-PRE shim); `mf` `pyscf._native.pbc.dft.KRKS` | `gto.M`, `dft.KRKS` native (`pyscf.M` is root, not a `pyscf.pbc` family: `which_impl` raises `KeyError` by design) | none | none |
| 40-custom_gdf | not recorded (run killed before the wrapper reported); a separate probe shows `cell.KRHF(kpts=).density_fit().with_df` is `pyscf._native.pbc.df.GDF` | — | — | — |

Notes. The ksymm example's 1.695e-6 is the same number 20-18-PRE measured (1.70e-6) and is inside
the 2e-6 KMP2 floor by only 1.18×; its KRHF sits 2.77e-8 from upstream (conv_tol 1e-7 and an Å
lattice — CODATA-2014 vs 2010 in the 8th digit). The 23-smearing numbers are bit-identical to
20-19 D's post-fix run; before D the σ=0.1 free energy was 5.04e-3 off.

**Item (4) candidate search.** `examples/pbc/*.py` that use `BeckeGrids`/`density_fit`/`KRKS` were
read: `21-k_points_all_electron_scf.py` (8-atom, 64 k, `mix_density_fit` unbound, `jk_method`,
`.newton()`), `35-gaussian_density_fit.py` (explicit auxbasis tuples, `df.aug_etb`),
`35-range_separated_density_fit.py` (`rs_density_fit`, `with_df.omega`), `26-linear_dep.py`
(molecular `scf.addons.remove_linear_dep_`) all hit unbound surface; **`40-custom_gdf.py`**
qualifies on paper: diamond `sto3g` all-electron 2×2×2, `cell.KRHF(kpts).density_fit()` with
`_cderi_to_save`/`build()`, KRHF from the saved `_cderi`, then `cell.KRKS(kpts).density_fit()` PBE
(a GGA, Becke grids via `density_fit`, 20-13) with `_j_only = False` over the same file.
It does not complete natively (table above). Probe (`target/p20-18-final/ex40_probe.py`, same cell, the
saved `_cderi`): `type(mf.with_df)` is native `GDF`, `conv_tol 1e-7`, `max_cycle 50`,
`exxdiv 'ewald'`; ONE `with_df.get_jk(init_guess, kpts, exxdiv='ewald')` was still running when
`timeout 580` killed it (concurrently with the 82-min run). **No bounded example exercising
KRHF + KRKS(GGA) + Becke/`density_fit` runs natively in reasonable time today**, so leg 4 is
recorded NOT MET and the GDF exchange cost joins `20-fftdf-jk-performance.md`.

### §3.3 — Upstream suite (Task 3)

Cited measurements: 20-18 `measurements/upstream-pbc-suite.md` (1 / 815, `.so` 17:42:00) and
20-19 AC `measurements/upstream-pbc-suite-after-20-19.md` (**11 / 815**, `.so` 18:49:53; control
808 / 815). Denominator: 920 collected → −105 `-k "not _high_cost and not _skip"` → −0 from the two
ignored `cc/test/test_h_*.py` (scripts with no tests) = **815** in 111 files; the other `pytest.ini`
globs (`*_slow*`, `*test_kproxy*`, `*test_proxy*`, `*test_bz*`, `*test_ks_noimport*`) match no file
in the 2.12.1 `pbc` tree; `eph`, `geomopt`, `mpicc`, `mpitools`, `tddft` have no test directory.
Nothing else excluded.

**Final-tree re-run (D-20-18-5).** The `.so` was rebuilt by 20-19 D (19:52:34) after the AC run
(18:49:53), so the whole suite was re-run with the identical harness:
`cd target/p20-19-suite && systemd-run --user --scope -p MemoryMax=11G ./run_families.sh overlay 2.14.0 run-overlay-final 3 3`
(2026-09-14 20:52:54–20:57:42, box loaded by concurrent runs), then `aggregate.py run run-overlay-final`
and `target/p20-18-final/suite_compare.py` (per-family tables via the 20-18 `report.py`, plus a
per-node-id diff against `run-overlay`). The `.so` mtime was 19:52:34 at the start and end of all
15 families (`logs/run-overlay-final.families`, 30/30 stamps).

| run | passed / collected | ported families | failed | errors | collection errors | crash/timeout |
|---|---:|---:|---:|---:|---:|---:|
| 20-19 AC (`.so` 18:49:53) | 11 / 815 | 11 / 660 | 104 | 16 | 684 | 0 |
| **final tree (`.so` 19:52:34)** | **11 / 815** (1.35 %) | **11 / 660** | 104 | 16 | 684 | 0 |

**Per-test diff: 0 outcome changes** across all 815 node ids. The 20-19 D overlap fix did NOT move
any of the 13 numerical mismatches out of failure; their values are unchanged at the reported
precision (e.g. `test_kuks_as_kuhf` −4.204045409901784 vs −4.204045409900913 before; `test_density_fit`
still −7.852143828842326; `test_klda8_cubic_kpt_222` still 4.321e-3) — so D is not their cause.
Per-family table identical to `upstream-pbc-suite-after-20-19.md`.

**Verdict: ≥ 80 % NOT MET — 11 / 815.**

Taxonomy after 20-19 (804 non-passing): unported family 155; import, identity-mismatch
(`gto.Mole` not subclassable 444 + native `scf.uhf.UHF` in `rohf.py` 240, minus the unported share)
529; runtime import, same mismatch 15; native API-surface gap 74; refused input 18;
**numerical 13**; crash / signal / timeout **0**. The 13 numerical node ids are in carryover
`20-upstream-suite-numerical-mismatches.md`.

**Harness finding (must survive):** `pytest.ini`'s `--import-mode=importlib` makes pytest import
the *vendored* `pyscf/pbc/__init__.py` for these files even with the overlay first on `PYTHONPATH`
(0 collected, 111/111 collection errors). Every number here uses `--import-mode=prepend` after
`-c pytest.ini`, with a plugin that asserts `pyscf.__file__` per process and strips site-packages
2.14.0 from `extend_path` (0 leaks in 111/111 processes).

### §3.4 — Determinism (Task 4)

New test `crates/pyscf-pbc-scf/tests/krhf_threads.rs::krhf_e_tot_is_bit_identical_at_1_and_8_rayon_threads`
(separate test file, AGENTS.md §2; oracle-free, not ignored). Fixture = `krhf_bands_oracle.rs`'s:
He-fcc `sto-3g` all-electron, Bohr, 2×2×2, FFTDF mesh `[15]*3`, `conv_tol 1e-12`,
`conv_tol_grad 1e-8`. It runs the converged KRHF under explicit `rayon::ThreadPool`s of 1 and 8
inside one process (the `cphf_k.rs` / `fft_jk_threads.rs` pattern) and once on the global pool, and
asserts `e_tot`, `e_elec`, `cycles` and every `mo_energy` equal by `to_bits()`. Non-vacuity: pool(8)
reports 8 workers, 8 k blocks, 2 cycles.

```
CARGO_TARGET_DIR=$PWD/target/gate CARGO_PROFILE_RELEASE_LTO=false RAYON_NUM_THREADS={1,8} \
  cargo test --release -p pyscf-pbc-scf --test krhf_threads -j 6 -- --exact \
  krhf_e_tot_is_bit_identical_at_1_and_8_rayon_threads --nocapture --test-threads=1
RAYON_NUM_THREADS=1: exit 0, ok. 1 passed — KRHF_DETERMINISM e_tot_bits=0xc0067587e69e6ce6 e_tot=-2.80738811655975251e0 cycles=2 global_threads=1
RAYON_NUM_THREADS=8: exit 0, ok. 1 passed — KRHF_DETERMINISM e_tot_bits=0xc0067587e69e6ce6 e_tot=-2.80738811655975251e0 cycles=2 global_threads=8
```

CI: a `PBC determinism` step in the push `pbc-oracle` job runs the same command at
`RAYON_NUM_THREADS=1` and `8`, requires `ok. 1 passed` in both, and compares the printed
`e_tot_bits` (D-20-18-4). `actionlint 1.7.12` on `ci.yml`: only the pre-existing `:435 if: false`
finding. Limitation recorded: the printed pool wall times (3.12 s / 3.50 s and 2.84 s / 2.62 s on a
loaded box) do not by themselves show parallel speed-up on this 1-AO fixture; rayon is on the
FFTDF J/K path by construction (W-02b, gated bitwise across 1/2/3/8 threads by `fft_jk_threads.rs`).
The value differs from the pre-20-19-D print (−2.8073881165599626) by the overlap-precision fix,
as expected.

### §3.5 — Performance sanity (Task 5, orchestrator request)

One FFTDF `get_jk(dm, hermi=1, kpts, exxdiv='ewald')` on the example-22 diamond cell (`gth-szv` /
`gth-pade`, Bohr), 2×2×2, default mesh **[47,47,47]** on both sides; the SAME complex128 density
(upstream `KRHF.get_init_guess()` saved to `jk_dm.npy`); two warm worker processes (native
`PYTHONPATH=python`, upstream `PYTHONPATH=$REPO`, version 2.12.1 asserted) driven alternately by
one controller; default threads (16). Before starting: load average 1.91 falling, only
desktop processes; during the run `ps` showed only the two workers (the `load1` column is the
1-minute average lagging behind the previous native call's 16 threads, not other load).
Scripts: `target/p20-18-final/jk_{controller,worker}.py`, log `jk_perf.log`.

| rep | native (s) | upstream (s) |
|---|---:|---:|
| 0 (warm-up, not counted) | 116.783 | 19.360 |
| 1 | 114.668 | 19.060 |
| 2 | 114.441 | 19.088 |
| 3 | 113.660 | 19.423 |
| **mean of 1–3** | **114.256** | **19.190** |

**Ratio 5.95×** (range 5.85–6.02×). Results agree: `|vj|` = 1.464697396574e+00 and
`|vk|` = 2.812059547747e+01 on both sides to all 13 printed digits. The earlier loaded-box
measurement (115–121 s vs 12–24 s) was therefore NOT a load artefact. > 2× → carryover
`20-fftdf-jk-performance.md`; no fix in this phase.

---

## §4 — Defects found and fixed during Phase 20

| # | defect | found by | fix | measured after |
|---|---|---|---|---|
| 1 | **FFTDF `coulG` cache keyed on the wrapped `kdiff_index`** (uncommitted 18-04 hunk): `coulG` is not per-grid-point invariant under `dk → dk + b`; `kscf::supercell_equivalence_holds` −10.347315387196 vs −10.531064341613 | 20-06 regression run; bisect `measurements/kscf-supercell-regression.md` | D-20-E: re-keyed on raw `dk` bits (`crates/pyscf-pbc-df/src/fftdf.rs`); `fft_jk_grad::coulg_build_counter_reads_nkpts` restated | Δ **1.5650591933535907e-10**, k-point energy bit-identical to HEAD; KRHF Si floor P1 2.884e-1 FAIL → P2 4.158e-12 PASS |
| 2 | **KS drivers gridded XC on the DF mesh** (`Krks/Kuks/Kroks/Kgks::from_df` used `with_df.mesh()`: GDF RS mesh `[13]³` vs `cell.mesh` `[35]³`) — the cause of 17-08's GDF Gate C 1.432e-06 | 20-04 bisect (`measurements/gdf-ksymm-bisect.md`; steps 1–5 0e0, step 6 `vxc` 1.810e-05) | 20-04-FIX: `PeriodicGrids::uniform(cell, None)`; callers pin `mf.grids` | **Gate C GDF 2.1997e-10 MET** (bound 1e-8) |
| 3 | GDF/MDF `get_jk(omega)` refused; then **SR 3-centre tensor truncated** (`incore::int3c::aux_e2_intor` overlap-radius screening under `erfc`) and **RSMDF metric radius at `precision**1.5`** instead of `mdf.py:264`'s `precision` | 20-05 (first SR run 1.414e-4 / 5.243e-3) | RSH dispatch + `range_coulomb`; SR neighbour list + `prescreen_exponent_sr`; mixed-scheme `j2c` radius | four routes ≤ 1.939e-9 (floor 1.353e-08); `gate3_rsdf` 2.396e-10, RSMDF 3.211e-10/1.897e-11/7.806e-12, band 1.4116e-9 |
| 4 | **`rsjk` `guess_omega` returned RSDF's formula** (He-fcc ω 0.73936/mesh 11 vs upstream 1.312754030266949/mesh 15) | 20-06 sufficiency check | ported `rsjk.py:1263/1293/1306` | ω at 1e-12, ke at 1e-10 (7 cases) |
| 5 | **`df_ao2mo` oracle compared port RSGDF with upstream CCGDF** (default flipped in `a423c0e`) | 20-02 tiers §5a; bisect `measurements/gdf-ao2mo-regression.md` (7 steps, cintx exonerated) | test-only `df.prefer_ccdf = true` | He-fcc `get_eri` 1.666666804567285e-12, `ao2mo_7d` 1.9839130338539235e-12 (Phase 14: 1.667e-12 / 1.984e-12) |
| 6 | **KUKSpU Hubbard term used the restricted formula per spin** (`kspu::add_vhubbard_weighted`; converged He `E_U` 9.187e-2 vs upstream 5.6e-14) | 20-13 D3 (binding refused it) | 20-13-FIX: 2 channels → `kukspu.py:96-102`; 1 channel bit-for-bit unchanged | `kukspu_he_matches_upstream` e_tot 6.08e-14, `E_U` 3.45e-10 / 3.10e-10 |
| 7 | **ksymm GGA on an s-only basis aborted the interpreter** (`KPoints::symmetrize_density_vec` indexed `dmats()[iop][1]`; release `panic = "abort"`) | 20-13 D4 | 20-13-FIX: build `l = 1` via `make_dmats(cell, &[rot], Some(1))` | KRKS 8.44e-14, KUKS 8.48e-14 vs full BZ; s-only == p-basis twin bitwise |
| 8 | **Python `get_veff` override left KS energy tags stale** (|dE| **6.2e-3 Ha** vs unbridged) | 20-13 override test | binding `TagGuard` (upstream's untagged-`vhf` recompute rule, `krks.py:118-119`) | override-then-kernel **bitwise** == unbridged |
| 9 | **SCF overlap integrated at `cell.precision`**, upstream at `precision*1e-5` with widened `rcut` (`pbc/scf/hf.py:47-55`); 23-smearing σ=0.1 free energy **5.04e-3** | 20-18-PRE bisect (S differs 2.0e-9; Γ p band 0.77 vs 0.20 occupied) | 20-19 D: `get_ovlp_scf` in `pyscf-pbc-gto`, used by every SCF `get_ovlp` hook + `get_bands` `s1e` + `kccsd_rhf_ksymm` | smearing free energy **2.2e-9**; floors fell ~100×: **KRHF Si gth 4.158e-12 → 2.931e-14**, **KRKS Si PBE 6.451e-12 → 4.086e-14**, bands mo/band 6.10e-11/1.68e-11 → 3.72e-13/4.23e-13, KUHF Li γ 1.493e-11 → 3.490e-12 |
| 10 | `use_ao_symmetry` forced on (upstream ANDs `not time_reversal and symmorphic and little_cogroup_ops`); `sigma` read-only; `KohnShamDFT.smearing_method` missing | 20-18-PRE example runs | binding predicate + setters | 22-ksymm exit 0; test_pbc_example_shims 8 passed |
| 11 | molecular overlay packages hid upstream names PBC modules import (`gto.basis` 204, `dft.radi` 86, …); `gen_uniform_grids` missing | 20-18 suite | 20-19 A/C `_passthrough.py` + `extend_path` | 0 missing-name import errors; import probe 41 → 73 / 195 |
| 12 | native `Cell` rejected `cell.output=`, list-form atoms, per-element pseudo dicts (same name), explicit shell lists | 20-18 suite class 5 (68 tests) | 20-19 B | 68/68 past `Cell` construction; H₂ explicit shells vs upstream ΔS 1.8e-15 |
| 13 | flat `_native.<mod>` never in `sys.modules` (`'pyscf._native' is not a package`) | 20-08 D4 | 20-09 step 0b | `import pyscf.dft, pyscf.mp, pyscf.cc` ok |
| 14 | `pyscf.pbc.gto` `hasattr` raised `ImportError` | 20-17 | `__getattr__` refuses dunders, maps `ImportError` → `AttributeError` | test_pbc_unported |
| 15 | numpy linking OpenBLAS moves upstream's own `fft_jk` answer by 1.712e-7 (CI venv would be red) | 20-03 | CI oracle venv builds numpy 1.26.4 with `-Dblas=none` and asserts it | local replay 74/74 T1 green |

Also closed: `project_mo_nr2nr` implemented (was a `phase: 20` refusal; 3.22e-15 vs upstream);
PyO3 wall D-PBC-14 machine-checked (20-16, proven red on an injected dep).

---

## §5 — Gates NOT met (recorded, not absorbed) and the carryover ledger

| item | measured | carryover |
|---|---|---|
| (a) upstream suite ≥ 80 % | 11 / 815 (20-19 and final tree); counterfactual 49 / 815 | `20-upstream-pbc-suite-80pct-structural.md` |
| (b) periodic meta-GGA | `M06` parse failure after 126 s; `Family::Mgga` refused | `20-periodic-meta-gga.md` |
| (c) Newton-AH k-point driver | `.newton` unbound; `newton_ah.rs` dense real test driver | `20-periodic-newton-ah.md` |
| (d) GDF `df_ao2mo` diamond gamma | 3.8427e-8 vs 1e-11, 2793 s (T3) | `20-gdf-df-ao2mo-diamond-gamma.md` |
| (e) KMP2Stagger mesh parity | 1.136e-3 with only `with_df.mesh` pinned (4.3e-7 with `cell.mesh`) | `20-post-scf-convention-gaps.md` |
| (f) `pbc.ao2mo` pair Ω/N scaling | binding rescales by N/Ω; 1.211e-11 vs upstream | `20-post-scf-convention-gaps.md` |
| (g) ksymm SCF He-fcc mesh 15 | 1.354e-6 vs full BZ (3.6e-14 at default mesh; KMP2 6.84e-9) | `20-ksymm-scf-he-fcc-coarse-mesh.md` |
| (h) `krks_stress` (Phase 18) | 1.841e-8 / 1e-8; 2.362e-9 / 2e-9; 3.061e-9 / 2e-9 | `20-foreign-phase-gate-misses.md` |
| (i) gto oracle gates unpinned | `gth_pp_loc`/`gth_pp_nl`/`pbc_intor` compared vs 2.14.0 locally | `20-oracle-harness-and-ci.md` |
| (j) Phase-19 stubs | adc `gate_d_ip_spec_factors` 9.358e-1 vs 5e-4; tdscf `live_bigbox` panic; scf `newton_ah` live panic | `20-foreign-phase-gate-misses.md` |
| (k) molecular Python drift | 10 failed + 9 errors, pre-existing (strip A/B identical) | `20-molecular-python-suite-drift.md` |
| (l) `pyscf-bench` build | `krks_profile.rs:1694` `KsNumInt::unfold_kdms` → exit 101 | `20-tree-hygiene.md` |
| (m) CI | `schedule` runs un-guarded jobs nightly; cube-math/rmath not cloned by `setup-sibling-crates`; first-push run unobserved | `20-oracle-harness-and-ci.md` |
| (n) per-element pseudo dict | different names per element refused | `20-per-element-pseudo-dict.md` |
| (o) 13 numerical mismatches | e.g. `dft/test_rks::test_density_fit` 3.134 Ha, `test_klda8_cubic_*` 4.32e-3 / 3.39e-3 | `20-upstream-suite-numerical-mismatches.md` |
| (p) FFTDF JK performance | 5.95× upstream | `20-fftdf-jk-performance.md` |
| example gate leg 4 (bounded native replacement for example 20) | none qualifies: `40-custom_gdf.py` killed at 4927 s vs upstream 53.5 s; one GDF `get_jk` > 580 s | `20-fftdf-jk-performance.md` (GDF section), `20-periodic-meta-gga.md`, `20-periodic-newton-ah.md` |
| (q) stale floor quotes | `gate.rs`/`gate_openshell.rs` comments, `12-VERIFICATION`, `pbc-oracle-tiers.md`, `docs/pbc-status.md` | `20-tree-hygiene.md` |
| rsjk (20-06 branch B) | still refused | `D-PBC-24-cintx-range-omega-PLAN.md` (updated by 20-06) |

---

## §6 — Floor table updates

`measurements/README.md` §1 rows 1, 3, 3a and 12, and §2's band-energy row, now carry the
post-20-19-D numbers with the old values struck through and marked superseded (not deleted); row 2
(diamond `gth-pade` 4.00e-12) is marked "not re-measured; its Si analogue fell to 2.931e-14". §6 of
that file adds the Phase-20 example and performance numbers.

---

## §7 — Final checks (Task 7)

| check | command | result |
|---|---|---|
| xtask (all checks) | `cargo run -p xtask` | **exit 0**: `check-no-fma` PASS, `check-forbidden-paths` PASS (410 files), `check-catch-unwind` PASS (856 files), `check-dependency-wall` PASS (cubecl ALG-06 + PyO3 D-PBC-14), `check-cubecl-pin` PASS (6 crates at 0.10.0), `check-orphan-modules` PASS (449 files) — `target/p20-18-final/xtask-all.log` |
| Python suite, as found | `.venv/bin/pytest python/pyscf/tests -q -p no:cacheprovider -rfE` (`.so` 19:52:34 start = end; 1490.78 s, loaded box) | **14 failed, 407 passed, 3 xfailed, 9 errors**, exit 1 — `pytest-full.log`. vs the triage baseline set (`target/p20-19-suite/pysuite/after.log`: 10 failed + 9 errors) the diff is **4 extra failures**, all 20-19 D fallout (below) |
| the 4 extra failures | `test_pbc_scf.py::test_krhf_energy_is_the_rust_kernel_bitwise`, `test_pbc_dft.py::test_krks_si_pbe_is_the_rust_kernel_bitwise`, `test_pbc_override_dispatch.py::test_kmats_payload_is_row_major_and_keeps_the_phase[_KRHFBridgeSelftest, KRHF]` | binding == Rust stayed **bitwise** in all of them; what failed were **stale references** from before 20-19 D moved the SCF overlap bits: the printed Rust `e_tot` (`-2.807388116559963` → now `-2.807388116559753`; `-7.785668903719571` → now `-7.785668903725981`, both from the post-D gate logs `target/p20-19-D/after/{scf,dft}.log`) and a plain-precision `pbc_intor('int1e_ovlp')` layout reference (1.01e-10 away, because the SCF hook now integrates at `precision*1e-5`) |
| restated (20-18) | references updated with provenance comments; the layout test now compares against `pbc_intor('int1e_ovlp', hermi=0)` on a cell built at `precision*1e-5` (upstream `pbc/scf/hf.py:47-55`), which the payload matches at **0.0**; **no tolerance changed** (5e-15, exact string, 1e-14) | `pytest test_pbc_override_dispatch.py test_pbc_scf.py test_pbc_dft.py` → **140 passed**, exit 0 (622.56 s) — `pytest-restated-files.log` |
| Python suite, after the restatement | the four cases pass and nothing else changed | **10 failed, 411 passed, 3 xfailed, 9 errors** by composition — the FAILED/ERROR ids equal the triage baseline set exactly (`pytest-final-ids.txt` minus the 4 restated ids = `pytest-baseline-ids.txt`). The full directory was not re-run a second time. |
| identity gate | `pytest test_pbc_identity_gate.py` | 18 passed |
| determinism | §3.4 | exit 0 at `RAYON_NUM_THREADS=1` and `8`, same bits |
| rustfmt | `rustfmt --edition 2024 --check crates/pyscf-pbc-scf/tests/krhf_threads.rs` | clean (only file of Rust touched) |
| actionlint | `target/p20-03/actionlint .github/workflows/ci.yml` | 1 finding, pre-existing (`:435 if: false`) |
| git status (for the record) | `git status --short \| wc -l` | **268** entries (2026-09-14 ~21:45, after all 20-18 writes) |

---

## §8 — What this phase licenses, and what it does not

**Licensed.** `from pyscf.pbc import gto, df, scf, dft, symm, mp, cc, ci, ao2mo, tools, lib` in an
overlay process serves native Rust objects for the bound surface, with the subclass-override contract
(11 hooks, bitwise when not overridden), the complex k-resolved NumPy boundary, and the numbers of
§3.2 against upstream on real example scripts; unported families announce themselves once
(D-PBC-35). The periodic T1 oracle gates run in CI for the first time.

**Not licensed.**
* **Running upstream PBC Python tests or modules on top of the overlay** — 11 / 815; upstream PBC
  Python cannot subclass the native molecular classes.
* **Anything needing `.newton()`, meta-GGA, `mix_density_fit`, gamma-point post-HF shapes,
  `scf.addons` conversions, or periodic gradients from Python** (Phase 18 unbound).
* **Periodic exchange at upstream's speed** — FFTDF 5.95× slower on the example-22 cell; GDF far
  worse (`40-custom_gdf.py` > 82 min vs 53.5 s).
* **GDF `get_eri` on diamond at 1e-11.**
