# 20-18 SUMMARY — verify + rollup (Phase 20 rolled up, NOT closed)

**Shipped:** 2026-09-14. All seven orchestrator tasks done. No git add/commit/stash/restore/checkout
(D-20-A). `.so` 2026-09-14 19:52:34 (20-19 D's build) for every Python measurement; no `.so`
rebuild in this plan. Raw logs: `target/p20-18-final/`. The authority is `20-VERIFICATION.md`.

## Gate table

| gate | measured | verdict |
|---|---|---|
| G1 identity contract (18 names) | final tree **18 passed**; pre-binding D-20-C capture **18 failed** | **MET**, discriminating |
| G2.1 `22-k_points_mp2.py` unmodified → upstream's terminal state | KMP2 `e_tot` 2×2×2 **4.558e-7**, 1 k **6.543e-9** (bound 2e-6); `NotImplementedError` at **line 62**; all touched names native, `mp.RMP2` refused; 1637.7 s (loaded) | **MET** |
| G2.2 `22-k_points_mp2_ksymm.py` fully native | exit 0; KMP2 `e_tot` **1.695e-6** | **MET** (1.18× margin) |
| G2.3 `23-smearing.py` fully native | exit 0; σ=0.1 free energy **2.2e-9**, entropy 2.3e-8 | **MET** |
| G2.4 bounded native replacement for example 20 | only `40-custom_gdf.py` qualifies; natively **killed at 4927 s** in KRHF-over-GDF (line 35) vs upstream **53.5 s**; one GDF `get_jk` > 580 s | **NOT MET — none qualifies** |
| G3 upstream suite ≥ 80 % | **11 / 815** on the final tree (whole-suite re-run, 0 per-test changes vs 20-19 AC); control 808 / 815 | **NOT MET** (structural) |
| G4 periodic determinism | KRHF He-fcc 2×2×2 `e_tot_bits=0xc0067587e69e6ce6` at `RAYON_NUM_THREADS=1` and `8`; pool(1)==pool(8)==global bitwise | **MET** |
| P1 FFTDF `get_jk` perf (quiet box, 3 interleaved reps) | native **114.26 s** vs upstream **19.19 s** = **5.95×** | **NOT MET** → carryover |

## What was done

1. **Identity gate** run on the final tree (18 passed, 0.04 s); D-20-C's
   `measurements/identity-gate-pre-20-09.out` cited as the discriminating half (D-20-18-3).
2. **Example gate** per the orchestrator's restatement (D-20-18-1). New wrapper
   `measurements/example_gate_wrapper.py` (exec of the unmodified file + module-frame line tracer →
   types, `which_impl`, upstream `pyscf.pbc` modules, fallthrough warnings, terminal state + line).
   Ran 22 (detached, `target/p20-18-final/run_ex22.sh`), 22-ksymm, 23-smearing, and
   `40-custom_gdf.py` native + upstream. Candidate search for item 4 recorded in VERIFICATION §3.2.
3. **Upstream suite**: `.so` changed after the AC report (18:49:53 → 19:52:34), whole suite is
   cheap, so re-ran all 15 families with the unchanged harness (`run-overlay-final`, `.so` stamped
   30/30): 11 / 815, identical per-family table, 0 per-node-id outcome changes, the 13 numerical
   mismatches unchanged → 20-19 D is not their cause. `--import-mode=importlib` harness issue noted.
4. **Determinism**: new `crates/pyscf-pbc-scf/tests/krhf_threads.rs` (separate test file,
   oracle-free, not ignored); new `PBC determinism` step in the push `pbc-oracle` job
   (`.github/workflows/ci.yml`) running it at `RAYON_NUM_THREADS=1|8` and comparing printed bits.
   Local: `cargo test --release -p pyscf-pbc-scf --test krhf_threads` exit 0 both ways.
5. **Performance**: `target/p20-18-final/jk_{controller,worker}.py` (two warm workers, same dm,
   interleaved), load 1.91 at start, only the workers running. 5.95×; `|vj|`,`|vk|` agree to 13 digits.
6. **Rollup**: `20-VERIFICATION.md`; floor table rows 1/3/3a/12 + §2 band row marked superseded,
   row 2 marked not re-measured, new §6 (rows 15–21); `STATE.md` (counters 12/7/120/91/76 %, focus);
   `ROADMAP.md` Phase-20 row (`[ ]`, honest status); `FEATURES` (new PBC-bindings section in the
   file's `* Section` / `  - item` format); `20-EXECUTION-NOTES.md` §4 closing.
   **13 carryover files** `.planning/carryovers/20-*.md`: `upstream-pbc-suite-80pct-structural` (a),
   `periodic-meta-gga` (b), `periodic-newton-ah` (c), `gdf-df-ao2mo-diamond-gamma` (d),
   `post-scf-convention-gaps` (e, f), `ksymm-scf-he-fcc-coarse-mesh` (g),
   `foreign-phase-gate-misses` (h, j), `oracle-harness-and-ci` (i, m),
   `molecular-python-suite-drift` (k), `tree-hygiene` (l, q), `per-element-pseudo-dict` (n),
   `upstream-suite-numerical-mismatches` (o), `fftdf-jk-performance` (p + GDF, item 4).
7. **Final checks**: `cargo run -p xtask` exit 0 (6 checks PASS); Python suite as found
   **14 failed / 407 passed / 3 xfailed / 9 errors** — 4 more failures than the triage baseline,
   all 20-19 D fallout (see Deviations); after the restatement those files give 140 passed, so the
   suite is 10 failed / 411 passed / 9 errors with ids equal to the baseline set;
   `git status --short | wc -l` recorded in VERIFICATION §7.

## Deviations

- **D1 — three Phase-20 Python tests re-referenced (not loosened).** 20-19 D changed every
  periodic SCF bit but did not run the full Python suite; four cases then failed on stale
  references while their binding==Rust bitwise halves still passed:
  `test_pbc_scf.py` `RUST_PRINTED_E_TOT` −2.807388116559963 → **−2.807388116559753** and
  `test_pbc_dft.py` `RUST_PRINTED_SI_PBE` −7.785668903719571 → **−7.785668903725981** (both from the
  post-D gate logs `target/p20-19-D/after/{scf,dft}.log`; KRHF also printed by `krhf_threads.rs`);
  `test_pbc_override_dispatch.py::test_kmats_payload_is_row_major_and_keeps_the_phase` now lays
  its reference out on a cell built at `precision*1e-5` with `hermi=0` (upstream `pbc/scf/hf.py:47-55`),
  matched at 0.0 (plain-precision reference 1.01e-10 away). Tolerances unchanged (exact string, 5e-15, 1e-14).
- **D2 — item (4) has no qualifying script** (orchestrator allowed "state none qualifies"); the
  measured GDF slowness is recorded, not diagnosed.
- **D3 — `40-custom_gdf.py` killed by hand at 4927 s** (D-20-D spirit; it outlived every other
  task). Its Python stack was captured with `gdb` + `PyGILState_Ensure`/`PyRun_SimpleString`
  (the GIL was already held by the driver, `PyGILState_Ensure` returned LOCKED).
- **D4 — timings under load.** Examples 22/22-ksymm/23/40 ran concurrently with builds and suites;
  their wall times are upper bounds. Task 5 ran alone.
- **D5 — the determinism test's pool wall times** (3.12/3.50 s, 2.84/2.62 s) do not show a speed-up
  on the 1-AO fixture; rayon is on the FFTDF J/K path by construction (`fft_jk_threads.rs`).
- **D6 — CubeCL manual (AGENTS.md §3) not consulted:** no kernel written; one host-side Rust test.
