# Follow-ups exposed by fixing the molecular drift carryover (2026-09-20)

**Source:** `20-molecular-python-suite-drift.md` items 1–4 are FIXED (loader
retired to `upstream_eval`, cross-dispatch test updated, `PyGHF`/`PyUHF`
kernel+run bound, `ConvergenceFailure` mapping + overlay-subclass raise).
Running the suite against the real vendored-2.12.1 oracle for the first time
surfaced the gaps below. The loader previously masked every one of them
(fixture error before any number was compared). No tolerance was loosened;
no test was taught to pass.

## (a) Native benzene/6-31G* RHF makes no progress in 100 s (scaling gap)

- `test_scf_rhf_benzene.py` (converted to `run_upstream`, honest gate).
- Measured 2026-09-20, `.so` 16:06: native `scf.RHF(benzene).kernel()` prints
  nothing (verbose=4) and returns nothing within 100 s / 500 s (SIGABRT shows
  the main thread inside `mf.kernel()`, i.e. Rust compute, not deadlock —
  the 8m20s probe burned 10m59s user CPU). Upstream 2.12.1 on the same
  geometry: 8 cycles, `e_tot = -230.7014617460919`, ~2 s wall.
- Cliff is between 24 AOs (H2O/cc-pVDZ, seconds) and 96 AOs (benzene).
  First suspects: minao init guess at scale, first JK build, DIIS — unowned.
  SCF-01's benzene claim still has no passing Python gate.

## (b) Native DF-HF reproduces conventional exactly — upstream's 2.09e-5 fitting error is absent (SCF-07)

- `test_scf_density_fit_uhartree_oracle`: native DF `-76.02676567287806`
  vs upstream DF `-76.026744737355`, `|Δ| = 2.094e-05` vs 1e-6 gate.
- Native DF == native conventional to all 14 digits, with default AND
  explicit `auxbasis='cc-pvdz-jkfit'`; upstream DF differs from upstream
  conventional by the same 2.09e-5. So the native DF path carries ~zero
  fitting error where upstream's carries 2e-5 — either near-exact fitting
  (implausible for JKFIT) or a silent conventional fallback. SCF-07 owner
  to determine; the gate stands as written.

## (c) Native `dip_moment` disagrees structurally (SCF-09)

- `test_dip_moment_h2o_ccpvdz`: native `[-0.008427, 0.909672, 0.030750]`
  vs upstream `[-5.9e-15, 2.05843942, -1.7e-15]`, max abs diff 1.149.
- Two stacked issues: (1) the binding takes no `unit` argument while
  upstream defaults to Debye (`analyze.rs:259` documents atomic units), so
  the test compares a.u. against Debye; (2) even in a.u. (0.9102 vs 0.8099
  magnitude, plus nonzero x/z on a symmetric molecule) the value is 12% off.
  Energies agree to 2.4e-10, so the density is right and the contraction is
  suspect. SCF-09 owner to decide the unit contract (`unit=` passthrough vs
  documented a.u.) and fix the contraction.

## (d) Native UHF cannot run open-shell: RHF Aufbau rejects odd nelec (SCF-02)

- `test_scf_uhf_open_shell_oracle` (NH2 doublet, 9 e): `PyUHF::kernel` (newly
  bound) fails with `Core(InvalidMolecule: RHF Aufbau requires even nelec;
  got 9)`. The generic kernel path uses the default (RHF) Aufbau even for
  UHF; `UhfRust::kernel` has the same shape (alpha/beta packing deferred in
  the driver). No Rust UHF odd-electron gate exists either. SCF-02 owner to
  wire UHF occupancy handling through the kernel.

## (e) Mulliken charges miss 1e-6 by 1.24× (marginal, SCF-09)

- `test_mulliken_pop_h2o_ccpvdz`: max abs diff **1.235e-6** (oxygen charge
  −0.30608703 vs −0.30608826) vs 1e-6 gate. One 24%-over row; could be
  convergence noise or a genuine 1e-6-level `mulliken_pop` gap. Cheapest of
  the five; re-measure at tighter `conv_tol` before touching the driver.

## Suite state after the drift fix

- Fixed: `test_scf_cross_dispatch` (5), `test_scf_ghf` (1),
  `test_panic_to_exception` (2), `test_scf_rhf_h2o` (1),
  `test_scf_rhf_ccpvdz` (3), `test_scf_diis` (1), `test_scf_xplat_uhartree`
  (1), `test_scf_chkfile` (4/4 incl. direction B) — all green on `.so` 16:06.
- Still red, all real gaps above: analyze ×3 (c+e), df ×1 (b), uhf ×1 (d),
  benzene ×1 (a, hangs — bound the runner before adding to CI).
