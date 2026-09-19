# 20-11 SUMMARY — `KPyOverrideBridge`: the k-point subclass-override contract

**Shipped:** 2026-09-14. All four tasks are done, in TDD order. Every plan verification command is green; the deviations are recorded below. No driver is bound: 20-12 binds the drivers.

## Harness choice (orchestrator latitude)

**Choice: a private native base class, not a free function.** `pyscf._native.pbc.scf._KRHFBridgeSelftest(with_df)` is a `#[pyclass(subclass, dict)]` in `crates/pyscf-py/src/pbc/scf.rs`.
- Its eleven hook methods run the Rust `Krhf` defaults through the bridge's payload codec. `Krhf` is built per call from `extract_df(with_df)`, which is the 20-10 ownership contract.
- Its `kernel()` drives `pyscf_pbc_scf::kernel(&bridge, &cfg)`. `kernel(use_bridge=False)` runs `Krhf::kernel` with no bridge; the negative tests use that as the reference.
- A Python subclass of it is dispatched exactly as a subclass of 20-12's `KRHF` will be, and `super().get_veff(...)` works.

A plain `_kbridge_selftest(obj)` function could not test that, because an override would have no native `super()` to delegate to. This harness also exercises real `call_method1` dispatch, MRO resolution, both payload directions and exception propagation now, and nothing is skipped.

The test file has `DRIVERS = [nscf._KRHFBridgeSelftest]`. 20-12 appends `nscf.KRHF`, and every case then runs against both classes. One placeholder case is skipped with the reason `needs 20-12 KRHF`.

## Task 1 — tests first

`python/pyscf/tests/test_pbc_override_dispatch.py` has 44 cases plus 1 skipped. RED: collection failed with `AttributeError: module 'pyscf._native.pbc.scf' has no attribute '_KRHFBridgeSelftest'`.

The fixture is He-fcc `cc-pvdz` (5 AO), all-electron, Bohr, mesh `[15]*3`, on a **3×1×1** k-mesh.
- The first draft used `sto-3g` on 2×1×1 and passed everything. It was vacuous: with 1 AO a transposition cannot be seen, and on a TRIM mesh every k-matrix is real (memory `trim-meshes-make-phase-gates-vacuous`).
- `test_fixture_is_not_vacuous` now asserts `max|Im S| > 1e-3` and a non-symmetric `mo_coeff`.

The cases cover:
- **Payload layout.** The codec `get_ovlp` equals the independently laid-out 20-09 `cell.pbc_intor('int1e_ovlp')` to within 1e-14, and differs from its transpose on the complex blocks. `eig` output satisfies `F c = S c e` column-wise, so `mo_coeff` is column-major.
- **Negative contract.** An unsubclassed instance, and subclasses overriding nothing (including one two levels deep), report `overridden == []`. Their `e_tot` is **bitwise** equal to raw `Krhf::kernel`, with equal cycle counts.
- **Per hook (11, parametrised).** A pass-through override is invoked (counter > 0), `overridden == [hook]`, and `e_tot` is **bitwise** equal to raw Rust.
- **Other dispatch cases:**
  - all 11 overridden at once: bitwise;
  - override reached through an intermediate Python class;
  - `get_veff` count == `cycles + 1`.
- **The override changes the physics.**
  - `get_hcore + c·S` moves `e_tot` by `2c` (N=2) and every orbital energy by `c`.
  - `get_veff + c·S` moves `e_tot` by `c`.
  - `energy_nuc → 0` gives `e_tot == e_elec` bitwise.
  - A stacked real ndarray return is accepted, still bitwise.
  - An instance attribute (`mf.energy_nuc = f`) is seen.
- **Exceptions.**
  - A custom `Boom` raised in any of the 11 hooks surfaces as `Boom`, with the same type and message.
  - A `KeyError` on the 2nd `get_grad` call propagates, and the kernel stops at once (`n == 2`).
  - Seven malformed returns raise `TypeError`/`ValueError`.
- **Probe cache.** `_kbridge_probe_count()` rises by exactly 1 for a new type and by 0 over two more kernels.

**Mutation check:** writing `mo_coeff` C-order instead of F-order in `mo_coeff_to_py` makes **4 tests fail**: the `eig`, `make_rdm1` and `get_grad` bitwise cases and `test_mo_coeff_payload_is_column_major`. With the change reverted and rebuilt, the file is green again.

## Tasks 2–4 — the bridge (`crates/pyscf-py/src/pbc/kbridge.rs`) and the cached probe (`caches.rs`)

### Public Rust API that 20-12 consumes

```rust
pub const K_HOOKS: [&str; 11];   // get_ovlp get_hcore get_init_guess get_veff get_fock eig get_occ make_rdm1 energy_elec energy_nuc get_grad
pub struct KPyOverrideBridge<'a, D: KOverrideHooks + ?Sized> { pub slf: Py<PyAny>, pub py_cell: Py<PyAny>, /* inner: &'a D, mask, stash */ }
impl KPyOverrideBridge<'a, D> {
    pub fn new(py, slf: Py<PyAny>, py_cell: Py<PyAny>, base: &Bound<PyType>, inner: &'a D) -> PyResult<Self>;
    pub fn finish<T>(&self, res: Result<T, PyscfRsError>) -> PyResult<T>;   // re-raises the ORIGINAL Python exception
    pub fn take_py_err(&self) -> Option<PyErr>;
    pub fn overrides(&self, hook: &str) -> bool;
    pub fn overridden_hooks(&self) -> Vec<&'static str>;
    pub fn inner(&self) -> &D;
}
impl KOverrideHooks for KPyOverrideBridge<'_, D>
// payload codec — the ONE layout the default hook pymethods must also use
pub fn kmats_to_py(py, &[CTensor], nao) -> PyResult<Bound<PyAny>>                // per-k list, row-major
pub fn kdms_to_py(py, &[KMats], nao) -> PyResult<Bound<PyAny>>                   // 1 channel: per-k list; else list of lists
pub fn mo_coeff_to_py(py, &[CTensor], nch, nao) -> PyResult<Bound<PyAny>>        // (nao,nmo), column-major
pub fn mo_values_to_py(py, &[Vec<f64>], nch) -> PyResult<Bound<PyAny>>
pub fn kmats_from_py(obj, nk, nao, what) -> PyResult<KMats>
pub fn kdms_from_py(obj, nch, nk, nao, what) -> PyResult<KDms>
pub fn mo_coeff_from_py(obj, nch, nk, nao, what) -> PyResult<Vec<CTensor>>
pub fn mo_values_from_py(obj, nch, nk, what) -> PyResult<Vec<Vec<f64>>>
// caches.rs
pub fn k_override_mask(py, ty, base, probe: FnOnce() -> PyResult<u16>) -> PyResult<u16>
pub fn k_override_probes_run() -> usize
```

**20-12 driver pattern** (it is `_KRHFBridgeSelftest::kernel`):
1. `let d = Krhf::from_df(extract_df(with_df)?)`
2. `KPyOverrideBridge::new(py, slf.clone().into_any().unbind(), py_cell, &py.get_type::<Self>(), &d)?`
3. `bridge.finish(pyscf_pbc_scf::kernel(&bridge, &cfg))?`

Keep the GIL: `Krhf` is not `Sync`.

### Semantics

- **Dispatch.** Each of the 11 hooks: if Python overrides it, `slf.call_method1(hook, args)`; otherwise the Rust default of `inner` runs directly (the `bridge.rs:100-104` fallback, applied to every hook).
  - The accessors (`cell`, `kpts`, `nset`, `nfock`, `nao`) and the driver-internal `diis_dms` / `free_energy` always delegate.
  - Call signatures are upstream-positional: `get_veff(cell, dm)`, `get_fock(h1e, None, vhf, dm)` (upstream `cycle=-1` means the bare Fock), `get_grad(mo_coeff, mo_occ, fock)`, and so on. The table is in the file header.
- **Probe.**
  - A hook is overridden if the raw `__mro__`/`__dict__` entry on `type(slf)` is not the object `base` resolves (compared by identity, with no descriptor invocation), or if the instance `__dict__` shadows it.
  - The type half is cached in a `PyOnceLock<Mutex<HashMap<(type, base), mask>>>`. The cache holds strong references to both types, so a garbage-collected class can never pass its address and stale mask to a new class. The probe runs with the lock released.
  - The instance half is re-read once per `kernel()`. In the loop, a hook check is a bit test.
- **Payloads.** 20-07's `kmats_to_pylist` / `kdms_to_pylist` / `to_ctensor` / `ctensor_to_pyarray`. One channel drops its axis (upstream KRHF `(nk,nao,nao)`); more than one is nested (KUHF).
  - Reads accept a list, a tuple or a stacked ndarray, complex or real. Real input is widened exactly by `numpy.asarray`.
  - Shapes, k counts and channel counts are validated: a mismatch raises `TypeError`/`ValueError` and never truncates.
- **Exceptions.**
  - The first `PyErr` is stashed, and a converted `PyscfRsError` `?`-propagates (the `call_hook` pattern).
  - `finish` re-raises the original exception.
  - `get_grad` is infallible in the trait. It returns `[NaN]` and poisons the bridge, so the next hook, overridden or not, returns `Err`.

## Verification (2026-09-14)

| command | result |
|---|---|
| `cargo check -p pyscf-py --release` (`target/py`, LTO=false) | exit 0, no warnings in touched files |
| `maturin develop --release --skip-install` (EXECUTION-NOTES §2 env) | **maturin exit=0**, three builds: `Finished` 34.82 s (first); 8.93 s (mutant); 9.12 s (restored) |
| `.venv/bin/pytest python/pyscf/tests/test_pbc_override_dispatch.py -q -p no:cacheprovider` | **44 passed, 1 skipped in 19.06 s**, exit 0 |
| get_veff override counter > 0 / no-override subclass bitwise | yes / yes (`to_bits` via `.view(uint64)`, vs raw `Krhf::kernel`) |
| ad-hoc He `sto-3g` 2×2×2 (8 k): no-override bridge and all-11-overridden, vs raw Rust | both **bitwise**, Δ = 0.0; `e_tot = -2.8073881165599626`, 2 cycles; per-hook calls `get_veff 3, get_fock 2, …` (per cycle, not per k) |
| `cargo run -p xtask --bin check-catch-unwind` | exit 0 (843 files) |
| `check-forbid-lazy-static` | exit 0 |
| `check-dependency-wall` | exit 0, both PASS lines (cubecl, PyO3 D-PBC-14) |
| `check-orphan-modules` | exit 0 (444 files) |
| `grep -c 'call_method1' crates/pyscf-py/src/pbc/kbridge.rs` | **11** == hooks in `KOverrideHooks` (get_ovlp … get_grad) |
| full `.venv/bin/pytest python/pyscf/tests -q -p no:cacheprovider` | **192 passed, 21 failed, 9 errors, 5 skipped, 3 xfailed** (254 s). See D4 |

## Deviations

- **D1 — the premise "the probe is multiplied by nkpts" is wrong for this trait.** `KOverrideHooks` hooks are batched over k: one `get_veff` call per cycle carries every k-point. An uncached probe would therefore cost hooks × cycles, not × nkpts. The cache is still implemented as the plan asks: once per `(type, base)`, not once per hook per cycle.
- **D2 — the probe is not `hasattr`.** The plan named the `bridge.rs:102` `hasattr` probe. Once 20-12 binds native default pymethods, `hasattr` is always true, so every hook would round-trip through Python. The probe asks "resolved differently from the native base class" instead, and keeps the fast Rust path for non-overridden hooks.
  - Limitation: a class mutated **after** its first kernel (`Cls.get_veff = f`) keeps its cached mask. Instance attributes are always seen.
- **D3 — scope beyond the plan's file list.**
  - `crates/pyscf-py/src/pbc/scf.rs` was created for the harness (allowed by the orchestrator).
  - `pbc/mod.rs` got 3 additive lines: `pub mod kbridge; pub mod scf;` and the `"scf" => scf::register`. The concurrent 20-14 agent's lines landed alongside and were not touched.
  - Python `get_occ` overrides return occupations only (upstream's signature). The cycle's Fermi level is then derived as the highest occupied energy per channel, which is what the Rust aufbau default computes. For a smearing driver it differs from the smeared μ, and only the reported `fermi` is affected.
  - `KInitGuess::Chkfile(path)` passes the key `"chkfile"` without the path.
- **D4 — full suite: nothing attributable to 20-11, but there is no prior full-suite baseline.** No earlier Phase-20 plan ran the whole directory. Breakdown:
  - `test_panic_to_exception.py` ×2 — known (20-08 D3).
  - `test_pbc_identity_gate.py` ×10 — scf ×4, dft ×4, mp, cc, as expected until 20-12/13/15. `symm` now passes because of 20-14.
  - `test_pbc_cell.py::test_make_kpts_with_symmetry_points_at_pbc_symm` — restated in flight by the concurrent 20-14 agent, whose `gto.rs`/`symm.rs` edits are newer than this `.so`.
  - `test_scf_cross_dispatch.py` ×3 — `DID NOT RAISE PyscfRsError`, the same subclass defect as 20-08 D3.
  - `test_scf_ghf.py` — `GHF` has no `run`.
  - `test_scf_rhf_ccpvdz.py` ×3 and `test_scf_rhf_h2o.py` — `M()` rejects `verbose`, and SCF did not converge after 1 cycle.
  - 9 errors in `test_scf_{analyze,chkfile,df,diis,rhf_benzene,uhf,xplat_uhartree}.py` — overlay imports: `pyscf.gto.moleintor`, `ATM_SLOTS`, `pyscf.dft.radi`, `pyscf.cc.rccsd`.

  None of these tests imports `pyscf._native.pbc.scf` or reaches `kbridge.rs`/`caches.rs`'s new code. The only shared runtime path is `pbc::register` at import, and that succeeds.
- **D5 — CubeCL manual (AGENTS.md §3) not consulted.** This plan writes no compute kernel. It is PyO3 glue over existing Rust.
