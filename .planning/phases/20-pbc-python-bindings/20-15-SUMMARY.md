# 20-15 SUMMARY — `pbc.mp` / `pbc.cc` / `pbc.ci` / `pbc.ao2mo`

**Shipped:** 2026-09-14. All five tasks are done. Every plan verification command is green. Deviations are listed below.

## Task 1 — the tests

`python/pyscf/tests/test_pbc_post_scf.py` has 28 cases and runs in 48.3 s (`target/py-20-15-post-scf4.log`).

**Not strictly RED-first.** The file was written after the first `.so` with the bindings had been built. The first two runs failed 3 distinct cases, and each was a real finding: a wrong per-k `Tr(γ)` assumption in the test (the electron count is conserved only over the zone, as 17-VERIFICATION records), D6 (`ao2mo` pair scaling) and O1 (fcc ksymm SCF at mesh 15). D5 was found while the full-mesh stagger case was being timed.

**Fixture.** He-fcc, all-electron, Bohr, `6-31g`, so `nocc = nvir = 1` per k-point. This is 16 measurements §4's all-electron control: `[1,1,2]`, mesh `[15]*3`, `exxdiv=None`. `sto-3g` cannot host CC (`nvir = 0`).

Three kinds of assertion are kept apart:

1. **Binding == Rust, bitwise.** Two private functions, `_rust_reference_kmp2` and `_rust_reference_krccsd`, run `Krhf::kernel` followed by `Kmp2::kernel` or `Krccsd::ao2mo` + `kernel_with` + `kccsd_t_rhf::kernel`. They use fresh Rust builders of the same kind (FFTDF at `with_df.mesh`, or GDF), with no Python object in between. This is `krccsd_smoke.rs`'s sequence. The binding path (bridged `KRHF` → `KMP2`/`KRCCSD`) must reproduce `e_hf`, `e_corr` and `e_t` to the bit.
2. **vs upstream.** One subprocess on the vendored 2.12.1 tree (`PYTHONPATH=REPO`, version asserted), **FFTDF route only**, at the 16 §1 gates.
3. **Port-internal identities:** G3, G4, G5, ksymm vs full BZ.

Every CC test id names its route (`fftdf` or `gdf`), and no assertion compares across routes. The **GDF route is bitwise-vs-Rust only**: its `get_eri`/`ao2mo_7d` oracle gates currently fail at 2.33e-10 / 8.2e-11 against 1e-11, a bisect is pending, and this plan does not paper over that.

## Task 2 — `pbc.mp` (`crates/pyscf-py/src/pbc/mp.rs`)

| Python | Rust |
|---|---|
| `KMP2(mf, frozen=None)` / `KRMP2` | `Kmp2`. A `KsymAdaptedKRHF` mean field (`KRHF(cell, kpts=KPoints)`) is unfolded with `unfold_kscf_result` and runs `KsymAdaptedKmp2`. |
| `KsymAdaptedKMP2(KMP2)` | `KsymAdaptedKmp2`. Requires a ksymm mean field. |
| `KUMP2(mf, frozen=None)` | `Kump2`. `get_nocc`/`get_nmo`/`get_frozen_mask` work; `kernel` raises `NotImplementedError` (upstream `kump2.py:38`). |
| `KMP2_stagger(mf, frozen=None, flag_submesh=False)` | `Kmp2Stagger::new` / `new_full_mesh` |

- **KMP2 surface.** `kernel(mo_energy=None, mo_coeff=None, with_t2=None)` returns `(e_corr, t2)`. The per-route default for `with_t2` is upstream's: `True` for full BZ, `False` for ksymm. `with_t2=True` on the ksymm route calls `kernel_with_t2`.
- **Results:** `e_hf`, `e_corr`, `e_corr_ss`, `e_corr_os`, `e_tot`, and `t2` as a complex `(nk,nk,nk,no,no,nv,nv)` array. Also `nocc`, `nmo`, `get_frozen_mask`, `with_df_ints`, `max_memory`, `run`.
- **RDMs.** `make_rdm1(t2=None, kind)` returns per-k arrays; on the ksymm route it calls `make_t2_for_rdm1` and returns IBZ-sized blocks. `make_rdm2(t2=None, kind)` returns a stacked 7-D array, or a flat list of blocks when padding makes the blocks ragged. The ksymm route raises, as upstream does.
- **Frozen specs** (`FrozenK`): `None`, an `int`, a list of `int`, or per-k lists. KUMP2 (`FrozenU`) additionally takes a 2-sequence of per-spin specs. `'auto'` raises.
- **Mean-field access.** The Rust `KScfResult` is reached through a new additive accessor, `PyKscf::post_scf_input` → `PostScfInput { kind, result, with_df, kpoints, exxdiv }`, appended to `pbc/scf.rs`. The DF object goes through 20-10's `extract_df`. Nothing re-runs SCF. The GIL is released around every kernel.

## Task 3 — `pbc.cc` (`crates/pyscf-py/src/pbc/cc.rs`)

**Class layout.** `_KCCSD` is the native base, with `KRCCSD` (alias `KCCSD`), `KUCCSD`, `KGCCSD` and `KsymAdaptedRCCSD` as `#[new]`-only subclasses (20-12's pattern). Each requires exactly its mean field: `KRHF`, `KUHF`, `KGHF`, or ksymm `KRHF`. `KRCCSD(ksymm_mf)` raises a `TypeError` that names `KsymAdaptedRCCSD`.

**Kernel and stored state.** `kernel(t1=None, t2=None, eris=None, mbpt2=False)` returns `(e_corr, t1, t2)`; `mbpt2=True` is KRCCSD `init_amps` only. The integrals are kept with the amplitudes (`KEris` / `KuEris` / `KgEris` + `PaddedMos` + `KptsHelper`), so `ccsd_t()` and EOM reuse them.

**Options.** The Python attributes cover all of `KrccsdOpts`, plus `keep_exxdiv` (passed into `KErisOpts` together with the mean field's `exxdiv`) and `max_space`.

**Results.** `e_hf`, `e_corr`, `emp2`, `e_tot`, `converged`, `cycles`. `t1`/`t2` are tuples for KUCCSD and unfolded full-BZ arrays for ksymm.

**(T).** `ccsd_t()` is the blocked `kccsd_t_rhf` for KRCCSD and the spin-orbital `kccsd_t` for KGCCSD. `_ccsd_t_slow()` is `kccsd_t_rhf_slow`. KUCCSD and ksymm raise.

**EOM.** `ipccsd`/`eaccsd(nroots, left, koopmans, guess, partition, eris, kptlist)` return `(e (nkshift,nroots), v list)`.
- There are native EOM classes, `EOMIP`, `EOMEA`, `EOMEESinglet` (KRCCSD only) and `EOMEE` (KGCCSD only). Their attributes are `partition`, `conv_tol`, `max_cycle`, `max_space`, `e`, `v` and `converged`.
- Behind them sit `EomOpts`, `EomRoots` and `Excitation`, through `eom_kccsd_{rhf,uhf,ghf}::eom_kernel`.
- **`partition='mp'` and `'full'` RAISE `NotImplementedError`** with the Rust upstream-parity refusal (`eom_kccsd_ghf.rs:2385` / `eom_kccsd_uhf.rs:2690`), checked first, as upstream does. Anything else raises `ValueError`.

## Task 4 — `pbc.ci` and `pbc.ao2mo`

- **`pbc.ci`.** `KCIS(mf)` (alias `CIS`) runs `kernel(nroots, eris, kptlist)` → `kernel_at_kshift` with `KcisOpts` (`max_space`, `max_cycle`, `conv_tol`, `davidson`, `build_full_H`) over `Krccsd::ao2mo`'s `KEris`. Upstream's `_CIS_ERIS` is the same Fock/Madelung build. `RCISD`/`CISD`/`UCISD`/`GCISD` raise `NotImplementedError`: **`pbc/ci/cisd.py` is deferred by design** because there is no molecular CI crate (`pyscf-pbc-ci/src/lib.rs:3-22`).
- **`pbc.ao2mo`** is a new 11th nested child (`PBC_CHILDREN` 10 → 11, additive). It holds the **seven** `eris.rs` wrappers (the plan says nine; the crate has seven): `general`, `get_mo_eri`, `get_mo_pairs_G`, `get_mo_pairs_invG`, `assemble_eri`, `get_ao_pairs_G`, `get_ao_eri`.
- **Overlays** (identity re-exports plus lazy upstream-submodule fallthrough): `python/pyscf/pbc/{mp,cc,ci,ao2mo}/__init__.py`. The gamma-point `RMP2`/`UMP2`/`GMP2` and `RCCSD`/`UCCSD`/`GCCSD` raise `NotImplementedError` instead of handing a native mean field to upstream Python.

## Task 5 — gates (measured 2026-09-14, `target/py-20-15-post-scf3.log`)

| gate | fixture / route | tolerance | measured |
|---|---|---|---|
| KMP2 binding == `Kmp2::kernel` | He `[1,1,2]` FFTDF / GDF | bitwise | **bit-identical** (`-0.018005430877292998` / `-0.01698936919891853`) |
| KRCCSD + (T) binding == Rust, and repeat determinism | FFTDF / GDF | bitwise | **bit-identical** `e_hf`, `e_corr`, `e_t`, `t1`, `t2` |
| KMP2 vs upstream | FFTDF | 2e-6 | **1.378e-9** (mean field 9.12e-12) |
| KMP2 `frozen=[4]` vs upstream; per-k spec == uniform | cc-pvdz FFTDF | 2e-6 / bitwise | **1.576e-10** / bit-identical |
| KMP2 rdm1+rdm2 contraction == `e_tot` (`test_dm.py`) | FFTDF | 1e-8 | **1.158e-9** |
| G1 KRCCSD vs upstream | FFTDF | 1e-7 | **7.186e-10** |
| \|t1\|, \|t2\| element-wise per (ki,kj,ka) vs upstream | FFTDF | 1e-6 | pass (`t2` spread checked non-uniform) |
| G3 KGCCSD / KUCCSD vs KRCCSD | FFTDF | 1e-8 | **4.978e-12 / 3.386e-11** |
| G4 (T) fast vs slow (rel.); (T) vs upstream | FFTDF | 1e-13 / 1e-7 | **0e0** / 1.922e-12 |
| G5 spin-orbital (T) vs RHF (T) | FFTDF | 1e-9 | **7.485e-13** |
| G6 EOM-IP / EA vs upstream | FFTDF | 1e-5 | **9.372e-10 / 8.000e-10** |
| G7 KCIS vs upstream (Davidson; dense within 1e-6) | FFTDF | 1e-5 | **8.799e-9** |
| KMP2_stagger submesh / full mesh vs upstream | He `[2,2,2]`, `cell.mesh` 15 | 2e-6 | **4.332e-7 / 4.158e-7** |
| ksymm KMP2 vs full BZ | He simple-cubic `6-31g` `[2,2,2]`, FFTDF, mesh 15 | 5e-10 (`kmp2_ksymm.rs` `E_CORR_TOL`) | **2.297e-11** (SCF 1.8e-15); `KsymAdaptedKMP2` == `KMP2(ksymm)` bitwise |
| ksymm KRCCSD vs full BZ (two SCFs) | same | 1e-8 | **7.807e-11**, 75 `ao2mo` transforms |
| ao2mo wrappers vs upstream `eris.py` | He `sto-3g` Γ mesh 9 | 1e-10 | ≤ **1.211e-11**; `assemble_eri(G, invG)` == `get_mo_eri` 4e-14 |
| `partition='mp'`/`'full'` raise | KRCCSD/KUCCSD/KGCCSD, each via `ipccsd`, `eaccsd`, `EOMEA.partition` | — | 6 parametrized cases pass |

G2 (GDF vs upstream) is deliberately not asserted (see Task 1). G8–G11 are not binding-level gates; they stay in the Rust suites.

## Verification

| command | result |
|---|---|
| `maturin develop --release --skip-install` (EXECUTION-NOTES §2 env) | exit 0: `Finished` 16.57 s, then 17.18 s (`target/py-20-15-build{1,2}.log`) |
| `.venv/bin/pytest python/pyscf/tests/test_pbc_post_scf.py -q -p no:cacheprovider` | **28 passed in 48.29 s**, exit 0 |
| `.venv/bin/pytest python/pyscf/tests/test_pbc_identity_gate.py` | **18 passed** (mp/cc pass; dft now also passes with 20-13's work) |
| `check-catch-unwind` / `check-forbid-lazy-static` / `check-dependency-wall` / `check-orphan-modules` | exit 0 (849 files) / 0 / 0 (both PASS) / 0 (449 files) |
| `cargo clippy -p pyscf-py` (dev profile) | 0 warnings in `pbc/{mp,cc,ci,ao2mo,scf,mod}.rs` |
| `rustfmt --edition 2024 --check` on the touched files | exit 0 |
| full `.venv/bin/pytest python/pyscf/tests -q -p no:cacheprovider` (`.so` 16:58:45, build 2) | **10 failed, 323 passed, 4 skipped, 3 xfailed, 9 errors in 283.08 s**, exit 1 (`target/py-20-15-fullsuite.log`) — every failure/error is a known molecular one (see below) |

## Deviations

- **D1 — the `KPoints` dispatch is in the `KMP2` constructor.** Same reason as 20-12 D1: the identity gate needs `pyscf.pbc.mp.KMP2` to be the native class. `type(KMP2(ksymm_mf))` is `KMP2`. Upstream `KRCCSD` does not dispatch, so `KRCCSD(ksymm_mf)` raises and names `KsymAdaptedRCCSD`.
- **D2 — "bitwise vs Rust" uses the private `_rust_reference_*` functions**, not a value printed by a Rust test. A Rust test binary would rebuild the libxc tree (memory: gate-target-dir-lto-spelling).
- **D3 — upstream-parity gaps:**
  - CC `frozen=`, KCIS `frozen=`, and full-mesh stagger with `frozen` raise. The Rust drivers pad with `FrozenK::default()`.
  - `mo_coeff=`/`mo_occ=`/`eris=`/`guess=`/`imds=`/`kernel(t1=, t2=)` raise. `ccsd_t(t1, t2)` is accepted.
  - No `mf.to_rhf()`-style conversion.
  - CC/EOM option defaults are the Rust ones: `conv_tol` 1e-9 and `conv_tol_normt` 1e-7, not upstream's 1e-7 / 1e-5.
  - `KCIS.kernel` returns `(e, None)`; there are no eigenvectors in Rust.
  - Not bound: `KUCCSD` (T), EOM on ksymm, EOM-star / `t3p2` / left-EOM helpers beyond the `left` flag, `EOMEETriplet`/`EOMEESpinFlip`, the KUCCSD rdm, and the Γ-point `ccsd.rs` shim.
  - KS mean fields from `pbc.dft` are not accepted: `mean_field` casts to `PyKscf` only.
  - `ao2mo` results are always complex and unpacked.
- **D4 — `pbc/mod.rs` and `pbc/scf.rs`.** `mod.rs` got the four `pub mod`s, the `mp`/`cc`/`ci`/`ao2mo` register arms and the 11th child. `scf.rs` got `PostScfInput` + `post_scf_input` appended. Both edits are additive; no existing line of 20-12 changed.
- **D5 — Rust gap: `Kmp2Stagger::integral_df` reuses the mean field's FFTDF.** It does this whenever the k-points match (`kmp2_stagger.rs`). Upstream instead rebuilds `FFTDF(cell, kpts)` at `cell.mesh` (`kmp2_stagger.py:74`).
  - With `with_df.mesh=[15]*3` over a default-mesh (`[99]^3`) cell, submesh `e_corr` is off by **1.136e-3** (-0.03706278503825 vs -0.03592704457336).
  - With `cell.mesh` pinned, the two agree to 4.3e-7.
  - The test pins `cell.mesh`. The fix belongs to `pyscf-pbc-mp`, which this plan may not edit.
- **D6 — Rust convention gap in `pyscf-pbc-ao2mo`.** `get_mo_pairs_g`/`get_mo_pairs_invg`/`get_ao_pairs_g` return `FFT·Ω/N` (`fft_ao_pairs_g`'s `scale`), while upstream returns the bare `tools.fft`. Both `assemble_eri`s apply `Ω/N²`, so composing the Rust functions was off by `Ω/N` (0.00145 vs 0.37096). The binding multiplies the pair arrays by `N/Ω`, which reproduces upstream to 1.2e-11. This is a scaling, not bit-preserving. Documented in `pbc/ao2mo.rs`.
- **D7 — CubeCL manual (AGENTS.md §3) not consulted.** No compute kernel was written, only PyO3 glue.

## Observations (not defects of this plan; for 20-18 / Phase 17)

- **O1 — He-fcc `6-31g` `[2,2,2]`, ksymm vs full-BZ `KRHF`.**
  - At `mesh = [15]*3` the two differ by **1.354e-6**, with `ops_outside_kmesh_subgroup` empty and `use_ao_symmetry` on or off.
  - At the default `[99]^3` mesh they agree to 3.6e-14, but ksymm KMP2 is still **6.84e-9** from full BZ.
  - On a simple-cubic cell at mesh 15 (the fixture used above) KMP2 agrees to 2.3e-11.
  - Not investigated further.

## Full suite

`10 failed, 323 passed, 4 skipped, 3 xfailed, 9 errors`. **Zero PBC failures.** The 10 failures are `test_panic_to_exception` ×2, `test_scf_cross_dispatch` ×3, `test_scf_ghf`, `test_scf_rhf_ccpvdz` ×3 and `test_scf_rhf_h2o`. The 9 errors are the `moleintor` overlay-import errors in `test_scf_{analyze ×3, chkfile, df, diis, rhf_benzene, uhf, xplat_uhartree}`. This is exactly the set that `measurements/python-suite-triage.md` attributes to pre-existing causes. Against 20-12's run (16 failed / 263 passed), the 6 fewer failures are the identity-gate dft/mp/cc cases, and the +60 passes include `test_pbc_post_scf`'s 28 and 20-13's `test_pbc_dft` (not itemised).
