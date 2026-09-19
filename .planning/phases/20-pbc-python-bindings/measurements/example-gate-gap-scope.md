# Example-script gate: gap scope for `20-k_points_scf.py` and `22-k_points_mp2.py`

**Written:** 2026-09-14. This is a read-only scoping pass for the 20-18 Task 2 gate.
- No Rust or Python source was edited. No `.so` was rebuilt, and nothing was staged.
- The only file written is this one. Raw logs are in `target/` (gitignored): `target/scope-ex22/upstream.log`.
- Probe scripts are in the session scratchpad (`probe20.py`, `probe22.py`, `probe_other.py`, `nat_jk.py`). They are not kept in the repo.

**How the probes were run.**
- Native: `PYTHONPATH=$REPO/python .venv/bin/python <script in scratchpad>`. Every probe printed `pyscf.__file__ == python/pyscf/__init__.py`.
- A `-c` one-liner run from the repo root silently imports the vendored tree (20-17 D1). One such run was discarded and redone from the scratchpad cwd.
- Upstream: `PYTHONPATH=$REPO .venv/bin/python` with `assert pyscf.__version__ == '2.12.1'`.
- Host: 16 cores. The `.so` is dated 2026-09-14 16:58.

**Effort classes.**

| class | meaning |
|---|---|
| **S** | Bind or shim existing Rust; under 1 day |
| **M** | Small Rust port plus binding |
| **L** | New periodic capability |
| **0** | Native today |

---

## 0. Headline findings

1. **Upstream 2.12.1 cannot run `22-k_points_mp2.py` unmodified either.**
   - Measured: `exit=1` after **554 s**, at line 62 `mp.RMP2(mf).run()`.
   - The error is `NotImplementedError`, raised by `pyscf/pbc/mp/mp2.py:23-24` (`if abs(mf.kpt).max() > 1e-9: raise NotImplementedError`). The script's `kpt` is `get_abs_kpts([.25,.25,.25])`, which is not Γ.
   - `mp.UMP2` has the same guard (`mp2.py:36-37`). Everything from line 62 on is unreachable upstream, so there is no oracle for it.
   - Native today stops at the same line with the same exception class, `NotImplementedError`, from the overlay refusal `python/pyscf/pbc/mp/__init__.py:50`.
   - Upstream numbers before the crash (the floors for any restated gate):
     - `KMP2 e_tot` at 2×2×2 = **-11.02605159764625**, with `E_corr` -0.0951932619095835 and SCF -10.9308583357367.
     - One k-point: `KMP2 e_tot` = **-11.01065485358922**, with SCF -10.9379203952462.
   - Upstream stage times: KRHF 2×2×2 126 s, KMP2 2×2×2 408 s, 1-k KRHF about 15 s.
2. **Neither upstream nor native can finish `20-k_points_scf.py` in reasonable time on this box.**
   - The cost is K builds: 64 k-points × 8 atoms × a 65³ mesh.
   - Micro-measured JK calls on the example-20 cell at 2 k-points and the default mesh 65³:

     | build | seconds per JK call |
     |---|---:|
     | upstream, mo-tagged dm | 13.0 |
     | upstream, plain dm | 20.3 |
     | native (`with_df.get_jk`) | 95–102 |

   - K scales as nk². At 64 k-points that is ×1024, or about 3.7 h per upstream JK call and about 27 h per native one.
   - The script needs about 10 SCF cycles for KRHF, the same again for KRKS-M06 (M06 is a 27 % hybrid, so it also builds K), plus Newton, whose Hessian-vector products each build a JK response. The estimate is **≥ 4–6 days upstream** and **weeks native**.
   - This holds even after every binding gap is closed.
3. **Example 20 has three real gaps, two of them L.**
   - `.newton()`: the Rust driver is a dense, real-only, test-model Augmented Hessian (AH). No k-point model exists.
   - Periodic meta-GGA: there is no τ path in either the periodic or the molecular NumInt.
   - `xc='m06,m06'` fails at **parse** time (unknown token `M06`), not only at the meta-GGA refusal. The parse is an S fix.

---

## 1. `examples/pbc/20-k_points_scf.py` — per construct

The native columns were probed on the example's own 8-atom cell with `mesh=[11]*3` and `kpts=make_kpts([1,1,2])`; the timings in brackets come from that probe.

| line | construct | (a) Python binding today | (b) Rust | (c) effort |
|---|---|---|---|---|
| 13-26 | `gto.M(a=, atom=, basis='gth-szv', pseudo='gth-pade', verbose=4)` | **native** `pyscf._native.pbc.gto.Cell`, nao 32, default mesh [65,65,65] (upstream: same 65³) | exists (`pyscf-pbc-gto`) | 0 |
| 29 | `cell.make_kpts([4,4,4])` | **native**, (64,3) | exists | 0 |
| 34-35 | `scf.KRHF(cell, kpts).kernel()` | **native**; kernel OK (-45.0508, 103 s at mesh 11 / 2 k) | exists `pyscf-pbc-scf/src/krhf.rs` | 0, but see §3 wall time |
| 37 | `dft.KRKS(cell, kpts)` | **native** | exists `pyscf-pbc-dft/src/krks.rs` | 0 |
| 39 | `kmf.grids = dft.gen_grid.BeckeGrids(cell)` | **native** shim (20-17). PBE on Becke ran OK (161 s at mesh 11) | exists `pyscf-pbc-dft/src/gen_grid.rs:36-285`; binding `pyscf-py/src/pbc/dft.rs:365-402,1235-1249` | 0 |
| 40 | `kmf.xc = 'm06,m06'` (assignment) | accepted | — | 0 |
| 41 | KRKS `m06,m06` `.kernel()` | **FAIL after 126 s**: `unknown XC functional token 'M06' (XC string: 'M06,M06')` | parser: `pyscf-dft/src/parser/libxc.rs:389-437` tries only the inline `XC_CODES` for the family-prefix search, then `libxc_rs::lookup_by_name("M06")`. libxc_rs **does** register `XC_HYB_MGGA_X_M06` (449) and `XC_MGGA_C_M06` (235) (`libxc_rs/crates/libxc-core/src/registry/by_name.rs:423,519`) | **S**: run the prefix search against the libxc_rs registry |
| 41 | same, after parsing | would hit `pyscf-pbc-dft/src/xc.rs:103-112` (`Family::Mgga` refused) | **absent**: `XcType` is `Lda`/`Gga` only (`xc.rs:60-90`, `RhoEff.nvar` 1 or 4); molecular NumInt also refuses MGGA (`pyscf-dft/src/numint.rs:899`) and UKS MGGA eval is refused (`xc_backend.rs:857`). libxc_rs MGGA eval exists (`xc_backend.rs:1075-1098`). τ needs only deriv-1 AOs, which exist | **L**: periodic MGGA NumInt (τ build, `vtau` Fock contraction, uniform and Becke grids, KRKS plus hybrid-K path, parity gate). Risk: libxc_rs vendors a pre-7.0.0 master; M06 parity against PySCF 7.0.0 is unmeasured |
| 49 | `scf.KRHF(cell, kpts).newton()` | **FAIL** `AttributeError: 'pyscf._native.pbc.scf.KRHF' object has no attribute 'newton'`; `scf.newton(mf)` falls through to upstream and raises `ImportError: cannot import name 'chkfile' from 'pyscf.scf'` | **partial (unusable)**: `pyscf-pbc-scf/src/newton_ah.rs` (246 l, committed in 0f8b58d; the working-tree diff is rustfmt only). `ah_step` (`:73-145`) **materializes the full Hessian column by column** (dim = Σₖ nocc·nvir = 64·16·16 = 16 384 hops per step, each a JK build); `f64` only (KRHF rotations are complex); `NewtonModel` (`:203-214`) is implemented only by the test `TwoLevel` model; no `gen_g_hop` for KRHF, no CIAH Davidson, no complex `expmat`. The response seam `KFvind` is `dyn Fn(usize, &[f64])`, real only (`cphf.rs:49`) | **L**: complex k-point `gen_g_hop` over the FFTDF/GDF JK response, CIAH Davidson micro-iterations, complex `expmat`/`rotate_mo`, the `_SecondOrderKRHF` wrapper class (`.newton()`, `._scf`, `.kernel`), and the identity-gate decision for a new class |
| 50 | `mf.kernel()` (Newton) | — | as above | L (counted once) |
| 60 | `scf.KRHF(cell, kpts, exxdiv=None)` | **native** | exists | 0 |
| 60 | `.density_fit()` | **native** (returns native KRHF over GDF) | exists | 0 |
| 60,63,68 | `.density_fit().newton()` (constructed only, never run) | FAIL (no `newton`) | as line 49 | S once L lands (object construction only) |
| 64 | `mf_opt2.exxdiv = None` | setter **native** on KRHF | exists | 0 |
| 65 | `mf_opt2._scf.exxdiv = None` | FAIL: no `_scf` (it belongs to the Newton wrapper) | absent | part of the Newton L |

**`which_impl` for example 20:**
- `native`: `gto.M`, `scf.KRHF`, `dft.KRKS`, `dft.gen_grid.BeckeGrids`.
- `upstream`: `scf.newton_ah`.

**Effort total for example 20:** 1 S (XC parser) + 2 L (periodic MGGA NumInt; periodic Newton-AH). The run is then still infeasible (§3).

---

## 2. `examples/pbc/22-k_points_mp2.py` — per construct

The native probe used the example's own cell with `mesh=[11]*3` and `kpts=[1,1,2]`. Upstream ran the whole script unmodified at the default mesh 47³ and 2×2×2.

| line | construct | (a) Python binding today | (b) Rust | upstream 2.12.1 | (c) effort |
|---|---|---|---|---|---|
| 11-24 | `gto.Cell()` attribute build, `cell.unit='B'`, `a` as string, `cell.build()` | **native** (nao 8) | exists | ok | 0 |
| 29 | `cell.make_kpts([2,2,2])` | **native** | exists | ok | 0 |
| 30-32 | `scf.KRHF(cell)`; `kmf.kpts = kpts`; `kmf.kernel()` | **native** (7.5 s at mesh 11 / 2 k) | exists | 126 s | 0, but native JK is 115–121 s per call at 2×2×2 / 47³ vs upstream 12–24 s (§3) |
| 34-36 | `mp.KMP2(kmf).kernel()`, `.e_tot` | **native** | exists `pyscf-pbc-mp/src/kmp2.rs`; binding `pyscf-py/src/pbc/mp.rs:461-511` | 408 s, e_tot -11.02605159764625 | 0 |
| 41-43 | `cell.get_abs_kpts([.25]*3)`; `kmf.kpts = (3,)`; kernel | **native** | exists | ok | 0 |
| 45-47 | `mp.KMP2` at 1 k-point | **native** | exists | e_tot -11.01065485358922 | 0 |
| 58-59 | `scf.RHF(cell, kpt=kpt).kernel()` | constructs and runs, but returns **native `KRHF` at nk=1** (per-k lists; `mo_coeff` is `list[(8,8)]`, no `.kpt`) | `pyscf-pbc-scf/src/gamma.rs:35` `rhf_at` → `Krhf` | ok (single-k `pbc.scf.hf.RHF`, 2-D complex arrays) | **M** for the upstream shape: a single-k SCF class with 2-D `mo_coeff`, `get_hcore`, `.kpt`. This is a 20-12 design decision (overlay docstring: "results are per-k lists of length 1") |
| **62** | `mp.RMP2(mf).run()` | **FAIL** `NotImplementedError` (overlay `pbc/mp/__init__.py:50`, `which_impl='refused'`) | molecular `pyscf-mp2` is real-only (`mp2.rs:286`, `Mp2Reference`); complex nk=1 is equivalent to `KMP2`, which exists | **FAIL `NotImplementedError`** (`pbc/mp/mp2.py:24`, k ≠ 0). **The script ends here upstream (measured, 554 s, exit 1)** | no oracle; M if pursued |
| 64-65 | `make_rdm1()/make_rdm2()` (gamma RMP2) | unreachable | nk=1 equivalent exists `pyscf-pbc-mp/src/krdm.rs:109,170` | unreachable | M (with RMP2) |
| 66 | `mf.mo_coeff.shape[1]` | would FAIL (`list` has no `.shape`) | — | unreachable | part of the single-k-class M |
| 67 | `mf.with_df.ao2mo(mf.mo_coeff, kpts=kpt)` | **FAIL** `ValueError: mo_coeffs must be one block or four` (for the list), then `kpts must be four k-points, shape (4, 3)` for a single `(3,)` k-point | Rust `ao2mo` exists; the binding `quad()` does not broadcast one k-point (`pyscf-py/src/pbc/df.rs:1082-1106`); upstream broadcasts it | unreachable | **S** |
| 68 | `mf.get_hcore()` 2-D | returns `list[(8,8)]` | exists | unreachable | part of M |
| 70 | `mf.energy_nuc()` | **native** | exists | unreachable | 0 |
| 73 | `scf.addons.convert_to_uhf(mf)` | **FAIL** `ImportError: cannot import name 'addons' from 'pyscf.scf'` (upstream fallthrough under the molecular overlay) | only DM converters exist, `pyscf-pbc-scf/src/addons.rs:43-104` (`rhf_dm_to_uhf`, `uhf_dm_to_ghf`); no MO/object conversion | unreachable | **M** (addons shim + object conversion; the GHF side needs an `orbspin`-tagged `mo_coeff`) |
| 74-90 | `mp.UMP2(mf).run()`, `make_rdm1/2`, `ao2mo` (4-tuple), `get_hcore` | `UMP2` **refused** | molecular `ump2.rs:334` is real-only; `KUMP2` kernel refused as upstream does (`kump2.rs:1-2`) | unreachable; would also raise (k ≠ 0 guard `mp2.py:37`) | no oracle; M |
| 93 | `scf.addons.convert_to_ghf(mf)` | FAIL (as line 73) | absent | unreachable | M (with line 73) |
| 94-96 | `mp.GMP2(mf).run()`, `make_rdm1/2` | **refused** | **absent** (no `gmp2` in `pyscf-mp2` or `pyscf-pbc-mp`) | unreachable (upstream GMP2 has no k guard) | M |
| 97 | `cell.nao_nr()` | **native** | exists | — | 0 |
| 101 | `mf.mo_coeff.orbspin` | absent | — (`lib.tag_array` semantics) | unreachable | S (with line 93) |

**`which_impl` for example 22:**
- `native`: `gto.Cell`, `scf.KRHF`, `mp.KMP2`, `scf.RHF`, `df.FFTDF`.
- `refused`: `mp.RMP2`, `mp.UMP2`, `mp.GMP2`.
- `upstream`: `scf.addons`, `scf.addons.convert_to_uhf`.

**Effort total for example 22:**
- Lines 1–58, everything upstream can run: **0**. It is all native today; only the default-mesh wall time is unmeasured.
- Lines 62–110: 1 S (single-k-point `ao2mo`) and 4 M (single-k SCF shape; complex gamma RMP2 plus RDMs; complex UMP2; addons conversion plus GMP2 with `orbspin`).
- **None of that has an upstream oracle for this script**, because upstream raises at line 62.

---

## 3. Wall time

| run | measured / estimated | basis |
|---|---|---|
| upstream `22-k_points_mp2.py` | **554 s, exit 1 at line 62** (measured) | KRHF 126 s (7 JK builds of 12–24 s each, 103 823 PWs), KMP2 408 s, 1-k KRHF about 15 s, gamma RHF about 5 s |
| native `22-k_points_mp2.py` through line 58 | **estimated 20–60 min** | native JK on this cell at 2×2×2 / 47³ = 115.5 s cold, 120.6 s warm (measured) × about 8 builds ≈ 16 min, plus KMP2 (upstream 408 s; native unmeasured at this size) |
| upstream `20-k_points_scf.py` | **≥ 4–6 days** (estimated, not run) | upstream JK on the 8-atom cell at 2 k / 65³ = 13.0 s mo-tagged (measured). K ∝ nk² gives ×1024 ≈ 3.7 h per JK at 64 k. KRHF ≈ 10 cycles ≈ 1.5 days; KRKS-M06 (27 % HF, so K each cycle, plus MGGA on about 118 k Becke points × 64 k) ≈ 1.5+ days; Newton ≈ several JK responses per macro step ≈ 2+ days |
| native `20-k_points_scf.py` | **weeks** (estimated) | native JK on the same cell at 2 k / 65³ = 102 s cold, 95 s warm (measured), 7.3× upstream, so about 27 h per JK at 64 k |

Upstream does **not** complete example 20 in reasonable time on this 16-core box. The script's own comment (lines 31-33) says the default JK builder is slow for KHF.

---

## 4. Other unmodified `examples/pbc/*.py` candidates

These were checked by static reading plus construct/attribute probes only; no candidate kernel was run.

| script | first native blocker(s) | upstream runs it? | class to make it fully native |
|---|---|---|---|
| `22-k_points_mp2_ksymm.py` | **none found.** `gto.M(space_group_symmetry=True, mesh=[24]*3)`, `make_kpts(..., space_group_symmetry, time_reversal_symmetry)`, `KRHF(cell, KPoints)`, `mp.KMP2` (dispatches to `KsymAdaptedKMP2`; `kernel`/`e_tot` bound) all construct natively | expected yes (Si, 2 atoms, 24³, IBZ of 2×2×2): cheap | **0** (predicted; needs one run to confirm) |
| `23-smearing.py` | `pyscf.M` missing from the molecular overlay top level (`AttributeError`); `Cell.KRKS(xc=, kpts=)` method missing; `KRKS.entropy` missing (it exists on the SCF side, `pyscf-py/src/pbc/scf.rs:1349`). `mf.smearing(sigma, method)`, `e_free` and the `sigma` setter are bound | yes (Al, 1 atom, 4×4×4, PBE, no K) | **S** (3 shims) |
| `22-k_points_ccsd.py` | lines through `KRCCSD` at one k-point are native; gamma `cc.RCCSD/UCCSD/GCCSD` are refused; then the same `scf.RHF(kpt=)` shape, single-k `ao2mo` and `scf.addons` gaps as example 22 | upstream `pbc/cc/ccsd.py` has no k ≠ 0 guard; whether upstream completes is unmeasured | **M+** |
| `24-k_points_vs_gamma.py` | `pbccc.RCCSD(gamma supercell mf)` refused; `.ipccsd/.eaccsd` on gamma; `KRCCSD.ecc` attribute missing (`ccsd`/`ipccsd`/`eaccsd` bound) | yes (mesh 24, [1,1,2]) | **M** (gamma CCSD plus EOM shim over the K drivers at nk=1) |
| `22-dft+u.py` | `U_idx=['1 C 2p']` and `['2p','2s']` refused (`pyscf-py/src/pbc/dft.rs:214`: only `'<El> <n><l>'` is ported) | **no**: the second half calls `make_kpts(space_group_symmetry=True)` on a cell built without `space_group_symmetry`, which raises `RuntimeError` at `pyscf/pbc/gto/cell.py:876` (native reproduces it, `gto.rs:925`) | M (AO-label U_idx search); the script is broken upstream |
| `20-k_points_scf_ksymm.py` | `KRHF(..., use_ao_symmetry=False)` `TypeError` (the kwarg is internal, `scf.rs:602`, not exposed to `__new__`); `.to_khf` missing; `.newton()`; `KUHF(cell, KPoints)` refused (no `kuhf_ksymm`); `scf.addons.smearing_` `ImportError` | yes | **L** (Newton, KUHF ksymm) |
| `10-gamma_point_scf.py` | `dft.RKS` `xc='m06,m06'` (parse S, then MGGA L); `scf.RHF(cell).newton()` | yes (gamma, but slow: 8 atoms, 65³ K) | **L** |
| `27-multigrid.py` | `from pyscf.pbc.dft import multigrid` `ImportError: cannot import name 'ATOM_OF' from 'pyscf.gto'` (upstream fallthrough); `KRKS.newton()`; `KRKS.Gradients()` (Phase 18 grad unbound/uncommitted). `UKS(cell).multigrid_numint()` and `KRKS.multigrid_numint()` are native | yes, but 4×4×4 gth-dzvp 8-atom KRKS is expensive | **L** |

---

## 5. Recommendation

**Do not plan gap closure to make the two named scripts run natively in Phase 20. It is not realistic, and for example 22 it is not even well-defined.**

- **Example 22.** The literal gate "runs unmodified" is **unattainable by the oracle itself**: upstream 2.12.1 exits with `NotImplementedError` at line 62 (measured).
  - Closing the 1 S + 4 M items would make pyscf-rs run *further* than upstream, with no upstream number to land on.
  - Everything upstream can execute (lines 1–58) is already native.
- **Example 20.** It needs 2 L capabilities (periodic meta-GGA NumInt; a production complex k-point Newton-AH, where the current `newton_ah.rs` is a dense real test driver) plus 1 S.
  - Even then, the run is days upstream and weeks native, so it could never be a Phase-20 or CI gate.
  - Carry MGGA and Newton-AH as `.planning/carryovers/` entries with these numbers.

**Proposed restated Task-2 gate** (keeps the 20-01 intent: unmodified upstream scripts, identity holding, `which_impl` native, landing on floors):

1. **`examples/pbc/22-k_points_mp2.py` unmodified reaches the same terminal state as upstream 2.12.1.**
   - Lines 1–58 run with every touched name `which_impl == 'native'`.
   - `KMP2 e_tot` lands within floor row 7 (2e-6 per DF route) of -11.02605159764625 (2×2×2) and -11.01065485358922 (one k).
   - Line 62 raises `NotImplementedError`, as upstream does.
   - Measure the native wall time once first (estimated 20–60 min). If it is too slow for CI, gate it as a recorded one-off run.
2. **`examples/pbc/22-k_points_mp2_ksymm.py` unmodified**, fully native. Predicted 0 gaps and cheap; confirm with one run before adopting.
3. Optionally **`examples/pbc/23-smearing.py`** after three S shims: `pyscf.M` → `pbc.gto.M` when `a=` is given; `Cell.KRKS`/`Cell.KRHF` methods; `KRKS.entropy`.
4. Replace `20-k_points_scf.py` with a named, bounded native script that exercises the bound surface: KRHF + KRKS (GGA) + `BeckeGrids` + `density_fit`/`exxdiv` on a small cell. Record example 20's gaps (MGGA, Newton) and cost in carryovers.

**Cheap S items worth folding into a small 20-18 pre-task regardless** (each under 1 day, each unblocks several scripts):
- XC parser family-prefix fallback through the libxc_rs registry (`M06` → 449/235).
- Single-k-point broadcast in the periodic DF `ao2mo`/`get_eri` binding (`df.rs:1082-1106`).
- A `pyscf.pbc.scf.addons` overlay re-exporting native `smearing_`/`project_mo_nr2nr`.
- A `pyscf.pbc.dft.multigrid` shim re-exporting `MultiGridNumInt`/`MultiGridNumInt2`.
- The `KRCCSD.ecc` alias.
- `KRKS.entropy`.

**Effort totals:**

| scope | S | M | L |
|---|---:|---:|---:|
| ex 20 | 1 | 0 | 2 |
| ex 22 (post-crash, no oracle) | 1 | 4 | 0 |
| other 8 candidates (distinct items) | ~6 | ~4 | 3 (Newton shared; KUHF-ksymm; grad binding) |
