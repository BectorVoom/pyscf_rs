# PBC Master Implementation Plan — pyscf_rs v2.0

**Created:** 2026-08-22
**Milestone:** v2.0 — Periodic Boundary Conditions (full `pyscf/pbc/*` parity)
**Upstream surface to port:** 186 Python modules / ~78,000 lines under `pyscf/pbc/` (excluding `test/`)
**Target:** 12 phases (Phase 9 … Phase 20), 19 new workspace crates (20 → 39 members)
**Audience:** an execution agent that follows instructions literally and does NOT infer.

---

## 0. HOW TO EXECUTE THIS PLAN — READ THIS FIRST

You are implementing periodic boundary conditions in a Rust rewrite of PySCF.
Follow these rules **exactly**. Do not improvise. Do not skip steps.

### 0.1 The nine standing rules

**RULE 1 — One plan at a time.**
Work through `§8 Phase Breakdown` in order. Phase 9 plan 01, then 09-02, then 09-03 …
Never start plan `N+1` before plan `N` has a green `cargo test -p <crate>`.

**RULE 2 — Always read the upstream Python first.**
Every task names an upstream file and line range, e.g. `pyscf/pbc/gto/cell.py:539-605`.
Before writing Rust, run:
```bash
sed -n '539,605p' pyscf/pbc/gto/cell.py
```
Port that function **line by line**. Same variable names. Same order of operations.
Do not "improve" the algorithm. Bit-exactness with upstream is the acceptance criterion.

**RULE 3 — Use CodeGraph before grep.**
The repo has a `.codegraph/` index. To find anything, call:
```
codegraph_explore("<symbol or question>")
```
It returns verbatim source + callers + call paths in one call. Do NOT run grep/find loops.
Shell fallback: `codegraph explore "<symbols>"`.

**RULE 4 — Tests live in separate files. ALWAYS.**
`AGENTS.md` forbids `mod tests` at the bottom of a production source file.
Production code → `crates/<crate>/src/<name>.rs`.
Tests → `crates/<crate>/tests/<name>.rs` (integration) or `crates/<crate>/src/<name>_test.rs` (unit, declared with `#[cfg(test)] #[path = "<name>_test.rs"] mod <name>_test;`).

**RULE 5 — cubecl rules.**
Before writing ANY cubecl kernel, read the manual:
`/home/user/Documents/workspace/cubecl_manual/manual/manual/Cubecl/INDEX.md`
Every kernel MUST be generic over the device float: `#[cube(launch_unchecked)] fn k<F: Float>(...)`.
On ANY cubecl build/link/feature error, STOP and read
`/home/user/Documents/workspace/cubecl_manual/manual/cubecl_error_guideline.md`
before touching the code. Blind fixes are a protocol violation.

**RULE 6 — The algebra wall (ALG-06) is absolute.**
Only `pyscf-algebra`, `pyscf-runtime` and `pyscf-kernels` may declare `cubecl-*`
dependencies. Every new `pyscf-pbc-*` crate consumes `pyscf_algebra::*` and
`pyscf_kernels::*` only. `xtask/src/bin/check_dependency_wall.rs` enforces this in CI.
If you need a new device kernel, it goes in `pyscf-kernels`, never in a method crate.

**RULE 7 — Un-gate the forbidden-paths lint ONCE, in plan 09-01.**
`xtask/src/bin/check_forbidden_paths.rs:34` currently bans the string `use pyscf::pbc`.
Plan 09-01 removes exactly the `pbc` needle and leaves `x2c`/`mcscf`/`adc`/`gw`/`eom`/`NAC`/`EPH` in place
**for molecular crates only** — the ban becomes path-scoped: crates matching `pyscf-pbc-*` are exempt
from all of them, because their periodic variants ARE v2.0 scope.

**RULE 8 — Complex numbers never cross the algebra wall as `Complex<f64>`.**
Use the planar `CTensor { re, im }` split representation defined in `§5`.
Interleaved `[re, im, re, im, …]` appears ONLY at the PyO3/NumPy boundary in `pyscf-py`.

**RULE 9 — Every plan ends with a SUMMARY.**
Write `.planning/phases/<NN>-<name>/<NN>-<PP>-SUMMARY.md` recording: what shipped,
the exact `cargo test` command that is green, deviations from the plan, and carry-overs.
Then update `.planning/STATE.md`.

### 0.2 Definition of done for any task

A task is done when **all four** hold:
1. `cargo build --workspace` is clean (no warnings in the new crate).
2. `cargo test -p <crate>` is green.
3. `cargo clippy -p <crate> -- -D warnings` is clean.
4. The numeric acceptance value stated in the task matches to the stated tolerance.

### 0.3 Vocabulary you must not confuse

| Term | Meaning | Shape |
|---|---|---|
| `a` | lattice vectors, row-major, Bohr | `[3][3]` — `a[i]` is lattice vector i |
| `b` | reciprocal vectors, `b = 2π·inv(a).T` | `[3][3]` |
| `vol` | `abs(det(a))` | scalar |
| `Ls` | real-space lattice translation vectors | `[nimgs][3]` |
| `Gv` | reciprocal grid vectors | `[ngrids][3]` |
| `mesh` | FFT grid dims `(nx, ny, nz)`; `ngrids = nx*ny*nz` | `[3]` |
| `kpts` | absolute k-points (1/Bohr) | `[nkpts][3]` |
| `kpts_scaled` | fractional k-points, `kpts_scaled = kpts · a.T / 2π` | `[nkpts][3]` |
| `nao` | AOs per unit cell | scalar |
| `dm_kpts` | density matrices, complex | `[nkpts][nao][nao]` |
| `exxdiv` | divergence treatment at G=0 for exchange: `None` or `"ewald"` | enum |

---

## 1. Scope

**In scope (everything under `pyscf/pbc/`):**

| Subpackage | Files | Lines | Phase |
|---|---:|---:|---|
| `pbc/gto` | 16 | 4,593 | 9, 10 |
| `pbc/tools` | 9 | 3,102 | 9, 20 |
| `pbc/lib` | 7 | 3,447 | 9, 17 |
| `pbc/scf` | 21 | 7,693 | 11, 17, 19 |
| `pbc/dft` | 25 | 9,313 | 12, 17 |
| `pbc/df` | 21 | 13,666 | 11, 13, 14 |
| `pbc/ao2mo` | 2 | 273 | 15 |
| `pbc/mp` | 6 | 2,085 | 15 |
| `pbc/cc` | 19 | 13,675 | 16 |
| `pbc/ci` | 3 | 852 | 16 |
| `pbc/symm` | 8 | 1,767 | 17 |
| `pbc/grad` | 15 | 2,848 | 18 |
| `pbc/geomopt` | 2 | 269 | 18 |
| `pbc/tdscf` + `tddft` | 10 | 1,900 | 19 |
| `pbc/gw` | 7 | 2,504 | 19 |
| `pbc/adc` | 7 | 3,437 | 19 |
| `pbc/x2c` | 3 | 654 | 19 |
| `pbc/eph` | 2 | 181 | 19 |
| `pbc/mpicc` + `mpitools` | 10 | 5,797 | 20 |

Per-file → per-crate → per-phase mapping: see `PBC-DRIVER-INVENTORY.md`.

**Out of scope for v2.0:** nothing under `pyscf/pbc/`. Optional third-party bridges
(`tools/pywannier90.py` needs `wannier90`, `symm/pyscf_spglib.py` needs `spglib`)
ship as **thin optional shims**, not reimplementations — see plan 20-05.

---

## 2. Ground truth — what the Rust workspace already gives you

Read this section before designing anything. **These already exist. Reuse them. Do not re-implement.**

### 2.1 Crates you will build on

| Crate | Gives you | Key entry points |
|---|---|---|
| `pyscf-core` | `Mole` (63 fields incl. `_atm`/`_bas`/`_env`/`basis_set: Arc<BasisSet>`), `Density`, `Energy`, `MOCoefficients`, `canonicalize_signs`, `Scalar`, error types | `pyscf_core::{Mole, Density, Energy, PyscfRsError}` |
| `pyscf-runtime` | `BackendKind`, `select_backend()` from `PYSCF_BACKEND`, `WorkspacePool`, `DType` | `pyscf_runtime::*` |
| `pyscf-algebra` | **the only cubecl gateway**: `gemm_dense`, `gemv_dense`, `axpy_dense`, `scal_dense`, `dot_dense`, `reduce_sum_dense`, `transpose_dense`, `eigh_gen`, `solve_linear`, `cholesky/eigh/qr/svd` (faer host), `oracle_sum/oracle_dot/oracle_einsum`, `dispatch_backend!` macro | `pyscf_algebra::*` |
| `pyscf-kernels` | cubecl `eval_gto` kernels (2,564 lines, s/p/d + deriv1, sph+cart) | `pyscf_kernels::eval_gto::*` |
| `pyscf-gto` | `Mole` build, 207 basis files, **CP2K/GTH basis parser** (`basis/cp2k.rs`), **GTH pseudopotential parser** (`basis/cp2k_pp.rs`), `intor(mol,name)`, `intor_with_auxmol`, `projection::build_combined_basis`, `eval_gto`, `cart2sph_coeff` | `pyscf_gto::*` |
| `pyscf-scf` | `kernel()` SCF driver, `OverrideHooks` trait (11 hooks), DIIS adapter, `init_guess` (5 modes), chkfile, `analyze`, `as_scanner` | `pyscf_scf::*` |
| `pyscf-dft` | `NumInt` (`eval_rho`, `eval_xc`, `nr_rks`, `nr_uks`), `XcSpec` parser, libxc/xcfun backends, `RKS`, `UKS`, VV10 | `pyscf_dft::*` |
| `pyscf-grids` | Becke molecular grids: `Grids`, `gen_atomic_grids`, Lebedev, radial, pruning | `pyscf_grids::*` |
| `pyscf-df` | `auxbasis`, `cholesky_eri`, `df_jk` | `pyscf_df::*` |
| `pyscf-diis` | Pulay C-DIIS (`cdiis.rs`) | `pyscf_diis::*` |
| `pyscf-mp2`, `pyscf-ccsd`, `pyscf-grad`, `pyscf-geomopt`, `pyscf-ao2mo` | molecular analogues you will k-point-ize | — |
| `cintx` (sibling, `../cintx`) | libcint-equivalent integrals over an arbitrary `BasisSet`; `SessionRequest::new(op, rep, basis, shells, opts).query_workspace()?.evaluate()?` evaluates ONE shell tuple | `cintx_core`, `cintx_ops`, `cintx_rs` |

### 2.2 The three facts that make periodic integrals possible today

1. **`cintx` evaluates one shell tuple at a time** against an arbitrary `BasisSet`
   (`crates/pyscf-gto/src/intor.rs:424-446`). You can therefore evaluate
   `(shell i in cell 0 | shell j in image L)` directly.
2. **`pyscf_gto::projection::build_combined_basis`** already concatenates two basis
   sets into one `cintx_core::BasisSet` (used by DF `int3c2e` with an auxmol,
   `intor.rs:987`). This is exactly the "cell + images" construction you need.
3. **`_env` holds atom coordinates**, so an image atom is the same shell with a
   shifted `PTR_COORD` — no new basis parsing.

### 2.3 The one big hole: complex arithmetic

`pyscf-algebra` today is **real-only** (`f32`/`f64` via `DeviceScalar: Scalar + cubecl::Float + Pod`).
Every k-point quantity in PBC is `complex128`. **Phase 9 plan 09-02 fixes this** — see `§5`.

### 2.4 cintx dependency matrix — READ BEFORE PLANNING ANY PHASE

**Status verified against the cintx working tree on 2026-08-22.** Re-verify with:
```bash
cd /home/user/Documents/workspace/cintx
grep -n 'symbol_name: "<sym>' crates/cintx-ops/src/generated/api_manifest.rs
grep -n 'op_name == "<operator>"\|operator_name() == "<operator>"' crates/cintx-cubecl/src/kernels/*.rs
```
A symbol is **usable** only when it has (a) a manifest row for the representation you
need, (b) a dispatch arm in the owning family launcher, and (c) `oracle_covered: true`.
Two of three is not enough.

| Phase | cintx symbols needed | Status | Consequence |
|---|---|---|---|
| **10** — periodic 1e integrals | `int1e_ovlp`, `int1e_kin`, `int1e_nuc` (cart/sph) | ✅ shipped, oracle-covered | none |
| **10** — GTH PP `get_pp_loc_part2` | `int3c2e`, `int3c1e` | ✅ shipped | none |
| **10** — GTH PP `get_pp_loc_part2` | `int3c1e_r2_origk`, `int3c1e_r4_origk`, `int3c1e_r6_origk` | ❌ **declared, `oracle_covered:false`, NO dispatch arm** (`center_3c1e.rs:1469`) | **BLOCKS plan 10-05** |
| **10** — GTH PP `_int_vnl` | `int1e_ovlp` | ✅ shipped | none |
| **10** — GTH PP `_int_vnl` | `int1e_r2_origi`, `int1e_r4_origi` | ❌ **declared, `oracle_covered:false`, NO dispatch arm** | **BLOCKS plan 10-06** |
| **11** — FFTDF J/K | *(none — FFTDF is grid-based)* | — | none |
| **13** — AFTDF | `int3c1e_r{2,4,6}_origk` (via `df/aft.py:335`) | ❌ same as Phase 10 | shares the Phase-10 blocker |
| **13** — `ft_ao` | *(none — `GTO_ft_*` is a cintx-free cubecl kernel, K-15)* | — | none |
| **14** — GDF/MDF/RSDF | `int3c2e`, `int2c2e`, `int3c2e_ip1`, `int3c2e_ip2` | ✅ shipped, oracle-covered | none |
| **18** — periodic gradients | `int1e_ipovlp`, `int1e_ipkin` | ✅ shipped, oracle-covered | none |
| **18** — PP gradient `vpploc_part2_nuc_grad` | `int3c2e_ip1`, `int3c1e_ip1` | ✅ shipped | none |
| **18** — PP gradient `vpploc_part2_nuc_grad` | `int3c1e_ip1_r2_origk`, `_r4_origk`, `_r6_origk` | ❌ **declared, unverified, no dispatch arm** | **BLOCKS plan 18-01** |
| **18** — PP gradient `vppnl_nuc_grad` | `int1e_ipovlp` | ✅ shipped | none |
| **18** — PP gradient `vppnl_nuc_grad` | `int1e_r2_origi_ip2`, `int1e_r4_origi_ip2` | ❌ **declared, unverified, no dispatch arm** | **BLOCKS plan 18-01** |
| **18** — stress tensor | *(none — strain derivatives are finite-differenced `pbc_intor` + analytic `coulG`/weight/AO strain terms)* | — | none |
| **19** — periodic ECP | `int1e_ecp`, `int1e_ecp_ipnuc`, `int1e_ecp_iprinv` (cart/sph) | ✅ shipped | none |
| **19** — `ppnl_velgauge` | `GTO_ft_ovlp`, `GTO_ft_r2_origi`, `GTO_ft_r4_origi` | ⚠️ cubecl-side, extends K-15 | plan 19-08 owns it |

**The single takeaway: the cintx blocker for PBC is NOT gradients.**
Every gradient family this milestone needs already ships and is oracle-covered.
The blocker is a set of **10 moment-weighted (`r^{2n}`-weighted) 3-center and 1-electron
families** that gate **GTH pseudopotential evaluation in Phase 10** — the second phase,
not the eighteenth. Without them no periodic SCF runs at all, because every reference
system in §9.2 uses `gth-pade`.

Those 10 symbols are specified for implementation as **Wave 0.5** of
`/home/user/Documents/workspace/cintx/.planning/notes/gradient-family-gap-closure-PLAN.md` §1.3b.

**⚠️ Do not assume a declared symbol errors when unimplemented.** `center_3c1e.rs:1469`
falls through (`_ => {}`) to the plain 3-center overlap for any unrecognised operator
name. If that fall-through is reachable, requesting `int3c1e_r2_origk_sph` returns the
**unweighted** integral silently. cintx plan task W0-05 proves or disproves this.
**Until it is disproved, treat every `oracle_covered:false` cintx row as
"may return a plausible wrong number", not "will error".** Plan 10-05 must assert
`int3c1e_r2_origk != int3c1e` on a fixture before trusting any value.

---

---

## 3. Architecture decisions

These are LOCKED. Do not revisit them mid-implementation. If one turns out wrong,
stop, write a deviation note in the plan's SUMMARY, and escalate.

| ID | Decision | Rationale |
|---|---|---|
| **D-PBC-01** | `Cell` is a **new struct that OWNS a `Mole`** (`pub struct Cell { pub mol: Mole, pub a: [[f64;3];3], … }`) with `Deref<Target = Mole>`, NOT a duplicate of `Mole`. | Upstream `Cell(MoleBase)` inherits. Rust has no inheritance; `Deref` gives `cell.nao_nr` for free and keeps ONE `Mole` build path. |
| **D-PBC-02** | Complex numbers use a **planar split representation** `CTensor { re: Vec<f64>, im: Vec<f64> }` inside the workspace. Interleaved `Complex<f64>` appears only at the PyO3 boundary. | Reuses every existing real cubecl kernel unchanged; keeps the ALG-06 wall intact; no new cubecl numeric type needed. |
| **D-PBC-03** | Complex GEMM = **4 real GEMMs** (`C_re = A_re·B_re − A_im·B_im`, `C_im = A_re·B_im + A_im·B_re`), NOT the 3-multiplication Karatsuba/Gauss trick. | Karatsuba changes the summation order and breaks bit-parity with NumPy/BLAS. Oracle-exactness beats 25% FLOPs. |
| **D-PBC-04** | Complex Hermitian generalized eigh (`F C = S C ε`) uses **faer `c64` `SelfAdjointEigen` if the plan-09-02 smoke test passes**; otherwise the mandated fallback is the **real 2n×2n embedding** `[[Re, −Im],[Im, Re]]` fed to the existing real `eigh_gen`, taking every other eigenpair. | Guarantees a shipping path regardless of faer's complex coverage. |
| **D-PBC-05** | 3-D FFT ships in **two implementations**: `fft_blas` (port of `pyscf/pbc/tools/pbc.py:_fftn_blas`, three batched complex GEMMs against explicit DFT matrices) first, then `fft_stockham` (native cubecl radix-2/3/5 Stockham autosort) as a perf-only swap behind an env flag. | `_fftn_blas` is an upstream-supported engine (`FFT_ENGINE='NUMPY+BLAS'`), so it is bit-comparable AND it maps onto the GEMM kernel that already exists. Arbitrary (odd, non-smooth) mesh sizes work with zero extra code. |
| **D-PBC-06** | The FFT default engine is chosen by `PYSCF_PBC_FFT_ENGINE` ∈ `{blas, stockham}`, default `blas` until plan 11-03 proves `stockham` matches `blas` to 1e-13 on 200 random meshes. | Mirrors upstream's `pbc_tools_pbc_fft_engine` config knob. |
| **D-PBC-07** | Periodic 1-electron integrals (`pbc_intor`) are built by the **image-expansion route**: build a combined `BasisSet` of cell-0 shells + shells translated by each `L ∈ Ls`, evaluate molecular `cintx` shell-pair blocks, and accumulate `Σ_L exp(i·k·L)·block`. NO new cintx operator is required for 1e. | `cintx` is molecular-only and adding a periodic operator to a sibling crate is out of this project's control. |
| **D-PBC-08** | Shell-pair screening uses a **neighbor list** (port of `pyscf/pbc/gto/neighborlist.py`) plus the `rcut`-from-`precision` estimator (`cell.py` `rcut`/`estimate_rcut`). Any pair whose estimated max integral < `cell.precision` is skipped. | Without screening the lattice sum is O(nimgs·nbas²) and unusable. |
| **D-PBC-09** | Density fitting lands in **strict dependency order**: FFTDF (Phase 11) → AFTDF (Phase 13) → GDF/MDF (Phase 14) → RSDF/RSJK (Phase 14). | FFTDF needs only `eval_gto` on a uniform grid + FFT. AFTDF needs `ft_ao`. GDF needs AFTDF + `int3c2e` over images. Building them in any other order blocks on missing primitives. |
| **D-PBC-10** | `ft_aopair` (analytic Fourier transform of a Gaussian AO pair) is implemented as a **new cubecl kernel in `pyscf-kernels`**, ported from `pyscf/lib/pbc/ft_ao.c` semantics via `pyscf/pbc/df/ft_ao.py`. It is the single largest new kernel in the milestone. | No cintx equivalent; it is the hot loop of AFTDF/GDF; it is perfectly data-parallel over `(shell-pair, G)`. |
| **D-PBC-11** | GTH pseudopotentials reuse the **already-landed** `crates/pyscf-gto/src/basis/cp2k_pp.rs` parser. `Mole.pseudo: Option<()>` (currently a placeholder at `mole.rs:281`) is widened to `Option<PseudoData>` in plan 10-01. | Parser exists; only the consumer is missing. |
| **D-PBC-12** | k-point SCF is a **new driver** `pyscf_pbc_scf::kscf::kernel()`, NOT a generalization of `pyscf_scf::kernel()`. The molecular driver stays untouched. | Occupation is global across k (one Fermi level for all k); DIIS extrapolates the stacked `[nkpts][nao][nao]` complex Fock; the loop shapes differ enough that generalizing would destabilize v1.0. |
| **D-PBC-13** | The `OverrideHooks` pattern from `pyscf-scf` is **replicated, not reused**, as `KOverrideHooks` with k-point signatures. | Same reason as D-PBC-12; keeps the v1.0 PyO3 subclass contract stable. |
| **D-PBC-14** | Every `pyscf-pbc-*` method crate is **pyo3-free**. All Python bindings live in `pyscf-py` under `_native.pbc.*`, with `python/pyscf/pbc/*.py` re-export shims. | Matches the v1.0 rule that made Phase 3 work. |
| **D-PBC-15** | k-point symmetry (`*_ksymm`) is a **Phase 17 add-on layer**, never a fork of the Phase 11/12 drivers. It wraps them by mapping IBZ↔BZ. | Upstream does the same (`khf_ksymm.KsymAdaptedKSCF(khf.KSCF)`). |
| **D-PBC-16** | `exxdiv` defaults to `"ewald"` for k-point HF (upstream default) and the Madelung constant is computed by `pyscf_pbc_tools::madelung` on a Monkhorst-Pack supercell, exactly as `pyscf/pbc/tools/pbc.py:548-586`. | Silent exxdiv mismatch is the #1 source of "my periodic energy is wrong by 0.1 Ha". |
| **D-PBC-17** | All complex reductions go through **new ordered primitives** `oracle_zsum`, `oracle_zdot` in `pyscf-algebra`, mirroring `oracle_sum`/`oracle_dot`. Thread-count-independent bit-identity is a CI gate, same as FOUND-06. | v1.0 already pays for this invariant; PBC must not break it. |
| **D-PBC-18** | MPI paths (Phase 20) ship as a **single-rank-correct implementation** behind a `mpi` cargo feature that is OFF by default; multi-rank uses `mpi` crate bindings only if the feature is on. | Keeps the default wheel dependency-free, matching v1.0's distribution goal. |
| **D-PBC-19** | Every new crate gets its **numeric acceptance test written before the implementation** (TDD), using a hard-coded reference value from a one-off upstream PySCF run recorded in the plan. | The oracle venv is CI-gated; in-tree gates must not depend on it. |
| **D-PBC-20** | Cell dimension support ships **3D first, then 2D, then 1D/0D**. `cell.dimension < 3` paths return `NotYetImplemented { phase: 12 }` until plan 12-08. | 3D is 95% of real use; low-dim Coulomb truncation is a separate, subtle body of work. |
| **D-PBC-21** | `ft_aopair` is a **direct lattice sum** `Σ_L e^{ikL} ∫ φ_μ(r) φ_ν(r−L) e^{−i(G+q)r}` over `get_lattice_Ls(rcut = ft_ao.estimate_rcut(cell).max())`, with upstream's own per-shell-pair Schwarz screen. Upstream's `_RangeSeparatedCell` + `ExtendedMole` BvK supermole (~600 of `ft_ao.py`'s 790 lines) is **not ported**; the BvK bucket contraction is a deferred performance optimisation, revisited in Phase 14. | The RS/BvK machinery is numerically transparent — it decontracts and recontracts, and only *drops* terms below `cell.precision·1e-2`. Porting the definition instead makes the screen never tighter than upstream's, so the 1e-10 oracle agreement is a convergence statement rather than a coincidence. Recorded 2026-08-28, plan 13-01. |
| **D-PBC-22** | `with_df` on every k-point driver is **`Box<dyn PeriodicDf>`**, not a concrete builder and not a generic parameter `D: PeriodicDf`. `pyscf_pbc_df::get_{nuc,pp,hcore}` take `&dyn PeriodicDf`. | Phase 11 hard-wired `Fftdf` into eight drivers, so nothing downstream can be handed an AFTDF — Phase 13's own gate is unmeasurable without this. The trait is already object-safe. A generic parameter would monomorphise the whole SCF machinery once per builder for no measured gain, against one vtable hop per SCF iteration. Recorded 2026-08-28, plan 13-07. |
| **D-PBC-23** | `exclude_dd_block` is **DEFERRED**: Phase 14 ports the *definition* of the 3-centre lattice sum (upstream's `exclude_dd_block = False` route) and does not port `ft_ao._RangeSeparatedCell` / `_int_dd_block` / `merge_diffused_block` / `ExtendedMole.strip_basis`. A caller that explicitly asks for `true` gets `NotYetImplemented { phase: 17 }`. | **This one needed a MEASUREMENT, not a judgement.** D-PBC-21 could argue the RS machinery is numerically transparent; that argument does NOT carry here, because `exclude_dd_block` is not screening — it RE-ROUTES the smooth–smooth block of `(ij\|L)` out of the real-space sum and into an FFT, and upstream's default is the *more* accurate route. `measurements/ddblock.py` prices it: **1.835e-8 Ha** (diamond 2×2×2), **2.900e-8** (gamma), **exactly 0** (He-fcc/`sto-3g`, which has no smooth shell — `bas_type = [1]`). So Gate 1 is stated on He-fcc, where the deferral is provably inert, and Gate 1b asserts diamond against upstream run with `exclude_dd_block=False`. Plan 14-05 then measured a SECOND piece of the same machinery: `ExtendedMole.strip_basis`'s per-shell-pair radii are worth **1.054e-09** in `j3c` and **2.750e-09** in the ERI, and the port — which evaluates every pair to the maximum radius — is the MORE converged of the two. Recorded 2026-08-29, plans 14-01/14-05. |
| **D-PBC-24** | Everything range-separated in Phase 14 — `rsdf_builder._RSGDFBuilder`, `mdf._RSMDFBuilder`, `rsdf.RSGDF`, `scf.rsjk` — is **BLOCKED on `cintx`, not deferred by choice**, and returns `NotYetImplemented { phase: 14 }` naming the gap. The ω machinery (all twelve estimators, `weighted_coulG_LR/_SR`, `_gaussian_int`) ships and is gated. | Range separation is not a distinct integral symbol: upstream never calls an `int2e_sr_*`. It is libcint's `PTR_RANGE_OMEGA` (`env[8]`) toggle around the STANDARD `int3c2e`/`int2c2e`/`int2e`. `cintx_runtime::ExecutionOptions` has `f12_zeta` (`env[9]`), `rinv_orig` and `common_orig` and **no `range_omega`**; no kernel reads `env[8]`; and `incore::aux_e2` reaches cintx through `build_image_expanded_with_aux`, which builds its `BasisSet` from the parsed per-element basis rather than from an `_env` array, so even `pyscf-gto`'s own direct-`_env` workaround is unreachable. **This is Phase 4's Open Question A5 / cintx#11**, already documented in `crates/pyscf-gto/src/range_coulomb.rs`. `14-07-PLAN.md` Task 7b required this be REPORTED rather than worked around with a numerically different kernel — a full-range substitute would run, converge, and be silently a different method, and for `rsjk` (which is EXACT) the wrong answer would land inside GDF's 1.2e-3 fitting error and look plausible. Consequence: **Gate 3 is unreachable** and `GDF._prefer_ccdf` stays `true`. **Lifting it is planned in `.planning/carryovers/D-PBC-24-cintx-range-omega-PLAN.md`**, whose sizing finding is that `rys_order = (sum l_ceil)/2 + 1` is `<= 3` on every system this milestone gates, and that libcint computes the short-range integral in that regime as `full - LR` with DOUBLED Rys roots using only the standard root finder — so `CINTsr_rys_roots` is a later stage, not a prerequisite. Recorded 2026-08-29, plans 14-07/14-08; planned 2026-08-30. |
| **D-PBC-25** | `KPoints` lives in **`pyscf-pbc-symm`**, not `pyscf-pbc-lib`, and holds a `Symmetry` by **composition**. `pyscf-pbc-df` and `pyscf-pbc-dft` gain a `pyscf-pbc-symm` dependency. | Upstream declares `class KPoints(symm.Symmetry, lib.StreamObject)` (`kpts.py:847`) — `KPoints` *is a* `Symmetry`. Mirroring the file name would put it in `pyscf-pbc-lib`, which already sits *below* `pyscf-pbc-symm` in the dependency graph (`pyscf-pbc-symm/Cargo.toml`) and cannot see `Cell` at all, so the natural placement is a **cycle**. The `df`/`dft` dependency is forced by upstream's seven `isinstance(kpts, KPoints)` branches in `pbc/dft/numint.py` (`:328, 431, 647, 779, 859, 908, 956`) and by every DF builder's `kpts` setter (`fft.py:230-246`, `aft.py:613-641`, `df.py:189-217`, `mdf.py:59`). Verified acyclic. `xtask check_dependency_wall` polices only cubecl deps (`check_dependency_wall.rs:28-60`), so no exemption is needed. **Corollary:** `pyscf-pbc-dft` may not name cubecl (ALG-06), so multigrid's kernels go in `pyscf-kernels` (plans 17-11/17-12). Recorded 2026-08-31. |
| **D-PBC-27** | Plan 17-10 ports `ft_ao._RangeSeparatedCell` (`RsCell`) by copying ALREADY-NORMALISED `cintx::Shell`s out of the built `basis_set` and reordering/slicing them — never through `Cell::build`, which would re-normalise a decontracted (smaller-`nprim`) shell's coefficients a second time and silently break upstream's `_env`-splice invariant. `ExtendedMole` is represented as `(rs_cell, Ls, bvkmesh_Ls, bas_mask)`, NOT as a literal replicated `Mole` the way upstream builds it. `exclude_dd_block`'s closure (`gdf_builder::dd_block::fft_dd_block` + `j3c::make_j3c_scheme_dd`) is a POST-HOC scatter-add correction applied after the existing real-space pass, not a rewrite of it. | `Cell::build`'s normalisation path (`pyscf_gto::make_env::normalise_contractions`) rescales each shell's contraction column by `1/sqrt(cᵀSc)` over THAT shell's own primitive set; routing a smaller decontracted shell through it changes the rescale factor, which is exactly the corruption upstream's raw `_env` splice avoids. This port's dual (raw-array + cintx `BasisSet`) representation needs the analogous fix at the `Shell` level — see `AuxCell::modrho_scale`'s precedent for the same class of problem. `ExtendedMole`'s quantities (`strip_basis`'s surviving triples, `get_ovlp_mask`'s screen) are functions of shell parameters + geometry only, never of an actually-cint-drivable molecule, so materialising `nimgs · bvk_ncells · rs_cell.nbas` real shells would cost real time and buy nothing this plan gates. The post-hoc correction shape means `make_j3c`/`make_j3c_scheme`'s existing callers and tests are behaviourally UNCHANGED (`dd_correction = None` is a no-op path), which is what let `exclude_dd_block = true` ship as a fully-working OPT-IN without touching the already-shipped, already-tested real-space pipeline; the crate-wide DEFAULT was deliberately left at `false` rather than flipped, pending a full-suite regression run — see `17-10-SUMMARY.md`. Recorded 2026-09-01, plan 17-10. Gated: `crates/pyscf-pbc-df/tests/{rs_cell,extended_mole,exclude_dd_block}.rs`, `crates/pyscf-pbc-scf/tests/exclude_dd_block_energy.rs`. |
| **D-PBC-28** | **Phase 15 speed ruling.** Parallelise independent k-work with rayon; retain ordered `oracle_*` reductions; store Lov as `(ia,L)` with `L` fastest; never use `zgemm_dense`/`gemm_dense` on Phase-15 paths; use MO-first FFTDF/AFTDF AO2MO; cache Coulomb/G-vector data by momentum transfer; include concurrent scratch in memory preflight. Recorded in full at `15-CONTEXT.md §7`; implemented 2026-09-05. | Correctness is bit-gated independently of speed. **Measured, 2026-09-05** (`15-VERIFICATION.md §8`): upstream's small He/6-31g fixture puts GDF/Lov **5.6313x** ahead of FFTDF/MO-first; this port's **MO-first FFT AO2MO is 9.784x** faster than its AO-ERI route on diamond `gth-szv` `[1,1,2]` (78.289 s -> 8.002 s over 8 conserving quadruples, residual 1.735e-17), against §7.0's `(nao²/(nocc·nvir))² = 16x` prediction — **the prediction over-states it**, because both routes pay the same grid-side cost, and that correction is the useful part of the measurement. `build_symm_map` costs 0.143 ms / 5.59 ms / 245 ms / 3.040 s at `nkpts = 8 / 27 / 64 / 125`, growing **faster** than `n³` at the two larger steps (43.78x and 12.41x measured against 13.32x and 7.45x predicted) — which is why `Kmp2::new` builds `kconserv` only, and why Phase 16, which does want the map, should build it once. The `[2,2,2]` and `gth-dzvp` legs of the MO-first sweep were NOT reached (multi-hour at ~26 s/quadruple) and are reported unreached rather than extrapolated; the test that runs them is committed. No speed number is inferred from the cost model. |
| **D-PBC-29** | **Phase 16's speed + memory ruling, four clauses.** **(1) Complex tensors get their own arena.** `ZWorkspacePool` in `pyscf-runtime` with `shape_bytes = product * 16`, non-copying block access and per-buffer locking; `WorkspacePool`'s f64 API stays byte-for-byte unchanged. Reinterpreting `Box<[f64]>` as complex pairs is FORBIDDEN. **(2) Contractions are host rayon loops over k-triples with `oracle_*` accumulators, not `zgemm_dense`**, and every site NAMES its primitive (`oracle_zdot` = `zdotc` vs `oracle_zdotu`). **(3) `symm_map` is used from the FIRST version of the ERI build**, not retrofitted. **(4) Storage tiers are selected from an exact per-tensor byte count, never from upstream's `_mem_usage`**, and at least one green test must cross a tier boundary. Recorded 2026-09-02, plans 16-02/16-05, `16-REVIEW.md §6`. | **(1)** `workspace_pool.rs:278-280` computes `shape_bytes` as `product * 8` and feeds `try_reserve` (`:266-274`), the HARD `MemoryLimitExceeded` refusal — a complex tensor sized with `* 8` under-reports by 2× to the one mechanism whose job is to refuse before an OOM, on the machine where 17-12's host suite already exit-137s. `as_slice` (`:397`) is `Ok(b.to_vec())`, a full copy per access, and `with_mut_slice` (`:461-483`, the closure runs at `:475`) holds the pool's single `Mutex` across the caller's closure, which would serialise every rayon k-loop at one core. `pyscf-ccsd` has zero complex arithmetic, so only the arena's SHAPE is reusable, not its element type. **(2)** Standing measurements `zgemm-dense-loses-to-host-rayon` (6-12× slower on the CPU backend AND 1.35e-10 off, outside the 1e-11 gate) and `pyscf-algebra-cpu-is-default-backend`. Re-measured against this phase's own shapes at 16-14 Task 4.2, and amended if it wins there. The naming requirement is `15-REVIEW.md D-15-R-02`'s lesson: a plan saying only "route through `oracle_dot`" yields `Σ x·x` instead of `Σ conj(x)·y` — a plausible wrong number only the final energy gate catches. **(3)** `kpts_helper.py:583-630` puts each of the `nkpts³` triples in an orbit of ≤4; `kccsd_rhf.py:783`/`:798-805`/`:909` transforms `≈nkpts³/4` blocks and transposes the rest — a derived ~4× on the phase's dominant step, which the original §8.8 never mentions. **`15-REVIEW.md D-15-R-04`'s ≤2× ruling does NOT carry over**: it held because KMP2 only wants `(ov|ov)` blocks, whereas KCCSD's `_ERIS` wants the full general block (`kccsd_rhf.py:789-794`), so all four operations land inside the set it needs. Measured by 16-01 Task 6, not asserted cold. **(4)** `kccsd_rhf.py:1100-1107` is `nkpts³·nmo⁴·4·16` carrying its own `# TODO`, and over-estimates by **9.1×** (diamond `gth-szv`) to **6.2×** (`gth-dzvp`) against the seven blocks actually allocated — porting it would make this port's HARD refusal reject jobs that fit. Derived `vvvv`: `gth-szv` 2×2×2 **2.0 MiB**, 4×4×4 **1.0 GiB**; `gth-dzvp` 2×2×2 **1.79 GiB**, 3×3×3 **68.7 GiB**; ×3 for KUCCSD's three `Wvvvv`-class tensors, **×16 for KGCCSD**. Every `§9.2` fixture is `gth-szv`, where `vvvv` at 2×2×2 is 2 MiB — so a gate on those alone ships the HDF5 spill path never once executed, which is 17-12's exit-137 shape. Gated: `crates/pyscf-runtime/tests/zworkspace_pool.rs`, `crates/pyscf-pbc-lib/tests/symm_map.rs`, `crates/pyscf-pbc-cc/tests/{ktensor,keris_tiers,kccsd_rhf,kccsd_t}.rs`. |

---

## 4. New crates (workspace 20 → 39)

Add these to `[workspace] members` in root `Cargo.toml`, in this order:

```
crates/pyscf-pbc-lib      # kpts_helper, KPoints, ktensor, linalg_helper, arnoldi
crates/pyscf-pbc-tools    # fft/ifft, get_coulG, madelung, lattice_Ls, super_cell, k2gamma
crates/pyscf-pbc-gto      # Cell, Gv, SI, ewald, GTH pseudo, pbc_intor, eval_gto periodic, neighborlist
crates/pyscf-pbc-df       # FFTDF, AFTDF, GDF, MDF, RSDF, ft_ao, jk builders
crates/pyscf-pbc-scf      # SCF (gamma) + KSCF/KRHF/KUHF/KROHF/KGHF, smearing, addons, stability, newton
crates/pyscf-pbc-dft      # gen_grid, numint, KRKS/KUKS/KROKS/KGKS + gamma, DFT+U, multigrid, cdft
crates/pyscf-pbc-ao2mo    # periodic MO transforms
crates/pyscf-pbc-mp       # KMP2, KUMP2, kmp2_stagger
crates/pyscf-pbc-cc       # KCCSD RHF/UHF/GHF, (T), EOM
crates/pyscf-pbc-ci       # KCIS
crates/pyscf-pbc-symm     # space group, Symmetry, k-point symmetry adapters
crates/pyscf-pbc-grad     # periodic gradients + stress tensor
crates/pyscf-pbc-geomopt  # cell + geometry optimization
crates/pyscf-pbc-tdscf    # periodic TDA/TDHF/TDDFT
crates/pyscf-pbc-gw       # KGW-AC, KGW-CD
crates/pyscf-pbc-adc      # KADC IP/EA
crates/pyscf-pbc-x2c      # periodic sfx2c1e, x2c1e
crates/pyscf-pbc-eph      # electron-phonon (finite difference)
crates/pyscf-pbc-mpi      # mpicc + mpitools (feature-gated)
```

**Dependency DAG** (a crate may depend only on crates ABOVE it in this list, plus
the molecular crates and `pyscf-algebra`/`pyscf-kernels`):

```
pyscf-algebra ─┬─> pyscf-pbc-lib ──> pyscf-pbc-tools ──> pyscf-pbc-gto ──> pyscf-pbc-df
pyscf-kernels ─┘                                              │                 │
                                                              ▼                 ▼
                                                        pyscf-pbc-symm     pyscf-pbc-scf
                                                                                │
                                        ┌───────────────┬───────────────┬───────┴────────┐
                                        ▼               ▼               ▼                ▼
                                  pyscf-pbc-dft   pyscf-pbc-ao2mo  pyscf-pbc-grad   pyscf-pbc-x2c
                                        │               │               │
                                        │               ▼               ▼
                                        │         pyscf-pbc-mp    pyscf-pbc-geomopt
                                        │               │
                                        │               ▼
                                        │         pyscf-pbc-cc ──> pyscf-pbc-ci
                                        ▼               │
                                  pyscf-pbc-tdscf       ▼
                                  pyscf-pbc-gw    pyscf-pbc-mpi
                                  pyscf-pbc-adc
                                  pyscf-pbc-eph
```

Every crate's `Cargo.toml` template (plan 09-01 creates all 19 as stubs at once):

```toml
[package]
name = "pyscf-pbc-<name>"
version.workspace = true
edition.workspace = true
rust-version.workspace = true
license.workspace = true
repository.workspace = true
authors.workspace = true

[dependencies]
pyscf-core    = { path = "../pyscf-core" }
pyscf-algebra = { path = "../pyscf-algebra" }
# ... only crates above this one in the DAG ...
tracing = { workspace = true }

# FORBIDDEN in this file: pyo3, cubecl-*, hdf5-metno (except pyscf-pbc-df, which
# may use hdf5-metno for on-disk GDF _cderi storage — same allowance pyscf-ccsd has).
```

---

## 5. The complex-algebra contract (Phase 9, plan 09-02)

This is the single most important new API. Everything downstream depends on it.
Implement it EXACTLY as specified.

### 5.1 Host type — `crates/pyscf-algebra/src/complex.rs`

```rust
/// Planar (split) complex matrix/vector. `re` and `im` always have equal length.
/// Row-major unless a function explicitly documents F-order.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct CTensor {
    pub re: Vec<f64>,
    pub im: Vec<f64>,
}

impl CTensor {
    pub fn zeros(n: usize) -> Self;
    pub fn from_interleaved(z: &[f64]) -> Self;   // [re0,im0,re1,im1,...] -> planar
    pub fn to_interleaved(&self) -> Vec<f64>;     // planar -> [re0,im0,...]
    pub fn from_real(re: &[f64]) -> Self;         // im = zeros
    pub fn len(&self) -> usize;                   // == re.len()
    pub fn conj(&self) -> Self;                   // im negated
    pub fn is_real(&self, tol: f64) -> bool;      // max|im| < tol
}
```

### 5.2 Operations — `crates/pyscf-algebra/src/zgemm.rs`, `zblas.rs`, `zeigh.rs`

| Function | Signature | Implementation (mandatory) |
|---|---|---|
| `zgemm_dense` | `(client, a: &CTensor, b: &CTensor, m, k, n) -> Result<CTensor, AlgebraError>` | Four calls to the existing `gemm_dense`: `t1=Ar·Br`, `t2=Ai·Bi`, `t3=Ar·Bi`, `t4=Ai·Br`; `re = t1 − t2`, `im = t3 + t4`. **In that order.** (D-PBC-03) |
| `zgemm_h_dense` | same, but `a` conjugate-transposed | Build `Aᴴ` explicitly via `transpose_dense` + negate `im`, then call `zgemm_dense`. Do not fuse. |
| `zaxpy_dense` | `(client, alpha: (f64,f64), x: &CTensor, y: &mut CTensor)` | `y.re += ar*x.re − ai*x.im`; `y.im += ar*x.im + ai*x.re` via 4 `axpy_dense` calls |
| `zscal_dense` | `(client, alpha: (f64,f64), x: &mut CTensor)` | as above with a temp |
| `zdotc_dense` | `(client, x, y) -> (f64, f64)` | `re = dot(xr,yr) + dot(xi,yi)`; `im = dot(xr,yi) − dot(xi,yr)` |
| `zdotu_dense` | `(client, x, y) -> (f64, f64)` | `re = dot(xr,yr) − dot(xi,yi)`; `im = dot(xr,yi) + dot(xi,yr)` |
| `zreduce_sum_dense` | `(client, x) -> (f64, f64)` | two `reduce_sum_dense` |
| `ztranspose_dense` | `(client, x, rows, cols) -> CTensor` | two `transpose_dense` |
| `zhadamard_dense` | `(client, x, y) -> CTensor` | new cubecl kernel `zhadamard_kernel<F: Float>` in `pyscf-kernels` (§6 K-04) |
| `oracle_zsum` | `(x: &CTensor) -> (f64,f64)` | two `oracle_sum` — **ordered, thread-count-independent** (D-PBC-17) |
| `oracle_zdot` | `(x, y) -> (f64,f64)` | four `oracle_dot`, combined as `zdotc` |
| `zeigh_gen` | `(f: &CTensor, s: &CTensor, n: usize) -> Result<(Vec<f64>, CTensor), AlgebraError>` | **D-PBC-04**: try faer `c64`; on failure use the real 2n×2n embedding (§5.3) |
| `zcholesky` | `(a: &CTensor, n) -> Result<CTensor, AlgebraError>` | faer `c64` LLT, or embedding fallback |
| `zsolve_linear` | `(a: &CTensor, b: &CTensor, n) -> Result<CTensor, AlgebraError>` | faer `c64` LU, or 2n×2n real embedding |

### 5.3 The mandated eigh fallback (write this even if faer c64 works — it is the CI cross-check)

For Hermitian `H = Hr + i·Hi` (n×n), build the **real symmetric** 2n×2n matrix
```
M = [ Hr   −Hi ]
    [ Hi    Hr ]
```
`M` is symmetric because `Hr` is symmetric and `Hi` is antisymmetric.
Its eigenvalues are those of `H`, **each appearing twice**. Same embedding for `S`.
Solve the real generalized problem with the existing `eigh_gen`, then:
1. Sort eigenvalues ascending (already sorted by faer).
2. Take indices `0, 2, 4, …, 2n−2` (one of each degenerate pair).
3. Eigenvector `k` of `H` is `C[0..n] + i·C[n..2n]` from column `2k`.
4. Normalize so that `Cᴴ S C = I`, then apply `pyscf_core::canonicalize_signs`
   on the **real part** to get vendor-stable phases (Pitfall 4 mitigation).

**Acceptance test** (`crates/pyscf-algebra/tests/zeigh.rs`):
For a random 8×8 Hermitian `H` and `S = I`, `zeigh_gen` eigenvalues match
`faer` on the 16×16 embedding to 1e-12, and `Cᴴ H C` is diagonal to 1e-12.

---

## 6. cubecl kernel inventory

All new kernels live in `crates/pyscf-kernels/src/pbc/`, are `#[cube(launch_unchecked)]`,
and are generic over `F: Float` (AGENTS.md §3). Each gets a host launcher that fans out
via `pyscf_algebra::dispatch_backend!`. Each gets a CPU-vs-host-reference test in
`crates/pyscf-kernels/tests/pbc_<name>.rs`.

| ID | Kernel | File | Parallel over | Formula | Phase |
|---|---|---|---|---|---|
| **K-01** | `gv_kernel` | `pbc/gv.rs` | `(x,y,z)` grid points | `Gv[x,y,z,:] = rx[x]·b[0] + ry[y]·b[1] + rz[z]·b[2]` — verbatim port of `pyscf/lib/pbc/cell.c:122-146` | 9 |
| **K-02** | `struct_factor_kernel` | `pbc/struct_factor.rs` | `(atom, G)` | `SI[a,g] = exp(−i·Gv[g]·R_a)` → `re = cos(−θ)`, `im = sin(−θ)` | 9 |
| **K-03** | `coulg_kernel` | `pbc/coulg.rs` | `G` | `coulG[g] = 4π/|k+G|²`; `=0` at `|k+G|=0`; ×`exp(−|k+G|²/(4ω²))` if LR; ×`(1−exp(…))` if SR | 9 |
| **K-04** | `zhadamard_kernel` | `pbc/zhadamard.rs` | element | `(ar·br − ai·bi, ar·bi + ai·br)` | 9 |
| **K-05** | `ewald_real_kernel` | `pbc/ewald.rs` | `(L, i, j)` | `q_i q_j erfc(η·r)/r`, `r<1e-16 → r=1e200` | 9 |
| **K-06** | `ewald_recip_kernel` | `pbc/ewald.rs` | `G` | `|ZS(G)|²·exp(−G²/4η²)·4π/G²·w` | 9 |
| **K-07** | `bloch_phase_kernel` | `pbc/bloch.rs` | `(L, k)` then `(elem)` | `out[k] += (cos(k·L), sin(k·L)) ⊙ block[L]` | 10 |
| **K-08** | `eval_ao_kpts_kernel` | `pbc/eval_ao_k.rs` | `(grid pt, AO)` | reuses `pyscf_kernels::eval_gto` per image, ×`exp(i k·L)` accumulate | 10 |
| **K-09** | `gth_vloc_kernel` | `pbc/gth_vloc.rs` | `G` | `pyscf/pbc/gto/pseudo/pp.py:58-95` `get_gth_vlocG` |10 |
| **K-10** | `gth_projg_kernel` | `pbc/gth_projg.rs` | `(proj, G)` | `projG_li`, `_qli` — `pp.py:107-195` | 10 |
| **K-11** | `fft_blas` (host, no new kernel) | `pyscf-pbc-tools/src/fft.rs` | — | three batched `zgemm_dense` vs explicit DFT matrices; port of `pbc.py:30-68` | 11 |
| **K-12** | `fft_stockham_kernel` | `pbc/fft.rs` | butterflies | radix-2/3/5 Stockham autosort, per-axis; Bluestein for prime factors > 5 | 11 (perf) |
| **K-13** | `rho_k_kernel` | `pbc/rho_k.rs` | grid pt | `ρ(r) = (1/N_k)·Σ_k Σ_μν ao*_μk(r)·D^k_μν·ao_νk(r)` — done as `zgemm` + row-wise `zdotc` | 11 |
| **K-14** | `vmat_ao_kernel` | `pbc/vmat.rs` | `(AO, AO)` | `V^k_μν = Σ_r ao*_μk(r)·v(r)·ao_νk(r)·w` — `zgemm_h_dense` | 11 |
| **K-15** | `ft_aopair_kernel` | `pbc/ft_aopair.rs` | `(shell-pair, G)` | analytic FT of a contracted Gaussian pair; see `§8.13` for the derivation and the exact recursion | 13 |
| **K-16** | `kconserv_kernel` | `pbc/kconserv.rs` | `(ki,kj,kk)` | integer table `kl = ki − kj + kk (mod BZ)` | 15 |
| **K-17** | `kbatched_zgemm` | `pbc/kbatched.rs` | `(k, tile)` | batched complex GEMM over the k-index for KCCSD contractions | 16 |

**Kernels you must NOT write:** anything for which `pyscf-algebra` already has a real
primitive that you can call four times. Adding a bespoke complex kernel where
`4 × gemm_dense` works is a D-PBC-03 violation.

---

## 7. Phase roadmap (Phase 9 → Phase 20)

| Phase | Name | Delivers | Gate (must be TRUE to close) |
|---|---|---|---|
| **9** | PBC Foundation + Complex Algebra | 19 crate stubs, complex algebra, `Cell`, lattice/reciprocal/Gv/SI, `Ls`, Ewald, k-point meshes, super_cell, cutoff↔mesh | `cell.ewald()` for diamond C2 matches upstream to 1e-9 Ha; `zeigh_gen` matches faer to 1e-12 |
| **10** | Periodic integrals + GTH PP | `pbc_intor` (ovlp/kin/nuc at k), neighbor list + screening, GTH `get_pp` (loc part1/part2 + nl), periodic `eval_ao_kpts` | `cell.pbc_intor('int1e_ovlp', kpts=…)` matches upstream to 1e-10 for diamond 2×2×2. **cintx prereq: Wave 0.5 (10 moment-weighted families) — see §2.4** |
| **11** | FFT + FFTDF + periodic HF | complex 3-D FFT, `get_coulG`, `madelung`, `fft_jk`, `FFTDF`, gamma-point RHF/UHF/ROHF/GHF, `KRHF`/`KUHF`/`KROHF`/`KGHF`, smearing, chkfile | `KRHF(diamond, 2×2×2 kpts, gth-szv/gth-pade).kernel()` matches upstream to 1e-7 Ha |
| **12** | Periodic DFT | `UniformGrids`+`BeckeGrids`, periodic `NumInt` (`eval_ao_kpts`, `nr_rks`/`nr_uks` at k), `KRKS`/`KUKS`/`KROKS`/`KGKS` + gamma variants, DFT+U, cdft, 2D/1D/0D dimensions | `KRKS(Si, 2×2×2, PBE).kernel()` matches upstream to 1e-7 Ha |
| **13** | ft_ao + AFTDF | `ft_ao`/`ft_aopair` cubecl kernel, `AFTDF` `get_nuc`/`get_pp`/`get_jk`, `aft_jk`, `aft_ao2mo` | `ft_aopair` matches upstream to 1e-10; `AFTDF` KRHF energy == FFTDF KRHF energy to 1e-6 |
| **14** | GDF / MDF / RSDF / RSJK | `gdf_builder`, `GDF` (+ HDF5 `_cderi`), `incore`/`outcore`, `MDF`, `rsdf_builder`, `rsdf_helper`, `RSDF`, `df_jk`, `rsjk` | `GDF` KRHF energy == FFTDF to 1e-6 with 10× less memory on a 4×4×4 mesh |
| **15** | Periodic AO2MO + KMP2 | `pbc/ao2mo/eris`, per-DF `*_ao2mo`, `KMP2`, `KUMP2`, `kmp2_stagger` | `KMP2` correlation energy matches upstream to 1e-8 |
| **16** | Periodic CC + CI | `KCCSD` (RHF/UHF/GHF), `kintermediates`, `KCCSD(T)`, `EOM-KCCSD` IP/EA (+EE: GHF full, RHF singlet-only, **UHF absent upstream** — see §8.8), `KCIS` | ~~`KRCCSD` `e_corr` matches upstream to 1e-8 on He 1×1×2~~ **UNMEASURED, and it contradicts `ROADMAP.md`'s 1e-14 for the same number — six orders apart, the fourth instance of this defect (cf. Phases 14, 15, 17).** Upstream's own suite asserts `KRCCSD` `e_corr` at **6 decimals** (`pbc/cc/test/test_krccsd.py:180`, `:226`, `:232`, `:338`, `:356`) and EOM roots at **3** (`:359-366`), so 1e-14 is eight orders tighter than upstream and 1e-8 is two. **Plan 16-01 measures the floor before the gate is written** (`.planning/phases/16-periodic-cc-ci/16-01-PLAN.md`) and restates it here, in `ROADMAP.md` and in `16-CONTEXT §2` together; the gate must name its DF route (`kccsd_rhf.py:37` branches on `isinstance(with_df, GDF)`, the split Phase 14 measured at 4.5e-6 Ha). |
| **17** | k-point symmetry + multigrid | `pbc/symm/*`, `KPoints` IBZ machinery, all `*_ksymm` adapters, `dft/multigrid` | ~~`KRHF` with `space_group_symmetry=True` equals the no-symmetry energy to 1e-9`~~ **WRONG, replaced by five measured gates (17-01, `.planning/phases/17-ksymm-multigrid/measurements/README.md`; §8.9 below).** Gate A (IBZ integers, exact): 145/145/245/408/816/2052, reproduced bit-for-bit on `si`/`diamond`. Gate B (transforms vs one converged SCF): ≤1e-9 at default `cell.precision`, ≤1e-13 at `cell.precision=1e-13`. Gate C/D (energy, symmetry vs no-symmetry, mesh pinned): FFTDF ≤5.985e-11, GDF ≤3.433e-09, both inside upstream's own 5e-8/5e-7. Gate E (multigrid vs reference `numint`): v1 exact to 1e-12; v2 carries a mesh-independent ~2e-8…2e-7 floor and is NOT held to Gate B-D. |
| **18** | Periodic gradients + stress + geomopt | `grad/krhf`, `kuhf`, `krks`, `kuks`, `krkspu`, `kukspu`, gamma variants, `*_stress`, `geomopt` | analytic gradient matches central-difference `verify_fd` to 1e-6 Ha/Bohr. **cintx prereq: the `_ip1_r{2,4,6}_origk` + `r{2,4}_origi_ip2` half of Wave 0.5 — see §2.4** |
| **19** | Periodic response + relativistic | `tdscf` (k + gamma), `gw` (AC + CD), `adc` (IP/EA), `x2c`, `newton_ah`, `stability`, `cphf`, `eph` | KRHF-TDA lowest excitation matches upstream to 1e-6 eV |
| **20** | MPI + tools + bindings + ship | `mpicc`/`mpitools` (feature-gated), `k2gamma`, `lattice`, `pyscf_ase`, `pywannier90` shim, full `pyscf.pbc.*` PyO3 surface, oracle CI, benchmarks | `from pyscf.pbc import gto, scf, dft` runs an unmodified upstream script |

---

## 8. Phase breakdown — the actual work

Notation for every plan:
- **`FILES`** — files you create or modify. Create them all; leave `todo!()`-free stubs.
- **`PORT`** — the upstream Python you must read and translate, with line numbers.
- **`STEPS`** — do these in order.
- **`TEST`** — the file to write and the exact assertion.
- **`DONE`** — the command that must be green.

---

### 8.1 Phase 9 — PBC Foundation + Complex Algebra

**Goal:** the workspace has 39 members, complex linear algebra works and is
bit-reproducible, and a `Cell` can be built, produce its reciprocal lattice, its
G-vector grid, its lattice-image list, its k-point mesh, and its Ewald energy.

**Plans:** 9 (09-01 … 09-09), 4 waves.

---

#### Plan 09-01 — Workspace scaffolding (Wave 1)

**FILES**
- `Cargo.toml` (root) — add the 19 `crates/pyscf-pbc-*` members from §4
- `crates/pyscf-pbc-*/Cargo.toml` × 19 — from the §4 template
- `crates/pyscf-pbc-*/src/lib.rs` × 19 — `//! <one-line purpose>` + `pub mod error;`
- `crates/pyscf-pbc-*/src/error.rs` × 19 — a `thiserror` enum with one `Core(#[from] PyscfRsError)` variant
- `xtask/src/bin/check_forbidden_paths.rs` — RULE 7 change
- `xtask/src/bin/check_dependency_wall.rs` — add `pyscf-kernels` and nothing else to the cubecl allowlist (already there); assert no `pyscf-pbc-*` names a `cubecl-*` dep
- `docs/design/pbc-architecture.md` — copy §2–§6 of this document

**STEPS**
1. `sed -n '1,30p' Cargo.toml` — see the existing members list and its comment style.
2. Append the 19 members with a comment `# v2.0 PBC milestone (D-PBC-*). Workspace grows 20 → 39.`
3. For each crate: `mkdir -p crates/pyscf-pbc-<n>/src`, write `Cargo.toml` + `lib.rs` + `error.rs`.
4. In `check_forbidden_paths.rs`: change the needle list so `"use pyscf::pbc"` is
   **removed**, and add a path predicate: files whose path contains `crates/pyscf-pbc-`
   are exempt from every needle. Keep the molecular ban intact for all other crates.
5. `cargo build --workspace` — must be clean.
6. `cargo run -p xtask --bin check_forbidden_paths` — must pass.
7. `cargo run -p xtask --bin check_dependency_wall` — must pass.

**TEST** `xtask/tests/forbidden_paths_pbc_exempt.rs` — a fixture file under a fake
`crates/pyscf-pbc-foo/` path containing `use pyscf::pbc` is NOT flagged; the same
content under `crates/pyscf-scf/` IS flagged.

**DONE** `cargo build --workspace && cargo test -p xtask`

---

#### Plan 09-02 — Complex algebra in `pyscf-algebra` (Wave 1)

**FILES**
- `crates/pyscf-algebra/src/complex.rs` — `CTensor` (§5.1)
- `crates/pyscf-algebra/src/zgemm.rs` — `zgemm_dense`, `zgemm_h_dense`
- `crates/pyscf-algebra/src/zblas.rs` — `zaxpy_dense`, `zscal_dense`, `zdotc_dense`, `zdotu_dense`, `zreduce_sum_dense`, `ztranspose_dense`, `zhadamard_dense`
- `crates/pyscf-algebra/src/zeigh.rs` — `zeigh_gen`, `zcholesky`, `zsolve_linear` (§5.3)
- `crates/pyscf-algebra/src/zoracle.rs` — `oracle_zsum`, `oracle_zdot`
- `crates/pyscf-algebra/src/lib.rs` — declare + re-export all of the above
- `crates/pyscf-kernels/src/pbc/mod.rs`, `crates/pyscf-kernels/src/pbc/zhadamard.rs` — K-04

**STEPS**
1. **First**, write a 15-line throwaway binary that constructs a `faer::Mat<faer::c64>`
   and calls `SelfAdjointEigen`. If it compiles and runs → set `const FAER_C64: bool = true`
   in `zeigh.rs`. If it does NOT → set `false`. Record the outcome in the SUMMARY.
   This decides D-PBC-04 for the whole milestone.
2. Implement `CTensor` exactly as §5.1. `from_interleaved`/`to_interleaved` must be
   exact inverses; assert that in a test.
3. Implement `zgemm_dense` as FOUR `gemm_dense` calls in the stated order (D-PBC-03).
   Do not reorder. Do not fuse. Add a doc comment citing D-PBC-03.
4. Implement the rest of `zblas.rs` in terms of existing real primitives.
5. Implement `zhadamard_kernel<F: Float>` per AGENTS.md §3 and launch it via
   `pyscf_algebra::dispatch_backend!`. Read the cubecl manual first (RULE 5).
6. Implement `zeigh_gen`: the faer-`c64` branch if step 1 said yes, and the
   2n×2n real-embedding branch ALWAYS (it is the cross-check). If `FAER_C64` is
   true, a debug-only assertion compares the two to 1e-11 on `n ≤ 16`.
7. Implement `oracle_zsum`/`oracle_zdot` as two/four ordered real calls.

**TEST**
- `crates/pyscf-algebra/tests/ctensor.rs` — interleaved round-trip, `conj`, `is_real`
- `crates/pyscf-algebra/tests/zgemm.rs` — 64×64 random complex; compare against a
  naive triple-loop host reference to 1e-12
- `crates/pyscf-algebra/tests/zeigh.rs` — §5.3 acceptance test
- `crates/pyscf-algebra/tests/zoracle_determinism.rs` — `oracle_zsum` on a 1e6-element
  `CTensor` is **bit-identical** under `RAYON_NUM_THREADS=1` and `=8` (D-PBC-17)

**DONE** `cargo test -p pyscf-algebra && cargo test -p pyscf-kernels`

---

#### Plan 09-03 — `Cell` type (Wave 2)

**FILES**
- `crates/pyscf-pbc-gto/src/cell.rs`
- `crates/pyscf-pbc-gto/src/types.rs` — `CellBuildArgs`, `LowDimFtType`, `DimensionKind`

**PORT** `pyscf/pbc/gto/cell.py:1250-1600` (class body + `build`), `:1811-1975` (properties)

**STEPS**
1. Define:
```rust
pub struct Cell {
    pub mol: pyscf_core::Mole,        // D-PBC-01: OWNED, not duplicated
    pub a: [[f64; 3]; 3],             // lattice vectors, Bohr, row-major
    pub mesh: [usize; 3],
    pub dimension: u8,                // 0..=3, default 3
    pub low_dim_ft_type: LowDimFtType,// None | InfVacuum
    pub precision: f64,               // default 1e-8
    pub ke_cutoff: Option<f64>,
    pub rcut: f64,                    // computed by estimate_rcut
    pub ew_eta: Option<f64>,
    pub ew_cut: Option<f64>,
    pub pseudo: Option<PseudoData>,   // filled in Phase 10 (D-PBC-11)
    pub exp_to_discard: Option<f64>,
    pub use_particle_mesh_ewald: bool,
    pub space_group_symmetry: bool,
    pub _built: bool,
}
impl std::ops::Deref for Cell { type Target = Mole; fn deref(&self) -> &Mole { &self.mol } }
impl std::ops::DerefMut for Cell { fn deref_mut(&mut self) -> &mut Mole { &mut self.mol } }
```
2. `pub fn lattice_vectors(&self) -> [[f64;3];3]` — returns `a` (already Bohr).
   Upstream also accepts a string/`Å`; handle unit conversion in `build`, not here.
3. `pub fn vol(&self) -> f64` — `det(a).abs()`.
4. `pub fn reciprocal_vectors(&self, norm_to: f64) -> [[f64;3];3]`
   — `inv(a.T) * norm_to`. Default `norm_to = 2π`. **Port `cell.py:1896-1917` exactly**;
   note the `dimension < 3` branch that zeroes out the non-periodic rows.
5. `pub fn get_scaled_kpts(&self, abs_kpts) -> Vec<[f64;3]>` — `abs_kpts · a.T / 2π`
6. `pub fn get_abs_kpts(&self, scaled) -> Vec<[f64;3]>` — `scaled · b`
7. `pub fn get_scaled_atom_coords(&self) -> Vec<[f64;3]>` — `coords · inv(a)`
8. `pub fn build(args: CellBuildArgs) -> Result<Cell, PyscfRsError>` — delegates the
   molecular half to `pyscf_gto::build_from`, then computes `rcut`, `mesh` (from
   `ke_cutoff` or `precision`), `ew_eta`/`ew_cut`.
9. `pub fn tot_electrons(&self, nkpts: usize) -> usize` — port `cell.py:957-967`
   (**note: PP-adjusted electron count once Phase 10 lands `pseudo`**).

**TEST** `crates/pyscf-pbc-gto/tests/cell_build.rs`
- Diamond: `a = 3.5668 Å` fcc, 2 C atoms at `(0,0,0)` and `(0.25,0.25,0.25)` scaled.
- Assert `vol` == 45.2445… Bohr³ to 1e-6 (compute the exact value with upstream once
  and hard-code it — D-PBC-19).
- Assert `reciprocal_vectors(2π) · a.T == 2π·I` to 1e-12.
- Assert `get_abs_kpts(get_scaled_kpts(k)) == k` to 1e-12 for 10 random k.

**DONE** `cargo test -p pyscf-pbc-gto --test cell_build`

---

#### Plan 09-04 — Cutoffs, rcut, mesh (Wave 2)

**FILES** `crates/pyscf-pbc-gto/src/cutoff.rs`, `crates/pyscf-pbc-tools/src/mesh.rs`

**PORT**
- `cell.py:373-497` — `get_nimgs`, `_estimate_rcut`, `bas_rcut`, `estimate_rcut`,
  `_estimate_ke_cutoff`, `estimate_ke_cutoff`, `_extract_pgto_params`, `error_for_ke_cutoff`
- `cell.py:499-523` — `get_bounding_sphere`
- `cell.py:974-1025` — `pgf_rcut`, `rcut_by_shells`
- `tools/pbc.py:787-835` — `cutoff_to_mesh`, `mesh_to_cutoff`, `cutoff_to_gs`, `gs_to_cutoff`

**STEPS** Port each function one-to-one. These are pure scalar math — no GPU.
Keep the same clamping and `np.ceil` semantics; use `f64::ceil` and `as usize`.

**TEST** `crates/pyscf-pbc-gto/tests/cutoff.rs` — for diamond/gth-szv,
`estimate_rcut(1e-8)` and `cutoff_to_mesh(a, 100.0)` match hard-coded upstream values
(`rcut ≈ 15.6` Bohr, `mesh == [15,15,15]` — regenerate and hard-code these before writing the test).

---

#### Plan 09-05 — G-vectors, structure factors, uniform grids (Wave 2) — **GPU**

**FILES**
- `crates/pyscf-kernels/src/pbc/gv.rs` (K-01), `struct_factor.rs` (K-02)
- `crates/pyscf-pbc-gto/src/gv.rs` — host wrappers `get_Gv`, `get_Gv_weights`, `get_SI`, `get_uniform_grids`

**PORT** `cell.py:525-648` (`get_Gv`, `get_Gv_weights`, `_non_uniform_Gv_base`, `get_SI`),
`cell.py:886-912` (`get_uniform_grids`), and the C kernel `pyscf/lib/pbc/cell.c:122-146`.

**STEPS**
1. `fftfreq(n, 1/n)` in Rust = `(0..n).map(|i| if i <= (n-1)/2 { i as f64 } else { i as f64 - n as f64 })`.
   **Write this as a single helper `fn fftfreq_scaled(n: usize) -> Vec<f64>` and unit-test it
   against `numpy.fft.fftfreq(n, 1./n)` for n = 1..=32.** Getting this wrong silently
   corrupts every FFT downstream.
2. K-01 `gv_kernel<F: Float>(rx, ry, rz, b, gv, mx, my, mz)` — one thread per `(x,y,z)`;
   body is the 9-line C loop verbatim. Grid: `CubeCount::Static(ceil(ngrids/256),1,1)`,
   `CubeDim { x: 256, y: 1, z: 1 }`.
3. `weights = |det(b)| / (2π)³` for the uniform 3D case. The `dimension ≤ 2 &&
   low_dim_ft_type == InfVacuum` branches return `NotYetImplemented { phase: 12 }` (D-PBC-20).
4. K-02 `struct_factor_kernel<F: Float>(coords, gv, si_re, si_im, natm, ngrids)`:
   `θ = −(Gv[g]·R_a)`, `si_re = cos θ`, `si_im = sin θ`. One thread per `(a,g)`.
5. `get_uniform_grids(cell, mesh, wrap_around)` — the real-space grid
   `r[i,j,k] = (i/nx)·a[0] + (j/ny)·a[1] + (k/nz)·a[2]`, with `wrap_around` folding
   indices `> n/2` to negative. Port `cell.py:886-912` exactly.

**TEST** `crates/pyscf-pbc-gto/tests/gv.rs`
- `fftfreq_scaled(n)` matches a hard-coded table for n=1..8.
- For diamond mesh `[5,5,5]`, `get_Gv` row 0 is `[0,0,0]` and `get_Gv` matches a
  hard-coded 125×3 array from upstream to 1e-12.
- `get_SI` at `G=0` equals `1+0i` for every atom.
- `|SI[a,g]| == 1` for all `a,g` to 1e-14.

**DONE** `cargo test -p pyscf-pbc-gto --test gv`

---

#### Plan 09-06 — Lattice sums: `get_lattice_Ls`, `super_cell`, `cell_plus_imgs` (Wave 3)

**FILES** `crates/pyscf-pbc-tools/src/lattice.rs`, `crates/pyscf-pbc-tools/src/supercell.rs`

**PORT** `tools/pbc.py:601-786` (`get_lattice_Ls`, `check_lattice_sum_range`,
`super_cell`, `cell_plus_imgs`, `_build_supcell_`), `tools/pbc.py:587-600`
(`get_monkhorst_pack_size`), `tools/pbc.py:836-840` (`round_to_cell0`).

**STEPS**
1. `get_lattice_Ls(cell, nimgs, rcut, dimension, discard)`:
   a. Default `dimension`: `if cell.dimension < 2 || low_dim == InfVacuum { cell.dimension } else { 3 }`.
   b. `dR = (scaled_max − scaled_min)[..dim] · a[..dim]`; `dR_basis = diag(dR)`.
   c. `find_boundary(a_perm)`: QR of `vstack([a_perm, dR_basis]).T`, take `R`;
      `ub = (rcut + |R[2,3..]|.sum()) / |R[2,2]|`. **Use `pyscf_algebra::qr`.**
   d. `bounds = ceil([xb, yb, zb])`; cartesian product of `-b..=b` per axis; `Ls = Ts · a`.
   e. If `discard && Ls.len() > 1`: keep `|L| < rcut + max_ij |r_i − r_j|`.
2. `super_cell(cell, ncopy, wrap_around)` — build a new `Cell` with `natm × Πncopy` atoms
   and `a_super[i] = ncopy[i] · a[i]`. Reuse `pyscf_gto::build_from` for the molecular half.
3. `get_monkhorst_pack_size(cell, kpts, tol)` — port `pbc.py:587-600` exactly, including
   the `tol = max(10^(-int(-log10(1/nkpts))-2), 1e-5)` line.

**TEST** `crates/pyscf-pbc-tools/tests/lattice.rs`
- Diamond, `rcut = 10.0` → `Ls.len()` equals the hard-coded upstream count; `Ls[0] == [0,0,0]`.
- `super_cell(diamond, [2,2,2]).natm == 16`; its `vol == 8 × cell.vol` to 1e-9.

---

#### Plan 09-07 — k-point meshes: `make_kpts` (Wave 3)

**FILES** `crates/pyscf-pbc-gto/src/kpts_mesh.rs`, `crates/pyscf-pbc-lib/src/kpts_helper.rs`

**PORT** `cell.py:827-885` (`make_kpts`), `lib/kpts_helper.py` (`get_kconserv`,
`get_kconserv3`, `member`, `is_zero`, `round_to_fbz`, `unique`, `KPT_DIFF_TOL`).

**STEPS**
1. `make_kpts(cell, nks: [usize;3], wrap_around: bool, with_gamma_point: bool, scaled_center: Option<[f64;3]>) -> Vec<[f64;3]>`:
   - `ks_each_axis[i] = if with_gamma_point { (0..nks[i]).map(|x| x as f64 / nks[i] as f64) }
     else { (arange(nks[i]) + 0.5)/nks[i] − 0.5 }`
   - cartesian product → `scaled_kpts`; `+ scaled_center` if given
   - `if wrap_around { scaled_kpts = round_to_fbz(scaled_kpts) }` (fold to `[-0.5, 0.5)`)
   - return `cell.get_abs_kpts(scaled_kpts)`
2. `is_zero(k) -> bool` — `k.iter().map(|x| x.abs()).sum::<f64>() < 1e-9`.
   **This exact threshold decides real-vs-complex code paths everywhere. Do not change it.**
3. `get_kconserv(cell, kpts) -> Vec<i32>` shaped `[nk][nk][nk]`:
   for `(ki,kj,kk)`, find `kl` with `ki − kj + kk − kl ≡ 0 (mod b)`. Port
   `kpts_helper.py` verbatim including `KPT_DIFF_TOL = 1e-6`.

**TEST** `crates/pyscf-pbc-gto/tests/kpts_mesh.rs`
- `make_kpts(diamond,[2,2,2])` returns 8 k-points, the first being `[0,0,0]`.
- `get_kconserv` for a 2×2×2 mesh matches a hard-coded 8×8×8 int array from upstream.

---

#### Plan 09-08 — Ewald summation (Wave 4) — **GPU**

**FILES**
- `crates/pyscf-kernels/src/pbc/ewald.rs` (K-05, K-06)
- `crates/pyscf-pbc-gto/src/ewald.rs` — `get_ewald_params`, `ewald`
- `crates/pyscf-pbc-gto/src/ewald_pme.rs` — particle-mesh Ewald (b-splines)

**PORT** `cell.py:650-695` (`get_ewald_params`), `cell.py:696-826` (`ewald`, all four
dimension branches), `gto/ewald_methods.py:32-292` (b-splines + PME).

**STEPS** (3D branch only in this plan; 2D/1D/0D deferred to plan 12-08 per D-PBC-20)
1. `get_ewald_params(cell, precision, mesh) -> (ew_eta, ew_cut)` — port `cell.py:650-695`.
2. `ewald(cell, ew_eta, ew_cut) -> f64`:
   - `chargs = cell.atom_charges()` (PP-adjusted once Phase 10 lands)
   - `log_precision = ln(precision / (Σq · 16π²))`; `ke_cutoff = −2·η²·log_precision`;
     `mesh = cutoff_to_mesh(a, ke_cutoff)`
   - `Ls = get_lattice_Ls(cell, rcut = ew_cut)`
   - **`ewovrl`** via K-05: `0.5 · Σ_{L,i,j} q_i q_j erfc(η r)/r`, where
     `r = |R_i − R_j + L|` and `r < 1e-16 → r = 1e200`. Reduce with `oracle_sum`.
     `erfc` on device: implement `erfc_f64` as a `#[cube]` helper using the
     Abramowitz–Stegun 7.1.26 rational approximation **only if** a device `erfc`
     is unavailable; otherwise compute `erfc` on host and pass the array in.
     *(Preferred: compute `r` on device, `erfc` on host — precision matters more
     than the extra transfer here. Record the choice in the SUMMARY.)*
   - **`ewself`** = `−0.5 · (Σ q²) · 2η/√π`; plus `−0.5 · (Σq)² · π/(η²·vol)` when `dimension == 3`
   - **`ewg`** via K-06: `Gv, Gvbase, weights = get_Gv_weights(mesh)`;
     `absG2[absG2 == 0] = 1e200`; `coulG = 4π/absG2 · weights`;
     `ZSI = Σ_a q_a · SI[a,:]`; `ewg = 0.5 · Re Σ_g ZSI*[g]·ZSI[g]·exp(−absG2[g]/(4η²))·coulG[g]`
   - return `ewovrl + ewself + ewg`
3. `Cell::energy_nuc()` returns `ewald()` when `a` is set, else `Mole::enuc()`.

**TEST** `crates/pyscf-pbc-gto/tests/ewald.rs`
- Diamond primitive cell (2 C, `a = 3.5668 Å` fcc): `ewald()` == hard-coded upstream
  value to **1e-9 Ha**. Generate that number once:
  ```python
  from pyscf.pbc import gto
  c = gto.M(atom='C 0 0 0; C 0.8917 0.8917 0.8917', a=[[0,1.7834,1.7834],[1.7834,0,1.7834],[1.7834,1.7834,0]], basis='gth-szv', pseudo='gth-pade')
  print(repr(c.ewald()))
  ```
- Ewald energy is invariant to `ew_eta` within `[0.5η₀, 2η₀]` to 1e-8 (self-consistency
  gate that needs no oracle).

**DONE** `cargo test -p pyscf-pbc-gto --test ewald`

---

#### Plan 09-09 — Phase 9 verification rollup (Wave 4)

**FILES** `.planning/phases/09-pbc-foundation/09-VERIFICATION.md`, `python/pyscf/pbc/__init__.py` (empty shim), `crates/pyscf-pbc-gto/tests/oracle_phase9.rs` (venv-gated)

**STEPS**
1. Write the pytest oracle comparing `Cell` construction, `Gv`, `SI`, `Ls`, `kpts`,
   and `ewald` against upstream for 5 systems (diamond, Si, LiF, graphene 2D-marked-3D, He fcc).
2. Gate it behind `#[ignore]` + `PYSCF_ORACLE_VENV`, exactly as `crates/pyscf-gto/tests/`
   already does. Record all 5 reference numbers in the verification doc.
3. Update `.planning/STATE.md` and `.planning/ROADMAP.md` phase-9 checkbox.

---

### 8.2 Phase 10 — Periodic integrals + GTH pseudopotentials

**Goal:** `cell.pbc_intor(name, kpts)` returns k-resolved 1-electron matrices that match
upstream, GTH pseudopotentials are consumed, and AOs can be evaluated on a periodic grid
at arbitrary k.

**Plans:** 8 (10-01 … 10-08).

---

#### Plan 10-01 — GTH pseudopotential data model

**FILES** `crates/pyscf-pbc-gto/src/pseudo/mod.rs`, `pseudo/data.rs`, and
`crates/pyscf-core/src/mole.rs` (widen `pseudo: Option<()>` → `Option<PseudoData>`)

**PORT** `pyscf/gto/basis/parse_cp2k_pp.py` semantics (already parsed by
`crates/pyscf-gto/src/basis/cp2k_pp.rs` — **read that file first**), plus
`pyscf/pbc/gto/pseudo/__init__.py`.

**STEPS**
1. `codegraph_explore("cp2k_pp parse_cp2k_pp ParsedPseudo")` — learn the existing output type.
2. Define:
```rust
pub struct GthPseudo {
    pub symbol: String,
    pub nelec_per_shell: Vec<i32>,   // [ns, np, nd, nf]
    pub rloc: f64,
    pub cexp: Vec<f64>,              // C1..C4 local coefficients (0..4 entries)
    pub nl_blocks: Vec<GthNlBlock>,  // one per l
}
pub struct GthNlBlock { pub l: usize, pub rl: f64, pub nproj: usize, pub hl: Vec<f64> /* nproj×nproj, row-major */ }
pub struct PseudoData { pub per_symbol: std::collections::HashMap<String, GthPseudo> }
```
3. `Cell::build` resolves `pseudo='gth-pade'` through
   `pyscf_gto::basis::path::pbc_pseudo_dir()` (already exists) + `cp2k_pp` parser.
4. **`Cell::atom_charges()` must return the PP valence charge** `Σ nelec_per_shell`
   for PP'd atoms, not `Z`. This single line changes Ewald, `tot_electrons`, and every
   energy downstream. Port `cell.py` `atom_charges` override.

**TEST** `crates/pyscf-pbc-gto/tests/pseudo_parse.rs` — `gth-pade` for C gives
`nelec = [2,2,0,0]`, `rloc = 0.34883045`, `cexp = [-8.51377110, 1.22843203]`,
one `l=0` NL block with `rl = 0.30455321`, `hl = [[9.52284179]]`.
(Verify these against `pyscf/pbc/gto/pseudo/gth-pade.dat` before writing the test:
`grep -A5 '^C GTH-PADE' pyscf/pbc/gto/pseudo/gth-pade.dat`.)

---

#### Plan 10-02 — Neighbor list + shell screening

**FILES** `crates/pyscf-pbc-gto/src/neighborlist.rs`

**PORT** `pyscf/pbc/gto/neighborlist.py:1-200` (whole file), `cell.py:993-1025` (`rcut_by_shells`)

**STEPS**
1. `pub struct NeighborPair { pub ish: usize, pub jsh: usize, pub l_idx: usize }`
2. `pub fn build_neighbor_list(cell: &Cell, ls: &[[f64;3]], rcut_shell: &[f64]) -> Vec<NeighborPair>`
   — include `(ish, jsh, L)` iff `|R_ish − R_jsh − L| < rcut_shell[ish] + rcut_shell[jsh]`.
3. `rcut_by_shells(cell, precision)` → per-shell radii from `pgf_rcut` (plan 09-04).

**TEST** `crates/pyscf-pbc-gto/tests/neighborlist.rs` — for diamond/gth-szv with
`rcut = 15`, the pair count matches a hard-coded upstream number, and every excluded
pair has an overlap integral below `cell.precision`.

---

#### Plan 10-03 — `pbc_intor` / `intor_cross` (the lattice-sum driver) — **the core of Phase 10**

**FILES** `crates/pyscf-pbc-gto/src/pbc_intor.rs`, `crates/pyscf-kernels/src/pbc/bloch.rs` (K-07)

**PORT** `cell.py:184-288` (`intor_cross`), `cell.py:289-372` (`_intor_cross_screened`),
`cell.py:2018-2042` (`Cell.pbc_intor`), and the C driver semantics of
`PBCnr2c_drv` / `PBCnr2c_fill_ks1` in `pyscf/lib/pbc/`.

**ALGORITHM — implement exactly this (D-PBC-07):**
```
fn pbc_intor(cell, intor_name, comp, hermi, kpts) -> Vec<CTensor>   // one per k, [nao×nao]
  1. name  = add_suffix(intor_name, cell.cart)            // reuse pyscf_gto::intor::add_suffix
  2. Ls    = get_lattice_Ls(cell, rcut = cell.rcut)       // plan 09-06
  3. pairs = build_neighbor_list(cell, Ls, rcut_by_shells(cell))   // plan 10-02
  4. out[k] = CTensor::zeros(nao*nao) for each k
  5. for each distinct L in Ls:
       a. shifted = cell.mol.clone_with_atom_shift(L)     // shift PTR_COORD by L in _env
       b. combined = pyscf_gto::projection::build_combined_basis(&cell.mol, &shifted)
          // cell-0 shells occupy [0, nbas); image shells occupy [nbas, 2*nbas)
       c. for (ish, jsh, l_idx) in pairs where l_idx == index_of(L):
            block = cintx SessionRequest(op, rep, combined, [ish, nbas + jsh]).evaluate()
            // block is F-order [ni, nj] (or [comp, ni, nj])
            for k in 0..nkpts:
              phase = (cos(kpts[k]·L), sin(kpts[k]·L))    // K-07
              out[k][rows, cols] += phase * block
  6. if hermi != 0: hermi_triu(out[k])                    // fill the upper triangle
  7. if is_zero(kpts[k]): drop the imaginary part (assert |im| < 1e-9 first)
```
**Optimization that MUST be in the first version:** hoist step 5b out of the `L`
loop when `nbas` is large by building ONE combined basis containing cell-0 shells
plus all image shells (`nbas × (1 + nimgs)` shells) and indexing `[ish, nbas + l_idx*nbas + jsh]`.
Choose the one-big-basis form if `nimgs · nbas ≤ 20_000`, otherwise the per-L form
(memory guard). Put the threshold in a named `const PBC_INTOR_ONE_SHOT_SHELL_LIMIT: usize = 20_000;`.

**Supported names in this plan:** `int1e_ovlp`, `int1e_kin`, `int1e_nuc`, `int1e_r`, `int1e_ipovlp`, `int1e_ipkin`, `int1e_ipnuc`.
Anything else → `NotYetImplemented { phase: 13 }`.

**TEST** `crates/pyscf-pbc-gto/tests/pbc_intor.rs`
- Diamond/gth-szv, `kpts = make_kpts([2,2,2])`:
  `pbc_intor("int1e_ovlp")[0]` is real (max |im| < 1e-10) and Hermitian to 1e-12.
- `S^k` is positive definite for every k (`zcholesky` succeeds).
- Gamma-point `S` matches the hard-coded upstream 8×8 matrix to 1e-10.
- **Self-consistency gate (no oracle needed):** `pbc_intor("int1e_ovlp", kpts=[k])` computed
  with `rcut` and with `1.5·rcut` agree to 1e-9.

**DONE** `cargo test -p pyscf-pbc-gto --test pbc_intor`

---

#### Plan 10-04 — Periodic `eval_gto` / `eval_ao_kpts` — **GPU**

**FILES** `crates/pyscf-kernels/src/pbc/eval_ao_k.rs` (K-08), `crates/pyscf-pbc-gto/src/eval_gto.rs`

**PORT** `pyscf/pbc/gto/eval_gto.py:1-257` (whole file), `cell.py:2043-2053`.

**ALGORITHM**
```
eval_ao_kpts(cell, coords[ngrids][3], kpts[nk][3], deriv) -> Vec<CTensor>  // [nk] of [ngrids × nao × ncomp]
  Ls = get_lattice_Ls(cell)
  for L in Ls:
     ao_L = pyscf_gto::eval_gto(cell.mol, name, coords − L)   // reuse the EXISTING molecular kernel
     for k: out[k] += exp(i·kpts[k]·L) * ao_L
```
Do **not** write a new AO evaluator. `crates/pyscf-kernels/src/eval_gto.rs` (2,564 lines)
already handles s/p/d + deriv1, sph + cart. K-08 is only the phase-accumulate step.

**TEST** `crates/pyscf-pbc-gto/tests/eval_ao_kpts.rs`
- At `k = 0`, `eval_ao_kpts` is real to 1e-12.
- **Bloch periodicity:** `ao_k(r + L) == exp(i k·L) · ao_k(r)` to 1e-10 for random `r`, `L`, `k`.
  This is a complete correctness gate requiring no oracle. Make it the primary test.

---

#### Plan 10-05 — GTH local pseudopotential (`vloc`) — **GPU**

**FILES** `crates/pyscf-kernels/src/pbc/gth_vloc.rs` (K-09), `crates/pyscf-pbc-gto/src/pseudo/vloc.rs`

**PORT** `pseudo/pp.py:33-95` (`get_alphas`, `get_alphas_gth`, `get_vlocG`, `get_gth_vlocG`),
`pseudo/pp_int.py:47-117` (`get_pp_loc_part1`, `get_gth_vlocG_part1`),
`pseudo/pp_int.py:118-170` (`get_pp_loc_part2`, `get_pp_loc_part2_gamma`),
`pseudo/pp_int.py:511-576` (`fake_cell_vloc`).

**⚠️ cintx PREREQUISITE (§2.4).** `get_pp_loc_part2` (`pp_int.py:150-151`) calls the
intor tuple `('int3c2e', 'int3c1e', 'int3c1e_r2_origk', 'int3c1e_r4_origk',
'int3c1e_r6_origk')`. The last three are **declared in the cintx manifest with
`oracle_covered: false` and have NO dispatch arm** (`center_3c1e.rs:1469` matches only
`ip1`/`iprinv` and falls through on `_ => {}`). They are cintx Wave 0.5.

**Task 0 of this plan, before any porting — the fail-open check:**
```rust
// crates/pyscf-pbc-gto/tests/cintx_moment_weighted_available.rs
#[test]
fn int3c1e_r2_origk_is_available_and_not_the_unweighted_parent() {
    let weighted   = cintx_eval("int3c1e_r2_origk_sph", &fx, shls);
    let unweighted = cintx_eval("int3c1e_sph",          &fx, shls)
        .expect("int3c1e is oracle-covered and must succeed");
    match weighted {
        Err(e) => panic!("BLOCKED on cintx Wave 0.5: {e}"),
        Ok(v)  => assert_ne!(v, unweighted,
            "cintx FAIL-OPEN: int3c1e_r2_origk returned the unweighted int3c1e"),
    }
}
```
Repeat for `_r4_origk` and `_r6_origk`.
- **Err** → the family is genuinely unimplemented. Implement everything else in this
  plan, `#[ignore = "blocked on cintx Wave 0.5"]` the `get_pp_loc_part2` numeric gate,
  and record the blocker in `10-05-SUMMARY.md` + `.planning/STATE.md`.
- **Equal to the unweighted parent** → STOP. This is a silent-wrong-answer bug in a
  shipped cintx API. Escalate to cintx plan task W0-05/W0-06 before writing another line.
- **Different and non-trivial** → proceed and gate on upstream byte-identity as normal.

**Formula for `get_gth_vlocG` (port verbatim, `pp.py:58-95`):**
```
G2 = |G|²;  G = sqrt(G2)
vlocG = 4π/G2 · exp(−G2·rloc²/2) · Zion
      − (2π)^{3/2} · rloc³ · exp(−G2·rloc²/2) · [ C1
        + C2·(3 − G2·rloc²)
        + C3·(15 − 10·G2·rloc² + (G2·rloc²)²)
        + C4·(105 − 105·G2·rloc² + 21·(G2·rloc²)² − (G2·rloc²)³) ]
at G=0:  vlocG[0] = 2π·rloc²·Zion + (2π)^{3/2}·rloc³·(C1 + 3·C2 + 15·C3 + 105·C4)
```
`get_pp_loc_part2` uses **`fake_cell_vloc`**: an auxiliary `Cell` whose "basis" encodes
the Gaussian `C_i r^{2i} exp(−r²/2rloc²)` terms, evaluated with `int1e_ovlp`/`int1e_r2_origk`
style lattice-sum integrals through `pbc_intor` (plan 10-03).

**TEST** `crates/pyscf-pbc-gto/tests/gth_vloc.rs` — `get_gth_vlocG` for C/gth-pade on a
`[5,5,5]` mesh matches a hard-coded upstream array to 1e-12.

---

#### Plan 10-06 — GTH nonlocal pseudopotential (`vnl`) — **GPU**

**FILES** `crates/pyscf-kernels/src/pbc/gth_projg.rs` (K-10), `crates/pyscf-pbc-gto/src/pseudo/vnl.rs`

**PORT** `pseudo/pp.py:96-218` (`get_projG`, `get_gth_projG`, `projG_li`, `_qli`, `Ylm_real`, `cart2polar`),
`pseudo/pp_int.py:408-442` (`get_pp_nl`), `:211-299` (`_prepare_hl_data`, `_contract_ppnl`),
`:577-674` (`fake_cell_vnl`, `_int_vnl`).

**⚠️ cintx PREREQUISITE (§2.4).** `_int_vnl` (`pp_int.py:626`) calls
`('int1e_ovlp', 'int1e_r2_origi', 'int1e_r4_origi')`. The last two are in the same
Wave-0.5 state as plan 10-05's `origk` family: declared, `oracle_covered: false`, no
dispatch arm. Run the same Task-0 fail-open check (comparing against `int1e_ovlp`)
before porting, and apply the same three-way disposition.

**STEPS**
1. `fake_cell_vnl(cell)` builds a fake `Cell` whose shells are the GTH projectors
   `p_i^l(r) ∝ r^{l+2i} exp(−r²/2rl²)`.
2. `ppnl_half = pbc_intor_cross("int1e_ovlp", cell, fakecell, kpts)` — the projector overlaps.
3. `vppnl[k] = Σ_atoms Σ_l  Pᴴ · h_l · P` where `P = ppnl_half` block for that atom/l.
   Port `_contract_ppnl` (`pp_int.py:232-299`) literally, including the `hl` block layout.
4. `get_pp(cell, kpts) = get_pp_loc_part1 + get_pp_loc_part2 + get_pp_nl`.

**TEST** `crates/pyscf-pbc-gto/tests/gth_pp.rs` — `get_pp(diamond/gth-pade, kpts=[0,0,0])`
matches a hard-coded 8×8 upstream matrix to 1e-9; the matrix is Hermitian to 1e-12.

---

#### Plan 10-07 — `get_hcore` / `get_ovlp` assembly

**FILES** `crates/pyscf-pbc-gto/src/hcore.rs`

**STEPS**
- `get_ovlp(cell, kpts) = pbc_intor("int1e_ovlp", kpts)`
- `get_hcore(cell, kpts)`:
  - `T = pbc_intor("int1e_kin", kpts)`
  - `V = if cell.pseudo.is_some() { get_pp(cell, kpts) } else { <FFTDF/AFTDF get_nuc — Phase 11> }`
  - Until Phase 11 lands `get_nuc`, the all-electron branch returns
    `NotYetImplemented { phase: 11 }`. **PP path must work in Phase 10.**

**TEST** `crates/pyscf-pbc-gto/tests/hcore.rs` — Hermiticity + real-at-gamma.

---

#### Plan 10-08 — Phase 10 verification rollup

Same shape as 09-09. Oracle-gated pytest for `pbc_intor` × 4 intors × 3 systems,
`eval_ao_kpts`, and `get_pp`.

---

### 8.3 Phase 11 — FFT + FFTDF + periodic Hartree–Fock

**Goal:** a user runs `KRHF(cell, kpts).kernel()` and gets upstream's energy.
This is the milestone's first end-to-end periodic result.

**Plans:** 12 (11-01 … 11-12).

---

#### Plan 11-01 — Complex 3-D FFT, BLAS engine (D-PBC-05)

**FILES** `crates/pyscf-pbc-tools/src/fft.rs`

**PORT** `pyscf/pbc/tools/pbc.py:30-68` (`_fftn_blas`, `_ifftn_blas`),
`:157-236` (`fft`, `ifft`, `fftk`, `ifftk`).

**ALGORITHM — `fft_blas(f: &CTensor, n_batch, mesh) -> CTensor`:**
```
mx,my,mz = mesh
expRGx[r,g] = exp(-2πi · r · fftfreq(mx)[g])      // mx × mx complex DFT matrix
expRGy, expRGz likewise
g = transpose(f, [n_batch, mx*my*mz] -> [mx*my*mz, n_batch])
g = zgemm( reshape(g,[mx, -1]).T , expRGx )       // contracts x
g = zgemm( reshape(g,[my, -1]).T , expRGy )       // contracts y
g = zgemm( reshape(g,[mz, -1]).T , expRGz )       // contracts z
return reshape(g, [n_batch, mx, my, mz])
```
`ifft_blas` is identical with `exp(+2πi …)` and a `1/mx`, `1/my`, `1/mz` scale applied
per stage (**not** a single `1/ngrids` at the end — matching upstream's staging keeps
rounding identical).

`fftk(f, mesh, expmikr) = fft(f * expmikr, mesh)`;
`ifftk(g, mesh, expikr) = ifft(g, mesh) * expikr`.

**STEPS**
1. Build the three DFT matrices once per `(mesh, direction)` and cache them in a
   `pyo3`-free `std::sync::OnceLock<Mutex<HashMap<([usize;3], bool), Arc<DftMats>>>>`.
2. All three contractions go through `pyscf_algebra::zgemm_dense` (D-PBC-03).
3. Batch size: process `n_batch` rows in chunks of `max(1e5/ngrids, 8)*4`, as upstream does.

**TEST** `crates/pyscf-pbc-tools/tests/fft.rs`
- `ifft(fft(x)) == x` to 1e-12 for 50 random `(mesh, n_batch)` with mesh dims in 1..=17
  (**include odd and prime dims — 3, 5, 7, 11, 13, 17**).
- `fft` of a delta function is all-ones.
- `fft` of a constant is `ngrids·δ_{G,0}`.
- `fft` matches a hard-coded 3×3×3 upstream array to 1e-13.

**DONE** `cargo test -p pyscf-pbc-tools --test fft`

---

#### Plan 11-02 — `get_coulG`, `madelung`, exxdiv — **GPU (K-03)**

**FILES** `crates/pyscf-kernels/src/pbc/coulg.rs`, `crates/pyscf-pbc-tools/src/coulg.rs`, `crates/pyscf-pbc-tools/src/madelung.rs`

**PORT** `tools/pbc.py:237-257` (`_Gv_wrap_around`), `:258-486` (`get_coulG` — ALL branches),
`:487-547` (`precompute_exx`), `:548-586` (`madelung`).

**STEPS**
1. `get_coulG(cell, k, exx, mf, mesh, Gv, wrap_around, omega) -> Vec<f64>`.
   Implement, in this order:
   - `dimension == 3` full-range: `4π/|k+G|²`, `0` at `|k+G| = 0`
   - `omega != 0`: `× exp(−|k+G|²/(4ω²))` (LR, `ω > 0`) or `× (1 − exp(…))` (SR, `ω < 0`)
   - `exxdiv == "ewald"`: add `madelung(cell, kpts) · ngrids · vol/ngrids` at `G+k = 0`
     — port `pbc.py` `_ewald_exxdiv_for_G0` (`df/df_jk.py`) rather than inventing it
   - `exxdiv == "vcut_sph"` / `"vcut_ws"` / `"2d"` / `"1d"` / `"0d"` →
     `NotYetImplemented { phase: 12 }` (D-PBC-20)
2. `madelung(cell, kpts, omega)` — port `pbc.py:548-586` exactly: build the
   Monkhorst-Pack supercell `a_super[i] = Nk[i]·a[i]` with ONE probe charge, return `−2·ewald()`.

**TEST** `crates/pyscf-pbc-tools/tests/coulg.rs`
- `coulG[0] == 0` for `k = 0, exxdiv = None`.
- `coulG` is invariant under `G → −G` to 1e-14.
- `madelung(diamond, make_kpts([2,2,2]))` == hard-coded upstream value to 1e-9.

---

#### Plan 11-03 — Stockham FFT kernel (perf, optional gate) — **GPU (K-12)**

**FILES** `crates/pyscf-kernels/src/pbc/fft.rs`

Implement a radix-2/3/5 Stockham autosort FFT as `#[cube(launch_unchecked)]`, one
axis at a time, with a Bluestein chirp-z fallback for axes whose length has a prime
factor > 5. Selected by `PYSCF_PBC_FFT_ENGINE=stockham` (D-PBC-06).

**TEST** must match `fft_blas` to **1e-13** on 200 random `(mesh, n_batch)`.
Until that test is green, `blas` stays the default. **This plan may be deferred to
Phase 20 without blocking anything.**

---

#### Plan 11-04 — Periodic uniform grids

**FILES** `crates/pyscf-pbc-dft/src/gen_grid.rs` (yes — `pbc-dft`, used by `pbc-df`;
add `pyscf-pbc-dft` as a dep of `pyscf-pbc-df`? **NO** — put `UniformGrids` in
`crates/pyscf-pbc-gto/src/grids.rs` to keep the DAG acyclic, and re-export it from
`pyscf-pbc-dft`).

**PORT** `pyscf/pbc/dft/gen_grid.py:1-294` (`UniformGrids`, `BeckeDFTGrids`, `gen_becke_grids`).

**STEPS**
1. `pub struct UniformGrids { pub cell_mesh: [usize;3], pub coords: Vec<[f64;3]>, pub weights: Vec<f64> }`
   with `weights[i] = vol / ngrids` (uniform).
2. `BeckeDFTGrids` reuses `pyscf_grids::gen_atomic_grids` over the atoms of the
   cell **plus their images within `rcut`**, then applies the Becke partition
   (`pyscf_grids::partition::original_becke`). Port `gen_becke_grids` exactly.

**TEST** `Σ weights == vol` to 1e-10 for both grid types.

---

#### Plan 11-05 — `FFTDF` skeleton + `get_nuc`

**FILES** `crates/pyscf-pbc-df/src/fftdf.rs`, `crates/pyscf-pbc-df/src/traits.rs`

**PORT** `pyscf/pbc/df/fft.py:40-80` (`get_nuc`), `:185-405` (`FFTDF` class).

**Define the DF trait every builder implements** (this is the seam AFTDF/GDF/MDF/RSDF plug into):
```rust
pub trait PeriodicDf {
    fn cell(&self) -> &Cell;
    fn mesh(&self) -> [usize; 3];
    fn kpts(&self) -> &[[f64; 3]];
    fn build(&mut self) -> Result<(), DfError>;
    fn get_nuc(&self, kpts: &[[f64;3]]) -> Result<Vec<CTensor>, DfError>;
    fn get_pp (&self, kpts: &[[f64;3]]) -> Result<Vec<CTensor>, DfError>;
    fn get_jk (&self, dm_kpts: &[CTensor], hermi: i32, kpts: &[[f64;3]],
               kpts_band: Option<&[[f64;3]]>, with_j: bool, with_k: bool,
               exxdiv: Option<ExxDiv>) -> Result<(Option<Vec<CTensor>>, Option<Vec<CTensor>>), DfError>;
}
```
`get_nuc` for FFTDF:
```
Gv, _, _ = cell.get_Gv_weights(mesh)
SI       = cell.get_SI(Gv)
charges  = cell.atom_charges()
rhoG     = −Σ_a charges[a] · SI[a, :]            // nuclear charge density in G space
coulG    = get_coulG(cell, mesh=mesh, Gv=Gv)
vneG     = rhoG * coulG
vneR     = ifft(vneG, mesh).real
vne[k]   = Σ_r ao_k(r)ᴴ · vneR[r] · ao_k(r) · (vol/ngrids)     // K-14
```

**TEST** `get_nuc` at gamma for an all-electron He cell matches upstream to 1e-9.

---

#### Plan 11-06 — `fft_jk::get_j_kpts` — **GPU (K-13, K-14)**

**FILES** `crates/pyscf-pbc-df/src/fft_jk.rs`, `crates/pyscf-kernels/src/pbc/rho_k.rs`, `crates/pyscf-kernels/src/pbc/vmat.rs`

**PORT** `pyscf/pbc/df/fft_jk.py:33-112` (`get_j_kpts`) — **read the whole function, it is short and you must match it exactly.**

**ALGORITHM**
```
1. coulG = get_coulG(cell, mesh=mesh)                       // real, ngrids
2. rhoR  = 0                                                 // ngrids
   for each grid block (p0,p1) and each k:
       ao   = eval_ao_kpts(cell, coords[p0:p1], [k])[0]      // [nblk × nao] complex
       aodm = zgemm(ao, dm_kpts[k])                          // [nblk × nao]
       rhoR[p0:p1] += rowwise_zdotc(aodm, ao)                // Σ_μ aodm[r,μ]·conj(ao[r,μ])
   rhoR /= nkpts
   // if hermi==1 or is_zero(kpts): rhoR is REAL — take .re and assert |.im| < 1e-10
3. rhoG = fft(rhoR, mesh);  vG = coulG ⊙ rhoG;  vR = ifft(vG, mesh)
4. vR *= vol/ngrids
5. for each k:  vj[k] = zgemm_h(ao_k, diag(vR) · ao_k)       // K-14
```

**TEST** `crates/pyscf-pbc-df/tests/fft_j.rs`
- `vj` is Hermitian to 1e-12 for every k.
- `Σ_k Tr(vj[k]·dm[k]).real / nkpts` equals `Σ_r vR[r]·rhoR[r]·ngrids/vol` to 1e-9
  (an internal consistency identity — no oracle needed).

---

#### Plan 11-07 — `fft_jk::get_k_kpts` + exxdiv — **GPU**

**FILES** `crates/pyscf-pbc-df/src/fft_jk.rs` (continued), `crates/pyscf-pbc-df/src/exxdiv.rs`

**PORT** `fft_jk.py:181-309` (`get_k_kpts`), `df/df_jk.py` `_ewald_exxdiv_for_G0`,
`_format_dms`, `_format_kpts_band`, `_format_jks`.

**ALGORITHM (per band k2 and each k1)**
```
for k2 in kpts_band:
  for k1 in kpts:
    coulG = get_coulG(cell, k = k1 − k2, exx = exxdiv, mf, mesh)   // COMPLEX-valued k offset
    expmikr = exp(−i·(k1 − k2)·r)                                   // ngrids
    for each occupied-orbital block of dm[k1]:
       rho_pair[i, r] = conj(ao_k1[r,:]·mo[:,i]) · ao_k2[r,:]       // [nmo × ngrids]
       vG   = fftk(rho_pair, mesh, expmikr) * coulG
       vR   = ifftk(vG, mesh, conj(expmikr))
       vk[k2] += zgemm_h(ao_k2, diag(vR)·ao_k1) ... (accumulate per orbital)
    vk[k2] *= 1/nkpts · vol/ngrids
if exxdiv == Ewald: _ewald_exxdiv_for_G0(cell, kpts, dms, vk, kpts_band)
```
**Warning:** `get_k_kpts` is the most bug-prone routine in the whole milestone.
Port it statement-by-statement. Do not restructure the loops. Keep upstream's
variable names in the Rust code as comments.

**TEST**
- `vk` Hermitian to 1e-12.
- For a 1×1×1 (gamma-only) cell, `vk` from `get_k_kpts` equals the molecular
  `pyscf_scf::fock::get_k` of the equivalent supercell to 1e-8 — **this is the
  strongest oracle-free gate available; make it a required test.**

---

#### Plan 11-08 — `FFTDF::get_jk` + gamma-point `get_jk`

**FILES** `crates/pyscf-pbc-df/src/fftdf.rs` (finish), `crates/pyscf-pbc-df/src/df_jk.rs`

**PORT** `fft_jk.py:411-520` (`get_jk`, `get_j`, `get_k`), `fft.py:345-405`.

---

#### Plan 11-09 — `KSCF` driver (D-PBC-12)

**FILES**
- `crates/pyscf-pbc-scf/src/kscf.rs` — the driver
- `crates/pyscf-pbc-scf/src/khooks.rs` — `KOverrideHooks` (D-PBC-13)
- `crates/pyscf-pbc-scf/src/kocc.rs`, `krdm.rs`, `kenergy.rs`, `kdiis.rs`

**PORT** `pyscf/pbc/scf/khf.py:52-436` (all module functions) + `:437-788` (`class KSCF`).

**THE DRIVER — implement exactly:**
```
kernel(cell, kpts, conf) -> KScfResult
 1. s1e   = get_ovlp(cell, kpts)                       // Vec<CTensor>, nkpts
 2. h1e   = get_hcore(cell, kpts)
 3. dm    = get_init_guess(cell, kpts, conf.init_guess)
 4. e_nuc = cell.energy_nuc()                          // Ewald
 5. loop cycle in 0..max_cycle:
      vhf = get_veff(cell, dm, kpts)                   // via PeriodicDf::get_jk
      fock = h1e + vhf                                  // per k
      if diis: fock = kdiis.extrapolate(fock, dm, s1e) // stacked over k
      for k: (mo_e[k], mo_c[k]) = zeigh_gen(fock[k], s1e[k])
      mo_occ = get_occ(mo_e)                            // GLOBAL aufbau across all k
      dm     = make_rdm1(mo_c, mo_occ)                  // Σ_i occ · C[:,i] Cᴴ[:,i]
      e_tot  = energy_elec(dm, h1e, vhf) + e_nuc
      if |e_tot − e_last| < conv_tol && |grad| < conv_tol_grad: converged
```
**`get_occ` (khf.py:184-225) — the k-point-specific part you MUST get right:**
concatenate all `mo_energy[k]`, sort ascending, take the lowest
`nelectron · nkpts` levels (÷2 for RHF), then scatter the occupations back per k.
There is ONE Fermi level for the whole BZ. Getting this per-k instead of global is
the classic periodic-SCF bug.

**`energy_elec` (khf.py:249-268):**
`e1 = (1/nkpts) Σ_k Tr(dm[k] · h1e[k]).real`;
`e_coul = (1/nkpts) Σ_k Tr(dm[k] · vhf[k]).real · 0.5`.

**DIIS:** stack all k-blocks into one long `CTensor` and run the existing
`pyscf_diis::cdiis` on `[re; im]` concatenated — the error vector is
`Σ_k (F S D − D S F)[k]`, flattened. Port `khf.py:133-160` (`get_fock`).

**TEST** `crates/pyscf-pbc-scf/tests/krhf_diamond.rs`
- `KRHF(diamond, make_kpts([1,1,1]), gth-szv/gth-pade)` converges and
  `e_tot` == hard-coded upstream value to **1e-7 Ha**.
- `KRHF` with `kpts=[2,2,2]` converges in ≤ 30 cycles.
- **Supercell equivalence gate (no oracle):** `KRHF(cell, kpts=[2,1,1]).e_tot`
  equals `RHF(super_cell(cell,[2,1,1])).e_tot / 2` to 1e-7.

**DONE** `cargo test -p pyscf-pbc-scf`

---

#### Plan 11-10 — `KRHF` / `KUHF` / `KROHF` / `KGHF` + gamma-point `RHF`/`UHF`/`ROHF`/`GHF`

**FILES** `crates/pyscf-pbc-scf/src/{krhf,kuhf,krohf,kghf,hf,uhf,rohf,ghf}.rs`

**PORT** `khf.py:789-864` (`KRHF`), `kuhf.py` (635 l), `krohf.py` (386 l), `kghf.py` (323 l),
`hf.py` (1003 l — the single-k / gamma-point SCF), `uhf.py`, `rohf.py`, `ghf.py`.

Each is a thin struct over `kscf::kernel` with its own `get_veff`, `get_occ`,
`make_rdm1`, `energy_elec`. Follow the molecular crate's `rhf.rs`/`uhf.rs` structure.

---

#### Plan 11-11 — Smearing, addons, chkfile, `init_guess`

**FILES** `crates/pyscf-pbc-scf/src/{smearing,addons,chkfile,init_guess}.rs`

**PORT** `scf/smearing.py` (191 l — Fermi–Dirac + Gaussian smearing, `sigma`,
entropy term `−σ·S` in `e_free`), `scf/addons.py` (379 l — `smearing_`, `canonical_occ_`,
`convert_to_*`, `project_mo_nr2nr`), `scf/chkfile.py`, `khf.py:345-386`
(`_cast_mol_init_guess`, `init_guess_by_minao/atom/chkfile`).

**Note on init_guess:** the periodic minao/atom guesses REUSE the molecular
`pyscf_scf::init_guess` on `cell.to_mol()` and then replicate the resulting real dm
to every k-point (`_cast_mol_init_guess`, khf.py:345-362). Do exactly that.

---

#### Plan 11-12 — Phase 11 verification rollup

Oracle pytest: `KRHF`/`KUHF` on diamond, Si, LiF, graphene at 1×1×1, 2×2×2, 3×3×3;
FFTDF `get_nuc`/`get_pp`/`get_jk`. Record every reference number in `11-VERIFICATION.md`.

---

### 8.4 Phase 12 — Periodic DFT

**Goal:** `KRKS(cell, kpts, xc='pbe').kernel()` matches upstream.

**Plans:** 9 (12-01 … 12-09).

| Plan | Content | Port from |
|---|---|---|
| **12-01** | Periodic `NumInt`: `eval_ao_kpts` block loop, `eval_rho` at k (complex), `nr_rks`, `nr_uks`, `nr_rks_fxc`, `eval_mat` | `pbc/dft/numint.py:1-700` |
| **12-02** | `NumInt` continued: `nr_uks_fxc`, `cache_xc_kernel`, `get_rho`, `_format_uks_dm`, block-size heuristics | `pbc/dft/numint.py:700-1346` |
| **12-03** | `KRKS` driver + `get_veff` (XC + J − hyb·K + RSH) | `pbc/dft/krks.py` (292 l) |
| **12-04** | `KUKS`, `KROKS`, `KGKS` | `kuks.py`, `kroks.py`, `kgks.py` |
| **12-05** | Gamma-point `RKS`, `UKS`, `ROKS`, `GKS` | `dft/rks.py` (447 l), `uks.py`, `roks.py`, `gks.py` |
| **12-06** | DFT+U: `KRKSpU`, `KUKSpU` (Hubbard U on projected local orbitals) | `krkspu.py` (325 l), `kukspu.py` (301 l) |
| **12-07** | `numint2c` (2-component / non-collinear) + `cdft` (constrained DFT) | `numint2c.py` (642 l), `cdft.py` (154 l) |
| **12-08** | **Low-dimension support (D-PBC-20 closure):** `dimension ∈ {0,1,2}` branches of `get_Gv_weights`, `get_coulG` (`vcut_sph`, `vcut_ws`, 2D truncated Coulomb), `ewald` 2D branch, `_mesh_inf_vaccum` | `cell.py:560-582`, `cell.py:770-826`, `tools/pbc.py:300-486` |
| **12-09** | Verification rollup + oracle pytest for KRKS/KUKS × {LDA, PBE, PBE0, HSE06} × {diamond, Si, graphene} | — |

**Key implementation notes for 12-01 (the hard one):**
- `eval_rho` at k-points: `ρ(r) = Σ_k w_k Σ_μν ao*_μk(r) D^k_μν ao_νk(r)`. The result
  is **real** — assert `|im| < 1e-10` and drop it. For GGA you also need
  `∇ρ`, which comes from `deriv=1` AOs; the periodic phase factor differentiates too:
  `∇[e^{ikL} ao] = e^{ikL} ∇ao` (the phase is r-independent within a cell), so no extra term.
- `eval_mat` (XC potential → AO matrix) is `zgemm_h(ao_k, diag(w·vxc)·ao_k)` — the same
  K-14 kernel as `get_j`.
- Reuse `pyscf_dft::NumInt::eval_xc` unchanged: XC is evaluated on the real density,
  so libxc/xcfun need no periodic changes at all.

---

### 8.5 Phase 13 — `ft_ao` + AFTDF

**Goal:** analytic Fourier transforms of AO pairs, and the analytic-FT density-fitting
builder that GDF/MDF/RSDF all sit on. At the end of this phase `KRHF(cell, kpts)`
runs on **either** FFTDF or AFTDF with no driver change, and the two agree.

**Plans:** 8 (13-01 … 13-08).

| Wave | Plans |
|---|---|
| 1 | 13-01 (`ft_aopair` MD kernel, K-15), 13-02 (`ft_ao`, single-centre FT) |
| 2 | 13-03 (`ft_aopair_kpts` + `FtKernel` dispatch: k-resolution, aosym, screening, G-blocking) |
| 3 | 13-04 (`AFTDF`: `build`/`ft_loop`/`pw_loop`/`weighted_coulG`/`get_nuc`/`get_pp`) |
| 4 | 13-05 (`aft_jk`), 13-06 (`fft_ao2mo` + `aft_ao2mo`) |
| 5 | 13-07 (`Box<dyn PeriodicDf>` — make every K-driver builder-agnostic) |
| 6 | 13-08 (verification rollup) |

---

#### 8.5.0 Two decisions that govern the whole phase

**D-PBC-21 — `ft_aopair` is a DIRECT lattice sum, not a port of the BvK supermole.**
Upstream spends ~600 of `ft_ao.py`'s 790 lines on `_RangeSeparatedCell` (partial
basis de-contraction into steep/local/smooth blocks) and `ExtendedMole` (a
Born–von-Kármán supercell `Mole` whose shells carry a `bas_mask` back to the
primitive cell). **Neither changes the answer.** `_RangeSeparatedCell` decontracts
and then `recontract`s; `ExtendedMole.strip_basis` only *drops* image shells whose
Schwarz bound is below `cell.precision * 1e-2`. Both are screening and
cache-blocking devices for a C loop over a supermole.

This port implements the mathematical definition instead —

```
ft_aopair_kpts[k, μν, G] = Σ_L e^{i k·L} ∫ φ_μ(r) φ_ν(r − L) e^{−i(G+q)·r} dr
```

— over `Ls = get_lattice_Ls(cell, rcut = estimate_rcut(cell).max())`, with the SAME
per-shell-pair Schwarz screen upstream uses (`ft_ao.py:744-790`, ported verbatim as
`estimate_rcut`). The de-contraction is therefore never needed: a contracted shell
is screened by its **most diffuse** primitive, which is exactly what
`_extract_pgto_params(cell, 'min')` selects.

Consequence: the port is smaller AND its screening is never *tighter* than
upstream's, so the 1e-10 oracle gate is a convergence statement, not a
coincidence. The BvK bucket contraction is recorded as a **deferred performance
optimisation** in plan 13-03 STEP 2, not a correctness item.

> **MEASURED 2026-08-28/29, and it forces a three-part Gate 1.**
> `ft_ao.estimate_rcut` is **looser** than `cell.rcut` — on diamond/`gth-szv` they
> are **20.420** and **21.319** Bohr. So upstream's own `ft_aopair[G=0]` does NOT
> reproduce `pbc_intor("int1e_ovlp")` to 1e-10: it lands at **1.554e-9** at gamma
> and **5.322e-10** at `k ≠ 0`.
>
> Scaling `ft_ao.estimate_rcut` on upstream and re-measuring, at mesh 31:
>
> | `rcut` | Gate 1 residual | `dvj` | `dvk` |
> |---|---|---|---|
> | ×1.0 = 20.42 | 1.554e-9 | 1.996e-11 | 6.487e-10 |
> | ×1.5 = 30.63 | **1.472e-10** | **7.727e-13** | **1.609e-10** |
> | ×2.0 = 40.84 | **1.472e-10** | **7.726e-13** | **1.609e-10** |
>
> **×1.5 and ×2.0 are identical to four digits.** The FT lattice sum is fully
> converged at ~30.6 Bohr, so beyond that `ft_aopair` stops changing — and yet the
> Gate 1 residual sits at 1.472e-10 and will not move.
>
> **That residual is therefore NOT `ft_aopair`'s. It is the reference side's.**
> `pbc_intor("int1e_ovlp")` runs its own lattice sum out to `cell.rcut` = 21.319
> at `cell.precision` = 1e-8. Once the FT sum is converged, Gate 1 is measuring the
> truncation error in the OVERLAP, not in the Fourier transform. **A Gate 1 stated
> as "≤ 1e-11 against `pbc_intor`" is unachievable no matter how correct the
> kernel is.**
>
> **The fix is available and already in the port.**
> `pyscf_pbc_gto::pbc_intor::intor_cross_with_images` takes an explicit image list
> — plan 10-06 added it precisely so several operators could share one `Ls`. Gate 1
> must therefore be run in three parts:
>
> | | `rcut` for `ft_aopair` | reference overlap | target |
> |---|---|---|---|
> | **1a** | `estimate_rcut` (upstream default) | `pbc_intor`, default `Ls` | **≤ 2e-9** — reproduces upstream's 1.554e-9; this is the setting Gate 3 uses |
> | **1b** | `1.5 × cell.rcut` | `pbc_intor`, default `Ls` | **≤ 2e-10** — hits the reference's own floor at 1.472e-10; NOT a kernel gate |
> | **1c** | `1.5 × cell.rcut` | `intor_cross_with_images` over the **SAME `Ls`** | **≤ 1e-13** — both sides converged over one identical image list; **this is the real gate on the McMurchie–Davidson algebra** |
>
> Only 1c isolates the kernel. 1a pins agreement with upstream, 1b is a
> consistency check with a known floor, and 1c is what fails if the recursion is
> wrong.

**D-PBC-22 — `with_df` becomes `Box<dyn PeriodicDf>`.**
Phase 11 hard-wired the concrete `Fftdf` into every driver
(`krhf.rs:29 pub with_df: Fftdf`, and the same in `kuhf`/`krohf`/`kghf` and the
Phase-12 KS drivers), plus the free functions `pyscf_pbc_df::get_{nuc,pp,hcore}`
which take `&Fftdf`. `PeriodicDf` is already object-safe — every method takes
`&self`/`&mut self`, none is generic, none returns `Self`. Plan 13-07 replaces the
field with `Box<dyn PeriodicDf>` and re-signs the three free functions to
`&dyn PeriodicDf`. **Do not make the drivers generic over `D: PeriodicDf`**: eight
driver types × four builders monomorphises the whole SCF machinery four times for
no measured gain, and the `Box` is dereferenced once per SCF iteration.

---

#### Plan 13-01 — `ft_aopair` McMurchie–Davidson kernel (K-15) — **the biggest new kernel in v2.0**

**FILES** `crates/pyscf-kernels/src/pbc/ft_aopair.rs` (device),
`crates/pyscf-pbc-df/src/ft_ao/mcmurchie.rs` (host E-coefficient recursion),
`crates/pyscf-pbc-df/src/ft_ao/mod.rs` (driver),
`crates/pyscf-kernels/tests/pbc_ft_aopair.rs`

**PORT** `pyscf/pbc/df/ft_ao.py:48-61` (`ft_aopair`), `:744-790` (`estimate_rcut`);
the integral itself is `pyscf/lib/gto/ft_ao.c` `GTO_ft_ovlp`, reproduced from the
formula below rather than transliterated from C.

**THE MATH — implement exactly this (McMurchie–Davidson / Hermite expansion):**

For two primitive Cartesian Gaussians centred at `A` (exponent `a`, angular `i`) and
`B` (exponent `b`, angular `j`):
```
p     = a + b
P     = (a·A + b·B) / p
K_AB  = exp(−(a·b/p)·|A − B|²)                       // pre-exponential factor
```
The product `φ_i(r−A)·φ_j(r−B)` expands in Hermite Gaussians `Λ_t(r; P, p)`:
```
φ_i φ_j = Σ_{t=0}^{i+j} E_t^{ij} · Λ_t(r; P, p)      // per Cartesian axis
```
with the standard McMurchie–Davidson recursion (implement it verbatim; `x_PA = P−A`,
`x_PB = P−B`):
```
E_0^{00} = K_AB
E_t^{i+1,j} = (1/(2p))·E_{t−1}^{ij} + x_PA·E_t^{ij} + (t+1)·E_{t+1}^{ij}
E_t^{i,j+1} = (1/(2p))·E_{t−1}^{ij} + x_PB·E_t^{ij} + (t+1)·E_{t+1}^{ij}
E_t^{ij} = 0  for  t < 0  or  t > i+j
```
Because `∫ dr e^{−iG·r} Λ_t(r;P,p) = (−i·G)^t · (π/p)^{3/2} · exp(−G²/(4p)) · exp(−i·G·P)`,
the AO-pair Fourier transform is
```
ft_aopair[μν, G] = Σ_{t,u,v} E_t^{ij,x}·E_u^{ij,y}·E_v^{ij,z}
                   · (−i·G_x)^t·(−i·G_y)^u·(−i·G_z)^v
                   · (π/p)^{3/2} · exp(−|G|²/(4p)) · exp(−i·G·P)
```
Note `K_AB` is carried INSIDE `E_0^{00}`, so it must not be multiplied in a second
time. The one-axis `E` table has `(i+1)(j+1)(i+j+1)` entries; the three axes share
`p`, `K_AB` is put on the x axis only and the y/z tables start from `E_0^{00} = 1`.

**HOST/DEVICE SPLIT**
- **Host** (`mcmurchie.rs`): for every surviving `(shell-pair, image L, primitive
  pair)` triple, build `P`, `p`, `(π/p)^{3/2}`, and the three `E` tables. This is
  `O(nprim² · l⁴)` per pair and independent of `nG` — it must never run on device.
  Flatten into one `f64` upload plus an `i32` index table.
- **Device** (`ft_aopair.rs`): a pure polynomial-times-exponential evaluation. No
  recursion, no branching on `l` beyond the comptime-bounded `t` loop.

**KERNEL DESIGN**
- Thread = one `(cartesian AO pair c, G index g)`. Cube = `(record block, G block)`,
  `CubeDim { x: 256, y: 1, z: 1 }` with `x` running over `g` — a coalesced read of
  `Gv`, which is the only large input.
- The thread loops over the primitive-pair records of its shell pair and accumulates
  `(re, im)` in registers, so there are **no atomics**.
- Output is PLANAR (`D-PBC-02`/RULE 8): two `Array<F>`, `ft_re`/`ft_im`, never
  interleaved.
- Generic over `F: Float` per AGENTS.md §3, launched through
  `pyscf_algebra::dispatch_backend!`, exactly as `pbc/struct_factor.rs` does.
- **All scalar math is `cube-math`** (`/home/user/Documents/workspace/cube-math`):
  `cube_math::double::exp::exp` for `exp(−|G|²/4p)` and
  `cube_math::double::trig::sincos` for `exp(−i·G·P)`, both with
  `cube_math::MathConfig::EXACT`. Do not call `f64::exp`/`sin`/`cos` inside a
  `#[cube]` body — the existing PBC kernels (`struct_factor.rs:46`,
  `ewald.rs:217`, `eval_gto.rs:1235`) set the precedent and the FMA-free
  `release-oracle` gate depends on it.
- Screening: skip a `(shell-pair, L)` whose Schwarz bound
  `K_AB · (π/p)^{3/2}` is below `cell.precision * 1e-2` — the same constant
  upstream's `estimate_rcut` uses, and for the same stated reason (hermitian
  symmetry of the result).

**STEPS**
1. Add `pyscf-kernels` to `crates/pyscf-pbc-df/Cargo.toml`. `pyscf-pbc-gto` already
   depends on it, so `xtask check_dependency_wall` needs no new exemption — run it
   and confirm.
2. `estimate_rcut(cell, precision) -> Vec<f64>` in `ft_ao/mod.rs`, a line-by-line
   port of `ft_ao.py:744-790` **including the two-pass `r0` refinement** and the
   default `precision = cell.precision * 1e-2`.
3. `mcmurchie::e_coefficients(li, lj, a, b, ax, bx) -> Vec<f64>` — the one-axis
   recursion above, indexed `[i][j][t]` flattened row-major.
4. `ft_aopair_prims(...)` — assemble the record table for ONE `(q, single k)` call.
5. The device kernel + `launch_ft_aopair_on_handles` following the
   `struct_factor.rs` shape exactly (core launch on resident handles, `Runtime`
   never escaping into a public signature).
6. Cartesian → spherical: reuse `pyscf_gto::cart2sph_coeff`; apply it as two real
   GEMMs on each of `ft_re`/`ft_im` (D-PBC-03: four real GEMMs, never Karatsuba).
7. This plan ships `q = 0`, one k-point, `aosym = 's1'`, `comp = 1` only. Everything
   else is 13-03.

**TEST** `crates/pyscf-kernels/tests/pbc_ft_aopair.rs`
1. **s-s analytic check (no oracle):** for two s primitives, compare against the
   closed form `(π/p)^{3/2}·exp(−G²/4p)·exp(−iG·P)·K_AB` to 1e-14.
2. **Numerical-FT check (no oracle):** on a dense real-space grid,
   `Σ_r ao_μ(r)·ao_ν(r)·exp(−iG·r)·(vol/ngrids)` matches `ft_aopair` to 1e-6 for a
   `[40,40,40]` mesh, for `l` up to 2. Uses `pyscf_pbc_gto::eval_ao_kpts`, so this
   test also cross-checks Phase 10.
3. **G = 0 identity — RUN IT THREE WAYS.** See §8.5.0's boxed table for why two
   is not enough. `ft_aopair[μν, G=0]` vs the periodic overlap:
   - **1a, screening-faithful** — `rcut = estimate_rcut(cell).max()` (upstream's
     default), reference `pbc_intor("int1e_ovlp")`: **≤ 2e-9**. Upstream itself
     measures **1.554e-9** at gamma and **5.322e-10** at `k ≠ 0`. Do NOT gate this
     at 1e-10; it is not achievable and it is not a defect. This is the `rcut`
     Gate 3 uses.
   - **1b, converged FT against the default overlap** — `rcut = 1.5 × cell.rcut`:
     **≤ 2e-10**. The measured value is 1.472e-10 and it does not improve at
     `2.0 × cell.rcut`, because at that point the residual belongs to
     `pbc_intor`'s own truncation (`cell.rcut` = 21.319, `precision` = 1e-8), not
     to `ft_aopair`. A consistency check with a known floor — NOT a kernel gate.
   - **1c, both sides over ONE image list** — `rcut = 1.5 × cell.rcut` for
     `ft_aopair`, and the reference built with
     `pyscf_pbc_gto::pbc_intor::intor_cross_with_images` over **the same `Ls`**:
     **≤ 1e-13**. **This is the real gate on the McMurchie–Davidson recursion**
     (risk R-08) — it is the only one of the three in which both sides are
     converged over identical images, so nothing but the algebra can move it.
   **All three need no oracle.**
4. **Hermiticity:** `ft_aopair[μν, G] == conj(ft_aopair[νμ, −G])` to 1e-13.
5. Oracle-gated (`PYSCF_ORACLE_VENV`): match
   `pyscf.pbc.df.ft_ao.ft_aopair(cell, Gv)` to 1e-10 on diamond and on He-fcc.

**DONE** `cargo test -p pyscf-kernels --test pbc_ft_aopair`

---

#### Plan 13-02 — `ft_ao`: the single-centre AO Fourier transform

**FILES** `crates/pyscf-gto/src/ft_ao.rs` (molecular — mirrors upstream's own
location), `crates/pyscf-pbc-df/src/ft_ao/mod.rs` (the k-point wrapper),
`crates/pyscf-gto/tests/ft_ao.rs`

**PORT** `pyscf/gto/ft_ao.py` (`ft_ao`, the `GTO_ft_ovlp` single-centre branch) and
`pyscf/pbc/df/ft_ao.py:93-100` (the periodic wrapper).

**THE MATH.** With one centre this collapses out of the MD recursion entirely:
```
ft_ao[μ, G] = Σ_prim  c_prim · (π/α)^{3/2} · exp(−|G|²/(4α))
                     · (Cartesian angular factor in (−iG)) · exp(−i·G·A)
```
The angular factor is the `l = i, j = 0` special case of plan 13-01's `E` table, so
implement it by CALLING `mcmurchie::e_coefficients(l, 0, α, 0, 0, 0)` rather than by
writing a second recursion — one recursion, one place to be wrong.

**THE PERIODIC WRAPPER IS TWO LINES AND BOTH MATTER** (`ft_ao.py:93-100`):
```rust
if gamma_point(kpt) { mol_ft_ao(mol, Gv) } else { mol_ft_ao(mol, Gv + kpt) }
```
There is no lattice sum here. `ft_ao` is evaluated on the *fake* nuclear cell whose
`rcut = 0.1`, so the images never contribute.

**STEPS**
1. `pyscf_gto::ft_ao(mol, gv) -> CTensor` (`nao × nG`, planar).
2. `pyscf_pbc_df::ft_ao::ft_ao(mol, gv, kpt)` — the shift-by-`kpt` wrapper.
3. `_fake_nuc(cell, with_pseudo)` — port `aft.py:247-274` verbatim, including
   `eta = 0.5/rloc²` when the atom has a GTH pseudopotential and `eta = 1e16`
   otherwise, and `norm = half_sph_norm / gaussian_int(2, eta)`. Build it as a real
   `Mole` (`_atm`/`_bas`/`_env` rows) so `ft_ao` consumes it unchanged.

**TEST** `crates/pyscf-gto/tests/ft_ao.rs`
1. s-function closed form to 1e-14.
2. `ft_ao[μ, G=0] == ∫φ_μ` — for `l > 0` that is exactly 0; for `l = 0` it is
   `c·(π/α)^{3/2}`. To 1e-13.
3. Numerical FT on a dense grid to 1e-8, `l` up to 3.
4. Oracle-gated: `pyscf.gto.ft_ao.ft_ao` to 1e-12 on `cc-pVDZ` water.
5. `_fake_nuc(diamond).charge`-weighted `ft_ao` reproduces
   `get_gth_vlocG_part1` × `SI` to 1e-12 — the identity that makes 13-04's
   two `get_nuc` branches consistent.

**DONE** `cargo test -p pyscf-gto --test ft_ao`

---

#### Plan 13-03 — `ft_aopair_kpts`, `FtKernel` dispatch, aosym, G-blocking

**FILES** `crates/pyscf-pbc-df/src/ft_ao/mod.rs`, `crates/pyscf-pbc-df/tests/ft_ao.rs`

**PORT** `ft_ao.py:63-90` (`ft_aopair_kpts`), `:102-250` (`gen_ft_kernel` and the
inner `ft_kernel` closure — port the SHAPE and the `aosym` handling, not the
`libpbc` plumbing).

**STEPS**
1. `struct FtKernel` replaces upstream's closure: it owns the cell, the screened
   `(shell-pair, image)` record table, the uploaded E-coefficient buffer and the
   device client. Building it is the expensive half; `FtKernel::eval(gv, q, kpts,
   shls_slice, aosym)` is the cheap half, called once per G block.
2. **k-resolution.** One kernel launch per k-point, with the Bloch phase
   `e^{i k·L}` folded into the per-image record on the host. Upstream instead
   accumulates into `bvk_ncells` buckets and contracts with `expLk` — that is a
   pure optimisation and is DEFERRED (see D-PBC-21); record it in the module docs
   with the measured cost so Phase 14 can revisit it.
3. **`q` handling.** `GvT = Gv.T + q[:, None]` (`ft_ao.py:163`). `q` is
   `kptj − kpti` and enters ONLY through this shift — never through the lattice sum.
4. **`aosym`.**
   - `s1` — the full `ni × nj` block.
   - `s2` — the packed lower triangle, `nij = i1(i1+1)/2 − i0(i0+1)/2`. Assert
     `shls_slice[2] == 0`, as upstream does.
   - `s1hermi` — gamma point only; assert `is_zero(q) && is_zero(kptjs) && ni == nj`,
     compute the `i ≥ j` half and mirror it (`ft_ao.py:234-236`). Note upstream's
     own comment: it mirrors WITHOUT conjugation because at `q = k = 0` the pair
     density is real-symmetric.
5. **`shls_slice`** as a 4-tuple over the PRIMITIVE cell's shells, with `ni`/`nj`
   taken from `cell0_ao_loc`.
6. **Output layout.** Upstream returns `[nkpts, nGv, ni, nj]` after its
   `np.rollaxis(out, -1, 2)`. This port returns G-major
   (`ft[k][(g, μ, ν)]`, `g` slowest) because every consumer in 13-04/13-05
   contracts over `G` in blocks; the layout is stated once in the module docs and
   never re-derived at a call site.
7. **G-blocking.** `eval` takes a `Gv` slice, so `ft_loop` can walk the mesh in
   blocks sized from `PYSCF_MAX_MEMORY`. The record table is built once, outside
   the loop.

**TEST** `crates/pyscf-pbc-df/tests/ft_ao.rs`
1. **Gate 1, k-resolved:** `ft_aopair_kpts[k, G=0]` vs the periodic overlap for
   diamond at `2×2×2` and He-fcc at gamma, in all three variants of plan 13-01
   test 3 — **1a ≤ 2e-9**, **1b ≤ 2e-10**, **1c ≤ 1e-13** (1c against
   `intor_cross_with_images` over the same `Ls`). No oracle. 1c is the one that
   must hold at every k-point, not just gamma.
2. `s2` unpacks to `s1` element-for-element (bit-identical, not "to 1e-15").
3. `s1hermi` at gamma equals the `s1` result to 1e-13.
4. Block invariance: evaluating the mesh in blocks of 64 gives a **bit-identical**
   result to one whole-mesh call.
5. `q`-shift identity: `ft_aopair_kpts(Gv, q=k)` equals `ft_aopair_kpts(Gv + k, q=0)`
   to 1e-13.
6. Oracle-gated: `ft_ao.ft_aopair_kpts` to 1e-10, diamond `2×2×2`, mesh `[11,11,11]`.

**DONE** `cargo test -p pyscf-pbc-df --test ft_ao`

---

#### Plan 13-04 — `AFTDF`: `build`, `ft_loop`, `pw_loop`, `weighted_coulG`, `get_nuc`, `get_pp`

**FILES** `crates/pyscf-pbc-df/src/aftdf.rs`, `crates/pyscf-pbc-df/tests/aftdf.rs`

**PORT** `pyscf/pbc/df/aft.py:104-165` (`_get_pp_loc_part1`), `:186-234`
(`get_pp`, `get_nuc`), `:236-245` (`weighted_coulG`), `:247-274` (`_fake_nuc`),
`:276-301` (`estimate_ke_cutoff`), `:421-551` (`AFTDFMixin.pw_loop`, `.ft_loop`),
`:585-770` (`AFTDF`).

**STEPS**
1. `struct Aftdf { cell, kpts, mesh, eta, ft_kernel: OnceCell<FtKernel>, … }` and
   `impl PeriodicDf for Aftdf`. `build()` constructs the `FtKernel` and caches it.
2. `estimate_ke_cutoff(cell, precision)` — port `aft.py:276-301` (`_estimate_ke_cutoff`
   with its **two** Newton passes) — and warn, as upstream does, when
   `mesh < mesh_guess * KE_SCALING`. `KE_SCALING = 0.75`; confirm the constant
   against `pyscf/pbc/df/aft.py`'s import before hard-coding it.
3. `weighted_coulG(kpt, exx, mesh, omega)` = `get_coulG(...) * kws`, where `kws`
   comes from the EXISTING `pyscf_pbc_gto::get_gv_weights`. **`exx` is threaded
   through to `get_coulG`** — see §8.5.0's note on the exxdiv divergence.
4. `ft_loop(mesh, q, kpts, shls_slice, aosym) -> impl Iterator<Item = (FtBlock, p0, p1)>`
   and `pw_loop(...)` (the `s2`, real/imag-split, transposed variant `_get_pp_loc_part1`
   uses). Both are thin generators over `FtKernel::eval`.
5. `_get_pp_loc_part1(with_pseudo)`:
   - `with_pseudo = true` → `vpplocG = −Σ_i SI[i,G] · get_gth_vlocG_part1[i,G]`
     (both already exist in `pyscf_pbc_gto::pseudo::vloc`).
   - `with_pseudo = false` → `vpplocG[G] = Σ_i (−Z_i) · ft_ao(fakenuc)[i,G] · coulG[G]`,
     using plan 13-02's `ft_ao` and `_fake_nuc`.
   - Then `× kws`, and contract with the `aosym='s2'` `ft_loop` block exactly as
     `aft.py:141-154` does — **keep upstream's real/imaginary bookkeeping literally**,
     including the `if not is_zero(kpts[k])` guard that leaves `vjI` at zero for a
     gamma k-point.
   - Unpack the triangle (`lib.unpack_tril`) to a full `nao × nao` matrix.
6. `get_nuc(kpts)` = `_get_pp_loc_part1(with_pseudo=false)`.
7. `get_pp(kpts)` = `_get_pp_loc_part1(with_pseudo=true)`
   `+ pyscf_pbc_gto::pseudo::get_pp_loc_part2(cell, kpts)`
   `+ pyscf_pbc_gto::pseudo::get_pp_nl(cell, kpts)`.
   **Both addends already ship** (Phase 10 plans 10-05/10-06). AFTDF and FFTDF
   therefore differ ONLY in part 1 — which is what makes the `get_pp` cross-check
   in 13-08 a clean measurement of `ft_aopair` itself.

**TEST** `crates/pyscf-pbc-df/tests/aftdf.rs`
1. `get_nuc`/`get_pp` are Hermitian to 1e-12 and real at gamma.
2. **Cross-builder, no oracle:** `Aftdf::get_pp` vs `Fftdf::get_pp` on diamond
   `2×2×2` at meshes `[15,21,31,41]`. Assert the deviation **falls until it reaches
   a plateau, then STAYS on that plateau** — measured on upstream, the J/K analogue
   is flat to three digits between mesh 31 and 41 (see plan 13-05's boxed table).
   **Do NOT assert monotone convergence to a small constant; it is false for a
   correct implementation.** What the test must catch is a plateau at the WRONG
   level: pin it against the recorded upstream floor. This is the phase's
   early-warning system, and it must be characterised at mesh ≥ 31 — at mesh 21 the
   general screening residual still masks the effect.
3. `get_nuc` on He-fcc (all-electron — the branch a `gth-pade` cell never reaches)
   vs `Fftdf::get_nuc`, same monotonicity assertion.
4. Oracle-gated: `AFTDF(cell, kpts).get_nuc()` and `.get_pp()` to 1e-11.

**DONE** `cargo test -p pyscf-pbc-df --test aftdf`

---

#### Plan 13-05 — `aft_jk`

**FILES** `crates/pyscf-pbc-df/src/aft_jk.rs`, `crates/pyscf-pbc-df/tests/aft_jk.rs`

**PORT** `pyscf/pbc/df/aft_jk.py:41-94` (`get_j_kpts`, `_update_vj_`), `:96-133`
(`get_j_for_bands`), `:135-293` (`get_k_kpts`), `:295-364` (`get_k_for_bands`),
`:366-418` (`_update_vk_`), `:641-753` (`get_jk`, `_format_dms`).

**PORT THIS ONE STATEMENT BY STATEMENT.** Same warning as plan 11-06 for
`fft_jk.get_k_kpts`: keep upstream's identifiers (`vkR`, `vkI`, `wcoulG`,
`kpti_idx`, `swap_2e`, `k_to_compute`) as Rust identifiers so a reviewer can diff
against the Python line for line.

**THE ONE PLACE AFTDF IS NOT FFTDF — read before writing `get_k_kpts`.**
`aft_jk.py:285` applies the Ewald exchange correction only
`if exxdiv == 'ewald' and cell.low_dim_ft_type == 'inf_vacuum'`. For an ordinary
3-D cell that is FALSE, and the correction instead arrives through
`weighted_coulG(kpt, exxdiv, mesh)` → `get_coulG(..., exx='ewald')`, which adds
`Nk · vol · madelung` at `G+k = 0` (`tools/pbc.py:480-484`). FFTDF in 2.12.1 does
the opposite: no `exx` in `coulG`, and `_ewald_exxdiv_for_G0` applied analytically
afterwards. **The two agree exactly to the extent that `ft_aopair[G=0] == S`** —
which is Gate 1. This is the same difference PySCF **2.14** later imposed on
`fft_jk` (Phase 11 measured it at 1.7e-5 in `vk`), so it is a real, documented
divergence and NOT a bug to "fix". `pyscf_pbc_gto::get_coulg` already implements
the folding (`coulg.rs:198-203`); do not add a second path.

> **MEASURED 2026-08-28/29 — this term DOMINATES the AFTDF/FFTDF difference, and
> it is a first-order gate concern, not a footnote.** Diamond/`gth-szv` 2×2×2,
> fixed init-guess density, `max|vj_AFT − vj_FFT|` / `max|vk_AFT − vk_FFT|`:
>
> | mesh | `exxdiv=None` dvj / dvk | `exxdiv='ewald'` dvj / dvk |
> |---|---|---|
> | 15 | — | 3.827e-7 / 1.112e-6 |
> | 21 | 2.365e-10 / 1.690e-9 | 2.365e-10 / 1.804e-9 |
> | 31 | 1.996e-11 / **2.653e-11** | 1.996e-11 / **6.487e-10** |
> | 41 | — | 1.996e-11 / 6.485e-10 |
>
> Three things to take from this table.
>
> 1. **The difference FLOORS by mesh 31 and does not improve at mesh 41** —
>    1.996e-11 and 6.487e-10 → 6.485e-10, identical to three digits. Above mesh
>    ~31 this is not a mesh effect at all. Any test that asserts "monotone
>    convergence to 1e-13 at a large mesh" will fail, and it will fail for a
>    correct implementation.
> 2. **With `exxdiv='ewald'` (the SCF default) the G=0 term is ~96% of the
>    residual at mesh 31** — 6.2e-10 of 6.487e-10. Turning exxdiv off drops `dvk`
>    by 25×, to 2.653e-11.
> 3. **The mesh-21 row is a trap.** There the general screening residual
>    (1.690e-9) still dominates and the two error sources partially cancel in the
>    max-abs norm, making the exxdiv term look like ~1.1e-10 — negligible. It is
>    not. Do not characterise this effect at mesh 21; use mesh 31 or above.
>
> The mechanism is exact, not empirical. `_ewald_exxdiv_for_G0`
> (`df_jk.py:1480`) builds its correction from `s = cell.pbc_intor('int1e_ovlp')`
> — the **exact** overlap — while AFTDF's folded `coulG[G+k=0]` route effectively
> uses `ft_aopair[G=0] = S + δ`. So
>
> ```
> floor(|vk_AFT − vk_FFT|)  ≈  2 · madelung · ‖S·D‖ · ‖δ‖
> ```
>
> with `madelung = 0.3400910` on this cell and `δ` the **Gate-1a screening
> residual** (1.554e-9). Gate 1 and Gate 2 are therefore coupled quantitatively:
> the only lever on the Gate-2 floor is `rcut`.

**STEPS**
1. `_format_dms` / `_format_jks` / `_format_kpts_band` — reuse
   `crate::df_jk`'s, which plan 11-07 already put in one place. Do not duplicate.
2. `get_j_kpts`: one `ft_loop` pass, `_update_vj_`'s four real `einsum`s expressed
   as `pyscf_algebra` GEMMs on the planar halves.
3. `get_k_kpts`: iterate `kk_adapted_iter(cell, kpts)`. **`kk_adapted_iter` and
   `group_by_conj_pairs` are NOT yet ported** — `pyscf-pbc-lib::kpts_helper` has
   `unique`/`member`/`get_kconserv` but neither of these. Port
   `pyscf/pbc/lib/kpts_helper.py:170-219` and `:221-268` into
   `crates/pyscf-pbc-lib/src/kpts_helper.rs` as part of THIS plan, with their own
   tier-1 tests (every k-point covered exactly once; conjugate pairs are mutual).
4. Time-reversal symmetry: implement the `k_to_compute` mask and the final
   `vk[k_conj] = conj(vk[k])` fill. Upstream disables it when the input density
   matrices break the symmetry by more than 1e-6 — port that check, it is load
   bearing for KUHF.
5. `_update_vk_` only. The four other update variants (`_update_vk1_`,
   `_update_vk_dmf`, `_update_vk1_dmf`, `_update_vk_fake_gamma`) are thread-count
   and MO-factorisation optimisations that produce the same numbers; return
   `NotYetImplemented { phase: 14 }` from any code path that would select them,
   rather than silently taking a different route (D-PBC-20).
6. `get_jk` (single-k) and the two `*_for_bands` entry points.

**TEST** `crates/pyscf-pbc-df/tests/aft_jk.rs`
1. `vj`, `vk` Hermitian to 1e-12; real at gamma.
2. Coulomb-energy symmetry in its two densities (the plan 11-06 identity), no oracle.
3. Explicit `kpts_band` equal to `kpts` reproduces the default path bit-identically.
4. **Cross-builder, pinned to the measured upstream floor** (the boxed table
   above): `Aftdf::get_jk` vs `Fftdf::get_jk` from a fixed init-guess density on
   diamond `2×2×2` at meshes `[15,21,31,41]`. Assert (a) mesh 31 and mesh 41 agree
   to within 1% of each other — the plateau is real and the port sits on it; (b)
   the `exxdiv='ewald'` `dvk` plateau is **~6.5e-10** and the `exxdiv=None` `dvk`
   plateau is **~2.7e-11**, i.e. the port reproduces upstream's 25× exxdiv gap
   rather than accidentally cancelling it; (c) `dvj` plateaus at **~2.0e-11** under
   both. Deviating from these levels means the port's `rcut` differs from
   upstream's — which is a legitimate choice, but it must be a DECLARED one, not a
   surprise.
5. Oracle-gated: `AFTDF.get_jk` to 1e-11.

**DONE** `cargo test -p pyscf-pbc-df --test aft_jk`

---

#### Plan 13-06 — `fft_ao2mo` + `aft_ao2mo`

**FILES** `crates/pyscf-pbc-df/src/fft_ao2mo.rs`, `crates/pyscf-pbc-df/src/aft_ao2mo.rs`,
`crates/pyscf-pbc-df/tests/pbc_ao2mo.rs`

**PORT** `pyscf/pbc/df/aft_ao2mo.py` (all of it) and `pyscf/pbc/df/fft_ao2mo.py`.

**WHY `fft_ao2mo` IS IN THIS PHASE.** It is not in the roadmap text, and Phase 11
skipped it. It costs ~200 lines, it reuses machinery that already ships
(`Fftdf::ao_kpts`, `fft`, `get_coulG`), and it turns 13-06's acceptance from "one
oracle number" into "two independent builders that must agree" — the same
cross-check that makes 13-04 and 13-05 trustworthy. Ship it here.

**STEPS**
1. `get_ao_pairs_G(kpts, q, aosym)` and `get_mo_pairs_G` for both builders.
2. `get_eri(kpts, compact)` — the 4-index `(kp kq | kr ks)` block, momentum
   conservation checked through `pyscf_pbc_lib::get_kconserv`.
3. `general(mo_coeffs, kpts)` — the four-index MO transform.
4. `ao2mo_7d(mo_coeff_kpts, kpts)` — the `[nk,nk,nk,nmo,nmo,nmo,nmo]` tensor
   Phase 15's KMP2 and Phase 16's KCCSD consume. Shape and index order are a
   downstream contract: state them in the module docs and assert them in a test.

**TEST** `crates/pyscf-pbc-df/tests/pbc_ao2mo.rs`
1. 8-fold permutational symmetry of `get_eri` at gamma, to 1e-12.
2. `general` with identity MO coefficients reproduces `get_eri` bit-identically.
3. `ao2mo_7d` slices agree with the corresponding `general` call to 1e-13.
4. **Cross-builder:** `Aftdf::get_eri` vs `Fftdf::get_eri`, monotone in mesh.
5. Oracle-gated: both against upstream to 1e-11.

**DONE** `cargo test -p pyscf-pbc-df --test pbc_ao2mo`

---

#### Plan 13-07 — `Box<dyn PeriodicDf>`: make every K-driver builder-agnostic (D-PBC-22)

**FILES** `crates/pyscf-pbc-df/src/fftdf.rs`, `crates/pyscf-pbc-scf/src/{krhf,kuhf,krohf,kghf,gamma}.rs`,
`crates/pyscf-pbc-dft/src/{krks,kuks,kroks,kgks}.rs`, `crates/pyscf-py/src/bridge.rs`

**THE PROBLEM.** `KRHF { pub with_df: Fftdf }` (`krhf.rs:29`, and the same in
`kuhf`/`krohf`/`kghf` and every Phase-12 KS driver), plus
`pyscf_pbc_df::get_{nuc,pp,hcore}(df: &Fftdf, …)` (`fftdf.rs:271/310/359`). Nothing
downstream can be handed an AFTDF today, so **without this plan Phase 13's gate
cannot even be measured.**

**STEPS**
1. Re-sign `get_nuc`/`get_pp`/`get_hcore` to take `&dyn PeriodicDf`. `get_hcore` is
   the only one with real logic (it adds `int1e_kin` and picks pp vs nuc from
   `cell.pseudo`); the other two become trait-method forwards, with the FFT bodies
   moving to `impl PeriodicDf for Fftdf`.
2. `with_df: Box<dyn PeriodicDf>` in all eight drivers. `KRHF::new` keeps building an
   `Fftdf` (unchanged default, matching upstream); `KRHF::from_df` takes the box.
3. Add `PeriodicDf: Send + Sync` if — and only if — the drivers are used across
   threads today; check before widening the bound.
4. `Fftdf` and `Aftdf` both get a `fn name(&self) -> &'static str` for `dump_flags`
   and chkfile provenance.
5. `pyscf-py`: `mf.with_df = AFTDF(cell, kpts)` must work from Python. Phase 20 owns
   the full `pyscf.pbc.df` surface — here, only wire the assignment and the
   subclass-override dispatch Phase 3 established.

**TEST** `crates/pyscf-pbc-scf/tests/df_swap.rs`
1. `KRHF` built with an explicitly constructed `Fftdf` gives a **bit-identical**
   energy to `KRHF::new` (proves the box changed nothing).
2. The same for `KUHF`, `KROHF`, `KGHF`, `KRKS`, `KUKS`.
3. `KRHF` with an `Aftdf` converges and reports `with_df.name() == "AFTDF"`.

**DONE** `cargo test -p pyscf-pbc-scf --test df_swap && cargo test -p pyscf-pbc-dft`

---

#### Plan 13-08 — Verification rollup

**FILES** `.planning/phases/13-ft-ao-aftdf/13-VERIFICATION.md`, `.planning/ROADMAP.md`,
`.planning/STATE.md`

**THE THREE GATES, AND WHAT EACH ACTUALLY MEASURES**

| # | Gate | Oracle? | What a failure means |
|---|---|---|---|
| 1a | `ft_aopair[G=0] == pbc_intor("int1e_ovlp")` to **≤ 2e-9** at `rcut = estimate_rcut` (upstream measures 1.554e-9 / 5.322e-10) | no | the screening does not match upstream's |
| 1b | the same identity to **≤ 2e-10** at `rcut = 1.5 × cell.rcut` (measured floor 1.472e-10, set by `pbc_intor`'s OWN truncation) | no | the FT lattice sum is not converged |
| 1c | `ft_aopair[G=0]` vs `intor_cross_with_images` over the **same `Ls`**, to **≤ 1e-13** | no | the MD recursion, the contraction, or the cart→sph transform is wrong — **this is the kernel gate** |
| 2 | AFTDF `KRHF` == FFTDF `KRHF`, converged in mesh | no | see below |
| 3 | AFTDF `KRHF`/`get_nuc`/`get_pp`/`get_jk`/`get_eri` vs upstream 2.12.1 to **1e-11** | yes | anything |

**Gate 2 is not a mesh sweep. It is a `(rcut, mesh)` surface with two floors.**
The roadmap says 1e-13; the master plan's original 13-06 said 1e-6; the first
rewrite of this section said "monotone convergence to 1e-13 at the largest mesh".
**All three are wrong, and the third is wrong in a way that would fail a correct
implementation.** Measured on upstream 2026-08-28/29, diamond/`gth-szv`/`gth-pade`
`2×2×2`, fixed init-guess density:

| mesh | `dvj` | `dvk` (`exxdiv='ewald'`) | `dvk` (`exxdiv=None`) |
|---|---|---|---|
| 15 | 3.827e-7 | 1.112e-6 | — |
| 21 | 2.365e-10 | 1.804e-9 | 1.690e-9 |
| 31 | 1.996e-11 | 6.487e-10 | 2.653e-11 |
| 41 | 1.996e-11 | 6.485e-10 | — |

and, at fixed mesh 31, scaling `ft_ao.estimate_rcut`:

| `rcut` | Gate 1 residual | `dvj` | `dvk` |
|---|---|---|---|
| ×1.0 = 20.42 | 1.554e-9 | 1.996e-11 | 6.487e-10 |
| ×1.5 = 30.63 | 1.472e-10 | 7.727e-13 | 1.609e-10 |
| ×2.0 = 40.84 | 1.472e-10 | 7.726e-13 | 1.609e-10 |

**Read both tables together and the structure is unambiguous.**

- Down the first table (fix `rcut`, raise the mesh): everything **plateaus at mesh
  31** and mesh 41 is identical to three digits. Above mesh ~31 the mesh is not the
  controlling parameter at all.
- Down the second (fix the mesh, raise `rcut`): everything plateaus at
  **`rcut` ≈ 1.5 × cell.rcut** and ×2.0 is identical to four digits.
- So there are **two independent floors** — AFTDF's `rcut` screening and FFTDF's
  mesh aliasing — and lowering either one alone stalls against the other. At
  (×1.5, mesh 31) `dvj` reaches 7.7e-13 while `dvk` stops at 1.6e-10, which is
  FFTDF's mesh-31 aliasing and no `rcut` can touch it.
- The `exxdiv` column is the other half of the story: with `exxdiv='ewald'` — the
  SCF default — the G=0 asymmetry of risk R-15 is **~96% of `dvk` at mesh 31**
  (6.2e-10 of 6.487e-10), and switching it off drops `dvk` 25×. **Do not
  characterise any of this at mesh 21**: there the general screening residual still
  dominates, the two error sources partially cancel in the max-abs norm, and the
  exxdiv term misleadingly looks like ~1.1e-10.

**Therefore state Gate 2 as a paired ladder, not a sweep:**

> On diamond/`gth-szv`/`gth-pade` at `2×2×2`, with both SCFs at `conv_tol = 1e-12`,
> `|E_AFTDF − E_FFTDF|` is measured on the **pairs** `(rcut, mesh)` =
> `(×1.0, 21)`, `(×1.0, 31)`, `(×1.5, 31)`, `(×1.5, 41)`. It must fall as the pair
> tightens and then sit on the recorded upstream floor for that pair — the port
> reproduces upstream's floor, it does not beat it and it does not miss it.

**The upstream energy series measured so far** (mesh ≥ 31 rows were still running
when this section was written — plan 13-08 Task 0 completes the table and pastes it
into `13-VERIFICATION.md` §1):

| mesh | `E_FFTDF` | `E_AFTDF` | `|ΔE|` |
|---|---|---|---|
| 15 | −10.93090153682113 | −10.93087523588556 | 2.630e-5 |
| 21 | −10.93087319091901 | −10.93087316555834 | 2.536e-8 |
| 31 | −10.93087316795859 | −10.93087316798466 | **2.607e-11** |
| 41 | −10.93087316795859 | −10.93087316798466 | **2.607e-11** |

**Mesh 31 and mesh 41 are BIT-IDENTICAL — all 14 printed digits of both
energies.** The energy floor at upstream's default `rcut` is **2.607e-11 Ha**, and
the roadmap's 1e-13 is unreachable — three orders out. Meshes 55 and 71 add
nothing and should not be run; the ladder stops at 41. Note
it lands with `dvj`'s plateau (1.996e-11) and the `exxdiv=None` `dvk` plateau
(2.653e-11), NOT with the `exxdiv='ewald'` `dvk` plateau (6.487e-10): the G=0
asymmetry is a near-uniform shift and largely cancels in `Tr(D·vk)`, so it
dominates the MATRIX difference while contributing little to the ENERGY. Gate the
matrices and the energy at different levels; do not carry one number across both.
Correct the ROADMAP to 2.607e-11 at `(×1.0, 31)`, exactly as Phase 12 §1d did for
the proposed 1e-15 gate.

The measurement scripts and the full recorded results are committed at
`.planning/phases/13-ft-ao-aftdf/measurements/` (`README.md` + four scripts);
re-run them rather than re-deriving them.

**Also record**
- the `get_pp` cross-builder convergence table from 13-04 test 2 (this isolates
  `ft_aopair` from the SCF);
- the wall-clock cost of `FtKernel::build` and of one `ft_loop` pass at the default
  mesh, so Phase 14's decision about the BvK bucket contraction (D-PBC-21) has a
  number behind it;
- whether `xtask check-orphan-modules` and `check_dependency_wall` stayed green —
  Phase 12's lesson (its source had never been compiled) applies to every new
  module this phase adds.

**DONE** `cargo test --workspace` green; `13-VERIFICATION.md` written; ROADMAP
Phase 13 ticked; `STATE.md` advanced.

---

### 8.6 Phase 14 — GDF / MDF / RSDF / RSJK

**Goal:** production density fitting — the builders real solid-state calculations use.

**Plans:** 9 (14-01 … 14-09).

| Plan | Content | Port from | Notes |
|---|---|---|---|
| **14-01** | `incore`: `aux_e2`, `fill_2c2e`, `_Int3cBuilder` over lattice images (`int3c2e` with a *cell* and an *auxcell*, summed over `L` with Bloch phases) | `df/incore.py` (731 l) | Reuses `pyscf_gto::intor_with_auxmol("int3c2e_sph")` per image pair, same pattern as plan 10-03 |
| **14-02** | `gdf_builder`: `_CCGDFBuilder`, `get_2c2e`, `get_3c2e`, the compensating-charge (Gaussian-nuclear-model) scheme, `weighted_ft_ao` | `df/gdf_builder.py` (1062 l) | The Coulomb divergence is handled by subtracting a smooth compensating charge whose FT is analytic |
| **14-03** | `GDF` class: `build`, `_make_j3c`, `sr_loop`, `get_naoaux`, HDF5 `_cderi` on-disk store | `df/df.py` (1029 l) | HDF5 via `hdf5-metno`, same RAII-spill pattern as `pyscf-ccsd` |
| **14-04** | `df_jk`: `get_j_kpts`, `get_k_kpts`, `get_jk`, `_ewald_exxdiv_for_G0`, `_format_dms/_format_jks/_format_kpts_band` | `df/df_jk.py` (1552 l) | `_ewald_exxdiv_for_G0` is shared by FFTDF/AFTDF — **move it here in 11-07 and re-export**, do not duplicate |
| **14-05** | `df_ao2mo`, `outcore` — **and Phase 13's `ao2mo_7d` carry-over, closed for all three builders** | `df/df_ao2mo.py` (351 l), `df/outcore.py` (250 l) | SHIPPED. `ao2mo_7d`'s index order is now a fixed contract: `eri[ki,kj,kk][i,j,k,l]`, `kl = kconserv[ki,kj,kk]`, chemists' notation. `outcore` blocks over the k-point pair, not the auxiliary shell (stated deviation). |
| **14-06** | `MDF` (mixed density fitting = GDF + AFT residual) | `df/mdf.py`, `mdf_jk.py`, `mdf_ao2mo.py` (785 l) | SHIPPED as `_CCMDFBuilder`, composed over 14-02's `make_j3c` with a `Scheme` tag rather than a second driver. `_RSMDFBuilder` is blocked by D-PBC-24. It caught the phase's largest defect — `decompose_j2c` read `zeigh_gen`'s COLUMN-MAJOR eigenvectors row-major, worth **6.3e6 Ha**, on a branch no earlier gate had reached. |
| **14-07** | `rsdf_helper` + `rsdf_builder` (range-separated GDF) | `df/rsdf_helper.py` (1348 l), `rsdf_builder.py` (1631 l) | **7a SHIPPED** (all twelve ω estimators + `weighted_coulG_LR/_SR` + `_gaussian_int`, gated at 1e-12 against `measurements/omega.out`). **7b/7c/7d BLOCKED — D-PBC-24**, the cintx `range_omega` (`env[8]`) gap. Task 7d's flip of `GDF._prefer_ccdf` does not happen. |
| **14-08** | `RSDF` class + `rsdf_jk`; `scf/rsjk.py` (range-separated JK build, no DF) | `df/rsdf.py` (680 l), `rsdf_jk.py`, `scf/rsjk.py` (1355 l) | `get_aux_chg` and ONE shared `density_fit` for all four shims SHIPPED. `RSGDF` and `rsjk` BLOCKED — D-PBC-24. `rsjk` lives in `pyscf-pbc-scf` and is deliberately NOT a `PeriodicDf`: it has no `cderi` and no auxiliary count, so the impl would have to lie in two methods. |
| **14-09** | Verification rollup | — | SHIPPED — `14-VERIFICATION.md`. **This row's stated gate was wrong too**: "the same `KRHF` energy to 1e-6" softens the ROADMAP's 1e-15 but keeps its category error. GDF is an APPROXIMATION; `\|E_FFTDF − E_GDF\|` on diamond 2×2×2 is **1.222e-03**, and upstream's own two GDF builders disagree by up to **4.502e-06**. The five gates that replace it are in `14-CONTEXT.md`: **1** (MET, 2.750e-10 vs upstream on the all-electron control), **1b** (PARTIAL — diamond's `make_j3c` wall time is unmeasured), **2** (MET, MDF's mesh ladder), **3** (UNREACHABLE — D-PBC-24), **4** (MET, 6.08 % at a PINNED 2×2×2; the same system is 20.50 % at 3×3×3, which is why the mesh is pinned). |

---

### 8.7 Phase 15 — Periodic AO2MO + KMP2

**Plans: 8** (~~5~~). **Implemented 2026-09-05; verification CLOSED 2026-09-05.**
The original five-plan table mixed already-shipped work with Phase 17's
`ktensor`; the corrected scope and rationale are in `15-CONTEXT.md`. Three
corrections, all made BEFORE any code: `kconserv`/`kconserv3` shipped in 09-07,
most of the original 15-02 shipped in 14-05, and `KUMP2`'s energy kernel does
not exist upstream at all. `ktensor` moved to §8.9 (Phase 17), where it shipped.

| Plan | Content | Status |
|---|---|---|
| **15-01** | Measure anchor, per-route energies, padding, and timing before setting gates | measured core matrix; oversized optional matrix omitted |
| **15-02** | `KptsHelper`, ordered `symm_map`, operations and transforms | shipped; oracle-gated exactly, `[1,1,2]` and `[2,2,2]` |
| **15-03** | restricted frozen/padding bookkeeping and the SCF→AO2MO layout seam | shipped; oracle-gated exactly on all four `frozen` forms |
| **15-04** | `PeriodicDf` AO2MO dispatch, `oracle_zdotu`, Lov, legacy wrappers | shipped |
| **15-05** | restricted KMP2, T2, RDM1/RDM2, deterministic reductions | shipped; FFTDF `5.4e-11`/`2.7e-11`, inherited GDF gap open |
| **15-06** | KUMP2 bookkeeping/refusal and staggered KMP2 | shipped; the stagger energy oracle now runs and **caught three defects** |
| **15-07** | verification rollup and opt-in oracle harness | **complete** — nine-part matrix green, `15-VERIFICATION.md` |
| **15-08** | MO-first FFTDF/AFTDF transform and reusable Coulomb caches | shipped; measured **9.784×** over the AO-first route |

The original `1e-8` here and `1e-14` in ROADMAP were unmeasured guesses. The
measured headline gate is **2e-6 Ha per DF route**, and the FFTDF route clears
it by five orders (`5.418e-11` on the diamond anchor, `2.719e-11` on He/6-31g).
The GDF route does **not** clear it (`1.289e-1` on diamond, `1.417e-3` on He)
and that is Phase 14's carry-over, not KMP2's: `Lov` and forced AO2MO agree
within `2e-15 Ha` on the same mean field, and upstream's own two routes agree
to `6.9e-13` on its.

**Key formula (KMP2):**
```
e_corr = (1/nkpts) Σ_{ki,kj,ka} Σ_{ijab}
         (ia|jb)·[2·(ia|jb)* − (ib|ja)*] / (ε_i^{ki} + ε_j^{kj} − ε_a^{ka} − ε_b^{kb})
```
with `kb` fixed by momentum conservation `kconserv[ki, ka, kj]`.

---

### 8.8 Phase 16 — Periodic Coupled Cluster + CI

**Plans: 14** (~~10~~ — the original ten-plan table was an undercount and was
wrong about the starting state in seven ways; see `16-CONTEXT.md §1`, corrected
2026-09-02, and `16-REVIEW.md`). Still the largest phase by line count
(13,675 + 852 upstream). **`16-01` MUST run before any plan below writes a line
of Rust** — its measured gates replace the placeholder numbers this section and
`§7` carried before it.

**Phase 16 is HARD-BLOCKED on Phase 15** (`16-CONTEXT §1.1`): all nine k-point
CC/CI modules import `padding_k_idx` / `padded_mo_coeff` / `padded_mo_energy` /
`get_nocc` / `get_nmo` / `get_frozen_mask` from `pbc.mp.kmp2` / `kump2`, and
`crates/pyscf-pbc-mp` is a 13-line stub. Waves 0 (16-01/02/03) have no such
dependency and start immediately; wave 1 onward waits, and defers explicitly
rather than reimplementing the padding surface — the same ruling 17-09 made.

| Plan | Content | Port from |
|---|---|---|
| **16-01** | Measure the floor (`e_corr`, EOM roots, (T), the DF-route split, the storage-tier crossover, the `symm_map` ratio); restate every gate in four documents | — (measurement only, no Rust) |
| **16-02** | Substrate: `KptsHelper::build_symm_map`/`transform_symm`; the **complex** tensor arena `ZWorkspacePool`; `KTensor` (**costed at zero by the original table** — see D-PBC-29) | `lib/kpts_helper.py:544-630` |
| **16-03** | `davidson_nosym1` + `pick_real_eigs` — iterative NON-symmetric Davidson (**omitted entirely**; four plans are dead without it) | `lib/linalg_helper.py:741` |
| **16-04** | `kintermediates_rhf` at k | `cc/kintermediates_rhf.py` (926 l) |
| **16-05** | `KRCCSD`: `_ERIS` (3 storage tiers + `symm_map`), `update_amps`, `energy`, `kernel` | `cc/kccsd_rhf.py` (1203 l) |
| **16-06** | `kintermediates_uhf` + `KUCCSD` (kernel EXISTS upstream, unlike `KUMP2`) | `cc/kintermediates_uhf.py` (1225 l), `cc/kccsd_uhf.py` (1116 l) |
| **16-07** | `KGCCSD` + `kintermediates` + **the narrow molecular `gccsd` surface it inherits** (`kccsd.py:332`/`:339`/`:352`/`:477`) | `cc/kccsd.py` (833 l), `cc/kintermediates.py` (529 l), + `cc/gccsd.py` (partial) |
| **16-08** | `KCCSD(T)`: `kccsd_t`, `kccsd_t_rhf`, gated against **`kccsd_t_rhf_slow` — the file the original table omitted** (`kccsd_t_rhf.py:236` runs on the C kernel `CCsd_zcontract_t3T`, which this port has not got) | `cc/kccsd_t.py` (319 l) + `kccsd_t_rhf.py` (651 l) + `kccsd_t_rhf_slow.py` (271 l) |
| **16-09** | `EOM-KCCSD-GHF` IP/EA/EE + **the narrow molecular `eom_rccsd` base** — **REORDERED FIRST**: `eom_kccsd_rhf.py:25` and `eom_kccsd_uhf.py:29` both inherit from it | `cc/eom_kccsd_ghf.py` (2011 l), + `cc/eom_rccsd.py` (partial) |
| **16-10** | `EOM-KCCSD-RHF` IP/EA + EE-**singlet**; Triplet/SpinFlip ship as upstream's shells with upstream's refusals | `cc/eom_kccsd_rhf.py` (1716 l) + ip/ea (158 l) |
| **16-11** | `EOM-KCCSD-UHF` IP/EA; **EE refuses — upstream has no `EOMEE` class and `_IMDS.make_ee` (`:1120`) raises** | `cc/eom_kccsd_uhf.py` (1275 l) |
| **16-12** | `kuccsd_rdm` + the Γ-point `pbc/cc/ccsd.py` shim (both halves of the `exxdiv`/Madelung treatment) | `cc/kuccsd_rdm.py` (157 l), `cc/ccsd.py` (157 l) |
| **16-13** | `KCIS` (k-point CI **singles**); **`pbc/ci/cisd.py` DEFERRED EXPLICITLY** — it is a Γ-only shim over molecular RCISD/UCISD/GCISD, and this port has no molecular CI crate | `ci/kcis_rhf.py` (700 l) |
| **16-14** | Verification against the restated gates + re-measurement of D-PBC-29's four claims | — |

**Reuse note — CORRECTED.** ~~`pyscf-ccsd` already owns the molecular
`WorkspacePool` tensor arena and `PYSCF_MAX_MEMORY` pre-flight refusal.
`pyscf-pbc-cc` MUST reuse both.~~ Three corrections (`16-CONTEXT §1.3`,
`16-REVIEW §2`): the arena is in **`pyscf-runtime`**, not `pyscf-ccsd`;
`pyscf-ccsd` contains **zero** complex arithmetic (`grep -c "Complex64" ` is 0
on every file) while every k-point CC tensor is `complex128`; and the pool is
f64-typed all the way down (`shape_bytes = product * 8`,
`InMemory(Box<[f64]>)`, `as_slice -> Vec<f64>`). What is reusable is the
**shape** — budget ceiling, free-list, `InMemory | Spilled`, the HARD refusal
with no silent downgrade. See **D-PBC-29**.

**Droppable half if the phase overruns:** 16-09/10/11/13 (EOM + KCIS). Nothing
in Phases 17-20 needs excited states for correctness, and they are ordered
last. Do **not** drop 16-05 or 16-07 instead — §8.9 blocks 17-09's CC half on
`KRCCSD` by name, and `scf.kghf.KGHF.CCSD` (`kccsd.py:805`) is a surface
Phase 19 reads.

### 8.9 Phase 17 — k-point symmetry + multigrid

**Plans:** 13 (§8.9's original eight was an undercount, not a compression —
see `17-CONTEXT.md §1`, corrected 2026-08-31, and `17-01`'s measurements,
2026-09-01). `17-01` MUST run before any of the plans below writes a line of
Rust — its measured gates (`.planning/phases/17-ksymm-multigrid/measurements/README.md`)
are what 17-02 through 17-13 are held to, replacing every placeholder number
that was in this table before.

| Plan | Content | Port from |
|---|---|---|
| **17-01** | Measure the floor; fixtures; restate every gate (Gates A-E) in `ROADMAP.md`, this file and `17-CONTEXT.md` together | — (measurement only, no Rust) |
| **17-02** | `symm/geom.py` + `tables.py` + `group.py` (`PGElement`, `PointGroup`, `Representation`); records D-PBC-25 (crate-layering ruling) | `symm/geom.py`, `tables.py`, `group.py` (821 l) |
| **17-03** | `symm/space_group.py` + `symmetry.py` (`Symmetry` base); `Cell` symmetry fields, `symmetrize_mesh`, `build_lattice_symmetry` | `space_group.py` + `symmetry.py` (717 l) |
| **17-04** | `symm/basis.py` — `symm_adapted_basis`, `Cell::_build_symmetry` (**omitted from the original 8-plan table**; the default `use_ao_symmetry=True` SCF path needs it) | (161 l) |
| **17-05** | `lib/kpts.py` `KPoints`: `make_kpts_ibz`, `stars`, `bz2ibz`, `addition_table`, `inverse_table`, `get_kconserv`, `transform_*` (dm/fock/mo_coeff/mo_energy); closes `make_kpts_with_symmetry`; `is_trim` | `lib/kpts.py` (1223 l) |
| **17-06** | `lib/ktensor.py` `KsymmArray` (**moved here by `15-CONTEXT §1.1`**; the original table never recorded the move) | (386 l) |
| **17-07** | `khf_ksymm`/`kuhf_ksymm`/`kghf_ksymm` as `KOverrideHooks` impls; records D-PBC-26 (the IBZ-restricted `get_jk` fast path — see below) | (840 l) |
| **17-08** | `krks_ksymm`/`kuks_ksymm`/`krkspu_ksymm`/`kukspu_ksymm` | (422 l) |
| **17-09** | `kmp2_ksymm` (+ `kccsd_rhf_ksymm` if Phase 16 has shipped `KRCCSD` by then — otherwise defer the CC half explicitly, do not guess) | 285 l (+1071 l if Phase 16 has landed) |
| **17-10** | `ft_ao._RangeSeparatedCell` + `ExtendedMole` (**omitted from the original 8-plan table entirely**; closes `exclude_dd_block`/`strip_basis`/the Phase-13 `ft_aopair` residual for Phase 14; independent track, no shared code/fixture/gate with the rest of the phase, can run in parallel from wave 1) | ~600 l of `ft_ao.py` |
| **17-11** | multigrid **v1** — `multigrid.py`, 2 cubecl kernels (`NUMINT_fill`, `NUMINT_rho_drv`) | 1962 l |
| **17-12** | multigrid **v2** — `multigrid_pair.py` + `pp.py` + `utils.py`, 12 cubecl kernels via `_backend_c.py`; this is the half Phase 18's `grad/rhf.py`/`grad/uhf.py` assert `isinstance(ni, MultiGridNumInt2)` on | 1257+256+70 l |
| **17-13** | Verification against the restated gates (Gates A-E, D-PBC-26's fast-path equivalence, the multigrid speed corollary) | — |

**Multigrid is the droppable half if the phase overruns** — it is a speed
feature (17-01 measured upstream's OWN multigrid as SLOWER than reference
`numint` on this repo's reference-system scale: 0.18-0.49x, see below), and
17-11/17-12 are ordered last so dropping them costs nothing already built. Do
**not** drop 17-10 instead — seven live Rust sites' `NotYetImplemented { phase: 17 }`
payloads point at it by number.

**17-10 status (2026-09-01): Tasks 1/2/3/5 SHIPPED, Task 4 CARRIED OVER.**
`RsCell`/`ExtendedMole` exist and are gated (D-PBC-27); `exclude_dd_block`'s
refusal is closed on both `CcGdfBuilder` and `RsGdfBuilder` — setting it
`true` builds and runs a correct, tested result — but the CRATE'S DEFAULT
was deliberately left at `false` rather than flipped to match upstream, because
flipping it touches every pre-existing test that builds a
`CcGdfBuilder`/`RsGdfBuilder`/`Gdf` without naming the flag (several gated
tighter than D-PBC-23's own ~1e-8 deltas), and a full-suite regression run
to confirm that is safe did not finish inside 17-10's time budget. He-fcc's
zero-cost claim holds bit-identically and is CONFIRMED green; diamond's two
Ha-level numbers have a written, `#[ignore]`d oracle test whose live run did
not finish this session (the oracle/target side was independently
re-confirmed; this port's own number was not). **NOT closed**: the two
band-k-point `_cderi`-rebuild refusals (`gdf/jk.rs:243-253`,
`mdf/mdf_jk.rs:80-90`) and the MO-factorised `get_k_kpts` (no
`force_dm_kbuild`-equivalent parameter exists yet) — `grep -rn "phase: 17"
crates/pyscf-pbc-df/` still returns two lines. A follow-on plan should (a)
run the diamond oracle test to completion and, if it passes, run a full
`cargo test -p pyscf-pbc-df` with the default flipped before committing to
that flip, and (b) size closing the two band-k-point refusals (asymmetric
`kpts`-vs-`kpts_band` contraction in `get_j_kpts`/`get_k_kpts`), separately
from `rsjk` below. Full detail: `17-10-SUMMARY.md`.

**`rsjk` sizing, recorded by 17-10 Task 5**: 17-10 closed `rsjk`'s FIRST of
two blockers (the `RsCell`/`ExtendedMole` supermole) but did not — and could
not, on its own — unblock `rsjk` itself. Blocker 2, a screened periodic
4-centre `int2e` driver (`PBCVHF_direct_drv1`, `rsjk.py:267-436`), has no
implementation anywhere in this workspace and, because `rsjk` is EXACT, no
correct-but-slow fallback can stand in for it (a wrong answer would land
inside a correct GDF's DF-fitting error and look plausible). **`rsjk` should
be sized as its own plan, in a phase after 17** — `crates/pyscf-pbc-scf/src/rsjk.rs`
still refuses and must keep refusing until that plan lands.

**`spglib` note (hardened by 17-CONTEXT §1.5):** the native detection
(17-02/17-03) IS upstream's own default (`space_group.py:264`); spglib is an
opt-in `backend='spglib'` upstream reaches only when asked, and it cannot
serve `cell.dimension < 3` at all (`space_group.py:288-290`), so it cannot
cover the `graphene` reference system. **Recommendation: do not ship the
`spglib` feature in v2.0** — record it as a v2.1 nicety, not a required
dependency.

**D-PBC-26 (the `get_jk` fast path, recorded by 17-07):** upstream's own
`khf_ksymm.get_jk` runs the DF build at the FULL `nkpts`, not `nkpts_ibz` —
for Si `[16,16,16]`, `145` vs `4096`, a 28x gap upstream leaves on the table.
17-07 adds a second, faster route (call the DF builder only at `kpts_ibz`,
then unfold with `transform_1e_operator`), validated against the port's own
reference route at 1e-13 (not against upstream). **17-01 measured the
achievable bound**: on `si [4,4,4]` (`nkpts=64`, `nkpts_ibz=8`), a full-BZ
vs IBZ-subset `get_jk` wall-clock ratio came out at **223x (FFTDF)** /
**40x (GDF)** — both well above the naive `nkpts/nkpts_ibz=8x` estimate,
because exact-exchange cost scales closer to quadratically than linearly in
`nkpts`. Target the GDF number (~40x) as the realistic floor since GDF is the
port's default route.

> ### ERRATUM — D-PBC-26 point 1 is WRONG. Recorded 2026-09-02, MEASURED.
>
> Plan item S-02 of
> [`KUKS-KSYMM-MULTIGRID-OPTIMISATION-PLAN.md`](./KUKS-KSYMM-MULTIGRID-OPTIMISATION-PLAN.md)
> §2.2.3, measured by
> `crates/pyscf-pbc-scf/tests/khf_ksymm.rs::ibz_only_get_jk_is_not_an_identity`.
>
> **Calling the DF builder at `kpts_ibz` only does not compute the same `vj`
> or `vk`.** `get_j_kpts` forms `rho(r) = (1/N) Σ_{k ∈ list}` over whatever
> k-list it is handed; over the IBZ list that is `Σ_{k∈IBZ} rho_k / N_ibz`,
> while the true density is `Σ_{k∈IBZ} w_k <rho_k>_star` — and `rho_k(r)` is
> NOT point-group invariant (`rho_{Rk}(r) = rho_k(R^-1 r)`). The two agree
> only when every star has one member. Unfolding the RESULT with
> `transform_1e_operator` rotates a potential built from the wrong density; it
> does not repair it. For `K` the argument is sharper: restoring the dropped
> `k2` terms by equivariance needs `K` evaluated at every `R^-1 k1`, which for
> `k1` over the IBZ is the whole zone again.
>
> **Measured on `si [2,2,2]`** (stars `[1, 3, 4]`, so the fixture can see the
> difference): `max |d veff|` between the IBZ-only route and the reference
> route is **9.486e-2 Ha** — not a tolerance question.
>
> **So the 223x / 40x above compared two DIFFERENT QUANTITIES**, and neither
> number is a speed target. The attainable bound is `nkpts / nkpts_ibz`, and
> it is reached exactly — **bit-identically**, measured `max |d| = 0e0` — by
> restricting the OUTPUT set instead of the sampling set, i.e. `kpts_band =
> kpts_ibz`. The exchange sum still runs over every `k2` (that is the
> physics); only the free output index `k1` is restricted, which is
> `nkpts · nkpts_ibz` pairs instead of `nkpts²`. On `si [2,2,2]` that is
> **64 -> 24 pairs, 2.67x fewer**, and the reference route was computing those
> extra output points only to discard them in `fold_to_ibz`.
>
> `JkRoute::Fast` is renamed `JkRoute::IbzOnly` and kept non-default solely so
> this measurement stays reproducible; `JkRoute::Band` is the correct route.
> Points 2, 4, 5 and 6 of the ruling are unaffected — and the DFT k-symmetric
> adapters have taken the band route since 17-08 Task 2, so point 4 was
> already true.

---

### 8.10 Phase 18 — Periodic gradients + stress + geomopt

**Plans:** 7.

| Plan | Content | Port from |
|---|---|---|
| **18-01** | Gradient integral infrastructure: `pbc_intor` derivative families (`int1e_ipovlp`, `int1e_ipkin` — both cintx-ready) over lattice images; `vpploc_part2_nuc_grad` (**needs cintx `int3c1e_ip1_r{2,4,6}_origk`**), `vppnl_nuc_grad` (**needs cintx `int1e_r{2,4}_origi_ip2`**), `ewald_nuc_grad` | `pp_int.py:171-210, 300-407, 443-510`, `ewald_methods.py:101-122, 256-292` |
| **18-02** | `grad/krhf` (418 l) + `grad/rhf` gamma (188 l) | |
| **18-03** | `grad/kuhf` (124 l), `grad/uhf` (103 l) | |
| **18-04** | `grad/krks` (141 l), `grad/kuks` (135 l), `grad/rks`, `grad/uks`, `grad/krkspu`, `grad/kukspu` | |
| **18-05** | **Stress tensor**: `krks_stress` (404 l), `kuks_stress` (308 l), `rks_stress` (462 l), `uks_stress` (246 l) | Includes the lattice-vector derivative `∂E/∂a_ij` |
| **18-06** | `pbc/geomopt/geometric_solver` (246 l) — reuse `pyscf-geomopt`'s native BFGS+RFO engine; add lattice degrees of freedom | |
| **18-07** | Verification: `verify_fd` central-difference gate at 1e-6 Ha/Bohr for every gradient body; stress vs finite-difference of `E(a)` at 1e-5 Ha | |

**cintx posture for this phase (verified 2026-08-22 — supersedes the v1.0 Phase-7 audit):**
Every *gradient* integral family this phase needs already ships in cintx and is
oracle-covered: `int1e_ipovlp`, `int1e_ipkin`, `int3c2e_ip1`, `int3c2e_ip2`,
`int3c1e_ip1`, `ECPscalar_ipnuc`, `ECPscalar_iprinv`, plus `Builder::with_rinv_origin`.
**Phase 18 is NOT structurally-gated on a cintx gradient workstream** — that framing
came from `07-01-PLAN.md:46-48` and is stale.

Two further facts shrink this phase's integral surface considerably:
  * **FFTDF J/K gradients are grid-based, not integral-based.** `get_j_e1_kpts` /
    `get_k_e1_kpts` (`pyscf/pbc/df/fft_jk.py:113`, `:310`) differentiate the AOs on the
    real-space grid via `eval_ao_kpts(deriv=1)` and reuse the same FFT/`coulG` pipeline
    as the energy. **`int2e_ip1` is never called on the FFTDF path.**
  * **Stress needs no derivative integrals at all.** `krks_stress.py:95-112` obtains
    `∂S/∂ε` and `∂T/∂ε` by finite-differencing `pbc_intor('int1e_ovlp'/'int1e_kin')`
    between two strained cells, and computes the `coulG` / grid-weight / AO strain terms
    analytically.

The remaining cintx dependency is the PP-gradient half of §2.4:
`int3c1e_ip1_r{2,4,6}_origk` (`pp_int.py:187`) and `int1e_r{2,4}_origi_ip2`
(`pp_int.py:454`). These are cintx Wave 0.5. If they have not landed, plan 18-01 ships
the assembly and `#[ignore]`-gates only the PP-gradient contribution — the rest of the
phase, including `verify_fd`, runs.

**Multigrid dependency (recorded by plan 17-12, 2026-09-02):** `grad/rhf.py:44` and
`grad/uhf.py:40` `assert isinstance(ni, MultiGridNumInt2)` — that is upstream's **v2**
multigrid (`multigrid_pair.py`, ported as `pyscf_pbc_dft::multigrid::MultiGridNumInt2`,
`crates/pyscf-pbc-dft/src/multigrid/pair.rs`), NOT v1 (`MultiGridNumInt`,
`multigrid.py`, plan 17-11). Plans 18-02/18-03 inherit the v2 half: gamma-point only,
not yet wired into the KRKS driver as a selectable `numint`, and (17-01's
measurement) slower than the reference `numint` at Gate-E scale — require it for
`isinstance` fidelity, do not expect a speed win from it. Its accuracy floor against
the reference route is the screening floor `precision · EXTRA_PREC` (~1e-6 on the
electron count), see `17-12-SUMMARY.md`.

**Standing rule from v1.0 that applies here:** `pyscf-grad`'s always-on
central-difference `verify_fd` harness (GRAD-09, D-01) is the primary numeric gate.
`pyscf-pbc-grad` MUST expose the same `verify_fd` on every gradient it ships, using
`Cell`-aware scanners. Do not ship an analytic gradient without it.

---

### 8.11 Phase 19 — Periodic response + relativistic

**Plans:** 8.

| Plan | Content | Port from |
|---|---|---|
| **19-01** | `scf/cphf` (176 l), `_response_functions` (47 l), `scf/newton_ah` (303 l), `scf/stability` (329 l) | Reuse `pyscf-grad`'s single matrix-free Krylov CPHF solver (GRAD-10) — **one implementation only, enforced by the existing source-scan gate** |
| **19-02** | `tdscf/krhf` (537 l) + `tdscf/rhf` (238 l) — TDA/TDHF at k | |
| **19-03** | `tdscf/kuhf` (540 l), `tdscf/uhf` (268 l), `tdscf/krks`, `kuks`, `rks`, `uks` (205 l) | |
| **19-04** | `gw/krgw_ac` (644 l) — analytic continuation G0W0 | |
| **19-05** | `gw/krgw_cd` (704 l) — contour deformation; `gw/kugw_ac` (784 l) | |
| **19-06** | `gw/kgw_slow`, `kgw_slow_supercell`, `gw_slow` (328 l) | |
| **19-07** | `adc/kadc_rhf` + `kadc_ao2mo` + `kadc_rhf_amplitudes` + IP (1061 l) + EA (1324 l) + `dfadc` | |
| **19-08** | `x2c/sfx2c1e` (355 l), `x2c/x2c1e` (286 l), `eph/eph_fd` (181 l), verification | |

---

### 8.12 Phase 20 — MPI, tools, bindings, ship

**Plans:** 8.

| Plan | Content |
|---|---|
| **20-01** | `pbc/tools`: `k2gamma` (345 l — k-point → supercell gamma transform), `lattice` (171 l — built-in lattice constants), `tril`, `print_funcs`, `make_test_cell` |
| **20-02** | `tools/pyscf_ase` (286 l) — ASE `Calculator` bridge; `tools/pywannier90` (1184 l) as an **optional feature-gated shim** over the external `wannier90` binary |
| **20-03** | `mpitools` (821 l): `mpi`, `mpi_pool`, `mpi_helper`, `mpi_load_balancer`, `mpi_blksize` — behind a `mpi` cargo feature, default OFF (D-PBC-18) |
| **20-04** | `mpicc` (4976 l): `kccsd_rhf`, `kintermediates_rhf`, `mpi_kpoint_helper` |
| **20-05** | **PyO3 bindings for the whole `pyscf.pbc.*` surface** in `pyscf-py` + `python/pyscf/pbc/*.py` shims (D-PBC-14). Mirror the Phase-3 subclass-override dispatch contract exactly: every overrideable hook goes through `slf.call_method1`, never Rust MRO. |
| **20-06** | Oracle CI hardening: the full `pyscf/pbc/*/test/` suite run against `pyscf-rs` as the import target; target ≥ 80% pass |
| **20-07** | Benchmarks: `pyscf-bench` periodic suite (KRHF/KRKS on diamond/Si/MOF-5 at 1×1×1 … 4×4×4), CPU vs CUDA vs ROCm, 2–5× claim validation |
| **20-08** | Milestone verification rollup + `FEATURES` file update + `PROJECT.md` Active→Validated moves |

---

### 8.13 Cross-cutting: what to do when you hit a missing cintx integral

Several periodic paths need integral families from `cintx`.

**Status correction (verified 2026-08-22):** the v1.0 Phase-7 audit claim that 6 of 8
gradient families are missing from cintx is **STALE**. `int1e_ip{ovlp,kin,nuc,rinv}`,
`int2e_ip1/ip2`, `int3c2e_ip1/ip2`, `int2c2e_ip1/ip2`, `ECPscalar_ipnuc`,
`ECPscalar_iprinv` and the rinv-origin shift (`Builder::with_rinv_origin`) all ship
today. See `/home/user/Documents/workspace/cintx/.planning/notes/gradient-family-gap-closure-PLAN.md`
§0 for the per-family evidence table. Phase 18 should therefore expect the gradient
integrals to be AVAILABLE; re-verify before gating anything.

**What IS genuinely missing, and what it blocks — see §2.4 for the full matrix:**

1. **10 moment-weighted families** (`int3c1e_r{2,4,6}_origk`, `int1e_r{2,4}_origi`, and
   their `_ip1_`/`_ip2` derivative variants). Declared in the manifest,
   `oracle_covered: false`, **no dispatch arm**. These gate **GTH pseudopotentials
   (Phase 10)** — i.e. every periodic calculation in this milestone. This is the real
   critical path, and it is cintx **Wave 0.5**.
2. **Hessian families** (`int1e_iprinvip`, `int1e_ipipr`, `int2e_ipvip1ipvip2`,
   `int2c2e_ip1ip2`, `int3c2e_ip1ip2`, `int3c2e_ipvip1`) plus 33 derivative symbols
   libcint exports from `src/autocode/*.c` without declaring in `cint_funcs.h`.
   **These block nothing in v2.0** — `pyscf/pbc` has no Hessian module. They matter to
   molecular `pyscf.hessian` / `pyscf.df.hessian`, not to PBC.

**Procedure — do this, do not improvise. Three states, not two.**

1. **Check the manifest row.**
   ```bash
   grep -n 'symbol_name: "<sym>' ../cintx/crates/cintx-ops/src/generated/api_manifest.rs
   ```
   No row → the symbol does not exist in cintx. Go to step 4.

2. **Check for a dispatch arm** in the owning family launcher:
   ```bash
   grep -n 'op_name == "<operator>"\|operator_name() == "<operator>"' \
        ../cintx/crates/cintx-cubecl/src/kernels/*.rs
   ```
   No arm → the manifest row is a *declaration only*. **This is the dangerous state**:
   the launcher may fall through and return a different operator's result rather than
   erroring (R-14). Go to step 3.

3. **Check `oracle_covered`.** `true` → usable. `false` → **not usable, and possibly
   fail-open.** Before consuming the value anywhere, add a test asserting it differs
   from the unweighted / underived parent integral on a real fixture. See plan 10-05
   Task 0 for the exact shape.

4. **Disposition when unusable:** implement the periodic body anyway, gate ONLY the
   affected numeric test behind `#[ignore = "blocked on cintx <symbol> (Wave N)"]`,
   record the blocker in the plan SUMMARY plus a row in `.planning/STATE.md`, and add
   it to the cintx plan's wave table if it is not already there.

5. **Never hand-roll a replacement integral inside `pyscf-pbc-*`.** Integrals belong in
   `cintx`. A one-off Rust Rys implementation in a method crate is a design violation.

**Current known-unusable set (2026-08-22):** the 10 moment-weighted families in §2.4.
Everything else this milestone needs is green.

---

## 9. Testing and oracle strategy

### 9.1 Three tiers of test — every plan must ship tier 1 and tier 2

| Tier | Name | Runs when | Example |
|---|---|---|---|
| **1** | **Invariant tests** — no upstream needed | every `cargo test` | Hermiticity, `ifft(fft(x))==x`, Bloch periodicity `ao_k(r+L)=e^{ikL}ao_k(r)`, `ft_aopair[G=0]==S`, supercell equivalence `E_KRHF(k-mesh) == E_RHF(supercell)/N` |
| **2** | **Hard-coded reference tests** — one number from one upstream run, committed | every `cargo test` | `cell.ewald() == -12.775… Ha` |
| **3** | **Live oracle** — upstream PySCF in the same process | CI, venv-gated, `#[ignore]` by default | full array byte-comparison |

**Tier 1 is the most valuable thing in this plan.** The supercell-equivalence identity
alone catches almost every k-point indexing bug. Write it early, run it always.

### 9.2 The reference systems — use these five everywhere

| Name | Cell | Basis / PP | Why |
|---|---|---|---|
| `diamond` | C₂, fcc `a = 3.5668 Å` | `gth-szv` / `gth-pade` | smallest realistic 3D insulator; 8 AOs |
| `si` | Si₂, fcc `a = 5.4306 Å` | `gth-szv` / `gth-pade` | narrow gap, tests occupation edge cases |
| `lif` | LiF, rocksalt `a = 4.03 Å` | `gth-szv` / `gth-pade` | strongly ionic; large Ewald term |
| `he_fcc` | He, fcc `a = 3.0 Å` | `gth-szv` (all-electron variant too) | tiny; the all-electron `get_nuc` path |
| `graphene` | C₂, hexagonal, 20 Å vacuum | `gth-szv` / `gth-pade` | `dimension = 2` path (Phase 12) |

Put them in `crates/pyscf-pbc-gto/tests/common/systems.rs` as constructor functions and
re-export from a `dev-dependencies` test-support module. **Do not redefine them per crate.**

### 9.3 Determinism gates (inherited from v1.0, non-negotiable)

- FMA-free `release-oracle` profile: `cargo build --profile release-oracle` +
  `xtask check_no_fma` must find zero `llvm.fmuladd` in every new crate.
- `oracle_zsum` bit-identical at `RAYON_NUM_THREADS=1` and `=8`.
- Complex eigenvector phases canonicalized via `canonicalize_signs` on the real part.
- Cross-platform: Linux x86_64 vs macOS aarch64 `KRHF` energies agree to 1 µHartree.

---

## 10. Risk register

| # | Risk | Severity | Mitigation | Owning plan |
|---|---|---|---|---|
| R-01 | `faer` has no complex `SelfAdjointEigen` → k-point diagonalization blocked | **SHOWSTOPPER** | The 2n×2n real-embedding fallback is mandated and written regardless (D-PBC-04) | 09-02 |
| R-02 | `cintx` cannot evaluate a shell pair across two basis sets | **SHOWSTOPPER** | Already proven possible: `build_combined_basis` + `int3c2e` with an auxmol ships in v1.0 (`intor.rs:987`). Smoke-test it in 09-01 before committing to D-PBC-07. | 09-01 |
| R-03 | Lattice-sum integrals are too slow (O(nimgs·nbas²) cintx calls) | **MAJOR** | Neighbor-list screening (10-02) + one-shot combined basis + `PBC_INTOR_ONE_SHOT_SHELL_LIMIT`. Benchmark in 10-03; if > 10× upstream, escalate to a batched cintx request API. | 10-02, 10-03 |
| R-04 | GEMM-based FFT is O(N^{4/3}) not O(N log N) — too slow at mesh ≥ 60³ | **MAJOR** | Stockham kernel (11-03). Blas engine ships first for correctness; swap on the 1e-13 match. | 11-03 |
| R-05 | `get_k_kpts` wrong by a phase convention → silently wrong energies | **MAJOR** | Supercell-equivalence tier-1 test (11-07). Never merge `get_k_kpts` without it. | 11-07 |
| R-06 | Global-aufbau `get_occ` implemented per-k → converges to the wrong state | **MAJOR** | Explicit test: for a metal at 2×2×2, occupancies differ between k-points | 11-09 |
| R-07 | exxdiv/Madelung convention mismatch → energy off by ~0.1 Ha | **MAJOR** | `madelung` hard-coded reference test (11-02) + `exxdiv=None` vs `"ewald"` difference equals `−madelung·nelec/2` exactly | 11-02 |
| R-08 | `ft_aopair` McMurchie–Davidson recursion wrong at `l ≥ 2` | **MAJOR** | Numerical-FT tier-1 test on a dense grid (13-01 test 2) + the `G=0 == S` identity (test 3) | 13-01 |
| R-15 | AFTDF and FFTDF treat `exxdiv='ewald'` DIFFERENTLY (AFTDF folds `Nk·vol·madelung` into `coulG[G+k=0]`; 2.12.1's FFTDF applies `_ewald_exxdiv_for_G0` analytically from the EXACT overlap, `df_jk.py:1480`). Read as a bug, "fixing" it would break the upstream match on one side or the other. **MEASURED: this is ~96% of `max\|vk_AFT − vk_FFT\|` at mesh 31** (6.2e-10 of 6.487e-10; `exxdiv=None` drops it 25× to 2.653e-11) — first-order, not a footnote. | **MAJOR** | §8.5 plan 13-05's boxed table; the floor is `≈ 2·madelung·‖S·D‖·‖δ‖` with `δ` = the Gate-1a residual, so Gates 1 and 2 are coupled quantitatively; 13-05 test 4 pins BOTH plateau levels. Characterise at mesh ≥ 31 only — at mesh 21 the two error sources cancel in the max-abs norm and hide it | 13-05 |
| R-16 | The Phase-13 gate "AFTDF KRHF == FFTDF KRHF to 1e-13" is unreachable, and so is any monotone-in-mesh restatement of it. **MEASURED: the difference plateaus at mesh 31 and is identical at mesh 41 to three digits, and separately plateaus at `rcut` ×1.5 (unchanged at ×2.0).** Two independent floors — AFTDF's `rcut` screening and FFTDF's mesh aliasing — and lowering one alone stalls against the other, so a monotone-convergence assertion would fail a CORRECT implementation. | **MAJOR** | §8.5 plan 13-08 states Gate 2 as a `(rcut, mesh)` **ladder** over the pairs (×1.0,21), (×1.0,31), (×1.5,31), (×1.5,41), pinned to the recorded upstream floor per pair; measurements committed at `.planning/phases/13-ft-ao-aftdf/measurements/` | 13-08 |
| R-09 | KCCSD memory blowup (`nkpts³·nvir⁴`) | **MAJOR** | Reuse `pyscf-ccsd`'s `WorkspacePool` + `PYSCF_MAX_MEMORY` pre-flight refusal + HDF5 spill | 16-02 |
| R-10 | ~~Missing cintx derivative integrals block Phase 18~~ **RETIRED 2026-08-22** — every gradient family Phase 18 needs ships and is oracle-covered (§2.4). | — | superseded by R-13/R-14 | — |
| R-13 | cintx moment-weighted families (`int3c1e_r{2,4,6}_origk`, `int1e_r{2,4}_origi`) are unimplemented → **GTH pseudopotentials do not work → no periodic SCF at all** | **SHOWSTOPPER** | They are cintx Wave 0.5 (10 symbols, one engine class, sph-only). §2.4 records the status; plans 10-05/10-06 carry a Task-0 availability check that fails loudly. All five reference systems (§9.2) use `gth-pade`, so this gates Phase 11 onward. **Land cintx Wave 0.5 before starting Phase 10.** | 10-05, 10-06 |
| R-14 | cintx family launchers fall through (`_ => {}`) on an unrecognised `operator_name`, returning the **unweighted parent integral instead of an error** | **SHOWSTOPPER if reachable** | cintx plan W0-05 proves/disproves; W0-06 lands a generic fail-closed guard. Until disproved, every pyscf-rs call site consuming an `oracle_covered:false` cintx symbol MUST assert the result differs from the unweighted parent before trusting it. | 10-05, 10-06, 18-01 |
| R-11 | 19 new crates explode CI time | MINOR | Group `pyscf-pbc-*` into one CI job with `cargo test --workspace --exclude`-style sharding by phase | 09-01 |
| R-12 | Low-dimension (`dimension < 3`) Coulomb truncation is subtly wrong | MINOR | D-PBC-20: explicitly `NotYetImplemented` until 12-08; never silently return a 3D answer | 12-08 |

---

## 11. Formula appendix (quick reference)

```
b            = 2π · inv(a)ᵀ                       reciprocal lattice
vol          = |det(a)|
Gv[x,y,z]    = rx[x]·b₀ + ry[y]·b₁ + rz[z]·b₂     rx = fftfreq(nx)·nx
weights      = |det(b)| / (2π)³ = 1/vol
SI[a,g]      = exp(−i·Gv[g]·R_a)
coulG[g]     = 4π / |k+G|²                        (0 at |k+G| = 0)
coulG_LR     = coulG · exp(−|k+G|²/4ω²)
coulG_SR     = coulG · (1 − exp(−|k+G|²/4ω²))
E_ewald      = ½ Σ_{L,i≠j} qᵢqⱼ erfc(η r)/r
             + ½ (4π/Ω) Σ_{G≠0} |ZS(G)|² exp(−G²/4η²)/G²
             − η/√π Σ qᵢ²  −  π (Σqᵢ)² / (2 η² Ω)
madelung     = −2 · ewald(MP-supercell with one probe charge)
ρ(r)         = (1/N_k) Σ_k Σ_μν ao*_μk(r) D^k_μν ao_νk(r)
vJ(r)        = ifft( coulG · fft(ρ) )
J^k_μν       = Σ_r ao*_μk(r) vJ(r) ao_νk(r) · (Ω/N_g)
E_elec       = (1/N_k) Σ_k [ Tr(D^k h^k) + ½ Tr(D^k v^k) ].real
ft_aopair    = Σ_{tuv} E_t E_u E_v (−iG)^{tuv} (π/p)^{3/2} e^{−G²/4p} e^{−iG·P}
kconserv     : k_l = k_i − k_j + k_k  (mod reciprocal lattice)
Bloch        : ao_k(r + L) = e^{i k·L} ao_k(r)
```

---

## 12. Where to look when you are stuck

| Problem | Where to look |
|---|---|
| "What does this Rust symbol do?" | `codegraph_explore("<name>")` — returns verbatim source + callers |
| "How do I write this cubecl kernel?" | `/home/user/Documents/workspace/cubecl_manual/manual/manual/Cubecl/INDEX.md` |
| cubecl build/link/feature error | `/home/user/Documents/workspace/cubecl_manual/manual/cubecl_error_guideline.md` — **mandatory, before any fix** |
| "What is the upstream algorithm?" | the `PORT` line of the plan — read that exact line range |
| "What are the repo conventions?" | `AGENTS.md`, `CONTRIBUTING.md`, `docs/rust_crate_test_guideline.md` |
| "What already exists?" | §2.1 of this document, then `codegraph_explore` |
| "Is this in scope?" | §1 — everything under `pyscf/pbc/` is in scope for v2.0 |
| An existing plan's format | `.planning/phases/07-gradients-geomopt/07-02-PLAN.md` |
