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
| **16** | Periodic CC + CI | `KCCSD` (RHF/UHF/GHF), `kintermediates`, `KCCSD(T)`, `EOM-KCCSD` IP/EA/EE, `KCIS` | `KRCCSD` `e_corr` matches upstream to 1e-8 on He 1×1×2 |
| **17** | k-point symmetry + multigrid | `pbc/symm/*`, `KPoints` IBZ machinery, all `*_ksymm` adapters, `dft/multigrid` | `KRHF` with `space_group_symmetry=True` equals the no-symmetry energy to 1e-9 |
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
builder that GDF/MDF/RSDF all sit on.

**Plans:** 6 (13-01 … 13-06).

---

#### Plan 13-01 — `ft_aopair` cubecl kernel (K-15) — **the biggest new kernel in v2.0**

**FILES** `crates/pyscf-kernels/src/pbc/ft_aopair.rs`, `crates/pyscf-pbc-df/src/ft_ao.rs`

**PORT** `pyscf/pbc/df/ft_ao.py:1-790` (whole file: `ft_aopair`, `ft_aopair_kpts`,
`ft_ao`, `gen_ft_kernel`, `_ft_aopair_kpts`, `_RangeSeparatedCell` helpers).

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
For the **periodic, k-resolved** form (`ft_aopair_kpts`), the second centre runs over
lattice images and carries the Bloch phase:
```
ft_aopair_kpts[k, μν, G] = Σ_L exp(i·k·L) · ft_aopair(A, B + L)[μν, G]
```
Contraction over primitives and the Cartesian→spherical transform
(`pyscf_gto::cart2sph_coeff`) are applied afterwards, exactly as in the molecular path.

**KERNEL DESIGN**
- One cube per `(shell-pair, G-block)`. `CubeDim { x: 256, y: 1, z: 1 }`, one thread per G.
- Precompute `E_t` on the **host** (small, `(i+1)(j+1)(i+j+1)` per axis per primitive
  pair) and upload as a flat `f64` array. The device loop is then a pure
  polynomial-times-exponential evaluation — no recursion on device.
- Generic over `F: Float` per AGENTS.md §3.
- Screening: skip a `(shell-pair, L)` whose `K_AB · (π/p)^{3/2}` is below
  `cell.precision`.

**TEST** `crates/pyscf-kernels/tests/pbc_ft_aopair.rs`
1. **s-s analytic check (no oracle):** for two s primitives, compare against the
   closed form `(π/p)^{3/2}·exp(−G²/4p)·exp(−iG·P)·K_AB` to 1e-14.
2. **Numerical-FT check (no oracle):** on a dense real-space grid,
   `Σ_r ao_μ(r)·ao_ν(r)·exp(−iG·r)·(vol/ngrids)` matches `ft_aopair` to 1e-6 for a
   `[40,40,40]` mesh, for `l` up to 2.
3. **G = 0 identity:** `ft_aopair[μν, G=0]` equals the periodic overlap
   `pbc_intor("int1e_ovlp")[μν]` to 1e-10. **This is the single best correctness gate.**
4. Oracle-gated: match `pyscf.pbc.df.ft_ao.ft_aopair` to 1e-10.

**DONE** `cargo test -p pyscf-kernels --test pbc_ft_aopair`

---

#### Plans 13-02 … 13-06

| Plan | Content | Port from |
|---|---|---|
| **13-02** | `ft_ao` (single-AO FT, `Σ_L e^{ikL} ∫ φ_μ(r−L) e^{−iGr}`), `gen_ft_kernel` dispatch, range-separated cell splitting | `df/ft_ao.py` (rest) |
| **13-03** | `AFTDF`: `build`, `get_nuc`, `get_pp`, `pw_loop`, `ft_loop`, `weighted_coulG` | `df/aft.py` (776 l) |
| **13-04** | `aft_jk`: `get_j_kpts`, `get_k_kpts`, `get_jk`, `_format_dms` | `df/aft_jk.py` (753 l) |
| **13-05** | `aft_ao2mo`: `get_eri`, `general`, `ao2mo_7d` | `df/aft_ao2mo.py` (434 l) |
| **13-06** | Verification: AFTDF `KRHF` energy == FFTDF `KRHF` energy to 1e-6 on 3 systems | — |

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
| **14-05** | `df_ao2mo`, `outcore` | `df/df_ao2mo.py` (351 l), `df/outcore.py` (250 l) | |
| **14-06** | `MDF` (mixed density fitting = GDF + AFT residual) | `df/mdf.py`, `mdf_jk.py`, `mdf_ao2mo.py` (785 l) | |
| **14-07** | `rsdf_helper` + `rsdf_builder` (range-separated GDF) | `df/rsdf_helper.py` (1348 l), `rsdf_builder.py` (1631 l) | Largest single porting task in the milestone; split into 3 sub-tasks by function group |
| **14-08** | `RSDF` class + `rsdf_jk`; `scf/rsjk.py` (range-separated JK build, no DF) | `df/rsdf.py` (680 l), `rsdf_jk.py`, `scf/rsjk.py` (1355 l) | |
| **14-09** | Verification: every DF builder gives the same `KRHF` energy to 1e-6 on diamond 2×2×2; GDF uses < 20% of FFTDF memory at mesh `[40,40,40]` | — | |

---

### 8.7 Phase 15 — Periodic AO2MO + KMP2

**Plans:** 5.

| Plan | Content | Port from |
|---|---|---|
| **15-01** | `kconserv` tables (K-16), `KptsHelper`, `ktensor` | `lib/kpts_helper.py`, `lib/ktensor.py` (386 l) |
| **15-02** | `pbc/ao2mo/eris`: `general`, `get_eri` at k-quadruples; per-DF `fft_ao2mo`, `df_ao2mo` wiring | `pbc/ao2mo/eris.py` (258 l), `df/fft_ao2mo.py` (484 l) |
| **15-03** | `KMP2`: `kernel`, `_gamma1_intermediates`, `make_rdm1`, frozen core at k | `mp/kmp2.py` (821 l) |
| **15-04** | `KUMP2` + `kmp2_stagger` (staggered-mesh finite-size correction) | `mp/kump2.py` (423 l), `mp/kmp2_stagger.py` (419 l) |
| **15-05** | Verification: `KMP2(diamond, 2×2×2)` `e_corr` matches upstream to 1e-8 | — |

**Key formula (KMP2):**
```
e_corr = (1/nkpts) Σ_{ki,kj,ka} Σ_{ijab}
         (ia|jb)·[2·(ia|jb)* − (ib|ja)*] / (ε_i^{ki} + ε_j^{kj} − ε_a^{ka} − ε_b^{kb})
```
with `kb` fixed by momentum conservation `kconserv[ki, ka, kj]`.

---

### 8.8 Phase 16 — Periodic Coupled Cluster + CI

**Plans:** 10. This is the largest phase by line count (13,675 + 852 lines upstream).

| Plan | Content | Port from |
|---|---|---|
| **16-01** | `kintermediates_rhf` (Foo/Fvv/Fov/Woooo/Wvvvv/Wvoov…) at k | `cc/kintermediates_rhf.py` (926 l) |
| **16-02** | `KRCCSD`: `update_amps`, `energy`, `kernel`, k-batched contractions (K-17) | `cc/kccsd_rhf.py` (1203 l) |
| **16-03** | `kintermediates_uhf` + `KUCCSD` | `cc/kintermediates_uhf.py` (1225 l), `cc/kccsd_uhf.py` (1116 l) |
| **16-04** | `KGCCSD` (spin-orbital) + `kintermediates` | `cc/kccsd.py` (833 l), `cc/kintermediates.py` (529 l) |
| **16-05** | `KCCSD(T)`: `kccsd_t`, `kccsd_t_rhf` | `cc/kccsd_t.py` + `kccsd_t_rhf.py` (970 l) |
| **16-06** | `EOM-KCCSD-RHF` IP/EA | `cc/eom_kccsd_rhf.py` (1716 l) + ip/ea (158 l) |
| **16-07** | `EOM-KCCSD-UHF` | `cc/eom_kccsd_uhf.py` (1275 l) |
| **16-08** | `EOM-KCCSD-GHF` (incl. EE) | `cc/eom_kccsd_ghf.py` (2011 l) |
| **16-09** | `KCIS` + `pbc/ci/cisd` | `ci/kcis_rhf.py` (700 l), `ci/cisd.py` (116 l) |
| **16-10** | `kuccsd_rdm`, `pbc/cc/ccsd.py` gamma shim, verification | `cc/kuccsd_rdm.py`, `cc/ccsd.py` |

**Reuse note:** `pyscf-ccsd` already owns the molecular `WorkspacePool` tensor arena and
`PYSCF_MAX_MEMORY` pre-flight refusal. `pyscf-pbc-cc` MUST reuse both — a k-point CCSD
`Wvvvv` is `nkpts³ × nvir⁴` and will OOM without them.

---

### 8.9 Phase 17 — k-point symmetry + multigrid

**Plans:** 8.

| Plan | Content | Port from |
|---|---|---|
| **17-01** | `pbc/symm`: `geom` (lattice symmetry detection), `tables`, `group` (`PGElement`, `PointGroup`, `Representation`) | `symm/geom.py`, `tables.py`, `group.py` (821 l) |
| **17-02** | `symm/space_group.py` (SpaceGroup detection + operations) + `symmetry.py` (`Symmetry` base) | (717 l) |
| **17-03** | `lib/kpts.py` `KPoints`: `make_kpts_ibz`, `stars`, `bz2ibz`, `addition_table`, `inverse_table`, `get_kconserv`, `transform_*` (dm/fock/mo_coeff/mo_energy) | `lib/kpts.py` (1223 l) |
| **17-04** | `khf_ksymm`, `kuhf_ksymm`, `kghf_ksymm` (IBZ-restricted SCF) | (840 l) |
| **17-05** | `krks_ksymm`, `kuks_ksymm`, `krkspu_ksymm`, `kukspu_ksymm` | (422 l) |
| **17-06** | `kmp2_ksymm`, `kccsd_rhf_ksymm` | (1091 l) |
| **17-07** | `dft/multigrid/multigrid.py` (1962 l) + `multigrid_pair.py` (1257 l) + `pp.py` + `utils.py` | Multi-level real-space collocation — port the `_backend_c.py` reference semantics into cubecl kernels |
| **17-08** | Verification: symmetry-restricted energies equal full-BZ energies to 1e-9 for 5 systems | — |

**`spglib` note:** `symm/pyscf_spglib.py` is an *optional* bridge. Implement the
native detection (17-01/17-02) and expose a `spglib` feature that, when off,
returns the native result. Do NOT make `spglib` a required dependency.

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
