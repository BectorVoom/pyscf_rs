# PBC Architecture Design (v2.0 Milestone)

This document contains Sections 2 through 6 of the Periodic Boundary Conditions Master Plan (`.planning/pbc/PBC-MASTER-PLAN.md`).

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
